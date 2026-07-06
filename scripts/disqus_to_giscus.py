#!/usr/bin/env python3
"""
Migrate Disqus XML export to GitHub Discussions for giscus.

Usage:
  python3 scripts/disqus_to_giscus.py \\
      --xml /path/to/disqus-export.xml \\
      --token ghp_xxxx \\
      --category "General"

  To delete previously migrated discussions:
  python3 scripts/disqus_to_giscus.py --delete --token ghp_xxxx
"""

import argparse
import gzip
import os
import sys
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from datetime import datetime

import requests

GRAPHQL_URL = "https://api.github.com/graphql"
NS = {"d": "http://disqus.com"}
DSQ_NS = "{http://disqus.com/disqus-internals}"


def graphql(token, query, variables=None):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(GRAPHQL_URL, json={"query": query, "variables": variables or {}}, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"GraphQL HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data["data"]


def get_repo_id(token, repo_full_name):
    owner, repo = repo_full_name.split("/")
    return graphql(token, """
        query($owner: String!, $repo: String!) {
            repository(owner: $owner, name: $repo) { id }
        }
    """, {"owner": owner, "repo": repo})["repository"]["id"]


def get_category_id(token, repo_full_name, category_name):
    owner, repo = repo_full_name.split("/")
    data = graphql(token, """
        query($owner: String!, $repo: String!) {
            repository(owner: $owner, name: $repo) {
                discussionCategories(first: 20) { nodes { id name } }
            }
        }
    """, {"owner": owner, "repo": repo})
    for cat in data["repository"]["discussionCategories"]["nodes"]:
        if cat["name"].lower() == category_name.lower():
            return cat["id"]
    raise ValueError(
        f"Category '{category_name}' not found. "
        f"Available: {[c['name'] for c in data['repository']['discussionCategories']['nodes']]}"
    )


def create_discussion(token, repo_id, category_id, title, body):
    data = graphql(token, """
        mutation($repoId: ID!, $categoryId: ID!, $title: String!, $body: String!) {
            createDiscussion(input: {
                repositoryId: $repoId, categoryId: $categoryId,
                title: $title, body: $body
            }) { discussion { id } }
        }
    """, {"repoId": repo_id, "categoryId": category_id, "title": title, "body": body})
    return data["createDiscussion"]["discussion"]["id"]


def add_comment(token, discussion_id, body, reply_to_id=None):
    args = {"discussionId": discussion_id, "body": body}
    if reply_to_id:
        args["replyToId"] = reply_to_id
    data = graphql(token, """
        mutation($discussionId: ID!, $body: String!, $replyToId: ID) {
            addDiscussionComment(input: {
                discussionId: $discussionId, body: $body, replyToId: $replyToId
            }) { comment { id } }
        }
    """, args)
    return data["addDiscussionComment"]["comment"]["id"]


def delete_discussion(token, discussion_id):
    graphql(token, """
        mutation($id: ID!) {
            deleteDiscussion(input: { id: $id }) { discussion { id } }
        }
    """, {"id": discussion_id})


def list_discussions(token, repo_full_name, category_id, first=50):
    owner, repo = repo_full_name.split("/")
    data = graphql(token, """
        query($owner: String!, $repo: String!, $categoryId: ID!, $first: Int!) {
            repository(owner: $owner, name: $repo) {
                discussions(first: $first, categoryId: $categoryId) {
                    nodes { id title }
                }
            }
        }
    """, {"owner": owner, "repo": repo, "categoryId": category_id, "first": first})
    return data["repository"]["discussions"]["nodes"]


def parse_disqus_xml(path):
    # Handle .gz files transparently
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            tree = ET.parse(f)
    else:
        tree = ET.parse(path)
    root = tree.getroot()

    # Collect all threads, keyed by slug (path) for deduplication
    thread_map = {}
    for el in root.findall("d:thread", NS):
        tid = el.get(DSQ_NS + "id", "")
        link = (el.findtext("d:link", "", NS) or "").strip()
        title = (el.findtext("d:title", "", NS) or "").strip()
        if not link or not tid:
            continue

        # Use the URL path as the key (strip query params and scheme/host)
        from urllib.parse import urlparse
        parsed = urlparse(link)
        path = parsed.path.rstrip("/")
        has_query = bool(parsed.query)
        is_prod = link.startswith("https://kharshit.github.io")

        if path not in thread_map:
            thread_map[path] = {"id": tid, "title": title, "link": link}
        elif is_prod and not has_query:
            # Prefer clean production URL over ones with query params
            thread_map[path] = {"id": tid, "title": title, "link": link}

    # Collect posts by thread id (with parent info)
    posts_by_tid = {}
    for el in root.findall("d:post", NS):
        thread_el = el.find("d:thread", NS)
        tid = thread_el.get(DSQ_NS + "id") if thread_el is not None else None
        if not tid:
            continue
        msg = (el.findtext("d:message", "", NS) or "").strip()
        author_el = el.find("d:author", NS)
        author = "Anonymous"
        if author_el is not None:
            author = author_el.findtext("d:name", "Anonymous", NS) or "Anonymous"
        created = (el.findtext("d:createdAt", "", NS) or "").strip()
        parent_el = el.find("d:parent", NS)
        parent = parent_el.get(DSQ_NS + "id") if parent_el is not None else None
        pid = el.get(DSQ_NS + "id", "")

        if msg:
            posts_by_tid.setdefault(tid, []).append({
                "id": pid, "message": msg, "author": author.strip(),
                "created": created, "parent": parent,
            })

    # Build final thread list: deduped, only threads that have posts
    seen_slugs = set()
    final_threads = []
    for slug, t in thread_map.items():
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        if t["id"] in posts_by_tid and posts_by_tid[t["id"]]:
            final_threads.append(t)

    return final_threads, posts_by_tid


def format_post_body(post, depth=0):
    indent = "> " * depth
    ts = ""
    if post["created"]:
        try:
            dt = datetime.fromisoformat(post["created"].replace("Z", "+00:00"))
            ts = dt.strftime("%b %d, %Y at %H:%M UTC")
        except ValueError:
            ts = post["created"]
    header = f"**{post['author']}** commented on {ts} *(migrated from Disqus)*" if ts else f"**{post['author']}** *(migrated from Disqus)*"
    return f"{indent}{header}\n\n{indent}{post['message']}\n"


def handle_delete(args):
    """Delete all previously migrated discussions from the target category."""
    print("Fetching existing discussions...")
    repo_id = get_repo_id(args.token, args.repo)
    category_id = get_category_id(args.token, args.repo, args.category)
    discussions = list_discussions(args.token, args.repo, category_id)

    if not discussions:
        print("No discussions found to delete.")
        return

    print(f"Found {len(discussions)} discussions:")
    for d in discussions:
        print(f"  {d['id']}: {d['title'][:80]}")
    print()

    if not args.yes:
        confirm = input("Delete all these discussions? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Cancelled.")
            return

    for d in discussions:
        print(f"  Deleting: {d['title'][:60]}...")
        try:
            delete_discussion(args.token, d["id"])
            time.sleep(1)
        except Exception as e:
            print(f"  Error deleting {d['id']}: {e}", file=sys.stderr)

    print("Done.")


def handle_migrate(args):
    """Migrate Disqus export to GitHub Discussions."""
    if not os.path.exists(args.xml):
        print(f"Error: XML file not found: {args.xml}", file=sys.stderr)
        sys.exit(1)

    print("Parsing Disqus export...")
    threads, posts_by_tid = parse_disqus_xml(args.xml)
    print(f"Found {len(threads)} active threads (with comments) after deduplication")
    total_comments = sum(len(posts_by_tid.get(t["id"], [])) for t in threads)
    print(f"Total comments to migrate: {total_comments}")

    if not threads:
        print("Nothing to migrate. Exiting.")
        return

    print("\nThreads to migrate:")
    for t in threads:
        count = len(posts_by_tid.get(t["id"], []))
        print(f"  {t['link'][:80]:80s} ({count} comments)")

    if args.dry_run:
        print("\nDry run complete.")
        return

    print("\nFetching repo and category IDs from GitHub...")
    repo_id = get_repo_id(args.token, args.repo)
    category_id = get_category_id(args.token, args.repo, args.category)
    print(f"Repo ID: {repo_id}")
    print(f"Category ID: {category_id}")

    stats = {"created": 0, "comments": 0, "errors": 0}

    for i, thread in enumerate(threads, 1):
        url = thread["link"]
        title = urlparse(url).path or url
        thread_posts = posts_by_tid.get(thread["id"], [])

        # Sort posts by created date
        thread_posts.sort(key=lambda p: p["created"])

        # Use a generic discussion body so ALL posts become comments
        # (this allows proper threading including replies to the first comment)
        body = f"Discussion for [{title}]({url})"
        all_posts = thread_posts  # all posts added as comments
        posts_lookup = {p["id"]: p for p in all_posts}

        def top_level_ancestor(pid):
            """Walk up the parent chain to find the first ancestor with no parent."""
            visited = set()
            cur = pid
            while cur and cur not in visited:
                visited.add(cur)
                p = posts_lookup.get(cur)
                if not p or not p["parent"]:
                    return cur
                cur = p["parent"]
            return None

        time.sleep(args.rate_limit)
        print(f"\n[{i}/{len(threads)}] Creating discussion for: {title}")
        try:
            disc_id = create_discussion(args.token, repo_id, category_id, title, body)
            stats["created"] += 1
            print(f"  Created: {disc_id}")

            # Track mapping: disqus_post_id -> github_comment_id for threading
            id_map = {}

            for j, post in enumerate(all_posts, 1):
                # Find effective reply target: if the direct parent is itself
                # a reply (GitHub only supports one level), walk up to the
                # top-level ancestor and reply there instead.
                reply_to = None
                if post["parent"]:
                    target = top_level_ancestor(post["parent"])
                    if target in id_map:
                        reply_to = id_map[target]

                try:
                    gh_comment_id = add_comment(args.token, disc_id, format_post_body(post), reply_to)
                    id_map[post["id"]] = gh_comment_id
                    stats["comments"] += 1
                except Exception:
                    # Reply failed — fall back to top-level
                    try:
                        gh_comment_id = add_comment(args.token, disc_id, format_post_body(post))
                        id_map[post["id"]] = gh_comment_id
                        stats["comments"] += 1
                    except Exception as e2:
                        print(f"  Error on comment {j}: {e2}", file=sys.stderr)
                        stats["errors"] += 1
                time.sleep(args.rate_limit)

            time.sleep(args.rate_limit)
        except Exception as e:
            print(f"  Error creating discussion: {e}", file=sys.stderr)
            stats["errors"] += 1

    print("\n=== Migration complete ===")
    print(f"Discussions created: {stats['created']}")
    print(f"Comments migrated: {stats['comments']}")
    print(f"Errors: {stats['errors']}")


def main():
    parser = argparse.ArgumentParser(description="Migrate Disqus export to GitHub Discussions")
    parser.add_argument("--xml", help="Path to Disqus XML export")
    parser.add_argument("--token", required=True, help="GitHub PAT with repo scope")
    parser.add_argument("--repo", default="kHarshit/kHarshit.github.io")
    parser.add_argument("--category", default="General")
    parser.add_argument("--dry-run", action="store_true", help="Parse XML but don't create discussions")
    parser.add_argument("--rate-limit", type=int, default=1, help="Seconds between API calls")
    parser.add_argument("--delete", action="store_true", help="Delete all existing discussions in the category")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation for --delete")
    args = parser.parse_args()

    if args.delete:
        handle_delete(args)
    elif args.xml:
        handle_migrate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
