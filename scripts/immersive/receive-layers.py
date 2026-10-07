#!/usr/bin/env python3
"""
Receive image layers rendered by scripts/immersive/capture-layers.js and save
them into the repo.

The capture runs in the browser (it needs WebGL), so it POSTs each layer
here instead of downloading files one by one.

Usage:
  python3 scripts/immersive/receive-layers.py img/poems/immersive/snowy-woods
  # then, on the poem page served by `bundle exec jekyll serve`, in the console:
  #   (await import('/scripts/immersive/capture-layers.js')).capture()

Only listens on localhost, and only writes `<name>.webp` / `<name>.json`
files with simple names into the given directory.
"""

import os
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

PORT = 8765
NAME = re.compile(r"^[a-z0-9-]+\.(webp|json)$")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out_dir = os.path.abspath(sys.argv[1])
    os.makedirs(out_dir, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Private-Network", "true")

        def do_OPTIONS(self):
            self.send_response(204)
            self.cors()
            self.end_headers()

        def do_POST(self):
            name = parse_qs(urlparse(self.path).query).get("name", [""])[0]
            if not NAME.match(name):
                self.send_response(400)
                self.cors()
                self.end_headers()
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            with open(os.path.join(out_dir, name), "wb") as f:
                f.write(body)
            print(f"saved {name} ({len(body) / 1024:.0f} KB)", flush=True)
            self.send_response(200)
            self.cors()
            self.end_headers()

        def log_message(self, *args):
            pass

    print(f"Saving layers to {out_dir} on http://localhost:{PORT}", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
