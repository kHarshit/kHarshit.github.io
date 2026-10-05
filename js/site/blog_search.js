/*
 * Search for the blog page.
 *
 * /blog/ paginates at 10 of 84 posts, so filtering what is already in the DOM
 * would only ever search one page. Instead this fetches /search.json (every
 * post: title, excerpt, categories and section headings, ~12KB gzipped) and
 * swaps the paginated list for a result list while a query is active.
 *
 * No search library: at 84 records over four short fields the whole corpus
 * scans in well under a millisecond, and lunr/minisearch/fuse would each cost
 * more bytes than the entire index.
 */
(function () {
    var box = document.getElementById('blog-search');
    var input = document.getElementById('blog-search-input');
    var clearBtn = document.getElementById('blog-search-clear');
    var status = document.getElementById('blog-search-status');
    var results = document.getElementById('blog-search-results');
    var paginated = document.getElementById('blog-paginated-list');
    // Absent when there is only one page of posts.
    var pagination = document.querySelector('#main .pagination');

    if (!box || !input || !results || !paginated) return;

    // The box ships hidden so that a missing or broken script leaves the normal
    // paginated page intact rather than an input that does nothing.
    box.hidden = false;

    var MIN_QUERY = 2;
    var index = null;
    var indexPromise = null;
    var attempts = 0;

    /* ---------------------------------------------------------------- index */

    function loadIndex() {
        if (indexPromise) return indexPromise;
        attempts++;
        indexPromise = fetch(box.dataset.index)
            .then(function (r) {
                if (!r.ok) throw new Error('HTTP ' + r.status);
                return r.json();
            })
            .then(function (data) {
                // Lowercase once here, not on every keystroke.
                for (var i = 0; i < data.length; i++) {
                    var it = data[i];
                    it._t = (it.t || '').toLowerCase();
                    it._e = (it.e || '').toLowerCase();
                    it._h = (it.h || '').toLowerCase();
                    it._c = (it.c || []).join(' ').toLowerCase();
                }
                index = data;
                return data;
            })
            .catch(function (err) {
                // Let a later keystroke retry once, but do not retry per key.
                indexPromise = null;
                throw err;
            });
        return indexPromise;
    }

    /* --------------------------------------------------------------- search */

    // Substring matching is directional: "benchmark" finds "Benchmarks" but
    // "benchmarks" would miss "Benchmark". Dropping a trailing "s" covers the
    // common plural case. This is not stemming - "training" still will not find
    // "train" - which is the accepted trade for having no dependency.
    function normalize(token) {
        return token.length > 4 && token.charAt(token.length - 1) === 's'
            ? token.slice(0, -1)
            : token;
    }

    // Every token must land in some field (AND), so a second word narrows the
    // result set rather than widening it. Each token scores by the strongest
    // field it hits.
    function score(item, tokens, query) {
        var total = 0;
        for (var i = 0; i < tokens.length; i++) {
            var tok = tokens[i], s;
            if (item._t.indexOf(tok) > -1) s = 10;          // title
            else if (item._c.indexOf(tok) > -1) s = 6;      // categories
            else if (item._h.indexOf(tok) > -1) s = 4;      // headings
            else if (item._e.indexOf(tok) > -1) s = 2;      // excerpt
            else return -1;
            total += s;
        }
        if (item._t.indexOf(query) > -1) total += 15;       // whole phrase in title
        if (item._t.indexOf(query) === 0) total += 10;      // title starts with it
        return total;
    }

    function search(query) {
        var raw = query.toLowerCase();
        var parts = raw.split(/\s+/);
        var tokens = [];
        for (var i = 0; i < parts.length; i++) {
            if (parts[i]) tokens.push(normalize(parts[i]));
        }

        var scored = [];
        for (var j = 0; j < index.length; j++) {
            var s = score(index[j], tokens, raw);
            // Keep the source order as a tie-break rather than relying on sort
            // stability; site.posts is newest-first.
            if (s >= 0) scored.push({ item: index[j], score: s, order: j });
        }
        scored.sort(function (a, b) {
            return b.score - a.score || a.order - b.order;
        });
        return scored;
    }

    /* --------------------------------------------------------------- render */

    function el(tag, className, text) {
        var node = document.createElement(tag);
        if (className) node.className = className;
        if (text != null) node.textContent = text;
        return node;
    }

    // Mirrors the Liquid markup in blog/index.html so the existing
    // _sass/_blog-list.scss rules (including dark mode) apply unchanged.
    function renderItem(item) {
        var li = el('li', 'blog-list-item');
        var a = el('a', 'glass-card');
        a.href = item.u;
        a.title = item.t;

        var thumb = el('div', 'blog-list-thumb');
        if (item.i) {
            var img = document.createElement('img');
            img.src = item.i;
            img.alt = item.t;
            img.loading = 'lazy';
            thumb.appendChild(img);
        } else {
            var ph = el('div', 'blog-list-thumb-placeholder');
            ph.appendChild(el('i', 'fa-regular fa-file-lines'));
            thumb.appendChild(ph);
        }

        var body = el('div', 'blog-list-body');
        if (item.c && item.c.length) {
            body.appendChild(el('span', 'blog-list-category', item.c[0]));
        }
        body.appendChild(el('h3', 'blog-list-title', item.t));
        body.appendChild(el('p', 'blog-list-excerpt', item.e));

        var meta = el('div', 'blog-list-meta');
        var date = el('span', 'blog-list-date');
        date.appendChild(el('i', 'fa-regular fa-calendar'));
        date.appendChild(document.createTextNode(' ' + item.d));
        var read = el('span', 'blog-list-readtime');
        read.appendChild(el('i', 'fa-regular fa-clock'));
        read.appendChild(document.createTextNode(' ' + item.r + ' min read'));
        meta.appendChild(date);
        meta.appendChild(read);
        body.appendChild(meta);

        a.appendChild(thumb);
        a.appendChild(body);
        li.appendChild(a);
        return li;
    }

    function renderResults(scored, query) {
        var frag = document.createDocumentFragment();
        if (!scored.length) {
            frag.appendChild(el('li', 'blog-list-empty', 'No posts match “' + query + '”.'));
        } else {
            for (var i = 0; i < scored.length; i++) {
                frag.appendChild(renderItem(scored[i].item));
            }
        }
        results.innerHTML = '';
        results.appendChild(frag);

        status.textContent = scored.length
            ? scored.length + ' result' + (scored.length === 1 ? '' : 's') + ' for “' + query + '”'
            : 'No posts match “' + query + '”';
    }

    /* ----------------------------------------------------------- view modes */

    function showResults() {
        paginated.hidden = true;
        results.hidden = false;
        if (pagination) pagination.style.display = 'none';
    }

    function showPaginated() {
        paginated.hidden = false;
        results.hidden = true;
        results.innerHTML = '';
        if (pagination) pagination.style.display = '';
        status.textContent = '';
    }

    /* ---------------------------------------------------------------- input */

    function run() {
        var query = input.value.trim();
        clearBtn.hidden = !input.value;

        // A single character matches most of the archive, and flipping the
        // whole page on a stray keypress is jarring.
        if (query.length < MIN_QUERY) {
            showPaginated();
            return;
        }

        if (index) {
            showResults();
            renderResults(search(query), query);
            return;
        }

        status.textContent = 'Loading search…';
        loadIndex().then(function () {
            // The query may have grown while the fetch was in flight.
            if (input.value.trim().length >= MIN_QUERY) run();
            else showPaginated();
        }, function () {
            status.textContent = attempts > 1
                ? 'Search is unavailable right now.'
                : 'Could not load search — try again.';
        });
    }

    function reset() {
        input.value = '';
        clearBtn.hidden = true;
        showPaginated();
        input.focus();
    }

    input.addEventListener('input', run);
    // Prefetch so the index has usually landed before the first character does.
    input.addEventListener('focus', function () { loadIndex().catch(function () {}); });
    input.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') reset();
    });
    clearBtn.addEventListener('click', reset);
})();
