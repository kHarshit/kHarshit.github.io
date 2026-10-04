(function() {
  // Reading progress bar
  var bar = document.getElementById('reading-progress-bar');
  if (bar) {
    window.addEventListener('scroll', function() {
      var doc = document.documentElement;
      var scrolled = doc.scrollTop || document.body.scrollTop;
      var total = doc.scrollHeight - doc.clientHeight;
      bar.style.width = (total > 0 ? (scrolled / total) * 100 : 0) + '%';
    }, { passive: true });
  }

  // Code copy buttons
  document.querySelectorAll('figure.highlight').forEach(function(block) {
    var btn = document.createElement('button');
    btn.className = 'code-copy-btn';
    btn.textContent = 'Copy';
    block.appendChild(btn);
    btn.addEventListener('click', function() {
      var code = block.querySelector('code') || block.querySelector('pre');
      navigator.clipboard.writeText(code.innerText).then(function() {
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(function() {
          btn.textContent = 'Copy';
          btn.classList.remove('copied');
        }, 2000);
      });
    });
  });

  // Lazy-load images inside post content
  document.querySelectorAll('.post-content img').forEach(function(img) {
    if (!img.getAttribute('loading')) img.setAttribute('loading', 'lazy');
  });

  // Heading anchor links + Table of Contents
  var headings = Array.from(document.querySelectorAll('.post-content h1, .post-content h2, .post-content h3'));

  headings.forEach(function(h) {
    if (!h.id) return;
    var a = document.createElement('a');
    a.className = 'heading-anchor';
    a.href = '#' + h.id;
    a.setAttribute('aria-label', 'Link to this section');
    a.textContent = '#';
    h.appendChild(a);
  });

  var tocHeadings = headings.filter(function(h) { return !!h.id && !h.hasAttribute('data-toc-exclude'); });
  if (tocHeadings.length >= 2) {
    var block = document.getElementById('toc-block');
    var list = document.getElementById('toc-list');
    if (block && list) {
      block.removeAttribute('hidden');
      // At <=768px the TOC stacks above the article (see _post.scss), and a long
      // one pushed the first paragraph about two screens down. Start it closed
      // there; one tap on "Contents" opens it.
      if (window.matchMedia && window.matchMedia('(max-width: 768px)').matches) {
        block.open = false;
      }

      var tocLinks = [];
      var ol = document.createElement('ol');
      tocHeadings.forEach(function(h) {
        var li = document.createElement('li');
        li.className = h.tagName === 'H1' ? 'toc-h1' : h.tagName === 'H3' ? 'toc-h3' : 'toc-h2';
        var a = document.createElement('a');
        a.href = '#' + h.id;
        a.textContent = h.textContent.replace(/#\s*$/, '').trim();
        li.appendChild(a);
        ol.appendChild(li);
        tocLinks.push({ heading: h, link: a });
      });
      list.appendChild(ol);

      tocLinks.forEach(function(item) {
        item.link.addEventListener('click', function(e) {
          e.preventDefault();
          item.heading.scrollIntoView({ behavior: 'smooth', block: 'start' });
          history.pushState(null, null, '#' + item.heading.id);
        });
      });

      var spyOffset = 100;
      function absoluteTop(el) {
        var top = 0;
        while (el) { top += el.offsetTop; el = el.offsetParent; }
        return top;
      }
      function updateActive() {
        var current = null;
        tocLinks.forEach(function(item) {
          if (window.scrollY >= absoluteTop(item.heading) - spyOffset) {
            current = item;
          }
        });
        // current stays null while scrolled above the first heading.
        tocLinks.forEach(function(item) {
          item.link.classList.toggle('active', current !== null && item.link === current.link);
        });
      }

      window.addEventListener('scroll', updateActive, { passive: true });
      updateActive();
    }
  }
})();
