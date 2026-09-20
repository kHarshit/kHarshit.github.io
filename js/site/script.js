document.addEventListener('DOMContentLoaded', function () {

  // ── Scroll-to-top button ──────────────────────────────────
  window.addEventListener('scroll', function () {
    var btn = document.getElementById('scroll_top');
    if (!btn) return;
    var scrolled = document.body.scrollTop || document.documentElement.scrollTop;
    btn.style.display = scrolled > 600 ? 'block' : 'none';
  }, { passive: true });

  // ── Typing loop for hero title ────────────────────────────
  var el = document.getElementById('hero-title');
  if (el) {
    var titles = [
      'Machine Learning Engineer',
      'AI Research Engineer',
      'Generative AI Engineer',
      'Computer Vision Engineer',
      'LLM Engineer',
      'Deep Learning Engineer'
    ];
    var titleIndex = 0;
    var charIndex = 0;
    var isDeleting = false;
    var typeSpeed = 45;
    var deleteSpeed = 25;
    var pauseAfterType = 2000;
    var pauseAfterDelete = 500;

    function typeLoop() {
      var currentText = titles[titleIndex];
      if (!isDeleting) {
        el.textContent = currentText.substring(0, charIndex + 1);
        charIndex++;
        if (charIndex === currentText.length) {
          isDeleting = true;
          setTimeout(typeLoop, pauseAfterType);
          return;
        }
        setTimeout(typeLoop, typeSpeed);
      } else {
        el.textContent = currentText.substring(0, charIndex - 1);
        charIndex--;
        if (charIndex === 0) {
          isDeleting = false;
          titleIndex = (titleIndex + 1) % titles.length;
          setTimeout(typeLoop, pauseAfterDelete);
          return;
        }
        setTimeout(typeLoop, deleteSpeed);
      }
    }
    typeLoop();
  }

  // ── "More" nav dropdown ───────────────────────────────────
  // CSS opens the menu on :hover and :focus-within; this adds click/tap
  // toggling (the only thing that works on touch) and Escape to close.
  (function () {
    var dropdown = document.querySelector('.nav-dropdown');
    var toggle = dropdown && dropdown.querySelector('.nav-dropdown-toggle');
    if (!dropdown || !toggle) return;

    function setOpen(open) {
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    }

    toggle.addEventListener('click', function (e) {
      e.preventDefault();
      setOpen(toggle.getAttribute('aria-expanded') !== 'true');
    });

    // A mouse user who clicked leaves aria-expanded set; clear it on the way out
    // so hover stays in charge.
    dropdown.addEventListener('mouseleave', function () { setOpen(false); });

    dropdown.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') {
        setOpen(false);
        toggle.focus();
      }
    });

    document.addEventListener('click', function (e) {
      if (!dropdown.contains(e.target)) setOpen(false);
    });
  })();

});

function topFunction() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
