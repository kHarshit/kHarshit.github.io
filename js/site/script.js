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

});

function topFunction() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
