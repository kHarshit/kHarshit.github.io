document.addEventListener('DOMContentLoaded', function () {

  // ── Scroll-to-top button ──────────────────────────────────
  window.addEventListener('scroll', function () {
    var btn = document.getElementById('scroll_top');
    if (!btn) return;
    var scrolled = document.body.scrollTop || document.documentElement.scrollTop;
    btn.style.display = scrolled > 600 ? 'block' : 'none';
  }, { passive: true });

  // ── Typing effect for hero title ──────────────────────────
  var el = document.getElementById('hero-title');
  if (el) {
    var text = 'Machine Learning Engineer';
    var i = 0;
    var speed = 45;
    function type() {
      if (i < text.length) {
        el.textContent += text.charAt(i);
        i++;
        setTimeout(type, speed);
      }
    }
    type();
  }

});

function topFunction() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
