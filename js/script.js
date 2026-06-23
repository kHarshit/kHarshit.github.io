// Show/hide scroll-to-top button — appears after scrolling 400px
window.addEventListener('scroll', function() {
    var btn = document.getElementById('scroll_top');
    if (!btn) return;
    var scrolled = document.body.scrollTop || document.documentElement.scrollTop;
    btn.style.display = scrolled > 600 ? 'block' : 'none';
}, { passive: true });

// Scroll to top on button click
function topFunction() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
