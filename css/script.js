// Show/hide scroll-to-top button
window.addEventListener('scroll', function() {
    var btn = document.getElementById('scroll_top');
    if (!btn) return;
    btn.style.display = (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20)
        ? 'block' : 'none';
}, { passive: true });

// Scroll to top on button click
function topFunction() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
