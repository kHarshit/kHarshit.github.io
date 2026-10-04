// stats.js — stat strip in the homepage hero.
// The figures render final in the HTML; this only counts them up from zero the
// first time the strip is on screen. Skipped for reduced-motion visitors and
// browsers without IntersectionObserver.
(function () {
    var strip = document.querySelector('.hero-stats');
    if (!strip || !('IntersectionObserver' in window)) return;
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    var values = strip.querySelectorAll('.hero-stat-value');

    function countUp(el) {
        var target = parseInt(el.getAttribute('data-count'), 10) || 0;
        var suffix = el.getAttribute('data-suffix') || '';
        var duration = 1100;
        var start = null;
        function frame(ts) {
            if (start === null) start = ts;
            var t = Math.min((ts - start) / duration, 1);
            var eased = 1 - Math.pow(1 - t, 3);
            el.textContent = Math.round(target * eased) + suffix;
            if (t < 1) requestAnimationFrame(frame);
        }
        requestAnimationFrame(frame);
    }

    // Hold each figure at its final width so the strip doesn't shift as
    // "8" becomes "84", then reset to zero ready to count.
    for (var i = 0; i < values.length; i++) {
        values[i].style.minWidth = values[i].getBoundingClientRect().width + 'px';
        values[i].style.textAlign = 'center';
        values[i].textContent = '0' + (values[i].getAttribute('data-suffix') || '');
    }

    var observer = new IntersectionObserver(function (entries) {
        if (!entries[0].isIntersecting) return;
        observer.disconnect();
        for (var j = 0; j < values.length; j++) countUp(values[j]);
    }, { threshold: 0.5 });
    observer.observe(strip);
})();
