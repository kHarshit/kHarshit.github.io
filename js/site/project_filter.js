// project-filter.js
(function() {
    function filterProjects(category) {
        var items = document.querySelectorAll('#projects-list .project-list-item');
        for (var i = 0; i < items.length; i++) {
            var cats = items[i].dataset.category || '';
            items[i].style.display = (category === 'all' || cats.indexOf(category) > -1) ? '' : 'none';
        }
        var buttons = document.querySelectorAll('.project-filters .filter-btn');
        for (var i = 0; i < buttons.length; i++) {
            buttons[i].classList.toggle('active', buttons[i].dataset.filter === category);
        }
    }

    // Append "N" to each chip, using the same match as filterProjects so the
    // number always equals what the chip shows when clicked.
    function addCounts() {
        var items = document.querySelectorAll('#projects-list .project-list-item');
        var buttons = document.querySelectorAll('.project-filters .filter-btn');
        for (var i = 0; i < buttons.length; i++) {
            var f = buttons[i].dataset.filter, n = 0;
            for (var j = 0; j < items.length; j++) {
                if (f === 'all' || (items[j].dataset.category || '').indexOf(f) > -1) n++;
            }
            var badge = document.createElement('span');
            badge.className = 'filter-count';
            badge.textContent = n;
            buttons[i].appendChild(badge);
        }
    }

    document.addEventListener('DOMContentLoaded', function() {
        addCounts();
        var filters = document.querySelector('.project-filters');
        if (filters) {
            filters.addEventListener('click', function(e) {
                var btn = e.target.closest('.filter-btn');
                if (btn && btn.dataset.filter) {
                    filterProjects(btn.dataset.filter);
                }
            });
        }
        filterProjects('all');
    });
})();
