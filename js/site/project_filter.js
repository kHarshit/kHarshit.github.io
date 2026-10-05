// project-filter.js — filter chips on /projects/.
// Every project, featured card or archive row, is a .project-item with a
// space-separated data-category list (from _data/projects.yml).
(function() {
    function categoriesOf(item) {
        return (item.dataset.category || '').split(/\s+/);
    }

    function matches(item, category) {
        return category === 'all' || categoriesOf(item).indexOf(category) > -1;
    }

    function filterProjects(category) {
        var items = document.querySelectorAll('.project-item');
        for (var i = 0; i < items.length; i++) {
            items[i].style.display = matches(items[i], category) ? '' : 'none';
        }

        // Hide a year, or the featured row, once a filter leaves it empty, so
        // no label sits over nothing.
        var groups = document.querySelectorAll('.archive-group, .projects-featured');
        for (var g = 0; g < groups.length; g++) {
            var visible = groups[g].querySelectorAll('.project-item:not([style*="display: none"])').length;
            groups[g].style.display = visible ? '' : 'none';
        }

        var buttons = document.querySelectorAll('.project-filters .filter-btn');
        for (var b = 0; b < buttons.length; b++) {
            buttons[b].classList.toggle('active', buttons[b].dataset.filter === category);
        }
    }

    // Append "N" to each chip, using the same match as filterProjects so the
    // number always equals what the chip shows when clicked.
    function addCounts() {
        var items = document.querySelectorAll('.project-item');
        var buttons = document.querySelectorAll('.project-filters .filter-btn');
        for (var i = 0; i < buttons.length; i++) {
            var f = buttons[i].dataset.filter, n = 0;
            for (var j = 0; j < items.length; j++) {
                if (matches(items[j], f)) n++;
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
