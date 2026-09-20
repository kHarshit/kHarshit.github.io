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

    document.addEventListener('DOMContentLoaded', function() {
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
