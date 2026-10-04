function filterSelection(category) {
    var items = document.getElementsByClassName('text-category-title');
    var showAll = (category === 'all' || category === '');
    for (var i = 0; i < items.length; i++) {
        if (showAll || items[i].classList.contains(category)) {
            items[i].classList.add('show');
        } else {
            items[i].classList.remove('show');
        }
    }
}

// Active button highlight
var btnContainer = document.getElementById('category-filters');
var btns = btnContainer ? btnContainer.getElementsByClassName('filter-btn') : [];
for (var i = 0; i < btns.length; i++) {
    btns[i].addEventListener('click', function() {
        var current = btnContainer.querySelector('.active');
        if (current) current.classList.remove('active');
        this.classList.add('active');
    });
}

// Append "N" to each chip: how many items that filter shows.
(function() {
    var items = document.getElementsByClassName('text-category-title');
    for (var i = 0; i < btns.length; i++) {
        var m = /filterSelection\('([^']*)'\)/.exec(btns[i].getAttribute('onclick') || '');
        if (!m) continue;
        var cat = m[1], n = 0;
        for (var j = 0; j < items.length; j++) {
            if (cat === 'all' || cat === '' || items[j].classList.contains(cat)) n++;
        }
        var badge = document.createElement('span');
        badge.className = 'filter-count';
        badge.textContent = n;
        btns[i].appendChild(badge);
    }
})();

// On load, apply filter from URL hash if present
(function() {
    var hash = window.location.hash.replace('#', '');
    if (hash) {
        filterSelection(hash);
        for (var i = 0; i < btns.length; i++) {
            if (btns[i].getAttribute('onclick') === "filterSelection('" + hash + "')") {
                var current = btnContainer.querySelector('.active');
                if (current) current.classList.remove('active');
                btns[i].classList.add('active');
                btns[i].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                return;
            }
        }
    }
    filterSelection('all');
})();
