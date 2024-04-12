// project-filter.js
function filterProjects(category) {
    var projects = document.getElementsByClassName('project-card');
    var buttons = document.getElementsByClassName('filter-btn');
    
    // Remove the 'active' class from all buttons
    for (var i = 0; i < buttons.length; i++) {
        buttons[i].classList.remove('active');
    }
    
    // Add the 'active' class to the currently clicked button
    // The 'this' keyword represents the clicked button
    // Make sure to pass 'this' from the onclick attribute in the HTML
    document.querySelector('.filter-btn.' + category).classList.add('active');
    
    for (var i = 0; i < projects.length; i++) {
        var categories = projects[i].dataset.category.split(' ');
        if (category === 'all' || categories.includes(category)) {
            projects[i].style.display = 'block';
        } else {
            projects[i].style.display = 'none';
        }
    }
}

// Add a load event listener
window.addEventListener('load', function() {
    filterProjects('all');
});
