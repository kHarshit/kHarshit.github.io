// project-filter.js
function filterProjects(category) {
    var projects = document.getElementsByClassName('project-card');
    for (var i = 0; i < projects.length; i++) {
        // Get the project's categories and split into an array
        var categories = projects[i].dataset.category.split(' ');
        if (category === 'all' || categories.includes(category)) {
            projects[i].style.display = 'block';
        } else {
            projects[i].style.display = 'none';
        }
    }
}

// Initialize to show all projects
window.onload = function() {
    filterProjects('all');
};
