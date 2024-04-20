document.querySelectorAll('.floating-sidebar a').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault(); // Stop the default jump to section behavior

        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth' // Smoothly scroll to the section
        });
    });
});

// window.onscroll = function() {
//     var sidebar = document.querySelector('.floating-sidebar');
//     if (window.pageYOffset > 400) { // Only show the sidebar after scrolling down 300px
//         sidebar.style.display = 'block';
//     } else {
//         sidebar.style.display = 'none';
//     }
// };