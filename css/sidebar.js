document.querySelectorAll('.floating-sidebar a').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Active state: highlight the sidebar link for the section currently in view
const sectionIds = ['showcase', 'experience', 'education', 'skills', 'projects'];
const sidebarLinks = document.querySelectorAll('.floating-sidebar .sidebar-link');

function updateActive() {
    let current = sectionIds[0];
    sectionIds.forEach(id => {
        const el = document.getElementById(id);
        if (el && window.scrollY >= el.offsetTop - 160) {
            current = id;
        }
    });
    sidebarLinks.forEach(link => {
        const href = link.getAttribute('href').replace('#', '');
        link.classList.toggle('active', href === current);
    });
}

window.addEventListener('scroll', updateActive, { passive: true });
updateActive();
