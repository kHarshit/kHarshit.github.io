(function() {
  var activeSound = 'off';
  var audioEls = {};

  var soundscapeToggle = document.getElementById('poem-soundscape-toggle');
  var soundscapePanel = document.getElementById('poem-soundscape-panel');

  if (!soundscapeToggle) return;

  var sources = {
    rain: '/audio/rain.mp3',
    ocean: '/audio/ocean.mp3',
    wind: '/audio/wind.mp3'
  };

  function preload(sound) {
    if (!audioEls[sound]) {
      audioEls[sound] = new Audio(sources[sound]);
      audioEls[sound].loop = true;
      audioEls[sound].volume = 0.4;
    }
  }

  function stopCurrent() {
    for (var s in audioEls) {
      if (audioEls[s]) { try { audioEls[s].pause(); } catch(e) {} try { audioEls[s].currentTime = 0; } catch(e) {} }
    }
  }

  function playAudio(sound) {
    stopCurrent();
    if (sound === 'off') { activeSound = 'off'; return; }
    preload(sound);
    audioEls[sound].play().catch(function() {});
    activeSound = sound;
  }

  function setActive(sound) {
    var opts = soundscapePanel.querySelectorAll('.poem-soundscape-option');
    opts.forEach(function(opt) {
      opt.classList.toggle('active', opt.getAttribute('data-sound') === sound);
    });
    soundscapeToggle.classList.toggle('active', sound !== 'off');
    playAudio(sound);
    try { localStorage.setItem('poem-soundscape', sound); } catch(e) {}
  }

  // Close panel when clicking outside
  document.addEventListener('click', function(e) {
    if (soundscapePanel.classList.contains('open') &&
        !soundscapePanel.contains(e.target) &&
        e.target !== soundscapeToggle &&
        !soundscapeToggle.contains(e.target)) {
      soundscapePanel.classList.remove('open');
    }
  });

  // Toggle: if sound is playing, turn it off; otherwise open panel
  soundscapeToggle.addEventListener('click', function(e) {
    e.stopPropagation();
    if (activeSound !== 'off') {
      setActive('off');
      soundscapePanel.classList.remove('open');
    } else {
      soundscapePanel.classList.toggle('open');
    }
  });

  // Close when cursor leaves the panel
  soundscapePanel.addEventListener('mouseleave', function() {
    soundscapePanel.classList.remove('open');
  });

  soundscapePanel.addEventListener('click', function(e) {
    e.stopPropagation();
    var opt = e.target.closest('.poem-soundscape-option');
    if (!opt) return;
    var sound = opt.getAttribute('data-sound');
    setActive(sound);
  });

  // Restore saved preference
  try {
    var saved = localStorage.getItem('poem-soundscape');
    if (saved) {
      setActive(saved);
    }
  } catch(e) {}

  // Stop on page unload
  window.addEventListener('pagehide', function() {
    stopCurrent();
  });
})();
