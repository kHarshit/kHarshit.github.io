/*
 * Scene for "Stopping by Woods on a Snowy Evening" made of five picture
 * layers (mountains, far, mid, near, foreground) in
 * img/poems/immersive/snowy-woods/, rendered from the three.js scene by
 * scripts/immersive/capture-layers.js. The engine draws the sky, stars,
 * moon and snowfall around them.
 *
 * Pictures blur when magnified, so the camera only creeps forward (the
 * foreground grows to about 2.3x by the end) and the depth comes mostly from
 * parallax between the layers. Selected with `?scene=snowy-woods-layers`.
 */
(function () {
  'use strict';

  var PI = window.PoemImmersive;
  if (!PI) return;

  PI.register('snowy-woods-layers', PI.imageScene('/img/poems/immersive/snowy-woods/', {
    align: ['right', 'right', 'left', 'center'],
    // [unit, camera depth, darkness, snowfall, wind]
    keys: [
      [0.0, 0, 0.00, 0.45, 0.10],
      [1.3, 20, 0.00, 0.60, 0.12],
      [2.3, 30, 0.04, 0.95, 0.12],   // "...fill up with snow"
      [2.9, 45, 0.08, 0.70, 0.10],
      [3.9, 52, 0.12, 0.70, 0.10],
      [4.5, 62, 0.14, 0.80, 0.35],
      [5.0, 66, 0.15, 1.00, 1.00],   // "the sweep of easy wind"
      [5.5, 70, 0.20, 0.90, 0.50],
      [6.1, 80, 0.35, 0.80, 0.20],
      [7.1, 90, 0.55, 0.70, 0.12],   // "lovely, dark and deep"
      [8.6, 100, 0.90, 0.50, 0.05]
    ],
    sound: {
      src: '/audio/wind.mp3',
      label: 'Play wind and sleigh bells',
      cues: [{ stanza: 2, at: 0.35, play: PI.sounds.sleighBells }]
    },
    // Matches the three.js sky so the layers sit in the same night.
    sky: ['#02040b', '#0a1430', '#1a2a50', '#25365c'],
    stars: 220,
    moon: { x: 0.74, y: 0.27 }
  }));
})();
