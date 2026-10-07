/*
 * Scene for "Stopping by Woods on a Snowy Evening": a moonlit walk past a
 * village and a frozen lake into deep pine woods. Loaded by
 * _layouts/poem.html for poems with `immersive: snowy-woods`; the engine is
 * js/site/immersive/engine.js.
 */
(function () {
  'use strict';

  var PI = window.PoemImmersive;
  if (!PI) return;
  var rng = PI.util.rng, S = PI.shapes;
  var ridgePath = S.ridgePath, treeRow = S.treeRow, groundPath = S.groundPath, vgrad = S.vgrad;

  // Sleigh bells for "He gives his harness bells a shake".
  function jingle(ac, out) {
    var now = ac.currentTime;
    for (var i = 0; i < 9; i++) {
      var t = now + i * 0.065 + Math.random() * 0.035;
      var f = 2300 + Math.random() * 1100;
      [1, 2.76, 5.4].forEach(function (m, j) {
        var o = ac.createOscillator(), g = ac.createGain();
        o.type = 'sine';
        o.frequency.value = f * m;
        g.gain.setValueAtTime(0.0001, t);
        g.gain.exponentialRampToValueAtTime(0.07 / (j + 1), t + 0.004);
        g.gain.exponentialRampToValueAtTime(0.0001, t + 0.7 / (j + 1));
        o.connect(g);
        g.connect(out);
        o.start(t);
        o.stop(t + 0.75);
      });
    }
  }

  PI.register('snowy-woods', {
    align: ['left', 'right', 'left', 'center'],
    // [unit, camera depth, darkness, snowfall, wind]
    keys: [
      [0.0, 0, 0.00, 0.45, 0.10],
      [0.7, 40, 0.00, 0.50, 0.10],
      [1.3, 400, 0.00, 0.60, 0.12],
      [2.3, 470, 0.04, 0.95, 0.12],   // "...fill up with snow"
      [2.9, 950, 0.08, 0.70, 0.10],
      [3.9, 1020, 0.12, 0.70, 0.10],  // by the frozen lake
      [4.5, 1250, 0.14, 0.80, 0.35],
      [5.0, 1300, 0.15, 1.00, 1.00],  // "the sweep of easy wind"
      [5.5, 1350, 0.20, 0.90, 0.50],
      [6.1, 1650, 0.35, 0.80, 0.20],
      [7.1, 1800, 0.55, 0.70, 0.12],  // "lovely, dark and deep"
      [8.6, 1950, 0.90, 0.50, 0.05]
    ],
    sound: {
      src: '/audio/wind.mp3',
      label: 'Play wind and sleigh bells',
      cues: [{ stanza: 2, at: 0.35, play: jingle }]
    },
    stars: 220,
    sky: ['#03060f', '#0b1530', '#22335c', '#3b4d78'],
    moon: { x: 0.74, y: 0.19 },

    build: function (ctx) {
      var r = rng(11), L = [];

      var far = ridgePath(r, 705, 300, 1);
      L.push({ z: 6000, items: [
        { path: far.fill, fill: vgrad(ctx, 405, 720, ['#7487b0', '#4b5c86', '#36466e']) },
        { path: far.line, stroke: 'rgba(200,214,240,0.35)', width: 3 }
      ] });

      var near = ridgePath(r, 722, 170, 1.4);
      L.push({ z: 4200, items: [
        { path: near.fill, fill: vgrad(ctx, 550, 730, ['#56688f', '#2d3b60']) },
        { path: near.line, stroke: 'rgba(190,205,235,0.3)', width: 2.5 }
      ] });

      // Village on the low hills: "His house is in the village though".
      var hills = ridgePath(r, 742, 40, 0.4);
      var houses = [], windows = [];
      [[730, 1], [770, 0.8], [815, 1.15], [860, 0.75], [905, 0.9]].forEach(function (hs, i) {
        var x = hs[0], s = hs[1] * 12, y = 712 + (i % 2) * 4;
        houses.push('M' + (x - s) + ',' + y + 'v' + (-s * 0.9) + 'l' + s + ',' + (-s * 0.8) + 'l' + s + ',' + (s * 0.8) + 'v' + (s * 0.9) + 'Z');
        windows.push([x - s * 0.35, y - s * 0.55, s * 0.4]);
      });
      L.push({ z: 3000, items: [
        { path: hills.fill, fill: vgrad(ctx, 700, 800, ['#6c7fa8', '#50628c']) },
        { path: new Path2D(houses.join('')), fill: '#1b2440' }
      ], glow: windows });

      L.push({ z: 2600, items: [], fog: { cy: 730, rx: 1300, ry: 40, a: 0.16, drift: 6 } });

      // The deep woods the poem ends in. Only a narrow path through them.
      L.push({ z: 2300, items: treeRow(ctx, r, {
        base: 744, h: [60, 125], gap: [9, 20], clear: 55, jitter: 6,
        color: '#141c33', snow: 'rgba(150,168,208,0.55)',
        ground: ['#7c8fb6', '#5e709a'], wave: 3
      }) });

      L.push({ z: 1950, items: [], fog: { cy: 760, rx: 1500, ry: 55, a: 0.14, drift: 10 } });

      // Frozen lake with the moon on the ice, trees along both shores.
      var lake = new Path2D();
      lake.ellipse(1060, 806, 620, 46, 0, 0, Math.PI * 2);
      var shine = new Path2D();
      shine.ellipse(1330, 800, 150, 9, 0, 0, Math.PI * 2);
      var lakeItems = [
        { path: groundPath(r, 762, 3), fill: vgrad(ctx, 760, 1100, ['#8a9cc4', '#6d80a9']) },
        { path: lake, fill: vgrad(ctx, 760, 852, ['#40548a', '#7287b4', '#5a6f9e']) },
        { path: lake, stroke: 'rgba(20,28,50,0.35)', width: 2 },
        { path: shine, fill: 'rgba(222,232,255,0.55)' }
      ];
      L.push({ z: 1500, items: lakeItems.concat(treeRow(ctx, r, {
        base: 768, h: [110, 210], gap: [16, 34], clear: 640, jitter: 8,
        color: '#10182b', snow: 'rgba(170,188,225,0.6)'
      })) });

      L.push({ z: 1350, items: [], fog: { cy: 780, rx: 1400, ry: 70, a: 0.12, drift: -8 } });

      L.push({ z: 1150, items: treeRow(ctx, r, {
        base: 792, h: [170, 300], gap: [30, 60], clear: 520, jitter: 10,
        color: '#0d1426', snow: 'rgba(185,200,235,0.65)',
        ground: ['#93a5cb', '#7a8db6'], wave: 5
      }) });

      L.push({ z: 800, items: treeRow(ctx, r, {
        base: 842, h: [280, 470], gap: [45, 90], clear: 470, jitter: 14,
        color: '#0a1020', snow: 'rgba(195,210,240,0.72)',
        ground: ['#a1b2d5', '#8293bb'], wave: 7
      }) });

      L.push({ z: 600, items: [], fog: { cy: 860, rx: 1500, ry: 110, a: 0.10, drift: 14 } });

      L.push({ z: 330, items: treeRow(ctx, r, {
        base: 965, h: [540, 820], gap: [90, 170], clear: 600, jitter: 20,
        color: '#070c18', snow: 'rgba(200,214,242,0.75)',
        ground: ['#adbddf', '#8b9cc3'], wave: 10
      }) });

      // Foreground framing: a few huge dark trunks right at the reader.
      L.push({ z: 170, items: treeRow(ctx, r, {
        base: 1190, h: [1000, 1450], gap: [150, 260], clear: 780, jitter: 30,
        color: '#03060d', snow: 'rgba(190,205,236,0.6)',
        ground: ['#b4c3e3', '#94a5cb'], wave: 14
      }) });

      L.sort(function (a, b) { return b.z - a.z; });
      return L;
    }
  });
})();
