/*
 * Scene for "Perhaps" (Poem_for_your_sprog): a child's bedroom window at
 * night, and the dreams that rise out of it.
 *
 * I    (two parts) The casement swings open on a starry sky over the
 *      rooftops. From a jar of light on the sill each dream flies out as a
 *      spark and draws itself among the stars as a constellation: a sailing
 *      ship; a queen's crown with needle and thread; a dancer's spotlight;
 *      a flask for the cure; a theatre curtain for the play.
 * II   (two parts) A quill and page; a fire engine's ladder; the scales of
 *      justice; a star destroyer; a flag on a summit, every test conquered.
 * III  A heart; a balloon to journey and discover; a house with its
 *      windows lit; and tomorrow, rising.
 * IV   "He smiled and looked ahead": leaning out, the whole sky of
 *      possibilities hangs there, linked and glowing.
 * V    "He softly sighed with sorrow": the dreams drift away and dim.
 * VI   "... or maybe not," he said: the jar goes out, the town goes dark,
 *      and back in the room the window holds a single star.
 *
 * Each dream is born at its line from the scroll position itself (the keys
 * function hands over the timeline), so fourteen births don't put fourteen
 * stops in the camera. Columns:
 *   [unit, camZ, dim, (unused), breeze, yaw, pitch, open, hang, away, out]
 */
import { THREE, isSmall, makeRenderer, fitCamera, softSprite, skyDome, terrain, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout (metres; the window wall is z = 0, the room is z > 0) ─────────
var WIN = { x: 0.75, y0: 0.95, y1: 2.45 }, ROOM = { w: 2.7, h: 2.75, d: 4.8 }, WALL_T = 0.3, EYE = 1.42;
var JAR = new THREE.Vector3(0.36, WIN.y0, -0.1);
var STREET = -10;                     // three storeys up, above the roofs
var TL = null;                        // the timeline, kept by keys()

// ── The dreams, as constellations drawn in a unit box ────────────────────
function P(pts, nodes) { return { pts: pts, nodes: nodes !== false }; }
function C(pts, n, closed) {
  var curve = new THREE.CatmullRomCurve3(pts.map(function (p) { return new THREE.Vector3(p[0], p[1], 0); }), !!closed);
  return { pts: curve.getPoints(n).map(function (v) { return [v.x, v.y]; }), nodes: false };
}
function arc(cx, cy, rx, ry, a0, a1, n) {
  var pts = [];
  for (var i = 0; i <= n; i++) { var a = a0 + (a1 - a0) * i / n; pts.push([cx + Math.cos(a) * rx, cy + Math.sin(a) * ry]); }
  return { pts: pts, nodes: false };
}
function ring(cx, cy, r) { return arc(cx, cy, r, r, 0, Math.PI * 2, 14); }

var SHAPES = {
  ship: function () {
    return { strokes: [
      P([[-1.0, -0.25], [1.05, -0.25], [0.72, -0.6], [-0.78, -0.6], [-1.0, -0.25]]),
      P([[0, -0.25], [0, 1.1]]),
      P([[0.07, 1.0], [0.07, -0.1], [0.85, -0.1], [0.07, 1.0]]),
      P([[-0.07, 0.86], [-0.07, -0.1], [-0.72, -0.1], [-0.07, 0.86]]),
      P([[0, 1.1], [0.34, 1.0], [0, 0.9]]),
      C([[-1.35, -0.8], [-1.0, -0.72], [-0.6, -0.82], [-0.2, -0.72], [0.2, -0.82], [0.6, -0.72], [1.0, -0.82], [1.35, -0.74]], 40)
    ], lights: [[0, 1.1, 1.3]] };
  },
  crown: function () {
    return { strokes: [
      P([[-0.75, -0.35], [-0.88, 0.45], [-0.42, 0.06], [0, 0.62], [0.42, 0.06], [0.88, 0.45], [0.75, -0.35], [-0.75, -0.35]]),
      P([[-0.78, -0.13], [0.78, -0.13]], false),
      P([[0.62, -1.0], [1.3, 0.22]]),
      C([[1.25, 0.12], [1.42, -0.3], [1.05, -0.62], [0.45, -0.58], [-0.1, -0.5], [-0.6, -0.66], [-1.05, -0.52], [-1.3, -0.7]], 46)
    ], lights: [[-0.88, 0.45, 1.2], [0, 0.62, 1.5], [0.88, 0.45, 1.2], [0, 0.22, 1.0]] };
  },
  spotlight: function () {
    var twirl = [];
    for (var i = 0; i <= 40; i++) { var t = i / 40; twirl.push([Math.sin(t * Math.PI * 4.5) * 0.36 * (1 - t * 0.45), -0.6 + t * 1.15]); }
    return { strokes: [
      P([[-0.28, 1.25], [0.28, 1.25]]),
      P([[-0.2, 1.2], [-0.95, -0.66]], false),
      P([[0.2, 1.2], [0.95, -0.66]], false),
      arc(0, -0.7, 0.98, 0.2, 0, Math.PI * 2, 30),
      { pts: twirl, nodes: false }
    ], lights: [[0, 1.3, 1.4], [0.3, 0.45, 0.9], [-0.25, 0.1, 0.8]] };
  },
  flask: function () {
    return { strokes: [
      P([[-0.33, 1.0], [0.33, 1.0]]),
      P([[-0.22, 1.0], [-0.22, 0.3], [-0.82, -0.78], [0.82, -0.78], [0.22, 0.3], [0.22, 1.0]]),
      C([[-0.57, -0.3], [-0.25, -0.22], [0.1, -0.32], [0.57, -0.24]], 16),
      ring(0.05, 0.02, 0.07), ring(-0.08, 0.42, 0.06), ring(0.1, 0.78, 0.07), ring(-0.05, 1.28, 0.09)
    ], lights: [[0, 1.6, 1.6]] };
  },
  curtain: function () {
    var strokes = [P([[-1.15, 1.0], [1.15, 1.0]]), P([[-1.25, -0.92], [1.25, -0.92]])];
    for (var k = 0; k < 4; k++) strokes.push(arc(-0.86 + k * 0.575, 1.0, 0.29, 0.16, Math.PI, Math.PI * 2, 10));
    strokes.push(C([[-1.05, 0.95], [-1.0, 0.2], [-0.82, -0.45], [-1.0, -0.92]], 18));
    strokes.push(C([[-0.38, 0.95], [-0.58, 0.3], [-0.8, -0.38]], 14));
    strokes.push(C([[1.05, 0.95], [1.0, 0.2], [0.82, -0.45], [1.0, -0.92]], 18));
    strokes.push(C([[0.38, 0.95], [0.58, 0.3], [0.8, -0.38]], 14));
    return { strokes: strokes, lights: [[-0.8, -0.38, 1.1], [0.8, -0.38, 1.1], [0, -0.35, 1.6]] };
  },
  quill: function () {
    return { strokes: [
      P([[-0.9, -0.85], [0.3, -0.85], [0.3, 0.55], [-0.9, 0.55], [-0.9, -0.85]]),
      P([[-0.75, 0.32], [0.12, 0.32]], false), P([[-0.75, 0.1], [0.05, 0.1]], false),
      P([[-0.75, -0.12], [0.15, -0.12]], false), P([[-0.75, -0.34], [-0.15, -0.34]], false),
      P([[0.02, -0.5], [1.05, 1.15]], false),
      C([[0.22, -0.12], [0.62, 0.22], [0.92, 0.7], [1.05, 1.15], [0.62, 0.86], [0.36, 0.42], [0.22, -0.12]], 34)
    ], lights: [[0.02, -0.5, 1.2], [1.05, 1.15, 1.0]] };
  },
  ladder: function () {
    var strokes = [
      P([[-1.25, -0.8], [1.05, -0.8], [1.05, -0.48], [0.88, -0.28], [0.5, -0.28], [0.5, -0.48], [-1.25, -0.48], [-1.25, -0.8]]),
      ring(-0.82, -0.86, 0.15), ring(0.7, -0.86, 0.15)
    ];
    var a = [-0.95, -0.48], d = [0.62, 0.78], n = [-0.78 * 0.17, 0.62 * 0.17], L = 2.0;
    strokes.push(P([[a[0], a[1]], [a[0] + d[0] * L, a[1] + d[1] * L]]));
    strokes.push(P([[a[0] - n[0], a[1] - n[1]], [a[0] - n[0] + d[0] * L, a[1] - n[1] + d[1] * L]]));
    for (var k = 1; k < 10; k++) {
      var t = k / 10 * L;
      strokes.push(P([[a[0] + d[0] * t, a[1] + d[1] * t], [a[0] - n[0] + d[0] * t, a[1] - n[1] + d[1] * t]], false));
    }
    return { strokes: strokes, lights: [[0.95, -0.25, 1.3], [a[0] + d[0] * L, a[1] + d[1] * L, 1.1]] };
  },
  scales: function () {
    var strokes = [
      P([[0, -0.85], [0, 0.85]]), P([[-0.45, -0.85], [0.45, -0.85]]), P([[-0.95, 0.62], [0.95, 0.62]])
    ];
    [-1, 1].forEach(function (s) {
      strokes.push(P([[s * 0.95, 0.62], [s * 1.2, -0.05]], false));
      strokes.push(P([[s * 0.95, 0.62], [s * 0.7, -0.05]], false));
      strokes.push(arc(s * 0.95, -0.05, 0.3, 0.16, Math.PI, Math.PI * 2, 12));
    });
    return { strokes: strokes, lights: [[0, 0.95, 1.4]] };
  },
  destroyer: function () {
    return { strokes: [
      P([[-1.4, 0.0], [1.05, 0.44], [1.05, -0.44], [-1.4, 0.0]]),
      P([[-1.0, 0.0], [1.05, 0.0]], false),
      P([[0.3, 0.12], [0.45, 0.22], [0.9, 0.22], [0.9, 0.12]]),
      P([[0.58, 0.22], [0.58, 0.34], [0.8, 0.34], [0.8, 0.22]])
    ], lights: [[1.1, 0.24, 1.2], [1.1, 0.0, 1.4], [1.1, -0.24, 1.2]] };
  },
  summit: function () {
    return { strokes: [
      P([[-1.2, -0.75], [-0.45, 0.35], [-0.2, 0.1], [0.3, 0.85], [1.2, -0.75], [-1.2, -0.75]]),
      P([[0.05, 0.45], [0.2, 0.36], [0.32, 0.5], [0.46, 0.4], [0.55, 0.55]], false),
      P([[0.3, 0.85], [0.3, 1.38]]),
      P([[0.3, 1.38], [0.72, 1.24], [0.3, 1.1]])
    ], lights: [[0.3, 1.38, 1.3]] };
  },
  heart: function () {
    var pts = [];
    for (var i = 0; i <= 48; i++) {
      var t = i / 48 * Math.PI * 2, x = 16 * Math.pow(Math.sin(t), 3);
      var y = 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t);
      pts.push([x / 16, y / 16 + 0.1]);
    }
    return { strokes: [{ pts: pts, nodes: false }], lights: [[0, 0.9, 0.9], [0, -0.95, 1.2]] };
  },
  balloon: function () {
    return { strokes: [
      C([[0, -0.35], [-0.45, 0.05], [-0.68, 0.5], [-0.55, 0.95], [0, 1.15], [0.55, 0.95], [0.68, 0.5], [0.45, 0.05]], 44, true),
      C([[0, 1.15], [-0.28, 0.6], [-0.16, 0.0]], 12), C([[0, 1.15], [0.28, 0.6], [0.16, 0.0]], 12),
      P([[-0.2, -0.22], [-0.17, -0.65]], false), P([[0.2, -0.22], [0.17, -0.65]], false),
      P([[-0.21, -0.65], [0.21, -0.65], [0.17, -0.92], [-0.17, -0.92], [-0.21, -0.65]])
    ], lights: [[0, -0.42, 1.3]] };
  },
  house: function () {
    return { strokes: [
      P([[-0.75, -0.8], [0.75, -0.8], [0.75, 0.15], [-0.75, 0.15], [-0.75, -0.8]]),
      P([[-0.95, 0.1], [0, 0.9], [0.95, 0.1]]),
      P([[0.42, 0.45], [0.42, 0.82], [0.62, 0.82], [0.62, 0.3]]),
      P([[-0.14, -0.8], [-0.14, -0.36], [0.14, -0.36], [0.14, -0.8]]),
      ring(0.6, 1.0, 0.06), ring(0.7, 1.18, 0.08)
    ], lights: [], fills: [[-0.43, -0.22, 0.28, 0.26], [0.43, -0.22, 0.28, 0.26], [0, 0.42, 0.2, 0.16]] };
  },
  tomorrow: function () {
    var strokes = [P([[-1.3, -0.3], [1.3, -0.3]], false), arc(0, -0.3, 0.58, 0.58, 0, Math.PI, 22)];
    for (var k = 0; k < 7; k++) {
      var a = (15 + k * 25) * Math.PI / 180;
      strokes.push(P([[Math.cos(a) * 0.75, -0.3 + Math.sin(a) * 0.75], [Math.cos(a) * 1.1, -0.3 + Math.sin(a) * 1.1]], false));
    }
    return { strokes: strokes, lights: [] };
  }
};

// Which panel and when in it (fraction of the panel's 1.6 units) each dream
// is born; its place in the sky, its size and its colour. Panels: 0-1 the
// first stanza, 2-3 the second, 4 the third, 5-7 the three single lines.
var DREAMS = [
  // az/el: degrees from straight out of the window; d: distance
  { shape: 'ship',      panel: 0, at: 0.36, az: 8,   el: 8,  d: 60, size: 6.0, tint: '#ffe2a8' },
  { shape: 'crown',     panel: 0, at: 0.86, az: 18,  el: 19, d: 64, size: 5.0, tint: '#ffd27a' },
  { shape: 'spotlight', panel: 1, at: 0.36, az: -12, el: 7,  d: 60, size: 6.0, tint: '#fff0d6' },
  { shape: 'flask',     panel: 1, at: 0.74, az: -27, el: 17, d: 62, size: 5.0, tint: '#cfe8ff' },
  { shape: 'curtain',   panel: 1, at: 0.98, az: -15, el: 23, d: 66, size: 6.0, tint: '#ffc2b0' },
  { shape: 'quill',     panel: 2, at: 0.36, az: 25,  el: 7,  d: 62, size: 5.6, tint: '#ffe8c0' },
  { shape: 'ladder',    panel: 2, at: 0.86, az: 33,  el: 19, d: 64, size: 6.0, tint: '#ffb98a' },
  { shape: 'scales',    panel: 3, at: 0.36, az: -33, el: 5,  d: 62, size: 6.0, tint: '#ffe2a8' },
  { shape: 'destroyer', panel: 3, at: 0.74, az: -20, el: 35, d: 70, size: 7.0, tint: '#d8e4ff' },
  { shape: 'summit',    panel: 3, at: 0.98, az: -41, el: 21, d: 64, size: 5.6, tint: '#fff0d6' },
  { shape: 'heart',     panel: 4, at: 0.3,  az: 11,  el: 34, d: 66, size: 5.0, tint: '#ffb0b8' },
  { shape: 'balloon',   panel: 4, at: 0.56, az: 29,  el: 36, d: 68, size: 5.6, tint: '#ffd27a' },
  { shape: 'house',     panel: 4, at: 0.78, az: 42,  el: 8,  d: 62, size: 5.6, tint: '#ffe2a8' },
  { shape: 'tomorrow',  panel: 4, at: 1.0,  az: 0,   el: 25, d: 76, size: 6.0, tint: '#ffd9a0' }
];
// A place in the sky, seen from the window. Phones squeeze the azimuth and
// part the dreams above and below the centred verse (`keep` stays put).
function skyAt(az, el, d, sx, out, keep) {
  if (sx < 1 && !keep) el = el < 16 ? 3 + (el - 7) * 0.4 : 25 + (el - 16);
  var a = az * sx * Math.PI / 180, e = el * Math.PI / 180;
  return out.set(Math.sin(a) * Math.cos(e) * d, EYE + Math.sin(e) * d, -Math.cos(a) * Math.cos(e) * d);
}
var ONE = DREAMS.length - 1;          // tomorrow: its heart is the star that stays

// ── Sound: tender chimes as the dreams appear, a sigh, one low note ──────
function chime(freq, gain) {
  return function (ac, out) {
    var t = ac.currentTime;
    [[1, 1], [2.01, 0.35], [3.02, 0.12]].forEach(function (p) {
      var o = ac.createOscillator(), g = ac.createGain();
      o.type = 'sine';
      o.frequency.value = freq * p[0];
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime((gain || 0.06) * p[1], t + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 2.6 / p[0]);
      o.connect(g); g.connect(out);
      o.start(t); o.stop(t + 2.7);
    });
  };
}
function sigh(ac, out) {
  [784, 659, 587, 494].forEach(function (f, i) { setTimeout(function () { chime(f, 0.045)(ac, out); }, i * 260); });
}
var NOTES = [392, 440, 523.25, 587.33, 659.25, 523.25, 587.33, 659.25, 783.99, 880, 659.25, 783.99, 880, 1046.5];

// ── Renderer ─────────────────────────────────────────────────────────────
function canvasTexture(w, h, paint) {
  var c = document.createElement('canvas');
  c.width = w; c.height = h;
  paint(c.getContext('2d'), w, h);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

var DOT_VS = 'attribute vec3 aDot; uniform float uTime; uniform float uReveal; uniform float uAlpha; uniform float uPx; uniform float uScale;\n' +
  'varying float vA;\n' +
  'void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;\n' +
  ' float on = smoothstep(aDot.y - 0.02, aDot.y + 0.002, uReveal);\n' +
  ' float front = on * smoothstep(0.08, 0.0, uReveal - aDot.y) * step(uReveal, 0.999);\n' +
  ' float tw = 0.72 + 0.28 * sin(uTime * (1.1 + fract(aDot.z) * 1.4) + aDot.z * 6.0);\n' +
  ' vA = uAlpha * on * tw * (1.0 + front * 1.5);\n' +
  ' gl_PointSize = aDot.x * 0.24 * uScale * uPx / -mv.z * (1.0 + front * 1.8); }';
var DOT_FS = 'uniform vec3 uColor; varying float vA;\n' +
  'void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;\n' +
  ' float a = (pow(smoothstep(0.5, 0.0, d), 2.0) * 0.6 + smoothstep(0.16, 0.0, d)) * vA;\n' +
  ' gl_FragColor = vec4(uColor, a);\n #include <colorspace_fragment>\n }';

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(53);
  var gl = makeRenderer(canvas, { clear: '#070a1c' });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#1c2147', 0.0065);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.03, 3000);
  var pxScale = { value: 800 };

  // ── Sky: a deep night with a soft glow over the town ────────────────────
  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#04071a', mid: '#0d1536', horizon: '#2a2c58' }, 1500);
  sky.add(dome.mesh);

  var SN = small ? 2600 : 5200, sPos = [], sAttr = [], v = new THREE.Vector3(), band = new THREE.Euler(1.0, 0.3, 0.45);
  for (var i = 0; i < SN; i++) {
    var milky = i < SN * 0.35, th = r() * Math.PI * 2, y;
    if (milky) { v.set(Math.cos(th), (r() + r() + r() - 1.5) * 0.12, Math.sin(th)).normalize().applyEuler(band); if (v.y < 0.03) continue; }
    else { y = 0.03 + r() * 0.97; v.set(Math.sqrt(1 - y * y) * Math.cos(th), y, Math.sqrt(1 - y * y) * Math.sin(th)); }
    sPos.push(v.x * 1300, v.y * 1300, v.z * 1300);
    sAttr.push(milky ? 0.6 + r() * 0.8 : 0.8 + Math.pow(r(), 3) * 3, r() * 6.28, r());
  }
  var starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.Float32BufferAttribute(sPos, 3));
  starGeo.setAttribute('star', new THREE.Float32BufferAttribute(sAttr, 3));
  var starMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
    uniforms: { uTime: { value: 0 }, uDim: { value: 0 }, uScale: { value: 1 } },
    vertexShader: 'attribute vec3 star; uniform float uTime; uniform float uDim; uniform float uScale; varying float vA;\n' +
      'void main(){ gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);\n' +
      ' float tw = 0.7 + 0.3 * sin(uTime * (1.2 + star.z * 2.0) + star.y);\n' +
      ' float gone = smoothstep(star.z - 0.1, star.z + 0.05, uDim);\n' +
      ' vA = tw * (1.0 - gone * 0.92); gl_PointSize = star.x * uScale; }',
    fragmentShader: 'varying float vA; void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;\n' +
      ' float a = smoothstep(0.5, 0.0, d) * vA; gl_FragColor = vec4(vec3(0.84, 0.88, 1.0) * a, a);\n #include <colorspace_fragment>\n }'
  });
  var stars = new THREE.Points(starGeo, starMat);
  stars.frustumCulled = false;
  sky.add(stars);

  // ── Outside: the town below, dark trees, far hills ──────────────────────
  world.add(new THREE.HemisphereLight('#5a6aa8', '#141020', 1.3));
  var moon = new THREE.DirectionalLight('#aebcff', 0.9);
  moon.position.set(-4, 9, -10);
  moon.target.position.set(0, 0, 3);
  world.add(moon, moon.target);

  var ground = new THREE.Mesh(new THREE.PlaneGeometry(900, 900).rotateX(-Math.PI / 2), new THREE.MeshLambertMaterial({ color: '#0b0d18' }));
  ground.position.set(0, STREET, -200);
  world.add(ground);
  world.add(terrain(2400, small ? 90 : 140, 0, -700, function (x, z) {
    var d = -z - 260, a = Math.atan2(x, 600);
    return STREET + smooth(0, 260, d) * (26 + 14 * Math.sin(a * 9 + 1) + 7 * Math.sin(a * 23)) - 1;
  }, new THREE.MeshLambertMaterial({ color: '#0a0c1a' })));

  // Houses: a box with a gabled roof, instanced, roofs just under the sill.
  var houseGeo = new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0);
  var roofGeo = new THREE.CylinderGeometry(0.577, 0.577, 1.02, 3).rotateZ(Math.PI / 2).rotateX(-Math.PI / 2).scale(1, 0.75, 1).translate(0, 1.2, 0);
  var townMat = new THREE.MeshLambertMaterial({ color: '#141a30' }), roofMat = new THREE.MeshLambertMaterial({ color: '#1d2340' });
  var HN = small ? 160 : 320, houses = new THREE.InstancedMesh(houseGeo, townMat, HN), roofs = new THREE.InstancedMesh(roofGeo, roofMat, HN);
  var m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s3 = new THREE.Vector3(), p3 = new THREE.Vector3(), Y = new THREE.Vector3(0, 1, 0);
  var winPos = [], winAttr = [], hn = 0;
  for (var t = 0; t < HN * 3 && hn < HN; t++) {
    var row = Math.floor(r() * 9), hz = -18 - row * 13 - r() * 4, hx = (r() - 0.5) * (60 + row * 22);
    if (Math.abs(hx) < 5 && row < 2) continue;                    // keep the near view open
    var w = 4 + r() * 4, h = 3.5 + r() * 3 + (row === 0 ? 0 : r() * 2.5), d = 5 + r() * 3, rot = (r() - 0.5) * 0.3;
    q.setFromAxisAngle(Y, rot);
    p3.set(hx, STREET, hz);
    houses.setMatrixAt(hn, m4.compose(p3, q, s3.set(w, h, d)));
    roofs.setMatrixAt(hn, m4.compose(p3, q, s3.set(w, h, d)));
    // Lit windows on the street side; each goes dark at its own moment.
    var nw = 1 + Math.floor(r() * 4);
    for (var k = 0; k < nw; k++) {
      if (r() > 0.7) continue;
      var lx = (r() - 0.5) * w * 0.7, ly = 0.8 + r() * (h - 1.6);
      winPos.push(hx + lx * Math.cos(rot), STREET + ly, hz + d / 2 + 0.05 - lx * Math.sin(rot));
      winAttr.push(0.35 + r() * 0.25, r(), r() * 6.28);
    }
    hn++;
  }
  houses.count = roofs.count = hn;
  world.add(houses, roofs);
  var winGeo = new THREE.BufferGeometry();
  winGeo.setAttribute('position', new THREE.Float32BufferAttribute(winPos, 3));
  winGeo.setAttribute('aWin', new THREE.Float32BufferAttribute(winAttr, 3));
  var winMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uOut: { value: 0 }, uPx: pxScale, uTime: { value: 0 } },
    vertexShader: 'attribute vec3 aWin; uniform float uOut; uniform float uPx; uniform float uTime; varying float vA;\n' +
      'void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;\n' +
      ' vA = smoothstep(aWin.y - 0.04, aWin.y + 0.02, 1.0 - uOut) * (0.85 + 0.15 * sin(uTime * 0.7 + aWin.z));\n' +
      ' gl_PointSize = aWin.x * uPx / -mv.z; }',
    fragmentShader: 'varying float vA; void main(){ vec2 c = abs(gl_PointCoord - 0.5); if (max(c.x * 1.5, c.y) > 0.5) discard;\n' +
      ' float a = vA * smoothstep(0.5, 0.3, max(c.x * 1.5, c.y)); gl_FragColor = vec4(vec3(1.0, 0.72, 0.38), a);\n #include <colorspace_fragment>\n }'
  });
  var winPts = new THREE.Points(winGeo, winMat);
  winPts.frustumCulled = false;
  world.add(winPts);

  var treeGeo = new THREE.IcosahedronGeometry(1, 1);
  var trees = new THREE.InstancedMesh(treeGeo, new THREE.MeshLambertMaterial({ color: '#0e1424' }), small ? 120 : 240), tn = 0;
  for (t = 0; t < 600 && tn < trees.count; t++) {
    var tx = (r() - 0.5) * 220, tz = -10 - r() * 130;
    if (Math.abs(tx) < 6 && tz > -30) continue;
    var ts = 1.6 + r() * 1.8;
    p3.set(tx, STREET + ts * 1.4, tz);
    trees.setMatrixAt(tn++, m4.compose(p3, q.identity(), s3.set(ts, ts * 1.3, ts)));
  }
  trees.count = tn;
  world.add(trees);

  // ── The room: walls round the window, frame, sill, curtains ─────────────
  var wallMat = new THREE.MeshStandardMaterial({ color: '#283052', roughness: 0.95 });
  var room = new THREE.Group();
  function box(w, h, d, x, y, z, mat) {
    var m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
    m.position.set(x, y, z);
    room.add(m);
    return m;
  }
  var sideW = ROOM.w - WIN.x;
  box(sideW, ROOM.h, WALL_T, -WIN.x - sideW / 2, ROOM.h / 2, -WALL_T / 2, wallMat);
  box(sideW, ROOM.h, WALL_T, WIN.x + sideW / 2, ROOM.h / 2, -WALL_T / 2, wallMat);
  box(WIN.x * 2, WIN.y0, WALL_T, 0, WIN.y0 / 2, -WALL_T / 2, wallMat);
  box(WIN.x * 2, ROOM.h - WIN.y1, WALL_T, 0, (ROOM.h + WIN.y1) / 2, -WALL_T / 2, wallMat);
  box(WALL_T, ROOM.h, ROOM.d, -ROOM.w, ROOM.h / 2, ROOM.d / 2, wallMat);
  box(WALL_T, ROOM.h, ROOM.d, ROOM.w, ROOM.h / 2, ROOM.d / 2, wallMat);
  box(ROOM.w * 2, 0.1, ROOM.d, 0, -0.05, ROOM.d / 2, new THREE.MeshStandardMaterial({ color: '#2a2230', roughness: 0.8 }));
  box(ROOM.w * 2, 0.1, ROOM.d, 0, ROOM.h + 0.05, ROOM.d / 2, new THREE.MeshStandardMaterial({ color: '#222844', roughness: 1 }));
  var paint = new THREE.MeshStandardMaterial({ color: '#c9cdd8', roughness: 0.7 });
  box(WIN.x * 2 + 0.5, 0.06, 0.55, 0, WIN.y0 - 0.03, 0.0, paint);                      // sill
  box(0.08, WIN.y1 - WIN.y0, 0.1, -WIN.x + 0.04, (WIN.y0 + WIN.y1) / 2, -0.25, paint);   // frame
  box(0.08, WIN.y1 - WIN.y0, 0.1, WIN.x - 0.04, (WIN.y0 + WIN.y1) / 2, -0.25, paint);
  box(WIN.x * 2, 0.08, 0.1, 0, WIN.y1 - 0.04, -0.25, paint);

  // Casement panes, hinged at the outer edges, swinging out.
  var glass = new THREE.MeshStandardMaterial({ color: '#a8b8e8', transparent: true, opacity: 0.12, roughness: 0.05, metalness: 0.3 });
  var paneW = WIN.x - 0.08, paneH = WIN.y1 - WIN.y0 - 0.08;
  var panes = [-1, 1].map(function (side) {
    var hinge = new THREE.Group(), pane = new THREE.Group();
    [[paneW / 2, paneH - 0.03, paneW, 0.06], [paneW / 2, 0.03, paneW, 0.06], [0.03, paneH / 2, 0.06, paneH], [paneW - 0.03, paneH / 2, 0.06, paneH],
     [paneW / 2, paneH / 2, paneW, 0.035], [paneW / 2, paneH / 2, 0.035, paneH]].forEach(function (b) {
      var m = new THREE.Mesh(new THREE.BoxGeometry(b[2], b[3], 0.05), paint);
      m.position.set(b[0], b[1], 0);
      pane.add(m);
    });
    var g = new THREE.Mesh(new THREE.PlaneGeometry(paneW, paneH), glass);
    g.position.set(paneW / 2, paneH / 2, 0);
    pane.add(g);
    pane.scale.x = -side;                       // left pane grows right, right pane grows left
    hinge.add(pane);
    hinge.position.set(side * (WIN.x - 0.08), WIN.y0 + 0.04, -0.27);
    room.add(hinge);
    return { hinge: hinge, side: side };
  });

  // Curtains printed with little stars and moons, stirring in the breeze.
  var print = canvasTexture(256, 512, function (x, w, h) {
    x.fillStyle = '#6f7aa4';
    x.fillRect(0, 0, w, h);
    x.fillStyle = 'rgba(255,236,190,0.75)';
    for (var k = 0; k < 40; k++) {
      var cx = r() * w, cy = r() * h, s = 4 + r() * 5;
      x.beginPath();
      for (var j = 0; j < 10; j++) { var a = j * Math.PI / 5 - Math.PI / 2, rr = j % 2 ? s * 0.45 : s; x.lineTo(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr); }
      x.fill();
    }
  });
  print.repeat.set(1, 1.2);
  var breeze = { value: 0 }, clock = { value: 0 };
  var curtainMat = new THREE.MeshLambertMaterial({ map: print, side: THREE.DoubleSide, emissive: '#0c0f1e' });
  curtainMat.onBeforeCompile = function (sh) {
    sh.uniforms.uBreeze = breeze;
    sh.uniforms.uClock = clock;
    sh.vertexShader = 'uniform float uBreeze; uniform float uClock;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n float hang = clamp((1.25 - position.y) / 2.5, 0.0, 1.0);\n' +
      ' transformed.z += sin(position.x * 19.0) * 0.045 + uBreeze * hang * hang * (0.16 + 0.08 * sin(uClock * 1.7 + position.x * 3.0)) * (0.7 + 0.3 * sin(uClock * 0.9));\n' +
      ' transformed.x += uBreeze * hang * 0.05 * sin(uClock * 1.3 + position.y);');
  };
  [-1, 1].forEach(function (side) {
    var c = new THREE.Mesh(new THREE.PlaneGeometry(0.95, 2.5, 40, 16), curtainMat);
    c.position.set(side * (WIN.x + 0.32), 1.37, 0.14);
    room.add(c);
  });
  box(WIN.x * 2 + 1.6, 0.04, 0.04, 0, 2.64, 0.14, new THREE.MeshStandardMaterial({ color: '#8a8f9e', roughness: 0.5 }));

  // On the sill: a toy boat, two books and the jar of light.
  var toy = new THREE.MeshStandardMaterial({ color: '#b54a3a', roughness: 0.7 }), sail = new THREE.MeshStandardMaterial({ color: '#e8e2d4', roughness: 0.8 });
  var boat = new THREE.Group();
  var hull = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.04, 0.26, 8, 1, false, 0, Math.PI).rotateZ(Math.PI / 2).rotateX(Math.PI), toy);
  var mast = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.005, 0.24), paint);
  mast.position.y = 0.12;
  var sailGeo = new THREE.BufferGeometry();
  sailGeo.setAttribute('position', new THREE.Float32BufferAttribute([0.005, 0.03, 0, 0.005, 0.23, 0, 0.11, 0.03, 0], 3));
  sailGeo.computeVertexNormals();
  var sailM = new THREE.Mesh(sailGeo, new THREE.MeshStandardMaterial({ color: '#e8e2d4', side: THREE.DoubleSide }));
  boat.add(hull, mast, sailM);
  boat.position.set(-0.42, WIN.y0 + 0.04, 0.02);
  boat.rotation.y = 0.5;
  room.add(boat);
  [['#5a6a9a', 0.02], ['#9a5a6a', 0.05]].forEach(function (b, k) {
    var bk = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.035, 0.16), new THREE.MeshStandardMaterial({ color: b[0], roughness: 0.8 }));
    bk.position.set(-0.85, WIN.y0 + 0.018 + k * 0.036, 0.08);
    bk.rotation.y = 0.2 - k * 0.35;
    room.add(bk);
  });
  var jarProfile = [[0, 0], [0.07, 0], [0.08, 0.02], [0.08, 0.14], [0.065, 0.17], [0.055, 0.18], [0.055, 0.2]].map(function (p) { return new THREE.Vector2(p[0], p[1]); });
  var jar = new THREE.Mesh(new THREE.LatheGeometry(jarProfile, 20), new THREE.MeshStandardMaterial({ color: '#d8e4ff', transparent: true, opacity: 0.28,
    roughness: 0.1, emissive: '#ffb860', emissiveIntensity: 0.25, depthWrite: false }));
  jar.position.copy(JAR);
  var lid = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.03, 16), new THREE.MeshStandardMaterial({ color: '#9a8a6a', roughness: 0.6 }));
  lid.position.set(JAR.x, JAR.y + 0.21, JAR.z);
  var warmTex = softSprite('rgba(255,214,150,1)', 'rgba(255,170,90,0)');
  var jarGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: warmTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
  jarGlow.position.set(JAR.x, JAR.y + 0.09, JAR.z);
  jarGlow.scale.setScalar(0.42);
  var jarHalo = new THREE.Sprite(new THREE.SpriteMaterial({ map: warmTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0.25 }));
  jarHalo.position.copy(jarGlow.position);
  jarHalo.scale.setScalar(1.6);
  var jarLight = new THREE.PointLight('#ffb766', 2.2, 5, 1.4);
  jarLight.position.set(JAR.x, JAR.y + 0.12, JAR.z + 0.05);
  room.add(jar, lid, jarGlow, jarHalo, jarLight);
  world.add(room);

  // The star that stays, at the heart of tomorrow.
  var oneTex = canvasTexture(128, 128, function (x, w) {
    var g = x.createRadialGradient(64, 64, 0, 64, 64, 64);
    g.addColorStop(0, 'rgba(255,255,255,1)'); g.addColorStop(0.06, 'rgba(240,245,255,0.95)');
    g.addColorStop(0.18, 'rgba(190,205,255,0.25)'); g.addColorStop(1, 'rgba(160,180,255,0)');
    x.fillStyle = g; x.fillRect(0, 0, w, w);
    x.globalCompositeOperation = 'lighter';
    [[128, 3], [3, 128]].forEach(function (s) {
      var lg = x.createRadialGradient(64, 64, 0, 64, 64, 64);
      lg.addColorStop(0, 'rgba(230,238,255,0.8)'); lg.addColorStop(1, 'rgba(230,238,255,0)');
      x.fillStyle = lg; x.fillRect(64 - s[0] / 2, 64 - s[1] / 2, s[0], s[1]);
    });
  });
  var oneStar = new THREE.Sprite(new THREE.SpriteMaterial({ map: oneTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
  world.add(oneStar);

  // ── The dreams ──────────────────────────────────────────────────────────
  var dreams = DREAMS.map(function (def, n) {
    var shape = SHAPES[def.shape](), total = 0, acc = 0;
    shape.strokes.forEach(function (s) {
      for (var i = 0; i < s.pts.length - 1; i++) total += Math.hypot(s.pts[i + 1][0] - s.pts[i][0], s.pts[i + 1][1] - s.pts[i][1]);
    });
    var dotPos = [], dotAttr = [], segPos = [], segAlong = [];
    shape.strokes.forEach(function (s) {
      var pts = s.pts;
      for (var i = 0; i < pts.length - 1; i++) {
        var a = pts[i], b = pts[i + 1], len = Math.hypot(b[0] - a[0], b[1] - a[1]), steps = Math.max(1, Math.round(len / 0.075));
        var node = s.nodes || i === 0;
        dotPos.push(a[0], a[1], 0);
        dotAttr.push(node ? 0.5 : 0.2, acc / total, r() * 6.28);
        for (var k = 1; k < steps; k++) {
          dotPos.push(lerp(a[0], b[0], k / steps), lerp(a[1], b[1], k / steps), 0);
          dotAttr.push(0.17 + r() * 0.07, (acc + len * k / steps) / total, r() * 6.28);
        }
        segPos.push(a[0], a[1], 0, b[0], b[1], 0);
        acc += len;
        segAlong.push(acc / total);
      }
      var last = pts[pts.length - 1];
      dotPos.push(last[0], last[1], 0);
      dotAttr.push(0.5, acc / total, r() * 6.28);
    });
    shape.lights.forEach(function (l) { dotPos.push(l[0], l[1], 0.01); dotAttr.push(0.5 * l[2], 0.9 + r() * 0.1, r() * 6.28); });

    var g = new THREE.Group(), color = new THREE.Color(def.tint);
    var dotGeo = new THREE.BufferGeometry();
    dotGeo.setAttribute('position', new THREE.Float32BufferAttribute(dotPos, 3));
    dotGeo.setAttribute('aDot', new THREE.Float32BufferAttribute(dotAttr, 3));
    var dotMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
      uniforms: { uTime: clock, uReveal: { value: 0 }, uAlpha: { value: 0 }, uPx: pxScale, uScale: { value: def.size }, uColor: { value: color } },
      vertexShader: DOT_VS, fragmentShader: DOT_FS
    });
    var dots = new THREE.Points(dotGeo, dotMat);
    dots.frustumCulled = false;
    var segGeo = new THREE.BufferGeometry();
    segGeo.setAttribute('position', new THREE.Float32BufferAttribute(segPos, 3));
    var lines = new THREE.LineSegments(segGeo, new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
    lines.frustumCulled = false;
    var halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: warmTex, color: color, blending: THREE.AdditiveBlending, depthWrite: false,
      transparent: true, opacity: 0, fog: false }));
    halo.scale.setScalar(3.4);
    g.add(halo, lines, dots);
    (shape.fills || []).forEach(function (fl) {
      var pane = new THREE.Mesh(new THREE.PlaneGeometry(fl[2], fl[3]), new THREE.MeshBasicMaterial({ color: '#ffc070', transparent: true, opacity: 0,
        blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
      pane.position.set(fl[0], fl[1], 0);
      g.add(pane);
    });
    g.scale.setScalar(def.size);
    g.visible = false;
    world.add(g);

    var spark = new THREE.Sprite(new THREE.SpriteMaterial({ map: oneTex, color: color, blending: THREE.AdditiveBlending, depthWrite: false,
      transparent: true, fog: false }));
    spark.visible = false;
    world.add(spark);

    return { def: def, group: g, dots: dotMat, lines: lines, segAlong: segAlong, halo: halo, spark: spark,
             home: skyAt(def.az, def.el, def.d, 1, new THREE.Vector3()), at: new THREE.Vector3(), order: n === ONE ? 2 : r(),
             drift: new THREE.Vector3((r() - 0.5) * 50, 30 + r() * 25, -50 - r() * 60), phase: r() * 6.28, fills: g.children.slice(3) };
  });


  // Faint threads between neighbouring dreams once the sky is full.
  var linkPairs = [];
  dreams.forEach(function (a, i) {
    var near = dreams.map(function (b, j) { return [j, a.home.distanceTo(b.home)]; }).filter(function (e) { return e[0] !== i; })
      .sort(function (x, y) { return x[1] - y[1]; });
    [near[0][0], near[1][0]].forEach(function (j) { if (j > i || !linkPairs.some(function (p) { return p[0] === j && p[1] === i; })) linkPairs.push([i, j]); });
  });
  var linkPos = new Float32Array(linkPairs.length * 6), linkGeo = new THREE.BufferGeometry();
  linkGeo.setAttribute('position', new THREE.BufferAttribute(linkPos, 3));
  var links = new THREE.LineSegments(linkGeo, new THREE.LineBasicMaterial({ color: '#ffe0b0', transparent: true, opacity: 0,
    blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
  links.frustumCulled = false;
  world.add(links);

  // Phones see a narrow slice of sky, so the dreams gather closer in.
  var layout = { sx: 1, size: 1 }, ctrl = new THREE.Vector3();

  function frame(f) {
    var row = f.row, time = f.time, u = f.u, dim = f.dark;
    var open = row[6], hang = row[7], away = row[8], out = row[9];
    clock.value = time;
    breeze.value = f.wind * open;

    camera.position.set(Math.sin(time * 0.21) * 0.015, EYE + Math.sin(time * 0.6) * 0.008, f.cam);
    camera.rotation.set(0, 0, 0);
    camera.rotateY(row[4] - f.mx * 0.1);
    camera.rotateX(row[5] - f.my * 0.05);
    sky.position.copy(camera.position);

    panes.forEach(function (p) { p.hinge.rotation.y = p.side * -open * 1.95; });

    // The jar dims with each dream let go, and goes out at the end.
    var flick = 0.92 + 0.08 * Math.sin(time * 5.3) * Math.sin(time * 2.1);
    var jarOn = (1 - away * 0.35) * (1 - out) * flick;
    jarGlow.material.opacity = jarOn;
    jarHalo.material.opacity = jarOn * 0.3;
    jarLight.intensity = 2.2 * jarOn;
    jar.material.emissiveIntensity = 0.25 * jarOn;

    starMat.uniforms.uTime.value = time;
    starMat.uniforms.uDim.value = clamp(dim, 0, 1.2);
    winMat.uniforms.uOut.value = Math.max(away * 0.3, out);
    winMat.uniforms.uTime.value = time;

    dreams.forEach(function (d, n) {
      var def = d.def, born = TL ? TL.start(def.panel) + def.at * 1.6 : 0;
      var fly = clamp((u - born + 0.32) / 0.32, 0, 1), reveal = smooth(born - 0.02, born + 0.42, u);
      var gone = n === ONE ? 0 : smooth(0, 1, clamp(away * 1.7 - d.order * 0.7, 0, 1));
      var lastOut = n === ONE ? smooth(0, 0.7, out) : out;
      var alpha = (1 - gone * 0.94) * (1 - lastOut);

      skyAt(def.az, def.el, def.d, layout.sx, d.at, n === ONE).addScaledVector(d.drift, gone);
      // Each dream breathes in its own way.
      var bob = n === ONE ? 0 : Math.sin(time * 0.5 + d.phase) * 0.4 * (1 + hang);
      d.group.position.set(d.at.x, d.at.y + bob, d.at.z);
      d.group.rotation.z = def.shape === 'ship' ? Math.sin(time * 0.9) * 0.06 : def.shape === 'scales' ? Math.sin(time * 0.6) * 0.04 : 0;
      var beat = def.shape === 'heart' ? 1 + 0.06 * Math.max(0, Math.sin(time * 3.2)) : 1;
      if (def.shape === 'destroyer') d.group.position.x += Math.sin(time * 0.08) * 3;
      if (def.shape === 'balloon') d.group.position.y += Math.sin(time * 0.3) * 0.8;
      var sc = def.size * layout.size * (0.35 + 0.65 * reveal) * (1 - gone * 0.4) * beat;
      d.group.scale.setScalar(sc);
      d.group.visible = reveal > 0.001 && alpha > 0.002;
      d.dots.uniforms.uReveal.value = reveal;
      d.dots.uniforms.uAlpha.value = alpha * (1 + hang * 0.25);
      d.dots.uniforms.uScale.value = sc;
      var segs = 0;
      while (segs < d.segAlong.length && d.segAlong[segs] <= reveal + 0.001) segs++;
      d.lines.geometry.setDrawRange(0, segs * 2);
      d.lines.material.opacity = alpha * 0.32;
      d.halo.material.opacity = alpha * reveal * (0.08 + hang * 0.05);
      d.fills.forEach(function (fm) { fm.material.opacity = alpha * smooth(0.6, 1, reveal) * (0.75 + 0.1 * Math.sin(time * 2 + fm.position.x * 5)); });

      // The spark: out of the jar, through the window, up into place.
      d.spark.visible = fly > 0 && fly < 1;
      if (d.spark.visible) {
        var e = fly * fly * (3 - 2 * fly);
        ctrl.set(d.at.x * 0.15, EYE + 1.5, -6);
        var a0 = (1 - e) * (1 - e), a1 = 2 * e * (1 - e), a2 = e * e;
        d.spark.position.set(JAR.x * a0 + ctrl.x * a1 + d.at.x * a2, (JAR.y + 0.12) * a0 + ctrl.y * a1 + d.at.y * a2, JAR.z * a0 + ctrl.z * a1 + d.at.z * a2);
        d.spark.scale.setScalar(0.1 + e * 3.2);
        d.spark.material.opacity = 1;
      }
    });

    // The one star: a bright star like any other, until it is the only one.
    var od = dreams[ONE];
    oneStar.position.set(od.at.x, od.at.y - 0.08 * od.group.scale.y, od.at.z - 0.5);
    oneStar.scale.setScalar(2.4 + out * 2.6 + 0.3 * Math.sin(time * 1.7));
    oneStar.material.opacity = 0.75 + out * 0.25;

    // Threads between the dreams while the whole sky hangs there.
    var lo = hang * (1 - away);
    links.visible = lo > 0.01;
    if (links.visible) {
      linkPairs.forEach(function (pr, k) {
        var a = dreams[pr[0]].group.position, b = dreams[pr[1]].group.position;
        linkPos[k * 6] = a.x; linkPos[k * 6 + 1] = a.y; linkPos[k * 6 + 2] = a.z;
        linkPos[k * 6 + 3] = b.x; linkPos[k * 6 + 4] = b.y; linkPos[k * 6 + 5] = b.z;
      });
      linkGeo.attributes.position.needsUpdate = true;
      links.material.opacity = lo * 0.16;
    }

    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) {
      fitCamera(gl, camera, w, h, dpr, small);
      var pr = gl.getPixelRatio();
      pxScale.value = h * pr / (2 * Math.tan(camera.fov * Math.PI / 360));
      starMat.uniforms.uScale.value = pr * (h / 900 + 0.35);
      layout.sx = w / h < 1 ? 0.45 : 1;
      layout.size = w / h < 1 ? 0.75 : 1;
    },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('perhaps', {
  renderer: renderer3d,
  maxLines: 5,
  scrim: 0.62,
  align: ['left', 'right', 'left', 'right', 'left', 'center', 'center', 'center'],
  keys: function (T) {
    TL = T;
    var n = T.count;
    function at(i, frac) { i = Math.min(i, n - 1); return lerp(T.start(i), T.end(i), frac); }
    //       unit          camZ   dim  -  breeze yaw    pitch  open hang away out
    return [
      [0,                  3.60, 0.00, 0, 0.0, 0.00,  0.04,  0.0, 0.0, 0.0, 0.0],
      [0.55,               3.45, 0.00, 0, 0.2, 0.00,  0.05,  0.1, 0.0, 0.0, 0.0],
      [at(0, 0.1),         2.60, 0.00, 0, 0.6, 0.02,  0.08,  1.0, 0.0, 0.0, 0.0],   // the window swings open
      [at(0, 0.9),         1.80, 0.00, 0, 0.5, 0.05,  0.12,  1.0, 0.0, 0.0, 0.0],
      [at(1, 0.6),         1.00, 0.00, 0, 0.5, -0.06, 0.15,  1.0, 0.0, 0.0, 0.0],
      [at(2, 0.6),         0.30, 0.00, 0, 0.5, 0.08,  0.17,  1.0, 0.0, 0.0, 0.0],
      [at(3, 0.6),        -0.20, 0.00, 0, 0.6, -0.08, 0.22,  1.0, 0.0, 0.0, 0.0],
      [at(4, 0.6),        -0.50, 0.00, 0, 0.6, 0.06,  0.30,  1.0, 0.0, 0.0, 0.0],   // leaning out into the dreams
      [at(5, 0.5),        -0.70, 0.00, 0, 0.4, 0.00,  0.26,  1.0, 1.0, 0.0, 0.0],   // "He smiled and looked ahead."
      [at(6, 0.15),       -0.70, 0.05, 0, 0.5, 0.00,  0.27,  1.0, 0.8, 0.05, 0.0],
      [at(6, 0.9),        -0.55, 0.30, 0, 0.7, 0.00,  0.28,  1.0, 0.3, 0.75, 0.0],  // "He softly sighed with sorrow."
      [at(7, 0.3),        -0.20, 0.55, 0, 0.4, 0.00,  0.24,  1.0, 0.0, 1.0, 0.1],
      [at(7, 0.85),        1.10, 0.92, 0, 0.1, 0.00,  0.22,  1.0, 0.0, 1.0, 1.0],   // "... or maybe not," he said.
      [T.total,            1.60, 1.00, 0, 0.0, 0.00,  0.22,  1.0, 0.0, 1.0, 1.0]
    ];
  },
  sound: {
    src: '/audio/wind.mp3',
    label: 'Play the night breeze and chimes',
    volume: function (row) { return 0.04 + 0.08 * row[3]; },
    cues: DREAMS.map(function (d, k) { return { stanza: d.panel, at: d.at * 1.6 - 0.05, play: chime(NOTES[k]) }; })
      .concat([{ stanza: 6, at: 0.5, play: sigh }, { stanza: 7, at: 1.0, play: chime(196, 0.07) }])
  }
});
