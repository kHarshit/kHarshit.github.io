/*
 * Scene for "If Tomorrow Starts Without Me" (anonymous): one journey from a
 * window at dawn up into the light and home again, told with light, never
 * figures.
 *
 * I-III     Inside, at a cottage window at dawn. Rain runs down the glass;
 *           the sun rises low behind it, "and find your eyes all filled with
 *           tears". The rain slowly eases.
 * IV        "An angel came and called my name": a warm light comes down out
 *           of the overcast to the window, "took me by the hand", and you
 *           follow it out through the glass and up.
 * V-VI      Turning back as you rise, "a tear fell from my eye": one drop of
 *           light falls towards the house; its window glows below as you
 *           climb away ("that I was leaving you").
 * VII-IX    "All the yesterdays": soft glowing orbs drift past, one comes
 *           close ("and maybe see you smile"), then they drift away into
 *           the cloud ("emptiness and memories").
 * X         Inside the grey cloud, "my heart was filled with sorrow".
 * XI        Out above it into gold: "heaven's gates", an arch of light on the
 *           cloud sea that opens, with a great radiance beyond it.
 * XII-XV    Through the gate, "this is eternity": an endless golden
 *           cloudscape where the sun never moves; lights rise from the
 *           clouds; the guiding light returns ("come and take my hand").
 * XVI       Brightness to white, and back at the window: the rain has
 *           stopped, morning sun, and a candle glowing on the sill: "I'm
 *           right here, in your heart."
 *
 * The sky is a palette index ("look") blended between five looks. Columns:
 *   [unit, way, look, rain, wind, yaw, pitch, memories, guide, lead, tear,
 *    gate, candle, flash, motes, near]
 */
import { THREE, isSmall, makeRenderer, fitCamera, broadleafGeometry, tinted, merge, softSprite, skyDome,
         terrain, scatter, particleField, rainField, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout (metres): the cottage's front wall is z = 0, facing -z ────────
var WIN = { w: 1.4, y0: 1.0, y1: 2.5 };            // the window opening
var CLOUD_Y = 610;                                  // the top of the overcast
var GATE = new THREE.Vector3(0, 615, -1500);        // the arch stands on the cloud sea
// Waypoints: the "way" column is an index into these (fractional between).
var WAY = [[0, 1.6, 2.35], [0, 1.65, 0.6], [0, 2.4, -4], [0, 9, -22], [0, 40, -70], [0, 120, -160], [0, 260, -280],
           [0, 545, -460], [0, 665, -660], [0, 668, -1050], [0, 668, -1500], [0, 672, -2100], [0, 672, -2800]];
var curve = new THREE.CatmullRomCurve3(WAY.map(function (p) { return new THREE.Vector3(p[0], p[1], p[2]); }), false, 'centripetal');

function land(x, z) {
  var d = Math.hypot(x, z - 3);
  return smooth(14, 80, d) * (3 * Math.sin(x * 0.011) * Math.cos(z * 0.009) + 2 * Math.sin(x * 0.027 + z * 0.019) - 1.5) +
         smooth(500, 1400, d) * 60 * (0.6 + 0.4 * Math.sin(Math.atan2(z, x) * 5));
}

var LOOKS = [
  { // 0 rainy dawn
    top: '#3a4258', mid: '#7a7f92', horizon: '#d9a98a', glow: '#ffc890', glowAmt: 0.55, el: 0.02, sunI: 0.7, az: 0.25,
    hemiSky: '#9aa2b8', hemiGnd: '#2a2a26', hemiI: 1.0, fog: '#8a8a96', fogD: 0.0012, lit: '#9a9aa8', shade: '#585a68', exp: 1 },
  { // 1 the dawn clearing
    top: '#4a5a88', mid: '#c89aa0', horizon: '#ffcfa0', glow: '#ffd8a0', glowAmt: 1.0, el: 0.05, sunI: 1.6, az: 0.25,
    hemiSky: '#e0c8cc', hemiGnd: '#5a5038', hemiI: 1.6, fog: '#c8a8a4', fogD: 0.0009, lit: '#e8c4b4', shade: '#7a7088', exp: 1 },
  { // 2 inside the cloud
    top: '#b0b0ba', mid: '#c4c4cc', horizon: '#cfcfd6', glow: '#ffffff', glowAmt: 0.15, el: 0.3, sunI: 0.8, az: 0.25,
    hemiSky: '#d0d0d8', hemiGnd: '#9a9aa0', hemiI: 1.4, fog: '#bcbcc6', fogD: 0.018, lit: '#e0e0e6', shade: '#a0a0aa', exp: 1 },
  { // 3 golden eternity
    top: '#34467e', mid: '#c48c66', horizon: '#f2b878', glow: '#ffe2a8', glowAmt: 0.55, el: 0.1, sunI: 2.0, az: 0.34,
    hemiSky: '#f0c8a0', hemiGnd: '#9a6a48', hemiI: 1.2, fog: '#e0a474', fogD: 0.00028, lit: '#ffd496', shade: '#b06c50', exp: 0.95 },
  { // 4 the morning after, at the window
    top: '#5a7ab8', mid: '#a8c4e0', horizon: '#ffe2bc', glow: '#fff0d0', glowAmt: 1.0, el: 0.12, sunI: 2.2, az: 0.25,
    hemiSky: '#d8e4f8', hemiGnd: '#4a4a38', hemiI: 1.3, fog: '#d4d8e2', fogD: 0.0008, lit: '#f4f4f8', shade: '#a8b0c4', exp: 1 }
];
var COLOR_KEYS = ['top', 'mid', 'horizon', 'glow', 'hemiSky', 'hemiGnd', 'fog', 'lit', 'shade'];
var NUM_KEYS = ['glowAmt', 'el', 'az', 'sunI', 'hemiI', 'fogD', 'exp'];
LOOKS.forEach(function (l) { COLOR_KEYS.forEach(function (k) { l[k] = new THREE.Color(l[k]); }); });

var NOISE = [
  'float hh(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }',
  'float vn(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);',
  '  return mix(mix(hh(i), hh(i + vec2(1.0, 0.0)), f.x), mix(hh(i + vec2(0.0, 1.0)), hh(i + vec2(1.0, 1.0)), f.x), f.y); }',
  'float fbm(vec2 p){ float s = 0.0, a = 0.5; for (int k = 0; k < 5; k++) { s += a * vn(p); p = p * 2.07 + 3.1; a *= 0.5; } return s; }'
].join('\n');

// The gates opening: a bright, slow bell chord.
function gateChime(ac, out) {
  var t = ac.currentTime;
  [523.25, 659.25, 783.99, 1046.5, 1318.5].forEach(function (f, k) {
    [1, 2.01, 3.02].forEach(function (m, j) {
      var o = ac.createOscillator(), g = ac.createGain(), at = t + k * 0.18;
      o.type = 'sine';
      o.frequency.value = f * m;
      g.gain.setValueAtTime(0.0001, at);
      g.gain.exponentialRampToValueAtTime(0.06 / (j + 1) / (1 + k * 0.2), at + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, at + 5 / (j + 1));
      o.connect(g); g.connect(out);
      o.start(at); o.stop(at + 5.2);
    });
  });
}

// The light filling the open gate: an arch, soft-edged, brightest low down.
function doorTexture() {
  var c = document.createElement('canvas'), d = document.createElement('canvas');
  c.width = d.width = 256; c.height = d.height = 142;
  var x = c.getContext('2d');
  var g = x.createRadialGradient(128, 150, 10, 128, 100, 150);
  g.addColorStop(0, 'rgba(255,252,240,1)'); g.addColorStop(1, 'rgba(255,226,170,0.35)');
  x.fillStyle = g;
  x.beginPath();
  x.moveTo(14, 142); x.lineTo(14, 128); x.arc(128, 128, 114, Math.PI, 0); x.lineTo(242, 142); x.closePath();
  x.fill();
  var y = d.getContext('2d');
  y.filter = 'blur(9px)';
  y.drawImage(c, 0, 0);
  var t = new THREE.CanvasTexture(d);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// Rays round a soft centre, for the radiance beyond the gate.
function rayTexture() {
  var c = document.createElement('canvas');
  c.width = c.height = 256;
  var x = c.getContext('2d'), r = rng(12);
  var g = x.createRadialGradient(128, 128, 0, 128, 128, 128);
  g.addColorStop(0, 'rgba(255,248,230,1)'); g.addColorStop(0.1, 'rgba(255,228,180,0.7)'); g.addColorStop(0.35, 'rgba(255,200,140,0.18)'); g.addColorStop(1, 'rgba(255,190,130,0)');
  x.fillStyle = g;
  x.fillRect(0, 0, 256, 256);
  x.globalCompositeOperation = 'lighter';
  for (var i = 0; i < 48; i++) {
    var a = i / 48 * Math.PI * 2 + r() * 0.05, len = 70 + r() * 58, w = 0.012 + r() * 0.02;
    var lg = x.createLinearGradient(128, 128, 128 + Math.cos(a) * len, 128 + Math.sin(a) * len);
    lg.addColorStop(0, 'rgba(255,226,170,0.35)'); lg.addColorStop(1, 'rgba(255,215,150,0)');
    x.fillStyle = lg;
    x.beginPath();
    x.moveTo(128, 128);
    x.lineTo(128 + Math.cos(a - w) * len, 128 + Math.sin(a - w) * len);
    x.lineTo(128 + Math.cos(a + w) * len, 128 + Math.sin(a + w) * len);
    x.fill();
  }
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(31);
  var gl = makeRenderer(canvas, { clear: '#8a8a96' });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#8a8a96', 0.0012);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.05, 9000);
  world.add(camera);

  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#3a4258', mid: '#7a7f92', horizon: '#d9a98a', sun: '#ffc890' }, 3000);
  sky.add(dome.mesh);
  var sunDir = new THREE.Vector3();

  var hemi = new THREE.HemisphereLight('#9aa2b8', '#2a2a26', 1);
  var sun = new THREE.DirectionalLight('#ffd8b0', 0.7);
  world.add(hemi, sun, sun.target);

  // ── The land below: fields, hedges of trees, a village ──
  var ground = terrain(4000, small ? 120 : 180, 0, -600, land, new THREE.MeshLambertMaterial({ vertexColors: true }), (function () {
    var c = new THREE.Color(), a = new THREE.Color('#5a7238'), b = new THREE.Color('#6e7c42');
    // Fields in long strips, close in tone.
    return function (x, z) { var strip = Math.floor((z + 30 * Math.sin(x * 0.004)) / 38); return c.copy(a).lerp(b, ((strip * 0.618) % 1 + 1) % 1 * 0.8); };
  })());
  world.add(ground);
  var treeMat = new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true }), up = new THREE.Vector3(0, 1, 0);
  var trees = new THREE.InstancedMesh(broadleafGeometry(rng(4), '#3e3026'), treeMat, small ? 260 : 600);
  scatter(trees, 9000, function (k, p, q, s, c) {
    var x = (r() - 0.5) * 900, z = 40 - r() * 900;
    if (Math.hypot(x, z) < 16 || (Math.abs(x) < 6 && z < 0 && z > -40)) return false;
    // Hedgerows along field edges, and a few in the open.
    if (r() < 0.75 && Math.abs(Math.sin(x * 0.02)) > 0.12 && Math.abs(Math.sin(z * 0.025)) > 0.12) return false;
    p.set(x, land(x, z) - 0.2, z);
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar(0.9 + r() * 0.7);
    c.setHSL(0.24 + r() * 0.08, 0.35, 0.3 + r() * 0.12);
  });
  world.add(trees);
  // Village houses with lit windows, scattered down the valley.
  var hut = merge([tinted(new THREE.BoxGeometry(6, 4, 5).translate(0, 2, 0), '#c8bca8'),
                   tinted(new THREE.ConeGeometry(4.6, 2.6, 4).rotateY(Math.PI / 4).scale(1, 1, 0.8).translate(0, 5.3, 0), '#6a4a3a')]);
  var huts = new THREE.InstancedMesh(hut, new THREE.MeshLambertMaterial({ vertexColors: true }), small ? 40 : 80), hutPos = [];
  scatter(huts, 3000, function (k, p, q, s) {
    var x = (r() - 0.5) * 500, z = -60 - r() * 500;
    if (Math.abs(x) < 20) return false;
    p.set(x, land(x, z), z);
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar(0.8 + r() * 0.5);
    hutPos.push(x, land(x, z) + 2, z);
  });
  world.add(huts);
  var hutLightGeo = new THREE.BufferGeometry();
  hutLightGeo.setAttribute('position', new THREE.Float32BufferAttribute(hutPos, 3));
  var hutLights = new THREE.Points(hutLightGeo, new THREE.PointsMaterial({ color: '#ffc27a', size: 2.2, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, map: softSprite('rgba(255,215,150,1)', 'rgba(255,190,110,0)') }));
  world.add(hutLights);

  // ── The cottage: front wall with the window, a roof, the sill ──
  var wallMat = new THREE.MeshLambertMaterial({ color: '#cfc4b2' }), woodMat = new THREE.MeshLambertMaterial({ color: '#5a4434' });
  var cottage = new THREE.Group(), hw = WIN.w / 2;
  [[-4, 0, -hw, 3.2], [hw, 0, 4, 3.2]].forEach(function (b) {             // left and right of the window
    var m = new THREE.Mesh(new THREE.BoxGeometry(b[2] - b[0], 3.2, 0.3), wallMat);
    m.position.set((b[0] + b[2]) / 2, 1.6, -0.15);
    cottage.add(m);
  });
  var below = new THREE.Mesh(new THREE.BoxGeometry(WIN.w, WIN.y0, 0.3), wallMat);
  below.position.set(0, WIN.y0 / 2, -0.15);
  var above = new THREE.Mesh(new THREE.BoxGeometry(WIN.w, 3.2 - WIN.y1, 0.3), wallMat);
  above.position.set(0, (3.2 + WIN.y1) / 2, -0.15);
  var back = new THREE.Mesh(new THREE.BoxGeometry(8, 3.2, 0.3), wallMat);
  back.position.set(0, 1.6, 6.15);
  var sideL = new THREE.Mesh(new THREE.BoxGeometry(0.3, 3.2, 6.6), wallMat);
  sideL.position.set(-4.15, 1.6, 3);
  var sideR = sideL.clone();
  sideR.position.x = 4.15;
  var roofShape = new THREE.Shape();
  roofShape.moveTo(-0.7, -0.15); roofShape.lineTo(7.2, -0.15); roofShape.lineTo(3.25, 2.7); roofShape.lineTo(-0.7, -0.15);
  var roof = new THREE.Mesh(new THREE.ExtrudeGeometry(roofShape, { depth: 9.4, bevelEnabled: false }), new THREE.MeshLambertMaterial({ color: '#4a3a34' }));
  roof.rotation.y = -Math.PI / 2;
  roof.position.set(4.7, 3.2, 0);
  // From outside: a chimney, a door, two more lit windows, a path and a fence.
  var stone = new THREE.MeshLambertMaterial({ color: '#7a7068' });
  var chimney = new THREE.Mesh(new THREE.BoxGeometry(0.7, 2.4, 0.7), stone);
  chimney.position.set(2.6, 5.1, 4.2);
  var door = new THREE.Mesh(new THREE.BoxGeometry(0.95, 2.05, 0.06), woodMat);
  door.position.set(-2.4, 1.03, -0.33);
  var lintel = new THREE.Mesh(new THREE.BoxGeometry(1.25, 0.12, 0.12), woodMat);
  lintel.position.set(-2.4, 2.12, -0.34);
  var step = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.16, 0.6), stone);
  step.position.set(-2.4, 0.06, -0.6);
  cottage.add(chimney, door, lintel, step);
  var litMat = new THREE.MeshBasicMaterial({ color: '#ffc988', transparent: true, opacity: 0 }), outerWins = [];
  [[2.4, 1.75, -0.31], [-4.31, 1.75, 2.4]].forEach(function (w, k) {
    var m = new THREE.Mesh(new THREE.PlaneGeometry(1.0, 1.1), litMat);
    m.position.set(w[0], w[1], w[2]);
    m.rotation.y = k ? -Math.PI / 2 : Math.PI;
    var frame = new THREE.Mesh(new THREE.BoxGeometry(1.15, 1.25, 0.04), woodMat);
    frame.position.copy(m.position);
    frame.rotation.y = m.rotation.y;
    frame.translateZ(-0.03);
    var bar = new THREE.Mesh(new THREE.BoxGeometry(0.04, 1.1, 0.02), woodMat);
    bar.position.copy(m.position);
    bar.rotation.y = m.rotation.y;
    bar.translateZ(0.01);
    cottage.add(frame, m, bar);
    outerWins.push(m);
  });
  var gravel = new THREE.MeshLambertMaterial({ color: '#9a8e7a' });
  var pathGeo = new THREE.PlaneGeometry(1.2, 22, 1, 22).rotateX(-Math.PI / 2), pp = pathGeo.attributes.position;
  for (var pk = 0; pk < pp.count; pk++) {
    var pz0 = pp.getZ(pk) - 11.9, px0 = pp.getX(pk) - 2.4 + Math.sin(pz0 * 0.18) * 0.6;
    pp.setXYZ(pk, px0, land(px0, pz0) - land(0, 0) + 0.03, pz0);
  }
  pathGeo.computeVertexNormals();
  cottage.add(new THREE.Mesh(pathGeo, gravel));
  // A low picket fence round the front garden, with a gap for the path.
  var picket = new THREE.InstancedMesh(new THREE.BoxGeometry(0.07, 0.8, 0.04).translate(0, 0.4, 0), new THREE.MeshLambertMaterial({ color: '#d8d2c4' }), 160);
  var fm4 = new THREE.Matrix4(), fn = 0, fenceZ = -7;
  for (var fx = -6; fx <= 6.01; fx += 0.22) {
    if (Math.abs(fx + 2.4 - Math.sin(fenceZ * 0.18) * 0.6) < 0.8) continue;
    picket.setMatrixAt(fn++, fm4.makeTranslation(fx, land(fx, fenceZ) - land(0, 0), fenceZ));
  }
  for (var fz = fenceZ + 0.25; fz < -0.4; fz += 0.25) {
    [-6, 6].forEach(function (sx) { picket.setMatrixAt(fn++, fm4.makeTranslation(sx, land(sx, fz) - land(0, 0), fz)); });
  }
  picket.count = fn;
  var rail = new THREE.MeshLambertMaterial({ color: '#d8d2c4' });
  [[-6, -3.25, 12, fenceZ], [-6, -3.6, 0, 0], [6, -3.6, 0, 0]].forEach(function (rl, k) {
    var m = new THREE.Mesh(k ? new THREE.BoxGeometry(0.04, 0.06, Math.abs(fenceZ) - 0.4) : new THREE.BoxGeometry(12.1, 0.06, 0.04), rail);
    m.position.set(k ? rl[0] : 0, 0.55, k ? fenceZ / 2 - 0.2 : fenceZ);
    cottage.add(m);
  });
  cottage.add(picket);
  var sill = new THREE.Mesh(new THREE.BoxGeometry(WIN.w + 0.3, 0.06, 0.42), woodMat);
  sill.position.set(0, WIN.y0, 0.06);
  // Glazing bars: one upright, one across.
  var barV = new THREE.Mesh(new THREE.BoxGeometry(0.05, WIN.y1 - WIN.y0, 0.06), woodMat);
  barV.position.set(0, (WIN.y0 + WIN.y1) / 2, -0.15);
  var barH = new THREE.Mesh(new THREE.BoxGeometry(WIN.w, 0.05, 0.06), woodMat);
  barH.position.set(0, WIN.y0 + (WIN.y1 - WIN.y0) * 0.58, -0.15);
  var floor = new THREE.Mesh(new THREE.BoxGeometry(8, 0.1, 6.3), woodMat);
  floor.position.set(0, 0.05, 3);
  var ceiling = new THREE.Mesh(new THREE.BoxGeometry(8, 0.1, 6.3), wallMat);
  ceiling.position.set(0, 3.15, 3);
  cottage.add(below, above, back, sideL, sideR, roof, sill, barV, barH, floor, ceiling);
  cottage.position.y = land(0, 0);
  world.add(cottage);
  var interior = new THREE.PointLight('#ffcf9a', 0, 9, 1.5);      // the room, warm, seen from outside
  interior.position.set(0, 2, 2.5);
  cottage.add(interior);

  // ── Rain on the glass ──
  var glassU = { uTime: { value: 0 }, uRain: { value: 1 }, uTint: { value: new THREE.Color() }, uSun: { value: 0 } };
  var glass = new THREE.Mesh(new THREE.PlaneGeometry(WIN.w, WIN.y1 - WIN.y0), new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, uniforms: glassU,
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform float uTime; uniform float uRain; uniform vec3 uTint; uniform float uSun; varying vec2 vUv;\n' + NOISE + '\n' +
      // A drop is clear: a darker rim, the sky refracted (upside down, so
      // bright at the bottom), a small sharp highlight up and left.
      'vec4 drop(vec2 f, float r){ float d = length(f * vec2(1.0, 0.85)); float body = smoothstep(r, r * 0.88, d);\n' +
      ' float rim = body * smoothstep(r * 0.5, r * 0.95, d);\n' +
      ' float sky = body * smoothstep(r * 0.1, -r * 0.9, f.y) * (1.0 - rim);\n' +
      ' float hi = smoothstep(r * 0.2, r * 0.05, length(f + vec2(r * 0.32, -r * 0.4)));\n' +
      ' return vec4(body, rim, sky, hi * body); }\n' +
      'void main(){ vec2 p = vUv * vec2(' + WIN.w.toFixed(2) + ', ' + (WIN.y1 - WIN.y0).toFixed(2) + ');\n' +
      ' vec4 acc = vec4(0.0);\n' +
      // Beads that sit on the pane.
      ' for (int k = 0; k < 2; k++) { float sc = k == 0 ? 20.0 : 44.0; vec2 g = p * sc + float(k) * 7.3; vec2 id = floor(g); vec2 f = fract(g) - 0.5;\n' +
      '  vec2 o = (vec2(hh(id + 1.7), hh(id + 3.1)) - 0.5) * 0.5; float on = step(1.0 - uRain * (k == 0 ? 0.28 : 0.32), hh(id));\n' +
      '  acc = max(acc, drop(f - o, 0.14 + 0.2 * hh(id + 5.3)) * on); }\n' +
      // Runnels: drops sliding down columns, leaving a wet streak.
      ' float cw = 0.07; float col = floor(p.x / cw); float cx = (fract(p.x / cw) - 0.5) * cw;\n' +
      ' float sp = 0.08 + 0.18 * hh(vec2(col, 2.0)), ph = hh(vec2(col, 9.0));\n' +
      ' float wig = sin(p.y * 40.0 + col) * 0.004;\n' +
      ' float yd = 1.6 - fract(uTime * sp + ph) * 2.0;\n' +
      ' float live = step(1.0 - uRain * 0.4, hh(vec2(col, 4.0)));\n' +
      ' acc = max(acc, drop(vec2(cx - wig, (p.y - yd) * 0.7) / cw * 0.5, 0.3) * live);\n' +
      ' float streak = step(yd, p.y) * smoothstep(yd + 0.6, yd, p.y) * smoothstep(0.006, 0.001, abs(cx - wig)) * live;\n' +
      ' float dark = acc.y * 0.42 + streak * 0.12, light = acc.z * 0.3 + acc.w * 0.9 + streak * 0.18;\n' +
      ' float a = clamp(0.02 + dark + light, 0.0, 1.0);\n' +
      ' vec3 c = (vec3(0.03, 0.04, 0.06) * dark + (uTint * (acc.z * 0.3 + streak * 0.18) + vec3(1.0) * acc.w * 0.9 * (0.7 + uSun))) / a;\n' +
      ' gl_FragColor = vec4(c, a);\n #include <colorspace_fragment>\n }'
  }));
  glass.position.set(0, (WIN.y0 + WIN.y1) / 2, -0.16);
  cottage.add(glass);
  // From outside the pane glows with the lit room.
  var pane = new THREE.Mesh(new THREE.PlaneGeometry(WIN.w, WIN.y1 - WIN.y0), new THREE.MeshBasicMaterial({ color: '#ffcf96', transparent: true, opacity: 0 }));
  pane.rotation.y = Math.PI;
  pane.position.set(0, (WIN.y0 + WIN.y1) / 2, -0.17);
  cottage.add(pane);
  var paneGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,210,150,1)', 'rgba(255,170,100,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0 }));
  paneGlow.position.set(0, (WIN.y0 + WIN.y1) / 2, -0.8);
  paneGlow.scale.setScalar(5);
  cottage.add(paneGlow);

  // ── The candle on the sill ──
  var candle = new THREE.Group();
  candle.add(new THREE.Mesh(new THREE.CylinderGeometry(0.032, 0.035, 0.17, 16).translate(0, 0.085, 0), new THREE.MeshLambertMaterial({ color: '#f2ead8', emissive: '#3a2a18' })));
  var flameTex = softSprite('rgba(255,236,190,1)', 'rgba(255,160,70,0)');
  var flame = new THREE.Sprite(new THREE.SpriteMaterial({ map: flameTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
  flame.position.y = 0.21;
  var halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: flameTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
  halo.position.y = 0.21;
  var candleLight = new THREE.PointLight('#ffb466', 0, 4, 1.6);
  candleLight.position.y = 0.3;
  candle.add(flame, halo, candleLight);
  candle.position.set(0.48, WIN.y0 + 0.03, 0.12);
  cottage.add(candle);

  // ── Clouds: the overcast's underside and top, and puffs to climb through ──
  var cloudU = { uLit: { value: new THREE.Color() }, uShade: { value: new THREE.Color() }, uSun: { value: sunDir }, uGlow: { value: new THREE.Color() },
                 uFog: { value: new THREE.Color() }, uFogD: { value: 0 }, uTime: { value: 0 } };
  var sea = new THREE.Mesh(new THREE.PlaneGeometry(16000, 16000).rotateX(-Math.PI / 2), new THREE.ShaderMaterial({
    side: THREE.DoubleSide, uniforms: cloudU,
    vertexShader: 'varying vec3 vW; varying float vDepth; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz;\n' +
      ' vec4 mv = viewMatrix * w; vDepth = -mv.z; gl_Position = projectionMatrix * mv; }',
    fragmentShader: 'uniform vec3 uLit; uniform vec3 uShade; uniform vec3 uSun; uniform vec3 uGlow; uniform vec3 uFog; uniform float uFogD; uniform float uTime;\n' +
      'varying vec3 vW; varying float vDepth;\n' + NOISE + '\n' +
      'float billow(vec2 p){ return fbm(p) * 0.6 + fbm(p * 3.1 + 5.0) * 0.4; }\n' +
      'void main(){ vec2 p = vW.xz * 0.005 + vec2(uTime * 0.003, 0.0);\n' +
      // Light from the sun's side of each billow; tops brighter than the hollows.
      ' vec2 sd = normalize(uSun.xz + vec2(0.0001)) * 0.03;\n' +
      ' float n = billow(p), lit = clamp(0.55 + (n - billow(p - sd)) * 9.0, 0.0, 1.0);\n' +
      ' float k = clamp(smoothstep(0.3, 0.72, n) * 0.55 + lit * 0.65 - 0.2, 0.0, 1.0);\n' +
      ' vec3 c = mix(uShade * 0.8, uLit, k * k * (3.0 - 2.0 * k)) * 0.9;\n' +
      ' vec3 V = normalize(vW - cameraPosition);\n' +
      ' c += uGlow * pow(max(dot(normalize(vec3(V.x, 0.0, V.z)), normalize(vec3(uSun.x, 0.0, uSun.z))), 0.0), 24.0) * 0.22 * smoothstep(0.5, 0.8, n);\n' +
      ' if (!gl_FrontFacing) c = uShade * (0.55 + 0.25 * n);\n' +
      ' gl_FragColor = vec4(c, 1.0);\n #include <colorspace_fragment>\n' +
      ' gl_FragColor.rgb = mix(gl_FragColor.rgb, uFog, 1.0 - exp(-uFogD * uFogD * vDepth * vDepth)); }'
  }));
  sea.position.y = CLOUD_Y;
  sea.frustumCulled = false;
  world.add(sea);
  var puffTex = softSprite('rgba(255,255,255,0.85)', 'rgba(255,255,255,0)'), puffs = [];
  for (var i = 0; i < (small ? 120 : 240); i++) {
    var climb = i < (small ? 70 : 140);
    var pm = new THREE.Sprite(new THREE.SpriteMaterial({ map: puffTex, transparent: true, depthWrite: false, fog: true }));
    if (climb) pm.position.set((r() - 0.5) * 260, 520 + r() * 110, -360 - r() * 380);   // round the climb
    else pm.position.set((r() - 0.5) * 2400, CLOUD_Y - 25 + r() * 25, -700 - r() * 3200);   // billows on the sea
    var ps = climb ? 50 + r() * 70 : 120 + r() * 200;
    pm.scale.set(ps, ps * (climb ? 0.6 : 0.35), 1);
    pm.userData.base = climb ? 0.5 + r() * 0.45 : 0.25 + r() * 0.3;
    pm.userData.climb = climb;
    world.add(pm);
    puffs.push(pm);
  }

  // ── Memories: soft orbs drifting where you climb ──
  var MN = small ? 70 : 140, mPos = [], mCol = [], mSeed = [], palette = ['#ffd49a', '#ffb8a0', '#fff0c8', '#f8c0d0', '#ffe0a8'];
  var tmpC = new THREE.Color();
  for (i = 0; i < MN; i++) {
    var mt = 0.3 + r() * 0.42, mp = curve.getPoint(mt);
    var ma = r() * 6.28, md = 8 + r() * 40;
    mPos.push(mp.x + Math.cos(ma) * md, mp.y + (r() - 0.5) * 30, mp.z + Math.sin(ma) * md * 0.6);
    tmpC.set(palette[i % palette.length]);
    mCol.push(tmpC.r, tmpC.g, tmpC.b);
    mSeed.push(r() * 6.28, 1.5 + r() * 2.5);
  }
  var memGeo = new THREE.BufferGeometry();
  memGeo.setAttribute('position', new THREE.Float32BufferAttribute(mPos, 3));
  memGeo.setAttribute('aCol', new THREE.Float32BufferAttribute(mCol, 3));
  memGeo.setAttribute('aSeed', new THREE.Float32BufferAttribute(mSeed, 2));
  var memMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uTime: { value: 0 }, uAmt: { value: 0 }, uAway: { value: 0 }, uScale: { value: 1 } },
    vertexShader: 'attribute vec3 aCol; attribute vec2 aSeed; uniform float uTime; uniform float uAmt; uniform float uAway; uniform float uScale;\n' +
      'varying vec3 vC; varying float vA;\n' +
      'void main(){ vec3 p = position + vec3(sin(uTime * 0.21 + aSeed.x) * 3.0, sin(uTime * 0.3 + aSeed.x * 2.0) * 2.0 + uAway * (40.0 + aSeed.y * 20.0), cos(uTime * 0.17 + aSeed.x) * 3.0);\n' +
      ' vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_Position = projectionMatrix * mv;\n' +
      ' vC = aCol; vA = uAmt * (0.75 + 0.25 * sin(uTime * 1.3 + aSeed.x * 3.0)) * smoothstep(2.0, 10.0, -mv.z);\n' +
      ' gl_PointSize = aSeed.y * uScale * 900.0 / -mv.z; }',
    fragmentShader: 'varying vec3 vC; varying float vA; void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;\n' +
      ' float a = (smoothstep(0.5, 0.0, d) * 0.6 + smoothstep(0.16, 0.0, d) * 0.8) * vA; gl_FragColor = vec4(vC * a, a);\n #include <colorspace_fragment>\n }'
  });
  var memories = new THREE.Points(memGeo, memMat);
  memories.frustumCulled = false;
  world.add(memories);
  // One that comes close: "and maybe see you smile".
  var nearOrb = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,214,190,1)', 'rgba(255,170,150,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0 }));
  camera.add(nearOrb);

  // ── The guiding light, the tear ──
  var glowTex = softSprite('rgba(255,240,210,1)', 'rgba(255,200,140,0)');
  var guide = new THREE.Group(), guideCore = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
  var guideHalo = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
  guideCore.scale.setScalar(0.9);
  guideHalo.scale.setScalar(5);
  var guideLight = new THREE.PointLight('#ffd8a8', 0, 14, 1.5);
  guide.add(guideCore, guideHalo, guideLight);
  world.add(guide);
  var tear = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(220,235,255,1)', 'rgba(170,200,255,0)'), blending: THREE.AdditiveBlending,
    depthWrite: false, transparent: true, opacity: 0 }));
  world.add(tear);

  // ── Heaven's gate: an arch of light on the cloud sea, and the radiance ──
  var gate = new THREE.Group();
  var archMat = new THREE.MeshBasicMaterial({ color: '#fff6e0', transparent: true, depthWrite: false, fog: false, toneMapped: false });
  var arch = new THREE.Mesh(new THREE.TorusGeometry(110, 3.2, 10, 120, Math.PI), archMat);
  gate.add(arch);
  [-1, 1].forEach(function (sd) {
    var pillar = new THREE.Mesh(new THREE.CylinderGeometry(3.2, 3.2, 60, 10).translate(sd * 110, -30, 0), archMat);
    gate.add(pillar);
  });
  var archGlows = [];
  for (i = 0; i <= 24; i++) {
    var aa = i / 24 * Math.PI, gs = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
    gs.position.set(Math.cos(aa) * 110, Math.sin(aa) * 110, 0);
    gs.scale.setScalar(34);
    gate.add(gs);
    archGlows.push(gs);
  }
  var veil = new THREE.Mesh(new THREE.PlaneGeometry(220, 122), new THREE.MeshBasicMaterial({ map: doorTexture(), blending: THREE.AdditiveBlending,
    depthWrite: false, transparent: true, fog: false }));
  veil.position.set(0, 49, -1);
  gate.add(veil);
  gate.position.copy(GATE);
  world.add(gate);
  var radiance = new THREE.Sprite(new THREE.SpriteMaterial({ map: rayTexture(), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
  radiance.scale.setScalar(520);
  sky.add(radiance);

  // ── Weather and motes ──
  var rain = rainField({ count: small ? 700 : 1500, box: [18, 16, 28], color: '#aab4c4', opacity: 0.3, speed: 10, windSpeed: 1.5 });
  world.add(rain.lines);
  var motes = particleField({ count: small ? 300 : 700, box: [60, 30, 60], fall: [-1.4, -0.4], size: 0.35, color: '#fff0c8',
                              map: softSprite('rgba(255,245,215,1)', 'rgba(255,225,170,0)'), sway: 0.6, windSpeed: 0.4 });
  motes.points.material.blending = THREE.AdditiveBlending;
  world.add(motes.points);

  // ── A flash of white-gold to carry you home ──
  var flash = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), new THREE.MeshBasicMaterial({ color: '#fff6e6', transparent: true, opacity: 0,
                                                                                         depthTest: false, depthWrite: false, fog: false }));
  flash.position.z = -0.3;
  flash.renderOrder = 999;
  camera.add(flash);

  var cur = {};
  COLOR_KEYS.forEach(function (k) { cur[k] = new THREE.Color(); });
  function blend(L) {
    L = clamp(L, 0, LOOKS.length - 1);
    var i0 = Math.min(Math.floor(L), LOOKS.length - 2), t = L - i0, a = LOOKS[i0], b = LOOKS[i0 + 1];
    for (var k = 0; k < COLOR_KEYS.length; k++) cur[COLOR_KEYS[k]].copy(a[COLOR_KEYS[k]]).lerp(b[COLOR_KEYS[k]], t);
    for (k = 0; k < NUM_KEYS.length; k++) cur[NUM_KEYS[k]] = lerp(a[NUM_KEYS[k]], b[NUM_KEYS[k]], t);
  }

  var look = new THREE.Vector3(), tmp = new THREE.Vector3(), tmp2 = new THREE.Vector3(), down = new THREE.Vector3();
  var HIGH = new THREE.Vector3(0, 70, -60), OUTSIDE = new THREE.Vector3(0, 1.95, -2.4), rainAt = new THREE.Vector3();
  var portrait = false, H = 800;

  function frame(f) {
    var row = f.row, time = f.time, way = f.cam, slow = env.reduceMotion;
    var yaw = row[4], pitch = row[5], memAmt = row[6], guideAmt = row[7], lead = row[8], tearT = row[9], gateAmt = row[10];
    var candleAmt = row[11], flashAmt = row[12], moteAmt = row[13], nearAmt = row[14];
    blend(row[1]);

    // ── Camera along the way ──
    var n = WAY.length - 1, t = clamp(way / n, 0, 1);
    curve.getPoint(t, camera.position);
    curve.getPoint(Math.min(t + 0.012, 1), look);
    if (t > 0.985) look.z -= 40;
    look.y = camera.position.y + (look.y - camera.position.y) * 0.3;
    camera.position.y += Math.sin(time * 0.6) * 0.01 * (1 + smooth(2, 4, way) * 20);
    // A phone is narrow: stand back from the window so the sill and candle stay in shot.
    if (portrait) { camera.position.z += 0.5 * (1 - smooth(0.5, 1.1, way)); look.z += 0.5 * (1 - smooth(0.5, 1.1, way)); }
    camera.lookAt(look);
    var gold = smooth(2.3, 3, row[1]) * (1 - smooth(3.2, 3.6, row[1]));
    var goldTurn = gold;
    camera.rotateY(yaw - (portrait ? 0.16 * goldTurn : 0) - f.mx * 0.1);
    camera.rotateX(pitch - f.my * 0.05);
    sky.position.copy(camera.position);

    // ── Light for the look ──
    // On a phone the verse spans the middle: lift the eternal sun above it.
    sunDir.set(cur.az, Math.sin(cur.el + (portrait ? 0.2 * goldTurn : 0)), -1).normalize();
    dome.uniforms.sunDir.value.copy(sunDir);
    dome.uniforms.top.value.copy(cur.top);
    dome.uniforms.mid.value.copy(cur.mid);
    dome.uniforms.horizon.value.copy(cur.horizon);
    dome.uniforms.sunColor.value.copy(cur.glow).multiplyScalar(cur.glowAmt);
    sun.position.copy(camera.position).addScaledVector(sunDir, 200);
    sun.target.position.copy(camera.position);
    sun.color.copy(cur.glow);
    sun.intensity = cur.sunI;
    hemi.color.copy(cur.hemiSky);
    hemi.groundColor.copy(cur.hemiGnd);
    hemi.intensity = cur.hemiI;
    world.fog.color.copy(cur.fog);
    world.fog.density = cur.fogD;
    gl.setClearColor(cur.fog);
    gl.toneMappingExposure = cur.exp + flashAmt * 0.3;

    cloudU.uLit.value.copy(cur.lit);
    cloudU.uShade.value.copy(cur.shade);
    cloudU.uGlow.value.copy(cur.glow).multiplyScalar(cur.glowAmt);
    cloudU.uFog.value.copy(cur.fog);
    cloudU.uFogD.value = cur.fogD;
    cloudU.uTime.value = time;
    var morning = smooth(3.5, 3.9, row[1]);
    sea.visible = morning < 0.99;
    for (var p = 0; p < puffs.length; p++) {
      var pu = puffs[p];
      pu.material.color.copy(cur.shade).lerp(cur.lit, pu.position.y > CLOUD_Y ? 0.6 : 0.35 + 0.4 * (pu.position.y - 520) / 110);
      pu.material.opacity = pu.userData.base * (1 - morning);
    }

    // ── The window: rain on the glass, the candle ──
    glassU.uTime.value = slow ? time * 0.5 : time;
    glassU.uRain.value = f.snow;
    glassU.uTint.value.copy(cur.mid).lerp(cur.horizon, 0.3).lerp(tmpC.set('#ffffff'), 0.35).multiplyScalar(1.15);
    glassU.uSun.value = smooth(0.5, 1, cur.glowAmt) * 0.5;
    var flick = 0.92 + 0.08 * Math.sin(time * 9.1) * Math.sin(time * 5.3 + 1);
    flame.scale.set(0.045 * candleAmt * flick, 0.09 * candleAmt * (0.95 + 0.1 * Math.sin(time * 7)), 1);
    flame.material.opacity = candleAmt;
    halo.scale.setScalar(0.55 * flick);
    halo.material.opacity = candleAmt * 0.55;
    candleLight.intensity = candleAmt * 2.2 * flick;
    // The room glows to whoever looks back at it.
    var outside = smooth(1.2, 2.5, way);
    pane.material.opacity = outside * 0.9;
    interior.intensity = outside * 6;
    paneGlow.material.opacity = outside * 0.8;
    litMat.opacity = outside;
    paneGlow.scale.setScalar(4 + smooth(3, 5, way) * 10);

    // ── Rain outside, kept out of the room ──
    rainAt.copy(camera.position);
    if (camera.position.z > -1) rainAt.z = -15;
    rain.update(f, rainAt, f.snow * (1 - smooth(2.5, 4, way)) + 0.0, slow);
    rain.lines.visible = f.snow > 0.01 && way < 4;

    // ── Memories ──
    memMat.uniforms.uTime.value = time;
    memMat.uniforms.uAmt.value = memAmt;
    memMat.uniforms.uAway.value = smooth(5.9, 6.9, way);
    memMat.uniforms.uScale.value = Math.min(window.devicePixelRatio || 1, 2) * H / 800;
    memories.visible = memAmt > 0.01;
    nearOrb.material.opacity = nearAmt;
    nearOrb.visible = nearAmt > 0.01;
    nearOrb.position.set(-1.6 + nearAmt * 0.3, 0.2 + Math.sin(time * 0.8) * 0.05, -4.5 + nearAmt * 1.0);
    nearOrb.scale.setScalar(1.3 + 0.1 * Math.sin(time * 1.3));

    // ── The guide: down from the sky to the window, then ahead of you ──
    if (lead <= 1) guide.position.lerpVectors(HIGH, OUTSIDE, smooth(0, 1, lead));
    else {
      curve.getPoint(clamp((way + 0.9) / n, 0, 1), tmp);
      tmp.y += 1.5;
      guide.position.lerpVectors(OUTSIDE, tmp, smooth(1, 2, lead));
    }
    var pulse = 0.9 + 0.1 * Math.sin(time * 2.2);
    guideCore.material.opacity = guideAmt;
    guideHalo.material.opacity = guideAmt * 0.6 * pulse;
    guideHalo.scale.setScalar(3 + 1.5 * pulse + smooth(2, 6, way) * 10);
    guideCore.scale.setScalar(0.8 + smooth(2, 6, way) * 3);
    guideLight.intensity = guideAmt * 40;
    guide.visible = guideAmt > 0.01;

    // ── The tear: a drop of light falling from just in front of you ──
    tear.visible = tearT > 0.01 && tearT < 0.99;
    if (tear.visible) {
      camera.getWorldDirection(tmp2);
      tear.position.copy(camera.position).addScaledVector(tmp2, 3);
      down.set(0, -1, 0);
      tear.position.addScaledVector(down, 0.2 + tearT * tearT * 3.5);
      tear.scale.set(0.12, 0.18 + tearT * 0.3, 1);
      tear.material.opacity = smooth(0, 0.08, tearT) * (1 - smooth(0.8, 1, tearT));
    }

    // ── The gate and the radiance beyond ──
    gold *= smooth(70, 320, camera.position.distanceTo(GATE));      // dissolving as you pass through
    archMat.opacity = gold * (0.35 + gateAmt * 0.65);
    for (var g = 0; g < archGlows.length; g++) archGlows[g].material.opacity = gold * (0.2 + gateAmt * 0.6) * (0.85 + 0.15 * Math.sin(time * 1.5 + g));
    veil.material.opacity = gold * gateAmt * 0.5;
    gate.visible = gold > 0.01;
    radiance.position.copy(sunDir).multiplyScalar(1400);
    radiance.material.opacity = gold * (0.25 + gateAmt * 0.6);
    radiance.material.rotation = time * 0.01;
    radiance.visible = gold > 0.01;

    // ── Lights rising out of the clouds ──
    pf.snow = moteAmt; pf.wind = f.wind; pf.dt = f.dt; pf.time = time;
    motes.update(pf, camera.position, slow);
    motes.points.visible = moteAmt > 0.01;
    motes.points.material.opacity = 0.8;

    hutLights.material.opacity = 1 - smooth(0.6, 1.4, row[1]) * 0.6;
    flash.material.opacity = flashAmt;
    flash.visible = flashAmt > 0.001;

    gl.render(world, camera);
  }
  var pf = { snow: 0, wind: 0, dt: 0, time: 0 };

  return {
    resize: function (w, h, dpr) { H = h; portrait = w < h; fitCamera(gl, camera, w, h, dpr, small); },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('tomorrow-light', {
  renderer: renderer3d,
  scrim: 0.62,
  align: ['left', 'right', 'left', 'right', 'left', 'right', 'left', 'right', 'center', 'center', 'left', 'left', 'left', 'left', 'left', 'left'],
  // One panel per stanza (16). Each row is pinned to the stanza it answers.
  keys: function (T) {
    function at(i, d) { return T.start(i) + d; }   // d units into stanza i (0..1.6)
    var P = Math.PI;
    //  unit          way    look  rain wind  yaw    pitch  mem  guide lead tear gate candle flash motes near
    return [
      [0,             0.00,  0.00, 1.0, 0.10, 0.00,  0.00, 0,   0,   0,   0,   0,   0,     0,    0,    0],
      [0.7,           0.02,  0.00, 1.0, 0.10, 0.00,  0.00, 0,   0,   0,   0,   0,   0,     0,    0,    0],
      [at(0, 0.8),    0.10,  0.35, 1.0, 0.10, 0.00,  0.01, 0,   0,   0,   0,   0,   0,     0,    0,    0],  // "if the sun should rise"
      [at(1, 0.8),    0.22,  0.45, 0.9, 0.10, 0.00,  0.00, 0,   0,   0,   0,   0,   0,     0,    0,    0],  // "the way you did today"
      [at(2, 0.8),    0.34,  0.60, 0.6, 0.10, 0.00,  0.01, 0,   0,   0,   0,   0,   0,     0,    0,    0],  // "I know how much you love me"
      [at(3, 0.2),    0.40,  0.65, 0.5, 0.10, 0.00,  0.03, 0,   0.3, 0,   0,   0,   0,     0,    0,    0],
      [at(3, 0.6),    0.45,  0.70, 0.4, 0.10, 0.00,  0.04, 0,   1,   1,   0,   0,   0,     0,    0,    0],  // "an angel came and called my name"
      [at(3, 0.95),   1.10,  0.75, 0.3, 0.10, 0.00,  0.05, 0,   1,   1.4, 0,   0,   0,     0,    0,    0],  // "took me by the hand"
      [at(3, 1.4),    2.40,  0.85, 0.2, 0.10, 0.00,  0.35, 0,   1,   2,   0,   0,   0,     0,    0,    0],  // "in heaven far above"
      [at(4, 0.25),   2.60,  0.90, 0.1, 0.10, P - 0.1, -0.20, 0, 1,  2,   0,   0,   0,     0,    0,    0],  // "as I turned to walk away"
      [at(4, 0.4),    2.70,  0.90, 0.1, 0.10, P - 0.1, -0.22, 0, 1,  2,   0.05, 0,  0,     0,    0,    0],
      [at(4, 1.0),    2.95,  0.92, 0.0, 0.10, P - 0.1, -0.30, 0, 1,  2,   0.98, 0,  0,     0,    0,    0],  // "a tear fell from my eye"
      [at(5, 0.4),    3.30,  0.95, 0.0, 0.10, P - 0.05, -0.38, 0, 1, 2,   1,   0,   0,     0,    0,    0],  // "so much left yet to do"
      [at(5, 1.3),    4.00,  1.00, 0.0, 0.10, P,     -0.45, 0,   1,   2,   1,   0,   0,     0,    0,    0],  // "that I was leaving you"
      [at(6, 0.3),    4.50,  1.00, 0.0, 0.15, 0.10,  0.05, 0.6, 0.6, 2,   1,   0,   0,     0,    0,    0],  // "all the yesterdays"
      [at(6, 1.0),    5.00,  1.00, 0.0, 0.15, 0.00,  0.08, 1.0, 0.4, 2,   1,   0,   0,     0,    0,    0],
      [at(7, 0.5),    5.50,  1.05, 0.0, 0.15, -0.05, 0.08, 1.0, 0.3, 2,   1,   0,   0,     0,    0,    1],  // "and maybe see you smile"
      [at(7, 1.3),    5.90,  1.10, 0.0, 0.15, 0.00,  0.10, 1.0, 0.3, 2,   1,   0,   0,     0,    0,    0.3],
      [at(8, 0.7),    6.40,  1.40, 0.0, 0.20, 0.00,  0.12, 0.4, 0.2, 2,   1,   0,   0,     0,    0,    0],  // "emptiness and memories"
      [at(8, 1.5),    6.85,  2.00, 0.0, 0.25, 0.00,  0.12, 0.0, 0.0, 2,   1,   0,   0,     0,    0,    0],  // into the cloud
      [at(9, 1.2),    7.40,  2.00, 0.0, 0.30, 0.00,  0.10, 0,   0,   2,   1,   0,   0,     0,    0,    0],  // "my heart was filled with sorrow"
      [at(10, 0.3),   7.95,  3.00, 0.0, 0.20, 0.00,  0.04, 0,   0,   2,   1,   0,   0,     0,    0,    0],  // "through heaven's gates"
      [at(10, 0.75),  8.50,  3.00, 0.0, 0.15, 0.00,  0.05, 0,   0,   2,   1,   1,   0,     0,    0,    0],  // the gates open
      [at(10, 1.4),   9.45,  3.00, 0.0, 0.10, 0.00,  0.12, 0,   0,   2,   1,   1,   0,     0,    0,    0],  // "His great golden throne"
      [at(11, 0.6),   10.0,  3.00, 0.0, 0.10, 0.00,  0.03, 0,   0,   2,   1,   1,   0,     0,    0.2,  0],  // "this is eternity"
      [at(11, 1.4),   10.7,  3.00, 0.0, 0.10, 0.05,  0.00, 0,   0,   2,   1,   1,   0,     0,    0.3,  0],
      [at(12, 1.2),   11.2,  3.00, 0.0, 0.10, -0.08, 0.00, 0,   0,   2,   1,   1,   0,     0,    0.4,  0],  // "today will always last"
      [at(13, 1.2),   11.6,  3.00, 0.0, 0.10, 0.00,  0.06, 0,   0,   2,   1,   1,   0,     0,    1.0,  0],  // "so trusting and so true"
      [at(14, 0.4),   11.8,  3.00, 0.0, 0.10, 0.00,  0.04, 0,   0.8, 2,   1,   1,   0,     0,    1.0,  0],  // "you have been forgiven"
      [at(14, 1.0),   11.95, 3.00, 0.0, 0.10, 0.00,  0.04, 0,   1,   2,   1,   1,   0,     0.3,  1.0,  0],  // "come and take my hand"
      [at(14, 1.5),   12.0,  3.00, 0.0, 0.10, 0.00,  0.04, 0,   1,   2,   1,   1,   0,     1,    1.0,  0],
      [at(14, 1.501), 0.30,  4.00, 0.0, 0.05, 0.00,  0.00, 0,   0,   0,   1,   0,   0.6,   1,    0,    0],  // home, behind the flash
      [at(15, 0.35),  0.30,  4.00, 0.0, 0.05, 0.00,  0.00, 0,   0,   0,   1,   0,   0.8,   0,    0,    0],  // the window, morning, a candle
      [at(15, 1.2),   0.38,  4.00, 0.0, 0.05, 0.03,  -0.03, 0,  0,   0,   1,   0,   1,     0,    0,    0],  // "I'm right here, in your heart"
      [T.total,       0.44,  4.00, 0.0, 0.05, 0.05,  -0.05, 0,  0,   0,   1,   0,   1,     0,    0,    0]
    ];
  },
  sound: {
    src: '/audio/wind.mp3',
    label: 'Play the wind and the light',
    volume: function (row) { return 0.04 + 0.22 * smooth(1.5, 2.5, row[1]) * (1 - smooth(3.6, 4, row[1])) + 0.08 * row[2]; },
    cues: [{ stanza: 10, at: 0.6, play: gateChime }]
  }
});
