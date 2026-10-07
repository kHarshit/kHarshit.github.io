/*
 * Scene for "If—" (Rudyard Kipling): a climb up a mountain ridge marked by
 * small cairns, to the cairn on the summit. Each stanza is split in two.
 *
 * I    "Keep your head when all about you are losing theirs": a storm on the
 *      lower ridge, cloud whirling round you while the view stays level.
 * ·    "Wait and not be tired by waiting": you shelter by a boulder while the
 *      storm blows itself out and the day goes; the summit appears above.
 * II   "If you can dream": a glide up the ridge under the stars to the
 *      summit; "triumph and disaster ... just the same": a golden sunburst
 *      and a lightning flash, equal in strength and length, and you don't
 *      turn your head to either.
 * ·    "Watch the things you gave your life to, broken": the cairn tumbles;
 *      "stoop and build 'em up": stone by stone it goes back up.
 * III  "One heap of all your winnings ... one turn of pitch-and-toss": a coin
 *      spins up off the cairn and falls away into the dark; "and lose, and
 *      start again at your beginnings": you slide all the way back down.
 * ·    "Hold on": the knife-edge before dawn in a gale, spindrift tearing past.
 * IV   "Talk with crowds ... walk with kings": above the valley mist at blue
 *      hour, town lights below, the high peaks all round.
 * ·    "The unforgiving minute": sixty lights tick on in a ring round the
 *      summit cairn as you run the last of the ridge; then sunrise, and
 *      "yours is the Earth" spread out below.
 *
 * Columns: [unit, path, time of day (0 storm, 1 dusk, 2 night, 3 before dawn,
 *           4 blue hour, 5 sunrise), snow, wind, yaw, pitch, storm, eye,
 *           gold, flash, fall, rebuild, coin, ring, shake, near]
 */
import { THREE, isSmall, makeRenderer, fitCamera, softSprite, skyDome, starField, scatter,
         particleField, rainField, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── The mountain (metres; you climb towards -z) ─────────────────────────
// A long ridge rises from Z0 to the summit at ZS: broad at the foot, a
// knife-edge in the middle, then a steep drop on the far side into a wide
// valley with a river and towns, ringed by higher peaks except straight
// ahead, where the land opens out towards the sunrise.
var Z0 = 150, ZS = -110;
function spine(z) { return 10 * Math.sin(z * 0.012 + 1.0); }
function addc(c, d, s) { c.r += d.r * s; c.g += d.g * s; c.b += d.b * s; return c; }
function smin(a, b, k) { var h = clamp(0.5 + 0.5 * (b - a) / k, 0, 1); return lerp(b, a, h) - k * h * (1 - h); }
function smax(a, b, k) { return -smin(-a, -b, k); }
function ridgeT(z) { return clamp((Z0 - z) / (Z0 - ZS), 0, 1); }

function crest(z) {
  var t = Math.max((Z0 - z) / (Z0 - ZS), 0);
  return smin(8 + 112 * (0.55 * t + 0.45 * t * t), 120 + 0.9 * (z - ZS), 18);
}
function rough(x, z) {
  return 2.4 * Math.sin(x * 0.21 + z * 0.17) * Math.sin(z * 0.29 - x * 0.13) + 1.3 * Math.sin(x * 0.53 - z * 0.41 + 1.7) +
         0.6 * Math.sin(x * 1.1 + z * 0.9);
}
function knifeAt(t) { return smooth(0.22, 0.34, t) * (1 - smooth(0.58, 0.7, t)); }
function ridge(x, z) {
  var t = ridgeT(z), d = Math.abs(x - spine(z)), knife = knifeAt(t);
  var k = 0.4 + 0.95 * knife + 0.35 * smooth(0.6, 0.85, t), c = 2.6 - 1.5 * knife;
  return crest(z) - (Math.sqrt(d * d + c * c) - c) * k * (1 + d * 0.006) + rough(x, z) * smooth(1.5, 12, d);
}
function river(z) { return 260 * Math.sin(z * 0.0021 + 0.7) + 110 * Math.sin(z * 0.0053 + 2.0) - 150; }
function valley(x, z) {
  var h = -74 + 4 * Math.sin(x * 0.011) * Math.cos(z * 0.009) + 2.5 * Math.sin(x * 0.031 + z * 0.023);
  var hills = Math.max(Math.sin(x * 0.0047 + 0.5) * Math.sin(z * 0.0051 + 1.2), 0);
  h += 70 * hills * hills;
  // The kings: a ring of high peaks, low straight ahead (-z).
  var rr = Math.hypot(x, z + 250), a = Math.atan2(z + 250, x);
  var peaks = 0.4 + 0.4 * (1 - Math.abs(Math.sin(a * 3.5 + 0.4))) + 0.2 * (1 - Math.abs(Math.sin(a * 8.3 + 1.1))) + 0.05 * Math.sin(a * 23);
  var da = Math.atan2(Math.sin(a + Math.PI / 2), Math.cos(a + Math.PI / 2));
  h += smooth(650, 1500, rr) * 300 * peaks * (1 - 0.8 * Math.exp(-da * da / 0.45));
  var dr = Math.abs(x - river(z));
  return lerp(h, -84, smooth(110, 18, dr) * (1 - smooth(1000, 1400, rr)));
}
function height(x, z) { return smax(valley(x, z), ridge(x, z), 16); }

// The trail along the crest, and where things stand on it.
var trail = [];
for (var tz = Z0 + 4; tz > ZS + 8; tz -= 8) trail.push(new THREE.Vector3(spine(tz), 0, tz));
trail.push(new THREE.Vector3(spine(ZS + 5), 0, ZS + 5));
var PATH = new THREE.CatmullRomCurve3(trail);
var SUMMIT = new THREE.Vector3(spine(ZS - 3), 0, ZS - 3);
SUMMIT.y = height(SUMMIT.x, SUMMIT.z);

// ── Procedural pieces ───────────────────────────────────────────────────
// A dense grid in the middle, coarse at the edges, so the ridge is detailed
// and the valley and peaks still reach the horizon.
function warpedTerrain(R, seg, cx, cz, colorFn, material) {
  var geo = new THREE.PlaneGeometry(2, 2, seg, seg).rotateX(-Math.PI / 2);
  var p = geo.attributes.position, cols = new Float32Array(p.count * 3);
  function w(a) { return R * (0.22 * a + 0.78 * a * a * a); }
  for (var i = 0; i < p.count; i++) {
    var x = cx + w(p.getX(i)), z = cz + w(p.getZ(i)), y = height(x, z), c = colorFn(x, z, y);
    p.setXYZ(i, x, y, z);
    cols[i * 3] = c.r; cols[i * 3 + 1] = c.g; cols[i * 3 + 2] = c.b;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  geo.computeVertexNormals();
  return new THREE.Mesh(geo, material);
}

// A rough stone: an icosahedron with its corners pushed in and out (by
// position, so shared corners move together and the faces stay closed).
function stoneGeometry(seed) {
  var geo = new THREE.IcosahedronGeometry(1, 1), p = geo.attributes.position, v = new THREE.Vector3();
  for (var i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    var n = Math.sin(v.x * 12.9898 + v.y * 78.233 + v.z * 37.719 + seed) * 43758.5453;
    v.multiplyScalar(0.82 + 0.3 * (n - Math.floor(n)));
    p.setXYZ(i, v.x, v.y, v.z);
  }
  geo.computeVertexNormals();
  return geo;
}

function canvasTex(size, paint) {
  var c = document.createElement('canvas');
  c.width = c.height = size;
  paint(c.getContext('2d'), size);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// A golden sunburst: a soft core and rays.
function burstTexture() {
  return canvasTex(256, function (x, s) {
    x.globalCompositeOperation = 'lighter';
    for (var i = 0; i < 28; i++) {
      var a = i / 28 * Math.PI * 2, w = i % 2 ? 0.035 : 0.06, len = s * (i % 2 ? 0.36 : 0.5);
      var g = x.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, len);
      g.addColorStop(0, 'rgba(255,220,140,0.55)');
      g.addColorStop(1, 'rgba(255,190,90,0)');
      x.fillStyle = g;
      x.beginPath();
      x.moveTo(s / 2, s / 2);
      x.lineTo(s / 2 + Math.cos(a - w) * len, s / 2 + Math.sin(a - w) * len);
      x.lineTo(s / 2 + Math.cos(a + w) * len, s / 2 + Math.sin(a + w) * len);
      x.fill();
    }
    var c = x.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
    c.addColorStop(0, 'rgba(255,240,200,1)');
    c.addColorStop(0.12, 'rgba(255,210,130,0.8)');
    c.addColorStop(0.45, 'rgba(255,170,80,0.15)');
    c.addColorStop(1, 'rgba(255,150,60,0)');
    x.fillStyle = c;
    x.fillRect(0, 0, s, s);
  });
}

// A forked lightning bolt as flat ribbons in a vertical plane facing +z.
function boltGeometry(r, top, bottom) {
  var pos = [];
  function jag(a, b, depth, width, out) {
    if (depth === 0) { out.push(a, b); return; }
    var m = new THREE.Vector3().lerpVectors(a, b, 0.5), len = a.distanceTo(b);
    m.x += (r() - 0.5) * len * 0.45;
    jag(a, m, depth - 1, width, out);
    jag(m, b, depth - 1, width, out);
  }
  function strip(a, b, width) {
    var segs = [];
    jag(a, b, 6, width, segs);
    for (var i = 0; i < segs.length; i += 2) {
      var p = segs[i], q = segs[i + 1], dx = q.x - p.x, dy = q.y - p.y, l = Math.hypot(dx, dy) || 1;
      var nx = -dy / l * width, ny = dx / l * width;
      pos.push(p.x - nx, p.y - ny, 0, q.x - nx, q.y - ny, 0, q.x + nx, q.y + ny, 0,
               p.x - nx, p.y - ny, 0, q.x + nx, q.y + ny, 0, p.x + nx, p.y + ny, 0);
    }
  }
  strip(top, bottom, 2.2);
  var mid = new THREE.Vector3().lerpVectors(top, bottom, 0.45);
  strip(mid, new THREE.Vector3(mid.x - 70, mid.y - 110, 0), 1.2);
  var low = new THREE.Vector3().lerpVectors(top, bottom, 0.7);
  strip(low, new THREE.Vector3(low.x + 50, low.y - 60, 0), 0.9);
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  return geo;
}

// ── Synthesised sound cues ───────────────────────────────────────────────
function noiseBuffer(ac, secs, shape) {
  var b = ac.createBuffer(1, Math.floor(ac.sampleRate * secs), ac.sampleRate), d = b.getChannelData(0);
  for (var i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * (shape ? shape(i / d.length) : 1);
  return b;
}
function thunder(ac, out) {
  var t = ac.currentTime, src = ac.createBufferSource(), lp = ac.createBiquadFilter(), g = ac.createGain();
  src.buffer = noiseBuffer(ac, 3.5, function (k) { return Math.pow(1 - k, 2); });
  lp.type = 'lowpass';
  lp.frequency.setValueAtTime(800, t);
  lp.frequency.exponentialRampToValueAtTime(110, t + 1.4);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.8, t + 0.06);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 3.4);
  src.connect(lp); lp.connect(g); g.connect(out);
  src.start(t);
}
// Stones knocking together as the cairn falls.
function clatter(ac, out) {
  for (var i = 0; i < 11; i++) {
    var t = ac.currentTime + i * 0.09 + Math.random() * 0.07, src = ac.createBufferSource(), bp = ac.createBiquadFilter(), g = ac.createGain();
    src.buffer = noiseBuffer(ac, 0.12, function (k) { return Math.pow(1 - k, 6); });
    bp.type = 'bandpass';
    bp.frequency.value = 500 + Math.random() * 1400;
    bp.Q.value = 4;
    g.gain.value = 0.5 * (1 - i / 14);
    src.connect(bp); bp.connect(g); g.connect(out);
    src.start(t);
  }
}
// A flipped coin: a bright metallic ring that wavers as it spins.
function coinRing(ac, out) {
  var t = ac.currentTime;
  [2637, 3952, 5274].forEach(function (f, j) {
    var o = ac.createOscillator(), g = ac.createGain(), lfo = ac.createOscillator(), lg = ac.createGain();
    o.type = 'sine';
    o.frequency.value = f * (1 + j * 0.003);
    lfo.frequency.value = 9;
    lg.gain.value = 0.03 / (j + 1);
    lfo.connect(lg); lg.connect(g.gain);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.06 / (j + 1), t + 0.005);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 1.6 / (j + 1));
    o.connect(g); g.connect(out);
    o.start(t); lfo.start(t);
    o.stop(t + 1.7); lfo.stop(t + 1.7);
  });
}

// ── Light through the day: storm, dusk, night, before dawn, blue hour, sunrise
var TOD = [
  { top: '#4c5464', mid: '#6a7484', hor: '#8e96a3', fog: '#7c8694', sky: '#a4aec0', gnd: '#2c2c32', hemi: 1.25,
    key: '#d0d6e2', keyI: 0.5, dir: [0.3, 0.8, 0.4], glow: 0, stars: 0, density: 0.03 },
  { top: '#1c2646', mid: '#4a4a6a', hor: '#c47e5e', fog: '#56526a', sky: '#8a86a8', gnd: '#221e26', hemi: 0.75,
    key: '#ffa070', keyI: 1.3, dir: [1, 0.1, 0.25], glow: 0.55, stars: 0.12, density: 0.004 },
  { top: '#01030a', mid: '#06102a', hor: '#16243f', fog: '#0c1527', sky: '#4a5a8c', gnd: '#05070c', hemi: 0.5,
    key: '#a4b8ff', keyI: 0.7, dir: [-0.5, 0.75, 0.45], glow: 0, stars: 1, density: 0.0016 },
  { top: '#030716', mid: '#0c1838', hor: '#2c3762', fog: '#151d37', sky: '#5a6aa4', gnd: '#080a12', hemi: 0.75,
    key: '#9ab0f0', keyI: 0.75, dir: [1, 0.45, 0.1], glow: 0, stars: 0.75, density: 0.002 },
  { top: '#132148', mid: '#36487c', hor: '#d28c86', fog: '#4c5478', sky: '#8e9cd0', gnd: '#1a1a28', hemi: 0.95,
    key: '#ffb4a0', keyI: 0.55, dir: [0.25, -0.03, -1], glow: 0.35, stars: 0.25, density: 0.0011 },
  { top: '#36589a', mid: '#94acd2', hor: '#ffcf96', fog: '#d6b49c', sky: '#c8d4f0', gnd: '#3a3028', hemi: 1.2,
    key: '#ffc890', keyI: 2.8, dir: [0.25, 0.07, -1], glow: 1, stars: 0, density: 0.0008 }
].map(function (p) {
  ['top', 'mid', 'hor', 'fog', 'sky', 'gnd', 'key'].forEach(function (k) { p[k] = new THREE.Color(p[k]); });
  p.dir = new THREE.Vector3().fromArray(p.dir).normalize();
  return p;
});

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(1910);
  var gl = makeRenderer(canvas, { clear: '#7c8694' });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#7c8694', 0.03);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.1, 5000);
  camera.rotation.order = 'YXZ';

  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#4c5464', mid: '#6a7484', horizon: '#8e96a3', sun: '#ffc890' }, 1500);
  sky.add(dome.mesh);
  var stars = starField(r, small ? 2200 : 4500, 1300, -0.02, 1.5);
  sky.add(stars);

  var hemi = new THREE.HemisphereLight('#a4aec0', '#2c2c32', 1.2);
  var key = new THREE.DirectionalLight('#ffffff', 0.5);
  world.add(hemi, key, key.target);

  // Snow on the gentler slopes high up, rock where it is steep, scree and
  // alpine grass lower down, then fields, woods and the river in the valley.
  var C = {
    snow: new THREE.Color('#e6ecf5'), snowShade: new THREE.Color('#c4cfe0'), rock: new THREE.Color('#3e3b3b'),
    rock2: new THREE.Color('#5a544e'), scree: new THREE.Color('#7a7268'), alp: new THREE.Color('#6a6a44'),
    field: new THREE.Color('#6c7c3a'), field2: new THREE.Color('#8f8a48'), wood: new THREE.Color('#2e3e28'),
    bed: new THREE.Color('#6f8296')
  };
  var col = new THREE.Color(), col2 = new THREE.Color();
  var terrainMesh = warpedTerrain(1900, small ? 220 : 320, 0, 20, function (x, z, y) {
    var sx = height(x + 1.5, z) - height(x - 1.5, z), sz = height(x, z + 1.5) - height(x, z - 1.5);
    var slope = Math.hypot(sx, sz) / 3;
    var n = 0.5 + 0.5 * Math.sin(x * 0.07 + Math.sin(z * 0.05) * 2) * Math.cos(z * 0.06 - x * 0.03);
    // Lowland: a patchwork of fields with dark woods.
    var cells = Math.sin(x * 0.021 + 0.3) * Math.sin(z * 0.019 + 1.1) + 0.5 * Math.sin(x * 0.05 - z * 0.043);
    col.copy(C.field2).lerp(C.field, smooth(0, 0.4, cells));
    col.lerp(C.wood, 0.85 * smooth(0.25, 0.6, Math.sin(x * 0.009 + 2) * Math.cos(z * 0.011) + 0.4 * Math.sin(x * 0.033 + z * 0.027)));
    col.lerp(C.bed, smooth(-79, -83, y));
    // Up the mountain: alpine grass, scree, rock, snow.
    col.lerp(C.alp, smooth(-60, -45, y));
    col.lerp(C.scree, smooth(-30, -5, y) * (0.5 + 0.5 * n));
    var rockAmt = smooth(0.62, 1.0, slope) * smooth(-55, -35, y);
    col.lerp(col2.copy(C.rock).lerp(C.rock2, n), rockAmt);
    var line = -20 + n * 22 + Math.max(0, valley(x, z) + 74) * 0.25;
    var snow = smooth(line - 6, line + 6, y) * (1 - smooth(0.7, 1.05, slope));
    col.lerp(col2.copy(C.snow).lerp(C.snowShade, n * 0.6), snow);
    return col;
  }, new THREE.MeshLambertMaterial({ vertexColors: true }));
  world.add(terrainMesh);

  // Boulders on the flanks, and a big one to shelter behind.
  var stoneGeo = stoneGeometry(3.1);
  var rocks = new THREE.InstancedMesh(stoneGeo, new THREE.MeshLambertMaterial({ flatShading: true }), small ? 260 : 520);
  var up = new THREE.Vector3(0, 1, 0), e = new THREE.Euler();
  scatter(rocks, 5000, function (i, p, q, s, c) {
    var z = Z0 + 20 - r() * (Z0 - ZS + 60), side = r() < 0.5 ? -1 : 1, d = 2.2 + Math.pow(r(), 1.6) * 40;
    var x = spine(z) + side * d, sz = 0.3 + Math.pow(r(), 2) * (d < 5 ? 0.9 : 2.6);
    p.set(x, height(x, z) - sz * 0.3, z);
    q.setFromEuler(e.set(r() * 3, r() * 3, r() * 3));
    s.set(sz * (0.8 + r() * 0.5), sz * (0.5 + r() * 0.4), sz * (0.8 + r() * 0.5));
    var v = 0.32 + r() * 0.2;
    c.setRGB(v, v * 0.97, v * 0.95);
  });
  world.add(rocks);
  var LEE = PATH.getPointAt(0.2);
  var lee = new THREE.Mesh(stoneGeo, new THREE.MeshLambertMaterial({ color: '#6e6964', flatShading: true }));
  lee.position.set(LEE.x - 5.2, height(LEE.x - 5.2, LEE.z - 2.5) + 0.5, LEE.z - 2.5);
  lee.scale.set(2.3, 1.8, 2.0);
  lee.rotation.set(0.3, 0.8, 0.1);
  world.add(lee);

  // Little cairns mark the way up.
  var marks = new THREE.InstancedMesh(stoneGeo, new THREE.MeshLambertMaterial({ flatShading: true }), 9 * 4);
  var m4 = new THREE.Matrix4(), q4 = new THREE.Quaternion(), s4 = new THREE.Vector3(), p4 = new THREE.Vector3(), mi = 0;
  [0.06, 0.15, 0.28, 0.4, 0.52, 0.63, 0.74, 0.85].forEach(function (t, k) {
    var a = PATH.getPointAt(t), side = k % 2 ? 1 : -1, x = a.x + side * 1.6, z = a.z, y = height(x, z) - 0.05;
    [[0.24, 0.13], [0.19, 0.11], [0.14, 0.09], [0.09, 0.07]].forEach(function (st) {
      q4.setFromEuler(e.set((r() - 0.5) * 0.3, r() * 6, (r() - 0.5) * 0.3));
      marks.setMatrixAt(mi, m4.compose(p4.set(x + (r() - 0.5) * 0.05, y + st[1], z), q4, s4.set(st[0], st[1], st[0] * 0.9)));
      var v = 0.38 + r() * 0.15;
      marks.setColorAt(mi++, col.setRGB(v, v, v * 0.96));
      y += st[1] * 1.7;
    });
  });
  marks.count = mi;
  world.add(marks);

  // The summit cairn: sixteen stones, each with a place in the stack and a
  // place on the ground where it lands when the cairn falls.
  var cairn = [], stoneMats = ['#7c766e', '#8a8379', '#6c6862'].map(function (c) { return new THREE.MeshLambertMaterial({ color: c, flatShading: true, emissive: '#10131c' }); });
  var geos = [stoneGeometry(1.7), stoneGeometry(5.3), stoneGeometry(8.9)];
  var cy = SUMMIT.y - 0.06;
  [5, 4, 3, 2, 1, 1].forEach(function (n, L) {
    var sz = 0.36 - L * 0.05, hh = sz * 0.55, rad = n > 1 ? sz * 0.95 : 0;
    for (var j = 0; j < n; j++) {
      var a = j / n * Math.PI * 2 + L * 0.6, k = cairn.length;
      var m = new THREE.Mesh(geos[k % 3], stoneMats[k % 3]);
      var home = new THREE.Vector3(SUMMIT.x + Math.cos(a) * rad, cy + hh, SUMMIT.z + Math.sin(a) * rad);
      var fa = r() * Math.PI * 2, fd = 0.7 + r() * 1.6, fx = SUMMIT.x + Math.cos(fa) * fd, fz = SUMMIT.z + Math.sin(fa) * fd;
      m.userData = {
        home: home, homeQ: new THREE.Quaternion().setFromEuler(e.set((r() - 0.5) * 0.15, r() * 6, (r() - 0.5) * 0.15)),
        down: new THREE.Vector3(fx, height(fx, fz) + hh * 0.6, fz),
        downQ: new THREE.Quaternion().setFromEuler(e.set(r() * 3, r() * 6, r() * 3)),
        at: (5 - L) / 6 * 0.6 + r() * 0.08          // the top stones go first
      };
      m.scale.set(sz * (0.95 + r() * 0.2), hh, sz * (0.85 + r() * 0.2));
      world.add(m);
      cairn.push(m);
    }
    cy += hh * 1.7;
  });
  var CAIRN_TOP = cy;

  // Sixty lights in a ring round it: the unforgiving minute.
  var ringPos = [], ringBase = [], ringCol = new Float32Array(60 * 3);
  for (var i = 0; i < 60; i++) {
    var ra = Math.PI / 2 + i / 60 * Math.PI * 2, rx = SUMMIT.x + Math.cos(ra) * 3.4, rz = SUMMIT.z + Math.sin(ra) * 3.4;
    ringPos.push(rx, height(rx, rz) + 0.18, rz);
    ringBase.push(ringPos[ringPos.length - 2]);
  }
  var ringGeo = new THREE.BufferGeometry();
  ringGeo.setAttribute('position', new THREE.Float32BufferAttribute(ringPos, 3));
  ringGeo.setAttribute('color', new THREE.BufferAttribute(ringCol, 3));
  var ring = new THREE.Points(ringGeo, new THREE.PointsMaterial({ size: 0.24, vertexColors: true, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, map: softSprite('rgba(255,236,190,1)', 'rgba(255,200,120,0)') }));
  world.add(ring);

  // The coin.
  var coin = new THREE.Group();
  var coinMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.03, 28),
    new THREE.MeshPhongMaterial({ color: '#d8a640', specular: '#fff0b0', shininess: 80, emissive: '#5a3a08' }));
  coinMesh.rotation.x = Math.PI / 2;
  var coinGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,225,150,1)', 'rgba(255,190,90,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
  coinGlow.scale.setScalar(1.5);
  var coinLight = new THREE.PointLight('#ffcf80', 0, 5, 1.6);
  coin.add(coinMesh, coinGlow, coinLight);
  world.add(coin);

  // Town lights on the valley floor.
  var towns = [];
  [[-420, -520, 90], [380, -700, 120], [-160, -1000, 80], [620, -380, 60], [-720, -260, 70], [180, -1320, 100],
   [-560, -1380, 70], [60, -560, 40], [-900, -800, 60], [880, -1000, 70]].forEach(function (tw) {
    var n = small ? Math.round(tw[2] * 0.6) : tw[2];
    for (var k = 0; k < n; k++) {
      var a = r() * Math.PI * 2, d = Math.pow(r(), 0.7) * 70, x = tw[0] + Math.cos(a) * d, z = tw[1] + Math.sin(a) * d, y = height(x, z);
      if (y > -40) continue;
      towns.push(x, Math.max(y, -83) + 1.5, z);
    }
  });
  var townGeo = new THREE.BufferGeometry();
  townGeo.setAttribute('position', new THREE.Float32BufferAttribute(towns, 3));
  var townMat = new THREE.PointsMaterial({ color: '#ffc27a', size: 9, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, map: softSprite('rgba(255,215,150,1)', 'rgba(255,190,110,0)') });
  var townPts = new THREE.Points(townGeo, townMat);
  world.add(townPts);

  // Valley mist: a noisy sheet lying in the low ground.
  var mistMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, fog: false,
    uniforms: { uTime: { value: 0 }, uLit: { value: new THREE.Color() }, uShade: { value: new THREE.Color() }, uAlpha: { value: 0.5 },
                uCam: { value: new THREE.Vector3() } },
    vertexShader: 'varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }',
    fragmentShader: 'uniform float uTime; uniform vec3 uLit; uniform vec3 uShade; uniform float uAlpha; uniform vec3 uCam; varying vec3 vW;\n' +
      'float h(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }\n' +
      'float n(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);\n' +
      ' return mix(mix(h(i), h(i + vec2(1, 0)), f.x), mix(h(i + vec2(0, 1)), h(i + vec2(1, 1)), f.x), f.y); }\n' +
      'float fbm(vec2 p){ float v = 0.0, a = 0.5; for (int i = 0; i < 5; i++){ v += a * n(p); p = p * 2.03 + vec2(1.7, 9.2); a *= 0.5; } return v; }\n' +
      'void main(){ vec2 p = vW.xz * 0.0035 + vec2(uTime * 0.004, uTime * 0.002);\n' +
      ' float m = fbm(p); float a = smoothstep(0.45, 0.8, m) * uAlpha;\n' +
      ' a *= 1.0 - smoothstep(1700.0, 2600.0, distance(vW.xz, uCam.xz));\n' +
      ' gl_FragColor = vec4(mix(uShade, uLit, smoothstep(0.45, 0.85, m)), a);\n #include <colorspace_fragment>\n }'
  });
  var mist = new THREE.Mesh(new THREE.PlaneGeometry(6000, 6000).rotateX(-Math.PI / 2), mistMat);
  mist.position.y = -66;
  mist.renderOrder = 2;
  world.add(mist);

  // Storm: whirling cloud wisps round you, sleet, snow.
  var wispTex = softSprite('rgba(255,255,255,0.75)', 'rgba(255,255,255,0)'), wisps = [];
  for (i = 0; i < (small ? 26 : 44); i++) {
    var w = new THREE.Sprite(new THREE.SpriteMaterial({ map: wispTex, transparent: true, depthWrite: false, opacity: 0 }));
    w.userData = { a: r() * Math.PI * 2, rad: 7 + r() * 26, y: -6 + r() * 16, spd: 0.5 + r() * 0.8, base: 0.25 + r() * 0.35, tint: 0.85 + r() * 0.3 };
    w.scale.set(14 + r() * 18, 8 + r() * 8, 1);
    world.add(w);
    wisps.push(w);
  }
  var sleet = rainField({ count: small ? 900 : 2200, box: [18, 14, 24], speed: 9, windSpeed: 9, color: '#dfe6f0', opacity: 0.28 });
  world.add(sleet.lines);
  var snow = particleField({ count: small ? 1500 : 3500, box: [30, 16, 30], fall: [0.6, 1.5], size: 0.07, color: '#f2f6ff',
                             map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)'), sway: 0.8, windSpeed: 13 });
  world.add(snow.points);

  // Triumph and disaster: a golden sunburst ahead-left, lightning ahead-right,
  // a bank of cloud along the horizon for both to light up.
  var burst = new THREE.Sprite(new THREE.SpriteMaterial({ map: burstTexture(), blending: THREE.AdditiveBlending, depthWrite: false,
    transparent: true, fog: false, opacity: 0 }));
  burst.position.set(-0.42, 0.07, -1).normalize().multiplyScalar(1200);
  burst.scale.setScalar(620);
  sky.add(burst);
  var bolt = new THREE.Mesh(boltGeometry(rng(7), new THREE.Vector3(0, 330, 0), new THREE.Vector3(30, -40, 0)),
    new THREE.MeshBasicMaterial({ color: '#e8eeff', transparent: true, opacity: 0, blending: THREE.AdditiveBlending,
                                  depthWrite: false, fog: false, side: THREE.DoubleSide }));
  bolt.position.set(SUMMIT.x + 440, 0, SUMMIT.z - 1050);
  world.add(bolt);
  var boltGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(220,230,255,1)', 'rgba(160,180,255,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false, opacity: 0 }));
  boltGlow.position.set(bolt.position.x, 300, bolt.position.z);
  boltGlow.scale.setScalar(620);
  world.add(boltGlow);
  var bank = [], bankTex = softSprite('rgba(255,255,255,0.85)', 'rgba(255,255,255,0)');
  for (i = 0; i < 26; i++) {
    var b = new THREE.Sprite(new THREE.SpriteMaterial({ map: bankTex, transparent: true, depthWrite: false, fog: false, opacity: 0 }));
    var ba = -Math.PI / 2 + (r() - 0.5) * 2.2, be = 0.02 + r() * 0.09;
    b.position.set(Math.cos(ba) * Math.cos(be), Math.sin(be), Math.sin(ba) * Math.cos(be)).multiplyScalar(1250);
    b.scale.set(260 + r() * 260, 70 + r() * 60, 1);
    b.userData = { side: Math.cos(ba) };
    sky.add(b);
    bank.push(b);
  }

  // The sun, for dusk and sunrise.
  var sunDisc = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,244,220,1)', 'rgba(255,200,120,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false, opacity: 0 }));
  sunDisc.scale.setScalar(130);
  sky.add(sunDisc);

  // ── Per frame ─────────────────────────────────────────────────────────
  var pos = new THREE.Vector3(), tan = new THREE.Vector3(), ahead = new THREE.Vector3(), dir = new THREE.Vector3();
  var tmp = new THREE.Color(), gold = new THREE.Color('#ffc066'), cold = new THREE.Color('#c8d4ff');
  var yawScale = 1, lastLift = -1, lastPath = 0, speed = 0, step = 0, L = PATH.getLength();
  var P = { top: new THREE.Color(), mid: new THREE.Color(), hor: new THREE.Color(), fog: new THREE.Color(), sky: new THREE.Color(),
            gnd: new THREE.Color(), key: new THREE.Color(), dir: new THREE.Vector3() };

  function palette(tod) {
    var a = TOD[Math.floor(clamp(tod, 0, 4.999))], b = TOD[Math.min(Math.floor(clamp(tod, 0, 4.999)) + 1, 5)], t = clamp(tod, 0, 5) - Math.floor(clamp(tod, 0, 4.999));
    ['top', 'mid', 'hor', 'fog', 'sky', 'gnd', 'key'].forEach(function (k) { P[k].copy(a[k]).lerp(b[k], t); });
    P.dir.copy(a.dir).lerp(b.dir, t).normalize();
    P.hemi = lerp(a.hemi, b.hemi, t); P.keyI = lerp(a.keyI, b.keyI, t); P.glow = lerp(a.glow, b.glow, t);
    P.stars = lerp(a.stars, b.stars, t);
    P.density = Math.exp(lerp(Math.log(a.density), Math.log(b.density), t));
    return P;
  }

  function frame(f) {
    var row = f.row, time = f.time, dt = f.dt, t = clamp(f.cam, 0, 1), tod = row[1], storm = row[6];
    var goldAmt = row[8], flash = row[9], fall = row[10], rebuild = row[11], coinT = row[12], ringT = row[13];
    var calm = env.reduceMotion ? 0 : 1, shake = row[14] * f.wind * calm;

    // On the trail: footsteps bob with your speed; the gale shakes you.
    speed = lerp(speed, Math.abs(t - lastPath) * L / Math.max(dt, 0.001), 1 - Math.exp(-dt * 4));
    lastPath = t;
    step += Math.min(speed, 6) * dt * 4.2;
    PATH.getPointAt(t, pos);
    PATH.getTangentAt(t, tan);
    PATH.getPointAt(Math.min(t + 0.02, 1), ahead);
    var slope = t < 0.98 ? Math.atan2(height(ahead.x, ahead.z) - height(pos.x, pos.z), Math.hypot(ahead.x - pos.x, ahead.z - pos.z)) : 0;
    var bob = Math.sin(step) * Math.min(speed * 0.012, 0.05) * calm;
    camera.position.set(pos.x, height(pos.x, pos.z) + 1.7 + row[7] + bob + Math.sin(time * 1.1) * 0.02, pos.z);
    camera.rotation.set(slope * 0.45 + row[5] + (1 - yawScale) * 0.32 * clamp(row[15] / 0.8, 0, 1) - f.my * 0.07 + Math.sin(time * 7.3) * Math.sin(time * 3.1) * 0.012 * shake,
                        Math.atan2(-tan.x, -tan.z) + row[4] * lerp(yawScale, 1, smooth(0.4, 0.7, Math.abs(row[4]))) - f.mx * 0.14 + Math.sin(time * 5.1) * 0.01 * shake,
                        (Math.sin(time * 9.1) * 0.6 + Math.sin(time * 15.7) * 0.4) * 0.014 * shake);
    var near = row[15];
    if (near > 0) {
      dir.set(SUMMIT.x - camera.position.x, 0, SUMMIT.z - camera.position.z);
      var gap = dir.length();
      camera.position.addScaledVector(dir.normalize(), near * Math.max(gap - 3.6, 0));
      camera.position.y = height(camera.position.x, camera.position.z) + 1.7 + row[7] + bob;
    }
    camera.position.x += Math.sin(time * 11.3) * 0.03 * shake;
    sky.position.copy(camera.position);

    // Light.
    var p = palette(tod);
    var flashCol = tmp.copy(cold).multiplyScalar(flash * 0.7);
    addc(dome.uniforms.top.value.copy(p.top), gold, goldAmt * 0.12).add(flashCol);
    addc(dome.uniforms.mid.value.copy(p.mid), gold, goldAmt * 0.25).add(flashCol);
    addc(dome.uniforms.horizon.value.copy(p.fog).lerp(p.hor, 0.5 + 0.5 * (1 - storm)), gold, goldAmt * 0.35).add(flashCol);
    dome.uniforms.sunDir.value.copy(p.dir);
    dome.uniforms.sunColor.value.set('#ffc890').multiplyScalar(p.glow * (1 - storm) * smooth(0.28, 0.12, p.dir.y));
    addc(addc(world.fog.color.copy(p.fog), gold, goldAmt * 0.12), cold, flash * 0.12);
    world.fog.density = lerp(p.density, 0.032, storm);
    gl.setClearColor(world.fog.color);
    hemi.color.copy(p.sky);
    hemi.groundColor.copy(p.gnd);
    hemi.intensity = p.hemi + goldAmt * 0.9 + flash * 0.9;
    key.color.copy(p.key);
    key.intensity = p.keyI * (1 - storm * 0.7);
    key.position.copy(camera.position).addScaledVector(p.dir, 200);
    key.target.position.copy(camera.position);
    stars.material.opacity = 0.9 * p.stars * (1 - storm);
    sunDisc.position.copy(p.dir).multiplyScalar(1200);
    sunDisc.material.opacity = p.glow * (1 - storm) * smooth(-0.06, 0.02, p.dir.y) * smooth(0.3, 0.15, p.dir.y) * smooth(0.3, 0.55, p.glow);

    // Town lights show in the dark; the mist glows with the dawn.
    townMat.opacity = smooth(0.8, 1.8, tod) * (1 - smooth(4.4, 5, tod)) * (1 - storm);
    mistMat.uniforms.uTime.value = time;
    mistMat.uniforms.uCam.value.copy(camera.position);
    mistMat.uniforms.uLit.value.copy(p.hor).lerp(p.sky, 0.25);
    mistMat.uniforms.uShade.value.copy(p.fog).lerp(p.sky, 0.4);
    mistMat.uniforms.uAlpha.value = 0.3 + 0.35 * smooth(2.5, 4, tod);

    // Triumph and disaster, the same size and the same length.
    burst.material.opacity = goldAmt;
    burst.material.rotation = time * 0.03;
    bolt.material.opacity = flash * (0.75 + 0.25 * Math.sin(time * 60));
    boltGlow.material.opacity = flash * 0.8;
    bank.forEach(function (b) {
      addc(addc(b.material.color.copy(p.fog).multiplyScalar(0.7), gold, goldAmt * Math.max(-b.userData.side, 0.15) * 1.2),
           cold, flash * Math.max(b.userData.side, 0.15) * 1.2);
      b.material.opacity = 0.75 * smooth(1.2, 2, tod) * (1 - smooth(4.2, 5, tod));
    });

    // The storm whirls round you.
    wisps.forEach(function (w) {
      var u = w.userData;
      u.a += dt * u.spd * (0.15 + f.wind * 0.6) * (env.reduceMotion ? 0.3 : 1);
      w.position.set(camera.position.x + Math.cos(u.a) * u.rad, camera.position.y + u.y + Math.sin(u.a * 2 + u.rad) * 1.5,
                     camera.position.z + Math.sin(u.a) * u.rad);
      w.material.color.copy(p.fog).multiplyScalar(u.tint);
      w.material.opacity = storm * u.base;
    });
    sleet.update(f, camera.position, storm * 0.9 + (1 - storm) * smooth(0.5, 1, row[14]) * f.wind * 0.6, env.reduceMotion);
    snow.update(f, camera.position, env.reduceMotion);

    // The cairn: the top stones fall first; it is rebuilt from the bottom.
    var n = cairn.length;
    for (var k = 0; k < n; k++) {
      var m = cairn[k], u = m.userData;
      var down = smooth(u.at, u.at + 0.32, fall), back = smooth(k / (n + 1), (k + 2) / (n + 1), rebuild), s = down * (1 - back);
      m.position.copy(u.home).lerp(u.down, s);
      m.position.y = lerp(u.home.y, u.down.y, down * down * (1 - back)) + Math.sin(back * Math.PI) * 0.5 * down;
      m.quaternion.copy(u.homeQ).slerp(u.downQ, s);
    }

    // Pitch-and-toss: up off the cairn, spinning, then down past the edge.
    coin.visible = coinT > 0.01 && coinT < 0.99;
    if (coin.visible) {
      var ct = coinT;
      coin.position.set(SUMMIT.x - ct * 2.6, CAIRN_TOP + 0.2 + Math.sin(Math.min(ct, 0.5) * Math.PI) * 3.4 - smooth(0.5, 1, ct) * (3.4 + 16 * ct * ct),
                        SUMMIT.z + Math.sin(ct * Math.PI) * 3);
      coinMesh.rotation.set(Math.PI / 2 + ct * 34, 0, ct * 3);
      var face = Math.abs(Math.cos(ct * 34));
      coinGlow.material.opacity = 0.25 + 0.75 * Math.pow(face, 6);
      coinLight.intensity = 2 + face * 3;
    }

    // The unforgiving minute.
    var lit = ringT * 60;
    for (var j = 0; j < 60; j++) {
      var on = clamp(lit - j, 0, 1), fresh = j < lit && j > lit - 1.5 ? 1.6 : 1;
      var v = (0.04 + on * 0.96 * fresh) * (1 - smooth(4.7, 5, tod) * 0.55);
      ringCol[j * 3] = v; ringCol[j * 3 + 1] = v * 0.82; ringCol[j * 3 + 2] = v * 0.55;
    }
    ringGeo.attributes.color.needsUpdate = true;
    var lift = 2.4 * (1 - smooth(4.6, 5, tod)), rp = ringGeo.attributes.position;
    if (Math.abs(lift - lastLift) > 0.001) {
      for (j = 0; j < 60; j++) rp.setY(j, ringBase[j] + lift + Math.sin(j / 60 * Math.PI * 2) * 0.15 * lift);
      rp.needsUpdate = true;
      lastLift = lift;
    }
    ring.visible = ringT > 0.001;

    gl.toneMappingExposure = 1 + goldAmt * 0.15 + flash * 0.15;
    gl.render(world, camera);
  }

  return {
    // Portrait screens are narrow and centre the text: turn less, so what
    // the view turns towards stays in frame.
    resize: function (w, h, dpr) { fitCamera(gl, camera, w, h, dpr, small); yawScale = w / h < 1 ? 0.5 : 1; },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('cairn', {
  renderer: renderer3d,
  maxLines: 4,
  scrim: 0.62,
  align: ['left', 'right', 'center', 'left', 'right', 'right', 'left', 'left'],
  // Panels: each stanza in two halves, 0-1 I, 2-3 II, 4-5 III, 6-7 IV.
  keys: function (T) {
    var n = T.count;
    function s(i) { return T.start(Math.min(i, n - 1)); }
    //  unit          path   tod  snow wind  yaw    pitch  storm eye   gold flash fall rebuild coin ring shake near
    return [
      [0,             0.000, 0.0, 0.70, 0.85, 0.00,  0.02, 1.00, 0.0,  0, 0, 0, 0, 0, 0, 0.5, 0],
      [0.7,           0.010, 0.0, 0.70, 0.85, 0.00,  0.02, 1.00, 0.0,  0, 0, 0, 0, 0, 0, 0.5, 0],
      [s(0) + 0.3,    0.040, 0.0, 0.80, 0.95, 0.00,  0.00, 1.00, 0.0,  0, 0, 0, 0, 0, 0, 0.5, 0],   // "keep your head"
      [s(0) + 1.3,    0.160, 0.1, 0.75, 0.90, -0.05, 0.02, 1.00, 0.0,  0, 0, 0, 0, 0, 0, 0.5, 0],
      [s(1) + 0.2,    0.190, 0.2, 0.55, 0.70, 0.08,  0.05, 0.85, -0.2, 0, 0, 0, 0, 0, 0, 0.3, 0],   // shelter by the boulder
      [s(1) + 0.8,    0.195, 0.6, 0.20, 0.35, 0.04,  0.12, 0.40, -0.2, 0, 0, 0, 0, 0, 0, 0.1, 0],   // "wait and not be tired by waiting"
      [s(1) + 1.4,    0.200, 1.0, 0.00, 0.15, 0.00,  0.14, 0.00, 0.0,  0, 0, 0, 0, 0, 0, 0.0, 0],   // the summit appears
      [s(2) + 0.1,    0.260, 1.8, 0.00, 0.10, 0.00,  0.30, 0.00, 0.0,  0, 0, 0, 0, 0, 0, 0.0, 0],
      [s(2) + 0.45,   0.700, 2.0, 0.00, 0.10, 0.00,  0.42, 0.00, 0.0,  0, 0, 0, 0, 0, 0, 0.0, 0],   // "if you can dream"
      [s(2) + 0.72,   0.995, 2.0, 0.00, 0.10, 0.00,  0.10, 0.00, 0.0,  0, 0, 0, 0, 0, 0, 0.0, 0],   // on the summit
      [s(2) + 0.82,   0.995, 2.0, 0.00, 0.10, 0.00,  0.10, 0.00, 0.0,  1, 0, 0, 0, 0, 0, 0.0, 0],   // "triumph"
      [s(2) + 0.94,   0.995, 2.0, 0.00, 0.10, 0.00,  0.10, 0.00, 0.0,  0, 0, 0, 0, 0, 0, 0.0, 0],
      [s(2) + 1.04,   0.995, 2.0, 0.00, 0.10, 0.00,  0.10, 0.00, 0.0,  0, 1, 0, 0, 0, 0, 0.0, 0],   // "and disaster"
      [s(2) + 1.16,   0.995, 2.0, 0.00, 0.10, 0.00,  0.10, 0.00, 0.0,  0, 0, 0, 0, 0, 0, 0.0, 0],
      [s(3) + 0.3,    0.995, 2.0, 0.00, 0.10, 0.36,  -0.22, 0.00, -0.3, 0, 0, 0, 0, 0, 0, 0.0, 0.5],
      [s(3) + 0.55,   0.995, 2.0, 0.00, 0.10, 0.38,  -0.30, 0.00, -0.3, 0, 0, 0, 0, 0, 0, 0.0, 0.8],
      [s(3) + 0.85,   0.995, 2.0, 0.00, 0.10, 0.38,  -0.30, 0.00, -0.5, 0, 0, 1, 0, 0, 0, 0.0, 0.85],   // "broken"
      [s(3) + 1.5,    0.995, 2.0, 0.00, 0.10, 0.38,  -0.30, 0.00, -0.7, 0, 0, 1, 1, 0, 0, 0.0, 0.85],   // "stoop and build 'em up"
      [s(4) + 0.3,    0.995, 2.0, 0.00, 0.10, -0.14, -0.16, 0.00, 0.0,  0, 0, 1, 1, 0, 0, 0.0, 0.4],   // "one heap of all your winnings"
      [s(4) + 0.6,    0.995, 2.0, 0.00, 0.10, -0.10, 0.22, 0.00, 0.0,  0, 0, 1, 1, 0.5, 0, 0.0, 0.3], // "pitch-and-toss"
      [s(4) + 0.85,   0.995, 2.0, 0.00, 0.20, -0.06, -0.34, 0.00, 0.0,  0, 0, 1, 1, 1, 0, 0.0, 0.2],
      [s(4) + 1.25,   0.020, 2.2, 0.00, 0.30, 0.00,  0.22, 0.00, 0.0,  0, 0, 1, 1, 1, 0, 0.0, 0],   // "start again at your beginnings"
      [s(4) + 1.45,   0.020, 2.4, 0.00, 0.02, 0.00,  0.20, 0.00, 0.0,  0, 0, 1, 1, 1, 0, 0.0, 0],   // "never breathe a word"
      [s(5) + 0.25,   0.060, 2.9, 0.40, 0.80, 0.00,  0.05, 0.00, -0.3, 0, 0, 1, 1, 1, 0, 1.0, 0],
      [s(5) + 0.8,    0.300, 3.0, 0.90, 1.00, -0.30, -0.10, 0.00, -0.4, 0, 0, 1, 1, 1, 0, 1.0, 0],  // the knife-edge in a gale
      [s(5) + 1.4,    0.480, 3.2, 0.60, 1.00, -0.15,  0.02, 0.00, -0.3, 0, 0, 1, 1, 1, 0, 1.0, 0],   // "Hold on"
      [s(6) + 0.45,   0.580, 3.8, 0.10, 0.35, -0.60,  -0.30, 0.00, 0.0,  0, 0, 1, 1, 1, 0, 0.3, 0],  // "talk with crowds"
      [s(6) + 1.1,    0.760, 4.0, 0.05, 0.25, -0.85, 0.06, 0.00, 0.0,  0, 0, 1, 1, 1, 0, 0.2, 0],   // "walk with kings"
      [s(7) + 0.15,   0.820, 4.1, 0.00, 0.20, 0.00,  0.02, 0.00, 0.0,  0, 0, 1, 1, 1, 0, 0.0, 0],
      [s(7) + 0.7,    0.985, 4.5, 0.00, 0.20, 0.00,  -0.06, 0.00, 0.0,  0, 0, 1, 1, 1, 1, 0.0, 0],  // "sixty seconds' worth of distance run"
      [s(7) + 1.2,    0.995, 5.0, 0.00, 0.15, 0.00,  -0.10, 0.00, 0.2,  0, 0, 1, 1, 1, 1, 0.0, 0],  // "yours is the Earth"
      [T.total,       0.998, 5.0, 0.00, 0.12, 0.00,  -0.08, 0.00, 0.3,  0, 0, 1, 1, 1, 1, 0.0, 0.15]
    ];
  },
  sound: {
    src: '/audio/wind.mp3',
    label: 'Play the mountain wind',
    volume: function (row) { return 0.06 + 0.5 * row[3] + 0.15 * row[6]; },
    cues: [
      { stanza: 2, at: 1.04, play: thunder },
      { stanza: 3, at: 0.6, play: clatter },
      { stanza: 4, at: 0.45, play: coinRing }
    ]
  }
});
