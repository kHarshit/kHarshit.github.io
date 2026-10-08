/*
 * Scene for "Man Vs Nature" (Norman Littleford): one green valley, seen
 * from the hill above it, as it is dug, felled and built over.
 *
 * I   "The heavens roared with thunder as lightning filled the skies": a
 *     storm over rolling countryside, seen from below the crest of the hill,
 *     bolts striking the far hills, rain curtains hanging from the cloud.
 * II  "Is Mother Nature telling us": over the crest, and the storm rolls on
 *     down the valley; sunlight follows it across the fields and woods.
 * III "We marvel at her beauty each time we look around": the valley clear
 *     and lovely, fair-weather clouds and their shadows drifting over it;
 *     "then dig up all her treasures": an open-pit mine opens in the
 *     hillside and terraces down into the earth, to a deep blast.
 * IV  "We take it all and still want more": the forest on the far slope
 *     falls, tree by tree, to a chainsaw, leaving stumps and bare mud.
 * V   "We build across the countryside": roads draw themselves along the
 *     valley, pylons march over it, and grey blocks rise out of the fields
 *     and over the cleared land, until little but the hill you stand on is
 *     green, under a dim brown haze. "Man is in the way": far off, through
 *     the smog, the lightning flickers once more.
 *
 * Columns: [unit, z, dark, rain, wind, storm, front, fair, mine, cut,
 *           city, haze, yaw, pitch, lift]
 * `front` is the z of the storm's trailing edge (clouds lie beyond it, at
 * lower z); `mine`, `cut` and `city` run 0..1 as the pit is dug, the forest
 * felled and the valley built over.
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, softSprite, skyDome, scatter,
         rainField, ribbon, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── The valley (metres; you look down it towards -z) ────────────────────
// You stand on a round hill at z ~ 165; below it the valley floor runs away
// between long slopes to far mountains. The river swings out to the right
// past the foot of the hill. The mine is dug into the left-hand slope, the
// forest covers the right-hand one.
var PIT = { x: -135, z: -55 };
var HILL = { x: 0, z: 170 };
function riverX(z) { return 26 * Math.sin(z * 0.006 + 0.8) + 12 * Math.sin(z * 0.017 + 2.1) + 10 + smooth(-10, 140, z) * 260; }
function land(x, z) {
  var rx = riverX(z), d = Math.abs(x - rx);
  var side = 95 * smooth(50, 430, d) * (0.82 + 0.22 * Math.sin(z * 0.004 + (x > rx ? 1.3 : 0.2))) * smooth(90, -40, z);
  var roll = 5 * Math.sin(x * 0.013 + 0.4) * Math.cos(z * 0.011) + 2.5 * Math.sin(x * 0.031 + z * 0.027);
  var far = smooth(-650, -1500, z) * (200 + 80 * Math.sin(x * 0.004 + 1) + 45 * Math.sin(x * 0.011 + 0.3) +
            26 * Math.abs(Math.sin(x * 0.019 + Math.sin(z * 0.005) * 1.6)) + 14 * Math.sin(x * 0.061 - z * 0.02));
  var hx = x - HILL.x, hz = z - HILL.z;
  var hill = 76 * Math.exp(-(hx * hx / (2 * 95 * 95) + hz * hz / (2 * 72 * 72)));
  return side + roll * smooth(15, 110, d) + far + hill - 3.4 * smooth(17, 6, d);
}
var PIT_TOP = land(PIT.x, PIT.z) + 0.5;

// The order things go in, as numbers 0..1 the timeline sweeps past.
// Building spreads out from the middle of the valley floor; nothing is built
// on the hill you stand on, in the pit, on the river or on the high tops.
function bornAt(x, z) {
  var h = land(x, z);
  if (z > 25 && h > 16) return 9;
  if (Math.hypot(x - PIT.x, (z - PIT.z) * 0.8) < 128) return 9;
  if (Math.abs(x - riverX(z)) < 15) return 9;
  if (h > 80) return 9;
  var d = Math.hypot((x - 30) * 0.85, (z + 330) * 0.68) / 430;
  return d + 0.12 * Math.sin(x * 0.02 + z * 0.013) * Math.cos(z * 0.017 - x * 0.006) + h * 0.002;
}
// The forest: the right-hand slope (felled first) and a band beyond the pit.
function forestAt(x, z) {
  var rx = riverX(z), n = Math.sin(x * 0.021 + 1.3) * Math.cos(z * 0.017) + 0.6 * Math.sin(x * 0.047 - z * 0.039 + 0.7);
  var right = smooth(60, 95, x - rx) * smooth(-560, -520, z) * smooth(30, -10, z) * smooth(-0.55, -0.15, n);
  var left = smooth(-240, -280, x - rx) * smooth(-620, -580, z) * smooth(60, 20, z) * smooth(-0.4, 0, n) *
             (Math.hypot(x - PIT.x, (z - PIT.z) * 0.8) > 140 ? 1 : 0);
  return Math.max(right, left * 0.9);
}
function cutAt(x, z) {
  var wobble = 0.12 * Math.sin(x * 0.05 + z * 0.03) + 0.06 * Math.sin(x * 0.17 - z * 0.11);
  if (x > riverX(z)) return clamp(Math.hypot(x - 120, (z + 130) * 0.8) / 330 + wobble, 0.02, 0.98);
  return 0.62 + Math.hypot(x + 300, z + 200) / 900 + wobble;     // the far wood: only partly felled
}

// ── Shaders ──────────────────────────────────────────────────────────────
// One cloud field, shared by the cloud deck and the shadows it casts on the
// land: the storm (thick, everywhere beyond `front`) and fair-weather cumulus.
var CLOUD_GLSL =
  'uniform float uCover; uniform float uFront; uniform float uFair; uniform vec2 uDrift;\n' +
  'float ch(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }\n' +
  'float cn(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);\n' +
  ' return mix(mix(ch(i), ch(i + vec2(1, 0)), f.x), mix(ch(i + vec2(0, 1)), ch(i + vec2(1, 1)), f.x), f.y); }\n' +
  'float cfbm(vec2 p){ float v = 0.0, a = 0.5; for (int i = 0; i < 5; i++){ v += a * cn(p); p = p * 2.03 + vec2(1.7, 9.2); a *= 0.5; } return v; }\n' +
  // x: cover, y: how much of it is storm, z: the raw noise (for shading)
  'vec3 clouds(vec2 w){ float n = cfbm(w * 0.0016 + uDrift);\n' +
  ' float s = uCover * (1.0 - smoothstep(uFront - 380.0, uFront + 380.0, w.y));\n' +
  ' float c = max(s, uFair);\n' +
  ' float a = smoothstep(0.9 - c, 1.1 - 0.8 * c, n);\n' +
  ' return vec3(a, s / max(c, 0.001), n); }\n';

// The open pit: benches stepping down from the rim, cut into the slope.
// It is its own mesh, in rings that sit exactly on the bench edges and
// widen as it is dug, so the terraces stay crisp; the land around it is
// drawn by the terrain, each mesh discarding the other's side of the cut.
var PIT_GLSL =
  'uniform float uMine; uniform vec2 uPit; uniform float uPitTop;\n' +
  'float sm(float a, float b, float v){ float t = clamp((v - a) / (b - a), 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }\n' +
  // land() as in the JS, near the pit (the far mountains don't reach it)
  'float landAt(vec2 p){ float x = p.x, z = p.y;\n' +
  ' float rx = 26.0 * sin(z * 0.006 + 0.8) + 12.0 * sin(z * 0.017 + 2.1) + 10.0 + sm(-10.0, 140.0, z) * 260.0, d = abs(x - rx);\n' +
  ' float side = 95.0 * sm(50.0, 430.0, d) * (0.82 + 0.22 * sin(z * 0.004 + (x > rx ? 1.3 : 0.2))) * sm(90.0, -40.0, z);\n' +
  ' float roll = 5.0 * sin(x * 0.013 + 0.4) * cos(z * 0.011) + 2.5 * sin(x * 0.031 + z * 0.027);\n' +
  ' vec2 hh = p - vec2(0.0, 170.0);\n' +
  ' float hill = 76.0 * exp(-(hh.x * hh.x / 18050.0 + hh.y * hh.y / 10368.0));\n' +
  ' return side + roll * sm(15.0, 110.0, d) + hill - 3.4 * sm(17.0, 6.0, d); }\n' +
  'float pitR(){ return 16.0 + 96.0 * uMine; }\n' +
  'float pitDist(vec2 p){ return length((p - uPit) * vec2(1.0, 0.8)); }\n' +
  // s counts benches in from the rim (0) to the floor (7..9); h counts
  // them up the cut face beyond the rim, where the slope rises
  'float bench(float f){ return floor(f) + smoothstep(0.5, 0.96, fract(f)); }\n' +
  'float pitY(vec2 p, out float level){ float R = pitR(), d = pitDist(p);\n' +
  ' float s = min(clamp(1.0 - d / R, 0.0, 1.0) * 9.0, 7.0), h = max(d - R, 0.0) * 1.15 / 9.0;\n' +
  ' level = d > R ? -bench(h) : bench(s);\n' +
  ' return uPitTop - 62.0 * uMine * bench(s) / 7.0 + bench(h) * 9.0; }\n' +
  // cut deeper than `t` m here? The two meshes overlap a little at the edge, so no seam shows
  'bool inPit(vec2 p, float t){ float lv; return uMine > 0.04 && pitDist(p) < pitR() + 90.0 && pitY(p, lv) < landAt(p) - t; }\n';

// Ring positions for the pit mesh: s (>= 0) in from the rim, -h out from it.
function pitRings() {
  var out = [9, 8.5, 8, 7.5, 7], F = [0.96, 0.88, 0.8, 0.7, 0.6, 0.5, 0.33, 0.17, 0];
  for (var k = 6; k >= 0; k--) F.forEach(function (f) { out.push(k + f); });
  for (var j = 0; j < 10; j++) F.slice().reverse().forEach(function (f) { if (j + f > 0) out.push(-(j + f)); });
  return out;
}

// ── Pieces ───────────────────────────────────────────────────────────────
// A grid that is fine (`fine` m) over the valley, coarser towards the far
// mountains: xs and zs are the grid lines.
function axis(spans) {
  var out = [];
  spans.forEach(function (s) { for (var v = s[0]; v < s[1] - 1e-6; v += s[2]) out.push(v); });
  out.push(spans[spans.length - 1][1]);
  return out;
}
function gridTerrain(xs, zs, colorFn, infoFn, material) {
  var nx = xs.length, nz = zs.length, pos = new Float32Array(nx * nz * 3), cols = new Float32Array(nx * nz * 3),
      info = new Float32Array(nx * nz * 3), idx = new Uint32Array((nx - 1) * (nz - 1) * 6), k = 0;
  for (var j = 0; j < nz; j++) {
    for (var i = 0; i < nx; i++) {
      var x = xs[i], z = zs[j], y = land(x, z), c = colorFn(x, z, y), f = infoFn(x, z), o = (j * nx + i) * 3;
      pos[o] = x; pos[o + 1] = y; pos[o + 2] = z;
      cols[o] = c.r; cols[o + 1] = c.g; cols[o + 2] = c.b;
      info[o] = f[0]; info[o + 1] = f[1]; info[o + 2] = f[2];
    }
  }
  for (j = 0; j < nz - 1; j++) {
    for (i = 0; i < nx - 1; i++) {
      var a = j * nx + i, b = a + 1, c2 = a + nx, d = c2 + 1;
      idx[k++] = a; idx[k++] = c2; idx[k++] = b;
      idx[k++] = b; idx[k++] = c2; idx[k++] = d;
    }
  }
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  geo.setAttribute('aInfo', new THREE.BufferAttribute(info, 3));
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  geo.computeVertexNormals();
  var mesh = new THREE.Mesh(geo, material);
  mesh.frustumCulled = false;
  return mesh;
}

// Trees, in metres, with a stump below 0.75 m that stays when they fall.
// Crowns are white so the instance colour tints them; bark keeps its own.
var STUMP = 0.75;
function conifer() {
  return merge([
    tinted(new THREE.CylinderGeometry(0.3, 0.34, STUMP, 6).translate(0, STUMP / 2, 0), '#4a3a2c'),
    tinted(new THREE.CylinderGeometry(0.3, 0.3, 0.02, 6).translate(0, STUMP - 0.03, 0), '#d6bf92'),
    tinted(new THREE.CylinderGeometry(0.16, 0.3, 4.6, 6).translate(0, STUMP + 2.3, 0), '#4a3a2c'),
    tinted(new THREE.ConeGeometry(2.5, 5.4, 7).translate(0, 4.9, 0), '#ffffff'),
    tinted(new THREE.ConeGeometry(1.95, 4.6, 7).translate(0, 7.5, 0), '#ffffff'),
    tinted(new THREE.ConeGeometry(1.3, 3.8, 7).translate(0, 9.9, 0), '#ffffff')
  ]);
}
function broadleaf(r) {
  var parts = [
    tinted(new THREE.CylinderGeometry(0.32, 0.38, STUMP, 6).translate(0, STUMP / 2, 0), '#4e3e30'),
    tinted(new THREE.CylinderGeometry(0.32, 0.32, 0.02, 6).translate(0, STUMP - 0.03, 0), '#d6bf92'),
    tinted(new THREE.CylinderGeometry(0.18, 0.32, 3.4, 6).translate(0, STUMP + 1.7, 0), '#4e3e30')
  ];
  for (var k = 0; k < 5; k++) {
    var a = r() * Math.PI * 2, d = k ? 0.9 + r() * 0.6 : 0;
    parts.push(tinted(new THREE.IcosahedronGeometry(1.5 + r() * 0.7, 0).scale(1, 0.85, 1)
      .translate(Math.cos(a) * d, 4.6 + r() * 1.3 + (k ? 0 : 0.8), Math.sin(a) * d), '#ffffff'));
  }
  return merge(parts);
}

// A lattice pylon about 30 m tall: tapering legs, braces and two crossarms.
function beam(a, b, w) {
  var dir = new THREE.Vector3().subVectors(b, a), len = dir.length();
  var g = new THREE.BoxGeometry(w, len, w);
  g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize()));
  return g.translate((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2);
}
function pylonGeometry() {
  var V = function (x, y, z) { return new THREE.Vector3(x, y, z); }, parts = [], col = '#9a9ea4';
  function legAt(sx, sz, y) { var w = lerp(3.2, 0.7, Math.pow(y / 31, 0.8)); return V(sx * w, y, sz * w); }
  var corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
  corners.forEach(function (c) { parts.push(tinted(beam(legAt(c[0], c[1], 0), legAt(c[0], c[1], 31), 0.32), col)); });
  [0, 6, 12, 17.5, 22.5, 27.5].forEach(function (y0, li, arr) {
    var y1 = arr[li + 1] || 31;
    for (var k = 0; k < 4; k++) {
      var c = corners[k], d = corners[(k + 1) % 4];
      parts.push(tinted(beam(legAt(c[0], c[1], y0), legAt(d[0], d[1], y1), 0.14), col));
      parts.push(tinted(beam(legAt(d[0], d[1], y0), legAt(c[0], c[1], y1), 0.14), col));
      parts.push(tinted(beam(legAt(c[0], c[1], y1), legAt(d[0], d[1], y1), 0.16), col));
    }
  });
  [[21.5, 7.5], [26.5, 5.5]].forEach(function (arm) {
    parts.push(tinted(beam(V(-arm[1], arm[0], -0.5), V(arm[1], arm[0], -0.5), 0.3), col));
    parts.push(tinted(beam(V(-arm[1], arm[0], 0.5), V(arm[1], arm[0], 0.5), 0.3), col));
    parts.push(tinted(beam(V(-arm[1], arm[0], 0), V(-1, arm[0] + 1.8, 0), 0.16), col));
    parts.push(tinted(beam(V(arm[1], arm[0], 0), V(1, arm[0] + 1.8, 0), 0.16), col));
  });
  parts.push(tinted(beam(V(0, 31, 0), V(0, 33, 0), 0.25), col));
  return merge(parts);
}
// Where the wires hang from a pylon, across the line (x) and up (y).
var ATTACH = [[-7.2, 21.0], [7.2, 21.0], [-5.2, 26.0], [5.2, 26.0], [0, 33]];

// A tuft of grass blades about 0.6 m tall.
function tuft(r) {
  var parts = [];
  for (var k = 0; k < 5; k++) {
    var a = r() * Math.PI, h = 0.35 + r() * 0.45, lean = (r() - 0.5) * 0.5, x = (r() - 0.5) * 0.25, z = (r() - 0.5) * 0.25;
    var g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute([-0.035, 0, 0, 0.035, 0, 0, lean * h, h, 0], 3));
    g.computeVertexNormals();
    g.rotateY(a).translate(x, 0, z);
    parts.push(tinted(g, k % 2 ? '#ffffff' : '#d8e6c0'));
  }
  return merge(parts);
}

function canvasTex(w, h, paint) {
  var c = document.createElement('canvas');
  c.width = w; c.height = h;
  paint(c.getContext('2d'), w, h);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
// A curtain of rain hanging from the cloud: soft vertical streaks.
function curtainTexture(r) {
  return canvasTex(128, 256, function (x, w, h) {
    for (var i = 0; i < 160; i++) {
      var u = 0.5 + (r() - 0.5) * (0.4 + r() * 0.6), top = r() * 0.25;
      var g = x.createLinearGradient(0, top * h, 0, h);
      var a = (0.05 + r() * 0.08) * (1 - Math.abs(u - 0.5) * 1.6);
      g.addColorStop(0, 'rgba(200,208,220,0)');
      g.addColorStop(0.25, 'rgba(200,208,220,' + a.toFixed(3) + ')');
      g.addColorStop(0.85, 'rgba(200,208,220,' + (a * 0.8).toFixed(3) + ')');
      g.addColorStop(1, 'rgba(200,208,220,0)');
      x.fillStyle = g;
      x.fillRect(u * w + (r() - 0.5) * 6, top * h, 1 + r() * 3, h);
    }
  });
}

// ── Sound ────────────────────────────────────────────────────────────────
function noiseBuffer(ac, secs, shape) {
  var b = ac.createBuffer(1, Math.floor(ac.sampleRate * secs), ac.sampleRate), d = b.getChannelData(0);
  for (var i = 0, last = 0; i < d.length; i++) {
    last = last * 0.96 + (Math.random() * 2 - 1) * 0.04;
    d[i] = (last * 6 + (Math.random() * 2 - 1) * 0.25) * (shape ? shape(i / d.length) : 1);
  }
  return b;
}
// Thunder: a crack, then a long low roll. `far` softens and delays it.
function thunderAt(far) {
  return function (ac, out) {
    var t = ac.currentTime + 0.3 + far * 1.2, len = 4.5 + far * 1.5;
    var src = ac.createBufferSource(), lp = ac.createBiquadFilter(), g = ac.createGain();
    src.buffer = noiseBuffer(ac, len, function (k) { return (k < 0.03 && far < 0.5 ? 1.6 : 1) * Math.pow(1 - k, 1.6); });
    lp.type = 'lowpass';
    lp.frequency.setValueAtTime(lerp(1500, 420, far), t);
    lp.frequency.exponentialRampToValueAtTime(lerp(160, 90, far), t + 1.6);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(lerp(1.0, 0.35, far), t + 0.05 + far * 0.4);
    g.gain.exponentialRampToValueAtTime(0.0001, t + len);
    src.connect(lp); lp.connect(g); g.connect(out);
    src.start(t);
  };
}
// A quarry blast: a deep thump and the rumble of falling rock.
function blast(ac, out) {
  var t = ac.currentTime, o = ac.createOscillator(), og = ac.createGain();
  o.type = 'sine';
  o.frequency.setValueAtTime(70, t);
  o.frequency.exponentialRampToValueAtTime(32, t + 0.8);
  og.gain.setValueAtTime(0.0001, t);
  og.gain.exponentialRampToValueAtTime(0.9, t + 0.02);
  og.gain.exponentialRampToValueAtTime(0.0001, t + 1.2);
  o.connect(og); og.connect(out);
  o.start(t); o.stop(t + 1.3);
  var src = ac.createBufferSource(), lp = ac.createBiquadFilter(), g = ac.createGain();
  src.buffer = noiseBuffer(ac, 3, function (k) { return Math.pow(1 - k, 2.2); });
  lp.type = 'lowpass';
  lp.frequency.value = 380;
  g.gain.value = 0.7;
  src.connect(lp); lp.connect(g); g.connect(out);
  src.start(t + 0.05);
}
// A chainsaw that bites, then the creak and crash of the tree.
function timber(ac, out) {
  var t = ac.currentTime, o = ac.createOscillator(), lp = ac.createBiquadFilter(), g = ac.createGain();
  var lfo = ac.createOscillator(), lg = ac.createGain();
  o.type = 'sawtooth';
  o.frequency.setValueAtTime(120, t);
  o.frequency.linearRampToValueAtTime(155, t + 0.4);
  o.frequency.linearRampToValueAtTime(105, t + 1.7);
  lfo.frequency.value = 23;
  lg.gain.value = 0.02;
  lfo.connect(lg); lg.connect(g.gain);
  lp.type = 'lowpass';
  lp.frequency.value = 1600;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.06, t + 0.08);
  g.gain.setValueAtTime(0.06, t + 1.6);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 1.9);
  o.connect(lp); lp.connect(g); g.connect(out);
  o.start(t); lfo.start(t); o.stop(t + 2); lfo.stop(t + 2);
  var crash = ac.createBufferSource(), bp = ac.createBiquadFilter(), cg = ac.createGain();
  crash.buffer = noiseBuffer(ac, 1.6, function (k) { return k < 0.05 ? k / 0.05 : Math.pow(1 - k, 2.5); });
  bp.type = 'lowpass';
  bp.frequency.value = 520;
  cg.gain.value = 0.8;
  crash.connect(bp); bp.connect(cg); cg.connect(out);
  crash.start(t + 2.6);
}

// ── Light ────────────────────────────────────────────────────────────────
var SUN = new THREE.Vector3(-0.8, 0.5, 0.18).normalize();      // afternoon, from the left and a little behind
var SKY = {
  clear: ['#3e6fb2', '#8db1d8', '#e4ecef'], storm: ['#171b22', '#262c35', '#434b55'], haze: ['#5a5650', '#857d70', '#ab9d86']
};
var BOLTS = [];                                                 // [timeline point, kind] of scripted strikes
var OUTRO_BOLT = 2.15;                                          // after the start of panel V

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(1970);
  var gl = makeRenderer(canvas, { clear: '#434b55' });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#434b55', 0.0015);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.5, 9000);
  camera.rotation.order = 'YXZ';

  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: SKY.storm[0], mid: SKY.storm[1], horizon: SKY.storm[2], sun: '#fff0d0' }, 6000);
  dome.uniforms.sunDir.value.copy(SUN);
  sky.add(dome.mesh);
  var sunDisc = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,250,235,1)', 'rgba(255,230,180,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false, opacity: 0 }));
  sunDisc.position.copy(SUN).multiplyScalar(5000);
  sunDisc.scale.setScalar(420);
  sky.add(sunDisc);

  var hemi = new THREE.HemisphereLight('#b8c8e0', '#3a4a2c', 0.6);
  var sun = new THREE.DirectionalLight('#fff0d8', 0);
  world.add(hemi, sun, sun.target);

  // Uniforms shared by the land, trees, blocks, roads and the cloud deck.
  var U = {
    uCover: { value: 1 }, uFront: { value: 900 }, uFair: { value: 0 }, uDrift: { value: new THREE.Vector2() },
    uMine: { value: 0 }, uPit: { value: new THREE.Vector2(PIT.x, PIT.z) }, uPitTop: { value: PIT_TOP },
    uCut: { value: 0 }, uCity: { value: 0 }, uRoad: { value: 0 }, uTime: { value: 0 }, uWind: { value: 0 },
    uShade: { value: 0 }, uShadowOff: { value: new THREE.Vector2(SUN.x / SUN.y * 300, SUN.z / SUN.y * 300) },
    uCell: { value: small ? 32 : 26 }, uLights: { value: 0 }
  };
  function hook(mat, fn) { mat.onBeforeCompile = function (sh) { Object.keys(U).forEach(function (k) { sh.uniforms[k] = U[k]; }); fn(sh); }; return mat; }

  // ── The land ──
  var C = {
    meadow: new THREE.Color('#5c8a38'), bright: new THREE.Color('#7da54a'), wheat: new THREE.Color('#b9a95c'),
    deep: new THREE.Color('#4a7430'), hedge: new THREE.Color('#355a26'), slope: new THREE.Color('#6a8c44'),
    upper: new THREE.Color('#7e8d58'), forest: new THREE.Color('#2b4523'), bank: new THREE.Color('#5d6a40'),
    mount: new THREE.Color('#41573e'), rock: new THREE.Color('#6e6a60'), hill: new THREE.Color('#5a8f38')
  };
  var col = new THREE.Color();
  function landColor(x, z, y) {
    var rx = riverX(z), d = Math.abs(x - rx);
    // Fields on the valley floor: a patchwork, hedged.
    var fx = (x + 13) / 64, fz = (z + 7) / 82, cx = Math.floor(fx), cz = Math.floor(fz);
    var hsh = Math.abs(Math.sin(cx * 12.99 + cz * 78.23) * 43758.54) % 1;
    col.copy(hsh < 0.3 ? C.meadow : hsh < 0.55 ? C.bright : hsh < 0.75 ? C.deep : C.wheat);
    var edge = Math.max(Math.abs(fx - cx - 0.5), Math.abs(fz - cz - 0.5));
    col.lerp(C.hedge, smooth(0.46, 0.495, edge) * 0.7);
    col.lerp(C.slope, smooth(10, 40, y));
    col.lerp(C.upper, smooth(55, 90, y) * 0.7);
    col.lerp(C.bank, smooth(20, 9, d));
    col.lerp(C.forest, clamp(forestAt(x, z) * 1.6, 0, 1));
    col.lerp(C.hill, smooth(14, 40, y) * smooth(30, 80, z));
    var m = smooth(-650, -1100, z);
    col.lerp(C.mount, m * 0.85);
    col.lerp(C.rock, m * smooth(170, 260, y) * 0.6);
    // mottled, so the grass isn't one flat green
    return col.multiplyScalar(0.84 + 0.2 * Math.sin(x * 0.11 + Math.sin(z * 0.07) * 2) * Math.cos(z * 0.09 - x * 0.05) +
                              0.08 * Math.sin(x * 0.37 + z * 0.29));
  }
  var landMat = hook(new THREE.MeshLambertMaterial({ vertexColors: true }), function (sh) {
    sh.vertexShader = 'attribute vec3 aInfo; varying vec3 vInfo; varying vec3 vWP;\n' +
      sh.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\n vInfo = aInfo; vWP = transformed;');
    sh.fragmentShader = CLOUD_GLSL + PIT_GLSL + 'uniform float uCut; uniform float uCity; uniform float uShade; uniform vec2 uShadowOff; uniform float uCell;\n' +
      'varying vec3 vInfo; varying vec3 vWP;\n' +
      sh.fragmentShader
        .replace('void main() {', 'void main() {\n if (inPit(vWP.xz, 2.0)) discard;')
        .replace('#include <color_fragment>',
          '#include <color_fragment>\n' +
          ' vec3 c = diffuseColor.rgb;\n' +
          // felled: mud and slash where the forest stood
          ' float g = cn(vWP.xz * 0.15);\n' +
          ' float felled = vInfo.x * smoothstep(vInfo.y - 0.02, vInfo.y + 0.03, uCut);\n' +
          ' c = mix(c, mix(vec3(0.33, 0.26, 0.18), vec3(0.45, 0.39, 0.28), g), felled * 0.9);\n' +
          // built over: concrete and tarmac, with a grid of streets
          ' float built = smoothstep(vInfo.z - 0.015, vInfo.z + 0.015, uCity);\n' +
          ' vec2 st = abs(fract(vWP.xz / uCell) - 0.5) * uCell; vec2 fw = fwidth(vWP.xz) + 0.001;\n' +
          ' vec3 conc = mix(vec3(0.42, 0.41, 0.39), vec3(0.5, 0.49, 0.46), g);\n' +
          ' float edgeX = smoothstep(uCell * 0.5 - 2.2 - fw.x, uCell * 0.5 - 2.2, st.x), edgeZ = smoothstep(uCell * 0.5 - 2.2 - fw.y, uCell * 0.5 - 2.2, st.y);\n' +
          ' conc = mix(conc, vec3(0.2, 0.2, 0.21), max(edgeX, edgeZ));\n' +
          ' c = mix(c, conc, built);\n' +
          ' diffuseColor.rgb = c;')
        .replace('#include <lights_fragment_end>',
          '#include <lights_fragment_end>\n reflectedLight.directDiffuse *= 1.0 - clouds(vWP.xz + uShadowOff).x * uShade;');
  });
  var F = small ? 2 : 1;
  var xs = axis([[-3200, -1400, 300], [-1400, -560, 40 * F], [-560, -264, 4 * F], [-264, -8, 1.6 * F], [-8, 560, 4 * F],
                 [560, 1400, 40 * F], [1400, 3200, 300]]);
  var zs = axis([[-4200, -1600, 300], [-1600, -720, 40 * F], [-720, -210, 6 * F], [-210, 90, 2 * F], [90, 290, 4 * F],
                 [290, 700, 40 * F], [700, 1600, 300]]);
  var ground = gridTerrain(xs, zs, landColor, function (x, z) {
    return [forestAt(x, z) > 0.3 ? 1 : 0, cutAt(x, z), bornAt(x, z)];
  }, landMat);
  world.add(ground);

  // The pit itself: rings round its centre (x = angle, aRing = ring), laid
  // on the benches in the shader and shaded flat so every edge is crisp.
  var rings = pitRings(), NT = small ? 160 : 300, NR = rings.length;
  var pitPos = new Float32Array(NR * (NT + 1) * 3), pitRing = new Float32Array(NR * (NT + 1)), pitIdx = new Uint32Array((NR - 1) * NT * 6);
  for (var ri = 0, pk = 0; ri < NR; ri++) {
    for (var ti = 0; ti <= NT; ti++) {
      var vi = ri * (NT + 1) + ti;
      pitPos[vi * 3] = ti / NT * Math.PI * 2;
      pitRing[vi] = rings[ri];
      if (ri < NR - 1 && ti < NT) {
        var a0 = vi, b0 = vi + 1, c0 = vi + NT + 1, d0 = c0 + 1;
        pitIdx[pk++] = a0; pitIdx[pk++] = b0; pitIdx[pk++] = c0;
        pitIdx[pk++] = b0; pitIdx[pk++] = d0; pitIdx[pk++] = c0;
      }
    }
  }
  var pitGeo = new THREE.BufferGeometry();
  pitGeo.setAttribute('position', new THREE.BufferAttribute(pitPos, 3));
  pitGeo.setAttribute('aRing', new THREE.BufferAttribute(pitRing, 1));
  pitGeo.setIndex(new THREE.BufferAttribute(pitIdx, 1));
  var pitMat = hook(new THREE.MeshLambertMaterial({ color: '#ffffff', flatShading: true, side: THREE.DoubleSide,
                                                          polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 2 }), function (sh) {
    sh.vertexShader = PIT_GLSL + 'attribute float aRing; varying vec3 vWP; varying float vLevel;\n' +
      sh.vertexShader.replace('#include <begin_vertex>',
        ' float pr = pitR(), th = position.x, pd = aRing >= 0.0 ? pr * (1.0 - aRing / 9.0) : pr - aRing * 9.0 / 1.15;\n' +
        ' vec2 pxz = uPit + vec2(cos(th), sin(th) / 0.8) * pd;\n' +
        ' float lv, py = pitY(pxz, lv);\n' +
        ' vec3 transformed = vec3(pxz.x, min(py, landAt(pxz)), pxz.y); vWP = transformed; vLevel = lv;');
    sh.fragmentShader = CLOUD_GLSL + PIT_GLSL + 'uniform float uShade; uniform vec2 uShadowOff; varying vec3 vWP; varying float vLevel;\n' +
      sh.fragmentShader
        .replace('void main() {', 'void main() {\n if (!inPit(vWP.xz, -4.0)) discard;')
        .replace('#include <color_fragment>',
          '#include <color_fragment>\n' +
          // bands of rock and earth, darker on the faces, a sour green pool on the floor
          ' vec3 wn = normalize(cross(dFdx(vWP), dFdy(vWP))); float steep = 1.0 - abs(wn.y);\n' +
          ' float band = floor(vLevel + 0.02), bf = fract(band * 0.37 + 0.1), g = cn(vWP.xz * 0.15);\n' +
          ' vec3 pc = bf < 0.33 ? vec3(0.62, 0.5, 0.36) : bf < 0.66 ? vec3(0.52, 0.48, 0.44) : vec3(0.58, 0.4, 0.28);\n' +
          ' pc *= mix(1.08, 0.7, smoothstep(0.2, 0.7, steep)) * (0.9 + 0.2 * g) * (0.94 + 0.06 * sin(vWP.y * 3.1));\n' +
          ' pc = mix(pc, vec3(0.32, 0.52, 0.48), step(6.98, vLevel) * step(0.5, uMine));\n' +
          ' diffuseColor.rgb = pc;')
        .replace('#include <lights_fragment_end>',
          '#include <lights_fragment_end>\n reflectedLight.directDiffuse *= 1.0 - clouds(vWP.xz + uShadowOff).x * uShade;');
  });
  var pitMesh = new THREE.Mesh(pitGeo, pitMat);
  pitMesh.frustumCulled = false;
  pitMesh.visible = false;
  world.add(pitMesh);

  // The river: a ribbon of water lying in its channel.
  var riverPts = [];
  for (var rz = 260; rz > -1500; rz -= 8) riverPts.push(new THREE.Vector3(riverX(rz), 0, rz));
  var water = new THREE.Mesh(ribbon(riverPts, 0, 26, function () { return -1.7; }, 0),
    new THREE.MeshPhongMaterial({ color: '#2c4654', specular: '#c8d8e8', shininess: 90, emissive: '#1a2a34' }));
  world.add(water);

  // ── Trees: the forest, and hedgerow trees down the valley ──
  var treeMat = hook(new THREE.MeshLambertMaterial({ vertexColors: true }), function (sh) {
    sh.vertexShader = 'uniform float uCut; uniform float uCity; uniform float uTime; uniform float uWind; attribute vec3 aTree;\n' +
      sh.vertexShader
        .replace('#include <color_vertex>', 'vColor = vec3(1.0); vColor *= color; if (color.g > 0.99) vColor *= instanceColor.xyz;')
        .replace('#include <begin_vertex>',
          'vec3 transformed = vec3(position);\n' +
          ' float fall = clamp((uCut - aTree.x) * 16.0, 0.0, 1.0), gone = clamp((uCut - aTree.x) * 16.0 - 1.8, 0.0, 1.0);\n' +
          ' float built = clamp((uCity - aTree.y) * 14.0, 0.0, 1.0);\n' +
          ' if (position.y > ' + STUMP.toFixed(2) + ') {\n' +
          '  vec3 q = transformed - vec3(0.0, ' + STUMP.toFixed(2) + ', 0.0);\n' +
          '  float ph = instanceMatrix[3][0] * 0.21 + instanceMatrix[3][2] * 0.17;\n' +
          '  q.x += (sin(uTime * (1.3 + uWind * 2.0) + ph) * 0.6 + 0.5) * uWind * 0.004 * q.y * q.y;\n' +
          '  float ang = fall * fall * 1.52; vec2 dir = vec2(cos(aTree.z), sin(aTree.z));\n' +
          '  float al = dot(q.xz, dir), al2 = al * cos(ang) + q.y * sin(ang);\n' +
          '  q.y = -al * sin(ang) + q.y * cos(ang); q.xz += dir * (al2 - al);\n' +
          '  transformed = q * (1.0 - gone) + vec3(0.0, ' + STUMP.toFixed(2) + ', 0.0);\n' +
          ' }\n' +
          ' transformed *= 1.0 - built;');
  });
  var trees = [];
  [[conifer(), small ? 1300 : 3000], [broadleaf(rng(5)), small ? 500 : 1100]].forEach(function (spec, kind) {
    var mesh = new THREE.InstancedMesh(spec[0], treeMat, spec[1]);
    var attr = new Float32Array(spec[1] * 3), e = new THREE.Euler();
    scatter(mesh, spec[1] * 40, function (i, p, q, s, c) {
      var x, z, f;
      if (kind === 0 || r() < 0.45) {
        // the woods
        x = -700 + r() * 1300; z = 40 - r() * 680;
        f = forestAt(x, z);
        if (r() > f) return false;
        if (kind === 1 && x > riverX(z)) return false;
      } else {
        // hedgerow trees along the field edges
        var cx = Math.floor((-650 + r() * 1300 + 13) / 64), cz = Math.floor((60 - r() * 760 + 7) / 82);
        if (r() < 0.5) { x = cx * 64 - 13 + r() * 64; z = cz * 82 - 7 + (r() - 0.5) * 3; }
        else { x = cx * 64 - 13 + (r() - 0.5) * 3; z = cz * 82 - 7 + r() * 82; }
        if (Math.abs(x - riverX(z)) < 18 || land(x, z) > 45 || forestAt(x, z) > 0.2) return false;
        if (Math.hypot(x - PIT.x, (z - PIT.z) * 0.8) < 120) return false;
      }
      if (z > 30 && land(x, z) > 16) return false;
      p.set(x, land(x, z) - 0.15, z);
      q.setFromEuler(e.set((r() - 0.5) * 0.06, r() * 6.28, (r() - 0.5) * 0.06));
      var sc = kind ? 1.4 + r() * 0.8 : 1.15 + r() * 0.7;
      s.set(sc, sc * (0.9 + r() * 0.25), sc);
      if (kind === 0) c.setRGB(0.16 + r() * 0.06, 0.3 + r() * 0.08, 0.17 + r() * 0.05);
      else c.setRGB(0.26 + r() * 0.12, 0.44 + r() * 0.12, 0.17 + r() * 0.06);
      var felled = forestAt(x, z) > 0.2;
      attr[i * 3] = felled ? cutAt(x, z) : 9;
      attr[i * 3 + 1] = bornAt(x, z);
      attr[i * 3 + 2] = r() * 6.28;
    });
    mesh.geometry.setAttribute('aTree', new THREE.InstancedBufferAttribute(attr, 3));
    mesh.frustumCulled = false;
    world.add(mesh);
    trees.push(mesh);
  });

  // ── Grass on the hill you stand on ──
  var grassMat = hook(new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }), function (sh) {
    sh.vertexShader = 'uniform float uTime; uniform float uWind;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n float gph = instanceMatrix[3][0] * 0.4 + instanceMatrix[3][2] * 0.3;\n' +
      ' float gb = (sin(uTime * (1.4 + uWind * 3.0) + gph) * 0.6 + 0.4 + uWind * 0.8) * (0.05 + uWind * 0.3) * position.y * position.y;\n' +
      ' transformed.x += gb; transformed.z += gb * 0.3;');
  });
  var UP = new THREE.Vector3(0, 1, 0);
  var grass = new THREE.InstancedMesh(tuft(rng(9)), grassMat, small ? 5000 : 14000);
  scatter(grass, 60000, function (i, p, q, s, c) {
    var x = (r() - 0.5) * 160, z = 95 + r() * 110;
    var y = land(x, z);
    if (y < 40 || (Math.abs(x) < 6 && z > 150 && r() < 0.85)) return false;
    p.set(x, y - 0.05, z);
    q.setFromAxisAngle(UP, r() * 6.28);
    s.setScalar(0.55 + r() * 0.6);
    var g = r();
    c.setRGB(0.3 + g * 0.18, 0.5 + g * 0.16, 0.18 + g * 0.08);
  });
  grass.frustumCulled = false;
  world.add(grass);
  // Wildflowers in it.
  var flowerPos = [], flowerCol = [], fc = new THREE.Color(), FLOWERS = ['#f2e9c8', '#f0c84a', '#c8a0e0', '#f4f4f0', '#e88a6a'];
  for (var fi = 0; fi < (small ? 900 : 2200); fi++) {
    var fx = (r() - 0.5) * 140, fz = 100 + r() * 100, fy = land(fx, fz);
    if (fy < 42) continue;
    flowerPos.push(fx, fy + 0.35 + r() * 0.3, fz);
    fc.set(FLOWERS[Math.floor(r() * FLOWERS.length)]);
    flowerCol.push(fc.r, fc.g, fc.b);
  }
  var flowerGeo = new THREE.BufferGeometry();
  flowerGeo.setAttribute('position', new THREE.Float32BufferAttribute(flowerPos, 3));
  flowerGeo.setAttribute('color', new THREE.Float32BufferAttribute(flowerCol, 3));
  var flowers = new THREE.Points(flowerGeo, new THREE.PointsMaterial({ size: 0.13, vertexColors: true, transparent: true,
    map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)'), depthWrite: false }));
  world.add(flowers);

  // ── Roads: a highway down the valley, a cross road, a haul road to the pit ──
  function road(ctrl) {
    var cv = new THREE.CatmullRomCurve3(ctrl.map(function (p) { return new THREE.Vector3(p[0], 0, p[1]); }));
    return cv.getSpacedPoints(Math.ceil(cv.getLength() / 7));
  }
  var roadLines = [
    road([[riverX(-1500) + 70, -1500], [riverX(-1100) + 60, -1100], [riverX(-700) + 62, -700], [riverX(-400) + 58, -400],
          [riverX(-150) + 60, -150], [riverX(-20) + 66, -20], [150, 40], [330, 78], [600, 96]]),
    road([[-900, -400], [-450, -385], [-120, -372], [riverX(-365), -365], [300, -352], [700, -330], [1100, -340]]),
    road([[riverX(-240) + 60, -240], [riverX(-200), -200], [-40, -150], [-70, -95], [PIT.x + 70, PIT.z + 10]]),
    road([[-700, -760], [-300, -720], [riverX(-700), -700], [400, -690], [900, -700]])
  ];
  function roadHeight(x, z) { return Math.max(land(x, z), 0.4); }
  var roadGeos = roadLines.map(function (pts, k) {
    var g = ribbon(pts, 0, k === 0 ? 15 : 10, roadHeight, 0.35), n = g.attributes.position.count, t = new Float32Array(n);
    for (var j = 0; j < pts.length - 1; j++) {
      var a = j / (pts.length - 1), b = (j + 1) / (pts.length - 1), o = j * 6;
      t[o] = a; t[o + 1] = b; t[o + 2] = a; t[o + 3] = a; t[o + 4] = b; t[o + 5] = b;
    }
    // the cross roads are laid a little after the highway
    for (var m = 0; m < n; m++) t[m] = k === 0 ? t[m] * 0.8 : 0.25 + t[m] * 0.75;
    g.setAttribute('aT', new THREE.BufferAttribute(t, 1));
    return g;
  });
  var roadGeo = new THREE.BufferGeometry(), total = 0;
  roadGeos.forEach(function (g) { total += g.attributes.position.count; });
  ['position', 'normal', 'aT'].forEach(function (name) {
    var size = name === 'aT' ? 1 : 3, arr = new Float32Array(total * size), o = 0;
    roadGeos.forEach(function (g) { arr.set(g.attributes[name].array, o); o += g.attributes[name].array.length; });
    roadGeo.setAttribute(name, new THREE.BufferAttribute(arr, size));
  });
  roadGeos.forEach(function (g) { g.dispose(); });
  var roads = new THREE.Mesh(roadGeo, hook(new THREE.MeshLambertMaterial({ color: '#3a3a3c', polygonOffset: true, polygonOffsetFactor: -2 }), function (sh) {
    sh.vertexShader = 'attribute float aT; varying float vT;\n' + sh.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\n vT = aT;');
    sh.fragmentShader = 'uniform float uRoad; varying float vT;\n' + sh.fragmentShader.replace('void main() {', 'void main() {\n if (vT > uRoad) discard;');
  }));
  world.add(roads);

  // ── Blocks: grey city on a grid, rising in the order they are built ──
  var blockGeo = new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0);
  var blockMat = hook(new THREE.MeshLambertMaterial({ color: '#ffffff' }), function (sh) {
    sh.vertexShader = 'uniform float uCity; attribute float aBorn; varying vec3 vWP; varying vec3 vWN; varying float vRise;\n' +
      sh.vertexShader.replace('#include <begin_vertex>',
        '#include <begin_vertex>\n float rise = smoothstep(aBorn, aBorn + 0.05, uCity); transformed.y *= rise; vRise = rise;\n' +
        ' vec4 wq = modelMatrix * instanceMatrix * vec4(transformed, 1.0); vWP = wq.xyz;\n' +
        ' vWN = normalize((modelMatrix * instanceMatrix * vec4(normal, 0.0)).xyz);');
    sh.fragmentShader = 'uniform float uLights; varying vec3 vWP; varying vec3 vWN; varying float vRise;\n' +
      sh.fragmentShader.replace('#include <emissivemap_fragment>',
        '#include <emissivemap_fragment>\n' +
        ' float side = 1.0 - abs(vWN.y);\n' +
        ' vec2 cell = vec2((abs(vWN.x) > abs(vWN.z) ? vWP.z : vWP.x) / 3.4, vWP.y / 3.6);\n' +
        ' vec2 f = fract(cell); vec2 w = fwidth(cell);\n' +
        ' float aa = clamp(1.6 - max(w.x, w.y) * 3.0, 0.0, 1.0);\n' +
        ' float win = mix(0.24, step(0.22, f.x) * step(f.x, 0.78) * step(0.3, f.y) * step(f.y, 0.78), aa) * side;\n' +
        ' float on = mix(0.3, step(0.7, fract(sin(dot(floor(cell), vec2(12.9898, 78.233))) * 43758.5453)), aa);\n' +
        ' diffuseColor.rgb *= 1.0 - win * 0.45;\n' +
        ' totalEmissiveRadiance += vec3(1.0, 0.72, 0.4) * win * on * uLights;');
  });
  var CELL = U.uCell.value, BLOCKS = small ? 1300 : 3000, blocks = new THREE.InstancedMesh(blockGeo, blockMat, BLOCKS);
  var born = new Float32Array(BLOCKS), GREYS = ['#b4b3ae', '#c2beb4', '#a2a3a5', '#cfcbc0', '#8e9196', '#d6d2c8', '#96a2ac'];
  var cells = [];
  for (var gx = Math.ceil(-620 / CELL); gx * CELL < 620; gx++) for (var gz = Math.ceil(-1050 / CELL); gz * CELL < 60; gz++) cells.push([gx * CELL, gz * CELL]);
  var ci = 0;
  scatter(blocks, cells.length, function (i, p, q, s, c) {
    var cell = cells[ci++], x = cell[0], z = cell[1], b = bornAt(x, z);
    if (b > 1.02 || r() < 0.08) return false;
    var w = CELL * (0.42 + r() * 0.34), d = CELL * (0.42 + r() * 0.34);
    var h0 = Math.min(land(x - w / 2, z - d / 2), land(x + w / 2, z - d / 2), land(x - w / 2, z + d / 2), land(x + w / 2, z + d / 2));
    var h1 = Math.max(land(x - w / 2, z - d / 2), land(x + w / 2, z - d / 2), land(x - w / 2, z + d / 2), land(x + w / 2, z + d / 2));
    if (h1 - h0 > 9 || h0 < -0.5) return false;
    var tall = 7 + Math.pow(r(), 2.4) * 26 + Math.exp(-b * b / 0.05) * Math.pow(r(), 1.5) * 75;
    p.set(x + (r() - 0.5) * (CELL - w - 5), h0 - 3, z + (r() - 0.5) * (CELL - d - 5));
    q.identity();
    s.set(w, tall + 3 + (h1 - h0), d);
    c.set(GREYS[Math.floor(r() * GREYS.length)]);
    born[i] = clamp(b, 0, 1) * 0.94 + r() * 0.03;
  });
  blockGeo.setAttribute('aBorn', new THREE.InstancedBufferAttribute(born, 1));
  blocks.frustumCulled = false;
  world.add(blocks);

  // ── Pylons striding across the valley, and their wires ──
  var PYL_A = new THREE.Vector3(-640, 0, -820), PYL_B = new THREE.Vector3(470, 0, 30), NP = 15;
  var pylDir = new THREE.Vector3().subVectors(PYL_B, PYL_A).normalize(), pylSide = new THREE.Vector3(-pylDir.z, 0, pylDir.x);
  var pylonMesh = new THREE.InstancedMesh(pylonGeometry(), new THREE.MeshLambertMaterial({ vertexColors: true }), NP);
  pylonMesh.frustumCulled = false;
  var pylons = [], pylQ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.atan2(-pylDir.z, pylDir.x));
  for (var pi = 0; pi < NP; pi++) {
    var pp = new THREE.Vector3().lerpVectors(PYL_A, PYL_B, pi / (NP - 1));
    pp.x += Math.sin(pi * 1.7) * 8;
    pp.y = land(pp.x, pp.z) - 0.3;
    pylons.push(pp);
  }
  world.add(pylonMesh);
  var SAG = 14, wirePos = new Float32Array((NP - 1) * ATTACH.length * SAG * 6), wv = 0;
  for (pi = 0; pi < NP - 1; pi++) {
    ATTACH.forEach(function (at) {
      var a = pylons[pi], b = pylons[pi + 1];
      for (var k = 0; k < SAG; k++) {
        [k / SAG, (k + 1) / SAG].forEach(function (t) {
          var sag = 4 * t * (1 - t) * (at[1] > 30 ? 4.5 : 6.5);
          wirePos[wv++] = lerp(a.x, b.x, t) + pylSide.x * at[0];
          wirePos[wv++] = lerp(a.y, b.y, t) + at[1] - sag;
          wirePos[wv++] = lerp(a.z, b.z, t) + pylSide.z * at[0];
        });
      }
    });
  }
  var wireGeo = new THREE.BufferGeometry();
  wireGeo.setAttribute('position', new THREE.BufferAttribute(wirePos, 3));
  var wires = new THREE.LineSegments(wireGeo, new THREE.LineBasicMaterial({ color: '#3c3e42', transparent: true, opacity: 0.8 }));
  wires.frustumCulled = false;
  world.add(wires);

  // ── Weather ──
  // The cloud deck: a sheet high over the valley, lit from inside by lightning.
  var cloudMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, fog: false,
    uniforms: Object.assign({
      uStormLit: { value: new THREE.Color('#4a525e') }, uStormDark: { value: new THREE.Color('#1c2028') },
      uFairLit: { value: new THREE.Color('#ffffff') }, uFairShade: { value: new THREE.Color('#aab4c4') },
      uHorizon: { value: new THREE.Color() }, uCam: { value: new THREE.Vector3() },
      uFlash: { value: 0 }, uFlashPos: { value: new THREE.Vector2() }, uFlashCol: { value: new THREE.Color('#b8c4ff') }
    }, U),
    vertexShader: 'varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }',
    fragmentShader: CLOUD_GLSL + 'uniform vec3 uStormLit; uniform vec3 uStormDark; uniform vec3 uFairLit; uniform vec3 uFairShade; uniform vec3 uHorizon;\n' +
      'uniform vec3 uCam; uniform float uFlash; uniform vec2 uFlashPos; uniform vec3 uFlashCol; varying vec3 vW;\n' +
      'void main(){ vec3 cl = clouds(vW.xz);\n' +
      ' vec3 storm = mix(uStormLit, uStormDark, smoothstep(0.35, 0.75, cl.z));\n' +
      ' vec3 fair = mix(uFairShade, uFairLit, smoothstep(0.5, 0.8, cl.z));\n' +
      ' vec3 col = mix(fair, storm, cl.y);\n' +
      ' col += uFlashCol * uFlash * exp(-pow(distance(vW.xz, uFlashPos) / 520.0, 2.0)) * (0.6 + cl.z);\n' +
      ' float d = distance(vW.xz, uCam.xz);\n' +
      ' col = mix(col, uHorizon, smoothstep(500.0, 4200.0, d) * 0.85);\n' +
      ' gl_FragColor = vec4(col, cl.x * (1.0 - smoothstep(4200.0, 6200.0, d)));\n' +
      ' #include <tonemapping_fragment>\n #include <colorspace_fragment>\n }'
  });
  var cloudDeck = new THREE.Mesh(new THREE.PlaneGeometry(14000, 14000).rotateX(Math.PI / 2), cloudMat);
  cloudDeck.position.y = 340;
  cloudDeck.frustumCulled = false;
  world.add(cloudDeck);

  // Rain curtains hanging under the storm, carried down the valley with it.
  var curtTex = curtainTexture(r), curtains = [];
  for (var k = 0; k < (small ? 9 : 14); k++) {
    var cm = new THREE.Sprite(new THREE.SpriteMaterial({ map: curtTex, transparent: true, depthWrite: false, opacity: 0, color: '#9aa4b2' }));
    cm.userData = { x: (r() - 0.5) * 1400, dz: 120 + r() * 900, w: 180 + r() * 220 };
    cm.scale.set(cm.userData.w, 360, 1);
    world.add(cm);
    curtains.push(cm);
  }
  var rain = rainField({ count: small ? 1600 : 3600, box: [40, 26, 50], speed: 22, windSpeed: 6, opacity: 0.34, color: '#aab6c8' });
  world.add(rain.lines);

  // Lightning: a forked bolt, redrawn for every strike, turned to face you.
  var BOLT_MAX = 900, boltPos = new Float32Array(BOLT_MAX * 3), boltGeo = new THREE.BufferGeometry();
  boltGeo.setAttribute('position', new THREE.BufferAttribute(boltPos, 3));
  var bolt = new THREE.Mesh(boltGeo, new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0, side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
  bolt.frustumCulled = false;
  world.add(bolt);
  var boltGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(220,228,255,1)', 'rgba(160,180,255,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false, opacity: 0 }));
  world.add(boltGlow);
  var flash = 0, restrike = 0, boltFade = 1, nb = 0;
  function seg(ax, ay, bx, by, w) {
    if (nb + 6 > BOLT_MAX) return;
    var dx = bx - ax, dy = by - ay, l = Math.hypot(dx, dy) || 1, nx = -dy / l * w, ny = dx / l * w, o = nb * 3;
    boltPos.set([ax - nx, ay - ny, 0, bx - nx, by - ny, 0, bx + nx, by + ny, 0, ax - nx, ay - ny, 0, bx + nx, by + ny, 0, ax + nx, ay + ny, 0], o);
    nb += 6;
  }
  function jag(ax, ay, bx, by, depth, w, spread) {
    if (depth === 0) { seg(ax, ay, bx, by, w); return; }
    var mx = (ax + bx) / 2 + (Math.random() - 0.5) * Math.abs(by - ay) * spread, my = (ay + by) / 2 + (Math.random() - 0.5) * Math.abs(by - ay) * 0.1;
    jag(ax, ay, mx, my, depth - 1, w, spread);
    jag(mx, my, bx, by, depth - 1, w, spread);
  }
  function strike(kind, cam) {
    var x, z;
    if (kind === 'near') { x = cam.x - 420 + Math.random() * 500; z = -220 - Math.random() * 500; }
    else if (kind === 'far') { x = cam.x - 150 + Math.random() * 750; z = Math.min(U.uFront.value - 350, -500) - Math.random() * 500; }
    else { x = cam.x + 60 + Math.random() * 200; z = -760 - Math.random() * 120; }
    var y = land(x, z), H = cloudDeck.position.y - y;
    nb = 0;
    jag(0, H, (Math.random() - 0.5) * H * 0.25, 0, 6, H * 0.014, 0.55);
    for (var b = 0; b < 2; b++) {
      var t = 0.25 + Math.random() * 0.4, sx = (Math.random() - 0.5) * H * 0.12, sy = H * (1 - t);
      jag(sx, sy, sx + (Math.random() < 0.5 ? -1 : 1) * H * (0.15 + Math.random() * 0.2), sy - H * (0.2 + Math.random() * 0.25), 4, H * 0.005, 0.6);
    }
    boltPos.fill(0, nb * 3);
    boltGeo.attributes.position.needsUpdate = true;
    boltGeo.computeBoundingSphere();
    bolt.position.set(x, y, z);
    bolt.rotation.y = Math.atan2(cam.x - x, cam.z - z);
    boltGlow.position.set(x, cloudDeck.position.y - 30, z);
    boltGlow.scale.setScalar(H * 2.2);
    cloudMat.uniforms.uFlashPos.value.set(x, z);
    boltFade = kind === 'outro' ? 0.7 : kind === 'far' ? 0.75 : 1;
    flash = 1;
    restrike = 0.09 + Math.random() * 0.08;
  }

  // ── Per frame ─────────────────────────────────────────────────────────
  var tmp2 = new THREE.Color(), horizon = new THREE.Color();
  var m4 = new THREE.Matrix4(), s3 = new THREE.Vector3(), p3 = new THREE.Vector3();
  var aspect = 1.6, lastU = 0, drift = 0;

  function skyColor(k, storm, haze, dark, out) {
    out.set(SKY.clear[k]).lerp(tmp2.set(SKY.storm[k]), storm).lerp(tmp2.set(SKY.haze[k]), haze);
    return out.multiplyScalar(1 - dark * 0.6);
  }

  function frame(f) {
    var row = f.row, time = f.time, dt = f.dt;
    var camZ = row[0], dark = row[1], rainAmt = row[2], wind = row[3], storm = row[4], front = row[5], fair = row[6];
    var mine = row[7], cut = row[8], city = row[9], haze = row[10];
    var portrait = aspect < 1, calm = env.reduceMotion ? 0.3 : 1;

    // Camera: on the hill, turning to what each stanza names. A tall phone
    // screen sees mostly hill, so stand further out and higher there.
    if (portrait) camZ -= 22;
    camera.position.set(Math.sin(time * 0.05) * 1.5 * calm, land(0, camZ) + 2.6 + row[13] + (portrait ? 12 : 0), camZ);
    camera.rotation.set(row[12] - (portrait ? 0.08 : 0) - f.my * 0.06, row[11] * (portrait ? 1.35 : 1) - f.mx * 0.12, 0);
    sky.position.copy(camera.position);
    cloudDeck.position.x = camera.position.x;
    cloudDeck.position.z = camera.position.z;

    // Shared uniforms.
    drift += dt * (0.004 + wind * 0.01) * calm;
    U.uDrift.value.set(drift, drift * 0.35);
    U.uTime.value = time;
    U.uWind.value = wind * calm;
    U.uCover.value = storm;
    U.uFront.value = front;
    U.uFair.value = fair;
    U.uMine.value = mine;
    pitMesh.visible = mine > 0.04;
    U.uCut.value = cut;
    U.uCity.value = smooth(0.12, 1, city) * 1.0;
    U.uRoad.value = smooth(0, 0.45, city) * 1.01;
    U.uLights.value = haze * (0.2 + dark * 2.2);

    // Light: storm, then clear afternoon, then a dim brown haze.
    var stormSky = storm * smooth(-1200, 300, front - camZ);        // the storm overhead, not just beyond
    var sunlit = clamp(1 - storm * (1 - smooth(camZ - 200, camZ - 900, front)), 0, 1);
    var clear = sunlit * (1 - haze * 0.55);
    skyColor(0, stormSky, haze, dark, dome.uniforms.top.value);
    skyColor(1, stormSky, haze, dark, dome.uniforms.mid.value);
    skyColor(2, stormSky, haze, dark, horizon);
    dome.uniforms.horizon.value.copy(horizon);
    dome.uniforms.sunColor.value.set('#fff0d0').multiplyScalar(sunlit * (1 - haze * 0.6) * 0.8);
    sunDisc.material.opacity = sunlit * (1 - stormSky) * (0.9 - haze * 0.5);
    sunDisc.material.color.set('#ffffff').lerp(tmp2.set('#ffc890'), haze);
    world.fog.color.copy(horizon).lerp(dome.uniforms.mid.value, 0.45);
    world.fog.density = lerp(0.0007, 0.0015, stormSky) + haze * 0.0004;
    gl.setClearColor(world.fog.color);
    cloudMat.uniforms.uHorizon.value.copy(horizon);
    water.material.emissive.copy(dome.uniforms.mid.value).lerp(horizon, 0.4).multiplyScalar(0.45);
    cloudMat.uniforms.uCam.value.copy(camera.position);
    cloudMat.uniforms.uFairLit.value.set('#ffffff').lerp(tmp2.set('#a8a090'), haze);
    cloudMat.uniforms.uFairShade.value.set('#a7b2c4').lerp(tmp2.set('#7a7266'), haze);

    sun.position.copy(camera.position).addScaledVector(SUN, 500);
    sun.target.position.copy(camera.position);
    sun.intensity = lerp(0.25, 2.6, sunlit) * (1 - haze * 0.55) * (1 - dark * 0.7);
    sun.color.set('#fff0d8').lerp(tmp2.set('#e8b888'), haze);
    U.uShade.value = lerp(0.75, 0.45, haze) * clamp(0.4 + clear, 0, 1);

    // Lightning.
    for (var b = 0; b < BOLTS.length; b++) if (lastU < BOLTS[b][0] && f.u >= BOLTS[b][0]) strike(BOLTS[b][1], camera.position);
    lastU = f.u;
    if (storm > 0.75 && front > -300 && Math.random() < dt * 0.4 * storm) strike(front > 300 ? 'near' : 'far', camera.position);
    if (restrike > 0) { restrike -= dt; if (restrike <= 0) flash = Math.max(flash, 0.85); }
    flash *= Math.exp(-dt * (flash > 0.6 ? 7 : 3.5));
    var flick = flash > 0.15 ? 0.65 + 0.35 * Math.sin(time * 85) : 1, fl = flash * flick * boltFade;
    bolt.material.opacity = smooth(0.25, 0.75, flash) * flick * boltFade;
    boltGlow.material.opacity = fl * 0.55;
    cloudMat.uniforms.uFlash.value = fl * 1.3;
    hemi.intensity = lerp(0.8, 1.15, clear) * (1 - dark * 0.6) + fl * 1.6;
    hemi.color.set('#b8c8e0').lerp(tmp2.set('#c8b8a0'), haze);
    hemi.groundColor.set('#3a4a2c').lerp(tmp2.set('#4a4438'), haze);
    dome.uniforms.horizon.value.lerp(tmp2.set('#8a94b0'), fl * 0.35);

    // Rain, near and hanging from the storm.
    rain.update(f, camera.position, rainAmt, env.reduceMotion);
    curtains.forEach(function (cm) {
      var ud = cm.userData, z = Math.min(front - 260 - ud.dz, -150 - ud.dz * 0.6);
      var y = land(ud.x, z);
      cm.position.set(ud.x, y + 170, z);
      cm.material.opacity = storm * 0.85 * (1 - smooth(-1800, -2600, z));
      cm.material.color.copy(horizon).lerp(tmp2.set('#c8d0e0'), fl * 0.6).multiplyScalar(1.1);
    });

    // Pylons go up one after another, far to near; wires follow.
    var pylT = smooth(0.18, 0.75, city) * NP, nWire = 0;
    for (var k = 0; k < NP; k++) {
      var g = smooth(k, k + 1.4, pylT);
      pylonMesh.setMatrixAt(k, m4.compose(p3.copy(pylons[k]), pylQ, s3.set(1, Math.max(g, 0.001), 1)));
      if (k > 0 && g >= 0.999) nWire = k;
    }
    pylonMesh.count = pylT > 0.01 ? NP : 0;
    pylonMesh.instanceMatrix.needsUpdate = true;
    wireGeo.setDrawRange(0, nWire * ATTACH.length * SAG * 2);

    gl.toneMappingExposure = 1.0 - haze * 0.08 + fl * 0.25;
    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) { aspect = w / h; fitCamera(gl, camera, w, h, dpr, small); },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('encroach', {
  renderer: renderer3d,
  align: ['right', 'left', 'right', 'left', 'left'],
  scrim: 0.62,
  keys: function (T) {
    var n = T.count;
    function s(i) { return T.start(Math.min(i, n - 1)); }
    BOLTS = [[s(0) + 0.2, 'near'], [s(0) + 0.55, 'near'], [s(0) + 0.8, 'near'], [s(0) + 1.15, 'near'],
             [s(1) + 0.45, 'far'], [s(1) + 1.0, 'far'], [s(4) + OUTRO_BOLT, 'outro']];
    //  unit         z    dark rain wind storm front  fair mine cut  city haze yaw    pitch  lift
    return [
      [0,            174, 0.18, 0.6, 0.7, 1.0,  900, 0.0, 0,   0,   0,   0,   0.00,  0.02,  0],
      [0.7,          173, 0.18, 0.7, 0.7, 1.0,  900, 0.0, 0,   0,   0,   0,   0.00,  0.02,  0],
      [s(0) + 0.3,   172, 0.14, 0.9, 0.8, 1.0,  900, 0.0, 0,   0,   0,   0,   0.04,  0.0,   0],    // "the heavens roared"
      [s(0) + 1.3,   170, 0.12, 1.0, 0.9, 1.0,  860, 0.0, 0,   0,   0,   0,  -0.04,  -0.02, 0],    // "lightning filled the skies"
      [s(1) + 0.25,  168, 0.08, 0.7, 0.8, 1.0,  500, 0.0, 0,   0,   0,   0,   0.00,  -0.02, 0],    // over the crest
      [s(1) + 0.9,   166, 0.04, 0.2, 0.5, 0.95, -150, 0.1, 0,  0,   0,   0,   0.00,  -0.08, 0],    // the storm rolls on
      [s(1) + 1.5,   164, 0.00, 0.0, 0.3, 0.9, -900, 0.25, 0,  0,   0,   0,   0.00,  -0.09, 0],
      [s(2) + 0.25,  163, 0.00, 0.0, 0.2, 0.0, -2600, 0.32, 0, 0,   0,   0,  -0.10,  -0.10, 0],    // "we marvel at her beauty"
      [s(2) + 0.6,   162, 0.00, 0.0, 0.2, 0.0, -3000, 0.32, 0.02, 0, 0,   0,   0.20,  -0.13, 0],    // "look around"
      [s(2) + 1.35,  161, 0.00, 0.0, 0.2, 0.0, -3000, 0.30, 1, 0,   0,   0,   0.30,  -0.17, 4],    // "dig up all her treasures"
      [s(3) + 0.3,   161, 0.00, 0.0, 0.2, 0.0, -3000, 0.30, 1, 0.02, 0,  0.05, -0.12, -0.13, 4],
      [s(3) + 1.35,  161, 0.00, 0.0, 0.25, 0.0, -3000, 0.30, 1, 1,  0,   0.12, -0.45, -0.12, 4],    // "never think of giving"
      [s(4) + 0.2,   166, 0.00, 0.0, 0.2, 0.0, -3000, 0.35, 1, 1,   0.05, 0.25, -0.08, -0.15, 16],  // "we build across the countryside"
      [s(4) + 1.35,  178, 0.10, 0.0, 0.15, 0.0, -3000, 0.5, 1, 1,   1,   0.85, 0.00,  -0.2,  30],   // "man is in the way"
      [T.total,      190, 0.32, 0.0, 0.1, 0.0, -3000, 0.6, 1, 1,    1,   1.0,  0.00,  -0.18, 40]
    ];
  },
  sound: {
    src: '/audio/rain.mp3',
    label: 'Play the storm and the valley',
    volume: function (row) { return 0.03 + 0.55 * row[2] + 0.08 * row[4] * (row[5] > 0 ? 1 : 0); },
    cues: [
      { stanza: 0, at: 0.2, play: thunderAt(0) }, { stanza: 0, at: 0.55, play: thunderAt(0.1) },
      { stanza: 0, at: 0.8, play: thunderAt(0) }, { stanza: 0, at: 1.15, play: thunderAt(0.2) },
      { stanza: 1, at: 0.45, play: thunderAt(0.5) }, { stanza: 1, at: 1.0, play: thunderAt(0.7) },
      { stanza: 2, at: 0.75, play: blast },
      { stanza: 3, at: 0.25, play: timber },
      { stanza: 4, at: OUTRO_BOLT, play: thunderAt(1) }
    ]
  }
});
