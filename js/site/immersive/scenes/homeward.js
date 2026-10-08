/*
 * Scene for "Have You Earned Your Tomorrow" (Edgar Guest): walking home down
 * the sidewalk of a small-town street of clapboard houses at the end of the
 * day, from golden hour into night, and a first hint of the morning.
 *
 * I   "This day is almost over": an elm-shaded street of porches and picket
 *     fences, mailboxes and telephone poles; a low gold sun at the end of it
 *     throws long shadows back up the street, and the church bell tolls. The
 *     view turns to the doors and windows you pass.
 * II  "A cheerful greeting to the friend who came along": as the sun goes
 *     down, the windows light up one by one as you pass them, each a small
 *     flare of warmth: "a deed you did today".
 * III "That you helped a single brother": one house has stayed dark. At its
 *     gate a lantern on the step comes alight, then the porch light and the
 *     windows ("a single heart rejoicing"), fireflies rise from the lawn,
 *     and "with courage look ahead": the street lamps come on down the street.
 * IV  At your own gate you turn and look back: "a trail of kindness", warm
 *     lights all down the sidewalk the way you came, under the lit windows.
 *     "As you close your eyes in slumber": the houses go dark one by one and
 *     the night deepens; then, at the far end of the street, the first pale
 *     light of "one more tomorrow".
 *
 * Columns: [unit, walk, dark, flies, wind, yaw, pitch, eve, lit, deed,
 *           trail, sleep, dawn, side, zoom, lamps]
 *   walk  0..1 along the sidewalk to your gate; eve golden hour (0) to night
 *   (1); lit how far along the walk the windows have come on; deed the
 *   lantern and the dark house; trail the lights behind you; sleep the
 *   houses going dark and the dimming; dawn the glow in the east; side a
 *   step towards a gate; zoom a longer lens; lamps the street lamps ahead.
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, softSprite, starField, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout (metres). The street runs west towards -z; the right-hand
// sidewalk is yours. Road, curb, grass strip with the elms, sidewalk, the
// fence line, front lawns, porches, houses. ───────────────────────────────
var ROAD = 4, CURB = 4.2, WALK_IN = 6.0, WALK_OUT = 7.5, FENCE = 7.8, FACE = 17.6, LOT = 19;
var YARD = 0.12, PAVE = 0.15, EYE = 1.62;
var CAM_X = 6.75, GATE_STEP = 1.15;
var Z_START = 14, DEED_Z = -46, HOME_Z = -103, TOWN_N = 110, TOWN_S = -130;
var WALK_LEN = Z_START - HOME_Z;
var RIGHT_LOTS = [87, 68, 49, 30, 11, -8, -27, -46, -65, -84, -103];
var LEFT_LOTS = [81, 62, 43, 24, 5, -14, -33, -52, -71, -90];
var CHURCH_Z = -117;
var POLE_X = -5.6, POLES = [];
for (var pz0 = 101; pz0 > -420; pz0 -= 38) POLES.push(pz0);
function walkAt(z) { return (Z_START - z) / WALK_LEN; }

// ── Painted textures ─────────────────────────────────────────────────────
function canvasTex(w, h, paint, rep) {
  var c = document.createElement('canvas');
  c.width = w; c.height = h;
  paint(c.getContext('2d'), w, h);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  if (rep) t.repeat.set(rep[0], rep[1]);
  t.anisotropy = 8;
  return t;
}

function speckle(x, w, h, r, n, cols, s0, s1) {
  for (var i = 0; i < n; i++) {
    x.fillStyle = cols[Math.floor(r() * cols.length)];
    var s = s0 + r() * (s1 - s0);
    x.fillRect(r() * w, r() * h, s, s);
  }
}

// Mown grass: a warm green with paler and darker tufts.
function grassPaint(r) {
  return function (x, w, h) {
    x.fillStyle = '#6d8a3c';
    x.fillRect(0, 0, w, h);
    for (var i = 0; i < 70; i++) {
      var g = x.createRadialGradient(0, 0, 0, 0, 0, 1), cx = r() * w, cy = r() * h, rad = 10 + r() * 40;
      x.save(); x.translate(cx, cy); x.scale(rad, rad * (0.6 + r() * 0.8));
      g.addColorStop(0, r() < 0.5 ? 'rgba(140,160,70,0.35)' : 'rgba(60,85,35,0.35)');
      g.addColorStop(1, 'rgba(0,0,0,0)');
      x.fillStyle = g; x.fillRect(-1, -1, 2, 2); x.restore();
    }
    speckle(x, w, h, r, 2600, ['#7f9c46', '#5a7832', '#8aa652', '#4e6a2c', '#93a85a'], 1, 3);
  };
}

// Old asphalt: grey with grit, tar-sealed cracks and a patch or two.
function asphaltPaint(r) {
  return function (x, w, h) {
    x.fillStyle = '#5a5754';
    x.fillRect(0, 0, w, h);
    speckle(x, w, h, r, 5000, ['#686460', '#4c4a48', '#73706a', '#44423f'], 1, 2.5);
    x.fillStyle = 'rgba(40,38,36,0.35)';
    x.fillRect(w * 0.15, h * 0.55, w * 0.3, h * 0.2);
    x.strokeStyle = 'rgba(28,26,24,0.8)';
    x.lineWidth = 2;
    for (var k = 0; k < 4; k++) {
      x.beginPath();
      var px = r() * w, py = r() * h;
      x.moveTo(px, py);
      for (var j = 0; j < 6; j++) { px += (r() - 0.5) * 60; py += 10 + r() * 30; x.lineTo(px, py); }
      x.stroke();
    }
  };
}

// Concrete sidewalk slabs, one expansion joint per tile.
function concretePaint(r) {
  return function (x, w, h) {
    x.fillStyle = '#c2b9aa';
    x.fillRect(0, 0, w, h);
    speckle(x, w, h, r, 1500, ['#cbc3b4', '#b4ab9c', '#d4ccbe', '#aaa192'], 1, 2);
    x.fillStyle = 'rgba(70,64,56,0.75)';
    x.fillRect(0, 0, w, 3);
  };
}

// A double-hung sash window, six over six. Instance colours make it a lit
// room or dark glass holding the evening.
function windowTex() {
  return canvasTex(64, 112, function (x, w, h) {
    var g = x.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, '#fff2dc');
    g.addColorStop(0.55, '#ffffff');
    g.addColorStop(1, '#e8d8c0');
    x.fillStyle = g;
    x.fillRect(0, 0, w, h);
    x.fillStyle = '#e8e0d0';
    x.fillRect(0, 0, w, 5); x.fillRect(0, h - 6, w, 6); x.fillRect(0, 0, 5, h); x.fillRect(w - 5, 0, 5, h);
    x.fillRect(0, h / 2 - 3, w, 6);
    x.fillStyle = '#3a3430';
    [w / 3, 2 * w / 3].forEach(function (mx) { x.fillRect(mx - 1, 5, 2, h - 11); });
    [h / 4, 3 * h / 4].forEach(function (my) { x.fillRect(5, my - 1, w - 10, 2); });
  });
}

// ── Geometry helpers ─────────────────────────────────────────────────────
var UP = new THREE.Vector3(0, 1, 0);
function box(w, h, d, x, y, z, col) { return tinted(new THREE.BoxGeometry(w, h, d).translate(x, y, z), col); }

// A tapered cylinder from a to b (for trunks and limbs).
function limb(a, b, r0, r1, col) {
  var d = new THREE.Vector3().subVectors(b, a), len = d.length();
  var g = new THREE.CylinderGeometry(r1, r0, len, 6).translate(0, len / 2, 0);
  g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(UP, d.normalize()));
  return tinted(g.translate(a.x, a.y, a.z), col);
}

// A lumpy leaf clump: a round icosahedron pushed in and out by a smooth
// function of position, so shared vertices stay together.
function clump(rad, x, y, z, sy, seed, col) {
  var g = new THREE.IcosahedronGeometry(rad, 1), p = g.attributes.position;
  for (var i = 0; i < p.count; i++) {
    var vx = p.getX(i), vy = p.getY(i), vz = p.getZ(i);
    var k = 1 + 0.2 * Math.sin(vx * 2.9 + seed) * Math.cos(vz * 2.5 + seed * 1.3) + 0.1 * Math.sin(vy * 4.1 + seed * 0.7);
    p.setXYZ(i, vx * k, vy * k * sy, vz * k);
  }
  return tinted(g.translate(x, y, z), col || '#ffffff');
}

// An American elm: a short trunk splitting into limbs that rise and spread
// like a vase, holding a broad umbrella crown about 9 to 15 m up.
function elmGeometry(r) {
  var wood = [], leaf = [], bark = '#4e463e';
  var fork = new THREE.Vector3(0, 3.2, 0);
  wood.push(limb(new THREE.Vector3(0, 0, 0), fork, 0.3, 0.24, bark));
  var n = 4;
  for (var k = 0; k < n; k++) {
    var a = k / n * Math.PI * 2 + r() * 0.6, out = 2.2 + r() * 1.0;
    var mid = new THREE.Vector3(Math.cos(a) * out, 8.0 + r() * 1.0, Math.sin(a) * out);
    wood.push(limb(fork, mid, 0.15, 0.09, bark));
    var tip = new THREE.Vector3(Math.cos(a + 0.2) * (out + 1.6), mid.y + 1.8, Math.sin(a + 0.2) * (out + 1.6));
    wood.push(limb(mid, tip, 0.08, 0.04, bark));
  }
  for (k = 0; k < 12; k++) {
    var b = k / 12 * Math.PI * 2 + r() * 0.3, rr = 3.2 + r() * 1.5;
    leaf.push(clump(2.0 + r() * 0.7, Math.cos(b) * rr, 10.0 + r() * 1.6 - rr * 0.1, Math.sin(b) * rr, 0.7, k * 1.7));
  }
  for (k = 0; k < 6; k++) {
    var c = r() * Math.PI * 2, cr = r() * 2.2;
    leaf.push(clump(2.3 + r() * 0.6, Math.cos(c) * cr, 11.8 + r() * 1.4, Math.sin(c) * cr, 0.65, 20 + k));
  }
  // A few clumps hanging lower round the rim, the elm's weeping edge.
  for (k = 0; k < 5; k++) {
    var e = r() * Math.PI * 2, er = 4.6 + r() * 0.8;
    leaf.push(clump(1.4 + r() * 0.4, Math.cos(e) * er, 8.6 + r() * 0.8, Math.sin(e) * er, 0.8, 40 + k));
  }
  return { wood: merge(wood), leaf: merge(leaf) };
}

// A rounder maple or oak for the back yards and the fields.
function mapleGeometry(r) {
  var wood = [limb(new THREE.Vector3(0, 0, 0), new THREE.Vector3(0.2, 4.2, 0), 0.32, 0.18, '#3e342c')], leaf = [];
  for (var k = 0; k < 7; k++) {
    var a = r() * Math.PI * 2, d = k === 0 ? 0 : 1.2 + r() * 1.1;
    leaf.push(clump(1.8 + r() * 0.7, Math.cos(a) * d, 5.4 + r() * 2.2 + (k === 0 ? 0.8 : 0), Math.sin(a) * d, 0.85, k * 2.3));
  }
  return { wood: merge(wood), leaf: merge(leaf) };
}

// Clapboard: darken the lower edge of each board by world height, fading the
// lines out where they would be too fine to draw.
function lapMaterial(spacing, depth, rough, wallsOnly) {
  var m = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: rough });
  m.onBeforeCompile = function (sh) {
    sh.vertexShader = 'varying vec3 vWP; varying vec3 vWN;\n' + sh.vertexShader.replace('#include <project_vertex>',
      '#include <project_vertex>\n vWP = (modelMatrix * vec4(transformed, 1.0)).xyz; vWN = normalize(mat3(modelMatrix) * objectNormal);');
    sh.fragmentShader = 'varying vec3 vWP; varying vec3 vWN;\n' + sh.fragmentShader.replace('#include <color_fragment>',
      '#include <color_fragment>\n float lapU = vWP.y / ' + spacing.toFixed(3) + ', lap = fract(lapU);\n' +
      ' float lapK = mix(1.0 - ' + depth.toFixed(3) + ', 1.0, smoothstep(0.0, 0.25, lap));\n' +
      ' lapK = mix(lapK, 1.0 - ' + (depth * 0.4).toFixed(3) + ', smoothstep(0.25, 0.7, fwidth(lapU)));\n' +
      ' diffuseColor.rgb *= mix(1.0, lapK, ' + (wallsOnly ? 'step(abs(vWN.y), 0.5)' : '1.0') + ');');
  };
  return m;
}

// ── The street's houses, decided once so the keys know where things are ─
var SIDING = ['#f1eee4', '#ece0b4', '#b8c7ae', '#a9bfd0', '#e9dcc4', '#b06a56', '#cfc8b8', '#7f93a6', '#e2cfa0'];
var ROOFS = ['#4a4744', '#5a4a40', '#3c4046', '#64503e', '#4a4038'];
var DOORS = ['#7a2a24', '#2a4434', '#22304a', '#6a4a2a', '#1f1f1f', '#8a5a2a'];
var SHUTTERS = ['#2a3a2e', '#1e2630', '#3a2a24', '#2a3240'];
var TRIM = '#f3f0e8';

function lotPlan() {
  var r = rng(41), lots = [];
  function plan(side, zc, k) {
    var deed = side > 0 && zc === DEED_Z, home = side > 0 && zc === HOME_Z;
    var two = r() < 0.6, W = 8.2 + r() * 2.2;
    var full = r() < 0.6 || deed || home, PW = full ? W : W * 0.62;
    var pz = full ? 0 : (r() < 0.5 ? -1 : 1) * (W - PW) / 2;
    lots.push({
      side: side, zc: zc, deed: deed, home: home, W: W, D: 9 + r() * 2, H: two ? 5.8 : 3.7,
      pitch: two ? 0.62 + r() * 0.12 : 0.82 + r() * 0.08, PW: PW, pz: pz, doorZ: pz,
      siding: deed ? '#cfc6b2' : home ? '#f1eee4' : SIDING[(k * 4 + (side > 0 ? 0 : 3) + Math.floor(r() * 2)) % SIDING.length],
      roof: ROOFS[Math.floor(r() * ROOFS.length)], door: DOORS[Math.floor(r() * DOORS.length)],
      shutters: r() < 0.55 ? SHUTTERS[Math.floor(r() * SHUTTERS.length)] : null,
      porchFloor: r() < 0.5 ? '#8a8e94' : '#9a8a74', chim: r() < 0.5 ? -1 : 1,
      front: deed || home ? 'picket' : (function (v) { return v < 0.55 ? 'picket' : v < 0.8 ? 'hedge' : 'none'; })(r()),
      mailbox: r() < 0.7 || home
    });
  }
  RIGHT_LOTS.forEach(function (z, k) { plan(1, z, k); });
  LEFT_LOTS.forEach(function (z, k) { plan(-1, z, k); });
  return lots;
}

// ── Synthesised sound cues ───────────────────────────────────────────────
// A church bell: hum, prime, minor-third tierce, quint and nominal, each
// with its own decay.
function bellStroke(ac, out, f, t, gain) {
  [[0.5, 0.55, 7], [1, 0.8, 5], [1.19, 0.45, 3.5], [1.5, 0.3, 2.6], [2, 0.5, 2.4], [2.52, 0.2, 1.5], [3.01, 0.15, 1.1]].forEach(function (p) {
    var o = ac.createOscillator(), g = ac.createGain();
    o.type = 'sine';
    o.frequency.value = f * p[0];
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain * p[1], t + 0.008);
    g.gain.exponentialRampToValueAtTime(0.0001, t + p[2]);
    o.connect(g); g.connect(out);
    o.start(t); o.stop(t + p[2] + 0.1);
  });
}

// The evening bell from the white church down the street: five slow strokes,
// softened by distance.
function eveningBell(ac, out) {
  var lp = ac.createBiquadFilter(), now = ac.currentTime + 0.05;
  lp.type = 'lowpass';
  lp.frequency.value = 1800;
  lp.connect(out);
  for (var i = 0; i < 5; i++) bellStroke(ac, lp, 233.1, now + i * 2.7, 0.1);
}

// The wind chime on the porch of the house that lights up.
function windChime(ac, out) {
  var now = ac.currentTime + 0.05, notes = [1568, 1760, 2093, 2349, 2637, 3136];
  for (var i = 0; i < 7; i++) {
    var t = now + i * 0.32 + Math.random() * 0.25, f = notes[Math.floor(Math.random() * notes.length)];
    [[1, 0.05, 3.2], [2.76, 0.018, 1.4], [5.4, 0.008, 0.6]].forEach(function (p) {
      var o = ac.createOscillator(), g = ac.createGain();
      o.type = 'sine';
      o.frequency.value = f * p[0];
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(p[1], t + 0.003);
      g.gain.exponentialRampToValueAtTime(0.0001, t + p[2]);
      o.connect(g); g.connect(out);
      o.start(t); o.stop(t + p[2] + 0.05);
    });
  }
}

// Crickets in the dark yards: three, each a carrier pulsed in chirps of three.
function crickets(ac, out) {
  var now = ac.currentTime + 0.05, LEN = 18;
  [[4300, 0.86, 0.012], [4720, 0.71, 0.009], [3950, 1.12, 0.008]].forEach(function (c, k) {
    var o = ac.createOscillator(), g = ac.createGain(), env = ac.createGain();
    o.type = 'sine';
    o.frequency.value = c[0];
    g.gain.value = 0;
    env.gain.setValueAtTime(0.0001, now);
    env.gain.exponentialRampToValueAtTime(1, now + 3 + k);
    env.gain.setValueAtTime(1, now + LEN - 5);
    env.gain.exponentialRampToValueAtTime(0.0001, now + LEN);
    for (var t = now + k * 0.37; t < now + LEN; t += c[1] * (0.9 + Math.random() * 0.2)) {
      for (var p = 0; p < 3; p++) {
        var s = t + p * 0.045;
        g.gain.setValueAtTime(0, s);
        g.gain.linearRampToValueAtTime(c[2], s + 0.006);
        g.gain.linearRampToValueAtTime(0, s + 0.024);
      }
    }
    o.connect(g); g.connect(env); env.connect(out);
    o.start(now); o.stop(now + LEN + 0.1);
  });
}

// ── Light through the evening: golden hour, sunset, dusk, night ─────────
var EVE = [0, 0.35, 0.65, 1];
var LOOKS = {
  zen:   ['#5f86bc', '#4a5c98', '#202c58', '#060b18'],
  mid:   ['#c4bcb4', '#c58a86', '#5d5a86', '#0e1530'],
  hor:   ['#eec08a', '#e8a27a', '#6a6890', '#18203a'],
  glow:  ['#ff9a36', '#ff6a1c', '#c24a26', '#000000'],
  anti:  ['#000000', '#5a3a58', '#3a2c4c', '#000000'],
  cloud: ['#ffe2b0', '#ff8a64', '#9a5a72', '#0c1020'],
  cdark: ['#b89884', '#8a5a6a', '#3a3450', '#080b16'],
  hemiS: ['#f4d4a8', '#d8a08a', '#6a76a8', '#2c3a64'],
  hemiG: ['#5a4a30', '#3e2c28', '#1e1c28', '#0a0a12'],
  sun:   ['#ffc27a', '#ff7a3a', '#000000', '#000000'],
  refl:  ['#5a524c', '#4a3c40', '#262838', '#0a0c14']
};
var HEMI_I = [1.15, 0.85, 0.62, 0.32];

// ── Renderer ─────────────────────────────────────────────────────────────
function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(23), lots = lotPlan(), k;
  var gl = makeRenderer(canvas, { clear: '#141a2c', shadows: !small });
  var world = new THREE.Scene();
  world.fog = new THREE.Fog('#e8c890', 40, 900);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.05, 3200);
  camera.rotation.order = 'YXZ';
  world.add(camera);

  var looks = {};
  Object.keys(LOOKS).forEach(function (k) { looks[k] = LOOKS[k].map(function (h) { return new THREE.Color(h); }); });
  function look(name, eve, out) {
    var L = looks[name];
    for (var i = 0; i < EVE.length - 1; i++) {
      if (eve <= EVE[i + 1] || i === EVE.length - 2) return out.copy(L[i]).lerp(L[i + 1], clamp((eve - EVE[i]) / (EVE[i + 1] - EVE[i]), 0, 1));
    }
    return out;
  }

  // ── Sky: gradient, sunset band, the sun, streaks of cloud, and a dawn ──
  var sky = new THREE.Group();
  world.add(sky);
  var skyU = {
    uZen: { value: new THREE.Color() }, uMid: { value: new THREE.Color() }, uHor: { value: new THREE.Color() },
    uGlow: { value: new THREE.Color() }, uAnti: { value: new THREE.Color() }, uCloud: { value: new THREE.Color() },
    uCDark: { value: new THREE.Color() }, uSunCol: { value: new THREE.Color() }, uSun: { value: new THREE.Vector3(0, 0.1, -1) },
    uDisc: { value: 1 }, uDawn: { value: 0 }, uBright: { value: 1 }, uTime: { value: 0 }
  };
  var dome = new THREE.Mesh(new THREE.SphereGeometry(1500, 48, 24), new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false, uniforms: skyU,
    vertexShader: 'varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform vec3 uZen, uMid, uHor, uGlow, uAnti, uCloud, uCDark, uSunCol, uSun; uniform float uDisc, uDawn, uBright, uTime; varying vec3 vP;\n' +
      'float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }\n' +
      'float noise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);\n' +
      ' return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y); }\n' +
      'float fbm(vec2 p){ float v = 0.0, a = 0.5; for (int i = 0; i < 5; i++){ v += a * noise(p); p = p * 2.03 + vec2(1.7, 9.2); a *= 0.5; } return v; }\n' +
      'void main(){ vec3 d = normalize(vP); float h = max(d.y, 0.0);\n' +
      ' vec3 c = mix(uHor, uMid, smoothstep(0.0, 0.2, h)); c = mix(c, uZen, smoothstep(0.16, 0.75, h));\n' +
      ' vec2 dz = normalize(d.xz + vec2(1e-5)); vec3 sd = normalize(uSun); float az = dot(dz, normalize(sd.xz + vec2(1e-5)));\n' +
      ' float toward = 0.5 + 0.5 * az;\n' +
      ' c += uGlow * 0.6 * pow(toward, 5.0) * exp(-h * 6.5);\n' +
      ' c += uAnti * pow(1.0 - toward, 3.0) * smoothstep(0.0, 0.06, h) * exp(-h * 7.0);\n' +
      // dawn: low in the east (+z), a pale gold notch widening into rose
      ' float east = 0.5 + 0.5 * dz.y;\n' +
      ' c += uDawn * (vec3(1.0, 0.66, 0.38) * pow(east, 10.0) * exp(-h * 11.0) * 2.0 + vec3(0.5, 0.36, 0.52) * pow(east, 2.5) * exp(-h * 4.0) * 0.8 + vec3(0.03, 0.06, 0.14) * smoothstep(0.0, 0.8, h));\n' +
      // streaky clouds near the horizon, lit from the sun's side
      ' vec2 sp = d.xz / (d.y + 0.1);\n' +
      ' float n = fbm(vec2(sp.x * 0.45 + uTime * 0.004, sp.y * 1.9));\n' +
      ' float cl = smoothstep(0.54, 0.78, n) * smoothstep(0.015, 0.07, h) * (1.0 - smoothstep(0.22, 0.45, h));\n' +
      ' vec3 cc = mix(uCDark, uCloud, smoothstep(0.2, 1.0, pow(toward, 2.0)) * smoothstep(0.5, 0.85, n) + 0.15);\n' +
      ' cc = mix(cc, cc + vec3(0.5, 0.36, 0.3) * uDawn, pow(east, 6.0));\n' +
      ' c = mix(c, cc, cl * 0.85);\n' +
      ' float s = max(dot(d, sd), 0.0);\n' +
      ' c += uSunCol * (pow(s, 900.0) * 0.8 + pow(s, 30.0) * 0.3);\n' +
      ' c = mix(c, vec3(1.0, 0.9, 0.68) * 1.6, smoothstep(0.99984, 0.99989, s) * uDisc);\n' +
      ' gl_FragColor = vec4(c * uBright, 1.0);\n #include <colorspace_fragment>\n }'
  }));
  dome.frustumCulled = false;
  sky.add(dome);
  var stars = starField(r, small ? 700 : 1500, 1300, 0.04, 1.4);
  stars.material.opacity = 0;
  sky.add(stars);

  var hemi = new THREE.HemisphereLight('#f4d4a8', '#5a4a30', 1.1);
  var sunLight = new THREE.DirectionalLight('#ffc27a', 3);
  world.add(hemi, sunLight, sunLight.target);
  if (!small) {
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.set(2048, 2048);
    var sc = sunLight.shadow.camera;
    sc.left = -34; sc.right = 34; sc.top = 34; sc.bottom = -34; sc.near = 1; sc.far = 420;
    sunLight.shadow.bias = -0.0006;
    sunLight.shadow.normalBias = 0.06;
    sc.updateProjectionMatrix();
  }

  // ── Ground: lawns, the road, curbs, sidewalks ─────────────────────────
  var grassTex = canvasTex(256, 256, grassPaint(r)), roadTex = canvasTex(256, 256, asphaltPaint(r)), slabTex = canvasTex(128, 128, concretePaint(r));
  function flat(w, d, x, y, z, tex, rep, color, rough) {
    var t = tex.clone();
    t.needsUpdate = true;
    t.repeat.set(rep[0], rep[1]);
    var m = new THREE.Mesh(new THREE.PlaneGeometry(w, d).rotateX(-Math.PI / 2),
      new THREE.MeshStandardMaterial({ map: t, color: color || '#ffffff', roughness: rough || 0.95 }));
    m.position.set(x, y, z);
    m.receiveShadow = true;
    world.add(m);
    return m;
  }
  [-1, 1].forEach(function (s) { flat(1500, 1600, s * (CURB + 750), YARD, -100, grassTex, [1500 / 4, 1600 / 4]); });
  flat(ROAD * 2, 1600, 0, 0, -100, roadTex, [1, 200]);
  var walkLen = TOWN_N - TOWN_S, walkMid = (TOWN_N + TOWN_S) / 2;
  [-1, 1].forEach(function (s) { flat(WALK_OUT - WALK_IN, walkLen, s * (WALK_IN + WALK_OUT) / 2, PAVE, walkMid, slabTex, [1, walkLen / 1.5], '#ffffff', 0.9); });

  // Painted and plain surfaces share vertex-coloured merged meshes: clapboard
  // walls, shingled roofs, and everything else (trim, porches, fences).
  var walls = [], roofs = [], trim = [], foliage = [], windows = [], lamps = [];
  [-1, 1].forEach(function (s) {
    trim.push(box(0.2, 0.17, 1600, s * (CURB - 0.1), 0.085, -100, '#a8a296'));
  });

  var pickets = [];
  var LANTERN = null;

  function addLot(L) {
    var s = L.side, zc = L.zc, W = L.W, D = L.D, H = L.H, p = L.pitch, G = W / 2 * Math.tan(p), top = 0.7 + H;
    var PD = 2.4, FL = 0.75, PW = L.PW, pz = L.pz, dz = L.doorZ;
    var S = [], R = [], T = [], F = [];
    function wx(lx) { return s * (FACE + lx); }
    function wz(lz) { return zc + s * lz; }
    function win(lx, y, lz, nx, nz, w, h, kind) {
      windows.push({ x: wx(lx) + s * nx * 0.03, y: y, z: wz(lz) + s * nz * 0.03, nx: s * nx, nz: s * nz, w: w, h: h,
                     deed: L.deed, home: L.home, kind: kind || 0 });
      // casing, head and sill in white; shutters on some houses
      var sx = nx !== 0, cw = 0.12, dep = 0.07;
      T.push(sx ? box(dep, h + 0.2, cw, lx + nx * 0.02, y, lz - w / 2 - cw / 2, TRIM) : box(cw, h + 0.2, dep, lx - w / 2 - cw / 2, y, lz + nz * 0.02, TRIM));
      T.push(sx ? box(dep, h + 0.2, cw, lx + nx * 0.02, y, lz + w / 2 + cw / 2, TRIM) : box(cw, h + 0.2, dep, lx + w / 2 + cw / 2, y, lz + nz * 0.02, TRIM));
      T.push(sx ? box(dep + 0.04, 0.16, w + 0.42, lx + nx * 0.04, y + h / 2 + 0.12, lz, TRIM) : box(w + 0.42, 0.16, dep + 0.04, lx, y + h / 2 + 0.12, lz + nz * 0.04, TRIM));
      T.push(sx ? box(0.14, 0.07, w + 0.3, lx + nx * 0.07, y - h / 2 - 0.06, lz, TRIM) : box(w + 0.3, 0.07, 0.14, lx, y - h / 2 - 0.06, lz + nz * 0.07, TRIM));
      if (L.shutters && kind === 0) {
        [-1, 1].forEach(function (k) {
          T.push(sx ? box(0.04, h, 0.42, lx + nx * 0.03, y, lz + k * (w / 2 + 0.36), L.shutters) : box(0.42, h, 0.04, lx + k * (w / 2 + 0.36), y, lz + nz * 0.03, L.shutters));
        });
      }
    }

    // Body, foundation and the front gable.
    S.push(box(D, H, W, D / 2, 0.7 + H / 2, 0, L.siding));
    T.push(box(D + 0.1, 0.6, W + 0.1, D / 2, 0.42, 0, '#8a8178'));
    var tri = new THREE.Shape();
    tri.moveTo(-W / 2, 0); tri.lineTo(W / 2, 0); tri.lineTo(0, G); tri.lineTo(-W / 2, 0);
    S.push(tinted(new THREE.ExtrudeGeometry(tri, { depth: D, bevelEnabled: false }).rotateY(Math.PI / 2).translate(0, top, 0), L.siding));
    // Corner boards and the frieze under the gable.
    [-1, 1].forEach(function (k) { T.push(box(0.16, H, 0.16, -0.03, 0.7 + H / 2, k * (W / 2 - 0.05), TRIM)); });
    T.push(box(0.14, 0.24, W + 0.1, -0.05, top, 0, TRIM));
    // Roof slopes with white rake boards at the front, and a ridge cap.
    var Ls = (W / 2 + 0.45) / Math.cos(p);
    [-1, 1].forEach(function (k) {
      var cy = top + G - Math.sin(p) * Ls / 2 + Math.cos(p) * 0.07, cz = k * (Math.cos(p) * Ls / 2 + Math.sin(p) * 0.07);
      R.push(tinted(new THREE.BoxGeometry(D + 0.7, 0.14, Ls).rotateX(k * p).translate(D / 2, cy, cz), L.roof));
      T.push(tinted(new THREE.BoxGeometry(0.1, 0.26, Ls).rotateX(k * p).translate(-0.32, cy - 0.1, cz - k * Math.sin(p) * 0.02), TRIM));
    });
    R.push(box(D + 0.7, 0.16, 0.3, D / 2, top + G + 0.1, 0, L.roof));
    T.push(box(0.7, G + 1.8, 0.9, D * 0.68, top + (G + 1.8) / 2 - 0.2, L.chim * W * 0.2, '#7a4a3a'));

    // The porch: floor, columns, a shed roof, railings, steps.
    T.push(box(PD, 0.15, PW, -PD / 2, FL - 0.075, pz, L.porchFloor));
    T.push(box(PD - 0.1, FL - 0.15 - YARD, PW - 0.1, -PD / 2, (FL - 0.15 + YARD) / 2, pz, '#4a443c'));
    var ncol = PW > 6.5 ? 4 : 3;
    for (var k = 0; k < ncol; k++) {
      T.push(box(0.2, 2.55, 0.2, -PD + 0.15, FL + 1.275, pz - PW / 2 + 0.2 + k * (PW - 0.4) / (ncol - 1), TRIM));
    }
    R.push(tinted(new THREE.BoxGeometry(PD + 0.5, 0.14, PW + 0.4).rotateZ(0.13).translate(-PD / 2 - 0.12, FL + 2.68, pz), L.roof));
    T.push(box(0.1, 0.28, PW + 0.42, -PD - 0.35, FL + 2.5, pz, TRIM));
    function rail(x0, z0, x1, z1) {
      var len = Math.hypot(x1 - x0, z1 - z0), along = Math.abs(x1 - x0) < 0.01;
      if (len < 0.3) return;
      var mx = (x0 + x1) / 2, mz = (z0 + z1) / 2;
      T.push(along ? box(0.1, 0.07, len, mx, FL + 0.9, mz, TRIM) : box(len, 0.07, 0.1, mx, FL + 0.9, mz, TRIM));
      T.push(along ? box(0.06, 0.06, len, mx, FL + 0.12, mz, TRIM) : box(len, 0.06, 0.06, mx, FL + 0.12, mz, TRIM));
      for (var b = 0.12; b < len - 0.05; b += 0.17) {
        var t = b / len;
        T.push(box(0.045, 0.74, 0.045, lerp(x0, x1, t), FL + 0.5, lerp(z0, z1, t), TRIM));
      }
    }
    var fx = -PD + 0.15, zA = pz - PW / 2 + 0.25, zB = pz + PW / 2 - 0.25;
    rail(fx, zA, fx, dz - 0.85);
    rail(fx, dz + 0.85, fx, zB);
    rail(fx, zA, -0.15, zA);
    rail(fx, zB, -0.15, zB);
    T.push(box(0.66, 0.21, 1.7, -PD - 0.33, YARD + 0.105, dz, L.porchFloor));
    T.push(box(0.33, 0.42, 1.7, -PD - 0.165, YARD + 0.21, dz, L.porchFloor));
    T.push(box(0.025, 0.21, 1.7, -PD - 0.66, YARD + 0.105, dz, TRIM));

    // Front door with its casing, the porch lamp beside it.
    T.push(box(0.08, 2.1, 1.0, -0.04, FL + 1.05, dz, L.door));
    T.push(box(0.1, 2.3, 0.14, -0.05, FL + 1.15, dz - 0.57, TRIM));
    T.push(box(0.1, 2.3, 0.14, -0.05, FL + 1.15, dz + 0.57, TRIM));
    T.push(box(0.12, 0.18, 1.3, -0.06, FL + 2.3, dz, TRIM));
    T.push(box(0.14, 0.26, 0.14, -0.12, FL + 1.95, dz - 0.9, '#2a2622'));
    lamps.push({ x: wx(-0.22), y: FL + 1.95, z: wz(dz - 0.9), deed: L.deed, home: L.home, wf: walkAt(wz(dz)), l: 0, thr: 0 });

    // Windows: either side of the door, upstairs or in the gable, and down
    // the sides of the house.
    [-2.0, 2.0].forEach(function (o) {
      var z = dz + o;
      if (Math.abs(z) < W / 2 - 0.7) win(-0.02, FL + 1.3, z, -1, 0, 0.95, 1.5, 0);
    });
    if (H > 5) {
      [-W / 4, W / 4].forEach(function (z) { win(-0.02, 0.7 + H * 0.76, z, -1, 0, 0.9, 1.35, 0); });
      win(-0.02, top + G * 0.36, 0, -1, 0, 0.6, 0.85, 1);
    } else {
      [-0.65, 0.65].forEach(function (z) { win(-0.02, top + G * 0.28, z, -1, 0, 0.8, 1.2, 1); });
    }
    [-1, 1].forEach(function (k) {
      [D * 0.3, D * 0.72].forEach(function (x) {
        win(x, FL + 1.3, k * (W / 2 + 0.02), 0, k, 0.9, 1.45, 0);
        if (H > 5) win(x, 0.7 + H * 0.76, k * (W / 2 + 0.02), 0, k, 0.85, 1.3, 0);
      });
    });

    // Shrubs along the porch, a walk to the gate.
    for (var zz = pz - PW / 2 + 0.5; zz < pz + PW / 2 - 0.4; zz += 0.85) {
      if (Math.abs(zz - dz) < 1.25) continue;
      F.push(clump(0.5 + r() * 0.18, -PD - 0.45, YARD + 0.35, zz, 0.85, zz * 3.1 + zc, r() < 0.5 ? '#3e5a2c' : '#4a6630'));
    }
    var walkFrom = -(FACE - FENCE), walkTo = -PD - 0.66;
    T.push(box(walkTo - walkFrom, 0.04, 1.1, (walkFrom + walkTo) / 2, YARD + 0.02, dz, '#b8b0a2'));

    // Move the local pieces into place.
    function settle(list, into) {
      list.forEach(function (g) { if (s < 0) g.rotateY(Math.PI); into.push(g.translate(s * FACE, 0, zc)); });
    }
    settle(S, walls); settle(R, roofs); settle(T, trim); settle(F, foliage);

    // The street frontage: picket fence with a gate, or a hedge, or open lawn.
    var gateZ = wz(dz), z0 = zc - LOT / 2 + 0.15, z1 = zc + LOT / 2 - 0.15, fxw = s * FENCE;
    if (L.front === 'picket') {
      var step = small ? 0.17 : 0.13;
      for (var pzz = z0; pzz < z1; pzz += step) if (Math.abs(pzz - gateZ) > 0.65) pickets.push(fxw, pzz);
      [[z0, gateZ - 0.65], [gateZ + 0.65, z1]].forEach(function (seg) {
        var len = seg[1] - seg[0], mid = (seg[0] + seg[1]) / 2;
        [0.42, 0.85].forEach(function (y) { trim.push(box(0.05, 0.08, len, fxw + s * 0.05, YARD + y, mid, TRIM)); });
        for (var q = seg[0]; q <= seg[1] + 0.01; q += len / Math.max(1, Math.round(len / 2.4))) trim.push(box(0.11, 1.12, 0.11, fxw + s * 0.06, YARD + 0.56, q, TRIM));
      });
      if (L.deed) LANTERN = { x: wx(-PD - 0.5), y: YARD + 0.21, z: wz(dz - 0.55) };
    } else if (L.front === 'hedge') {
      for (var hz = z0 + 0.3; hz < z1; hz += 0.62) {
        if (Math.abs(hz - gateZ) < 0.9) continue;
        foliage.push(clump(0.55, fxw + s * 0.4, YARD + 0.45, hz, 0.95, hz * 1.7, r() < 0.5 ? '#36502a' : '#3e5a2e'));
      }
    }
    if (L.mailbox) {
      var mz = gateZ + s * 1.3, mx = s * (CURB + 0.45);
      trim.push(box(0.09, 1.05, 0.09, mx, YARD + 0.52, mz, '#5a4a3a'));
      trim.push(box(0.5, 0.2, 0.24, mx, YARD + 1.13, mz, L.home ? '#2c3a4c' : '#3a3a3a'));
      trim.push(tinted(new THREE.CylinderGeometry(0.12, 0.12, 0.5, 10, 1, false, 0, Math.PI).rotateZ(Math.PI / 2).translate(mx, YARD + 1.23, mz), L.home ? '#2c3a4c' : '#3a3a3a'));
      trim.push(box(0.03, 0.2, 0.05, mx + s * 0.13, YARD + 1.33, mz + 0.14, '#b02a20'));
    }
  }
  lots.forEach(addLot);

  // ── The white church, the last building on the left ───────────────────
  (function church() {
    var zc = CHURCH_Z, s = -1, W = 11, D = 20, H = 6.4, p = 0.8, G = W / 2 * Math.tan(p), top = 0.7 + H, CF = 15;
    var S = [], R = [], T = [];
    S.push(box(D, H, W, D / 2, 0.7 + H / 2, 0, '#f2efe6'));
    T.push(box(D + 0.1, 0.6, W + 0.1, D / 2, 0.42, 0, '#8a8178'));
    var tri = new THREE.Shape();
    tri.moveTo(-W / 2, 0); tri.lineTo(W / 2, 0); tri.lineTo(0, G); tri.lineTo(-W / 2, 0);
    S.push(tinted(new THREE.ExtrudeGeometry(tri, { depth: D, bevelEnabled: false }).rotateY(Math.PI / 2).translate(0, top, 0), '#f2efe6'));
    var Ls = (W / 2 + 0.5) / Math.cos(p);
    [-1, 1].forEach(function (k) {
      R.push(tinted(new THREE.BoxGeometry(D + 0.6, 0.15, Ls).rotateX(k * p)
        .translate(D / 2, top + G - Math.sin(p) * Ls / 2 + Math.cos(p) * 0.07, k * (Math.cos(p) * Ls / 2 + Math.sin(p) * 0.07)), '#3c3e44'));
    });
    // The tower: base, belfry with dark louvres, cornices, the spire and cross.
    S.push(box(3.8, 13.6, 3.8, -1.2, 0.7 + 6.8, 0, '#f2efe6'));
    T.push(box(4.2, 0.35, 4.2, -1.2, 14.4, 0, TRIM));
    S.push(box(3.2, 3.0, 3.2, -1.2, 16.0, 0, '#f2efe6'));
    [[-1, 0], [1, 0], [0, -1], [0, 1]].forEach(function (n) {
      T.push(n[0] ? box(0.05, 2.0, 1.5, -1.2 + n[0] * 1.62, 16.0, 0, '#2a2622') : box(1.5, 2.0, 0.05, -1.2, 16.0, n[1] * 1.62, '#2a2622'));
    });
    T.push(box(3.6, 0.3, 3.6, -1.2, 17.6, 0, TRIM));
    R.push(tinted(new THREE.ConeGeometry(2.25, 9.5, 4).rotateY(Math.PI / 4).translate(-1.2, 17.75 + 4.75, 0), '#3c3e44'));
    T.push(box(0.12, 1.4, 0.12, -1.2, 27.9, 0, '#d8c890'));
    T.push(box(0.12, 0.12, 0.7, -1.2, 28.2, 0, '#d8c890'));
    T.push(box(0.1, 2.8, 1.8, -3.12, 0.7 + 1.4, 0, '#7a2a24'));
    T.push(box(0.14, 0.2, 2.1, -3.14, 0.7 + 2.9, 0, TRIM));
    T.push(box(1.6, 0.6, 3.4, -4.0, 0.42, 0, '#a8a296'));
    function cw(lx, y, lz, nx, nz, w, h) { windows.push({ x: s * (CF + lx) + s * nx * 0.03, y: y, z: zc + s * lz + s * nz * 0.03, nx: s * nx, nz: s * nz, w: w, h: h, church: true, kind: 2 }); }
    cw(-3.12, 9.0, 0, -1, 0, 0.9, 1.6);
    [-1, 1].forEach(function (k) { cw(-0.02, 3.6, k * 3.6, -1, 0, 1.0, 2.6); });
    [-1, 1].forEach(function (k) { for (var i = 0; i < 4; i++) cw(3.5 + i * 4.2, 3.7, k * (W / 2 + 0.02), 0, k, 1.1, 3.0); });
    function settle(list, into) { list.forEach(function (g) { if (s < 0) g.rotateY(Math.PI); into.push(g.translate(s * CF, 0, zc)); }); }
    settle(S, walls); settle(R, roofs); settle(T, trim);
  })();

  // ── Far off: a red barn in the fields and the town water tower ────────
  (function farm() {
    var bx = -64, bz = -280;
    trim.push(box(14, 7, 22, bx, 3.5, bz, '#8a2e24'));
    var tri = new THREE.Shape();
    tri.moveTo(-7.4, 0); tri.lineTo(7.4, 0); tri.lineTo(0, 5.5); tri.lineTo(-7.4, 0);
    trim.push(tinted(new THREE.ExtrudeGeometry(tri, { depth: 22.6, bevelEnabled: false }).translate(bx, 7, bz - 11.3), '#4a3e38'));
    var tx = 30, tz = -440;
    for (var k = 0; k < 4; k++) {
      var a = k / 4 * Math.PI * 2 + Math.PI / 4;
      trim.push(limb(new THREE.Vector3(tx + Math.cos(a) * 6.5, 0, tz + Math.sin(a) * 6.5), new THREE.Vector3(tx + Math.cos(a) * 4.8, 25, tz + Math.sin(a) * 4.8), 0.35, 0.3, '#b8b6ae'));
    }
    trim.push(tinted(new THREE.CylinderGeometry(4.6, 6.2, 0.6, 16).translate(tx, 25, tz), '#c4c2ba'));
    trim.push(tinted(new THREE.CylinderGeometry(7, 7, 7.5, 24).translate(tx, 29, tz), '#cfcdc4'));
    trim.push(tinted(new THREE.ConeGeometry(7.2, 3.2, 24).translate(tx, 34.3, tz), '#b4b2aa'));
    trim.push(tinted(new THREE.SphereGeometry(0.5, 8, 6).translate(tx, 36.1, tz), '#b4b2aa'));
  })();

  // ── Telephone poles down the left side, with their wires and lamps ────
  var wireP = [], poleLamps = [];
  POLES.forEach(function (z, k) {
    trim.push(limb(new THREE.Vector3(POLE_X, 0, z), new THREE.Vector3(POLE_X, 10.2, z), 0.16, 0.12, '#4e4034'));
    trim.push(box(2.4, 0.13, 0.13, POLE_X, 9.4, z, '#4e4034'));
    if (z > TOWN_S - 10) {
      trim.push(limb(new THREE.Vector3(POLE_X, 7.6, z), new THREE.Vector3(POLE_X + 2.2, 7.95, z), 0.05, 0.04, '#3a3a3a'));
      trim.push(box(0.75, 0.16, 0.32, POLE_X + 2.45, 7.92, z, '#56585a'));
      poleLamps.push({ x: POLE_X + 2.45, y: 7.8, z: z, k: k });
    }
    var n = POLES[k + 1];
    if (n == null) return;
    [-1.05, 0, 1.05].forEach(function (o) {
      var y0 = o === 0 ? 10.05 : 9.5;
      for (var i = 0; i < 10; i++) {
        var t0 = i / 10, t1 = (i + 1) / 10;
        wireP.push(POLE_X + o, y0 - Math.sin(Math.PI * t0) * 0.7, lerp(z, n, t0), POLE_X + o, y0 - Math.sin(Math.PI * t1) * 0.7, lerp(z, n, t1));
      }
    });
  });
  var wireGeo = new THREE.BufferGeometry();
  wireGeo.setAttribute('position', new THREE.Float32BufferAttribute(wireP, 3));
  world.add(new THREE.LineSegments(wireGeo, new THREE.LineBasicMaterial({ color: '#1e1a16' })));

  var wallMesh = new THREE.Mesh(merge(walls), lapMaterial(0.19, 0.2, 0.85, true));
  var roofMesh = new THREE.Mesh(merge(roofs), lapMaterial(0.24, 0.3, 0.92, false));
  var trimMesh = new THREE.Mesh(merge(trim), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.8 }));
  var bushMesh = new THREE.Mesh(merge(foliage), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95 }));
  [wallMesh, roofMesh, trimMesh, bushMesh].forEach(function (m) { m.castShadow = m.receiveShadow = true; world.add(m); });

  var picketGeo = merge([tinted(new THREE.BoxGeometry(0.075, 0.9, 0.022).translate(0, YARD + 0.45, 0), TRIM),
                         tinted(new THREE.ConeGeometry(0.053, 0.09, 4).rotateY(Math.PI / 4).translate(0, YARD + 0.945, 0), TRIM)]);
  var picketMesh = new THREE.InstancedMesh(picketGeo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.8 }), pickets.length / 2);
  var m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s3 = new THREE.Vector3(), p3 = new THREE.Vector3();
  for (var i = 0; i < pickets.length / 2; i++) picketMesh.setMatrixAt(i, m4.makeRotationY(Math.PI / 2).setPosition(pickets[i * 2], 0, pickets[i * 2 + 1]));
  picketMesh.castShadow = picketMesh.receiveShadow = true;
  world.add(picketMesh);

  // ── Trees: elms along the grass strips, maples behind the houses and out
  // in the fields. Backlit leaves glow where the sun shines through them. ─
  var leafU = { uSunV: { value: new THREE.Vector3() }, uTrans: { value: 1 }, uTransCol: { value: new THREE.Color('#ffc860') } };
  var barkMat = new THREE.MeshStandardMaterial({ color: '#ffffff', vertexColors: true, roughness: 0.95 });
  var leafMat = new THREE.MeshStandardMaterial({ color: '#ffffff', vertexColors: true, roughness: 0.9 });
  leafMat.onBeforeCompile = function (sh) {
    Object.assign(sh.uniforms, leafU);
    sh.fragmentShader = 'uniform vec3 uSunV; uniform float uTrans; uniform vec3 uTransCol;\n' + sh.fragmentShader.replace('#include <emissivemap_fragment>',
      '#include <emissivemap_fragment>\n float trn = pow(max(dot(normalize(-vViewPosition), uSunV), 0.0), 9.0);\n' +
      ' totalEmissiveRadiance += uTransCol * diffuseColor.rgb * trn * uTrans;');
  };
  function forest(geo, list) {
    var wood = new THREE.InstancedMesh(geo.wood, barkMat, list.length), leaf = new THREE.InstancedMesh(geo.leaf, leafMat, list.length);
    var c = new THREE.Color();
    list.forEach(function (t, k) {
      q.setFromAxisAngle(UP, t.rot);
      m4.compose(p3.set(t.x, YARD, t.z), q, s3.setScalar(t.scale));
      wood.setMatrixAt(k, m4);
      leaf.setMatrixAt(k, m4);
      leaf.setColorAt(k, c.setHSL(0.22 + r() * 0.06, 0.36 + r() * 0.12, 0.23 + r() * 0.06));
    });
    [wood, leaf].forEach(function (m) { m.castShadow = true; m.receiveShadow = true; world.add(m); });
    return leaf;
  }
  var elmsA = [], elmsB = [];
  [-1, 1].forEach(function (s) {
    for (var z = TOWN_N - 2 - r() * 4; z > TOWN_S + 4; z -= 13.5 + r() * 3) {
      if (s < 0 && POLES.some(function (pz) { return Math.abs(pz - z) < 3.5; })) continue;
      if (s > 0 && z < Z_START - 3 && z > Z_START - 24) continue;      // keep the first view down the street open
      (r() < 0.5 ? elmsA : elmsB).push({ x: s * (5.1 + (r() - 0.5) * 0.3), z: z, rot: r() * 6.28, scale: 0.9 + r() * 0.22 });
    }
  });
  forest(elmGeometry(rng(5)), elmsA);
  forest(elmGeometry(rng(9)), elmsB);
  var maples = [];
  [-1, 1].forEach(function (s) {
    for (var z = TOWN_N + 30; z > TOWN_S - 10; z -= 9 + r() * 8) maples.push({ x: s * (31 + r() * 16), z: z, rot: r() * 6.28, scale: 1.2 + r() * 0.6 });
  });
  for (var fi = 0; fi < 46; fi++) {
    var fz = fi < 30 ? -160 - r() * 300 : TOWN_N + 20 + r() * 220, fx = (r() < 0.5 ? -1 : 1) * ((fi < 30 ? 14 : 45) + r() * 150);
    maples.push({ x: fx, z: fz, rot: r() * 6.28, scale: 1.1 + r() * 0.9 });
  }
  forest(mapleGeometry(rng(13)), maples);

  // A dark line of woods all round the horizon.
  (function treeLine() {
    var pos = [], N = 220;
    for (var k = 0; k <= N; k++) {
      var a = k / N * Math.PI * 2, rad = 620 + 40 * Math.sin(a * 5.0), h = 9 + 7 * Math.abs(Math.sin(a * 23.0)) + 5 * Math.sin(a * 61.0) * Math.sin(a * 7.0);
      pos.push(Math.cos(a) * rad, -2, -20 + Math.sin(a) * rad, Math.cos(a) * rad, h, -20 + Math.sin(a) * rad);
    }
    var idx = [];
    for (k = 0; k < N; k++) { var b = k * 2; idx.push(b, b + 2, b + 1, b + 1, b + 2, b + 3); }
    var g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setIndex(idx);
    world.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: '#2a3020', side: THREE.DoubleSide })));
  })();

  // ── Windows, lit rooms and their glow ─────────────────────────────────
  var winMesh = new THREE.InstancedMesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: windowTex() }), windows.length);
  var glowTex = softSprite('rgba(255,206,140,1)', 'rgba(255,160,80,0)');
  var winGlowPos = new Float32Array(windows.length * 3), winGlowCol = new Float32Array(windows.length * 3);
  var dr = rng(77);
  windows.forEach(function (w, k) {
    q.setFromAxisAngle(UP, Math.atan2(w.nx, w.nz));
    winMesh.setMatrixAt(k, m4.compose(p3.set(w.x, w.y, w.z), q, s3.set(w.w, w.h, 1)));
    w.wf = walkAt(w.z);
    w.thr = 0.08 + dr() * 0.7;
    w.hue = 0.07 + dr() * 0.04;
    w.dim = 0.8 + dr() * 0.45;
    w.d0 = 0.42 + dr() * 0.38;
    w.l = 0;
    winGlowPos[k * 3] = w.x + w.nx * 0.35; winGlowPos[k * 3 + 1] = w.y; winGlowPos[k * 3 + 2] = w.z + w.nz * 0.35;
  });
  world.add(winMesh);
  lamps.forEach(function (lp) { lp.thr = 0.1 + dr() * 0.6; });

  function glowPoints(posArr, colArr, size) {
    var g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    g.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
    var pts = new THREE.Points(g, new THREE.PointsMaterial({ size: size, map: glowTex, vertexColors: true, transparent: true,
      depthWrite: false, blending: THREE.AdditiveBlending, fog: false }));
    pts.frustumCulled = false;
    world.add(pts);
    return g;
  }
  var winGlow = glowPoints(winGlowPos, winGlowCol, 3.0);

  // Porch lamps and street lamps share one set of glows.
  var NLAMP = lamps.length + poleLamps.length;
  var lampPos = new Float32Array(NLAMP * 3), lampCol = new Float32Array(NLAMP * 3);
  lamps.forEach(function (lp, k) { lampPos[k * 3] = lp.x; lampPos[k * 3 + 1] = lp.y; lampPos[k * 3 + 2] = lp.z; });
  poleLamps.forEach(function (lp, k) { var j = lamps.length + k; lampPos[j * 3] = lp.x; lampPos[j * 3 + 1] = lp.y - 0.2; lampPos[j * 3 + 2] = lp.z; });
  var lampGlow = glowPoints(lampPos, lampCol, 2.2);
  // Pools of street-lamp light on the road.
  var pools = new THREE.InstancedMesh(new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({
    map: softSprite('rgba(255,200,130,0.6)', 'rgba(255,170,100,0)'), transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, polygonOffset: true, polygonOffsetFactor: -2 }), poleLamps.length);
  poleLamps.forEach(function (lp, k) { pools.setMatrixAt(k, m4.compose(p3.set(lp.x, 0.02, lp.z), q.identity(), s3.set(11, 1, 11))); pools.setColorAt(k, new THREE.Color(0, 0, 0)); });
  world.add(pools);
  // Real light from the few lit porches nearest you.
  var NL = small ? 2 : 4, porchLights = [];
  for (i = 0; i < NL; i++) { var pl = new THREE.PointLight('#ffb468', 0, 14, 1.7); world.add(pl); porchLights.push(pl); }

  // ── The lantern on the step ───────────────────────────────────────────
  var lantern = new THREE.Group();
  var tin = new THREE.MeshStandardMaterial({ color: '#3a3632', roughness: 0.5, metalness: 0.6 });
  var chimney = new THREE.MeshBasicMaterial({ color: '#4a4038' });
  lantern.add(new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.11, 0.07, 14).translate(0, 0.035, 0), tin));
  lantern.add(new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.075, 0.2, 14).translate(0, 0.17, 0), chimney));
  lantern.add(new THREE.Mesh(new THREE.ConeGeometry(0.095, 0.08, 14).translate(0, 0.31, 0), tin));
  for (k = 0; k < 4; k++) {
    var ga = k / 4 * Math.PI * 2;
    lantern.add(new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.24, 4).translate(Math.cos(ga) * 0.085, 0.18, Math.sin(ga) * 0.085), tin));
  }
  lantern.add(new THREE.Mesh(new THREE.TorusGeometry(0.075, 0.006, 6, 20, Math.PI).translate(0, 0.35, 0), tin));
  lantern.position.set(LANTERN.x, LANTERN.y, LANTERN.z);
  world.add(lantern);
  var flame = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0, fog: false }));
  flame.position.set(LANTERN.x, LANTERN.y + 0.17, LANTERN.z);
  world.add(flame);
  var flameHalo = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0, fog: false }));
  flameHalo.position.copy(flame.position);
  world.add(flameHalo);
  var lanternLight = new THREE.PointLight('#ffa850', 0, 9, 1.6);
  lanternLight.position.set(LANTERN.x - 0.3, LANTERN.y + 0.45, LANTERN.z);
  world.add(lanternLight);

  // ── Fireflies over the lawns ──────────────────────────────────────────
  var NF = small ? 140 : 320, flyPos = new Float32Array(NF * 3), flyCol = new Float32Array(NF * 3), flyHome = [];
  for (i = 0; i < NF; i++) {
    var side = r() < 0.5 ? -1 : 1;
    flyHome.push({ x: side * (8.3 + r() * 8.5), y: 0.35 + r() * 2.2, z: (r() - 0.5) * 70, ph: r() * 6.28, sp: 0.6 + r() * 1.2, bl: 0.5 + r() * 0.9 });
  }
  var flyGeo = glowPoints(flyPos, flyCol, 0.15);

  // ── The trail of kindness: lights down the sidewalk behind you ────────
  var NT = 0, trailZ = [];
  for (var tz = HOME_Z + 1.2; tz < TOWN_N; tz += 0.95) { trailZ.push(tz); NT++; }
  var trailPos = new Float32Array(NT * 3), trailCol = new Float32Array(NT * 3);
  trailZ.forEach(function (z, k) { trailPos[k * 3] = CAM_X + (k % 2 ? 0.2 : -0.2) + (dr() - 0.5) * 0.08; trailPos[k * 3 + 1] = PAVE + 0.13; trailPos[k * 3 + 2] = z; });
  var trailGeo = glowPoints(trailPos, trailCol, 0.42);

  // ── Per-frame state ───────────────────────────────────────────────────
  var tmp = new THREE.Color(), tmp2 = new THREE.Color(), warm = new THREE.Color(), refl = new THREE.Color(), fogC = new THREE.Color();
  var sunDir = new THREE.Vector3(), fwd = new THREE.Vector3(), tgt = new THREE.Vector3();
  var GLASS = new THREE.Color('#4a4038'), FLAME = new THREE.Color('#ffd08a').multiplyScalar(1.6), PREDAWN = new THREE.Color('#8a96c8');
  var baseFov = 55, portrait = 0, near = [], nearD = [];
  for (i = 0; i < NL; i++) { near.push(-1); nearD.push(0); }

  function frame(f) {
    var row = f.row, time = f.time;
    var walk = row[0], flies = row[2], yaw = row[4], pitch = row[5], eve = row[6], litF = row[7], deed = row[8];
    var trail = row[9], sleep = row[10], dawn = row[11], side = row[12], zoom = row[13], lampsOn = row[14];

    // ── Camera: on the sidewalk, a step towards a gate, a gentle stride ──
    var z = lerp(Z_START, HOME_Z, walk), dist = walk * WALK_LEN;
    var bob = env.reduceMotion ? 0 : Math.abs(Math.sin(dist / 0.75 * Math.PI)) * 0.025;
    camera.position.set(CAM_X + side * GATE_STEP, PAVE + EYE + bob, z);
    // Portrait screens are narrow: walking, keep the houses on the right in
    // view; at the gate, look a little up so the lantern sits under the verse.
    var gate = Math.min(side, 1), ahead = 1 - smooth(0.3, 1.2, Math.abs(yaw));
    camera.rotation.set(pitch + portrait * (0.05 * ahead + 0.11 * gate) - f.my * 0.07,
                        yaw - portrait * (0.16 * ahead * (1 - gate) + 0.17 * gate) - f.mx * 0.16, 0);
    camera.fov = baseFov * (1 - 0.26 * zoom * (1 - 0.5 * portrait));
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld();
    sky.position.copy(camera.position);

    // ── The sun going down at the end of the street ──
    var elev = (10 - 30 * eve) * Math.PI / 180, az = 0.1;
    sunDir.set(Math.sin(az) * Math.cos(elev), Math.sin(elev), -Math.cos(az) * Math.cos(elev));
    var bright = (1 - 0.6 * smooth(0, 1, sleep)) + 0.35 * dawn;
    skyU.uSun.value.copy(sunDir);
    look('zen', eve, skyU.uZen.value);
    look('mid', eve, skyU.uMid.value);
    look('hor', eve, skyU.uHor.value);
    look('glow', eve, skyU.uGlow.value);
    look('anti', eve, skyU.uAnti.value);
    look('cloud', eve, skyU.uCloud.value);
    look('cdark', eve, skyU.uCDark.value);
    look('sun', eve, skyU.uSunCol.value);
    skyU.uDawn.value = dawn;
    skyU.uDisc.value = 1 - smooth(0.28, 0.36, eve);
    skyU.uBright.value = bright;
    skyU.uTime.value = time;
    stars.material.opacity = smooth(0.6, 0.95, eve) * (0.55 + 0.45 * sleep) * (1 - 0.6 * dawn);

    var sunAmt = 1 - smooth(0.2, 0.36, eve);
    sunLight.intensity = 3.1 * sunAmt;
    look('sun', Math.min(eve, 0.34), sunLight.color);
    hemi.color.copy(look('hemiS', eve, tmp));
    hemi.groundColor.copy(look('hemiG', eve, tmp));
    var hi = eve < 0.35 ? lerp(HEMI_I[0], HEMI_I[1], eve / 0.35) : eve < 0.65 ? lerp(HEMI_I[1], HEMI_I[2], (eve - 0.35) / 0.3) : lerp(HEMI_I[2], HEMI_I[3], (eve - 0.65) / 0.35);
    if (dawn > 0) hemi.color.lerp(PREDAWN, dawn * 0.6);
    hemi.intensity = hi * (1 + dawn * 1.4);
    gl.toneMappingExposure = (1 - 0.62 * smooth(0, 1, sleep)) + 0.4 * dawn;
    leafU.uTrans.value = sunAmt * 0.9;
    leafU.uSunV.value.copy(sunDir).transformDirection(camera.matrixWorldInverse);

    // Fog takes the colour of the horizon in the direction you face.
    fwd.set(-Math.sin(yaw), 0, -Math.cos(yaw));
    var toward = 0.5 + 0.5 * (fwd.x * sunDir.x + fwd.z * sunDir.z) / Math.hypot(sunDir.x, sunDir.z);
    fogC.copy(skyU.uHor.value).add(tmp.copy(skyU.uGlow.value).multiplyScalar(Math.pow(toward, 5) * 0.55))
      .add(tmp.copy(skyU.uAnti.value).multiplyScalar(Math.pow(1 - toward, 3) * 0.3));
    var east = 0.5 + 0.5 * fwd.z;
    fogC.r += dawn * 0.5 * Math.pow(east, 6); fogC.g += dawn * 0.3 * Math.pow(east, 6); fogC.b += dawn * 0.26 * Math.pow(east, 6);
    fogC.multiplyScalar(bright);
    tmp.copy(fogC).convertLinearToSRGB();
    world.fog.color.setRGB(tmp.r * 0.92, tmp.g * 0.92, tmp.b * 0.92);

    // Shadows from the low sun, over the stretch of street in view.
    if (!small) {
      tgt.copy(camera.position).addScaledVector(fwd, 22);
      tgt.y = 0;
      sunLight.target.position.copy(tgt);
      sunLight.position.copy(tgt).addScaledVector(sunDir, 200);
      sunLight.target.updateMatrixWorld();
    } else {
      sunLight.position.copy(sunDir).multiplyScalar(100);
    }

    // ── Windows light as you pass; the dark house waits for the lantern ──
    warm.setHSL(0.085, 0.95, 0.6);
    look('refl', eve, refl);
    for (var k = 0; k < windows.length; k++) {
      var w = windows[k], l, ign;
      if (w.deed) ign = smooth(w.d0, w.d0 + 0.1, deed);
      else ign = smooth(0, 1, clamp((litF - w.wf) / 0.03, 0, 1));
      l = ign * (1 - smooth(w.thr, w.thr + 0.07, sleep));
      var flare = 4 * ign * (1 - ign);
      tmp.setHSL(w.hue, 0.92, 0.58).multiplyScalar((1.25 + flare * 0.9) * w.dim);
      tmp2.copy(refl).multiplyScalar(w.dim).lerp(tmp, l);
      winMesh.setColorAt(k, tmp2);
      var gk = l * (0.32 + flare * 1.4) * w.dim;
      winGlowCol[k * 3] = tmp.r * gk * 0.6; winGlowCol[k * 3 + 1] = tmp.g * gk * 0.6; winGlowCol[k * 3 + 2] = tmp.b * gk * 0.6;
    }
    winMesh.instanceColor.needsUpdate = true;
    winGlow.attributes.color.needsUpdate = true;

    // Porch lamps: with their windows, the deed house's after the lantern,
    // yours already on, waiting.
    var dusk = smooth(0.25, 0.5, eve);
    for (k = 0; k < lamps.length; k++) {
      var lp = lamps[k], on;
      if (lp.home) on = dusk;
      else if (lp.deed) on = smooth(0.3, 0.42, deed);
      else on = smooth(0, 1, clamp((litF - lp.wf) / 0.03, 0, 1)) * dusk;
      lp.l = on * (1 - smooth(lp.thr, lp.thr + 0.08, sleep) * (lp.home ? 0.7 : 1));
      var flareL = lp.deed ? 1 + 3 * smooth(0.3, 0.38, deed) * (1 - smooth(0.38, 0.6, deed)) : 1;
      lampCol[k * 3] = lp.l * 1.0 * flareL; lampCol[k * 3 + 1] = lp.l * 0.72 * flareL; lampCol[k * 3 + 2] = lp.l * 0.4 * flareL;
    }
    // Street lamps come on down the street ahead from the deed house.
    for (k = 0; k < poleLamps.length; k++) {
      var pk = poleLamps[k], order = clamp((DEED_Z - pk.z) / 120, -1, 1), sOn = smooth(order * 0.7 + 0.05, order * 0.7 + 0.2, lampsOn);
      if (pk.z > DEED_Z) sOn = smooth(0, 0.3, lampsOn);
      var j = lamps.length + k;
      lampCol[j * 3] = sOn * 1.0; lampCol[j * 3 + 1] = sOn * 0.78; lampCol[j * 3 + 2] = sOn * 0.5;
      pools.setColorAt(k, tmp.setRGB(sOn * 0.8, sOn * 0.6, sOn * 0.36));
    }
    lampGlow.attributes.color.needsUpdate = true;
    pools.instanceColor.needsUpdate = true;

    // The nearest lit porches get real light.
    for (i = 0; i < NL; i++) { near[i] = -1; nearD[i] = 1e9; }
    for (k = 0; k < lamps.length; k++) {
      if (lamps[k].l < 0.05) continue;
      var dd = Math.abs(lamps[k].z - z) + Math.abs(lamps[k].x - camera.position.x) * 0.5;
      for (i = 0; i < NL; i++) {
        if (dd < nearD[i]) {
          for (var m = NL - 1; m > i; m--) { near[m] = near[m - 1]; nearD[m] = nearD[m - 1]; }
          near[i] = k; nearD[i] = dd;
          break;
        }
      }
    }
    for (i = 0; i < NL; i++) {
      var pl = porchLights[i];
      if (near[i] < 0) { pl.intensity = 0; continue; }
      var L = lamps[near[i]];
      pl.position.set(L.x - Math.sign(L.x) * 0.5, L.y + 0.1, L.z);
      pl.intensity = 7 * L.l;
    }

    // ── The lantern: lit as you reach the dark house ──
    var flick = env.reduceMotion ? 1 : 0.9 + 0.06 * Math.sin(time * 13) + 0.04 * Math.sin(time * 23.7 + 1.3);
    var lit = smooth(0.02, 0.28, deed) * (1 - 0.35 * smooth(0.3, 1, sleep));
    chimney.color.copy(GLASS).lerp(FLAME, lit);
    flame.material.opacity = lit * flick;
    flame.scale.setScalar(0.45);
    flameHalo.material.opacity = lit * 0.45 * flick;
    flameHalo.scale.setScalar(2.2 + 0.5 * smooth(0.1, 0.5, deed));
    lanternLight.intensity = lit * 4.5 * flick;

    // ── Fireflies drifting over the lawns round you ──
    var fa = flies * (1 - 0.85 * smooth(0.2, 0.9, sleep)), rate = env.reduceMotion ? 0.3 : 1;
    for (k = 0; k < NF; k++) {
      var fh = flyHome[k], o3 = k * 3;
      var fzr = z + ((((fh.z - z + 35) % 70) + 70) % 70) - 35;
      flyPos[o3] = fh.x + Math.sin(time * 0.3 * fh.sp * rate + fh.ph) * 1.2;
      flyPos[o3 + 1] = fh.y + Math.sin(time * 0.5 * fh.sp * rate + fh.ph * 2) * 0.35 + fa * 0.3;
      flyPos[o3 + 2] = fzr + Math.cos(time * 0.27 * fh.sp * rate + fh.ph) * 1.2;
      var blink = Math.pow(Math.max(0, Math.sin(time * fh.bl * 1.7 + fh.ph * 3)), 4) * fa * 1.8;
      flyCol[o3] = blink * 0.75; flyCol[o3 + 1] = blink * 1.0; flyCol[o3 + 2] = blink * 0.3;
    }
    flyGeo.attributes.position.needsUpdate = true;
    flyGeo.attributes.color.needsUpdate = true;

    // ── The trail: lights running back from your gate along the sidewalk ──
    var reach = trail * (TOWN_N - HOME_Z), tdim = (1 - 0.5 * smooth(0.3, 1, sleep)) * (1 + 0.4 * dawn);
    for (k = 0; k < NT; k++) {
      var d = trailZ[k] - HOME_Z, a = clamp((reach - d) / 6, 0, 1);
      var head = Math.exp(-Math.abs(reach - d) * 0.25) * (trail < 0.99 ? 1 : 0);
      var far = 1 - smooth(Z_START, TOWN_N, trailZ[k]) * 0.85;
      var tw = env.reduceMotion ? 1 : 0.8 + 0.2 * Math.sin(time * 2.1 + k * 1.7);
      var v = (a * tw * far + head * 1.5) * tdim;
      trailCol[k * 3] = v * 1.0; trailCol[k * 3 + 1] = v * 0.7; trailCol[k * 3 + 2] = v * 0.36;
    }
    trailGeo.attributes.color.needsUpdate = true;

    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) { portrait = w < h ? 1 : 0; fitCamera(gl, camera, w, h, dpr, small); baseFov = camera.fov; },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('homeward', {
  renderer: renderer3d,
  align: ['left', 'left', 'left', 'right'],
  scrim: 0.62,
  accent: '#ffc070',
  keys: function (T) {
    var S = T.start, E = T.end, DW = walkAt(DEED_Z);
    //    unit          walk  dark flies wind  yaw    pitch  eve   lit    deed  trail sleep dawn  side  zoom  lamps
    return [
      [0,             0.000, 0.0, 0.0, 0.15,  0.00,  0.04, 0.00, -1.0, 0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0],
      [0.7,           0.010, 0.0, 0.0, 0.15,  0.00,  0.04, 0.02, -1.0, 0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0],
      [S(0) + 0.45,   0.070, 0.0, 0.0, 0.15,  0.06,  0.00, 0.06, -1.0, 0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0],  // "is anybody happier"
      [E(0) - 0.15,   0.160, 0.0, 0.0, 0.15, -0.08,  0.03, 0.14, -0.2, 0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0],  // "this day is almost over"
      [S(1) + 0.2,    0.190, 0.1, 0.0, 0.15, -0.16,  0.02, 0.18,  0.30, 0.0, 0.0,  0.0, 0.0,  0.0,  0.0,  0.0],  // "a cheerful greeting"
      [S(1) + 0.8,    0.270, 0.2, 0.0, 0.15, -0.30,  0.00, 0.27,  0.48, 0.0, 0.0,  0.0, 0.0,  0.0,  0.0,  0.0],  // windows lighting as you come
      [E(1) - 0.1,    0.350, 0.3, 0.1, 0.15, -0.08,  0.03, 0.38,  0.66, 0.0, 0.0,  0.0, 0.0,  0.0,  0.0,  0.0],  // "a deed you did today"
      [S(2) + 0.3,    DW - 0.012, 0.4, 0.2, 0.15, -0.95, -0.04, 0.46, 0.76, 0.12, 0.0, 0.0, 0.0, 0.6, 0.3, 0.0],  // "can you say tonight"
      [S(2) + 0.7,    DW,    0.5, 0.5, 0.15, -1.32, -0.07, 0.52,  0.76, 0.62, 0.0, 0.0,  0.0,  1.0,  0.6,  0.0],  // "helped a single brother"
      [E(2) - 0.4,    DW + 0.004, 0.55, 1.0, 0.15, -1.2, -0.04, 0.58, 0.78, 1.0, 0.0, 0.0, 0.0, 0.9, 0.45, 0.1], // "a single heart rejoicing"
      [E(2) - 0.05,   0.600, 0.6, 1.0, 0.15,  0.02,  0.06, 0.64,  1.0,  1.0, 0.0,  0.0, 0.0,  0.0,  0.0,  1.0],  // "with courage look ahead"
      [S(3) + 0.3,    0.900, 0.7, 0.8, 0.15,  0.25,  0.03, 0.72,  1.4,  1.0, 0.0,  0.0, 0.0,  0.0,  0.0,  1.0],  // "did you waste the day"
      [S(3) + 0.6,    1.000, 0.75, 0.8, 0.15, 3.02,  0.02, 0.78,  1.4,  1.0, 0.45, 0.0, 0.0,  0.0,  0.0,  1.0],  // turning round at the gate
      [S(3) + 0.85,   1.000, 0.8, 0.7, 0.15,  3.02,  0.02, 0.84,  1.4,  1.0, 1.0,  0.0, 0.0,  0.0,  0.0,  1.0],  // "a trail of kindness"
      [E(3) - 0.15,   1.000, 0.9, 0.3, 0.10,  3.02,  0.04, 0.95,  1.4,  1.0, 1.0,  0.8, 0.0,  0.0,  0.0,  1.0],  // "close your eyes in slumber"
      [E(3) + 0.2,    1.000, 1.0, 0.1, 0.10,  3.02,  0.04, 1.00,  1.4,  1.0, 1.0,  1.0, 0.0,  0.0,  0.0,  1.0],
      [T.total,       1.000, 0.8, 0.0, 0.10,  3.02,  0.07, 1.00,  1.4,  1.0, 1.0,  1.0, 1.0,  0.0,  0.0,  1.0]   // "one more tomorrow"
    ];
  },
  sound: {
    src: '/audio/birds.mp3',
    label: 'Play the evening street',
    volume: function (row) { return (0.04 + 0.2 * (1 - smooth(0.3, 0.8, row[6]))) * (1 - 0.9 * row[10]) + 0.22 * row[11]; },
    cues: [
      { stanza: 0, at: 0.35, play: eveningBell },
      { stanza: 2, at: 0.6, play: windChime },
      { stanza: 3, at: 0.1, play: crickets }
    ]
  }
});
