/*
 * Scene for "Where the Mind is Without Fear" (Rabindranath Tagore): first
 * light over the red laterite plain of Birbhum, palmyra palms and a banyan,
 * a wide sandy riverbed, and the sun coming up over it all.
 *
 * I    "the head is held high": looking down at the red earth, the head
 *      lifts to the open plain under a clear pre-dawn sky and the morning
 *      star; "knowledge is free": egrets lift out of the banyan and fly off
 *      into the open sky.
 * II   "broken up into fragments by narrow domestic walls": the low laterite
 *      walls that parcel up the fields shiver, topple and sink into the
 *      earth in puffs of red dust, nearest first, and the plain lies open.
 * III  "tireless striving stretches its arms towards perfection": rays of
 *      the unrisen sun reach up the sky; "the clear stream of reason": down
 *      the bank into a broad riverbed of grey sand, following a bright
 *      thread of water as it winds through it and never loses its way.
 * IV   "ever-widening thought and action": rising above the river, rings of
 *      light widen out over the land and wake it gold behind them; "into
 *      that heaven of freedom ... let my country awake": the sun comes up.
 *
 * The poem is one stanza; maxLines 2 makes it four panels.
 * Columns: [unit, path, -, motes, wind, lift, yaw, pitch, dawn, sun, walls,
 *           birds, rings, rays]
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, softSprite, starField, particleField,
         disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout (metres; you travel towards -z, where the sun will rise) ─────
// A river runs down the land in a broad sandy bed DEPTH below the plain; it
// passes off to the right at the start and curves in under the path, and a
// narrow stream winds along its bed.
var DEPTH = 3.2, BANK = 11, SUN_AZ = 0.3;
function riverX(z) { return 95 * (1 - smooth(-60, -200, z)) + 10 * Math.sin(z * 0.006 + 0.5); }
function bedHalf(z) { return 42 + 9 * Math.sin(z * 0.011 + 1); }
function streamX(z) { return riverX(z) + 14 * Math.sin(z * 0.024 + 2) + 4 * Math.sin(z * 0.057 + 0.3); }
function inBed(x, z) { return 1 - smooth(bedHalf(z), bedHalf(z) + BANK, Math.abs(x - riverX(z))); }
function plain(x, z) {
  return 0.35 * Math.sin(x * 0.05 + z * 0.03) + 0.2 * Math.sin(x * 0.13 - z * 0.07) +
         smooth(420, 1000, Math.abs(x)) * (34 + 20 * Math.sin(z * 0.004 + x * 0.002) + 9 * Math.sin(z * 0.013 + 1));
}
// Low dunes in the sand, flattening out beside the stream, which runs in a
// shallow channel.
function bedFloor(x, z) {
  var s = Math.abs(x - streamX(z));
  return -DEPTH + 0.55 * Math.sin(x * 0.19 + z * 0.05) * Math.sin(z * 0.11 - x * 0.04) * smooth(5, 14, s) -
         0.3 * Math.exp(-s * s / 6);
}
function land(x, z) { var b = inBed(x, z); return b > 0 ? lerp(plain(x, z), bedFloor(x, z), b) : plain(x, z); }
// The camera rides the land without the dunes' bumps.
function camGround(x, z) { return lerp(plain(x, z), -DEPTH, inBed(x, z)); }

// Across the fields, over the bank, then along the stream.
var pathPts = [[0, 8], [0, -12], [1, -40], [4, -70], [10, -102]];
for (var pz = -140; pz >= -780; pz -= 20) pathPts.push([streamX(pz), pz]);
var curve = new THREE.CatmullRomCurve3(pathPts.map(function (p) { return new THREE.Vector3(p[0], 0, p[1]); }));
var RING_AT = 0.8;                      // the rings spread from the stream here
var VILLAGES = [[-125, -265], [150, -520], [-210, -640], [100, -860], [-120, -1060], [240, -300], [-300, -120]];
var BANYAN = [[-46, -80, 1], [-78, -205, 0.9], [82, -430, 1.1], [-170, -880, 1]];

function hash2(a, b) { var s = Math.sin(a * 127.1 + b * 311.7) * 43758.5453; return s - Math.floor(s); }
function vnoise(x, z) {
  return 0.5 + 0.25 * Math.sin(x * 0.071 + Math.sin(z * 0.053) * 2) + 0.25 * Math.sin(z * 0.067 + Math.sin(x * 0.041) * 2.3);
}

// ── Pieces ───────────────────────────────────────────────────────────────
// Push a geometry in or out along a smooth function of position.
function lumpy(geo, amp, seed) {
  var p = geo.attributes.position, v = new THREE.Vector3();
  for (var i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    var k = 1 + amp * (Math.sin(v.x * 3.1 + seed) * Math.cos(v.y * 2.7 - seed) + 0.5 * Math.sin(v.z * 4.3 + v.x * 1.3));
    p.setXYZ(i, v.x * k, v.y * k, v.z * k);
  }
  geo.computeVertexNormals();
  return geo;
}

// Lean a geometry's top by `lean` metres per metre of height squared.
function leaned(geo, lx, lz, h) {
  var p = geo.attributes.position;
  for (var i = 0; i < p.count; i++) {
    var y = p.getY(i) / h;
    p.setX(i, p.getX(i) + lx * y * y * h);
    p.setZ(i, p.getZ(i) + lz * y * y * h);
  }
  geo.computeVertexNormals();
  return geo;
}

// A palmyra palm: a tall grey trunk, swollen a little at the foot, with a
// round head of stiff fan leaves; the oldest hang brown below the rest.
function palmGeometry(r, h) {
  var parts = [], lx = (r() - 0.5) * 0.08, lz = (r() - 0.5) * 0.08;
  var trunk = new THREE.CylinderGeometry(0.2, 0.3, h, 7, 8).translate(0, h / 2, 0);
  var tp = trunk.attributes.position;
  for (var i = 0; i < tp.count; i++) {
    var y = tp.getY(i) / h, k = 1 + 0.35 * Math.exp(-y * 14);
    tp.setX(i, tp.getX(i) * k);
    tp.setZ(i, tp.getZ(i) * k);
  }
  parts.push(tinted(leaned(trunk, lx, lz, h), '#4a4038'));
  var top = new THREE.Vector3(lx * h, h, lz * h), X = new THREE.Vector3(1, 0, 0), q = new THREE.Quaternion();
  parts.push(tinted(new THREE.IcosahedronGeometry(0.75, 0).translate(top.x, top.y + 0.2, top.z), '#3a3a24'));
  var n = 20;
  for (var k = 0; k < n; k++) {
    var a = k * 2.399 + r() * 0.3, f = k / (n - 1), el = lerp(1.25, -0.75, f) + (r() - 0.5) * 0.25;
    var dead = f > 0.82, R = 1.35 + r() * 0.35, pet = 0.9 + r() * 0.5;
    if (dead) el = -1.25 + r() * 0.2;
    // The fan in its own plane: a stalk along +x and a ring of stiff points.
    var pos = [];
    var spikes = 22, prev = null;
    for (var s = 0; s <= spikes; s++) {
      var th = -1.35 + 2.7 * s / spikes, rad = (s % 2 ? 0.78 : 1) * R;
      var pt = [pet + Math.cos(th) * rad, Math.sin(th) * rad];
      if (prev) pos.push(pet, 0, 0, prev[0], prev[1], 0, pt[0], pt[1], 0);
      prev = pt;
    }
    pos.push(0, -0.04, 0, pet, -0.04, 0, pet, 0.04, 0);
    var fan = new THREE.BufferGeometry();
    fan.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    fan.rotateX(Math.PI / 2 + (r() - 0.5) * 1.2);
    fan.computeVertexNormals();
    var dir = new THREE.Vector3(Math.cos(a) * Math.cos(el), Math.sin(el), Math.sin(a) * Math.cos(el));
    fan.applyQuaternion(q.setFromUnitVectors(X, dir));
    fan.translate(top.x, top.y + 0.3, top.z);
    parts.push(tinted(fan, dead ? '#6e5a3a' : (r() < 0.5 ? '#3c5a2a' : '#46642e')));
  }
  return merge(parts);
}

// A banyan: a knot of fused trunks, long level limbs, prop roots dropped to
// the ground all round, and a broad low dome of foliage.
function banyanGeometry(r) {
  var parts = [], bark = '#5e5448', Y = new THREE.Vector3(0, 1, 0), q = new THREE.Quaternion();
  for (var k = 0; k < 5; k++) {
    var a = k / 5 * 6.28;
    parts.push(tinted(new THREE.CylinderGeometry(0.45, 0.85, 7.5, 7).translate(Math.cos(a) * 0.7, 3.75, Math.sin(a) * 0.7), bark));
  }
  for (k = 0; k < 8; k++) {
    var la = k / 8 * 6.28 + r() * 0.5, len = 8 + r() * 5;
    var g = new THREE.CylinderGeometry(0.18, 0.4, len, 6).translate(0, len / 2, 0);
    g.applyQuaternion(q.setFromUnitVectors(Y, new THREE.Vector3(Math.cos(la), 0.35 + r() * 0.2, Math.sin(la)).normalize()));
    parts.push(tinted(g.translate(0, 6, 0), bark));
  }
  for (k = 0; k < 34; k++) {
    var ra = r() * 6.28, rd = 3 + r() * 10.5, rh = 7.4 - rd * 0.12;
    parts.push(tinted(new THREE.CylinderGeometry(0.05 + r() * 0.08, 0.1 + r() * 0.14, rh, 5)
      .translate(Math.cos(ra) * rd, rh / 2, Math.sin(ra) * rd), bark));
  }
  var greens = ['#2d4a26', '#355427', '#28431f', '#3a5a2a'];
  for (k = 0; k < 64; k++) {
    // Clumps over the surface of a low dome, a few hanging under its rim.
    var ca = r() * 6.28, ph = k < 52 ? Math.acos(1 - r() * 0.95) : 1.45 + r() * 0.2, rad = 2.0 + r() * 1.6;
    parts.push(tinted(lumpy(new THREE.IcosahedronGeometry(rad, 1), 0.16, k).scale(1.15, 0.85, 1.15)
      .translate(Math.cos(ca) * Math.sin(ph) * 13, 8.2 + Math.cos(ph) * 5.5, Math.sin(ca) * Math.sin(ph) * 13), greens[k % 4]));
  }
  return merge(parts);
}

// A neem or mango tree: a short trunk and a dense round crown, white so
// instance colours tint it.
function roundTreeGeometry(r) {
  var parts = [tinted(new THREE.CylinderGeometry(0.16, 0.26, 2.8, 6).translate(0, 1.4, 0), '#5a4a3c')];
  parts.push(tinted(lumpy(new THREE.IcosahedronGeometry(1.9, 1), 0.12, 3).scale(1.15, 0.95, 1.15).translate(0, 4.4, 0), '#ffffff'));
  for (var k = 0; k < 13; k++) {
    var a = r() * 6.28, ph = 0.3 + r() * 1.5, rad = 0.9 + r() * 0.6;
    parts.push(tinted(lumpy(new THREE.IcosahedronGeometry(rad, 1), 0.15, k).scale(1.1, 0.9, 1.1)
      .translate(Math.cos(a) * Math.sin(ph) * 2.0, 4.3 + Math.cos(ph) * 1.6, Math.sin(a) * Math.sin(ph) * 2.0), '#ffffff'));
  }
  return merge(parts);
}

// A village hut: mud walls under a four-sided thatch roof.
function hutGeometry() {
  return merge([tinted(new THREE.BoxGeometry(4, 2.2, 3).translate(0, 1.1, 0), '#7e5238'),
                tinted(new THREE.ConeGeometry(3.4, 2.2, 4, 1).rotateY(Math.PI / 4).scale(1.2, 1, 0.95).translate(0, 3.2, 0), '#8a7848')]);
}

// A clump of kash grass: tall thin blades, each tipped with a pale plume.
function kashGeometry(r) {
  var pos = [], col = [], nor = [], base = new THREE.Color('#5a5a2c'), mid = new THREE.Color('#9a8e5a'), plume = new THREE.Color('#e6ded0');
  for (var i = 0; i < 16; i++) {
    var a = r() * 6.28, h = 1.0 + r() * 0.9, lean = 0.1 + r() * 0.4, w = 0.016;
    var ox = Math.cos(a) * 0.1, oz = Math.sin(a) * 0.1, tx = ox + Math.cos(a) * lean * 0.6, tz = oz + Math.sin(a) * lean * 0.6;
    var px = -Math.sin(a) * w, pz = Math.cos(a) * w;
    pos.push(ox - px, 0, oz - pz, ox + px, 0, oz + pz, tx, h * 0.62, tz);
    col.push(base.r, base.g, base.b, base.r, base.g, base.b, mid.r, mid.g, mid.b);
    // Every other blade carries a plume: a slim feather nodding over.
    if (i % 2) continue;
    var ex = tx + Math.cos(a) * lean * 0.7, ez = tz + Math.sin(a) * lean * 0.7;
    pos.push(tx - px * 2.2, h * 0.6, tz - pz * 2.2, tx + px * 2.2, h * 0.6, tz + pz * 2.2, ex, h, ez);
    col.push(mid.r, mid.g, mid.b, mid.r, mid.g, mid.b, plume.r, plume.g, plume.b);
  }
  for (i = 0; i < pos.length / 3; i++) nor.push(0, 1, 0);
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  return geo;
}

// A tuft of short dry grass.
function tuftGeometry(r) {
  var pos = [], nor = [], col = [], root = new THREE.Color('#4a3a1e'), tip = new THREE.Color('#b89a5a');
  for (var i = 0; i < 7; i++) {
    var a = r() * 6.28, w = 0.014 + r() * 0.01, h = 0.25 + r() * 0.3, lean = 0.06 + r() * 0.14;
    var ox = Math.cos(a) * 0.05, oz = Math.sin(a) * 0.05, px = -Math.sin(a) * w, pz = Math.cos(a) * w;
    pos.push(ox - px, 0, oz - pz, ox + px, 0, oz + pz, ox + Math.cos(a) * lean, h, oz + Math.sin(a) * lean);
    nor.push(0, 1, 0, 0, 1, 0, 0, 1, 0);
    col.push(root.r, root.g, root.b, root.r, root.g, root.b, tip.r, tip.g, tip.b);
  }
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  return geo;
}

// An egret in flight, facing +z: two long wings and a slim body. `wing` is
// how far out along the wing a vertex is, for the flap.
function birdGeometry() {
  var pos = [0, 0, 0.32, -0.05, 0, -0.3, 0.05, 0, -0.3,
             0, 0, 0.1, 0, 0, -0.12, -0.62, 0.02, -0.08,
             0, 0, 0.1, 0.62, 0.02, -0.08, 0, 0, -0.12];
  var wing = [0, 0, 0, 0, 0, 1, 0, 1, 0];
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('wing', new THREE.Float32BufferAttribute(wing, 1));
  return geo;
}

// A soft-edged disc for the sun: solid to about half its radius, then a
// quick fall-off and a faint halo.
function sunTexture() {
  var c = document.createElement('canvas');
  c.width = c.height = 256;
  var x = c.getContext('2d'), g = x.createRadialGradient(128, 128, 0, 128, 128, 128);
  [[0, 1], [0.3, 1], [0.36, 0.55], [0.45, 0.16], [0.7, 0.04], [1, 0]].forEach(function (s) {
    g.addColorStop(s[0], 'rgba(255,255,255,' + s[1] + ')');
  });
  x.fillStyle = g;
  x.fillRect(0, 0, 256, 256);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// Rays of an unrisen sun: long soft wedges fanning out from the centre.
function raysTexture(r) {
  var c = document.createElement('canvas');
  c.width = c.height = 512;
  var x = c.getContext('2d');
  x.translate(256, 256);
  x.globalCompositeOperation = 'lighter';
  for (var i = 0; i < 26; i++) {
    var a = -Math.PI + (i + r() * 0.6) / 26 * Math.PI, len = 170 + r() * 86, wd = 0.03 + r() * 0.05;
    var g = x.createLinearGradient(0, 0, Math.cos(a) * len, Math.sin(a) * len);
    g.addColorStop(0, 'rgba(255,240,220,0.5)');
    g.addColorStop(0.5, 'rgba(255,240,220,0.18)');
    g.addColorStop(1, 'rgba(255,240,220,0)');
    x.fillStyle = g;
    x.beginPath();
    x.moveTo(0, 0);
    x.arc(0, 0, len, a - wd, a + wd);
    x.closePath();
    x.fill();
  }
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// A long thin band of cloud, for the streaks low over the dawn.
function streakTexture(r) {
  var c = document.createElement('canvas');
  c.width = 512;
  c.height = 64;
  var x = c.getContext('2d');
  for (var i = 0; i < 40; i++) {
    var px = 40 + r() * 432, py = 26 + (r() - 0.5) * 18, rw = 30 + r() * 70, rh = 5 + r() * 7;
    var g = x.createRadialGradient(0, 0, 0, 0, 0, 1);
    g.addColorStop(0, 'rgba(255,255,255,0.32)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    x.save();
    x.translate(px, py);
    x.scale(rw, rh);
    x.fillStyle = g;
    x.fillRect(-1, -1, 2, 2);
    x.restore();
  }
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// ── Synthesised sound cues ───────────────────────────────────────────────
// The koel's call, "ku-oo", climbing a little each time.
function koel(ac, out) {
  var now = ac.currentTime;
  for (var i = 0; i < 5; i++) {
    var t = now + i * 0.72, f = 640 + i * 60;
    [[0, 0.11, f, f * 1.04], [0.17, 0.32, f * 1.22, f * 1.5]].forEach(function (n) {
      var o = ac.createOscillator(), g = ac.createGain(), s = t + n[0];
      o.type = 'sine';
      o.frequency.setValueAtTime(n[2], s);
      o.frequency.exponentialRampToValueAtTime(n[3], s + n[1]);
      g.gain.setValueAtTime(0.0001, s);
      g.gain.exponentialRampToValueAtTime(0.06, s + 0.03);
      g.gain.exponentialRampToValueAtTime(0.0001, s + n[1] + 0.06);
      o.connect(g);
      g.connect(out);
      o.start(s);
      o.stop(s + n[1] + 0.1);
    });
  }
}

// The walls going down: a low rumble and a scatter of stones knocking.
function crumble(ac, out) {
  var t = ac.currentTime, len = 3, b = ac.createBuffer(1, ac.sampleRate * len, ac.sampleRate), d = b.getChannelData(0);
  for (var i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  var src = ac.createBufferSource(), lp = ac.createBiquadFilter(), g = ac.createGain();
  src.buffer = b;
  lp.type = 'lowpass';
  lp.frequency.value = 260;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.3, t + 0.5);
  g.gain.exponentialRampToValueAtTime(0.0001, t + len);
  src.connect(lp); lp.connect(g); g.connect(out);
  src.start(t);
  for (i = 0; i < 16; i++) {
    var s = t + 0.2 + Math.random() * 1.8, n = ac.createBufferSource(), bp = ac.createBiquadFilter(), k = ac.createGain();
    n.buffer = b;
    bp.type = 'bandpass';
    bp.frequency.value = 900 + Math.random() * 1600;
    bp.Q.value = 3;
    k.gain.setValueAtTime(0.0001, s);
    k.gain.exponentialRampToValueAtTime(0.12, s + 0.004);
    k.gain.exponentialRampToValueAtTime(0.0001, s + 0.09);
    n.connect(bp); bp.connect(k); k.connect(out);
    n.start(s, Math.random() * 2, 0.12);
  }
}

// A tanpura's cycle under the sunrise: Pa, Sa, Sa, low Sa, each pluck
// bright and buzzing at first, then mellowing as it rings.
function tanpura(ac, out) {
  var now = ac.currentTime, sa = 138.59;
  [[sa * 0.75, 0], [sa, 1.0], [sa, 1.9], [sa * 0.5, 2.9]].forEach(function (p) {
    var t = now + p[1], lp = ac.createBiquadFilter(), g = ac.createGain();
    lp.type = 'lowpass';
    lp.Q.value = 3;
    lp.frequency.setValueAtTime(3200, t);
    lp.frequency.exponentialRampToValueAtTime(700, t + 3.5);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.05, t + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 5.5);
    [0, 4].forEach(function (cents) {
      var o = ac.createOscillator();
      o.type = 'sawtooth';
      o.frequency.value = p[0];
      o.detune.value = cents;
      o.connect(lp);
      o.start(t);
      o.stop(t + 5.6);
    });
    lp.connect(g);
    g.connect(out);
  });
}

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(1913);
  var gl = makeRenderer(canvas, { clear: '#1c1a34' });
  var world = new THREE.Scene();
  world.fog = new THREE.Fog('#3a3050', 140, 2700);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.1, 8000);
  var tmp = new THREE.Color(), tmp2 = new THREE.Color(), up = new THREE.Vector3(0, 1, 0);
  var m4 = new THREE.Matrix4(), qt = new THREE.Quaternion(), v3 = new THREE.Vector3(), sc = new THREE.Vector3();

  // ── Sky: night still in the west, the east kindling ──
  var sky = new THREE.Group();
  world.add(sky);
  var skyU = { uTop: { value: new THREE.Color() }, uMid: { value: new THREE.Color() }, uHor: { value: new THREE.Color() },
               uWest: { value: new THREE.Color() }, uGlow: { value: new THREE.Color() },
               uSunDir: { value: new THREE.Vector3(0, 0, -1) }, uSunCol: { value: new THREE.Color() } };
  var dome = new THREE.Mesh(new THREE.SphereGeometry(3400, 40, 20), new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false, uniforms: skyU,
    vertexShader: 'varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform vec3 uTop; uniform vec3 uMid; uniform vec3 uHor; uniform vec3 uWest; uniform vec3 uGlow; uniform vec3 uSunDir; uniform vec3 uSunCol; varying vec3 vP;\n' +
      'void main(){ vec3 d = normalize(vP); float h = max(d.y, 0.0);\n' +
      ' float az = dot(normalize(d.xz + vec2(1e-5)), normalize(uSunDir.xz)) * 0.5 + 0.5;\n' +
      ' vec3 hor = mix(uWest, uHor, smoothstep(0.1, 0.9, az));\n' +
      ' vec3 c = mix(hor, uMid, smoothstep(0.0, 0.22, h)); c = mix(c, uTop, smoothstep(0.14, 0.7, h));\n' +
      ' c += uGlow * pow(az, 5.0) * exp(-h * 6.0);\n' +
      ' float s = max(dot(d, normalize(uSunDir)), 0.0);\n' +
      ' c += uSunCol * (pow(s, 1500.0) * 0.8 + pow(s, 60.0) * 0.3 + pow(s, 6.0) * 0.16);\n' +
      ' gl_FragColor = vec4(c, 1.0);\n #include <tonemapping_fragment>\n #include <colorspace_fragment>\n }'
  }));
  sky.add(dome);
  var stars = starField(r, small ? 900 : 1800, 3300, 0.1, 1.4);
  sky.add(stars);
  var venus = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,250,240,1)', 'rgba(255,240,220,0)'),
    transparent: true, depthWrite: false, fog: false, blending: THREE.AdditiveBlending }));
  venus.position.set(Math.sin(-0.42) * 3200 * 0.97, Math.sin(0.2) * 3200, -Math.cos(-0.42) * 3200 * 0.97);
  venus.scale.setScalar(38);
  sky.add(venus);

  // Thin streaks of cloud low over the east, lit from beneath at dawn.
  var streakTex = streakTexture(r), streaks = [];
  for (var i = 0; i < 14; i++) {
    var sa = SUN_AZ + (r() - 0.5) * (i < 6 ? 0.6 : 1.8), se = 0.02 + r() * 0.08;
    var st = new THREE.Sprite(new THREE.SpriteMaterial({ map: streakTex, transparent: true, depthWrite: false, fog: false }));
    st.position.set(Math.sin(sa) * 3150, Math.sin(se) * 3150, -Math.cos(sa) * 3150);
    st.scale.set(700 + r() * 900, 45 + r() * 40, 1);
    sky.add(st);
    streaks.push(st);
  }

  // The sun, its halo, and the rays it sends up before it rises.
  var sunDisc = new THREE.Sprite(new THREE.SpriteMaterial({ map: sunTexture(), transparent: true, depthWrite: false, fog: false,
    blending: THREE.AdditiveBlending }));
  var sunHalo = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,220,160,0.6)', 'rgba(255,170,90,0)'),
    transparent: true, depthWrite: false, fog: false, blending: THREE.AdditiveBlending }));
  var rays = new THREE.Sprite(new THREE.SpriteMaterial({ map: raysTexture(r), transparent: true, depthWrite: false, fog: false,
    blending: THREE.AdditiveBlending }));
  sky.add(rays, sunHalo, sunDisc);

  // ── Light ──
  var hemi = new THREE.HemisphereLight('#7080b0', '#5a3a28', 1.6);
  var sunL = new THREE.DirectionalLight('#ff9a5a', 0.2);
  world.add(hemi, sunL, sunL.target);

  // ── Land ──
  // The rings of light are drawn in the ground's own shader: bright bands
  // spreading from a point on the stream, waking the land warm behind them.
  var LU = { uRing: { value: 0 }, uRingC: { value: new THREE.Vector2() }, uRingCol: { value: new THREE.Color('#ffb24a') } };
  var groundMat = new THREE.MeshLambertMaterial({ vertexColors: true });
  groundMat.onBeforeCompile = function (sh) {
    Object.keys(LU).forEach(function (k) { sh.uniforms[k] = LU[k]; });
    sh.vertexShader = 'varying vec3 vWP;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n vWP = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    sh.fragmentShader = 'uniform float uRing; uniform vec2 uRingC; uniform vec3 uRingCol; varying vec3 vWP;\n' +
      sh.fragmentShader.replace('#include <emissivemap_fragment>',
      '#include <emissivemap_fragment>\n' +
      ' float rd = distance(vWP.xz, uRingC), glow = 0.0, front = -1.0;\n' +
      ' for (int k = 0; k < 4; k++) {\n' +
      '  float age = uRing - float(k) * 0.15;\n' +
      '  if (age > 0.0) {\n' +
      '   float R = age * 220.0 + age * age * 1500.0, w = 1.6 + R * 0.012;\n' +
      '   glow += exp(-pow((rd - R) / w, 2.0)) * (1.0 - smoothstep(0.55, 1.15, age)) * (1.0 - float(k) * 0.2);\n' +
      '   if (k == 0) front = R;\n' +
      '  }\n' +
      ' }\n' +
      ' float woke = front > 0.0 ? smoothstep(front + 10.0, front - 160.0, rd) : 0.0;\n' +
      ' totalEmissiveRadiance += uRingCol * (glow * 1.4 + woke * 0.1);\n' +
      ' diffuseColor.rgb *= 1.0 + woke * 0.5;');
  };

  var RED = new THREE.Color('#7c4430'), RED2 = new THREE.Color('#5e3424'), DRY = new THREE.Color('#857246'),
      SCRUBC = new THREE.Color('#525c2e'), SAND = new THREE.Color('#b2a48a'), SAND2 = new THREE.Color('#968a74'),
      DAMP = new THREE.Color('#6a6152'), BANKC = new THREE.Color('#86482e'), HILL = new THREE.Color('#5a5446');
  var PADDY = [new THREE.Color('#4e6a2c'), new THREE.Color('#678234'), new THREE.Color('#a08a3e'), new THREE.Color('#5a742e')];
  var sandC = new THREE.Color();
  function groundColor(x, z, y, out) {
    var n1 = vnoise(x, z), n2 = vnoise(x * 2.3 + 40, z * 2.1 - 17), b = inBed(x, z);
    out.copy(RED).lerp(RED2, n1).lerp(DRY, smooth(0.5, 0.75, n2) * 0.8).lerp(SCRUBC, smooth(0.7, 0.9, n1 * 0.6 + n2 * 0.5) * 0.6);
    // Away from the walk, a patchwork of paddy: green, young green and ripe.
    var away = Math.abs(x - (z > -110 ? -0.08 * z : streamX(z))), fu = Math.floor((x * 0.96 + z * 0.28) / 46), fv = Math.floor((z * 0.96 - x * 0.28) / 34);
    var cell = hash2(fu, fv), crop = smooth(60, 150, away) * smooth(bedHalf(z) + BANK + 6, bedHalf(z) + BANK + 30, Math.abs(x - riverX(z)));
    if (cell < 0.68) out.lerp(PADDY[Math.floor(cell / 0.17)], crop * (0.75 + 0.25 * n2));
    if (b > 0) {
      var s = Math.abs(x - streamX(z)), rip = 0.5 + 0.5 * Math.sin(x * 0.9 + z * 0.35 + Math.sin(z * 0.2) * 2);
      sandC.copy(SAND).lerp(SAND2, rip * 0.5 + n2 * 0.3).lerp(DAMP, Math.exp(-s * s / 40));
      out.lerp(BANKC, Math.min(1, b * (1 - b) * 3)).lerp(sandC, smooth(0.55, 1, b));
    }
    return out.lerp(HILL, smooth(380, 800, Math.abs(x)));
  }

  // Near ground in detail round the path; a coarse skirt out to the horizon,
  // sunk where the near ground covers it.
  var NX = 320, NZ0 = 110, NZ1 = -1150;
  function slab(w, d, sx, sz, cx, cz, height) {
    var geo = new THREE.PlaneGeometry(w, d, sx, sz).rotateX(-Math.PI / 2).translate(cx, 0, cz);
    var p = geo.attributes.position, cols = new Float32Array(p.count * 3);
    for (var k = 0; k < p.count; k++) {
      var x = p.getX(k), z = p.getZ(k), y = height(x, z);
      p.setY(k, y);
      groundColor(x, z, y, tmp);
      cols[k * 3] = tmp.r; cols[k * 3 + 1] = tmp.g; cols[k * 3 + 2] = tmp.b;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
    geo.computeVertexNormals();
    return new THREE.Mesh(geo, groundMat);
  }
  var step = small ? 4.5 : 3;
  world.add(slab(NX * 2, NZ0 - NZ1, Math.round(NX * 2 / step), Math.round((NZ0 - NZ1) / step), 0, (NZ0 + NZ1) / 2, land));
  world.add(slab(7000, 7000, small ? 140 : 220, small ? 140 : 220, 0, -1300, function (x, z) {
    var inside = Math.min(NX - Math.abs(x), NZ0 - z, z - NZ1);
    return land(x, z) - smooth(0, 30, inside) * 6;
  }));

  // ── The stream: a bright ribbon down the middle of the sand ──
  var sPos = [], sUv = [], sIdx = [], n = 0, along = 0, lx = streamX(120), lz = 120;
  for (var z = 120; z > -3400; z -= z > -900 ? 1.5 : 8) {
    var x = streamX(z), w = 2.7 + 0.9 * Math.sin(z * 0.043) + Math.max(0, -z - 800) * 0.006;
    var tx = streamX(z - 1) - x, tl = Math.hypot(tx, 1), nx = 1 / tl, nz = tx / tl;
    var y = -DEPTH + 0.08 + smooth(-1150, -1400, z) * 1.6;
    along += Math.hypot(x - lx, z - lz);
    lx = x; lz = z;
    sPos.push(x - nx * w / 2, y, z - nz * w / 2, x + nx * w / 2, y, z + nz * w / 2);
    sUv.push(along, 0, along, 1);
    if (n) sIdx.push(n * 2 - 2, n * 2 - 1, n * 2, n * 2 - 1, n * 2 + 1, n * 2);
    n++;
  }
  var streamGeo = new THREE.BufferGeometry();
  streamGeo.setAttribute('position', new THREE.Float32BufferAttribute(sPos, 3));
  streamGeo.setAttribute('uv', new THREE.Float32BufferAttribute(sUv, 2));
  streamGeo.setIndex(sIdx);
  var streamMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, fog: true, side: THREE.DoubleSide,
    uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, {
      uLo: { value: new THREE.Color() }, uHi: { value: new THREE.Color() }, uGlint: { value: new THREE.Color() },
      uDeep: { value: new THREE.Color() }, uSunDir: { value: new THREE.Vector3(0, 0, -1) }, uTime: { value: 0 }, uBright: { value: 1 } }]),
    vertexShader: 'varying vec2 vUv; varying vec3 vWP;\n#include <fog_pars_vertex>\n' +
      'void main(){ vUv = uv; vec4 wp = modelMatrix * vec4(position, 1.0); vWP = wp.xyz;\n' +
      ' vec4 mvPosition = viewMatrix * wp; gl_Position = projectionMatrix * mvPosition;\n #include <fog_vertex>\n }',
    fragmentShader: 'uniform vec3 uLo; uniform vec3 uHi; uniform vec3 uGlint; uniform vec3 uDeep; uniform vec3 uSunDir; uniform float uTime; uniform float uBright;\n' +
      'varying vec2 vUv; varying vec3 vWP;\n#include <fog_pars_fragment>\n' +
      'float h21(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }\n' +
      'float vn(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);\n' +
      ' return mix(mix(h21(i), h21(i + vec2(1.0, 0.0)), f.x), mix(h21(i + vec2(0.0, 1.0)), h21(i + vec2(1.0, 1.0)), f.x), f.y); }\n' +
      'void main(){ vec3 V = normalize(cameraPosition - vWP); vec3 R = reflect(-V, vec3(0.0, 1.0, 0.0));\n' +
      ' float L = vUv.x, yy = vUv.y;\n' +
      // Ripples drawn out along the flow, as reflections are on moving water.
      ' float n = vn(vec2(L * 0.22 - uTime * 0.5, yy * 7.0)) * 0.6 + vn(vec2(L * 0.8 - uTime * 1.2, yy * 15.0)) * 0.4;\n' +
      // Looking down you see into the water; towards the distance it turns
      // to a mirror of the dawn.
      ' float fres = mix(0.4, 1.0, pow(1.0 - clamp(V.y, 0.0, 1.0), 4.0));\n' +
      ' vec3 refl = mix(uLo, uHi, smoothstep(0.0, 0.35, R.y + (n - 0.5) * 0.12)) * (0.7 + 0.6 * n);\n' +
      ' vec3 col = mix(uDeep, refl, fres) * uBright;\n' +
      ' col += uGlint * pow(max(dot(R, normalize(uSunDir)), 0.0), 60.0) * (0.6 + 2.4 * smoothstep(0.45, 0.8, n));\n' +
      ' float edge = smoothstep(0.0, 0.14, yy) * smoothstep(1.0, 0.86, yy);\n' +
      ' gl_FragColor = vec4(col, edge);\n' +
      ' #include <tonemapping_fragment>\n #include <colorspace_fragment>\n #include <fog_fragment>\n }'
  });
  var stream = new THREE.Mesh(streamGeo, streamMat);
  stream.frustumCulled = false;
  stream.renderOrder = 1;
  world.add(stream);

  // ── The walls: dry laterite blocks parcelling the fields ──
  // Each block knows when it falls (nearest the start first); the shader
  // shivers it, tips it over its foot and sinks it into the earth.
  var blocks = [], ROT = 0.13, CW = 14, CD = 11, cR = Math.cos(ROT), sR = Math.sin(ROT);
  var BL = small ? 0.8 : 0.58, COURSES = small ? 2 : 3, BH = small ? 0.42 : 0.3;
  function corner(i, j) {
    var gx = -84 + i * CW + (hash2(i, j) - 0.5) * 3.5, gz = -16 - j * CD + (hash2(j + 3, i + 7) - 0.5) * 3;
    return [gx * cR - gz * sR, gx * sR + gz * cR];
  }
  function wallFrom(a, b, kind) {
    var dx = b[0] - a[0], dz = b[1] - a[1], len = Math.hypot(dx, dz), cnt = Math.floor(len / BL), rot = Math.atan2(-dz, dx);
    var gate = r() < 0.3 ? Math.floor(r() * (cnt - 4)) : -99;
    for (var c = 0; c < COURSES; c++) {
      for (var k = 0; k < cnt; k++) {
        if (k >= gate && k < gate + 3) continue;
        var u = (k + 0.5 + (c % 2 ? 0.5 : 0)) / cnt;
        if (u > 1) continue;
        if (c === COURSES - 1 && r() < 0.35) continue;        // the top course long broken
        var x = a[0] + dx * u, zz = a[1] + dz * u;
        if (inBed(x, zz) > 0 || Math.abs(x - riverX(zz)) < bedHalf(zz) + BANK + 4) continue;
        if ((zz > -10 && Math.abs(x) < 8) || Math.hypot(x - BANYAN[0][0], zz - BANYAN[0][1]) < 16) continue;
        var dist = Math.hypot(x - 2, zz + 6);
        blocks.push({ x: x, y: land(x, zz) + BH * 0.5 + c * BH - 0.04, z: zz, rot: rot + (r() - 0.5) * 0.08,
                      at: clamp(dist / 150, 0, 1) * 0.72 + r() * 0.18 + (COURSES - 1 - c) * 0.03, seed: r(), kind: kind });
      }
    }
  }
  for (var gi = 0; gi <= 10; gi++) {
    for (var gj = 0; gj <= 8; gj++) {
      if (gi < 10 && r() < 0.85) wallFrom(corner(gi, gj), corner(gi + 1, gj), 0);
      if (gj < 8 && r() < 0.85) wallFrom(corner(gi, gj), corner(gi, gj + 1), 1);
    }
  }
  var blockGeo = new THREE.BoxGeometry(BL * 0.96, BH * 0.94, 0.42);
  var fallArr = new Float32Array(blocks.length * 2);
  var WU = { uFall: { value: 0 }, uTime: { value: 0 } };
  var wallMat = new THREE.MeshLambertMaterial();
  var TIP = 'attribute vec2 aFall; uniform float uFall; uniform float uTime;\n' +
    'float fallK(){ return smoothstep(aFall.x, aFall.x + 0.14, uFall); }\n' +
    'mat3 tipM(float k){ float a = k * (1.1 + aFall.y) * (aFall.y > 0.5 ? 1.0 : -1.0); float c = cos(a), s = sin(a);\n' +
    ' return mat3(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c); }\n';
  wallMat.onBeforeCompile = function (sh) {
    Object.keys(WU).forEach(function (k) { sh.uniforms[k] = WU[k]; });
    sh.vertexShader = TIP + sh.vertexShader
      .replace('#include <beginnormal_vertex>', 'vec3 objectNormal = tipM(fallK()) * vec3(normal);')
      .replace('#include <begin_vertex>',
        'float fk = fallK(), shiver = smoothstep(aFall.x - 0.07, aFall.x, uFall) * (1.0 - fk);\n' +
        'vec3 transformed = position + vec3(0.0, ' + (BH / 2).toFixed(3) + ', 0.0);\n' +
        'transformed = tipM(fk) * transformed;\n' +
        'transformed.y -= ' + (BH / 2).toFixed(3) + ' + fk * fk * 2.4;\n' +
        'transformed.x += sin(uTime * 41.0 + aFall.y * 60.0) * 0.02 * shiver;\n' +
        'transformed *= 1.0 - step(0.999, fk);');
  };
  var walls = new THREE.InstancedMesh(blockGeo, wallMat, blocks.length);
  blocks.forEach(function (b, k) {
    m4.compose(v3.set(b.x, b.y, b.z), qt.setFromAxisAngle(up, b.rot), sc.set(1, 1, 1));
    walls.setMatrixAt(k, m4);
    walls.setColorAt(k, tmp.setHSL(0.03 + b.seed * 0.025, 0.45 + hash2(k, 3) * 0.12, 0.25 + hash2(k, 9) * 0.1));
    fallArr[k * 2] = b.at;
    fallArr[k * 2 + 1] = b.seed;
  });
  blockGeo.setAttribute('aFall', new THREE.InstancedBufferAttribute(fallArr, 2));
  walls.instanceMatrix.needsUpdate = true;
  walls.frustumCulled = false;
  world.add(walls);

  // Red dust puffing up where the blocks go down.
  var dustPos = [], dustA = [];
  blocks.forEach(function (b, k) {
    if (k % 5) return;
    dustPos.push(b.x + (r() - 0.5) * 0.8, b.y, b.z + (r() - 0.5) * 0.8);
    dustA.push(b.at + 0.02, r());
  });
  var dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute('position', new THREE.Float32BufferAttribute(dustPos, 3));
  dustGeo.setAttribute('aFall', new THREE.Float32BufferAttribute(dustA, 2));
  var dustMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false,
    uniforms: { uFall: WU.uFall, uScale: { value: 1 }, uColor: { value: new THREE.Color('#a8603e') }, uMap: { value: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)') } },
    vertexShader: 'attribute vec2 aFall; uniform float uFall; uniform float uScale; varying float vA;\n' +
      'void main(){ float k = smoothstep(aFall.x, aFall.x + 0.3, uFall);\n' +
      ' vec3 p = position + vec3((aFall.y - 0.5) * 2.0 * k, k * (0.8 + aFall.y * 1.4), (fract(aFall.y * 7.3) - 0.5) * 2.0 * k);\n' +
      ' vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_Position = projectionMatrix * mv;\n' +
      ' vA = sin(k * 3.14159) * 0.2;\n' +
      ' gl_PointSize = (0.8 + k * 2.0) * uScale / max(-mv.z, 0.5); }',
    fragmentShader: 'uniform vec3 uColor; uniform sampler2D uMap; varying float vA;\n' +
      'void main(){ float a = texture2D(uMap, gl_PointCoord).a * vA; if (a < 0.005) discard; gl_FragColor = vec4(uColor, a);\n #include <colorspace_fragment>\n }'
  });
  var dust = new THREE.Points(dustGeo, dustMat);
  dust.frustumCulled = false;
  world.add(dust);

  // ── Trees ──
  var plantMat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
  function nearPath(x, z) {
    if (z > -125 && z < 20 && Math.abs(x - z * -0.08) < 10) return true;      // the walk across the fields
    return false;
  }
  // Palmyras along the banks and standing about the plain, alone or in twos.
  var palmSpots = [[17, -34], [-15, -64], [27, -80], [-48, -52], [34, 2]];
  for (i = 0; i < (small ? 80 : 150); i++) {
    var pzz = 80 - r() * 1350, side = r() < 0.5 ? -1 : 1, dd = bedHalf(pzz) + BANK + 3 + Math.pow(r(), 2) * 70;
    palmSpots.push([riverX(pzz) + side * dd, pzz]);
  }
  for (i = 0; i < (small ? 40 : 80); i++) {
    var px = (r() - 0.5) * 900, pz2 = 90 - r() * 1500;
    palmSpots.push([px, pz2]);
    if (r() < 0.3) palmSpots.push([px + 3 + r() * 4, pz2 + (r() - 0.5) * 5]);
  }
  palmSpots = palmSpots.filter(function (p) { return inBed(p[0], p[1]) <= 0 && !nearPath(p[0], p[1]) && Math.abs(p[0]) < 1600; });
  [11, 14, 17].forEach(function (h, k) {
    var mine = palmSpots.filter(function (p, j) { return j % 3 === k; });
    var mesh = new THREE.InstancedMesh(palmGeometry(rng(70 + k), h), plantMat, mine.length);
    mine.forEach(function (p, j) {
      var s = 0.85 + r() * 0.3;
      m4.compose(v3.set(p[0], land(p[0], p[1]) - 0.1, p[1]), qt.setFromAxisAngle(up, r() * 6.28), sc.set(s, s * (0.9 + r() * 0.25), s));
      mesh.setMatrixAt(j, m4);
      mesh.setColorAt(j, tmp.setScalar(0.85 + r() * 0.2));
    });
    world.add(mesh);
  });

  // The banyans: one by the path, where the egrets roost; more far off.
  var banyanMesh = new THREE.InstancedMesh(banyanGeometry(rng(8)), plantMat, BANYAN.length);
  BANYAN.forEach(function (b, k) {
    m4.compose(v3.set(b[0], land(b[0], b[1]) - 0.2, b[1]), qt.setFromAxisAngle(up, k * 1.7), sc.setScalar(b[2]));
    banyanMesh.setMatrixAt(k, m4);
    banyanMesh.setColorAt(k, tmp.setScalar(1));
  });
  world.add(banyanMesh);

  // Villages out on the plain, a thread of cooking smoke over each.
  var hutSpots = [];
  VILLAGES.forEach(function (v) {
    for (var k = 0; k < 9; k++) hutSpots.push([v[0] + (r() - 0.5) * 50, v[1] + (r() - 0.5) * 36]);
  });
  var huts = new THREE.InstancedMesh(hutGeometry(), new THREE.MeshLambertMaterial({ vertexColors: true }), hutSpots.length);
  hutSpots.forEach(function (h, k) {
    m4.compose(v3.set(h[0], land(h[0], h[1]) - 0.1, h[1]), qt.setFromAxisAngle(up, r() * 6.28), sc.setScalar(0.8 + r() * 0.4));
    huts.setMatrixAt(k, m4);
    huts.setColorAt(k, tmp.setScalar(0.85 + r() * 0.25));
  });
  world.add(huts);
  var SMOKE = 70, smPos = [], smSeed = [];
  VILLAGES.forEach(function (v, j) {
    for (var k = 0; k < SMOKE; k++) {
      smPos.push(v[0] + (j % 2 ? 8 : -6), land(v[0], v[1]) + 3, v[1] + 4);
      smSeed.push(k / SMOKE, r());
    }
  });
  var smokeGeo = new THREE.BufferGeometry();
  smokeGeo.setAttribute('position', new THREE.Float32BufferAttribute(smPos, 3));
  smokeGeo.setAttribute('seed', new THREE.Float32BufferAttribute(smSeed, 2));
  var smokeMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false,
    uniforms: { uTime: { value: 0 }, uScale: { value: 1 }, uColor: { value: new THREE.Color('#9aa0b4') }, uAlpha: { value: 0.2 },
                uMap: { value: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)') } },
    vertexShader: 'attribute vec2 seed; uniform float uTime; uniform float uScale; varying float vA;\n' +
      'void main(){ float t = fract(seed.x + uTime * 0.012);\n' +
      ' vec3 p = position + vec3(t * t * 22.0 + sin(t * 7.0 + seed.y * 6.0) * 2.5 * t, t * 40.0, -t * t * 8.0 + (seed.y - 0.5) * 7.0 * t);\n' +
      ' vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_Position = projectionMatrix * mv;\n' +
      ' vA = smoothstep(0.0, 0.1, t) * (1.0 - t) * (1.0 - t);\n' +
      ' gl_PointSize = (4.0 + t * 26.0) * uScale / max(-mv.z, 1.0); }',
    fragmentShader: 'uniform vec3 uColor; uniform float uAlpha; uniform sampler2D uMap; varying float vA;\n' +
      'void main(){ float a = texture2D(uMap, gl_PointCoord).a * vA * uAlpha; if (a < 0.003) discard; gl_FragColor = vec4(uColor, a);\n #include <colorspace_fragment>\n }'
  });
  var smoke = new THREE.Points(smokeGeo, smokeMat);
  smoke.frustumCulled = false;
  world.add(smoke);

  // Neem and babool scattered over the plain and along the banks.
  var scrub = new THREE.InstancedMesh(roundTreeGeometry(rng(5)), plantMat, small ? 260 : 520), sn = 0;
  for (i = 0; i < 4000 && sn < scrub.count; i++) {
    var bx, bz, vg = VILLAGES[i % VILLAGES.length];
    if (r() < 0.35) { bx = vg[0] + (r() - 0.5) * 90; bz = vg[1] + (r() - 0.5) * 70; }
    else if (r() < 0.5) { bz = 60 - r() * 1300; bx = riverX(bz) + (r() < 0.5 ? -1 : 1) * (bedHalf(bz) + BANK + 2 + r() * 50); }
    else { bx = (r() - 0.5) * 800; bz = 90 - r() * 1400; }
    if (inBed(bx, bz) > 0 || nearPath(bx, bz)) continue;
    var s2 = 0.8 + r() * 0.7;
    m4.compose(v3.set(bx, land(bx, bz) - 0.1, bz), qt.setFromAxisAngle(up, r() * 6.28), sc.set(s2, s2 * (0.9 + r() * 0.25), s2));
    scrub.setMatrixAt(sn, m4);
    scrub.setColorAt(sn, tmp.setHSL(0.24 + r() * 0.06, 0.42, 0.14 + r() * 0.07));
    sn++;
  }
  scrub.count = sn;
  world.add(scrub);

  // Grass that sways: dry tufts on the plain, kash along the riverbed.
  var GU = { uClock: { value: 0 }, uWind: { value: 0.2 } };
  function swaying(mat, amt) {
    mat.onBeforeCompile = function (sh) {
      Object.keys(GU).forEach(function (k) { sh.uniforms[k] = GU[k]; });
      sh.vertexShader = 'uniform float uClock; uniform float uWind;\n' + sh.vertexShader.replace('#include <begin_vertex>',
        '#include <begin_vertex>\n float gph = instanceMatrix[3][0] * 0.4 + instanceMatrix[3][2] * 0.3;\n' +
        ' float gb = (sin(uClock * (1.3 + uWind * 2.0) + gph) * 0.6 + 0.4 + uWind * 0.6) * (' + amt + ' + uWind * 0.12) * position.y * position.y;\n' +
        ' transformed.x += gb; transformed.z += gb * 0.4;');
    };
    return mat;
  }
  var tufts = new THREE.InstancedMesh(tuftGeometry(rng(3)), swaying(new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }), '0.05'),
    small ? 5000 : 14000), tn = 0;
  for (i = 0; i < 60000 && tn < tufts.count; i++) {
    var gx = (r() - 0.5) * (r() < 0.6 ? 40 : 110), gz = 14 - r() * 150;
    if (inBed(gx, gz) > 0.02 || (Math.abs(gx - gz * -0.08) < 0.8 && r() < 0.7)) continue;
    m4.compose(v3.set(gx, land(gx, gz), gz), qt.setFromAxisAngle(up, r() * 6.28), sc.set(1.1, 0.45 + r() * 0.6, 1.1));
    tufts.setMatrixAt(tn, m4);
    tufts.setColorAt(tn, tmp.setHSL(0.1 + r() * 0.04, 0.3, 0.55 + r() * 0.35));
    tn++;
  }
  tufts.count = tn;
  world.add(tufts);

  var kash = new THREE.InstancedMesh(kashGeometry(rng(4)), swaying(new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }), '0.03'),
    small ? 1600 : 4200), kn = 0;
  for (i = 0; i < 40000 && kn < kash.count; i++) {
    var kz = -60 - r() * 950, kx, sd = r() < 0.5 ? -1 : 1;
    if (r() < 0.55) kx = streamX(kz) + sd * (5 + Math.pow(r(), 1.6) * 26);     // beside the stream
    else kx = riverX(kz) + sd * (bedHalf(kz) - 4 + r() * 12);                  // along the foot of the banks
    var cl = vnoise(kx * 1.7, kz * 1.3);
    if (cl < 0.45 || Math.abs(kx - streamX(kz)) < 4.5) continue;
    var s3 = 0.7 + r() * 0.6;
    m4.compose(v3.set(kx, land(kx, kz) - 0.05, kz), qt.setFromAxisAngle(up, r() * 6.28), sc.set(s3, s3 * (0.8 + r() * 0.5), s3));
    kash.setMatrixAt(kn, m4);
    kash.setColorAt(kn, tmp.setScalar(0.8 + r() * 0.25));
    kn++;
  }
  kash.count = kn;
  world.add(kash);

  // ── The egrets ──
  var BIRDS = small ? 22 : 36, roost = new THREE.Vector3(BANYAN[0][0], land(BANYAN[0][0], BANYAN[0][1]) + 10, BANYAN[0][1]);
  var birdGeo = birdGeometry(), birdPhase = new Float32Array(BIRDS), birdData = [];
  for (i = 0; i < BIRDS; i++) {
    birdPhase[i] = r() * 6.28;
    birdData.push({ o: new THREE.Vector3((r() - 0.5) * 16, (r() - 0.5) * 4, (r() - 0.5) * 12),
                    spread: new THREE.Vector3((r() - 0.5) * 30, (r() - 0.5) * 16, (r() - 0.5) * 24), delay: r() * 0.35 });
  }
  birdGeo.setAttribute('aPhase', new THREE.InstancedBufferAttribute(birdPhase, 1));
  var BU = { uTime: { value: 0 } };
  var birdMat = new THREE.MeshBasicMaterial({ color: '#d9d4cc', side: THREE.DoubleSide });
  birdMat.onBeforeCompile = function (sh) {
    sh.uniforms.uTime = BU.uTime;
    sh.vertexShader = 'uniform float uTime; attribute float aPhase; attribute float wing;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n transformed.y += sin(uTime * 7.5 + aPhase) * wing * 0.45;');
  };
  var birds = new THREE.InstancedMesh(birdGeo, birdMat, BIRDS);
  birds.frustumCulled = false;
  world.add(birds);
  var bp = new THREE.Vector3(), bv = new THREE.Vector3(), dummy = new THREE.Object3D();
  function flight(t, d, out) {
    // Up out of the tree, then away across the plain towards the dawn.
    return out.set(t * 45 + t * t * 40, t * 50 - t * t * 10, -t * 120).add(roost).add(d.o).addScaledVector(d.spread, t);
  }

  // ── Air ──
  var motes = particleField({ count: small ? 200 : 450, box: [40, 14, 40], fall: [-0.08, 0.06], size: 0.05, color: '#ffd08a',
                              map: softSprite('rgba(255,230,180,1)', 'rgba(255,200,130,0)'), sway: 0.3, windSpeed: 0.6 });
  motes.points.material.blending = THREE.AdditiveBlending;
  world.add(motes.points);
  var pf = { snow: 0, wind: 0, dt: 0, time: 0 };

  // ── Frame ──
  var look = new THREE.Vector3(), sunDir = new THREE.Vector3(), H = 800, portrait = false;
  var ringC = curve.getPointAt(RING_AT);
  LU.uRingC.value.set(ringC.x, ringC.z);
  var cTop = new THREE.Color(), cMid = new THREE.Color(), cHor = new THREE.Color(), cWest = new THREE.Color();

  // Dawn in three steps: the clear blue hour, the saffron before the sun,
  // and gold sunrise.
  function mix3(out, a, b, c, t) { return t < 0.5 ? out.set(a).lerp(tmp2.set(b), t * 2) : out.set(b).lerp(tmp2.set(c), t * 2 - 1); }

  function frame(f) {
    var row = f.row, time = f.time, dt = f.dt;
    var motesAmt = row[2], wind = row[3], lift = row[4], dawn = row[7], sunUp = row[8];
    var fall = row[9], flock = row[10], rings = row[11], rayAmt = row[12];

    // Across the fields and along the stream, looking down the path.
    var t = clamp(f.cam, 0, 0.999);
    curve.getPointAt(t, camera.position);
    camera.position.y = camGround(camera.position.x, camera.position.z) + 1.6 + lift + Math.sin(time * 0.9) * 0.02;
    curve.getPointAt(Math.min(t + 0.03, 1), look);
    // Once up in the air, stop turning with every bend of the stream and
    // face the dawn, keeping the sun to the right of the verse.
    var aerial = smooth(4, 14, lift), head = portrait ? 0.2 : 0.08;
    look.sub(camera.position).setY(0).normalize();
    look.x = lerp(look.x, Math.sin(head), aerial);
    look.z = lerp(look.z, -Math.cos(head), aerial);
    look.multiplyScalar(50).add(camera.position);
    look.y = camera.position.y;
    camera.lookAt(look);
    camera.rotateY(row[5] - f.mx * 0.14);
    camera.rotateX(row[6] + (portrait ? lerp(0.04, -0.06, aerial) : 0) - f.my * 0.06);
    sky.position.copy(camera.position);

    // The sun climbs from below the horizon.
    var el = lerp(-0.07, 0.08, sunUp);
    sunDir.set(Math.sin(SUN_AZ) * Math.cos(el), Math.sin(el), -Math.cos(SUN_AZ) * Math.cos(el));
    sunDisc.position.copy(sunDir).multiplyScalar(3250);
    sunHalo.position.copy(sunDisc.position);
    rays.position.copy(sunDir).multiplyScalar(3250);
    rays.position.y = Math.min(rays.position.y, -60);
    sunDisc.scale.setScalar(500 - sunUp * 60);
    sunDisc.material.color.set('#ff7a2a').lerp(tmp.set('#fff0c8'), smooth(0.3, 1, sunUp));
    sunDisc.material.opacity = smooth(0.2, 0.45, sunUp);
    sunHalo.scale.setScalar(520 + sunUp * 200);
    sunHalo.material.opacity = smooth(0.3, 0.9, dawn) * (0.12 + sunUp * 0.2);
    rays.scale.setScalar(2600);
    rays.material.opacity = rayAmt * 0.55;
    rays.material.rotation = Math.sin(time * 0.05) * 0.02;
    rays.material.color.set('#ffb070').lerp(tmp.set('#ffe0a0'), sunUp);

    // Sky: the clear blue hour, the saffron before sunrise, then gold.
    mix3(cTop, '#0a0f2e', '#1d2c62', '#2e5090', dawn);
    mix3(cMid, '#1e2558', '#5a5488', '#c47a52', dawn);
    mix3(cHor, '#7a4c64', '#f4803a', '#ff9a3a', dawn);
    mix3(cWest, '#1a1c3c', '#3c3a66', '#8a92b0', dawn);
    skyU.uTop.value.copy(cTop);
    skyU.uMid.value.copy(cMid);
    skyU.uHor.value.copy(cHor);
    skyU.uWest.value.copy(cWest);
    mix3(skyU.uGlow.value, '#4a2030', '#a8441a', '#e07a2a', dawn).multiplyScalar(0.7);
    skyU.uSunDir.value.copy(sunDir);
    skyU.uSunCol.value.set('#ff8a3a').lerp(tmp.set('#ffd890'), sunUp).multiplyScalar(0.1 + dawn * 0.25 + sunUp * 0.3);
    stars.material.opacity = 0.85 * (1 - smooth(0.05, 0.45, dawn));
    venus.material.opacity = 1 - smooth(0.3, 0.75, dawn);
    streaks.forEach(function (s, k) {
      mix3(s.material.color, '#3a3050', '#ff8a5a', '#ffd090', clamp(dawn + (k % 3) * 0.05, 0, 1));
      s.material.opacity = 0.5 + dawn * 0.5;
    });

    // Fog is the haze over the eastern plain.
    world.fog.color.copy(cHor).lerp(cMid, 0.35).multiplyScalar(0.9);
    world.fog.near = 140 + dawn * 60;
    world.fog.far = 2700 - dawn * 300;
    gl.setClearColor(world.fog.color);
    gl.toneMappingExposure = 1.0;

    // Light: the cool blue hour, warming; the low sun rakes across at the end.
    hemi.color.set('#7080b0').lerp(tmp.set('#d8c0b0'), dawn);
    hemi.groundColor.set('#5a3a28').lerp(tmp.set('#6a4a34'), dawn);
    hemi.intensity = 1.6 - smooth(0.5, 1, dawn) * 0.75;
    sunL.position.copy(camera.position).addScaledVector(sunDir.y > 0.02 ? sunDir : v3.copy(sunDir).setY(0.02), 300);
    sunL.target.position.copy(camera.position);
    sunL.color.set('#ff8a4a').lerp(tmp.set('#ffd090'), sunUp);
    sunL.intensity = 0.3 + dawn * 0.5 + smooth(0.25, 0.8, sunUp) * 1.2;

    // The stream reflects the brightest of the sky, and the sun's glitter.
    streamMat.uniforms.uLo.value.copy(cHor).multiplyScalar(1.15);
    // Once the sun is up the whole stream runs gold with it.
    streamMat.uniforms.uHi.value.copy(cTop).lerp(cMid, 0.6).lerp(cHor, smooth(0.1, 0.7, sunUp) * 0.75);
    streamMat.uniforms.uDeep.value.set('#1c2030').lerp(tmp.set('#4a4038'), dawn);
    streamMat.uniforms.uGlint.value.set('#ffe0a8').multiplyScalar(0.25 + sunUp * 1.2);
    streamMat.uniforms.uSunDir.value.copy(sunDir).setY(Math.max(sunDir.y, 0.01));
    streamMat.uniforms.uTime.value = env.reduceMotion ? time * 0.3 : time;
    streamMat.uniforms.uBright.value = 1.25 + dawn * 0.2;

    smokeMat.uniforms.uTime.value = time;
    smokeMat.uniforms.uScale.value = H * 0.9;
    smokeMat.uniforms.uColor.value.set('#4a4c62').lerp(tmp.set('#b89c88'), dawn);

    // Walls and their dust.
    WU.uFall.value = fall;
    WU.uTime.value = time;
    dustMat.uniforms.uScale.value = H * 0.9;
    walls.visible = fall < 1.2;
    dust.visible = fall > 0 && fall < 1.3;

    // Rings of light widen out over the land.
    LU.uRing.value = rings;
    LU.uRingCol.value.set('#ff9a3a').lerp(tmp.set('#ffd27a'), sunUp);

    // The egrets: each leaves the banyan in its turn and flies off.
    BU.uTime.value = time;
    for (var k = 0; k < BIRDS; k++) {
      var d = birdData[k], bt = clamp((flock - d.delay) / 0.65, 0, 1);
      if (bt <= 0) { m4.makeScale(0, 0, 0); birds.setMatrixAt(k, m4); continue; }
      flight(bt, d, bp);
      bp.x += Math.sin(time * 0.5 + birdPhase[k]) * 1.2 * bt;
      bp.y += Math.sin(time * 0.7 + birdPhase[k] * 2) * 0.6 * bt;
      flight(Math.min(bt + 0.02, 1.02), d, bv);
      dummy.position.copy(bp);
      dummy.lookAt(bv.x + (bt >= 1 ? 1.1 : 0), bv.y, bv.z - (bt >= 1 ? 3 : 0));
      dummy.scale.setScalar(2.4 * smooth(0, 0.04, bt) * (1 - smooth(0.9, 1, bt)));
      dummy.updateMatrix();
      birds.setMatrixAt(k, dummy.matrix);
    }
    birds.instanceMatrix.needsUpdate = true;
    birds.visible = flock > 0 && flock < 1;
    birdMat.color.set('#c8c4c8').lerp(tmp.set('#fff0dc'), dawn);

    GU.uClock.value = time;
    GU.uWind.value = wind;
    pf.dt = dt; pf.time = time; pf.snow = motesAmt; pf.wind = wind * 0.4;
    motes.update(pf, camera.position, env.reduceMotion);

    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) {
      H = h;
      portrait = w < h;
      fitCamera(gl, camera, w, h, dpr, small);
    },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('without-fear', {
  renderer: renderer3d,
  maxLines: 2,
  align: ['right', 'left', 'right', 'left'],
  scrim: 0.6,
  keys: function (T) {
    var n = T.count;
    function at(i, frac) { i = Math.min(i, n - 1); return lerp(T.start(i), T.end(i), frac); }
    //  unit          path   -  motes wind  lift  yaw    pitch  dawn  sun   walls birds rings rays
    return [
      [0,             0.000, 0, 0.0, 0.15, 0.0, -0.10, -0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
      [0.7,           0.002, 0, 0.0, 0.15, 0.0, -0.10, -0.19, 0.02, 0.00, 0.00, 0.00, 0.00, 0.0],
      [at(0, 0.2),    0.006, 0, 0.0, 0.15, 0.0, -0.14, -0.06, 0.06, 0.00, 0.00, 0.00, 0.00, 0.0],
      [at(0, 0.45),   0.011, 0, 0.0, 0.18, 0.0, -0.18, 0.12, 0.10, 0.00, 0.00, 0.00, 0.00, 0.0],   // "the head is held high"
      [at(0, 0.62),   0.014, 0, 0.0, 0.20, 0.0, -0.22, 0.14, 0.12, 0.00, 0.00, 0.04, 0.00, 0.0],   // "knowledge is free"
      [at(0, 1.0),    0.020, 0, 0.0, 0.22, 0.4, -0.08, 0.10, 0.16, 0.00, 0.00, 0.62, 0.00, 0.0],
      [at(1, 0.15),   0.026, 0, 0.0, 0.22, 2.0, 0.00, -0.12, 0.25, 0.00, 0.00, 1.00, 0.00, 0.0],
      [at(1, 0.3),    0.032, 0, 0.0, 0.22, 2.2, 0.02, -0.13, 0.28, 0.00, 0.06, 1.00, 0.00, 0.0],   // "narrow domestic walls"
      [at(1, 0.8),    0.070, 0, 0.0, 0.25, 2.4, 0.02, -0.10, 0.34, 0.00, 1.00, 1.00, 0.00, 0.0],
      [at(1, 1.0),    0.110, 0, 0.0, 0.25, 2.4, 0.00, -0.08, 0.37, 0.00, 1.25, 1.00, 0.00, 0.0],
      [at(2, 0.25),   0.190, 0, 0.0, 0.25, 2.6, 0.00, -0.10, 0.40, 0.00, 1.25, 1.00, 0.00, 0.5],   // "stretches its arms"
      [at(2, 0.45),   0.290, 0, 0.0, 0.25, 2.6, 0.00, -0.12, 0.46, 0.00, 1.25, 1.00, 0.00, 1.0],
      [at(2, 0.75),   0.460, 0, 0.0, 0.25, 2.8, 0.00, -0.13, 0.52, 0.00, 1.25, 1.00, 0.00, 0.9],   // "the clear stream of reason"
      [at(2, 1.0),    0.600, 0, 0.1, 0.25, 3.5, 0.00, -0.12, 0.58, 0.05, 1.25, 1.00, 0.00, 0.8],
      [at(3, 0.15),   0.660, 0, 0.2, 0.22, 7.0, 0.00, -0.12, 0.66, 0.12, 1.25, 1.00, 0.04, 0.7],
      [at(3, 0.5),    0.740, 0, 0.5, 0.20, 16.0, 0.00, -0.15, 0.80, 0.30, 1.25, 1.00, 0.45, 0.5],  // "ever-widening thought and action"
      [at(3, 0.85),   0.810, 0, 0.8, 0.18, 23.0, 0.00, -0.08, 0.94, 0.62, 1.25, 1.00, 0.80, 0.5],  // "let my country awake"
      [T.total,       0.880, 0, 0.5, 0.15, 27.0, 0.00, -0.04, 1.00, 1.00, 1.25, 1.00, 1.15, 0.6]
    ];
  },
  sound: {
    src: '/audio/birds.mp3',
    label: 'Play the dawn birdsong',
    // The dawn chorus swells as the light comes.
    volume: function (row) { return 0.05 + row[7] * 0.17; },
    cues: [
      { stanza: 0, at: 0.75, play: koel },
      { stanza: 1, at: 0.5, play: crumble },
      { stanza: 3, at: 0.25, play: tanpura }
    ]
  }
});
