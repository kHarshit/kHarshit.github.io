/*
 * Scene for "If Only She Knew" (Kiara Wilson): a terrible grey day in a
 * little town, turned around. Seen first-person; "she" is never shown,
 * only felt in the light and in what changes.
 *
 * Title  Heavy rain on a narrow street of terraced houses: the lamps lit in
 *        the gloom, the wet road holding their light, puddles ringing.
 * I      "Turning around my terrible day": walking down the wet sidewalk in
 *        the downpour. "The second that she says hey" the rain stops at
 *        once, the clouds break and sunlight floods the facades; the last
 *        drops hang glittering in the air and a rainbow rises over the
 *        roofs.
 * II     "When I look into her eyes": two upper windows ahead glow warm, side
 *        by side, and as the haze lifts the town shows its true colours.
 *        Looking down, the two lights shine in a still puddle, and the
 *        heartbeat rises: warm rings of light pulse out through the street,
 *        quickening.
 * III    "How much my love for her grew": a sapling cracks up through the
 *        asphalt in the middle of the road and grows into a cherry tree in
 *        full blossom; flowers spread along the kerbs, the doorsteps and the
 *        window boxes, and the blue cast of the rainy town warms away.
 * IV     "If beauty were inches, she'd go on for miles": at the end of the
 *        street the road turns west and runs for miles into a golden
 *        evening, its centre line a yellow tape measure and mile posts
 *        counting away. "Let my heart be my dial": the hands of the old
 *        street clock on the verge spin and settle together on the heart at
 *        twelve, which glows. The outro glides on down the road into the
 *        low sun.
 *
 * The wet road and sidewalks mirror the world: each frame it is drawn once
 * more from a camera mirrored in the ground into a target the ground
 * samples, rippled in the puddles while it rains. Colours are graded per
 * material (grey-blue in the rain, warm by the end); lamps and lit windows
 * are left ungraded so they stay warm in the grey.
 *
 * Columns: [unit, walk, dark, rain, wind, yaw, pitch, lift, light, glitter,
 *           bow, eyes, beat, grow, spread, dial]
 *   walk path progress; dark the gloom (lamps and lit windows); rain the
 *   downpour; light the mood in LIGHT (0 rain, 1 sunburst, 2 clear, 3 warm,
 *   4 golden evening); glitter the hanging drops; bow the rainbow; eyes the
 *   two warm windows; beat the heartbeat (0..1 strength, 1..2 quickening);
 *   grow the tree; spread the flowers; dial the clock hands settling.
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, broadleafGeometry, softSprite,
         rainField, scatter, followPath, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout (metres). The street runs north towards -z between terraces
// whose fronts stand at x = ±FACE; at its end it meets a road that runs
// west (-x) along z = ROAD_Z into open country. ─────────────────────────
var KERB = 3.5, FACE = 7.2, PAVE = 0.14, EYE = 1.65;
var GROUND = 3.8, STOREY = 3.1, PARAPET = 0.9;
var TOWN_N = 64, TOWN_S = -94, STREET_END = -98.5, ROAD_Z = -102, ROAD_W = 3.5;
var TREE = { x: 0, z: -66 };
var CLOCK = { x: -31, z: -107.6, y: 4.1, R: 1.05 };
var EYES_Z = -20;                       // the two warm windows, on the left
var EYES_CAM = { x: 3.0, z: 1 };       // where you stand to see them in the puddle

var PATH = [[5.0, 46], [5.0, 22], [4.6, 11], [3.0, 1], [2.4, -9], [2.0, -20], [1.7, -32], [2.3, -48],
            [3.6, -64], [3.3, -80], [1.4, -92], [-3.4, -100.6], [-12, -103.4], [-30, -103.5], [-110, -103.5],
            [-290, -103.5]];
var curve = new THREE.CatmullRomCurve3(PATH.map(function (p) { return new THREE.Vector3(p[0], 0, p[1]); }), false, 'centripetal');
var AHEAD = 10 / curve.getLength();

// Path progress nearest a point on the ground.
var SAMPLES = (function () {
  var out = [], p = new THREE.Vector3();
  for (var i = 0; i <= 1200; i++) { curve.getPointAt(i / 1200, p); out.push(p.x, p.z); }
  return out;
})();
function tNear(x, z) {
  var best = 1e18, bi = 0;
  for (var i = 0; i <= 1200; i++) {
    var dx = SAMPLES[i * 2] - x, dz = SAMPLES[i * 2 + 1] - z, d = dx * dx + dz * dz;
    if (d < best) { best = d; bi = i; }
  }
  return bi / 1200;
}
function nearPath(x, z, d) {
  for (var i = 0; i <= 1200; i++) if (Math.hypot(SAMPLES[i * 2] - x, SAMPLES[i * 2 + 1] - z) < d) return true;
  return false;
}

// Standing height: the sidewalks are a kerb above the road.
function baseY(x, z) {
  return z > TOWN_S - 2 ? PAVE * smooth(KERB - 0.1, KERB + 0.25, Math.abs(x)) : 0;
}

// ── The terraces, decided once so the keys know where the windows are ──
var WALLS = ['#e8dcc4', '#d8b48a', '#c47a5a', '#9fb09a', '#a8bccc', '#e4c6c0', '#f0ece2', '#c9a86a', '#b7a3b8', '#d9cfa8'];
var AWNINGS = ['#b0342e', '#2c6a46', '#28406e', '#c09030', '#8a3a5a'];
var ROOFS = ['#4a4e58', '#3e424a', '#8a4c38', '#56504c'];

function planTown() {
  var r = rng(19), out = [];
  [-1, 1].forEach(function (side) {
    var z = TOWN_N, k = 0;
    while (z > TOWN_S + 1) {
      var bays = 2 + Math.floor(r() * 3), bayW = 2.3 + r() * 0.6, W = bays * bayW, eyes = false;
      var floors = side < 0 ? 2 + (r() < 0.55 ? 1 : 0) : 3 + (r() < 0.35 ? 1 : 0);
      if (side < 0 && z - W < EYES_Z + 2.6 && z > EYES_Z + 2.6) {
        // Fill up to the building with the two windows, then place it.
        W = z - (EYES_Z + 2.6);
        if (W < 2) { out[out.length - 1].z0 -= W; out[out.length - 1].W += W; out[out.length - 1].bayW = out[out.length - 1].W / out[out.length - 1].bays; z = EYES_Z + 2.6; W = 0; }
        else { bays = Math.max(1, Math.round(W / 2.5)); bayW = W / bays; }
      } else if (side < 0 && Math.abs(z - (EYES_Z + 2.6)) < 0.001) {
        bays = 2; bayW = 2.6; W = 5.2; floors = 3; eyes = true;
      }
      if (W <= 0) continue;
      if (z - W < TOWN_S) { W = z - TOWN_S; bays = Math.max(1, Math.round(W / 2.5)); bayW = W / bays; }
      out.push({
        side: side, z0: z - W, z1: z, W: W, bays: bays, bayW: bayW, floors: floors,
        H: GROUND + floors * STOREY + PARAPET, D: 9 + r() * 2, seed: Math.floor(r() * 97),
        wall: WALLS[(k * 3 + (side > 0 ? 1 : 6) + Math.floor(r() * 3)) % WALLS.length],
        roof: ROOFS[Math.floor(r() * ROOFS.length)], pitched: r() < 0.6,
        awning: !eyes && r() < 0.5 ? AWNINGS[Math.floor(r() * AWNINGS.length)] : null, eyes: eyes
      });
      z -= W;
      k++;
    }
  });
  return out;
}
var TOWN = planTown();
var EYES = (function () {
  var b = TOWN.filter(function (t) { return t.eyes; })[0];
  var y = GROUND + 2 * STOREY + 0.5 * STOREY;
  return [0, 1].map(function (i) { return new THREE.Vector3(-FACE + 0.04, y, b.z0 + (i + 0.5) * b.bayW); });
})();
// The puddle that holds them: where the line from your eye to their mirror
// image below the ground meets the road.
var PUDDLE = (function () {
  var e = EYE + baseY(EYES_CAM.x, EYES_CAM.z), mx = (EYES[0].x + EYES[1].x) / 2, my = EYES[0].y, mz = (EYES[0].z + EYES[1].z) / 2;
  var t = e / (e + my);
  return new THREE.Vector3(EYES_CAM.x + (mx - EYES_CAM.x) * t, 0, EYES_CAM.z + (mz - EYES_CAM.z) * t);
})();

// ── Light through the day, in the order the `light` column visits it ─────
var LC = ['top', 'hor', 'cloud', 'shade', 'sun', 'fog', 'sunC', 'sky', 'ground'];
var LN = ['fogD', 'sunI', 'hemiI', 'cover', 'rift', 'sat', 'exp', 'disc', 'wet'];
var SUN_DAY = [-0.3, 0.45, 0.84], SUN_EVE = [-1, 0.045, 0.02];
var LIGHT = [
  // the downpour: low dark cloud, grey-blue, the lamps on
  { top: '#4f5862', hor: '#7c858e', cloud: '#6c747c', shade: '#3e444c', sun: '#000000', fog: '#6f7881', fogD: 0.026,
    sunC: '#b8c4d0', sunI: 0.2, sky: '#9aa8ba', ground: '#2e3238', hemiI: 1.5, dir: SUN_DAY, cover: 1.0, rift: 0.0,
    sat: 0.32, tint: [0.84, 0.92, 1.08], exp: 0.95, disc: 0, wet: 1 },
  // the turn: sun flooding in under breaking cloud
  { top: '#4c7cb4', hor: '#d6d0c0', cloud: '#fff0d8', shade: '#5c6272', sun: '#ffc890', fog: '#b4b4ac', fogD: 0.0105,
    sunC: '#ffd29a', sunI: 4.2, sky: '#a8b8d4', ground: '#4a4440', hemiI: 0.85, dir: SUN_DAY, cover: 0.8, rift: 1.0,
    sat: 0.75, tint: [0.98, 0.98, 1.0], exp: 1.12, disc: 0, wet: 1 },
  // clear: the true colours
  { top: '#3c74bc', hor: '#cddcea', cloud: '#ffffff', shade: '#94a0b4', sun: '#ffe2b8', fog: '#c2d0dc', fogD: 0.0046,
    sunC: '#fff0d8', sunI: 3.3, sky: '#c0d4ee', ground: '#5a524a', hemiI: 1.1, dir: SUN_DAY, cover: 0.5, rift: 0.6,
    sat: 1.0, tint: [1, 1, 1], exp: 1.0, disc: 0, wet: 0.9 },
  // warm: the blue washed out
  { top: '#4a7aba', hor: '#f0dcc2', cloud: '#fff4e2', shade: '#b4a8b0', sun: '#ffd8a8', fog: '#e2d4c0', fogD: 0.0042,
    sunC: '#ffe0b0', sunI: 3.1, sky: '#f0dcc8', ground: '#665644', hemiI: 1.15, dir: SUN_DAY, cover: 0.36, rift: 0.5,
    sat: 1.08, tint: [1.05, 1.0, 0.93], exp: 1.0, disc: 0, wet: 0.75 },
  // golden evening down the long road
  { top: '#3a5a96', hor: '#ffb46a', cloud: '#ffc890', shade: '#8a6676', sun: '#ff9a40', fog: '#e9b07c', fogD: 0.0017,
    sunC: '#ffaa5a', sunI: 2.7, sky: '#ffd2a0', ground: '#5a3e2c', hemiI: 0.95, dir: SUN_EVE, cover: 0.3, rift: 0.3,
    sat: 1.12, tint: [1.08, 0.98, 0.86], exp: 1.05, disc: 1, wet: 0.55 }
].map(function (L) {
  var o = { dir: new THREE.Vector3().fromArray(L.dir).normalize(), tint: new THREE.Vector3().fromArray(L.tint) };
  LC.forEach(function (k) { o[k] = new THREE.Color(L[k]); });
  LN.forEach(function (k) { o[k] = L[k]; });
  return o;
});

// ── Sound cues ───────────────────────────────────────────────────────────
function noiseBuffer(ac, sec) {
  var b = ac.createBuffer(1, Math.floor(ac.sampleRate * sec), ac.sampleRate), d = b.getChannelData(0);
  for (var i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  return b;
}

// "Hey": a warm major chord blooming as the sun breaks, with a few bright
// glints like light on drops.
function sunburst(ac, out) {
  var now = ac.currentTime + 0.02;
  [220, 277.2, 329.6, 440, 554.4].forEach(function (f, i) {
    var o = ac.createOscillator(), g = ac.createGain();
    o.type = i < 2 ? 'triangle' : 'sine';
    o.frequency.value = f;
    g.gain.setValueAtTime(0.0001, now);
    g.gain.exponentialRampToValueAtTime(0.035 / (1 + i * 0.3), now + 0.9);
    g.gain.exponentialRampToValueAtTime(0.0001, now + 5.5);
    o.connect(g); g.connect(out);
    o.start(now); o.stop(now + 5.6);
  });
  for (var k = 0; k < 14; k++) {
    var t = now + 0.3 + k * 0.17 + Math.random() * 0.12, o = ac.createOscillator(), g = ac.createGain();
    o.type = 'sine';
    o.frequency.value = [1760, 2217, 2637, 3520, 4435][Math.floor(Math.random() * 5)];
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.018, t + 0.005);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.9);
    o.connect(g); g.connect(out);
    o.start(t); o.stop(t + 1);
  }
}

// A heartbeat that rises: soft low lub-dubs, quickening from about 62 to 100
// beats a minute.
function heartbeat(ac, out) {
  var lp = ac.createBiquadFilter(), t = ac.currentTime + 0.1;
  lp.type = 'lowpass';
  lp.frequency.value = 220;
  lp.connect(out);
  function thump(at, f, gain) {
    var o = ac.createOscillator(), g = ac.createGain();
    o.type = 'sine';
    o.frequency.setValueAtTime(f, at);
    o.frequency.exponentialRampToValueAtTime(f * 0.6, at + 0.16);
    g.gain.setValueAtTime(0.0001, at);
    g.gain.exponentialRampToValueAtTime(gain, at + 0.012);
    g.gain.exponentialRampToValueAtTime(0.0001, at + 0.22);
    o.connect(g); g.connect(lp);
    o.start(at); o.stop(at + 0.25);
  }
  for (var i = 0; i < 16; i++) {
    var bpm = 62 + 38 * Math.min(i / 11, 1), swell = Math.min(1, 0.45 + i * 0.08) * (i > 13 ? 0.6 : 1);
    thump(t, 62, 0.55 * swell);
    thump(t + 0.24 * 60 / bpm + 0.05, 78, 0.32 * swell);
    t += 60 / bpm;
  }
}

// Tuning an old radio: hiss and a whistle sweeping across the band, a
// crackle, then a warm tone locking in.
function radioTune(ac, out) {
  var now = ac.currentTime + 0.02, src = ac.createBufferSource(), bp = ac.createBiquadFilter(), ng = ac.createGain();
  src.buffer = noiseBuffer(ac, 2.4);
  bp.type = 'bandpass';
  bp.Q.value = 6;
  bp.frequency.setValueAtTime(500, now);
  bp.frequency.exponentialRampToValueAtTime(3200, now + 0.8);
  bp.frequency.exponentialRampToValueAtTime(900, now + 1.7);
  ng.gain.setValueAtTime(0.0001, now);
  ng.gain.exponentialRampToValueAtTime(0.16, now + 0.15);
  ng.gain.setValueAtTime(0.16, now + 1.3);
  ng.gain.exponentialRampToValueAtTime(0.0001, now + 2.2);
  src.connect(bp); bp.connect(ng); ng.connect(out);
  src.start(now); src.stop(now + 2.4);
  var w = ac.createOscillator(), wg = ac.createGain();
  w.type = 'sine';
  w.frequency.setValueAtTime(700, now);
  w.frequency.exponentialRampToValueAtTime(2600, now + 0.7);
  w.frequency.exponentialRampToValueAtTime(1100, now + 1.3);
  w.frequency.exponentialRampToValueAtTime(523.3, now + 1.75);
  wg.gain.setValueAtTime(0.0001, now);
  wg.gain.exponentialRampToValueAtTime(0.025, now + 0.2);
  wg.gain.setValueAtTime(0.025, now + 1.5);
  wg.gain.exponentialRampToValueAtTime(0.0001, now + 1.9);
  w.connect(wg); wg.connect(out);
  w.start(now); w.stop(now + 2);
  // Locked on: a warm third that swells and fades.
  [523.3, 659.3, 784].forEach(function (f, i) {
    var o = ac.createOscillator(), g = ac.createGain(), at = now + 1.7;
    o.type = 'sine';
    o.frequency.value = f;
    g.gain.setValueAtTime(0.0001, at);
    g.gain.exponentialRampToValueAtTime(0.04 / (1 + i * 0.5), at + 0.35);
    g.gain.exponentialRampToValueAtTime(0.0001, at + 4);
    o.connect(g); g.connect(out);
    o.start(at); o.stop(at + 4.1);
  });
}

// ── Shared shader pieces ─────────────────────────────────────────────────
var NOISE = [
  'float rhash(vec3 p){ return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }',
  'float rnoise(vec3 p){ vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);',
  ' return mix(mix(mix(rhash(i), rhash(i + vec3(1.0, 0.0, 0.0)), f.x), mix(rhash(i + vec3(0.0, 1.0, 0.0)), rhash(i + vec3(1.0, 1.0, 0.0)), f.x), f.y),',
  '            mix(mix(rhash(i + vec3(0.0, 0.0, 1.0)), rhash(i + vec3(1.0, 0.0, 1.0)), f.x), mix(rhash(i + vec3(0.0, 1.0, 1.0)), rhash(i + vec3(1.0, 1.0, 1.0)), f.x), f.y), f.z); }',
  'float band(float v, float a, float b, float w){ return smoothstep(a - w, a + w, v) - smoothstep(b - w, b + w, v); }',
  'vec3 hueC(float h){ return clamp(vec3(abs(h * 6.0 - 3.0) - 1.0, 2.0 - abs(h * 6.0 - 2.0), 2.0 - abs(h * 6.0 - 4.0)), 0.0, 1.0); }',
  'vec3 gradeC(vec3 c){ float l = dot(c, vec3(0.299, 0.587, 0.114)); return max(mix(vec3(l), c, uSat), 0.0) * uTint; }',
  // Warm rings of light pulsing out from a point with each heartbeat.
  'float pulseAt(vec3 p){ float d = length(p - uPulseO), s = 0.0;',
  ' for (int i = 0; i < 4; i++) { float a = uPulse[i]; float x = (d - a * 8.0) / 0.75; s += exp(-x * x) * exp(-a * 1.1) * exp(-d * 0.03); }',
  ' return s * uBeat; }'
].join('\n') + '\n';
var HEAD = 'uniform float uSat; uniform vec3 uTint; uniform float uTime; uniform float uWet; uniform vec3 uPulseO; uniform float uPulse[4]; uniform float uBeat;\n' +
           'varying vec3 vWP; varying vec3 vWN;\n';
var WORLD_V = '\n vec4 rwp = vec4(transformed, 1.0);\n#ifdef USE_INSTANCING\n rwp = instanceMatrix * rwp;\n#endif\n vWP = (modelMatrix * rwp).xyz; vWN = normalize(mat3(modelMatrix) * objectNormal);\n';

// Patch a built-in material: world position and normal for the fragment,
// the shared uniforms, the grade on the albedo, and a material's own code
// (`color` runs after the albedo is known, `emissive` adds glow, `begin`
// moves vertices).
function patch(mat, key, G, o) {
  mat.onBeforeCompile = function (sh) {
    Object.keys(G).forEach(function (k) { sh.uniforms[k] = G[k]; });
    if (o.uniforms) Object.keys(o.uniforms).forEach(function (k) { sh.uniforms[k] = o.uniforms[k]; });
    sh.vertexShader = HEAD + (o.vHead || '') + sh.vertexShader
      .replace('#include <begin_vertex>', '#include <begin_vertex>\n' + (o.begin || ''))
      .replace('#include <project_vertex>', '#include <project_vertex>' + WORLD_V + (o.vEnd || ''));
    if (o.color_vertex) sh.vertexShader = sh.vertexShader.replace('#include <color_vertex>', o.color_vertex);
    sh.fragmentShader = HEAD + NOISE + (o.fHead || '') + sh.fragmentShader
      .replace('#include <color_fragment>', '#include <color_fragment>\n vec3 rEm = vec3(0.0); float rRough = -1.0;\n' + (o.color || '') +
               '\n diffuseColor.rgb = gradeC(diffuseColor.rgb);')
      .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\n if (rRough >= 0.0) roughnessFactor = rRough;')
      .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\n' + (o.emissive || '') + '\n totalEmissiveRadiance += rEm;');
  };
  mat.customProgramCacheKey = function () { return 'rainy-' + key; };
  return mat;
}

// ── Geometry helpers ─────────────────────────────────────────────────────
var UP = new THREE.Vector3(0, 1, 0);
function box(w, h, d, x, y, z, col) { return tinted(new THREE.BoxGeometry(w, h, d).translate(x, y, z), col); }

// Concatenate geometries with any of the named attributes (missing ones
// are zero-filled), so facades can carry their window plan.
function mergeWith(geos, attrs) {
  var total = 0;
  geos.forEach(function (g) { total += g.attributes.position.count; });
  var out = new THREE.BufferGeometry();
  attrs.forEach(function (a) {
    var arr = new Float32Array(total * a[1]), o = 0;
    geos.forEach(function (g) {
      var src = g.attributes[a[0]], n = g.attributes.position.count;
      if (src) arr.set(src.array, o);
      o += n * a[1];
    });
    out.setAttribute(a[0], new THREE.BufferAttribute(arr, a[1]));
  });
  geos.forEach(function (g) { g.dispose(); });
  return out;
}
function withAttr(geo, name, vals) {
  var n = geo.attributes.position.count, a = new Float32Array(n * vals.length);
  for (var i = 0; i < n; i++) for (var k = 0; k < vals.length; k++) a[i * vals.length + k] = vals[k];
  geo.setAttribute(name, new THREE.BufferAttribute(a, vals.length));
  return geo;
}

// A tapered cylinder from a to b (for the tree's limbs).
function limb(a, b, r0, r1, col) {
  var d = new THREE.Vector3().subVectors(b, a), len = d.length();
  var g = new THREE.CylinderGeometry(r1, r0, len, 7).translate(0, len / 2, 0);
  g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(UP, d.normalize()));
  return tinted(g.translate(a.x, a.y, a.z), col);
}
// A lumpy clump of blossom, pushed in and out by a smooth function.
function clump(rad, x, y, z, sy, seed, col) {
  var g = new THREE.IcosahedronGeometry(rad, 2), p = g.attributes.position;
  for (var i = 0; i < p.count; i++) {
    var vx = p.getX(i), vy = p.getY(i), vz = p.getZ(i);
    var k = 1 + 0.24 * Math.sin(vx * 3.1 + seed) * Math.cos(vz * 2.7 + seed * 1.3) + 0.12 * Math.sin(vy * 4.3 + seed * 0.7) +
            0.07 * Math.sin(vx * 9.0 + vy * 7.0 + seed) * Math.cos(vz * 8.0 - seed);
    p.setXYZ(i, vx * k, vy * k * sy, vz * k);
  }
  return tinted(g.translate(x, y, z), col);
}

// A cherry tree about 8 m tall and 10 m across: a short trunk forking into
// five spreading limbs under a broad crown of blossom.
function cherryGeometry(r) {
  var wood = [], bloom = [], pts = [], bark = '#4a3430', fork = new THREE.Vector3(0.1, 2.0, 0);
  wood.push(limb(new THREE.Vector3(0, -0.2, 0), fork, 0.34, 0.26, bark));
  var cols = ['#ffb0d4', '#f8a0c8', '#ffc0dc', '#f090c0', '#ffcce2'];
  // Six limbs, each forking into two or three branches that end in twigs;
  // small clusters of blossom sit along the twigs, loose enough for the
  // branches to show between them.
  var n = 0;
  for (var k = 0; k < 6; k++) {
    var a = k / 6 * Math.PI * 2 + r() * 0.5, out = 1.5 + r() * 0.7;
    var mid = new THREE.Vector3(Math.cos(a) * out, 3.4 + r() * 0.7, Math.sin(a) * out);
    wood.push(limb(fork, mid, 0.17, 0.11, bark));
    var nb = 2 + (r() < 0.5 ? 1 : 0);
    for (var j = 0; j < nb; j++) {
      var a2 = a + (j - (nb - 1) / 2) * 0.55 + (r() - 0.5) * 0.2, o2 = out + 1.4 + r() * 0.9;
      var br2 = new THREE.Vector3(Math.cos(a2) * o2, mid.y + 1.0 + r() * 0.8, Math.sin(a2) * o2);
      wood.push(limb(mid, br2, 0.1, 0.06, bark));
      for (var t = 0; t < 2; t++) {
        var a3 = a2 + (t - 0.5) * 0.7, o3 = o2 + 0.8 + r() * 0.7;
        var tw = new THREE.Vector3(Math.cos(a3) * o3, br2.y + 0.5 + r() * 0.7 - (o3 > 4.6 ? 0.6 : 0), Math.sin(a3) * o3);
        wood.push(limb(br2, tw, 0.05, 0.02, bark));
        for (var q = 0; q < 3; q++) {
          var f = 0.35 + q * 0.32;
          bloom.push(clump(0.55 + r() * 0.35, lerp(br2.x, tw.x, f) + (r() - 0.5) * 0.5, lerp(br2.y, tw.y, f) + 0.15 + r() * 0.3,
                           lerp(br2.z, tw.z, f) + (r() - 0.5) * 0.5, 0.78, n * 1.3, cols[n % cols.length]));
          n++;
        }
      }
    }
  }
  for (k = 0; k < 8; k++) {
    var b = r() * Math.PI * 2, bb = 0.6 + r() * 2.2;
    bloom.push(clump(0.5 + r() * 0.3, Math.cos(b) * bb, 5.6 + r() * 1.2, Math.sin(b) * bb, 0.75, 90 + k, cols[k % cols.length]));
  }
  // Loose blossom round the crown's edge for the points.
  for (k = 0; k < 2600; k++) {
    var c = r() * Math.PI * 2, cr = 1.5 + Math.sqrt(r()) * 4.4, cy = 4.4 + r() * 3.0 - (cr - 1.5) * 0.15;
    pts.push(Math.cos(c) * cr, cy, Math.sin(c) * cr);
  }
  return { wood: merge(wood), bloom: merge(bloom), pts: pts };
}

// A flower tuft: three stems with round heads. Heads are marked with a red
// vertex colour; the material gives them the instance colour.
function tuftGeometry() {
  var parts = [];
  for (var k = 0; k < 3; k++) {
    var a = k * 2.1 + 0.3, lean = 0.12 + k * 0.05, h = 0.22 + k * 0.07, x = Math.cos(a) * 0.05, z = Math.sin(a) * 0.05;
    parts.push(tinted(new THREE.CylinderGeometry(0.006, 0.008, h, 3).translate(0, h / 2, 0).rotateZ(lean).rotateY(a).translate(x, 0, z), '#000000'));
    var hx = x - Math.sin(lean) * h * Math.cos(a), hz = z + Math.sin(lean) * h * Math.sin(a);
    parts.push(tinted(new THREE.IcosahedronGeometry(0.036, 0).scale(1, 0.55, 1).translate(hx, h * Math.cos(lean), hz), '#ff0000'));
  }
  for (k = 0; k < 4; k++) {
    parts.push(tinted(new THREE.PlaneGeometry(0.035, 0.16).translate(0, 0.08, 0).rotateX(0.5).rotateY(k * 1.6 + 0.4), '#000000'));
  }
  return merge(parts);
}

// A street lamp: a post with a lantern; the lantern glass is separate.
function lampGeometry() {
  var c = '#22262a';
  return merge([
    tinted(new THREE.CylinderGeometry(0.16, 0.2, 0.5, 8).translate(0, 0.25, 0), c),
    tinted(new THREE.CylinderGeometry(0.06, 0.08, 3.6, 8).translate(0, 2.3, 0), c),
    tinted(new THREE.CylinderGeometry(0.11, 0.07, 0.25, 8).translate(0, 4.1, 0), c),
    tinted(new THREE.ConeGeometry(0.34, 0.3, 6).translate(0, 4.95, 0), c),
    tinted(new THREE.CylinderGeometry(0.04, 0.04, 0.2, 6).translate(0, 5.18, 0), c)
  ]);
}

// ── Painted textures ─────────────────────────────────────────────────────
function canvasTex(w, h, paint) {
  var c = document.createElement('canvas');
  c.width = w; c.height = h;
  paint(c.getContext('2d'), w, h);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

// The tape measure down the long road's centre: sixteen "inches" per tile,
// numbered so they read upright from behind them, counting away from you.
function tapeTex() {
  var t = canvasTex(1024, 64, function (x, w, h) {
    x.fillStyle = '#f2c62e';
    x.fillRect(0, 0, w, h);
    x.fillStyle = 'rgba(120,80,0,0.18)';
    x.fillRect(0, 0, w, 3); x.fillRect(0, h - 3, w, 3);
    x.fillStyle = '#1a1a1a';
    for (var i = 0; i < 128; i++) {
      var px = i * 8, len = i % 8 === 0 ? 30 : i % 4 === 0 ? 20 : i % 2 === 0 ? 13 : 8;
      x.fillRect(px, 0, i % 8 === 0 ? 3 : 2, len);
      x.fillRect(px, h - len, i % 8 === 0 ? 3 : 2, len);
    }
    x.font = 'bold 30px Georgia, serif';
    x.textAlign = 'center';
    x.textBaseline = 'middle';
    for (var n = 1; n <= 16; n++) {
      x.save();
      x.translate(n * 64 - 30, h / 2);
      x.rotate(Math.PI / 2);
      x.fillStyle = n % 12 === 0 ? '#b8231e' : '#1a1a1a';
      x.fillText(String(n), 0, 1);
      x.restore();
    }
  });
  t.wrapS = THREE.RepeatWrapping;
  return t;
}

// Mile plates, 1 to 12, in one strip.
function mileTex() {
  return canvasTex(1536, 128, function (x, w, h) {
    for (var n = 0; n < 12; n++) {
      var ox = n * 128;
      x.fillStyle = '#1f6a44';
      x.fillRect(ox, 0, 128, h);
      x.strokeStyle = '#f4f4ee';
      x.lineWidth = 5;
      x.strokeRect(ox + 7, 7, 114, h - 14);
      x.fillStyle = '#f4f4ee';
      x.textAlign = 'center';
      x.font = 'bold 58px Helvetica, Arial, sans-serif';
      x.fillText(String(n + 1), ox + 64, 70);
      x.font = 'bold 22px Helvetica, Arial, sans-serif';
      x.fillText(n ? 'MILES' : 'MILE', ox + 64, 104);
    }
  });
}

// The sign where the town ends: its name, struck through in red.
function townSignTex() {
  return canvasTex(512, 256, function (x, w, h) {
    x.fillStyle = '#f2c84a';
    x.fillRect(0, 0, w, h);
    x.strokeStyle = '#1c1c1c';
    x.lineWidth = 14;
    x.strokeRect(14, 14, w - 28, h - 28);
    x.fillStyle = '#1c1c1c';
    x.font = 'bold 92px Helvetica, Arial, sans-serif';
    x.textAlign = 'center';
    x.textBaseline = 'middle';
    x.fillText('RAINFORD', w / 2, h / 2 + 4);
    x.strokeStyle = '#c8261e';
    x.lineWidth = 22;
    x.beginPath(); x.moveTo(40, h - 40); x.lineTo(w - 40, 40); x.stroke();
  });
}

// The clock's face: cream enamel, Roman hours, and a heart at twelve.
function clockTex() {
  return canvasTex(512, 512, function (x, w) {
    var c = w / 2;
    var g = x.createRadialGradient(c, c, 0, c, c, c);
    g.addColorStop(0, '#fbf4e2'); g.addColorStop(0.85, '#f0e4c6'); g.addColorStop(1, '#c8b48a');
    x.fillStyle = g;
    x.beginPath(); x.arc(c, c, c, 0, Math.PI * 2); x.fill();
    x.strokeStyle = '#2a2622';
    x.lineWidth = 6;
    x.beginPath(); x.arc(c, c, c - 22, 0, Math.PI * 2); x.stroke();
    for (var m = 0; m < 60; m++) {
      var a = m / 60 * Math.PI * 2, r0 = c - 24, r1 = m % 5 ? c - 36 : c - 50;
      x.lineWidth = m % 5 ? 3 : 7;
      x.beginPath(); x.moveTo(c + Math.sin(a) * r0, c - Math.cos(a) * r0); x.lineTo(c + Math.sin(a) * r1, c - Math.cos(a) * r1); x.stroke();
    }
    var ROM = ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI'];
    x.fillStyle = '#2a2622';
    x.font = '600 50px Georgia, serif';
    x.textAlign = 'center';
    x.textBaseline = 'middle';
    for (var hNum = 1; hNum < 12; hNum++) {
      var b = hNum / 12 * Math.PI * 2, rr = c - 94;
      x.fillText(ROM[hNum], c + Math.sin(b) * rr, c - Math.cos(b) * rr);
    }
    // A faint heart drawn at twelve; the glowing one sits in front of it.
    heartPath(x, c, 94, 34);
    x.fillStyle = '#c0504a';
    x.fill();
  });
}
function heartPath(x, cx, cy, s) {
  x.beginPath();
  x.moveTo(cx, cy + s * 0.9);
  x.bezierCurveTo(cx - s * 1.3, cy + s * 0.1, cx - s * 0.9, cy - s * 0.9, cx, cy - s * 0.35);
  x.bezierCurveTo(cx + s * 0.9, cy - s * 0.9, cx + s * 1.3, cy + s * 0.1, cx, cy + s * 0.9);
  x.closePath();
}
function heartShape(s) {
  var h = new THREE.Shape();
  h.moveTo(0, -s * 0.9);
  h.bezierCurveTo(-s * 1.3, -s * 0.1, -s * 0.9, s * 0.9, 0, s * 0.35);
  h.bezierCurveTo(s * 0.9, s * 0.9, s * 1.3, -s * 0.1, 0, -s * 0.9);
  return h;
}

// A star glint: a soft core with long thin rays across it.
function glintTex() {
  return canvasTex(256, 256, function (x, w) {
    var c = w / 2, g = x.createRadialGradient(c, c, 0, c, c, c);
    g.addColorStop(0, 'rgba(255,240,210,1)'); g.addColorStop(0.12, 'rgba(255,200,130,0.7)');
    g.addColorStop(0.4, 'rgba(255,160,80,0.18)'); g.addColorStop(1, 'rgba(255,140,60,0)');
    x.fillStyle = g;
    x.fillRect(0, 0, w, w);
    [[1, 0.02], [0.55, 0.012]].forEach(function (ray, k) {
      for (var a = 0; a < 4; a++) {
        x.save();
        x.translate(c, c);
        x.rotate(a * Math.PI / 2 + k * Math.PI / 4);
        var lg = x.createLinearGradient(0, 0, c * ray[0], 0);
        lg.addColorStop(0, 'rgba(255,236,200,0.9)'); lg.addColorStop(1, 'rgba(255,200,140,0)');
        x.fillStyle = lg;
        x.beginPath(); x.moveTo(0, -w * ray[1]); x.lineTo(c * ray[0], 0); x.lineTo(0, w * ray[1]); x.fill();
        x.restore();
      }
    });
  });
}

// ── Sky: gradient, clouds that break open towards the sun, a rainbow
// opposite it and the sun's disc. Colours are as displayed. ─────────────
function skyDome(radius) {
  var u = {
    uTop: { value: new THREE.Color() }, uHor: { value: new THREE.Color() }, uCloud: { value: new THREE.Color() },
    uShade: { value: new THREE.Color() }, uSun: { value: new THREE.Color() }, uSunDir: { value: new THREE.Vector3(0, 1, 0) },
    uCover: { value: 1 }, uRift: { value: 0 }, uBow: { value: 0 }, uDisc: { value: 0 }, uTime: { value: 0 }
  };
  var mat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false, uniforms: u,
    vertexShader: 'varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: [
      'uniform vec3 uTop, uHor, uCloud, uShade, uSun, uSunDir; uniform float uCover, uRift, uBow, uDisc, uTime; varying vec3 vP;',
      'float hh(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }',
      'float nn(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);',
      ' return mix(mix(hh(i), hh(i + vec2(1.0, 0.0)), f.x), mix(hh(i + vec2(0.0, 1.0)), hh(i + vec2(1.0, 1.0)), f.x), f.y); }',
      'float fbm(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 5; i++) { s += a * nn(p); p = p * 2.07 + vec2(3.1, 1.7); a *= 0.5; } return s; }',
      'vec3 hue(float h){ return clamp(vec3(abs(h * 6.0 - 3.0) - 1.0, 2.0 - abs(h * 6.0 - 2.0), 2.0 - abs(h * 6.0 - 4.0)), 0.0, 1.0); }',
      'void main(){',
      ' vec3 d = normalize(vP); float h = max(d.y, 0.0); vec3 sd = normalize(uSunDir);',
      ' vec3 c = mix(uHor, uTop, smoothstep(0.0, 0.6, pow(h, 0.75)));',
      ' float s = max(dot(d, sd), 0.0);',
      ' c += uSun * (pow(s, 6.0) * 0.35 + pow(s, 60.0) * 0.6);',
      ' c = mix(c, vec3(1.0, 0.86, 0.62) * 1.5, smoothstep(0.99955, 0.9998, s) * uDisc);',
      ' vec2 p = d.xz / (h + 0.1) * 0.75 + vec2(uTime * 0.012, uTime * 0.004);',
      ' float f = fbm(p) * 0.62 + fbm(p * 2.3 + 7.0) * 0.38;',
      ' float cover = uCover * (1.0 - uRift * smoothstep(-0.35, 0.75, dot(d, sd)));',
      ' float cov = smoothstep(1.0 - cover, 1.0 - cover + 0.26, f) * smoothstep(-0.02, 0.15, h);',
      ' vec3 cl = mix(uCloud, uShade, smoothstep(0.38, 0.9, f));',
      ' cl += uSun * pow(s, 3.0) * 0.55 * (1.0 - smoothstep(0.35, 0.8, f));',
      ' c = mix(c, cl, cov);',
      // The rainbow: the primary bow 40-42 degrees from the antisolar
      // point, the sky a little brighter inside it, a faint second bow.
      ' float a = acos(clamp(dot(d, -sd), -1.0, 1.0));',
      ' float bt = (a - 0.698) / 0.044, bw = smoothstep(0.0, 0.3, bt) * smoothstep(1.0, 0.7, bt);',
      ' float bt2 = (0.94 - a) / 0.06, bw2 = smoothstep(0.0, 0.3, bt2) * smoothstep(1.0, 0.7, bt2);',
      ' float vis = uBow * smoothstep(-0.01, 0.12, d.y) * (0.6 + 0.4 * cov);',
      ' c += hue((1.0 - bt) * 0.78) * bw * vis * 0.42 + hue((1.0 - bt2) * 0.78) * bw2 * vis * 0.12;',
      ' c += vec3(0.06) * vis * smoothstep(0.7, 0.6, a);',
      ' c = mix(c, uHor, 1.0 - smoothstep(-0.03, 0.05, d.y));',
      ' gl_FragColor = vec4(c, 1.0);',
      ' #include <colorspace_fragment>',
      '}'
    ].join('\n')
  });
  return { mesh: new THREE.Mesh(new THREE.SphereGeometry(radius, 48, 24), mat), uniforms: u };
}

// ── Renderer ─────────────────────────────────────────────────────────────
function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(31), i, k;
  var gl = makeRenderer(canvas, { shadows: !small, clear: '#6f7881' });
  gl.shadowMap.autoUpdate = false;
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#6f7881', 0.026);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.1, 4000);

  // Uniforms every patched material shares.
  var G = {
    uSat: { value: 0.32 }, uTint: { value: new THREE.Vector3(1, 1, 1) }, uTime: { value: 0 }, uWet: { value: 1 },
    uPulseO: { value: PUDDLE.clone() }, uPulse: { value: [9, 9, 9, 9] }, uBeat: { value: 0 }
  };

  // ── Sky and light ──────────────────────────────────────────────────────
  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome(1800);
  dome.mesh.frustumCulled = false;
  sky.add(dome.mesh);

  var hemi = new THREE.HemisphereLight('#9aa8ba', '#2e3238', 1.5);
  var sun = new THREE.DirectionalLight('#ffffff', 0.2);
  world.add(hemi, sun, sun.target);
  if (!small) {
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    var sc = sun.shadow.camera;
    sc.left = -45; sc.right = 45; sc.top = 45; sc.bottom = -45; sc.near = 1; sc.far = 600;
    sun.shadow.bias = -0.0005;
    sun.shadow.normalBias = 0.05;
    sc.updateProjectionMatrix();
  }

  // ── The mirror: the world drawn again from below the ground ────────────
  var RT_SCALE = small ? 0.35 : 0.5;
  var reflRT = new THREE.WebGLRenderTarget(4, 4, { type: THREE.HalfFloatType });
  var reflCam = new THREE.PerspectiveCamera();
  var R = { tRefl: { value: reflRT.texture }, uTexMat: { value: new THREE.Matrix4() }, uRain: { value: 1 },
            uBig: { value: new THREE.Vector3(PUDDLE.x, PUDDLE.z, 1.6) }, uTree: { value: new THREE.Vector2(TREE.x, TREE.z) },
            uCrack: { value: 0 } };
  var BIAS = new THREE.Matrix4().set(0.5, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0, 1);

  // ── Ground: the street, its sidewalks and the long road west ──────────
  // `kind` 0 is asphalt, 1 paving. Puddles gather in the gutters; one sits
  // where the two windows will show.
  var groundMat = patch(new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.85, metalness: 0 }), 'ground', G, {
    uniforms: R,
    vHead: 'attribute float kind; varying float vKind; varying vec4 vRefl; uniform mat4 uTexMat;\n',
    vEnd: ' vKind = kind; vRefl = uTexMat * vec4(vWP, 1.0);\n',
    fHead: 'varying float vKind; varying vec4 vRefl; uniform sampler2D tRefl; uniform float uRain; uniform vec3 uBig; uniform vec2 uTree; uniform float uCrack;\n' +
      'vec2 ripples(vec2 p){ vec2 g = vec2(0.0), b = floor(p);\n' +
      ' for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {\n' +
      '  vec2 c = b + vec2(float(i), float(j)); float k = rhash(vec3(c, 3.0));\n' +
      '  if (k > uRain) continue;\n' +
      '  vec2 o = c + 0.25 + 0.5 * vec2(rhash(vec3(c, 1.7)), rhash(vec3(c, 4.3)));\n' +
      '  float t = fract(uTime * (0.9 + k * 0.8) + k * 7.0);\n' +
      '  vec2 d = p - o; float r = length(d) + 0.0001; float x = (r - t * 1.2) * 10.0;\n' +
      '  g += d / r * sin(x * 2.4) * exp(-x * x) * (1.0 - t) * (1.0 - t);\n' +
      ' }\n return g; }\n',
    color: [
      ' vec2 p = vWP.xz; float ax = abs(vWP.x);',
      ' float g1 = rnoise(vec3(p * 2.7, 0.0)), g2 = rnoise(vec3(p * 11.0, 1.0)), g3 = rnoise(vec3(p * 0.3, 2.0));',
      ' vec3 asph = vec3(0.17, 0.17, 0.18) * (0.78 + 0.3 * g1 + 0.2 * g2) * (0.9 + 0.2 * g3);',
      // paving slabs, a granite kerb along the edge
      ' vec2 sl = vec2((ax - KERB_) / 0.75, vWP.z / 0.9 + floor((ax - KERB_) / 0.75) * 0.5); vec2 fs = abs(fract(sl) - 0.5), fw = fwidth(sl);',
      ' float joint = 1.0 - smoothstep(0.0, 0.03 + fw.x, 0.5 - fs.x) * smoothstep(0.0, 0.03 + fw.y, 0.5 - fs.y);',
      ' vec3 pave = vec3(0.47, 0.45, 0.42) * (0.82 + 0.3 * rhash(vec3(floor(sl), 5.0)) + 0.12 * g2);',
      ' pave = mix(pave, vec3(0.2, 0.19, 0.18), joint * 0.8);',
      ' pave = mix(pave, vec3(0.52, 0.52, 0.53) * (0.85 + 0.2 * g2), step(ax, KERB_ + 0.32));',
      ' vec3 base = mix(asph, pave, vKind);',
      // a dashed white centre line in the town
      ' float dash = (1.0 - vKind) * step(STREET_END_, vWP.z) * band(vWP.x, -0.07, 0.07, fwidth(vWP.x)) * step(fract(vWP.z / 6.0), 0.5);',
      ' base = mix(base, vec3(0.72, 0.72, 0.68) * (0.8 + 0.2 * g2), dash);',
      // the tree cracking the road open, soil and grass round its foot
      ' vec2 tq = p - uTree; float td = length(tq), ta = atan(tq.y, tq.x);',
      ' float cr = abs(rnoise(vec3(cos(ta) * 2.4, sin(ta) * 2.4, td * 0.9)) - 0.5) + td * 0.004;',
      ' float crack = (1.0 - vKind) * smoothstep(0.035, 0.0, cr) * smoothstep(uCrack * 6.5, uCrack * 6.5 - 1.5, td) * step(0.01, uCrack);',
      ' float soil = smoothstep(uCrack * 1.5, uCrack * 1.5 - 0.4, td + (g1 - 0.5) * 0.5);',
      ' base = mix(base, vec3(0.05, 0.045, 0.04), crack);',
      ' base = mix(base, mix(vec3(0.2, 0.15, 0.1), vec3(0.22, 0.34, 0.12), smoothstep(0.4, 0.7, g2) * smoothstep(0.3, 1.0, uCrack)), soil);',
      // puddles: in the gutters, in low spots, and the one for the windows
      ' float pn = rnoise(vec3(p * 0.21, 3.0)) * 0.62 + rnoise(vec3(p * 0.63, 4.0)) * 0.38;',
      ' float gut = (1.0 - vKind) * step(STREET_END_, vWP.z) * smoothstep(KERB_ - 1.1, KERB_ - 0.15, ax);',
      ' float big = 1.0 - smoothstep(uBig.z * 0.55, uBig.z, length((p - uBig.xy) * vec2(1.0, 0.75)) + (g1 - 0.5) * 0.5);',
      ' float pud = max(smoothstep(0.66, 0.76, pn + gut * 0.2 - vKind * 0.08) * step(STREET_END_ + 1.0, vWP.z), big);',
      ' pud *= (1.0 - soil) * smoothstep(0.35, 0.6, uWet);',
      ' float wet = uWet * (0.65 + 0.35 * g3);',
      ' base *= mix(1.0, 0.6, wet) * mix(1.0, 0.45, pud);',
      ' diffuseColor.rgb = base;',
      ' rRough = mix(mix(0.92, 0.5, wet), 0.06, pud);'
    ].join('\n').replace(/KERB_/g, KERB.toFixed(2)).replace(/STREET_END_/g, STREET_END.toFixed(2)),
    emissive: [
      ' vec2 rg = ripples(p * 1.7);',
      ' vec4 ru = vRefl;',
      ' ru.xy += (rg * 0.035 * pud + vec2(g2 - 0.5, g1 - 0.5) * 0.02 * (1.0 - pud)) * ru.w;',
      ' vec3 refl = texture2DProj(tRefl, ru).rgb;',
      // wet asphalt smears what it holds into vertical streaks
      ' vec3 blur = (texture2DProj(tRefl, ru + vec4(0.0, 0.025, 0.0, 0.0) * ru.w).rgb + texture2DProj(tRefl, ru - vec4(0.0, 0.025, 0.0, 0.0) * ru.w).rgb +',
      '              texture2DProj(tRefl, ru + vec4(0.0, 0.055, 0.0, 0.0) * ru.w).rgb + refl) * 0.25;',
      ' float cosv = clamp(dot(normal, normalize(vViewPosition)), 0.0, 1.0), fres = pow(1.0 - cosv, 3.0);',
      ' float rk = mix(mix(0.04, 0.36, fres) * wet, mix(0.32, 0.78, fres), pud) * (1.0 - soil) * (0.5 + 0.5 * smoothstep(0.6, 0.9, uWet));',
      ' rEm += mix(blur, refl, pud) * rk;',
      ' rEm += vec3(1.0, 0.46, 0.2) * pulseAt(vWP) * (0.35 + 0.65 * pud) * 0.9;'
    ].join('\n')
  });

  function flat(w, d, x, y, z, kind) {
    var g = new THREE.PlaneGeometry(w, d).rotateX(-Math.PI / 2).translate(x, y, z);
    withAttr(g, 'kind', [kind]);
    return g;
  }
  var streetLen = TOWN_N + 40 - STREET_END;
  var groundParts = [
    flat(KERB * 2, streetLen, 0, 0, (TOWN_N + 40 + STREET_END) / 2, 0),
    flat(4400, ROAD_W * 2, -1900, 0, ROAD_Z, 0)
  ];
  [-1, 1].forEach(function (s) {
    var len = TOWN_N + 40 - (TOWN_S - 2), g = new THREE.BoxGeometry(FACE - KERB + 0.4, PAVE, len)
      .translate(s * ((KERB + FACE + 0.4) / 2), PAVE / 2, (TOWN_N + 40 + TOWN_S - 2) / 2);
    withAttr(g, 'kind', [1]);
    groundParts.push(g.toNonIndexed());
  });
  groundParts = groundParts.map(function (g) { return g.index ? g.toNonIndexed() : g; });
  var ground = new THREE.Mesh(mergeWith(groundParts, [['position', 3], ['normal', 3], ['kind', 1]]), groundMat);
  ground.receiveShadow = true;
  world.add(ground);

  // The yellow tape measure down the long road, and its hook at the start.
  var tape = tapeTex(), tapeLen = 3600, tapeGeo = new THREE.BufferGeometry(), tx0 = -6;
  tapeGeo.setAttribute('position', new THREE.Float32BufferAttribute([tx0, 0.012, ROAD_Z - 0.17, tx0 - tapeLen, 0.012, ROAD_Z - 0.17,
    tx0, 0.012, ROAD_Z + 0.17, tx0 - tapeLen, 0.012, ROAD_Z + 0.17], 3));
  tapeGeo.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, tapeLen / 8, 0, 0, 1, tapeLen / 8, 1], 2));
  tapeGeo.setAttribute('normal', new THREE.Float32BufferAttribute([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], 3));
  tapeGeo.setIndex([0, 1, 2, 2, 1, 3]);
  var tapeMat = patch(new THREE.MeshStandardMaterial({ map: tape, roughness: 0.4, polygonOffset: true, polygonOffsetFactor: -2 }), 'tape', G, {});
  var tapeMesh = new THREE.Mesh(tapeGeo, tapeMat);
  tapeMesh.receiveShadow = true;
  world.add(tapeMesh);

  // ── Fields beyond the town: a patchwork of crops and green verges ──────
  var fieldMat = patch(new THREE.MeshLambertMaterial({ color: '#ffffff' }), 'field', G, {
    color: [
      ' vec2 p = vWP.xz;',
      ' vec2 q = p / vec2(150.0, 110.0) + vec2(rnoise(vec3(p * 0.004, 1.0)), rnoise(vec3(p * 0.004, 2.0))) * 0.6;',
      ' float k = rhash(vec3(floor(q), 7.0));',
      ' vec3 crop = k < 0.3 ? vec3(0.58, 0.47, 0.2) : k < 0.55 ? vec3(0.28, 0.4, 0.15) : k < 0.75 ? vec3(0.5, 0.5, 0.26) : vec3(0.2, 0.31, 0.13);',
      ' float ang = k * 6.28; crop *= 0.9 + 0.1 * sin(dot(p, vec2(cos(ang), sin(ang))) * 2.2);',
      ' crop *= 0.82 + 0.3 * rnoise(vec3(p * 0.05, 3.0)) + 0.1 * rnoise(vec3(p * 0.6, 4.0));',
      ' vec2 e = abs(fract(q) - 0.5); crop = mix(crop, vec3(0.16, 0.24, 0.1), smoothstep(0.47, 0.495, max(e.x, e.y)) * 0.8);',
      ' float verge = 1.0 - smoothstep(4.5, 9.0, abs(vWP.z - ROADZ_));',
      ' crop = mix(crop, vec3(0.3, 0.42, 0.16) * (0.8 + 0.3 * rnoise(vec3(p * 0.8, 5.0))), verge);',
      ' diffuseColor.rgb = crop;'
    ].join('\n').replace(/ROADZ_/g, ROAD_Z.toFixed(1))
  });
  var fields = new THREE.Mesh(new THREE.PlaneGeometry(9000, 9000).rotateX(-Math.PI / 2).translate(-1500, -0.03, -400), fieldMat);
  fields.receiveShadow = true;
  world.add(fields);

  // ── The terraces ───────────────────────────────────────────────────────
  // Facades are plain boxes; the shader draws shopfronts, sash windows,
  // string courses and grime from each building's plan (`fac`: start of
  // its front, bay width, height, seed). Upper windows light in the gloom.
  var FU = { uLit: { value: 1 }, uShop: { value: 1 }, uGlass: { value: new THREE.Color('#b8c8d8') } };
  var facadeMat = patch(new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.92 }), 'facade', G, {
    uniforms: FU,
    vHead: 'attribute vec4 fac; varying vec4 vFac;\n',
    vEnd: ' vFac = fac;\n',
    fHead: 'varying vec4 vFac; uniform float uLit; uniform float uShop; uniform vec3 uGlass;\n',
    color: [
      ' vec3 wall = diffuseColor.rgb; float y = vWP.y;',
      ' float n1 = rnoise(vWP * 0.8), n2 = rnoise(vWP * 5.0);',
      ' wall *= 0.86 + 0.2 * n1 - 0.06 * n2;',
      ' wall *= mix(1.0, 0.8 - 0.16 * rnoise(vec3(vWP.z * 2.5, y * 0.2, vWP.x * 2.5)), clamp((uWet - 0.85) / 0.15, 0.0, 1.0));',
      ' wall *= mix(0.72, 1.0, smoothstep(0.0, 1.2, y));',
      ' float endW = step(vWN.z, -0.7) * step(vWP.z, TOWNS_ + 0.3);',
      ' float front = max(smoothstep(0.6, 0.85, abs(vWN.x)), endW);',
      ' float bu = mix(vWP.z - vFac.x, abs(vWP.x) - FACE_ + 0.4, endW) / vFac.y, fu = fract(bu), bi = floor(bu), fwu = max(fwidth(bu), 0.002);',
      ' float sv = (y - GROUND_) / STOREY_, fv = fract(sv), fl = floor(sv), fwv = max(fwidth(sv), 0.002), fwy = max(fwidth(y), 0.002);',
      ' float upper = step(GROUND_, y) * step(y, vFac.z - PARAPET_) * front;',
      ' float shop = step(y, GROUND_ - 0.3) * front;',
      ' vec3 col = wall;',
      // stone courses over the shops and under the parapet
      ' float course = band(y, GROUND_ - 0.3, GROUND_ + 0.05, fwy) + band(y, vFac.z - PARAPET_ + 0.05, vFac.z - PARAPET_ + 0.35, fwy);',
      ' col = mix(col, wall * 1.15 + 0.06, course * front);',
      // sash windows with white frames, glazing bars and a sill
      ' float win = band(fu, 0.3, 0.7, fwu) * band(fv, 0.2, 0.8, fwv) * upper;',
      ' float frame = band(fu, 0.27, 0.73, fwu) * band(fv, 0.17, 0.83, fwv) * upper;',
      ' float bars = max(band(fu, 0.49, 0.51, fwu), band(fv, 0.51, 0.535, fwv));',
      ' float sill = band(fu, 0.25, 0.75, fwu) * band(fv, 0.12, 0.17, fwv) * upper;',
      ' float wk = rhash(vec3(bi, fl, vFac.w));',
      ' vec3 glass = mix(vec3(0.03, 0.035, 0.045), uGlass * 0.32, 0.25 + 0.6 * fv) + vec3(0.03) * step(0.6, wk);',
      ' col = mix(col, vec3(0.86, 0.84, 0.8), max(frame - win, sill));',
      ' col = mix(col, mix(glass, vec3(0.86, 0.84, 0.8), bars), win);',
      // shopfronts: a painted fascia, a big lit pane or a door per bay
      ' float sk = rhash(vec3(vFac.w, 2.0, 1.0)), isDoor = step(0.7, rhash(vec3(bi, 3.0, vFac.w)));',
      ' vec3 paint = sk < 0.17 ? vec3(0.08, 0.17, 0.11) : sk < 0.34 ? vec3(0.27, 0.07, 0.07) : sk < 0.5 ? vec3(0.07, 0.11, 0.2) : sk < 0.67 ? vec3(0.05) : sk < 0.84 ? vec3(0.32, 0.22, 0.06) : vec3(0.08, 0.18, 0.2);',
      ' float shopF = band(fu, 0.04, 0.96, fwu) * band(y, 0.0, GROUND_ - 0.35, fwy) * shop;',
      ' float pane = mix(band(fu, 0.1, 0.9, fwu) * band(y, 0.55, GROUND_ - 1.3, fwy), band(fu, 0.3, 0.7, fwu) * band(y, 0.02, 2.35, fwy), isDoor) * shop;',
      ' float letters = band(y, GROUND_ - 1.0, GROUND_ - 0.65, fwy) * step(0.5, rnoise(vec3(vWP.z * 7.0, 0.0, vFac.w))) * band(fract((vWP.z - vFac.x) / (vFac.y * 4.0) ), 0.2, 0.8, 0.01) * shop;',
      ' col = mix(col, paint * (0.9 + 0.2 * n2), shopF);',
      ' col = mix(col, vec3(0.85, 0.75, 0.5), letters * 0.8);',
      // inside the shop: shelves of goods under warm ceiling light
      ' float shelf = fract(y * 2.6), cz = vWP.z * 9.0;',
      ' float item = step(0.5, rhash(vec3(floor(cz), floor(y * 2.6), vFac.w))) * smoothstep(0.1, 0.18, fract(cz)) * smoothstep(0.9, 0.82, fract(cz)) * step(0.18, shelf) * step(shelf, 0.3 + 0.6 * rhash(vec3(floor(cz), floor(y * 2.6), 3.0)));',
      ' vec3 goods = hueC(rhash(vec3(floor(cz), floor(y * 2.6), vFac.w + 1.0))) * 0.5 + 0.2;',
      ' item *= step(0.9, y) * step(y, GROUND_ - 1.5);',
      ' vec3 inside = mix(vec3(0.03, 0.026, 0.024), goods * 0.3, item);',
      ' inside = mix(inside, vec3(0.012), step(shelf, 0.06) * step(0.9, y));',
      ' col = mix(col, isDoor > 0.5 ? paint * 0.7 + vec3(0.04) : inside, pane);',
      ' diffuseColor.rgb = col;',
      ' float lit = step(0.62, wk) * uLit;',
      ' rEm += vec3(1.0, 0.68, 0.36) * win * (1.0 - bars) * lit * (0.55 + 0.5 * fv);',
      ' float ceil = smoothstep(GROUND_ - 2.4, GROUND_ - 1.35, y);',
      ' rEm += vec3(1.0, 0.72, 0.42) * pane * (1.0 - isDoor) * (0.04 + 0.32 * ceil * ceil + item * 0.12) * uShop;',
      ' rEm += uGlass * pane * (1.0 - isDoor) * 0.06 * smoothstep(0.3, 1.0, fract((vWP.z * 0.6 + y * 0.4)));',
      ' rRough = mix(0.92, 0.12, max(win * (1.0 - bars), pane * (1.0 - isDoor)));'
    ].join('\n').replace(/GROUND_/g, GROUND.toFixed(2)).replace(/STOREY_/g, STOREY.toFixed(2)).replace(/PARAPET_/g, PARAPET.toFixed(2))
      .replace(/TOWNS_/g, TOWN_S.toFixed(1)).replace(/FACE_/g, FACE.toFixed(2)),
    emissive: ' rEm += vec3(1.0, 0.5, 0.24) * pulseAt(vWP) * 0.35;'
  });

  var bodies = [], roofs = [], trim = [], awnings = [], sills = [];
  TOWN.forEach(function (b) {
    var s = b.side, x0 = s * FACE, xc = s * (FACE + b.D / 2), zc = (b.z0 + b.z1) / 2;
    var fac = [b.z0, b.bayW, b.H, b.seed];
    bodies.push(withAttr(tinted(new THREE.BoxGeometry(b.D, b.H, b.W - 0.02).translate(xc, b.H / 2, zc), b.wall), 'fac', fac));
    // Cornice and a ledge over the shops.
    trim.push(box(0.3, 0.22, b.W, x0 + s * 0.08, b.H - PARAPET + 0.05, zc, '#d8d2c6'));
    trim.push(box(0.22, 0.16, b.W, x0 - s * 0.06, GROUND - 0.25, zc, '#d0cabe'));
    if (b.pitched) {
      var G2 = 2.4 + (b.seed % 5) * 0.2, Ls = Math.hypot(b.D / 2 + 0.3, G2), ang = Math.atan2(G2, b.D / 2 + 0.3);
      var tri = new THREE.Shape();
      tri.moveTo(-b.D / 2, 0); tri.lineTo(b.D / 2, 0); tri.lineTo(0, G2); tri.lineTo(-b.D / 2, 0);
      bodies.push(withAttr(tinted(new THREE.ExtrudeGeometry(tri, { depth: b.W - 0.06, bevelEnabled: false }).translate(0, 0, -(b.W - 0.06) / 2)
        .translate(xc, b.H - 0.02, zc), b.wall), 'fac', [b.z0, b.bayW, -100, b.seed]));
      [-1, 1].forEach(function (k2) {
        roofs.push(tinted(new THREE.BoxGeometry(Ls + 0.1, 0.14, b.W + 0.1).rotateZ(k2 * ang)
          .translate(xc - k2 * (Math.cos(ang) * Ls / 2 - 0.15), b.H + G2 / 2 + 0.05, zc), b.roof));
      });
      if (b.seed % 3 === 0) trim.push(box(0.7, 2.2, 0.9, xc + s * 1.2, b.H + G2 - 0.3, zc + (b.seed % 2 ? 1 : -1) * b.W * 0.3, '#8a5444'));
    } else {
      roofs.push(box(b.D, 0.1, b.W, xc, b.H - 0.05, zc, '#3a3c40'));
    }
    // Window sills and, under some, boxes for flowers.
    for (var bi = 0; bi < b.bays; bi++) {
      var wz = b.z0 + (bi + 0.5) * b.bayW;
      for (var fl = 0; fl < b.floors; fl++) {
        var wy = GROUND + fl * STOREY + 0.17 * STOREY;
        sills.push(box(0.16, 0.08, b.bayW * 0.5, x0 - s * 0.06, wy - 0.04, wz, '#dcd6ca'));
        if (fl === 0 && (b.seed + bi) % 3 !== 2) {
          trim.push(box(0.3, 0.24, b.bayW * 0.44, x0 - s * 0.2, wy + 0.06, wz, (b.seed + bi) % 2 ? '#7a4a30' : '#3a4a3a'));
        }
      }
    }
    // A striped awning over some shops.
    if (b.awning) {
      var aw = b.W - 0.5, out = s < 0 ? 1.15 : 1.25;
      var ag = new THREE.PlaneGeometry(aw, Math.hypot(out, 0.6), 1, 1);
      var uv = ag.attributes.uv;
      for (var q = 0; q < uv.count; q++) uv.setX(q, uv.getX(q) * aw / 0.42);
      ag.rotateX(-Math.PI / 2 + Math.atan2(0.6, out)).rotateY(s < 0 ? Math.PI / 2 : -Math.PI / 2)
        .translate(x0 - s * out / 2, GROUND - 0.6 - 0.3, zc);
      awnings.push(tinted(ag, b.awning));
      var vg = new THREE.PlaneGeometry(aw, 0.28).rotateY(s < 0 ? Math.PI / 2 : -Math.PI / 2).translate(x0 - s * out, GROUND - 1.35, zc);
      var vuv = vg.attributes.uv;
      for (q = 0; q < vuv.count; q++) vuv.setX(q, vuv.getX(q) * aw / 0.42);
      awnings.push(tinted(vg, b.awning));
    }
  });
  var facades = new THREE.Mesh(mergeWith(bodies.map(function (g) { return g.index ? g.toNonIndexed() : g; }),
    [['position', 3], ['normal', 3], ['color', 3], ['fac', 4]]), facadeMat);
  facades.castShadow = facades.receiveShadow = true;
  world.add(facades);

  var plainMat = patch(new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.8 }), 'plain', G, {});
  var roofMat = patch(new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6 }), 'roof', G, {
    color: ' diffuseColor.rgb *= 0.85 + 0.25 * rnoise(vec3(vWP.x * 4.0, vWP.y * 6.0, vWP.z * 0.8)); rRough = mix(0.8, 0.3, uWet);'
  });
  var roofMesh = new THREE.Mesh(merge(roofs), roofMat);
  roofMesh.castShadow = roofMesh.receiveShadow = true;
  var trimMesh = new THREE.Mesh(merge(trim.concat(sills)), plainMat);
  trimMesh.castShadow = trimMesh.receiveShadow = true;
  world.add(roofMesh, trimMesh);

  var stripe = canvasTex(64, 8, function (x, w, h) {
    x.fillStyle = '#ffffff'; x.fillRect(0, 0, w, h);
    x.fillStyle = '#f4ece0'; x.fillRect(w / 2, 0, w / 2, h);
  });
  stripe.wrapS = stripe.wrapT = THREE.RepeatWrapping;
  var awnMat = patch(new THREE.MeshStandardMaterial({ vertexColors: true, map: stripe, roughness: 0.7, side: THREE.DoubleSide }), 'awning', G, {
    color: ' diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.92, 0.9, 0.85), step(0.5, fract(vMapUv.x)) * 0.85);'
  });
  var awnMesh = new THREE.Mesh(mergeWith(awnings, [['position', 3], ['normal', 3], ['color', 3], ['uv', 2]]), awnMat);
  awnMesh.castShadow = awnMesh.receiveShadow = true;
  world.add(awnMesh);

  // The two warm windows, and their halos.
  var eyeMat = new THREE.MeshBasicMaterial({ color: '#ffb860', fog: false });
  var glint = glintTex();
  var halo = softSprite('rgba(255,214,150,1)', 'rgba(255,170,90,0)');
  var eyeHalos = [];
  var eyeWin = mergeWith(EYES.map(function (e) {
    return new THREE.PlaneGeometry(2.6 * 0.4 - 0.06, STOREY * 0.6 - 0.06).rotateY(Math.PI / 2).translate(e.x, e.y, e.z).toNonIndexed();
  }), [['position', 3], ['normal', 3]]);
  var eyeMesh = new THREE.Mesh(eyeWin, eyeMat);
  world.add(eyeMesh);
  EYES.forEach(function (e) {
    var sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: glint, blending: THREE.AdditiveBlending, transparent: true, depthWrite: false, opacity: 0, fog: false }));
    sp.position.copy(e).x += 0.4;
    sp.scale.setScalar(6);
    world.add(sp);
    eyeHalos.push(sp);
  });

  // ── Street lamps along the kerbs ───────────────────────────────────────
  var lampSpots = [];
  for (var lz = TOWN_N - 6; lz > TOWN_S + 4; lz -= 17) {
    [-1, 1].forEach(function (s) {
      var z = lz + (s > 0 ? 8.5 : 0), x = s * (KERB + 0.35);
      if (!nearPath(x, z, 1.1) && Math.abs(z - TREE.z) > 6 && !(s > 0 && z > 26 && z < 46)) lampSpots.push(new THREE.Vector3(x, PAVE, z));
    });
  }
  lampSpots.push(new THREE.Vector3(-(KERB + 0.6), PAVE, TOWN_S - 1.2), new THREE.Vector3(KERB + 0.6, PAVE, TOWN_S - 1.2));
  var lampMat = patch(new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.4, metalness: 0.3 }), 'lamp', G, {});
  var lamps = new THREE.InstancedMesh(lampGeometry(), lampMat, lampSpots.length);
  lamps.castShadow = true;
  scatter(lamps, lampSpots.length, function (n, p, q, s) { p.copy(lampSpots[n]); q.identity(); s.setScalar(1); });
  world.add(lamps);
  var lanternMat = new THREE.MeshBasicMaterial({ color: '#ffd090' });
  var lanterns = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.24, 0.13, 0.62, 6).translate(0, 4.5, 0), lanternMat, lampSpots.length);
  scatter(lanterns, lampSpots.length, function (n, p, q, s) { p.copy(lampSpots[n]); q.identity(); s.setScalar(1); });
  world.add(lanterns);
  var glowMat = new THREE.SpriteMaterial({ map: halo, blending: THREE.AdditiveBlending, transparent: true, depthWrite: false, opacity: 1 });
  lampSpots.forEach(function (p) {
    var g = new THREE.Sprite(glowMat);
    g.position.set(p.x, p.y + 4.5, p.z);
    g.scale.setScalar(3.2);
    world.add(g);
  });

  // ── The tree that grows through the road ───────────────────────────────
  var cherry = cherryGeometry(rng(5));
  var treeG = new THREE.Group();
  treeG.position.set(TREE.x, 0, TREE.z);
  var TREE_S = 1.2;
  var woodMat = patch(new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9 }), 'wood', G, {});
  var bloomMat = patch(new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, emissive: '#3c1a26' }), 'bloom', G, {
    color: ' diffuseColor.rgb *= 0.66 + 0.4 * rnoise(vWP * 4.0); diffuseColor.rgb = mix(diffuseColor.rgb, vec3(1.0, 0.86, 0.9), smoothstep(0.6, 0.9, rnoise(vWP * 11.0)) * 0.35);'
  });
  var woodMesh = new THREE.Mesh(cherry.wood, woodMat), bloomMesh = new THREE.Mesh(cherry.bloom, bloomMat);
  woodMesh.castShadow = bloomMesh.castShadow = true;
  var bloomGroup = new THREE.Group();
  bloomGroup.add(bloomMesh);
  var bloomPtsGeo = new THREE.BufferGeometry();
  bloomPtsGeo.setAttribute('position', new THREE.Float32BufferAttribute(cherry.pts, 3));
  var bloomPts = new THREE.Points(bloomPtsGeo, new THREE.PointsMaterial({ color: '#ffb8cc', size: 0.16, map: softSprite('rgba(255,255,255,1)', 'rgba(255,230,238,0)'),
                                                                          transparent: true, depthWrite: false, alphaTest: 0.05 }));
  bloomGroup.add(bloomPts);
  treeG.add(woodMesh, bloomGroup);
  world.add(treeG);

  // ── Flowers: along the kerbs and doorsteps, round the tree, in the
  // window boxes. Each opens as the spread passes it. ──────────────────
  var FLOWERS = ['#e8304a', '#ff7aa0', '#ffd23a', '#ffffff', '#b066e0', '#ff8a2a', '#f4a0c8'];
  var FLU = { uSpread: { value: 0 }, uTreeP: { value: new THREE.Vector2(TREE.x, TREE.z) } };
  var flowerMat = patch(new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }), 'flower', G, {
    uniforms: FLU,
    vHead: 'uniform float uSpread; uniform vec2 uTreeP;\n' +
           'float rhashV(vec3 p){ return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }\n',
    begin: ' vec3 ip = vec3(instanceMatrix[3][0], instanceMatrix[3][1], instanceMatrix[3][2]);\n' +
           ' float fd = length(ip.xz - uTreeP) + rhashV(ip) * 6.0;\n' +
           ' transformed *= smoothstep(uSpread * 165.0, uSpread * 165.0 - 7.0, fd);\n',
    color_vertex: ' vColor = mix(vec3(0.16, 0.3, 0.1), instanceColor.xyz, color.r);',
    color: ''
  });
  var flowerSpots = [];
  var NF = small ? 1900 : 4200;
  for (i = 0; i < NF * 3 && flowerSpots.length < NF; i++) {
    var kind = r(), fx, fz, fy;
    if (kind < 0.38) {          // kerbside, in clumps
      var side = r() < 0.5 ? -1 : 1;
      fz = TOWN_N - r() * (TOWN_N - TOWN_S);
      if (Math.sin(fz * 0.7) + Math.sin(fz * 0.23) < 0.2) continue;
      fx = side * (KERB + 0.42 + r() * 0.35); fy = PAVE;
    } else if (kind < 0.7) {    // at the foot of the facades
      side = r() < 0.5 ? -1 : 1;
      fz = TOWN_N - r() * (TOWN_N - TOWN_S);
      if (Math.sin(fz * 0.5 + side) < -0.2) continue;
      fx = side * (FACE - 0.18 - r() * 0.3); fy = PAVE;
    } else if (kind < 0.86) {   // round the tree, out through the cracks
      var a = r() * Math.PI * 2, d = 0.5 + Math.pow(r(), 1.6) * 7;
      fx = TREE.x + Math.cos(a) * d; fz = TREE.z + Math.sin(a) * d; fy = 0;
      if (Math.abs(fx) > KERB) fy = PAVE;
    } else if (kind < 0.9) {    // out along the verges where the town ends
      fx = -6 - Math.pow(r(), 1.5) * 150;
      fz = ROAD_Z + (r() < 0.5 ? -1 : 1) * (ROAD_W + 0.5 + r() * 1.6); fy = 0;
      if (Math.abs(fx - CLOCK.x) < 1.2 && fz < ROAD_Z) continue;
    } else {                    // in the window boxes
      var b = TOWN[Math.floor(r() * TOWN.length)], bi2 = Math.floor(r() * b.bays);
      if ((b.seed + bi2) % 3 === 2) continue;
      fz = b.z0 + (bi2 + 0.5) * b.bayW + (r() - 0.5) * b.bayW * 0.38;
      fx = b.side * (FACE - 0.2); fy = GROUND + 0.17 * STOREY + 0.2;
    }
    if (nearPath(fx, fz, 0.5)) continue;
    flowerSpots.push([fx, fy, fz]);
  }
  var flowers = new THREE.InstancedMesh(tuftGeometry(), flowerMat, flowerSpots.length);
  scatter(flowers, flowerSpots.length, function (n, p, q, s, c) {
    var f = flowerSpots[n];
    p.set(f[0], f[1], f[2]);
    q.setFromAxisAngle(UP, r() * 6.28);
    s.setScalar(0.85 + r() * 0.7);
    c.set(FLOWERS[Math.floor(r() * FLOWERS.length)]);
  });
  world.add(flowers);

  // ── The long road: telegraph poles, mile posts, trees in the fields ────
  var poleParts = [], wires = [];
  var POLE_Z = ROAD_Z + ROAD_W + 2.2;
  for (var px = -14; px > -2400; px -= 46) {
    poleParts.push(box(0.24, 8, 0.24, px, 4, POLE_Z, '#4a3c30'));
    poleParts.push(box(0.16, 0.14, 2.2, px, 7.4, POLE_Z, '#4a3c30'));
  }
  var poleMat = patch(new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9 }), 'pole', G, {});
  var poles = new THREE.Mesh(merge(poleParts), poleMat);
  poles.castShadow = true;
  world.add(poles);
  [-0.9, 0, 0.9].forEach(function (off) {
    for (var wx = -14; wx > -2400 + 46; wx -= 46) {
      for (var sgm = 0; sgm < 8; sgm++) {
        var a0 = sgm / 8, a1 = (sgm + 1) / 8;
        wires.push(wx - a0 * 46, 7.5 - Math.sin(a0 * Math.PI) * 0.7, POLE_Z + off, wx - a1 * 46, 7.5 - Math.sin(a1 * Math.PI) * 0.7, POLE_Z + off);
      }
    }
  });
  var wireGeo = new THREE.BufferGeometry();
  wireGeo.setAttribute('position', new THREE.Float32BufferAttribute(wires, 3));
  var wireMat = new THREE.LineBasicMaterial({ color: '#1e1a18', transparent: true, opacity: 0.7 });
  world.add(new THREE.LineSegments(wireGeo, wireMat));

  var mileT = mileTex(), mileMat = patch(new THREE.MeshStandardMaterial({ map: mileT, roughness: 0.5 }), 'mile', G, {});
  var postParts = [], plates = [];
  for (var m = 0; m < 12; m++) {
    var mx = -52 - m * 58, mz = ROAD_Z - ROAD_W - 1.4;
    postParts.push(box(0.1, 1.5, 0.1, mx - 0.06, 0.75, mz, '#e8e8e2'));
    var pg = new THREE.PlaneGeometry(0.62, 0.52).rotateY(Math.PI / 2).translate(mx, 1.45, mz);
    var puv = pg.attributes.uv;
    for (var q2 = 0; q2 < puv.count; q2++) puv.setX(q2, (m + puv.getX(q2)) / 12);
    plates.push(pg.toNonIndexed());
  }
  world.add(new THREE.Mesh(mergeWith(plates, [['position', 3], ['normal', 3], ['uv', 2]]), mileMat));
  var posts = new THREE.Mesh(merge(postParts), plainMat);
  posts.castShadow = true;
  world.add(posts);

  var treeMat = patch(new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true }), 'trees', G, {});
  var greens = ['#5f8a3a', '#4c7a34', '#6e9440', '#3e6a2e', '#7a9a48'];
  [broadleafGeometry(rng(7), '#4a3a2e'), broadleafGeometry(rng(13), '#4a3a2e')].forEach(function (geo) {
    var trees = new THREE.InstancedMesh(geo, treeMat, small ? 160 : 360);
    trees.castShadow = true;
    scatter(trees, 9000, function (n, p, q, s, c) {
      var x = 300 - r() * 2200, z = 300 - r() * 1400;
      if (x > -40 && x < 40 && z > -100 && z < 90) return false;          // the town
      if (Math.abs(x) < 50 && z > -190) return false;                    // the view up the street
      if (Math.abs(z - ROAD_Z) < (x > -160 ? 34 : 12)) return false;      // the road
      if (Math.abs(x - CLOCK.x) < 14 && Math.abs(z - CLOCK.z) < 14) return false;
      // along the field edges, as hedgerow trees
      var e = Math.min(Math.abs(((x / 150) % 1 + 1) % 1 - 0.5), Math.abs(((z / 110) % 1 + 1) % 1 - 0.5));
      if (e < 0.44 && r() > 0.04) return false;
      p.set(x, -0.1, z);
      q.setFromAxisAngle(UP, r() * 6.28);
      var sc2 = 1.2 + r() * 1.2;
      s.set(sc2, sc2 * (0.85 + r() * 0.4), sc2);
      c.set(greens[Math.floor(r() * greens.length)]);
    });
    world.add(trees);
  });

  // ── Where the town ends: hedges round the corner and along the verges,
  // a field gate opposite the end of the street, the town's sign ────────
  function hedgeRow(out, x0, x1, z, h0, h1, seed) {
    var len = Math.abs(x1 - x0), steps = Math.ceil(len / 0.55), hr = rng(seed);
    for (var hs = 0; hs <= steps; hs++) {
      var hx = lerp(x0, x1, hs / steps), hh2 = lerp(h0, h1, hs / steps) * (0.85 + hr() * 0.3);
      var g = new THREE.IcosahedronGeometry(0.75, 1), pp = g.attributes.position;
      for (var v = 0; v < pp.count; v++) {
        var kk = 1 + 0.18 * Math.sin(pp.getX(v) * 5 + hs) * Math.cos(pp.getZ(v) * 4 + hs * 0.7);
        pp.setXYZ(v, pp.getX(v) * kk * 0.9, Math.min(pp.getY(v), 0.55) * kk, pp.getZ(v) * kk * 0.75);
      }
      g.scale(1, hh2 / 1.5, 1).translate(hx, hh2 / 2 - 0.05, z + (hr() - 0.5) * 0.2);
      out.push(tinted(g, ['#2f4a24', '#3a5a2a', '#2a4220', '#44622e'][hs % 4]));
    }
  }
  var hedgeParts = [], gateParts = [], NZ = ROAD_Z - ROAD_W - 1.7, SZ = ROAD_Z + ROAD_W + 2.6;
  hedgeRow(hedgeParts, 70, 2.6, NZ, 1.5, 1.5, 3);
  hedgeRow(hedgeParts, -2.6, -24, NZ, 1.5, 0.7, 4);
  hedgeRow(hedgeParts, -FACE - 0.6, -46, SZ, 1.6, 0.6, 5);
  hedgeRow(hedgeParts, FACE + 0.6, 60, SZ, 1.6, 1.4, 6);
  var hedgeMat = patch(new THREE.MeshLambertMaterial({ vertexColors: true }), 'hedge', G, {
    color: ' diffuseColor.rgb *= 0.7 + 0.5 * rnoise(vWP * 3.5); diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.9, 0.85, 0.6), smoothstep(0.82, 0.95, rnoise(vWP * 9.0)) * 0.5);'
  });
  var hedges = new THREE.Mesh(merge(hedgeParts), hedgeMat);
  hedges.castShadow = hedges.receiveShadow = true;
  world.add(hedges);
  // A five-bar field gate, weathered silver-brown.
  var gw = '#8c7e68';
  [-2.3, 2.3].forEach(function (gx) { gateParts.push(box(0.2, 1.6, 0.2, gx, 0.8, NZ, '#6e604c')); });
  for (var gb = 0; gb < 5; gb++) gateParts.push(box(4.4, 0.1, 0.06, 0, 0.3 + gb * 0.24, NZ + 0.06, gw));
  gateParts.push(tinted(new THREE.BoxGeometry(4.6, 0.09, 0.05).rotateZ(Math.atan2(0.96, 4.2)).translate(0, 0.78, NZ + 0.1), gw));
  [-2.1, 0, 2.1].forEach(function (gx) { gateParts.push(box(0.1, 1.06, 0.06, gx, 0.78, NZ + 0.06, gw)); });
  // The town sign on two posts at the corner, facing the way out.
  var SIGN_X = -12, SIGN_Z = NZ + 1.0;
  [-0.55, 0.55].forEach(function (dz) { gateParts.push(box(0.09, 2.3, 0.09, SIGN_X - 0.05, 1.15, SIGN_Z + dz, '#9a9a96')); });
  var gate = new THREE.Mesh(merge(gateParts), plainMat);
  gate.castShadow = true;
  world.add(gate);
  var signMat = patch(new THREE.MeshStandardMaterial({ map: townSignTex(), roughness: 0.5 }), 'sign', G, {});
  var sign = new THREE.Mesh(new THREE.PlaneGeometry(1.5, 0.75).rotateY(Math.PI / 2).translate(SIGN_X, 1.95, SIGN_Z), signMat);
  world.add(sign);
  var signBack = new THREE.Mesh(tinted(new THREE.PlaneGeometry(1.5, 0.75).rotateY(-Math.PI / 2).translate(SIGN_X - 0.02, 1.95, SIGN_Z), '#8a8a86'), plainMat);
  world.add(signBack);

  // ── The street clock on the verge ──────────────────────────────────────
  var clockG = new THREE.Group();
  clockG.position.set(CLOCK.x, 0, CLOCK.z);
  var iron = '#1e3a2c', brass = '#b08a3a';
  var clockBody = new THREE.Mesh(merge([
    tinted(new THREE.CylinderGeometry(0.42, 0.5, 0.5, 12).translate(0, 0.25, 0), iron),
    tinted(new THREE.CylinderGeometry(0.3, 0.38, 0.4, 12).translate(0, 0.7, 0), iron),
    tinted(new THREE.CylinderGeometry(0.13, 0.17, 2.4, 10).translate(0, 2.1, 0), iron),
    tinted(new THREE.CylinderGeometry(0.22, 0.16, 0.3, 10).translate(0, 3.0, 0), brass),
    tinted(new THREE.CylinderGeometry(CLOCK.R + 0.14, CLOCK.R + 0.14, 0.46, 32).rotateZ(Math.PI / 2).translate(0, CLOCK.y, 0), iron),
    tinted(new THREE.TorusGeometry(CLOCK.R + 0.06, 0.06, 8, 40).rotateY(Math.PI / 2).translate(0.24, CLOCK.y, 0), brass),
    tinted(new THREE.TorusGeometry(CLOCK.R + 0.06, 0.06, 8, 40).rotateY(Math.PI / 2).translate(-0.24, CLOCK.y, 0), brass),
    tinted(new THREE.ConeGeometry(0.28, 0.5, 10).translate(0, CLOCK.y + CLOCK.R + 0.42, 0), iron),
    tinted(new THREE.SphereGeometry(0.1, 8, 6).translate(0, CLOCK.y + CLOCK.R + 0.72, 0), brass)
  ]), patch(new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.45, metalness: 0.4 }), 'clock', G, {}));
  clockBody.castShadow = true;
  clockG.add(clockBody);
  var faceMat = patch(new THREE.MeshStandardMaterial({ map: clockTex(), roughness: 0.5, emissive: '#ffcf90', emissiveIntensity: 0 }), 'face', G, {});
  var hourHands = [], minHands = [];
  [1, -1].forEach(function (side) {
    var face = new THREE.Mesh(new THREE.CircleGeometry(CLOCK.R, 48).rotateY(side * Math.PI / 2), faceMat);
    face.position.set(side * 0.235, CLOCK.y, 0);
    clockG.add(face);
    var hm = new THREE.MeshStandardMaterial({ color: '#141414', roughness: 0.4 });
    var hourS = new THREE.Shape();
    hourS.moveTo(-0.045, -0.12); hourS.lineTo(0.045, -0.12); hourS.lineTo(0.03, 0.5); hourS.lineTo(0, 0.6); hourS.lineTo(-0.03, 0.5); hourS.lineTo(-0.045, -0.12);
    var minS = new THREE.Shape();
    minS.moveTo(-0.03, -0.16); minS.lineTo(0.03, -0.16); minS.lineTo(0.018, 0.78); minS.lineTo(0, 0.86); minS.lineTo(-0.018, 0.78); minS.lineTo(-0.03, -0.16);
    var pivot = new THREE.Group();
    pivot.position.set(side * 0.25, CLOCK.y, 0);
    pivot.rotation.y = side * Math.PI / 2;
    var hh = new THREE.Group(), mm = new THREE.Group();
    hh.add(new THREE.Mesh(new THREE.ShapeGeometry(hourS), hm));
    mm.add(new THREE.Mesh(new THREE.ShapeGeometry(minS).translate(0, 0, 0.008), hm));
    mm.add(new THREE.Mesh(new THREE.CircleGeometry(0.05, 12).translate(0, 0, 0.012), hm));
    pivot.add(hh, mm);
    clockG.add(pivot);
    hourHands.push({ g: hh, side: side });
    minHands.push({ g: mm, side: side });
  });
  // The heart at twelve that lights when the hands find it.
  var heartMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(1.6, 0.06, 0.12), transparent: true, opacity: 0 });
  var heart = new THREE.Mesh(new THREE.ShapeGeometry(heartShape(0.2)).rotateY(Math.PI / 2), heartMat);
  heart.position.set(0.245, CLOCK.y + CLOCK.R * 0.632, 0);
  clockG.add(heart);
  var heartGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,120,130,1)', 'rgba(255,60,80,0)'),
    blending: THREE.AdditiveBlending, transparent: true, depthWrite: false, opacity: 0 }));
  heartGlow.position.set(0.4, CLOCK.y + CLOCK.R * 0.62, 0);
  heartGlow.scale.setScalar(1.1);
  var faceGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: halo, blending: THREE.AdditiveBlending, transparent: true, depthWrite: false, opacity: 0 }));
  // Behind the face, so only a halo shows round the drum.
  faceGlow.position.set(0, CLOCK.y, 0);
  faceGlow.scale.setScalar(4.6);
  clockG.add(heartGlow, faceGlow);
  world.add(clockG);

  // ── Weather and light in the air ───────────────────────────────────────
  var rain = rainField({ count: small ? 2200 : 5200, box: [22, 16, 34], speed: 14, windSpeed: 3, color: '#c4ccd6', opacity: 0.32 });
  world.add(rain.lines);

  // Splashes: little rings flashing on the ground round you.
  var NS = small ? 500 : 1300, splashSeed = new Float32Array(NS);
  for (i = 0; i < NS; i++) splashSeed[i] = r() * 100;
  var splashGeo = new THREE.BufferGeometry();
  splashGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NS * 3), 3));
  splashGeo.setAttribute('seed', new THREE.BufferAttribute(splashSeed, 1));
  var splashU = { uTime: { value: 0 }, uRain: { value: 1 }, uCenter: { value: new THREE.Vector3() }, uScale: { value: 400 },
                  uColor: { value: new THREE.Color('#dfe6ee') } };
  var splashes = new THREE.Points(splashGeo, new THREE.ShaderMaterial({
    uniforms: splashU, transparent: true, depthWrite: false,
    vertexShader: [
      'attribute float seed; uniform float uTime, uRain, uScale; uniform vec3 uCenter; varying float vPh; varying float vA;',
      'float h1(float n){ return fract(sin(n) * 43758.5453); }',
      'void main(){',
      ' float rate = 1.4 + h1(seed * 3.1); float cyc = uTime * rate + seed * 17.0; float k = floor(cyc); vPh = fract(cyc);',
      ' vec2 off = vec2(h1(seed * 7.3 + k * 1.7), h1(seed * 2.9 + k * 3.1)) - 0.5;',
      ' vec3 p = vec3(uCenter.x + off.x * 22.0, 0.0, uCenter.z + off.y * 30.0);',
      ' p.y = (abs(p.x) > ' + KERB.toFixed(2) + ' && p.z > ' + (TOWN_S - 2).toFixed(1) + ') ? ' + (PAVE + 0.02).toFixed(2) + ' : 0.02;',
      ' vA = step(h1(seed * 5.7 + k), uRain);',
      ' vec4 mv = modelViewMatrix * vec4(p, 1.0);',
      ' gl_PointSize = (0.12 + 0.28 * vPh) * uScale / -mv.z;',
      ' gl_Position = projectionMatrix * mv;',
      '}'
    ].join('\n'),
    fragmentShader: [
      'uniform vec3 uColor; varying float vPh; varying float vA;',
      'void main(){ vec2 c = gl_PointCoord * 2.0 - 1.0; c.y *= 2.6; float r = length(c);',
      ' float a = exp(-pow((r - 0.75) / 0.16, 2.0)) * (1.0 - vPh) * vA * 0.55;',
      ' if (a < 0.01) discard; gl_FragColor = vec4(uColor, a); }'
    ].join('\n')
  }));
  splashes.frustumCulled = false;
  world.add(splashes);

  // Hanging drops: when the rain stops they stay in the air, glittering.
  var NG = small ? 1400 : 3200, gPos = new Float32Array(NG * 3), gSeed = new Float32Array(NG);
  for (i = 0; i < NG; i++) {
    gPos[i * 3] = (r() - 0.5) * 13;
    gPos[i * 3 + 1] = 0.4 + Math.pow(r(), 1.4) * 9;
    gPos[i * 3 + 2] = 34 - r() * 52;
    gSeed[i] = r() * 100;
  }
  var glitGeo = new THREE.BufferGeometry();
  glitGeo.setAttribute('position', new THREE.BufferAttribute(gPos, 3));
  glitGeo.setAttribute('seed', new THREE.BufferAttribute(gSeed, 1));
  var glitU = { uTime: { value: 0 }, uAmt: { value: 0 }, uScale: { value: 400 } };
  var glitter = new THREE.Points(glitGeo, new THREE.ShaderMaterial({
    uniforms: glitU, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: [
      'attribute float seed; uniform float uTime, uAmt, uScale; varying float vTw; varying vec3 vC;',
      'void main(){',
      ' vec3 p = position; p.y += sin(uTime * 0.4 + seed) * 0.05 - uTime * 0.0;',
      ' vTw = pow(0.5 + 0.5 * sin(uTime * (2.0 + mod(seed, 3.0)) + seed * 5.0), 6.0) * 0.85 + 0.15;',
      ' float hh = fract(seed * 0.618);',
      ' vC = mix(vec3(1.0, 0.95, 0.85), clamp(vec3(abs(hh * 6.0 - 3.0) - 1.0, 2.0 - abs(hh * 6.0 - 2.0), 2.0 - abs(hh * 6.0 - 4.0)), 0.0, 1.0), 0.45);',
      ' vec4 mv = modelViewMatrix * vec4(p, 1.0);',
      ' gl_PointSize = max((0.035 + 0.05 * vTw) * uScale / -mv.z, 1.5) * step(fract(seed * 1.37), uAmt);',
      ' gl_Position = projectionMatrix * mv;',
      '}'
    ].join('\n'),
    fragmentShader: [
      'uniform float uAmt; varying float vTw; varying vec3 vC;',
      'void main(){ vec2 c = gl_PointCoord * 2.0 - 1.0; float r = length(c);',
      ' float a = (exp(-r * r * 6.0) + max(0.0, 1.0 - abs(c.x) * 9.0) * max(0.0, 1.0 - abs(c.y)) * 0.5 + max(0.0, 1.0 - abs(c.y) * 9.0) * max(0.0, 1.0 - abs(c.x)) * 0.5) * vTw;',
      ' if (a < 0.01) discard; gl_FragColor = vec4(vC * a * 1.4, 1.0); }'
    ].join('\n')
  }));
  glitter.frustumCulled = false;
  world.add(glitter);

  // Petals drifting from the tree.
  var petalTex = canvasTex(32, 32, function (x) { x.fillStyle = '#fff'; x.beginPath(); x.ellipse(16, 16, 12, 7, 0.6, 0, Math.PI * 2); x.fill(); });
  var NP = small ? 260 : 600, pPos = new Float32Array(NP * 3), pVel = new Float32Array(NP * 3);
  for (i = 0; i < NP; i++) {
    pPos[i * 3] = TREE.x + (r() - 0.5) * 30; pPos[i * 3 + 1] = r() * 8; pPos[i * 3 + 2] = TREE.z + (r() - 0.3) * 40;
    pVel[i * 3] = 0.3 + r() * 0.5; pVel[i * 3 + 1] = 0.25 + r() * 0.35; pVel[i * 3 + 2] = (r() - 0.5) * 0.4;
  }
  var petalGeo = new THREE.BufferGeometry();
  petalGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
  var petals = new THREE.Points(petalGeo, new THREE.PointsMaterial({ color: '#ffd0dc', size: 0.09, map: petalTex, transparent: true,
                                                                      depthWrite: false, alphaTest: 0.4 }));
  petals.frustumCulled = false;
  world.add(petals);

  // Things the mirror leaves out: the ground itself and the weather.
  var notInMirror = [ground, tapeMesh, fields, rain.lines, splashes, glitter, petals], hidden = [];

  // ── Per frame ─────────────────────────────────────────────────────────
  var L = { dir: new THREE.Vector3(), tint: new THREE.Vector3() };
  LC.forEach(function (key) { L[key] = new THREE.Color(); });
  function mixLight(v) {
    var a0 = clamp(Math.floor(v), 0, LIGHT.length - 2), t = clamp(v - a0, 0, 1), A = LIGHT[a0], B = LIGHT[a0 + 1], n;
    for (n = 0; n < LC.length; n++) L[LC[n]].copy(A[LC[n]]).lerp(B[LC[n]], t);
    for (n = 0; n < LN.length; n++) L[LN[n]] = lerp(A[LN[n]], B[LN[n]], t);
    L.dir.copy(A.dir).lerp(B.dir, t).normalize();
    L.tint.copy(A.tint).lerp(B.tint, t);
  }
  var fwd = new THREE.Vector3(), tgt = new THREE.Vector3(), upv = new THREE.Vector3(), WHITE = new THREE.Color('#ffffff');
  var warmLamp = new THREE.Color('#ffc880'), offLamp = new THREE.Color('#3a3a36'), eyeOn = new THREE.Color(1.0, 0.52, 0.16);
  var beats = [-99, -99, -99, -99], beatPh = 0, dubDone = true, portrait = false, viewH = 800;

  function pushBeat(t) { beats[3] = beats[2]; beats[2] = beats[1]; beats[1] = beats[0]; beats[0] = t; }

  function frame(f) {
    var row = f.row, time = f.time, dt = f.dt;
    var walk = row[0], dark = row[1], rainAmt = row[2], lift = row[6], light = row[7], glit = row[8], bow = row[9],
        eyes = row[10], beat = row[11], grow = row[12], spread = row[13], dial = row[14];

    // On a portrait screen the verse sits mid-frame and the view is narrow:
    // turn further towards the clock, less away from the tree and the sun,
    // and a little off the near facade early on.
    var yaw = row[4];
    if (portrait) yaw = (yaw < 0 ? yaw * (light > 3.5 ? 1.7 : 1) : yaw * (eyes > 0.5 ? 1 : 0.4)) + 0.12 * (1 - smooth(0.3, 0.5, walk));
    followPath(camera, curve, baseY, clamp(walk, 0, 1), { eye: EYE, lift: lift, yaw: yaw, pitch: row[5], ahead: AHEAD,
                                                          mx: f.mx, my: f.my, time: time });
    camera.updateMatrixWorld();
    sky.position.copy(camera.position);

    // Light for the mood.
    mixLight(light);
    var du = dome.uniforms;
    du.uTop.value.copy(L.top); du.uHor.value.copy(L.hor); du.uCloud.value.copy(L.cloud); du.uShade.value.copy(L.shade);
    du.uSun.value.copy(L.sun); du.uSunDir.value.copy(L.dir);
    du.uCover.value = L.cover; du.uRift.value = L.rift; du.uBow.value = bow; du.uDisc.value = Math.pow(L.disc, 4); du.uTime.value = time;
    world.fog.color.copy(L.fog);
    world.fog.density = L.fogD + rainAmt * 0.004;
    gl.setClearColor(L.fog);
    hemi.color.copy(L.sky); hemi.groundColor.copy(L.ground); hemi.intensity = L.hemiI;
    sun.color.copy(L.sunC); sun.intensity = L.sunI;
    camera.getWorldDirection(fwd);
    sun.target.position.copy(camera.position).addScaledVector(fwd, 22);
    sun.target.position.y = 0;
    sun.position.copy(sun.target.position).addScaledVector(L.dir, 260);
    gl.toneMappingExposure = L.exp;

    G.uSat.value = L.sat; G.uTint.value.copy(L.tint); G.uTime.value = time; G.uWet.value = L.wet;
    R.uRain.value = rainAmt * 0.9;
    R.uCrack.value = smooth(0.0, 0.35, grow);

    // Lamps and lit windows in the gloom.
    lanternMat.color.copy(offLamp).lerp(warmLamp, dark).multiplyScalar(0.6 + 1.4 * dark);
    glowMat.opacity = dark * 0.9;
    FU.uLit.value = dark;
    FU.uShop.value = 0.35 + 0.65 * dark;
    FU.uGlass.value.copy(L.top).lerp(L.hor, 0.5);

    // The heartbeat: lub-dubs whose rate climbs with `beat` past 1.
    if (beat > 0.01) {
      var bpm = 62 + 40 * clamp(beat - 1, 0, 1);
      beatPh += dt * bpm / 60;
      if (beatPh >= 1) { beatPh -= 1; pushBeat(time); dubDone = false; }
      if (!dubDone && beatPh > 0.3) { pushBeat(time); dubDone = true; }
    }
    for (k = 0; k < 4; k++) G.uPulse.value[k] = time - beats[k];
    G.uBeat.value = clamp(beat, 0, 1);
    var throb = Math.exp(-(time - beats[0]) * 4) * clamp(beat, 0, 1);

    // The two warm windows.
    eyeMat.color.copy(eyeOn).multiplyScalar(eyes * (1.3 + 0.8 * throb));
    eyeMesh.visible = eyes > 0.01;
    for (k = 0; k < 2; k++) {
      eyeHalos[k].material.opacity = Math.min(1, eyes * (0.85 + 0.4 * throb));
      eyeHalos[k].scale.setScalar(5.5 + 2.5 * throb + Math.sin(time * 1.7 + k * 2) * 0.4);
      eyeHalos[k].material.rotation = Math.sin(time * 0.3 + k) * 0.1;
    }

    // The tree, and the flowers spreading out from it.
    var g = smooth(0, 1, grow);
    treeG.scale.setScalar(Math.max(0.0001, (0.04 + 0.96 * g) * TREE_S));
    treeG.visible = grow > 0.002;
    treeG.rotation.y = (1 - g) * 0.6;
    bloomGroup.scale.setScalar(Math.max(0.0001, smooth(0.45, 1, grow)));
    FLU.uSpread.value = spread;

    // The clock: the hands sweep round and settle together on the heart.
    var e = smooth(0, 1, dial);
    for (k = 0; k < 2; k++) {
      hourHands[k].g.rotation.z = lerp(-4.6 / 12 * Math.PI * 2, -Math.PI * 2, e);
      minHands[k].g.rotation.z = lerp(-38 / 60 * Math.PI * 2, -Math.PI * 6, e);
    }
    var tuned = smooth(0.85, 1, dial), hp = 0.75 + 0.25 * Math.sin(time * 6.5);
    heartMat.opacity = tuned;
    heartGlow.material.opacity = tuned * hp * 0.55;
    faceGlow.material.opacity = tuned * 0.7;
    faceMat.emissiveIntensity = tuned * 0.12;

    // Weather.
    rain.update(f, camera.position, rainAmt, env.reduceMotion);
    splashU.uTime.value = time; splashU.uRain.value = rainAmt; splashU.uCenter.value.copy(camera.position).addScaledVector(fwd, 9);
    splashU.uColor.value.copy(L.hor).lerp(WHITE, 0.5);
    splashes.visible = rainAmt > 0.01;
    glitU.uTime.value = time; glitU.uAmt.value = glit;
    glitter.visible = glit > 0.01;
    var pa = smooth(0.55, 0.9, grow) * (1 - smooth(3.7, 4.0, light));
    petals.visible = pa > 0.01;
    petals.material.opacity = pa;
    if (petals.visible) {
      var rate = env.reduceMotion ? 0.4 : 1;
      for (i = 0; i < NP; i++) {
        var o = i * 3;
        pPos[o] += (pVel[o] * (0.6 + f.wind) + Math.sin(time * 1.3 + i) * 0.25) * dt * rate;
        pPos[o + 1] -= pVel[o + 1] * dt * rate;
        pPos[o + 2] += (pVel[o + 2] + Math.cos(time * 0.9 + i) * 0.2) * dt * rate;
        if (pPos[o + 1] < 0.02) { pPos[o] = TREE.x + (r() - 0.5) * 9; pPos[o + 1] = 3 + r() * 5; pPos[o + 2] = TREE.z + (r() - 0.5) * 9; }
      }
      petalGeo.attributes.position.needsUpdate = true;
    }

    // The mirror: the camera reflected in the ground plane.
    reflCam.position.copy(camera.position); reflCam.position.y = -reflCam.position.y;
    tgt.copy(camera.position).add(fwd); tgt.y = -tgt.y;
    upv.set(0, 1, 0).applyQuaternion(camera.quaternion); upv.y = -upv.y;
    reflCam.up.copy(upv);
    reflCam.lookAt(tgt);
    reflCam.projectionMatrix.copy(camera.projectionMatrix);
    reflCam.projectionMatrixInverse.copy(camera.projectionMatrixInverse);
    reflCam.updateMatrixWorld();
    R.uTexMat.value.copy(BIAS).multiply(reflCam.projectionMatrix).multiply(reflCam.matrixWorldInverse);
    for (k = 0; k < notInMirror.length; k++) { hidden[k] = notInMirror[k].visible; notInMirror[k].visible = false; }
    gl.shadowMap.needsUpdate = true;
    gl.setRenderTarget(reflRT);
    gl.render(world, reflCam);
    gl.setRenderTarget(null);
    for (k = 0; k < notInMirror.length; k++) notInMirror[k].visible = hidden[k];
    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) {
      portrait = w < h;
      fitCamera(gl, camera, w, h, dpr, small);
      var pr = gl.getPixelRatio();
      reflRT.setSize(Math.max(4, Math.round(w * pr * RT_SCALE)), Math.max(4, Math.round(h * pr * RT_SCALE)));
      viewH = h * pr;
      var proj = viewH / (2 * Math.tan(camera.fov * Math.PI / 360));
      splashU.uScale.value = proj;
      glitU.uScale.value = proj;
    },
    frame: frame,
    destroy: function () { reflRT.dispose(); disposeAll(world, gl); }
  };
}

PI.register('rainy-turnaround', {
  renderer: renderer3d,
  align: ['left', 'right', 'left', 'left'],
  scrim: 0.62,
  accent: '#ffcf7a',
  emphasis: /^(hey|dial)[.,]?$/i,
  keys: function (T) {
    var S = T.start, E = T.end;
    //  unit          walk                   dark rain  wind  yaw    pitch  lift light glit bow  eyes beat grow  spread dial
    return [
      [0,             tNear(5, 40),          1.0, 1.0,  0.30, 0.00,  0.02, 0,   0.0, 0,   0,   0,   0,   0,    0,     0],
      [0.7,           tNear(5, 38),          1.0, 1.0,  0.30, 0.00,  0.02, 0,   0.0, 0,   0,   0,   0,   0,    0,     0],
      [S(0) + 0.5,    tNear(5, 30),          1.0, 1.0,  0.35, 0.06, -0.10, 0,   0.0, 0,   0,   0,   0,   0,    0,     0],  // walking in the downpour
      [S(0) + 1.0,    tNear(5, 25),          1.0, 1.0,  0.35, 0.00, -0.02, 0,   0.0, 0,   0,   0,   0,   0,    0,     0],
      [S(0) + 1.15,   tNear(5, 24),          0.7, 0.0,  0.10, -0.05, 0.06, 0,   0.6, 1,   0.1, 0,   0,   0,    0,     0],  // "the second that she says hey"
      [S(0) + 1.45,   tNear(4.9, 19),        0.0, 0.0,  0.05, -0.18, 0.14, 0,   1.0, 0.9, 0.85, 0,  0,   0,    0,     0],  // sun floods in; the rainbow
      [S(1) + 0.15,   tNear(4.5, 10),        0.0, 0.0,  0.05, 0.05, 0.12, 0,    1.3, 0.5, 1.0, 0.3, 0,   0,    0,     0],
      [S(1) + 0.5,    tNear(3.6, 4),         0.0, 0.0,  0.05, 0.16, 0.30, 0,    1.7, 0.1, 1.0, 1.0, 0,   0,    0,     0],  // "when I look into her eyes"
      [S(1) + 0.85,   tNear(EYES_CAM.x, EYES_CAM.z), 0, 0, 0.05, 0.10, -0.50, 0, 2.0, 0,   0.9, 1.0, 1.0, 0,    0,     0],  // in the puddle; the heartbeat
      [E(1) - 0.15,   tNear(2.6, -4),        0.0, 0.0,  0.05, 0.04, -0.14, 0,   2.0, 0,   0.8, 1.0, 2.0, 0,    0,     0],  // "my heartbeat rise"
      [S(2) + 0.25,   tNear(1.9, -24),       0.0, 0.0,  0.05, 0.00,  0.05, 0,   2.2, 0,   0.3, 0.4, 0.3, 0.06, 0,     0],  // "if only she knew"
      [S(2) + 0.75,   tNear(1.7, -34),       0.0, 0.0,  0.10, 0.16,  0.10, 0,   2.6, 0,   0,   0.2, 0,   0.7,  0.25,  0],  // "how much my love for her grew"
      [E(2) - 0.15,   tNear(2.4, -50),       0.0, 0.0,  0.15, 0.26,  0.14, 0,   3.0, 0,   0,   0.1, 0,   1.0,  0.85,  0],  // "never feel blue"
      [S(3) + 0.2,    tNear(-3.4, -100.6),    0.0, 0.0,  0.15, 0.00,  0.02, 0,   3.7, 0,   0,   0,   0,   1.0,  1.0,   0],  // the road runs for miles
      [S(3) + 0.55,   tNear(-15, -103.5),    0.0, 0.0,  0.15, -0.08, 0.15, 0,   3.85, 0,  0,   0,   0,   1.0,  1.0,   0],  // the clock
      [S(3) + 1.0,    tNear(-20, -103.5),    0.0, 0.0,  0.15, -0.22, 0.2, 0,    3.95, 0,  0,   0,   0,   1.0,  1.0,   1],  // "let my heart be my dial"
      [E(3) - 0.1,    tNear(-23.5, -103.5),  0.0, 0.0,  0.15, -0.42, 0.24, 0,   4.0, 0,   0,   0,   0,   1.0,  1.0,   1],  // the heart glowing at twelve
      [T.total,       tNear(-120, -103.5),   0.0, 0.0,  0.15, 0.2, -0.1, 2.2,  4.0, 0,   0,   0,   0,   1.0,  1.0,   1]   // into the golden evening
    ];
  },
  sound: {
    src: '/audio/rain.mp3',
    label: 'Play the rain, a heartbeat and the radio dial',
    volume: function (row) { return 0.02 + 0.72 * row[2]; },
    cues: [
      { stanza: 0, at: 1.08, play: sunburst },
      { stanza: 1, at: 0.7, play: heartbeat },
      { stanza: 3, at: 0.55, play: radioTune }
    ]
  }
});
