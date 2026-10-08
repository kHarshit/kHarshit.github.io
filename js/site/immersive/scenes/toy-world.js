/*
 * Scene for "Let Kids Be Kids" (Jennifer Caldwell): one long childhood day
 * in a toy world, seen at a child's eye height. The world is a little round
 * island: an open-fronted dollhouse with its garden, a lane looping round
 * it, a neighbour's cottage, a beach, a lawn for races and a field for
 * kites. Nobody is in it; the child is the eye, and their things are left
 * everywhere.
 *
 * (title) A summer morning over the garden; the view drops into the
 *      dollhouse's bathroom.
 * I    "Princesses, pirates; let the bath be the sea!": a pirate ship, a
 *      duck and a paper crown in the tub; down at water level the walls
 *      melt away and the bath is the open sea.
 * II   Out on the lawn, looking up: the clouds slowly become a rabbit, an
 *      elephant and a whale. Then rain, rings in the puddles on the path, a
 *      red umbrella twirling by a pair of yellow wellies.
 * III  The back seat of the toy car at dusk, the window wound down: a
 *      wishing star streaks across it; then the wind, the hedges and lamps
 *      of the lane smeared past, a pinwheel spinning on the sill and a
 *      rainbow ribbon streaming out.
 * IV   A jam jar of dandelions on the garden stump; "grow up in what seems
 *      like just an hour": the sun races across the sky, the flowers turn
 *      to clocks and their seeds blow away into the evening.
 * V    Bedtime upstairs: a storybook's pages turning with a little gold dust,
 *      the star mobile, the moon night-light throwing stars on the walls.
 *      Lightning at the window, and the view settles into the middle of the
 *      big bed between two pillows.
 * VI   The night garden: the dollhouse glows and fireflies keep watch. At the
 *      gate, a lantern comes alight on the neighbour's step beside a basket.
 * VII  Morning in the kitchen: the crowded calendar's pages tear off and
 *      whirl, the clock spins; then it all calms, the pages settle and the
 *      calendar is clear but for a sun.
 * VIII The beach: a sandcastle rises tower by tower; "give them a hand": the
 *      upturned bucket lifts away from the keep and a flag pops up.
 * IX   Down in the grass among daisies and kicked-off sandals, blades
 *      swaying, a ladybird climbing, a butterfly.
 * X    Night: the couch-cushion fort under a blanket, fairy lights along its
 *      edge, a glowing play tent; the view crawls inside.
 * XI   Breakfast in bed: a tray with burnt toast still smoking, a smiling
 *      egg, juice and a dandelion; the morning light turns golden.
 * XII  A tea party on the lawn: the pot pours for the teddy, a slice of cake
 *      goes to his plate, and a heart balloon floats up on its string.
 * XIII A race of wind-up cars: red snaps the tape; blue comes second and
 *      gets a smiling rosette of its own.
 * XIV  A piggy bank and an unopened present on the grass; then up the string
 *      to a homemade kite, high in the wind.
 * XV   Rising at golden hour over the whole toy world at once; balloons go
 *      up and the lights come on.
 *
 * Beats inside a panel come from the scroll position itself (beat(), from
 * the timeline the keys function hands over); the keys carry the camera,
 * which travels between the places, and the light of the day.
 * Columns: [unit, x, dark, rain, wind, y, z, yaw, pitch, sun, gold, blink, car, py, pp]
 *   x, y, z, yaw, pitch the eye; sun 0 sunrise .. 1 sunset; gold the warm
 *   low light; blink a quick dip that covers a cut between panels; car riding in the car; py, pp
 *   a turn and tilt that keep the subject clear of the verse on a phone.
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, softSprite, skyDome, starField, terrain,
         ribbon, scatter, particleField, rainField, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;
var TL = null;                          // the timeline, kept by keys()

// ── Layout (metres; the dollhouse faces south, +z) ───────────────────────
// The house spans x -6.5..6.5 and z -9 (back wall)..-3 (the open front);
// floors and ceilings of the two storeys:
var FRONT = -3, F0 = 0.3, C0 = 3.3, F1 = 3.5, C1 = 6.3;
var TUB = { x: -3.3, z: -8.2, w: 1.9, d: 0.86, h: 0.6 }, WATER = F1 + 0.47;
var BED = { x: 3.3, z0: -8.8, z1: -6.6, w: 1.9, top: F1 + 0.62 };
var LAMP = new THREE.Vector3(4.95, F1 + 0.78, -8.35);
var GATE_Z = 18, FENCE_X = 12, SEA_Y = -0.35;
var JAR = new THREE.Vector3(6.2, 0.56, 9.2);
var TEA = new THREE.Vector3(-6.5, 0, 7);
var NEIGH = new THREE.Vector3(6, 0, 31.5);
var CASTLE = new THREE.Vector3(22.5, 0, 40);
var LAWN = new THREE.Vector3(7.6, 0, 13.6);
var RACE_X = -42, FINISH = 2;
var PIGGY = new THREE.Vector3(-24.4, 0, 33.6), STAKE = new THREE.Vector3(-25.2, 0, 32.6);

// The beach lies to the south-east; elsewhere the lawn ends in a low bank.
function beachAmt(x, z) {
  var a = Math.atan2(x, z);
  return smooth(38, 45, Math.hypot(x, z)) * smooth(0.12, 0.42, a) * smooth(1.62, 1.3, a);
}
function land(x, z) {
  var rr = Math.hypot(x, z), b = beachAmt(x, z);
  var bumps = (0.32 * Math.sin(x * 0.11 + 1.3) * Math.cos(z * 0.09) + 0.16 * Math.sin(x * 0.23 - z * 0.19)) *
              smooth(18, 30, Math.hypot(x, (z - 4) * 1.15)) * (1 - b) * smooth(4, 9, Math.abs(x - RACE_X) + Math.abs(z + 8) * 0.4);
  return bumps - smooth(53, 61, rr) * 3.2 * (1 - b) - smooth(42, 70, rr) * 1.6 * b;
}

// The lane: a loop round the house and garden, passing the gate.
var ROAD = new THREE.CatmullRomCurve3((function () {
  var pts = [];
  for (var i = 0; i < 64; i++) { var a = i / 64 * Math.PI * 2; pts.push(new THREE.Vector3(30 * Math.sin(a), 0, -6 + 28 * Math.cos(a))); }
  return pts;
})(), true);

function hash2(a, b) { var s = Math.sin(a * 127.1 + b * 311.7) * 43758.5453; return s - Math.floor(s); }

// Yaw and pitch that look from p to t (the camera looks down -z at yaw 0).
function aim(p, t) {
  var dx = t[0] - p[0], dy = t[1] - p[1], dz = t[2] - p[2];
  return [Math.atan2(-dx, -dz), Math.atan2(dy, Math.hypot(dx, dz))];
}

// ── Geometry helpers ─────────────────────────────────────────────────────
// A box with rounded edges: corners pushed out onto a radius, normals
// smoothed, so toys read soft and plastic. Few segments = a pillowy face.
function rbox(w, h, d, rad, seg) {
  seg = seg || 4;
  var g = new THREE.BoxGeometry(w, h, d, seg, seg, seg), p = g.attributes.position, n = g.attributes.normal;
  var ex = Math.max(w / 2 - rad, 0), ey = Math.max(h / 2 - rad, 0), ez = Math.max(d / 2 - rad, 0);
  for (var i = 0; i < p.count; i++) {
    var x = p.getX(i), y = p.getY(i), z = p.getZ(i);
    var cx = clamp(x, -ex, ex), cy = clamp(y, -ey, ey), cz = clamp(z, -ez, ez);
    var dx = x - cx, dy = y - cy, dz = z - cz, l = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (l < 1e-6) continue;
    dx /= l; dy /= l; dz /= l;
    p.setXYZ(i, cx + dx * rad, cy + dy * rad, cz + dz * rad);
    n.setXYZ(i, dx, dy, dz);
  }
  return g;
}
function sph(rad, w, h) { return new THREE.SphereGeometry(rad, w || 12, h || 9); }
function cyl(rt, rb, h, s) { return new THREE.CylinderGeometry(rt, rb, h, s || 12); }
function box(w, h, d) { return new THREE.BoxGeometry(w, h, d); }

var _m = new THREE.Matrix4(), _e = new THREE.Euler();
// Transform a geometry, colour it and add it to a list for merging.
function put(list, geo, col, x, y, z, rx, ry, rz, sx, sy, sz) {
  if (sx != null) geo.scale(sx, sy == null ? sx : sy, sz == null ? sx : sz);
  if (rx || ry || rz) geo.applyMatrix4(_m.makeRotationFromEuler(_e.set(rx || 0, ry || 0, rz || 0)));
  geo.translate(x || 0, y || 0, z || 0);
  list.push(tinted(geo, col));
}

function starShape(ro, ri, n) {
  var s = new THREE.Shape();
  for (var i = 0; i <= n * 2; i++) {
    var a = i / (n * 2) * Math.PI * 2 + Math.PI / 2, rr = i % 2 ? ri : ro;
    if (!i) s.moveTo(Math.cos(a) * rr, Math.sin(a) * rr); else s.lineTo(Math.cos(a) * rr, Math.sin(a) * rr);
  }
  return s;
}
function heartShape(k) {
  var h = new THREE.Shape();
  h.moveTo(0, -k);
  h.bezierCurveTo(-k * 0.25, -k * 0.62, -k * 1.1, -k * 0.3, -k, k * 0.35);
  h.bezierCurveTo(-k * 0.9, k * 0.98, -k * 0.15, k * 1.02, 0, k * 0.5);
  h.bezierCurveTo(k * 0.15, k * 1.02, k * 0.9, k * 0.98, k, k * 0.35);
  h.bezierCurveTo(k * 1.1, -k * 0.3, k * 0.25, -k * 0.62, 0, -k);
  return h;
}
function puffy(shape, depth, bevel) {
  return new THREE.ExtrudeGeometry(shape, { depth: depth, bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel,
                                            bevelSegments: 3, curveSegments: 10 }).translate(0, 0, -depth / 2);
}

function canvasTex(w, h, paint) {
  var c = document.createElement('canvas');
  c.width = w; c.height = h;
  paint(c.getContext('2d'), w, h);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

// ── Water: one shader for the island's sea and the bath that becomes one.
// Flat, with ripple normals summed in the fragment shader (so it can be
// toy-sized in the tub and broad round the island), a fresnel mirror of the
// sky and a sun glint. uRect is the tub, always shown; uSea opens the rest.
var SEA_VS = 'varying vec3 vW;\n#include <fog_pars_vertex>\n' +
  'void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; vec4 mvPosition = viewMatrix * w;\n' +
  ' gl_Position = projectionMatrix * mvPosition;\n#include <fog_vertex>\n}';
var SEA_FS = 'uniform float uTime; uniform float uScale; uniform float uChop; uniform float uSea; uniform vec4 uRect;\n' +
  'uniform vec3 uDeep; uniform vec3 uSky; uniform vec3 uSun; uniform vec3 uSunDir; varying vec3 vW;\n#include <fog_pars_fragment>\n' +
  'void main(){\n' +
  ' float inside = step(uRect.x, vW.x) * step(vW.x, uRect.z) * step(uRect.y, vW.z) * step(vW.z, uRect.w);\n' +
  ' float a = max(inside, uSea); if (a < 0.003) discard;\n' +
  ' vec2 p = vW.xz * uScale; float t = uTime;\n' +
  ' vec2 g = vec2(0.8, 0.6) * cos(dot(p, vec2(0.8, 0.6)) + t * 1.1) * 0.5\n' +
  '        + vec2(-0.5, 0.86) * cos(dot(p, vec2(-0.5, 0.86)) * 1.7 + t * 1.5) * 0.35\n' +
  '        + vec2(0.2, -0.98) * cos(dot(p, vec2(0.2, -0.98)) * 2.9 + t * 2.2) * 0.22\n' +
  '        + vec2(-0.9, -0.4) * cos(dot(p, vec2(-0.9, -0.4)) * 4.6 + t * 2.9) * 0.14;\n' +
  ' float dist = length(cameraPosition - vW);\n' +
  ' float chop = uChop / (1.0 + dist * uScale * 0.06);\n' +
  ' vec3 n = normalize(vec3(-g.x * chop, 1.0, -g.y * chop));\n' +
  ' vec3 v = normalize(cameraPosition - vW);\n' +
  ' float fres = pow(1.0 - max(dot(n, v), 0.0), 3.0);\n' +
  ' vec3 col = mix(uDeep, uSky, 0.06 + 0.55 * fres);\n' +
  ' col += uSun * pow(max(dot(reflect(-v, n), normalize(uSunDir)), 0.0), 80.0) * 1.3;\n' +
  ' col += vec3(0.08) * smoothstep(0.7, 1.0, sin(dot(p, vec2(0.8, 0.6)) + t * 1.1)) / (1.0 + dist * uScale * 0.1);\n' +
  ' gl_FragColor = vec4(col, a);\n#include <tonemapping_fragment>\n#include <colorspace_fragment>\n#include <fog_fragment>\n}';

function seaMaterial(o) {
  return new THREE.ShaderMaterial({
    transparent: true, fog: true,
    uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, {
      uTime: { value: 0 }, uScale: { value: o.scale }, uChop: { value: o.chop }, uSea: { value: o.sea },
      uRect: { value: o.rect || new THREE.Vector4(0, 0, 0, 0) },
      uDeep: { value: new THREE.Color(o.deep) }, uSky: { value: new THREE.Color('#bfe4f7') },
      uSun: { value: new THREE.Color('#fff2d0') }, uSunDir: { value: new THREE.Vector3(0, 1, 0) }
    }]),
    vertexShader: SEA_VS, fragmentShader: SEA_FS
  });
}

// Puddles: a wet mirror of the sky with rings spreading in the rain.
var PUDDLE_FS = 'uniform float uTime; uniform float uRain; uniform vec3 uSky; uniform float uSeed; varying vec2 vUv;\n' +
  'float h1(float n){ return fract(sin(n) * 43758.5453); }\n' +
  'void main(){ vec2 q = vUv * 2.0 - 1.0; float d = length(q);\n' +
  ' float edge = 0.86 + 0.1 * sin(atan(q.y, q.x) * 3.0 + uSeed) + 0.05 * sin(atan(q.y, q.x) * 7.0 + uSeed * 2.0);\n' +
  ' float a = smoothstep(edge, edge - 0.12, d) * (0.15 + 0.7 * uRain);\n' +
  ' float ring = 0.0;\n' +
  ' for (int k = 0; k < 5; k++) { float fk = float(k); float tt = uTime * 1.1 + fk * 0.2 + uSeed;\n' +
  '  float ph = fract(tt), id = floor(tt);\n' +
  '  vec2 c = vec2(h1(id * 3.1 + fk * 7.7 + uSeed), h1(id * 5.3 + fk * 1.9 + uSeed)) * 1.2 - 0.6;\n' +
  '  float rr = length(q - c);\n' +
  '  ring += exp(-pow((rr - ph * 0.55) * 30.0, 2.0)) * (1.0 - ph); }\n' +
  ' vec3 col = mix(uSky * vec3(0.48, 0.55, 0.62), vec3(0.92, 0.95, 1.0), ring * 0.6 * uRain);\n' +
  ' gl_FragColor = vec4(col, a);\n#include <tonemapping_fragment>\n#include <colorspace_fragment>\n}';

// Dandelion seeds: each sits on its clock, grows with uGrow and, past its
// own moment in uBlow, lifts away on the wind.
var SEED_VS = 'attribute vec3 aHead; attribute vec3 aRand; uniform float uGrow; uniform float uBlow; uniform float uTime; uniform float uPx;\n' +
  'uniform vec3 uWind; varying float vA;\n' +
  'void main(){ float rel = clamp((uBlow - aRand.x * 0.55) / 0.45, 0.0, 1.0); float e = rel * rel;\n' +
  ' vec3 p = aHead + position * 0.07 * uGrow;\n' +
  ' p += uWind * e * (2.0 + aRand.y * 5.0);\n' +
  ' p.y += e * (0.8 + aRand.y * 2.4) + sin(uTime * 1.5 + aRand.z * 6.28) * 0.12 * rel;\n' +
  ' p.x += sin(uTime * 0.8 + aRand.z * 12.0) * 0.25 * rel;\n' +
  ' vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_Position = projectionMatrix * mv;\n' +
  ' vA = smoothstep(0.05, 0.3, uGrow) * (1.0 - smoothstep(0.7, 1.0, rel));\n' +
  ' gl_PointSize = (0.026 + 0.014 * rel) * uPx / -mv.z; }';
var SEED_FS = 'varying float vA; void main(){ vec2 q = gl_PointCoord - 0.5; float d = length(q); if (d > 0.5) discard;\n' +
  ' float spokes = pow(abs(cos(atan(q.y, q.x) * 6.0)), 6.0) * smoothstep(0.5, 0.1, d);\n' +
  ' float a = (smoothstep(0.5, 0.0, d) * 0.45 + spokes * 0.5 + smoothstep(0.1, 0.0, d)) * vA;\n' +
  ' gl_FragColor = vec4(vec3(1.0, 0.99, 0.95), a);\n#include <colorspace_fragment>\n}';

// The clouds that become animals, as circles [x, y, r] facing you.
var ANIMALS = {
  rabbit: [[0, 0, 1.6], [-1.3, -0.2, 1.3], [0.4, 0.8, 1.1], [1.7, 0.9, 1.05], [2.35, 0.65, 0.62], [1.3, 2.0, 0.5], [1.15, 2.7, 0.48],
           [1.05, 3.4, 0.45], [1.0, 4.05, 0.38], [1.95, 2.0, 0.5], [2.05, 2.7, 0.48], [2.15, 3.4, 0.45], [2.2, 4.05, 0.38],
           [-2.85, 0.4, 0.72], [1.1, -1.4, 0.62], [-1.8, -1.4, 0.66], [-0.4, -0.9, 0.9]],
  elephant: [[0, 0, 2.0], [-1.6, 0.1, 1.7], [1.3, 0.2, 1.7], [0.2, 1.4, 1.3], [-1.0, 1.2, 1.2], [3.0, 0.9, 1.35], [2.4, 0.7, 1.3],
             [4.2, 0.5, 0.62], [4.7, -0.2, 0.55], [4.9, -1.0, 0.5], [5.0, -1.7, 0.45], [5.4, -2.2, 0.4],
             [-1.7, -1.9, 0.8], [-0.5, -2.0, 0.8], [0.9, -2.0, 0.8], [2.0, -1.9, 0.8], [-1.7, -2.65, 0.7], [-0.5, -2.75, 0.7],
             [0.9, -2.75, 0.7], [2.0, -2.65, 0.7], [-3.4, 0.2, 0.42], [-3.8, -0.35, 0.32]],
  whale: [[0, 0, 1.9], [1.8, 0.2, 1.7], [-1.8, 0, 1.5], [-3.2, 0.2, 1.0], [-4.1, 0.5, 0.7], [-4.8, 1.2, 0.6], [-5.4, 1.8, 0.55],
          [-4.9, -0.2, 0.6], [-5.5, -0.7, 0.5], [3.2, 0.0, 1.2], [4.0, -0.2, 0.8], [2.0, 2.3, 0.4], [1.7, 3.0, 0.38],
          [2.4, 3.0, 0.38], [1.3, 3.55, 0.32], [2.8, 3.55, 0.32], [0.4, -1.3, 1.1], [1.8, -1.0, 1.0]]
};
// Where they hang, seen from the lawn: [azimuth left of the view, elevation, distance, scale];
// phones stack them up the narrow sky above and below the verse.
var CLOUD_AT = { rabbit: [0.5, 0.3, 62, 2.4], elephant: [0.2, 0.72, 70, 2.4], whale: [-0.36, 0.67, 72, 2.2] };
var CLOUD_AT_PHONE = { rabbit: [0.04, 0.12, 62, 1.9], elephant: [0.02, 0.68, 70, 1.8], whale: [-0.02, 0.98, 76, 1.7] };
var LAWN_EYE = new THREE.Vector3(-1, 1.0, 4.0);

// ── Sound ────────────────────────────────────────────────────────────────
function tone(ac, out, f, t, dur, gain, type) {
  var o = ac.createOscillator(), g = ac.createGain();
  o.type = type || 'sine';
  o.frequency.value = f;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(gain, t + 0.008);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g); g.connect(out);
  o.start(t); o.stop(t + dur + 0.05);
}
function noise(ac, len) {
  var b = ac.createBuffer(1, Math.floor(ac.sampleRate * len), ac.sampleRate), d = b.getChannelData(0);
  for (var i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  var s = ac.createBufferSource();
  s.buffer = b;
  return s;
}
function midi(n) { return 440 * Math.pow(2, (n - 69) / 12); }
// A music box: plucked tines, a bright partial and a quick metallic tick.
function musicBox(notes, step, gain) {
  return function (ac, out) {
    var t0 = ac.currentTime + 0.05;
    notes.forEach(function (n, i) {
      if (n == null) return;
      var t = t0 + i * step, f = midi(n);
      tone(ac, out, f, t, 1.8, gain || 0.07);
      tone(ac, out, f * 3.0, t, 0.6, (gain || 0.07) * 0.25);
      tone(ac, out, f * 5.4, t, 0.15, (gain || 0.07) * 0.1);
    });
  };
}
function wishChime(ac, out) {
  var t0 = ac.currentTime + 0.02;
  [84, 88, 91, 96, 100].forEach(function (n, i) { tone(ac, out, midi(n), t0 + i * 0.07, 1.6, 0.05); tone(ac, out, midi(n) * 2.01, t0 + i * 0.07, 0.6, 0.015); });
}
function rainPatter(ac, out) {
  var t0 = ac.currentTime;
  for (var i = 0; i < 70; i++) {
    var t = t0 + Math.random() * 2.4, s = noise(ac, 0.03), bp = ac.createBiquadFilter(), g = ac.createGain();
    bp.type = 'bandpass'; bp.frequency.value = 2500 + Math.random() * 3500; bp.Q.value = 3;
    g.gain.setValueAtTime(0.05 + Math.random() * 0.05, t);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.03);
    s.connect(bp); bp.connect(g); g.connect(out);
    s.start(t);
  }
}
function rumble(ac, out) {
  var t = ac.currentTime, s = noise(ac, 3.5), lp = ac.createBiquadFilter(), g = ac.createGain();
  lp.type = 'lowpass'; lp.frequency.value = 160;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.5, t + 0.25);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 3.4);
  s.connect(lp); lp.connect(g); g.connect(out);
  s.start(t);
}
// The kitchen clock racing and then slowing to an ordinary tick.
function ticking(ac, out) {
  var t = ac.currentTime + 0.05;
  for (var i = 0; i < 34; i++) {
    var gap = i < 22 ? 0.06 : 0.06 + (i - 21) * 0.07;
    tone(ac, out, i % 2 ? 2400 : 2000, t, 0.03, 0.03, 'square');
    t += gap;
  }
}
function toasterPop(ac, out) {
  var t = ac.currentTime;
  var o = ac.createOscillator(), g = ac.createGain();
  o.frequency.setValueAtTime(220, t);
  o.frequency.exponentialRampToValueAtTime(70, t + 0.15);
  g.gain.setValueAtTime(0.3, t);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 0.2);
  o.connect(g); g.connect(out);
  o.start(t); o.stop(t + 0.25);
  tone(ac, out, 1318.5, t + 0.12, 1.6, 0.05);
  tone(ac, out, 1318.5 * 2.76, t + 0.12, 0.5, 0.012);
}
function clink(ac, out) {
  var t = ac.currentTime;
  [[2650, 0], [3420, 0.0], [2650, 0.16], [3980, 0.16]].forEach(function (p) { tone(ac, out, p[0], t + p[1], 0.35, 0.025); });
}
function partyHorn(ac, out) {
  var t = ac.currentTime, o = ac.createOscillator(), lp = ac.createBiquadFilter(), g = ac.createGain();
  o.type = 'sawtooth';
  o.frequency.setValueAtTime(380, t);
  o.frequency.linearRampToValueAtTime(560, t + 0.35);
  lp.type = 'lowpass'; lp.frequency.value = 1400;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.05, t + 0.05);
  g.gain.setValueAtTime(0.05, t + 0.4);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 0.6);
  o.connect(lp); lp.connect(g); g.connect(out);
  o.start(t); o.stop(t + 0.65);
}

// ── Renderer ─────────────────────────────────────────────────────────────
function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(29);
  var gl = makeRenderer(canvas, { shadows: !small, clear: '#cfe8f7' });
  var world = new THREE.Scene();
  world.fog = new THREE.Fog('#dff2ff', 60, 340);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.05, 1600);
  camera.rotation.order = 'YXZ';
  world.add(camera);
  var pxScale = { value: 800 };
  var tc = new THREE.Color(), v1 = new THREE.Vector3(), v2 = new THREE.Vector3(), v3 = new THREE.Vector3();
  var q1 = new THREE.Quaternion(), m1 = new THREE.Matrix4(), e1 = new THREE.Euler(), s1 = new THREE.Vector3(), UP = new THREE.Vector3(0, 1, 0);
  var glowTex = softSprite('rgba(255,236,190,1)', 'rgba(255,200,120,0)');
  var dotTex = softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)');

  // Materials: plastic toys, matte walls and grass, the dollhouse shell
  // (which melts away for the sea) and things that glow.
  var toy = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.5, metalness: 0 });
  var toy2 = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, side: THREE.DoubleSide });
  var matte = new THREE.MeshLambertMaterial({ vertexColors: true });
  var shellMat = new THREE.MeshLambertMaterial({ vertexColors: true, transparent: true, emissive: '#ffb878', emissiveIntensity: 0 });
  var bathMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.3, transparent: true });
  var fading = [shellMat, bathMat];
  function mesh(list, mat, parent, shadow) {
    var m = new THREE.Mesh(merge(list), mat);
    m.castShadow = !!shadow && !small;
    m.receiveShadow = !small;
    (parent || world).add(m);
    return m;
  }
  function glowSprite(col, size, parent, x, y, z) {
    var s = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: col, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
    s.scale.setScalar(size);
    s.position.set(x, y, z);
    (parent || world).add(s);
    return s;
  }
  function basic(col) { return new THREE.MeshBasicMaterial({ color: col }); }

  // ── Sky ────────────────────────────────────────────────────────────────
  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#4f9be8', mid: '#8cc8f5', horizon: '#dff2ff', sun: '#fff1d0' }, 1400);
  sky.add(dome.mesh);
  var stars = starField(r, small ? 900 : 1800, 1300, 0.03, 1.7);
  sky.add(stars);
  var sunSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,252,240,1)', 'rgba(255,240,200,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
  var sunHalo = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
  sky.add(sunHalo, sunSprite);
  var moonTex = canvasTex(128, 128, function (x) {
    var g = x.createRadialGradient(64, 64, 0, 64, 64, 64);
    g.addColorStop(0, 'rgba(255,250,225,0.5)'); g.addColorStop(0.35, 'rgba(255,245,215,0.12)'); g.addColorStop(1, 'rgba(255,245,215,0)');
    x.fillStyle = g; x.fillRect(0, 0, 128, 128);
    x.fillStyle = '#fff6da'; x.beginPath(); x.arc(64, 64, 22, 0, Math.PI * 2); x.fill();
    x.globalCompositeOperation = 'destination-out';
    x.beginPath(); x.arc(76, 58, 20, 0, Math.PI * 2); x.fill();
  });
  var moon = new THREE.Sprite(new THREE.SpriteMaterial({ map: moonTex, depthWrite: false, transparent: true, fog: false }));
  moon.position.set(0.42, 0.36, 1).normalize().multiplyScalar(1000);
  moon.scale.setScalar(150);
  sky.add(moon);

  // Clouds: cotton-ball puffs, instanced. The first ones are the animals.
  var puffGeo = new THREE.IcosahedronGeometry(1, 2);
  var cloudMat = new THREE.MeshLambertMaterial({ color: '#ffffff', emissive: '#ffffff', emissiveIntensity: 0.45, fog: false });
  var animalPuffs = [], dayPuffs = [];
  Object.keys(ANIMALS).forEach(function (name) {
    ANIMALS[name].forEach(function (c, k) {
      animalPuffs.push({ name: name, a: c, blob: [(r() - 0.5) * 7.5, (r() - 0.35) * 1.8, 1.1 + r() * 0.8], z: (r() - 0.5) * 1.6, lag: r() * 0.35 });
    });
  });
  for (var ci = 0; ci < 16; ci++) {
    var ca = r() * Math.PI * 2, cd = 190 + r() * 160, ch = 75 + r() * 50, cs = 8 + r() * 8;
    for (var cj = 0; cj < 9; cj++) {
      dayPuffs.push({ x: Math.cos(ca) * cd + (cj - 4) * cs * 0.75 + (r() - 0.5) * cs, y: ch + Math.sin(cj / 8 * Math.PI) * cs * 0.6 + r() * cs * 0.3,
                      z: Math.sin(ca) * cd + (r() - 0.5) * cs, s: cs * (0.6 + r() * 0.5) * (0.6 + Math.sin(cj / 8 * Math.PI) * 0.5) });
    }
  }
  var puffs = new THREE.InstancedMesh(puffGeo, cloudMat, animalPuffs.length + dayPuffs.length);
  puffs.frustumCulled = false;
  world.add(puffs);
  var cloudPos = {};            // each animal's centre and facing, set on resize
  function placeAnimals(sx) {
    var y0 = aim([LAWN_EYE.x, LAWN_EYE.y, LAWN_EYE.z], [-3, 30, 60]);
    var table = sx < 1 ? CLOUD_AT_PHONE : CLOUD_AT;
    Object.keys(table).forEach(function (name) {
      var c = table[name], yaw = y0[0] + c[0], pitch = c[1];
      var dir = new THREE.Vector3(-Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw) * Math.cos(pitch));
      var at = LAWN_EYE.clone().addScaledVector(dir, c[2]);
      var F = dir.clone().negate(), R = new THREE.Vector3().crossVectors(UP, F).normalize(), U = new THREE.Vector3().crossVectors(F, R);
      cloudPos[name] = { at: at, R: R, U: U, F: F, s: c[3] };
    });
  }
  placeAnimals(1);

  // ── Light ──────────────────────────────────────────────────────────────
  var hemi = new THREE.HemisphereLight('#e4f2ff', '#8aa86a', 1.25);
  var sun = new THREE.DirectionalLight('#fff4e0', 2.4);
  sun.castShadow = !small;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.bias = -0.0005;
  sun.shadow.normalBias = 0.03;
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 240;
  var moonLight = new THREE.DirectionalLight('#9fb4ff', 0);
  moonLight.position.set(30, 50, 60);
  var nightLight = new THREE.PointLight('#ffcf8a', 0, 7.5, 1.3);
  nightLight.position.copy(LAMP).add(v1.set(-0.1, 0.1, 0.25));
  var fortLight = new THREE.PointLight('#ffb867', 0, 4.5, 1.6);
  fortLight.position.set(-3.6, 0.75, -6.7);
  var lanternLight = new THREE.PointLight('#ffc278', 0, 7, 1.6);
  world.add(hemi, sun, sun.target, moonLight, nightLight, fortLight, lanternLight);

  // ── The island and the sea ─────────────────────────────────────────────
  var grassA = new THREE.Color('#7cc556'), grassB = new THREE.Color('#6db84c'), grassC = new THREE.Color('#8acd5e');
  var sand = new THREE.Color('#f3dda4'), wet = new THREE.Color('#d6bd86'), earth = new THREE.Color('#b98b5c');
  var ground = terrain(150, small ? 130 : 190, 0, 0, land, new THREE.MeshLambertMaterial({ vertexColors: true }), function (x, z, y) {
    var c = new THREE.Color().copy(grassA).lerp(grassB, hash2(Math.floor(x / 6), Math.floor(z / 6)) * 0.8);
    if (Math.abs(x) < FENCE_X && z > FRONT - 1 && z < GATE_Z) c.copy(Math.floor((x + 30) / 1.6) % 2 ? grassA : grassC);
    c.lerp(sand, beachAmt(x, z));
    c.lerp(wet, smooth(-0.15, -0.32, y) * beachAmt(x, z));
    c.lerp(earth, smooth(-0.2, -1.0, y) * (1 - beachAmt(x, z)));
    return c;
  });
  world.add(ground);
  var sea = new THREE.Mesh(new THREE.CircleGeometry(800, 64).rotateX(-Math.PI / 2),
                           seaMaterial({ scale: 0.55, chop: 0.28, sea: 1, deep: '#1fa6c8' }));
  sea.position.y = SEA_Y;
  sea.material.transparent = false;
  world.add(sea);

  // Everything outdoors goes in `island`, hidden while the bath is the sea.
  var island = new THREE.Group();
  world.add(island);
  island.add(ground);

  // ── The lane, its lamps and hedges ─────────────────────────────────────
  var roadPts = ROAD.getSpacedPoints(240);
  var roadMesh = new THREE.Mesh(ribbon(roadPts, 0, 4.2, land, 0.04), new THREE.MeshLambertMaterial({ color: '#7c8396' }));
  roadMesh.receiveShadow = true;
  island.add(roadMesh);
  var dash = [];
  for (var di = 0; di < 240; di += 3) dash.push(roadPts[di], roadPts[di + 1]);
  var dashGeo = [];
  for (di = 0; di < dash.length; di += 2) dashGeo.push(ribbon([dash[di], dash[di + 1]], 0, 0.16, land, 0.06));
  var dashPos = [];
  dashGeo.forEach(function (g) { dashPos.push.apply(dashPos, g.attributes.position.array); g.dispose(); });
  var dashG = new THREE.BufferGeometry();
  dashG.setAttribute('position', new THREE.Float32BufferAttribute(dashPos, 3));
  dashG.computeVertexNormals();
  island.add(new THREE.Mesh(dashG, new THREE.MeshLambertMaterial({ color: '#f6f1df' })));

  var lampSpots = [], hedgeSpots = [], tan = new THREE.Vector3(), nrm = new THREE.Vector3();
  var roadLen = ROAD.getLength();
  for (var li = 0; li < Math.floor(roadLen / 1.9); li++) {
    var lt = li / Math.floor(roadLen / 1.9);
    ROAD.getPointAt(lt, v1);
    ROAD.getTangentAt(lt, tan);
    nrm.set(-tan.z, 0, tan.x);                 // outward, to the right of travel
    if (li % 6 === 0) lampSpots.push(v1.x + nrm.x * 2.7, v1.z + nrm.z * 2.7);
    var hx = v1.x + nrm.x * 3.9, hz = v1.z + nrm.z * 3.9;
    if (Math.hypot(hx - NEIGH.x, hz - NEIGH.z + 2) < 7 || li % 6 === 0 || hash2(li, 3) < 0.15 || Math.abs(Math.atan2(hx, hz)) < 0.1) continue;
    hedgeSpots.push(hx, hz, Math.atan2(tan.x, tan.z));
  }
  var postGeo = merge([tinted(cyl(0.05, 0.07, 3.2, 8).translate(0, 1.6, 0), '#41725e'), tinted(sph(0.09).translate(0, 3.22, 0), '#41725e'),
                       tinted(cyl(0.2, 0.26, 0.12, 10).translate(0, 3.35, 0), '#41725e'), tinted(sph(0.16, 12, 8).scale(1, 0.8, 1).translate(0, 3.22, 0), '#fff4d6')]);
  var posts = new THREE.InstancedMesh(postGeo, toy, lampSpots.length / 2);
  scatter(posts, lampSpots.length / 2, function (n, p, q, s) { var x = lampSpots[n * 2], z = lampSpots[n * 2 + 1]; p.set(x, land(x, z), z); q.identity(); s.setScalar(1); });
  posts.castShadow = !small;
  island.add(posts);
  var lampGlowPos = [];
  for (li = 0; li < lampSpots.length; li += 2) lampGlowPos.push(lampSpots[li], land(lampSpots[li], lampSpots[li + 1]) + 3.2, lampSpots[li + 1]);
  var lampGeo = new THREE.BufferGeometry();
  lampGeo.setAttribute('position', new THREE.Float32BufferAttribute(lampGlowPos, 3));
  var lampGlow = new THREE.Points(lampGeo, new THREE.PointsMaterial({ map: glowTex, color: '#ffc77a', size: 2.6, transparent: true, opacity: 0,
    blending: THREE.AdditiveBlending, depthWrite: false }));
  island.add(lampGlow);
  var hedges = new THREE.InstancedMesh(rbox(1.9, 1.05, 0.9, 0.4, 2), new THREE.MeshLambertMaterial({ color: '#ffffff' }), hedgeSpots.length / 3);
  scatter(hedges, hedgeSpots.length / 3, function (n, p, q, s, c) {
    var x = hedgeSpots[n * 3], z = hedgeSpots[n * 3 + 1];
    p.set(x, land(x, z) + 0.45, z);
    q.setFromAxisAngle(UP, hedgeSpots[n * 3 + 2] + Math.PI / 2);
    s.set(1, 0.85 + r() * 0.3, 1);
    c.setHSL(0.29 + r() * 0.04, 0.45, 0.32 + r() * 0.06);
  });
  hedges.castShadow = !small;
  hedges.receiveShadow = !small;
  island.add(hedges);

  // ── Trees round the island: lollipops and little cone pines ────────────
  function freeSpot(x, z) {
    var rr = Math.hypot(x, z);
    if (rr > 50 || beachAmt(x, z) > 0.02) return false;
    if (Math.abs(x) < 14 && z > -12 && z < 20) return false;
    if (Math.hypot(x - NEIGH.x, z - NEIGH.z) < 7) return false;
    if (Math.abs(x - RACE_X) < 6 && z > -24 && z < 8) return false;
    if (Math.hypot(x - PIGGY.x, z - PIGGY.z) < 6) return false;
    var e = Math.hypot(x / 30, (z + 6) / 28);
    return Math.abs(e - 1) * 29 > (e > 1 ? 9 : 5.5);
  }
  var trunkGeo = cyl(0.12, 0.18, 1.6, 7).translate(0, 0.8, 0);
  var crownGeo = new THREE.IcosahedronGeometry(1, 1).scale(1, 1.05, 1).translate(0, 2.4, 0);
  var pineGeo = merge([tinted(new THREE.ConeGeometry(1.1, 1.6, 8).translate(0, 1.3, 0), '#ffffff'),
                       tinted(new THREE.ConeGeometry(0.85, 1.4, 8).translate(0, 2.1, 0), '#ffffff'),
                       tinted(new THREE.ConeGeometry(0.6, 1.2, 8).translate(0, 2.85, 0), '#ffffff')]);
  var NT = small ? 60 : 100;
  var trunks = new THREE.InstancedMesh(trunkGeo, new THREE.MeshLambertMaterial({ color: '#9a6a44' }), NT);
  var crowns = new THREE.InstancedMesh(crownGeo, new THREE.MeshLambertMaterial({ color: '#ffffff' }), NT);
  var pines = new THREE.InstancedMesh(pineGeo, new THREE.MeshLambertMaterial({ vertexColors: true }), NT);
  var nt = 0, np = 0;
  var treeCols = ['#5fb04a', '#4ea544', '#79c24d', '#3f9a4a', '#f5a3b8', '#f0b84e', '#8fcf4f'];
  for (var ti = 0; ti < 4000 && (nt < NT || np < NT); ti++) {
    var tx = (r() - 0.5) * 104, tz = (r() - 0.5) * 104;
    if (!freeSpot(tx, tz) || hash2(Math.floor(tx / 13) + 3, Math.floor(tz / 13) + 7) < 0.45 - smooth(30, 48, Math.hypot(tx, tz)) * 0.4) continue;
    var tsc = 0.8 + r() * 0.7, ty = land(tx, tz) - 0.05;
    if (r() < 0.6 && nt < NT) {
      m1.compose(v1.set(tx, ty, tz), q1.setFromAxisAngle(UP, r() * 6.28), s1.set(tsc, tsc * (0.9 + r() * 0.3), tsc));
      trunks.setMatrixAt(nt, m1); crowns.setMatrixAt(nt, m1);
      crowns.setColorAt(nt, tc.set(treeCols[Math.floor(r() * treeCols.length)]));
      nt++;
    } else if (np < NT) {
      m1.compose(v1.set(tx, ty, tz), q1.setFromAxisAngle(UP, r() * 6.28), s1.set(tsc, tsc * (1 + r() * 0.4), tsc));
      pines.setMatrixAt(np, m1);
      pines.setColorAt(np, tc.setHSL(0.36 + r() * 0.05, 0.5, 0.3 + r() * 0.08));
      np++;
    }
  }
  trunks.count = crowns.count = nt;
  pines.count = np;
  [trunks, crowns, pines].forEach(function (m) { m.castShadow = !small; m.receiveShadow = !small; island.add(m); });

  // Flowers in the beds by the fence and in patches round the island.
  var flowerGeo = merge((function () {
    var l = [];
    put(l, cyl(0.012, 0.014, 0.32, 5), '#3f8a3a', 0, 0.16, 0);
    for (var k = 0; k < 5; k++) { var a = k / 5 * Math.PI * 2; put(l, sph(0.045, 7, 5), '#ffffff', Math.cos(a) * 0.05, 0.34, Math.sin(a) * 0.05, 0, 0, 0, 1, 0.5, 1); }
    put(l, sph(0.03, 7, 5), '#ffd23a', 0, 0.355, 0);
    return l;
  })());
  var flowerCols = ['#ff7aa2', '#ffd84a', '#ffffff', '#b98cff', '#ff9a4a', '#ff5f6d', '#7ac8ff'];
  var flowers = new THREE.InstancedMesh(flowerGeo, toy, small ? 500 : 1100);
  scatter(flowers, 20000, function (n, p, q, s, c) {
    var x, z, k = r();
    if (k < 0.45) { x = (r() - 0.5) * 22; z = 16.6 + r() * 1.0; if (Math.abs(x) < 1.6) return false; }
    else if (k < 0.75) { x = (r() < 0.5 ? -1 : 1) * (10.6 + r() * 1.0); z = -1 + r() * 18; }
    else { x = (r() - 0.5) * 100; z = (r() - 0.5) * 100; if (!freeSpot(x, z) || hash2(Math.floor(x / 7), Math.floor(z / 7)) < 0.75) return false; }
    p.set(x, land(x, z), z);
    q.setFromAxisAngle(UP, r() * 6.28);
    s.setScalar(0.8 + r() * 0.6);
    c.set(flowerCols[Math.floor(r() * flowerCols.length)]);
  });
  island.add(flowers);

  // Grass tufts over the lawns; a dense patch where you lie in the grass.
  var tuft = (function () {
    var pos = [], nor = [], col = [], root = new THREE.Color('#3f8a2c'), tip = new THREE.Color('#b8e07a'), rr = rng(5);
    for (var i = 0; i < 6; i++) {
      var a = rr() * 6.28, w = 0.014 + rr() * 0.01, h = 0.16 + rr() * 0.16, lean = 0.03 + rr() * 0.07;
      var ox = Math.cos(a) * 0.04, oz = Math.sin(a) * 0.04, px = -Math.sin(a) * w, pz = Math.cos(a) * w;
      pos.push(ox - px, 0, oz - pz, ox + px, 0, oz + pz, ox + Math.cos(a) * lean, h, oz + Math.sin(a) * lean);
      nor.push(0, 1, 0, 0, 1, 0, 0, 1, 0);
      col.push(root.r, root.g, root.b, root.r, root.g, root.b, tip.r, tip.g, tip.b);
    }
    var g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    return g;
  })();
  var U = { uClock: { value: 0 }, uWind: { value: 0.2 } };
  var grassMat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
  grassMat.onBeforeCompile = function (sh) {
    sh.uniforms.uClock = U.uClock; sh.uniforms.uWind = U.uWind;
    sh.vertexShader = 'uniform float uClock; uniform float uWind;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n float gph = instanceMatrix[3][0] * 1.7 + instanceMatrix[3][2] * 1.3;\n' +
      ' float gb = (sin(uClock * (1.6 + uWind * 2.0) + gph) * 0.6 + 0.3 + uWind * 0.5) * (0.25 + uWind * 0.6) * position.y * position.y;\n' +
      ' transformed.x += gb; transformed.z += gb * 0.4;');
  };
  var lawnTufts = new THREE.InstancedMesh(tuft, grassMat, small ? 5000 : 12000);
  scatter(lawnTufts, 60000, function (n, p, q, s, c) {
    var x, z;
    var inGarden = r() < 0.6;
    if (inGarden) { x = (r() - 0.5) * 24; z = FRONT + r() * 21; if (Math.abs(x) < 0.9 || Math.hypot(x - JAR.x, z - JAR.z) < 1.6 || Math.hypot(x - TEA.x, z - TEA.z) < 1.6 || Math.hypot(x - LAWN.x, z - LAWN.z) < 2.2) return false; }
    else { x = (r() - 0.5) * 100; z = (r() - 0.5) * 100; if (!freeSpot(x, z) && !(Math.abs(x - RACE_X) < 6 && z > -24 && z < 8)) return false; }
    p.set(x, land(x, z), z);
    q.setFromAxisAngle(UP, r() * 6.28);
    s.set(1.2, inGarden ? 0.3 + r() * 0.35 : 0.6 + r() * 0.6, 1.2);
    c.setHSL(0.24 + r() * 0.06, 0.4, 0.7 + r() * 0.3);
  });
  island.add(lawnTufts);
  // The patch you lie in: fine blades, finer near the middle.
  var fine = (function () {
    var pos = [], nor = [], colr = [], root = new THREE.Color('#2f7a2a'), rr = rng(8), tip = new THREE.Color();
    for (var i = 0; i < 7; i++) {
      var a = rr() * 6.28, w = 0.0035 + rr() * 0.003, h = 0.07 + rr() * 0.13, lean = 0.01 + rr() * 0.05;
      var ox = Math.cos(a) * 0.025, oz = Math.sin(a) * 0.025, px = -Math.sin(a) * w, pz = Math.cos(a) * w;
      tip.setHSL(0.22 + rr() * 0.06, 0.55, 0.55 + rr() * 0.15);
      pos.push(ox - px, 0, oz - pz, ox + px, 0, oz + pz, ox + Math.cos(a + 1.3) * lean, h, oz + Math.sin(a + 1.3) * lean);
      nor.push(0, 1, 0, 0, 1, 0, 0, 1, 0);
      colr.push(root.r, root.g, root.b, root.r, root.g, root.b, tip.r, tip.g, tip.b);
    }
    var g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(colr, 3));
    return g;
  })();
  var patch = new THREE.InstancedMesh(fine, grassMat, small ? 2500 : 6000);
  scatter(patch, 20000, function (n, p, q, s, c) {
    var x = LAWN.x + (r() - 0.5) * 4.4, z = LAWN.z + (r() - 0.5) * 4.4, d = Math.hypot(x - LAWN.x, z - LAWN.z);
    if (d > 2.2) return false;
    p.set(x, 0, z);
    q.setFromAxisAngle(UP, r() * 6.28);
    s.set(1, (0.7 + r() * 0.8) * lerp(0.35, 1, smooth(0.25, 0.9, d)), 1);
    c.setHSL(0, 0, 0.85 + r() * 0.3);
  });
  island.add(patch);

  // ── The dollhouse ──────────────────────────────────────────────────────
  // Two storeys, open at the front like a toy: living room and kitchen
  // below, bathroom and bedroom above, a red roof over.
  var house = new THREE.Group();
  world.add(house);
  var shell = [], ROOMS = { living: '#f7c9a6', kitchen: '#fbe6a0', bath: '#bfe7ec', bed: '#d6c7ef' };
  // A wall along x (at z0..z1) or along z (at x0..x1), with one window hole.
  function wallX(col, x0, x1, y0, y1, z0, z1, hole) {
    var zc = (z0 + z1) / 2, t = z1 - z0;
    if (!hole) { put(shell, box(x1 - x0, y1 - y0, t), col, (x0 + x1) / 2, (y0 + y1) / 2, zc); return; }
    put(shell, box(hole[0] - x0, y1 - y0, t), col, (x0 + hole[0]) / 2, (y0 + y1) / 2, zc);
    put(shell, box(x1 - hole[1], y1 - y0, t), col, (hole[1] + x1) / 2, (y0 + y1) / 2, zc);
    put(shell, box(hole[1] - hole[0], hole[2] - y0, t), col, (hole[0] + hole[1]) / 2, (y0 + hole[2]) / 2, zc);
    put(shell, box(hole[1] - hole[0], y1 - hole[3], t), col, (hole[0] + hole[1]) / 2, (hole[3] + y1) / 2, zc);
    winFrame(hole[0], hole[1], hole[2], hole[3], zc, 'x', t);
  }
  function wallZ(col, z0, z1, y0, y1, x0, x1, hole) {
    var xc = (x0 + x1) / 2, t = x1 - x0;
    if (!hole) { put(shell, box(t, y1 - y0, z1 - z0), col, xc, (y0 + y1) / 2, (z0 + z1) / 2); return; }
    put(shell, box(t, y1 - y0, hole[0] - z0), col, xc, (y0 + y1) / 2, (z0 + hole[0]) / 2);
    put(shell, box(t, y1 - y0, z1 - hole[1]), col, xc, (y0 + y1) / 2, (hole[1] + z1) / 2);
    put(shell, box(t, hole[2] - y0, hole[1] - hole[0]), col, xc, (y0 + hole[2]) / 2, (hole[0] + hole[1]) / 2);
    put(shell, box(t, y1 - hole[3], hole[1] - hole[0]), col, xc, (hole[3] + y1) / 2, (hole[0] + hole[1]) / 2);
    winFrame(hole[0], hole[1], hole[2], hole[3], xc, 'z', t);
  }
  function winFrame(a0, a1, b0, b1, c, axis, t) {
    var w = 0.07, d = t + 0.08, W = '#ffffff';
    var bars = [[(a0 + a1) / 2, b0, a1 - a0 + w * 2, w * 1.6], [(a0 + a1) / 2, b1, a1 - a0 + w * 2, w], [a0, (b0 + b1) / 2, w, b1 - b0],
                [a1, (b0 + b1) / 2, w, b1 - b0], [(a0 + a1) / 2, (b0 + b1) / 2, 0.035, b1 - b0], [(a0 + a1) / 2, (b0 + b1) / 2, a1 - a0, 0.035]];
    bars.forEach(function (b) {
      if (axis === 'x') put(shell, box(b[2], b[3], d), W, b[0], b[1], c);
      else put(shell, box(d, b[3], b[2]), W, c, b[1], b[0]);
    });
  }
  // Plinth, floors, ceilings
  put(shell, rbox(13.7, 0.3, 6.7, 0.08), '#efe2c8', 0, 0.15, -6);
  put(shell, box(6.45, 0.02, 5.8), '#e0ab70', -3.25, F0 + 0.01, -5.9);
  put(shell, box(6.45, 0.02, 5.8), '#f4ead6', 3.25, F0 + 0.01, -5.9);
  put(shell, box(13.2, 0.2, 6.0), '#fbf6ec', 0, C0 + 0.1, -6);
  put(shell, box(6.45, 0.02, 5.8), '#e6f6f4', -3.25, F1 + 0.01, -5.9);
  put(shell, box(6.45, 0.02, 5.8), '#cdb7d8', 3.25, F1 + 0.01, -5.9);
  put(shell, box(13.2, 0.16, 6.0), '#fbf6ec', 0, C1 + 0.08, -6);
  // Back walls (with windows), outer cladding, side walls, partition
  wallX(ROOMS.living, -6.5, 0, F0, C0, -9, -8.8, [-6.0, -5.1, 1.35, 2.45]);
  wallX(ROOMS.kitchen, 0, 6.5, F0, C0, -9, -8.8, [4.65, 5.95, 1.45, 2.45]);
  wallX(ROOMS.bath, -6.5, 0, F1, C1, -9, -8.8, [-1.6, -0.7, 4.65, 5.55]);
  wallX(ROOMS.bed, 0, 6.5, F1, C1, -9, -8.8);
  put(shell, box(13.3, 6.2, 0.06), '#f8f1e4', 0, 3.4, -9.03);
  wallZ(ROOMS.living, -9, -3, F0, C0, -6.5, -6.3);
  wallZ(ROOMS.bath, -9, -3, F1, C1, -6.5, -6.3);
  wallZ(ROOMS.kitchen, -9, -3, F0, C0, 6.3, 6.5);
  wallZ(ROOMS.bed, -9, -3, F1, C1, 6.3, 6.5, [-6.6, -4.8, 4.3, 5.5]);
  put(shell, box(0.06, 6.2, 6.1), '#f8f1e4', -6.53, 3.4, -6);
  put(shell, box(0.06, 2.9, 6.1), '#f8f1e4', 6.53, 1.8, -6);
  put(shell, box(0.06, 0.8, 6.1), '#f8f1e4', 6.53, 3.9, -6);
  put(shell, box(0.06, 0.8, 6.1), '#f8f1e4', 6.53, 5.9, -6);
  put(shell, box(0.06, 1.2, 1.6), '#f8f1e4', 6.53, 4.9, -7.8);
  put(shell, box(0.06, 1.2, 1.7), '#f8f1e4', 6.53, 4.9, -3.95);
  [[F0, C0, 'living', 'kitchen'], [F1, C1, 'bath', 'bed']].forEach(function (f) {
    put(shell, box(0.075, f[1] - f[0], 5.8), ROOMS[f[2]], -0.0375, (f[0] + f[1]) / 2, -5.9);
    put(shell, box(0.075, f[1] - f[0], 5.8), ROOMS[f[3]], 0.0375, (f[0] + f[1]) / 2, -5.9);
  });
  // The front: white posts and the edges of the floors, like a cabinet
  [-6.55, 0, 6.55].forEach(function (x) { put(shell, rbox(0.26, 6.5, 0.26, 0.05), '#ffffff', x, 3.25, -2.95); });
  put(shell, rbox(13.4, 0.26, 0.26, 0.05), '#ffffff', 0, C0 + 0.1, -2.95);
  put(shell, rbox(13.4, 0.22, 0.26, 0.05), '#ffffff', 0, C1 + 0.1, -2.95);
  // Roof: two red slopes, gable ends and a chimney
  var ridge = 8.5, eave = C1 + 0.16, slope = Math.atan2(ridge - eave, 3.4);
  put(shell, box(14.2, 0.22, Math.hypot(ridge - eave, 3.4) + 0.6), '#e7604f', 0, (ridge + eave) / 2 + 0.05, -4.3, slope);
  put(shell, box(14.2, 0.22, Math.hypot(ridge - eave, 3.4) + 0.6), '#d9564a', 0, (ridge + eave) / 2 + 0.05, -7.7, -slope);
  [-6.5, 6.5].forEach(function (x) {
    var g = new THREE.Shape();
    g.moveTo(-3.0, 0); g.lineTo(3.0, 0); g.lineTo(0, ridge - eave); g.lineTo(-3.0, 0);
    put(shell, new THREE.ExtrudeGeometry(g, { depth: 0.2, bevelEnabled: false }).rotateY(Math.PI / 2), '#f8f1e4', x - 0.1, eave, -6);
  });
  put(shell, rbox(0.7, 1.6, 0.7, 0.06), '#d98d6a', 3.8, 8.3, -7.6);
  put(shell, rbox(0.85, 0.15, 0.85, 0.04), '#c47a5a', 3.8, 9.1, -7.6);
  var shellMesh = mesh(shell, shellMat, house, false);

  // Window glass where the night flashes: the bedroom's side window.
  var flashPane = new THREE.Mesh(new THREE.PlaneGeometry(1.8, 1.2), new THREE.MeshBasicMaterial({ color: '#dfe8ff', transparent: true, opacity: 0, depthWrite: false }));
  flashPane.rotation.y = -Math.PI / 2;
  flashPane.position.set(6.42, 4.9, -5.7);
  house.add(flashPane);

  // ── Bathroom: the tub, its toys and the sea ────────────────────────────
  var bath = new THREE.Group();
  house.add(bath);
  var bl = [], tx0 = TUB.x - TUB.w / 2, tx1 = TUB.x + TUB.w / 2, tz0 = TUB.z - TUB.d / 2, tz1 = TUB.z + TUB.d / 2, rim = 0.09;
  put(bl, rbox(TUB.w, 0.12, TUB.d, 0.05), '#fdfdfb', TUB.x, F1 + 0.12, TUB.z);
  put(bl, rbox(TUB.w, TUB.h - 0.1, rim, 0.04), '#fdfdfb', TUB.x, F1 + 0.1 + (TUB.h - 0.1) / 2, tz0 + rim / 2);
  put(bl, rbox(TUB.w, TUB.h - 0.1, rim, 0.04), '#fdfdfb', TUB.x, F1 + 0.1 + (TUB.h - 0.1) / 2, tz1 - rim / 2);
  put(bl, rbox(rim * 1.4, TUB.h - 0.1, TUB.d, 0.05), '#fdfdfb', tx0 + rim * 0.7, F1 + 0.1 + (TUB.h - 0.1) / 2, TUB.z);
  put(bl, rbox(rim * 1.4, TUB.h - 0.1, TUB.d, 0.05), '#fdfdfb', tx1 - rim * 0.7, F1 + 0.1 + (TUB.h - 0.1) / 2, TUB.z);
  put(bl, rbox(TUB.w + 0.05, 0.06, 0.13, 0.03), '#ffffff', TUB.x, F1 + TUB.h + 0.01, tz0 + 0.04);
  put(bl, rbox(TUB.w + 0.05, 0.06, 0.13, 0.03), '#ffffff', TUB.x, F1 + TUB.h + 0.01, tz1 - 0.04);
  put(bl, rbox(0.16, 0.06, TUB.d + 0.05, 0.03), '#ffffff', tx0 + 0.06, F1 + TUB.h + 0.01, TUB.z);
  put(bl, rbox(0.16, 0.06, TUB.d + 0.05, 0.03), '#ffffff', tx1 - 0.06, F1 + TUB.h + 0.01, TUB.z);
  put(bl, box(TUB.w - 0.2, 0.02, TUB.d - 0.2), '#bfe9f2', TUB.x, F1 + 0.19, TUB.z);
  [[tx0 + 0.15, tz0 + 0.12], [tx1 - 0.15, tz0 + 0.12], [tx0 + 0.15, tz1 - 0.12], [tx1 - 0.15, tz1 - 0.12]].forEach(function (p) {
    put(bl, sph(0.07, 8, 6), '#e8c15a', p[0], F1 + 0.06, p[1]);
  });
  put(bl, cyl(0.025, 0.025, 0.3, 8), '#c9d2d8', tx0 + 0.25, F1 + TUB.h + 0.15, tz0 + 0.05);
  put(bl, cyl(0.02, 0.02, 0.22, 8), '#c9d2d8', tx0 + 0.25, F1 + TUB.h + 0.28, tz0 + 0.15, Math.PI / 2);
  // Sink, mirror, towel, mat, a cup of toothbrushes, a little stool
  put(bl, cyl(0.12, 0.1, 0.75, 12), '#fdfdfb', -6.0, F1 + 0.37, -5.4);
  put(bl, rbox(0.55, 0.16, 0.5, 0.07), '#fdfdfb', -6.0, F1 + 0.8, -5.4);
  put(bl, cyl(0.3, 0.3, 0.03, 20), '#d8f2fb', -6.27, F1 + 1.55, -5.4, 0, 0, Math.PI / 2);
  put(bl, cyl(0.33, 0.33, 0.025, 20), '#ffffff', -6.29, F1 + 1.55, -5.4, 0, 0, Math.PI / 2);
  put(bl, cyl(0.03, 0.035, 0.12, 8), '#ff9db0', -6.05, F1 + 0.94, -5.65);
  put(bl, cyl(0.006, 0.006, 0.16, 4), '#7ad0ff', -6.04, F1 + 1.0, -5.64, 0.15);
  put(bl, cyl(0.006, 0.006, 0.16, 4), '#ffd54a', -6.06, F1 + 1.0, -5.66, -0.15);
  put(bl, cyl(0.015, 0.015, 0.9, 6), '#c9d2d8', -1.0, F1 + 1.25, -8.72, 0, 0, Math.PI / 2);
  for (var st = 0; st < 5; st++) put(bl, box(0.16, 0.6, 0.03), st % 2 ? '#ffffff' : '#ff8fa8', -1.32 + st * 0.16, F1 + 0.95, -8.7);
  put(bl, rbox(1.1, 0.02, 0.6, 0.01, 2), '#ffb3c6', TUB.x, F1 + 0.03, TUB.z + 1.1);
  put(bl, rbox(0.42, 0.3, 0.32, 0.06), '#7ccfe0', -1.6, F1 + 0.15, -7.0);
  var bathFurn = mesh(bl, bathMat, bath, true);
  var tiles = canvasTex(256, 128, function (x, w, h) {
    x.fillStyle = '#aee0e8'; x.fillRect(0, 0, w, h);
    x.strokeStyle = '#ffffff'; x.lineWidth = 3;
    for (var i = 0; i <= 16; i++) { x.beginPath(); x.moveTo(i * 16, 0); x.lineTo(i * 16, h); x.stroke(); }
    for (var j = 0; j <= 8; j++) { x.beginPath(); x.moveTo(0, j * 16); x.lineTo(w, j * 16); x.stroke(); }
    x.fillStyle = '#ffffff';
    x.fillRect(0, 0, w, 6);
  });
  tiles.wrapS = THREE.RepeatWrapping;
  tiles.repeat.set(2, 1);
  var tileMat = new THREE.MeshLambertMaterial({ map: tiles, transparent: true });
  fading.push(tileMat);
  var tileWall = new THREE.Mesh(new THREE.PlaneGeometry(6.4, 1.4), tileMat);
  tileWall.position.set(-3.25, F1 + 0.7, -8.79);
  bath.add(tileWall);
  var floorTex = canvasTex(128, 128, function (x) {
    for (var i = 0; i < 8; i++) for (var j = 0; j < 8; j++) { x.fillStyle = (i + j) % 2 ? '#ffffff' : '#9fdbe6'; x.fillRect(i * 16, j * 16, 16, 16); }
  });
  floorTex.wrapS = floorTex.wrapT = THREE.RepeatWrapping;
  floorTex.repeat.set(5, 4.5);
  var floorMat = new THREE.MeshLambertMaterial({ map: floorTex, transparent: true });
  fading.push(floorMat);
  var bathFloor = new THREE.Mesh(new THREE.PlaneGeometry(6.4, 5.8).rotateX(-Math.PI / 2), floorMat);
  bathFloor.position.set(-3.25, F1 + 0.025, -5.9);
  bathFloor.receiveShadow = !small;
  bath.add(bathFloor);
  var tileSide = new THREE.Mesh(new THREE.PlaneGeometry(5.8, 1.4), tileMat);
  tileSide.rotation.y = Math.PI / 2;
  tileSide.position.set(-6.29, F1 + 0.7, -5.9);
  bath.add(tileSide);

  // The bath water (the sea in waiting) and the toys on it.
  var bathSea = new THREE.Mesh(new THREE.CircleGeometry(400, 64).rotateX(-Math.PI / 2),
    seaMaterial({ scale: 9, chop: 0.32, sea: 0, deep: '#2fb2d4', rect: new THREE.Vector4(tx0 + rim, tz0 + rim, tx1 - rim, tz1 - rim) }));
  bathSea.position.set(TUB.x, WATER, TUB.z);
  bathSea.renderOrder = 2;
  world.add(bathSea);
  var foamL = [];
  for (var fi = 0; fi < 40; fi++) {
    var fx = fi < 20 ? tx0 + 0.16 + r() * 0.35 : tx1 - 0.16 - r() * 0.3, fz = tz0 + rim + 0.05 + r() * (TUB.d - rim * 2 - 0.1);
    put(foamL, sph(0.03 + r() * 0.05, 8, 6), '#ffffff', fx, WATER + 0.01, fz);
  }
  var foam = mesh(foamL, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.2, transparent: true, opacity: 0.92 }), world);

  var ship = new THREE.Group(), sl = [];
  put(sl, rbox(0.3, 0.07, 0.1, 0.03), '#a0603a', 0, 0.02, 0);
  put(sl, new THREE.ConeGeometry(0.05, 0.1, 8).rotateZ(-Math.PI / 2).scale(1, 0.7, 1), '#a0603a', 0.19, 0.025, 0);
  put(sl, rbox(0.1, 0.06, 0.1, 0.02), '#8a4e2e', -0.11, 0.07, 0);
  put(sl, box(0.32, 0.015, 0.105), '#f2c14e', 0, 0.045, 0);
  put(sl, cyl(0.006, 0.006, 0.34, 6), '#5a3a22', 0.02, 0.2, 0);
  put(sl, cyl(0.005, 0.005, 0.24, 6), '#5a3a22', -0.09, 0.16, 0);
  put(sl, box(0.13, 0.12, 0.006), '#2b2f3a', 0.02, 0.22, 0);
  put(sl, box(0.1, 0.09, 0.006), '#2b2f3a', 0.02, 0.33, 0);
  put(sl, box(0.1, 0.09, 0.006), '#f6efe0', -0.09, 0.19, 0);
  put(sl, sph(0.016, 8, 6), '#ffffff', 0.02, 0.225, 0.006, 0, 0, 0, 1, 1, 0.3);
  put(sl, box(0.06, 0.035, 0.003), '#e5484d', 0.05, 0.36, 0);
  put(sl, cyl(0.012, 0.012, 0.02, 8), '#2b2f3a', 0.09, 0.05, 0.05, Math.PI / 2);
  put(sl, cyl(0.012, 0.012, 0.02, 8), '#2b2f3a', 0.0, 0.05, 0.05, Math.PI / 2);
  ship.add(new THREE.Mesh(merge(sl), toy));
  world.add(ship);
  var duck = new THREE.Group(), dl = [];
  put(dl, sph(0.06, 12, 9), '#ffd23a', 0, 0.03, 0, 0, 0, 0, 1.25, 0.8, 1);
  put(dl, sph(0.04, 12, 9), '#ffd23a', 0.045, 0.1, 0);
  put(dl, new THREE.ConeGeometry(0.018, 0.04, 8).rotateZ(-Math.PI / 2), '#ff8a2a', 0.095, 0.095, 0);
  put(dl, sph(0.007, 6, 5), '#222222', 0.07, 0.115, 0.025);
  put(dl, sph(0.007, 6, 5), '#222222', 0.07, 0.115, -0.025);
  put(dl, new THREE.ConeGeometry(0.03, 0.05, 6).rotateZ(Math.PI / 2.6), '#ffc21a', -0.075, 0.06, 0);
  duck.add(new THREE.Mesh(merge(dl), toy));
  world.add(duck);
  // The paper crown: a gold band with points.
  var crownGeoP = (function () {
    var pos = [], N = 7, R0 = 0.07;
    for (var i = 0; i < N * 2; i++) {
      var a0 = i / (N * 2) * Math.PI * 2, a1 = (i + 1) / (N * 2) * Math.PI * 2;
      var h0 = i % 2 ? 0.06 : 0.11, h1 = i % 2 ? 0.11 : 0.06;
      var x0 = Math.cos(a0) * R0, z0 = Math.sin(a0) * R0, x1 = Math.cos(a1) * R0, z1 = Math.sin(a1) * R0;
      pos.push(x0, 0, z0, x1, 0, z1, x1, h1, z1, x0, 0, z0, x1, h1, z1, x0, h0, z0);
    }
    var g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.computeVertexNormals();
    return g;
  })();
  var crown = new THREE.Mesh(crownGeoP, new THREE.MeshStandardMaterial({ color: '#f5c542', roughness: 0.4, metalness: 0.3, side: THREE.DoubleSide }));
  [0, 2, 4, 6, 8, 10, 12].forEach(function (k) {
    var a = k / 14 * Math.PI * 2, gem = new THREE.Mesh(sph(0.009, 6, 5), basic(['#ff5f8a', '#5fc8ff', '#7be07b'][k % 3]));
    gem.position.set(Math.cos(a) * 0.071, 0.035, Math.sin(a) * 0.071);
    crown.add(gem);
  });
  world.add(crown);
  var bubbles = particleField({ count: 60, box: [1.6, 1.4, 0.8], fall: [-0.18, -0.08], size: 0.05, map: dotTex, color: '#e8fbff', sway: 0.12, windSpeed: 0 });
  bubbles.points.material.opacity = 0.55;
  world.add(bubbles.points);
  var bubbleAt = new THREE.Vector3(TUB.x, WATER + 0.25, TUB.z);

  // ── Bedroom ────────────────────────────────────────────────────────────
  var bedroom = new THREE.Group();
  house.add(bedroom);
  var bd = [], bx0 = BED.x - BED.w / 2, bx1 = BED.x + BED.w / 2, bzc = (BED.z0 + BED.z1) / 2, blen = BED.z1 - BED.z0;
  put(bd, rbox(BED.w + 0.1, 0.32, blen + 0.06, 0.06), '#e4a96e', BED.x, F1 + 0.22, bzc);
  [[bx0, BED.z1], [bx1, BED.z1]].forEach(function (p) { put(bd, cyl(0.05, 0.05, 0.12, 8), '#c98c55', p[0], F1 + 0.06, p[1] - 0.05); });
  put(bd, rbox(BED.w, 0.2, blen, 0.08, 3), '#ffffff', BED.x, F1 + 0.48, bzc);
  put(bd, rbox(BED.w + 0.2, 1.1, 0.12, 0.05), '#e4a96e', BED.x, F1 + 0.6, BED.z0 + 0.06);
  put(bd, new THREE.CylinderGeometry(BED.w / 2 + 0.1, BED.w / 2 + 0.1, 0.12, 28, 1, false, Math.PI / 2, Math.PI).rotateX(Math.PI / 2).scale(1, 0.4, 1), '#e4a96e', BED.x, F1 + 1.14, BED.z0 + 0.06);
  put(bd, rbox(BED.w + 0.04, 0.3, 0.12, 0.05), '#e4a96e', BED.x, F1 + 0.42, BED.z1 + 0.02);
  // Duvet folded back, two big pillows and a small one in the middle
  put(bd, rbox(BED.w + 0.06, 0.1, blen - 0.75, 0.05, 3), '#8fc3ff', BED.x, F1 + 0.6, BED.z1 - (blen - 0.75) / 2);
  put(bd, rbox(BED.w + 0.06, 0.13, 0.3, 0.06, 3), '#ffffff', BED.x, F1 + 0.62, BED.z1 - (blen - 0.75) - 0.08);
  put(bd, rbox(0.78, 0.2, 0.45, 0.1, 2), '#fff7ea', BED.x - 0.48, F1 + 0.66, BED.z0 + 0.38, 0.12);
  put(bd, rbox(0.78, 0.2, 0.45, 0.1, 2), '#fff7ea', BED.x + 0.48, F1 + 0.66, BED.z0 + 0.38, 0.12);
  put(bd, rbox(0.36, 0.14, 0.28, 0.07, 2), '#ffd6e2', BED.x, F1 + 0.64, BED.z0 + 0.66, 0.1);
  // Bedside table, rug, toy chest, blocks
  put(bd, rbox(0.62, 0.62, 0.5, 0.05), '#f4d7a4', LAMP.x, F1 + 0.31, LAMP.z);
  put(bd, sph(0.025, 8, 6), '#c98c55', LAMP.x, F1 + 0.45, LAMP.z + 0.26);
  put(bd, cyl(1.1, 1.1, 0.02, 32), '#f7b7c9', 2.4, F1 + 0.02, -4.9);
  put(bd, cyl(0.75, 0.75, 0.025, 32), '#ffe08a', 2.4, F1 + 0.025, -4.9);
  put(bd, rbox(1.0, 0.55, 0.55, 0.06), '#7fc8a9', 1.0, F1 + 0.28, -8.4);
  put(bd, rbox(1.06, 0.1, 0.6, 0.05), '#5fb08f', 1.0, F1 + 0.6, -8.4);
  [['#ff6b6b', 1.3, -4.3, 0], ['#5aa9ff', 1.52, -4.32, 0.3], ['#ffd23a', 1.4, -4.1, 0.7], ['#7be07b', 1.41, -4.25, 0.2]].forEach(function (b, k) {
    put(bd, rbox(0.18, 0.18, 0.18, 0.03), b[0], b[1], F1 + 0.09 + (k === 3 ? 0.18 : 0), b[2], 0, b[3]);
  });
  addTeddy(bd, BED.x, F1 + 0.66, BED.z0 + 0.86, 0, 0.55);
  mesh(bd, toy, bedroom, true);
  // Star-and-moon wallpaper
  var paper = canvasTex(256, 256, function (x, w, h) {
    x.fillStyle = '#d6c7ef'; x.fillRect(0, 0, w, h);
    x.fillStyle = 'rgba(255,248,220,0.85)';
    for (var i = 0; i < 26; i++) {
      var cx = (i * 97) % 256, cy = (i * 61 + (i % 3) * 40) % 256, s = 5 + (i % 4);
      x.beginPath();
      for (var j = 0; j < 10; j++) { var a = j * Math.PI / 5 - Math.PI / 2, rr = j % 2 ? s * 0.45 : s; x.lineTo(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr); }
      x.fill();
    }
  });
  paper.wrapS = paper.wrapT = THREE.RepeatWrapping;
  paper.repeat.set(4, 2);
  var paperMat = new THREE.MeshLambertMaterial({ map: paper });
  var paperBack = new THREE.Mesh(new THREE.PlaneGeometry(6.3, 2.8), paperMat);
  paperBack.position.set(3.25, F1 + 1.4, -8.79);
  bedroom.add(paperBack);
  // The moon night-light: a glowing ball on a little stand
  var lampMat = new THREE.MeshBasicMaterial({ color: '#fff0c4' });
  var lampBall = new THREE.Mesh(sph(0.12, 16, 12), lampMat);
  lampBall.position.copy(LAMP);
  bedroom.add(lampBall);
  var lampBase = new THREE.Mesh(cyl(0.07, 0.09, 0.06, 12), new THREE.MeshStandardMaterial({ color: '#c9a77a' }));
  lampBase.position.set(LAMP.x, F1 + 0.65, LAMP.z);
  bedroom.add(lampBase);
  var lampGlowS = glowSprite('#ffd79a', 1.1, bedroom, LAMP.x, LAMP.y, LAMP.z);
  // The stars it throws on the walls and ceiling.
  var NS = 70, projDir = [], projPos = new Float32Array(NS * 3);
  for (var pi2 = 0; pi2 < NS; pi2++) {
    var py = 0.05 + r() * 0.9, pa = r() * 6.28;
    projDir.push(Math.sqrt(1 - py * py) * Math.cos(pa), py, Math.sqrt(1 - py * py) * Math.sin(pa), 0.5 + r() * 0.8);
  }
  var projGeo = new THREE.BufferGeometry();
  projGeo.setAttribute('position', new THREE.BufferAttribute(projPos, 3));
  var projStars = new THREE.Points(projGeo, new THREE.PointsMaterial({ map: canvasTex(32, 32, function (x) {
    x.fillStyle = '#fff'; x.beginPath();
    for (var j = 0; j < 10; j++) { var a = j * Math.PI / 5 - Math.PI / 2, rr = j % 2 ? 6 : 14; x.lineTo(16 + Math.cos(a) * rr, 16 + Math.sin(a) * rr); }
    x.fill();
  }), color: '#ffe7a8', size: 0.09, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending }));
  projStars.frustumCulled = false;
  bedroom.add(projStars);
  // The mobile over the bed
  var mobile = new THREE.Group();
  mobile.position.set(BED.x, C1, -7.3);
  bedroom.add(mobile);
  var ml = [];
  put(ml, cyl(0.004, 0.004, 0.5, 4), '#dddddd', 0, -0.25, 0);
  put(ml, new THREE.TorusGeometry(0.36, 0.012, 6, 32).rotateX(Math.PI / 2), '#ffffff', 0, -0.5, 0);
  var hang = [['star', '#ffd75a', 0.32], ['moon', '#fff3c4', 0.42], ['star', '#ffb3c6', 0.26], ['cloud', '#ffffff', 0.36], ['star', '#9fd8ff', 0.4], ['star', '#ffd75a', 0.3]];
  hang.forEach(function (h, k) {
    var a = k / hang.length * Math.PI * 2, x = Math.cos(a) * 0.36, z = Math.sin(a) * 0.36, y = -0.5 - h[2];
    put(ml, cyl(0.003, 0.003, h[2], 4), '#dddddd', x, -0.5 - h[2] / 2, z);
    if (h[0] === 'star') put(ml, puffy(starShape(0.09, 0.04, 5), 0.02, 0.012), h[1], x, y - 0.08, z, 0, -a);
    else if (h[0] === 'moon') {
      var mo = new THREE.Shape();
      mo.absarc(0, 0, 0.1, Math.PI * 0.25, Math.PI * 1.75, false);
      mo.absarc(0.055, 0, 0.08, Math.PI * 1.6, Math.PI * 0.4, true);
      put(ml, puffy(mo, 0.02, 0.012), h[1], x, y - 0.1, z, 0, -a);
    } else { put(ml, sph(0.06, 10, 8), h[1], x - 0.05, y - 0.06, z); put(ml, sph(0.075, 10, 8), h[1], x + 0.04, y - 0.05, z); put(ml, sph(0.05, 10, 8), h[1], x + 0.11, y - 0.07, z); }
  });
  mobile.add(new THREE.Mesh(merge(ml), toy));
  // The storybook: open on the bed, a page turning, gold dust rising
  var book = new THREE.Group();
  book.position.set(BED.x - 0.35, F1 + 0.67, -7.2);
  book.rotation.y = 0.35;
  bedroom.add(book);
  var bkl = [];
  put(bkl, rbox(0.5, 0.012, 0.34, 0.005, 2), '#3f6fd8', 0, 0, 0);
  put(bkl, box(0.23, 0.025, 0.31), '#fffaf0', -0.12, 0.018, 0, 0, 0, 0.04);
  put(bkl, box(0.23, 0.025, 0.31), '#fffaf0', 0.12, 0.018, 0, 0, 0, -0.04);
  book.add(new THREE.Mesh(merge(bkl), toy));
  var pageTex = canvasTex(256, 172, function (x, w, h) {
    x.fillStyle = '#fffaf0'; x.fillRect(0, 0, w, h);
    x.fillStyle = '#2b3a6a'; x.beginPath(); x.arc(64, 60, 34, 0, Math.PI * 2); x.fill();
    x.fillStyle = '#ffe17a'; x.beginPath(); x.arc(64, 60, 18, 0, Math.PI * 2); x.fill();
    x.fillStyle = '#2b3a6a'; x.beginPath(); x.arc(72, 54, 16, 0, Math.PI * 2); x.fill();
    x.fillStyle = '#c4c0b4';
    for (var i = 0; i < 7; i++) { x.fillRect(150, 30 + i * 18, 80 - (i % 3) * 12, 5); x.fillRect(20, 110 + (i % 3) * 18, 100 - (i % 2) * 20, 5); }
    x.fillStyle = '#e5484d'; x.beginPath(); x.moveTo(190, 150); x.lineTo(210, 160); x.lineTo(170, 160); x.fill();
  });
  var pagesTop = new THREE.Mesh(new THREE.PlaneGeometry(0.46, 0.31).rotateX(-Math.PI / 2), new THREE.MeshLambertMaterial({ map: pageTex }));
  pagesTop.position.y = 0.032;
  book.add(pagesTop);
  var leafPivot = new THREE.Group();
  leafPivot.position.y = 0.034;
  book.add(leafPivot);
  var leaf = new THREE.Mesh(new THREE.PlaneGeometry(0.22, 0.3).translate(0.11, 0, 0).rotateX(-Math.PI / 2),
                            new THREE.MeshLambertMaterial({ color: '#fff6e6', side: THREE.DoubleSide }));
  leafPivot.add(leaf);
  var dustN = 60, dustPos = new Float32Array(dustN * 3), dustSeed = [];
  for (var dk = 0; dk < dustN; dk++) dustSeed.push([r(), r(), r()]);
  var dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPos, 3));
  var dust = new THREE.Points(dustGeo, new THREE.PointsMaterial({ map: dotTex, color: '#ffd77a', size: 0.03, transparent: true, opacity: 0,
    blending: THREE.AdditiveBlending, depthWrite: false }));
  dust.frustumCulled = false;
  bedroom.add(dust);
  // Breakfast in bed: a tray with burnt toast, a smiling egg, juice and a flower
  var tray = new THREE.Group();
  tray.position.set(BED.x + 0.05, F1 + 0.66, -7.15);
  tray.rotation.y = -0.15;
  bedroom.add(tray);
  var tl = [];
  put(tl, rbox(0.64, 0.025, 0.42, 0.01, 2), '#f2c27a', 0, 0.012, 0);
  put(tl, rbox(0.64, 0.05, 0.02, 0.01, 2), '#e3a95e', 0, 0.04, 0.2);
  put(tl, rbox(0.64, 0.05, 0.02, 0.01, 2), '#e3a95e', 0, 0.04, -0.2);
  put(tl, rbox(0.02, 0.05, 0.42, 0.01, 2), '#e3a95e', 0.31, 0.04, 0);
  put(tl, rbox(0.02, 0.05, 0.42, 0.01, 2), '#e3a95e', -0.31, 0.04, 0);
  put(tl, cyl(0.12, 0.1, 0.015, 24), '#ffffff', -0.12, 0.035, 0.02);
  put(tl, rbox(0.12, 0.018, 0.12, 0.012, 2), '#2a1a12', -0.16, 0.052, 0.04, 0, 0.3);
  put(tl, rbox(0.115, 0.012, 0.115, 0.01, 2), '#3a2416', -0.16, 0.062, 0.04, 0, 0.3);
  put(tl, rbox(0.12, 0.018, 0.12, 0.012, 2), '#6b3d1e', -0.08, 0.05, -0.02, 0.05, -0.2);
  put(tl, cyl(0.08, 0.08, 0.008, 18), '#ffffff', 0.12, 0.03, 0.08, 0, 0, 0, 1, 1, 0.8);
  put(tl, sph(0.03, 12, 9), '#ffc21a', 0.12, 0.034, 0.08, 0, 0, 0, 1, 0.55, 1);
  put(tl, sph(0.006, 6, 5), '#3a2a1a', 0.105, 0.05, 0.075);
  put(tl, sph(0.006, 6, 5), '#3a2a1a', 0.135, 0.05, 0.075);
  put(tl, new THREE.TorusGeometry(0.014, 0.003, 4, 12, Math.PI).rotateX(-Math.PI / 2).rotateY(Math.PI), '#3a2a1a', 0.12, 0.045, 0.09);
  put(tl, cyl(0.035, 0.03, 0.11, 14), '#ff9a2a', 0.2, 0.08, -0.1);
  put(tl, cyl(0.022, 0.018, 0.045, 10), '#ffffff', 0.05, 0.045, -0.12);
  put(tl, cyl(0.002, 0.002, 0.12, 4), '#3f8a3a', 0.05, 0.11, -0.12);
  put(tl, sph(0.022, 10, 8), '#ffd23a', 0.05, 0.175, -0.12, 0, 0, 0, 1, 0.6, 1);
  var trayMesh = new THREE.Mesh(merge(tl), toy);
  trayMesh.castShadow = !small;
  tray.add(trayMesh);
  var juiceGlass = new THREE.Mesh(cyl(0.04, 0.034, 0.13, 16, 1, true), new THREE.MeshStandardMaterial({ color: '#ffffff', transparent: true, opacity: 0.35, roughness: 0.1 }));
  juiceGlass.position.set(0.2, 0.09, -0.1);
  tray.add(juiceGlass);
  var card = new THREE.Mesh(new THREE.PlaneGeometry(0.16, 0.12), new THREE.MeshLambertMaterial({ side: THREE.DoubleSide, map: canvasTex(128, 96, function (x) {
    x.fillStyle = '#fffdf6'; x.fillRect(0, 0, 128, 96);
    x.fillStyle = '#ff5a7a'; x.beginPath(); x.moveTo(64, 78); x.bezierCurveTo(20, 50, 30, 14, 64, 34); x.bezierCurveTo(98, 14, 108, 50, 64, 78); x.fill();
    x.strokeStyle = '#5aa9ff'; x.lineWidth = 3; x.strokeRect(6, 6, 116, 84);
  }) }));
  card.position.set(-0.02, 0.09, -0.15);
  card.rotation.set(-0.25, 0.25, 0);
  tray.add(card);
  var smokeN = 26, smokePos = new Float32Array(smokeN * 3), smokeSeed = [];
  for (var sk = 0; sk < smokeN; sk++) smokeSeed.push([r(), r()]);
  var smokeGeo = new THREE.BufferGeometry();
  smokeGeo.setAttribute('position', new THREE.BufferAttribute(smokePos, 3));
  var smoke = new THREE.Points(smokeGeo, new THREE.PointsMaterial({ map: dotTex, color: '#8b8580', size: 0.07, transparent: true, opacity: 0.5, depthWrite: false }));
  smoke.frustumCulled = false;
  tray.add(smoke);
  // Morning sun through the side window: a soft shaft of light.
  var beamGeo = (function () {
    var d = new THREE.Vector3(-0.73, -0.36, -0.5).normalize().multiplyScalar(3.6), c = [[6.4, 4.32, -6.58], [6.4, 4.32, -4.82], [6.4, 5.48, -4.82], [6.4, 5.48, -6.58]];
    var pos = [], al = [];
    for (var k = 0; k < 4; k++) {
      var a = c[k], b = c[(k + 1) % 4];
      [[a, 0], [b, 0], [b, 1], [a, 0], [b, 1], [a, 1]].forEach(function (v) {
        pos.push(v[0][0] + d.x * v[1], v[0][1] + d.y * v[1], v[0][2] + d.z * v[1]);
        al.push(v[1]);
      });
    }
    var g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('aT', new THREE.Float32BufferAttribute(al, 1));
    return g;
  })();
  var beamMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    uniforms: { uA: { value: 0 } },
    vertexShader: 'attribute float aT; varying float vT; void main(){ vT = aT; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform float uA; varying float vT; void main(){ gl_FragColor = vec4(vec3(1.0, 0.86, 0.6) * uA * (1.0 - vT) * (1.0 - vT) * 0.22, 1.0); }'
  });
  var beam = new THREE.Mesh(beamGeo, beamMat);
  beam.frustumCulled = false;
  bedroom.add(beam);
  var motes = particleField({ count: small ? 120 : 260, box: [3, 2.2, 3], fall: [-0.03, 0.03], size: 0.018, map: dotTex, color: '#ffe2a6', sway: 0.06, windSpeed: 0.1 });
  motes.points.material.blending = THREE.AdditiveBlending;
  world.add(motes.points);
  var moteAt = new THREE.Vector3(BED.x + 0.6, F1 + 1.1, -6.6);

  // ── Kitchen: the calendar, the clock, the whirl of pages ───────────────
  var kitchen = new THREE.Group();
  house.add(kitchen);
  var kl = [];
  put(kl, rbox(2.2, 0.88, 0.62, 0.04), '#ffffff', 5.2, F0 + 0.44, -8.48);
  put(kl, rbox(2.26, 0.06, 0.68, 0.02), '#7fc8a9', 5.2, F0 + 0.91, -8.47);
  for (var dw = 0; dw < 3; dw++) put(kl, sph(0.025, 8, 6), '#7fc8a9', 4.55 + dw * 0.65, F0 + 0.62, -8.15);
  put(kl, cyl(0.25, 0.2, 0.12, 16), '#e5e9ee', 4.6, F0 + 1.0, -8.5);
  put(kl, rbox(0.9, 2.0, 0.7, 0.1), '#bfe8f0', 0.6, F0 + 1.0, -8.42);
  put(kl, rbox(0.05, 0.4, 0.04, 0.02), '#8fb8c2', 0.98, F0 + 1.4, -8.04);
  put(kl, cyl(0.42, 0.42, 0.05, 24), '#ff9a76', 3.0, F0 + 0.58, -5.9);
  put(kl, cyl(0.05, 0.08, 0.56, 10), '#ffffff', 3.0, F0 + 0.28, -5.9);
  [[2.35, -5.9, Math.PI / 2], [3.65, -5.9, -Math.PI / 2]].forEach(function (c) {
    put(kl, rbox(0.34, 0.05, 0.34, 0.03), '#5aa9ff', c[0], F0 + 0.34, c[1]);
    put(kl, rbox(0.05, 0.34, 0.34, 0.03), '#5aa9ff', c[0] + (c[0] < 3 ? -0.15 : 0.15), F0 + 0.55, c[1]);
    [[-0.13, -0.13], [0.13, -0.13], [-0.13, 0.13], [0.13, 0.13]].forEach(function (l) { put(kl, cyl(0.018, 0.018, 0.32, 6), '#4a8fe0', c[0] + l[0], F0 + 0.16, c[1] + l[1]); });
  });
  put(kl, box(0.3, 0.004, 0.22), '#ffffff', 2.9, F0 + 0.61, -5.8, 0, 0.3);
  ['#e5484d', '#5aa9ff', '#ffd23a', '#7be07b'].forEach(function (c, k) { put(kl, cyl(0.008, 0.008, 0.12, 6), c, 3.2 + k * 0.03, F0 + 0.612, -6.05 + k * 0.02, 0, 0, Math.PI / 2); });
  put(kl, rbox(0.34, 0.24, 0.16, 0.05), '#ff7aa2', 3.3, F0 + 0.73, -5.75);
  put(kl, new THREE.TorusGeometry(0.06, 0.012, 6, 12, Math.PI), '#e5487d', 3.3, F0 + 0.85, -5.75);
  mesh(kl, toy, kitchen, true);
  function calendarTex(busy) {
    return canvasTex(256, 300, function (x, w, h) {
      x.fillStyle = '#fffdf6'; x.fillRect(0, 0, w, h);
      x.fillStyle = busy ? '#ff7a5a' : '#7fc8a9'; x.fillRect(0, 0, w, 48);
      x.fillStyle = '#ffffff'; x.font = 'bold 26px sans-serif'; x.textAlign = 'center'; x.fillText(busy ? 'BUSY' : 'PLAY', w / 2, 34);
      x.strokeStyle = '#d8d2c4'; x.lineWidth = 2;
      var cw = w / 7, ch = (h - 56) / 5, cols = ['#ff7aa2', '#5aa9ff', '#ffd23a', '#7be07b', '#b98cff', '#ff9a4a'];
      for (var i = 0; i < 35; i++) {
        var cx = (i % 7) * cw, cy = 56 + Math.floor(i / 7) * ch;
        x.strokeRect(cx + 2, cy + 2, cw - 4, ch - 4);
        if (busy) for (var k = 0; k < 2 + (i * 7) % 3; k++) { x.fillStyle = cols[(i + k * 2) % cols.length]; x.fillRect(cx + 5, cy + 6 + k * 12, cw - 10 - ((i + k) % 3) * 6, 8); }
      }
      if (!busy) {
        var sx = 3.5 * cw, sy = 56 + 2.5 * ch;
        x.fillStyle = '#ffc21a'; x.beginPath(); x.arc(sx, sy, 20, 0, Math.PI * 2); x.fill();
        x.strokeStyle = '#ffc21a'; x.lineWidth = 4;
        for (var k2 = 0; k2 < 8; k2++) { var a = k2 / 8 * Math.PI * 2; x.beginPath(); x.moveTo(sx + Math.cos(a) * 26, sy + Math.sin(a) * 26); x.lineTo(sx + Math.cos(a) * 36, sy + Math.sin(a) * 36); x.stroke(); }
        x.fillStyle = '#ff5a7a'; x.beginPath(); x.moveTo(5.5 * cw, 56 + 4.7 * ch); x.bezierCurveTo(5.1 * cw, 56 + 4.3 * ch, 5.2 * cw, 56 + 3.95 * ch, 5.5 * cw, 56 + 4.2 * ch);
        x.bezierCurveTo(5.8 * cw, 56 + 3.95 * ch, 5.9 * cw, 56 + 4.3 * ch, 5.5 * cw, 56 + 4.7 * ch); x.fill();
      }
    });
  }
  var calBusy = new THREE.Mesh(new THREE.PlaneGeometry(1.0, 1.17), new THREE.MeshLambertMaterial({ map: calendarTex(true), transparent: true }));
  var calCalm = new THREE.Mesh(new THREE.PlaneGeometry(1.0, 1.17), new THREE.MeshLambertMaterial({ map: calendarTex(false), transparent: true, opacity: 0 }));
  calBusy.position.set(2.1, F0 + 1.6, -8.78);
  calCalm.position.set(2.1, F0 + 1.6, -8.775);
  kitchen.add(calBusy, calCalm);
  var clockFace = new THREE.Mesh(cyl(0.32, 0.32, 0.05, 32).rotateX(Math.PI / 2), new THREE.MeshLambertMaterial({ map: canvasTex(128, 128, function (x) {
    x.fillStyle = '#ffffff'; x.fillRect(0, 0, 128, 128);
    x.fillStyle = '#3a4a6a';
    for (var i = 0; i < 12; i++) { var a = i / 12 * Math.PI * 2; x.beginPath(); x.arc(64 + Math.cos(a) * 50, 64 + Math.sin(a) * 50, i % 3 ? 3 : 6, 0, Math.PI * 2); x.fill(); }
  }) }));
  var clock = new THREE.Group();
  clock.position.set(3.5, F0 + 2.3, -8.74);
  clock.add(clockFace);
  var clockRim = new THREE.Mesh(new THREE.TorusGeometry(0.33, 0.035, 8, 32), new THREE.MeshStandardMaterial({ color: '#ff7a5a', roughness: 0.5 }));
  clock.add(clockRim);
  var minHand = new THREE.Group(), hourHand = new THREE.Group();
  var handMat = basic('#2b3a5a');
  var mh = new THREE.Mesh(box(0.02, 0.26, 0.01).translate(0, 0.11, 0), handMat), hh = new THREE.Mesh(box(0.03, 0.17, 0.01).translate(0, 0.07, 0), handMat);
  minHand.add(mh); hourHand.add(hh);
  minHand.position.z = hourHand.position.z = 0.035;
  clock.add(minHand, hourHand);
  kitchen.add(clock);
  var NPAGE = 40, pageGeo = new THREE.PlaneGeometry(0.2, 0.17);
  var pageMap = canvasTex(64, 54, function (x) {
    x.fillStyle = '#fffdf6'; x.fillRect(0, 0, 64, 54);
    x.fillStyle = '#ffffff'; x.fillRect(0, 0, 64, 12);
    x.fillStyle = 'rgba(0,0,0,0.25)';
    for (var i = 0; i < 12; i++) x.fillRect(4 + (i % 4) * 15, 16 + Math.floor(i / 4) * 12, 11, 7);
  });
  var pages = new THREE.InstancedMesh(pageGeo, new THREE.MeshLambertMaterial({ map: pageMap, side: THREE.DoubleSide }), NPAGE);
  var pageData = [], pageCols = ['#ff9a8a', '#8fc3ff', '#ffe07a', '#9fe39f', '#d2b3ff', '#ffbf8a'];
  for (var pk = 0; pk < NPAGE; pk++) {
    pageData.push({ a: r() * 6.28, rad: 0.7 + r() * 1.3, h: 0.5 + r() * 2.1, sp: 0.7 + r() * 0.8, tum: r() * 6.28,
                    fx: 0.6 + r() * 5.2, fz: -8.3 + r() * 4.5, fr: r() * 6.28 });
    pages.setColorAt(pk, tc.set(pageCols[pk % pageCols.length]));
  }
  pages.frustumCulled = false;
  kitchen.add(pages);

  // ── Living room: the cushion fort and the tent ─────────────────────────
  var living = new THREE.Group();
  house.add(living);
  var ll = [];
  put(ll, rbox(2.9, 0.42, 0.95, 0.1), '#4fb3a9', -3.6, F0 + 0.21, -8.3);
  put(ll, rbox(2.9, 0.72, 0.3, 0.12, 3), '#4fb3a9', -3.6, F0 + 0.75, -8.62);
  put(ll, rbox(0.3, 0.62, 0.95, 0.12, 3), '#46a39a', -5.0, F0 + 0.42, -8.3);
  put(ll, rbox(0.3, 0.62, 0.95, 0.12, 3), '#46a39a', -2.2, F0 + 0.42, -8.3);
  // Cushions stood on edge for walls, chairs at the front holding the roof
  [[-4.85, -7.25, 0], [-4.85, -6.45, 0], [-2.35, -7.25, 0], [-2.35, -6.45, 0]].forEach(function (c, k) {
    put(ll, rbox(0.2, 0.7, 0.78, 0.1, 2), k % 2 ? '#ffb347' : '#ff7aa2', c[0], F0 + 0.35, c[1], 0, 0, (k < 2 ? -0.08 : 0.08));
  });
  [[-4.85, -5.55], [-2.35, -5.55]].forEach(function (c) {
    put(ll, rbox(0.4, 0.05, 0.4, 0.03), '#e7b67c', c[0], F0 + 0.45, c[1]);
    put(ll, rbox(0.4, 0.5, 0.05, 0.03), '#e7b67c', c[0], F0 + 0.72, c[1] + 0.18);
    [[-0.17, -0.17], [0.17, -0.17], [-0.17, 0.17], [0.17, 0.17]].forEach(function (l) { put(ll, cyl(0.02, 0.02, 0.44, 6), '#cf9a5c', c[0] + l[0], F0 + 0.22, c[1] + l[1]); });
  });
  // Inside: pillows, a torch, a book
  put(ll, rbox(0.7, 0.16, 0.45, 0.08, 2), '#ffffff', -3.95, F0 + 0.08, -7.4, 0, 0.2);
  put(ll, rbox(0.65, 0.16, 0.45, 0.08, 2), '#bfe0ff', -3.2, F0 + 0.08, -7.3, 0, -0.2);
  put(ll, rbox(1.6, 0.05, 1.5, 0.03, 2), '#ffe08a', -3.6, F0 + 0.025, -6.7);
  put(ll, cyl(0.035, 0.03, 0.18, 10), '#5aa9ff', -3.3, F0 + 0.08, -6.3, 0, 0.6, Math.PI / 2);
  put(ll, rbox(0.22, 0.03, 0.16, 0.01, 2), '#e5484d', -3.9, F0 + 0.07, -6.2, 0, 0.4);
  // The room: rug, floor lamp, pictures, the tent's poles
  put(ll, cyl(1.5, 1.5, 0.015, 32), '#ffd6a5', -3.0, F0 + 0.01, -4.8);
  put(ll, cyl(0.015, 0.02, 1.5, 6), '#ffffff', -5.9, F0 + 0.75, -6.4);
  put(ll, cyl(0.15, 0.24, 0.3, 14, 1, true), '#ffe3a3', -5.9, F0 + 1.6, -6.4);
  put(ll, cyl(0.2, 0.2, 0.03, 14), '#ffffff', -5.9, F0 + 0.015, -6.4);
  mesh(ll, toy, living, true);
  function drawing(paint) { return new THREE.MeshLambertMaterial({ map: canvasTex(128, 100, function (x) { x.fillStyle = '#fffdf6'; x.fillRect(0, 0, 128, 100); x.lineWidth = 5; x.lineCap = 'round'; paint(x); }) }); }
  var pic1 = new THREE.Mesh(new THREE.PlaneGeometry(0.55, 0.43), drawing(function (x) {
    x.fillStyle = '#ffc21a'; x.beginPath(); x.arc(96, 28, 14, 0, Math.PI * 2); x.fill();
    x.fillStyle = '#ff7a5a'; x.fillRect(26, 50, 40, 32); x.fillStyle = '#e5484d'; x.beginPath(); x.moveTo(20, 52); x.lineTo(46, 28); x.lineTo(72, 52); x.fill();
    x.strokeStyle = '#4fb34f'; x.beginPath(); x.moveTo(0, 88); x.lineTo(128, 88); x.stroke();
  }));
  pic1.position.set(-4.3, F0 + 1.75, -8.79);
  var pic2 = new THREE.Mesh(new THREE.PlaneGeometry(0.5, 0.4), drawing(function (x) {
    x.strokeStyle = '#5aa9ff'; x.beginPath(); x.moveTo(10, 70); x.quadraticCurveTo(64, 20, 118, 70); x.stroke();
    x.strokeStyle = '#ff7aa2'; x.beginPath(); x.moveTo(10, 80); x.quadraticCurveTo(64, 30, 118, 80); x.stroke();
    x.strokeStyle = '#ffd23a'; x.beginPath(); x.moveTo(10, 90); x.quadraticCurveTo(64, 40, 118, 90); x.stroke();
  }));
  pic2.position.set(-2.9, F0 + 1.85, -8.79);
  living.add(pic1, pic2);
  // The blanket roof: draped from the sofa back over the chairs.
  var quilt = canvasTex(256, 256, function (x) {
    var cols = ['#ff9bb5', '#ffd27a', '#9fd8ff', '#b7e59a', '#d4b8ff', '#ffb38a'];
    for (var i = 0; i < 8; i++) for (var j = 0; j < 8; j++) {
      x.fillStyle = cols[(i * 3 + j * 5) % cols.length]; x.fillRect(i * 32, j * 32, 32, 32);
      x.fillStyle = 'rgba(255,255,255,0.35)'; if ((i + j) % 2) { x.beginPath(); x.arc(i * 32 + 16, j * 32 + 16, 7, 0, Math.PI * 2); x.fill(); }
    }
    x.strokeStyle = 'rgba(255,255,255,0.6)'; x.setLineDash([4, 4]); x.lineWidth = 2;
    for (var k = 0; k <= 8; k++) { x.beginPath(); x.moveTo(k * 32, 0); x.lineTo(k * 32, 256); x.stroke(); x.beginPath(); x.moveTo(0, k * 32); x.lineTo(256, k * 32); x.stroke(); }
  });
  var blanketGeo = new THREE.PlaneGeometry(2.9, 2.9, 20, 20).rotateX(-Math.PI / 2), bp = blanketGeo.attributes.position;
  for (var bi = 0; bi < bp.count; bi++) {
    var bu = (bp.getX(bi) + 1.45) / 2.9, bv = (bp.getZ(bi) + 1.45) / 2.9;     // u across, v from back to front
    var by = lerp(F0 + 1.12, F0 + 0.98, bv) - 0.16 * Math.sin(Math.PI * bu) * Math.sin(Math.PI * Math.min(bv * 1.1, 1));
    by -= (smooth(0.12, 0, bu) + smooth(0.88, 1, bu)) * 0.62;
    by -= smooth(0.92, 1, bv) * 0.18 * Math.abs(Math.sin(bu * Math.PI * 3));
    bp.setXYZ(bi, -3.6 + (bu - 0.5) * 2.65 * (1 + (smooth(0.12, 0, bu) + smooth(0.88, 1, bu)) * 0.05), by, lerp(-8.4, -5.35, bv));
  }
  blanketGeo.computeVertexNormals();
  var blanket = new THREE.Mesh(blanketGeo, new THREE.MeshLambertMaterial({ map: quilt, side: THREE.DoubleSide }));
  blanket.castShadow = !small;
  living.add(blanket);
  // Fairy lights along the blanket's front edge
  var fairyCols = ['#ffd56a', '#ff8fa8', '#8fd8ff', '#b6f08a', '#ffb070'], fairy = [];
  for (var fl = 0; fl < 15; fl++) {
    var fu = fl / 14, fxp = -3.6 + (fu - 0.5) * 2.7, fyp = F0 + 0.92 - 0.12 * Math.sin(Math.PI * fu) - Math.abs(Math.sin(fu * Math.PI * 3)) * 0.06;
    var fb = new THREE.Mesh(sph(0.028, 8, 6), basic(fairyCols[fl % fairyCols.length]));
    fb.position.set(fxp, fyp, -5.3);
    living.add(fb);
    fairy.push(fb);
  }
  var fairyGlow = new THREE.Points(new THREE.BufferGeometry().setFromPoints(fairy.map(function (f) { return f.position; })),
    new THREE.PointsMaterial({ map: glowTex, color: '#ffcf8a', size: 0.35, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
  living.add(fairyGlow);
  var fortGlow = glowSprite('#ffb867', 2.2, living, -3.6, F0 + 0.5, -6.8);
  var floorLampGlow = glowSprite('#ffd9a0', 1.4, living, -5.9, F0 + 1.55, -6.4);
  // The tent: a striped cone that glows from inside
  var stripes = canvasTex(256, 64, function (x) {
    for (var i = 0; i < 12; i++) { x.fillStyle = i % 2 ? '#fff4dc' : '#ff8a6a'; x.fillRect(i * 256 / 12, 0, 256 / 12, 64); }
    x.fillStyle = '#ffd690'; x.beginPath(); x.moveTo(40, 64); x.lineTo(64, 18); x.lineTo(88, 64); x.fill();
  });
  var tentMat = new THREE.MeshLambertMaterial({ map: stripes, side: THREE.DoubleSide, emissive: '#ffb060', emissiveIntensity: 0 });
  var tent = new THREE.Mesh(new THREE.ConeGeometry(0.75, 1.75, 12, 1, true), tentMat);
  tent.position.set(-1.05, F0 + 0.875, -7.6);
  tent.rotation.y = -1.9;
  living.add(tent);
  var tentTip = new THREE.Mesh(cyl(0.01, 0.01, 0.35, 5), basic('#c98c55'));
  tentTip.position.set(-1.05, F0 + 1.85, -7.6);
  living.add(tentTip);
  var tentFlag = new THREE.Mesh(new THREE.PlaneGeometry(0.16, 0.1).translate(0.08, 0, 0), new THREE.MeshLambertMaterial({ color: '#ffd23a', side: THREE.DoubleSide }));
  tentFlag.position.set(-1.05, F0 + 1.97, -7.6);
  living.add(tentFlag);
  var tentGlow = glowSprite('#ffb060', 2.4, living, -1.05, F0 + 0.6, -7.6);

  // ── The garden: fence and gate, path and puddles, things left out ──────
  var garden = new THREE.Group();
  island.add(garden);
  var gl2 = [];
  put(gl2, box(1.3, 0.02, GATE_Z - FRONT + 0.5), '#eadcb9', 0, 0.012, (GATE_Z + FRONT) / 2);
  for (var stp = 0; stp < 9; stp++) put(gl2, cyl(0.28, 0.3, 0.05, 9), '#d8d0c4', (stp % 2 ? 0.18 : -0.18), 0.03, FRONT + 1.2 + stp * 2.3, 0, stp);
  var picket = merge([tinted(box(0.09, 0.9, 0.035).translate(0, 0.45, 0), '#ffffff'), tinted(new THREE.ConeGeometry(0.064, 0.1, 4).rotateY(Math.PI / 4).translate(0, 0.95, 0), '#ffffff')]);
  var picketAt = [];
  for (var px = -FENCE_X; px <= FENCE_X; px += 0.22) if (Math.abs(px) > 0.85) picketAt.push(px, GATE_Z, 0);
  for (var pz = FRONT + 0.2; pz < GATE_Z; pz += 0.22) { picketAt.push(-FENCE_X, pz, 1); picketAt.push(FENCE_X, pz, 1); }
  var pickets = new THREE.InstancedMesh(picket, toy, picketAt.length / 3);
  scatter(pickets, picketAt.length / 3, function (n, p, q) { p.set(picketAt[n * 3], 0, picketAt[n * 3 + 1]); q.setFromAxisAngle(UP, picketAt[n * 3 + 2] ? Math.PI / 2 : 0); });
  pickets.castShadow = !small;
  garden.add(pickets);
  [[-FENCE_X, 0.55, GATE_Z / 2 + FRONT / 2, 'z'], [FENCE_X, 0.55, GATE_Z / 2 + FRONT / 2, 'z']].forEach(function (rl) {
    put(gl2, box(0.03, 0.06, GATE_Z - FRONT), '#f4f0e8', rl[0], 0.3, rl[2]); put(gl2, box(0.03, 0.06, GATE_Z - FRONT), '#f4f0e8', rl[0], 0.68, rl[2]);
  });
  [[-1, 1]].forEach(function () {
    put(gl2, box(FENCE_X - 0.85, 0.06, 0.03), '#f4f0e8', -(FENCE_X + 0.85) / 2, 0.3, GATE_Z - 0.03); put(gl2, box(FENCE_X - 0.85, 0.06, 0.03), '#f4f0e8', -(FENCE_X + 0.85) / 2, 0.68, GATE_Z - 0.03);
    put(gl2, box(FENCE_X - 0.85, 0.06, 0.03), '#f4f0e8', (FENCE_X + 0.85) / 2, 0.3, GATE_Z - 0.03); put(gl2, box(FENCE_X - 0.85, 0.06, 0.03), '#f4f0e8', (FENCE_X + 0.85) / 2, 0.68, GATE_Z - 0.03);
  });
  [-0.85, 0.85].forEach(function (x) { put(gl2, rbox(0.16, 1.25, 0.16, 0.03), '#ffffff', x, 0.62, GATE_Z); put(gl2, sph(0.1, 10, 8), '#ffffff', x, 1.3, GATE_Z); });
  // The apple tree in the corner, with a bench under it
  put(gl2, cyl(0.2, 0.32, 2.6, 8), '#a06a40', -9.5, 1.3, 14);
  [[0, 3.4, 0, 1.7], [-1.0, 3.0, 0.4, 1.2], [1.0, 3.1, -0.3, 1.25], [0.2, 3.0, 1.0, 1.15], [-0.3, 4.2, -0.4, 1.1]].forEach(function (c) {
    put(gl2, new THREE.IcosahedronGeometry(c[3], 1), '#5fb04a', -9.5 + c[0], c[1], 14 + c[2]);
  });
  for (var ap = 0; ap < 16; ap++) {
    var aa = r() * 6.28, ae = r() * 1.2 - 0.3, ar = 1.8;
    put(gl2, sph(0.09, 8, 6), '#e5484d', -9.5 + Math.cos(aa) * Math.cos(ae) * ar, 3.4 + Math.sin(ae) * ar * 0.8, 14 + Math.sin(aa) * Math.cos(ae) * ar);
  }
  // The dandelion stump-table
  put(gl2, cyl(0.42, 0.48, 0.5, 14), '#b07a4a', JAR.x, 0.25, JAR.z);
  put(gl2, cyl(0.43, 0.43, 0.04, 14), '#e8c48a', JAR.x, 0.51, JAR.z);
  // Sandals kicked off in the grass
  [[LAWN.x - 0.05, LAWN.z - 0.15, 0.4, 0], [LAWN.x + 0.22, LAWN.z + 0.12, -0.5, Math.PI]].forEach(function (s) {
    put(gl2, rbox(0.1, 0.02, 0.22, 0.01, 2), '#ff8fb0', s[0], 0.02, s[1], s[3], s[2]);
    put(gl2, new THREE.TorusGeometry(0.045, 0.01, 5, 12, Math.PI).rotateY(Math.PI / 2), '#ff5f8a', s[0], 0.025, s[1], s[3] ? Math.PI : 0, s[2]);
  });
  // Daisies near the sandals
  for (var dz2 = 0; dz2 < 18; dz2++) {
    var ddx = LAWN.x + (r() - 0.5) * 2.4, ddz = LAWN.z + (r() - 0.5) * 2.4;
    put(gl2, cyl(0.004, 0.004, 0.14, 4), '#3f8a3a', ddx, 0.07, ddz);
    for (var dpet = 0; dpet < 8; dpet++) { var dpa = dpet / 8 * Math.PI * 2; put(gl2, box(0.03, 0.004, 0.01), '#ffffff', ddx + Math.cos(dpa) * 0.018, 0.14, ddz + Math.sin(dpa) * 0.018, 0, -dpa); }
    put(gl2, sph(0.009, 6, 4), '#ffd23a', ddx, 0.142, ddz);
  }
  // The tea party table and chairs, a pole for the bunting
  put(gl2, cyl(0.06, 0.1, 0.5, 10), '#ffffff', TEA.x, 0.25, TEA.z);
  [-0.72, 0.72].forEach(function (dz) {
    var cz = TEA.z + dz;
    put(gl2, rbox(0.36, 0.05, 0.36, 0.03), '#ff9bb5', TEA.x, 0.32, cz);
    put(gl2, rbox(0.36, 0.42, 0.05, 0.03), '#ff9bb5', TEA.x, 0.55, cz + (dz < 0 ? -0.17 : 0.17));
    [[-0.15, -0.15], [0.15, -0.15], [-0.15, 0.15], [0.15, 0.15]].forEach(function (l) { put(gl2, cyl(0.018, 0.018, 0.3, 6), '#ffffff', TEA.x + l[0], 0.15, cz + l[1]); });
  });
  put(gl2, cyl(0.04, 0.05, 2.6, 8), '#ffffff', -3.6, 1.3, 3.4);
  addTeddy(gl2, TEA.x, 0.36, TEA.z + 0.68, Math.PI / 2, 1);
  mesh(gl2, toy, garden, true);
  var cloth = new THREE.Mesh(cyl(0.48, 0.52, 0.12, 24, 1, true), new THREE.MeshLambertMaterial({ side: THREE.DoubleSide, map: canvasTex(128, 32, function (x) {
    for (var i = 0; i < 16; i++) for (var j = 0; j < 4; j++) { x.fillStyle = (i + j) % 2 ? '#ffffff' : '#ff6b6b'; x.fillRect(i * 8, j * 8, 8, 8); }
  }) }));
  cloth.position.set(TEA.x, 0.46, TEA.z);
  var clothTop = new THREE.Mesh(new THREE.CircleGeometry(0.48, 24).rotateX(-Math.PI / 2), new THREE.MeshLambertMaterial({ map: canvasTex(128, 128, function (x) {
    for (var i = 0; i < 12; i++) for (var j = 0; j < 12; j++) { x.fillStyle = (i + j) % 2 ? '#ffffff' : '#ff6b6b'; x.fillRect(i * 11, j * 11, 11, 11); }
  }) }));
  clothTop.position.set(TEA.x, 0.521, TEA.z);
  garden.add(cloth, clothTop);
  // Cups, plates, cake and the teapot that pours
  var tea = new THREE.Group();
  tea.position.copy(TEA);
  garden.add(tea);
  var tel = [];
  [-1, 1].forEach(function (side) {
    put(tel, cyl(0.09, 0.08, 0.01, 18), '#ffffff', -0.12, 0.53, side * 0.3);
    put(tel, cyl(0.04, 0.03, 0.05, 14, 1, true), '#9fd8ff', 0.14, 0.555, side * 0.27);
    put(tel, cyl(0.03, 0.03, 0.004, 14), '#9fd8ff', 0.14, 0.531, side * 0.27);
    put(tel, new THREE.TorusGeometry(0.016, 0.004, 4, 10), '#9fd8ff', 0.185, 0.56, side * 0.27);
  });
  put(tel, cyl(0.12, 0.12, 0.01, 20), '#ffffff', -0.2, 0.535, 0);
  put(tel, cyl(0.02, 0.03, 0.06, 10), '#ffffff', -0.2, 0.56, 0);
  put(tel, cyl(0.14, 0.14, 0.01, 22), '#ffffff', -0.2, 0.595, 0);
  tea.add(new THREE.Mesh(merge(tel), toy));
  var cake = new THREE.Mesh(merge([tinted(new THREE.CylinderGeometry(0.11, 0.11, 0.1, 24, 1, false, 0.5, Math.PI * 2 - 0.9).translate(-0.2, 0.65, 0), '#ffd6e2'),
                                   tinted(new THREE.CylinderGeometry(0.112, 0.112, 0.02, 24, 1, false, 0.5, Math.PI * 2 - 0.9).translate(-0.2, 0.7, 0), '#ff8fb0'),
                                   tinted(sph(0.018, 8, 6).translate(-0.22, 0.73, -0.02), '#e5484d')]), toy);
  tea.add(cake);
  var slice = new THREE.Mesh(merge([tinted(new THREE.CylinderGeometry(0.11, 0.11, 0.1, 6, 1, false, -0.4, 0.9), '#ffd6e2'),
                                    tinted(new THREE.CylinderGeometry(0.112, 0.112, 0.02, 6, 1, false, -0.4, 0.9).translate(0, 0.05, 0), '#ff8fb0')]), toy);
  tea.add(slice);
  var teapot = new THREE.Group(), tpl = [];
  put(tpl, sph(0.08, 16, 12), '#7fd3c4', 0, 0, 0, 0, 0, 0, 1, 0.85, 1);
  put(tpl, cyl(0.04, 0.05, 0.03, 12), '#7fd3c4', 0, 0.07, 0);
  put(tpl, sph(0.015, 8, 6), '#ffffff', 0, 0.095, 0);
  put(tpl, cyl(0.012, 0.02, 0.11, 8), '#7fd3c4', 0.09, 0.02, 0, 0, 0, -0.9);
  put(tpl, new THREE.TorusGeometry(0.04, 0.01, 6, 12, Math.PI), '#7fd3c4', -0.08, 0.01, 0, 0, 0, Math.PI / 2);
  put(tpl, sph(0.06, 10, 6), '#ffffff', 0, 0.0, 0.0, 0, 0, 0, 1.02, 0.3, 1.02);
  teapot.add(new THREE.Mesh(merge(tpl), toy));
  teapot.position.set(0.16, 0.62, 0.02);
  tea.add(teapot);
  var stream = new THREE.Mesh(cyl(0.006, 0.008, 1, 6).translate(0, -0.5, 0), new THREE.MeshStandardMaterial({ color: '#c98a4a', roughness: 0.2, transparent: true, opacity: 0.85 }));
  tea.add(stream);
  var teaLevel = new THREE.Mesh(new THREE.CircleGeometry(0.035, 14).rotateX(-Math.PI / 2), basic('#b8783a'));
  teaLevel.position.set(0.14, 0.535, 0.27);
  tea.add(teaLevel);
  // Bunting from the apple tree to the pole
  var buntA = new THREE.Vector3(-9.1, 2.9, 13.2), buntB = new THREE.Vector3(-3.6, 2.55, 3.4), bunt = [];
  for (var bk = 0; bk < 15; bk++) {
    var bt = (bk + 0.5) / 15;
    v1.lerpVectors(buntA, buntB, bt);
    v1.y -= Math.sin(bt * Math.PI) * 0.55;
    var tri = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-0.12, 0, 0), new THREE.Vector3(0.12, 0, 0), new THREE.Vector3(0, -0.24, 0)]);
    tri.computeVertexNormals();
    var flag = new THREE.Mesh(tri, new THREE.MeshLambertMaterial({ color: flowerCols[bk % flowerCols.length], side: THREE.DoubleSide }));
    flag.position.copy(v1);
    flag.rotation.y = Math.atan2(buntB.x - buntA.x, buntB.z - buntA.z) + Math.PI / 2;
    garden.add(flag);
    bunt.push(flag);
  }
  var buntLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(bunt.map(function (b) { return b.position; })), new THREE.LineBasicMaterial({ color: '#ffffff' }));
  garden.add(buntLine);
  // The heart balloon tied to the teddy's chair
  var balloon = new THREE.Mesh(puffy(heartShape(0.16), 0.08, 0.05), new THREE.MeshStandardMaterial({ color: '#ff4f7a', roughness: 0.25 }));
  garden.add(balloon);
  var balloonStr = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), new THREE.LineBasicMaterial({ color: '#ffffff' }));
  balloonStr.frustumCulled = false;
  garden.add(balloonStr);
  var balloonTie = new THREE.Vector3(TEA.x + 0.16, 0.76, TEA.z + 0.89);
  // The umbrella that twirls in the rain, by a pair of wellies
  var rainGear = new THREE.Group(), wl = [];
  garden.add(rainGear);
  [[-1.0, 7.3, 0.2], [-1.25, 7.15, 0.5]].forEach(function (w) {
    put(wl, cyl(0.07, 0.075, 0.32, 12), '#ffd23a', w[0], 0.18, w[1]);
    put(wl, rbox(0.15, 0.08, 0.26, 0.04), '#ffd23a', w[0], 0.04, w[1] + 0.05, 0, w[2]);
    put(wl, cyl(0.075, 0.075, 0.03, 12), '#f2b81a', w[0], 0.33, w[1]);
  });
  mesh(wl, toy, rainGear, true);
  var umbrella = new THREE.Group();
  umbrella.position.set(1.65, 0.33, 7.7);
  umbrella.rotation.set(0.9, 0.3, 0.5);
  rainGear.add(umbrella);
  var umbSpin = new THREE.Group();
  umbrella.add(umbSpin);
  umbSpin.add(new THREE.Mesh(new THREE.ConeGeometry(0.62, 0.28, 8, 1, true).translate(0, 0.14, 0), new THREE.MeshStandardMaterial({ color: '#ff4f5a', roughness: 0.4, side: THREE.DoubleSide })));
  var ul = [];
  put(ul, cyl(0.012, 0.012, 0.8, 6), '#d8d8d8', 0, -0.25, 0);
  put(ul, new THREE.TorusGeometry(0.06, 0.016, 6, 12, Math.PI), '#7a4a2a', 0.06, -0.65, 0, 0, 0, Math.PI);
  put(ul, sph(0.025, 8, 6), '#ffffff', 0, 0.3, 0);
  umbSpin.add(new THREE.Mesh(merge(ul), toy));
  // Puddles on the path
  var puddles = [[0.1, 6.8, 0.75, 1.3], [-0.35, 8.7, 0.5, 2.2], [0.45, 10.3, 0.58, 3.7], [-0.6, 5.5, 0.4, 4.4]].map(function (pd) {
    var m = new THREE.Mesh(new THREE.PlaneGeometry(pd[2] * 2, pd[2] * 2).rotateX(-Math.PI / 2), new THREE.ShaderMaterial({
      transparent: true, depthWrite: false,
      uniforms: { uTime: { value: 0 }, uRain: { value: 0 }, uSky: { value: new THREE.Color('#cfe3f0') }, uSeed: { value: pd[3] } },
      vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
      fragmentShader: PUDDLE_FS
    }));
    m.scale.set(1, 1, 1.3);
    m.position.set(pd[0], 0.03, pd[1]);
    garden.add(m);
    return m;
  });
  // The dandelions in their jam jar: yellow heads that become clocks.
  var jarGlass = new THREE.Mesh(new THREE.LatheGeometry([[0, 0], [0.08, 0], [0.09, 0.02], [0.09, 0.15], [0.075, 0.18], [0.075, 0.2]].map(function (p) { return new THREE.Vector2(p[0], p[1]); }), 20),
    new THREE.MeshStandardMaterial({ color: '#dff4ff', transparent: true, opacity: 0.32, roughness: 0.05, depthWrite: false }));
  jarGlass.position.copy(JAR);
  garden.add(jarGlass);
  var jarWater = new THREE.Mesh(cyl(0.083, 0.078, 0.12, 18), new THREE.MeshStandardMaterial({ color: '#bfe8ff', transparent: true, opacity: 0.35, roughness: 0.1 }));
  jarWater.position.set(JAR.x, JAR.y + 0.07, JAR.z);
  garden.add(jarWater);
  var bow = new THREE.Mesh(new THREE.TorusGeometry(0.084, 0.012, 6, 24).rotateX(Math.PI / 2), basic('#ff7aa2'));
  bow.position.set(JAR.x, JAR.y + 0.16, JAR.z);
  garden.add(bow);
  var heads = [], stemL = [], seedPos = [], seedHead = [], seedRand = [], SPH = small ? 45 : 80;
  for (var hk = 0; hk < 9; hk++) {
    var ha = hk / 9 * Math.PI * 2 + r() * 0.4, hrad = hk === 0 ? 0 : 0.07 + r() * 0.08, hy = 0.34 + r() * 0.16 + (hk === 0 ? 0.08 : 0);
    var head = new THREE.Vector3(JAR.x + Math.cos(ha) * hrad * 1.6, JAR.y + hy, JAR.z + Math.sin(ha) * hrad * 1.6);
    var base = new THREE.Vector3(JAR.x + Math.cos(ha) * 0.03, JAR.y + 0.05, JAR.z + Math.sin(ha) * 0.03);
    var curve = new THREE.QuadraticBezierCurve3(base, new THREE.Vector3(lerp(base.x, head.x, 0.3), lerp(base.y, head.y, 0.6), lerp(base.z, head.z, 0.3)), head);
    stemL.push(tinted(new THREE.TubeGeometry(curve, 8, 0.004, 4, false), '#4f9a3a'));
    heads.push(head);
    for (var sd = 0; sd < SPH; sd++) {
      var sy = r() * 2 - 1, sa = r() * 6.28, sr = Math.sqrt(1 - sy * sy);
      seedPos.push(sr * Math.cos(sa), sy, sr * Math.sin(sa));
      seedHead.push(head.x, head.y, head.z);
      seedRand.push(r(), r(), r());
    }
  }
  garden.add(new THREE.Mesh(merge(stemL), toy));
  var headGeo = merge((function () {
    var l = [];
    put(l, sph(0.042, 10, 6), '#ffc21a', 0, 0, 0, 0, 0, 0, 1, 0.55, 1);
    for (var k = 0; k < 18; k++) { var a = k / 18 * Math.PI * 2; put(l, box(0.05, 0.008, 0.016), '#ffd23a', Math.cos(a) * 0.05, 0.004, Math.sin(a) * 0.05, 0, -a, 0.25); }
    for (k = 0; k < 12; k++) { var b = k / 12 * Math.PI * 2 + 0.3; put(l, box(0.034, 0.008, 0.014), '#ffe14a', Math.cos(b) * 0.032, 0.016, Math.sin(b) * 0.032, 0, -b, 0.5); }
    put(l, sph(0.03, 8, 6), '#7fae4a', 0, -0.024, 0, 0, 0, 0, 1, 0.6, 1);
    return l;
  })());
  var yellow = new THREE.InstancedMesh(headGeo, toy, heads.length);
  garden.add(yellow);
  var seedGeo = new THREE.BufferGeometry();
  seedGeo.setAttribute('position', new THREE.Float32BufferAttribute(seedPos, 3));
  seedGeo.setAttribute('aHead', new THREE.Float32BufferAttribute(seedHead, 3));
  seedGeo.setAttribute('aRand', new THREE.Float32BufferAttribute(seedRand, 3));
  var seedMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false,
    uniforms: { uGrow: { value: 0 }, uBlow: { value: 0 }, uTime: { value: 0 }, uPx: pxScale, uWind: { value: new THREE.Vector3(-0.9, 0.0, 0.5) } },
    vertexShader: SEED_VS, fragmentShader: SEED_FS
  });
  var seeds = new THREE.Points(seedGeo, seedMat);
  seeds.frustumCulled = false;
  garden.add(seeds);
  // Butterflies about the garden, and a ladybird climbing a blade
  var wingGeo = new THREE.CircleGeometry(0.06, 10).scale(1, 0.75, 1).translate(0.055, 0, 0);
  var butterflies = [['#ffb347', 0.2], ['#9fd8ff', 2.1], ['#ff8fb0', 4.0]].map(function (b) {
    var g = new THREE.Group(), mat = new THREE.MeshLambertMaterial({ color: b[0], side: THREE.DoubleSide });
    var w1 = new THREE.Mesh(wingGeo, mat), w2 = new THREE.Mesh(wingGeo, mat);
    w2.scale.x = -1;
    var body = new THREE.Mesh(cyl(0.008, 0.006, 0.08, 5).rotateX(Math.PI / 2), basic('#3a2a2a'));
    g.add(w1, w2, body);
    garden.add(g);
    return { g: g, w1: w1, w2: w2, ph: b[1] };
  });
  var lady = new THREE.Group(), ldl = [];
  put(ldl, sph(0.012, 10, 6, 0, Math.PI * 2, 0, Math.PI / 2), '#e5303a', 0, 0, 0, 0, 0, 0, 1, 0.8, 1.2);
  put(ldl, sph(0.006, 6, 5), '#1a1a1a', 0, 0.002, 0.012);
  [[0.005, 0.008, 0.002], [-0.005, 0.008, 0.002], [0.006, 0.006, -0.006], [-0.006, 0.006, -0.006]].forEach(function (d) { put(ldl, sph(0.0026, 5, 4), '#1a1a1a', d[0], d[1], d[2]); });
  lady.add(new THREE.Mesh(merge(ldl), toy));
  garden.add(lady);
  var ladyBlade = new THREE.Mesh(new THREE.ConeGeometry(0.008, 0.42, 4).translate(0, 0.21, 0), new THREE.MeshLambertMaterial({ color: '#7fc24a' }));
  ladyBlade.position.set(LAWN.x - 0.12, 0, LAWN.z - 0.55);
  ladyBlade.rotation.z = 0.12;
  garden.add(ladyBlade);
  // Fireflies keeping watch at night
  var FF = 70, ffPos = new Float32Array(FF * 3), ffSeed = [];
  for (var fk = 0; fk < FF; fk++) ffSeed.push((r() - 0.5) * 22, 0.3 + r() * 2.2, FRONT + 1 + r() * 18, r() * 6.28);
  var ffGeo = new THREE.BufferGeometry();
  ffGeo.setAttribute('position', new THREE.BufferAttribute(ffPos, 3));
  var fireflies = new THREE.Points(ffGeo, new THREE.PointsMaterial({ map: glowTex, color: '#e8ff9a', size: 0.32, transparent: true, opacity: 0,
    blending: THREE.AdditiveBlending, depthWrite: false }));
  fireflies.frustumCulled = false;
  garden.add(fireflies);
  // The gate, which swings open towards the neighbour
  var gate = new THREE.Group();
  gate.position.set(0.78, 0, GATE_Z);
  garden.add(gate);
  var gtl = [];
  for (var gp = 0; gp < 7; gp++) put(gtl, box(0.08, 0.85, 0.03), '#ffffff', -0.11 - gp * 0.215, 0.5, 0);
  put(gtl, box(1.5, 0.06, 0.035), '#f4f0e8', -0.76, 0.3, 0.01);
  put(gtl, box(1.5, 0.06, 0.035), '#f4f0e8', -0.76, 0.7, 0.01);
  put(gtl, box(0.06, 0.6, 0.035), '#f4f0e8', -0.76, 0.5, 0.012, 0, 0, 0.95);
  gate.add(new THREE.Mesh(merge(gtl), toy));

  // ── The neighbour's cottage across the lane ────────────────────────────
  var neigh = new THREE.Group();
  neigh.position.copy(NEIGH);
  neigh.position.y = land(NEIGH.x, NEIGH.z);
  neigh.rotation.y = Math.PI;
  island.add(neigh);
  var nl = [];
  put(nl, rbox(4.2, 2.7, 3.6, 0.08), '#f6d2c4', 0, 1.35, 0);
  put(nl, new THREE.CylinderGeometry(0.01, 2.8, 1.8, 4, 1).rotateY(Math.PI / 4).scale(1.08, 1, 0.92), '#5d8fd0', 0, 3.6, 0);
  put(nl, rbox(0.9, 1.7, 0.1, 0.04), '#3f9f8f', 0, 0.95, 1.82);
  put(nl, sph(0.04, 8, 6), '#ffd23a', 0.3, 0.95, 1.9);
  put(nl, rbox(1.5, 0.22, 0.8, 0.04), '#d8d0c4', 0, 0.11, 2.15);
  put(nl, rbox(0.6, 0.3, 0.4, 0.05), '#ffffff', -1.6, 0.15, 2.2);
  for (var nf = 0; nf < 6; nf++) put(nl, sph(0.08, 8, 6), flowerCols[nf], -1.75 + (nf % 3) * 0.15, 0.35, 2.15 + Math.floor(nf / 3) * 0.12);
  put(nl, cyl(0.04, 0.04, 1.0, 6), '#ffffff', 1.9, 0.5, 2.6);
  put(nl, rbox(0.4, 0.3, 0.25, 0.08), '#e5484d', 1.9, 1.1, 2.6);
  // The basket of kindness on the step
  put(nl, cyl(0.17, 0.13, 0.15, 14), '#c98c55', 0.42, 0.3, 2.3);
  put(nl, new THREE.TorusGeometry(0.15, 0.015, 6, 16, Math.PI), '#b07a4a', 0.42, 0.37, 2.3);
  put(nl, sph(0.05, 8, 6), '#e5484d', 0.38, 0.39, 2.28); put(nl, sph(0.05, 8, 6), '#ffd23a', 0.48, 0.39, 2.33);
  put(nl, rbox(0.14, 0.06, 0.08, 0.03), '#f2c27a', 0.42, 0.4, 2.22, 0, 0.4);
  mesh(nl, toy, neigh, true);
  var neighWinMat = new THREE.MeshBasicMaterial({ color: '#43506a' });
  [[-1.25, 1.55], [1.25, 1.55]].forEach(function (w) {
    var win = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 0.7), neighWinMat);
    win.position.set(w[0], w[1], 1.81);
    neigh.add(win);
    var wf = new THREE.Mesh(merge([tinted(box(0.9, 0.06, 0.06).translate(0, 0.38, 0), '#ffffff'), tinted(box(0.9, 0.06, 0.06).translate(0, -0.38, 0), '#ffffff'),
                                   tinted(box(0.06, 0.8, 0.06).translate(0.43, 0, 0), '#ffffff'), tinted(box(0.06, 0.8, 0.06).translate(-0.43, 0, 0), '#ffffff'),
                                   tinted(box(0.04, 0.7, 0.04), '#ffffff')]), toy);
    wf.position.set(w[0], w[1], 1.83);
    neigh.add(wf);
  });
  var lanternCore = new THREE.Mesh(sph(0.05, 10, 8), basic('#5a4a3a'));
  lanternCore.position.set(-0.45, 0.38, 2.3);
  var lanternCage = new THREE.Mesh(merge([tinted(cyl(0.09, 0.09, 0.02, 8).translate(0, 0.1, 0), '#3a3a3a'), tinted(cyl(0.08, 0.08, 0.02, 8).translate(0, -0.1, 0), '#3a3a3a'),
                                          tinted(new THREE.TorusGeometry(0.04, 0.008, 4, 10).translate(0, 0.15, 0), '#3a3a3a'),
                                          tinted(cyl(0.006, 0.006, 0.2, 4).translate(0.07, 0, 0), '#3a3a3a'), tinted(cyl(0.006, 0.006, 0.2, 4).translate(-0.07, 0, 0), '#3a3a3a'),
                                          tinted(cyl(0.006, 0.006, 0.2, 4).translate(0, 0, 0.07), '#3a3a3a'), tinted(cyl(0.006, 0.006, 0.2, 4).translate(0, 0, -0.07), '#3a3a3a')]), toy);
  lanternCage.position.copy(lanternCore.position);
  var lanternGlow = glowSprite('#ffc278', 1.6, neigh, -0.45, 0.4, 2.3);
  neigh.add(lanternCore, lanternCage);
  neigh.updateMatrixWorld(true);
  lanternLight.position.copy(lanternCore.position).add(v1.set(0, 0.2, 0.3)).applyMatrix4(neigh.matrixWorld);

  // ── The beach: a sandcastle, a bucket that helps, shells, a pinwheel ───
  var beach = new THREE.Group();
  beach.position.set(CASTLE.x, land(CASTLE.x, CASTLE.z), CASTLE.z);
  beach.rotation.y = 0.55;
  island.add(beach);
  var SANDC = '#e8c98a';
  var bel = [];
  put(bel, cyl(0.62, 0.8, 0.12, 24), '#e2c184', 0, 0.02, 0);
  put(bel, new THREE.TorusGeometry(0.85, 0.07, 6, 28).rotateX(Math.PI / 2), '#d9b676', 0, 0.0, 0);
  for (var shl = 0; shl < 9; shl++) {
    var sa2 = r() * 6.28, sd2 = 1.2 + r() * 1.6;
    put(bel, new THREE.ConeGeometry(0.06, 0.02, 9, 1, false, 0, Math.PI), ['#ffd6e2', '#fff4e0', '#ffc7a8'][shl % 3], Math.cos(sa2) * sd2, 0.01, Math.sin(sa2) * sd2, 0, r() * 6.28);
  }
  put(bel, puffy(starShape(0.09, 0.04, 5), 0.02, 0.015), '#ff8a4a', 1.1, 0.03, 0.9, -Math.PI / 2, 0, 0.3);
  put(bel, puffy(starShape(0.07, 0.03, 5), 0.02, 0.012), '#ffb347', -1.5, 0.03, 1.2, -Math.PI / 2, 0, 1.0);
  put(bel, cyl(0.025, 0.025, 0.55, 6), '#5aa9ff', -1.0, 0.2, 0.55, 0.3, 0, 0.4);
  put(bel, rbox(0.15, 0.02, 0.18, 0.01, 2), '#3f8fe0', -1.12, 0.03, 0.42, -0.4, 0.4);
  // A striped towel and a beach parasol
  for (var tw = 0; tw < 6; tw++) put(bel, box(0.2, 0.01, 1.3), tw % 2 ? '#ffffff' : '#5ac8e0', 1.9 + tw * 0.2, 0.01, -0.8, 0, 0.2);
  put(bel, cyl(0.03, 0.03, 2.2, 6), '#ffffff', 2.8, 1.1, -1.6);
  for (var pr = 0; pr < 8; pr++) put(bel, new THREE.ConeGeometry(1.2, 0.5, 8, 1, true, pr / 8 * Math.PI * 2, Math.PI / 4), pr % 2 ? '#ffffff' : '#ff6b6b', 2.8, 2.25, -1.6);
  mesh(bel, toy2, beach, true);
  // Towers, rising one by one
  function tower(rad, h, x, z) {
    var l = [];
    put(l, cyl(rad * 0.9, rad, h, 16), SANDC, 0, h / 2, 0);
    for (var k = 0; k < 8; k++) { var a = k / 8 * Math.PI * 2; if (k % 2) put(l, box(rad * 0.45, rad * 0.4, rad * 0.3), SANDC, Math.cos(a) * rad * 0.8, h + rad * 0.15, Math.sin(a) * rad * 0.8, 0, -a); }
    put(l, cyl(rad * 0.15, rad * 0.15, 0.01, 8), '#c9a96a', rad * 0.92, h * 0.6, 0, 0, 0, Math.PI / 2);
    var m = new THREE.Mesh(merge(l), toy);
    m.position.set(x, 0.05, z);
    m.castShadow = !small;
    beach.add(m);
    return m;
  }
  var towers = [tower(0.16, 0.42, -0.42, -0.42), tower(0.16, 0.42, 0.42, -0.42), tower(0.16, 0.42, -0.42, 0.42), tower(0.16, 0.42, 0.42, 0.42)];
  var walls = [[0, -0.42, 0], [0, 0.42, 0], [-0.42, 0, Math.PI / 2], [0.42, 0, Math.PI / 2]].map(function (w) {
    var l = [];
    put(l, box(0.7, 0.28, 0.1), SANDC, 0, 0.14, 0);
    for (var k = 0; k < 5; k++) put(l, box(0.08, 0.06, 0.1), SANDC, -0.3 + k * 0.15, 0.31, 0);
    var m = new THREE.Mesh(merge(l), toy);
    m.position.set(w[0], 0.05, w[1]);
    m.rotation.y = w[2];
    beach.add(m);
    return m;
  });
  // The keep: bucket-shaped, ridged, under the upturned bucket
  var kpl = [];
  put(kpl, cyl(0.17, 0.22, 0.42, 20), SANDC, 0, 0.21, 0);
  for (var kr = 0; kr < 3; kr++) put(kpl, new THREE.TorusGeometry(0.2 - kr * 0.016, 0.01, 4, 20).rotateX(Math.PI / 2), '#dcbd7e', 0, 0.08 + kr * 0.13, 0);
  var keep = new THREE.Mesh(merge(kpl), toy);
  keep.position.y = 0.05;
  beach.add(keep);
  var bucket = new THREE.Group(), bul = [];
  put(bul, cyl(0.235, 0.19, 0.44, 20, 1, true), '#ff5f5a', 0, 0.22, 0);
  put(bul, cyl(0.19, 0.19, 0.01, 20), '#ff5f5a', 0, 0.005, 0);
  put(bul, new THREE.TorusGeometry(0.235, 0.012, 5, 20).rotateX(Math.PI / 2), '#e5484d', 0, 0.44, 0);
  put(bul, new THREE.TorusGeometry(0.24, 0.008, 4, 16, Math.PI), '#ffd23a', 0, 0.42, 0);
  bucket.add(new THREE.Mesh(merge(bul), toy2));
  beach.add(bucket);
  var castleFlag = new THREE.Group();
  castleFlag.add(new THREE.Mesh(cyl(0.006, 0.006, 0.3, 5).translate(0, 0.15, 0), basic('#8a6a4a')));
  var cfl = new THREE.Mesh(new THREE.PlaneGeometry(0.14, 0.09, 6, 1).translate(0.07, 0.25, 0), new THREE.MeshLambertMaterial({ color: '#ff4f7a', side: THREE.DoubleSide }));
  castleFlag.add(cfl);
  castleFlag.position.y = 0.47;
  beach.add(castleFlag);
  var pinwheel = new THREE.Group();
  pinwheel.position.set(0.95, 0.62, -0.75);
  beach.add(pinwheel);
  var pinStick = new THREE.Mesh(cyl(0.008, 0.008, 0.62, 5).translate(0, -0.31, 0), basic('#ffffff'));
  pinwheel.add(pinStick);
  var pinHead = new THREE.Group();
  pinHead.position.z = 0.02;
  pinwheel.add(pinHead);
  ['#ff5f8a', '#5ac8ff', '#ffd23a', '#7be07b'].forEach(function (c, k) {
    var g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0.12, 0, 0.02), new THREE.Vector3(0.12, 0.12, -0.02)]);
    g.computeVertexNormals();
    var b = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ color: c, side: THREE.DoubleSide }));
    b.rotation.z = k * Math.PI / 2;
    pinHead.add(b);
  });
  var ball = new THREE.Mesh(merge(['#ff5f5a', '#ffffff', '#5aa9ff', '#ffffff', '#ffd23a', '#ffffff'].map(function (c, k) {
    return tinted(new THREE.SphereGeometry(0.17, 6, 10, k / 6 * Math.PI * 2, Math.PI / 3), c);
  })), toy);
  ball.position.set(-0.9, 0.17, -0.9);
  beach.add(ball);

  // ── The race lawn: two wind-up cars and a finish tape ──────────────────
  var race = new THREE.Group();
  island.add(race);
  var rcl = [];
  [-0.9, 0, 0.9].forEach(function (dx) { put(rcl, box(0.05, 0.01, 17), '#ffffff', RACE_X + dx, 0.03, FINISH - 6.5); });
  put(rcl, box(1.85, 0.01, 0.06), '#ffffff', RACE_X, 0.03, FINISH - 13);
  [-1.25, 1.25].forEach(function (dx) {
    for (var k = 0; k < 6; k++) put(rcl, cyl(0.04, 0.04, 0.18, 8), k % 2 ? '#ffffff' : '#e5484d', RACE_X + dx, 0.09 + k * 0.18, FINISH);
    put(rcl, sph(0.06, 8, 6), '#ffd23a', RACE_X + dx, 1.17, FINISH);
  });
  mesh(rcl, toy, race, false);
  var checker = new THREE.Mesh(new THREE.PlaneGeometry(1.85, 0.3).rotateX(-Math.PI / 2), new THREE.MeshLambertMaterial({ map: canvasTex(128, 24, function (x) {
    for (var i = 0; i < 16; i++) for (var j = 0; j < 3; j++) { x.fillStyle = (i + j) % 2 ? '#ffffff' : '#222222'; x.fillRect(i * 8, j * 8, 8, 8); }
  }) }));
  checker.position.set(RACE_X, 0.035, FINISH);
  race.add(checker);
  var tapeMat = new THREE.MeshLambertMaterial({ color: '#ff4f5a', side: THREE.DoubleSide });
  var tapeL = new THREE.Group(), tapeR = new THREE.Group();
  tapeL.position.set(RACE_X - 1.25, 0.62, FINISH);
  tapeR.position.set(RACE_X + 1.25, 0.62, FINISH);
  tapeL.add(new THREE.Mesh(new THREE.PlaneGeometry(1.25, 0.05).translate(0.625, 0, 0), tapeMat));
  tapeR.add(new THREE.Mesh(new THREE.PlaneGeometry(1.25, 0.05).translate(-0.625, 0, 0), tapeMat));
  race.add(tapeL, tapeR);
  function windUpCar(body, roof) {
    var g = new THREE.Group(), l = [];
    put(l, rbox(0.42, 0.13, 0.22, 0.06), body, 0, 0.11, 0);
    put(l, rbox(0.22, 0.12, 0.19, 0.06), roof, -0.03, 0.22, 0);
    put(l, rbox(0.08, 0.08, 0.17, 0.03), '#bfe6ff', 0.07, 0.22, 0);
    put(l, sph(0.022, 8, 6), '#fff4b0', 0.21, 0.12, 0.07); put(l, sph(0.022, 8, 6), '#fff4b0', 0.21, 0.12, -0.07);
    put(l, new THREE.TorusGeometry(0.05, 0.01, 4, 10, Math.PI).rotateY(Math.PI / 2).rotateX(Math.PI), '#ffffff', 0.215, 0.085, 0);
    var wheels = [];
    [[0.13, 0.1], [-0.13, 0.1], [0.13, -0.1], [-0.13, -0.1]].forEach(function (w) {
      var wh = new THREE.Mesh(merge([tinted(cyl(0.055, 0.055, 0.04, 14).rotateX(Math.PI / 2), '#2b2f3a'), tinted(cyl(0.025, 0.025, 0.045, 10).rotateX(Math.PI / 2), '#ffffff')]), toy);
      wh.position.set(w[0], 0.055, w[1] * 1.15);
      g.add(wh);
      wheels.push(wh);
    });
    g.add(new THREE.Mesh(merge(l), toy));
    var key = new THREE.Group();
    key.add(new THREE.Mesh(merge([tinted(cyl(0.008, 0.008, 0.06, 5).translate(0, 0.03, 0), '#c9d2d8'),
                                  tinted(new THREE.TorusGeometry(0.03, 0.008, 5, 10).translate(0.03, 0.08, 0), '#c9d2d8'),
                                  tinted(new THREE.TorusGeometry(0.03, 0.008, 5, 10).translate(-0.03, 0.08, 0), '#c9d2d8')]), toy));
    key.position.set(-0.1, 0.27, 0);
    g.add(key);
    g.children.forEach(function (c) { c.castShadow = !small; });
    g.scale.setScalar(1.7);
    race.add(g);
    return { g: g, wheels: wheels, key: key, z: 0 };
  }
  var redCar = windUpCar('#ff4f5a', '#ff7a7a'), blueCar = windUpCar('#4f8fff', '#7ab0ff');
  function rosette(col, label) {
    var g = new THREE.Group();
    var tex = canvasTex(64, 64, function (x) {
      x.fillStyle = col; x.beginPath(); x.arc(32, 32, 31, 0, Math.PI * 2); x.fill();
      x.fillStyle = '#fffdf6'; x.beginPath(); x.arc(32, 32, 20, 0, Math.PI * 2); x.fill();
      x.fillStyle = '#2b3a5a'; x.font = 'bold 18px sans-serif'; x.textAlign = 'center'; x.fillText(label, 32, 31);
      x.strokeStyle = '#2b3a5a'; x.lineWidth = 2.5; x.beginPath(); x.arc(32, 36, 8, 0.2, Math.PI - 0.2); x.stroke();
    });
    var face = new THREE.Mesh(new THREE.CircleGeometry(0.07, 20), new THREE.MeshBasicMaterial({ map: tex }));
    var frill = new THREE.Mesh(new THREE.CircleGeometry(0.09, 12), new THREE.MeshLambertMaterial({ color: col, side: THREE.DoubleSide }));
    frill.position.z = -0.004;
    var tails = new THREE.Mesh(merge([tinted(box(0.035, 0.12, 0.003).translate(-0.025, -0.1, -0.006).rotateZ(0.2), col), tinted(box(0.035, 0.12, 0.003).translate(0.025, -0.1, -0.006).rotateZ(-0.2), col)]), toy);
    var stick = new THREE.Mesh(cyl(0.004, 0.004, 0.16, 4).translate(0, -0.08, -0.01), basic('#ffffff'));
    g.add(stick, tails, frill, face);
    return g;
  }
  var rosetteRed = rosette('#ffc21a', '1st'), rosetteBlue = rosette('#5aa9ff', '2nd');
  redCar.g.add(rosetteRed); blueCar.g.add(rosetteBlue);
  rosetteRed.position.set(-0.02, 0.45, 0); rosetteBlue.position.set(-0.02, 0.45, 0);
  rosetteRed.rotation.y = rosetteBlue.rotation.y = Math.PI / 2;
  var CONF = small ? 80 : 160, confetti = new THREE.InstancedMesh(new THREE.PlaneGeometry(0.03, 0.05), new THREE.MeshLambertMaterial({ side: THREE.DoubleSide }), CONF), confSeed = [];
  for (var cf = 0; cf < CONF; cf++) {
    confSeed.push((r() - 0.5) * 1.6, 2.2 + r() * 2.4, (r() - 0.5) * 1.6, r() * 6.28, r() * 6.28);
    confetti.setColorAt(cf, tc.set(flowerCols[cf % flowerCols.length]));
  }
  confetti.frustumCulled = false;
  confetti.visible = false;
  race.add(confetti);

  // ── The kite field: a piggy bank, a present, the homemade kite ─────────
  var kiteField = new THREE.Group();
  island.add(kiteField);
  var pgy = land(PIGGY.x, PIGGY.z), kfl = [];
  put(kfl, sph(0.2, 16, 12), '#ff9fb8', PIGGY.x, pgy + 0.24, PIGGY.z, 0, 0.5, 0, 1.3, 1, 1);
  put(kfl, cyl(0.07, 0.075, 0.07, 14), '#ff8aa8', PIGGY.x + Math.cos(0.5) * 0.27, pgy + 0.24, PIGGY.z - Math.sin(0.5) * 0.27, 0, 0.5, Math.PI / 2);
  put(kfl, sph(0.012, 6, 5), '#c94a6a', PIGGY.x + Math.cos(0.5) * 0.31 + 0.01, pgy + 0.25, PIGGY.z - Math.sin(0.5) * 0.31 + 0.02);
  put(kfl, sph(0.012, 6, 5), '#c94a6a', PIGGY.x + Math.cos(0.5) * 0.31 - 0.01, pgy + 0.25, PIGGY.z - Math.sin(0.5) * 0.31 - 0.02);
  [[0.15, 0.1], [0.15, -0.1]].forEach(function (e) { put(kfl, new THREE.ConeGeometry(0.05, 0.08, 6), '#ff8aa8', PIGGY.x + e[0] * Math.cos(0.5) + e[1] * Math.sin(0.5), pgy + 0.42, PIGGY.z - e[0] * Math.sin(0.5) + e[1] * Math.cos(0.5)); });
  [[0.12, 0.1], [0.12, -0.1], [-0.12, 0.1], [-0.12, -0.1]].forEach(function (e) { put(kfl, cyl(0.04, 0.045, 0.1, 8), '#ff8aa8', PIGGY.x + e[0] * Math.cos(0.5) + e[1] * Math.sin(0.5), pgy + 0.05, PIGGY.z - e[0] * Math.sin(0.5) + e[1] * Math.cos(0.5)); });
  put(kfl, box(0.09, 0.012, 0.02), '#5a2a3a', PIGGY.x, pgy + 0.445, PIGGY.z, 0, 0.5);
  put(kfl, new THREE.TorusGeometry(0.03, 0.008, 5, 10, Math.PI * 1.5), '#ff8aa8', PIGGY.x - Math.cos(0.5) * 0.27, pgy + 0.28, PIGGY.z + Math.sin(0.5) * 0.27, 0, 0.5 + Math.PI / 2);
  for (var cn = 0; cn < 5; cn++) put(kfl, cyl(0.035, 0.035, 0.008, 14), '#ffcf3a', PIGGY.x + 0.3 + r() * 0.3, pgy + 0.01 + cn * 0.009, PIGGY.z + 0.25 + r() * 0.1);
  put(kfl, rbox(0.4, 0.34, 0.4, 0.03), '#7fc8ff', PIGGY.x + 0.9, pgy + 0.17, PIGGY.z - 0.1, 0, 0.3);
  put(kfl, box(0.41, 0.35, 0.07), '#ff5f8a', PIGGY.x + 0.9, pgy + 0.17, PIGGY.z - 0.1, 0, 0.3);
  put(kfl, box(0.07, 0.35, 0.41), '#ff5f8a', PIGGY.x + 0.9, pgy + 0.17, PIGGY.z - 0.1, 0, 0.3);
  put(kfl, new THREE.TorusGeometry(0.06, 0.02, 6, 12), '#ff5f8a', PIGGY.x + 0.9 - 0.05, pgy + 0.38, PIGGY.z - 0.1, 0, 0.3);
  put(kfl, new THREE.TorusGeometry(0.06, 0.02, 6, 12), '#ff5f8a', PIGGY.x + 0.9 + 0.05, pgy + 0.38, PIGGY.z - 0.1, 0, 0.3 + Math.PI / 2);
  put(kfl, cyl(0.025, 0.03, 0.45, 6), '#b07a4a', STAKE.x, land(STAKE.x, STAKE.z) + 0.2, STAKE.z);
  mesh(kfl, toy, kiteField, true);
  var kite = new THREE.Group();
  kiteField.add(kite);
  var kcols = ['#ff5f5a', '#ffd23a', '#5aa9ff', '#7be07b'], kpts = [[0, 0.75], [0.48, 0], [0, -1.0], [-0.48, 0]];
  kcols.forEach(function (c, k) {
    var a = kpts[k], b = kpts[(k + 1) % 4], g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(a[0], a[1], 0), new THREE.Vector3(b[0], b[1], 0)]);
    g.computeVertexNormals();
    kite.add(new THREE.Mesh(g, new THREE.MeshLambertMaterial({ color: c, side: THREE.DoubleSide })));
  });
  kite.add(new THREE.Mesh(merge([tinted(box(0.02, 1.75, 0.02).translate(0, -0.125, 0.01), '#8a6a4a'), tinted(box(0.96, 0.02, 0.02).translate(0, 0, 0.01), '#8a6a4a')]), toy));
  var KT = 12, tailPos = new Float32Array(KT * 3), tailBows = [];
  var tailLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: '#ffffff' }));
  tailLine.geometry.setAttribute('position', new THREE.BufferAttribute(tailPos, 3));
  tailLine.frustumCulled = false;
  kiteField.add(tailLine);
  for (var tb = 0; tb < 5; tb++) {
    var bowM = new THREE.Mesh(merge([tinted(new THREE.ConeGeometry(0.1, 0.17, 4).rotateZ(Math.PI / 2).translate(0.085, 0, 0), kcols[tb % 4]),
                                     tinted(new THREE.ConeGeometry(0.1, 0.17, 4).rotateZ(-Math.PI / 2).translate(-0.085, 0, 0), kcols[tb % 4])]), toy);
    kiteField.add(bowM);
    tailBows.push(bowM);
  }
  var SN = 40, strPos = new Float32Array(SN * 3);
  var kiteString = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: '#fff8e8' }));
  kiteString.geometry.setAttribute('position', new THREE.BufferAttribute(strPos, 3));
  kiteString.frustumCulled = false;
  kiteField.add(kiteString);
  var stakeTop = new THREE.Vector3(STAKE.x, land(STAKE.x, STAKE.z) + 0.42, STAKE.z);
  var kiteLow = new THREE.Vector3(-26.6, 5.2, 30.2), kiteHigh = new THREE.Vector3(-27.4, 7.8, 28.8), kitePos = new THREE.Vector3();
  kite.scale.setScalar(1.7);

  // Balloons that go up at the end
  var balloonsUp = [[-6.5, 7], [22, 39], [-41, 1], [-24, 32], [6, 29], [7, 12], [-8, -16], [16, -14]].map(function (p, k) {
    var g = new THREE.Group();
    var b = new THREE.Mesh(sph(1.1, 16, 12).scale(1, 1.2, 1), new THREE.MeshStandardMaterial({ color: flowerCols[k % flowerCols.length], roughness: 0.25 }));
    var knot = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.1, 6), basic(flowerCols[k % flowerCols.length]));
    knot.position.y = -1.33;
    knot.scale.setScalar(2.5);
    var str = new THREE.Mesh(cyl(0.014, 0.014, 2.6, 4).translate(0, -2.6, 0), basic('#ffffff'));
    g.add(b, knot, str);
    g.position.set(p[0], land(p[0], p[1]) + 1.6, p[1]);
    g.visible = false;
    island.add(g);
    return { g: g, x: p[0], z: p[1], y: g.position.y, ph: k * 1.3 };
  });

  // ── The car: a little toy one on the lane, and its back seat ───────────
  var carOut = new THREE.Group(), col = [];
  put(col, rbox(2.1, 0.65, 1.1, 0.3), '#5ab0ff', 0, 0.55, 0);
  put(col, rbox(1.1, 0.55, 1.0, 0.25), '#7cc4ff', -0.15, 1.05, 0);
  put(col, rbox(0.6, 0.38, 1.02, 0.12), '#d8f0ff', 0.2, 1.06, 0);
  put(col, rbox(0.5, 0.38, 1.02, 0.12), '#d8f0ff', -0.48, 1.06, 0);
  [[0.65, 0.5], [-0.65, 0.5], [0.65, -0.5], [-0.65, -0.5]].forEach(function (w) { put(col, cyl(0.25, 0.25, 0.18, 14).rotateX(Math.PI / 2), '#2b2f3a', w[0], 0.25, w[1]); });
  put(col, sph(0.08, 8, 6), '#fff4b0', 1.05, 0.6, 0.35); put(col, sph(0.08, 8, 6), '#fff4b0', 1.05, 0.6, -0.35);
  carOut.add(new THREE.Mesh(merge(col), toy));
  carOut.children[0].castShadow = !small;
  var headGlow = glowSprite('#fff0c0', 2.0, carOut, 1.4, 0.6, 0);
  island.add(carOut);
  // The back seat, built in the camera's own space (you look out of the
  // right-hand rear window; the front of the car is to the left). A toy
  // car's inside: navy pleated door, teal trim, a coral armrest, a crank
  // for the window, the seat's mustard bolster and a belt on the pillar.
  var carIn = new THREE.Group();
  camera.add(carIn);
  carIn.visible = false;
  var HX0 = 0.045, HX1 = 0.35, HY0 = -0.01, HY1 = 0.15, HR = 0.035, DZ = -0.42;
  function roundRect(p, x0, y0, x1, y1, rr) {
    p.moveTo(x0 + rr, y0); p.lineTo(x1 - rr, y0); p.quadraticCurveTo(x1, y0, x1, y0 + rr); p.lineTo(x1, y1 - rr);
    p.quadraticCurveTo(x1, y1, x1 - rr, y1); p.lineTo(x0 + rr * 2.6, y1); p.quadraticCurveTo(x0, y1, x0, y1 - rr * 2.2);
    p.lineTo(x0, y0 + rr); p.quadraticCurveTo(x0, y0, x0 + rr, y0);
    return p;
  }
  // Interior plastics keep a little of their own colour as light, as if the
  // dome light were on, so the dusk outside stays the brightest thing.
  var cabinMat = new THREE.MeshLambertMaterial({ vertexColors: true });
  cabinMat.onBeforeCompile = function (sh) {
    sh.fragmentShader = sh.fragmentShader.replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\n totalEmissiveRadiance += vColor * 0.42;');
  };
  // The pillars and headliner round the window, with the window cut out
  var frameShape = new THREE.Shape();
  frameShape.moveTo(-1.5, HY0 - 0.03); frameShape.lineTo(1.5, HY0 - 0.03); frameShape.lineTo(1.5, 1.2); frameShape.lineTo(-1.5, 1.2); frameShape.lineTo(-1.5, HY0 - 0.03);
  frameShape.holes.push(roundRect(new THREE.Path(), HX0, HY0, HX1, HY1, HR));
  var cl = [];
  put(cl, new THREE.ExtrudeGeometry(frameShape, { depth: 0.02, bevelEnabled: true, bevelThickness: 0.006, bevelSize: 0.006, bevelSegments: 2, curveSegments: 10 }), '#3d8f97', 0, 0, DZ - 0.02);
  // A rubber seal round the opening
  var sealShape = roundRect(new THREE.Shape(), HX0 - 0.012, HY0 - 0.012, HX1 + 0.012, HY1 + 0.012, HR + 0.012);
  sealShape.holes.push(roundRect(new THREE.Path(), HX0, HY0, HX1, HY1, HR));
  put(cl, new THREE.ExtrudeGeometry(sealShape, { depth: 0.008, bevelEnabled: false, curveSegments: 10 }), '#262a33', 0, 0, DZ + 0.004);
  // Headliner above, with a grab handle
  put(cl, box(3, 0.6, 0.01), '#d9d0bf', 0, HY1 + 0.345, DZ + 0.014);
  put(cl, box(0.006, 0.5, 0.004), '#1d5560', 0.0, HY0 - 0.08, DZ + 0.012);
  put(cl, rbox(0.16, 0.018, 0.025, 0.008, 2), '#c8bfae', (HX0 + HX1) / 2, HY1 + 0.03, DZ + 0.03);
  put(cl, rbox(0.02, 0.03, 0.02, 0.008, 2), '#c8bfae', (HX0 + HX1) / 2 - 0.07, HY1 + 0.042, DZ + 0.02);
  put(cl, rbox(0.02, 0.03, 0.02, 0.008, 2), '#c8bfae', (HX0 + HX1) / 2 + 0.07, HY1 + 0.042, DZ + 0.02);
  // The sill where the glass has wound down into the door, and a lock pin
  put(cl, rbox(3, 0.032, 0.06, 0.012, 2), '#2f7a83', 0, HY0 - 0.035, DZ + 0.02);
  put(cl, cyl(0.005, 0.005, 0.03, 6), '#d8dde2', HX1 - 0.03, HY0 - 0.012, DZ + 0.03);
  put(cl, sph(0.008, 8, 6), '#ff8a6a', HX1 - 0.03, HY0 + 0.004, DZ + 0.03);
  // Armrest and door pull, the window crank, a round speaker
  put(cl, rbox(0.36, 0.035, 0.07, 0.016), '#ef7d5f', 0.13, HY0 - 0.095, DZ + 0.04);
  put(cl, rbox(0.36, 0.01, 0.025, 0.005, 2), '#d9654a', 0.13, HY0 - 0.115, DZ + 0.072);
  put(cl, rbox(0.085, 0.026, 0.012, 0.008, 2), '#1f2b45', 0.24, HY0 - 0.055, DZ + 0.01);
  put(cl, rbox(0.06, 0.01, 0.014, 0.005, 2), '#d8dde2', 0.24, HY0 - 0.051, DZ + 0.02);
  put(cl, cyl(0.016, 0.016, 0.01, 14).rotateX(Math.PI / 2), '#d8dde2', -0.14, HY0 - 0.1, DZ + 0.01);
  put(cl, rbox(0.06, 0.01, 0.008, 0.004, 2), '#d8dde2', -0.117, HY0 - 0.088, DZ + 0.018, 0, 0, 0.45);
  put(cl, cyl(0.009, 0.009, 0.03, 10).rotateX(Math.PI / 2), '#ef7d5f', -0.092, HY0 - 0.076, DZ + 0.035);
  put(cl, cyl(0.045, 0.045, 0.006, 20).rotateX(Math.PI / 2), '#1f2b45', -0.11, HY0 - 0.16, DZ + 0.008);
  for (var sg = 0; sg < 12; sg++) {
    var sga = sg / 12 * Math.PI * 2;
    put(cl, cyl(0.0035, 0.0035, 0.004, 5).rotateX(Math.PI / 2), '#6a7aa0', -0.11 + Math.cos(sga) * 0.025, HY0 - 0.16 + Math.sin(sga) * 0.025, DZ + 0.013);
  }
  // The seat you sit on: its cushion running to the door, piped, and the
  // side bolster of the backrest at the rear
  put(cl, rbox(1.6, 0.12, 0.23, 0.05, 3), '#e3a83e', 0, HY0 - 0.215, DZ + 0.165);
  put(cl, rbox(1.6, 0.012, 0.012, 0.005, 2), '#c98f2a', 0, HY0 - 0.157, DZ + 0.06);
  for (var sm2 = -5; sm2 <= 5; sm2++) put(cl, box(0.006, 0.004, 0.2), '#c08a2c', sm2 * 0.12, HY0 - 0.154, DZ + 0.17);
  for (sm2 = -5; sm2 <= 5; sm2++) put(cl, sph(0.007, 6, 4), '#b07a22', sm2 * 0.12 + 0.06, HY0 - 0.154, DZ + 0.2);
  put(cl, rbox(0.07, 0.3, 0.2, 0.03, 3), '#e3a83e', 0.3, HY0 - 0.2, DZ + 0.16);
  // The seatbelt down the rear pillar, with its buckle
  var beltTop = new THREE.Vector3(HX1 + 0.022, HY1 + 0.07, DZ + 0.02), beltLow = new THREE.Vector3(HX1 + 0.012, HY0 - 0.1, DZ + 0.05);
  var beltLen = beltTop.distanceTo(beltLow);
  var belt = box(0.03, beltLen, 0.004);
  belt.applyQuaternion(q1.setFromUnitVectors(UP, v1.subVectors(beltTop, beltLow).normalize()));
  put(cl, belt, '#2b3140', (beltTop.x + beltLow.x) / 2, (beltTop.y + beltLow.y) / 2, (beltTop.z + beltLow.z) / 2);
  put(cl, rbox(0.045, 0.03, 0.012, 0.006, 2), '#3a4050', beltTop.x, beltTop.y + 0.01, beltTop.z);
  put(cl, rbox(0.04, 0.055, 0.012, 0.006, 2), '#d8dde2', beltLow.x - 0.004, beltLow.y + 0.04, beltLow.z - 0.01);
  put(cl, rbox(0.05, 0.035, 0.03, 0.01, 2), '#e5484d', beltLow.x - 0.004, beltLow.y, beltLow.z);
  // The wing mirror outside, on the front door
  put(cl, rbox(0.11, 0.07, 0.07, 0.028), '#5ab0ff', HX0 + 0.04, HY0 + 0.075, -0.84);
  put(cl, box(0.07, 0.014, 0.02), '#4a9ae8', HX0 - 0.02, HY0 + 0.06, -0.84);
  put(cl, box(0.08, 0.05, 0.004), '#d8ecff', HX0 + 0.095, HY0 + 0.075, -0.81, 0, -1.2);
  carIn.add(new THREE.Mesh(merge(cl), cabinMat));
  // The door below the window: pleated navy fabric
  var pleats = canvasTex(64, 64, function (x) {
    x.fillStyle = '#34406e'; x.fillRect(0, 0, 64, 64);
    x.fillStyle = '#2a3460'; x.fillRect(0, 0, 6, 64);
    x.fillStyle = 'rgba(255,255,255,0.12)'; x.fillRect(8, 0, 2, 64);
  });
  pleats.wrapS = pleats.wrapT = THREE.RepeatWrapping;
  pleats.repeat.set(16, 1);
  var doorShape = new THREE.Shape();
  doorShape.moveTo(-1.5, -1.2); doorShape.lineTo(1.5, -1.2); doorShape.lineTo(1.5, HY0 - 0.03); doorShape.lineTo(-1.5, HY0 - 0.03); doorShape.lineTo(-1.5, -1.2);
  var door = new THREE.Mesh(new THREE.ShapeGeometry(doorShape, 2), new THREE.MeshLambertMaterial({ map: pleats, emissive: '#ffffff', emissiveMap: pleats, emissiveIntensity: 0.4 }));
  door.position.z = DZ;
  carIn.add(door);
  // The top of the wound-down glass
  var glassTop = new THREE.Mesh(new THREE.PlaneGeometry(HX1 - HX0 - 0.03, 0.022), new THREE.MeshBasicMaterial({ color: '#bfe0ff', transparent: true, opacity: 0.4 }));
  glassTop.position.set((HX0 + HX1) / 2, HY0 + 0.01, DZ - 0.006);
  carIn.add(glassTop);
  // The world outside, rushing: hedges smeared into green streaks, and
  // the lamps of the lane drawn out into lines of light.
  function streakTex(cols, n, alpha, warm) {
    var t = canvasTex(512, 64, function (x) {
      for (var i = 0; i < n; i++) {
        var y = r() * 64, len = 40 + r() * 220, sx = r() * 512, g = x.createLinearGradient(sx, 0, sx + len, 0);
        var c = cols[Math.floor(r() * cols.length)];
        g.addColorStop(0, 'rgba(' + c + ',0)'); g.addColorStop(0.5, 'rgba(' + c + ',' + alpha + ')'); g.addColorStop(1, 'rgba(' + c + ',0)');
        x.fillStyle = g;
        x.fillRect(sx, y, len, warm ? 2 + r() * 2 : 3 + r() * 9);
        if (sx + len > 512) x.fillRect(sx - 512, y, len, warm ? 2 : 6);
      }
    });
    t.wrapS = THREE.RepeatWrapping;
    return t;
  }
  var hedgeBlur = new THREE.Mesh(new THREE.PlaneGeometry(1.6, 0.16), new THREE.MeshBasicMaterial({
    map: streakTex(['40,92,40', '58,120,48', '30,70,36', '84,140,60'], 90, 0.55), transparent: true, depthWrite: false, opacity: 0.85 }));
  hedgeBlur.position.set(0.3, -0.06, -1.0);
  var lampBlur = new THREE.Mesh(new THREE.PlaneGeometry(1.6, 0.12), new THREE.MeshBasicMaterial({
    map: streakTex(['255,214,140', '255,236,190'], 14, 0.9, true), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  lampBlur.position.set(0.3, 0.2, -1.0);
  carIn.add(hedgeBlur, lampBlur);
  // A pinwheel clipped to the sill, spinning, and a rainbow ribbon on it
  // streaming back in the wind
  var carPin = new THREE.Group();
  carPin.position.set(HX0 + 0.05, HY0 + 0.045, DZ - 0.07);
  carIn.add(carPin);
  var pinStick2 = new THREE.Mesh(cyl(0.0025, 0.0025, 0.08, 5).translate(0, -0.04, 0), basic('#ffffff'));
  pinStick2.rotation.x = -0.6;
  carPin.add(pinStick2);
  var carPinHead = new THREE.Group();
  carPinHead.position.z = 0.006;
  carPin.add(carPinHead);
  ['#ff5f8a', '#5ac8ff', '#ffd23a', '#7be07b'].forEach(function (c, k) {
    var g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0.028, 0, 0.005), new THREE.Vector3(0.028, 0.028, -0.005)]);
    g.computeVertexNormals();
    var b = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ color: c, emissive: c, emissiveIntensity: 0.35, side: THREE.DoubleSide }));
    b.rotation.z = k * Math.PI / 2;
    carPinHead.add(b);
  });
  carPinHead.add(new THREE.Mesh(sph(0.006, 6, 5), basic('#ffffff')));
  var RIB = 28, ribGeo = new THREE.PlaneGeometry(0.36, 0.012, RIB, 1).translate(0.18, 0, 0), ribBase = ribGeo.attributes.position.array.slice(0);
  var ribCol = new Float32Array(ribGeo.attributes.position.count * 3), rainbow = ['#ff5f5a', '#ffb347', '#ffd23a', '#7be07b', '#5ab0ff', '#b98cff'];
  for (var rv = 0; rv < ribGeo.attributes.position.count; rv++) {
    tc.set(rainbow[Math.min(5, Math.floor(ribBase[rv * 3] / 0.36 * 6))]);
    ribCol[rv * 3] = tc.r; ribCol[rv * 3 + 1] = tc.g; ribCol[rv * 3 + 2] = tc.b;
  }
  ribGeo.setAttribute('color', new THREE.BufferAttribute(ribCol, 3));
  var ribbonM = new THREE.Mesh(ribGeo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
  ribbonM.frustumCulled = false;
  ribbonM.position.set(HX0 + 0.05, HY0 + 0.025, DZ - 0.06);
  carIn.add(ribbonM);
  // Wind streaking past the opening
  var WS = 16, wsPos = new Float32Array(WS * 6), wsSeed = [];
  for (var ws = 0; ws < WS; ws++) wsSeed.push([r(), HY0 + 0.01 + r() * (HY1 - HY0 - 0.02), 0.6 + r() * 1.4, 0.03 + r() * 0.06]);
  var wsGeo = new THREE.BufferGeometry();
  wsGeo.setAttribute('position', new THREE.BufferAttribute(wsPos, 3));
  var windLines = new THREE.LineSegments(wsGeo, new THREE.LineBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false }));
  windLines.frustumCulled = false;
  carIn.add(windLines);
  var drive = 0.985, carLoop = 0.35;

  // ── Weather and the wishing star ───────────────────────────────────────
  var rain = rainField({ count: small ? 900 : 1800, box: [14, 12, 20], color: '#b8c8de', opacity: 0.45, speed: 10 });
  world.add(rain.lines);
  var TRN = 24, trailPos = new Float32Array(TRN * 3), trailCol = new Float32Array(TRN * 3);
  var trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
  trailGeo.setAttribute('color', new THREE.BufferAttribute(trailCol, 3));
  var trail = new THREE.Points(trailGeo, new THREE.PointsMaterial({ map: dotTex, size: 15, sizeAttenuation: false, vertexColors: true, transparent: true,
    blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
  trail.frustumCulled = false;
  world.add(trail);
  var wishStar = new THREE.Sprite(new THREE.SpriteMaterial({ map: canvasTex(128, 128, function (x) {
    var g = x.createRadialGradient(64, 64, 0, 64, 64, 64);
    g.addColorStop(0, 'rgba(255,255,255,1)'); g.addColorStop(0.08, 'rgba(255,246,210,0.9)'); g.addColorStop(0.25, 'rgba(255,220,150,0.2)'); g.addColorStop(1, 'rgba(255,220,150,0)');
    x.fillStyle = g; x.fillRect(0, 0, 128, 128);
    x.globalCompositeOperation = 'lighter'; x.fillStyle = 'rgba(255,240,200,0.6)';
    x.fillRect(62, 0, 4, 128); x.fillRect(0, 62, 128, 4);
  }), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
  world.add(wishStar);
  // Seen through the car window: [x, y] over distance in the camera's view.
  function starAt(t, out) {
    var a = layout.portrait ? [-0.25, 0.52, 0.25, 0.3] : [0.16, 0.36, 0.72, 0.2];
    return out.set(lerp(a[0], a[2], t), lerp(a[1], a[3], t), -1).multiplyScalar(400).applyQuaternion(camera.quaternion).add(camera.position);
  }

  // ── Blink: a soft dip of light over a cut ──────────────────────────────
  var blinkMat = new THREE.ShaderMaterial({
    transparent: true, depthTest: false, depthWrite: false,
    uniforms: { uColor: { value: new THREE.Color('#fff4e0') }, uA: { value: 0 } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
    fragmentShader: 'uniform vec3 uColor; uniform float uA; varying vec2 vUv; void main(){ float v = length(vUv - 0.5);\n' +
      ' gl_FragColor = vec4(uColor * (1.0 - v * 0.35), uA);\n#include <colorspace_fragment>\n}'
  });
  var blink = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), blinkMat);
  blink.frustumCulled = false;
  blink.renderOrder = 999;
  world.add(blink);

  // ── Helpers for the frame ──────────────────────────────────────────────
  var U_ = 0, prevU = 0, clockT = 0, fired = {};
  function beat(i, a, b) {
    if (!TL) return 0;
    i = Math.min(i, TL.count - 1);
    return smooth(TL.start(i) + a * 1.6, TL.start(i) + b * 1.6, U_);
  }
  function at(i, f) { i = Math.min(i, TL ? TL.count - 1 : 0); return TL ? TL.start(i) + f * 1.6 : 0; }
  // Seconds since the reader passed a moment (forwards), or -1.
  function since(key, u) {
    if (prevU < u && U_ >= u) fired[key] = clockT;
    if (U_ < u - 0.01) fired[key] = -1;
    return fired[key] != null && fired[key] >= 0 ? clockT - fired[key] : -1;
  }
  function vis(o, on) { o.visible = on; return on; }

  var skyDay = { top: new THREE.Color('#4f9be8'), mid: new THREE.Color('#8cc8f5'), hor: new THREE.Color('#e2f3ff') };
  var skyGold = { top: new THREE.Color('#5a78c8'), mid: new THREE.Color('#f2b27a'), hor: new THREE.Color('#ffd49a') };
  var skyNight = { top: new THREE.Color('#090d2a'), mid: new THREE.Color('#1a2152'), hor: new THREE.Color('#353c74') };
  var skyRain = { top: new THREE.Color('#7f8da2'), mid: new THREE.Color('#a6b0be'), hor: new THREE.Color('#c9d0d8') };
  var horizon = new THREE.Color(), sunDir = new THREE.Vector3(), fwd = new THREE.Vector3(), focus = new THREE.Vector3();
  var layout = { sx: 1, portrait: false };

  function frame(f) {
    var row = f.row, time = f.time, dt = f.dt;
    U_ = f.u;
    clockT = time;
    var dark = clamp(f.dark, 0, 1), rainAmt = clamp(f.snow, 0, 1), wind = f.wind;
    var sunT = row[8], gold = clamp(row[9], 0, 1), blinkA = row[10], inCar = row[11] > 0.5;
    var day = 1 - dark;
    U.uClock.value = time;
    U.uWind.value = wind;

    // ── Beats ──
    var seaAmt = beat(0, 0.48, 0.7) * (1 - beat(0, 0.86, 1.0));
    var morph = beat(1, 0.08, 0.34) * (1 - beat(2, 0.0, 0.3));
    var streak = beat(2, 0.16, 0.42), wished = beat(2, 0.36, 0.44) * (1 - beat(2, 0.62, 0.7));
    var grow = beat(3, 0.5, 0.72), blowAmt = beat(3, 0.66, 1.1);
    var turning = beat(4, 0.1, 0.52), magic = beat(4, 0.08, 0.2) * (1 - beat(4, 0.52, 0.62)), comfort = beat(4, 0.7, 0.9);
    var lit = beat(5, 0.55, 0.72), answer = beat(5, 0.78, 0.95), gateOpen = beat(5, 0.58, 0.85) * (1 - beat(6, 0.0, 0.2));
    var whirlIn = beat(6, 0.08, 0.3), calm = beat(6, 0.55, 0.92), whirl = whirlIn * (1 - calm);
    var build = beat(7, 0.08, 0.5), help = beat(7, 0.55, 0.75), flagUp = beat(7, 0.68, 0.8);
    var climb = beat(8, 0.15, 0.95);
    var trayIn = beat(10, 0.0, 0.3), nostalgia = beat(10, 0.6, 0.95);
    var pour = beat(11, 0.12, 0.3) * (1 - beat(11, 0.45, 0.55)), poured = beat(11, 0.15, 0.5), served = beat(11, 0.3, 0.55), lift = beat(11, 0.58, 0.92);
    var raceT = beat(12, 0.06, 0.52), second = beat(12, 0.58, 0.7);
    var soar = beat(13, 0.45, 0.85);
    var finale = beat(14, 0.45, 1.6);

    // ── Camera ──
    if (inCar) {
      drive = (drive + dt * 0.05 * (env.reduceMotion ? 0.5 : 1)) % 1;
      ROAD.getPointAt(drive, v1);
      ROAD.getTangentAt(drive, tan);
      nrm.set(-tan.z, 0, tan.x);
      camera.position.set(v1.x + nrm.x * 1.0, land(v1.x, v1.z) + 1.05 + Math.sin(time * 9) * 0.006 + Math.sin(time * 3.3) * 0.008, v1.z + nrm.z * 1.0);
      v2.copy(nrm).multiplyScalar(Math.cos(0.18)).addScaledVector(tan, Math.sin(0.18));
      camera.rotation.set(lerp(0.12, -0.03, beat(2, 0.45, 0.6)) - f.my * 0.03, Math.atan2(-v2.x, -v2.z) - f.mx * 0.05, 0);
      carIn.position.set(layout.portrait ? -0.12 : 0, layout.portrait ? 0.07 : 0, 0);
    } else {
      camera.position.set(f.cam, row[4], row[5]);
      var roll = Math.sin(time * 1.4) * 0.05 * whirl * (env.reduceMotion ? 0.3 : 1);
      camera.rotation.set(row[7] + (layout.portrait ? row[13] : 0) - f.my * 0.06, row[6] + (layout.portrait ? row[12] : 0) - f.mx * 0.12, roll);
      camera.position.y += Math.sin(time * 1.1) * 0.006;
    }
    carIn.visible = inCar;
    sky.position.copy(camera.position);
    camera.updateMatrixWorld();

    // ── Sun, sky, light ──
    var az = Math.PI * clamp(sunT, -0.05, 1.05);
    sunDir.set(Math.cos(az), Math.max(Math.sin(az), 0) * 0.62 + 0.02, Math.sin(az) * 0.85).normalize();
    var low = smooth(0.25, 0.04, sunDir.y);
    dome.uniforms.top.value.copy(skyDay.top).lerp(skyGold.top, gold * 0.6).lerp(skyRain.top, rainAmt * 0.9).lerp(skyNight.top, dark);
    dome.uniforms.mid.value.copy(skyDay.mid).lerp(skyGold.mid, gold).lerp(skyRain.mid, rainAmt * 0.9).lerp(skyNight.mid, dark * 0.9);
    horizon.copy(skyDay.hor).lerp(skyGold.hor, gold).lerp(skyRain.hor, rainAmt * 0.9).lerp(skyNight.hor, dark * 0.85);
    dome.uniforms.horizon.value.copy(horizon);
    dome.uniforms.sunDir.value.copy(sunDir);
    dome.uniforms.sunColor.value.set('#fff1d0').lerp(tc.set('#ff9a50'), gold).multiplyScalar((0.25 + gold * 0.35) * day * (1 - rainAmt));
    world.fog.color.copy(horizon);
    world.fog.near = lerp(lerp(90, 8, seaAmt), 130, smooth(10, 30, camera.position.y));
    world.fog.far = lerp(lerp(420, 170, seaAmt), 560, smooth(10, 30, camera.position.y));
    gl.setClearColor(horizon);
    sunSprite.position.copy(sunDir).multiplyScalar(1000);
    sunHalo.position.copy(sunSprite.position);
    sunSprite.scale.setScalar(42 + gold * 30);
    sunHalo.scale.setScalar(230 + gold * 330);
    sunSprite.material.opacity = day * (1 - rainAmt) * smooth(-0.05, 0.03, sunDir.y);
    sunHalo.material.opacity = sunSprite.material.opacity * (0.45 + gold * 0.3);
    sunSprite.material.color.set('#ffffff').lerp(tc.set('#ffc070'), gold);
    sunHalo.material.color.set('#fff2d0').lerp(tc.set('#ff9a4a'), gold);
    stars.material.opacity = smooth(0.35, 0.85, dark) * 0.9;
    moon.material.opacity = smooth(0.5, 0.9, dark);

    sun.position.copy(sunDir).multiplyScalar(80);
    fwd.set(0, 0, -1).applyQuaternion(camera.quaternion);
    var reach = lerp(12, 70, smooth(8, 30, camera.position.y));
    focus.copy(camera.position).addScaledVector(fwd, reach * 0.5);
    focus.y = 0;
    sun.position.add(focus);
    sun.target.position.copy(focus);
    sun.shadow.camera.left = sun.shadow.camera.bottom = -reach;
    sun.shadow.camera.right = sun.shadow.camera.top = reach;
    sun.shadow.camera.updateProjectionMatrix();
    sun.intensity = 2.5 * day * (1 - rainAmt * 0.75) * (1 - low * 0.5) * smooth(-0.02, 0.06, sunDir.y);
    sun.color.set('#fff4e0').lerp(tc.set('#ffb070'), Math.max(gold, low));
    hemi.intensity = lerp(0.48, 1.25, day) * (1 - rainAmt * 0.15);
    hemi.color.set('#e4f2ff').lerp(tc.set('#ffd8b0'), gold * 0.6).lerp(tc.set('#3a4a8a'), dark);
    hemi.groundColor.set('#8aa86a').lerp(tc.set('#a8805a'), gold * 0.5).lerp(tc.set('#1a1830'), dark);
    var flashT = since('flash', at(4, 0.55));
    var flash = flashT < 0 ? 0 : Math.max(Math.exp(-Math.pow((flashT - 0.05) / 0.06, 2)), 0.7 * Math.exp(-Math.pow((flashT - 0.35) / 0.08, 2)));
    moonLight.intensity = dark * 0.65 + flash * 3;
    flashPane.material.opacity = flash * 0.9;
    gl.toneMappingExposure = 1.0 + gold * 0.08 - rainAmt * 0.05;

    // ── Clouds ──
    var cloudGrey = rainAmt * 0.6;
    cloudMat.color.set('#ffffff').lerp(tc.set('#8a95a8'), cloudGrey).lerp(tc.set('#ffc8a0'), gold * 0.7).lerp(tc.set('#2a3060'), dark * 0.9);
    cloudMat.emissive.set('#ffffff').lerp(tc.set('#8a95a8'), cloudGrey).lerp(tc.set('#ffb890'), gold * 0.6).lerp(tc.set('#1a2050'), dark);
    cloudMat.emissiveIntensity = 0.45 - dark * 0.3;
    var n = 0;
    var animalsOn = U_ > at(0, 0.9) && U_ < at(2, 0.35);
    animalPuffs.forEach(function (p) {
      var cp = cloudPos[p.name], k = smooth(p.lag, p.lag + 0.65, morph);
      if (!animalsOn) { puffs.setMatrixAt(n++, m1.makeScale(0, 0, 0)); return; }
      var x = lerp(p.blob[0], p.a[0], k), y = lerp(p.blob[1], p.a[1], k), rad = lerp(p.blob[2], p.a[2], k);
      v1.copy(cp.at).addScaledVector(cp.R, (x + time * 0.05) * cp.s).addScaledVector(cp.U, y * cp.s).addScaledVector(cp.F, p.z * cp.s * (1 - k * 0.85));
      puffs.setMatrixAt(n++, m1.compose(v1, q1.identity(), s1.setScalar(rad * cp.s * (1 + 0.03 * Math.sin(time + p.lag * 9)))));
    });
    dayPuffs.forEach(function (p) {
      if (animalsOn && p.z > 0) { puffs.setMatrixAt(n++, m1.makeScale(0, 0, 0)); return; }
      v1.set(p.x + time * 0.4, p.y, p.z);
      puffs.setMatrixAt(n++, m1.compose(v1, q1.identity(), s1.setScalar(p.s)));
    });
    puffs.instanceMatrix.needsUpdate = true;

    // ── The bath and the sea ──
    var shellA = 1 - seaAmt;
    fading.forEach(function (m) { m.opacity = shellA; });
    shellMesh.visible = bathFurn.visible = tileWall.visible = tileSide.visible = bathFloor.visible = shellA > 0.01;
    bedroom.visible = living.visible = kitchen.visible = seaAmt < 0.02;
    island.visible = seaAmt < 0.6;
    bathSea.material.uniforms.uSea.value = seaAmt;
    bathSea.scale.setScalar(seaAmt > 0.001 ? 1 : 0.004);
    bathSea.material.uniforms.uChop.value = 0.32 + seaAmt * 0.4;
    [bathSea, sea].forEach(function (s) {
      var su = s.material.uniforms;
      su.uTime.value = time;
      su.uSky.value.copy(horizon).lerp(dome.uniforms.mid.value, 0.35);
      su.uSunDir.value.copy(sunDir);
      su.uSun.value.set('#fff2d0').lerp(tc.set('#ffb070'), gold).multiplyScalar(day * (1 - rainAmt));
    });
    sea.material.uniforms.uDeep.value.set('#1fa6c8').lerp(tc.set('#3a6f8a'), rainAmt * 0.5).lerp(tc.set('#0c1838'), dark * 0.9);
    var bob = (0.004 + seaAmt * 0.012), bathOn = U_ < at(1, 0.4);
    if (vis(ship, bathOn)) {
      ship.position.set(TUB.x + 0.42 + seaAmt * 0.5 + beat(0, 0.7, 0.95) * 0.6, WATER + Math.sin(time * 2.1) * bob, TUB.z - 0.12 + seaAmt * 0.28);
      ship.rotation.set(Math.sin(time * 1.7) * 0.05 * (1 + seaAmt * 2), 0.15 + seaAmt * 0.35, Math.sin(time * 2.1 + 1) * 0.06 * (1 + seaAmt * 2));
    }
    if (vis(duck, bathOn)) {
      duck.position.set(TUB.x + 0.02 + seaAmt * 0.25, WATER + Math.sin(time * 2.4 + 2) * bob, TUB.z + 0.14 + seaAmt * 0.12);
      duck.rotation.set(Math.sin(time * 1.9) * 0.08 * (1 + seaAmt), 0.6 + Math.sin(time * 0.3) * 0.3, Math.sin(time * 2.4) * 0.08);
    }
    if (vis(crown, bathOn)) {
      crown.position.set(TUB.x + 0.76 + seaAmt * 0.4, WATER - 0.035 + Math.sin(time * 2.0 + 4) * bob, TUB.z + 0.14 - seaAmt * 0.32);
      crown.rotation.set(0.25 + Math.sin(time * 1.5) * 0.08, time * 0.1, 0.15);
    }
    foam.visible = bathOn && seaAmt < 0.5;
    foam.material.opacity = 0.92 * (1 - seaAmt * 2);
    if (vis(bubbles.points, bathOn && seaAmt < 0.9)) bubbles.update(f, bubbleAt, env.reduceMotion);

    // ── Garden weather: rain, puddles, the umbrella ──
    rain.update(f, camera.position, rainAmt * (inCar ? 0 : 1), env.reduceMotion);
    puddles.forEach(function (p) { p.material.uniforms.uTime.value = time; p.material.uniforms.uRain.value = rainAmt; p.material.uniforms.uSky.value.copy(horizon); });
    rainGear.visible = U_ < at(3, 0);
    umbSpin.rotation.y += dt * (0.3 + rainAmt * 3.5) * (env.reduceMotion ? 0.3 : 1);
    umbrella.position.y = 0.33 + Math.abs(Math.sin(time * 3)) * 0.04 * rainAmt;

    // ── The wishing star ──
    var tv = trail.visible = streak > 0.001 && streak < 0.999;
    if (tv) {
      for (var k = 0; k < TRN; k++) {
        var tt = clamp(streak - k * 0.012, 0, 1), fade = Math.pow(1 - k / TRN, 1.6) * smooth(0, 0.08, streak) * (1 - smooth(0.85, 1, streak));
        starAt(tt, v1);
        trailPos[k * 3] = v1.x; trailPos[k * 3 + 1] = v1.y; trailPos[k * 3 + 2] = v1.z;
        trailCol[k * 3] = fade; trailCol[k * 3 + 1] = fade * 0.95; trailCol[k * 3 + 2] = fade * 0.8;
      }
      trailGeo.attributes.position.needsUpdate = true;
      trailGeo.attributes.color.needsUpdate = true;
    }
    if (vis(wishStar, wished > 0.01)) {
      starAt(1, wishStar.position);
      wishStar.scale.setScalar((22 + Math.sin(time * 3) * 4) * (0.6 + wished * 0.6));
      wishStar.material.opacity = wished;
    }

    // ── The car ──
    carLoop = (carLoop + dt * 0.03) % 1;
    ROAD.getPointAt(carLoop, v1);
    ROAD.getTangentAt(carLoop, tan);
    carOut.visible = !inCar;
    carOut.position.set(v1.x - tan.z * 1.0, land(v1.x, v1.z) + 0.04, v1.z + tan.x * 1.0);
    carOut.rotation.y = Math.atan2(-tan.z, tan.x);
    headGlow.material.opacity = smooth(0.3, 0.7, dark);
    if (inCar) {
      var pa = ribGeo.attributes.position;
      for (var pv2 = 0; pv2 < pa.count; pv2++) {
        var bx = ribBase[pv2 * 3], k2 = bx / 0.36;
        pa.setXYZ(pv2, bx * 0.8, ribBase[pv2 * 3 + 1] + Math.sin(bx * 30 - time * 24) * 0.022 * k2 + k2 * 0.03, -k2 * 0.1 + Math.sin(bx * 22 - time * 19) * 0.025 * k2);
      }
      pa.needsUpdate = true;
      carPinHead.rotation.z -= dt * 22 * (env.reduceMotion ? 0.3 : 1);
      var rush = env.reduceMotion ? 0.4 : 1;
      hedgeBlur.material.map.offset.x -= dt * 1.4 * rush;
      lampBlur.material.map.offset.x -= dt * 0.7 * rush;
      hedgeBlur.material.color.set('#ffffff').multiplyScalar(0.55 + day * 0.45);
      for (var w2 = 0; w2 < WS; w2++) {
        var sw = wsSeed[w2], ph2 = ((time * sw[2] + sw[0]) % 1) * 1.4 - 0.4;
        wsPos[w2 * 6] = ph2; wsPos[w2 * 6 + 1] = sw[1]; wsPos[w2 * 6 + 2] = -0.5;
        wsPos[w2 * 6 + 3] = ph2 + sw[3]; wsPos[w2 * 6 + 4] = sw[1]; wsPos[w2 * 6 + 5] = -0.5;
      }
      wsGeo.attributes.position.needsUpdate = true;
      windLines.material.opacity = 0.18 + 0.1 * Math.sin(time * 5);
    }

    // ── Dandelions ──
    yellow.count = heads.length;
    heads.forEach(function (h, k) {
      var sc = 1 - smooth(k * 0.05, 0.55 + k * 0.05, grow);
      m1.compose(h, q1.setFromAxisAngle(v1.set(1, 0, 0.3).normalize(), 0.2 + Math.sin(time * 1.3 + k) * 0.05 * (1 + wind)), s1.setScalar(Math.max(sc, 0.001)));
      yellow.setMatrixAt(k, m1);
    });
    yellow.instanceMatrix.needsUpdate = true;
    seedMat.uniforms.uGrow.value = grow;
    seedMat.uniforms.uBlow.value = blowAmt;
    seedMat.uniforms.uTime.value = time;

    // ── Bedroom ──
    var night = smooth(0.4, 0.9, dark);
    lampMat.color.set('#6a6050').lerp(tc.set('#fff0c4'), Math.max(night, 0.3));
    lampGlowS.material.opacity = night * (0.35 + comfort * 0.2);
    var downstairs = U_ > at(8, 0.9) && U_ < at(10, 0);
    if (downstairs) nightLight.position.set(-5.7, F0 + 1.45, -6.0);
    else nightLight.position.copy(LAMP).add(v1.set(-0.1, 0.1, 0.25));
    nightLight.intensity = downstairs ? smooth(0.4, 0.9, dark) * 2.6 : night * (3.4 + comfort * 1.8) * (bedroom.visible ? 1 : 0);
    floorLampGlow.material.opacity = downstairs ? smooth(0.4, 0.9, dark) * 0.7 : 0;
    projStars.material.opacity = night * 0.75;
    if (bedroom.visible && night > 0.01) {
      var rot = time * 0.06;
      for (var s2 = 0; s2 < NS; s2++) {
        var c0 = Math.cos(rot), s0 = Math.sin(rot), dx = projDir[s2 * 4] * c0 - projDir[s2 * 4 + 2] * s0, dy = projDir[s2 * 4 + 1], dz = projDir[s2 * 4] * s0 + projDir[s2 * 4 + 2] * c0;
        var t = 1e9, hidden = false, tx;
        if (dx > 0) { tx = (6.29 - LAMP.x) / dx; if (tx < t) t = tx; } else if (dx < 0) { tx = (0.08 - LAMP.x) / dx; if (tx < t) t = tx; }
        if (dz < 0) { tx = (-8.78 - LAMP.z) / dz; if (tx < t) t = tx; } else if (dz > 0) { tx = (FRONT - LAMP.z) / dz; if (tx < t) { t = tx; hidden = true; } }
        if (dy > 0) { tx = (C1 - 0.01 - LAMP.y) / dy; if (tx < t) { t = tx; hidden = false; } }
        if (hidden) { projPos[s2 * 3] = 0; projPos[s2 * 3 + 1] = -100; projPos[s2 * 3 + 2] = 0; continue; }
        projPos[s2 * 3] = LAMP.x + dx * t; projPos[s2 * 3 + 1] = LAMP.y + dy * t; projPos[s2 * 3 + 2] = LAMP.z + dz * t;
      }
      projGeo.attributes.position.needsUpdate = true;
    }
    mobile.rotation.y = time * 0.25;
    mobile.children[0].rotation.z = Math.sin(time * 0.7) * 0.02;
    var pagesP = turning * 3, pf = pagesP - Math.floor(pagesP);
    leafPivot.rotation.z = Math.PI * smooth(0.1, 0.9, pf) * (pagesP >= 3 ? 0 : 1);
    leaf.visible = pagesP > 0.01 && pagesP < 2.99;
    dust.material.opacity = magic * 0.9;
    if (magic > 0.01) {
      book.updateMatrixWorld();
      for (var d2 = 0; d2 < dustN; d2++) {
        var ds = dustSeed[d2], life = (time * (0.15 + ds[1] * 0.2) + ds[0]) % 1, ang = ds[2] * 6.28 + life * 4;
        v1.set(Math.cos(ang) * (0.05 + life * 0.25), 0.04 + life * 0.8, Math.sin(ang) * (0.05 + life * 0.25)).applyMatrix4(book.matrixWorld);
        bedroom.worldToLocal(v1);
        dustPos[d2 * 3] = v1.x; dustPos[d2 * 3 + 1] = v1.y; dustPos[d2 * 3 + 2] = v1.z;
      }
      dustGeo.attributes.position.needsUpdate = true;
    }
    var trayOn = U_ > at(9, 0.95);
    book.visible = !trayOn;
    if (vis(tray, trayOn)) {
      tray.position.x = BED.x + 0.05 - (1 - trayIn) * 1.6;
      tray.position.y = F1 + 0.66 + Math.sin(trayIn * Math.PI) * 0.12;
      for (var sm = 0; sm < smokeN; sm++) {
        var sd3 = smokeSeed[sm], life2 = (time * 0.22 + sd3[0]) % 1;
        smokePos[sm * 3] = -0.16 + Math.sin(life2 * 6 + sd3[1] * 6) * 0.03 * life2 + life2 * 0.04;
        smokePos[sm * 3 + 1] = 0.08 + life2 * 0.5;
        smokePos[sm * 3 + 2] = 0.04 + Math.cos(life2 * 5 + sd3[1] * 4) * 0.03 * life2;
      }
      smokeGeo.attributes.position.needsUpdate = true;
      smoke.material.opacity = 0.45 * trayIn;
    }
    beamMat.uniforms.uA.value = trayOn && U_ < at(11, 0.3) ? day * (1 + nostalgia * 0.8) : 0;
    beam.visible = beamMat.uniforms.uA.value > 0.01;
    motes.points.visible = trayOn && U_ < at(11, 0.3);
    if (motes.points.visible) { f.snow = 0.3 + nostalgia * 0.7; motes.update(f, moteAt, env.reduceMotion); f.snow = rainAmt; }

    // ── Night glow of the dollhouse, the fort and the tent ──
    var dusk = beat(14, 0.85, 1.6), glow = Math.max(smooth(0.35, 0.85, dark), dusk * 0.7);
    shellMat.emissiveIntensity = glow * 0.16;
    var fortOn = glow * (U_ > at(8, 0.9) ? 1 : 0.35);
    fortLight.intensity = fortOn * 2.4 * (living.visible ? 1 : 0);
    fortGlow.material.opacity = fortOn * 0.55;
    fairyGlow.material.opacity = fortOn * 0.9;
    fairy.forEach(function (b, k) { b.material.color.set(fairyCols[k % fairyCols.length]).multiplyScalar(0.35 + fortOn * (0.65 + 0.2 * Math.sin(time * 2 + k))); });
    tentMat.emissiveIntensity = fortOn * 0.75;
    tentGlow.material.opacity = fortOn * 0.5;
    tentFlag.rotation.y = Math.sin(time * 1.3) * 0.3;

    // ── Kitchen ──
    var spin = (0.08 + whirl * 24) * (env.reduceMotion ? 0.3 : 1);
    minHand.rotation.z -= dt * spin;
    hourHand.rotation.z -= dt * spin / 12;
    calCalm.material.opacity = calm;
    calBusy.material.opacity = 1 - calm * 0.95;
    pages.visible = whirlIn > 0.001;
    if (pages.visible) {
      for (var pk2 = 0; pk2 < NPAGE; pk2++) {
        var pd2 = pageData[pk2], ang2 = pd2.a + time * pd2.sp * (1 + whirl * 2) * (env.reduceMotion ? 0.3 : 1);
        v1.set(2.1 + (r2(pk2) - 0.5) * 0.7, F0 + 1.4 + r2(pk2 + 7) * 0.6, -8.7);
        v2.set(3.0 + Math.cos(ang2) * pd2.rad, F0 + pd2.h + Math.sin(ang2 * 2) * 0.2, -6.4 + Math.sin(ang2) * pd2.rad * 0.8);
        v3.set(pd2.fx, F0 + 0.03 + pk2 * 0.0005, pd2.fz);
        var lo = smooth(pk2 / NPAGE * 0.5, pk2 / NPAGE * 0.5 + 0.5, whirlIn);
        v1.lerp(v2, lo).lerp(v3, calm);
        e1.set(lerp(lerp(0, pd2.tum + time * 2.1, lo), -Math.PI / 2, calm), lerp(lerp(0, pd2.tum * 2 + time * 1.3, lo), pd2.fr, calm), lerp(pd2.tum * lo, 0, calm));
        pages.setMatrixAt(pk2, m1.compose(v1, q1.setFromEuler(e1), s1.setScalar(lerp(0.3, 1, Math.max(lo, calm)))));
      }
      pages.instanceMatrix.needsUpdate = true;
    }

    // ── Night garden: fireflies, the lantern, the gate ──
    fireflies.material.opacity = smooth(0.6, 1, dark) * (U_ > at(4, 0.9) && U_ < at(6, 0.2) ? 1 : 0.3);
    if (fireflies.material.opacity > 0.01) {
      for (var fq = 0; fq < FF; fq++) {
        var fs = ffSeed;
        ffPos[fq * 3] = fs[fq * 4] + Math.sin(time * 0.5 + fs[fq * 4 + 3]) * 0.8;
        ffPos[fq * 3 + 1] = fs[fq * 4 + 1] + Math.sin(time * 0.9 + fs[fq * 4 + 3] * 2) * 0.3;
        ffPos[fq * 3 + 2] = fs[fq * 4 + 2] + Math.cos(time * 0.4 + fs[fq * 4 + 3]) * 0.8;
      }
      ffGeo.attributes.position.needsUpdate = true;
      fireflies.material.size = 0.25 + 0.08 * Math.sin(time * 3);
    }
    var lantern = Math.max(lit, smooth(0.5, 0.9, dark) * (U_ > at(5, 0.6) ? 1 : 0)) * (0.92 + 0.08 * Math.sin(time * 7) * Math.sin(time * 3.1));
    lanternCore.material.color.set('#5a4a3a').lerp(tc.set('#fff0b0'), lantern);
    lanternGlow.material.opacity = lantern * 0.9;
    lanternLight.intensity = lantern * 3.2 * smooth(0.2, 0.6, dark);
    neighWinMat.color.set('#43506a').lerp(tc.set('#ffc878'), Math.max(answer, smooth(0.5, 0.9, dark) * (U_ > at(6, 0) ? 1 : 0)) * smooth(0.2, 0.6, dark));
    gate.rotation.y = -gateOpen * 0.9;
    lampGlow.material.opacity = Math.max(smooth(0.25, 0.6, dark), dusk);

    // ── Butterflies, the ladybird ──
    butterflies.forEach(function (b, k) {
      var tb2 = time * 0.25 + b.ph;
      if (k === 2) b.g.position.set(LAWN.x + Math.sin(tb2 * 1.3) * 0.8, 0.35 + Math.sin(tb2 * 2.7) * 0.15, LAWN.z + 0.6 + Math.cos(tb2) * 0.5);
      else b.g.position.set(Math.sin(tb2) * 6 + (k ? -4 : 4), 0.9 + Math.sin(tb2 * 3.1) * 0.4, 7 + Math.cos(tb2 * 0.7) * 5);
      b.g.rotation.y = -tb2 * 1.3 + Math.PI / 2;
      var flap = Math.sin(time * 14 + k) * 0.9;
      b.w1.rotation.y = flap; b.w2.rotation.y = -flap;
      b.g.visible = dark < 0.5 && rainAmt < 0.5;
    });
    lady.position.set(ladyBlade.position.x - Math.sin(0.12) * 0.36 * climb, 0.04 + climb * 0.34, ladyBlade.position.z + 0.012);
    lady.rotation.set(-1.45, 0, 0.12);

    // ── Tea party ──
    teapot.rotation.set(0, -Math.PI / 2, -pour * 0.8);
    teapot.position.set(0.16, 0.62 + pour * 0.06, 0.02 + pour * 0.03);
    stream.visible = pour > 0.45;
    if (stream.visible) {
      stream.position.set(0.15, 0.655, 0.2);
      stream.scale.set(1, 0.1, 1);
    }
    teaLevel.position.y = 0.535 + poured * 0.035;
    teaLevel.visible = poured > 0.05;
    slice.position.set(lerp(-0.2, -0.12, served), 0.65 + Math.sin(served * Math.PI) * 0.14 - served * 0.07, lerp(0, 0.3, served));
    slice.rotation.y = served * 0.8;
    var bh = lerp(0.6, 3.4, lift) + Math.sin(time * 1.1) * 0.08;
    balloon.position.set(balloonTie.x + Math.sin(time * 0.7) * 0.1 * (0.4 + lift), balloonTie.y + bh, balloonTie.z + Math.cos(time * 0.5) * 0.08);
    balloon.scale.setScalar(lerp(0.55, 1.3, lift));
    balloon.rotation.set(0, Math.PI + Math.sin(time * 0.6) * 0.3, Math.sin(time * 0.9) * 0.08);
    var bs = balloonStr.geometry.attributes.position;
    bs.setXYZ(0, balloonTie.x, balloonTie.y, balloonTie.z);
    bs.setXYZ(1, balloon.position.x, balloon.position.y - 0.16 * balloon.scale.y, balloon.position.z);
    bs.needsUpdate = true;
    bunt.forEach(function (b, k) { b.rotation.x = Math.sin(time * 1.6 + k * 0.7) * 0.25 * (0.3 + wind); });

    // ── Beach ──
    towers.forEach(function (tw2, k) { var s3 = smooth(k * 0.18, k * 0.18 + 0.35, build); tw2.scale.set(1, Math.max(s3, 0.001), 1); tw2.visible = s3 > 0.001; });
    walls.forEach(function (wl, k) { var s4 = smooth(0.3 + k * 0.12, 0.5 + k * 0.12, build); wl.scale.set(1, Math.max(s4, 0.001), 1); wl.visible = s4 > 0.001; });
    keep.visible = help > 0.02;
    var arc = smooth(0, 1, help);
    bucket.position.set(lerp(0, -0.85, arc), 0.05 + 0.44 + Math.sin(arc * Math.PI) * 0.45 - arc * 0.25, lerp(0, 0.3, arc));
    bucket.rotation.set(0, 0, lerp(Math.PI, Math.PI * 2.5, arc));
    bucket.visible = build > 0.05 || U_ > at(7, 0);
    castleFlag.visible = flagUp > 0.01;
    castleFlag.scale.setScalar(Math.max(flagUp, 0.01));
    cfl.rotation.y = Math.sin(time * 4) * 0.25;
    pinHead.rotation.z -= dt * (2 + wind * 6) * (env.reduceMotion ? 0.3 : 1);

    // ── Race ──
    var zR = -11 + 14.4 * smooth(0, 0.92, raceT), zB = -11 + 13.5 * smooth(0.03, 1.0, raceT);
    redCar.g.position.set(RACE_X - 0.45, land(RACE_X, zR), zR);
    blueCar.g.position.set(RACE_X + 0.45, land(RACE_X, zB) + Math.abs(Math.sin(time * 7)) * 0.03 * second, zB);
    redCar.g.rotation.y = blueCar.g.rotation.y = -Math.PI / 2;
    blueCar.g.rotation.y += Math.sin(time * 5) * 0.12 * second;
    [redCar, blueCar].forEach(function (c, k) {
      var z = k ? zB : zR;
      c.wheels.forEach(function (w) { w.rotation.z = -(z + 11) / 0.094; });
      c.key.rotation.x = time * (raceT > 0 && raceT < 1 ? 6 : 0.5);
    });
    var broke = clamp((zR - (FINISH - 0.2)) / 0.5, 0, 1);
    tapeL.rotation.z = -broke * 1.35;
    tapeR.rotation.z = broke * 1.35;
    tapeL.rotation.y = tapeR.rotation.y = broke * 0.3;
    var redWin = beat(12, 0.48, 0.56);
    rosetteRed.scale.setScalar(Math.max(redWin, 0.001)); rosetteRed.visible = redWin > 0.01;
    rosetteBlue.scale.setScalar(Math.max(second, 0.001)); rosetteBlue.visible = second > 0.01;
    rosetteBlue.rotation.z = Math.sin(time * 4) * 0.15 * second;
    var cAge = since('confetti', at(12, 0.48));
    if (vis(confetti, cAge >= 0 && cAge < 6 && race.visible)) {
      for (var cq = 0; cq < CONF; cq++) {
        var cs2 = confSeed, tq = Math.min(cAge, 6);
        var drop = Math.max(0, cs2[cq * 5 + 1] * tq - 1.6 * tq * tq);
        v1.set(RACE_X + cs2[cq * 5] * tq * 0.8 + Math.sin(time * 2 + cs2[cq * 5 + 3]) * 0.1 * tq, Math.max(1.2 + drop - (tq > 1.2 ? (tq - 1.2) * 0.25 : 0), 0.04), FINISH + cs2[cq * 5 + 2] * tq * 0.8);
        e1.set(time * 3 + cs2[cq * 5 + 3], time * 2 + cs2[cq * 5 + 4], 0);
        confetti.setMatrixAt(cq, m1.compose(v1, q1.setFromEuler(e1), s1.setScalar(1)));
      }
      confetti.instanceMatrix.needsUpdate = true;
    }

    // ── Kite ──
    kitePos.lerpVectors(kiteLow, kiteHigh, soar);
    kitePos.x += Math.sin(time * 0.6) * 0.6; kitePos.y += Math.sin(time * 0.9) * 0.4;
    kite.position.copy(kitePos);
    kite.rotation.set(-0.35 + Math.sin(time * 1.3) * 0.08, 0.7, Math.sin(time * 0.8) * 0.18);
    kite.updateMatrixWorld();
    for (var kk = 0; kk < SN; kk++) {
      var su2 = kk / (SN - 1);
      v1.lerpVectors(stakeTop, kitePos, su2);
      v1.y -= Math.sin(su2 * Math.PI) * (1.2 - soar * 0.5);
      strPos[kk * 3] = v1.x; strPos[kk * 3 + 1] = v1.y; strPos[kk * 3 + 2] = v1.z;
    }
    kiteString.geometry.attributes.position.needsUpdate = true;
    v2.set(0, -1.0, 0).applyMatrix4(kite.matrixWorld);
    for (var tk = 0; tk < KT; tk++) {
      var tq2 = tk / (KT - 1);
      tailPos[tk * 3] = v2.x + tq2 * 1.6 + Math.sin(time * 3 - tk * 0.7) * 0.25 * tq2;
      tailPos[tk * 3 + 1] = v2.y - tq2 * 3.4;
      tailPos[tk * 3 + 2] = v2.z + tq2 * 1.2 + Math.cos(time * 2.4 - tk * 0.6) * 0.2 * tq2;
    }
    tailLine.geometry.attributes.position.needsUpdate = true;
    tailBows.forEach(function (b, k) {
      var idx = 2 + k * 2;
      b.position.set(tailPos[idx * 3], tailPos[idx * 3 + 1], tailPos[idx * 3 + 2]);
      b.rotation.set(0, 0.7, Math.sin(time * 3 - idx) * 0.4);
    });

    // ── Finale balloons ──
    balloonsUp.forEach(function (b, k) {
      var rise = smooth(k * 0.05, 0.6 + k * 0.05, finale);
      b.g.visible = rise > 0.001;
      b.g.position.set(b.x + Math.sin(time * 0.5 + b.ph) * 0.8 * rise, b.y + 1.5 + rise * (9 + k * 1.5) + Math.sin(time * 0.9 + b.ph) * 0.3, b.z);
    });

    // ── Blink ──
    blinkMat.uniforms.uA.value = clamp(blinkA, 0, 1);
    blinkMat.uniforms.uColor.value.copy(horizon).lerp(tc.set('#fff4e0'), 0.5);
    blink.visible = blinkA > 0.002;

    prevU = U_;
    gl.render(world, camera);
  }
  function r2(k) { return hash2(k, 11); }

  // ── Small pieces used above ────────────────────────────────────────────
  // A teddy bear sitting, facing +x before `ry`.
  function addTeddy(list, x, y, z, ry, s) {
    var c = '#c98a52', lt = '#ecc89a', parts = [
      [sph(0.15), c, 0, 0.14, 0, 1, 1.05, 0.9], [sph(0.11), c, 0.02, 0.36, 0], [sph(0.04), c, 0.0, 0.46, 0.08], [sph(0.04), c, 0.0, 0.46, -0.08],
      [sph(0.022), lt, 0.01, 0.46, 0.08, 1, 1, 0.5], [sph(0.022), lt, 0.01, 0.46, -0.08, 1, 1, 0.5],
      [sph(0.05), lt, 0.1, 0.34, 0, 1, 0.8, 1], [sph(0.016), '#3a2418', 0.15, 0.36, 0], [sph(0.012), '#1a1410', 0.1, 0.39, 0.045], [sph(0.012), '#1a1410', 0.1, 0.39, -0.045],
      [sph(0.08), lt, 0.1, 0.13, 0, 0.5, 1, 1], [new THREE.CapsuleGeometry(0.045, 0.1, 4, 8), c, 0.05, 0.18, 0.15, 1, 1, 1, 0.5], [new THREE.CapsuleGeometry(0.045, 0.1, 4, 8), c, 0.05, 0.18, -0.15, 1, 1, 1, -0.5],
      [new THREE.CapsuleGeometry(0.05, 0.1, 4, 8), c, 0.12, 0.02, 0.08, 1, 1, 1, 0, 0, Math.PI / 2], [new THREE.CapsuleGeometry(0.05, 0.1, 4, 8), c, 0.12, 0.02, -0.08, 1, 1, 1, 0, 0, Math.PI / 2],
      [new THREE.ConeGeometry(0.04, 0.06, 6).rotateZ(Math.PI / 2), '#e5484d', 0.1, 0.26, 0.035], [new THREE.ConeGeometry(0.04, 0.06, 6).rotateZ(-Math.PI / 2), '#e5484d', 0.1, 0.26, -0.035]
    ];
    var cr = Math.cos(ry), sr = Math.sin(ry);
    parts.forEach(function (p) {
      var g = p[0];
      if (p[5] != null) g.scale(p[5], p[6], p[7]);
      if (p[8] || p[9] || p[10]) g.applyMatrix4(_m.makeRotationFromEuler(_e.set(p[8] || 0, p[9] || 0, p[10] || 0)));
      g.scale(s, s, s);
      var lx = p[2] * s, lz = p[4] * s;
      g.rotateY(ry);
      g.translate(x + lx * cr + lz * sr, y + p[3] * s, z - lx * sr + lz * cr);
      list.push(tinted(g, p[1]));
    });
  }

  return {
    resize: function (w, h, dpr) {
      fitCamera(gl, camera, w, h, dpr, small);
      var pr = gl.getPixelRatio();
      pxScale.value = h * pr / (2 * Math.tan(camera.fov * Math.PI / 360));
      layout.portrait = w < h;
      layout.sx = w < h ? 0.45 : 1;
      placeAnimals(layout.sx);
    },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('toy-world', {
  renderer: renderer3d,
  scrim: 0.62,
  accent: '#ffd27a',
  align: ['left', 'right', 'left', 'left', 'left', 'left', 'left', 'left', 'right', 'right', 'left', 'left', 'right', 'right', 'center'],
  keys: function (T) {
    TL = T;
    var n = T.count, rows = [], S = { dark: 0, rain: 0, wind: 0.2, sun: 0.28, gold: 0.15, blink: 0, car: 0, py: 0, pp: 0 }, lastYaw = null;
    function at(i, frac) { i = Math.min(i, n - 1); return T.start(i) + frac * 1.6; }
    function K(u, p, t, env) {
      Object.keys(env || {}).forEach(function (k) { S[k] = env[k]; });
      var a = aim(p, t);
      if (lastYaw != null) { while (a[0] - lastYaw > Math.PI) a[0] -= Math.PI * 2; while (a[0] - lastYaw < -Math.PI) a[0] += Math.PI * 2; }
      lastYaw = a[0];
      //         unit x     dark    rain    wind    y     z     yaw   pitch sun    gold    blink    car    py    pp
      rows.push([u, p[0], S.dark, S.rain, S.wind, p[1], p[2], a[0], a[1], S.sun, S.gold, S.blink, S.car, S.py, S.pp]);
    }
    var tub = [-3.3, 3.95, -8.3], bed = [3.35, 4.25, -7.6];
    // Title: a summer morning over the garden, then into the bathroom
    K(0, [8, 8.5, 24], [-1, 3.6, -6]);
    K(0.7, [5.5, 7.4, 17], [-2, 4, -6]);
    // I: the bath, then down to the water and out to sea
    K(at(0, 0.14), [-3.5, 4.7, -6.6], [-3.45, 3.95, -8.25], { py: -0.24 });
    K(at(0, 0.42), [-3.55, 4.72, -7.0], [-3.5, 3.95, -8.25]);
    K(at(0, 0.6), [-4.02, 4.035, -8.22], [-1, 4.0, -8.5], { py: -0.12, pp: -0.22 });
    K(at(0, 0.86), [-4.0, 4.04, -8.2], [-1, 4.02, -8.3]);
    K(at(0, 1.0), [-3.25, 4.7, -6.2], [-3.3, 3.9, -8.4]);
    // II: out on the lawn, the clouds; then the rain and the puddles
    K(at(1, 0.1), [-1.8, 4.2, -0.6], [-2.8, 3.4, -8], { sun: 0.25, gold: 0, py: 0, pp: 0 });
    K(at(1, 0.26), [-1.0, 1.0, 4.0], [-3, 30, 60]);
    K(at(1, 0.52), [-1.0, 1.0, 4.1], [-3.2, 31, 60]);
    K(at(1, 0.66), [-0.9, 1.05, 4.4], [0.1, 0.0, 8.0], { rain: 1, wind: 0.45 });
    K(at(1, 0.92), [-0.85, 1.05, 4.6], [0.15, 0.0, 8.2]);
    // III: in the back of the car at dusk, a wishing star outside, the wind.
    // The cuts in and out sit in the gap between panels, under a quick dip.
    var cut = T.start(Math.min(2, n - 1));
    K(cut - 0.07, [-0.85, 1.05, 4.6], [0.15, 0.0, 8.2]);
    K(cut - 0.004, [-0.85, 1.05, 4.6], [0.15, 0.0, 8.2], { blink: 0.8 });
    K(cut + 0.004, [-0.85, 1.05, 4.6], [0.15, 0.0, 8.2], { car: 1, rain: 0, wind: 0.8, dark: 0.55, gold: 1, sun: 0.97 });
    K(cut + 0.08, [-0.85, 1.05, 4.6], [0.15, 0.0, 8.2], { blink: 0 });
    K(at(2, 0.6), [-0.85, 1.05, 4.6], [0.15, 0.0, 8.2], { dark: 0.62 });
    cut = T.start(Math.min(3, n - 1));
    K(cut - 0.07, [-0.85, 1.05, 4.6], [0.15, 0.0, 8.2]);
    // IV: back in the garden, morning at the dandelion jar; the day races by
    K(cut - 0.004, [5.7, 0.95, 8.15], [6.5, 0.82, 9.05], { blink: 0.8 });
    K(cut + 0.004, [5.7, 0.95, 8.15], [6.5, 0.82, 9.05], { car: 0, dark: 0, gold: 0.15, sun: 0.12, wind: 0.25, py: -0.3 });
    K(cut + 0.08, [5.7, 0.95, 8.15], [6.5, 0.83, 9.05], { blink: 0 });
    K(at(3, 0.45), [5.75, 0.93, 8.25], [6.52, 0.84, 9.1]);
    K(at(3, 0.62), [5.7, 0.95, 8.1], [6.5, 1.5, 10.5], { sun: 0.4, gold: 0.1, wind: 0.5, py: 0 });
    K(at(3, 0.9), [5.6, 1.0, 8.0], [6.8, 2.2, 11.4], { sun: 1.0, gold: 1, dark: 0.3 });
    K(at(3, 1.0), [5.0, 1.5, 6.2], bed, { dark: 0.75, wind: 0.2 });
    // V: bedtime upstairs; lightning; into the middle of the big bed
    K(at(4, 0.16), [2.2, 4.6, -4.4], bed, { dark: 1, gold: 0.4 });
    K(at(4, 0.5), [2.45, 4.55, -4.9], [3.3, 4.3, -7.6]);
    K(at(4, 0.62), [2.5, 4.5, -4.6], [5.8, 4.9, -5.7]);
    K(at(4, 0.86), [3.3, 4.42, -6.25], [3.3, 4.32, -8.6]);
    K(at(4, 1.0), [3.3, 4.45, -6.2], [3.3, 4.34, -8.6]);
    // VI: down into the night garden, the house glowing; then the gate
    K(at(5, 0.12), [2.4, 3.6, 1.0], [0, 2.6, -6]);
    K(at(5, 0.3), [0.6, 1.05, 9.5], [0, 2.6, -6]);
    K(at(5, 0.48), [0.4, 1.05, 11.2], [0, 2.4, -6]);
    K(at(5, 0.64), [-0.2, 1.05, 16.9], [5.6, 0.7, 29.4]);
    K(at(5, 0.94), [0.5, 1.05, 19.4], [5.6, 0.5, 29.4]);
    // VII: dawn, and into the kitchen
    K(at(6, 0.06), [1.8, 1.5, 5.0], [2.4, 1.6, -8], { dark: 0.3, sun: 0.12, gold: 0.3 });
    K(at(6, 0.22), [2.6, 1.35, -3.9], [1.6, 1.85, -8.8], { dark: 0, sun: 0.24, gold: 0.15, py: -0.15, pp: -0.22 });
    K(at(6, 0.6), [2.7, 1.35, -4.1], [1.7, 1.85, -8.8]);
    K(at(6, 0.95), [2.8, 1.3, -4.3], [1.6, 1.8, -8.8]);
    // VIII: out over the garden to the beach
    K(at(7, 0.08), [10, 4.0, 20], [22.5, 0.4, 40], { sun: 0.45, gold: 0 });
    K(at(7, 0.25), [20.6, 0.95, 37.4], [23.1, 0.3, 39.7], { py: -0.2, pp: 0.15 });
    K(at(7, 0.5), [20.9, 0.95, 37.8], [23.1, 0.35, 39.7]);
    K(at(7, 0.7), [21.25, 0.85, 38.25], [22.95, 0.42, 39.75]);
    K(at(7, 0.95), [21.3, 0.85, 38.3], [22.95, 0.42, 39.75]);
    // IX: back to the garden, down in the grass
    K(at(8, 0.1), [14, 2.5, 22], [7.8, 0.3, 14], { py: 0, pp: 0 });
    K(at(8, 0.3), [7.2, 0.45, 12.35], [7.62, 0.0, 13.55], { sun: 0.55 });
    K(at(8, 0.95), [7.22, 0.42, 12.45], [7.66, 0.02, 13.6]);
    // X: evening, the cushion fort; crawl inside
    K(at(9, 0.06), [-1.0, 1.5, 4.0], [-3.6, 0.9, -7], { dark: 0.6, gold: 0.8, sun: 0.95 });
    K(at(9, 0.22), [-3.0, 1.3, -2.6], [-3.5, 0.85, -7.0], { dark: 0.9 });
    K(at(9, 0.5), [-3.1, 1.25, -3.0], [-3.4, 0.8, -7.0]);
    K(at(9, 0.86), [-3.6, 0.62, -4.85], [-3.6, 0.55, -7.6]);
    K(at(9, 1.0), [-3.6, 0.62, -4.95], [-3.6, 0.55, -7.6]);
    // XI: morning, breakfast in bed
    K(at(10, 0.06), [-0.6, 2.6, 0.6], [2.5, 3.6, -6], { dark: 0.3, sun: 0.12, gold: 0.3 });
    K(at(10, 0.2), [2.85, 4.55, -5.85], [3.05, 4.15, -7.2], { dark: 0, sun: 0.22, gold: 0.25, py: -0.12, pp: 0.2 });
    K(at(10, 0.6), [2.95, 4.5, -6.1], [3.1, 4.15, -7.2]);
    K(at(10, 0.95), [2.3, 4.7, -4.9], [3.2, 4.2, -7.3], { gold: 0.55 });
    // XII: the tea party on the lawn; the balloon rises
    K(at(11, 0.02), [2.2, 4.9, -1.6], [-6.5, 0.6, 7], { sun: 0.35, gold: 0.15 });
    K(at(11, 0.12), [-4.0, 2.0, 2.5], [-6.5, 0.6, 7], { sun: 0.4, gold: 0.1, py: -0.1, pp: 0.1 });
    K(at(11, 0.25), [-6.5, 0.92, 6.38], [-6.2, 0.6, 7.6], { py: -0.25, pp: 0.22 });
    K(at(11, 0.55), [-6.48, 0.92, 6.4], [-6.2, 0.62, 7.6]);
    K(at(11, 0.72), [-6.5, 0.92, 6.3], [-6.2, 2.6, 8.2], { py: -0.1, pp: 0 });
    K(at(11, 0.95), [-6.5, 0.92, 6.25], [-6.2, 3.4, 8.2]);
    // XIII: the race
    K(at(12, 0.1), [-24, 5, 2], [-42, 0.3, -6], { sun: 0.55, py: 0 });
    K(at(12, 0.25), [RACE_X + 1.9, 0.42, FINISH + 1.3], [RACE_X - 0.2, 0.25, FINISH - 8]);
    K(at(12, 0.55), [RACE_X + 1.85, 0.42, FINISH + 1.35], [RACE_X - 0.1, 0.25, FINISH - 6]);
    K(at(12, 0.7), [RACE_X + 1.7, 0.6, FINISH + 3.6], [RACE_X + 1.0, 0.3, FINISH + 0.3], { py: 0.24, pp: 0.2 });
    K(at(12, 0.95), [RACE_X + 1.65, 0.6, FINISH + 3.65], [RACE_X + 1.0, 0.3, FINISH + 0.3]);
    // XIV: the piggy bank, then up to the kite
    K(at(13, 0.12), [-30, 3, 26], [-24.3, 0.4, 33.7], { sun: 0.7, gold: 0.35, py: 0.15 });
    K(at(13, 0.25), [-23.0, 0.85, 35.4], [-23.25, 0.35, 33.4], { py: 0.3, pp: 0.24 });
    K(at(13, 0.48), [-23.0, 0.85, 35.45], [-23.25, 0.38, 33.4]);
    K(at(13, 0.72), [-22.8, 0.95, 36.0], [-25.6, 5.4, 28.6], { py: 0.1, pp: 0 });
    K(at(13, 0.95), [-22.8, 1.0, 36.2], [-25.8, 7.0, 27.4]);
    // XV: rising over the whole toy world at golden hour
    K(at(14, 0.2), [-6, 12, 52], [-2, 2, 0], { sun: 0.85, gold: 0.9, py: 0 });
    K(at(14, 0.6), [30, 28, 72], [0, 10, 0], { gold: 1, sun: 0.88 });
    K(at(14, 1.0), [46, 34, 74], [0, 16, -2], { dark: 0.12 });
    K(T.total, [52, 36, 70], [0, 17, -3], { dark: 0.35, sun: 0.94 });
    return rows;
  },
  sound: {
    src: '/audio/birds.mp3',
    label: 'Play the garden birdsong and the music box',
    // Birdsong by day, hushed by rain and gone at night.
    volume: function (row) { return (0.03 + 0.16 * (1 - row[1])) * (1 - row[2] * 0.7); },
    cues: [
      { stanza: 1, at: 0.95, play: rainPatter },
      { stanza: 2, at: 0.3, play: wishChime },
      { stanza: 4, at: 0.2, play: musicBox([72, 72, 79, 79, 81, 81, 79, null, 77, 77, 76, 76, 74, 74, 72], 0.42) },
      { stanza: 4, at: 0.88, play: rumble },
      { stanza: 6, at: 0.2, play: ticking },
      { stanza: 10, at: 0.3, play: toasterPop },
      { stanza: 11, at: 0.35, play: clink },
      { stanza: 12, at: 0.8, play: partyHorn },
      { stanza: 14, at: 0.9, play: musicBox([72, 76, 79, 84, 79, 76, 77, 81, 84, 88], 0.3, 0.06) }
    ]
  }
});
