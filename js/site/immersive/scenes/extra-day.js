/*
 * Scene for "With All That Extra Time" (Poem_for_your_sprog): one extra
 * day, lived from dawn to dawn in a little seaside world, seen first-person
 * and travelled in one unbroken journey.
 *
 * Title  A quiet study at dawn, the wall clock ticking.
 * I      "With all that extra time...": the clock's second hand slows and
 *        stops, and the day calendar's 28 lifts away: February 29.
 * II     The writing pad on the desk by the window: a pen writes "A
 *        Pantomime" by itself, and the toy theatre's curtains part.
 * ·      Out through the window into the park, as a flock lifts out of the
 *        trees and wheels across the sky.
 * ·      Sitting on the bench on the bluff: the sun sets over the sea, dark
 *        falls and you look up as the stars come out.
 * ·      On the beach two little handmade boats with lanterns slip into the
 *        water and sail away down the moon's path.
 * ·      "With me, and only me": one chair and a lantern at the end of the
 *        jetty.
 * ·      Rising and turning back, the moonlit coast of homes, hills and
 *        lakes becomes a silver painting.
 * ·      Down to a cottage window and into its warm kitchen before dawn: a
 *        cake on its stand, cupcakes, a glossy baking magazine.
 * ·      Out again and into the sea at sunrise, swimming at the waterline
 *        through foam and spray.
 * ·      A door standing in the sea swings open on the sun; through it you
 *        rise and turn to see the whole world of the day laid out.
 * III    Lights wake at each place the day went: the room, the bench, the
 *        boats, the chair, the kitchen, the door.
 * ·      "I'd live my dreams...": everything softens into a dream-light
 *        that never quite finishes.
 *
 * The world renders into a target and a final pass paints it silver or
 * melts it into dream-light. The sky is a palette index ("look") blended
 * between ten looks. Columns:
 *   [unit, way, look, swell, wind, yaw, pitch, tick, page, write, curtain,
 *    open, birds, boats, silver, door, lights, dream]
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, broadleafGeometry, softSprite, skyDome,
         terrain, ribbon, scatter, particleField, oceanMaterial, oceanMesh, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── The land (metres; the sea is -z, its surface y = 0) ─────────────────
var G = 2.2;                                            // the garden and the study floor
var ROOM = { hw: 2.6, d: 4.6, h: 2.7 }, WIN = { hw: 0.75, y0: 0.95, y1: 2.25 };
var KIT = { x: 20, z: -69.4, hw: 2.4, hd: 2.0, h: 2.6 }; // the kitchen; its sea wall is at z - hd
var JETTY = { x: 8, z0: -79, z1: -112, y: 1.4 };
var BENCH = { x: -1, z: -57.2 };
var DOOR = { x: 18, z: -137 };
var LAKES = [[62, -8, 15], [94, 24, 8], [36, 16, 6]];

function hills(x, z) {
  var inland = smooth(-76, -44, z), right = smooth(12, 64, x), left = smooth(-12, -64, x);
  var vale = 7 * Math.exp(-((x - 62) * (x - 62) + (z + 8) * (z + 8)) / 1800);   // the lake valley
  return inland * (right * (10 + 5 * Math.sin(z * 0.035 + 0.6) + 3 * Math.sin(x * 0.045) - vale) +
                   left * (8 + 4 * Math.sin(z * 0.05 + 2) + 3 * Math.cos(x * 0.04))) +
         smooth(30, 140, z) * (16 + 8 * Math.sin(x * 0.021 + 1));
}
// Garden, the park rising to the bluff, the beach running down into the sea.
function profile(z) {
  var h = G + 2.8 * smooth(-6, -56, z);
  h = lerp(h, 2.3, smooth(-57, -71, z));
  h = lerp(h, -0.25, smooth(-71, -88, z));
  return lerp(h, -3.5, smooth(-88, -115, z));
}
function rawLand(x, z) {
  return profile(z) + hills(x, z) +
         0.4 * Math.sin(x * 0.13 + 1) * Math.cos(z * 0.11) * smooth(-8, -16, z) * (1 - smooth(-46, -56, z));
}
var LAKE_Y = LAKES.map(function (l) { return rawLand(l[0], l[1]) - 0.5; });
function land(x, z) {
  var h = rawLand(x, z);
  for (var i = 0; i < LAKES.length; i++) {
    var L = LAKES[i], d = Math.hypot(x - L[0], z - L[1]) / L[2];
    if (d < 1.8) h = lerp(d < 1 ? LAKE_Y[i] - 1.4 * (1 - d * d * 0.6) : LAKE_Y[i] + 0.25, h, smooth(1.0, 1.8, d));
  }
  return h;
}
var KF = Math.max(land(KIT.x - KIT.hw, KIT.z - KIT.hd), land(KIT.x + KIT.hw, KIT.z - KIT.hd),
                  land(KIT.x - KIT.hw, KIT.z + KIT.hd), land(KIT.x + KIT.hw, KIT.z + KIT.hd)) + 0.15;

// The sea: swells rolling in to the shore (+z).
var WAVES = [[0.2, 1, 0.32, 0.22, -1.1], [-0.5, 1, 0.55, 0.12, -1.6], [0.7, 1, 0.95, 0.07, -2.2], [-0.3, 1, 1.7, 0.035, -3.0]];
var WN = WAVES.map(function (w) { var l = Math.hypot(w[0], w[1]); return [w[0] / l, w[1] / l]; });
function wave(x, z, time, amp, out) {
  var h = 0, dx = 0, dz = 0;
  for (var i = 0; i < WAVES.length; i++) {
    var w = WAVES[i], ux = WN[i][0], uz = WN[i][1], ph = (ux * x + uz * z) * w[2] + time * w[4], a = w[3] * amp;
    h += a * Math.sin(ph);
    var c = a * w[2] * Math.cos(ph);
    dx += c * ux; dz += c * uz;
  }
  out.h = h; out.dx = dx; out.dz = dz;
  return out;
}

// ── The way: camera positions and what it looks at, one per waypoint ────
function E(x, z, eye) { return land(x, z) + eye; }
var WAY = [
  [[0.3, G + 1.55, 3.95], [0.6, G + 1.42, 0]],                       // 0  the study at dawn
  [[0.8, G + 1.6, 2.5], [1.5, G + 1.58, 0]],                         // 1  the clock and the calendar
  [[-0.05, G + 1.36, 1.42], [0.02, G + 0.8, 0.3]],                   // 2  over the desk
  [[0, G + 1.55, 0.6], [0, G + 1.5, -6]],                            // 3  up to the open window
  [[0, G + 1.62, -1.4], [0, G + 1.45, -12]],                         // 4  through it
  [[-1.6, E(-1.6, -12, 1.65), -12], [1.5, E(1.5, -30, 2.2), -30]],   // 5  the garden path
  [[0.2, E(0.2, -26, 1.65), -26], [5, E(5, -42, 7), -42]],           // 6  the park; the birds
  [[-0.8, E(-0.8, -44, 1.65), -44], [-1.4, E(-1.4, -60, 1.2), -60]], // 7  on to the bench
  [[-1, E(-1, -56.95, 1.18), -56.95], [-3, 1.4, -130]],              // 8  sitting; the sunset
  [[-1, E(-1, -57, 1.18), -57], [-3, 4, -130]],                      // 9  still sitting; the stars
  [[-2.8, E(-2.8, -69, 1.65), -69], [-2, 0.6, -100]],                // 10 down the slope
  [[-3.6, E(-3.6, -80.5, 1.6), -80.5], [2, 0.4, -110]],              // 11 the boats at the waterline
  [[4.6, E(4.6, -79.5, 1.65), -79.5], [8.5, 1.8, -100]],             // 12 along the beach to the jetty
  [[8.2, JETTY.y + 1.6, -104.5], [9.0, JETTY.y + 0.5, -112]],        // 13 on the jetty: the chair
  [[11, 12, -84], [9, 3, -130]],                                     // 14 pulling back and up
  [[34, 26, -12], [32, 2, -110]],                                    // 15 over the coast
  [[60, 56, 100], [46, 4, -50]],                                     // 16 the silver scene
  [[20.6, KF + 2.0, -59], [20.6, KF + 1.35, -70]],                   // 17 down to the kitchen window
  [[20.6, KF + 1.5, -68.2], [19.3, KF + 0.85, -69.7]],               // 18 inside, the table
  [[20.6, KF + 1.55, -71.1], [20.4, KF + 1.3, -80]],                 // 19 to the sea window
  [[19.5, 1.8, -86], [19, 0.6, -112]],                               // 20 down the beach to the water
  [[19, 0.5, -97], [18.6, 0.7, -130]],                               // 21 swimming out
  [[17.8, 0.5, -116], [17.2, 0.9, -140]],                           // 22
  [[17.3, 0.5, -129.5], [16.4, 1.1, -140]],                          // 23 the door ahead
  [[18, 1.15, -137.4], [18, 2.0, -150]],                             // 24 through it
  [[16, 12, -152], [36, 12, -148]],                                  // 25 rising, turning back
  [[14, 34, -172], [18, 6, -50]],                                    // 26 the whole world
  [[14, 42, -160], [19, 10, -36]],                                   // 27 drifting
  [[14, 50, -148], [19, 16, -26]]                                    // 28 into the dream
];
var pathP = new THREE.CatmullRomCurve3(WAY.map(function (w) { return new THREE.Vector3(w[0][0], w[0][1], w[0][2]); }), false, 'centripetal');
var pathT = new THREE.CatmullRomCurve3(WAY.map(function (w) { return new THREE.Vector3(w[1][0], w[1][1], w[1][2]); }), false, 'centripetal');

// ── The day, in ten looks ────────────────────────────────────────────────
var LOOKS = [
  { // 0 dawn, in the study
    top: '#6a7fb2', mid: '#c4afc2', horizon: '#f7c9a4', glow: '#ffd2a0', glowAmt: 0.9, el: 0.035, az: 0.12, sunC: '#ffcf9e', sunI: 1.6,
    hemiSky: '#c0c8e2', hemiGnd: '#6a5a4a', hemiI: 1.0, fog: '#dcc6c4', fogD: 0.006, sea: '#3a4c6a', cloud: '#f4c8b4', night: 0, exp: 1.0 },
  { // 1 morning
    top: '#4f86cc', mid: '#a6c8ea', horizon: '#f2e8d2', glow: '#fff0d0', glowAmt: 1.0, el: 0.3, az: -0.4, sunC: '#fff0da', sunI: 2.4,
    hemiSky: '#d0e0f6', hemiGnd: '#6a6a44', hemiI: 1.3, fog: '#cfdcea', fogD: 0.004, sea: '#2f5a7a', cloud: '#ffffff', night: 0, exp: 1.0 },
  { // 2 the afternoon in the park
    top: '#4482d4', mid: '#98c2ee', horizon: '#e2ecf4', glow: '#fff6e0', glowAmt: 0.9, el: 0.75, az: -0.7, sunC: '#fff6ea', sunI: 2.6,
    hemiSky: '#d8e6fa', hemiGnd: '#5f6e3e', hemiI: 1.3, fog: '#c6d8ec', fogD: 0.0032, sea: '#2a5a80', cloud: '#ffffff', night: 0, exp: 1.0 },
  { // 3 sunset from the bench
    top: '#38467e', mid: '#c27a82', horizon: '#ffb070', glow: '#ffa858', glowAmt: 1.3, el: 0.025, az: -0.5, sunC: '#ffb478', sunI: 1.8,
    hemiSky: '#c89ab0', hemiGnd: '#4a3a3a', hemiI: 1.0, fog: '#d8a088', fogD: 0.004, sea: '#3a3a5a', cloud: '#ffb48a', night: 0.15, exp: 1.0 },
  { // 4 night, moonlit
    top: '#050918', mid: '#101a3a', horizon: '#26345e', glow: '#000000', glowAmt: 0, el: 0.2, az: 0.25, sunC: '#a8b8ea', sunI: 0.9,
    hemiSky: '#4a5a92', hemiGnd: '#141822', hemiI: 0.9, fog: '#18213e', fogD: 0.005, sea: '#0e1830', cloud: '#2a3458', night: 1, exp: 1.1 },
  { // 5 the silver scene
    top: '#182032', mid: '#4a5672', horizon: '#9aa6bc', glow: '#000000', glowAmt: 0, el: 0.085, az: -0.04, sunC: '#e2e8f8', sunI: 1.6,
    hemiSky: '#b2bcd6', hemiGnd: '#4a4e5a', hemiI: 1.7, fog: '#7a849c', fogD: 0.0035, sea: '#2a3448', cloud: '#7a86a2', night: 0.85, exp: 1.12 },
  { // 6 before dawn
    top: '#101634', mid: '#363a68', horizon: '#ae7a88', glow: '#ff9a70', glowAmt: 0.25, el: -0.02, az: 0.09, sunC: '#ff9a82', sunI: 0.45,
    hemiSky: '#5a5a8c', hemiGnd: '#2a2228', hemiI: 0.9, fog: '#4a4668', fogD: 0.005, sea: '#1c2240', cloud: '#5a4a6a', night: 0.6, exp: 1.08 },
  { // 7 sunrise at sea
    top: '#4a5c9a', mid: '#dc9ea2', horizon: '#ffcf92', glow: '#ffd084', glowAmt: 1.4, el: 0.05, az: 0.09, sunC: '#ffc890', sunI: 2.0,
    hemiSky: '#e8c0b8', hemiGnd: '#5a4a3a', hemiI: 1.2, fog: '#e8b8a2', fogD: 0.004, sea: '#1c3c70', cloud: '#ffc8a8', night: 0, exp: 1.0 },
  { // 8 the morning of the extra day
    top: '#3f7ad0', mid: '#94bce8', horizon: '#e4ecf2', glow: '#fff0c8', glowAmt: 1.0, el: 0.2, az: 0.05, sunC: '#fff0d8', sunI: 2.6,
    hemiSky: '#d0e0f6', hemiGnd: '#6a6448', hemiI: 1.25, fog: '#cfdcea', fogD: 0.0022, sea: '#24547e', cloud: '#ffffff', night: 0, exp: 1.0 },
  { // 9 dream-light
    top: '#5a5e9e', mid: '#d4a0a8', horizon: '#ffd2a2', glow: '#ffe2b0', glowAmt: 1.2, el: 0.12, az: 0.05, sunC: '#ffd8a8', sunI: 2.0,
    hemiSky: '#f0c8b8', hemiGnd: '#7a6050', hemiI: 1.2, fog: '#e2b49a', fogD: 0.0032, sea: '#3a4a78', cloud: '#ffd8c0', night: 0.1, exp: 0.88 }
];
var COLOR_KEYS = ['top', 'mid', 'horizon', 'glow', 'sunC', 'hemiSky', 'hemiGnd', 'fog', 'sea', 'cloud'];
var NUM_KEYS = ['glowAmt', 'el', 'az', 'sunI', 'hemiI', 'fogD', 'night', 'exp'];
LOOKS.forEach(function (l) { COLOR_KEYS.forEach(function (k) { l[k] = new THREE.Color(l[k]); }); });
function nightAt(L) {
  L = clamp(L, 0, LOOKS.length - 1);
  var i = Math.min(Math.floor(L), LOOKS.length - 2);
  return lerp(LOOKS[i].night, LOOKS[i + 1].night, L - i);
}

var NOISE = [
  'float hh(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }',
  'float vn(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);',
  '  return mix(mix(hh(i), hh(i + vec2(1.0, 0.0)), f.x), mix(hh(i + vec2(0.0, 1.0)), hh(i + vec2(1.0, 1.0)), f.x), f.y); }'
].join('\n');

// ── Sound: the clock, then small synthesised cues for each beat ──────────
function noiseBuf(ac, len, shape) {
  var b = ac.createBuffer(1, Math.ceil(ac.sampleRate * len), ac.sampleRate), d = b.getChannelData(0);
  for (var i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * shape(i / d.length);
  return b;
}
function click(ac, out, t, freq, gain) {
  var src = ac.createBufferSource(), bp = ac.createBiquadFilter(), g = ac.createGain();
  src.buffer = noiseBuf(ac, 0.035, function (p) { return Math.pow(1 - p, 6); });
  bp.type = 'bandpass'; bp.frequency.value = freq; bp.Q.value = 5;
  g.gain.value = gain;
  src.connect(bp); bp.connect(g); g.connect(out);
  src.start(t);
}
// Tick-tock at `times` (seconds from now).
function ticks(times, gain) {
  return function (ac, out) {
    var t0 = ac.currentTime;
    times.forEach(function (dt, k) { click(ac, out, t0 + dt, k % 2 ? 2500 : 3300, gain * (1 - 0.4 * k / times.length)); });
  };
}
function bell(ac, out, f, at, gain, len) {
  [[1, 1], [2.01, 0.3], [3.03, 0.1]].forEach(function (p) {
    var o = ac.createOscillator(), g = ac.createGain();
    o.type = 'sine';
    o.frequency.value = f * p[0];
    g.gain.setValueAtTime(0.0001, at);
    g.gain.exponentialRampToValueAtTime(gain * p[1], at + 0.015);
    g.gain.exponentialRampToValueAtTime(0.0001, at + len / p[0]);
    o.connect(g); g.connect(out);
    o.start(at); o.stop(at + len + 0.1);
  });
}
// The calendar page: a paper flick and a small bright bell.
function pageTurn(ac, out) {
  var t = ac.currentTime, src = ac.createBufferSource(), bp = ac.createBiquadFilter(), g = ac.createGain();
  src.buffer = noiseBuf(ac, 0.3, function (p) { return Math.sin(p * Math.PI) * (0.6 + 0.4 * Math.sin(p * 40)); });
  bp.type = 'bandpass'; bp.Q.value = 1.2;
  bp.frequency.setValueAtTime(1800, t);
  bp.frequency.exponentialRampToValueAtTime(4200, t + 0.25);
  g.gain.value = 0.25;
  src.connect(bp); bp.connect(g); g.connect(out);
  src.start(t);
  bell(ac, out, 1318.5, t + 0.3, 0.05, 3);
  bell(ac, out, 1975.5, t + 0.48, 0.03, 3);
}
// The toy theatre: a music-box phrase.
function musicBox(ac, out) {
  var t = ac.currentTime;
  [784, 987.8, 1174.7, 1568, 1318.5, 1174.7, 987.8, 1174.7].forEach(function (f, k) { bell(ac, out, f, t + k * 0.19, 0.045, 1.6); });
}
// Wings: a flurry of soft feathery bursts.
function flutter(ac, out) {
  var t = ac.currentTime;
  for (var k = 0; k < 26; k++) {
    var at = t + k * 0.055 + Math.random() * 0.03, src = ac.createBufferSource(), lp = ac.createBiquadFilter(), g = ac.createGain();
    src.buffer = noiseBuf(ac, 0.07, function (p) { return Math.sin(p * Math.PI); });
    lp.type = 'lowpass'; lp.frequency.value = 900 + Math.random() * 700;
    g.gain.value = 0.22 * Math.sin(k / 26 * Math.PI);
    src.connect(lp); lp.connect(g); g.connect(out);
    src.start(at);
  }
}
// The first stars: slow, high, far apart.
function starNotes(ac, out) {
  var t = ac.currentTime;
  [1567.98, 1174.66, 1760, 1318.51, 2093].forEach(function (f, k) { bell(ac, out, f, t + k * 0.7 + Math.random() * 0.2, 0.03, 3.5); });
}
// A long soft wash of water: boats launching, a wave through the swim.
function wash(gain, len) {
  return function (ac, out) {
    var t = ac.currentTime, src = ac.createBufferSource(), lp = ac.createBiquadFilter(), g = ac.createGain();
    src.buffer = noiseBuf(ac, len, function (p) { return Math.pow(Math.sin(p * Math.PI), 1.5); });
    lp.type = 'lowpass';
    lp.frequency.setValueAtTime(500, t);
    lp.frequency.linearRampToValueAtTime(1600, t + len * 0.4);
    lp.frequency.linearRampToValueAtTime(400, t + len);
    g.gain.value = gain;
    src.connect(lp); lp.connect(g); g.connect(out);
    src.start(t);
  };
}
function doorChord(ac, out) {
  var t = ac.currentTime;
  [523.25, 659.25, 783.99, 1046.5].forEach(function (f, k) { bell(ac, out, f, t + k * 0.12, 0.04, 5); });
}
// The dream: a suspended chord that swells and fades, never resolving.
function dreamPad(ac, out) {
  var t = ac.currentTime;
  [261.63, 392, 587.33, 783.99].forEach(function (f, k) {
    [0, 3].forEach(function (cents) {
      var o = ac.createOscillator(), g = ac.createGain();
      o.type = 'sine';
      o.frequency.value = f * Math.pow(2, cents / 1200);
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(0.022 / (1 + k * 0.4), t + 3 + k * 0.5);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 14);
      o.connect(g); g.connect(out);
      o.start(t); o.stop(t + 14.2);
    });
  });
}

// ── Canvas textures ──────────────────────────────────────────────────────
function canvasTex(w, h, paint, repeat) {
  var c = document.createElement('canvas');
  c.width = w; c.height = h;
  paint(c.getContext('2d'), w, h);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(repeat[0], repeat[1]); }
  return t;
}
var SERIF = 'Georgia, "Times New Roman", serif';

function clockFace(x, w) {
  var c = w / 2;
  x.fillStyle = '#f3ead6'; x.beginPath(); x.arc(c, c, c, 0, Math.PI * 2); x.fill();
  x.strokeStyle = '#3a3026';
  for (var k = 0; k < 60; k++) {
    var a = k / 60 * Math.PI * 2, r0 = k % 5 ? c * 0.86 : c * 0.8;
    x.lineWidth = k % 5 ? 1.5 : 4;
    x.beginPath(); x.moveTo(c + Math.sin(a) * r0, c - Math.cos(a) * r0); x.lineTo(c + Math.sin(a) * c * 0.92, c - Math.cos(a) * c * 0.92); x.stroke();
  }
  x.fillStyle = '#2e261e'; x.font = '600 ' + Math.round(w * 0.11) + 'px ' + SERIF; x.textAlign = 'center'; x.textBaseline = 'middle';
  for (k = 1; k <= 12; k++) { var b = k / 12 * Math.PI * 2; x.fillText(String(k), c + Math.sin(b) * c * 0.64, c - Math.cos(b) * c * 0.64 + 2); }
}
function calendarPage(day, weekday, red, note) {
  return canvasTex(160, 210, function (x, w, h) {
    x.fillStyle = '#fbf8f0'; x.fillRect(0, 0, w, h);
    x.fillStyle = '#b8322a'; x.fillRect(0, 0, w, 44);
    x.fillStyle = '#fff6ea'; x.font = '600 21px ' + SERIF; x.textAlign = 'center'; x.textBaseline = 'middle';
    x.fillText('FEBRUARY', w / 2, 23);
    x.fillStyle = red ? '#b8322a' : '#2a2622'; x.font = '700 92px ' + SERIF;
    x.fillText(day, w / 2, 112);
    x.fillStyle = '#5a5248'; x.font = 'italic 19px ' + SERIF;
    x.fillText(weekday, w / 2, 172);
    if (note) { x.fillStyle = '#b8322a'; x.font = 'italic 15px ' + SERIF; x.fillText(note, w / 2, 195); }
    x.strokeStyle = 'rgba(0,0,0,0.12)'; x.lineWidth = 2; x.strokeRect(1, 1, w - 2, h - 2);
  });
}
var PAD = { w: 0.22, h: 0.3, px: 256, py: 350, row0: 50, rowH: 26, rows: 10 };
function padPaper(x, w, h) {
  x.fillStyle = '#f8f4e8'; x.fillRect(0, 0, w, h);
  x.strokeStyle = 'rgba(120,150,200,0.55)'; x.lineWidth = 1.4;
  for (var k = 0; k <= PAD.rows + 1; k++) { var y = PAD.row0 + k * PAD.rowH + 22; x.beginPath(); x.moveTo(0, y); x.lineTo(w, y); x.stroke(); }
  x.strokeStyle = 'rgba(210,90,90,0.6)'; x.beginPath(); x.moveTo(34, 0); x.lineTo(34, h); x.stroke();
  x.fillStyle = '#3a3a44'; x.fillRect(0, 0, w, 16);
  x.fillStyle = '#c8c8d0';
  for (k = 0; k < 11; k++) { x.beginPath(); x.ellipse(16 + k * 22.5, 14, 4, 8, 0, 0, Math.PI * 2); x.fill(); }
}
// The words, written by themselves, one line at a time (alpha = ink).
function padInk(x, w, h) {
  var r = rng(77);
  x.fillStyle = '#1c2a5a'; x.strokeStyle = '#1c2a5a';
  x.font = 'italic 600 27px ' + SERIF; x.textBaseline = 'alphabetic';
  x.fillText('A Pantomime', 46, PAD.row0 + 19);
  x.lineWidth = 2; x.lineCap = 'round';
  for (var k = 2; k < PAD.rows; k++) {
    var y = PAD.row0 + k * PAD.rowH + 18, end = 210 + r() * 30;
    if (k === 5) end = 120;
    var px = 44;
    x.beginPath(); x.moveTo(px, y);
    // A cursive-looking scribble: loops and small gaps between "words".
    while (px < end) {
      var wl = 18 + r() * 34;
      for (var s = 0; s < wl; s += 2) {
        x.lineTo(px + s, y - Math.abs(Math.sin((px + s) * 0.55 + r() * 0.3)) * (5 + r() * 4) + Math.sin((px + s) * 0.21) * 1.5);
      }
      px += wl + 7;
      x.moveTo(px, y);
    }
    x.stroke();
  }
}
function wallpaper(x, w, h) {
  x.fillStyle = '#e9dcc4'; x.fillRect(0, 0, w, h);
  for (var k = 0; k < 8; k++) { x.fillStyle = k % 2 ? 'rgba(170,140,100,0.08)' : 'rgba(255,255,255,0.08)'; x.fillRect(k * w / 8, 0, w / 16, h); }
  for (k = 0; k < 16; k++) {
    var cx = (k % 4) * w / 4 + (Math.floor(k / 4) % 2) * w / 8 + w / 16, cy = Math.floor(k / 4) * h / 4 + h / 8;
    x.fillStyle = 'rgba(120,150,100,0.35)';
    x.beginPath(); x.ellipse(cx - 5, cy + 4, 5, 2, -0.6, 0, 6.3); x.fill();
    x.beginPath(); x.ellipse(cx + 5, cy + 4, 5, 2, 0.6, 0, 6.3); x.fill();
    x.fillStyle = 'rgba(200,120,120,0.4)';
    x.beginPath(); x.arc(cx, cy, 3.2, 0, 6.3); x.fill();
  }
}
function boards(x, w, h) {
  var r = rng(9);
  for (var k = 0; k < 8; k++) {
    var l = 0.36 + r() * 0.12;
    x.fillStyle = 'rgb(' + Math.round(150 * l + 40) + ',' + Math.round(100 * l + 28) + ',' + Math.round(62 * l + 18) + ')';
    x.fillRect(0, k * h / 8, w, h / 8);
    x.fillStyle = 'rgba(40,24,12,0.5)'; x.fillRect(0, k * h / 8, w, 2);
    x.fillRect(r() * w, k * h / 8, 2, h / 8);
  }
}
function tiles(base, line, n) {
  return function (x, w, h) {
    x.fillStyle = base; x.fillRect(0, 0, w, h);
    x.strokeStyle = line; x.lineWidth = 3;
    for (var k = 0; k <= n; k++) { x.beginPath(); x.moveTo(k * w / n, 0); x.lineTo(k * w / n, h); x.moveTo(0, k * h / n); x.lineTo(w, k * h / n); x.stroke(); }
  };
}
function checker(x, w, h) {
  for (var i = 0; i < 8; i++) for (var j = 0; j < 8; j++) { x.fillStyle = (i + j) % 2 ? '#e8dcc6' : '#a8563c'; x.fillRect(i * w / 8, j * h / 8, w / 8, h / 8); }
}
// The toy theatre's proscenium: red with gold, mapped over x -0.2..0.2, y 0..0.34.
function proscenium(x, w, h) {
  x.fillStyle = '#9c2a26'; x.fillRect(0, 0, w, h);
  function X(v) { return (v + 0.2) / 0.4 * w; }
  function Y(v) { return (1 - v / 0.34) * h; }
  x.strokeStyle = '#e2b450'; x.lineWidth = 6; x.strokeRect(6, 6, w - 12, h - 12);
  x.lineWidth = 7;
  x.beginPath(); x.moveTo(X(-0.15), Y(0.04)); x.lineTo(X(-0.15), Y(0.22)); x.quadraticCurveTo(X(0), Y(0.305), X(0.15), Y(0.22)); x.lineTo(X(0.15), Y(0.04)); x.stroke();
  x.fillStyle = '#e2b450';
  for (var k = 0; k < 9; k++) { x.beginPath(); x.arc(X(-0.18), Y(0.04 + k * 0.03), 3, 0, 6.3); x.fill(); x.beginPath(); x.arc(X(0.18), Y(0.04 + k * 0.03), 3, 0, 6.3); x.fill(); }
  x.beginPath(); x.arc(X(0), Y(0.318), 9, 0, 6.3); x.fill();
}
function backdrop(x, w, h) {
  var g = x.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0, '#1a2050'); g.addColorStop(0.7, '#4a3a78'); g.addColorStop(1, '#6a4a7a');
  x.fillStyle = g; x.fillRect(0, 0, w, h);
  var r = rng(5);
  x.fillStyle = '#fff4d0';
  for (var k = 0; k < 40; k++) { x.beginPath(); x.arc(r() * w, r() * h * 0.6, 0.8 + r() * 1.6, 0, 6.3); x.fill(); }
  x.beginPath(); x.arc(w * 0.72, h * 0.22, 18, 0, 6.3); x.fill();
  x.fillStyle = '#1e2454'; x.beginPath(); x.arc(w * 0.72 + 9, h * 0.22 - 5, 16, 0, 6.3); x.fill();
  ['#2a4a8a', '#3a64a8', '#4a7cc0'].forEach(function (c, j) {
    x.fillStyle = c; x.beginPath(); x.moveTo(0, h);
    for (var s = 0; s <= w; s += 8) x.lineTo(s, h * (0.68 + j * 0.1) + Math.sin(s * 0.08 + j * 2) * 6);
    x.lineTo(w, h); x.fill();
  });
}
function wingFlat(x, w, h) {
  x.fillStyle = '#2a5a3a';
  for (var k = 0; k < 4; k++) { x.beginPath(); x.ellipse(w * 0.5, h * (0.15 + k * 0.22), w * (0.42 - k * 0.04), h * 0.14, 0, 0, 6.3); x.fill(); }
  x.fillStyle = '#4a3020'; x.fillRect(w * 0.42, h * 0.75, w * 0.16, h * 0.25);
}
function curtainCloth(x, w, h) {
  for (var s = 0; s < w; s++) { var v = 0.55 + 0.45 * Math.sin(s / w * Math.PI * 7); x.fillStyle = 'rgb(' + Math.round(110 + 80 * v) + ',' + Math.round(14 + 16 * v) + ',' + Math.round(26 + 18 * v) + ')'; x.fillRect(s, 0, 1, h); }
  x.fillStyle = '#e2b450'; x.fillRect(0, h - 10, w, 10);
}
function cutout(kind) {
  return canvasTex(128, 64, function (x, w, h) {
    if (kind === 'boat') {
      x.fillStyle = '#f2ead6'; x.beginPath(); x.moveTo(30, 40); x.lineTo(98, 40); x.lineTo(86, 54); x.lineTo(40, 54); x.fill();
      x.beginPath(); x.moveTo(64, 6); x.lineTo(64, 38); x.lineTo(92, 38); x.fill();
      x.fillStyle = '#c84a3a'; x.beginPath(); x.moveTo(62, 10); x.lineTo(62, 38); x.lineTo(40, 38); x.fill();
    } else if (kind === 'wave') {
      x.fillStyle = '#5a8ad0'; x.beginPath(); x.moveTo(0, h);
      for (var s = 0; s <= w; s += 4) x.lineTo(s, 26 + Math.sin(s * 0.15) * 9);
      x.lineTo(w, h); x.fill();
      x.fillStyle = '#e8f0ff';
      for (s = 6; s < w; s += 42) { x.beginPath(); x.arc(s + 4, 22, 4, 0, 6.3); x.fill(); }
    } else {
      x.fillStyle = '#ffd860'; x.beginPath();
      for (var j = 0; j < 10; j++) { var a = j * Math.PI / 5 - Math.PI / 2, rr = j % 2 ? 11 : 26; x.lineTo(64 + Math.cos(a) * rr, 34 + Math.sin(a) * rr); }
      x.fill();
    }
  });
}
function sign(x, w, h) {
  x.fillStyle = '#f2e6c4'; x.fillRect(0, 0, w, h);
  x.strokeStyle = '#e2b450'; x.lineWidth = 6; x.strokeRect(3, 3, w - 6, h - 6);
  x.fillStyle = '#9c2a26'; x.font = '700 34px ' + SERIF; x.textAlign = 'center'; x.textBaseline = 'middle';
  x.fillText('PANTOMIME', w / 2, h / 2 + 2);
}
// A house front: two windows over a window and a door (the wall colour is per house).
function facade(x, w, h) {
  x.fillStyle = '#ffffff'; x.fillRect(0, 0, w, h);
  x.fillStyle = 'rgba(0,0,0,0.06)'; x.fillRect(0, h * 0.9, w, h * 0.1);
  function win(cx, cy) {
    x.fillStyle = '#e8e4dc'; x.fillRect(cx - 13, cy - 16, 26, 32);
    x.fillStyle = '#3a4656'; x.fillRect(cx - 10, cy - 13, 20, 26);
    x.fillStyle = '#e8e4dc'; x.fillRect(cx - 1, cy - 13, 2, 26); x.fillRect(cx - 10, cy - 1, 20, 2);
  }
  win(w * 0.28, h * 0.32); win(w * 0.72, h * 0.32); win(w * 0.28, h * 0.68);
  x.fillStyle = '#4a5a6a'; x.fillRect(w * 0.62, h * 0.55, 22, h * 0.36);
}
function magazine(x, w, h) {
  var g = x.createLinearGradient(0, 0, w, h);
  g.addColorStop(0, '#f6c4cc'); g.addColorStop(1, '#eea2b0');
  x.fillStyle = g; x.fillRect(0, 0, w, h);
  x.fillStyle = '#ffffff'; x.font = '700 70px ' + SERIF; x.textAlign = 'center'; x.textBaseline = 'alphabetic';
  x.fillText('Sweet', w / 2, 78);
  x.fillStyle = '#8a2a44'; x.font = '600 15px ' + SERIF; x.fillText('THE  BAKING  ISSUE', w / 2, 102);
  // The cover cake: three tiers, cream drips, a cherry.
  [['#fbe8d0', 150, 46], ['#f4a8b8', 186, 64], ['#fbe8d0', 226, 82]].forEach(function (t) {
    x.fillStyle = t[0]; x.fillRect(w / 2 - t[2], t[1], t[2] * 2, 38);
    x.fillStyle = '#ffffff';
    for (var s = -t[2]; s < t[2]; s += 12) { x.beginPath(); x.ellipse(w / 2 + s + 6, t[1] + 2, 6, 7, 0, 0, Math.PI); x.fill(); }
  });
  x.fillStyle = '#c8202e'; x.beginPath(); x.arc(w / 2, 142, 10, 0, 6.3); x.fill();
  x.fillStyle = '#ffffff'; x.fillRect(w / 2 - 100, 266, 200, 6);
  x.fillStyle = '#8a2a44'; x.font = 'italic 19px ' + SERIF; x.fillText('29 cakes for an extra day', w / 2, 300);
  x.fillStyle = '#ffffff'; x.font = '13px ' + SERIF; x.fillText('Spring  ·  No. 29', w / 2, 326);
}
function moonTex() {
  return canvasTex(256, 256, function (x, w) {
    var g = x.createRadialGradient(128, 128, 0, 128, 128, 128);
    g.addColorStop(0, 'rgba(255,255,250,1)'); g.addColorStop(0.16, 'rgba(244,246,255,1)'); g.addColorStop(0.19, 'rgba(210,220,255,0.45)');
    g.addColorStop(0.4, 'rgba(170,190,255,0.12)'); g.addColorStop(1, 'rgba(150,170,255,0)');
    x.fillStyle = g; x.fillRect(0, 0, w, w);
    x.fillStyle = 'rgba(180,186,206,0.35)';
    [[118, 116, 9], [138, 134, 7], [124, 140, 5], [140, 112, 5]].forEach(function (m) { x.beginPath(); x.arc(m[0], m[1], m[2], 0, 6.3); x.fill(); });
  });
}
function raysTex() {
  return canvasTex(256, 256, function (x) {
    var r = rng(12), g = x.createRadialGradient(128, 128, 0, 128, 128, 128);
    g.addColorStop(0, 'rgba(255,250,235,1)'); g.addColorStop(0.12, 'rgba(255,230,180,0.6)'); g.addColorStop(0.4, 'rgba(255,200,140,0.14)'); g.addColorStop(1, 'rgba(255,190,130,0)');
    x.fillStyle = g; x.fillRect(0, 0, 256, 256);
    x.globalCompositeOperation = 'lighter';
    for (var i = 0; i < 40; i++) {
      var a = i / 40 * Math.PI * 2 + r() * 0.05, len = 70 + r() * 58, wd = 0.012 + r() * 0.02;
      var lg = x.createLinearGradient(128, 128, 128 + Math.cos(a) * len, 128 + Math.sin(a) * len);
      lg.addColorStop(0, 'rgba(255,226,170,0.3)'); lg.addColorStop(1, 'rgba(255,215,150,0)');
      x.fillStyle = lg; x.beginPath(); x.moveTo(128, 128);
      x.lineTo(128 + Math.cos(a - wd) * len, 128 + Math.sin(a - wd) * len); x.lineTo(128 + Math.cos(a + wd) * len, 128 + Math.sin(a + wd) * len); x.fill();
    }
  });
}

// ── Geometry ─────────────────────────────────────────────────────────────
function birdGeometry() {
  var v = [];
  function tri(a, b, c) { [a, b, c].forEach(function (p) { v.push(p[0], p[1], p[2]); }); }
  [1, -1].forEach(function (s) {
    tri([0, 0.02, -0.16], [s * 0.04, 0, -0.02], [0, 0.015, 0.12]);
    tri([s * 0.03, 0, -0.06], [s * 0.22, 0.03, -0.02], [s * 0.03, 0, 0.05]);
    tri([s * 0.22, 0.03, -0.02], [s * 0.42, 0.0, 0.08], [s * 0.03, 0, 0.05]);
    tri([s * 0.01, 0, 0.1], [s * 0.07, 0, 0.24], [0, 0, 0.14]);
  });
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(v, 3));
  geo.computeVertexNormals();
  return geo;
}
function tuftGeometry() {
  var parts = [];
  for (var k = 0; k < 3; k++) {
    var a = k * 2.1;
    parts.push(tinted(new THREE.ConeGeometry(0.03, 0.42, 3, 1, true).translate(0, 0.21, 0).rotateZ((k - 1) * 0.3).rotateY(a)
      .translate(Math.cos(a) * 0.05, 0, Math.sin(a) * 0.05), '#ffffff'));
  }
  return merge(parts);
}
function flowerGeometry() {
  return merge([
    tinted(new THREE.CylinderGeometry(0.005, 0.007, 0.32, 3).translate(0, 0.16, 0), '#8aa070'),
    tinted(new THREE.CircleGeometry(0.055, 6).rotateX(-Math.PI / 2 + 0.4).translate(0, 0.32, 0), '#ffffff'),
    tinted(new THREE.CircleGeometry(0.018, 5).rotateX(-Math.PI / 2 + 0.4).translate(0, 0.323, 0.003), '#f4d060')
  ]);
}
// A small handmade sailing boat, bow towards -z, waterline at y = 0.
function boatGeometry(L, hull, sailCol) {
  var g = new THREE.BoxGeometry(L * 0.32, L * 0.16, L, 2, 1, 8).translate(0, L * 0.03, 0), p = g.attributes.position;
  for (var i = 0; i < p.count; i++) {
    var z = p.getZ(i) / L, y = p.getY(i), bow = smooth(0.05, 0.5, -z), stern = smooth(0.2, 0.5, z);
    var narrow = (1 - bow * 0.92) * (1 - stern * 0.3) * (y < 0 ? 0.55 : 1);
    p.setX(i, p.getX(i) * narrow);
    p.setY(i, y + bow * bow * L * 0.06);
  }
  g.computeVertexNormals();
  var mastH = L * 1.05, sail = new THREE.BufferGeometry();
  sail.setAttribute('position', new THREE.Float32BufferAttribute([0, L * 0.16, -L * 0.12, 0, mastH, -L * 0.1, 0, L * 0.16, L * 0.36,
                                                                   0, L * 0.18, -L * 0.16, 0, mastH * 0.86, -L * 0.15, 0, L * 0.2, -L * 0.48], 3));
  sail.computeVertexNormals();
  return merge([tinted(g, hull),
                tinted(new THREE.BoxGeometry(L * 0.3, L * 0.012, L * 0.86).translate(0, L * 0.112, 0), '#d8c8a8'),
                tinted(new THREE.CylinderGeometry(L * 0.012, L * 0.014, mastH, 5).translate(0, mastH / 2 + L * 0.08, -L * 0.12), '#6a4a30'),
                tinted(sail, sailCol)]);
}
function benchGeometry() {
  var wood = '#8a5a36', iron = '#2a2a2e', parts = [];
  for (var k = 0; k < 3; k++) parts.push(tinted(new THREE.BoxGeometry(1.7, 0.035, 0.12).translate(0, 0.45, -0.16 + k * 0.15), wood));
  for (k = 0; k < 2; k++) parts.push(tinted(new THREE.BoxGeometry(1.7, 0.1, 0.03).rotateX(-0.2).translate(0, 0.62 + k * 0.17, 0.18 + k * 0.035), wood));
  [-0.78, 0.78].forEach(function (x) {
    parts.push(tinted(new THREE.BoxGeometry(0.05, 0.45, 0.05).translate(x, 0.225, -0.18), iron));
    parts.push(tinted(new THREE.BoxGeometry(0.05, 0.85, 0.05).rotateX(-0.12).translate(x, 0.42, 0.18), iron));
    parts.push(tinted(new THREE.BoxGeometry(0.06, 0.04, 0.5).translate(x, 0.66, -0.02), iron));
  });
  return merge(parts);
}
function chairGeometry() {
  var c = '#e6e0d2', parts = [tinted(new THREE.BoxGeometry(0.46, 0.04, 0.44).translate(0, 0.45, 0), c)];
  [[-0.2, -0.19], [0.2, -0.19], [-0.2, 0.19], [0.2, 0.19]].forEach(function (l, k) {
    parts.push(tinted(new THREE.BoxGeometry(0.035, k > 1 ? 0.98 : 0.45, 0.035).translate(l[0], (k > 1 ? 0.98 : 0.45) / 2, l[1]), c));
  });
  for (var k = 0; k < 3; k++) parts.push(tinted(new THREE.BoxGeometry(0.4, 0.05, 0.02).translate(0, 0.62 + k * 0.13, 0.19), c));
  return merge(parts);
}

// ── Renderer ─────────────────────────────────────────────────────────────
function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(29), slow = env.reduceMotion;
  var gl = makeRenderer(canvas, { clear: '#dcc6c4', shadows: !small });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#dcc6c4', 0.006);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.03, 6000);
  world.add(camera);
  var up = new THREE.Vector3(0, 1, 0);
  var m4 = new THREE.Matrix4(), q4 = new THREE.Quaternion(), s3 = new THREE.Vector3(), p3 = new THREE.Vector3(), t3 = new THREE.Vector3();
  var W = { h: 0, dx: 0, dz: 0 };

  function mesh(geo, mat, x, y, z, parent) {
    var m = new THREE.Mesh(geo, mat);
    m.position.set(x || 0, y || 0, z || 0);
    m.castShadow = m.receiveShadow = true;
    (parent || world).add(m);
    return m;
  }
  function box(w, h, d, x, y, z, mat, parent) { return mesh(new THREE.BoxGeometry(w, h, d), mat, x, y, z, parent); }
  function lam(color, extra) { return new THREE.MeshLambertMaterial(Object.assign({ color: color }, extra || {})); }
  function glow(tex, opacity, scale, parent, pos) {
    var s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: opacity }));
    s.scale.setScalar(scale);
    if (pos) s.position.copy(pos);
    (parent || world).add(s);
    return s;
  }
  var warmTex = softSprite('rgba(255,214,150,1)', 'rgba(255,170,90,0)');
  var whiteTex = softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)');

  // ── Sky ──────────────────────────────────────────────────────────────
  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#6a7fb2', mid: '#c4afc2', horizon: '#f7c9a4', sun: '#ffd2a0' }, 4000);
  sky.add(dome.mesh);
  var SN = small ? 2400 : 4800, sPos = [], sAttr = [], v3 = new THREE.Vector3(), band = new THREE.Euler(1.1, 0.4, 0.5);
  for (var i = 0; i < SN; i++) {
    var milky = i < SN * 0.4, th = r() * Math.PI * 2, yy;
    if (milky) { v3.set(Math.cos(th), (r() + r() + r() - 1.5) * 0.12, Math.sin(th)).normalize().applyEuler(band); if (v3.y < 0.02) continue; }
    else { yy = 0.02 + r() * 0.98; v3.set(Math.sqrt(1 - yy * yy) * Math.cos(th), yy, Math.sqrt(1 - yy * yy) * Math.sin(th)); }
    sPos.push(v3.x * 3500, v3.y * 3500, v3.z * 3500);
    sAttr.push(milky ? 0.6 + r() * 0.8 : 0.8 + Math.pow(r(), 3) * 3, r() * 6.28, r());
  }
  var starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.Float32BufferAttribute(sPos, 3));
  starGeo.setAttribute('star', new THREE.Float32BufferAttribute(sAttr, 3));
  var starMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
    uniforms: { uTime: { value: 0 }, uAmt: { value: 0 }, uScale: { value: 1 } },
    vertexShader: 'attribute vec3 star; uniform float uTime; uniform float uAmt; uniform float uScale; varying float vA;\n' +
      'void main(){ gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);\n' +
      ' float tw = 0.7 + 0.3 * sin(uTime * (1.2 + star.z * 2.0) + star.y);\n' +
      ' vA = tw * smoothstep(star.z * 0.8, star.z * 0.8 + 0.2, uAmt); gl_PointSize = star.x * uScale; }',
    fragmentShader: 'varying float vA; void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;\n' +
      ' float a = smoothstep(0.5, 0.0, d) * vA; gl_FragColor = vec4(vec3(0.86, 0.9, 1.0) * a, a);\n #include <colorspace_fragment>\n }'
  });
  var stars = new THREE.Points(starGeo, starMat);
  stars.frustumCulled = false;
  sky.add(stars);
  var moon = new THREE.Sprite(new THREE.SpriteMaterial({ map: moonTex(), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
  moon.scale.setScalar(360);
  sky.add(moon);
  var cloudTex = softSprite('rgba(255,255,255,0.85)', 'rgba(255,255,255,0)'), clouds = [];
  for (i = 0; i < 26; i++) {
    var ca = (r() - 0.5) * Math.PI * 1.6 + (i % 3 === 0 ? Math.PI : 0), cd = 2400 + r() * 800;
    var cl = new THREE.Sprite(new THREE.SpriteMaterial({ map: cloudTex, transparent: true, depthWrite: false, fog: false, opacity: 0.6 }));
    cl.position.set(Math.sin(ca) * cd, 160 + r() * 420, -Math.cos(ca) * cd);
    cl.scale.set(600 + r() * 700, 120 + r() * 120, 1);
    sky.add(cl);
    clouds.push(cl);
  }

  // ── Light ────────────────────────────────────────────────────────────
  var hemi = new THREE.HemisphereLight('#c0c8e2', '#6a5a4a', 1);
  var sun = new THREE.DirectionalLight('#ffcf9e', 1.6);
  world.add(hemi, sun, sun.target);
  if (!small) {
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    var sc = sun.shadow.camera;
    sc.left = sc.bottom = -16; sc.right = sc.top = 16; sc.near = 1; sc.far = 160;
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.03;
  }
  var sunDir = new THREE.Vector3();

  // ── The land ─────────────────────────────────────────────────────────
  var cA = new THREE.Color('#6a9442'), cB = new THREE.Color('#8aae50'), cHill = new THREE.Color('#748a46'), cSand = new THREE.Color('#e0cb9a'),
      cWet = new THREE.Color('#a8946c'), cBed = new THREE.Color('#6a6048'), cc = new THREE.Color();
  var ground = terrain(560, small ? 150 : 260, 30, 40, land, lam('#ffffff', { vertexColors: true }), function (x, z, y) {
    var n = 0.5 + 0.5 * Math.sin(x * 0.31 + z * 0.17) * Math.sin(x * 0.07 - z * 0.23);
    cc.copy(cA).lerp(cB, n * 0.8);
    cc.lerp(cHill, smooth(2, 12, y - profile(z)) * 0.7);
    var sand = smooth(-63, -71, z) * (1 - smooth(3.2, 5.5, y));
    if (sand > 0) { cc.lerp(cSand, sand); cc.lerp(cWet, smooth(0.9, 0.1, y) * sand); cc.lerp(cBed, smooth(-0.2, -1.5, y) * sand); }
    return cc;
  });
  ground.castShadow = false;
  world.add(ground);

  // A gravel path from the study door through the park, past the bench, down to the beach.
  var PATH = new THREE.CatmullRomCurve3([[-1.8, -0.6], [-2.2, -8], [0.5, -18], [-1.5, -30], [0.6, -42], [-0.4, -52], [-1.6, -58.5],
    [-3.2, -66], [-4.2, -73]].map(function (p) { return new THREE.Vector3(p[0], 0, p[1]); }));
  var pathPts = PATH.getPoints(140);
  var pathMesh = new THREE.Mesh(ribbon(pathPts, 0, 1.4, land, 0.07), lam('#c8b894'));
  pathMesh.receiveShadow = true;
  world.add(pathMesh);
  function nearPath(x, z, d) { for (var k = 0; k < pathPts.length; k++) { if (Math.abs(pathPts[k].x - x) < d && Math.abs(pathPts[k].z - z) < d) return true; } return false; }

  // Trees: the park, the wooded hills to the left, a few among the village.
  var trees = new THREE.InstancedMesh(broadleafGeometry(rng(4), '#4a3a2c'), lam('#ffffff', { vertexColors: true, flatShading: true }), small ? 260 : 560);
  scatter(trees, 12000, function (k, p, q, s, c) {
    var x, z, zone = r();
    if (zone < 0.14) { x = (r() < 0.5 ? -1 : 1) * (8 + r() * 26); z = -6 - r() * 50; if (nearPath(x, z, 7)) return false; }
    else if (zone < 0.62) { x = -18 - r() * 180; z = -70 + r() * 260; }
    else if (zone < 0.7) { x = 22 + r() * 120; z = -60 + r() * 120; }
    else { x = (r() - 0.5) * 300; z = 14 + r() * 200; }
    if (Math.abs(x) < 9 && z > -6 && z < 10) return false;               // the cottage
    if (z < -64 && land(x, z) < 3.5) return false;                       // not on the beach
    for (var l = 0; l < LAKES.length; l++) if (Math.hypot(x - LAKES[l][0], z - LAKES[l][1]) < LAKES[l][2] * 1.9) return false;
    if (Math.hypot(x - KIT.x, z - KIT.z) < 8) return false;
    p.set(x, land(x, z) - 0.2, z);
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar(zone < 0.14 ? 0.75 + r() * 0.45 : 0.9 + r() * 0.7);
    c.setHSL(0.22 + r() * 0.08, 0.4, 0.3 + r() * 0.12);
  });
  trees.castShadow = true;
  trees.receiveShadow = true;
  world.add(trees);

  // Grass tufts and flowers along the path and round the cottage.
  var tufts = new THREE.InstancedMesh(tuftGeometry(), lam('#ffffff', { vertexColors: true, side: THREE.DoubleSide }), small ? 900 : 2600);
  scatter(tufts, 20000, function (k, p, q, s, c) {
    var x = (r() - 0.5) * 34, z = 2 - r() * 66;
    if (z > -0.5 && Math.abs(x) < 3) return false;
    if (nearPath(x, z, 0.9) || land(x, z) < 2.6) return false;
    p.set(x, land(x, z) - 0.02, z);
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar(0.7 + r() * 0.8);
    c.setHSL(0.2 + r() * 0.07, 0.45, 0.3 + r() * 0.12);
  });
  world.add(tufts);
  var FLOWER = ['#ffffff', '#ffe27a', '#f6a6c0', '#a8b8ff', '#ffb070'];
  var flowers = new THREE.InstancedMesh(flowerGeometry(), lam('#ffffff', { vertexColors: true, side: THREE.DoubleSide }), small ? 400 : 1000);
  scatter(flowers, 20000, function (k, p, q, s, c) {
    var x, z;
    if (r() < 0.45) { x = (r() - 0.5) * 7; z = -0.3 - r() * 4; }        // the bed under the window
    else { x = (r() - 0.5) * 26; z = -2 - r() * 54; if (!nearPath(x, z, 2.6)) return false; }
    if (nearPath(x, z, 0.85)) return false;
    p.set(x, land(x, z), z);
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar(0.8 + r() * 0.6);
    c.set(FLOWER[k % FLOWER.length]);
  });
  world.add(flowers);

  // Lamp posts along the path, lit at dusk.
  var lampGlows = [], lampHeads = [], lampMat = lam('#26262a');
  [0.22, 0.47, 0.72, 0.84].forEach(function (t, k) {
    var pp = PATH.getPointAt(t), x = pp.x + (k % 2 ? -1.3 : 1.3), z = pp.z, y = land(x, z);
    if (k === 3) { x = BENCH.x + 1.6; z = BENCH.z + 0.2; y = land(x, z); }
    mesh(new THREE.CylinderGeometry(0.05, 0.07, 3.0, 6), lampMat, x, y + 1.5, z);
    lampHeads.push(box(0.22, 0.3, 0.22, x, y + 3.1, z, lam('#fff0cc', { emissive: '#ffcf80', emissiveIntensity: 0 })));
    lampGlows.push(glow(warmTex, 0, 2.2, world, p3.set(x, y + 3.1, z)));
  });

  // The bench on the bluff, looking out to sea.
  var bench = mesh(benchGeometry(), lam('#ffffff', { vertexColors: true }), BENCH.x, land(BENCH.x, BENCH.z), BENCH.z);
  bench.rotation.y = 0.06;

  // ── The study ────────────────────────────────────────────────────────
  var home = new THREE.Group();
  home.position.y = G;
  world.add(home);
  var paper = lam('#ffffff', { map: canvasTex(256, 256, wallpaper, [3, 2]) });
  var wood = lam('#6a4630'), paint = lam('#efe8da');
  var hw = ROOM.hw, RH = ROOM.h, T = 0.2;
  box(hw - WIN.hw, RH, T, -(hw + WIN.hw) / 2, RH / 2, -T / 2, paper, home);
  box(hw - WIN.hw, RH, T, (hw + WIN.hw) / 2, RH / 2, -T / 2, paper, home);
  box(WIN.hw * 2, WIN.y0, T, 0, WIN.y0 / 2, -T / 2, paper, home);
  box(WIN.hw * 2, RH - WIN.y1, T, 0, (RH + WIN.y1) / 2, -T / 2, paper, home);
  box(T, RH, ROOM.d + T, -hw - T / 2, RH / 2, ROOM.d / 2, paper, home);
  box(T, RH, ROOM.d + T, hw + T / 2, RH / 2, ROOM.d / 2, paper, home);
  box(hw * 2 + T * 2, RH, T, 0, RH / 2, ROOM.d + T / 2, paper, home);
  box(hw * 2, 0.1, ROOM.d, 0, -0.02, ROOM.d / 2, lam('#ffffff', { map: canvasTex(256, 256, boards, [3, 4]) }), home);
  box(hw * 2 + 0.4, 0.08, ROOM.d + 0.4, 0, RH + 0.04, ROOM.d / 2, paint, home);
  // Outside: a slate roof, a chimney, the front door.
  var roofShape = new THREE.Shape();
  roofShape.moveTo(-hw - 0.5, 0); roofShape.lineTo(hw + 0.5, 0); roofShape.lineTo(0, 1.9); roofShape.lineTo(-hw - 0.5, 0);
  var roof = mesh(new THREE.ExtrudeGeometry(roofShape, { depth: ROOM.d + 1.0, bevelEnabled: false }), lam('#4a4e58'), 0, RH + 0.08, -0.6, home);
  mesh(new THREE.BoxGeometry(0.6, 1.8, 0.6), lam('#8a7a6a'), 1.3, RH + 1.6, 3.4, home);
  box(0.95, 2.05, 0.06, -1.8, 1.03, -T - 0.04, lam('#3e6a7a'), home);
  box(1.7, 0.05, 0.22, 0, WIN.y0 - 0.02, 0.06, wood, home);                       // the inside sill
  box(1.7, 0.05, 0.22, 0, WIN.y0 - 0.06, -T - 0.08, paint, home);                  // the outside sill

  // Casement panes swinging out, hinged at the sides.
  var glassMat = new THREE.MeshLambertMaterial({ color: '#c8d8f0', transparent: true, opacity: 0.12, depthWrite: false });
  var paneW = WIN.hw - 0.02, paneH = WIN.y1 - WIN.y0;
  var panes = [-1, 1].map(function (side) {
    var hinge = new THREE.Group(), pane = new THREE.Group();
    [[paneW / 2, paneH - 0.025, paneW, 0.05], [paneW / 2, 0.025, paneW, 0.05], [0.025, paneH / 2, 0.05, paneH], [paneW - 0.025, paneH / 2, 0.05, paneH],
     [paneW / 2, paneH * 0.6, paneW, 0.03]].forEach(function (b) { box(b[2], b[3], 0.04, b[0], b[1], 0, paint, pane); });
    var gm = new THREE.Mesh(new THREE.PlaneGeometry(paneW, paneH), glassMat);
    gm.position.set(paneW / 2, paneH / 2, 0);
    pane.add(gm);
    pane.scale.x = -side;
    hinge.add(pane);
    hinge.position.set(side * WIN.hw, WIN.y0, -T + 0.03);
    home.add(hinge);
    return { hinge: hinge, side: side };
  });

  // The desk under the window, a chair, a lamp, books, a plant.
  box(1.6, 0.05, 0.7, 0, 0.75, 0.42, wood, home);
  [[-0.75, 0.12], [0.75, 0.12], [-0.75, 0.72], [0.75, 0.72]].forEach(function (l) { box(0.05, 0.73, 0.05, l[0], 0.365, l[1], wood, home); });
  box(0.5, 0.12, 0.6, 0.45, 0.66, 0.42, lam('#5a3c28'), home);
  var deskChair = mesh(chairGeometry(), lam('#9a6a44', { vertexColors: true }), 0.05, 0, 1.25, home);
  deskChair.rotation.y = Math.PI + 0.15;
  var lampLight = new THREE.PointLight('#ffc27a', 1.4, 5, 1.6);
  lampLight.position.set(0.62, 1.12, 0.32);
  home.add(lampLight);
  mesh(new THREE.CylinderGeometry(0.06, 0.08, 0.02, 12), lam('#2a2a2e'), 0.62, 0.785, 0.28, home);
  mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.3, 6), lam('#2a2a2e'), 0.62, 0.93, 0.28, home);
  mesh(new THREE.CylinderGeometry(0.06, 0.13, 0.13, 16, 1, true), lam('#e8d8b0', { side: THREE.DoubleSide, emissive: '#a07a40', emissiveIntensity: 0.5 }), 0.62, 1.1, 0.28, home);
  var bookCols = ['#7a2e2e', '#2e4a6a', '#5a6a3a', '#8a6a2a', '#4a3a5a', '#aa7a5a', '#2e5a5a'];
  [1.25, 1.75].forEach(function (y, row) {
    box(1.1, 0.03, 0.22, -1.65, y, 0.12, wood, home);
    var bx = -2.15;
    for (var k = 0; k < 14 && bx < -1.15; k++) {
      var bw = 0.03 + r() * 0.035, bh = 0.18 + r() * 0.1;
      box(bw, bh, 0.16 + r() * 0.03, bx + bw / 2, y + 0.015 + bh / 2, 0.12, lam(bookCols[(k + row * 3) % bookCols.length]), home).rotation.z = r() < 0.12 ? 0.15 : 0;
      bx += bw + 0.004;
    }
  });
  mesh(new THREE.CylinderGeometry(0.06, 0.045, 0.1, 10), lam('#b0603a'), -0.62, WIN.y0 + 0.05, 0.07, home);
  mesh(new THREE.IcosahedronGeometry(0.1, 0), lam('#4a7a3a', { flatShading: true }), -0.62, WIN.y0 + 0.17, 0.07, home);
  box(1.4, 0.01, 1.0, 0.0, 0.036, 1.6, lam('#8a3a3a'), home);                       // a rug

  // The wall clock.
  var clock = new THREE.Group();
  clock.position.set(1.6, 1.95, 0.02);
  home.add(clock);
  mesh(new THREE.CylinderGeometry(0.19, 0.19, 0.05, 32).rotateX(Math.PI / 2), wood, 0, 0, 0.025, clock);
  mesh(new THREE.CircleGeometry(0.165, 32), lam('#ffffff', { map: canvasTex(256, 256, clockFace) }), 0, 0, 0.052, clock);
  var handMat = lam('#1e1a16'), hands = [[0.012, 0.095, 0.054], [0.008, 0.135, 0.056], [0.003, 0.15, 0.058]].map(function (h, k) {
    var m = new THREE.Mesh(new THREE.BoxGeometry(h[0], h[1], 0.003).translate(0, h[1] / 2 - 0.02, 0), k === 2 ? lam('#b0302a') : handMat);
    m.position.z = h[2];
    clock.add(m);
    return m;
  });
  mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.01, 10).rotateX(Math.PI / 2), lam('#b08a40'), 0, 0, 0.06, clock);

  // The day calendar: 28 lifts away to show 29.
  var cal = new THREE.Group();
  cal.position.set(1.6, 1.24, 0.012);
  home.add(cal);
  mesh(new THREE.BoxGeometry(0.26, 0.34, 0.012), lam('#8a6a48'), 0, -0.02, 0, cal);
  mesh(new THREE.PlaneGeometry(0.2, 0.262), lam('#ffffff', { map: calendarPage('29', 'Thursday', true, 'leap day') }), 0, -0.02, 0.0075, cal);
  var leaf = new THREE.Group();
  leaf.position.set(0, -0.02 + 0.131, 0.009);
  cal.add(leaf);
  var leafMat = lam('#ffffff', { map: calendarPage('28', 'Wednesday'), transparent: true, side: THREE.DoubleSide });
  var leafMesh = new THREE.Mesh(new THREE.PlaneGeometry(0.2, 0.262).translate(0, -0.131, 0), leafMat);
  leaf.add(leafMesh);
  mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.01, 8).rotateX(Math.PI / 2), lam('#c8c0b0'), 0, 0.17, 0.01, cal);

  // The writing pad, with the words that write themselves, and the pen.
  var pad = new THREE.Group();
  pad.position.set(0.0, 0.777, 0.42);
  pad.rotation.set(-Math.PI / 2, 0, 0.12);
  home.add(pad);
  mesh(new THREE.BoxGeometry(PAD.w, PAD.h, 0.008).translate(0, 0, -0.004), lam('#ffffff', { map: canvasTex(PAD.px, PAD.py, padPaper) }), 0, 0, 0, pad);
  var inkU = { tInk: { value: canvasTex(PAD.px, PAD.py, padInk) }, uWrite: { value: 0 } };
  var ink = new THREE.Mesh(new THREE.PlaneGeometry(PAD.w, PAD.h), new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, uniforms: inkU,
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform sampler2D tInk; uniform float uWrite; varying vec2 vUv;\n' +
      'void main(){ vec4 t = texture2D(tInk, vUv);\n' +
      ' float row = floor(((1.0 - vUv.y) * ' + PAD.py.toFixed(1) + ' - ' + PAD.row0.toFixed(1) + ') / ' + PAD.rowH.toFixed(1) + ');\n' +
      ' float at = row + vUv.x * 0.98; float on = smoothstep(at - 0.015, at, uWrite * ' + PAD.rows.toFixed(1) + ');\n' +
      ' gl_FragColor = vec4(vec3(0.07, 0.1, 0.24), t.a * on * 0.95);\n #include <colorspace_fragment>\n }'
  }));
  ink.position.z = 0.0005;
  pad.add(ink);
  var pen = new THREE.Group();
  mesh(new THREE.CylinderGeometry(0.0055, 0.0045, 0.13, 8).translate(0, 0.075, 0), lam('#1e2a4a'), 0, 0, 0, pen);
  mesh(new THREE.ConeGeometry(0.0045, 0.012, 8).rotateX(Math.PI).translate(0, 0.006, 0), lam('#c8a050'), 0, 0, 0, pen);
  home.add(pen);
  var penRest = new THREE.Vector3(0.17, 0.783, 0.42), penTip = new THREE.Vector3();

  // The toy theatre: a proscenium, curtains that part, a painted sea and moon.
  var theatre = new THREE.Group();
  theatre.position.set(-0.43, 0.775, 0.3);
  theatre.rotation.y = 0.22;
  home.add(theatre);
  box(0.42, 0.04, 0.24, 0, 0.02, 0, lam('#5a2e22'), theatre);
  var ps = new THREE.Shape();
  ps.moveTo(-0.2, 0); ps.lineTo(0.2, 0); ps.lineTo(0.2, 0.34); ps.lineTo(-0.2, 0.34); ps.lineTo(-0.2, 0);
  var hole = new THREE.Path();
  hole.moveTo(-0.14, 0.05); hole.lineTo(-0.14, 0.22); hole.quadraticCurveTo(0, 0.29, 0.14, 0.22); hole.lineTo(0.14, 0.05); hole.lineTo(-0.14, 0.05);
  ps.holes.push(hole);
  var proTex = canvasTex(256, 218, proscenium);
  proTex.repeat.set(2.5, 1 / 0.34);
  proTex.offset.set(0.5, 0);
  mesh(new THREE.ExtrudeGeometry(ps, { depth: 0.012, bevelEnabled: false, curveSegments: 10 }), lam('#ffffff', { map: proTex }), 0, 0, 0.1, theatre);
  mesh(new THREE.PlaneGeometry(0.4, 0.1), lam('#ffffff', { map: canvasTex(256, 64, sign) }), 0, 0.39, 0.113, theatre);
  box(0.3, 0.012, 0.2, 0, 0.046, 0, lam('#6a4a30'), theatre);
  mesh(new THREE.PlaneGeometry(0.3, 0.26), lam('#ffffff', { map: canvasTex(256, 220, backdrop), emissive: '#ffffff', emissiveIntensity: 0.12 }), 0, 0.17, -0.09, theatre);
  var wingMat = lam('#ffffff', { map: canvasTex(64, 128, wingFlat), alphaTest: 0.5, side: THREE.DoubleSide });
  [[-0.12, -0.03], [0.12, -0.05], [-0.115, -0.065], [0.115, 0.0]].forEach(function (w) { mesh(new THREE.PlaneGeometry(0.07, 0.17), wingMat, w[0], 0.135, w[1], theatre); });
  var cutMat = function (k) { return lam('#ffffff', { map: cutout(k), alphaTest: 0.5, side: THREE.DoubleSide, emissive: '#3a3020', emissiveIntensity: 0.3 }); };
  var toyBoat = mesh(new THREE.PlaneGeometry(0.08, 0.04), cutMat('boat'), 0, 0.075, -0.06, theatre);
  var toyWave = mesh(new THREE.PlaneGeometry(0.28, 0.06), cutMat('wave'), 0, 0.06, -0.045, theatre);
  var toyStar = mesh(new THREE.PlaneGeometry(0.05, 0.025), cutMat('star'), -0.05, 0.2, -0.03, theatre);
  var curtainMat = lam('#ffffff', { map: canvasTex(64, 128, curtainCloth), side: THREE.DoubleSide });
  var curtains = [-1, 1].map(function (s) {
    var m = mesh(new THREE.PlaneGeometry(0.15, 0.2).translate(-s * 0.075, 0.1, 0), curtainMat, s * 0.145, 0.048, 0.085, theatre);
    return m;
  });
  mesh(new THREE.PlaneGeometry(0.3, 0.06), curtainMat, 0, 0.255, 0.09, theatre);
  var stageLight = new THREE.PointLight('#ffcf8a', 0, 0.9, 1.5);
  stageLight.position.set(0, 0.12, 0.05);
  theatre.add(stageLight);
  var foot = [];
  for (i = 0; i < 5; i++) foot.push(glow(warmTex, 0, 0.03, theatre, p3.set(-0.1 + i * 0.05, 0.06, 0.07)));

  // Dust drifting in the dawn light.
  var dust = particleField({ count: small ? 90 : 180, box: [3.2, 2.2, 3.2], fall: [-0.02, 0.03], size: 0.012, color: '#fff2d8',
                             map: whiteTex, sway: 0.05, windSpeed: 0.02 });
  dust.points.material.blending = THREE.AdditiveBlending;
  world.add(dust.points);
  var dustAt = new THREE.Vector3(0, G + 0.6, 1.8);

  // ── The sea ──────────────────────────────────────────────────────────
  var oceanMat = oceanMaterial({ color: '#2a4a6a', specular: '#c8d4e8', shininess: 110, foam: '#e8eef4', waves: WAVES });
  var ocean = oceanMesh(oceanMat, small ? 180 : 220, small ? 120 : 200);
  ocean.receiveShadow = true;
  world.add(ocean);
  var farMat = new THREE.MeshPhongMaterial({ color: '#2a4a6a', specular: '#c8d4e8', shininess: 70, polygonOffset: true, polygonOffsetFactor: 4, polygonOffsetUnits: 4 });
  var farSea = new THREE.Mesh(new THREE.PlaneGeometry(9000, 9000).rotateX(-Math.PI / 2), farMat);
  farSea.position.y = -0.9;
  world.add(farSea);

  // The jetty, the chair and its lantern.
  var plankN = Math.floor((JETTY.z0 - JETTY.z1) / 0.3);
  var planks = new THREE.InstancedMesh(new THREE.BoxGeometry(1.8, 0.06, 0.26), lam('#ffffff', { vertexColors: false }), plankN);
  for (i = 0; i < plankN; i++) {
    var shade = 0.82 + r() * 0.18;
    planks.setMatrixAt(i, m4.compose(p3.set(JETTY.x + (r() - 0.5) * 0.04, JETTY.y, JETTY.z0 - 0.15 - i * 0.3), q4.setFromAxisAngle(up, (r() - 0.5) * 0.02), s3.set(1, 1, 1)));
    planks.setColorAt(i, cc.setRGB(0.55 * shade, 0.45 * shade, 0.36 * shade));
  }
  planks.castShadow = planks.receiveShadow = true;
  world.add(planks);
  var pileN = Math.floor((JETTY.z0 - JETTY.z1) / 3.3) + 1;
  var piles = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.11, 0.13, 5.4, 7).translate(0, -2.7, 0), lam('#4a3a2e'), pileN * 2);
  for (i = 0; i < pileN; i++) [-1, 1].forEach(function (s, k) {
    piles.setMatrixAt(i * 2 + k, m4.makeTranslation(JETTY.x + s * 0.85, JETTY.y + 0.1, JETTY.z0 - 1 - i * 3.3));
  });
  world.add(piles);
  [-0.6, 0.6].forEach(function (s) { box(0.1, 0.16, JETTY.z0 - JETTY.z1, JETTY.x + s, JETTY.y - 0.1, (JETTY.z0 + JETTY.z1) / 2, lam('#4a3a2e')); });
  var chair = mesh(chairGeometry(), lam('#ffffff', { vertexColors: true }), JETTY.x, JETTY.y + 0.03, JETTY.z1 + 2.0);
  chair.rotation.y = 0.12;
  var LANT = new THREE.Vector3(JETTY.x + 0.55, JETTY.y + 0.2, JETTY.z1 + 2.2);
  box(0.16, 0.24, 0.16, LANT.x, LANT.y, LANT.z, lam('#fff0c8', { emissive: '#ffb860', emissiveIntensity: 0.8, transparent: true, opacity: 0.9 }));
  box(0.19, 0.03, 0.19, LANT.x, LANT.y + 0.135, LANT.z, lam('#26262a'));
  box(0.19, 0.03, 0.19, LANT.x, LANT.y - 0.12, LANT.z, lam('#26262a'));
  var lantGlow = glow(warmTex, 0, 1.6, world, LANT);
  var lantLight = new THREE.PointLight('#ffb766', 0, 9, 1.4);
  lantLight.position.copy(LANT).add(t3.set(0, 0.3, 0));
  world.add(lantLight);

  // Two handmade boats with lanterns.
  var boatMat = lam('#ffffff', { vertexColors: true, side: THREE.DoubleSide, emissive: '#2a2016' });
  var boats = [{ L: 2.1, hull: '#a8562e', sail: '#f2ead6', from: [1.2, -85.6], to: [22, -215], lag: 0 },
               { L: 1.4, hull: '#3e6a8a', sail: '#f6e2c8', from: [3.8, -86.4], to: [15, -195], lag: 0.14 }].map(function (b) {
    var g = new THREE.Group(), m = new THREE.Mesh(boatGeometry(b.L, b.hull, b.sail), boatMat);
    m.castShadow = true;
    g.add(m);
    b.glow = glow(warmTex, 0, 0.9, g, p3.set(0, b.L * 0.7, b.L * 0.28));
    b.group = g;
    world.add(g);
    return b;
  });
  var boatLight = new THREE.PointLight('#ffb766', 0, 12, 1.4);
  world.add(boatLight);

  // ── The silver coast: hills of homes, and lakes ──────────────────────
  var houseGeo = new THREE.BoxGeometry(1, 1.6, 1).translate(0, 0.2, 0);
  var roofGeo = new THREE.CylinderGeometry(0.62, 0.62, 1.06, 3).rotateZ(Math.PI / 2).rotateX(-Math.PI / 2).scale(1, 0.6, 1).translate(0, 1.27, 0);
  roofGeo.rotateY(Math.PI / 2);
  var HN = small ? 45 : 85, houses = new THREE.InstancedMesh(houseGeo, lam('#ffffff', { map: canvasTex(128, 128, facade) }), HN), roofs = new THREE.InstancedMesh(roofGeo, lam('#ffffff'), HN);
  var winPos = [], WALLS = ['#f2ece0', '#ece2d0', '#e8eef2', '#f4e4d4', '#dfe6dc'], ROOFS = ['#5a4e4e', '#8a4a3a', '#4a5260', '#7a5a46'], hn = 0;
  for (var tries = 0; tries < 4000 && hn < HN; tries++) {
    var hx = 24 + Math.pow(r(), 1.3) * 110, hz = -62 + r() * 105;
    var bad = Math.hypot(hx - KIT.x, hz - KIT.z) < 9;
    for (var l = 0; l < LAKES.length; l++) if (Math.hypot(hx - LAKES[l][0], hz - LAKES[l][1]) < LAKES[l][2] * 1.9) bad = true;
    if (bad || land(hx, hz) < 3) continue;
    var w = 4 + r() * 3, h = 3 + r() * 2.5, d = 4 + r() * 3, rot = (r() - 0.5) * 0.5;
    q4.setFromAxisAngle(up, rot);
    p3.set(hx, land(hx, hz), hz);
    houses.setMatrixAt(hn, m4.compose(p3, q4, s3.set(w, h, d)));
    roofs.setMatrixAt(hn, m4.compose(p3, q4, s3.set(w, h, d)));
    houses.setColorAt(hn, cc.set(WALLS[hn % WALLS.length]));
    roofs.setColorAt(hn, cc.set(ROOFS[hn % ROOFS.length]));
    for (var k = 0; k < 4; k++) {
      if (r() < 0.4) continue;
      var lx = ((k % 2) - 0.5) * w * 0.5, lz = (k < 2 ? -1 : 1) * (d / 2 + 0.05), ly = land(hx, hz) + h * 0.52;
      winPos.push(hx + lx * Math.cos(rot) + lz * Math.sin(rot), ly, hz + lz * Math.cos(rot) - lx * Math.sin(rot));
    }
    hn++;
  }
  houses.count = roofs.count = hn;
  houses.castShadow = roofs.castShadow = true;
  world.add(houses, roofs);
  var winGeo = new THREE.BufferGeometry();
  winGeo.setAttribute('position', new THREE.Float32BufferAttribute(winPos, 3));
  var winPts = new THREE.Points(winGeo, new THREE.PointsMaterial({ color: '#ffc27a', size: 1.6, map: warmTex, transparent: true, depthWrite: false,
                                                                    blending: THREE.AdditiveBlending, opacity: 0 }));
  world.add(winPts);
  var lakeMat = new THREE.MeshPhongMaterial({ color: '#0e1828', specular: '#ffffff', shininess: 160, emissive: '#000000' });
  LAKES.forEach(function (L, k) {
    var m = new THREE.Mesh(new THREE.CircleGeometry(L[2] * 1.06, 40).rotateX(-Math.PI / 2), lakeMat);
    m.position.set(L[0], LAKE_Y[k], L[1]);
    world.add(m);
  });

  // ── The kitchen cottage ──────────────────────────────────────────────
  // Local x, z about KIT; the sea wall is at z = -hd. Windows front and
  // back line up at x = WX, so the way in from the hills runs straight on
  // out to the sea; the table stands to the left of it.
  var kit = new THREE.Group();
  kit.position.set(KIT.x, KF, KIT.z);
  world.add(kit);
  var kwall = lam('#ffffff', { map: canvasTex(128, 128, tiles('#f4efe6', '#d8d0c2', 4), [6, 2]) });
  var kout = lam('#f2ece2'), khw = KIT.hw, khd = KIT.hd, KH = KIT.h, KW = { x: 0.6, hw: 0.6, y0: 0.95, y1: 2.1 };
  [-khd, khd].forEach(function (z) {
    box(khw + KW.x - KW.hw, KH, T, (-khw + KW.x - KW.hw) / 2, KH / 2, z, kwall, kit);
    box(khw - KW.x - KW.hw, KH, T, (khw + KW.x + KW.hw) / 2, KH / 2, z, kwall, kit);
    box(KW.hw * 2, KW.y0, T, KW.x, KW.y0 / 2, z, kwall, kit);
    box(KW.hw * 2, KH - KW.y1, T, KW.x, (KH + KW.y1) / 2, z, kwall, kit);
    box(1.4, 0.05, 0.3, KW.x, KW.y0 - 0.02, z, lam('#e8e0d0'), kit);
    [-1, 1].forEach(function (s) {                                                          // shutters folded back
      var sh = box(0.58, KW.y1 - KW.y0, 0.04, KW.x + s * (KW.hw + 0.32), (KW.y0 + KW.y1) / 2, z + (z < 0 ? -1 : 1) * (T / 2 + 0.03), lam('#4a7a6a'), kit);
      sh.castShadow = false;
    });
  });
  box(T, KH, khd * 2 + T, -khw, KH / 2, 0, kwall, kit);
  box(T, KH, khd * 2 + T, khw, KH / 2, 0, kwall, kit);
  box(khw * 2, 0.06, khd * 2, 0, -0.03, 0, lam('#ffffff', { map: canvasTex(128, 128, checker, [4, 3]) }), kit);
  box(khw * 2 + 0.3, 0.08, khd * 2 + 0.3, 0, KH + 0.04, 0, kout, kit);
  box(khw * 2 + 0.4, 3.2, khd * 2 + 0.4, 0, -1.63, 0, lam('#9a8e80'), kit);                  // the plinth into the slope
  var kroofShape = new THREE.Shape();
  kroofShape.moveTo(-khd - 0.5, 0); kroofShape.lineTo(khd + 0.5, 0); kroofShape.lineTo(0, 1.7); kroofShape.lineTo(-khd - 0.5, 0);
  var kroof = mesh(new THREE.ExtrudeGeometry(kroofShape, { depth: khw * 2 + 0.8, bevelEnabled: false }), lam('#9a4a36'), -khw - 0.4, KH + 0.08, 0, kit);
  kroof.rotation.y = Math.PI / 2;
  box(0.55, 1.6, 0.55, -khw + 0.6, KH + 1.2, 0.8, lam('#8a7a6a'), kit);
  box(0.9, 2.0, 0.07, -1.45, 1.0, -khd - T / 2 - 0.02, lam('#4a7a6a'), kit);                  // the front door

  // The table, its cakes and the magazine.
  var TB = { x: -0.8, z: 0.1 }, kwood = lam('#8a5c38');
  box(0.9, 0.05, 1.45, TB.x, 0.78, TB.z, kwood, kit);
  [[-0.38, -0.65], [0.38, -0.65], [-0.38, 0.65], [0.38, 0.65]].forEach(function (l) { box(0.06, 0.76, 0.06, TB.x + l[0], 0.38, TB.z + l[1], kwood, kit); });
  box(0.94, 0.006, 1.5, TB.x, 0.806, TB.z, lam('#f6eef0'), kit);                              // a cloth
  box(0.3, 0.007, 1.5, TB.x, 0.81, TB.z, lam('#d88a9a'), kit);                                // a runner
  var cream = lam('#fbf2e2'), pink = lam('#f2a8b8'), stand = new THREE.MeshPhongMaterial({ color: '#f4f4f8', shininess: 90, specular: '#ffffff' });
  var cake = new THREE.Group();
  cake.position.set(TB.x + 0.05, 0.81, TB.z - 0.2);
  kit.add(cake);
  mesh(new THREE.CylinderGeometry(0.04, 0.07, 0.12, 16), stand, 0, 0.06, 0, cake);
  mesh(new THREE.CylinderGeometry(0.2, 0.2, 0.015, 32), stand, 0, 0.125, 0, cake);
  [[0.16, 0.09, cream], [0.135, 0.07, pink], [0.105, 0.065, cream]].reduce(function (y, t) {
    mesh(new THREE.CylinderGeometry(t[0], t[0], t[1], 32), t[2], 0, y + t[1] / 2, 0, cake);
    mesh(new THREE.TorusGeometry(t[0] - 0.004, 0.012, 6, 32).rotateX(Math.PI / 2), lam('#ffffff'), 0, y + t[1], 0, cake);
    return y + t[1];
  }, 0.133);
  mesh(new THREE.SphereGeometry(0.022, 12, 8), new THREE.MeshPhongMaterial({ color: '#c8202e', shininess: 120, specular: '#ffffff' }), 0, 0.37, 0, cake);
  var cup = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.035, 0.026, 0.04, 12).translate(0, 0.02, 0), lam('#e8a85a'), 7);
  var top = new THREE.InstancedMesh(new THREE.SphereGeometry(0.036, 12, 8).scale(1, 0.75, 1).translate(0, 0.045, 0), lam('#ffffff'), 7);
  var ICING = ['#f6b8c8', '#fff4e8', '#c8e0f8', '#f6b8c8', '#fff0b0', '#fff4e8', '#d8c0f0'];
  for (i = 0; i < 7; i++) {
    var ang = i / 6 * Math.PI * 2;
    p3.set(TB.x + 0.12 + (i ? Math.cos(ang) * 0.09 : 0), 0.825, TB.z + 0.42 + (i ? Math.sin(ang) * 0.09 : 0));
    cup.setMatrixAt(i, m4.makeTranslation(p3.x, p3.y, p3.z));
    top.setMatrixAt(i, m4.makeTranslation(p3.x, p3.y, p3.z));
    top.setColorAt(i, cc.set(ICING[i]));
  }
  kit.add(cup, top);
  mesh(new THREE.CylinderGeometry(0.17, 0.17, 0.012, 24), stand, TB.x + 0.12, 0.818, TB.z + 0.42, kit);
  // The glossy magazine, propped up facing the way in.
  var mag = mesh(new THREE.BoxGeometry(0.24, 0.32, 0.006), [lam('#f0b8c4'), lam('#f0b8c4'), lam('#f0b8c4'), lam('#f0b8c4'),
    new THREE.MeshPhongMaterial({ color: '#ffffff', map: canvasTex(256, 340, magazine), shininess: 140, specular: '#ffffff' }), lam('#f4ece4')],
    TB.x - 0.2, 0.95, TB.z + 0.25, kit);
  mag.rotation.set(-0.38, 1.08, 0, 'YXZ');
  box(0.05, 0.16, 0.05, TB.x - 0.3, 0.89, TB.z + 0.27, kwood, kit);                           // its little easel
  mesh(new THREE.SphereGeometry(0.12, 18, 10, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2), lam('#c8b8a0', { side: THREE.DoubleSide }), TB.x + 0.22, 0.93, TB.z - 0.55, kit);
  mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.36, 10).rotateX(Math.PI / 2), lam('#c89a6a'), TB.x + 0.3, 0.835, TB.z + 0.05, kit);
  // The left wall: shelves of jars, the range glowing.
  [1.75, 1.35].forEach(function (y) { box(0.24, 0.03, 1.8, -khw + 0.22, y, -0.6, kwood, kit); });
  var JAR = ['#d8a860', '#a8c8d8', '#e0d0b0', '#b86a4a', '#c8d8a0'];
  for (i = 0; i < 10; i++) {
    var jh = 0.12 + r() * 0.1;
    mesh(new THREE.CylinderGeometry(0.05, 0.05, jh, 12), lam(JAR[i % JAR.length]), -khw + 0.22, (i < 5 ? 1.765 : 1.365) + jh / 2, -1.35 + (i % 5) * 0.36, kit);
  }
  box(0.62, 0.9, 0.9, -khw + 0.42, 0.45, 1.3, lam('#2a2a30'), kit);                            // the range
  var ovenWin = box(0.02, 0.28, 0.4, -khw + 0.74, 0.48, 1.3, lam('#ff9a4a', { emissive: '#ff8a3a', emissiveIntensity: 1.4 }), kit);
  ovenWin.castShadow = false;
  mesh(new THREE.SphereGeometry(0.12, 14, 10).scale(1, 0.8, 1), new THREE.MeshPhongMaterial({ color: '#b04a3a', shininess: 80 }), -khw + 0.45, 1.0, 1.15, kit);
  var pendant = new THREE.PointLight('#ffc88a', 3.2, 7, 1.5);
  pendant.position.set(TB.x, 1.85, TB.z);
  kit.add(pendant);
  mesh(new THREE.ConeGeometry(0.2, 0.18, 18, 1, true), lam('#2e4a3a', { side: THREE.DoubleSide }), TB.x, 2.02, TB.z, kit);
  mesh(new THREE.CylinderGeometry(0.004, 0.004, 0.5, 4), lam('#222222'), TB.x, 2.36, TB.z, kit);
  glow(warmTex, 0.9, 0.35, kit, p3.set(TB.x, 1.95, TB.z));
  var ovenLight = new THREE.PointLight('#ff9a50', 1.2, 3, 1.6);
  ovenLight.position.set(-khw + 0.95, 0.5, 1.3);
  kit.add(ovenLight);
  var kitGlow = glow(warmTex, 0.7, 3.2, kit, p3.set(KW.x, 1.5, khd + 0.4));
  var flour = particleField({ count: small ? 70 : 140, box: [3.6, 2.2, 3.4], fall: [-0.02, 0.04], size: 0.01, color: '#fff6ea',
                              map: whiteTex, sway: 0.05, windSpeed: 0.02 });
  flour.points.material.blending = THREE.AdditiveBlending;
  world.add(flour.points);
  var flourAt = new THREE.Vector3(KIT.x, KF + 0.6, KIT.z);

  // ── The door in the sea ──────────────────────────────────────────────
  var door = new THREE.Group();
  door.position.set(DOOR.x, 0, DOOR.z);
  world.add(door);
  var frameMat = lam('#f4efe4');
  box(0.12, 4.0, 0.16, -0.56, 0.25, 0, frameMat, door);
  box(0.12, 4.0, 0.16, 0.56, 0.25, 0, frameMat, door);
  box(1.36, 0.14, 0.18, 0, 2.3, 0, frameMat, door);
  var leafHinge = new THREE.Group();
  leafHinge.position.set(-0.5, 0.05, 0);
  door.add(leafHinge);
  var doorPaint = lam('#3e6aa0');
  box(1.0, 2.18, 0.05, 0.5, 1.09, 0, doorPaint, leafHinge);
  [[0.5, 1.58, 0.72, 0.62], [0.5, 0.62, 0.72, 0.8]].forEach(function (pn) { box(pn[2], pn[3], 0.015, pn[0], pn[1], 0.032, lam('#4a7ab0'), leafHinge); });
  mesh(new THREE.SphereGeometry(0.035, 12, 8), new THREE.MeshPhongMaterial({ color: '#d8b050', shininess: 100, specular: '#fff0c0' }), 0.88, 1.05, 0.06, leafHinge);
  var doorLight = new THREE.Mesh(new THREE.PlaneGeometry(1.0, 2.2), new THREE.MeshBasicMaterial({ map: warmTex, color: '#fff2d8', transparent: true,
    blending: THREE.AdditiveBlending, depthWrite: false, opacity: 0, fog: false }));
  doorLight.position.set(0, 1.15, -0.25);
  door.add(doorLight);
  var rays = glow(raysTex(), 0, 9, door, p3.set(0, 1.2, -1.5));
  rays.material.fog = false;

  // ── Birds: a flock over the park, a few gulls over the sea at sunrise ─
  var birdMat = lam('#ffffff', { side: THREE.DoubleSide });
  var birdTime = { value: 0 };
  birdMat.onBeforeCompile = function (sh) {
    sh.uniforms.uTime = birdTime;
    sh.vertexShader = 'uniform float uTime;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n float sph = float(gl_InstanceID) * 1.37;\n' +
      ' float beat = sin(uTime * 11.0 + sph) * (0.45 + 0.55 * smoothstep(-0.3, 0.4, sin(uTime * 0.8 + sph * 0.7)));\n' +
      ' transformed.y += beat * max(abs(position.x) - 0.03, 0.0) * 0.9;');
  };
  var NB = small ? 26 : 44, NG = 6, flock = new THREE.InstancedMesh(birdGeometry(), birdMat, NB + NG), birds = [];
  flock.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  flock.frustumCulled = false;
  for (i = 0; i < NB + NG; i++) {
    birds.push({ o: new THREE.Vector3((r() - 0.5) * 9, (r() - 0.5) * 4, (r() - 0.5) * 9), ph: r() * 6.28, w: 0.5 + r() * 0.6, R: 1 + r() * 2.5, gull: i >= NB });
  }
  for (i = 0; i < NB + NG; i++) flock.setColorAt(i, cc.set(i < NB ? '#2a2a32' : '#e4e6ec'));
  world.add(flock);
  var FLIGHT = new THREE.CatmullRomCurve3([[-12, 6, -30], [-3, 11, -38], [6, 15, -44], [15, 18, -46], [26, 23, -40], [44, 30, -22]]
    .map(function (p) { return new THREE.Vector3(p[0], land(p[0], p[2]) + p[1], p[2]); }));
  var GULLS = new THREE.Vector3(DOOR.x + 4, 9, DOOR.z - 8);

  // ── Swimming: flecks of foam on the water, spray off the crests ──────
  var FN = small ? 260 : 520, fBase = new Float32Array(FN * 2), fPos = new Float32Array(FN * 3);
  for (i = 0; i < FN; i++) { fBase[i * 2] = (r() - 0.5) * 36; fBase[i * 2 + 1] = (r() - 0.5) * 36; }
  var foamGeo = new THREE.BufferGeometry();
  foamGeo.setAttribute('position', new THREE.BufferAttribute(fPos, 3));
  var foam = new THREE.Points(foamGeo, new THREE.PointsMaterial({ color: '#f2f6fa', size: 0.07, map: whiteTex, transparent: true, depthWrite: false, opacity: 0 }));
  foam.frustumCulled = false;
  world.add(foam);
  var SPN = small ? 200 : 400, spPos = new Float32Array(SPN * 3), spVel = new Float32Array(SPN * 3), spLife = new Float32Array(SPN), spNext = 0, spClock = 0;
  for (i = 0; i < SPN; i++) spPos[i * 3 + 1] = -99;
  var sprayGeo = new THREE.BufferGeometry();
  sprayGeo.setAttribute('position', new THREE.BufferAttribute(spPos, 3));
  var spray = new THREE.Points(sprayGeo, new THREE.PointsMaterial({ color: '#f4f8ff', size: 0.028, map: whiteTex, transparent: true, depthWrite: false, opacity: 0.9 }));
  spray.frustumCulled = false;
  world.add(spray);

  // ── The lights of the day, and the dream ────────────────────────────
  var SITES = [new THREE.Vector3(0, G + 1.6, -0.4), new THREE.Vector3(BENCH.x, land(BENCH.x, BENCH.z) + 1.2, BENCH.z), new THREE.Vector3(2.5, 1.0, -85),
               LANT, new THREE.Vector3(KIT.x + 0.6, KF + 1.5, KIT.z - KIT.hd - 0.4), new THREE.Vector3(DOOR.x, 1.2, DOOR.z + 0.2)];
  var siteGlows = SITES.map(function () { return glow(warmTex, 0, 1, world); });
  siteGlows.forEach(function (g) { g.material.fog = false; });
  var motes = particleField({ count: small ? 300 : 600, box: [70, 36, 70], fall: [-1.4, -0.4], size: 0.45, color: '#fff0c8',
                              map: softSprite('rgba(255,245,215,1)', 'rgba(255,225,170,0)'), sway: 0.6, windSpeed: 0.4 });
  motes.points.material.blending = THREE.AdditiveBlending;
  world.add(motes.points);

  // ── The final pass: silver painting, dream-light ─────────────────────
  var target = new THREE.WebGLRenderTarget(16, 16, { type: THREE.HalfFloatType, samples: small ? 0 : 4 });
  var postU = { tScene: { value: target.texture }, uRes: { value: new THREE.Vector2(16, 16) }, uSilver: { value: 0 }, uDream: { value: 0 }, uTime: { value: 0 } };
  var post = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
    uniforms: postU, depthTest: false, depthWrite: false,
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
    fragmentShader: 'uniform sampler2D tScene; uniform vec2 uRes; uniform float uSilver; uniform float uDream; uniform float uTime; varying vec2 vUv;\n' + NOISE + '\n' +
      'void main(){ vec2 px = vUv * uRes; vec2 uv = vUv;\n' +
      // Brushwork: the image smeared a little along short slanted strokes.
      ' float s = uRes.y / 800.0;\n' +
      ' vec2 bp = mat2(0.8, -0.6, 0.6, 0.8) * px / s;\n' +
      ' vec2 wob = vec2(vn(bp * vec2(0.03, 0.16)), vn(bp * vec2(0.16, 0.03) + 7.0)) - 0.5;\n' +
      ' uv += wob * 5.0 * s / uRes * uSilver;\n' +
      ' vec3 c = texture2D(tScene, uv).rgb;\n' +
      ' if (uDream > 0.001) { vec3 g = vec3(0.0); float rr = 0.01 + 0.025 * uDream;\n' +
      '  for (int k = 0; k < 10; k++) { float a = float(k) * 0.6283 + uTime * 0.05; g += texture2D(tScene, vUv + vec2(cos(a) * uRes.y / uRes.x, sin(a)) * rr * (0.5 + 0.5 * fract(float(k) * 0.37))).rgb; }\n' +
      '  g /= 10.0; c = mix(c, max(c, g), uDream * 0.7) + g * 0.35 * uDream; }\n' +
      ' #ifdef TONE_MAPPING\n c = toneMapping(c);\n #endif\n' +
      ' float l = dot(c, vec3(0.2126, 0.7152, 0.0722));\n' +
      ' vec3 silver = mix(vec3(0.03, 0.035, 0.05), vec3(0.97, 0.98, 1.0), smoothstep(0.02, 0.6, l));\n' +
      ' float weave = (sin(px.x * 1.9 / s) * sin(px.y * 1.9 / s)) * 0.012 + (vn(px * 0.35 / s) - 0.5) * 0.05;\n' +
      ' c = mix(c, silver * (1.0 + weave), uSilver);\n' +
      ' c = mix(c, vec3(1.0, 0.86, 0.66) * (0.55 + 0.45 * l), uDream * 0.18);\n' +
      ' gl_FragColor = vec4(c, 1.0);\n #include <colorspace_fragment>\n }'
  }));
  post.frustumCulled = false;
  var postScene = new THREE.Scene(), postCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  postScene.add(post);

  // ── Per frame ────────────────────────────────────────────────────────
  var cur = {};
  COLOR_KEYS.forEach(function (k) { cur[k] = new THREE.Color(); });
  function blend(L) {
    L = clamp(L, 0, LOOKS.length - 1);
    var i0 = Math.min(Math.floor(L), LOOKS.length - 2), t = L - i0, a = LOOKS[i0], b = LOOKS[i0 + 1];
    for (var k = 0; k < COLOR_KEYS.length; k++) cur[COLOR_KEYS[k]].copy(a[COLOR_KEYS[k]]).lerp(b[COLOR_KEYS[k]], t);
    for (k = 0; k < NUM_KEYS.length; k++) cur[NUM_KEYS[k]] = lerp(a[NUM_KEYS[k]], b[NUM_KEYS[k]], t);
  }
  var skyRefl = new THREE.Color(), tmpC = new THREE.Color();
  var look = new THREE.Vector3(), tmp = new THREE.Vector3(), white = new THREE.Color('#ffffff'), silverC = new THREE.Color('#cfd6e6');
  var pf = { snow: 1, wind: 0, dt: 0, time: 0 }, portrait = false, H = 800, secs = 12, bufSize = new THREE.Vector2();

  function frame(f) {
    var row = f.row, time = f.time, dt = f.dt, way = f.cam;
    var swell = row[2], yaw = row[4], pitch = row[5], tick = row[6], page = row[7], write = row[8], curtain = row[9], open = row[10];
    var birdT = row[11], boatT = row[12], silver = row[13], doorT = row[14], lightsT = row[15], dream = row[16];
    blend(row[1]);
    var night = cur.night, day = 1 - night;

    // ── Camera along the way; in the sea it rides the swell ──
    var n = WAY.length - 1, t = clamp(way / n, 0, 1);
    pathP.getPoint(t, camera.position);
    pathT.getPoint(t, look);
    var ride = (1 - smooth(0.8, 1.9, camera.position.y)) * smooth(-86, -92, camera.position.z);
    var roll = 0;
    if (ride > 0) {
      wave(camera.position.x, camera.position.z, time, swell, W);
      camera.position.y += W.h * ride;
      look.y += W.h * ride * 0.6;
      roll = W.dx * ride * 0.25;
    }
    camera.position.y += Math.sin(time * 0.6) * 0.008;
    camera.lookAt(look);
    if (roll) camera.rotateZ(roll);
    if (portrait) { yaw += portraitAim(way, 0); pitch += portraitAim(way, 1); }
    camera.rotateY(yaw - f.mx * 0.1);
    camera.rotateX(pitch - f.my * 0.05);
    sky.position.copy(camera.position);

    // ── Light for the look ──
    sunDir.set(cur.az, Math.sin(cur.el), -1).normalize();
    var du = dome.uniforms;
    du.sunDir.value.copy(sunDir);
    du.top.value.copy(cur.top);
    du.mid.value.copy(cur.mid);
    du.horizon.value.copy(cur.horizon);
    du.sunColor.value.copy(cur.glow).multiplyScalar(cur.glowAmt);
    var inRoom = camera.position.z > -0.2 && Math.abs(camera.position.x) < ROOM.hw ? 1 : 0;
    sun.color.copy(cur.sunC);
    sun.intensity = cur.sunI * (small ? 1 - inRoom * 0.55 : 1);
    sun.position.copy(camera.position).addScaledVector(sunDir, 70);
    sun.target.position.copy(camera.position);
    hemi.color.copy(cur.hemiSky);
    hemi.groundColor.copy(cur.hemiGnd);
    hemi.intensity = cur.hemiI;
    world.fog.color.copy(cur.fog).lerp(cur.horizon, 0.6);
    world.fog.density = cur.fogD;
    gl.setClearColor(world.fog.color);
    gl.toneMappingExposure = cur.exp;
    starMat.uniforms.uTime.value = time;
    starMat.uniforms.uAmt.value = night;
    stars.visible = night > 0.02;
    moon.position.copy(sunDir).multiplyScalar(3200);
    moon.material.opacity = smooth(0.4, 0.9, night);
    moon.visible = night > 0.4;
    for (var c = 0; c < clouds.length; c++) {
      clouds[c].material.color.copy(cur.cloud);
      clouds[c].material.opacity = 0.55 * (1 - night * 0.5);
    }
    oceanMat.color.copy(cur.sea);
    farMat.color.copy(cur.sea);
    // A low sun on a calm sea would glaze it all over: keep the glitter tight.
    oceanMat.specular.copy(cur.sunC).multiplyScalar(0.9 / Math.max(cur.sunI, 0.9));
    farMat.specular.copy(oceanMat.specular);
    var ou = oceanMat.userData.uniforms;
    ou.uTime.value = time;
    ou.uAmp.value = swell;
    ou.uFoamAmt.value = 0.4 + swell * 0.5;
    // The sea mirrors the sky overhead by day, the bright horizon by night.
    skyRefl.copy(cur.mid).lerp(cur.top, 0.45).multiplyScalar(0.52).lerp(tmpC.copy(cur.horizon).multiplyScalar(0.55), night);
    ou.uSky.value.copy(skyRefl);
    farMat.emissive.copy(ou.uSky.value).multiplyScalar(0.9);
    ocean.userData.follow(camera.position);
    farSea.position.x = camera.position.x;
    farSea.position.z = camera.position.z;
    lakeMat.emissive.copy(cur.mid).lerp(cur.horizon, 0.4).multiplyScalar(0.8);
    lakeMat.specular.copy(cur.sunC);

    // ── The study: the clock, the calendar, the pad, the theatre ──
    secs += dt * tick;
    var whole = Math.floor(secs), snap = whole + smooth(0, 0.1, secs - whole);
    hands[2].rotation.z = -snap / 60 * Math.PI * 2;
    hands[1].rotation.z = -(12 + secs / 60) / 60 * Math.PI * 2;
    hands[0].rotation.z = -(6 + 12 / 60) / 12 * Math.PI * 2;
    var lift = smooth(0, 0.75, page);
    leaf.rotation.x = -lift * 2.5;
    leaf.position.y = 0.111 + smooth(0.5, 1, page) * 0.12;
    leaf.position.z = 0.009 + lift * 0.06;
    leafMat.opacity = 1 - smooth(0.6, 1, page);
    leaf.visible = page < 0.999;
    inkU.uWrite.value = write;
    var wr = clamp(write, 0, 0.9999) * PAD.rows, wrow = Math.floor(wr), writing = smooth(0, 0.02, write) * (1 - smooth(0.97, 1, write));
    penTip.set((wr - wrow - 0.5) * PAD.w * 0.96, (0.5 - (PAD.row0 + wrow * PAD.rowH + 16) / PAD.py) * PAD.h, 0.002);
    pad.updateMatrix();
    penTip.applyMatrix4(pad.matrix);
    pen.position.lerpVectors(penRest, penTip, writing);
    pen.position.y += writing * Math.abs(Math.sin(time * 18)) * 0.003;
    pen.rotation.set(lerp(Math.PI / 2, 0.5, writing), lerp(0.4, 0, writing), lerp(Math.PI / 2 - 0.2, -0.4, writing));
    curtains.forEach(function (m) { m.scale.x = 1 - curtain * 0.8; });
    stageLight.intensity = 0.05 + curtain * 0.5;
    for (var k = 0; k < foot.length; k++) foot[k].material.opacity = 0.2 + curtain * 0.8;
    toyBoat.position.x = Math.sin(time * 0.7) * 0.06 * curtain;
    toyBoat.rotation.z = Math.sin(time * 1.6) * 0.12;
    toyWave.position.x = Math.sin(time * 1.1) * 0.012;
    toyStar.rotation.z = Math.sin(time * 1.3) * 0.25;
    panes.forEach(function (p) { p.hinge.rotation.y = p.side * open * 1.9; });
    var inside = smooth(-0.6, 0.6, camera.position.z) * (Math.abs(camera.position.x) < 3 ? 1 : 0);
    pf.dt = dt; pf.time = time; pf.wind = 0; pf.snow = 1;
    if (inside > 0.01) dust.update(pf, dustAt, slow);
    dust.points.visible = inside > 0.01;
    dust.points.material.opacity = inside * (1 - night) * 0.8;

    // ── The park: lamps and the flock ──
    var dusk = smooth(0.05, 0.6, night);
    for (k = 0; k < lampGlows.length; k++) {
      lampGlows[k].material.opacity = dusk;
      lampHeads[k].material.emissiveIntensity = dusk * 1.5;
    }
    birdTime.value = slow ? time * 0.5 : time;
    var park = 1 - smooth(0.92, 1, birdT), gullsOn = smooth(6.8, 7.3, row[1]) * (1 - smooth(8.4, 8.9, row[1]));
    var visBirds = park * day * smooth(-1, 4.5, way) * (1 - smooth(9, 10, way));
    FLIGHT.getPoint(clamp(birdT, 0, 1), p3);
    for (k = 0; k < birds.length; k++) {
      var b = birds[k], a = time * b.w * (slow ? 0.4 : 1) + b.ph, ctr = b.gull ? GULLS : p3, vis = b.gull ? gullsOn : visBirds;
      var R = b.gull ? 6 + b.R * 2 : b.R * (1 + (1 - birdT) * 1.5);
      t3.set(ctr.x + b.o.x * (b.gull ? 1.5 : 1) + Math.cos(a) * R, ctr.y + b.o.y + Math.sin(a * 1.7) * 0.6, ctr.z + b.o.z + Math.sin(a) * R);
      tmp.set(t3.x - Math.sin(a) * R * 0.1, t3.y, t3.z + Math.cos(a) * R * 0.1);
      if (!b.gull) { FLIGHT.getPoint(clamp(birdT + 0.03, 0, 1), look); tmp.x += (look.x - p3.x) * 2; tmp.z += (look.z - p3.z) * 2; tmp.y += (look.y - p3.y) * 2; }
      m4.lookAt(tmp, t3, up);
      s3.setScalar((b.gull ? 1.6 : 2.2) * vis + 0.0001);
      m4.scale(s3);
      m4.setPosition(t3);
      flock.setMatrixAt(k, m4);
    }
    flock.instanceMatrix.needsUpdate = true;

    // ── The boats, launched and sailing away down the moon path ──
    var lanterns = smooth(0.2, 0.8, night);
    for (k = 0; k < boats.length; k++) {
      var bt = boats[k], e = smooth(0, 1, clamp((boatT - bt.lag) / (1 - bt.lag), 0, 1));
      var bx = lerp(bt.from[0], bt.to[0], Math.pow(e, 2.2)), bz = lerp(bt.from[1], bt.to[1], Math.pow(e, 2.2));
      var floatIn = smooth(0, 0.06, e);
      wave(bx, bz, time, swell, W);
      bt.group.position.set(bx, lerp(-0.04, W.h, floatIn) + 0.02, bz);
      bt.group.rotation.set(-W.dz * 0.5 * floatIn, Math.atan2(bt.from[0] - bt.to[0], bt.from[1] - bt.to[1]) * 0.6 + Math.sin(time * 0.4 + k) * 0.05, W.dx * 0.6 * floatIn);
      bt.glow.material.opacity = lanterns * (0.85 + 0.15 * Math.sin(time * 5 + k * 2));
      bt.glow.scale.setScalar(0.6 + e * 1.4);
    }
    boatLight.position.copy(boats[0].group.position).add(t3.set(0, 1.2, 0));
    boatLight.intensity = lanterns * 3;
    lantGlow.material.opacity = lanterns * 0.9;
    lantLight.intensity = lanterns * 4;
    winPts.material.opacity = smooth(0.3, 0.9, night) * (1 - silver * 0.3);

    // ── The kitchen and its window ──
    var kin = Math.hypot(camera.position.x - KIT.x, camera.position.z - KIT.z) < 4 ? 1 : 0;
    if (kin) flour.update(pf, flourAt, slow);
    flour.points.visible = !!kin;
    flour.points.material.opacity = 0.6;
    kitGlow.material.opacity = 0.75 * (1 - smooth(6.5, 7.5, row[1])) * smooth(2.5, 4, row[1]);
    ovenLight.intensity = 1.2 * (0.9 + 0.1 * Math.sin(time * 3.1));

    // ── Swimming ──
    var wet = ride * smooth(0.3, 1.0, swell);
    foam.visible = spray.visible = wet > 0.01;
    if (wet > 0.01) {
      for (k = 0; k < FN; k++) {
        var fx = camera.position.x - 18 + ((fBase[k * 2] - camera.position.x + 18) % 36 + 36) % 36;
        var fz = camera.position.z - 18 + ((fBase[k * 2 + 1] - camera.position.z + 18) % 36 + 36) % 36;
        fx += Math.sin(time * 0.3 + k) * 0.3;
        wave(fx, fz, time, swell, W);
        var near = Math.abs(fx - camera.position.x) + Math.abs(fz - camera.position.z) < 3;
        fPos[k * 3] = fx; fPos[k * 3 + 1] = near ? -99 : W.h + 0.03; fPos[k * 3 + 2] = fz;
      }
      foamGeo.attributes.position.needsUpdate = true;
      foam.material.opacity = 0.75 * wet;
      spClock += dt * wet * 30;
      while (spClock > 1) {
        spClock -= 1;
        var sa = (Math.random() - 0.5) * 1.4, sd = 3 + Math.random() * 10;
        camera.getWorldDirection(tmp);
        var ex = camera.position.x + tmp.x * sd + Math.cos(sa) * (Math.random() - 0.5) * 6, ez = camera.position.z + tmp.z * sd;
        wave(ex, ez, time, swell, W);
        for (var sp = 0; sp < 8; sp++) {
          var j = (spNext++ % SPN) * 3;
          spPos[j] = ex + (Math.random() - 0.5) * 0.6; spPos[j + 1] = W.h + 0.05; spPos[j + 2] = ez + (Math.random() - 0.5) * 0.6;
          spVel[j] = (Math.random() - 0.5) * 1.6; spVel[j + 1] = 1 + Math.random() * 2.4; spVel[j + 2] = (Math.random() - 0.3) * 1.2;
          spLife[j / 3] = 1.2;
        }
      }
      for (k = 0; k < SPN; k++) {
        var o3 = k * 3;
        if (spLife[k] <= 0) { spPos[o3 + 1] = -99; continue; }
        spLife[k] -= dt;
        spVel[o3 + 1] -= 6 * dt;
        spPos[o3] += spVel[o3] * dt; spPos[o3 + 1] += spVel[o3 + 1] * dt; spPos[o3 + 2] += spVel[o3 + 2] * dt;
      }
      sprayGeo.attributes.position.needsUpdate = true;
    }

    // ── The door swings open on the sun ──
    leafHinge.rotation.y = smooth(0, 1, doorT) * 1.25;
    doorLight.material.opacity = doorT * 0.9;
    rays.material.opacity = doorT * 0.7 * (1 - smooth(0.6, 1, dream));
    rays.material.rotation = time * 0.02;
    rays.scale.setScalar(7 + doorT * 5);

    // ── The lights of the day, rising into the dream ──
    for (k = 0; k < siteGlows.length; k++) {
      var sg = siteGlows[k], at = SITES[k];
      var on = smooth(k / siteGlows.length * 0.8, k / siteGlows.length * 0.8 + 0.25, lightsT);
      sg.position.copy(at);
      sg.position.y += dream * (14 + k * 6) + Math.sin(time * 0.7 + k) * 0.3 * on;
      sg.scale.setScalar(Math.max(1, camera.position.distanceTo(sg.position) * 0.05) * (1 + dream));
      sg.material.opacity = on * (0.8 + 0.2 * Math.sin(time * 1.4 + k * 1.7));
      sg.visible = on > 0.005;
    }
    var moteAmt = Math.max(lightsT * 0.35, dream);
    pf.snow = moteAmt; pf.wind = f.wind;
    if (moteAmt > 0.01) motes.update(pf, camera.position, slow);
    motes.points.visible = moteAmt > 0.01;

    // ── Render, then paint ──
    postU.uSilver.value = silver;
    postU.uDream.value = dream;
    postU.uTime.value = time;
    gl.setRenderTarget(target);
    gl.render(world, camera);
    gl.setRenderTarget(null);
    gl.render(postScene, postCam);
  }

  // A phone sees a narrow slice and its verse sits mid-screen: turn towards
  // what each beat is about. [waypoint, yaw, pitch]
  var AIM = [[2, 0.15, 0.16], [11, -0.24, 0], [13, 0.1, 0.14], [18, 0.26, 0], [23, -0.12, 0]];
  function portraitAim(way, k) {
    var v = 0;
    for (var a = 0; a < AIM.length; a++) v += AIM[a][k + 1] * (1 - smooth(0, 0.6, Math.abs(way - AIM[a][0])));
    return v;
  }

  return {
    resize: function (w, h, dpr) {
      H = h; portrait = w < h;
      fitCamera(gl, camera, w, h, dpr, small);
      gl.getDrawingBufferSize(bufSize);
      target.setSize(bufSize.x, bufSize.y);
      postU.uRes.value.copy(bufSize);
      starMat.uniforms.uScale.value = gl.getPixelRatio() * (h / 900 + 0.35);
    },
    frame: frame,
    destroy: function () { target.dispose(); disposeAll(postScene); disposeAll(world, gl); }
  };
}

PI.register('extra-day', {
  renderer: renderer3d,
  maxLines: 2,
  scrim: 0.6,
  align: ['left', 'right', 'left', 'right', 'left', 'right', 'left', 'right', 'left', 'left', 'right', 'center'],
  // Panels: 0 the opening couplet; 1-9 the long stanza, one couplet each
  // (pad, park, stars, boats, alone, silver, cakes, swim, door); 10-11 the
  // last stanza, its trailing line on its own.
  keys: function (T) {
    function at(i, d) { return T.start(Math.min(i, T.count - 1)) + d; }
    //  unit          way    look  swell wind  yaw pitch  tick  page  write curt  open birds boats  silver door lights dream
    return [
      [0,             0.00,  0.00, 0.5, 0.15, 0, 0.00,  1.0,  0,    0,    0,    0,   0,    0,    0,     0,   0,    0],
      [0.6,           0.10,  0.05, 0.5, 0.15, 0, 0.00,  1.0,  0,    0,    0,    0,   0,    0,    0,     0,   0,    0],
      [at(0, 0.45),   0.85,  0.12, 0.5, 0.15, 0, 0.00,  0.8,  0,    0,    0,    0,   0,    0,    0,     0,   0,    0],
      [at(0, 0.95),   1.00,  0.22, 0.5, 0.15, 0, 0.00,  0.15, 0.04, 0,    0,    0,   0,    0,    0,     0,   0,    0],  // the clock slows
      [at(0, 1.3),    1.06,  0.30, 0.5, 0.15, 0, 0.00,  0.0,  1,    0,    0,    0,   0,    0,    0,     0,   0,    0],  // and stops: February 29
      [at(1, 0.3),    2.00,  0.45, 0.5, 0.15, 0, 0.00,  0.0,  1,    0,    0,    0,   0,    0,    0,     0,   0,    0],  // the writing pad
      [at(1, 1.15),   2.05,  0.70, 0.5, 0.15, 0, 0.00,  0.0,  1,    1,    1,    0,   0,    0,    0,     0,   0,    0],  // a pantomime
      [at(1, 1.5),    2.70,  0.80, 0.5, 0.15, 0, 0.00,  0.0,  1,    1,    1,    1,   0,    0,    0,     0,   0,    0],  // the window opens
      [at(2, 0.25),   4.40,  1.10, 0.5, 0.20, 0, 0.00,  0.0,  1,    1,    1,    1,   0,    0,    0,     0,   0,    0],  // out into the park
      [at(2, 0.55),   5.00,  1.40, 0.5, 0.20, 0, 0.00,  0.0,  1,    1,    1,    1,   0.08, 0,    0,     0,   0,    0],
      [at(2, 1.3),    6.00,  1.90, 0.5, 0.25, 0, 0.00,  0.0,  1,    1,    1,    1,   0.9,  0,    0,     0,   0,    0],  // the birds in flight
      [at(3, 0.15),   7.00,  2.40, 0.5, 0.20, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    0,    0,     0,   0,    0],
      [at(3, 0.55),   8.00,  3.00, 0.5, 0.15, 0, -0.02, 0.0,  1,    1,    1,    1,   1,    0,    0,     0,   0,    0],  // sitting; the sun sets
      [at(3, 1.0),    8.40,  3.70, 0.5, 0.10, 0, 0.18,  0.0,  1,    1,    1,    1,   1,    0,    0,     0,   0,    0],
      [at(3, 1.45),   8.90,  4.00, 0.5, 0.10, 0, 0.42,  0.0,  1,    1,    1,    1,   1,    0,    0,     0,   0,    0],  // the stars at night
      [at(4, 0.3),    10.6,  4.00, 0.5, 0.10, 0, 0.02,  0.0,  1,    1,    1,    1,   1,    0,    0,     0,   0,    0],
      [at(4, 0.55),   11.0,  4.00, 0.5, 0.10, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    0.04, 0,     0,   0,    0],  // the boats
      [at(4, 1.45),   11.3,  4.00, 0.5, 0.15, 0, 0.02,  0.0,  1,    1,    1,    1,   1,    0.4,  0,     0,   0,    0],  // sailing across the sea
      [at(5, 0.35),   12.5,  4.00, 0.5, 0.15, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    0.62, 0,     0,   0,    0],
      [at(5, 0.9),    13.0,  4.00, 0.5, 0.10, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    0.85, 0,     0,   0,    0],  // one chair
      [at(5, 1.45),   13.15, 4.10, 0.5, 0.10, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    1,    0,     0,   0,    0],
      [at(6, 0.3),    14.8,  4.70, 0.5, 0.10, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    1,    0.5,   0,   0,    0],  // pulling back and up
      [at(6, 0.9),    16.0,  5.00, 0.5, 0.10, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    1,    1,     0,   0,    0],  // a silver scene
      [at(6, 1.3),    16.15, 5.15, 0.5, 0.10, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    1,    0.9,   0,   0,    0],
      [at(7, 0.35),   17.0,  5.90, 0.5, 0.10, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    1,    0,     0,   0,    0],  // down to the kitchen
      [at(7, 0.65),   18.0,  6.00, 0.5, 0.10, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    1,    0,     0,   0,    0],  // cakes
      [at(7, 1.35),   18.1,  6.10, 0.6, 0.10, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    1,    0,     0,   0,    0],
      [at(8, 0.2),    19.0,  6.50, 0.8, 0.15, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    1,    0,     0,   0,    0],
      [at(8, 0.5),    20.3,  6.90, 1.0, 0.20, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    1,    0,     0,   0,    0],
      [at(8, 0.8),    21.2,  7.00, 1.1, 0.25, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    1,    0,     0,   0,    0],  // a swim beyond the shore
      [at(8, 1.5),    22.2,  7.10, 1.1, 0.25, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    1,    0,     0,   0,    0],
      [at(9, 0.35),   22.9,  7.30, 1.0, 0.20, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    1,    0,     0,   0,    0],  // the door
      [at(9, 0.85),   23.0,  7.40, 1.0, 0.20, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    1,    0,     1,   0,    0],  // opens
      [at(9, 1.4),    24.0,  7.60, 0.9, 0.20, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    1,    0,     1,   0,    0],  // through it
      [at(10, 0.25),  25.0,  7.90, 0.8, 0.20, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    1,    0,     1,   0,    0],
      [at(10, 0.7),   26.0,  8.00, 0.7, 0.20, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    1,    0,     1,   0.3,  0],  // the whole world
      [at(10, 1.4),   26.3,  8.10, 0.7, 0.20, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    1,    0,     1,   1,    0],
      [at(11, 0.3),   26.6,  8.40, 0.7, 0.20, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    1,    0,     1,   1,    0.25],  // I'd live my dreams...
      [at(11, 1.1),   27.2,  8.90, 0.7, 0.20, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    1,    0,     1,   1,    0.55],
      [T.total,       28.0,  9.00, 0.7, 0.20, 0, 0.00,  0.0,  1,    1,    1,    1,   1,    1,    0,     1,   1,    0.5]
    ];
  },
  sound: {
    src: '/audio/birds.mp3',
    label: 'Play the birds, the clock and the sea',
    volume: function (row) { return 0.03 + 0.24 * (1 - nightAt(row[1])) * (1 - 0.55 * smooth(17.2, 17.8, row[0]) * (1 - smooth(18.3, 18.9, row[0]))) * (1 - row[16] * 0.6); },
    cues: [
      { stanza: 0, at: -0.75, play: ticks([0, 1, 2, 3, 4, 5, 6, 7], 0.5) },
      { stanza: 0, at: 0.45, play: ticks([0, 1.15, 2.5, 4.2, 6.4], 0.45) },
      { stanza: 0, at: 1.0, play: pageTurn },
      { stanza: 1, at: 0.8, play: musicBox },
      { stanza: 2, at: 0.9, play: flutter },
      { stanza: 3, at: 1.2, play: starNotes },
      { stanza: 4, at: 0.6, play: wash(0.3, 3.0) },
      { stanza: 8, at: 0.65, play: wash(0.5, 2.6) },
      { stanza: 8, at: 1.15, play: wash(0.4, 2.2) },
      { stanza: 9, at: 0.6, play: doorChord },
      { stanza: 11, at: 0.3, play: dreamPad }
    ]
  }
});
