/*
 * Scene for "You Are Tired, (I think)" (E. E. Cummings): a grey evening
 * that turns into a dream, seen first-person.
 *
 * I    A dim room at dusk, "the always puzzle of living and doing": a desk
 *      under the window piled with letters and bills, an unsolved jigsaw of
 *      the moon over the sea (the moon is the piece that is missing), the
 *      clock ticking on. Everything tired and grey.
 * II   "Come with me, then": the casement swings open, the curtains stir and
 *      you drift over the desk and out of the window, turning to watch the
 *      tired lit window fall "far and far away".
 * III  (two parts) The broken toys in the long grass: a tin drum with its
 *      head cracked, a wooden horse whose pull-string has snapped, a kite
 *      with a broken spar and its tail in the grass. On "and—" everything
 *      holds still, the wind drops and the light sinks; "So am I", you sink
 *      down beside them.
 * IV   (two parts) An old iron gate in the garden wall, shut and grown over
 *      with ivy, a rose tucked through its bars. "A dream in my eyes" casts
 *      a soft light ahead; three knocks, and on "Open to me!" the gate swings
 *      wide on the places Nobody knows, glowing, as the evening turns to the
 *      night of Sleep.
 * V    (two parts) Through the gate onto the steppes of dream. A soap bubble
 *      swells before you, floats up and becomes the moon; the jacinth stars
 *      come out one by one, each with a note of their song, set higher for
 *      higher notes. You glide over the unstartled grass to the Only Flower,
 *      which opens and glows, its heart beating, and beyond it the real moon
 *      comes up out of the sea.
 *
 * The bubble is a thin-film shader: a film draining thin at the top, swirled
 * by noise, whose interference colours ride the Fresnel reflection of the
 * sky. Events with their own sound (knocks, the bubble, each star) are
 * pinned to the scroll through the timeline keys() hands over. Columns:
 *   [unit, x, dark, eye, wind, z, yaw, pitch, open, still, gate, dream, moon,
 *    phoneAz, phonePitch]
 * where "eye" is the height above the ground, "dark" deepens the dusk,
 * "dream" turns the world to the night of Sleep, and a portrait screen,
 * which keeps its verse mid-screen, turns by "phoneAz" and tilts by
 * "phonePitch" to keep the subject above or below it.
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, softSprite, skyDome, ribbon, scatter,
         broadleafGeometry, particleField, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout (metres; the cottage's window wall is z = 0, the room z > 0, the
// garden runs out to its wall at WALL_Z and the steppes lie beyond) ───────
var ROOM = { w: 2.6, h: 2.7, d: 4.6 }, WIN = { x: 0.8, y0: 0.95, y1: 2.25 }, WT = 0.25;
var DESK_Y = 0.76;
var TOYS = new THREE.Vector3(2.15, 0, -9.7);
var WALL_Z = -18, LEAF_W = 1.15, POST_X = 1.425;
var FLOWER = new THREE.Vector3(-2.7, 0, -64);
var SEA_Y = -1.2;
var TL = null, EV = null;                     // the timeline, and the events pinned to it

function ground(x, z) {
  if (z > WALL_Z + 0.4) return 0.025 * Math.sin(x * 0.9) * Math.cos(z * 0.7) - 0.035;
  var d = WALL_Z - z;
  var roll = 1.3 * Math.sin(x * 0.045 + 0.6) * Math.cos(z * 0.035) + 0.8 * Math.sin(x * 0.11 - z * 0.07 + 1.1) +
             0.3 * Math.sin(x * 0.3 + z * 0.23);
  var h = smooth(0, 25, d) * roll;
  var kx = x - FLOWER.x, kz = z - FLOWER.z;
  h += 1.5 * Math.exp(-(kx * kx + kz * kz) / 260);            // the knoll of the Only Flower
  var coast = -98 + 8 * Math.sin(x * 0.018 + 1) + 5 * Math.sin(x * 0.05);
  return h - smooth(coast, coast - 40, z) * 9;                // and down to the sea
}

// A direction in the sky: az degrees to the left of straight ahead (-z), el up.
function skyDir(az, el, out) {
  var a = az * Math.PI / 180, e = el * Math.PI / 180;
  return out.set(-Math.sin(a) * Math.cos(e), Math.sin(e), -Math.cos(a) * Math.cos(e));
}

// The jacinth song: one note per star, each set higher for a higher note,
// strung like a melody along the sky under the moon.
var SONG = [659.25, 739.99, 880, 987.77, 880, 1108.73, 987.77, 880, 1318.51];
var SONG_AZ = [60, 56, 52, 48, 44, 40, 36, 32, 28];

// ── Sound: the clock, a breath of air, knocks, the bubble, the song ──────
var noiseBuf = null;
function noise(ac) {
  if (noiseBuf && noiseBuf.sampleRate === ac.sampleRate) return noiseBuf;
  noiseBuf = ac.createBuffer(1, ac.sampleRate * 2, ac.sampleRate);
  var d = noiseBuf.getChannelData(0);
  for (var i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  return noiseBuf;
}
function hiss(ac, out, t, dur, type, freq, q, gain, attack) {
  var s = ac.createBufferSource(), f = ac.createBiquadFilter(), g = ac.createGain();
  s.buffer = noise(ac);
  f.type = type; f.frequency.value = freq; f.Q.value = q;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(gain, t + attack);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  s.connect(f); f.connect(g); g.connect(out);
  s.start(t, Math.random()); s.stop(t + dur + 0.05);
}
function tone(ac, out, t, freq, gain, dur, attack, type) {
  var o = ac.createOscillator(), g = ac.createGain();
  o.type = type || 'sine';
  o.frequency.value = freq;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(gain, t + attack);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g); g.connect(out);
  o.start(t); o.stop(t + dur + 0.05);
  return o;
}
function ticks(ac, out) {
  var t = ac.currentTime + 0.05;
  for (var k = 0; k < 7; k++) hiss(ac, out, t + k, 0.05, 'bandpass', k % 2 ? 2600 : 3300, 6, 0.12, 0.002);
}
function breath(ac, out) {
  hiss(ac, out, ac.currentTime, 2.2, 'lowpass', 700, 0.7, 0.16, 0.9);
}
var KNOCKS = [0, 0.42, 0.84];
function knock(ac, out) {
  var t0 = ac.currentTime + 0.05;
  KNOCKS.forEach(function (d) {
    var t = t0 + d, o = tone(ac, out, t, 150, 0.4, 0.22, 0.004);
    o.frequency.setValueAtTime(150, t);
    o.frequency.exponentialRampToValueAtTime(70, t + 0.12);
    hiss(ac, out, t, 0.08, 'lowpass', 900, 0.8, 0.18, 0.002);
    [523, 1187, 1873].forEach(function (f, j) { tone(ac, out, t + 0.004, f, 0.025 / (j + 1), 0.9, 0.003); });
  });
}
function swell(ac, out) {
  var t = ac.currentTime;
  [220, 329.63, 440, 554.37, 659.25].forEach(function (f, i) { tone(ac, out, t + i * 0.18, f, 0.035, 4.5, 1.4); });
}
function blow(ac, out) {
  var t = ac.currentTime;
  hiss(ac, out, t, 1.4, 'bandpass', 900, 1.2, 0.08, 0.5);
  var o = tone(ac, out, t + 1.1, 620, 0.03, 2.2, 0.4);
  o.frequency.setValueAtTime(620, t + 1.1);
  o.frequency.exponentialRampToValueAtTime(1240, t + 3.2);
}
function chime(freq, gain) {
  return function (ac, out) {
    var t = ac.currentTime;
    [[1, 1], [2.01, 0.32], [3.02, 0.1], [1.003, 0.6]].forEach(function (p) {
      tone(ac, out, t, freq * p[0], (gain || 0.055) * p[1], 3.2 / Math.sqrt(p[0]), 0.015);
    });
  };
}
function heartbeat(ac, out) {
  var t = ac.currentTime;
  for (var k = 0; k < 3; k++) {
    tone(ac, out, t + k * 1.1, 73, 0.16, 0.3, 0.02);
    tone(ac, out, t + k * 1.1 + 0.25, 65, 0.1, 0.3, 0.02);
  }
  [293.66, 440, 587.33].forEach(function (f, i) { tone(ac, out, t + 0.3 + i * 0.25, f, 0.03, 3.5, 0.6); });
}
function moonrise(ac, out) {
  var t = ac.currentTime;
  [146.83, 220, 293.66, 369.99, 440].forEach(function (f, i) { tone(ac, out, t + i * 0.35, f, 0.03, 6.5, 2.2); });
}

// ── Small pieces ─────────────────────────────────────────────────────────
function canvasTexture(w, h, paint, rep) {
  var c = document.createElement('canvas');
  c.width = w; c.height = h;
  paint(c.getContext('2d'), w, h, c);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  if (rep) { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(rep[0], rep[1]); }
  return t;
}

// A w x d rectangle of ground() centred on (cx, cz), coloured per vertex.
function land(w, d, sw, sd, cx, cz, color) {
  var geo = new THREE.PlaneGeometry(w, d, sw, sd).rotateX(-Math.PI / 2).translate(cx, 0, cz);
  var p = geo.attributes.position, cols = new Float32Array(p.count * 3);
  for (var i = 0; i < p.count; i++) {
    var x = p.getX(i), z = p.getZ(i), y = ground(x, z), c = color(x, z, y);
    p.setY(i, y);
    cols[i * 3] = c.r; cols[i * 3 + 1] = c.g; cols[i * 3 + 2] = c.b;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  geo.computeVertexNormals();
  return new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ vertexColors: true }));
}

// One jigsaw piece of side s at (x, y), y down as on a canvas; tabs are
// [top, right, bottom, left]: 1 sticks out, -1 cuts in, 0 is the border.
// `pen` is a canvas context or anything with the same path calls.
function piecePath(pen, x, y, s, tabs) {
  var cs = [[x, y], [x + s, y], [x + s, y + s], [x, y + s]], p = [0, 0];
  function at(a, dx, dy, t, u, v) { p[0] = a[0] + (dx * u + dy * v * t) * s; p[1] = a[1] + (dy * u - dx * v * t) * s; return p; }
  pen.moveTo(x, y);
  for (var e = 0; e < 4; e++) {
    var a = cs[e], b = cs[(e + 1) % 4], dx = (b[0] - a[0]) / s, dy = (b[1] - a[1]) / s, t = tabs[e];
    if (t) {
      at(a, dx, dy, t, 0.36, 0); pen.lineTo(p[0], p[1]);
      var c1 = at(a, dx, dy, t, 0.42, 0.12).slice(), c2 = at(a, dx, dy, t, 0.27, 0.3).slice(), m = at(a, dx, dy, t, 0.5, 0.3).slice();
      pen.bezierCurveTo(c1[0], c1[1], c2[0], c2[1], m[0], m[1]);
      c1 = at(a, dx, dy, t, 0.73, 0.3).slice(); c2 = at(a, dx, dy, t, 0.58, 0.12).slice(); m = at(a, dx, dy, t, 0.64, 0).slice();
      pen.bezierCurveTo(c1[0], c1[1], c2[0], c2[1], m[0], m[1]);
    }
    pen.lineTo(b[0], b[1]);
  }
}
// The same path as a THREE.Shape, y up.
function pieceShape(s, tabs) {
  var sh = new THREE.Shape();
  piecePath({
    moveTo: function (x, y) { sh.moveTo(x, -y); },
    lineTo: function (x, y) { sh.lineTo(x, -y); },
    bezierCurveTo: function (a, b, c, d, e, f) { sh.bezierCurveTo(a, -b, c, -d, e, -f); }
  }, -s / 2, -s / 2, s, tabs);
  return sh;
}

// A cup of petals: a lathe whose rim is scalloped into `lobes` petals,
// curling in (bud) or flaring out (open).
function petalCup(rad, h, lobes, curl, phase, segs) {
  var pts = [];
  for (var i = 0; i <= 8; i++) {
    var t = i / 8;
    pts.push(new THREE.Vector2(rad * (Math.sin(t * Math.PI * 0.62) + curl * t * t), h * t));
  }
  var geo = new THREE.LatheGeometry(pts, segs || 24), p = geo.attributes.position;
  for (i = 0; i < p.count; i++) {
    var x = p.getX(i), y = p.getY(i), z = p.getZ(i), a = Math.atan2(z, x), k = y / h;
    var lobe = Math.abs(Math.sin(lobes * a / 2 + phase));
    var s = 1 + 0.16 * k * lobe;
    p.setXYZ(i, x * s, y * (1 - 0.22 * k * (1 - lobe)), z * s);
  }
  geo.computeVertexNormals();
  return geo;
}

// ── Shaders ──────────────────────────────────────────────────────────────
var NOISE3 =
  'float hash3(vec3 p){ p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }\n' +
  'float noise3(vec3 x){ vec3 i = floor(x); vec3 f = fract(x); f = f * f * (3.0 - 2.0 * f);\n' +
  ' return mix(mix(mix(hash3(i), hash3(i + vec3(1,0,0)), f.x), mix(hash3(i + vec3(0,1,0)), hash3(i + vec3(1,1,0)), f.x), f.y),\n' +
  '            mix(mix(hash3(i + vec3(0,0,1)), hash3(i + vec3(1,0,1)), f.x), mix(hash3(i + vec3(0,1,1)), hash3(i + vec3(1,1,1)), f.x), f.y), f.z); }\n';

// The soap bubble. The film drains thin towards the top and thick at the
// bottom, swirled by slow domain-warped noise; light reflected from its two
// faces interferes, so each wavelength is reflected as sin^2(2 pi n d cos t / l).
// Those colours tint a Fresnel reflection of the sky with two soft
// highlights, a window up and to the left and the rising moon.
var BUBBLE_VS =
  'uniform float uTime; uniform float uWob;\n' +
  'varying vec3 vN; varying vec3 vW; varying vec3 vO;\n' +
  'void main(){ vec3 n = normalize(position);\n' +
  ' float w = sin(n.y * 3.0 + uTime * 5.3) * sin(n.x * 2.0 - uTime * 4.1) * 0.06 + sin(n.z * 4.0 + n.y * 2.0 + uTime * 6.7) * 0.035;\n' +
  ' vec3 p = position * (1.0 + w * 0.55 * uWob);\n' +
  ' vO = n; vec4 wp = modelMatrix * vec4(p, 1.0); vW = wp.xyz; vN = normalize(mat3(modelMatrix) * n);\n' +
  ' gl_Position = projectionMatrix * viewMatrix * wp; }';
var BUBBLE_FS =
  'uniform float uTime; uniform float uAlpha; uniform float uGlow; uniform float uBack;\n' +
  'uniform vec3 uTop; uniform vec3 uHor; uniform vec3 uMoonC; uniform vec3 uMoonDir;\n' +
  'varying vec3 vN; varying vec3 vW; varying vec3 vO;\n' + NOISE3 +
  'void main(){\n' +
  ' vec3 V = normalize(cameraPosition - vW); vec3 N = normalize(vN) * (gl_FrontFacing ? 1.0 : -1.0);\n' +
  ' float c = clamp(dot(N, V), 0.0, 1.0);\n' +
  ' vec3 q = vO * 1.3 + vec3(0.0, -uTime * 0.04, uTime * 0.015);\n' +
  ' float sw = noise3(q + vec3(noise3(q * 1.4 + uTime * 0.03), noise3(q * 1.4 + 7.0 - uTime * 0.035), 0.0) * 0.9);\n' +
  ' float d = mix(180.0, 950.0, smoothstep(0.98, -1.0, vO.y + (sw - 0.5) * 0.5)) + (sw - 0.5) * 160.0;\n' +
  ' d = max(d, 0.0) * (0.92 + 0.08 * sin(uTime * 0.7 + vO.x * 3.0));\n' +
  ' float sinT2 = (1.0 - c * c) / 1.7689, cosT = sqrt(1.0 - sinT2);\n' +
  ' vec3 film = 0.5 - 0.5 * cos(16.713 * d * cosT / vec3(650.0, 532.0, 450.0));\n' +
  ' film = mix(vec3(dot(film, vec3(0.333))), film, 1.2);\n' +
  ' film = clamp(film, 0.0, 1.0) * smoothstep(20.0, 120.0, d);\n' +       // the black film where it is thinnest
  ' float F = 0.04 + 0.96 * pow(1.0 - c, 5.0);\n' +
  ' vec3 R = reflect(-V, N);\n' +
  ' vec3 env = R.y > 0.0 ? mix(uHor * 1.2, uTop, smoothstep(0.0, 0.6, R.y)) : mix(uHor * 0.9, vec3(0.008, 0.01, 0.025), smoothstep(0.0, -0.22, R.y));\n' +
  ' env += uHor * 1.6 * exp(-R.y * R.y / 0.006) + vec3(0.012, 0.012, 0.02);\n' +
  ' vec3 L = normalize(vec3(-0.5, 0.62, 0.6));\n' +
  ' float win = smoothstep(0.972, 0.988, dot(R, L)) * (0.55 + 0.45 * smoothstep(0.2, 0.9, noise3(R * 9.0)));\n' +
  ' float mo = max(dot(R, uMoonDir), 0.0), rim = pow(1.0 - c, 2.2);\n' +
  // Colour gathers at the rim; the middle stays clear. As the moon, it glows
  // softly from the edge in.
  ' float edge = smoothstep(0.25, 0.85, 1.0 - c);\n' +
  ' vec3 col = film * env * 2.2 * F * mix(0.06, 1.0, edge) * (1.0 - 0.6 * uGlow);\n' +
  ' col += (vec3(1.0) * win * 1.2 + uMoonC * (pow(mo, 300.0) * 5.0 + pow(mo, 24.0) * 0.25)) * (0.5 + 0.5 * film);\n' +
  ' col += mix(vec3(0.8, 0.85, 1.0), film, 0.55) * uGlow * 0.3 * pow(1.0 - c, 4.0);\n' +
  ' gl_FragColor = vec4(col * uAlpha * uBack, 1.0);\n' +
  ' #include <tonemapping_fragment>\n #include <colorspace_fragment>\n }';

var STAR_VS = 'attribute vec3 star; attribute vec3 tint; uniform float uTime; uniform float uShow; uniform float uScale;\n' +
  'varying float vA; varying vec3 vC;\n' +
  'void main(){ gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);\n' +
  ' float tw = 0.65 + 0.35 * sin(uTime * (1.0 + star.z * 2.0) + star.z * 40.0);\n' +
  ' vA = tw * smoothstep(star.y, star.y + 0.12, uShow); vC = tint;\n' +
  ' gl_PointSize = star.x * uScale; }';
var STAR_FS = 'varying float vA; varying vec3 vC;\n' +
  'void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;\n' +
  ' float a = smoothstep(0.5, 0.0, d) * vA; gl_FragColor = vec4(vC * a, a);\n #include <colorspace_fragment>\n }';

// Points with a world size: dream flowers in the grass, motes round the flower.
var GLOW_VS = 'attribute vec3 aG; attribute vec3 tint; uniform float uTime; uniform float uShow; uniform float uPx;\n' +
  'varying float vA; varying vec3 vC;\n' +
  'void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;\n' +
  ' float tw = 0.6 + 0.4 * sin(uTime * (0.7 + aG.y * 1.3) + aG.y * 30.0);\n' +
  ' vA = tw * smoothstep(aG.z, aG.z + 0.15, uShow); vC = tint;\n' +
  ' gl_PointSize = clamp(aG.x * uPx / -mv.z, 0.0, 40.0); }';
var GLOW_FS = 'varying float vA; varying vec3 vC;\n' +
  'void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;\n' +
  ' float a = (pow(smoothstep(0.5, 0.0, d), 2.0) * 0.7 + smoothstep(0.15, 0.0, d)) * vA;\n' +
  ' gl_FragColor = vec4(vC * a, a);\n #include <colorspace_fragment>\n }';

// The sea: a dark swell that mirrors the sky, with the moon's glitter path.
var SEA_FS =
  'uniform float uTime; uniform vec3 uTop; uniform vec3 uHor; uniform vec3 uDeep; uniform vec3 uMoonC; uniform vec3 uMoonDir; uniform float uMoon;\n' +
  'uniform vec3 uFogColor; uniform float uFogDensity;\n' +
  'varying vec3 vW;\n' + NOISE3 +
  'void main(){\n' +
  ' vec3 toCam = cameraPosition - vW; float dist = length(toCam); vec3 V = toCam / dist;\n' +
  ' vec2 p = vW.xz * vec2(0.9, 0.35);\n' +
  ' float e = 0.06; vec3 a = vec3(p.x, p.y, uTime * 0.35);\n' +
  ' float h0 = noise3(a * 0.8) + 0.5 * noise3(a * 2.1 + 3.0) + 0.25 * noise3(a * 4.7 + 9.0);\n' +
  ' float hx = noise3((a + vec3(e, 0.0, 0.0)) * 0.8) + 0.5 * noise3((a + vec3(e, 0.0, 0.0)) * 2.1 + 3.0) + 0.25 * noise3((a + vec3(e, 0.0, 0.0)) * 4.7 + 9.0);\n' +
  ' float hz = noise3((a + vec3(0.0, e, 0.0)) * 0.8) + 0.5 * noise3((a + vec3(0.0, e, 0.0)) * 2.1 + 3.0) + 0.25 * noise3((a + vec3(0.0, e, 0.0)) * 4.7 + 9.0);\n' +
  ' float k = 0.5 / (1.0 + dist * 0.07);\n' +
  ' vec3 N = normalize(vec3(-(hx - h0) / e * k, 1.0, -(hz - h0) / e * k * 0.4));\n' +
  ' vec3 R = reflect(-V, N); R.y = abs(R.y);\n' +
  ' float fres = 0.02 + 0.98 * pow(1.0 - max(dot(N, V), 0.0), 5.0);\n' +
  ' vec3 col = mix(uDeep, mix(uHor, uTop, smoothstep(0.0, 0.5, R.y)), 0.35 + 0.65 * fres);\n' +
  ' float m = max(dot(R, uMoonDir), 0.0);\n' +
  ' col += uMoonC * uMoon * (pow(m, 1400.0) * 10.0 + pow(m, 60.0) * 0.4 + pow(m, 10.0) * 0.05);\n' +
  // The glitter path: a band in line with the moon, wider nearer to you,
  // broken into sparkles that come and go.
  ' vec2 md = normalize(uMoonDir.xz), vd = normalize(-V.xz);\n' +
  ' float az = acos(clamp(dot(md, vd), -1.0, 1.0)), wid = 0.012 + V.y * 0.9;\n' +
  ' float path = exp(-az * az / (wid * wid)) * smoothstep(0.0, 0.004, V.y);\n' +
  ' float spk = noise3(vec3(vW.x * 1.6, vW.z * 0.25, uTime * 1.3)); spk = mix(0.6, pow(spk, 4.0) * 5.0, 1.0 / (1.0 + dist * 0.004));\n' +
  ' col += uMoonC * uMoon * path * (0.12 + spk) * (0.5 + 0.5 * smoothstep(0.03, 0.0, V.y));\n' +
  ' gl_FragColor = vec4(col, 1.0);\n' +
  ' #include <tonemapping_fragment>\n #include <colorspace_fragment>\n' +
  ' float fogF = 1.0 - exp(-uFogDensity * uFogDensity * dist * dist);\n' +
  ' gl_FragColor.rgb = mix(gl_FragColor.rgb, uFogColor, fogF);\n}';

// Grass that sways with the wind (and holds still on "and—").
function swayMaterial(clock, wind, color) {
  var m = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, color: color || '#ffffff' });
  m.onBeforeCompile = function (sh) {
    sh.uniforms.uClock = clock;
    sh.uniforms.uWind = wind;
    sh.vertexShader = 'uniform float uClock; uniform float uWind;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n vec3 iP = instanceMatrix[3].xyz; float b = position.y * position.y * 3.0;\n' +
      ' transformed.x += (sin(uClock * 1.6 + iP.x * 0.6 + iP.z * 0.35) * 0.7 + sin(uClock * 2.9 + iP.z * 1.3) * 0.3) * uWind * 0.11 * b;\n' +
      ' transformed.z += cos(uClock * 1.3 + iP.x * 0.4) * uWind * 0.04 * b;');
  };
  return m;
}

// A tuft of blades, white so instance colours tint it, darker at the roots.
function tuftGeometry(r, blades, h) {
  var pos = [], col = [], nor = [];
  for (var b = 0; b < blades; b++) {
    var a = r() * Math.PI * 2, ox = (r() - 0.5) * 0.16, oz = (r() - 0.5) * 0.16, w = 0.006 + r() * 0.006, hh = h * (0.55 + r() * 0.65);
    var lean = (r() - 0.3) * 0.25, ca = Math.cos(a), sa = Math.sin(a);
    var P = [[-w, 0], [w, 0], [-w * 0.55, hh * 0.55], [w * 0.55, hh * 0.55], [lean * hh * 0.2, hh]];
    var tris = [[0, 1, 2], [1, 3, 2], [2, 3, 4]];
    tris.forEach(function (t) {
      t.forEach(function (k) {
        var px = P[k][0], py = P[k][1], bend = lean * py * py / hh;
        pos.push(ox + px * ca + bend * sa, py, oz - px * sa + bend * ca);
        var g = 0.35 + 0.65 * (py / hh);
        col.push(g, g, g);
        nor.push(0, 1, 0);
      });
    });
  }
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  return geo;
}

// ── Renderer ─────────────────────────────────────────────────────────────
function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(1923), slow = env.reduceMotion;
  var gl = makeRenderer(canvas, { clear: '#40434f' });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#5c5f6c', 0.028);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.03, 3000);
  var pxScale = { value: 800 }, clock = { value: 0 }, sway = { value: 0.3 };
  var tmpA = new THREE.Vector3(), tmpB = new THREE.Vector3(), tmpC = new THREE.Color(), tmpD = new THREE.Color();
  var m4 = new THREE.Matrix4(), q4 = new THREE.Quaternion(), s3 = new THREE.Vector3(), p3 = new THREE.Vector3(), eul = new THREE.Euler();
  var UP = new THREE.Vector3(0, 1, 0);

  // ── Sky: a grey dusk that deepens, then the night of Sleep ─────────────
  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#3c4258', mid: '#5e6278', horizon: '#8c8894', sun: '#000000' }, 1500);
  sky.add(dome.mesh);
  var SKY = {
    dusk:  [new THREE.Color('#3a4056'), new THREE.Color('#5c6076'), new THREE.Color('#8a8692')],
    dark:  [new THREE.Color('#12141c'), new THREE.Color('#1e2029'), new THREE.Color('#2e2f37')],
    dream: [new THREE.Color('#05061a'), new THREE.Color('#161444'), new THREE.Color('#3e3474')]
  };
  var MOONGLOW = new THREE.Color('#6a4a5c');

  var SN = small ? 2400 : 4800, sPos = [], sAttr = [], sTint = [], v = new THREE.Vector3();
  var STAR_TINTS = [new THREE.Color('#dfe6ff'), new THREE.Color('#ffb27a'), new THREE.Color('#9aa0ff'), new THREE.Color('#fff2dc')];
  for (var i = 0; i < SN; i++) {
    var y = 0.02 + Math.pow(r(), 0.8) * 0.98, th = r() * Math.PI * 2, sr = Math.sqrt(1 - y * y);
    v.set(sr * Math.cos(th), y, sr * Math.sin(th));
    sPos.push(v.x * 1300, v.y * 1300, v.z * 1300);
    sAttr.push(0.8 + Math.pow(r(), 4) * 3.2, r() * 0.85, r());
    var tc = STAR_TINTS[r() < 0.6 ? 0 : 1 + Math.floor(r() * 3)];
    sTint.push(tc.r, tc.g, tc.b);
  }
  var starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.Float32BufferAttribute(sPos, 3));
  starGeo.setAttribute('star', new THREE.Float32BufferAttribute(sAttr, 3));
  starGeo.setAttribute('tint', new THREE.Float32BufferAttribute(sTint, 3));
  var starMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
    uniforms: { uTime: clock, uShow: { value: 0 }, uScale: { value: 1 } }, vertexShader: STAR_VS, fragmentShader: STAR_FS
  });
  var stars = new THREE.Points(starGeo, starMat);
  stars.frustumCulled = false;
  sky.add(stars);

  // The singing stars, with four-point flares.
  var flareTex = canvasTexture(128, 128, function (x) {
    var g = x.createRadialGradient(64, 64, 0, 64, 64, 64);
    g.addColorStop(0, 'rgba(255,255,255,1)'); g.addColorStop(0.1, 'rgba(255,255,255,0.95)');
    g.addColorStop(0.22, 'rgba(255,255,255,0.3)'); g.addColorStop(0.5, 'rgba(255,255,255,0.06)'); g.addColorStop(1, 'rgba(255,255,255,0)');
    x.fillStyle = g; x.fillRect(0, 0, 128, 128);
    x.globalCompositeOperation = 'lighter';
    [[124, 5, 0], [5, 124, 0], [60, 3, 1], [3, 60, 1]].forEach(function (s) {
      var lg = x.createRadialGradient(0, 0, 0, 0, 0, s[0] > s[1] ? s[0] / 2 : s[1] / 2);
      lg.addColorStop(0, 'rgba(255,255,255,0.9)'); lg.addColorStop(0.4, 'rgba(255,255,255,0.3)'); lg.addColorStop(1, 'rgba(255,255,255,0)');
      x.save(); x.translate(64, 64); if (s[2]) x.rotate(Math.PI / 4);
      x.fillStyle = lg; x.fillRect(-s[0] / 2, -s[1] / 2, s[0], s[1]); x.restore();
    });
  });
  var singers = SONG.map(function (f, k) {
    var sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: flareTex, color: k % 2 ? '#9488ff' : '#ff9a5c',
      blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false, opacity: 0 }));
    sp.visible = false;
    sky.add(sp);
    return { sprite: sp, az: SONG_AZ[k], el: 7 + (Math.log(f / 600) / Math.log(2)) * 17 };
  });

  // A faint thread from note to note.
  var tunePos = new Float32Array((SONG.length - 1) * 6), tuneGeo = new THREE.BufferGeometry();
  tuneGeo.setAttribute('position', new THREE.BufferAttribute(tunePos, 3));
  var tune = new THREE.LineSegments(tuneGeo, new THREE.LineBasicMaterial({ color: '#c4b4ff', transparent: true, opacity: 0,
    blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
  tune.frustumCulled = false;
  sky.add(tune);

  // The real moon, rising out of the sea at the end.
  var moonTex = canvasTexture(256, 256, function (x) {
    var g = x.createRadialGradient(118, 112, 10, 128, 128, 118);
    g.addColorStop(0, '#fff3d8'); g.addColorStop(0.85, '#f6d9a6'); g.addColorStop(1, '#e8bf86');
    x.fillStyle = g; x.beginPath(); x.arc(128, 128, 118, 0, Math.PI * 2); x.fill();
    [[92, 98, 34], [150, 84, 26], [158, 150, 38], [104, 162, 22], [128, 118, 18]].forEach(function (m) {
      var mg = x.createRadialGradient(m[0], m[1], 0, m[0], m[1], m[2]);
      mg.addColorStop(0, 'rgba(180,140,100,0.16)'); mg.addColorStop(1, 'rgba(180,140,100,0)');
      x.fillStyle = mg; x.beginPath(); x.arc(m[0], m[1], m[2], 0, Math.PI * 2); x.fill();
    });
  });
  var moon = new THREE.Sprite(new THREE.SpriteMaterial({ map: moonTex, transparent: true, depthWrite: false, fog: false }));
  var warmTex = softSprite('rgba(255,226,180,1)', 'rgba(255,200,150,0)');
  var moonHalo = new THREE.Sprite(new THREE.SpriteMaterial({ map: warmTex, transparent: true, depthWrite: false, fog: false,
    blending: THREE.AdditiveBlending, opacity: 0 }));
  sky.add(moonHalo, moon);
  var MOON_C = new THREE.Color('#ffd9a6'), moonDir = new THREE.Vector3();

  // ── Light ───────────────────────────────────────────────────────────────
  var hemi = new THREE.HemisphereLight('#8a90a8', '#2a2828', 1.0);
  var key = new THREE.DirectionalLight('#a8acc0', 0.6);
  key.position.set(-6, 10, -20);
  var moonLight = new THREE.DirectionalLight('#ffd8a8', 0);
  // A cool early moonlight over the garden, low in II and III.
  var gardenMoon = new THREE.DirectionalLight('#9eaedc', 0);
  gardenMoon.position.set(-12, 14, 6);
  world.add(gardenMoon);
  world.add(hemi, key, moonLight);
  var KEY_DUSK = new THREE.Color('#a4a8bc'), KEY_DREAM = new THREE.Color('#bcc6ff');
  var HEMI_DUSK = new THREE.Color('#8a90a8'), HEMI_DREAM = new THREE.Color('#7a80cc');
  var GND_DUSK = new THREE.Color('#2a2828'), GND_DREAM = new THREE.Color('#1c1a3a');
  // "A dream in my eyes": a soft light that goes ahead of you.
  var eyeLight = new THREE.PointLight('#f0d8ff', 0, 4.5, 1.2);
  world.add(eyeLight);

  // ── The room ───────────────────────────────────────────────────────────
  var paperTex = canvasTexture(256, 256, function (x, w, h) {
    x.fillStyle = '#868892'; x.fillRect(0, 0, w, h);
    for (var k = 0; k < 8; k++) { x.fillStyle = k % 2 ? 'rgba(255,255,255,0.035)' : 'rgba(0,0,0,0.035)'; x.fillRect(k * 32, 0, 14, h); }
    x.fillStyle = 'rgba(52,54,62,0.2)';
    for (var j = 0; j < 16; j++) {
      var cx = (j % 4) * 64 + (Math.floor(j / 4) % 2) * 32 + 16, cy = Math.floor(j / 4) * 64 + 32;
      x.beginPath(); x.ellipse(cx, cy, 3.5, 7, 0.4, 0, 6.283); x.fill();
      x.beginPath(); x.ellipse(cx + 6, cy - 4, 3, 6, -0.6, 0, 6.283); x.fill();
    }
  }, [5, 2]);
  var wallpaper = new THREE.MeshLambertMaterial({ map: paperTex });
  var plaster = new THREE.MeshLambertMaterial({ color: '#8a8a88' });
  var whitewash = new THREE.MeshLambertMaterial({ map: canvasTexture(256, 256, function (x, w, h) {
    x.fillStyle = '#b4b0a8'; x.fillRect(0, 0, w, h);
    for (var k = 0; k < 220; k++) { x.fillStyle = 'rgba(' + (r() < 0.5 ? '90,86,80' : '255,255,250') + ',' + (0.04 + r() * 0.06) + ')'; x.fillRect(r() * w, r() * h, 4 + r() * 30, 2 + r() * 14); }
    var g = x.createLinearGradient(0, h * 0.7, 0, h);
    g.addColorStop(0, 'rgba(60,64,50,0)'); g.addColorStop(1, 'rgba(60,64,50,0.45)');
    x.fillStyle = g; x.fillRect(0, 0, w, h);
  }, [2, 1]) });
  var floorTex = canvasTexture(256, 256, function (x, w, h) {
    for (var k = 0; k < 8; k++) {
      x.fillStyle = 'hsl(28,' + (14 + r() * 6) + '%,' + (22 + r() * 6) + '%)';
      x.fillRect(0, k * 32, w, 32);
      x.fillStyle = 'rgba(0,0,0,0.35)'; x.fillRect(0, k * 32, w, 1.5);
      x.fillRect(r() * w, k * 32, 1.5, 32);
      for (var j = 0; j < 6; j++) { x.fillStyle = 'rgba(0,0,0,0.06)'; x.fillRect(0, k * 32 + 4 + r() * 24, w, 1); }
    }
  }, [3, 4]);
  var room = new THREE.Group();
  world.add(room);
  function box(w, h, d, x, y, z, mat, parent) {
    var m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
    m.position.set(x, y, z);
    (parent || room).add(m);
    return m;
  }
  // Faces: +x, -x, +y, -y, +z, -z. Walls show wallpaper inside, whitewash out.
  var frontMats = [plaster, plaster, plaster, plaster, wallpaper, whitewash];
  var sideW = ROOM.w - WIN.x, H = ROOM.h + 0.15;
  box(sideW + WT, H, WT, -WIN.x - (sideW + WT) / 2, H / 2, -WT / 2, frontMats);
  box(sideW + WT, H, WT, WIN.x + (sideW + WT) / 2, H / 2, -WT / 2, frontMats);
  box(WIN.x * 2, WIN.y0, WT, 0, WIN.y0 / 2, -WT / 2, frontMats);
  box(WIN.x * 2, H - WIN.y1, WT, 0, (H + WIN.y1) / 2, -WT / 2, frontMats);
  box(WT, H, ROOM.d + WT * 2, -ROOM.w - WT / 2, H / 2, ROOM.d / 2, [wallpaper, whitewash, plaster, plaster, plaster, plaster]);
  box(WT, H, ROOM.d + WT * 2, ROOM.w + WT / 2, H / 2, ROOM.d / 2, [whitewash, wallpaper, plaster, plaster, plaster, plaster]);
  box(ROOM.w * 2 + WT * 2, H, WT, 0, H / 2, ROOM.d + WT / 2, [plaster, plaster, plaster, plaster, whitewash, wallpaper]);
  box(ROOM.w * 2, 0.1, ROOM.d, 0, -0.05, ROOM.d / 2, new THREE.MeshLambertMaterial({ map: floorTex }));
  box(ROOM.w * 2, 0.08, ROOM.d, 0, ROOM.h + 0.04, ROOM.d / 2, new THREE.MeshLambertMaterial({ color: '#6c6c6e' }));
  box(ROOM.w * 2, 0.12, 0.05, 0, ROOM.h - 0.06, 0.03, plaster);
  // The roof, a chimney, the door and a step, seen when you look back.
  var roofShape = new THREE.Shape();
  roofShape.moveTo(0.6, 2.82); roofShape.lineTo(-ROOM.d - 0.6, 2.82); roofShape.lineTo(-ROOM.d / 2, 4.55); roofShape.lineTo(0.6, 2.82);
  var roofGeo = new THREE.ExtrudeGeometry(roofShape, { depth: ROOM.w * 2 + 1.0, bevelEnabled: false }).rotateY(Math.PI / 2).translate(-ROOM.w - 0.5, 0, 0);
  var slate = new THREE.MeshLambertMaterial({ map: canvasTexture(128, 128, function (x, w, h) {
    x.fillStyle = '#3a3c44'; x.fillRect(0, 0, w, h);
    for (var k = 0; k < 8; k++) for (var j = 0; j < 8; j++) {
      x.fillStyle = 'hsl(225,' + (6 + r() * 6) + '%,' + (20 + r() * 8) + '%)';
      x.fillRect(j * 16 + (k % 2) * 8, k * 16, 15, 15);
    }
  }, [6, 4]) });
  room.add(new THREE.Mesh(roofGeo, [whitewash, slate]));
  box(0.55, 1.6, 0.55, 1.6, 4.3, 3.0, whitewash);
  box(0.62, 0.1, 0.62, 1.6, 5.1, 3.0, plaster);
  box(0.92, 2.0, 0.06, -1.75, 1.0, -WT - 0.03, new THREE.MeshLambertMaterial({ color: '#5e544a' }));
  box(1.2, 0.12, 0.5, -1.75, 0.06, -WT - 0.3, plaster);
  box(WIN.x * 2 + 0.2, 0.07, 0.14, 0, WIN.y0 - 0.03, -WT - 0.05, plaster);           // outer sill

  // The window: frame, inner sill and casements hinged at the outer edges.
  var paint = new THREE.MeshLambertMaterial({ color: '#9a9a96' });
  box(WIN.x * 2 + 0.3, 0.05, 0.26, 0, WIN.y0 - 0.025, 0.04, paint);
  box(0.07, WIN.y1 - WIN.y0, 0.1, -WIN.x + 0.035, (WIN.y0 + WIN.y1) / 2, -0.2, paint);
  box(0.07, WIN.y1 - WIN.y0, 0.1, WIN.x - 0.035, (WIN.y0 + WIN.y1) / 2, -0.2, paint);
  box(WIN.x * 2, 0.07, 0.1, 0, WIN.y1 - 0.035, -0.2, paint);
  var glass = new THREE.MeshLambertMaterial({ color: '#9aa4bc', transparent: true, opacity: 0.14, depthWrite: false });
  var paneW = WIN.x - 0.07, paneH = WIN.y1 - WIN.y0 - 0.07;
  var panes = [-1, 1].map(function (side) {
    var hinge = new THREE.Group(), pane = new THREE.Group();
    [[paneW / 2, paneH - 0.025, paneW, 0.05], [paneW / 2, 0.025, paneW, 0.05], [0.025, paneH / 2, 0.05, paneH], [paneW - 0.025, paneH / 2, 0.05, paneH],
     [paneW / 2, paneH / 2, paneW, 0.03], [paneW / 2, paneH * 0.75, paneW, 0.025], [paneW / 2, paneH * 0.25, paneW, 0.025]].forEach(function (b) {
      box(b[2], b[3], 0.04, b[0], b[1], 0, paint, pane);
    });
    var g = new THREE.Mesh(new THREE.PlaneGeometry(paneW, paneH), glass);
    g.position.set(paneW / 2, paneH / 2, 0);
    pane.add(g);
    pane.scale.x = -side;
    hinge.add(pane);
    hinge.position.set(side * (WIN.x - 0.07), WIN.y0 + 0.035, -0.23);
    room.add(hinge);
    return { hinge: hinge, side: side };
  });
  // Curtains, a tired mauve, stirring when the window opens.
  var breeze = { value: 0 };
  var curtainMat = new THREE.MeshLambertMaterial({ color: '#7c7480', side: THREE.DoubleSide });
  curtainMat.onBeforeCompile = function (sh) {
    sh.uniforms.uBreeze = breeze;
    sh.uniforms.uClock = clock;
    sh.vertexShader = 'uniform float uBreeze; uniform float uClock;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n float hang = clamp((1.2 - position.y) / 2.4, 0.0, 1.0);\n' +
      ' transformed.z += sin(position.x * 21.0) * 0.04 + uBreeze * hang * hang * (0.22 + 0.1 * sin(uClock * 1.9 + position.x * 3.0)) * (0.7 + 0.3 * sin(uClock * 0.8));\n' +
      ' transformed.x += uBreeze * hang * 0.06 * sin(uClock * 1.4 + position.y);');
  };
  [-1, 1].forEach(function (side) {
    var c = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 2.4, 36, 14), curtainMat);
    c.position.set(side * (WIN.x + 0.3), 1.32, 0.12);
    room.add(c);
  });
  box(WIN.x * 2 + 1.4, 0.03, 0.03, 0, 2.52, 0.12, paint);

  // The desk and its clutter.
  var wood = new THREE.MeshLambertMaterial({ color: '#5a4c40' });
  box(1.76, 0.04, 0.76, 0, DESK_Y - 0.02, 0.43, wood);
  [[-0.82, 0.1], [0.82, 0.1], [-0.82, 0.76], [0.82, 0.76]].forEach(function (l) { box(0.05, DESK_Y - 0.04, 0.05, l[0], (DESK_Y - 0.04) / 2, l[1], wood); });
  box(1.6, 0.12, 0.02, 0, DESK_Y - 0.1, 0.8, new THREE.MeshLambertMaterial({ color: '#504438' }));
  box(0.12, 0.015, 0.015, 0, DESK_Y - 0.1, 0.815, paint);

  // The jigsaw, a moon over the sea, with the moon missing.
  var JW = 500, JH = 375, PS = 62.5, COLS = 8, ROWS = 6, tabH = [], tabV = [];
  for (var rr = 0; rr < ROWS; rr++) { tabH.push([]); tabV.push([]); for (var cc = 0; cc < COLS; cc++) { tabH[rr].push(r() < 0.5 ? 1 : -1); tabV[rr].push(r() < 0.5 ? 1 : -1); } }
  function tabsOf(rw, cl) {
    return [rw > 0 ? -tabH[rw - 1][cl] : 0, cl < COLS - 1 ? tabV[rw][cl] : 0, rw < ROWS - 1 ? tabH[rw][cl] : 0, cl > 0 ? -tabV[rw][cl - 1] : 0];
  }
  var MISSING = [[1, 5], [2, 5], [0, 1], [3, 2], [4, 6], [5, 3], [2, 0], [4, 4], [1, 2], [3, 7]];
  var pic = document.createElement('canvas');
  pic.width = JW; pic.height = JH;
  (function (x) {
    var g = x.createLinearGradient(0, 0, 0, JH * 0.56);
    g.addColorStop(0, '#4e5468'); g.addColorStop(1, '#a29ea4');
    x.fillStyle = g; x.fillRect(0, 0, JW, JH * 0.56);
    var s = x.createLinearGradient(0, JH * 0.56, 0, JH);
    s.addColorStop(0, '#5c6272'); s.addColorStop(1, '#262c38');
    x.fillStyle = s; x.fillRect(0, JH * 0.56, JW, JH);
    x.fillStyle = 'rgba(200,196,200,0.25)';
    [[60, 70, 90, 12], [190, 120, 120, 10], [400, 60, 80, 9]].forEach(function (c) { x.beginPath(); x.ellipse(c[0], c[1], c[2], c[3], 0, 0, 6.283); x.fill(); });
    x.fillStyle = '#ece6d4'; x.beginPath(); x.arc(342, 96, 24, 0, 6.283); x.fill();
    for (var k = 0; k < 40; k++) {
      var yy = JH * 0.57 + k * 4, ww = 6 + r() * 26 * (1 - k / 50);
      x.fillStyle = 'rgba(230,224,206,' + (0.5 - k / 90) + ')';
      x.fillRect(342 - ww / 2 + (r() - 0.5) * 18, yy, ww, 1.5);
    }
  })(pic.getContext('2d'));
  var pieceColor = function (rw, cl) {
    var d = pic.getContext('2d').getImageData(cl * PS + 8, rw * PS + 8, PS - 16, PS - 16).data, s = [0, 0, 0];
    for (var k = 0; k < d.length; k += 4) { s[0] += d[k]; s[1] += d[k + 1]; s[2] += d[k + 2]; }
    var n = d.length / 4;
    return new THREE.Color().setRGB(s[0] / n / 255, s[1] / n / 255, s[2] / n / 255, THREE.SRGBColorSpace);
  };
  var jigTex = canvasTexture(520, 395, function (x) {
    x.fillStyle = '#2e2620'; x.fillRect(0, 0, 520, 395);
    x.save(); x.translate(10, 10);
    x.drawImage(pic, 0, 0);
    MISSING.forEach(function (m) {
      x.beginPath(); piecePath(x, m[1] * PS, m[0] * PS, PS, tabsOf(m[0], m[1])); x.closePath();
      x.fillStyle = '#2a231d'; x.fill();
      x.strokeStyle = 'rgba(0,0,0,0.5)'; x.lineWidth = 2; x.stroke();
    });
    x.strokeStyle = 'rgba(20,18,16,0.35)'; x.lineWidth = 1;
    for (var rw = 0; rw < ROWS; rw++) for (var cl = 0; cl < COLS; cl++) {
      x.beginPath(); piecePath(x, cl * PS, rw * PS, PS, tabsOf(rw, cl)); x.closePath(); x.stroke();
    }
    x.restore();
  });
  var boardMat = new THREE.MeshLambertMaterial({ color: '#2e2620' });
  var board = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.008, 0.395), [boardMat, boardMat, new THREE.MeshLambertMaterial({ map: jigTex }), boardMat, boardMat, boardMat]);
  board.position.set(0.04, DESK_Y + 0.004, 0.44);
  board.rotation.y = 0.07;
  room.add(board);
  // Loose pieces on the desk (the moon's are not among them).
  var loose = [];
  MISSING.slice(2).forEach(function (m, k) {
    var g = new THREE.ExtrudeGeometry(pieceShape(0.0625, tabsOf(m[0], m[1])), { depth: 0.004, bevelEnabled: false, curveSegments: 4 });
    g.rotateX(-Math.PI / 2).rotateY(r() * 6.28);
    var px = k % 2 ? -0.3 - r() * 0.22 : 0.36 + r() * 0.2, pz = 0.22 + r() * 0.5;
    if (k === 3) { px = 0.02; pz = 0.72; }
    g.translate(px, DESK_Y + 0.001 + k * 0.0004, pz);
    loose.push(tinted(g, r() < 0.25 ? '#3a3430' : pieceColor(m[0], m[1])));
  });
  room.add(new THREE.Mesh(merge(loose), new THREE.MeshLambertMaterial({ vertexColors: true })));

  // Letters and bills.
  function sheet(kind) {
    return canvasTexture(128, 180, function (x, w, h) {
      x.fillStyle = kind === 2 ? '#c4c0b4' : '#d0ccc2'; x.fillRect(0, 0, w, h);
      x.strokeStyle = 'rgba(60,60,70,0.55)'; x.lineWidth = 1;
      if (kind === 1) {
        x.fillStyle = 'rgba(50,50,60,0.6)'; x.font = '9px monospace';
        x.fillRect(10, 10, 50, 6);
        for (var k = 0; k < 13; k++) { x.fillRect(10, 30 + k * 10, 40 + r() * 30, 2); x.fillText(String(Math.floor(r() * 900 + 10)) + '.' + Math.floor(r() * 90 + 10), 86, 34 + k * 10); }
        x.fillRect(80, 166, 40, 1.5);
      } else {
        for (k = 0; k < 15; k++) {
          var y0 = 16 + k * 10.5, xx = 10;
          x.beginPath(); x.moveTo(xx, y0);
          while (xx < w - 14 - (k === 14 ? 50 : 0)) { xx += 2 + r() * 5; x.lineTo(xx, y0 + (r() - 0.5) * 3.2); }
          x.stroke();
        }
      }
    });
  }
  var sheetTex = [sheet(0), sheet(1), sheet(2)];
  var sheetGeo = new THREE.PlaneGeometry(0.21, 0.297).rotateX(-Math.PI / 2);
  [[-0.58, 0.42, 0.3, 0], [-0.5, 0.56, -0.25, 1], [-0.66, 0.3, 0.7, 2], [0.56, 0.36, -0.4, 1], [0.62, 0.6, 0.15, 0], [-0.3, 0.68, 1.3, 2],
   [0.38, 0.62, 0.9, 0]].forEach(function (s, k) {
    var m = new THREE.Mesh(sheetGeo, new THREE.MeshLambertMaterial({ map: sheetTex[s[3]], polygonOffset: true, polygonOffsetFactor: -1 - k, polygonOffsetUnits: -1 - k }));
    m.position.set(s[0], DESK_Y + 0.0006 * (k + 1), s[1]);
    m.rotation.y = s[2];
    room.add(m);
  });
  // Books, a cold cup of tea, a pencil, two crumpled sheets.
  [['#4a4e5c', 0.05], ['#5c4a48', 0.04], ['#4e564a', 0.045]].forEach(function (b, k) {
    var bk = box(0.24, b[1], 0.17, 0.66, DESK_Y + b[1] / 2 + k * 0.046, 0.2, new THREE.MeshLambertMaterial({ color: b[0] }));
    bk.rotation.y = 0.15 - k * 0.22;
  });
  var china = new THREE.MeshLambertMaterial({ color: '#b6b4b0' });
  var cup = new THREE.Mesh(new THREE.LatheGeometry([[0, 0], [0.03, 0], [0.04, 0.02], [0.045, 0.065], [0.042, 0.065], [0.037, 0.022], [0, 0.012]].map(function (p) {
    return new THREE.Vector2(p[0], p[1]);
  }), 20), china);
  cup.position.set(0.7, DESK_Y + 0.006, 0.66);
  room.add(cup);
  var saucer = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.06, 0.008, 24), china);
  saucer.position.set(0.7, DESK_Y + 0.004, 0.66);
  room.add(saucer);
  var tea = new THREE.Mesh(new THREE.CircleGeometry(0.04, 20).rotateX(-Math.PI / 2), new THREE.MeshLambertMaterial({ color: '#3a2c22' }));
  tea.position.set(0.7, DESK_Y + 0.05, 0.66);
  room.add(tea);
  var pencil = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 0.17, 6).rotateZ(Math.PI / 2), new THREE.MeshLambertMaterial({ color: '#8a7a4a' }));
  pencil.position.set(-0.12, DESK_Y + 0.012, 0.74);
  pencil.rotation.y = 0.5;
  room.add(pencil);
  [[-0.78, 0.66], [0.26, 0.15]].forEach(function (b) {
    var g = new THREE.IcosahedronGeometry(0.032, 1), p = g.attributes.position;
    for (var k = 0; k < p.count; k++) { var s = 0.75 + r() * 0.45; p.setXYZ(k, p.getX(k) * s, p.getY(k) * s * 0.85, p.getZ(k) * s); }
    g.computeVertexNormals();
    var m = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ color: '#c8c4b8', flatShading: true }));
    m.position.set(b[0], DESK_Y + 0.026, b[1]);
    room.add(m);
  });
  // The lamp, turned low.
  var lampMetal = new THREE.MeshLambertMaterial({ color: '#6a6458' });
  var lampBase = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.08, 0.025, 20), lampMetal);
  lampBase.position.set(-0.64, DESK_Y + 0.012, 0.22);
  var stem = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.3), lampMetal);
  stem.position.set(-0.64, DESK_Y + 0.17, 0.22);
  var shade = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.28, 18, 1, true, Math.PI / 2, Math.PI).rotateZ(Math.PI / 2),
    new THREE.MeshLambertMaterial({ color: '#3e5046', side: THREE.DoubleSide, emissive: '#1a2a20' }));
  shade.position.set(-0.64, DESK_Y + 0.33, 0.27);
  room.add(lampBase, stem, shade);
  var lampGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: warmTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0.55 }));
  lampGlow.position.set(-0.64, DESK_Y + 0.29, 0.28);
  lampGlow.scale.setScalar(0.28);
  room.add(lampGlow);
  var lamp = new THREE.PointLight('#e2c49a', 2.2, 6, 1.3);
  lamp.position.set(-0.64, DESK_Y + 0.26, 0.32);
  room.add(lamp);
  // The window seen lit from the garden.
  var amberTex = softSprite('rgba(255,186,104,1)', 'rgba(255,140,50,0)');
  var winGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: amberTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0 }));
  winGlow.position.set(0, 1.58, -0.34);
  winGlow.scale.set(2.6, 2.0, 1);
  room.add(winGlow);
  // Seen from the garden the room is lamplit amber, and the light spills out
  // over the sill onto the grass.
  var roomWarm = new THREE.PointLight('#ffab5c', 0, 5.5, 1.2);
  roomWarm.position.set(0, 1.6, 1.0);
  var spill = new THREE.PointLight('#ffb062', 0, 7, 1.3);
  spill.position.set(0, 1.5, -0.9);
  room.add(roomWarm, spill);
  // Life about the cottage: shutters, the eaves, a window box, smoke.
  var shutterMat = new THREE.MeshLambertMaterial({ color: '#44524c' });
  [-1, 1].forEach(function (side) {
    var sh = box(0.42, WIN.y1 - WIN.y0 + 0.1, 0.04, side * (WIN.x + 0.27), (WIN.y0 + WIN.y1) / 2, -WT - 0.03, shutterMat);
    for (var k = 0; k < 6; k++) box(0.38, 0.02, 0.02, 0, -0.55 + k * 0.22, -0.02, shutterMat, sh);
  });
  var fascia = new THREE.MeshLambertMaterial({ color: '#2e2a26' });
  box(ROOM.w * 2 + 1.0, 0.12, 0.05, 0, 2.86, -0.62, fascia);
  box(WIN.x * 2 + 0.1, 0.16, 0.2, 0, WIN.y0 - 0.12, -WT - 0.14, new THREE.MeshLambertMaterial({ color: '#4a3a2e' }));
  var potFlowers = [];
  for (i = 0; i < 18; i++) {
    var pf = new THREE.Mesh(new THREE.IcosahedronGeometry(0.035 + r() * 0.025, 0), new THREE.MeshLambertMaterial({ color: i % 3 ? '#7a5866' : '#a07a5a' }));
    pf.position.set(-WIN.x + 0.05 + r() * (WIN.x * 2 - 0.1), WIN.y0 + 0.0 + r() * 0.06, -WT - 0.14 + (r() - 0.5) * 0.12);
    room.add(pf);
    potFlowers.push(pf);
  }
  var smokeTex = softSprite('rgba(150,150,160,0.5)', 'rgba(150,150,160,0)');
  var smoke = [];
  for (i = 0; i < 5; i++) {
    var sm = new THREE.Sprite(new THREE.SpriteMaterial({ map: smokeTex, transparent: true, depthWrite: false, opacity: 0 }));
    room.add(sm);
    smoke.push(sm);
  }

  // The clock on the wall beside the window.
  var faceTex = canvasTexture(256, 256, function (x) {
    x.fillStyle = '#cfc9b8'; x.beginPath(); x.arc(128, 128, 126, 0, 6.283); x.fill();
    x.strokeStyle = '#3a3630'; x.fillStyle = '#3a3630';
    for (var k = 0; k < 60; k++) {
      var a = k / 60 * 6.283, l = k % 5 ? 8 : 18;
      x.lineWidth = k % 5 ? 2 : 5;
      x.beginPath(); x.moveTo(128 + Math.sin(a) * 112, 128 - Math.cos(a) * 112); x.lineTo(128 + Math.sin(a) * (112 - l), 128 - Math.cos(a) * (112 - l)); x.stroke();
    }
    x.font = '26px Georgia, serif'; x.textAlign = 'center'; x.textBaseline = 'middle';
    ['XII', 'I', 'II', 'III', 'IIII', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI'].forEach(function (n, k) {
      var a = k / 12 * 6.283; x.fillText(n, 128 + Math.sin(a) * 78, 128 - Math.cos(a) * 78);
    });
  });
  var clockG = new THREE.Group();
  clockG.position.set(0.37, DESK_Y, 0.15);
  clockG.rotation.y = -0.22;
  room.add(clockG);
  var caseWood = new THREE.MeshLambertMaterial({ color: '#4e3e32' }), brass = new THREE.MeshLambertMaterial({ color: '#8a7a52' });
  box(0.2, 0.17, 0.08, 0, 0.095, 0, caseWood, clockG);
  var arch = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.08, 28).rotateX(Math.PI / 2), caseWood);
  arch.position.y = 0.18;
  clockG.add(arch);
  box(0.23, 0.012, 0.1, 0, 0.006, 0, caseWood, clockG);
  var bezel = new THREE.Mesh(new THREE.TorusGeometry(0.078, 0.007, 8, 32), brass);
  bezel.position.set(0, 0.165, 0.041);
  var face = new THREE.Mesh(new THREE.CircleGeometry(0.075, 32), new THREE.MeshLambertMaterial({ map: faceTex }));
  face.position.set(0, 0.165, 0.041);
  clockG.add(bezel, face);
  var handMat = new THREE.MeshLambertMaterial({ color: '#24221e' });
  function hand(len, wid) {
    var h = new THREE.Mesh(new THREE.BoxGeometry(wid, len, 0.002).translate(0, len / 2 - 0.008, 0), handMat);
    h.position.set(0, 0.165, 0.044);
    clockG.add(h);
    return h;
  }
  var hourH = hand(0.045, 0.005), minH = hand(0.064, 0.0035), secH = hand(0.068, 0.0015);
  secH.material = new THREE.MeshLambertMaterial({ color: '#6a3a32' });

  // ── The garden at dusk ─────────────────────────────────────────────────
  var gc = new THREE.Color(), lawn = new THREE.Color('#3e463c'), soil = new THREE.Color('#3a3630');
  world.add(land(150, 74, small ? 90 : 150, small ? 50 : 80, 0, WALL_Z - 0.2 + 37, function (x, z) {
    var n = 0.5 + 0.5 * Math.sin(x * 0.7 + Math.sin(z * 0.5) * 2) * Math.cos(z * 0.6);
    return gc.copy(lawn).lerp(soil, n * 0.5).multiplyScalar(0.85 + 0.3 * n);
  }));
  var pathCurve = new THREE.CatmullRomCurve3([new THREE.Vector3(-1.75, 0, -0.6), new THREE.Vector3(-1.5, 0, -4), new THREE.Vector3(-0.2, 0, -8.5),
    new THREE.Vector3(0.1, 0, -13), new THREE.Vector3(0, 0, -17.9)]);
  var pathPts = pathCurve.getPoints(60);
  world.add(new THREE.Mesh(ribbon(pathPts, 0, 0.95, ground, 0.02), new THREE.MeshLambertMaterial({ color: '#6c665c' })));
  function nearPath(x, z, d) {
    for (var k = 0; k < pathPts.length; k++) if (Math.abs(pathPts[k].x - x) < d && Math.abs(pathPts[k].z - z) < d) return true;
    return false;
  }
  var gardenGrass = new THREE.InstancedMesh(tuftGeometry(r, 8, 0.3), swayMaterial(clock, sway), small ? 2200 : 6000);
  scatter(gardenGrass, 60000, function (n, p, q, sc, c) {
    var x = (r() - 0.5) * 18, z = 6 - r() * 23.6, dt = Math.hypot(x - TOYS.x, z - TOYS.z);
    if (z > -1.5 && Math.abs(x) < 3.3) return false;        // the cottage
    if (nearPath(x, z, 0.6) || dt < 1.25) return false;     // the path, and the toys
    p.set(x, ground(x, z), z);
    q.setFromAxisAngle(UP, r() * 6.28);
    var tall = z < WALL_Z + 1.2 ? 1.6 : dt < 2.2 ? 1.3 : 1;
    sc.set(1, (0.7 + r() * 0.7) * tall, 1);
    c.setHSL(0.2 + r() * 0.07, 0.12 + r() * 0.08, 0.3 + r() * 0.14);
  });
  // Short, flattened grass where the toys lie.
  var lowGrass = new THREE.InstancedMesh(tuftGeometry(r, 7, 0.12), swayMaterial(clock, sway), small ? 400 : 900);
  scatter(lowGrass, 4000, function (n, p, q, sc, c) {
    var a = r() * 6.28, d = Math.sqrt(r()) * 1.35, x = TOYS.x + Math.cos(a) * d, z = TOYS.z + Math.sin(a) * d * 0.9;
    p.set(x, ground(x, z), z);
    q.setFromAxisAngle(UP, r() * 6.28);
    sc.set(1, 0.5 + r() * 0.8, 1);
    c.setHSL(0.2 + r() * 0.07, 0.1, 0.2 + r() * 0.08);
  });
  world.add(lowGrass);
  world.add(gardenGrass);
  // Hedges down the sides, trees beyond the cottage.
  var shrubGeo = new THREE.IcosahedronGeometry(1, 2), sp = shrubGeo.attributes.position;
  for (i = 0; i < sp.count; i++) {
    v.fromBufferAttribute(sp, i);
    var lump = 1 + 0.16 * Math.sin(v.x * 5.1 + v.y * 3.3) * Math.cos(v.z * 4.7 - v.y * 2.2) + 0.07 * Math.sin(v.x * 13 + v.z * 11);
    sp.setXYZ(i, v.x * lump, v.y * lump, v.z * lump);
  }
  shrubGeo.computeVertexNormals();
  var shrubs = new THREE.InstancedMesh(shrubGeo, new THREE.MeshLambertMaterial({ color: '#ffffff' }), small ? 70 : 120);
  scatter(shrubs, 3000, function (n, p, q, sc, c) {
    var side = r() < 0.5 ? -1 : 1, x = side * (8.5 + r() * 3), z = 14 - r() * 31;
    if (n > (small ? 56 : 96)) { x = (r() - 0.5) * 30; z = WALL_Z + 0.9; if (Math.abs(x) < 7) return false; }
    var s = 0.9 + r() * 1.1;
    p.set(x, s * 0.45, z);
    q.setFromEuler(eul.set(r(), r() * 6, r()));
    sc.set(s * 1.1, s * (0.8 + r() * 0.4), s);
    c.setHSL(0.28 + r() * 0.04, 0.16, 0.08 + r() * 0.05);
  });
  world.add(shrubs);
  var trees = new THREE.InstancedMesh(broadleafGeometry(r, '#3a342e'), new THREE.MeshLambertMaterial({ vertexColors: true }), 14);
  scatter(trees, 200, function (n, p, q, sc, c) {
    var x = (r() - 0.5) * 40, z = 7 + r() * 22;
    if (n < 4) { x = (n % 2 ? 1 : -1) * (10.5 + r() * 2); z = -4 - n * 3.5; }
    if (Math.abs(x) < 4 && z < 10) return false;
    p.set(x, 0, z);
    q.setFromAxisAngle(UP, r() * 6.28);
    sc.setScalar(1 + r() * 0.5);
    c.setHSL(0.28, 0.12, 0.07 + r() * 0.04);
  });
  world.add(trees);

  // ── The broken toys ─────────────────────────────────────────────────────
  var toys = new THREE.Group();
  toys.position.copy(TOYS);
  world.add(toys);
  // The last of the evening light, on them.
  var toyLight = new THREE.PointLight('#b4bcd8', 0, 5, 1.2);
  toyLight.position.set(TOYS.x - 0.6, 1.9, TOYS.z + 1.2);
  world.add(toyLight);
  var faded = { red: '#8a4c48', cream: '#b8ae98', blue: '#4c5a76', gold: '#9a8a5c', wood: '#7a6650', green: '#4e6450' };
  function mat(c, o) { return new THREE.MeshLambertMaterial(Object.assign({ color: c }, o || {})); }
  // The tin drum, its head cracked.
  var drumTex = canvasTexture(256, 64, function (x, w, h) {
    x.fillStyle = '#8a4c48'; x.fillRect(0, 0, w, h);
    x.fillStyle = '#b8ae98'; x.fillRect(0, 0, w, 7); x.fillRect(0, h - 7, w, 7);
    x.strokeStyle = '#c8bca0'; x.lineWidth = 2.5; x.beginPath();
    for (var k = 0; k <= 16; k++) x.lineTo(k * 16, k % 2 ? h - 9 : 9);
    x.stroke();
    x.fillStyle = 'rgba(40,30,25,0.25)';
    for (k = 0; k < 30; k++) x.fillRect(r() * w, r() * h, 2 + r() * 8, 1 + r() * 3);
  });
  var headTex = canvasTexture(128, 128, function (x) {
    x.fillStyle = '#c4baa4'; x.beginPath(); x.arc(64, 64, 64, 0, 6.283); x.fill();
    x.strokeStyle = 'rgba(40,30,24,0.85)'; x.lineWidth = 2;
    x.beginPath(); x.moveTo(30, 34); x.lineTo(48, 52); x.lineTo(56, 50); x.lineTo(70, 70); x.lineTo(92, 78); x.lineTo(108, 100); x.stroke();
    x.lineWidth = 1.2;
    x.beginPath(); x.moveTo(70, 70); x.lineTo(64, 92); x.lineTo(70, 110); x.stroke();
    x.beginPath(); x.moveTo(56, 50); x.lineTo(64, 28); x.stroke();
    x.fillStyle = 'rgba(50,40,30,0.35)'; x.beginPath(); x.ellipse(68, 72, 9, 6, 0.6, 0, 6.283); x.fill();
  });
  var drum = new THREE.Group();
  drum.add(new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.15, 28, 1, true), mat('#ffffff', { map: drumTex, side: THREE.DoubleSide })));
  var head = new THREE.Mesh(new THREE.CircleGeometry(0.158, 28).rotateX(-Math.PI / 2), mat('#ffffff', { map: headTex }));
  head.position.y = 0.075;
  drum.add(head);
  var tin = mat('#8a8064');
  [-0.073, 0.073].forEach(function (yy) { var t = new THREE.Mesh(new THREE.TorusGeometry(0.162, 0.009, 6, 32).rotateX(Math.PI / 2), tin); t.position.y = yy; drum.add(t); });
  drum.position.set(-0.15, 0.1, 0.12);
  drum.rotation.set(0.22, 0.4, -0.18);
  toys.add(drum);
  var stickMat = mat('#9a8460');
  var stick = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.26, 6).rotateZ(Math.PI / 2), stickMat);
  stick.position.set(0.12, 0.012, 0.36); stick.rotation.y = 0.7;
  var stickA = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.005, 0.14, 6).rotateZ(Math.PI / 2), stickMat);
  stickA.position.set(-0.42, 0.012, 0.3); stickA.rotation.y = -0.3;
  var stickB = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.006, 0.12, 6).rotateZ(Math.PI / 2), stickMat);
  stickB.position.set(-0.5, 0.012, 0.44); stickB.rotation.y = 1.1;
  toys.add(stick, stickA, stickB);
  // The wooden horse on wheels, tipped over, a wheel off, its string snapped.
  var horse = new THREE.Group(), cream = mat(faded.cream), red = mat(faded.red), dark = mat('#3a3430');
  box(0.32, 0.02, 0.12, 0, 0.045, 0, mat(faded.wood), horse);
  [[-0.11, 0.06], [0.11, 0.06], [-0.11, -0.06], [0.11, -0.06]].forEach(function (w, k) {
    if (k === 3) return;
    var wh = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.018, 14).rotateX(Math.PI / 2), red);
    wh.position.set(w[0], 0.035, w[1] * 1.15);
    horse.add(wh);
  });
  var body = new THREE.Mesh(new THREE.CapsuleGeometry(0.05, 0.13, 4, 10).rotateZ(Math.PI / 2), cream);
  body.position.set(0, 0.2, 0);
  horse.add(body);
  [[-0.07, 0.03], [0.07, 0.03], [-0.07, -0.03], [0.07, -0.03]].forEach(function (l) { box(0.022, 0.11, 0.022, l[0], 0.11, l[1], cream, horse); });
  var neck = box(0.05, 0.13, 0.05, 0.11, 0.27, 0, cream, horse);
  neck.rotation.z = -0.5;
  var hd = box(0.11, 0.05, 0.05, 0.17, 0.32, 0, cream, horse);
  hd.rotation.z = -0.25;
  box(0.012, 0.13, 0.054, 0.085, 0.29, 0, dark, horse).rotation.z = -0.5;      // mane
  box(0.1, 0.012, 0.11, 0, 0.255, 0, red, horse);                               // saddle cloth
  horse.position.set(0.5, 0.06, -0.08);
  horse.rotation.set(1.38, -0.5, 0.1);
  toys.add(horse);
  var lostWheel = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.018, 14), red);
  lostWheel.position.set(0.72, 0.01, 0.32);
  toys.add(lostWheel);
  var stringMat = mat('#c8bca0');
  toys.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3([new THREE.Vector3(0.62, 0.03, -0.18), new THREE.Vector3(0.8, 0.01, -0.3),
    new THREE.Vector3(1.0, 0.012, -0.22), new THREE.Vector3(1.12, 0.01, -0.35)]), 20, 0.003, 4), stringMat));
  toys.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3([new THREE.Vector3(1.32, 0.01, -0.5), new THREE.Vector3(1.5, 0.012, -0.42),
    new THREE.Vector3(1.75, 0.01, -0.62)]), 16, 0.003, 4), stringMat));
  // The kite with a broken spar, its tail in the grass.
  var kite = new THREE.Group();
  var K = { top: [0, 0.44], left: [-0.28, 0.13], right: [0.28, 0.13], bottom: [0, -0.34], c: [0, 0.13] };
  function tri(a, b, c, col) {
    var g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute([a[0], a[1], 0, b[0], b[1], 0, c[0], c[1], 0], 3));
    g.computeVertexNormals();
    return new THREE.Mesh(g, mat(col, { side: THREE.DoubleSide }));
  }
  kite.add(tri(K.top, K.left, K.c, faded.red), tri(K.left, K.bottom, K.c, faded.cream));
  var wing = new THREE.Group();                                  // the right wing, folded back where the spar broke
  wing.add(tri(K.top, K.c, K.right, faded.cream), tri(K.c, K.bottom, K.right, faded.red));
  wing.rotation.y = -0.85;
  kite.add(wing);
  var spar = mat('#8a7a5a');
  var spine = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.005, 0.78), spar);
  spine.position.set(0, 0.05, 0.004);
  kite.add(spine);
  var crossL = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.005, 0.28).rotateZ(Math.PI / 2), spar);
  crossL.position.set(-0.14, 0.13, 0.004);
  kite.add(crossL);
  var crossR = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.004, 0.27).rotateZ(Math.PI / 2).translate(0.135, 0, 0), spar);
  crossR.position.set(0, 0.13, 0.004);
  wing.add(crossR);
  kite.position.set(-0.4, 0.16, -0.42);
  kite.rotation.set(-1.15, 0.25, 0.35);
  toys.add(kite);
  // Its tail: a string of bows that stirs with the wind, and stops.
  var TAILN = 22, tailBase = [], tailPos = new Float32Array(TAILN * 3);
  var tailCurve = new THREE.CatmullRomCurve3([new THREE.Vector3(-0.66, 0.05, 0.05), new THREE.Vector3(-0.9, 0.02, 0.3), new THREE.Vector3(-0.7, 0.03, 0.62),
    new THREE.Vector3(-1.0, 0.02, 0.9), new THREE.Vector3(-1.35, 0.04, 0.85)]);
  tailCurve.getPoints(TAILN - 1).forEach(function (p) { tailBase.push(p); });
  var tailGeo = new THREE.BufferGeometry();
  tailGeo.setAttribute('position', new THREE.BufferAttribute(tailPos, 3));
  var tail = new THREE.Line(tailGeo, new THREE.LineBasicMaterial({ color: '#4c4840' }));
  tail.frustumCulled = false;
  toys.add(tail);
  var bowGeo = new THREE.BufferGeometry();
  bowGeo.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, -0.05, 0.022, 0, -0.05, -0.022, 0, 0, 0, 0, 0.05, -0.022, 0, 0.05, 0.022, 0], 3));
  bowGeo.computeVertexNormals();
  var bows = new THREE.InstancedMesh(bowGeo, mat('#ffffff', { side: THREE.DoubleSide }), 5);
  for (i = 0; i < 5; i++) bows.setColorAt(i, tmpC.set(i % 2 ? faded.blue : faded.red));
  toys.add(bows);
  // A spinning top on its side, and three building blocks.
  var topCols = [];
  var topGeo = new THREE.LatheGeometry([[0, 0], [0.02, 0.01], [0.06, 0.05], [0.065, 0.065], [0.04, 0.085], [0.01, 0.09], [0.008, 0.13], [0, 0.13]].map(function (p) {
    return new THREE.Vector2(p[0], p[1]);
  }), 20).toNonIndexed();
  for (i = 0; i < topGeo.attributes.position.count; i++) {
    var band = Math.floor(topGeo.attributes.position.getY(i) * 60) % 2;
    tmpC.set(band ? faded.blue : faded.gold);
    topCols.push(tmpC.r, tmpC.g, tmpC.b);
  }
  topGeo.setAttribute('color', new THREE.Float32BufferAttribute(topCols, 3));
  topGeo.computeVertexNormals();
  var top = new THREE.Mesh(topGeo, new THREE.MeshLambertMaterial({ vertexColors: true }));
  top.position.set(0.2, 0.06, 0.45);
  top.rotation.set(0.2, 0.4, 1.45);
  toys.add(top);
  [[0.05, 0.03, -0.32, faded.blue, 0.3], [0.13, 0.03, -0.38, faded.gold, 1.0], [0.08, 0.09, -0.35, faded.green, 0.6]].forEach(function (b) {
    box(0.06, 0.06, 0.06, b[0], b[1], b[2], mat(b[3]), toys).rotation.set(b[2] * 0.2, b[4], 0.05);
  });

  // ── The wall and the gate ───────────────────────────────────────────────
  var stoneTex = canvasTexture(512, 256, function (x, w, h) {
    x.fillStyle = '#3e3c38'; x.fillRect(0, 0, w, h);
    var yy = 0;
    while (yy < h) {
      var rh = 20 + r() * 16, xx = -r() * 40;
      while (xx < w) {
        var sw = 28 + r() * 46;
        x.fillStyle = 'hsl(' + (28 + r() * 20) + ',' + (5 + r() * 9) + '%,' + (32 + r() * 14) + '%)';
        x.beginPath(); x.ellipse(xx + sw / 2, yy + rh / 2, sw / 2 - 2, rh / 2 - 2, 0, 0, 6.283); x.fill();
        x.fillRect(xx + 6, yy + 3, sw - 12, rh - 6);
        xx += sw;
      }
      yy += rh;
    }
    for (var k = 0; k < 60; k++) { x.fillStyle = 'rgba(70,90,50,' + (0.08 + r() * 0.12) + ')'; x.beginPath(); x.arc(r() * w, r() * h, 6 + r() * 20, 0, 6.283); x.fill(); }
    var g = x.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, 'rgba(20,24,16,0.4)');
    x.fillStyle = g; x.fillRect(0, 0, w, h);
  });
  stoneTex.wrapS = stoneTex.wrapT = THREE.RepeatWrapping;
  var WALL_H = 2.3, WALL_T = 0.45;
  var wallMat = new THREE.MeshLambertMaterial({ map: stoneTex });
  var wallTexL = stoneTex.clone(); wallTexL.repeat.set(22, 1.15); wallTexL.needsUpdate = true;
  var longWall = new THREE.MeshLambertMaterial({ map: wallTexL });
  var coping = new THREE.MeshLambertMaterial({ color: '#4e4c48' });
  [-1, 1].forEach(function (side) {
    var len = 60, cx = side * (POST_X + 0.27 + len / 2);
    box(len, WALL_H, WALL_T, cx, WALL_H / 2, WALL_Z, longWall, world);
    box(len, 0.1, WALL_T + 0.1, cx, WALL_H + 0.05, WALL_Z, coping, world);
    var postTex = stoneTex.clone(); postTex.repeat.set(1.1, 2.6); postTex.offset.set(side * 0.3, 0); postTex.needsUpdate = true;
    box(0.56, 2.95, 0.6, side * POST_X, 1.475, WALL_Z, new THREE.MeshLambertMaterial({ map: postTex }), world);
    box(0.7, 0.12, 0.74, side * POST_X, 3.01, WALL_Z, coping, world);
    var ball = new THREE.Mesh(new THREE.SphereGeometry(0.17, 16, 12), coping);
    ball.position.set(side * POST_X, 3.24, WALL_Z);
    world.add(ball);
  });
  // The gate: two leaves of bars under a rising arch, rings along the foot,
  // a lock on the right leaf.
  var iron = new THREE.MeshLambertMaterial({ vertexColors: true });
  function archY(x) { return 2.0 + 0.42 * Math.sin(x / LEAF_W * Math.PI / 2); }
  function leafGeometry(withLock) {
    var parts = [], rust = ['#36302c', '#3e3530', '#2e2a28'];
    function bar(x0, y0, x1, y1, rad) {
      var len = Math.hypot(x1 - x0, y1 - y0), g = new THREE.CylinderGeometry(rad, rad, len, 6);
      g.rotateZ(-Math.atan2(x1 - x0, y1 - y0)).translate((x0 + x1) / 2, (y0 + y1) / 2, 0);
      parts.push(tinted(g, rust[Math.floor(r() * 3)]));
    }
    bar(0.03, 0.06, 0.03, archY(0.03) + 0.02, 0.022);
    bar(LEAF_W - 0.025, 0.06, LEAF_W - 0.025, archY(LEAF_W) + 0.02, 0.018);
    [0.1, 0.34, 1.0, 1.82].forEach(function (yy) { bar(0.03, yy, LEAF_W - 0.025, yy, 0.016); });
    var arch = [];
    for (var k = 0; k <= 16; k++) { var ax = 0.03 + (LEAF_W - 0.055) * k / 16; arch.push(new THREE.Vector3(ax, archY(ax), 0)); }
    parts.push(tinted(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(arch), 24, 0.018, 6), rust[0]));
    for (var b = 1; b <= 9; b++) {
      var bx = 0.03 + b * (LEAF_W - 0.055) / 10;
      bar(bx, 0.1, bx, archY(bx) + 0.05, 0.0105);
      parts.push(tinted(new THREE.ConeGeometry(0.022, 0.09, 6).translate(bx, archY(bx) + 0.1, 0), rust[1]));
      if (b < 9) {
        var mx = bx + (LEAF_W - 0.055) / 20;
        parts.push(tinted(new THREE.TorusGeometry(0.045, 0.006, 5, 14).translate(mx, 0.22, 0), rust[2]));
        // A C-scroll in each bay under the arch.
        var sc = [];
        for (var t = 0; t <= 14; t++) { var a = t / 14 * Math.PI * 1.6; sc.push(new THREE.Vector3(mx + Math.cos(a) * 0.035 * (1 - t / 30), 1.95 + Math.sin(a) * 0.05, 0)); }
        parts.push(tinted(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(sc), 14, 0.005, 4), rust[0]));
      }
    }
    if (withLock) {
      parts.push(tinted(new THREE.BoxGeometry(0.13, 0.17, 0.06).translate(LEAF_W - 0.08, 1.12, 0.01), '#3a322c'));
      parts.push(tinted(new THREE.TorusGeometry(0.04, 0.007, 6, 16).translate(LEAF_W - 0.17, 1.12, 0.035), '#4a3e32'));
    }
    return merge(parts);
  }
  // Ivy: one leaf shape, instanced over the wall, the posts and the gate.
  var leafShape = new THREE.Shape();
  leafShape.moveTo(0, 0); leafShape.bezierCurveTo(0.55, 0.15, 0.6, 0.75, 0, 1); leafShape.bezierCurveTo(-0.6, 0.75, -0.55, 0.15, 0, 0);
  var leafGeo = new THREE.ShapeGeometry(leafShape, 3);
  var ivyMat = new THREE.MeshLambertMaterial({ side: THREE.DoubleSide });
  function ivyOn(parent, count, place) {
    var ivy = new THREE.InstancedMesh(leafGeo, ivyMat, count);
    scatter(ivy, count * 4, function (n, p, q, sc, c) {
      if (place(p) === false) return false;
      q.setFromEuler(eul.set((r() - 0.5) * 0.9, (r() - 0.5) * 0.9, r() * 6.28));
      sc.setScalar(0.05 + r() * 0.05);
      c.setHSL(0.27 + r() * 0.05, 0.25 + r() * 0.15, 0.13 + r() * 0.09);
    });
    parent.add(ivy);
    return ivy;
  }
  ivyOn(world, small ? 1200 : 2600, function (p) {
    var side = r() < 0.5 ? -1 : 1, x = side * (POST_X - 0.28 + Math.pow(r(), 1.4) * 9), yy = Math.pow(r(), 0.6) * (WALL_H + 0.1);
    if (Math.abs(x) < POST_X + 0.3) yy = Math.pow(r(), 0.5) * 3.1;
    if (Math.abs(x) > 3 && r() > 0.3 + 0.7 * yy / WALL_H) return false;
    var face = Math.abs(x) < POST_X + 0.3 ? WALL_Z + 0.31 : WALL_Z + WALL_T / 2 + 0.01;
    p.set(x, yy, face + r() * 0.03);
  });
  var leaves = [-1, 1].map(function (side) {
    var hinge = new THREE.Group(), leaf = new THREE.Group();
    leaf.add(new THREE.Mesh(leafGeometry(side > 0), iron));
    ivyOn(leaf, small ? 70 : 140, function (p) {
      var x = Math.pow(r(), 1.8) * LEAF_W * 0.9, yy = Math.pow(r(), 0.9) * 2.2;
      if (x > 0.5 && yy > 0.9 && r() < 0.7) return false;
      p.set(x, yy, 0.03 + r() * 0.03);
    });
    leaf.scale.x = -side;
    hinge.add(leaf);
    hinge.position.set(side * LEAF_W, 0, WALL_Z);
    world.add(hinge);
    return { hinge: hinge, side: side, leaf: leaf };
  });
  // Ivy climbing the cottage front, by the door and up past the window.
  ivyOn(room, small ? 260 : 560, function (p) {
    var right = r() < 0.6, x = right ? WIN.x + 0.55 + Math.pow(r(), 1.5) * 1.6 : -ROOM.w - 0.2 + r() * 0.5;
    var yy = Math.pow(r(), right ? 0.8 : 0.6) * (right ? 2.8 : 2.4);
    if (right && x > 2.0 && yy > 1.6 + r()) return false;
    p.set(x, yy, -WT - 0.02 - r() * 0.03);
  });

  // The rose, tucked through the bars of the right leaf.
  var rose = new THREE.Group();
  var roseRed = new THREE.MeshLambertMaterial({ color: '#9a1426', side: THREE.DoubleSide, emissive: '#2a0208' });
  var roseDeep = new THREE.MeshLambertMaterial({ color: '#6a0a18', side: THREE.DoubleSide, emissive: '#1a0105' });
  [[0.012, 0.03, 3, -0.4, roseDeep], [0.021, 0.034, 4, -0.15, roseRed], [0.03, 0.034, 5, 0.1, roseRed], [0.04, 0.03, 5, 0.35, roseRed]].forEach(function (c, k) {
    var cupM = new THREE.Mesh(petalCup(c[0], c[1], c[2], c[3], k * 0.9, 20), c[4]);
    cupM.rotation.y = k * 1.1;
    cupM.position.y = -k * 0.004;
    rose.add(cupM);
  });
  var leafy = new THREE.MeshLambertMaterial({ color: '#2e4a2a', side: THREE.DoubleSide });
  var sepals = new THREE.Mesh(petalCup(0.022, 0.016, 5, 0.5, 0, 15), leafy);
  sepals.position.y = -0.016;
  rose.add(sepals);
  rose.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3([new THREE.Vector3(0, -0.01, 0), new THREE.Vector3(0.02, -0.12, -0.02),
    new THREE.Vector3(0.05, -0.28, -0.05), new THREE.Vector3(0.09, -0.4, -0.06)]), 16, 0.0045, 5), leafy));
  [[0.02, -0.15, 0.6], [0.06, -0.3, -0.7]].forEach(function (l) {
    var lf = new THREE.Mesh(leafGeo, leafy);
    lf.scale.set(0.035, 0.06, 0.035);
    lf.position.set(l[0], l[1], -0.02);
    lf.rotation.set(0.6, 0, l[2]);
    rose.add(lf);
  });
  rose.position.set(LEAF_W - 0.14, 1.36, 0.06);
  rose.rotation.set(1.0, 0, -0.5);
  rose.scale.setScalar(1.8);
  leaves[0].leaf.add(rose);
  var roseGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,120,140,1)', 'rgba(255,80,110,0)'), blending: THREE.AdditiveBlending,
    depthWrite: false, transparent: true, opacity: 0 }));
  roseGlow.scale.setScalar(0.5);
  world.add(roseGlow);
  // The light of the places beyond, spilling through the open gate.
  var beyondGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(214,196,255,1)', 'rgba(170,150,255,0)'), blending: THREE.AdditiveBlending,
    depthWrite: false, transparent: true, opacity: 0 }));
  beyondGlow.position.set(0, 1.3, WALL_Z - 1.6);
  beyondGlow.scale.set(6, 4.5, 1);
  world.add(beyondGlow);
  var gateLight = new THREE.PointLight('#d4c8ff', 0, 12, 1.4);
  gateLight.position.set(0, 1.6, WALL_Z - 1.2);
  world.add(gateLight);

  // ── The steppes of dream ───────────────────────────────────────────────
  var grassA = new THREE.Color('#2a4848'), grassB = new THREE.Color('#3e5a52'), grassC = new THREE.Color('#4a4a6a');
  var steppe = land(720, 300, small ? 120 : 200, small ? 70 : 110, 0, WALL_Z - 150 + 0.2, function (x, z, y) {
    var n = 0.5 + 0.5 * Math.sin(x * 0.13 + Math.sin(z * 0.09) * 2.5) * Math.cos(z * 0.11 - x * 0.03);
    gc.copy(grassA).lerp(grassB, n).lerp(grassC, 0.5 + 0.5 * Math.sin(x * 0.02 + z * 0.015));
    if (y < SEA_Y + 0.6) gc.lerp(tmpD.set('#5a5468'), smooth(SEA_Y + 0.6, SEA_Y - 0.2, y));
    return gc;
  });
  world.add(steppe);
  var dreamGrassMat = swayMaterial(clock, sway);
  var dreamGrass = new THREE.InstancedMesh(tuftGeometry(r, 8, 0.38), dreamGrassMat, small ? 4000 : 11000);
  scatter(dreamGrass, 60000, function (n, p, q, sc, c) {
    var x = (r() - 0.5) * 34, z = WALL_Z - 0.3 - Math.pow(r(), 1.25) * 62;
    var dx = x - FLOWER.x, dz = z - FLOWER.z;
    if (dx * dx + dz * dz < 0.5) return false;
    p.set(x, ground(x, z) - 0.02, z);
    q.setFromAxisAngle(UP, r() * 6.28);
    sc.set(1, 0.7 + r() * 0.8, 1);
    c.setHSL(0.46 + r() * 0.14, 0.22 + r() * 0.15, 0.4 + r() * 0.18);
  });
  world.add(dreamGrass);
  // Dream flowers glimmering in the grass: the places Nobody knows.
  var DF = small ? 1500 : 3600, dfPos = [], dfAttr = [], dfTint = [];
  var DF_TINTS = [new THREE.Color('#9ea2ff'), new THREE.Color('#ffd28e'), new THREE.Color('#ff9eb8'), new THREE.Color('#bff0ff')];
  for (i = 0; i < DF; i++) {
    var fx = (r() - 0.5) * (i < DF * 0.6 ? 40 : 160), fz = WALL_Z - 0.6 - Math.pow(r(), 0.8) * (i < DF * 0.6 ? 70 : 100);
    if (Math.hypot(fx - FLOWER.x, fz - FLOWER.z) < 1.2) continue;
    dfPos.push(fx, ground(fx, fz) + 0.15 + r() * 0.3, fz);
    dfAttr.push(0.05 + r() * 0.07, r(), r() * 0.7);
    var dt = DF_TINTS[Math.floor(r() * 4)];
    dfTint.push(dt.r, dt.g, dt.b);
  }
  var dfGeo = new THREE.BufferGeometry();
  dfGeo.setAttribute('position', new THREE.Float32BufferAttribute(dfPos, 3));
  dfGeo.setAttribute('aG', new THREE.Float32BufferAttribute(dfAttr, 3));
  dfGeo.setAttribute('tint', new THREE.Float32BufferAttribute(dfTint, 3));
  var dfMat = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uTime: clock, uShow: { value: 0 }, uPx: pxScale }, vertexShader: GLOW_VS, fragmentShader: GLOW_FS });
  var dreamFlowers = new THREE.Points(dfGeo, dfMat);
  dreamFlowers.frustumCulled = false;
  world.add(dreamFlowers);
  // Low mist over the far steppe, the perfect places of Sleep.
  var mistTex = softSprite('rgba(190,180,255,0.9)', 'rgba(160,150,230,0)');
  var mists = [];
  for (i = 0; i < 9; i++) {
    var ms = new THREE.Sprite(new THREE.SpriteMaterial({ map: mistTex, transparent: true, depthWrite: false, opacity: 0, fog: false }));
    var mx = (r() - 0.5) * 120, mz = WALL_Z - 30 - r() * 70;
    ms.position.set(mx, ground(mx, mz) + 1.5, mz);
    ms.scale.set(40 + r() * 30, 5 + r() * 3, 1);
    world.add(ms);
    mists.push(ms);
  }
  // Motes drifting up through the dream.
  var motes = particleField({ count: small ? 220 : 520, box: [26, 9, 34], fall: [-0.25, -0.08], size: 0.07, sway: 0.25,
    map: softSprite('rgba(255,240,220,1)', 'rgba(255,220,200,0)'), colors: ['#fff0c8', '#d8ccff', '#ffd0e0', '#c8f0ff'] });
  motes.points.material.blending = THREE.AdditiveBlending;
  world.add(motes.points);
  var moteF = { time: 0, dt: 0, wind: 0, snow: 0 };

  // The sea, far off where the steppes end.
  var seaMat = new THREE.ShaderMaterial({
    uniforms: { uTime: clock, uTop: { value: new THREE.Color() }, uHor: { value: new THREE.Color() }, uDeep: { value: new THREE.Color('#05061a') },
      uMoonC: { value: MOON_C }, uMoonDir: { value: moonDir }, uMoon: { value: 0 }, uFogColor: { value: new THREE.Color() }, uFogDensity: { value: 0.006 } },
    vertexShader: 'varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }',
    fragmentShader: SEA_FS
  });
  var sea = new THREE.Mesh(new THREE.PlaneGeometry(5000, 2600).rotateX(-Math.PI / 2), seaMat);
  sea.position.set(0, SEA_Y, -1300);
  world.add(sea);

  // ── The Only Flower ─────────────────────────────────────────────────────
  FLOWER.y = ground(FLOWER.x, FLOWER.z);
  var flower = new THREE.Group();
  flower.position.copy(FLOWER);
  flower.scale.setScalar(2);
  world.add(flower);
  var stemMat = new THREE.MeshLambertMaterial({ color: '#2e5a4a', emissive: '#0c2a22' });
  flower.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0.03, 0.18, 0.01),
    new THREE.Vector3(-0.02, 0.36, 0), new THREE.Vector3(0, 0.5, 0.02)]), 20, 0.008, 6), stemMat));
  [[0.6, 0.08], [-2.4, 0.16]].forEach(function (l) {
    var lf = new THREE.Mesh(leafGeo, stemMat);
    lf.scale.set(0.07, 0.17, 0.07);
    lf.position.set(0, l[1], 0);
    lf.rotation.set(0.9, l[0], 0);
    flower.add(lf);
  });
  var bloom = new THREE.Group();
  bloom.position.set(0, 0.5, 0.02);
  flower.add(bloom);
  // Petals: each a cupped, tapering blade with its own pivot so it can open,
  // pearl at the tips, rose towards the heart, lit from within.
  function petalGeo(len, wid) {
    var g = new THREE.PlaneGeometry(1, 1, 6, 10).translate(0, 0.5, 0), p = g.attributes.position, cols = new Float32Array(p.count * 3);
    var inner = new THREE.Color('#e8487a'), mid = new THREE.Color('#f4d2e4'), tip = new THREE.Color('#a898f0');
    for (var k = 0; k < p.count; k++) {
      var u = p.getX(k), t = p.getY(k), w = Math.pow(Math.sin(Math.PI * Math.min(t * 0.92 + 0.08, 1)), 0.7) * (1 - 0.25 * t);
      p.setXYZ(k, u * wid * w, t * len, -(u * u) * wid * 0.9 + Math.sin(t * Math.PI) * len * 0.12);
      gc.copy(inner).lerp(mid, smooth(0.05, 0.55, t)).lerp(tip, smooth(0.6, 1, t));
      cols[k * 3] = gc.r; cols[k * 3 + 1] = gc.g; cols[k * 3 + 2] = gc.b;
    }
    g.setAttribute('color', new THREE.BufferAttribute(cols, 3));
    g.computeVertexNormals();
    return g;
  }
  var petalMat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: 0.92, color: '#ffffff', toneMapped: false });
  var petals = [];
  [[7, 0.17, 0.075, 0], [6, 0.12, 0.06, 0.45]].forEach(function (ring, k) {
    var g = petalGeo(ring[1], ring[2]);
    for (var j = 0; j < ring[0]; j++) {
      var pivot = new THREE.Group(), pm = new THREE.Mesh(g, petalMat);
      pivot.rotation.y = j / ring[0] * Math.PI * 2 + ring[3];
      pivot.add(pm);
      bloom.add(pivot);
      petals.push({ mesh: pm, inner: k, phase: r() * 6.28 });
    }
  });
  var HEART_C = new THREE.Color('#ffd6a0'), heartMat = new THREE.MeshBasicMaterial({ color: HEART_C.clone() });
  var heart = new THREE.Mesh(new THREE.SphereGeometry(0.022, 14, 10), heartMat);
  heart.position.y = 0.012;
  bloom.add(heart);
  var stamens = [];
  for (i = 0; i < 9; i++) {
    var sa = i / 9 * 6.28, st = new THREE.Mesh(new THREE.SphereGeometry(0.006, 6, 4), heartMat);
    st.position.set(Math.cos(sa) * 0.03, 0.05 + r() * 0.015, Math.sin(sa) * 0.03);
    bloom.add(st);
    stamens.push(st);
  }
  var heartGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,140,170,1)', 'rgba(255,90,130,0)'), blending: THREE.AdditiveBlending,
    depthWrite: false, transparent: true, opacity: 0 }));
  heartGlow.position.y = 0.04;
  bloom.add(heartGlow);
  var flowerHalo = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(220,200,255,1)', 'rgba(180,160,255,0)'), blending: THREE.AdditiveBlending,
    depthWrite: false, transparent: true, opacity: 0 }));
  flowerHalo.position.y = 0.05;
  bloom.add(flowerHalo);
  var flowerLight = new THREE.PointLight('#ffb8d4', 0, 4, 1.3);
  flowerLight.position.y = 0.25;
  bloom.add(flowerLight);
  // Sparks rising from it once it is found.
  var FS = 60, fsPos = [], fsAttr = [], fsTint = [];
  for (i = 0; i < FS; i++) {
    fsPos.push(0, 0, 0);
    fsAttr.push(0.03 + r() * 0.03, r(), 0);
    tmpC.set(i % 3 ? '#ffd8e8' : '#d8d0ff');
    fsTint.push(tmpC.r, tmpC.g, tmpC.b);
  }
  var fsGeo = new THREE.BufferGeometry();
  fsGeo.setAttribute('position', new THREE.Float32BufferAttribute(fsPos, 3));
  fsGeo.setAttribute('aG', new THREE.Float32BufferAttribute(fsAttr, 3));
  fsGeo.setAttribute('tint', new THREE.Float32BufferAttribute(fsTint, 3));
  var fsMat = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uTime: clock, uShow: { value: 0 }, uPx: pxScale }, vertexShader: GLOW_VS, fragmentShader: GLOW_FS });
  var sparks = new THREE.Points(fsGeo, fsMat);
  sparks.frustumCulled = false;
  flower.add(sparks);
  var sparkSeed = [];
  for (i = 0; i < FS; i++) sparkSeed.push(r(), r() * 6.28, 0.15 + r() * 0.5);

  // ── The bubble ─────────────────────────────────────────────────────────
  var bubbleUniforms = {
    uTime: clock, uWob: { value: 1 }, uAlpha: { value: 0 }, uGlow: { value: 0 }, uBack: { value: 0.5 },
    uTop: { value: new THREE.Color() }, uHor: { value: new THREE.Color() }, uMoonC: { value: MOON_C }, uMoonDir: { value: moonDir }
  };
  var bubbleGeo = new THREE.SphereGeometry(1, 64, 40);
  var bubbleBack = new THREE.Mesh(bubbleGeo, new THREE.ShaderMaterial({ uniforms: Object.assign({}, bubbleUniforms, { uBack: { value: 0.55 } }),
    vertexShader: BUBBLE_VS, fragmentShader: BUBBLE_FS, side: THREE.BackSide, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  var bubbleFront = new THREE.Mesh(bubbleGeo, new THREE.ShaderMaterial({ uniforms: Object.assign({}, bubbleUniforms, { uBack: { value: 1.0 } }),
    vertexShader: BUBBLE_VS, fragmentShader: BUBBLE_FS, side: THREE.FrontSide, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  bubbleBack.renderOrder = 5;
  bubbleFront.renderOrder = 6;
  var bubble = new THREE.Group();
  bubble.add(bubbleBack, bubbleFront);
  bubble.visible = false;
  world.add(bubble);
  [bubbleBack, bubbleFront].forEach(function (b) { b.frustumCulled = false; });
  // Its halo is a ring, so the glow gathers at the rim and the bubble stays clear.
  var ringTex = canvasTexture(128, 128, function (x) {
    var g = x.createRadialGradient(64, 64, 0, 64, 64, 64);
    g.addColorStop(0, 'rgba(200,210,255,0)'); g.addColorStop(0.42, 'rgba(200,210,255,0.04)'); g.addColorStop(0.48, 'rgba(205,212,255,0.6)');
    g.addColorStop(0.6, 'rgba(190,200,255,0.22)'); g.addColorStop(1, 'rgba(180,190,255,0)');
    x.fillStyle = g; x.fillRect(0, 0, 128, 128);
  });
  var bubbleHalo = new THREE.Sprite(new THREE.SpriteMaterial({ map: ringTex, blending: THREE.AdditiveBlending,
    depthWrite: false, transparent: true, opacity: 0, fog: false }));
  bubbleHalo.renderOrder = 4;
  world.add(bubbleHalo);
  var BLOW_AT = new THREE.Vector3(-0.25, 1.22, WALL_Z - 3.2), BLOWN = new THREE.Vector3(-0.55, 1.4, WALL_Z - 4.6);
  var bubDir0 = new THREE.Vector3(), bubDir1 = new THREE.Vector3(), bubDir = new THREE.Vector3(), bubbleMoonDir = new THREE.Vector3();

  // Phones see a narrow slice of the sky: the sky's things gather closer in.
  var layout = { sx: 1, portrait: false };
  var knockT = -99, lastU = 0;
  var COL = { top: new THREE.Color(), mid: new THREE.Color(), hor: new THREE.Color() };

  function frame(f) {
    var row = f.row, time = f.time, u = f.u, dt = f.dt;
    var dark = row[1], eye = row[2], wind = row[3], open = row[7], still = row[8], gateOpen = row[9], dream = row[10], moonUp = row[11];
    clock.value = time;

    // ── Camera ──
    var cx = row[0], cz = row[4];
    camera.position.set(cx, ground(cx, cz) + eye + Math.sin(time * 0.6) * 0.01 * (1 - still), cz);
    camera.rotation.set(0, 0, 0);
    var pAz = layout.portrait ? row[12] * Math.PI / 180 : 0, pPitch = layout.portrait ? row[13] : 0;
    camera.rotateY(row[5] + pAz - f.mx * 0.1 * (1 - still));
    camera.rotateX(row[6] + pPitch - f.my * 0.05 * (1 - still));
    sky.position.copy(camera.position);
    var indoor = smooth(-0.4, 0.5, cz), outside = 1 - indoor;

    // ── Sky, fog and light ──
    var dk = clamp(dark, 0, 1);
    COL.top.copy(SKY.dusk[0]).lerp(SKY.dark[0], dk).lerp(SKY.dream[0], dream);
    COL.mid.copy(SKY.dusk[1]).lerp(SKY.dark[1], dk).lerp(SKY.dream[1], dream);
    COL.hor.copy(SKY.dusk[2]).lerp(SKY.dark[2], dk).lerp(SKY.dream[2], dream).lerp(MOONGLOW, moonUp * 0.45);
    dome.uniforms.top.value.copy(COL.top);
    dome.uniforms.mid.value.copy(COL.mid);
    dome.uniforms.horizon.value.copy(COL.hor);
    world.fog.color.copy(COL.mid).lerp(COL.top, 0.35).lerp(tmpC.copy(COL.hor).lerp(COL.mid, 0.4), dream);
    world.fog.density = lerp(lerp(0.03, 0.04, dk), 0.0055, dream);
    gl.setClearColor(world.fog.color);

    skyDir(-9 * layout.sx, lerp(-4.5, 5.5, moonUp), moonDir);
    dome.uniforms.sunDir.value.copy(moonDir);
    dome.uniforms.sunColor.value.copy(MOON_C).multiplyScalar(0.12 * moonUp);
    moon.position.copy(moonDir).multiplyScalar(1100);
    moon.scale.setScalar(96);
    moon.visible = moonUp > 0.001;
    moonHalo.position.copy(moonDir).multiplyScalar(1105);
    moonHalo.scale.setScalar(520);
    moonHalo.material.opacity = 0.35 * smooth(0, 0.6, moonUp);
    moonLight.position.copy(moonDir).multiplyScalar(50);
    moonLight.intensity = 0.9 * moonUp;
    starMat.uniforms.uShow.value = dream * 1.05 + dk * 0.08;

    var dimLight = 1 - 0.6 * dk;
    hemi.color.copy(HEMI_DUSK).lerp(HEMI_DREAM, dream);
    hemi.groundColor.copy(GND_DUSK).lerp(GND_DREAM, dream);
    hemi.intensity = lerp(1.2 * dimLight, 1.25, dream) * (1 - 0.3 * indoor);
    key.color.copy(KEY_DUSK).lerp(KEY_DREAM, dream);
    key.intensity = lerp(0.55 * dimLight, 1.4, dream) * (1 - 0.3 * indoor);
    lamp.intensity = 2.2 * (0.94 + 0.06 * Math.sin(time * 7.1) * Math.sin(time * 2.3));
    var fromGarden = smooth(-0.6, -3.5, cz) * (1 - dream);
    winGlow.material.opacity = 0.5 * fromGarden;
    roomWarm.intensity = 3.2 * fromGarden;
    spill.intensity = 2.2 * fromGarden;
    lamp.intensity *= 1 + fromGarden;
    smoke.forEach(function (sm, n) {
      var a = (time * 0.06 + n / smoke.length) % 1;
      sm.position.set(1.6 + Math.sin(a * 3 + n) * 0.25 + a * 0.6, 5.2 + a * 2.6, 3.0 - a * 0.4);
      sm.scale.setScalar(0.5 + a * 1.6);
      sm.material.opacity = 0.35 * Math.sin(a * Math.PI) * fromGarden;
    });

    // ── The room ──
    panes.forEach(function (p) { p.hinge.rotation.y = -p.side * open * 1.9; });
    breeze.value = open * (0.4 + wind);
    var secs = 6 * 3600 + 47 * 60 + 12 + Math.floor(time);
    secH.rotation.z = -(secs % 60) / 60 * Math.PI * 2;
    minH.rotation.z = -((secs / 60) % 60) / 60 * Math.PI * 2;
    hourH.rotation.z = -((secs / 3600) % 12) / 12 * Math.PI * 2;

    // ── The garden: the grass, the kite's tail, and "and—" ──
    sway.value = (0.25 + wind * 1.2) * (1 - still) * (slow ? 0.5 : 1);
    var flutter = wind * (1 - still);
    for (var k = 0; k < TAILN; k++) {
      var b = tailBase[k], w = k / (TAILN - 1);
      tailPos[k * 3] = b.x + Math.sin(time * 2.4 + k * 0.5) * 0.03 * w * flutter;
      tailPos[k * 3 + 1] = b.y + Math.max(0, Math.sin(time * 3.1 + k * 0.7)) * 0.05 * w * flutter;
      tailPos[k * 3 + 2] = b.z + Math.cos(time * 2.0 + k * 0.4) * 0.02 * w * flutter;
    }
    tailGeo.attributes.position.needsUpdate = true;
    for (k = 0; k < 5; k++) {
      var tk = 3 + k * 4, tp = tk * 3;
      p3.set(tailPos[tp], tailPos[tp + 1] + 0.005, tailPos[tp + 2]);
      q4.setFromEuler(eul.set(-Math.PI / 2 + Math.sin(time * 2.2 + k) * 0.3 * flutter, k * 0.9, 0));
      bows.setMatrixAt(k, m4.compose(p3, q4, s3.set(1, 1, 1)));
    }
    bows.instanceMatrix.needsUpdate = true;

    var inIII = TL ? smooth(TL.start(2), TL.start(2) + 0.4, u) * (1 - smooth(TL.start(4), TL.start(4) + 0.5, u)) : 0;
    toyLight.intensity = 2.6 * inIII * (1 - 0.55 * still);
    gardenMoon.intensity = 0.75 * outside * (1 - dream) * (1 - 0.5 * still) * (TL ? 1 - smooth(TL.start(4), TL.start(4) + 1, u) * 0.5 : 1);

    // ── The gate ──
    if (TL && lastU < EV.knock && u >= EV.knock && u - lastU < 0.5) knockT = time;
    var kt = time - knockT, shake = 0;
    KNOCKS.forEach(function (d) { var s = kt - d - 0.05; if (s > 0 && s < 0.6) shake += Math.exp(-s * 10) * Math.sin(s * 46); });
    var swing = smooth(0, 1, gateOpen);
    leaves.forEach(function (l) { l.hinge.rotation.y = l.side * (-swing * 1.55 - shake * 0.012 * (l.side > 0 ? 1 : 0.6)); });
    var inIV = TL ? smooth(TL.start(4) + 0.1, TL.start(4) + 0.6, u) * (1 - smooth(TL.start(6), TL.start(6) + 0.6, u)) : 0;
    eyeLight.position.set(camera.position.x * 0.5, 1.7, Math.max(camera.position.z - 2.4, WALL_Z + 0.9));
    eyeLight.intensity = 1.3 * inIV;
    leaves[0].leaf.localToWorld(roseGlow.position.copy(rose.position));
    roseGlow.material.opacity = 0.22 * inIV;
    var passed = smooth(WALL_Z + 1.4, WALL_Z - 1.4, cz);
    beyondGlow.material.opacity = 0.55 * swing * (1 - passed);
    gateLight.intensity = 3 * swing * (1 - passed * 0.7);

    // ── The dream ──
    var wake = lerp(0.3, 1, dream);
    steppe.material.color.setScalar(wake);
    dreamGrassMat.color.setScalar(wake);
    steppe.material.emissive.setRGB(0.012, 0.014, 0.04).multiplyScalar(dream);
    dfMat.uniforms.uShow.value = dream * 1.1 * (1 - 0.3 * moonUp);
    var mistA = 0.07 * dream;
    mists.forEach(function (m, n) { m.material.opacity = mistA * (0.7 + 0.3 * Math.sin(time * 0.2 + n)); m.position.x += Math.sin(time * 0.05 + n) * 0.01; });
    moteF.time = time; moteF.dt = dt; moteF.wind = 0.15; moteF.snow = dream * smooth(WALL_Z + 2, WALL_Z - 1, cz);
    motes.update(moteF, camera.position, slow);
    seaMat.uniforms.uTop.value.copy(COL.top);
    seaMat.uniforms.uHor.value.copy(COL.hor);
    seaMat.uniforms.uMoon.value = moonUp;
    seaMat.uniforms.uFogColor.value.copy(world.fog.color);
    seaMat.uniforms.uFogDensity.value = world.fog.density;

    // ── The bubble: blown, released, and the moon ──
    if (TL) {
      var inf = smooth(EV.blow0, EV.blow1, u), rise = smooth(EV.blow1, EV.rise1, u);
      bubble.visible = inf > 0.001;
      var rad;
      skyDir(36 * layout.sx + Math.sin(time * 0.07) * 0.6, (layout.portrait ? 42 : 34) + moonUp * 12 + Math.sin(time * 0.11) * 0.4, bubbleMoonDir);
      if (rise <= 0) {
        bubble.position.copy(BLOW_AT).lerp(BLOWN, inf);
        rad = lerp(0.02, 0.3, Math.pow(inf, 0.7));
      } else {
        // Out from where it was blown, along a widening arc, into the moon.
        bubDir0.copy(BLOWN).sub(camera.position);
        var d0 = bubDir0.length();
        bubDir0.multiplyScalar(1 / d0);
        var e = rise, lift = smooth(0, 0.7, e);
        bubDir1.copy(bubbleMoonDir);
        bubDir.copy(bubDir0).lerp(bubDir1, lift).normalize();
        var dist = Math.exp(lerp(Math.log(d0), Math.log(380), smooth(0.05, 1, e)));
        bubble.position.copy(camera.position).addScaledVector(bubDir, dist);
        var ang = lerp(0.3 / d0, Math.tan(7 * Math.PI / 180), e) - 0.06 * Math.sin(Math.PI * Math.min(e * 1.15, 1));
        rad = dist * Math.max(ang, 0.03);
      }
      bubble.scale.setScalar(rad);
      var a0 = smooth(0, 0.15, inf) * (1 - 0.3 * moonUp);
      bubbleBack.material.uniforms.uAlpha.value = bubbleFront.material.uniforms.uAlpha.value = a0;
      bubbleBack.material.uniforms.uWob.value = bubbleFront.material.uniforms.uWob.value = (slow ? 0.3 : 1) * (inf < 1 ? 1 : lerp(0.6, 0.12, rise));
      bubbleBack.material.uniforms.uGlow.value = bubbleFront.material.uniforms.uGlow.value = rise * 0.8;
      [bubbleBack, bubbleFront].forEach(function (b) { b.material.uniforms.uTop.value.copy(COL.top); b.material.uniforms.uHor.value.copy(COL.hor); });
      bubbleHalo.position.copy(bubble.position);
      bubbleHalo.scale.setScalar(rad * 4.2);
      bubbleHalo.material.opacity = 0.45 * smooth(0.4, 1, rise) * (1 - 0.3 * moonUp);
      key.position.copy(bubbleMoonDir).multiplyScalar(30).lerp(tmpA.set(-6, 10, -20), 1 - dream);

      // The singing stars, each born on its note.
      var sung = 0;
      singers.forEach(function (s, n) {
        var at = EV.stars[n], born = smooth(at - 0.012, at + 0.04, u);
        skyDir(s.az * layout.sx, layout.portrait ? 26 + (s.el - 7) * 0.5 : s.el, tmpA);
        s.sprite.position.copy(tmpA).multiplyScalar(1200);
        if (n > 0) { tmpA.multiplyScalar(1190).toArray(tunePos, n * 6 - 3); singers[n - 1].sprite.position.toArray(tunePos, n * 6 - 6); }
        if (born > 0.5 && n > 0) sung = n;
        s.sprite.visible = born > 0.001;
        if (!s.sprite.visible) return;
        var flare = Math.exp(-Math.pow((u - at - 0.03) / 0.05, 2)) * 1.6;
        var song = 0.85 + 0.15 * Math.sin(time * 1.4 - n * 0.8);
        s.sprite.scale.setScalar(1200 * 0.028 * (1 + flare) * song * (layout.portrait ? 1.2 : 1));
        s.sprite.material.opacity = born * (0.75 + 0.25 * song);
      });
      tuneGeo.attributes.position.needsUpdate = true;
      tuneGeo.setDrawRange(0, sung * 2);
      tune.material.opacity = sung ? 0.2 * (1 - moonUp) : 0;

      // The Only Flower: found, opening, glowing, its heart beating.
      var found = smooth(EV.flower0, EV.flower1, u), beat = smooth(EV.heart0, EV.heart1, u);
      var tb = time % 1.1, pulse = Math.exp(-Math.pow(tb / 0.07, 2)) + 0.6 * Math.exp(-Math.pow((tb - 0.25) / 0.07, 2));
      petals.forEach(function (p) {
        var openA = lerp(0.18, p.inner ? 0.75 : 1.2, found) + Math.sin(time * 0.8 + p.phase) * 0.03;
        p.mesh.rotation.x = openA;
      });
      petalMat.opacity = 0.55 + 0.4 * found;
      petalMat.color.setScalar(0.4 + 0.42 * found + 0.1 * beat * pulse);
      heartMat.color.copy(HEART_C).multiplyScalar(0.6 + 0.6 * found + 0.8 * beat * pulse);
      heartGlow.material.opacity = found * (0.45 + 0.55 * beat * pulse);
      heartGlow.scale.setScalar(0.16 + 0.16 * beat * pulse);
      flowerHalo.material.opacity = 0.1 + 0.3 * found;
      flowerHalo.scale.setScalar(0.7 + 0.5 * found);
      flowerLight.intensity = (0.6 + 2.4 * found) * (1 + 0.4 * beat * pulse) * dream;
      bloom.rotation.z = Math.sin(time * 0.6) * 0.03 * (1 - still);
      fsMat.uniforms.uShow.value = found;
      var fp = fsGeo.attributes.position.array;
      for (k = 0; k < FS; k++) {
        var lt = (time * 0.12 + sparkSeed[k * 3]) % 1, ra = sparkSeed[k * 3 + 2] * (0.3 + lt);
        fp[k * 3] = Math.cos(sparkSeed[k * 3 + 1] + time * 0.3) * ra;
        fp[k * 3 + 1] = 0.5 + lt * 1.6;
        fp[k * 3 + 2] = Math.sin(sparkSeed[k * 3 + 1] + time * 0.3) * ra;
      }
      fsGeo.attributes.position.needsUpdate = true;
    }
    lastU = u;

    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) {
      fitCamera(gl, camera, w, h, dpr, small);
      var pr = gl.getPixelRatio();
      pxScale.value = h * pr / (2 * Math.tan(camera.fov * Math.PI / 360));
      starMat.uniforms.uScale.value = pr * (h / 900 + 0.35);
      layout.portrait = w / h < 1;
      layout.sx = layout.portrait ? 0.5 : 1;
    },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('bubble-moon', {
  renderer: renderer3d,
  maxLines: 5,
  scrim: 0.6,
  accent: '#ff8a9c',
  emphasis: /^(rose|Flower)\W*$/,
  align: ['left', 'right', 'left', 'left', 'right', 'left', 'right', 'left'],
  // Panels: 0 I; 1 II; 2-3 III; 4-5 IV; 6-7 V.
  keys: function (T) {
    TL = T;
    function at(i, d) { return T.start(Math.min(i, T.count - 1)) + d; }   // d units into panel i (0..1.6)
    EV = {
      knock: at(4, 0.6),
      blow0: at(6, 0.3), blow1: at(6, 0.6), rise1: at(6, 1.05),
      stars: SONG.map(function (f, k) { return at(6, 0.78 + k * 0.042); }),
      flower0: at(7, 0.4), flower1: at(7, 0.62), heart0: at(7, 0.6), heart1: at(7, 0.85)
    };
    var A = Math.PI * 2;
    //  unit          x      dark  eye   wind  z       yaw         pitch  open still gate dream moon pAz  pPitch
    return [
      [0,             0.00,  0.10, 1.55, 0.00,  3.40,  0.00,       -0.20, 0,   0,    0,   0,    0,    0,  -0.05],
      [0.6,           0.00,  0.10, 1.55, 0.00,  3.30,  0.00,       -0.21, 0,   0,    0,   0,    0,    0,  -0.05],
      [at(0, 0.7),   -0.05,  0.12, 1.42, 0.05,  2.35,  0.06,       -0.42, 0,   0,    0,   0,    0,    0,   0.02],   // the desk: papers, jigsaw, clock
      [at(0, 1.45),   0.05,  0.15, 1.45, 0.05,  2.05, -0.04,       -0.36, 0,   0,    0,   0,    0,    0,   0.02],
      [at(1, 0.1),    0.00,  0.15, 1.50, 0.10,  1.85,  0.00,       -0.18, 0,   0,    0,   0,    0,    0,   0.00],
      [at(1, 0.42),   0.00,  0.12, 1.60, 0.60,  1.30,  0.00,       -0.04, 1,   0,    0,   0,    0,    0,   0.00],   // "Come with me, then": it opens
      [at(1, 0.70),   0.00,  0.12, 1.62, 0.60, -0.60,  0.00,       -0.02, 1,   0,    0,   0,    0,    0,   0.00],   // out of the window
      [at(1, 1.00),   0.20,  0.15, 1.95, 0.50, -3.40,  1.70,       -0.08, 1,   0,    0,   0,    0,    0,   0.00],
      [at(1, 1.35),   0.10,  0.20, 2.30, 0.45, -6.80,  2.98,       -0.10, 1,   0,    0,   0,    0,    0,  -0.04],   // "far and far away"
      [at(2, 0.00),   0.10,  0.26, 2.45, 0.42, -8.60,  3.20,       -0.11, 1,   0,    0,   0,    0,    0,  -0.04],
      [at(2, 0.50),   1.00,  0.42, 1.00, 0.35, -8.25,  5.90,       -0.47, 1,   0,    0,   0,    0,   -8,   0.30],   // the broken toys
      [at(2, 1.40),   1.10,  0.50, 0.95, 0.30, -8.40,  5.91,       -0.50, 1,   0,    0,   0,    0,   -8,   0.30],
      [at(3, 0.45),   1.15,  0.58, 0.92, 0.25, -8.50,  5.92,       -0.52, 1,   0,    0,   0,    0,   -8,   0.30],
      [at(3, 0.62),   1.17,  0.70, 0.90, 0.00, -8.55,  5.92,       -0.52, 1,   1,    0,   0,    0,   -8,   0.30],   // "and—": everything holds still
      [at(3, 0.85),   1.17,  0.74, 0.90, 0.00, -8.55,  5.92,       -0.52, 1,   1,    0,   0,    0,   -8,   0.30],
      [at(3, 1.30),   1.15,  0.80, 0.65, 0.00, -8.50,  5.87,       -0.42, 1,   1,    0,   0,    0,   -8,   0.26],   // "So am I."
      [at(4, 0.05),   0.80,  0.75, 1.00, 0.05, -10.0,  6.00,       -0.12, 1,   0.6,  0,   0,    0,    0,   0.00],
      [at(4, 0.45),   0.35,  0.65, 1.50, 0.15, -12.8,  A - 0.16,    0.04, 1,   0,    0,   0.12, 0,   15,   0.22],   // the gate, the rose
      [at(4, 1.00),   0.45,  0.60, 1.55, 0.15, -14.2,  A - 0.17,    0.05, 1,   0,    0,   0.18, 0,   15,   0.24],
      [at(5, 0.40),   0.00,  0.45, 1.55, 0.20, -14.9,  A,           0.04, 1,   0,    1,   0.50, 0,    0,  -0.04],   // "Open to me!"
      [at(5, 1.30),   0.00,  0.30, 1.55, 0.25, -16.2,  A,           0.03, 1,   0,    1,   0.85, 0,    0,  -0.02],
      [at(6, 0.22),   0.00,  0.20, 1.55, 0.25, -19.6,  A,           0.00, 1,   0,    1,   1.00, 0,    0,   0.00],   // "Ah, come with me!"
      [at(6, 0.32),   0.00,  0.20, 1.55, 0.20, -20.2,  A + 0.02,   -0.03, 1,   0,    1,   1.00, 0,    6,   0.18],
      [at(6, 0.60),   0.00,  0.20, 1.50, 0.20, -20.5,  A + 0.04,    0.02, 1,   0,    1,   1.00, 0,    8,   0.20],   // the bubble, blown
      [at(6, 1.05),  -0.30,  0.20, 1.50, 0.25, -21.5,  A + 0.40,    0.38, 1,   0,    1,   1.00, 0,    0,  -0.08],   // the moon; the stars sing
      [at(6, 1.40),  -0.50,  0.20, 1.60, 0.30, -24.0,  A + 0.36,    0.32, 1,   0,    1,   1.00, 0,    0,  -0.06],
      [at(7, 0.40),  -2.40,  0.20, 2.30, 0.35, -48.0,  A + 0.04,    0.02, 1,   0,    1,   1.00, 0,    0,   0.00],   // the unstartled steppes
      [at(7, 0.62),  -3.20,  0.20, 1.80, 0.30, -58.8,  A + 0.01,   -0.20, 1,   0,    1,   1.00, 0,   -8,   0.12],   // the Only Flower
      [at(7, 0.90),  -3.25,  0.20, 1.65, 0.25, -60.9,  A,          -0.20, 1,   0,    1,   1.00, 0.1,  -9,   0.15],
      [at(7, 1.25),  -3.38,  0.15, 1.75, 0.25, -61.1,  A,          -0.10, 1,   0,    1,   1.00, 0.5,  -8,   0.06],   // "the moon comes out of the sea"
      [T.total,      -3.40,  0.10, 1.75, 0.25, -61.3,  A,          -0.12, 1,   0,    1,   1.00, 1.0,  -8,   0.10]
    ];
  },
  sound: {
    src: '/audio/wind.mp3',
    label: 'Play the evening wind and the jacinth song',
    volume: function (row) { return (0.02 + 0.13 * row[3]) * (1 - 0.4 * row[1] * (1 - row[10])); },
    cues: [
      { stanza: 0, at: 0.25, play: ticks },
      { stanza: 1, at: 0.15, play: breath },
      { stanza: 4, at: 0.6, play: knock },
      { stanza: 4, at: 1.05, play: swell },
      { stanza: 6, at: 0.3, play: blow },
      { stanza: 7, at: 0.6, play: heartbeat },
      { stanza: 7, at: 0.9, play: moonrise }
    ].concat(SONG.map(function (f, k) { return { stanza: 6, at: 0.78 + k * 0.042, play: chime(f) }; }))
  }
});
