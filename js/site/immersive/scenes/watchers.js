/*
 * Scene for "Neither Out Far Or In Deep" (Robert Frost): a grey, quiet,
 * wide beach under an overcast sky, and a row of small, distant figures
 * standing at the edge of the water, all facing the sea.
 *
 * I   "The people along the sand all turn and look one way": seen from the
 *     dunes behind them, the scattered figures turn, one by one, to face
 *     the sea, their backs to the land (and to us).
 * II  "A ship keeps raising its hull": from behind the row, a steamer far
 *     out passes along the horizon, its hull rising into view. Then, low on
 *     the wet sand, "the wetter ground like glass reflects a standing gull":
 *     a real mirror (a reflection pass) in the film of water on the sand.
 * III "The land may vary more": looking along the row, cloud shadows and
 *     light run over the dunes; "the water comes ashore": a long swash runs
 *     up the sand to the watchers' feet.
 * IV  "They cannot look out far": out over the water, the haze closes in;
 *     "they cannot look in deep": under the surface, into the murk where the
 *     floor drops away. Then back up and out, high over the beach, to the
 *     tiny row still keeping its watch.
 *
 * Keyframe columns:
 *   [unit, camZ, dark, haze, wind, camX, camY, lookX, lookY, lookZ, turn, ship, surge, vary, pyaw, ppitch]
 * where "vary" walks a break in the cloud along the dunes, and pyaw/ppitch turn
 * the view on portrait screens, whose text sits in the middle, to bring the
 * subject (the gull, the ship, the row) out from behind it.
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, softSprite, scatter,
         particleField, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout (metres): the waterline runs along x at z = 0, the sea is -z ──
var SHORE = 0;
var GULL = new THREE.Vector3(16.2, 0, 3.4);
var SHIP_Z = -1400;
var SUN = new THREE.Vector3(0.25, 0.55, -0.8).normalize();   // a pale glow behind the clouds, over the sea

// ── Noise (JS, for the land) ─────────────────────────────────────────────
function hash(x, y) { var h = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return h - Math.floor(h); }
function noise(x, y) {
  var ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy;
  fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
  return lerp(lerp(hash(ix, iy), hash(ix + 1, iy), fx), lerp(hash(ix, iy + 1), hash(ix + 1, iy + 1), fx), fy) * 2 - 1;
}
function fbm(x, y) { return noise(x, y) * 0.55 + noise(x * 2.1 + 5.2, y * 2.1 + 1.3) * 0.28 + noise(x * 4.4 + 9.1, y * 4.4 + 7.7) * 0.14; }

// The ground: a sea floor that shelves, then drops into the deep; a flat
// band of wet sand; the dry beach rising to a belt of dunes; rolling land.
function ground(x, z) {
  var d = z - SHORE, y;
  if (d < 0) y = (d > -20 ? d * 0.09 : -1.8 + (d + 20) * 0.32) - 0.06;
  else y = -0.06 + Math.min(Math.max(0, d - 11), 19) * 0.07;
  y = Math.max(y, -40);
  var dn = smooth(22, 44, d);
  if (dn > 0) {
    var ridge = 1 - Math.abs(noise(x * 0.016 + 3.1, z * 0.024));
    y += dn * (3.6 + 3.2 * fbm(x * 0.011, z * 0.016) + 1.8 * Math.sin(x * 0.027 + z * 0.018 + 2.5 * fbm(x * 0.006, z * 0.009)) +
               3.2 * ridge * ridge) * (1 - 0.4 * smooth(110, 220, d));
    y -= 3.2 * Math.exp(-Math.pow((d - 70 - 8 * noise(x * 0.02, 4.2)) / 13, 2)) * (0.6 + 0.4 * noise(x * 0.03, 8.1));   // the slack behind the foredune
  }
  return y;
}

// Where the marram grows: patches on the dunes, with bare blowouts between.
function grassAt(x, z) {
  return smooth(18, 32, z - SHORE) * smooth(-0.3, 0.3, fbm(x * 0.03, z * 0.03) + 0.12);
}

// The same value noise in GLSL, for the sky, the water and cloud shadows.
var GLSL_NOISE =
  'float hash2(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }\n' +
  'float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);\n' +
  '  return mix(mix(hash2(i), hash2(i + vec2(1.0, 0.0)), f.x), mix(hash2(i + vec2(0.0, 1.0)), hash2(i + vec2(1.0, 1.0)), f.x), f.y); }\n' +
  'float fbm2(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 5; i++) { s += a * vnoise(p); p = p * 2.03 + vec2(1.7, 9.2); a *= 0.5; } return s; }\n';

// A break in the cloud: a warm patch of light (centre x, z, strength)
// that drifts over the land while the sea stays grey.
var BREAK_GLSL =
  'vec3 sunBreak(vec3 w, vec3 b){ vec2 q = (w.xz - b.xy) / vec2(46.0, 24.0);\n' +
  ' float land = smoothstep(9.0, 15.0, w.z), k = exp(-dot(q, q) * 1.4) * b.z * land;\n' +
  ' return vec3(1.0 - 0.4 * b.z * land) + k * vec3(0.72, 0.62, 0.42); }\n';

// ── Sound ────────────────────────────────────────────────────────────────
// A herring gull's call: a few falling "kyow"s, rough and nasal.
function gullCry(ac, out) {
  var now = ac.currentTime + 0.1;
  [0, 0.42, 0.8, 1.12].forEach(function (dt, k) {
    var t = now + dt, len = 0.34 - k * 0.03;
    var o = ac.createOscillator(), bp = ac.createBiquadFilter(), g = ac.createGain();
    var lfo = ac.createOscillator(), lg = ac.createGain();
    o.type = 'sawtooth';
    o.frequency.setValueAtTime(1150, t);
    o.frequency.linearRampToValueAtTime(1550 - k * 60, t + len * 0.3);
    o.frequency.exponentialRampToValueAtTime(820, t + len);
    lfo.frequency.value = 38;
    lg.gain.value = 40;
    lfo.connect(lg); lg.connect(o.frequency);
    bp.type = 'bandpass';
    bp.frequency.value = 2100;
    bp.Q.value = 2.2;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.16 / (1 + k * 0.4), t + 0.03);
    g.gain.exponentialRampToValueAtTime(0.0001, t + len);
    o.connect(bp); bp.connect(g); g.connect(out);
    o.start(t); lfo.start(t);
    o.stop(t + len + 0.05); lfo.stop(t + len + 0.05);
  });
}

// A long wash of water up the sand: filtered noise that swells and drains.
function wash(ac, out) {
  var t = ac.currentTime, len = 4.2, b = ac.createBuffer(1, ac.sampleRate * len, ac.sampleRate), d = b.getChannelData(0);
  for (var i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  var src = ac.createBufferSource(), lp = ac.createBiquadFilter(), g = ac.createGain();
  src.buffer = b;
  lp.type = 'lowpass';
  lp.frequency.setValueAtTime(500, t);
  lp.frequency.linearRampToValueAtTime(2600, t + 1.3);
  lp.frequency.exponentialRampToValueAtTime(380, t + len);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.35, t + 1.2);
  g.gain.exponentialRampToValueAtTime(0.0001, t + len);
  src.connect(lp); lp.connect(g); g.connect(out);
  src.start(t);
}

// ── Pieces ───────────────────────────────────────────────────────────────
// Overcast sky: a grey gradient with a slow, mottled cloud deck and a pale
// glow where the sun is behind it.
function skyMaterial() {
  return new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false,
    uniforms: {
      top: { value: new THREE.Color('#6f767d') }, mid: { value: new THREE.Color('#90969b') },
      horizon: { value: new THREE.Color('#b3b6b5') }, sunDir: { value: SUN.clone() },
      uTime: { value: 0 }, uDark: { value: 0 }, uFlat: { value: new THREE.Color() }, uFlatAmt: { value: 0 }
    },
    vertexShader: 'varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform vec3 top; uniform vec3 mid; uniform vec3 horizon; uniform vec3 sunDir; uniform float uTime; uniform float uDark; uniform vec3 uFlat; uniform float uFlatAmt; varying vec3 vP;\n' +
      GLSL_NOISE +
      'void main(){ vec3 d = normalize(vP); float h = max(d.y, 0.0);\n' +
      ' vec3 c = mix(horizon, mid, smoothstep(0.0, 0.2, h)); c = mix(c, top, smoothstep(0.2, 0.85, h));\n' +
      ' vec2 uv = d.xz / (h + 0.12) * 1.4 + vec2(uTime * 0.006, uTime * 0.0025);\n' +
      ' float n = fbm2(uv * 0.9), n2 = fbm2(uv * 3.1 + 4.0);\n' +
      ' float sh = (n - 0.5) * 0.22 + (n2 - 0.5) * 0.06;\n' +
      ' c *= 1.0 + sh * smoothstep(0.0, 0.12, h);\n' +
      ' float s = max(dot(d, normalize(sunDir)), 0.0);\n' +
      ' c += vec3(0.95, 0.93, 0.88) * (pow(s, 6.0) * 0.16 + pow(s, 40.0) * 0.12) * (0.6 + 0.8 * n);\n' +
      ' gl_FragColor = vec4(mix(c, vec3(0.03, 0.035, 0.04), uDark), 1.0);\n' +
      ' #include <tonemapping_fragment>\n #include <colorspace_fragment>\n' +
      ' gl_FragColor.rgb = mix(gl_FragColor.rgb, uFlat, uFlatAmt);\n }'
  });
}

// The sea and the wet sand, as one plane at y = 0. It samples a mirror
// render of the world: a rippled sea that turns to sky at grazing angles,
// a swash that runs up and drains back, and behind it a film of water on
// the sand that reflects like glass and dulls as the beach dries. Where
// the sand is only damp it fades out to show the ground beneath it.
function shoreMaterial(tRefl, feetCount) {
  var uniforms = {
    tRefl: { value: tRefl }, uTexMat: { value: new THREE.Matrix4() }, uHasRefl: { value: 0 },
    uTime: { value: 0 }, uSurge: { value: 0 }, uDark: { value: 0 },
    uSea: { value: new THREE.Color('#3e4a4c') }, uFoam: { value: new THREE.Color('#d9dcda') },
    uWet: { value: new THREE.Color('#5d584f') }, uSky: { value: new THREE.Color('#a9adad') },
    uMurk: { value: new THREE.Color('#33403f') },
    uFogColor: { value: new THREE.Color() }, uFogDensity: { value: 0.003 }
  };
  return new THREE.ShaderMaterial({
    uniforms: uniforms, transparent: true, side: THREE.DoubleSide,
    vertexShader: 'uniform mat4 uTexMat; varying vec3 vW; varying vec4 vR;\n' +
      'void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; vR = uTexMat * w; gl_Position = projectionMatrix * viewMatrix * w; }',
    fragmentShader:
      'uniform sampler2D tRefl; uniform float uHasRefl; uniform float uTime; uniform float uSurge; uniform float uDark;\n' +
      'uniform vec3 uSea; uniform vec3 uFoam; uniform vec3 uWet; uniform vec3 uSky; uniform vec3 uMurk; uniform vec3 uFogColor; uniform float uFogDensity;\n' +
      'varying vec3 vW; varying vec4 vR;\n' + GLSL_NOISE +
      // Small travelling waves: the slope of their sum (analytic).
      'vec2 waveGrad(vec2 p, float t, float fine){ vec2 g = vec2(0.0); vec2 d; float k;\n' +
      ' p += 9.0 * vec2(vnoise(p * 0.013), vnoise(p * 0.013 + 7.3)) + 2.5 * vec2(vnoise(p * 0.06 + 3.1), vnoise(p * 0.06 + 5.7));\n' +
      ' d = normalize(vec2(0.15, 1.0)); k = 0.35; g += d * cos(dot(d, p) * k + t * 1.1) * 0.10 * k;\n' +
      ' d = normalize(vec2(-0.5, 1.0)); k = 0.8; g += d * cos(dot(d, p) * k + t * 1.6) * 0.05 * k;\n' +
      ' d = normalize(vec2(0.8, 0.6)); k = 1.7; g += d * cos(dot(d, p) * k + t * 2.3) * 0.022 * k * fine;\n' +
      ' d = normalize(vec2(-0.9, 0.3)); k = 3.1; g += d * cos(dot(d, p) * k + t * 3.0) * 0.011 * k * fine;\n' +
      ' d = normalize(vec2(0.3, -1.0)); k = 5.3; g += d * cos(dot(d, p) * k + t * 3.9) * 0.006 * k * fine;\n' +
      ' return g; }\n' +
      'void main(){\n' +
      ' vec3 toCam = cameraPosition - vW; float dist = length(toCam); vec3 V = toCam / dist;\n' +
      ' float t = uTime;\n' +
      // The waterline: sets of swash running up and draining back, staggered along the beach.
      ' float ph = t * 0.5 + vW.x * 0.03 + 2.0 * vnoise(vec2(vW.x * 0.02, 3.0));\n' +
      ' float run = pow(0.5 + 0.5 * sin(ph), 2.2);\n' +
      ' float zW = ' + SHORE.toFixed(1) + ' + 0.4 + run * 3.0 + uSurge * 7.4 + 0.5 * (vnoise(vec2(vW.x * 0.25, t * 0.25)) - 0.5);\n' +
      ' float depth = zW - vW.z;\n' +
      ' float water = smoothstep(-0.05, 0.12, depth);\n' +
      ' float fine = 1.0 - smoothstep(25.0, 160.0, dist);\n' +
      ' vec2 g = waveGrad(vW.xz, t, fine) * (0.1 + 0.9 * smoothstep(0.0, 6.0, depth));\n' +
      ' g += (vec2(vnoise(vW.xz * 1.7 + t * 0.4), vnoise(vW.xz * 1.7 - t * 0.35)) - 0.5) * 0.03 * fine;\n' +
      ' vec3 N = normalize(vec3(-g.x, 1.0, -g.y));\n' +
      ' float cosv = max(dot(N, V), 0.0), fres = 0.02 + 0.98 * pow(1.0 - cosv, 5.0);\n' +
      ' vec4 rc = vR; rc.xy += g * 0.35 * rc.w;\n' +
      ' vec3 refl = uHasRefl > 0.5 ? texture2DProj(tRefl, rc).rgb : uSky;\n' +
      ' vec3 col; float a;\n' +
      ' if (!gl_FrontFacing) {\n' +
      // From below: a bright, rippling ceiling, brightest overhead.
      '   float up = smoothstep(0.0, 1.0, -V.y);\n' +
      '   col = mix(uMurk * 1.5, uSky * 1.05, pow(up, 0.7) * (0.7 + 0.3 * vnoise(vW.xz * 0.6 + t * 0.3))); a = 1.0;\n' +
      ' } else {\n' +
      // Sea: dark water under a sky-coloured sheen, with foam lace at the swash edge and broken lines further out.
      '   vec3 sea = mix(uSea, refl, clamp(fres * 1.05 + 0.1, 0.0, 1.0));\n' +
      '   float shallow = 1.0 - smoothstep(0.0, 5.0, depth);\n' +
      '   float lace = smoothstep(-0.02, 0.18, depth) * (1.0 - smoothstep(0.18, 1.3 + uSurge, depth));\n' +
      '   lace *= 0.35 + 0.65 * smoothstep(0.38, 0.62, fbm2(vW.xz * vec2(0.7, 1.6) + vec2(0.0, t * 0.25)));\n' +
      '   float off = ' + SHORE.toFixed(1) + ' - vW.z + 3.0 * vnoise(vec2(vW.x * 0.03, 2.0)) + 1.2 * vnoise(vec2(vW.x * 0.11, 5.0));\n' +
      '   float band = fract(off / 9.0 - t * 0.075);\n' +
      '   float breakers = smoothstep(0.86, 0.95, band) * (1.0 - smoothstep(0.95, 1.0, band)) * smoothstep(2.0, 7.0, off) * (1.0 - smoothstep(14.0, 32.0, off));\n' +
      '   breakers *= smoothstep(0.45, 0.75, fbm2(vW.xz * vec2(0.22, 0.5) + vec2(t * 0.05, 0.0)));\n' +
      '   sea = mix(sea, uFoam, clamp(lace * 0.85 + breakers * 0.45, 0.0, 1.0));\n' +
      '   float seaA = mix(1.0, 0.72, shallow * (1.0 - lace));\n' +
      // Wet sand: a mirror near the water that dulls in patches as it dries.
      '   float dry = smoothstep(0.0, 8.0, -depth);\n' +
      '   float edge = ' + (SHORE + 11.0).toFixed(1) + ' + 2.0 * (vnoise(vec2(vW.x * 0.06, 1.7)) - 0.5);\n' +
      '   float patchy = smoothstep(0.3, 0.62, fbm2(vW.xz * vec2(0.18, 0.42)));\n' +
      '   float gloss = (1.0 - 0.75 * dry) * (0.55 + 0.45 * patchy) * (1.0 - smoothstep(edge - 4.0, edge, vW.z));\n' +
      '   float sandA = gloss * (0.72 + 0.28 * fres);\n' +
      '   col = mix(refl, sea, water); a = mix(sandA, seaA, water);\n' +
      ' }\n' +
      ' col = mix(col, vec3(0.03, 0.035, 0.04), uDark);\n' +
      ' gl_FragColor = vec4(col, a);\n' +
      ' #include <tonemapping_fragment>\n #include <colorspace_fragment>\n' +
      // Fog last, in output colours, as three.js does for its own materials.
      ' float fogF = 1.0 - exp(-uFogDensity * uFogDensity * dist * dist);\n' +
      ' gl_FragColor.rgb = mix(gl_FragColor.rgb, uFogColor, fogF);\n}'
  });
}

// A standing figure, about 1.75 m, as plain shapes: legs, a coat, shoulders,
// a head. `long` gives a longer coat and a hat. Shoulders run along x, so a
// figure facing the sea (-z) is broader than one seen side-on.
function figureGeometry(long) {
  var c = '#ffffff', parts = [];
  [-1, 1].forEach(function (s) {
    parts.push(tinted(new THREE.CylinderGeometry(0.065, 0.055, long ? 0.6 : 0.86, 5).translate(s * 0.095, long ? 0.3 : 0.43, 0), c));
  });
  var coatH = long ? 0.98 : 0.66, coatY = long ? 0.55 + coatH / 2 : 0.82 + coatH / 2;
  parts.push(tinted(new THREE.CylinderGeometry(0.2, long ? 0.3 : 0.23, coatH, 8).scale(1, 1, 0.62).translate(0, coatY, 0), c));
  parts.push(tinted(new THREE.SphereGeometry(0.21, 8, 5).scale(1.05, 0.42, 0.62).translate(0, 1.48, 0), c));
  parts.push(tinted(new THREE.CylinderGeometry(0.05, 0.06, 0.1, 5).translate(0, 1.56, 0), c));
  parts.push(tinted(new THREE.SphereGeometry(0.105, 8, 6).scale(1, 1.12, 1).translate(0, 1.67, 0), c));
  if (long) {
    parts.push(tinted(new THREE.CylinderGeometry(0.17, 0.17, 0.02, 10).translate(0, 1.745, 0), c));
    parts.push(tinted(new THREE.CylinderGeometry(0.09, 0.1, 0.1, 8).translate(0, 1.8, 0), c));
  }
  return merge(parts);
}

// A herring gull standing on the wet sand, about 0.45 m tall, facing -x.
function gull() {
  var g = new THREE.Group(), white = new THREE.MeshLambertMaterial({ color: '#eceeed' });
  var mantle = new THREE.MeshLambertMaterial({ color: '#8f979c' }), black = new THREE.MeshLambertMaterial({ color: '#1c1d1f' });
  var yellow = new THREE.MeshLambertMaterial({ color: '#d9b545' }), pink = new THREE.MeshLambertMaterial({ color: '#c8a197' });
  var body = new THREE.Mesh(new THREE.SphereGeometry(0.1, 14, 10).scale(2.1, 0.95, 1.0), white);
  body.position.set(0, 0.25, 0);
  body.rotation.z = -0.18;
  var wings = new THREE.Mesh(new THREE.SphereGeometry(0.1, 14, 10).scale(2.15, 0.6, 1.08), mantle);
  wings.position.set(0.05, 0.285, 0);
  wings.rotation.z = -0.22;
  var tips = new THREE.Mesh(new THREE.ConeGeometry(0.045, 0.15, 6).rotateZ(-Math.PI / 2).scale(1, 0.45, 1.3), black);
  tips.position.set(0.25, 0.29, 0);
  tips.rotation.z = 0.1;
  var tail = new THREE.Mesh(new THREE.ConeGeometry(0.05, 0.12, 6).rotateZ(-Math.PI / 2).scale(1, 0.5, 1.1), white);
  tail.position.set(0.22, 0.25, 0);
  tail.rotation.z = 0.15;
  var head = new THREE.Group();
  head.position.set(-0.17, 0.36, 0);
  var skull = new THREE.Mesh(new THREE.SphereGeometry(0.058, 12, 9).scale(1.15, 1, 0.95), white);
  var beak = new THREE.Mesh(new THREE.ConeGeometry(0.016, 0.075, 6).rotateZ(Math.PI / 2), yellow);
  beak.position.set(-0.09, -0.01, 0);
  var eye = new THREE.Mesh(new THREE.SphereGeometry(0.008, 6, 4), black);
  eye.position.set(-0.035, 0.015, 0.05);
  var eye2 = eye.clone();
  eye2.position.z = -0.05;
  head.add(skull, beak, eye, eye2);
  var neck = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.06, 0.1, 8), white);
  neck.position.set(-0.15, 0.31, 0);
  neck.rotation.z = 0.5;
  g.add(body, wings, tail, tips, head, neck);
  [-1, 1].forEach(function (s) {
    var leg = new THREE.Mesh(new THREE.CylinderGeometry(0.007, 0.008, 0.17, 5), pink);
    leg.position.set(0.0, 0.085, s * 0.035);
    var foot = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.006, 0.035), pink);
    foot.position.set(-0.015, 0.003, s * 0.035);
    g.add(leg, foot);
  });
  return { group: g, head: head };
}

// A steamer seen side-on, as one flat-shaded silhouette: hull, bridge,
// funnel and two masts, extruded thin. y = 0 is the waterline.
function steamer() {
  var s = new THREE.Shape();
  s.moveTo(-58, -20);
  s.lineTo(-58, 7.5); s.lineTo(-52, 8.6); s.lineTo(-30, 8.2);
  // after deckhouse
  s.lineTo(-30, 12); s.lineTo(-18, 12); s.lineTo(-18, 8.2);
  // mainmast
  s.lineTo(-24.6, 8.2); s.lineTo(-24.6, 34); s.lineTo(-23.4, 34); s.lineTo(-23.4, 12.2);
  s.lineTo(-18, 12.2); s.lineTo(-18, 8.2);
  // midships house and bridge, funnel
  s.lineTo(-10, 8.2); s.lineTo(-10, 14); s.lineTo(-3, 14); s.lineTo(-3.5, 25.5); s.lineTo(4.5, 25.5); s.lineTo(4, 14);
  s.lineTo(10, 14); s.lineTo(10, 18); s.lineTo(18, 18); s.lineTo(18, 14); s.lineTo(20, 14); s.lineTo(20, 8.2);
  // foremast
  s.lineTo(30.4, 8.2); s.lineTo(30.4, 31); s.lineTo(31.6, 31); s.lineTo(31.6, 8.4);
  s.lineTo(48, 9.2); s.lineTo(60, 11.2); s.lineTo(56, -20); s.lineTo(-58, -20);
  return new THREE.ExtrudeGeometry(s, { depth: 14, bevelEnabled: false }).translate(0, 0, -7);
}

// A clump of marram grass: thin blades leaning out from a point.
function grassGeometry(r) {
  var pos = [], nor = [], col = [];
  for (var i = 0; i < 14; i++) {
    var a = r() * Math.PI * 2, lean = 0.12 + r() * 0.4, h = 0.35 + r() * 0.45, w = 0.022;
    var cx = Math.cos(a), cz = Math.sin(a), px = -cz * w, pz = cx * w;
    var tx = cx * lean * h, tz = cz * lean * h;
    pos.push(-px, 0, -pz, px, 0, pz, tx, h, tz);
    nor.push(0, 1, 0, 0, 1, 0, 0, 1, 0);
    col.push(0.5, 0.5, 0.48, 0.5, 0.5, 0.48, 1.05, 1.04, 1.0);       // darker at the root
  }
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  return geo;
}

// ── Renderer ─────────────────────────────────────────────────────────────
function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(1936);
  var gl = makeRenderer(canvas, { clear: '#a7acad' });
  gl.localClippingEnabled = true;
  gl.toneMappingExposure = 1.15;
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#a5aaab', 0.003);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.1, 6000);
  var U = { time: { value: 0 }, vary: { value: 0 }, wind: { value: 0.3 }, brk: { value: new THREE.Vector3() } };

  var sky = new THREE.Group();
  world.add(sky);
  var skyMat = skyMaterial();
  sky.add(new THREE.Mesh(new THREE.SphereGeometry(4000, 32, 16), skyMat));

  var hemi = new THREE.HemisphereLight('#e2e5e7', '#7a7569', 2.0);
  var sun = new THREE.DirectionalLight('#f1efe9', 0.9);
  sun.position.copy(SUN).multiplyScalar(100);
  world.add(hemi, sun);

  // ── The land ──────────────────────────────────────────────────────────
  var FEET = 16, feet = [];
  for (var i = 0; i < FEET; i++) feet.push(new THREE.Vector3(0, 0, 0));
  var feetU = { value: feet };
  var wetC = new THREE.Color('#6a645a'), dryC = new THREE.Color('#a39d8f'), duneC = new THREE.Color('#a9a393'),
      floorC = new THREE.Color('#5d5f58'), scrubC = new THREE.Color('#8a8a74'), tc = new THREE.Color();
  var landMat = new THREE.MeshLambertMaterial({ vertexColors: true });
  landMat.onBeforeCompile = function (sh) {
    sh.uniforms.uTime = U.time; sh.uniforms.uVary = U.vary; sh.uniforms.uFeet = feetU; sh.uniforms.uBreak = U.brk;
    sh.vertexShader = 'varying vec3 vW;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n vW = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    sh.fragmentShader = 'varying vec3 vW; uniform float uTime; uniform float uVary; uniform vec3 uBreak; uniform vec3 uFeet[' + FEET + '];\n' + GLSL_NOISE + BREAK_GLSL +
      sh.fragmentShader.replace('#include <color_fragment>', '#include <color_fragment>\n' +
      // Cloud shadows drifting over the land: faster and deeper when "the land may vary".
      ' vec2 cp = vW.xz * 0.016 + vec2(uTime * 0.02, uTime * 0.008) * (1.0 + uVary * 3.0);\n' +
      ' float cs = fbm2(cp);\n' +
      ' diffuseColor.rgb *= 1.0 - smoothstep(0.5, 0.66, cs) * (0.12 + 0.3 * uVary);\n' +
      ' diffuseColor.rgb *= sunBreak(vW, uBreak);\n' +
      // Soft contact shadows under the watchers and the gull.
      ' for (int i = 0; i < ' + FEET + '; i++) { vec2 dd = (vW.xz - uFeet[i].xy) / max(uFeet[i].z, 0.001);\n' +
      '   diffuseColor.rgb *= 1.0 - 0.5 * exp(-dot(dd, dd) * 2.5) * step(0.001, uFeet[i].z); }');
  };
  var LW = 1800, LD = 470, landGeo = new THREE.PlaneGeometry(LW, LD, small ? 220 : 420, small ? 100 : 170).rotateX(-Math.PI / 2).translate(0, 0, 25);
  var lp = landGeo.attributes.position, lcol = new Float32Array(lp.count * 3);
  for (i = 0; i < lp.count; i++) {
    var x = lp.getX(i), z = lp.getZ(i), y = ground(x, z), d = z - SHORE;
    lp.setY(i, y);
    if (d < 0) tc.copy(floorC).lerp(wetC, smooth(-14, 0, d));
    else {
      tc.copy(wetC).lerp(dryC, smooth(9, 16, d + 2 * fbm(x * 0.05, 3)));
      tc.lerp(duneC, smooth(24, 40, d));
      tc.lerp(scrubC, Math.max(smooth(40, 90, d) * (0.4 + 0.4 * fbm(x * 0.02, z * 0.02)), grassAt(x, z) * 0.55));
      tc.multiplyScalar(1 + 0.05 * fbm(x * 0.15, z * 0.15));
    }
    lcol[i * 3] = tc.r; lcol[i * 3 + 1] = tc.g; lcol[i * 3 + 2] = tc.b;
  }
  landGeo.setAttribute('color', new THREE.BufferAttribute(lcol, 3));
  landGeo.computeVertexNormals();
  // Shade the dunes by slope, so their shapes read under a flat grey light:
  // seaward faces catch the brightness, the landward lee is darker.
  var ln = landGeo.attributes.normal;
  for (i = 0; i < lp.count; i++) {
    var ny = ln.getY(i), nz = ln.getZ(i), k = clamp(1 - (1 - ny) * 2.2 - nz * 1.4, 0.62, 1.12);
    lcol[i * 3] *= k; lcol[i * 3 + 1] *= k; lcol[i * 3 + 2] *= k;
  }
  var land = new THREE.Mesh(landGeo, landMat);
  world.add(land);

  // Marram grass on the dunes, swaying in the wind.
  var grassMat = new THREE.MeshLambertMaterial({ color: '#ffffff', vertexColors: true, side: THREE.DoubleSide });
  grassMat.onBeforeCompile = function (sh) {
    sh.uniforms.uTime = U.time; sh.uniforms.uWind = U.wind; sh.uniforms.uBreak = U.brk;
    sh.vertexShader = 'uniform float uTime; uniform float uWind; varying vec3 vW;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n vec4 io = instanceMatrix[3]; vW = io.xyz; float sway = sin(uTime * 1.6 + io.x * 0.21 + io.z * 0.13) * (0.05 + 0.12 * uWind) * transformed.y;\n' +
      ' transformed.x += sway; transformed.z += sway * 0.4;');
    // Light both faces of a blade alike, as if from above.
    sh.fragmentShader = 'uniform vec3 uBreak; varying vec3 vW;\n' + BREAK_GLSL + sh.fragmentShader
      .replace('#include <normal_fragment_begin>', 'float faceDirection = 1.0; vec3 normal = normalize(vNormal); vec3 nonPerturbedNormal = normal;')
      .replace('#include <color_fragment>', '#include <color_fragment>\n diffuseColor.rgb *= sunBreak(vW, uBreak);');
  };
  var grass = new THREE.InstancedMesh(grassGeometry(r), grassMat, small ? 5000 : 14000);
  var gc1 = new THREE.Color('#8f8f74'), gc2 = new THREE.Color('#b8b297');
  scatter(grass, 160000, function (n, p, q, s, c) {
    // Densest near the middle of the beach, where the camera comes down.
    var gx = (r() - 0.5) * (r() < 0.6 ? 180 : 480), gz = SHORE + 17 + Math.pow(r(), 1.4) * 170;
    if (r() > grassAt(gx, gz)) return false;
    p.set(gx, ground(gx, gz) - 0.05, gz);
    q.setFromAxisAngle(THREE.Object3D.DEFAULT_UP, r() * 6.28);
    s.set(0.7 + r() * 0.6, 0.55 + r() * 0.6, 0.7 + r() * 0.6);
    c.copy(gc1).lerp(gc2, r());
  });
  world.add(grass);

  // Scrub and low pines on the land beyond the dunes.
  var pine = new THREE.InstancedMesh(merge([
    tinted(new THREE.CylinderGeometry(0.1, 0.16, 1.4, 5).translate(0, 0.7, 0), '#4a4438'),
    tinted(new THREE.ConeGeometry(1.5, 3.2, 7).translate(0, 2.6, 0), '#ffffff'),
    tinted(new THREE.ConeGeometry(1.1, 2.4, 7).translate(0, 3.8, 0), '#ffffff')
  ]), new THREE.MeshLambertMaterial({ vertexColors: true }), small ? 160 : 380);
  var pc1 = new THREE.Color('#5b6152'), pc2 = new THREE.Color('#717461');
  scatter(pine, 6000, function (n, p, q, s, c) {
    var px = (r() - 0.5) * 900, pz = SHORE + 95 + r() * 150;
    if (fbm(px * 0.012, pz * 0.012) < -0.05) return false;
    p.set(px, ground(px, pz) - 0.2, pz);
    q.setFromAxisAngle(THREE.Object3D.DEFAULT_UP, r() * 6.28);
    s.set(0.8 + r() * 0.8, 0.7 + r() * 1.0, 0.8 + r() * 0.8);
    c.copy(pc1).lerp(pc2, r());
  });
  world.add(pine);

  // A weathered sand fence along the foot of the dunes, with two wires.
  var posts = new THREE.InstancedMesh(new THREE.BoxGeometry(0.1, 1.25, 0.1), new THREE.MeshLambertMaterial({ color: '#5f584d' }), 140);
  var wire = [], m4 = new THREE.Matrix4(), qq = new THREE.Quaternion(), ee = new THREE.Euler(), ss = new THREE.Vector3(1, 1, 1), pp = new THREE.Vector3();
  var lastPost = null, np = 0;
  for (var fx = -170; fx < 170 && np < 140; fx += 2.6) {
    var fz = SHORE + 25.5 + 2.2 * Math.sin(fx * 0.03) + (r() - 0.5) * 0.5;
    if (r() < 0.12) { lastPost = null; continue; }           // a gap where the fence has gone
    var fy = ground(fx, fz), tall = 0.85 + r() * 0.4;
    ee.set((r() - 0.5) * 0.18, 0, (r() - 0.5) * 0.18);
    posts.setMatrixAt(np++, m4.compose(pp.set(fx, fy + tall * 0.5 - 0.15, fz), qq.setFromEuler(ee), ss.set(1, tall, 1)));
    if (lastPost) {
      [0.45, 0.8].forEach(function (h) { wire.push(lastPost.x, lastPost.y + h, lastPost.z, fx, fy + h, fz); });
    }
    lastPost = { x: fx, y: fy, z: fz };
  }
  posts.count = np;
  posts.instanceMatrix.needsUpdate = true;
  var wireGeo = new THREE.BufferGeometry();
  wireGeo.setAttribute('position', new THREE.Float32BufferAttribute(wire, 3));
  world.add(posts, new THREE.LineSegments(wireGeo, new THREE.LineBasicMaterial({ color: '#4f4a42', transparent: true, opacity: 0.7 })));

  // ── The watchers ──────────────────────────────────────────────────────
  // Small groups and singles along the wet sand, with a gap where the gull
  // stands. Each starts facing somewhere of its own and turns to the sea.
  var SPOTS = [[-58, 8.2, 0], [-47, 7.4, 1], [-38.5, 8.6, 0], [-37.4, 8.4, 0, 0.82], [-27, 7.6, 1], [-15, 8.0, 0],
               [-7.5, 7.2, 0], [-6.4, 7.5, 1], [-1.0, 8.6, 0], [31, 7.8, 1], [40, 8.6, 0], [41.1, 8.4, 0, 0.78],
               [52, 7.6, 1], [64, 8.2, 0]];
  var figMat = new THREE.MeshLambertMaterial({ color: '#ffffff' });
  var figs = [new THREE.InstancedMesh(figureGeometry(false), figMat, 14), new THREE.InstancedMesh(figureGeometry(true), figMat, 14)];
  var watchers = [];
  SPOTS.forEach(function (sp, k) {
    var from = (r() - 0.5) * 3.4;
    if (k % 4 === 1) from = Math.PI * (r() < 0.5 ? 0.9 : -0.9);
    watchers.push({ x: sp[0], z: sp[1], kind: sp[2], scale: (sp[3] || 0.95 + r() * 0.12), from: from, to: (r() - 0.5) * 0.25,
                    lag: r() * 0.45, slot: 0 });
    if (k < FEET - 1) feet[k].set(sp[0], sp[1], 0.42);
  });
  var counts = [0, 0];
  watchers.forEach(function (w) { w.slot = counts[w.kind]++; });
  figs.forEach(function (m, k) { m.count = counts[k]; world.add(m); });
  var figC = new THREE.Color('#2b2d30');
  watchers.forEach(function (w) { figs[w.kind].setColorAt(w.slot, tc.copy(figC).multiplyScalar(0.85 + r() * 0.35)); });
  figs.forEach(function (m) { if (m.instanceColor) m.instanceColor.needsUpdate = true; });
  var lastTurn = -1;
  function placeWatchers(turn) {
    if (Math.abs(turn - lastTurn) < 0.0005) return;
    lastTurn = turn;
    watchers.forEach(function (w) {
      var k = smooth(w.lag, w.lag + 0.55, turn);
      ee.set(0, lerp(w.from, w.to, k), 0);
      figs[w.kind].setMatrixAt(w.slot, m4.compose(pp.set(w.x, 0, w.z), qq.setFromEuler(ee), ss.setScalar(w.scale)));
    });
    figs.forEach(function (m) { m.instanceMatrix.needsUpdate = true; });
  }
  placeWatchers(0);

  var bird = gull();
  bird.group.position.copy(GULL);
  bird.group.rotation.y = -0.35;
  world.add(bird.group);
  feet[FEET - 1].set(GULL.x - 0.03, GULL.z, 0.17);

  // ── The sea ───────────────────────────────────────────────────────────
  var mirrorRT = new THREE.WebGLRenderTarget(16, 16, { type: THREE.HalfFloatType });
  var shoreMat = shoreMaterial(mirrorRT.texture);
  var SU = shoreMat.uniforms;
  var shore = new THREE.Mesh(new THREE.PlaneGeometry(6000, 3200).rotateX(-Math.PI / 2).translate(0, 0, SHORE + 15 - 1600), shoreMat);
  shore.frustumCulled = false;
  world.add(shore);

  // The steamer, far out, cut off at the sea line so its hull can rise.
  var seaLine = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  var shipMat = new THREE.MeshBasicMaterial({ color: '#555b60', fog: false, clippingPlanes: [seaLine] });
  var ship = new THREE.Mesh(steamer(), shipMat);
  ship.position.set(-400, -14, SHIP_Z);
  world.add(ship);
  var smokeTex = softSprite('rgba(255,255,255,0.7)', 'rgba(255,255,255,0)'), smoke = [];
  for (i = 0; i < 9; i++) {
    var sm = new THREE.Sprite(new THREE.SpriteMaterial({ map: smokeTex, color: '#7d8287', transparent: true, depthWrite: false, fog: false, opacity: 0 }));
    world.add(sm);
    smoke.push(sm);
  }

  // A few gulls wheeling far out over the water: two flapping wings each.
  var flyers = [], wingGeo = new THREE.PlaneGeometry(0.75, 0.16).translate(0.375, 0, 0).rotateX(-Math.PI / 2);
  var flyMat = new THREE.MeshBasicMaterial({ color: '#43484c', side: THREE.DoubleSide, fog: true });
  for (i = 0; i < 5; i++) {
    var fg = new THREE.Group(), wl = new THREE.Mesh(wingGeo, flyMat), wr = new THREE.Mesh(wingGeo, flyMat);
    wr.scale.x = -1;
    fg.add(wl, wr);
    fg.userData = { cx: -80 + r() * 200, cz: -90 - r() * 120, rad: 16 + r() * 26, h: 14 + r() * 20, sp: (0.05 + r() * 0.05) * (r() < 0.5 ? -1 : 1), ph: r() * 6.28, wl: wl, wr: wr };
    fg.scale.setScalar(1.4);
    world.add(fg);
    flyers.push(fg);
  }

  // ── Under the water: drifting motes and pale shafts of light ───────────
  var motes = particleField({ count: small ? 400 : 800, box: [26, 14, 26], fall: [-0.05, 0.08], size: 0.035, color: '#9fb0aa',
                              map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)'), sway: 0.25 });
  motes.points.material.fog = false;
  world.add(motes.points);
  var shaftTex = (function () {
    var c = document.createElement('canvas');
    c.width = 32; c.height = 256;
    var x = c.getContext('2d'), gr = x.createLinearGradient(0, 0, 0, 256);
    gr.addColorStop(0, 'rgba(255,255,255,0)'); gr.addColorStop(0.12, 'rgba(255,255,255,0.8)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
    x.fillStyle = gr; x.fillRect(0, 0, 32, 256);
    var gh = x.createLinearGradient(0, 0, 32, 0);
    gh.addColorStop(0, 'rgba(0,0,0,1)'); gh.addColorStop(0.5, 'rgba(0,0,0,0)'); gh.addColorStop(1, 'rgba(0,0,0,1)');
    x.globalCompositeOperation = 'destination-out';
    x.fillStyle = gh; x.fillRect(0, 0, 32, 256);
    var t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  })();
  var shafts = [];
  for (i = 0; i < 5; i++) {
    var sf = new THREE.Mesh(new THREE.PlaneGeometry(0.6 + r() * 1.2, 18).translate(0, -9, 0),
      new THREE.MeshBasicMaterial({ map: shaftTex, color: '#b9c9c4', transparent: true, opacity: 0, depthWrite: false,
                                    blending: THREE.AdditiveBlending, fog: false }));
    sf.userData = { x: -22 + r() * 24, z: -56 - r() * 26, tilt: 0.22 + r() * 0.08 };
    world.add(sf);
    shafts.push(sf);
  }

  // ── Mirror: the world rendered from below the water line ───────────────
  var mirrorCam = new THREE.PerspectiveCamera();
  var bias = new THREE.Matrix4().set(0.5, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0, 1);
  var mPlane = new THREE.Plane(), clipV = new THREE.Vector4(), qv = new THREE.Vector4();
  var UP_N = new THREE.Vector3(0, 1, 0), ORIGIN = new THREE.Vector3(), rot = new THREE.Matrix4();
  var look = new THREE.Vector3(), upv = new THREE.Vector3();
  function renderMirror() {
    camera.updateMatrixWorld();
    mirrorCam.position.set(camera.position.x, -camera.position.y, camera.position.z);
    rot.extractRotation(camera.matrixWorld);
    look.set(0, 0, -1).applyMatrix4(rot).add(camera.position);
    look.y = -look.y;
    upv.set(0, 1, 0).applyMatrix4(rot);
    upv.y = -upv.y;
    mirrorCam.up.copy(upv);
    mirrorCam.lookAt(look);
    mirrorCam.far = camera.far;
    mirrorCam.updateMatrixWorld();
    mirrorCam.projectionMatrix.copy(camera.projectionMatrix);
    SU.uTexMat.value.copy(bias).multiply(mirrorCam.projectionMatrix).multiply(mirrorCam.matrixWorldInverse);
    // Oblique near plane at the water, so nothing below it is reflected.
    mPlane.setFromNormalAndCoplanarPoint(UP_N, ORIGIN).applyMatrix4(mirrorCam.matrixWorldInverse);
    clipV.set(mPlane.normal.x, mPlane.normal.y, mPlane.normal.z, mPlane.constant);
    var e = mirrorCam.projectionMatrix.elements;
    qv.set((Math.sign(clipV.x) + e[8]) / e[0], (Math.sign(clipV.y) + e[9]) / e[5], -1, (1 + e[10]) / e[14]);
    clipV.multiplyScalar(2 / clipV.dot(qv));
    e[2] = clipV.x; e[6] = clipV.y; e[10] = clipV.z + 1 - 0.003; e[14] = clipV.w;
    mirrorCam.projectionMatrixInverse.copy(mirrorCam.projectionMatrix).invert();

    shore.visible = false;
    motes.points.visible = false;
    gl.setRenderTarget(mirrorRT);
    gl.render(world, mirrorCam);
    gl.setRenderTarget(null);
    shore.visible = true;
  }

  // ── Per frame ─────────────────────────────────────────────────────────
  var AIR_FOG = new THREE.Color('#a5aaab'), MURK = new THREE.Color('#2e3b3a'), tmpC = new THREE.Color(), shipBase = new THREE.Color('#3f454a');
  var aspect = 1.6, headTurn = 0, headClock = 0, headGoal = 0;

  function frame(f) {
    var row = f.row, dt = f.dt, time = env.reduceMotion ? f.time * 0.6 : f.time;
    var dark = row[1], haze = row[2], wind = row[3], turn = row[9], shipT = row[10], surge = row[11], vary = row[12];
    U.time.value = time;
    // The sun-break sweeps along the dunes as `vary` runs 0 to 1.
    var brk = smooth(0, 0.15, vary) * (1 - smooth(0.85, 1, vary));
    U.vary.value = brk;
    U.brk.value.set(lerp(170, -70, vary), SHORE + 30 + 9 * Math.sin(vary * 3), brk);
    U.wind.value = wind;

    // Camera: keyed position and target, kept above the sand on land.
    var cx = row[4], cy = row[5], cz = row[0];
    var gy = ground(cx, cz);
    if (cy > -0.2) cy = Math.max(cy, Math.max(gy, -0.2) + 0.32);
    camera.position.set(cx, cy + Math.sin(time * 0.9) * 0.012, cz);
    camera.lookAt(row[6], row[7], row[8]);
    if (aspect < 1) { camera.rotateY(row[13]); camera.rotateX(row[14]); }
    camera.rotateY(-f.mx * 0.1);
    camera.rotateX(-f.my * 0.05);
    sky.position.copy(camera.position);

    var under = smooth(0.02, -0.25, camera.position.y);
    var air = 1 - under;

    // Fog: sea haze above (thickened when they "cannot look out far"), murk below.
    // Darker the deeper you go.
    world.fog.color.copy(AIR_FOG).lerp(tmpC.copy(MURK).multiplyScalar(1 - 0.35 * smooth(-1, -7, camera.position.y)), under)
      .multiplyScalar(1 - dark * 0.75);
    world.fog.density = lerp(0.0026 * haze, 0.05, under);
    gl.setClearColor(world.fog.color);
    skyMat.uniforms.uTime.value = time;
    skyMat.uniforms.uDark.value = dark * 0.85;
    // Under water the dome becomes the murk itself, tone-mapped like the fogged world.
    skyMat.uniforms.uFlat.value.copy(world.fog.color).convertLinearToSRGB();
    skyMat.uniforms.uFlatAmt.value = under;
    hemi.intensity = lerp(2.0, 1.1, under) * (1 - dark * 0.7);
    sun.intensity = 0.9 * air * (1 - dark * 0.7);
    hemi.color.set('#e2e5e7').lerp(tmpC.set('#9fb3ad'), under);

    // The watchers turn; the gull now and then turns its head.
    placeWatchers(turn);
    headClock -= dt;
    if (headClock < 0) { headClock = 1.6 + Math.random() * 2.8; headGoal = (Math.random() - 0.5) * 1.6; }
    headTurn += (headGoal - headTurn) * (1 - Math.exp(-dt * 6));
    bird.head.rotation.y = headTurn;
    bird.head.position.y = 0.36 + Math.sin(time * 1.3) * 0.004;

    // The steamer passes along the horizon, its hull rising into view.
    var rise = smooth(0.04, 0.42, shipT);
    ship.position.set(lerp(700, 1600, shipT), lerp(-14.5, 0, rise), SHIP_Z);
    ship.visible = shipT > 0.001 && shipT < 0.999 && air > 0.5;
    shipMat.color.copy(shipBase).lerp(world.fog.color, 0.42 + haze * 0.12);
    smoke.forEach(function (sm, k) {
      var age = (k + (time * 0.25) % 1) / smoke.length;
      sm.position.set(ship.position.x + 0.5 - age * 120 - wind * 30 * age, ship.position.y + 25 + age * 22, SHIP_Z + 5);
      sm.scale.set(14 + age * 70, 8 + age * 26, 1);
      sm.material.opacity = ship.visible ? 0.4 * (1 - age) * smooth(0, 0.15, age) * (0.4 + 0.6 * rise) : 0;
      sm.material.color.copy(shipBase).lerp(world.fog.color, 0.6);
    });

    // Gulls wheeling out over the water.
    flyers.forEach(function (fg) {
      var u = fg.userData, a = u.ph + time * u.sp;
      fg.position.set(u.cx + Math.cos(a) * u.rad, u.h + Math.sin(time * 0.4 + u.ph) * 1.5, u.cz + Math.sin(a) * u.rad * 0.6);
      fg.rotation.set(0, -a + (u.sp > 0 ? 0 : Math.PI), Math.sin(time * 0.6 + u.ph) * 0.2);
      var flap = Math.sin(time * 4.5 + u.ph) * 0.45 * (0.4 + 0.6 * smooth(-0.3, 0.6, Math.sin(time * 0.5 + u.ph)));
      u.wl.rotation.z = flap; u.wr.rotation.z = -flap;
      fg.visible = air > 0.5;
    });

    // Sea: waves, the surge up the sand, fog, and the mirror.
    SU.uTime.value = time;
    SU.uSurge.value = surge;
    SU.uDark.value = dark * 0.8;
    SU.uFogColor.value.copy(world.fog.color).convertLinearToSRGB();
    SU.uFogDensity.value = world.fog.density;
    SU.uSky.value.copy(AIR_FOG);

    // Under water: motes round the camera and light falling from above.
    motes.update({ snow: under, wind: 0.05, dt: dt, time: time }, camera.position, env.reduceMotion);
    shafts.forEach(function (sf, k) {
      var u = sf.userData;
      sf.position.set(u.x, -0.1, u.z);
      sf.rotation.set(0, Math.atan2(camera.position.x - u.x, camera.position.z - u.z), 0);
      sf.rotateZ(u.tilt);
      sf.material.opacity = under * (0.16 + 0.1 * Math.sin(time * 0.7 + k * 1.7));
      sf.visible = under > 0.01;
    });

    if (air > 0.5) {
      renderMirror();
      SU.uHasRefl.value = 1;
    } else {
      SU.uHasRefl.value = 0;
    }
    motes.points.visible = under > 0.01;
    gl.render(world, camera);
  }

  function resize(w, h, dpr) {
    aspect = w / h;
    fitCamera(gl, camera, w, h, dpr, small);
    var ratio = gl.getPixelRatio(), k = small ? 0.5 : 0.6;
    mirrorRT.setSize(Math.max(16, Math.round(w * ratio * k)), Math.max(16, Math.round(h * ratio * k)));
  }

  return {
    resize: resize,
    frame: frame,
    destroy: function () { mirrorRT.dispose(); disposeAll(world, gl); }
  };
}

PI.register('watchers', {
  renderer: renderer3d,
  align: ['right', 'left', 'left', 'center'],
  scrim: 0.6,
  keys: function (T) {
    function at(i, frac) { i = Math.min(i, T.count - 1); return lerp(T.start(i), T.end(i), frac); }
    //  unit          camZ   dark  haze wind  camX   camY   lookX  lookY  lookZ   turn ship surge vary  pyaw   ppitch
    return [
      [0,             57,    0.0,  1.0, 0.30, -5,    11.6,  -2,    -6.0,  -60,    0.0, 0.0, 0.0, 0.0,  0.00, -0.08],
      [0.7,           56,    0.0,  1.0, 0.30, -5,    11.4,  -2,    -5.8,  -60,    0.0, 0.0, 0.0, 0.0,  0.00, -0.08],
      [at(0, 0.3),    50,    0.0,  1.0, 0.30, -6,    10.4,  -4,    -4.5,  -60,    0.15, 0.0, 0.0, 0.0,  0.00, -0.18],  // "all turn and look one way"
      [at(0, 0.7),    40,    0.0,  1.0, 0.30, -9,    8.2,   -6,    -3.0,  -60,    0.8, 0.0, 0.0, 0.0,  0.00, -0.24],   // "their back on the land"
      [at(0, 1.0),    31,    0.0,  1.0, 0.30, -12,   5.4,   -8,    -1.2,  -60,    1.0, 0.0, 0.0, 0.0,  0.00, -0.26],
      [at(1, 0.2),    21,    0.0,  1.0, 0.30, -22,   1.6,   29,    1.1,   -119,   1.0, 0.06, 0.0, 0.0, -0.24, -0.26],  // behind the row: the ship
      [at(1, 0.45),   20,    0.0,  1.0, 0.30, -21,   1.6,   30,    1.1,   -119,   1.0, 0.33, 0.0, 0.0, -0.24, -0.26],  // "keeps raising its hull"
      [at(1, 0.56),   21,    0.0,  1.0, 0.28, -6,    3.4,   24,    0.4,   -24,    1.0, 0.39, 0.0, 0.0, -0.30, -0.22],
      [at(1, 0.68),   6.4,   0.0,  1.0, 0.25, 12.3,  0.45,  18.8,  0.1,   -5,     1.0, 0.45, 0.0, 0.0, -0.36, -0.20],  // down on the wet sand
      [at(1, 1.0),    6.0,   0.0,  1.0, 0.25, 12.6,  0.42,  19.2,  0.08,  -5.4,   1.0, 0.55, 0.0, 0.0, -0.38, -0.20],  // "reflects a standing gull"
      [at(2, 0.06),   24,    0.0,  1.0, 0.30, 6,     5.0,   -20,   0.6,   -40,    1.0, 0.59, 0.0, 0.0,  0.00, -0.20],
      [at(2, 0.17),   21,    0.0,  1.0, 0.35, -45,   4.2,   20,    0.6,   -10,    1.0, 0.63, 0.0, 0.08,  0.00, -0.20],
      [at(2, 0.3),    16,    0.0,  1.0, 0.40, -86,   3.0,   10,    0.6,   -8,     1.0, 0.7, 0.0, 0.22, -0.12, -0.22],
      [at(2, 0.55),   15.5,  0.0,  1.0, 0.40, -83,   3.0,   10,    0.6,   -8,     1.0, 0.78, 0.0, 0.55, -0.12, -0.22],  // "the land may vary more"
      [at(2, 0.75),   14.5,  0.0,  1.0, 0.35, -80,   2.5,   10,    0.3,   -10,    1.0, 0.86, 1.0, 0.8, -0.15, -0.24],  // "the water comes ashore"
      [at(2, 1.0),    22,    0.0,  1.0, 0.30, -50,   2.4,   -20,   0.8,   -60,    1.0, 0.94, 0.1, 1.0,  0.00, -0.20],  // "the people look at the sea"
      [at(3, 0.12),   4,     0.0,  1.4, 0.30, -10,   1.4,   -10,   1.0,   -200,   1.0, 1.0, 0.0, 1.0,  0.00, -0.12],
      [at(3, 0.32),   -32,   0.0,  3.4, 0.30, -10,   0.9,   -10,   0.7,   -200,   1.0, 1.0, 0.0, 1.0,  0.00, -0.05],   // "cannot look out far"
      [at(3, 0.45),   -44,   0.0,  3.2, 0.30, -10,   -2.0,  -10,   -6,    -70,    1.0, 1.0, 0.0, 1.0,  0.00,  0.00],
      [at(3, 0.6),    -52,   0.0,  2.0, 0.30, -10,   -5.0,  -10,   -12,   -82,    1.0, 1.0, 0.0, 1.0,  0.00,  0.00],   // "cannot look in deep"
      [at(3, 0.78),   -24,   0.0,  1.2, 0.30, -12,   5.0,   -10,   -1.0,  -60,    1.0, 1.0, 0.0, 1.0,  0.00, -0.05],
      [at(3, 1.0),    30,    0.05, 1.0, 0.30, -24,   13,    -6,    -1.0,  -40,    1.0, 1.0, 0.0, 1.0,  0.00, -0.05],   // "any watch they keep"
      [T.total,       80,    0.25, 1.0, 0.30, -48,   28,    4,     -4,    -50,    1.0, 1.0, 0.0, 1.0,  0.00,  0.00]
    ];
  },
  sound: {
    src: '/audio/ocean.mp3',
    label: 'Play the sea',
    // Loudest down at the water, softer up on the dunes, muffled under it.
    volume: function (row) { return row[5] < -0.2 ? 0.06 : 0.12 + 0.3 * (1 - smooth(1, 20, row[5])) + 0.2 * row[11]; },
    cues: [{ stanza: 1, at: 1.0, play: gullCry }, { stanza: 2, at: 1.0, play: wash }]
  }
});
