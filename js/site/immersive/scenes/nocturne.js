/*
 * Scene for "Nocturne Varial" (Lewis Alexander): standing on a ledge above
 * a sleeping valley, a lake below and a river winding away, at night.
 *
 * I   "I came as a shadow": the night is deep and a darker mass of shadow
 *     hangs over the valley, blotting out the stars. "I stand now a light":
 *     a point kindles at its heart and grows; "the depth of my darkness
 *     transfigures your night": the shadow is lit from within, silvers
 *     and melts away, and the black night turns a deep, clear blue.
 * II  The light rises into the sky and spreads out sideways into the five
 *     lines of a musical staff arching across the night, with a treble
 *     clef written in light at its head. "My soul is a nocturne, each note
 *     is a star": stars appear one by one on the staff as notes, each with
 *     a faint stem, a thread of light joining them into a constellation
 *     melody, and each sounds a soft bell-piano note. "So look where you
 *     are": the view drops to the lake, which holds the whole score.
 * III "The radiance is soothing, there's warmth in the light": the staff and
 *     its stars warm to gold, cottage windows wake along the shore and a
 *     warm haze fills the valley. "I came as a shadow" hushes it for a
 *     breath, then "to dazzle your night!": the sky blooms with hundreds of
 *     new stars, most of them landing on the staff as notes, in a cascade
 *     of bells.
 *
 * The lake reflects the sky score through a stencil: the water marks its
 * pixels, then mirrored copies of the staff, stars and light draw only
 * there. Keyframes are a function of the panels so each note lands on its
 * line, and its time is shared with its sound cue. Columns:
 *   [unit, dolly, dark, (unused), wind, yaw, pitch, shadow, kindle, rise,
 *    staff, warm, bloom, phoneAz, phonePitch]
 * where "staff" writes the staff out from the light (lines, then the
 * clef), and a portrait screen, which sees only a narrow slice of the sky
 * and keeps its text mid-screen, turns to "phoneAz" (degrees) and tilts to
 * "phonePitch" to keep the subject above or below the verse.
 */
import { THREE, isSmall, fitCamera, tinted, merge, softSprite, skyDome, starField, terrain,
         scatter, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

var DEG = Math.PI / 180;
var TL = null;              // the engine's timeline, kept from keys() for the note times

// ── The score in the sky (degrees: azimuth to the right, elevation up) ──
var R = 600;                // radius the sky score is drawn at
var AZ0 = -34, AZ1 = 70;    // where the staff starts and runs out of sight
var SP = 2.2;               // degrees between staff lines
var LIGHT = { az: 12, el: 7 };
var SL = (LIGHT.az - AZ0) / (AZ1 - AZ0);   // where the light meets the staff
function arch(s) { return 9 + 12 * Math.sin(Math.PI * (0.15 + 0.85 * s)); }
function staffAz(s) { return AZ0 + s * (AZ1 - AZ0); }
function skyDir(az, el, out) {
  var a = az * DEG, e = el * DEG, c = Math.cos(e);
  return out.set(Math.sin(a) * c, Math.sin(e), -Math.cos(a) * c);
}

// The melody: [frequency, staff position in lines above the bottom line E4].
var MELODY = [[392.00, 1], [523.25, 2.5], [659.26, 3.5], [587.33, 3], [523.25, 2.5],
              [493.88, 2], [523.25, 2.5], [659.26, 3.5], [783.99, 4.5]];
var NOTE_S0 = 0.13, NOTE_DS = 0.059;
function noteAt(k) { return 0.25 + k * 0.075; }          // after the start of panel II

// A treble clef in staff units (x right, y in lines; it curls round the G line).
var CLEF = [[0.3, 1.3], [0.05, 1.5], [-0.3, 1.3], [-0.35, 0.8], [0.0, 0.3], [0.55, 0.22], [1.0, 0.8], [0.98, 1.6],
            [0.55, 2.1], [-0.15, 2.5], [-0.6, 3.1], [-0.65, 3.9], [-0.35, 4.8], [0.05, 5.5], [0.3, 5.75], [0.42, 5.35],
            [0.32, 4.7], [0.05, 4.0], [-0.12, 3.0], [-0.05, 1.6], [0.05, 0.2], [0.12, -0.9], [0.0, -1.55], [-0.3, -1.75],
            [-0.55, -1.55], [-0.5, -1.2]];
var CLEF_S = 0.035;

// ── The valley (metres; you stand at z = 22 looking north, -z) ───────────
var LAKE = { x: 6, z: -92, rx: 105, rz: 70 };
var CAM_Z = 22;
function riverX(z) { return 20 + 40 * Math.sin(-(z + 160) * 0.012); }
function ground(x, z) {
  var h = 1.4 + 0.6 * Math.sin(x * 0.05) * Math.cos(z * 0.04);
  var d = Math.abs(x - riverX(clamp(z, -640, -160)));
  h += smooth(90, 210, d) * 9 + smooth(150, 430, d) * (70 + 35 * Math.sin(z * 0.007 + x * 0.004));
  h += smooth(-700, -1150, z) * (90 + 40 * Math.sin(x * 0.006 + 1));
  var e = Math.hypot((x - LAKE.x) / LAKE.rx, (z - LAKE.z) / LAKE.rz) *
          (1 + 0.05 * Math.sin(Math.atan2(z - LAKE.z, x - LAKE.x) * 3 + 1));
  h = lerp(-3, h, smooth(0.86, 1.08, e));
  // The river winds in from the far hills, where it rises, to the lake.
  if (z < -110 && z > -700) {
    var w = 9 + (-z - 110) * 0.012, dr = Math.abs(x - riverX(z));
    h = lerp(-2, h, clamp(smooth(w * 0.6, w * 1.4, dr) + (1 - smooth(-110, -150, z)) + smooth(-560, -680, z), 0, 1));
  }
  return h;
}
// The lip of the bluff you stand on, a promontory over the lake.
function ledge(x, z) { var t = smooth(6, 19, z + 0.003 * x * x); return t * (30 + 0.3 * Math.sin(x * 0.7) * Math.cos(z * 0.9)); }

// ── Shaders ──────────────────────────────────────────────────────────────
var NOISE =
  'float hash2(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }\n' +
  'float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);\n' +
  '  return mix(mix(hash2(i), hash2(i + vec2(1.0, 0.0)), f.x), mix(hash2(i + vec2(0.0, 1.0)), hash2(i + vec2(1.0, 1.0)), f.x), f.y); }\n' +
  'float fbm(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 5; i++) { s += a * vnoise(p); p = p * 2.03 + vec2(1.7, 9.2); a *= 0.5; } return s; }\n';

// Glowing strokes (staff lines, clef, stems): a thin core and a soft glow
// across the strip; aInfo = (across, gate, fade). A stroke shows where the
// gate has passed, with a bright pen-tip while it is being written.
var STROKE_VS = 'attribute vec3 aInfo; uniform float uGate; uniform float uSoft; uniform float uMirror; uniform float uTime;\n' +
  'varying float vV; varying float vA; varying float vHead;\n' +
  'void main(){ vV = aInfo.x; float d = uGate - aInfo.y;\n' +
  ' vA = clamp(d / uSoft, 0.0, 1.0) * aInfo.z;\n' +
  ' vHead = d > 0.0 ? exp(-d / (uSoft * 1.5)) * step(uGate, 0.995) : 0.0;\n' +
  ' vec3 p = position; p.x += uMirror * sin(uTime * 1.3 + position.y * 0.2) * 0.8;\n' +
  ' gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0); }';
var STROKE_FS = 'uniform vec3 uColor; uniform float uAlpha; uniform float uMirror; uniform float uCore;\n' +
  'varying float vV; varying float vA; varying float vHead;\n' +
  'void main(){ float core = exp(-vV * vV * 26.0) * uCore, glow = exp(-vV * vV * 3.2) * 0.2;\n' +
  ' float a = (core + glow) * vA * uAlpha * (1.0 + vHead * 3.0) * (1.0 - uMirror * 0.3);\n' +
  ' gl_FragColor = vec4(uColor * a, a);\n #include <colorspace_fragment>\n }';

// Stars that appear at a moment (aBorn = the clock time, -1 while hidden)
// and flash as they do; aStar = (size, seed, warm bias, bloom star).
// In the water they stretch into vertical glints.
var STAR_VS = 'attribute vec4 aStar; attribute float aBorn; uniform float uTime; uniform float uScale; uniform float uWarm;\n' +
  'uniform float uBloom; uniform float uAlpha;\n' +
  'varying float vA; varying float vFlash; varying float vWarm;\n' +
  'void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;\n' +
  ' float on = step(0.0, aBorn), life = uTime - aBorn;\n' +
  ' vFlash = on * exp(-life * 1.3);\n' +
  ' float tw = 0.78 + 0.22 * sin(uTime * (1.2 + aStar.y * 2.2) + aStar.y * 50.0);\n' +
  ' vA = on * smoothstep(0.0, 0.1, life) * tw * uAlpha;\n' +
  ' vWarm = clamp(uWarm * (0.65 + 0.35 * aStar.z) + 0.12 * aStar.z, 0.0, 1.0);\n' +
  ' gl_PointSize = aStar.x * uScale * (1.0 + vFlash * 1.4) * (1.0 + uBloom * 0.3 * (1.0 - aStar.w)); }';
var STAR_FS = 'uniform float uMirror; varying float vA; varying float vFlash; varying float vWarm;\n' +
  'void main(){ vec2 c = gl_PointCoord - 0.5; if (uMirror > 0.5) c *= vec2(1.7, 0.6);\n' +
  ' float d = length(c);\n' +
  ' float core = exp(-d * d * 170.0), halo = exp(-d * d * 24.0) * 0.4;\n' +
  ' float spikes = exp(-abs(c.x) * 60.0) * exp(-abs(c.y) * 7.0) + exp(-abs(c.y) * 60.0) * exp(-abs(c.x) * 7.0);\n' +
  ' float a = (core + halo + spikes * (0.3 + vFlash * 0.7)) * vA * (1.0 - uMirror * 0.25);\n' +
  ' a *= 1.0 - smoothstep(0.38, 0.5, max(abs(c.x), abs(c.y)));\n' +
  ' vec3 col = mix(vec3(0.9, 0.94, 1.0), vec3(1.0, 0.78, 0.46), vWarm) * (1.0 + vFlash * 1.2);\n' +
  ' gl_FragColor = vec4(col * a, a);\n #include <colorspace_fragment>\n }';

// The light: a core that grows from a pin-point, a halo and slow rays.
var LIGHT_FS = 'uniform float uK; uniform float uTime; uniform float uAlpha; uniform float uMirror; uniform vec3 uCol; varying vec2 vUv;\n' +
  'void main(){ vec2 c = vUv - 0.5; if (uMirror > 0.5) c *= vec2(1.5, 0.7); float d = length(c), ang = atan(c.y, c.x);\n' +
  ' float core = exp(-d * d / (0.00003 + 0.00026 * uK));\n' +
  ' float halo = exp(-d * d / (0.0006 + 0.006 * uK)) * 0.55 + exp(-d * 10.0) * 0.12 * uK;\n' +
  ' float rays = pow(abs(sin(ang * 4.0 + uTime * 0.04)), 30.0) + 0.6 * pow(abs(sin(ang * 7.0 - uTime * 0.03 + 1.0)), 40.0);\n' +
  ' rays *= exp(-d * (16.0 - 8.0 * uK)) * 0.45 * uK;\n' +
  ' float a = (core * 1.3 + halo + rays) * uAlpha * (1.0 - smoothstep(0.32, 0.5, d)) * (1.0 - uMirror * 0.4);\n' +
  ' vec3 col = mix(uCol, vec3(1.0), clamp(core, 0.0, 1.0));\n' +
  ' gl_FragColor = vec4(col * a, a);\n #include <colorspace_fragment>\n }';

// The shadow: a ragged, slowly billowing mass darker than the night, with a
// faint rim. Lit from within as the light kindles, it silvers and thins.
var SHADOW_FS = 'uniform float uShadow; uniform float uK; uniform float uTime; varying vec2 vUv;\n' + NOISE +
  'void main(){ vec2 p = (vUv - 0.5) * vec2(2.6, 1.5); float d = length(p);\n' +
  ' float edge = smoothstep(0.0, 0.14, vUv.x) * smoothstep(1.0, 0.86, vUv.x) * smoothstep(0.0, 0.18, vUv.y) * smoothstep(1.0, 0.82, vUv.y);\n' +
  ' float n = fbm(p * 3.0 + vec2(uTime * 0.018, -uTime * 0.012));\n' +
  ' float n2 = fbm(p * 5.5 - vec2(uTime * 0.025, 0.0) + n * 1.5);\n' +
  ' float shape = smoothstep(0.55, 0.1, d + (n2 - 0.5) * 0.5) * edge;\n' +
  ' float dens = shape * (0.7 + 0.4 * n);\n' +
  ' float inner = exp(-d * d * 7.0) * uK * uK;\n' +
  ' float rim = smoothstep(0.0, 0.12, shape) * (1.0 - smoothstep(0.12, 0.4, shape));\n' +
  ' vec3 lit = mix(vec3(0.5, 0.56, 0.85), vec3(1.0, 0.86, 0.62), inner);\n' +
  ' vec3 col = vec3(0.0, 0.0005, 0.002) + lit * (inner * dens * 0.8 + rim * n2 * (0.02 + 0.35 * uK));\n' +
  ' float a = clamp(dens * 1.4, 0.0, 0.98) * uShadow * (1.0 - 0.5 * inner);\n' +
  ' gl_FragColor = vec4(col, a);\n #include <colorspace_fragment>\n }';

var PLANE_VS = 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }';

// The lake: dark water holding the horizon's glow at a grazing angle, a
// faint breathing sheen, and the valley's warmth. It also marks the stencil
// so the mirrored sky draws only on the water.
var WATER_VS = 'varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }';
var WATER_FS = 'uniform vec3 uHorizon; uniform vec3 uDeep; uniform vec3 uWarmC; uniform vec3 uFogColor;\n' +
  'uniform float uWarm; uniform float uTime; uniform float uFogDensity; varying vec3 vW;\n' + NOISE +
  'void main(){ vec3 toCam = cameraPosition - vW; float dist = length(toCam); vec3 V = toCam / dist;\n' +
  ' float fres = pow(1.0 - max(V.y, 0.0), 5.0);\n' +
  ' float n = vnoise(vW.xz * vec2(0.03, 0.12) + vec2(uTime * 0.05, uTime * 0.02));\n' +
  ' vec3 col = mix(uDeep, uHorizon, 0.02 + 0.4 * fres) * (0.85 + 0.3 * n);\n' +
  ' col += uWarmC * uWarm * (0.006 + 0.04 * fres);\n' +
  ' float fogF = 1.0 - exp(-uFogDensity * uFogDensity * dist * dist);\n' +
  ' gl_FragColor = vec4(mix(col, uFogColor, fogF), 1.0);\n' +
  ' #include <tonemapping_fragment>\n #include <colorspace_fragment>\n }';

// Cottage windows that wake one by one as the valley warms.
var WINDOW_VS = 'attribute float aThr; uniform float uWarm; uniform float uScale; varying float vA;\n' +
  'void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;\n' +
  ' vA = smoothstep(aThr, aThr + 0.12, uWarm);\n' +
  ' gl_PointSize = uScale * clamp(1700.0 / -mv.z, 3.0, 16.0); }';
var WINDOW_FS = 'varying float vA;\n' +
  'void main(){ float d = length(gl_PointCoord - 0.5); float a = (exp(-d * d * 60.0) + exp(-d * d * 10.0) * 0.3) * vA;\n' +
  ' gl_FragColor = vec4(vec3(1.0, 0.72, 0.38) * a, a);\n #include <colorspace_fragment>\n }';

// ── Geometry helpers ────────────────────────────────────────────────────
// Strokes in sky degrees: polys are { pts: [[az, el]], gate: [], fade: [], w }
// with w the half-width in degrees; the strip is placed on the sphere R.
function strokeGeometry(polys) {
  var pos = [], info = [], idx = [], v = new THREE.Vector3(), base = 0;
  polys.forEach(function (p) {
    var pts = p.pts, n = pts.length;
    for (var i = 0; i < n; i++) {
      var a = pts[Math.max(i - 1, 0)], b = pts[Math.min(i + 1, n - 1)];
      var tx = (b[0] - a[0]) * Math.cos(pts[i][1] * DEG), ty = b[1] - a[1], l = Math.hypot(tx, ty) || 1;
      var nx = -ty / l * p.w / Math.cos(pts[i][1] * DEG), ny = tx / l * p.w;
      var g = p.gate[i], fd = p.fade ? p.fade[i] : 1;
      skyDir(pts[i][0] + nx, pts[i][1] + ny, v).multiplyScalar(R);
      pos.push(v.x, v.y, v.z); info.push(1, g, fd);
      skyDir(pts[i][0] - nx, pts[i][1] - ny, v).multiplyScalar(R);
      pos.push(v.x, v.y, v.z); info.push(-1, g, fd);
      if (i < n - 1) { var q = base + i * 2; idx.push(q, q + 1, q + 2, q + 1, q + 3, q + 2); }
    }
    base += n * 2;
  });
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('aInfo', new THREE.Float32BufferAttribute(info, 3));
  geo.setIndex(idx);
  return geo;
}

// A Catmull-Rom pass through a list of 2D points.
function smoothPath(P, per) {
  var out = [];
  for (var i = 0; i < P.length - 1; i++) {
    var p0 = P[Math.max(i - 1, 0)], p1 = P[i], p2 = P[i + 1], p3 = P[Math.min(i + 2, P.length - 1)];
    for (var k = 0; k < per; k++) {
      var t = k / per, t2 = t * t, t3 = t2 * t, o = [];
      for (var j = 0; j < 2; j++) {
        o.push(0.5 * (2 * p1[j] + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2 +
                      (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3));
      }
      out.push(o);
    }
  }
  out.push(P[P.length - 1].slice());
  return out;
}

// A material and its mirror twin, sharing uniforms; the twin draws only
// where the water marked the stencil, flipped, dimmer, glinting.
function onWater(mir) {
  mir.depthTest = false;
  mir.stencilWrite = true;
  mir.stencilRef = 1;
  mir.stencilFunc = THREE.EqualStencilFunc;
  mir.stencilFail = mir.stencilZFail = mir.stencilZPass = THREE.KeepStencilOp;
  return mir;
}
function twin(o) {
  var mat = new THREE.ShaderMaterial(Object.assign({ transparent: true, depthWrite: false, fog: false,
                                                     side: THREE.DoubleSide }, o));
  var u2 = Object.assign({}, o.uniforms, { uMirror: { value: 1 } });
  var mir = new THREE.ShaderMaterial(Object.assign({ transparent: true, depthWrite: false, fog: false,
                                                     side: THREE.DoubleSide }, o, { uniforms: u2 }));
  return [mat, onWater(mir)];
}

// ── Synthesised sound ────────────────────────────────────────────────────
// A soft bell-piano: a few harmonics and one bright inharmonic tine, fast
// attack and a long fall, sent dry and through a dark echo for space.
var ECHO = new WeakMap();
function echoIn(ac, out) {
  var e = ECHO.get(ac);
  if (e) return e;
  var input = ac.createGain(), delay = ac.createDelay(1), fb = ac.createGain(), lp = ac.createBiquadFilter(), wet = ac.createGain();
  delay.delayTime.value = 0.37;
  fb.gain.value = 0.38;
  lp.type = 'lowpass';
  lp.frequency.value = 2400;
  wet.gain.value = 0.45;
  input.connect(delay); delay.connect(lp); lp.connect(fb); fb.connect(delay); lp.connect(wet); wet.connect(out);
  ECHO.set(ac, input);
  return input;
}
function tone(ac, out, freq, t, gain) {
  var echo = echoIn(ac, out);
  [[1, 1, 3.2], [2, 0.32, 1.8], [3, 0.12, 1.1], [4.2, 0.06, 0.45], [1.003, 0.4, 2.6]].forEach(function (p) {
    var o = ac.createOscillator(), g = ac.createGain();
    o.type = 'sine';
    o.frequency.value = freq * p[0];
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain * p[1], t + 0.008);
    g.gain.exponentialRampToValueAtTime(0.0001, t + p[2]);
    o.connect(g); g.connect(out); g.connect(echo);
    o.start(t);
    o.stop(t + p[2] + 0.05);
  });
}
function note(freq, gain) {
  return function (ac, out) { tone(ac, out, freq, ac.currentTime + 0.01, gain); };
}
// "I stand now a light": a low open fifth.
function kindleSound(ac, out) {
  var t = ac.currentTime + 0.01;
  tone(ac, out, 130.81, t, 0.12);
  tone(ac, out, 196.00, t + 0.25, 0.08);
}
// "To dazzle your night!": a long cascade up and down a major chord over a
// low C, then a scatter of high glints.
function dazzleSound(ac, out) {
  var t = ac.currentTime + 0.01, ch = [261.63, 329.63, 392.00, 523.25, 659.26, 783.99, 1046.5, 1318.5, 1568.0, 2093.0];
  tone(ac, out, 65.41, t, 0.12);
  tone(ac, out, 130.81, t, 0.1);
  for (var i = 0; i < ch.length; i++) tone(ac, out, ch[i], t + i * 0.07, 0.07);
  for (i = 0; i < 14; i++) tone(ac, out, ch[4 + Math.floor(Math.random() * 6)], t + 0.8 + i * 0.13 + Math.random() * 0.08, 0.035);
}

// ── Renderer ─────────────────────────────────────────────────────────────
function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(1127), i, k;
  // kit's renderer, plus a stencil buffer for the lake's reflection.
  var gl = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, stencil: true, powerPreference: 'high-performance' });
  gl.setClearColor('#020309');
  gl.toneMapping = THREE.ACESFilmicToneMapping;
  gl.outputColorSpace = THREE.SRGBColorSpace;

  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#0b1020', 0.0016);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.1, 3200);
  var eyeY = ledge(0, CAM_Z) + 1.7;

  var T = TL || { start: function (n) { return 1 + 1.6 * n; } };
  var noteU = MELODY.map(function (m, n) { return T.start(1) + noteAt(n); });

  // The sky score is drawn round the camera and mirrored about the water
  // from there; things in the valley are mirrored in place.
  var sky = new THREE.Group(), mirror = new THREE.Group(), wmirror = new THREE.Group();
  mirror.scale.y = wmirror.scale.y = -1;
  world.add(sky, mirror, wmirror);
  var dome = skyDome({ top: '#03050f', mid: '#0a1230', horizon: '#1c2a52' }, 1500);
  sky.add(dome.mesh);
  var bgStars = starField(r, small ? 1400 : 2600, 1300, 0.03, 1.3);
  sky.add(bgStars);

  // ── The shadow, and the light that kindles in it ──
  var tmpV = new THREE.Vector3(), tmpM = new THREE.Matrix4(), ORIGIN = new THREE.Vector3(), UP = new THREE.Vector3(0, 1, 0);
  function faceIn(obj) { tmpM.lookAt(ORIGIN, obj.position, UP); obj.quaternion.setFromRotationMatrix(tmpM); }

  // A faint cold glow low in the sky that the shadow stands against.
  var backGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(70,90,170,0.9)', 'rgba(40,50,120,0)'),
    blending: THREE.AdditiveBlending, transparent: true, depthWrite: false, fog: false, opacity: 0 }));
  skyDir(LIGHT.az, LIGHT.el + 2, backGlow.position).multiplyScalar(900);
  backGlow.scale.set(1500, 620, 1);
  sky.add(backGlow);

  var shadowU = { uShadow: { value: 1 }, uK: { value: 0 }, uTime: { value: 0 } };
  var shadowMat = new THREE.ShaderMaterial({ uniforms: shadowU, vertexShader: PLANE_VS, fragmentShader: SHADOW_FS,
                                             transparent: true, depthWrite: false, fog: false });
  var shadow = new THREE.Mesh(new THREE.PlaneGeometry(520, 300), shadowMat);
  shadow.renderOrder = 1;
  sky.add(shadow);

  var lightU = { uK: { value: 0 }, uTime: { value: 0 }, uAlpha: { value: 0 }, uCol: { value: new THREE.Color('#ffe2b0') },
                 uMirror: { value: 0 } };
  var lightMats = twin({ uniforms: lightU, vertexShader: PLANE_VS, fragmentShader: LIGHT_FS, blending: THREE.AdditiveBlending });
  var lightGeo = new THREE.PlaneGeometry(320, 320);
  var light = new THREE.Mesh(lightGeo, lightMats[0]), lightM = new THREE.Mesh(lightGeo, lightMats[1]);
  light.renderOrder = 2;
  sky.add(light);
  mirror.add(lightM);

  // ── The staff, written outwards from the light, then the clef ──
  var polys = [], ln, s, pts, gate, fade;
  for (ln = 0; ln < 5; ln++) {
    pts = []; gate = []; fade = [];
    for (k = 0; k <= 220; k++) {
      s = k / 220;
      pts.push([staffAz(s), arch(s) + ln * SP]);
      gate.push(Math.abs(s - SL) / Math.max(SL, 1 - SL) * 0.8);
      fade.push(smooth(0, 0.012, s) * (1 - smooth(0.8, 1, s)) * 0.8);
    }
    polys.push({ pts: pts, gate: gate, fade: fade, w: 0.32 });
  }
  var clefPts = smoothPath(CLEF, 10), clefLen = clefPts.length;
  var sPerUnit = SP / (AZ1 - AZ0);
  polys.push({ w: 0.42, pts: clefPts.map(function (p) {
    var cs = CLEF_S + p[0] * sPerUnit * 1.1;
    return [staffAz(cs), arch(cs) + p[1] * SP];
  }), gate: clefPts.map(function (p, n) { return 0.8 + 0.2 * n / (clefLen - 1); }), fade: null });
  // A soft aura along the staff for the finale.
  var auraPts = [], auraGate = [], auraFade = [];
  for (k = 0; k <= 120; k++) {
    s = k / 120;
    auraPts.push([staffAz(s), arch(s) + 2 * SP]);
    auraGate.push(0);
    auraFade.push(smooth(0, 0.08, s) * (1 - smooth(0.75, 1, s)));
  }
  var auraU = { uGate: { value: 1 }, uSoft: { value: 0.05 }, uTime: { value: 0 }, uAlpha: { value: 0 },
                uColor: { value: new THREE.Color('#ffb860') }, uMirror: { value: 0 }, uCore: { value: 0 } };
  var auraMats = twin({ uniforms: auraU, vertexShader: STROKE_VS, fragmentShader: STROKE_FS, blending: THREE.AdditiveBlending });
  var auraGeo = strokeGeometry([{ pts: auraPts, gate: auraGate, fade: auraFade, w: SP * 4.5 }]);
  var aura = new THREE.Mesh(auraGeo, auraMats[0]), auraM = new THREE.Mesh(auraGeo, auraMats[1]);
  auraM.renderOrder = 6;
  sky.add(aura);
  mirror.add(auraM);

  var staffU = { uGate: { value: 0 }, uSoft: { value: 0.05 }, uTime: { value: 0 }, uAlpha: { value: 1 },
                 uColor: { value: new THREE.Color('#9fb6ff') }, uMirror: { value: 0 }, uCore: { value: 1 } };
  var staffMats = twin({ uniforms: staffU, vertexShader: STROKE_VS, fragmentShader: STROKE_FS, blending: THREE.AdditiveBlending });
  var staffGeo = strokeGeometry(polys);
  sky.add(new THREE.Mesh(staffGeo, staffMats[0]));
  var staffM = new THREE.Mesh(staffGeo, staffMats[1]);
  staffM.renderOrder = 6;
  mirror.add(staffM);

  // ── The notes: stars on the staff, with stems and a thread between them ──
  var notePos = [], noteAE = [];
  MELODY.forEach(function (m, n) {
    var ns = NOTE_S0 + n * NOTE_DS, ae = [staffAz(ns), arch(ns) + m[1] * SP];
    noteAE.push(ae);
    skyDir(ae[0], ae[1], tmpV).multiplyScalar(R - 2);
    notePos.push(tmpV.x, tmpV.y, tmpV.z);
  });
  var stems = [];
  MELODY.forEach(function (m, n) {
    var ae = noteAE[n], up = m[1] < 2, dx = (up ? 0.55 : -0.55) * SP * 0.5, sp = [], sg = [];
    for (k = 0; k <= 6; k++) {
      sp.push([ae[0] + dx, ae[1] + (up ? 1 : -1) * SP * (0.25 + 2.5 * k / 6)]);
      sg.push(noteU[n] + 0.015 + 0.05 * k / 6);
    }
    stems.push({ pts: sp, gate: sg, fade: null, w: 0.26 });
    if (n === 0) return;
    var a = noteAE[n - 1], tp = [], tg = [], tf = [];
    for (k = 0; k <= 16; k++) {
      var t = k / 16;
      tp.push([lerp(a[0], ae[0], t), lerp(a[1], ae[1], t)]);
      tg.push(lerp(noteU[n - 1] + 0.02, noteU[n], t));
      tf.push(0.5 * smooth(0, 0.15, t) * (1 - smooth(0.85, 1, t)));
    }
    stems.push({ pts: tp, gate: tg, fade: tf, w: 0.2 });
  });
  var stemU = Object.assign({}, staffU, { uGate: { value: 0 }, uSoft: { value: 0.03 }, uAlpha: { value: 0.42 } });
  var stemMats = twin({ uniforms: stemU, vertexShader: STROKE_VS, fragmentShader: STROKE_FS, blending: THREE.AdditiveBlending });
  var stemGeo = strokeGeometry(stems);
  sky.add(new THREE.Mesh(stemGeo, stemMats[0]));
  var stemM = new THREE.Mesh(stemGeo, stemMats[1]);
  stemM.renderOrder = 6;
  mirror.add(stemM);

  // Notes and the bloom share one point field: the first entries are the
  // melody; the rest are the finale's stars, most landing on the staff.
  var NB = small ? 650 : 1400, NN = MELODY.length, NT = NN + NB;
  var sPos = new Float32Array(NT * 3), sAttr = new Float32Array(NT * 4), born = new Float32Array(NT), thr = new Float32Array(NT);
  for (i = 0; i < NN; i++) {
    sPos.set(notePos.slice(i * 3, i * 3 + 3), i * 3);
    sAttr.set([58, r(), 0.5, 0], i * 4);
    born[i] = -1;
  }
  for (i = NN; i < NT; i++) {
    var az, el, onStaff = r() < 0.6;
    if (onStaff) {
      s = 0.04 + r() * 0.96;
      az = staffAz(s);
      el = arch(s) + Math.round(r() * 13 - 3) * 0.5 * SP;
      thr[i] = 0.04 + 0.7 * (0.65 * s + 0.35 * r());
    } else {
      az = -85 + r() * 190;
      el = 4 + Math.pow(r(), 1.4) * 60;
      thr[i] = 0.1 + 0.8 * r();
    }
    skyDir(az, el, tmpV).multiplyScalar(R + 20);
    sPos.set([tmpV.x, tmpV.y, tmpV.z], i * 3);
    sAttr.set([(onStaff ? 10 : 6) + Math.pow(r(), 2.5) * 30, r(), r(), 1], i * 4);
    born[i] = -1;
  }
  var starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.BufferAttribute(sPos, 3));
  starGeo.setAttribute('aStar', new THREE.BufferAttribute(sAttr, 4));
  var bornAttr = new THREE.BufferAttribute(born, 1);
  starGeo.setAttribute('aBorn', bornAttr);
  var starU = { uTime: { value: 0 }, uScale: { value: 1 }, uWarm: { value: 0 }, uBloom: { value: 0 }, uAlpha: { value: 1 },
               uMirror: { value: 0 } };
  var starMats = twin({ uniforms: starU, vertexShader: STAR_VS, fragmentShader: STAR_FS, blending: THREE.AdditiveBlending });
  var stars = new THREE.Points(starGeo, starMats[0]), starsM = new THREE.Points(starGeo, starMats[1]);
  stars.frustumCulled = starsM.frustumCulled = false;
  starsM.renderOrder = 6;
  sky.add(stars);
  mirror.add(starsM);

  // ── The valley ──
  var hemi = new THREE.HemisphereLight('#5f70a8', '#0a0c10', 0.3);
  var glowLight = new THREE.DirectionalLight('#cfd8ff', 0);
  skyDir(LIGHT.az, LIGHT.el + 10, glowLight.position);
  var warmLight = new THREE.DirectionalLight('#ffbe7a', 0);
  skyDir(5, 24, warmLight.position);
  world.add(hemi, glowLight, warmLight);

  var cTmp = new THREE.Color(), cA = new THREE.Color('#1a2519'), cB = new THREE.Color('#1c2232'), cBed = new THREE.Color('#0b0d10');
  world.add(terrain(1700, small ? 150 : 230, 0, -620, ground,
    new THREE.MeshLambertMaterial({ vertexColors: true }), function (x, z, y) {
      cTmp.copy(cA).lerp(cB, smooth(10, 90, y));
      return y < 0.3 ? cTmp.lerp(cBed, smooth(0.3, -1.5, y)) : cTmp;
    }));

  // The bluff underfoot, finer than the valley so its lip stays smooth;
  // away from the bluff it sinks under the valley floor.
  world.add(terrain(130, small ? 110 : 170, 0, 5, function (x, z) {
    var l = ledge(x, z), g = ground(x, z);
    return l > g + 0.4 && l > 2 ? l : g - 0.6;
  }, new THREE.MeshLambertMaterial({ color: '#141c14' })));

  var waterU = { uHorizon: { value: new THREE.Color() }, uDeep: { value: new THREE.Color('#020409') },
                 uWarmC: { value: new THREE.Color('#ff9a4a') }, uFogColor: { value: new THREE.Color() },
                 uWarm: { value: 0 }, uTime: { value: 0 }, uFogDensity: { value: 0.0016 } };
  var waterMat = new THREE.ShaderMaterial({ uniforms: waterU, vertexShader: WATER_VS, fragmentShader: WATER_FS, transparent: true });
  waterMat.stencilWrite = true;
  waterMat.stencilRef = 1;
  waterMat.stencilFunc = THREE.AlwaysStencilFunc;
  waterMat.stencilZPass = THREE.ReplaceStencilOp;
  var water = new THREE.Mesh(new THREE.PlaneGeometry(1700, 1700).rotateX(-Math.PI / 2).translate(0, 0, -620), waterMat);
  water.renderOrder = 5;
  world.add(water);

  // Conifers on the hills and shores.
  var conifer = merge([tinted(new THREE.CylinderGeometry(0.18, 0.25, 1.6, 5).translate(0, 0.8, 0), '#3a3028'),
                       tinted(new THREE.ConeGeometry(1.7, 4.6, 6).translate(0, 3.4, 0), '#ffffff'),
                       tinted(new THREE.ConeGeometry(1.2, 3.4, 6).translate(0, 5.6, 0), '#ffffff')]);
  var trees = new THREE.InstancedMesh(conifer, new THREE.MeshLambertMaterial({ vertexColors: true }), small ? 500 : 1100);
  scatter(trees, 20000, function (n, p, q, sc, c) {
    var x = (r() - 0.5) * 1100, z = -16 - r() * 760, y = ground(x, z);
    if (y < 1.6 || (y < 8 && r() < 0.6) || Math.hypot(x, z - CAM_Z) < 25) return false;
    p.set(x, y - 0.2, z);
    q.setFromAxisAngle(UP, r() * 6.28);
    sc.setScalar(1.2 + r() * 1.4);
    c.setHSL(0.36 + r() * 0.06, 0.25, 0.09 + r() * 0.05);
  });
  world.add(trees);

  // Cottages along the far shore and the river, their windows dark.
  var house = merge([tinted(new THREE.BoxGeometry(5, 3.5, 6).translate(0, 1.75, 0), '#4a4440'),
                     tinted(new THREE.CylinderGeometry(2.9, 2.9, 6.4, 3).rotateX(-Math.PI / 2).translate(0, 4.95, 0), '#2e2a2e')]);
  var NH = small ? 30 : 48, houses = new THREE.InstancedMesh(house, new THREE.MeshLambertMaterial({ vertexColors: true }), NH);
  var winPos = [], winThr = [], hq = new THREE.Quaternion();
  scatter(houses, 6000, function (n, p, q, sc, c) {
    var z = -150 - r() * 260, x = (z > -175 ? LAKE.x + (r() - 0.5) * 160 : riverX(z) + (r() < 0.5 ? -1 : 1) * (16 + r() * 45));
    var y = ground(x, z);
    if (y < 1 || y > 12) return false;
    var yaw = (r() - 0.5) * 0.9;
    p.set(x, y - 0.3, z);
    q.setFromAxisAngle(UP, yaw);
    sc.setScalar(0.9 + r() * 0.4);
    c.setHSL(0.08, 0.1, 0.5 + r() * 0.3);
    hq.copy(q);
    var wins = r() < 0.5 ? [-1.2, 1.2] : [r() < 0.5 ? -1.2 : 1.2];
    var t0 = 0.05 + r() * 0.6;
    wins.forEach(function (wx) {
      tmpV.set(wx * sc.x, 1.7 * sc.y, 3.05 * sc.z).applyQuaternion(hq).add(p);
      winPos.push(tmpV.x, tmpV.y, tmpV.z);
      winThr.push(t0 + r() * 0.15);
    });
  });
  world.add(houses);
  var winGeo = new THREE.BufferGeometry();
  winGeo.setAttribute('position', new THREE.Float32BufferAttribute(winPos, 3));
  winGeo.setAttribute('aThr', new THREE.Float32BufferAttribute(winThr, 1));
  var winU = { uWarm: { value: 0 }, uScale: { value: 1 } };
  var winOpts = { uniforms: winU, vertexShader: WINDOW_VS, fragmentShader: WINDOW_FS, transparent: true, depthWrite: false,
                  blending: THREE.AdditiveBlending };
  var windows = new THREE.Points(winGeo, new THREE.ShaderMaterial(winOpts));
  var windowsM = new THREE.Points(winGeo, onWater(new THREE.ShaderMaterial(winOpts)));
  windows.renderOrder = 7;
  windowsM.renderOrder = 6;
  world.add(windows);
  wmirror.add(windowsM);

  // A warm haze lying along the river by the village, and its reflection.
  var hazeTex = softSprite('rgba(255,190,120,0.9)', 'rgba(255,150,80,0)'), haze = [];
  for (i = 0; i < 12; i++) {
    var hz = -165 - i * 32, hx = riverX(hz) + (r() - 0.5) * 30;
    var hs = new THREE.Sprite(new THREE.SpriteMaterial({ map: hazeTex, transparent: true, depthWrite: false, opacity: 0,
                                                         blending: THREE.AdditiveBlending }));
    hs.position.set(hx, 5 + r() * 4, hz);
    hs.scale.set(130 + r() * 60, 20 + r() * 10, 1);
    hs.renderOrder = 7;
    world.add(hs);
    var hm = new THREE.Sprite(onWater(hs.material.clone()));
    hm.position.copy(hs.position);
    hm.scale.copy(hs.scale);
    hm.renderOrder = 6;
    wmirror.add(hm);
    haze.push(hs, hm);
  }

  // Grass along the edge of the ledge, the near foreground.
  var blade = merge([tinted(new THREE.ConeGeometry(0.035, 0.7, 3).translate(0, 0.35, 0), '#ffffff')]);
  var grass = new THREE.InstancedMesh(blade, new THREE.MeshLambertMaterial({ vertexColors: true }), small ? 1200 : 2600);
  scatter(grass, 20000, function (n, p, q, sc, c) {
    var z = 12 + r() * 7, x = (r() - 0.5) * 2 * (0.9 * (CAM_Z - z) + 3), y = ledge(x, z);
    if (Math.hypot(x, z - CAM_Z) < 4 || y < 22) return false;
    p.set(x, y, z);
    q.setFromAxisAngle(UP, r() * 6.28);
    sc.set(0.7, 0.35 + r() * 0.45, 0.7);
    c.setHSL(0.27, 0.3, 0.12 + r() * 0.08);
  });
  world.add(grass);

  // ── Per-frame ──
  var tmp = new THREE.Color(), tmp2 = new THREE.Color(), silver = new THREE.Color('#9fb6ff'), gold = new THREE.Color('#ffc983');
  var portrait = false, H = 800, lastU = -1;

  function frame(f) {
    var row = f.row, time = f.time, dark = row[1], yaw = row[4], pitch = row[5], shadowA = row[6], kindle = row[7];
    var rise = row[8], staffD = row[9], warm = row[10], bloom = row[11], paz = row[12], ppitch = row[13];
    var leap = lastU < 0 || Math.abs(f.u - lastU) > 1;
    lastU = f.u;

    camera.position.set(0, eyeY - f.cam * 1.2, CAM_Z - f.cam * 5);
    camera.rotation.set(0, 0, 0);
    camera.rotateY(-(portrait ? paz * DEG : yaw) - f.mx * 0.1);
    camera.rotateX((portrait ? ppitch : pitch) - f.my * 0.05);
    sky.position.copy(camera.position);
    mirror.position.set(camera.position.x, -camera.position.y, camera.position.z);

    // The night: near-black at first, a clear deep blue once transfigured,
    // warming at the end.
    dome.uniforms.dark.value = dark * 0.45;
    dome.uniforms.mid.value.set('#0a1230').lerp(tmp.set('#1a1530'), warm * 0.8);
    var horizon = tmp2.set('#1c2a52').lerp(tmp.set('#4a3036'), warm * 0.85);
    dome.uniforms.horizon.value.copy(horizon);
    horizon.lerp(tmp.set('#0a0f26'), dark * 0.6);
    world.fog.color.copy(horizon).multiplyScalar(0.7);
    gl.setClearColor(world.fog.color);
    bgStars.material.opacity = 0.3 + (1 - dark) * 0.5 - bloom * 0.15;
    hemi.intensity = 0.12 + (1 - dark) * 0.4 + warm * 0.25;
    hemi.color.set('#5f70a8').lerp(tmp.set('#c99a7a'), warm * 0.7);
    hemi.groundColor.set('#0a0c10').lerp(tmp.set('#3a2216'), warm);

    // The shadow and the light kindling at its heart, rising to the staff.
    var drift = Math.sin(time * 0.05) * 0.6;
    skyDir(LIGHT.az + drift, LIGHT.el - 0.5, shadow.position).multiplyScalar(520);
    faceIn(shadow);
    shadow.visible = shadowA > 0.005;
    shadowU.uShadow.value = shadowA;
    backGlow.material.opacity = shadowA * 0.5;
    shadowU.uK.value = kindle;
    shadowU.uTime.value = env.reduceMotion ? 0 : time;
    var riseE = rise * rise * (3 - 2 * rise);
    skyDir(LIGHT.az, lerp(LIGHT.el, arch(SL) + 2 * SP, riseE), light.position).multiplyScalar(480);
    faceIn(light);
    light.scale.setScalar(1 - riseE * 0.5);
    lightM.position.copy(light.position);
    lightM.quaternion.copy(light.quaternion);
    lightM.scale.copy(light.scale);
    lightU.uK.value = kindle;
    lightU.uTime.value = time;
    lightU.uAlpha.value = smooth(0, 0.06, kindle) * (1 - smooth(0.25, 0.9, staffD));
    light.visible = lightM.visible = lightU.uAlpha.value > 0.002;
    lightU.uCol.value.set('#ffe2b0').lerp(tmp.set('#cfdcff'), riseE * 0.6);
    glowLight.intensity = kindle * 0.9 * (1 - smooth(0.3, 1, staffD)) + staffD * 0.35;

    // The staff, the notes as they sound, and the bloom.
    staffU.uGate.value = staffD;
    staffU.uTime.value = time;
    staffU.uAlpha.value = 0.85 + bloom * 0.5 - (dark - 0.12) * 0.4;
    staffU.uColor.value.copy(silver).lerp(gold, warm);
    stemU.uGate.value = f.u;
    auraU.uAlpha.value = warm * 0.12 + bloom * 0.3;
    aura.visible = auraM.visible = auraU.uAlpha.value > 0.002;
    var changed = false;
    for (i = 0; i < NT; i++) {
      var on = i < NN ? f.u >= noteU[i] : bloom > thr[i];
      if (on && born[i] < 0) { born[i] = leap ? time - 20 : time; changed = true; }
      else if (!on && born[i] >= 0) { born[i] = -1; changed = true; }
    }
    if (changed) bornAttr.needsUpdate = true;
    starU.uTime.value = time;
    starU.uWarm.value = warm;
    starU.uBloom.value = bloom;
    starU.uScale.value = gl.getPixelRatio() * (H / 800 * 0.6 + 0.4);

    // The valley warms.
    warmLight.intensity = warm * 1.1 + bloom * 0.3;
    winU.uWarm.value = warm;
    winU.uScale.value = gl.getPixelRatio();
    for (i = 0; i < haze.length; i++) haze[i].material.opacity = warm * (0.16 + 0.06 * Math.sin(time * 0.3 + (i >> 1))) * (i % 2 ? 0.6 : 1);
    waterU.uHorizon.value.copy(horizon).lerp(tmp.set('#0b0e1c'), warm * 0.55);
    waterU.uFogColor.value.copy(world.fog.color);
    waterU.uWarm.value = warm;
    waterU.uTime.value = time;

    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) { H = h; portrait = w / h < 1; fitCamera(gl, camera, w, h, dpr, small); },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('nocturne', {
  renderer: renderer3d,
  align: ['left', 'right', 'left'],
  keys: function (T) {
    TL = T;
    var s0 = T.start(0), s1 = T.start(1), s2 = T.start(2);
    //  unit        dolly dark  -  wind  yaw    pitch  shadow kindle rise staff warm bloom paz  ppitch
    return [
      [0,           0.00, 1.00, 0, 0.1, 0.00,  0.14, 0.80, 0.00, 0.0, 0.00, 0.0, 0.0, 12, -0.06],
      [0.7,         0.02, 1.00, 0, 0.1, 0.00,  0.14, 0.90, 0.00, 0.0, 0.00, 0.0, 0.0, 12, -0.06],
      [s0 + 0.3,    0.06, 1.00, 0, 0.1, -0.02, 0.13, 1.00, 0.00, 0.0, 0.00, 0.0, 0.0, 12, -0.08],  // "I came as a shadow"
      [s0 + 0.48,   0.08, 0.96, 0, 0.1, -0.02, 0.13, 1.00, 0.04, 0.0, 0.00, 0.0, 0.0, 12, -0.08],
      [s0 + 0.7,    0.10, 0.80, 0, 0.1, -0.03, 0.14, 0.95, 0.55, 0.0, 0.00, 0.0, 0.0, 12, -0.08],  // "I stand now a light"
      [s0 + 1.05,   0.14, 0.30, 0, 0.1, -0.03, 0.16, 0.45, 1.00, 0.0, 0.00, 0.0, 0.0, 12, -0.08],  // "transfigures your night"
      [s0 + 1.2,    0.16, 0.18, 0, 0.1, -0.02, 0.18, 0.15, 1.00, 0.0, 0.00, 0.0, 0.0, 12, -0.06],
      [s1 - 0.05,   0.20, 0.13, 0, 0.1, 0.00,  0.20, 0.00, 1.00, 1.0, 0.00, 0.0, 0.0, 6, -0.03],   // the light rises
      [s1 + 0.22,   0.24, 0.12, 0, 0.1, 0.00,  0.21, 0.00, 1.00, 1.0, 1.00, 0.0, 0.0, -16, -0.02], // and writes the staff
      [s1 + 0.55,   0.30, 0.12, 0, 0.1, 0.00,  0.21, 0.00, 1.00, 1.0, 1.00, 0.0, 0.0, 3, -0.02],   // "each note is a star"
      [s1 + 0.92,   0.34, 0.12, 0, 0.1, 0.00,  0.19, 0.00, 1.00, 1.0, 1.00, 0.0, 0.0, 22, -0.04],
      [s1 + 1.35,   0.40, 0.12, 0, 0.1, 0.00, -0.20, 0.00, 1.00, 1.0, 1.00, 0.0, 0.0, 22, -0.14],  // "so look where you are"
      [s2 + 0.15,   0.48, 0.10, 0, 0.1, 0.00, -0.17, 0.00, 1.00, 1.0, 1.00, 0.1, 0.0, 10, -0.14],
      [s2 + 0.6,    0.56, 0.05, 0, 0.1, 0.00,  0.02, 0.00, 1.00, 1.0, 1.00, 1.0, 0.0, 2, -0.12],   // "there's warmth in the light"
      [s2 + 0.82,   0.60, 0.28, 0, 0.1, 0.00,  0.08, 0.00, 1.00, 1.0, 1.00, 0.9, 0.0, 2, -0.12],   // "I came as a shadow": a hush
      [s2 + 1.12,   0.66, 0.00, 0, 0.1, 0.00,  0.11, 0.00, 1.00, 1.0, 1.00, 1.0, 1.0, 4, -0.10],   // "to dazzle your night!"
      [T.total,     0.80, 0.00, 0, 0.1, 0.00,  0.12, 0.00, 1.00, 1.0, 1.00, 1.0, 1.0, 4, -0.10]
    ];
  },
  sound: {
    src: '/audio/wind.mp3',
    label: 'Play the night wind and the nocturne',
    volume: function () { return 0.05; },
    cues: [{ stanza: 0, at: 0.52, play: kindleSound }]
      .concat(MELODY.map(function (m, n) { return { stanza: 1, at: noteAt(n), play: note(m[0], 0.11) }; }))
      .concat([{ stanza: 2, at: 0.93, play: dazzleSound }])
  }
});
