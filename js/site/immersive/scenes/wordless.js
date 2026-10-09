/*
 * Scene for "Because She Would Ask Me Why I Loved Her" (Christopher
 * Brennan): an old study at night, seen from the chair at the desk. Two
 * candles burn in wire-caged lanterns either side of a tall window, a glass
 * jar of earth stands between them, open books lie on the desk and the
 * shelves run off into the dark. Nobody is there; the words are.
 *
 * Opening  the lanterns come up in the dark and the first letters lift off
 *      the open pages.
 * I    "If questioning would make us wise": the air fills with drifting
 *      words, letters and question marks. "No eyes would ever gaze in
 *      eyes": the two flames lean towards each other and the words part
 *      between them. "If all our tale were told in speech / No mouths would
 *      wander each to each": the words fall silent and sink away.
 * II   "Were spirits free from mortal mesh": close on the wire cages that
 *      hold each flame. "No aching breasts would yearn to meet": their
 *      light strains out through the mesh towards the other, two reaching
 *      beams, and at "ecstasy complete" the beams meet and the two glows
 *      merge into one warm bloom.
 * III  "The secret powers by which he grows": the bloom sinks into the jar;
 *      a seed pressed against the glass splits, roots run down through the
 *      earth with currents of light pulsing up them and a shoot climbs out
 *      into a stem, leaves and a bud. "To thrill and faint and sweetly
 *      bleed": a red rose opens and trembles to a heartbeat, one candle
 *      faints to a blue bead, and a petal falls like a drop of red.
 * IV   "Seek not, sweet, the 'If' and 'Why'": the last two words rise from
 *      the book and crumble into sparks. Dawn starts at the window. "Life in
 *      me is what you give": a spark leaves the bright flame, crosses to the
 *      fainting one and it catches again.
 * Outro the room in warm, wordless morning light: the rose, the fallen
 *      petal and the two flames.
 *
 * The words are one instanced quad field, painted from a canvas atlas and
 * moved on the GPU (rise from the pages, drift, part, sink). Flames are
 * billboard shaders that lean and gutter; the cages are wire shaders drawn
 * in two passes (far half, flame, near half) so each flame sits inside its
 * mesh. Keyframes are a function of the panels. Columns (see COLS):
 *   [unit, dist, dark, motes, wind, az, el, tx, ty, tz, side, py, pd, pty, lamp,
 *    words, part, lean, sink, strain, merge, seed, grow, bud, open, thrill,
 *    faint, fall, ask, crumble, dawn, spark]
 * where a portrait screen, which keeps its verse mid-screen, scales the
 * distance by "pd", raises the target by "pty" and drops the picture by
 * "py" (a fraction of the height) to keep the subject below the verse.
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, softSprite, particleField, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

var COLS = ['dist', 'dark', 'motes', 'wind', 'az', 'el', 'tx', 'ty', 'tz', 'side', 'py', 'pd', 'pty', 'lamp',
            'words', 'part', 'lean', 'sink', 'strain', 'merge', 'seed', 'grow', 'bud', 'open', 'thrill',
            'faint', 'fall', 'ask', 'crumble', 'dawn', 'spark'];
var K = {};
COLS.forEach(function (c, i) { K[c] = i; });

// ── The study (metres; you sit at +z looking at the window, -z) ─────────
var DESK = 0.76;                                      // desk top
var FLAME_Y = 0.17;                                  // flame centre above a lantern's foot
var LANT_A = new THREE.Vector3(-0.5, DESK, -0.05);    // "me": the one that faints
var LANT_B = new THREE.Vector3(0.5, DESK, -0.05);     // "you": the one that gives
var JAR = new THREE.Vector3(0, DESK, -0.3);
var JAR_R = 0.068, SOIL_R = 0.061, SOIL_H = 0.1;
var SEED = new THREE.Vector3(0, DESK + 0.072, JAR.z + SOIL_R - 0.002);
var WIN = { x0: -0.5, x1: 0.5, y0: 0.94, y1: 2.12, z: -0.78 };
var PETAL_REST = new THREE.Vector3(0.14, DESK + 0.003, -0.13);

// ── Shaders ──────────────────────────────────────────────────────────────
var NOISE =
  'float hash(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }\n' +
  'float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);\n' +
  '  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y); }\n';

// A candle flame on an upright billboard: round below, drawn to a point,
// swaying, flickering and leaning (uLean along uLeanDir, seen on screen).
// uFaint shrinks it to a blue bead.
var FLAME_VS = 'uniform float uW; uniform float uH; uniform vec3 uLeanDir; varying vec2 vQ; varying float vLx;\n' +
  'void main(){ vec3 C = (modelMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz; vec3 toCam = cameraPosition - C; toCam.y = 0.0;\n' +
  ' vec3 R = normalize(cross(vec3(0.0, 1.0, 0.0), toCam));\n' +
  ' vLx = dot(uLeanDir, R); vQ = position.xy;\n' +
  ' vec3 w = C + R * position.x * uW + vec3(0.0, position.y * uH, 0.0);\n' +
  ' gl_Position = projectionMatrix * viewMatrix * vec4(w, 1.0); }';
var FLAME_FS = 'uniform float uTime; uniform float uSeed; uniform float uLean; uniform float uI; uniform float uFaint; uniform float uFlick;\n' +
  'varying vec2 vQ; varying float vLx;\n' + NOISE +
  'void main(){ float t = uTime;\n' +
  ' float hs = 1.0 + (vnoise(vec2(t * 5.0, uSeed)) - 0.5) * 0.22 * uFlick;\n' +
  ' float h = (vQ.y - 0.06) / (0.88 * hs);\n' +
  ' float sway = ((vnoise(vec2(t * 1.7 + uSeed, 3.0)) - 0.5) * 0.3 + sin(t * 8.0 + uSeed) * 0.03) * uFlick;\n' +
  ' float x = vQ.x - (uLean * vLx * 0.7 + sway) * pow(clamp(h, 0.0, 1.2), 1.6);\n' +
  ' float r = h < 0.26 ? 0.4 * sqrt(max(1.0 - (0.26 - h) * (0.26 - h) / 0.0676, 0.0)) : 0.4 * pow(max(1.0 - (h - 0.26) / 0.74, 0.0), 1.25);\n' +
  ' float d = abs(x) / max(r, 0.002);\n' +
  ' float body = smoothstep(1.0, 0.55, d) * step(0.0, h) * step(h, 1.0);\n' +
  ' float core = smoothstep(0.8, 0.15, d) * smoothstep(0.06, 0.3, h) * smoothstep(0.9, 0.42, h);\n' +
  ' float wick = exp(-(x * x) / 0.012 - (h - 0.13) * (h - 0.13) / 0.006);\n' +
  ' float blue = smoothstep(0.32, 0.02, h) * smoothstep(0.3, 0.95, d) * body;\n' +
  ' vec3 col = mix(vec3(1.0, 0.36, 0.07), vec3(1.0, 0.72, 0.32), smoothstep(1.0, 0.45, d));\n' +
  ' col = mix(col, vec3(1.0, 0.96, 0.86), core * (1.0 - uFaint));\n' +
  ' col += vec3(0.18, 0.32, 1.0) * blue * 1.4;\n' +
  ' col *= 1.0 - wick * 0.45;\n' +
  ' col = mix(col, vec3(0.32, 0.5, 1.0), uFaint * 0.8);\n' +
  ' float glow = exp(-d * d * 0.35) * 0.1 * smoothstep(1.15, 0.2, h) * smoothstep(-0.1, 0.1, h);\n' +
  ' float a = (body * (1.0 + core * 1.3) + glow) * uI;\n' +
  ' gl_FragColor = vec4(col * a * 1.5, a);\n #include <colorspace_fragment>\n }';

// A flame's glow on a camera-facing quad: a hot core and long soft skirts,
// stretched towards the other flame (uDir on screen) as it strains.
var GLOW_VS = 'uniform vec2 uSize; varying vec2 vO;\n' +
  'void main(){ vO = position.xy * uSize; vec4 mv = modelViewMatrix * vec4(0.0, 0.0, 0.0, 1.0); mv.xy += vO;\n' +
  ' gl_Position = projectionMatrix * mv; }';
var GLOW_FS = 'uniform vec3 uCol; uniform float uI; uniform float uR; uniform float uReach; uniform float uDir; uniform vec2 uSize; varying vec2 vO;\n' +
  'void main(){ vec2 p = vO / uR; if (p.x * uDir > 0.0) p.x /= 1.0 + uReach * 2.6;\n' +
  ' p.y *= 1.0 + uReach * 0.35; float d2 = dot(p, p);\n' +
  ' float a = (exp(-d2 * 16.0) * 0.7 + exp(-d2 * 3.5) * 0.3 + exp(-d2 * 0.9) * 0.1) * uI;\n' +
  ' vec2 e = abs(vO) / (uSize * 0.5); a *= 1.0 - smoothstep(0.75, 1.0, max(e.x, e.y));\n' +
  ' gl_FragColor = vec4(uCol * a, 1.0);\n #include <colorspace_fragment>\n }';

// The lantern's wire cage: a grid of wires on an open cylinder, lit from
// inside by its flame (the far half shows its lit inner side).
var CAGE_VS = 'varying vec2 vUv; varying vec3 vW;\n' +
  'void main(){ vUv = uv; vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }';
var CAGE_FS = 'uniform vec3 uFlame; uniform float uI; uniform vec3 uWarm; uniform vec3 uAmb; varying vec2 vUv; varying vec3 vW;\n' +
  'void main(){ vec2 g = vec2(vUv.x * 30.0, vUv.y * 15.0); vec2 fw = fwidth(g);\n' +
  ' vec2 d = abs(fract(g + 0.5) - 0.5);\n' +
  ' float wx = 1.0 - smoothstep(0.035, 0.035 + fw.x * 1.2, d.x), wy = 1.0 - smoothstep(0.04, 0.04 + fw.y * 1.2, d.y);\n' +
  ' float wire = max(wx, wy) * (1.0 - smoothstep(0.35, 0.7, max(fw.x, fw.y)) * 0.6);\n' +
  ' if (wire < 0.01) discard;\n' +
  ' float dl = length(vW - uFlame);\n' +
  ' float lit = uI * 0.0016 / (dl * dl + 0.0016);\n' +
  ' vec3 col = vec3(0.04, 0.026, 0.012) + uAmb + uWarm * vec3(1.0, 0.75, 0.45) * lit * (gl_FrontFacing ? 0.4 : 1.0);\n' +
  ' gl_FragColor = vec4(col, wire);\n #include <colorspace_fragment>\n }';

// Glass (the lantern chimneys and the jar): Fresnel edges and the glints
// of both flames, added on top.
var GLASS_VS = 'varying vec3 vN; varying vec3 vW;\n' +
  'void main(){ vN = normalize(mat3(modelMatrix) * normal); vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz;\n' +
  ' gl_Position = projectionMatrix * viewMatrix * w; }';
var GLASS_FS = 'uniform vec3 uA; uniform vec3 uB; uniform float uIA; uniform float uIB; uniform vec3 uWarm; uniform vec3 uAmb; uniform float uOp;\n' +
  'varying vec3 vN; varying vec3 vW;\n' +
  'void main(){ vec3 V = normalize(cameraPosition - vW); vec3 N = normalize(vN); if (!gl_FrontFacing) N = -N;\n' +
  ' float fres = pow(1.0 - abs(dot(N, V)), 3.0);\n' +
  ' vec3 R = reflect(-V, N);\n' +
  ' float sa = pow(max(dot(R, normalize(uA - vW)), 0.0), 90.0) * uIA, sb = pow(max(dot(R, normalize(uB - vW)), 0.0), 90.0) * uIB;\n' +
  ' vec3 col = uAmb * (0.3 + fres) + uWarm * ((sa + sb) * 2.5 + fres * 0.18 * (uIA + uIB));\n' +
  ' float a = clamp(0.25 + fres * 0.8 + sa + sb, 0.0, 1.0) * uOp;\n' +
  ' gl_FragColor = vec4(col, a);\n #include <colorspace_fragment>\n }';

// The words: instanced quads from the atlas. Each rises off a page in an
// arc (uEmerge past its threshold), drifts, is pushed out of the line
// between the flames (uPart), then sinks and fades (uSink). They glow where
// the candles light them.
var WORD_VS = 'attribute vec3 aHome; attribute vec3 aSrc; attribute vec4 aRect; attribute vec4 aInfo;\n' +
  'uniform float uTime; uniform float uEmerge; uniform float uSink; uniform float uPart; uniform float uWob;\n' +
  'uniform vec3 uA; uniform vec3 uB; uniform float uIA; uniform float uIB;\n' +
  'varying vec2 vUv; varying float vA; varying vec3 vCol;\n' +
  'void main(){ float sd = aInfo.w;\n' +
  ' float e = smoothstep(aInfo.z, aInfo.z + 0.22, uEmerge);\n' +
  ' vec3 p = mix(aSrc, aHome, e); p.y += sin(e * 3.14159) * 0.12;\n' +
  ' float t = uTime * uWob;\n' +
  ' p += vec3(sin(t * 0.31 + sd * 17.0), sin(t * 0.23 + sd * 29.0) * 0.7, cos(t * 0.27 + sd * 11.0)) * 0.04 * e;\n' +
  ' vec3 ab = uB - uA; float h = clamp(dot(p - uA, ab) / dot(ab, ab), 0.0, 1.0); vec3 dv = p - (uA + ab * h);\n' +
  ' float dl = length(dv) + 1e-4; p += dv / dl * uPart * 0.24 * exp(-dl * dl * 12.0);\n' +
  ' float s = smoothstep(sd * 0.45, sd * 0.45 + 0.55, uSink);\n' +
  ' p.y -= s * s * 0.55;\n' +
  ' float la = exp(-dot(p - uA, p - uA) * 3.0) * uIA, lb = exp(-dot(p - uB, p - uB) * 3.0) * uIB;\n' +
  ' vCol = vec3(1.0, 0.8, 0.52) * (0.2 + 1.5 * (la + lb)) * (1.0 - s * 0.6);\n' +
  ' vec4 mv = modelViewMatrix * vec4(p, 1.0);\n' +
  ' float ang = sin(t * 0.2 + sd * 40.0) * 0.25 + (sd - 0.5) * 0.4;\n' +
  ' vec2 c = position.xy * vec2(aInfo.x * aInfo.y, aInfo.x);\n' +
  ' mv.xy += vec2(c.x * cos(ang) - c.y * sin(ang), c.x * sin(ang) + c.y * cos(ang));\n' +
  ' vA = e * (1.0 - smoothstep(0.5, 1.0, s)) * smoothstep(0.3, 0.75, -mv.z);\n' +
  ' vUv = aRect.xy + (position.xy + 0.5) * aRect.zw;\n' +
  ' gl_Position = projectionMatrix * mv; }';
var WORD_FS = 'uniform sampler2D uAtlas; uniform float uAlpha; varying vec2 vUv; varying float vA; varying vec3 vCol;\n' +
  'void main(){ float a = texture2D(uAtlas, vUv).a * vA * uAlpha; if (a < 0.003) discard;\n' +
  ' gl_FragColor = vec4(vCol * a, a);\n #include <colorspace_fragment>\n }';

// "If" and "Why": one large billboard word each, crumbling from the left
// with an ember edge as uC rises.
var ASK_VS = 'uniform vec2 uSize; uniform float uTilt; varying vec2 vQ;\n' +
  'void main(){ vQ = position.xy + 0.5; vec4 mv = modelViewMatrix * vec4(0.0, 0.0, 0.0, 1.0);\n' +
  ' vec2 c = position.xy * uSize; mv.xy += vec2(c.x * cos(uTilt) - c.y * sin(uTilt), c.x * sin(uTilt) + c.y * cos(uTilt));\n' +
  ' gl_Position = projectionMatrix * mv; }';
var ASK_FS = 'uniform sampler2D uAtlas; uniform vec4 uRect; uniform float uC; uniform float uAlpha; uniform float uI;\n' +
  'varying vec2 vQ;\n' + NOISE +
  'void main(){ float m = texture2D(uAtlas, uRect.xy + vQ * uRect.zw).a;\n' +
  ' float n = 0.65 * vQ.x + 0.35 * vnoise(vQ * vec2(16.0, 9.0));\n' +
  ' float front = uC * 1.6 - 0.1;\n' +
  ' float keep = smoothstep(front, front + 0.02, n);\n' +
  ' float en = (n - front - 0.02) / 0.03, ember = exp(-en * en) * step(0.001, uC);\n' +
  ' vec3 col = vec3(1.0, 0.86, 0.6) * uI * keep + vec3(1.0, 0.45, 0.12) * ember * 3.0;\n' +
  ' float a = m * max(keep, ember) * uAlpha; if (a < 0.003) discard;\n' +
  ' gl_FragColor = vec4(col * a, a);\n #include <colorspace_fragment>\n }';

// The sparks they crumble into: glyph-shaped points that are born as the
// burning front passes them and drift up, cooling from gold to red.
var SPARK_VS = 'attribute vec2 aOff; attribute vec2 aInfo; uniform float uC; uniform float uTime; uniform float uScale;\n' +
  'varying float vA; varying float vAge;\n' +
  'void main(){ float front = uC * 1.6 - 0.1; float age = (front + 0.02 - aInfo.x) * 2.6;\n' +
  ' vAge = clamp(age, 0.0, 1.0); vA = step(0.0, age) * (1.0 - smoothstep(0.55, 1.0, age));\n' +
  ' float sd = aInfo.y;\n' +
  ' vec4 mv = modelViewMatrix * vec4(0.0, 0.0, 0.0, 1.0);\n' +
  ' mv.xy += aOff + vec2(sin(sd * 40.0) * 0.08 + sin(uTime * 1.3 + sd * 20.0) * 0.012, 0.22 + sd * 0.12) * vAge * vAge + vec2(0.0, -0.03) * vAge;\n' +
  ' gl_Position = projectionMatrix * mv;\n' +
  ' gl_PointSize = uScale * (1.0 - vAge * 0.6) * (0.6 + sd * 0.8) / -mv.z; }';
var SPARK_FS = 'varying float vA; varying float vAge;\n' +
  'void main(){ float d = length(gl_PointCoord - 0.5); float a = (exp(-d * d * 40.0) + exp(-d * d * 9.0) * 0.3) * vA;\n' +
  ' vec3 col = mix(vec3(1.0, 0.85, 0.55), vec3(0.95, 0.18, 0.08), vAge);\n' +
  ' gl_FragColor = vec4(col * a * 1.3, 1.0);\n #include <colorspace_fragment>\n }';

// A strip along an axis, turned to face the camera: the beams reaching
// between the flames, and the dawn's shaft of light.
var AXIS_VS = 'uniform vec3 uP0; uniform vec3 uP1; uniform float uW0; uniform float uW1; varying float vT; varying float vY;\n' +
  'void main(){ vT = position.x + 0.5; vY = position.y * 2.0;\n' +
  ' vec3 w = mix(uP0, uP1, vT); vec3 side = normalize(cross(cameraPosition - w, uP1 - uP0));\n' +
  ' w += side * position.y * mix(uW0, uW1, vT);\n' +
  ' gl_Position = projectionMatrix * viewMatrix * vec4(w, 1.0); }';
var BEAM_FS = 'uniform float uReach; uniform float uMerge; uniform float uTime; uniform float uAlpha; varying float vT; varying float vY;\n' + NOISE +
  'void main(){ float x = vT;\n' +
  ' float fa = smoothstep(uReach + 0.015, uReach - 0.07, x), fb = smoothstep(uReach + 0.015, uReach - 0.07, 1.0 - x);\n' +
  ' float m = max(max(fa, fb), uMerge);\n' +
  ' float ta = (x - uReach) / 0.035, tb = (1.0 - x - uReach) / 0.035, tc = (x - 0.5) / 0.18;\n' +
  ' float tip = (exp(-ta * ta) + exp(-tb * tb)) * (1.0 - uMerge);\n' +
  ' float across = exp(-vY * vY * 5.0);\n' +
  ' float fil = 0.55 + 0.45 * vnoise(vec2(x * 46.0 - uTime * 1.6 * sign(0.5 - x), vY * 3.5));\n' +
  ' float ends = smoothstep(0.03, 0.12, x) * smoothstep(0.97, 0.88, x);\n' +
  ' float core = exp(-vY * vY * 40.0);\n' +
  ' float a = (m * (across * fil * 0.45 + core * 0.6) + tip * (across * 1.6 + core * 1.5) + uMerge * exp(-tc * tc) * across * 0.8) * ends * uAlpha;\n' +
  ' gl_FragColor = vec4(vec3(1.0, 0.7, 0.38) * a, 1.0);\n #include <colorspace_fragment>\n }';
var SHAFT_FS = 'uniform float uAlpha; uniform float uTime; varying float vT; varying float vY;\n' + NOISE +
  'void main(){ float across = smoothstep(1.0, 0.25, abs(vY));\n' +
  ' float along = smoothstep(0.0, 0.18, vT) * smoothstep(1.0, 0.55, vT);\n' +
  ' float bars = 0.75 + 0.25 * vnoise(vec2(vY * 3.0 + uTime * 0.05, vT * 2.0));\n' +
  ' float a = across * along * bars * uAlpha;\n' +
  ' gl_FragColor = vec4(vec3(1.0, 0.78, 0.56) * a, 1.0);\n #include <colorspace_fragment>\n }';

// The sky beyond the window: night, a few stars and a far line of trees,
// warming to dawn.
var SKY_VS = 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }';
var SKY_FS = 'uniform float uDawn; uniform float uTime; varying vec2 vUv;\n' + NOISE +
  'void main(){ vec2 uv = vUv; float y = uv.y;\n' +
  ' vec3 night = mix(vec3(0.018, 0.03, 0.075), vec3(0.004, 0.007, 0.022), smoothstep(0.2, 0.9, y));\n' +
  ' vec3 dawn = mix(vec3(1.0, 0.58, 0.3), vec3(0.62, 0.36, 0.36), smoothstep(0.18, 0.45, y));\n' +
  ' dawn = mix(dawn, vec3(0.05, 0.07, 0.17), smoothstep(0.4, 0.95, y));\n' +
  ' vec3 c = mix(night, dawn, uDawn);\n' +
  ' c += vec3(1.0, 0.72, 0.42) * exp(-length((uv - vec2(0.6, 0.22)) * vec2(1.0, 1.8)) * 5.0) * uDawn * uDawn;\n' +
  ' vec2 g = uv * vec2(46.0, 40.0); vec2 id = floor(g); float hh = hash(id);\n' +
  ' vec2 f = fract(g) - 0.5 - (vec2(hash(id + 3.1), hash(id + 7.7)) - 0.5) * 0.6;\n' +
  ' c += vec3(0.8, 0.86, 1.0) * step(0.9, hh) * exp(-dot(f, f) * 140.0) * (0.55 + 0.45 * sin(uTime * 1.3 + hh * 40.0)) * (1.0 - uDawn) * smoothstep(0.3, 0.5, y);\n' +
  ' float x = uv.x;\n' +
  ' float hill = 0.2 + 0.022 * sin(x * 6.0 + 1.0) + 0.012 * sin(x * 15.0 + 2.0) + 0.016 * vnoise(vec2(x * 24.0, 1.0)) + 0.006 * vnoise(vec2(x * 70.0, 3.0));\n' +
  ' c = mix(c, mix(vec3(0.004, 0.005, 0.01), vec3(0.12, 0.07, 0.07), uDawn), smoothstep(hill + 0.003, hill - 0.003, y));\n' +
  ' gl_FragColor = vec4(c, 1.0);\n #include <colorspace_fragment>\n }';

// ── Canvas textures ──────────────────────────────────────────────────────
function canvasTex(w, h, draw, srgb) {
  var c = document.createElement('canvas');
  c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  var t = new THREE.CanvasTexture(c);
  if (srgb !== false) t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

// Walnut for the desk: long wavering grain.
function woodTexture(r, base) {
  return canvasTex(1024, 256, function (x, w, h) {
    x.fillStyle = base;
    x.fillRect(0, 0, w, h);
    for (var i = 0; i < 160; i++) {
      var y0 = r() * h, a = 0.03 + r() * 0.09, dark = r() < 0.6;
      x.strokeStyle = dark ? 'rgba(10,5,2,' + a + ')' : 'rgba(255,215,170,' + a * 0.6 + ')';
      x.lineWidth = 0.6 + r() * 2.2;
      x.beginPath();
      var ph = r() * 6.28, amp = 1 + r() * 5, fq = 0.004 + r() * 0.01;
      for (var px = 0; px <= w; px += 16) x.lineTo(px, y0 + Math.sin(px * fq + ph) * amp);
      x.stroke();
    }
  });
}

// A page: aged paper, faint lines of handwriting-grey text.
function pageTexture(r) {
  return canvasTex(256, 352, function (x, w, h) {
    x.fillStyle = '#e6d8b8';
    x.fillRect(0, 0, w, h);
    var g = x.createRadialGradient(w / 2, h / 2, w * 0.2, w / 2, h / 2, w * 0.8);
    g.addColorStop(0, 'rgba(0,0,0,0)');
    g.addColorStop(1, 'rgba(90,60,20,0.35)');
    x.fillStyle = g;
    x.fillRect(0, 0, w, h);
    x.fillStyle = 'rgba(60,40,25,0.55)';
    for (var ly = 34; ly < h - 26; ly += 11) {
      var px = 22 + (ly === 34 ? 40 : 0);
      while (px < w - 24) {
        var ww = 6 + r() * 24;
        if (px + ww > w - 22) break;
        x.fillRect(px, ly, ww, 2.4);
        px += ww + 4 + r() * 3;
      }
    }
    x.fillStyle = 'rgba(110,30,20,0.55)';
    x.fillRect(22, 30, 30, 24);
  });
}

// Book spines on the shelves: raised bands and a title label.
function spineTexture() {
  return canvasTex(64, 256, function (x, w, h) {
    x.fillStyle = '#c8c8c8';
    x.fillRect(0, 0, w, h);
    x.fillStyle = '#ffffff';
    [0.1, 0.14, 0.82, 0.86, 0.93].forEach(function (v) { x.fillRect(0, v * h, w, 3); });
    x.fillStyle = '#5a5a5a';
    x.fillRect(8, 0.22 * h, w - 16, 0.12 * h);
    x.fillStyle = 'rgba(255,255,255,0.6)';
    x.fillRect(14, 0.26 * h, w - 28, 3);
  });
}

function soilTexture(r) {
  var t = canvasTex(256, 128, function (x, w, h) {
    x.fillStyle = '#2a1a10';
    x.fillRect(0, 0, w, h);
    for (var i = 0; i < 2600; i++) {
      var l = r();
      x.fillStyle = l < 0.5 ? 'rgba(8,4,2,0.6)' : l < 0.9 ? 'rgba(80,55,35,0.5)' : 'rgba(150,130,110,0.6)';
      x.fillRect(r() * w, r() * h, 1 + r() * 2.5, 1 + r() * 2);
    }
  });
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

// ── The word atlas ──────────────────────────────────────────────────────
var WORDS = ['why', 'if', 'how', 'what', 'who', 'wise', 'speech', 'tell', 'know', 'because', 'reason', 'whether',
             'when', 'proof', 'tale', 'told', 'words', 'said', 'ask', 'answer', 'meaning', 'therefore', 'perhaps',
             'whence', 'wherefore', 'yet', 'but', 'so', 'thus', 'is it'];
var LETTERS = ['a', 'e', 'i', 'o', 's', 't', 'w', 'y', 'h', 'r', 'k', 'n', 'W', 'I', '&', ';', '¶', '§'];
var FONT = 'Georgia, "Times New Roman", serif';

function atlas() {
  var W = 1024, H = 1024, c = document.createElement('canvas');
  c.width = W; c.height = H;
  var x = c.getContext('2d'), rects = {}, cx = 0, cy = 0, rowH = 0;
  x.fillStyle = '#fff';
  x.shadowColor = 'rgba(255,255,255,0.55)';
  x.shadowBlur = 5;
  function put(key, text, font, size) {
    x.font = font;
    var w = Math.ceil(x.measureText(text).width) + 20, h = Math.ceil(size * 1.45);
    if (cx + w > W) { cx = 0; cy += rowH; rowH = 0; }
    x.fillText(text, cx + 10, cy + h * 0.74);
    rects[key] = { px: cx, py: cy, pw: w, ph: h, rect: [cx / W, 1 - (cy + h) / H, w / W, h / H], ratio: w / h };
    cx += w;
    rowH = Math.max(rowH, h);
  }
  put('If', 'If', 'italic 150px ' + FONT, 150);
  put('Why', 'Why', 'italic 150px ' + FONT, 150);
  ['?', '?', '?'].forEach(function (q, i) { put('?' + i, q, (i === 1 ? 'italic ' : '') + (68 + i * 8) + 'px ' + FONT, 76); });
  WORDS.forEach(function (w) { put('w:' + w, w, 'italic 58px ' + FONT, 58); });
  LETTERS.forEach(function (l) { put('l:' + l, l, '62px ' + FONT, 62); });
  var t = new THREE.CanvasTexture(c);
  t.anisotropy = 8;
  return { tex: t, rects: rects, canvas: c, ctx: x };
}

// Points inside a glyph's ink, in its quad's -0.5..0.5 units.
function glyphPoints(A, key, n, r) {
  var g = A.rects[key], data = A.ctx.getImageData(g.px, g.py, g.pw, g.ph).data, pts = [];
  for (var tries = 0; tries < n * 60 && pts.length < n; tries++) {
    var px = Math.floor(r() * g.pw), py = Math.floor(r() * g.ph);
    if (data[(py * g.pw + px) * 4 + 3] > 150) pts.push([px / g.pw - 0.5, 0.5 - py / g.ph]);
  }
  return pts;
}

// ── Geometry ─────────────────────────────────────────────────────────────
// A brass lantern round a candle, foot at y = 0: the brass in one merged
// piece; the cage, chimney, candle and flame are separate.
function lanternBrass() {
  var B = '#b48c4c', D = '#6a4c22', parts = [];
  function add(g, c, y) { parts.push(tinted(g.translate(0, y, 0), c)); }
  add(new THREE.CylinderGeometry(0.098, 0.108, 0.028, 40), D, 0.014);
  add(new THREE.TorusGeometry(0.092, 0.006, 6, 40).rotateX(Math.PI / 2), B, 0.03);
  add(new THREE.CylinderGeometry(0.03, 0.036, 0.022, 20), B, 0.04);
  add(new THREE.TorusGeometry(0.09, 0.006, 6, 40).rotateX(Math.PI / 2), B, 0.32);
  add(new THREE.CylinderGeometry(0.046, 0.094, 0.05, 40), D, 0.345);
  add(new THREE.CylinderGeometry(0.026, 0.03, 0.04, 20), B, 0.39);
  add(new THREE.ConeGeometry(0.044, 0.03, 20), D, 0.425);
  add(new THREE.TorusGeometry(0.046, 0.0045, 6, 24, Math.PI), B, 0.44);
  for (var k = 0; k < 4; k++) {
    var a = k / 4 * Math.PI * 2 + Math.PI / 4;
    parts.push(tinted(new THREE.CylinderGeometry(0.0045, 0.0045, 0.29, 6).translate(Math.cos(a) * 0.092, 0.175, Math.sin(a) * 0.092), B));
  }
  return merge(parts);
}

// An open book on the desk: covers, two page blocks and pages that dip into
// the gutter. Returned with a function for a random point on its pages.
function openBook(pageTex, cover) {
  var g = new THREE.Group();
  var coverMat = new THREE.MeshStandardMaterial({ color: cover, roughness: 0.7 });
  var c = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.012, 0.31), coverMat);
  c.position.y = 0.006;
  g.add(c);
  var edgeMat = new THREE.MeshStandardMaterial({ color: '#cbbb98', roughness: 0.9 });
  var pageMat = new THREE.MeshStandardMaterial({ map: pageTex, roughness: 0.92, side: THREE.DoubleSide });
  [-1, 1].forEach(function (s) {
    var blk = new THREE.Mesh(new THREE.BoxGeometry(0.212, 0.016, 0.29), edgeMat);
    blk.position.set(s * 0.108, 0.02, 0);
    g.add(blk);
    var pg = new THREE.PlaneGeometry(0.215, 0.29, 14, 1).rotateX(-Math.PI / 2);
    var p = pg.attributes.position;
    for (var i = 0; i < p.count; i++) {
      var xs = p.getX(i) + 0.1075;                       // 0 at the spine
      p.setY(i, 0.03 - 0.013 * Math.exp(-xs * 32) + 0.006 * Math.sin(Math.PI * xs / 0.215));
      p.setX(i, s * xs);
    }
    if (s < 0) {                                          // keep the winding facing up
      var idx = pg.index.array;
      for (i = 0; i < idx.length; i += 3) { var tmp = idx[i]; idx[i] = idx[i + 1]; idx[i + 1] = tmp; }
    }
    pg.computeVertexNormals();
    g.add(new THREE.Mesh(pg, pageMat));
  });
  g.userData.pagePoint = function (r, out) {
    var s = r() < 0.5 ? -1 : 1, xs = 0.03 + r() * 0.17;
    out.set(s * xs, 0.034, (r() - 0.5) * 0.24);
    return g.localToWorld(out);
  };
  return g;
}

// A rose petal: narrow at the claw, cupped, its tip turned out a little.
// Inner face towards -z; it stands along +y.
function petalGeometry() {
  var nu = 6, nv = 8, pos = [], col = [], idx = [], c = new THREE.Color();
  var base = new THREE.Color('#4a0610'), mid = new THREE.Color('#b8162b'), edge = new THREE.Color('#7a0a18');
  for (var j = 0; j <= nv; j++) {
    var v = j / nv, w = 0.5 * (0.28 + 0.72 * Math.sin(Math.PI * Math.min(1, v * 0.82 + 0.1)));
    if (v > 0.78) w *= Math.sqrt(Math.max(0, 1 - Math.pow((v - 0.78) / 0.24, 2)));
    for (var i = 0; i <= nu; i++) {
      var u = i / nu * 2 - 1;
      pos.push(u * w, v, -0.38 * u * u * w + 0.16 * v * v);
      c.copy(base).lerp(mid, smooth(0, 0.45, v)).lerp(edge, Math.abs(u) * 0.35 + v * v * 0.35);
      col.push(c.r, c.g, c.b);
    }
  }
  for (j = 0; j < nv; j++) for (i = 0; i < nu; i++) {
    var a = j * (nu + 1) + i, b = a + nu + 1;
    idx.push(a, b, a + 1, a + 1, b, b + 1);
  }
  var g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  g.setIndex(idx);
  g.scale(0.036, 0.042, 0.04);
  g.computeVertexNormals();
  return g;
}

// A leaf, folded along its midrib and drooping; stands along +y.
function leafGeometry(color, tip) {
  var nu = 4, nv = 7, pos = [], col = [], idx = [], c = new THREE.Color(color), dk = c.clone().multiplyScalar(0.55);
  for (var j = 0; j <= nv; j++) {
    var v = j / nv, w = 0.5 * Math.pow(Math.sin(Math.PI * v), 0.8) * (1 - 0.35 * v * tip);
    for (var i = 0; i <= nu; i++) {
      var u = i / nu * 2 - 1;
      pos.push(u * w, v, 0.3 * Math.abs(u) * w - 0.18 * v * v);
      var k = c.clone().lerp(dk, 1 - Math.abs(u));
      col.push(k.r, k.g, k.b);
    }
  }
  for (j = 0; j < nv; j++) for (i = 0; i < nu; i++) {
    var a = j * (nu + 1) + i, b = a + nu + 1;
    idx.push(a, b, a + 1, a + 1, b, b + 1);
  }
  var g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

// Tubes with a growth time per vertex (aT), merged into one indexed geometry.
function growTubes(paths) {
  var P = [], N = [], T = [], I = [], base = 0;
  paths.forEach(function (p) {
    var curve = new THREE.CatmullRomCurve3(p.pts);
    var g = new THREE.TubeGeometry(curve, Math.max(6, p.pts.length * 2), p.r, 5, false);
    var pa = g.attributes.position, na = g.attributes.normal, ua = g.attributes.uv;
    for (var i = 0; i < pa.count; i++) {
      P.push(pa.getX(i), pa.getY(i), pa.getZ(i));
      N.push(na.getX(i), na.getY(i), na.getZ(i));
      T.push(p.t0 + ua.getX(i) * (p.t1 - p.t0));
    }
    var ix = g.index.array;
    for (i = 0; i < ix.length; i++) I.push(ix[i] + base);
    base += pa.count;
    g.dispose();
  });
  var out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.Float32BufferAttribute(P, 3));
  out.setAttribute('normal', new THREE.Float32BufferAttribute(N, 3));
  out.setAttribute('aT', new THREE.Float32BufferAttribute(T, 1));
  out.setIndex(I);
  return out;
}

// A lit material that grows (vertices past uGrow are cut away), glows at
// its growing tip and carries pulses of light along it (uDir: +1 travels
// with aT, -1 against it, towards the seed).
function growMaterial(color, glow, dir) {
  var mat = new THREE.MeshStandardMaterial({ color: color, roughness: 0.7, envMapIntensity: 0.3 });
  var u = { uGrow: { value: 0 }, uTime: { value: 0 }, uPulse: { value: 0 }, uBeat: { value: 0 }, uGlow: { value: new THREE.Color(glow) },
            uDir: { value: dir } };
  mat.onBeforeCompile = function (sh) {
    Object.assign(sh.uniforms, u);
    sh.vertexShader = 'attribute float aT; varying float vT;\n' + sh.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\n vT = aT;');
    sh.fragmentShader = 'uniform float uGrow; uniform float uTime; uniform float uPulse; uniform float uBeat; uniform vec3 uGlow; uniform float uDir; varying float vT;\n' +
      sh.fragmentShader
        .replace('#include <clipping_planes_fragment>', 'if (vT > uGrow) discard;\n#include <clipping_planes_fragment>')
        .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\n' +
          ' float gPh = fract(vT * 7.0 - uTime * 0.45 * uDir);\n' +
          ' float gPulse = smoothstep(0.0, 0.04, gPh) * exp(-gPh * 7.0);\n' +
          ' float gTip = exp(-(uGrow - vT) * 60.0) * (1.0 - smoothstep(0.9, 0.995, uGrow)) * 0.7;\n' +
          ' totalEmissiveRadiance += uGlow * (gPulse * uPulse + gTip * 1.2 + uBeat * 0.5);');
  };
  mat.userData.uniforms = u;
  return mat;
}

// Roots: paths over the front of the soil, down from the seed, branching.
function makeRoots(r) {
  var paths = [], RS = SOIL_R + 0.0012, bottom = DESK + 0.012;
  function at(th, y) { return new THREE.Vector3(JAR.x + Math.sin(th) * RS, y, JAR.z + Math.cos(th) * RS); }
  function walk(th, y, lean, len, t0, speed, rad, depth) {
    var pts = [at(th, y)], step = 0.004, d = 0;
    for (; d < len; d += step) {
      lean += (r() - 0.5) * 0.5;
      lean *= 0.92;
      th += Math.sin(lean) * step / RS;
      y -= Math.cos(lean) * step * 0.95;
      if (y < bottom || Math.abs(th) > 1.25) break;
      pts.push(at(th, y));
      if (depth < 2 && d > 0.012 && r() < 0.12) {
        walk(th, y, (r() < 0.5 ? -1 : 1) * (0.7 + r() * 0.6), len * (0.3 + r() * 0.35), t0 + d / speed, speed * 0.9, rad * 0.65, depth + 1);
      }
    }
    if (pts.length > 2) paths.push({ pts: pts, t0: t0, t1: t0 + d / speed, r: rad });
  }
  var y0 = SEED.y - 0.004;
  walk(0.02, y0, 0.1, 0.075, 0, 0.11, 0.0013, 0);
  walk(-0.12, y0, -0.7, 0.06, 0.08, 0.1, 0.0009, 1);
  walk(0.12, y0, 0.75, 0.055, 0.12, 0.1, 0.0009, 1);
  return paths;
}

// ── Synthesised sound ────────────────────────────────────────────────────
var NOISE_BUF = new WeakMap();
function noiseBuf(ac) {
  var b = NOISE_BUF.get(ac);
  if (b) return b;
  b = ac.createBuffer(1, ac.sampleRate * 2, ac.sampleRate);
  var d = b.getChannelData(0);
  for (var i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  NOISE_BUF.set(ac, b);
  return b;
}
function noiseBurst(ac, out, t, dur, f, q, gain, type) {
  var s = ac.createBufferSource(), bp = ac.createBiquadFilter(), g = ac.createGain();
  s.buffer = noiseBuf(ac);
  bp.type = type || 'bandpass';
  bp.frequency.value = f;
  bp.Q.value = q;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(gain, t + dur * 0.25);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  s.connect(bp); bp.connect(g); g.connect(out);
  s.start(t, Math.random() * 1.5);
  s.stop(t + dur + 0.05);
}
// A page turning: a soft papery sweep and a few crackles.
function rustle(ac, out) {
  var t = ac.currentTime + 0.01;
  noiseBurst(ac, out, t, 0.55, 2600, 0.7, 0.05);
  noiseBurst(ac, out, t + 0.18, 0.35, 4200, 1.2, 0.03);
  for (var i = 0; i < 5; i++) noiseBurst(ac, out, t + 0.05 + Math.random() * 0.4, 0.04, 5000 + Math.random() * 2500, 3, 0.02);
}
// The two glows meeting: a chord that swells and opens.
function swell(ac, out) {
  var t = ac.currentTime + 0.01, lp = ac.createBiquadFilter();
  lp.type = 'lowpass';
  lp.frequency.setValueAtTime(380, t);
  lp.frequency.exponentialRampToValueAtTime(2600, t + 2.2);
  lp.connect(out);
  [196, 293.66, 392, 493.88, 587.33].forEach(function (f, k) {
    [0, 1.5].forEach(function (det) {
      var o = ac.createOscillator(), g = ac.createGain();
      o.type = k < 2 ? 'triangle' : 'sine';
      o.frequency.value = f;
      o.detune.value = det ? 7 : -5;
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(0.022 / (1 + k * 0.25), t + 1.6);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 6.5);
      o.connect(g); g.connect(lp);
      o.start(t); o.stop(t + 6.6);
    });
  });
}
// "To thrill and faint": a low heartbeat, fading.
function heartbeat(ac, out) {
  var t = ac.currentTime + 0.01;
  for (var k = 0; k < 5; k++) {
    [[0, 64, 0.2], [0.21, 56, 0.13]].forEach(function (b) {
      var at = t + k * 0.92 + b[0], o = ac.createOscillator(), g = ac.createGain();
      o.type = 'sine';
      o.frequency.setValueAtTime(b[1], at);
      o.frequency.exponentialRampToValueAtTime(b[1] * 0.62, at + 0.16);
      g.gain.setValueAtTime(0.0001, at);
      g.gain.exponentialRampToValueAtTime(b[2] * (1 - k * 0.15), at + 0.018);
      g.gain.exponentialRampToValueAtTime(0.0001, at + 0.26);
      o.connect(g); g.connect(out);
      o.start(at); o.stop(at + 0.3);
    });
  }
}
// "If" and "Why" crumbling: a dry crackle, like paper in a grate.
function crackle(ac, out) {
  var t = ac.currentTime + 0.01;
  for (var i = 0; i < 22; i++) noiseBurst(ac, out, t + Math.pow(Math.random(), 1.3) * 1.6, 0.03 + Math.random() * 0.04, 3000 + Math.random() * 4000, 2.5, 0.012 + Math.random() * 0.02);
  noiseBurst(ac, out, t, 1.6, 900, 0.6, 0.012);
}
// The candle catching again: a soft breath of flame and a small bright tone.
function kindle(ac, out) {
  var t = ac.currentTime + 0.01;
  noiseBurst(ac, out, t, 0.7, 700, 0.5, 0.06, 'lowpass');
  [1046.5, 1568].forEach(function (f, k) {
    var o = ac.createOscillator(), g = ac.createGain();
    o.type = 'sine';
    o.frequency.value = f;
    g.gain.setValueAtTime(0.0001, t + 0.12);
    g.gain.exponentialRampToValueAtTime(0.02 / (k + 1), t + 0.14);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 2.4);
    o.connect(g); g.connect(out);
    o.start(t + 0.1); o.stop(t + 2.5);
  });
}
// The morning: a warm, wide major ninth, slow to come and slow to go.
function morningChord(ac, out) {
  var t = ac.currentTime + 0.01;
  [87.31, 174.61, 261.63, 349.23, 440, 523.25, 659.26, 783.99].forEach(function (f, k) {
    [-4, 4].forEach(function (det) {
      var o = ac.createOscillator(), g = ac.createGain();
      o.type = 'sine';
      o.frequency.value = f;
      o.detune.value = det;
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(0.02 / (1 + k * 0.18), t + 1.8 + k * 0.12);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 8.5);
      o.connect(g); g.connect(out);
      o.start(t); o.stop(t + 8.6);
    });
  });
}

// A small room for reflections in the brass and glass: dark wood, two warm
// candle glows either side, the window's cool light behind.
function studyEnvironment(gl) {
  var room = new THREE.Scene(), geos = [], mats = [];
  function add(geo, rgb, at, side) {
    var m = new THREE.MeshBasicMaterial({ color: new THREE.Color(rgb[0], rgb[1], rgb[2]), side: side || THREE.DoubleSide });
    var mesh = new THREE.Mesh(geo, m);
    mesh.position.set(at[0], at[1], at[2]);
    mesh.lookAt(0, 0, 0);
    room.add(mesh);
    geos.push(geo); mats.push(m);
  }
  var box = new THREE.Mesh(new THREE.BoxGeometry(10, 6, 10), new THREE.MeshBasicMaterial({ color: new THREE.Color(0.025, 0.016, 0.01), side: THREE.BackSide }));
  room.add(box); geos.push(box.geometry); mats.push(box.material);
  add(new THREE.PlaneGeometry(1.2, 1.2), [3.2, 2.0, 0.9], [-3.5, 0.6, 1.0]);
  add(new THREE.PlaneGeometry(1.2, 1.2), [3.2, 2.0, 0.9], [3.5, 0.6, 1.0]);
  add(new THREE.PlaneGeometry(2.2, 2.6), [0.25, 0.32, 0.55], [0, 1.5, -4.5]);
  add(new THREE.PlaneGeometry(6, 3), [0.16, 0.1, 0.06], [0, -2.5, 0]);
  add(new THREE.PlaneGeometry(2.5, 1.5), [0.6, 0.42, 0.28], [0, 1.2, 4.5]);
  var pm = new THREE.PMREMGenerator(gl), rt = pm.fromScene(room, 0.04);
  pm.dispose();
  geos.forEach(function (g) { g.dispose(); });
  mats.forEach(function (m) { m.dispose(); });
  return rt;
}

// ── Renderer ─────────────────────────────────────────────────────────────
function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(4417), i, k;
  var gl = makeRenderer(canvas, { clear: '#050302' });
  gl.toneMappingExposure = 1.15;
  var world = new THREE.Scene();
  world.fog = new THREE.Fog('#070403', 2.6, 7.5);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.02, 40);
  var envRT = studyEnvironment(gl);
  world.environment = envRT.texture;
  world.environmentIntensity = 0.5;
  var UP = new THREE.Vector3(0, 1, 0);

  // ── The room ──
  var wallMat = new THREE.MeshStandardMaterial({ color: '#2a1d15', roughness: 0.95 });
  var darkWood = new THREE.MeshStandardMaterial({ color: '#2e1d12', roughness: 0.7 });
  function plane(w, h, mat, x, y, z, ry, rx) {
    var m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
    m.position.set(x, y, z);
    m.rotation.set(rx || 0, ry || 0, 0);
    world.add(m);
    return m;
  }
  // The back wall round the window opening, the side walls, floor and ceiling.
  plane(2.4, 2.8, wallMat, -1.7, 1.4, WIN.z);
  plane(2.4, 2.8, wallMat, 1.7, 1.4, WIN.z);
  plane(1.0, WIN.y0, wallMat, 0, WIN.y0 / 2, WIN.z);
  plane(1.0, 2.8 - WIN.y1, wallMat, 0, (2.8 + WIN.y1) / 2, WIN.z);
  plane(4.5, 2.8, wallMat, -2.9, 1.4, 1.4, Math.PI / 2);
  plane(4.5, 2.8, wallMat, 2.9, 1.4, 1.4, -Math.PI / 2);
  plane(6, 4.5, darkWood, 0, 0, 1.4, 0, -Math.PI / 2);
  plane(6, 4.5, wallMat, 0, 2.8, 1.4, 0, Math.PI / 2);

  // The window: the sky beyond, a deep reveal, frame, glazing bars and sill.
  var skyU = { uDawn: { value: 0 }, uTime: { value: 0 } };
  var sky = new THREE.Mesh(new THREE.PlaneGeometry(3.2, 2.6), new THREE.ShaderMaterial({ uniforms: skyU, vertexShader: SKY_VS, fragmentShader: SKY_FS, fog: false }));
  sky.position.set(0, 1.45, WIN.z - 0.9);
  world.add(sky);
  var FR = '#24160d', fr = [], wx = (WIN.x0 + WIN.x1) / 2, ww = WIN.x1 - WIN.x0, wh = WIN.y1 - WIN.y0, wy = (WIN.y0 + WIN.y1) / 2;
  function bar(w, h, d, x, y, z) { fr.push(tinted(new THREE.BoxGeometry(w, h, d).translate(x, y, z), FR)); }
  bar(0.07, wh + 0.07, 0.12, WIN.x0, wy, WIN.z - 0.02);
  bar(0.07, wh + 0.07, 0.12, WIN.x1, wy, WIN.z - 0.02);
  bar(ww + 0.07, 0.07, 0.12, wx, WIN.y1, WIN.z - 0.02);
  bar(ww + 0.07, 0.06, 0.12, wx, WIN.y0, WIN.z - 0.02);
  bar(0.03, wh, 0.05, wx, wy, WIN.z - 0.06);
  [1 / 3, 2 / 3].forEach(function (f) { bar(ww, 0.028, 0.05, wx, WIN.y0 + wh * f, WIN.z - 0.06); });
  bar(ww + 0.22, 0.035, 0.2, wx, WIN.y0 - 0.045, WIN.z + 0.06);
  [WIN.x0 - 0.035, WIN.x1 + 0.035].forEach(function (x) {                // the reveal, deep in the wall
    fr.push(tinted(new THREE.BoxGeometry(0.01, wh, 0.28).translate(x, wy, WIN.z - 0.16), '#1c120b'));
  });
  var winFrame = new THREE.Mesh(merge(fr), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6 }));
  world.add(winFrame);

  // The shelves either side of the window, full of books.
  var shelfParts = [], SH_Y = [0.78, 1.18, 1.58, 1.98, 2.38];
  [-1, 1].forEach(function (s) {
    var xin = s * 0.62, xout = s * 2.55, cx = (xin + xout) / 2, w = Math.abs(xout - xin);
    [xin, xout].forEach(function (x) { shelfParts.push(tinted(new THREE.BoxGeometry(0.035, 2.1, 0.32).translate(x, 1.65, WIN.z + 0.16), '#3a2416')); });
    SH_Y.forEach(function (y) { shelfParts.push(tinted(new THREE.BoxGeometry(w, 0.03, 0.32).translate(cx, y - 0.015, WIN.z + 0.16), '#3a2416')); });
  });
  world.add(new THREE.Mesh(merge(shelfParts), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.65 })));
  var bookGeo = new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0);
  var books = new THREE.InstancedMesh(bookGeo, new THREE.MeshStandardMaterial({ map: spineTexture(), roughness: 0.72 }), 460);
  var bm = new THREE.Matrix4(), bq = new THREE.Quaternion(), bs = new THREE.Vector3(), bp = new THREE.Vector3(), bc = new THREE.Color(), nb = 0;
  var AX_Z = new THREE.Vector3(0, 0, 1);
  var HUES = [[0.0, 0.55, 0.2], [0.02, 0.45, 0.16], [0.07, 0.4, 0.2], [0.08, 0.3, 0.12], [0.33, 0.3, 0.12], [0.6, 0.3, 0.13], [0.1, 0.25, 0.3]];
  [-1, 1].forEach(function (s) {
    SH_Y.slice(0, 4).forEach(function (y) {
      var x = 0.645;
      while (x < 2.52 && nb < 460) {
        var w = 0.024 + r() * 0.035, h = 0.22 + r() * 0.12, d = 0.18 + r() * 0.08;
        if (r() < 0.05) { x += 0.04 + r() * 0.08; continue; }        // a gap
        if (x + w > 2.52) break;
        var lean = r() < 0.06 ? (r() < 0.5 ? -1 : 1) * 0.2 : 0;
        bp.set(s * (x + w / 2), y, WIN.z + 0.03 + d / 2);
        bq.setFromAxisAngle(AX_Z, lean);
        bs.set(w, h, d);
        books.setMatrixAt(nb, bm.compose(bp, bq, bs));
        var hu = HUES[Math.floor(r() * HUES.length)];
        bc.setHSL(hu[0] + (r() - 0.5) * 0.03, hu[1], hu[2] * (0.75 + r() * 0.5));
        books.setColorAt(nb, bc);
        nb++;
        x += w + 0.002 + Math.abs(lean) * 0.1;
      }
    });
  });
  books.count = nb;
  world.add(books);

  // ── The desk and what is on it ──
  var deskTex = woodTexture(r, '#3a2214');
  deskTex.wrapS = deskTex.wrapT = THREE.RepeatWrapping;
  var desk = new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.05, 1.45),
    new THREE.MeshStandardMaterial({ map: deskTex, roughness: 0.62, envMapIntensity: 0.5 }));
  desk.position.set(0, DESK - 0.025, 0.22);
  world.add(desk);

  var pageTex = pageTexture(r);
  var bookL = openBook(pageTex, '#4a1612'), bookR = openBook(pageTex, '#1f2a1f');
  bookL.position.set(-0.92, DESK, 0.16);
  bookL.rotation.y = 0.26;
  bookR.position.set(0.9, DESK, 0.2);
  bookR.rotation.y = -0.22;
  world.add(bookL, bookR);
  bookL.updateMatrixWorld();
  bookR.updateMatrixWorld();

  // A stack of closed books and an inkwell.
  var props = [];
  [['#3b1d14', 0.3, 0.05, 0.22, 0.0], ['#20281e', 0.27, 0.045, 0.2, 0.15], ['#5a3a1c', 0.24, 0.04, 0.18, -0.12]].forEach(function (b, n) {
    var y = DESK + [0, 0.05, 0.095][n];
    props.push(tinted(new THREE.BoxGeometry(b[1], b[2], b[3]).rotateY(b[4]).translate(1.2, y + b[2] / 2, -0.24), b[0]));
    props.push(tinted(new THREE.BoxGeometry(b[1] * 0.97, b[2] * 0.8, b[3] * 0.94).rotateY(b[4]).translate(1.205, y + b[2] / 2, -0.236), '#cdbd98'));
  });
  props.push(tinted(new THREE.CylinderGeometry(0.03, 0.036, 0.05, 16).translate(-0.7, DESK + 0.025, 0.36), '#0e1418'));
  props.push(tinted(new THREE.CylinderGeometry(0.016, 0.02, 0.012, 16).translate(-0.7, DESK + 0.056, 0.36), '#8a6a34'));
  world.add(new THREE.Mesh(merge(props), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6 })));

  // ── The lanterns ──
  var brassGeo = lanternBrass();
  var brassMat = new THREE.MeshStandardMaterial({ vertexColors: true, metalness: 0.85, roughness: 0.32 });
  var candleMat = new THREE.MeshStandardMaterial({ color: '#efe4cc', roughness: 0.55, emissive: '#ff9a50', emissiveIntensity: 0 });
  var candleGeo = new THREE.CylinderGeometry(0.024, 0.025, 0.058, 24).translate(0, 0.08, 0);
  var wickGeo = new THREE.CylinderGeometry(0.0016, 0.0016, 0.012, 5).translate(0, 0.114, 0);
  var cageGeo = new THREE.CylinderGeometry(0.086, 0.086, 0.29, 56, 1, true).translate(0, 0.175, 0);
  var chimGeo = new THREE.CylinderGeometry(0.056, 0.056, 0.2, 36, 1, true).translate(0, 0.15, 0);
  var flameGeo = new THREE.PlaneGeometry(2, 1, 1, 1).translate(0, 0.5, 0);
  var quadGeo = new THREE.PlaneGeometry(1, 1);
  function glow(color, dir) {
    var u = { uCol: { value: new THREE.Color(color) }, uI: { value: 1 }, uR: { value: 0.1 }, uReach: { value: 0 }, uDir: { value: dir },
              uSize: { value: new THREE.Vector2(1, 1) } };
    var m = new THREE.Mesh(quadGeo, new THREE.ShaderMaterial({ uniforms: u, vertexShader: GLOW_VS, fragmentShader: GLOW_FS,
      transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending, fog: false }));
    m.frustumCulled = false;
    m.userData.u = u;
    return m;
  }
  var WARM = new THREE.Color('#ffb46a');
  var FA = new THREE.Vector3(LANT_A.x, LANT_A.y + FLAME_Y, LANT_A.z), FB = new THREE.Vector3(LANT_B.x, LANT_B.y + FLAME_Y, LANT_B.z);
  var lanterns = [LANT_A, LANT_B].map(function (P, n) {
    var g = new THREE.Group();
    g.position.copy(P);
    g.add(new THREE.Mesh(brassGeo, brassMat));
    var wax = candleMat.clone();
    g.add(new THREE.Mesh(candleGeo, wax));
    g.add(new THREE.Mesh(wickGeo, new THREE.MeshBasicMaterial({ color: '#111' })));
    var cu = { uFlame: { value: n ? FB : FA }, uI: { value: 1 }, uWarm: { value: WARM }, uAmb: { value: new THREE.Color(0, 0, 0) } };
    var back = new THREE.Mesh(cageGeo, new THREE.ShaderMaterial({ uniforms: cu, vertexShader: CAGE_VS, fragmentShader: CAGE_FS,
      transparent: true, depthWrite: false, side: THREE.BackSide, fog: false }));
    var front = new THREE.Mesh(cageGeo, new THREE.ShaderMaterial({ uniforms: cu, vertexShader: CAGE_VS, fragmentShader: CAGE_FS,
      transparent: true, depthWrite: false, side: THREE.FrontSide, fog: false }));
    back.renderOrder = 1;
    front.renderOrder = 4;
    g.add(back, front);
    var chim = new THREE.Mesh(chimGeo, glassMaterial(0.55));
    chim.renderOrder = 2;
    g.add(chim);
    var fu = { uW: { value: 0.048 }, uH: { value: 0.12 }, uLeanDir: { value: new THREE.Vector3(n ? -1 : 1, 0, 0) }, uTime: { value: 0 },
               uSeed: { value: n * 7.3 + 1.1 }, uLean: { value: 0 }, uI: { value: 1 }, uFaint: { value: 0 }, uFlick: { value: 1 } };
    var flame = new THREE.Mesh(flameGeo, new THREE.ShaderMaterial({ uniforms: fu, vertexShader: FLAME_VS, fragmentShader: FLAME_FS,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false }));
    flame.position.y = 0.103;
    flame.renderOrder = 3;
    flame.frustumCulled = false;
    g.add(flame);
    var halo = glow('#ffa850', n ? -1 : 1);
    halo.position.y = FLAME_Y;
    halo.renderOrder = 8;
    g.add(halo);
    var light = new THREE.PointLight('#ffb46a', 1, 0, 2);
    light.position.y = FLAME_Y + 0.06;
    g.add(light);
    world.add(g);
    return { g: g, cu: cu, fu: fu, halo: halo, light: light, chim: chim, wax: wax, pos: cu.uFlame.value };
  });
  var MID = new THREE.Vector3().addVectors(FA, FB).multiplyScalar(0.5);

  function glassMaterial(op) {
    return new THREE.ShaderMaterial({
      uniforms: { uA: { value: FA }, uB: { value: FB }, uIA: { value: 1 }, uIB: { value: 1 }, uWarm: { value: WARM },
                  uAmb: { value: new THREE.Color('#1a2030') }, uOp: { value: op } },
      vertexShader: GLASS_VS, fragmentShader: GLASS_FS, transparent: true, depthWrite: false, side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending, fog: false
    });
  }

  // ── The beams that reach between the flames, and the merged bloom ──
  var stripGeo = new THREE.PlaneGeometry(1, 1, 24, 1);
  var beamU = { uP0: { value: FA }, uP1: { value: FB }, uW0: { value: 0.16 }, uW1: { value: 0.16 }, uReach: { value: 0 },
                uMerge: { value: 0 }, uTime: { value: 0 }, uAlpha: { value: 0 } };
  var beam = new THREE.Mesh(stripGeo, new THREE.ShaderMaterial({ uniforms: beamU, vertexShader: AXIS_VS, fragmentShader: BEAM_FS,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false }));
  beam.frustumCulled = false;
  beam.renderOrder = 6;
  world.add(beam);
  var pool = glow('#ffb468', 1);
  pool.renderOrder = 8;
  world.add(pool);
  var poolLight = new THREE.PointLight('#ffb070', 0, 0, 2);
  world.add(poolLight);

  // ── The jar of earth, the seed, its roots, the stem and the rose ──
  var soil = new THREE.Mesh(new THREE.CylinderGeometry(SOIL_R, SOIL_R * 0.97, SOIL_H, 40).translate(0, SOIL_H / 2 + 0.006, 0),
    new THREE.MeshStandardMaterial({ map: soilTexture(r), roughness: 1 }));
  soil.material.map.repeat.set(3, 1);
  soil.position.copy(JAR);
  world.add(soil);
  var jarPts = [[0, 0.002], [0.058, 0.002], [0.068, 0.012], [0.069, 0.128], [0.064, 0.14], [0.062, 0.148], [0.066, 0.153], [0.064, 0.157]]
    .map(function (p) { return new THREE.Vector2(p[0], p[1]); });
  var jar = new THREE.Mesh(new THREE.LatheGeometry(jarPts, 56), glassMaterial(0.9));
  jar.position.copy(JAR);
  jar.renderOrder = 2;
  world.add(jar);

  var seedMat = new THREE.MeshStandardMaterial({ color: '#6b4a2b', roughness: 0.55, side: THREE.DoubleSide });
  var seedTop = new THREE.Mesh(new THREE.SphereGeometry(0.0085, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2).scale(1.35, 0.72, 0.9), seedMat);
  var seedBot = new THREE.Mesh(new THREE.SphereGeometry(0.0085, 16, 8, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2).scale(1.35, 0.72, 0.9), seedMat);
  var seedG = new THREE.Group();
  seedG.position.copy(SEED);
  seedG.add(seedTop, seedBot);
  world.add(seedG);

  var rootMat = growMaterial('#d8c9a8', '#ffb35c', -1), ru = rootMat.userData.uniforms;
  var roots = new THREE.Mesh(growTubes(makeRoots(r)), rootMat);
  world.add(roots);

  var soilTop = DESK + SOIL_H + 0.006, RS = SOIL_R + 0.0012;
  var stemCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, SEED.y + 0.004, JAR.z + RS),
    new THREE.Vector3(0.002, soilTop - 0.004, JAR.z + RS - 0.002),
    new THREE.Vector3(0.0, soilTop + 0.03, JAR.z + 0.03),
    new THREE.Vector3(-0.006, soilTop + 0.1, JAR.z + 0.008),
    new THREE.Vector3(0.006, soilTop + 0.18, JAR.z),
    new THREE.Vector3(0.0, soilTop + 0.255, JAR.z + 0.004)
  ]);
  var stemGeo = new THREE.TubeGeometry(stemCurve, 80, 0.0032, 6, false);
  var stemT = new Float32Array(stemGeo.attributes.uv.count);
  for (i = 0; i < stemT.length; i++) stemT[i] = stemGeo.attributes.uv.getX(i);
  stemGeo.setAttribute('aT', new THREE.BufferAttribute(stemT, 1));
  var stemMat = growMaterial('#24501c', '#ffc070', 1), su = stemMat.userData.uniforms;
  world.add(new THREE.Mesh(stemGeo, stemMat));

  // Leaves along the stem: two seed leaves, then three true leaves.
  var leafMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, side: THREE.DoubleSide });
  var LEAVES = [[0.24, 0.0, 0.9, 0.016, 0.022, 0], [0.24, Math.PI, 0.9, 0.016, 0.022, 0],
                [0.52, 0.6, 1.0, 0.024, 0.05, 1], [0.66, 0.6 + Math.PI * 0.9, 0.95, 0.022, 0.046, 1], [0.8, 2.6, 0.8, 0.018, 0.038, 1]];
  var leafGeos = [leafGeometry('#4f8a35', 0), leafGeometry('#2f5a24', 1)];
  var leaves = LEAVES.map(function (L) {
    var m = new THREE.Mesh(leafGeos[L[5]], leafMat);
    stemCurve.getPointAt(L[0], m.position);
    m.rotation.set(0, L[1], 0, 'YXZ');
    m.rotateX(L[2]);
    m.userData.s = [L[3], L[4]];
    m.scale.setScalar(0.0001);
    world.add(m);
    return m;
  });

  // The rose head: petals in a phyllotaxis spiral, sepals beneath.
  var NP = small ? 16 : 22;
  var head = new THREE.Group();
  stemCurve.getPointAt(1, head.position);
  head.rotation.x = 0.32;
  world.add(head);
  var petalMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.5, side: THREE.DoubleSide, emissive: '#5a0612', emissiveIntensity: 0 });
  var petalGeo = petalGeometry();
  var petals = new THREE.InstancedMesh(petalGeo, petalMat, NP);
  petals.frustumCulled = false;
  head.add(petals);
  var sepals = new THREE.InstancedMesh(leafGeometry('#2f5a24', 1), leafMat, 5);
  head.add(sepals);
  var fallen = new THREE.Mesh(petalGeo, petalMat);
  fallen.visible = false;
  world.add(fallen);
  var FALL_I = NP - 2;                                    // the petal that falls

  var pm4 = new THREE.Matrix4(), pq = new THREE.Quaternion(), pqx = new THREE.Quaternion(), pp = new THREE.Vector3(), ps = new THREE.Vector3();
  var AX_X = new THREE.Vector3(1, 0, 0);
  function petalPose(n, open, trem, time, outP, outQ, outS) {
    var k = n / (NP - 1), phi = n * 2.39996;
    var o = clamp((open - (1 - k) * 0.42) / 0.58, 0, 1);
    o = o * o * (3 - 2 * o);
    var tilt = lerp(0.04 + 0.22 * k, 0.3 + 1.2 * Math.pow(k, 1.15), o) + trem * Math.sin(time * 31 + n * 2.1) * 0.06;
    var rad = 0.002 + 0.0085 * k + 0.004 * k * o;
    outP.set(Math.sin(phi) * rad, -0.007 * k, Math.cos(phi) * rad);
    outQ.setFromAxisAngle(UP, phi);
    pqx.setFromAxisAngle(AX_X, tilt);
    outQ.multiply(pqx);
    outS.setScalar(0.5 + 0.62 * k);
  }

  // ── Words in the air ──
  var A = atlas();
  var NW = small ? 150 : 300;
  var wordGeo = new THREE.InstancedBufferGeometry();
  var quad = new THREE.PlaneGeometry(1, 1);
  wordGeo.index = quad.index;
  wordGeo.setAttribute('position', quad.attributes.position);
  var aHome = new Float32Array(NW * 3), aSrc = new Float32Array(NW * 3), aRect = new Float32Array(NW * 4), aInfo = new Float32Array(NW * 4);
  var tv = new THREE.Vector3();
  for (i = 0; i < NW; i++) {
    var kind = r(), key;
    if (kind < 0.34) key = '?' + Math.floor(r() * 3);
    else if (kind < 0.62) key = 'l:' + LETTERS[Math.floor(r() * LETTERS.length)];
    else key = 'w:' + WORDS[Math.floor(r() * WORDS.length)];
    var g = A.rects[key];
    (r() < 0.5 ? bookL : bookR).userData.pagePoint(r, tv);
    aSrc.set([tv.x, tv.y, tv.z], i * 3);
    var hx = (r() - 0.5) * 3.0, hz = -0.55 + r() * 1.3, hy = 0.86 + Math.pow(r(), 0.8) * 1.0;
    aHome.set([hx, hy, hz], i * 3);
    aRect.set(g.rect, i * 4);
    var hgt = (key[0] === '?' ? 0.05 : key[0] === 'l' ? 0.04 : 0.032) * (0.7 + r() * 0.6);
    aInfo.set([hgt, g.ratio, r() * 0.78, r()], i * 4);
  }
  wordGeo.setAttribute('aHome', new THREE.InstancedBufferAttribute(aHome, 3));
  wordGeo.setAttribute('aSrc', new THREE.InstancedBufferAttribute(aSrc, 3));
  wordGeo.setAttribute('aRect', new THREE.InstancedBufferAttribute(aRect, 4));
  wordGeo.setAttribute('aInfo', new THREE.InstancedBufferAttribute(aInfo, 4));
  wordGeo.instanceCount = NW;
  var wordU = { uAtlas: { value: A.tex }, uTime: { value: 0 }, uEmerge: { value: 0 }, uSink: { value: 0 }, uPart: { value: 0 },
                uWob: { value: env.reduceMotion ? 0.3 : 1 }, uA: { value: FA }, uB: { value: FB }, uIA: { value: 1 }, uIB: { value: 1 },
                uAlpha: { value: 1 } };
  var words = new THREE.Mesh(wordGeo, new THREE.ShaderMaterial({ uniforms: wordU, vertexShader: WORD_VS, fragmentShader: WORD_FS,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false }));
  words.frustumCulled = false;
  words.renderOrder = 5;
  world.add(words);

  // "If" and "Why", and the sparks they crumble into.
  var ASKS = [{ key: 'If', from: new THREE.Vector3(-0.97, DESK + 0.04, 0.12), to: new THREE.Vector3(-0.17, 1.36, -0.08), tilt: 0.06 },
              { key: 'Why', from: new THREE.Vector3(-0.86, DESK + 0.04, 0.2), to: new THREE.Vector3(0.17, 1.3, -0.04), tilt: -0.05 }];
  var NS = small ? 180 : 420, ASK_H = 0.13;
  ASKS.forEach(function (q) {
    var gr = A.rects[q.key], size = new THREE.Vector2(ASK_H * gr.ratio, ASK_H);
    q.u = { uAtlas: { value: A.tex }, uRect: { value: new THREE.Vector4().fromArray(gr.rect) }, uC: { value: 0 }, uAlpha: { value: 0 },
            uI: { value: 1 }, uSize: { value: size }, uTilt: { value: q.tilt } };
    q.mesh = new THREE.Mesh(quad, new THREE.ShaderMaterial({ uniforms: q.u, vertexShader: ASK_VS, fragmentShader: ASK_FS,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false }));
    q.mesh.frustumCulled = false;
    q.mesh.renderOrder = 5;
    world.add(q.mesh);
    var pts = glyphPoints(A, q.key, NS, r), off = new Float32Array(pts.length * 2), inf = new Float32Array(pts.length * 2);
    pts.forEach(function (p, n) {
      off[n * 2] = p[0] * size.x; off[n * 2 + 1] = p[1] * size.y;
      inf[n * 2] = 0.65 * (p[0] + 0.5) + 0.35 * r();
      inf[n * 2 + 1] = r();
    });
    var sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pts.length * 3), 3));
    sg.setAttribute('aOff', new THREE.BufferAttribute(off, 2));
    sg.setAttribute('aInfo', new THREE.BufferAttribute(inf, 2));
    q.su = { uC: { value: 0 }, uTime: { value: 0 }, uScale: { value: 1 } };
    q.sparks = new THREE.Points(sg, new THREE.ShaderMaterial({ uniforms: q.su, vertexShader: SPARK_VS, fragmentShader: SPARK_FS,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false }));
    q.sparks.frustumCulled = false;
    q.sparks.renderOrder = 6;
    world.add(q.sparks);
  });

  // The spark that carries the flame across, with a short trail.
  var NT = 10, trailPos = new Float32Array(NT * 3), trailCol = new Float32Array(NT * 3);
  var trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
  trailGeo.setAttribute('color', new THREE.BufferAttribute(trailCol, 3));
  var trail = new THREE.Points(trailGeo, new THREE.PointsMaterial({ size: 0.06, map: softSprite('rgba(255,230,180,1)', 'rgba(255,160,80,0)'),
    vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false }));
  trail.frustumCulled = false;
  trail.renderOrder = 7;
  world.add(trail);
  var sparkHead = glow('#ffd08a', 1);
  sparkHead.renderOrder = 8;
  world.add(sparkHead);
  var ARC = new THREE.QuadraticBezierCurve3(FB.clone(), new THREE.Vector3(0, FB.y + 0.32, FB.z + 0.1), FA.clone());

  // ── Light: the candles, a dim warm fill, and the dawn through the window ──
  var hemi = new THREE.HemisphereLight('#3a2a20', '#0a0604', 0.12);
  var dawnLight = new THREE.DirectionalLight('#ffc9a0', 0);
  dawnLight.position.set(0.5, 2.6, -3.0);
  dawnLight.target.position.set(0, DESK, 0.1);
  world.add(hemi, dawnLight, dawnLight.target);
  var shaftU = { uP0: { value: new THREE.Vector3(0.05, 1.55, WIN.z - 0.05) }, uP1: { value: new THREE.Vector3(-0.05, DESK, 0.55) },
                 uW0: { value: 0.95 }, uW1: { value: 1.5 }, uAlpha: { value: 0 }, uTime: { value: 0 } };
  var shaft = new THREE.Mesh(stripGeo, new THREE.ShaderMaterial({ uniforms: shaftU, vertexShader: AXIS_VS, fragmentShader: SHAFT_FS,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false }));
  shaft.frustumCulled = false;
  shaft.renderOrder = 6;
  world.add(shaft);
  var motes = particleField({ count: small ? 120 : 260, box: [1.6, 1.4, 1.6], fall: [-0.01, 0.02], size: 0.006,
    color: '#ffe0b8', map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)'), sway: 0.03, windSpeed: 0.1 });
  motes.points.material.blending = THREE.AdditiveBlending;
  world.add(motes.points);
  var moteC = new THREE.Vector3(0, 0.9, -0.1), pf = { snow: 0, wind: 0, dt: 0, time: 0 };

  // ── Per-frame state ──
  var W = 1, H = 1, portrait = false;
  var target = new THREE.Vector3(), tmpV = new THREE.Vector3(), tmpQ = new THREE.Quaternion(), restQ = new THREE.Quaternion();
  var fallP0 = new THREE.Vector3(), fallQ0 = new THREE.Quaternion(), headM = new THREE.Matrix4();
  var AMB_NIGHT = new THREE.Color('#0c1220'), AMB_DAWN = new THREE.Color('#5a3a30');
  var FOG_NIGHT = new THREE.Color('#070403'), FOG_DAWN = new THREE.Color('#2a1810');
  var HEMI_NIGHT = new THREE.Color('#3a2a20'), HEMI_DAWN = new THREE.Color('#ffcaa0');
  restQ.setFromAxisAngle(AX_X, -Math.PI / 2 + 0.25).premultiply(tmpQ.setFromAxisAngle(UP, 0.8));

  function pulse(t) {
    var p = t % 0.92;
    return Math.exp(-Math.pow(p / 0.06, 2)) + 0.6 * Math.exp(-Math.pow((p - 0.21) / 0.06, 2));
  }

  function frame(f) {
    var row = f.row, time = f.time, slow = env.reduceMotion;
    var lamp = row[K.lamp], dawn = row[K.dawn], strain = row[K.strain], mrg = row[K.merge], seed = row[K.seed];
    var grow = row[K.grow], open = row[K.open], thrill = row[K.thrill], faint = row[K.faint], spark = row[K.spark];

    // ── Camera: an orbit round the target, the subject set beside the verse ──
    var dist = row[K.dist] * (portrait ? row[K.pd] : 1), az = row[K.az] - f.mx * 0.05, el = row[K.el] + f.my * 0.025;
    target.set(row[K.tx], row[K.ty] + (portrait ? row[K.pty] : 0), row[K.tz]);
    camera.position.set(Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)).multiplyScalar(dist).add(target);
    camera.lookAt(target);
    if (portrait) camera.setViewOffset(W, H, 0, -H * row[K.py], W, H);
    else camera.setViewOffset(W, H, -row[K.side] * W, 0, W, H);

    // ── The flames ──
    var flick = slow ? 0.3 : 1;
    var fa = faint, beat = pulse(time) * thrill;
    var IA = lamp * (1 - 0.85 * fa) * (0.92 + 0.08 * Math.sin(time * 11.3) * flick);
    var IB = lamp * (1 + 0.1 * mrg * (1 - smooth(0, 0.4, seed))) * (0.92 + 0.08 * Math.sin(time * 12.7 + 1.3) * flick);
    var relit = 4 * fa * (1 - fa) * smooth(0.7, 1, spark);            // the flare as it catches again
    IA += relit * 0.8;
    var lean = row[K.lean] * 0.75 + strain * 0.25;
    for (k = 0; k < 2; k++) {
      var L = lanterns[k], I = k ? IB : IA, fn = k ? 0 : fa;
      L.fu.uTime.value = time;
      L.fu.uFlick.value = flick;
      L.fu.uLean.value = lean * (1 - fn * 0.7);
      L.fu.uI.value = Math.min(I * 1.1, 1.6) + 0.15 * fn;
      L.fu.uFaint.value = fn;
      L.fu.uW.value = 0.048 * (1 - 0.5 * fn);
      L.fu.uH.value = 0.12 * (1 - 0.72 * fn);
      L.cu.uI.value = I;
      L.cu.uAmb.value.copy(AMB_NIGHT).lerp(AMB_DAWN, dawn).multiplyScalar(0.12);
      L.light.intensity = 0.5 * I;
      L.wax.emissiveIntensity = 0.25 * I;
      // The glow, straining towards the other flame through the mesh.
      var reach = strain * (1 - mrg), hu = L.halo.userData.u;
      hu.uR.value = 0.075 * (0.55 + 0.45 * I) * (1 + 0.25 * strain);
      hu.uReach.value = reach;
      hu.uSize.value.set(hu.uR.value * 2.8 * (1 + 2.6 * reach) * 2, hu.uR.value * 2.8);
      hu.uI.value = I * (1 - dawn * 0.3) * (1 + 0.3 * reach);
      L.chim.material.uniforms.uIA.value = IA;
      L.chim.material.uniforms.uIB.value = IB;
    }
    beamU.uReach.value = 0.5 * strain;
    beamU.uMerge.value = mrg;
    beamU.uTime.value = slow ? 0 : time;
    beamU.uAlpha.value = smooth(0, 0.15, strain) * (1 - smooth(0.45, 1, mrg) * 0.85) * (1 - smooth(0, 0.25, seed));
    beam.visible = beamU.uAlpha.value > 0.003;

    // The merged bloom, which then sinks into the seed.
    var sinkIn = smooth(0, 0.3, seed), bump = 4 * mrg * (1 - mrg);
    pool.position.lerpVectors(MID, SEED, sinkIn);
    var pa = mrg * (0.9 + 1.3 * bump) * (1 - smooth(0.18, 0.4, seed));
    var pu = pool.userData.u;
    pu.uI.value = Math.min(pa, 2);
    pu.uR.value = (0.08 + 0.24 * mrg + 0.1 * bump) * (1 - 0.75 * sinkIn);
    pu.uSize.value.set(pu.uR.value * 2.8, pu.uR.value * 2.8);
    pool.visible = pa > 0.003;
    poolLight.position.copy(pool.position);
    poolLight.intensity = pa * 0.6 + smooth(0.1, 0.6, seed) * (1 - grow * 0.6) * 0.03;

    // ── Words ──
    wordU.uTime.value = time;
    wordU.uEmerge.value = row[K.words];
    wordU.uSink.value = row[K.sink];
    wordU.uPart.value = row[K.part];
    wordU.uIA.value = IA;
    wordU.uIB.value = IB;
    words.visible = row[K.words] > 0.001 && row[K.sink] < 0.999;

    // "If" and "Why" rise, then crumble.
    var ask = row[K.ask], cr = row[K.crumble];
    for (k = 0; k < 2; k++) {
      var q = ASKS[k], a = smooth(k * 0.15, 0.85 + k * 0.15, ask);
      q.mesh.position.lerpVectors(q.from, q.to, a);
      q.mesh.position.y += Math.sin(a * Math.PI) * 0.08 + (slow ? 0 : Math.sin(time * 0.7 + k * 2) * 0.008) * a;
      q.sparks.position.copy(q.mesh.position);
      var c = clamp(cr * 1.25 - k * 0.25, 0, 1);
      q.u.uC.value = c;
      q.u.uAlpha.value = smooth(0, 0.25, a);
      q.u.uI.value = 0.9 + 0.5 * lamp;
      q.mesh.visible = a > 0.001 && c < 0.7;
      q.su.uC.value = c;
      q.su.uTime.value = time;
      q.su.uScale.value = gl.getPixelRatio() * H * 0.012;
      q.sparks.visible = c > 0.001 && c < 0.999;
    }

    // ── The seed, roots, stem, leaves and rose ──
    var split = smooth(0.05, 0.35, seed);
    seedTop.rotation.z = split * 0.55;
    seedTop.position.set(-0.003 * split, 0.002 * split, 0);
    seedBot.rotation.z = -split * 0.35;
    var rootG = smooth(0.15, 1, seed);
    ru.uGrow.value = rootG * 1.02;
    ru.uTime.value = slow ? 0 : time;
    ru.uPulse.value = smooth(0.2, 0.5, seed) * (1 - grow * 0.4) * 1.4 + thrill * 0.6;
    ru.uBeat.value = beat * 0.5;
    roots.visible = rootG > 0.001;
    var stemG = smooth(0.25, 1, seed) * 0.12 + grow * 0.88;
    su.uGrow.value = stemG;
    su.uTime.value = slow ? 0 : time;
    su.uPulse.value = smooth(0.3, 0.6, seed) * (1 - smooth(0.6, 1, grow) * 0.85) * 1.1 + thrill * 0.35;
    su.uBeat.value = beat * 0.25;
    for (k = 0; k < leaves.length; k++) {
      var lf = leaves[k], ls = smooth(LEAVES[k][0] - 0.02, LEAVES[k][0] + 0.16, stemG);
      lf.scale.set(lf.userData.s[0] * ls, lf.userData.s[1] * ls, lf.userData.s[0] * ls);
      lf.visible = ls > 0.001;
    }
    var bud = row[K.bud];
    head.visible = bud > 0.001;
    head.scale.setScalar(0.25 + 0.75 * bud);
    head.rotation.z = (slow ? 0 : Math.sin(time * 26) * 0.012 * thrill) + Math.sin(time * 0.6) * 0.01;
    var fall = row[K.fall], trem = thrill * (slow ? 0.3 : 1);
    for (i = 0; i < NP; i++) {
      petalPose(i, open, trem, time, pp, pq, ps);
      if (i === FALL_I && fall > 0) ps.setScalar(0.00001);
      petals.setMatrixAt(i, pm4.compose(pp, pq, ps));
    }
    petals.instanceMatrix.needsUpdate = true;
    for (i = 0; i < 5; i++) {
      var sa = i / 5 * Math.PI * 2 + 0.3;
      pq.setFromAxisAngle(UP, sa);
      pqx.setFromAxisAngle(AX_X, lerp(0.35, 1.9, smooth(0, 0.6, open)));
      pq.multiply(pqx);
      pp.set(Math.sin(sa) * 0.004, -0.008, Math.cos(sa) * 0.004);
      ps.set(0.012, 0.026, 0.012);
      sepals.setMatrixAt(i, pm4.compose(pp, pq, ps));
    }
    sepals.instanceMatrix.needsUpdate = true;
    petalMat.emissiveIntensity = beat * 0.5 + 0.06 * bud;
    // A petal falls, turning, and comes to rest on the desk.
    fallen.visible = fall > 0 && bud > 0.5;
    if (fallen.visible) {
      petalPose(FALL_I, open, 0, time, pp, pq, ps);
      head.updateMatrixWorld();
      headM.compose(pp, pq, ps).premultiply(head.matrixWorld);
      headM.decompose(fallP0, fallQ0, tmpV);
      var e = fall * fall * (3 - 2 * fall);
      fallen.position.lerpVectors(fallP0, PETAL_REST, e);
      fallen.position.y = lerp(fallP0.y, PETAL_REST.y, fall * (2 - fall) * 0.4 + e * 0.6);
      fallen.position.x += Math.sin(fall * 9) * 0.03 * (1 - fall);
      fallen.position.z += Math.cos(fall * 7) * 0.02 * (1 - fall);
      fallen.quaternion.slerpQuaternions(fallQ0, restQ, smooth(0, 0.95, fall));
      tmpQ.setFromAxisAngle(UP, Math.sin(fall * 6) * 1.2 * (1 - fall));
      fallen.quaternion.premultiply(tmpQ);
      fallen.scale.copy(tmpV);
    }

    // ── The spark that carries the flame across ──
    var sv = spark > 0.001 && spark < 0.999;
    trail.visible = sparkHead.visible = sv;
    if (sv) {
      ARC.getPoint(spark, sparkHead.position);
      var sh = sparkHead.userData.u;
      sh.uR.value = 0.05;
      sh.uSize.value.set(0.14, 0.14);
      sh.uI.value = 1.3 * smooth(0, 0.08, spark) * (1 - smooth(0.92, 1, spark)) * (0.85 + 0.15 * Math.sin(time * 23));
      for (i = 0; i < NT; i++) {
        var st = clamp(spark - i * 0.018, 0, 1);
        ARC.getPoint(st, tmpV);
        trailPos[i * 3] = tmpV.x; trailPos[i * 3 + 1] = tmpV.y; trailPos[i * 3 + 2] = tmpV.z;
        var fd = (1 - i / NT) * smooth(0, 0.06, spark) * (1 - smooth(0.94, 1, spark)) * (st > 0 ? 1 : 0);
        trailCol[i * 3] = fd; trailCol[i * 3 + 1] = fd * 0.85; trailCol[i * 3 + 2] = fd * 0.6;
      }
      trailGeo.attributes.position.needsUpdate = true;
      trailGeo.attributes.color.needsUpdate = true;
    }

    // ── The night, and the dawn ──
    skyU.uDawn.value = dawn;
    skyU.uTime.value = time;
    dawnLight.intensity = dawn * dawn * 1.6;
    hemi.intensity = 0.1 + dawn * 0.5;
    hemi.color.copy(HEMI_NIGHT).lerp(HEMI_DAWN, dawn);
    world.environmentIntensity = 0.35 + 0.15 * lamp + dawn * 0.4;
    shaftU.uAlpha.value = dawn * dawn * 0.07;
    shaftU.uTime.value = time;
    shaft.visible = dawn > 0.02;
    world.fog.color.copy(FOG_NIGHT).lerp(FOG_DAWN, dawn * 0.6);
    gl.setClearColor(world.fog.color);
    jar.material.uniforms.uIA.value = IA;
    jar.material.uniforms.uIB.value = IB;
    jar.material.uniforms.uAmb.value.copy(AMB_NIGHT).lerp(AMB_DAWN, dawn);

    pf.snow = row[K.motes]; pf.wind = 0; pf.dt = f.dt; pf.time = time;
    motes.update(pf, moteC, slow);
    motes.points.material.opacity = 0.25 + dawn * 0.6;

    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) {
      W = w; H = h; portrait = w < h;
      fitCamera(gl, camera, w, h, dpr, small);
    },
    frame: frame,
    destroy: function () { envRT.dispose(); quad.dispose(); disposeAll(world, gl); }
  };
}

PI.register('wordless', {
  renderer: renderer3d,
  scrim: 0.55,
  accent: '#f2c27e',
  emphasis: /^([“"](if|why)[”"]|give\W*)$/i,
  align: ['left', 'right', 'left', 'right'],
  // key() carries every column forward, so each beat lists only what changes.
  keys: function (T) {
    function at(i, d) { return T.start(Math.min(i, T.count - 1)) + d; }   // d units into panel i (0..1.6)
    var rows = [], cur = {};
    function key(u, ch) {
      for (var c in ch) cur[c] = ch[c];
      rows.push([u].concat(COLS.map(function (c) { return cur[c]; })));
    }
    key(0, { dist: 2.75, dark: 0, motes: 0.15, wind: 0.1, az: 0.1, el: 0.2, tx: 0, ty: 1.08, tz: -0.15, side: 0, py: 0.1, pd: 1.2, pty: 0,
             lamp: 0.05, words: 0, part: 0, lean: 0, sink: 0, strain: 0, merge: 0, seed: 0, grow: 0, bud: 0, open: 0,
             thrill: 0, faint: 0, fall: 0, ask: 0, crumble: 0, dawn: 0, spark: 0 });
    key(0.7, { lamp: 1, dist: 2.45, az: 0.04, words: 0.16 });                          // the lanterns come up
    // I: the air fills with questions; the flames gaze; the words sink.
    key(at(0, 0.2), { dist: 2.05, az: -0.1, el: 0.12, side: 0.2, words: 0.4, py: 0.22, pd: 1.25 });
    key(at(0, 0.55), { words: 1 });                                                     // "If questioning would make us wise"
    key(at(0, 0.62), { lean: 0, part: 0 });
    key(at(0, 0.9), { lean: 1, part: 1, dist: 1.95 });                                  // "No eyes would ever gaze in eyes"
    key(at(0, 0.95), { sink: 0 });
    key(at(0, 1.38), { sink: 1, part: 0.6, dist: 1.85, az: -0.14 });                    // "No mouths would wander each to each"
    // II: close on the mortal mesh; back to see the light strain through; the glows merge.
    key(at(1, 0.15), { side: -0.2, tx: LANT_A.x, ty: DESK + FLAME_Y + 0.01, tz: LANT_A.z, dist: 0.62, az: 0.25, el: 0.1, part: 0, lean: 0.7, py: 0.16, pd: 1.4 });
    key(at(1, 0.5), { dist: 0.68, az: 0.34 });
    key(at(1, 0.65), { strain: 0 });
    key(at(1, 0.85), { tx: 0, ty: 0.97, tz: -0.08, dist: 1.7, az: 0.2, el: 0.08, side: -0.25, py: 0.22, pd: 1.15 });
    key(at(1, 1.05), { strain: 1, lean: 1, dist: 1.66 });                                // "yearn to meet"
    key(at(1, 1.32), { merge: 1, dist: 1.6 });                                          // "ecstasy complete"
    // III: the seed, the roots, the shoot; the rose; faint; a petal falls.
    key(at(2, 0.15), { side: 0.2, tx: 0, ty: 0.86, tz: JAR.z, dist: 0.5, az: -0.42, el: 0.1, lean: 0.2, strain: 0, seed: 0, py: 0.24, pd: 1.45 });
    key(at(2, 0.55), { seed: 1, grow: 0.4, ty: 0.9, dist: 0.52 });                       // "the secret powers by which he grows"
    key(at(2, 0.9), { grow: 1, bud: 1, ty: 1.0, dist: 0.62, el: 0.14 });
    key(at(2, 1.0), { open: 0, thrill: 0, faint: 0 });
    key(at(2, 1.25), { open: 1, thrill: 1, faint: 1 });                                   // "to thrill and faint"
    key(at(2, 1.5), { fall: 1, thrill: 0, dist: 0.72, az: -0.32 });                                 // "and sweetly bleed"
    // IV: "If" and "Why" crumble; dawn; one flame gives the other life.
    key(at(3, 0.12), { side: -0.25, dist: 1.95, az: 0.12, el: 0.12, ty: 1.1, tz: -0.18, ask: 0, py: 0.26, pd: 1.15 });
    key(at(3, 0.35), { ask: 1 });
    key(at(3, 0.72), { crumble: 1, dawn: 0.3 });
    key(at(3, 0.85), { spark: 0 });
    key(at(3, 1.15), { spark: 1, faint: 1 });                                             // "life in me is what you give"
    key(at(3, 1.3), { faint: 0, dawn: 0.55 });
    key(T.total - 0.45, { side: 0.25, dist: 1.68, az: -0.02, el: 0.02, ty: 1.34, tz: -0.22, dawn: 1, motes: 0.7, py: 0, pd: 1.25, pty: 0.26 });
    key(T.total, { dist: 1.72 });
    return rows;
  },
  sound: {
    src: '/audio/wind.mp3',
    label: 'Play the quiet room and the candles',
    volume: function (row) { return 0.022 + 0.02 * row[K.dawn]; },
    cues: [
      { stanza: 0, at: 0.1, play: rustle },
      { stanza: 0, at: 0.4, play: rustle },
      { stanza: 1, at: 1.12, play: swell },
      { stanza: 2, at: 1.0, play: heartbeat },
      { stanza: 3, at: 0.4, play: crackle },
      { stanza: 3, at: 1.1, play: kindle },
      { stanza: 3, at: 1.5, play: morningChord }
    ]
  }
});
