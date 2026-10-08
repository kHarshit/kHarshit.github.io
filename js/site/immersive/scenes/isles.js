/*
 * Scene for "If You Forget Me" (Pablo Neruda), seen from inside his
 * autumn: a stone room on a rocky shore, a window on the night sea, a red
 * tree on its rock below the house and, far out, the isles. No one is seen;
 * the things he names answer the words.
 *
 * Panels (maxLines 4 splits the long stanzas into fours):
 * I     "I want you to know one thing": a dark room at night, the window
 *       full of moonlit sea, its light lying across the floor.
 * II    "the crystal moon, the red branch of the slow autumn at my window":
 *       at the window, the moon and the tree's red branch;
 *  ·    "near the fire the impalpable ash": you turn to the hearth, where
 *       the fire stirs over the ash;
 *  ·    "the wrinkled body of the log, everything carries me to you ...
 *       aromas, light, metals": motes lift from the log's embers, the steam
 *       of a cup, the moonlight and the brass on the mantel, the casement
 *       swings open and they drift out;
 *  ·    "were little boats that sail toward those isles": out over the sea
 *       they settle on the water as glowing paper boats and sail towards
 *       the isles on the horizon, and you follow them out.
 * III   "little by little you stop loving me": the boats go dark one by one
 *       and the moon wanes.
 * IV    "If suddenly you forget me": a gust; the sea roughens, leaves tear
 *       past and the isles fade into mist.
 * V     "the wind of banners that passes through my life": you turn to the
 *       shore as long pennants of wind stream by;
 *  ·    "the shore of the heart where I have roots": the red tree on its
 *       rock below the house, its roots gripping the stone;
 *  ·    "I shall lift my arms and my roots will set off to seek another
 *       land": its branches rise, its roots let go of the rock and it sets
 *       off across the water.
 * VI    "But if each day, each hour": dawn begins, and out on the water the
 *       tree stops;
 *  ·    "a flower climbs up to your lips": it comes home to its rock and a
 *       flowering vine climbs its trunk;
 *  ·    "in me all that fire is repeated": the hearth blazes in the window
 *       and sparks rise from the chimney;
 *  ·    "without leaving mine": you come back to the open window at dawn,
 *       the boats relit and sailing on towards the gold isles.
 *
 * Inside, only the fire lights the room (its materials ignore the moon and
 * the dawn, which would shine through the walls); the moonlight on the
 * floor is painted. Columns:
 *   [unit, x, y, z, yaw, pitch, fire, open, journey, dim, moon, gust, mist,
 *    banners, lift, roots, go, dawn, flower, phoneYaw, phonePitch]
 * where "journey" carries the motes (0-1 from their sources to the window,
 * 1-2 out to the water where they become boats, beyond 2 sailing on), "dim"
 * puts the boats out one by one, "lift" raises the tree's branches, "roots"
 * frees its roots and lifts it off the rock, "go" takes it out over the
 * water, and phoneYaw/phonePitch turn the view on portrait screens, whose
 * text sits mid-screen.
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, softSprite, starField, terrain,
         scatter, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout (metres; the sea lies north, -z; the room's floor is F above it) ──
var F = 3.2;
var ROOM = { x0: -2.6, x1: 2.6, z0: 0, z1: 4.6, h: 2.7 };
var WALL = 0.32;                                   // the north wall runs z = -WALL..0
var WIN = { x0: -0.72, x1: 0.72, y0: F + 0.85, y1: F + 2.25 };
var FIRE = new THREE.Vector3(2.22, F + 0.3, 2.8);  // the hearth, in the east wall
var MOON = new THREE.Vector3(0.235, 0.42, -0.876).normalize();
var DAWN = new THREE.Vector3(-0.42, 0.04, -0.906).normalize();
var ROCK = { x: -7, y: 0.1, z: -8.6, rx: 2.4, ry: 2.0, rz: 2.1 };
// The isles: [x, z, radius, height].
var ISLES = [[-430, -770, 120, 78], [-195, -655, 72, 44], [35, -790, 165, 112], [255, -690, 96, 58],
             [450, -830, 128, 76], [-630, -870, 96, 52]];

// ── Ground: the rocky bank below the house, falling to the sea ───────────
function shoreZ(x) { return -5.6 + 1.4 * Math.sin(x * 0.21 + 0.5) + 0.7 * Math.sin(x * 0.57 + 2); }
function ground(x, z) {
  var t = smooth(shoreZ(x) - 5, shoreZ(x) + 6, z);
  var h = lerp(-4, F - 0.15, t) + Math.sin(t * Math.PI) * (0.5 * Math.sin(x * 1.3 + z * 0.9) + 0.35 * Math.sin(x * 2.9 - z * 2.1));
  h += smooth(8, 60, z) * (9 + 5 * Math.sin(x * 0.06 + 1));
  // The house stands on a level pad.
  var dx = Math.max(Math.abs(x) - 3.2, 0), dz = Math.max(-0.7 - z, z - 5.3, 0);
  return lerp(h, F - 0.1, 1 - smooth(0, 1.6, Math.hypot(dx, dz)));
}

// The tree's rock: a jittered ellipsoid, so the roots can follow its skin.
function rockR(x, y, z) {
  return 1 + 0.1 * Math.sin(x * 3.1 + y * 2.3 + 0.7) * Math.cos(z * 2.7 - y * 1.9 + 1.9) + 0.05 * Math.sin(x * 7 + z * 6);
}
function rockSurface(u, out) {
  var k = rockR(u.x, u.y, u.z);
  return out.set(ROCK.x + u.x * ROCK.rx * k, ROCK.y + u.y * ROCK.ry * k, ROCK.z + u.z * ROCK.rz * k);
}
var TREE_BASE = rockSurface(new THREE.Vector3(0, 1, 0), new THREE.Vector3());

// ── Shaders ──────────────────────────────────────────────────────────────
var NOISE =
  'float hash2(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }\n' +
  'float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);\n' +
  '  return mix(mix(hash2(i), hash2(i + vec2(1.0, 0.0)), f.x), mix(hash2(i + vec2(0.0, 1.0)), hash2(i + vec2(1.0, 1.0)), f.x), f.y); }\n' +
  'float fbm(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = p * 2.03 + vec2(1.7, 9.2); a *= 0.5; } return s; }\n';

// The sky: a night gradient that warms to dawn, the dawn's glow (and the
// sun's edge) low in the north, and a soft glow round the moon.
var SKY_VS = 'varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }';
var SKY_FS = 'uniform vec3 uTop; uniform vec3 uMid; uniform vec3 uHorizon; uniform vec3 uDawnDir; uniform vec3 uDawnCol;\n' +
  'uniform vec3 uMoonDir; uniform float uDawn; uniform float uMoonGlow; varying vec3 vP;\n' +
  'void main(){ vec3 d = normalize(vP); float h = clamp(d.y, 0.0, 1.0);\n' +
  ' vec3 c = mix(uHorizon, uMid, smoothstep(0.0, 0.25, h)); c = mix(c, uTop, smoothstep(0.25, 0.8, h));\n' +
  ' float s = max(dot(d, uDawnDir), 0.0), side = 0.5 + 0.5 * dot(normalize(d.xz + 1e-4), normalize(uDawnDir.xz));\n' +
  ' c += uDawnCol * uDawn * (pow(s, 5.0) * 0.25 + pow(s, 40.0) * 0.4 + pow(s, 2400.0) * 3.0 + exp(-h * 10.0) * 0.22 * side * side);\n' +
  ' float m = max(dot(d, uMoonDir), 0.0);\n' +
  ' c += vec3(0.5, 0.58, 0.8) * uMoonGlow * (pow(m, 14.0) * 0.07 + pow(m, 160.0) * 0.22);\n' +
  ' gl_FragColor = vec4(c, 1.0);\n #include <tonemapping_fragment>\n #include <colorspace_fragment>\n }';

// The moon: a disc lit from one side by uPhase (1 full, 0 new), with faint
// maria and earthshine.
var PLANE_VS = 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }';
var MOON_FS = 'uniform float uPhase; uniform float uAlpha; varying vec2 vUv;\n' + NOISE +
  'void main(){ vec2 c = (vUv - 0.5) * 2.0; float r = length(c); if (r > 1.0) discard;\n' +
  ' float z = sqrt(max(1.0 - r * r, 0.0)), ang = (1.0 - uPhase) * 3.14159;\n' +
  ' float lit = smoothstep(-0.04, 0.08, dot(vec3(c, z), vec3(-sin(ang), 0.0, cos(ang))));\n' +
  ' float maria = 0.78 + 0.22 * smoothstep(0.35, 0.65, fbm(c * 2.2 + 3.0));\n' +
  ' vec3 col = vec3(0.93, 0.96, 1.0) * maria * lit * 2.6 + vec3(0.035, 0.045, 0.07) * (1.0 - lit);\n' +
  ' float a = smoothstep(1.0, 0.95, r) * uAlpha;\n' +
  ' gl_FragColor = vec4(col, a);\n #include <colorspace_fragment>\n }';

// The sea: a flat plane whose normals carry swell, ripples and, in the
// gust, chop and foam; it mirrors the sky gradient, the moon's path and
// the dawn's.
var WATER_VS = 'varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }';
var WATER_FS = 'uniform vec3 uTop; uniform vec3 uMid; uniform vec3 uHorizon; uniform vec3 uDeep;\n' +
  'uniform vec3 uMoonDir; uniform vec3 uMoonCol; uniform float uMoon; uniform vec3 uDawnDir; uniform vec3 uDawnCol; uniform float uDawn;\n' +
  'uniform float uTime; uniform float uChop; uniform vec3 uFogColor; uniform float uFogDensity; varying vec3 vW;\n' + NOISE +
  'vec3 skyCol(vec3 d){ float h = clamp(d.y, 0.0, 1.0); vec3 c = mix(uHorizon, uMid, smoothstep(0.0, 0.25, h)); c = mix(c, uTop, smoothstep(0.25, 0.8, h));\n' +
  ' return c + uDawnCol * uDawn * pow(max(dot(d, uDawnDir), 0.0), 6.0) * 0.5; }\n' +
  'vec2 slope(vec2 q, float e){ float n0 = vnoise(q); return vec2(vnoise(q + vec2(e, 0.0)) - n0, vnoise(q + vec2(0.0, e)) - n0) / e; }\n' +
  'void main(){ vec3 toCam = cameraPosition - vW; float dist = length(toCam); vec3 V = toCam / dist;\n' +
  ' float fine = 1.0 - smoothstep(25.0, 400.0, dist), t = uTime; vec2 p = vW.xz; vec2 g = vec2(0.0);\n' +
  ' g += vec2(0.6, 0.8) * cos(dot(p, vec2(0.6, 0.8)) * 0.21 + t * 0.9) * 0.045;\n' +
  ' g += vec2(-0.3, 0.95) * cos(dot(p, vec2(-0.3, 0.95)) * 0.37 + t * 1.3) * 0.03;\n' +
  ' g += vec2(0.9, 0.44) * cos(dot(p, vec2(0.9, 0.44)) * 0.63 + t * 1.7) * 0.02;\n' +
  ' g += slope(p * vec2(0.9, 1.6) + vec2(t * 0.3, -t * 0.5), 0.1) * (0.035 + 0.08 * uChop) * fine;\n' +
  ' g += slope(p * 3.1 + vec2(-t * 0.8, t * 0.6), 0.1) * (0.014 + 0.05 * uChop) * fine;\n' +
  ' g *= 1.0 + uChop * 1.2;\n' +
  ' vec3 N = normalize(vec3(-g.x, 1.0, -g.y)), R = reflect(-V, N); R.y = abs(R.y);\n' +
  ' float fres = 0.02 + 0.98 * pow(1.0 - max(dot(N, V), 0.0), 5.0);\n' +
  ' vec3 col = mix(uDeep, skyCol(R), clamp(fres, 0.0, 1.0));\n' +
  ' float sm = max(dot(R, uMoonDir), 0.0), sd = max(dot(R, uDawnDir), 0.0);\n' +
  ' col += uMoonCol * uMoon * (pow(sm, 1600.0) * 8.0 + pow(sm, 120.0) * 0.2);\n' +
  ' col += uDawnCol * uDawn * (pow(sd, 600.0) * 5.0 + pow(sd, 30.0) * 0.3);\n' +
  ' float foam = smoothstep(0.8, 0.96, vnoise(p * vec2(0.5, 1.1) + vec2(t * 1.1, t * 0.2))) * uChop * fine;\n' +
  ' col = mix(col, vec3(0.5, 0.56, 0.66) * (0.35 + 0.65 * uMoon), foam * 0.4);\n' +
  ' float fogF = 1.0 - exp(-uFogDensity * uFogDensity * dist * dist);\n' +
  ' gl_FragColor = vec4(mix(col, uFogColor, fogF), 1.0);\n #include <tonemapping_fragment>\n #include <colorspace_fragment>\n }';

// The isles: dark shapes in the haze, rimmed and warmed gold by the dawn;
// the mist swallows them.
var ISLE_VS = 'varying vec3 vW; varying vec3 vN; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; vN = normal; gl_Position = projectionMatrix * viewMatrix * w; }';
var ISLE_FS = 'uniform vec3 uBase; uniform vec3 uHaze; uniform vec3 uGold; uniform vec3 uDawnDir; uniform vec3 uMoonDir;\n' +
  'uniform float uDawn; uniform float uMoon; uniform float uMist; uniform float uDens; varying vec3 vW; varying vec3 vN;\n' +
  'void main(){ vec3 n = normalize(vN), V = normalize(cameraPosition - vW); float dist = length(cameraPosition - vW);\n' +
  ' float rim = pow(1.0 - abs(dot(n, V)), 3.0), dl = max(dot(n, uDawnDir), 0.0), ml = max(dot(n, uMoonDir), 0.0);\n' +
  ' vec3 col = uBase * (0.7 + 0.3 * n.y) + vec3(0.45, 0.55, 0.8) * ml * 0.06 * uMoon;\n' +
  ' col += uGold * uDawn * (0.16 + 0.5 * dl + 0.9 * rim * smoothstep(0.0, 25.0, vW.y));\n' +
  ' float haze = 1.0 - exp(-pow(dist * uDens, 2.0)); haze = mix(haze, 1.0, uMist * 0.9);\n' +
  ' gl_FragColor = vec4(mix(col, uHaze, haze), 1.0);\n #include <tonemapping_fragment>\n #include <colorspace_fragment>\n }';

// Glowing points (motes, the boats' light, the isles' lamps), and their
// reflections stretched down the water when uMirror is set.
var GLOW_VS = 'attribute vec3 aCol; attribute float aA; attribute float aSize; uniform float uScale; uniform float uTime; uniform float uAmt; uniform float uMirror;\n' +
  'varying vec3 vCol; varying float vA;\n' +
  'void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;\n' +
  ' vCol = aCol; vA = aA * uAmt * (0.86 + 0.14 * sin(uTime * 2.6 + position.x * 3.0 + position.z * 1.7));\n' +
  ' gl_PointSize = clamp(aSize * uScale / -mv.z * (1.0 + uMirror * 1.5), 1.5, 160.0); }';
var GLOW_FS = 'uniform float uMirror; uniform float uTime; varying vec3 vCol; varying float vA;\n' +
  'void main(){ vec2 c = gl_PointCoord - 0.5; if (uMirror > 0.5) c *= vec2(4.0, 1.0);\n' +
  ' float d = length(c), a = (exp(-d * d * 70.0) * (1.0 - uMirror * 0.6) + exp(-d * d * 11.0) * 0.32) * vA * (1.0 - uMirror * 0.45);\n' +
  ' a *= 1.0 - smoothstep(0.4, 0.5, max(abs(c.x), abs(c.y)));\n' +
  ' gl_FragColor = vec4(vCol * a, a);\n #include <colorspace_fragment>\n }';

// Sparks rising (from the hearth, the chimney): each loops up its column,
// drifting and fading; `position` holds three seeds.
var EMBER_VS = 'uniform vec3 uOrigin; uniform float uTime; uniform float uAmt; uniform float uH; uniform float uSpread; uniform float uRise;\n' +
  'uniform float uSize; uniform float uScale; uniform float uWind; varying float vA; varying float vHot;\n' +
  'void main(){ vec3 s = position; float life = fract(uTime * uRise / uH + s.x);\n' +
  ' vec3 p = uOrigin + vec3((s.y - 0.5) * uSpread * (1.0 + life) + sin(uTime * 1.7 + s.z * 30.0) * 0.08 * uH * life + uWind * life * uH,\n' +
  '                         life * uH, (s.z - 0.5) * uSpread * (1.0 + life) + cos(uTime * 1.3 + s.y * 20.0) * 0.08 * uH * life);\n' +
  ' vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_Position = projectionMatrix * mv;\n' +
  ' vA = step(s.y * 0.6 + s.z * 0.4, uAmt) * smoothstep(0.0, 0.06, life) * (1.0 - smoothstep(0.5, 1.0, life)) * (0.6 + 0.4 * sin(uTime * 9.0 + s.x * 40.0));\n' +
  ' vHot = 1.0 - life;\n' +
  ' gl_PointSize = clamp(uSize * uScale / -mv.z, 1.0, 24.0); }';
var EMBER_FS = 'varying float vA; varying float vHot;\n' +
  'void main(){ float d = length(gl_PointCoord - 0.5); float a = exp(-d * d * 22.0) * vA;\n' +
  ' vec3 col = mix(vec3(0.9, 0.18, 0.02), vec3(1.0, 0.45, 0.08), vHot);\n' +
  ' gl_FragColor = vec4(col * a, a);\n #include <colorspace_fragment>\n }';

// Flames: noise climbing through a tongue-shaped mask.
var FLAME_FS = 'uniform float uTime; uniform float uFire; uniform float uSeed; varying vec2 vUv;\n' + NOISE +
  'void main(){ vec2 uv = vUv; float x = (uv.x - 0.5) * 2.0;\n' +
  ' float n = fbm(vec2(uv.x * 3.5 + uSeed, uv.y * 2.6 - uTime * 1.9)) * 0.65 + fbm(vec2(uv.x * 7.0 - uSeed, uv.y * 5.0 - uTime * 3.2)) * 0.35;\n' +
  ' float width = (1.0 - uv.y * 0.85) * (0.7 + 0.3 * uFire);\n' +
  ' float body = 1.0 - smoothstep(0.25 * width, 0.95 * width, abs(x + (n - 0.5) * 0.5 * uv.y));\n' +
  ' float f = body * smoothstep(uv.y * 1.25 - 0.05, uv.y * 1.25 + 0.3, n * (0.35 + 0.95 * uFire));\n' +
  ' f *= smoothstep(0.0, 0.08, uv.y);\n' +
  ' vec3 col = mix(vec3(0.75, 0.12, 0.02), vec3(1.0, 0.5, 0.1), smoothstep(0.1, 0.55, f));\n' +
  ' col = mix(col, vec3(1.0, 0.86, 0.55), smoothstep(0.6, 1.0, f));\n' +
  ' float a = clamp(f * 1.4, 0.0, 1.0) * min(uFire * 2.0, 1.0);\n' +
  ' gl_FragColor = vec4(col * a, a);\n #include <colorspace_fragment>\n }';

// Banners: ribbons streaming on the wind, tapering to a point, rippling
// and twisting along their length. aBan = (length, width, phase, colour),
// aLane = (start x, height, z, speed).
var BANNER_VS = 'attribute vec4 aBan; attribute vec4 aLane; uniform float uTime; uniform float uFlow; uniform float uAmt;\n' +
  'uniform float uX0; uniform float uSpan; uniform vec3 uC0; uniform vec3 uC1; uniform vec3 uC2; uniform vec3 uC3; uniform vec3 uC4;\n' +
  'varying vec3 vCol; varying float vA; varying float vShade; varying float vEdge;\n' +
  'void main(){ float s = position.x, L = aBan.x, ph = aBan.z;\n' +
  ' float hx = uX0 + mod(aLane.x + uFlow * aLane.w, uSpan), sx = hx - s * L;\n' +
  ' float wave = s * 8.0 - uTime * 4.5 * (0.6 + 0.1 * aLane.w) + ph;\n' +
  ' float tw = sin(wave * 0.45 + ph) * 1.1, w = aBan.y * (1.0 - 0.86 * s);\n' +
  ' vec3 p = vec3(sx, aLane.y + sin(wave) * 0.4 * s + sin(s * 2.5 + ph + uTime * 0.7) * 0.5 * s, aLane.z + sin(wave * 0.7 + 1.3) * 0.7 * s);\n' +
  ' p += vec3(0.0, cos(tw), sin(tw)) * position.y * w;\n' +
  ' vShade = 0.5 + 0.5 * cos(wave + tw);\n' +
  ' float u01 = (sx - uX0) / uSpan;\n' +
  ' vA = uAmt * smoothstep(0.0, 0.12, u01) * (1.0 - smoothstep(0.88, 1.0, (hx - uX0) / uSpan)) * (1.0 - smoothstep(0.7, 1.0, s));\n' +
  ' int ci = int(aBan.w + 0.5);\n' +
  ' vCol = ci == 0 ? uC0 : ci == 1 ? uC1 : ci == 2 ? uC2 : ci == 3 ? uC3 : uC4;\n' +
  ' vEdge = position.y;\n' +
  ' gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0); }';
var BANNER_FS = 'uniform float uLight; varying vec3 vCol; varying float vA; varying float vShade; varying float vEdge;\n' +
  'void main(){ float a = vA * smoothstep(0.5, 0.36, abs(vEdge)) * 0.85;\n' +
  ' vec3 col = vCol * (0.25 + 0.75 * vShade) * uLight;\n' +
  ' gl_FragColor = vec4(col, a);\n #include <tonemapping_fragment>\n #include <colorspace_fragment>\n }';

// Leaves torn from the tree, tumbling on the wind (faded out near the eye).
var LEAF_VS = 'attribute float aSeed; uniform float uAmt; uniform float uScale; uniform float uTime;\n' +
  'varying float vA; varying float vRot; varying vec3 vCol;\n' +
  'void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv; float d = -mv.z;\n' +
  ' vA = step(aSeed, uAmt) * smoothstep(0.8, 2.5, d); vRot = uTime * (1.5 + aSeed * 3.0) + aSeed * 30.0;\n' +
  ' vCol = aSeed < 0.3 ? vec3(0.62, 0.16, 0.07) : aSeed < 0.6 ? vec3(0.75, 0.3, 0.1) : aSeed < 0.85 ? vec3(0.5, 0.1, 0.05) : vec3(0.85, 0.5, 0.15);\n' +
  ' gl_PointSize = clamp(0.16 * uScale / d, 0.0, 16.0); }';
var LEAF_FS = 'uniform float uLight; varying float vA; varying float vRot; varying vec3 vCol;\n' +
  'void main(){ vec2 c = gl_PointCoord - 0.5; float cs = cos(vRot), sn = sin(vRot); c = vec2(cs * c.x - sn * c.y, sn * c.x + cs * c.y);\n' +
  ' c.x /= max(abs(sin(vRot * 0.7)), 0.25);\n' +
  ' float m = smoothstep(0.46, 0.38, length(vec2(c.x * 2.3, c.y)));\n' +
  ' if (m * vA < 0.03) discard;\n' +
  ' gl_FragColor = vec4(vCol * uLight, m * vA);\n #include <tonemapping_fragment>\n #include <colorspace_fragment>\n }';

// Firelight dancing on the room's walls, seen through the window from
// outside: warm bands that flicker and climb.
var BLAZE_FS = 'uniform float uTime; uniform float uAmt; varying vec2 vUv;\n' + NOISE +
  'void main(){ float n = fbm(vec2(vUv.x * 3.0, vUv.y * 2.0 - uTime * 1.6)), n2 = vnoise(vec2(uTime * 6.0, 1.0));\n' +
  ' float glow = 0.55 + 0.45 * n + 0.2 * n2 - 0.25 * vUv.y;\n' +
  ' vec3 col = mix(vec3(0.85, 0.22, 0.04), vec3(1.0, 0.62, 0.22), smoothstep(0.45, 0.95, glow));\n' +
  ' gl_FragColor = vec4(col * glow * 1.05, uAmt * 0.92);\n #include <colorspace_fragment>\n }';

// Blossoms on the climbing vine: five soft petals round a warm heart.
var BLOOM_VS = 'attribute float aT; attribute float aSize; uniform float uGrow; uniform float uScale; uniform float uTime;\n' +
  'varying float vA; varying float vRot;\n' +
  'void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;\n' +
  ' float open = smoothstep(aT, aT + 0.12, uGrow); vA = open; vRot = aT * 40.0 + uTime * 0.1;\n' +
  ' gl_PointSize = clamp(aSize * uScale / -mv.z * open, 0.0, 70.0); }';
var BLOOM_FS = 'uniform vec3 uPetal; uniform vec3 uHeart; uniform float uLight; varying float vA; varying float vRot;\n' +
  'void main(){ vec2 c = (gl_PointCoord - 0.5) * 2.0; float r = length(c); float a = atan(c.y, c.x) + vRot;\n' +
  ' float petal = 0.62 + 0.38 * cos(a * 5.0); float m = smoothstep(petal, petal - 0.18, r);\n' +
  ' if (m < 0.02) discard;\n' +
  ' vec3 col = mix(uHeart, uPetal, smoothstep(0.12, 0.42, r)) * (0.75 + 0.25 * (1.0 - r)) * uLight;\n' +
  ' gl_FragColor = vec4(col, m * vA);\n #include <tonemapping_fragment>\n #include <colorspace_fragment>\n }';

// ── Small pieces ─────────────────────────────────────────────────────────
function canvasTexture(w, h, paint) {
  var c = document.createElement('canvas');
  c.width = w; c.height = h;
  paint(c.getContext('2d'), w, h);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function puffTexture(r) {
  return canvasTexture(128, 128, function (x) {
    for (var i = 0; i < 26; i++) {
      var a = r() * 6.28, d = r() * 24, cx = 64 + Math.cos(a) * d, cy = 64 + Math.sin(a) * d * 0.6, rad = 16 + r() * 22;
      var g = x.createRadialGradient(cx, cy, 0, cx, cy, rad);
      g.addColorStop(0, 'rgba(255,255,255,0.2)');
      g.addColorStop(1, 'rgba(255,255,255,0)');
      x.fillStyle = g;
      x.fillRect(0, 0, 128, 128);
    }
  });
}

// Inside the room only the fire (and a little sky) lights things: these
// materials drop the directional lights, the moon and the dawn, which would
// otherwise shine straight through the walls.
function indoors(mat) {
  var prev = mat.onBeforeCompile, key = mat.customProgramCacheKey();
  mat.onBeforeCompile = function (sh, gl) {
    prev.call(this, sh, gl);
    sh.fragmentShader = sh.fragmentShader.replace('#include <lights_fragment_begin>',
      THREE.ShaderChunk.lights_fragment_begin.replace('getDirectionalLightInfo( directionalLight, directLight );',
        'getDirectionalLightInfo( directionalLight, directLight ); directLight.color *= 0.0;'));
  };
  mat.customProgramCacheKey = function () { return key + 'indoors'; };
  return mat;
}

// The house's stone: courses of rough blocks with dark mortar, laid out in
// world space on whichever plane a face lies in; `soft` is the whitewashed
// version inside.
function stonework(mat, soft) {
  var lo = soft ? '0.8' : '0.45', span = soft ? '0.18' : '0.45', base = soft ? '0.9' : '0.7';
  mat.onBeforeCompile = function (sh) {
    sh.vertexShader = 'varying vec3 vWP; varying vec3 vWN;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n vWP = (modelMatrix * vec4(transformed, 1.0)).xyz; vWN = normalize(mat3(modelMatrix) * objectNormal);');
    sh.fragmentShader = 'varying vec3 vWP; varying vec3 vWN;\n' +
      'float sh1(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }\n' +
      sh.fragmentShader.replace('#include <color_fragment>', '#include <color_fragment>\n' +
      ' vec2 q = abs(vWN.x) > 0.6 ? vWP.zy : abs(vWN.z) > 0.6 ? vWP.xy : vWP.xz;\n' +
      ' float row = floor(q.y / 0.27), w = 0.4 + 0.25 * sh1(vec2(row, 3.0));\n' +
      ' float qx = q.x / w + sh1(vec2(row, 7.0)) * 3.0, cell = floor(qx);\n' +
      ' vec2 f = vec2(fract(qx), fract(q.y / 0.27));\n' +
      ' float mortar = smoothstep(0.0, 0.07, f.x) * smoothstep(1.0, 0.93, f.x) * smoothstep(0.0, 0.11, f.y) * smoothstep(1.0, 0.89, f.y);\n' +
      ' diffuseColor.rgb *= mix(' + lo + ', ' + base + ' + ' + span + ' * sh1(vec2(cell, row)), mortar) * (0.92 + 0.16 * sh1(floor(q * 9.0)));');
  };
  mat.customProgramCacheKey = function () { return 'stonework' + lo; };
  return mat;
}

function box(w, h, d, x, y, z, color) { return tinted(new THREE.BoxGeometry(w, h, d).translate(x, y, z), color); }
// A box given by its extents.
function slab(x0, x1, y0, y1, z0, z1, color) { return box(x1 - x0, y1 - y0, z1 - z0, (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2, color); }

// A tube along points whose radius tapers from r0 to r1.
function tube(points, tub, rad, r0, r1, color) {
  var curve = new THREE.CatmullRomCurve3(points), g = new THREE.TubeGeometry(curve, tub, 1, rad, false);
  var p = g.attributes.position, c = new THREE.Vector3(), v = new THREE.Vector3();
  for (var j = 0; j <= tub; j++) {
    curve.getPointAt(j / tub, c);
    var s = lerp(r0, r1, j / tub);
    for (var k = 0; k <= rad; k++) {
      var i = j * (rad + 1) + k;
      v.fromBufferAttribute(p, i).sub(c).multiplyScalar(s).add(c);
      p.setXYZ(i, v.x, v.y, v.z);
    }
  }
  return tinted(g, color);
}

// A rough rock placed in world space, darker and wetter towards the sea.
function rockGeometry(r, rad, sx, sy, sz, x, y, z) {
  var geo = new THREE.IcosahedronGeometry(rad, 2), p = geo.attributes.position, v = new THREE.Vector3();
  var k1 = r() * 6, k2 = r() * 6, cols = new Float32Array(p.count * 3), c = new THREE.Color();
  for (var i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i).normalize();
    var n = rad * (1 + 0.16 * Math.sin(v.x * 4.1 + v.y * 2.3 + k1) * Math.cos(v.z * 3.7 - v.y * 2.1 + k2));
    p.setXYZ(i, x + v.x * n * sx, y + v.y * n * sy, z + v.z * n * sz);
  }
  geo.computeVertexNormals();
  for (i = 0; i < p.count; i++) {
    c.set('#4d4944').lerp(new THREE.Color('#1d1c1b'), smooth(0.9, -0.2, p.getY(i)));
    cols[i * 3] = c.r; cols[i * 3 + 1] = c.g; cols[i * 3 + 2] = c.b;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  geo.deleteAttribute('uv');
  return geo;
}

// ── The tree ─────────────────────────────────────────────────────────────
// Built in its own frame (origin at the foot of the trunk, on the rock) in
// three poses that share one topology: at rest, with its branches lifted,
// and with its roots let go. The mesh morphs between them.
var TRUNK = [[0, -0.25, 0], [0.04, 0.9, 0.03], [0.16, 1.9, 0.08], [0.22, 2.75, 0.12]];
// [azimuth (0 = +x, towards +z), elevation, length, height on the trunk, droop]
var BRANCHES = [[0.65, 0.36, 7.0, 0.8, 0.12],     // the branch that reaches towards the window
                [2.5, 0.62, 3.4, 0.92, 0.13], [-2.45, 0.5, 3.8, 0.84, 0.15], [-0.7, 0.64, 3.4, 0.96, 0.11],
                [0.3, 1.08, 2.8, 1.0, 0.05], [-1.65, 0.42, 3.1, 0.72, 0.17], [3.6, 0.85, 2.6, 0.98, 0.08]];
var LEAF_COLS = ['#a8301c', '#c2462a', '#8a2416', '#cf6a28', '#b43a22', '#e08a34'];
var NROOT = 8;

function treeParts(lift, free) {
  var r = rng(907), parts = [], bark = '#3a2c24', trunkCurve = new THREE.CatmullRomCurve3(TRUNK.map(function (p) { return new THREE.Vector3(p[0], p[1], p[2]); }));
  var base = new THREE.Vector3(), d = new THREE.Vector3(), v = new THREE.Vector3(), u = new THREE.Vector3();
  parts.push(tube(trunkCurve.getPoints(8), 14, 7, 0.36, 0.17, bark));

  function limb(from, az, el, len, droop, r0, segs, out) {
    var pts = [];
    for (var s = 0; s <= 5; s++) {
      var t = s / 5;
      d.set(Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az));
      pts.push(from.clone().addScaledVector(d, t * len).add(v.set(0, -droop * t * t * len, 0)));
    }
    parts.push(tube(pts, segs, 5, r0, r0 * 0.25, bark));
    if (out) out.curve = new THREE.CatmullRomCurve3(pts);
    return pts;
  }
  // Foliage: bunches of small faceted clumps round a point.
  function clump(at, rad) {
    var g = new THREE.IcosahedronGeometry(rad, 0), p = g.attributes.position;
    var k1 = r() * 6, sy = 0.7 + r() * 0.25;
    for (var i = 0; i < p.count; i++) {
      u.fromBufferAttribute(p, i);
      var n = 1 + 0.22 * Math.sin(u.x * 9 + k1) * Math.cos(u.z * 7 - u.y * 5);
      p.setXYZ(i, at.x + u.x * n, at.y + u.y * n * sy, at.z + u.z * n);
    }
    g.computeVertexNormals();
    parts.push(tinted(g, LEAF_COLS[Math.floor(r() * LEAF_COLS.length)]));
  }
  function bunch(at, spread, n, rad) {
    for (var i = 0; i < n; i++) {
      clump(v.set(at.x + (r() - 0.5) * spread * 2, at.y + (r() - 0.3) * spread, at.z + (r() - 0.5) * spread * 2).clone(), rad * (0.7 + r() * 0.6));
    }
  }

  BRANCHES.forEach(function (b) {
    trunkCurve.getPointAt(b[3], base);
    var el = b[1] + (1.42 - b[1]) * 0.6 * lift, droop = b[4] * (1 - 0.8 * lift), info = {};
    limb(base, b[0], el, b[2], droop, 0.13, 12, info);
    var c = info.curve;
    [0.5, 0.75].forEach(function (s, k) {
      var from = c.getPointAt(s), az = b[0] + (k ? -0.65 : 0.6) + (r() - 0.5) * 0.3;
      var sub = limb(from, az, el + 0.25 + 0.15 * lift, b[2] * (0.42 + r() * 0.1), droop * 0.6, 0.06, 8);
      bunch(sub[5], 0.45, 5, 0.3);
      bunch(sub[3], 0.4, 3, 0.26);
    });
    [0.55, 0.7, 0.85, 1].forEach(function (s) { bunch(c.getPointAt(s).add(v.set(0, 0.2, 0)), 0.5, s === 1 ? 6 : 4, 0.32); });
  });

  // Roots: gripping, they run out over the rock's skin and down into the
  // water; let go, they hang and curl beneath the trunk like tendrils.
  for (var k = 0; k < NROOT; k++) {
    var a = k / NROOT * Math.PI * 2 + (r() - 0.5) * 0.5, reach = 1.55 + r() * 0.35, pts = [];
    for (var s = 0; s <= 8; s++) {
      var t = s / 8;
      if (!free) {
        var phi = lerp(0.1, reach, t), aa = a + 0.25 * Math.sin(t * 4 + k);
        u.set(Math.sin(phi) * Math.cos(aa), Math.cos(phi), Math.sin(phi) * Math.sin(aa));
        rockSurface(u, v);
        v.addScaledVector(u, 0.05 + 0.1 * (1 - t)).sub(TREE_BASE);
        if (s === 0) v.set(Math.cos(a) * 0.12, 0.35, Math.sin(a) * 0.12);
        pts.push(v.clone());
      } else {
        var rr = 0.12 + 1.05 * Math.sin(t * 1.5), af = a + 0.7 * t;
        pts.push(new THREE.Vector3(Math.cos(af) * rr, 0.35 - 1.9 * t + 0.75 * t * t * t, Math.sin(af) * rr));
      }
    }
    parts.push(tube(pts, 18, 5, 0.17, 0.03, '#4e3c30'));
  }
  return merge(parts);
}

function treeGeometry() {
  var rest = treeParts(0, false), raised = treeParts(1, false), freed = treeParts(0, true);
  rest.morphAttributes.position = [raised.attributes.position, freed.attributes.position];
  rest.morphAttributes.normal = [raised.attributes.normal, freed.attributes.normal];
  return rest;
}

// A paper boat about a metre long: two hull sides, the ends and a sail.
function boatGeometry() {
  var K0 = [0, 0, -0.3], K1 = [0, 0, 0.3], A0 = [0.17, 0.2, -0.58], A1 = [0.17, 0.2, 0.58], B0 = [-0.17, 0.2, -0.58], B1 = [-0.17, 0.2, 0.58];
  var S0 = [0, 0.2, -0.3], S1 = [0, 0.2, 0.3], S = [0, 0.56, 0.02];
  var tris = [K0, K1, A1, K0, A1, A0, K0, B1, K1, K0, B0, B1, K0, A0, B0, K1, B1, A1, S0, S1, S];
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute([].concat.apply([], tris), 3));
  geo.computeVertexNormals();
  return geo;
}

// ── Synthesised sound ────────────────────────────────────────────────────
// A fire's crackle: a low breathing roar under sparse pops and snaps.
function crackle(dur, gain) {
  return function (ac, out) {
    var sr = ac.sampleRate, n = Math.floor(sr * dur), b = ac.createBuffer(1, n, sr), d = b.getChannelData(0), lp = 0, i;
    for (i = 0; i < n; i++) {
      lp += (Math.random() * 2 - 1 - lp) * 0.03;
      d[i] = lp * 0.35;
    }
    var pops = Math.floor(dur * 11);
    for (var k = 0; k < pops; k++) {
      var at = Math.floor(Math.random() * n), len = Math.floor(sr * (0.002 + Math.random() * 0.01)), amp = 0.15 + Math.pow(Math.random(), 3) * 0.85;
      var burst = Math.random() < 0.25 ? 3 + Math.floor(Math.random() * 4) : 1;
      for (var q = 0; q < burst; q++) {
        var o = at + q * Math.floor(sr * (0.01 + Math.random() * 0.03));
        for (var j = 0; j < len && o + j < n; j++) d[o + j] += (Math.random() * 2 - 1) * amp * Math.exp(-j / (len * 0.25)) * (q ? 0.5 : 1);
      }
    }
    for (i = 0; i < n; i++) d[i] *= Math.min(1, i / (sr * 0.6)) * Math.min(1, (n - i) / (sr * 1.5));
    var src = ac.createBufferSource(), hp = ac.createBiquadFilter(), g = ac.createGain();
    src.buffer = b;
    hp.type = 'highpass';
    hp.frequency.value = 180;
    g.gain.value = gain;
    src.connect(hp); hp.connect(g); g.connect(out);
    src.start(ac.currentTime + 0.02);
  };
}
// A gust: filtered noise swelling and falling as its pitch rises and sinks.
function gustSound(ac, out) {
  var t = ac.currentTime + 0.02, len = 5, sr = ac.sampleRate, b = ac.createBuffer(1, sr * len, sr), d = b.getChannelData(0);
  for (var i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  var src = ac.createBufferSource(), bp = ac.createBiquadFilter(), g = ac.createGain();
  src.buffer = b;
  bp.type = 'bandpass';
  bp.Q.value = 0.9;
  bp.frequency.setValueAtTime(260, t);
  bp.frequency.exponentialRampToValueAtTime(950, t + 1.4);
  bp.frequency.exponentialRampToValueAtTime(320, t + len);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.55, t + 1.2);
  g.gain.exponentialRampToValueAtTime(0.0001, t + len);
  src.connect(bp); bp.connect(g); g.connect(out);
  src.start(t);
}

// ── Renderer ─────────────────────────────────────────────────────────────
function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(4127), i;
  var gl = makeRenderer(canvas, { clear: '#05070f' });
  gl.toneMappingExposure = 1.15;
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#141c30', 0.0045);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.04, 5000);
  var pxScale = 700, portrait = false;
  var UP = new THREE.Vector3(0, 1, 0), tmpV = new THREE.Vector3(), tmpV2 = new THREE.Vector3(), tmpQ = new THREE.Quaternion();
  var tmpM = new THREE.Matrix4(), tmpE = new THREE.Euler(), tmpC = new THREE.Color(), tmpC2 = new THREE.Color();

  // ── Sky ──
  var sky = new THREE.Group();
  world.add(sky);
  var skyU = { uTop: { value: new THREE.Color() }, uMid: { value: new THREE.Color() }, uHorizon: { value: new THREE.Color() },
               uDawnDir: { value: DAWN }, uDawnCol: { value: new THREE.Color('#ff9a60') }, uMoonDir: { value: MOON },
               uDawn: { value: 0 }, uMoonGlow: { value: 1 } };
  var skyMesh = new THREE.Mesh(new THREE.SphereGeometry(2000, 48, 24), new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false, uniforms: skyU, vertexShader: SKY_VS, fragmentShader: SKY_FS }));
  skyMesh.renderOrder = -10;
  sky.add(skyMesh);
  var stars = starField(r, small ? 1400 : 2600, 1800, 0.03, 1.3);
  stars.renderOrder = -9;
  sky.add(stars);
  var moonU = { uPhase: { value: 1 }, uAlpha: { value: 1 } };
  var moon = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.ShaderMaterial({ uniforms: moonU, vertexShader: PLANE_VS,
    fragmentShader: MOON_FS, transparent: true, depthWrite: false, fog: false }));
  moon.position.copy(MOON).multiplyScalar(1500);
  moon.scale.setScalar(44);
  sky.add(moon);
  var moonHalo = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(190,210,255,0.55)', 'rgba(120,140,220,0)'),
    blending: THREE.AdditiveBlending, transparent: true, depthWrite: false, fog: false }));
  moonHalo.position.copy(moon.position);
  moonHalo.scale.setScalar(170);
  sky.add(moonHalo);

  // ── Light ──
  var hemi = new THREE.HemisphereLight('#4f5f86', '#16130f', 0.6);
  var moonLight = new THREE.DirectionalLight('#b8c6ff', 1.0);
  moonLight.position.copy(MOON).multiplyScalar(50);
  var dawnLight = new THREE.DirectionalLight('#ffb478', 0);
  dawnLight.position.copy(DAWN).multiplyScalar(50).setY(6);
  var fireLight = new THREE.PointLight('#ff9a4a', 0, 9, 1.4);
  fireLight.position.set(FIRE.x - 0.3, FIRE.y + 0.15, FIRE.z);
  // Firelight spilling out of the window onto the bank and the red branch.
  var spill = new THREE.PointLight('#ff9a50', 0, 22, 1.4);
  spill.position.set(0, F + 1.5, -1.2);
  world.add(hemi, moonLight, dawnLight, fireLight, spill);

  // ── The sea ──
  var waterU = { uTop: skyU.uTop, uMid: skyU.uMid, uHorizon: skyU.uHorizon, uDeep: { value: new THREE.Color('#02050b') },
                 uMoonDir: { value: MOON }, uMoonCol: { value: new THREE.Color('#dfe8ff') }, uMoon: { value: 1 },
                 uDawnDir: { value: DAWN }, uDawnCol: skyU.uDawnCol, uDawn: skyU.uDawn, uTime: { value: 0 }, uChop: { value: 0 },
                 uFogColor: { value: new THREE.Color() }, uFogDensity: { value: 0.0013 } };
  var water = new THREE.Mesh(new THREE.PlaneGeometry(6000, 6000).rotateX(-Math.PI / 2), new THREE.ShaderMaterial({
    uniforms: waterU, vertexShader: WATER_VS, fragmentShader: WATER_FS }));
  water.frustumCulled = false;
  world.add(water);

  // ── The isles, far out, and the few lamps that wait on them ──
  var isleParts = [], lampPos = [], lampCol = [], lampA = [], lampSize = [];
  ISLES.forEach(function (s, n) {
    var R = s[2], H = s[3], g = new THREE.PlaneGeometry(R * 2.2, R * 2.2, 32, 32).rotateX(-Math.PI / 2), p = g.attributes.position;
    var k1 = r() * 6, k2 = r() * 6;
    for (var j = 0; j < p.count; j++) {
      var x = p.getX(j), z = p.getZ(j), a = Math.atan2(z, x), d = Math.hypot(x, z) / (R * (1 + 0.18 * Math.sin(a * 3 + k1) + 0.08 * Math.sin(a * 7 + k2)));
      var h = d < 1 ? H * (0.5 * Math.pow(1 - d, 1.2) + 0.5 * Math.pow(1 - d * d, 2.2)) * (0.75 + 0.25 * Math.sin(x * 0.03 + k1) * Math.cos(z * 0.04 + k2)) + 2 : -8;
      h += d < 1 ? H * 0.35 * Math.exp(-Math.pow((x - R * 0.25 * Math.cos(k1)) / (R * 0.22), 2) - Math.pow((z - R * 0.2) / (R * 0.3), 2)) : 0;
      p.setY(j, h);
      if (d < 0.75 && r() < 0.012 && lampPos.length < 90) {
        lampPos.push(s[0] + x, h + 1.5, s[1] + z);
        lampCol.push(1, 0.78, 0.46);
        lampA.push(0.5 + r() * 0.5);
        lampSize.push(5 + r() * 4);
      }
    }
    g.translate(s[0], 0, s[1]);
    g.computeVertexNormals();
    isleParts.push(tinted(g, '#ffffff'));
  });
  var isleU = { uBase: { value: new THREE.Color('#0b1222') }, uHaze: { value: new THREE.Color() }, uGold: { value: new THREE.Color('#e08a40') },
                uDawnDir: { value: DAWN }, uMoonDir: { value: MOON }, uDawn: { value: 0 }, uMoon: { value: 1 }, uMist: { value: 0 },
                uDens: { value: 0.0011 } };
  var isles = new THREE.Mesh(merge(isleParts), new THREE.ShaderMaterial({ uniforms: isleU, vertexShader: ISLE_VS, fragmentShader: ISLE_FS }));
  world.add(isles);
  function glowPoints(pos, col, a, size, amt) {
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('aCol', new THREE.BufferAttribute(col, 3));
    geo.setAttribute('aA', new THREE.BufferAttribute(a, 1));
    geo.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
    var u = { uScale: { value: 700 }, uTime: { value: 0 }, uAmt: { value: amt }, uMirror: { value: 0 } };
    var pts = new THREE.Points(geo, new THREE.ShaderMaterial({ uniforms: u, vertexShader: GLOW_VS, fragmentShader: GLOW_FS,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false }));
    pts.frustumCulled = false;
    return pts;
  }
  var lamps = glowPoints(new Float32Array(lampPos), new Float32Array(lampCol), new Float32Array(lampA), new Float32Array(lampSize), 0.5);
  world.add(lamps);

  // ── The bank, its rocks and the tree's rock ──
  var gc = new THREE.Color(), cWet = new THREE.Color('#1e1d1c'), cRock = new THREE.Color('#4a4640'), cGrass = new THREE.Color('#4a4528'),
      cTurf = new THREE.Color('#5a4426');
  world.add(terrain(130, small ? 110 : 170, 0, -8, ground, new THREE.MeshLambertMaterial({ vertexColors: true }), function (x, z, h) {
    var n = 0.5 + 0.5 * Math.sin(x * 1.7 + Math.sin(z * 1.3) * 2), grassy = smooth(F - 1.2, F - 0.2, h) * (0.6 + 0.4 * n);
    gc.copy(cWet).lerp(cRock, smooth(-0.2, 0.9, h)).lerp(cGrass, grassy).lerp(cTurf, smooth(6, 12, h) * 0.6);
    return gc.multiplyScalar(0.8 + 0.4 * n);
  }));
  var rocks = [];
  var rg = new THREE.IcosahedronGeometry(1, 4), rp = rg.attributes.position, rc = new Float32Array(rp.count * 3);
  for (i = 0; i < rp.count; i++) {
    tmpV.fromBufferAttribute(rp, i).normalize();
    rockSurface(tmpV, tmpV2);
    rp.setXYZ(i, tmpV2.x, tmpV2.y, tmpV2.z);
  }
  rg.computeVertexNormals();
  for (i = 0; i < rp.count; i++) {
    var ry = rp.getY(i), lichen = 0.5 + 0.5 * Math.sin(rp.getX(i) * 6 + rp.getZ(i) * 5);
    tmpC.set('#6a645c').lerp(tmpC2.set('#4a4a36'), smooth(1.4, 2.0, ry) * lichen).lerp(tmpC2.set('#191817'), smooth(0.7, -0.1, ry));
    rc[i * 3] = tmpC.r; rc[i * 3 + 1] = tmpC.g; rc[i * 3 + 2] = tmpC.b;
  }
  rg.setAttribute('color', new THREE.BufferAttribute(rc, 3));
  rg.deleteAttribute('uv');
  rocks.push(rg);
  for (i = 0; i < 16; i++) {
    var bx = -26 + r() * 52, bz = shoreZ(bx) + 0.5 + (r() - 0.5) * 3, bs = 0.5 + r() * 1.3;
    if (Math.hypot(bx - ROCK.x, bz - ROCK.z) < 4) continue;
    rocks.push(rockGeometry(r, bs, 1.2, 0.7, 1, bx, ground(bx, bz) + bs * 0.2, bz));
  }
  world.add(new THREE.Mesh(merge(rocks), new THREE.MeshLambertMaterial({ vertexColors: true })));

  // Dark firs on the hill behind the house.
  var fir = merge([tinted(new THREE.CylinderGeometry(0.12, 0.2, 1.4, 5).translate(0, 0.7, 0), '#2a2018'),
                   tinted(new THREE.ConeGeometry(1.4, 3.8, 7).translate(0, 2.8, 0), '#ffffff'),
                   tinted(new THREE.ConeGeometry(1.0, 3.0, 7).translate(0, 4.7, 0), '#ffffff')]);
  var firs = new THREE.InstancedMesh(fir, new THREE.MeshLambertMaterial({ vertexColors: true }), small ? 70 : 140);
  scatter(firs, 4000, function (n, p, q, sc, c) {
    var x = (r() - 0.5) * 120, z = 7 + r() * 50;
    if (Math.abs(x) < 6 && z < 12) return false;
    p.set(x, ground(x, z) - 0.2, z);
    q.setFromAxisAngle(UP, r() * 6.28);
    sc.setScalar(0.9 + r() * 0.9);
    c.setHSL(0.36, 0.22, 0.07 + r() * 0.04);
  });
  world.add(firs);

  // ── The house: stone walls, slate roof, the chimney on the east gable ──
  var stone = '#8c867c', X0 = ROOM.x0 - WALL, X1 = ROOM.x1 + WALL, Z0 = -WALL, Z1 = ROOM.z1 + WALL, TOP = F + ROOM.h + 0.1, LOW = F - 1.3;
  var outer = [
    slab(X0, WIN.x0, LOW, TOP, Z0, -0.005, stone), slab(WIN.x1, X1, LOW, TOP, Z0, -0.005, stone),
    slab(WIN.x0, WIN.x1, LOW, WIN.y0, Z0, -0.005, stone), slab(WIN.x0, WIN.x1, WIN.y1, TOP, Z0, -0.005, stone),
    slab(X0, X1, LOW, TOP, ROOM.z1 + 0.005, Z1, stone),
    slab(X0, ROOM.x0 - 0.005, LOW, TOP, -0.005, ROOM.z1 + 0.005, stone),
    slab(ROOM.x1 + 0.005, X1, LOW, TOP, -0.005, ROOM.z1 + 0.005, stone),
    slab(X1 - 0.05, X1 + 0.35, LOW, F + 5.2, 2.35, 3.25, '#7a746a'),        // the chimney stack
    slab(WIN.x0 - 0.12, WIN.x1 + 0.12, WIN.y0 - 0.08, WIN.y0, Z0 - 0.1, Z0 + 0.02, '#6e685e')   // the outer sill
  ];
  // Gable ends and the roof (a ridge running east-west).
  var gableH = 1.9, zMid = (Z0 + Z1) / 2;
  var gable = new THREE.Shape();
  gable.moveTo(Z0 - 0.02, 0); gable.lineTo(Z1 + 0.02, 0); gable.lineTo(zMid, gableH); gable.lineTo(Z0 - 0.02, 0);
  [X0, X1 - 0.3].forEach(function (gx) {
    outer.push(tinted(new THREE.ExtrudeGeometry(gable, { depth: 0.3, bevelEnabled: false }).rotateY(-Math.PI / 2).translate(gx + 0.3, TOP, 0), stone));
  });
  var roof = [], roofW = Math.hypot((Z1 - Z0) / 2 + 0.45, gableH + 0.3);
  [-1, 1].forEach(function (sd) {
    var g = new THREE.BoxGeometry(X1 - X0 + 0.6, 0.14, roofW);
    g.rotateX(sd * Math.atan2(gableH + 0.3, (Z1 - Z0) / 2 + 0.45)).translate((X0 + X1) / 2, TOP + gableH / 2 - 0.08, zMid + sd * ((Z1 - Z0) / 4 + 0.22));
    roof.push(tinted(g, '#2b2c33'));
  });
  world.add(new THREE.Mesh(merge(outer), stonework(new THREE.MeshLambertMaterial({ vertexColors: true }))));
  world.add(new THREE.Mesh(merge(roof), new THREE.MeshLambertMaterial({ vertexColors: true })));
  // The glow of the lit window, seen from outside.
  var winGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,130,50,0.9)', 'rgba(255,90,30,0)'),
    blending: THREE.AdditiveBlending, transparent: true, depthWrite: false, opacity: 0 }));
  winGlow.position.set(0, (WIN.y0 + WIN.y1) / 2, Z0 - 0.4);
  winGlow.scale.set(3.6, 3.2, 1);
  world.add(winGlow);
  // Firelight filling the room behind the window, seen from outside.
  var blaze = new THREE.Mesh(new THREE.PlaneGeometry(WIN.x1 - WIN.x0 + 0.1, WIN.y1 - WIN.y0 + 0.1).translate(0, (WIN.y0 + WIN.y1) / 2, 0.35),
    new THREE.ShaderMaterial({ uniforms: { uTime: { value: 0 }, uAmt: { value: 0 } }, vertexShader: PLANE_VS, fragmentShader: BLAZE_FS,
      transparent: true, depthWrite: false, side: THREE.DoubleSide }));
  world.add(blaze);

  // ── The room ──
  var plaster = '#8c8478', wood = '#3a2a1e', soot = '#141110', hearthStone = '#5e5650';
  var floorTex = canvasTexture(512, 512, function (x, w, h) {
    for (var b = 0; b < 8; b++) {
      x.fillStyle = 'hsl(' + (22 + r() * 8) + ',32%,' + (16 + r() * 7) + '%)';
      x.fillRect(0, b * 64, w, 64);
      for (var q = 0; q < 26; q++) {
        x.strokeStyle = 'rgba(0,0,0,' + (0.08 + r() * 0.12) + ')';
        x.beginPath();
        var yy = b * 64 + 4 + r() * 56;
        x.moveTo(0, yy);
        x.bezierCurveTo(w * 0.3, yy + (r() - 0.5) * 8, w * 0.7, yy + (r() - 0.5) * 8, w, yy + (r() - 0.5) * 5);
        x.stroke();
      }
      x.fillStyle = 'rgba(0,0,0,0.55)';
      x.fillRect(0, b * 64, w, 2);
    }
  });
  floorTex.wrapS = floorTex.wrapT = THREE.RepeatWrapping;
  floorTex.repeat.set(1.4, 1.2);
  var floor = new THREE.Mesh(new THREE.PlaneGeometry(ROOM.x1 - ROOM.x0, ROOM.z1).rotateX(-Math.PI / 2).translate(0, F, ROOM.z1 / 2),
                             indoors(new THREE.MeshLambertMaterial({ map: floorTex })));
  world.add(floor);
  var walls = [
    tinted(new THREE.PlaneGeometry(ROOM.x1 - ROOM.x0, ROOM.h).rotateY(Math.PI).translate(0, F + ROOM.h / 2, ROOM.z1), plaster),
    tinted(new THREE.PlaneGeometry(ROOM.z1, ROOM.h).rotateY(Math.PI / 2).translate(ROOM.x0, F + ROOM.h / 2, ROOM.z1 / 2), plaster),
    tinted(new THREE.PlaneGeometry(ROOM.z1, ROOM.h).rotateY(-Math.PI / 2).translate(ROOM.x1, F + ROOM.h / 2, ROOM.z1 / 2), plaster),
    // The north wall, round the window.
    tinted(new THREE.PlaneGeometry(WIN.x0 - ROOM.x0, ROOM.h).translate((ROOM.x0 + WIN.x0) / 2, F + ROOM.h / 2, 0), plaster),
    tinted(new THREE.PlaneGeometry(ROOM.x1 - WIN.x1, ROOM.h).translate((ROOM.x1 + WIN.x1) / 2, F + ROOM.h / 2, 0), plaster),
    tinted(new THREE.PlaneGeometry(WIN.x1 - WIN.x0, WIN.y0 - F).translate(0, (F + WIN.y0) / 2, 0), plaster),
    tinted(new THREE.PlaneGeometry(WIN.x1 - WIN.x0, F + ROOM.h - WIN.y1).translate(0, (WIN.y1 + F + ROOM.h) / 2, 0), plaster),
    // The chimney breast and its firebox.
    slab(1.95, ROOM.x1, F, F + ROOM.h, 1.95, 2.35, hearthStone),
    slab(1.95, ROOM.x1, F, F + ROOM.h, 3.25, 3.65, hearthStone),
    slab(1.95, ROOM.x1, F + 0.9, F + ROOM.h, 2.35, 3.25, hearthStone),
    slab(2.5, ROOM.x1, F, F + 0.9, 2.35, 3.25, soot)
  ];
  world.add(new THREE.Mesh(merge(walls), indoors(stonework(new THREE.MeshLambertMaterial({ vertexColors: true }), true))));
  var inner = [
    tinted(new THREE.PlaneGeometry(ROOM.x1 - ROOM.x0, ROOM.z1).rotateX(Math.PI / 2).translate(0, F + ROOM.h, ROOM.z1 / 2), '#3a3026'),
    slab(WIN.x0 - 0.1, WIN.x1 + 0.1, WIN.y0 - 0.04, WIN.y0 + 0.02, -0.12, 0.2, wood),      // the sill
    // Ceiling beams.
    slab(ROOM.x0, ROOM.x1, F + ROOM.h - 0.2, F + ROOM.h, 1.2, 1.42, '#2a2018'),
    slab(ROOM.x0, ROOM.x1, F + ROOM.h - 0.2, F + ROOM.h, 3.0, 3.22, '#2a2018'),
    // The hearth slab and the mantel.
    slab(1.55, ROOM.x1, F, F + 0.05, 2.05, 3.55, '#3e3934'),
    slab(1.8, ROOM.x1, F + 1.12, F + 1.2, 1.85, 3.75, wood),
    // An empty chair turned to the fire.
    slab(1.1, 1.6, F + 0.45, F + 0.5, 1.5, 2.0, wood), slab(1.07, 1.13, F + 0.5, F + 1.25, 1.5, 2.0, wood),
    slab(1.11, 1.16, F, F + 0.45, 1.52, 1.57, wood), slab(1.11, 1.16, F, F + 0.45, 1.93, 1.98, wood),
    slab(1.54, 1.59, F, F + 0.45, 1.52, 1.57, wood), slab(1.54, 1.59, F, F + 0.45, 1.93, 1.98, wood),
    // Split logs stacked by the hearth.
    tinted(new THREE.CylinderGeometry(0.07, 0.07, 0.5, 7).rotateZ(Math.PI / 2).translate(2.3, F + 0.07, 1.62), '#4a3424'),
    tinted(new THREE.CylinderGeometry(0.07, 0.07, 0.5, 7).rotateZ(Math.PI / 2).translate(2.3, F + 0.07, 1.78), '#4e3828'),
    tinted(new THREE.CylinderGeometry(0.07, 0.07, 0.5, 7).rotateZ(Math.PI / 2).translate(2.3, F + 0.19, 1.7), '#463222'),
    // A rug.
    tinted(new THREE.PlaneGeometry(1.6, 1.1).rotateX(-Math.PI / 2).translate(0.3, F + 0.004, 2.8), '#4a1c1a')
  ];
  world.add(new THREE.Mesh(merge(inner), indoors(new THREE.MeshLambertMaterial({ vertexColors: true }))));

  // The cup on the sill, its steam (aromas); brass on the mantel (metals).
  var cupPts = [[0, 0], [0.035, 0], [0.042, 0.01], [0.046, 0.08], [0.044, 0.09]].map(function (p) { return new THREE.Vector2(p[0], p[1]); });
  var CUP = new THREE.Vector3(0.42, WIN.y0 + 0.02, 0.08);
  var cup = new THREE.Mesh(new THREE.LatheGeometry(cupPts, 14).translate(CUP.x, CUP.y, CUP.z), indoors(new THREE.MeshLambertMaterial({ color: '#d8d0c4', side: THREE.DoubleSide })));
  world.add(cup);
  var brass = indoors(new THREE.MeshStandardMaterial({ color: '#c09040', metalness: 0.5, roughness: 0.38, emissive: '#1a1006' }));
  var stickPts = [[0, 0], [0.07, 0], [0.065, 0.02], [0.02, 0.04], [0.016, 0.2], [0.035, 0.22], [0.03, 0.24], [0.015, 0.24], [0, 0.24]]
    .map(function (p) { return new THREE.Vector2(p[0], p[1]); });
  var CANDLE = new THREE.Vector3(2.2, F + 1.2, 2.25), TIN = new THREE.Vector3(2.25, F + 1.2, 3.3);
  world.add(new THREE.Mesh(new THREE.LatheGeometry(stickPts, 16).translate(CANDLE.x, CANDLE.y, CANDLE.z), brass));
  world.add(new THREE.Mesh(new THREE.CylinderGeometry(0.013, 0.013, 0.14, 8).translate(CANDLE.x, CANDLE.y + 0.31, CANDLE.z),
                           indoors(new THREE.MeshLambertMaterial({ color: '#e8dcc0' }))));
  world.add(new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.09, 14).translate(TIN.x, TIN.y + 0.045, TIN.z), brass));
  world.add(new THREE.Mesh(new THREE.TorusGeometry(0.11, 0.012, 6, 20).rotateY(Math.PI / 2).translate(2.55, F + 1.62, 2.8), brass));   // a hanging ring of keys

  // The fire: two logs (the near one wrinkled, its cracks glowing), the
  // ash, the flames and their sparks.
  var crackTex = canvasTexture(256, 128, function (x, w, h) {
    x.fillStyle = '#000'; x.fillRect(0, 0, w, h);
    x.strokeStyle = '#ff7a2a';
    x.shadowColor = '#ff5a10';
    x.shadowBlur = 6;
    for (var q = 0; q < 34; q++) {
      x.lineWidth = 0.8 + r() * 2.2;
      x.beginPath();
      var cx = r() * w, cy = r() * h;
      x.moveTo(cx, cy);
      for (var s = 0; s < 4; s++) { cx += (r() - 0.5) * 34; cy += (r() - 0.3) * 14; x.lineTo(cx, cy); }
      x.stroke();
    }
  });
  var logMat = indoors(new THREE.MeshLambertMaterial({ color: '#4a3424', emissive: '#ff7030', emissiveMap: crackTex, emissiveIntensity: 0 }));
  function logGeometry(rad, len, seed) {
    var g = new THREE.CylinderGeometry(rad, rad * 1.08, len, 18, 12), p = g.attributes.position;
    for (var j = 0; j < p.count; j++) {
      var x = p.getX(j), y = p.getY(j), z = p.getZ(j), a = Math.atan2(z, x), rr = Math.hypot(x, z);
      if (rr < 1e-4) continue;
      var w = 1 + 0.07 * Math.sin(a * 9 + y * 6 + seed) + 0.05 * Math.sin(a * 17 - y * 11) + 0.04 * Math.sin(y * 23 + a * 3);
      p.setX(j, x * w); p.setZ(j, z * w);
    }
    g.computeVertexNormals();
    return g.rotateX(Math.PI / 2);
  }
  var log1 = new THREE.Mesh(logGeometry(0.12, 0.78, 1), logMat);
  log1.position.set(FIRE.x - 0.08, F + 0.15, FIRE.z);
  log1.rotation.y = 0.12;
  var log2 = new THREE.Mesh(logGeometry(0.1, 0.7, 4), logMat);
  log2.position.set(FIRE.x + 0.12, F + 0.27, FIRE.z + 0.02);
  log2.rotation.set(0.2, -0.25, 0);
  var ash = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2).scale(0.3, 0.07, 0.42).translate(FIRE.x, F + 0.03, FIRE.z),
                           indoors(new THREE.MeshLambertMaterial({ color: '#6a645e' })));
  world.add(log1, log2, ash);
  var flameU = [], flames = [];
  [[0, 0.82, 0.95, 0.3], [0.55, 0.62, 0.75, 1.7], [-0.6, 0.6, 0.7, 3.1]].forEach(function (fl, n) {
    var u = { uTime: { value: 0 }, uFire: { value: 0 }, uSeed: { value: fl[3] } };
    var m = new THREE.Mesh(new THREE.PlaneGeometry(fl[1], fl[2]).translate(0, fl[2] / 2, 0), new THREE.ShaderMaterial({ uniforms: u,
      vertexShader: PLANE_VS, fragmentShader: FLAME_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide }));
    m.position.set(FIRE.x + 0.02 * n, F + 0.12, FIRE.z + (n - 1) * 0.08);
    m.rotation.y = -Math.PI / 2 + fl[0];
    flameU.push(u);
    flames.push(m);
    world.add(m);
  });
  var fireGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,140,60,0.9)', 'rgba(255,80,20,0)'),
    blending: THREE.AdditiveBlending, transparent: true, depthWrite: false, opacity: 0 }));
  fireGlow.position.set(FIRE.x - 0.15, F + 0.3, FIRE.z);
  fireGlow.scale.set(1.3, 1.0, 1);
  world.add(fireGlow);
  function emberField(n, origin, h, spread, rise, size) {
    var seeds = new Float32Array(n * 3);
    for (var j = 0; j < n * 3; j++) seeds[j] = r();
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(seeds, 3));
    var u = { uOrigin: { value: origin }, uTime: { value: 0 }, uAmt: { value: 0 }, uH: { value: h }, uSpread: { value: spread },
              uRise: { value: rise }, uSize: { value: size }, uScale: { value: 700 }, uWind: { value: 0 } };
    var pts = new THREE.Points(geo, new THREE.ShaderMaterial({ uniforms: u, vertexShader: EMBER_VS, fragmentShader: EMBER_FS,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
    pts.frustumCulled = false;
    world.add(pts);
    return u;
  }
  var hearthSparks = emberField(small ? 40 : 70, new THREE.Vector3(FIRE.x, F + 0.2, FIRE.z), 0.75, 0.35, 0.5, 0.025);
  var chimneySparks = emberField(small ? 90 : 180, new THREE.Vector3(X1 + 0.15, F + 5.2, 2.8), 8, 0.45, 1.4, 0.26);

  // The moonlight through the window, laid on the floor (the window's
  // shape cast along the moon's rays and clipped at the back wall), its
  // shaft in the air and the dust that drifts in it.
  var poolTex = canvasTexture(256, 256, function (x, w, h) {
    x.fillStyle = '#000'; x.fillRect(0, 0, w, h);
    if ('filter' in x) x.filter = 'blur(5px)';
    x.fillStyle = '#fff';
    [[16, 16], [134, 16], [16, 134], [134, 134]].forEach(function (q) { x.fillRect(q[0], q[1], 106, 106); });
  });
  function onFloor(px, py, out) {
    var t = (py - F) / MOON.y;
    return out.set(px - MOON.x * t, F + 0.006, -0.16 - MOON.z * t);
  }
  var c00 = onFloor(WIN.x0, WIN.y0, new THREE.Vector3()), c10 = onFloor(WIN.x1, WIN.y0, new THREE.Vector3());
  var c01 = onFloor(WIN.x0, WIN.y1, new THREE.Vector3()), c11 = onFloor(WIN.x1, WIN.y1, new THREE.Vector3());
  var vMax = clamp((ROOM.z1 - 0.05 - c00.z) / (c01.z - c00.z), 0, 1);
  c01.lerpVectors(c00, c01, vMax); c11.lerpVectors(c10, c11, vMax);
  var poolGeo = new THREE.BufferGeometry();
  poolGeo.setAttribute('position', new THREE.Float32BufferAttribute([c00.x, c00.y, c00.z, c10.x, c10.y, c10.z, c11.x, c11.y, c11.z,
                                                                     c00.x, c00.y, c00.z, c11.x, c11.y, c11.z, c01.x, c01.y, c01.z], 3));
  poolGeo.setAttribute('uv', new THREE.Float32BufferAttribute([0, 1, 1, 1, 1, 1 - vMax, 0, 1, 1, 1 - vMax, 0, 1 - vMax], 2));
  var pool = new THREE.Mesh(poolGeo, new THREE.MeshBasicMaterial({ map: poolTex, color: '#7f94c8', transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, side: THREE.DoubleSide }));
  world.add(pool);
  // The shaft: the frustum between the window and the pool.
  var w00 = new THREE.Vector3(WIN.x0, WIN.y0, -0.16), w10 = new THREE.Vector3(WIN.x1, WIN.y0, -0.16);
  var w01 = new THREE.Vector3(WIN.x0, WIN.y1, -0.16), w11 = new THREE.Vector3(WIN.x1, WIN.y1, -0.16);
  w01.lerpVectors(w00, w01, vMax); w11.lerpVectors(w10, w11, vMax);
  var shaftPos = [];
  [[w00, w10, c10, c00], [w10, w11, c11, c10], [w11, w01, c01, c11], [w01, w00, c00, c01]].forEach(function (q) {
    [q[0], q[1], q[2], q[0], q[2], q[3]].forEach(function (p) { shaftPos.push(p.x, p.y, p.z); });
  });
  var shaftGeo = new THREE.BufferGeometry();
  shaftGeo.setAttribute('position', new THREE.Float32BufferAttribute(shaftPos, 3));
  var shaft = new THREE.Mesh(shaftGeo, new THREE.MeshBasicMaterial({ color: '#6a80c0', transparent: true, opacity: 0.03, depthWrite: false,
    blending: THREE.AdditiveBlending, side: THREE.DoubleSide, fog: false }));
  world.add(shaft);
  var ND = small ? 160 : 320, dustPos = new Float32Array(ND * 3), dustCol = new Float32Array(ND * 3), dustA = new Float32Array(ND), dustSize = new Float32Array(ND);
  var dustHome = [];
  for (i = 0; i < ND; i++) {
    var du = r(), dv = r() * vMax, dt0 = 0.1 + r() * 0.85;
    tmpV.set(lerp(WIN.x0, WIN.x1, du), lerp(WIN.y0, WIN.y1, dv), -0.16);
    tmpV2.copy(c00).lerp(c10, du).lerp(tmpV2.copy(c01).lerp(c11, du), dv / vMax);
    dustHome.push(tmpV.clone().lerp(onFloor(tmpV.x, tmpV.y, tmpV2), dt0));
    dustCol.set([0.75, 0.82, 1.0], i * 3);
    dustA[i] = 0.25 + r() * 0.5;
    dustSize[i] = 0.006 + r() * 0.01;
  }
  var dust = glowPoints(dustPos, dustCol, dustA, dustSize, 1);
  world.add(dust);

  // Steam from the cup.
  var steamTex = puffTexture(r), steam = [];
  for (i = 0; i < 6; i++) {
    var sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: steamTex, color: '#d8d4e8', transparent: true, depthWrite: false, opacity: 0 }));
    sp.userData.ph = i / 6;
    steam.push(sp);
    world.add(sp);
  }

  // ── The casement: two leaves of four panes, hinged at the outer jambs ──
  var leafGeo = merge([slab(0, 0.72, -0.7, -0.65, -0.03, 0.03, wood), slab(0, 0.72, 0.65, 0.7, -0.03, 0.03, wood),
                       slab(0, 0.05, -0.7, 0.7, -0.03, 0.03, wood), slab(0.67, 0.72, -0.7, 0.7, -0.03, 0.03, wood),
                       slab(0.34, 0.38, -0.7, 0.7, -0.02, 0.02, wood), slab(0, 0.72, -0.02, 0.02, -0.02, 0.02, wood)]);
  var glassGeo = new THREE.PlaneGeometry(0.67, 1.35).translate(0.36, 0, 0);
  var leafMat = new THREE.MeshLambertMaterial({ vertexColors: true });
  var glassMat = new THREE.MeshBasicMaterial({ color: '#8ea0c0', transparent: true, opacity: 0.06, depthWrite: false, side: THREE.DoubleSide });
  var leaves = [-1, 1].map(function (sd) {
    var g = new THREE.Group();
    g.add(new THREE.Mesh(leafGeo, leafMat), new THREE.Mesh(glassGeo, glassMat));
    g.position.set(sd < 0 ? WIN.x0 : WIN.x1, (WIN.y0 + WIN.y1) / 2, -WALL + 0.06);
    g.scale.x = sd < 0 ? 1 : -1;
    world.add(g);
    return g;
  });

  // ── The tree on its rock ──
  var tree = new THREE.Group();
  var treeMesh = new THREE.Mesh(treeGeometry(), new THREE.MeshLambertMaterial({ vertexColors: true, emissive: '#2a0806' }));
  treeMesh.morphTargetInfluences = [0, 0];
  tree.add(treeMesh);
  world.add(tree);
  // A flowering vine, climbing the trunk as uGrow rises.
  var vinePts = [], trunkC = new THREE.CatmullRomCurve3(TRUNK.map(function (p) { return new THREE.Vector3(p[0], p[1], p[2]); }));
  for (i = 0; i <= 60; i++) {
    var vt = i / 60, va = vt * Math.PI * 2 * 2.4 + 0.6, vr = lerp(0.38, 0.2, vt);
    trunkC.getPointAt(clamp(0.12 + vt * 0.86, 0, 1), tmpV);
    vinePts.push(new THREE.Vector3(tmpV.x + Math.cos(va) * vr, tmpV.y, tmpV.z + Math.sin(va) * vr));
  }
  var vineGeo = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(vinePts), 120, 0.03, 4, false);
  var vineT = new Float32Array(vineGeo.attributes.position.count);
  for (i = 0; i < vineT.length; i++) vineT[i] = Math.floor(i / 5) / 120;
  vineGeo.setAttribute('aT', new THREE.BufferAttribute(vineT, 1));
  var vineGrow = { value: 0 };
  var vineMat = new THREE.MeshLambertMaterial({ color: '#3d5a2a' });
  vineMat.onBeforeCompile = function (sh) {
    sh.uniforms.uGrow = vineGrow;
    sh.vertexShader = 'attribute float aT; varying float vT;\n' + sh.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\n vT = aT;');
    sh.fragmentShader = 'uniform float uGrow; varying float vT;\n' + sh.fragmentShader.replace('void main() {', 'void main() {\n if (vT > uGrow) discard;');
  };
  var vine = new THREE.Mesh(vineGeo, vineMat);
  vine.visible = false;
  tree.add(vine);
  var bloomPos = [], bloomT = [], bloomSize = [], vc = new THREE.CatmullRomCurve3(vinePts);
  for (i = 0; i < 26; i++) {
    var bt = 0.05 + i / 26 * 0.92;
    vc.getPointAt(bt, tmpV);
    bloomPos.push(tmpV.x, tmpV.y, tmpV.z);
    bloomT.push(bt);
    bloomSize.push(0.3 + r() * 0.12);
  }
  for (i = 0; i < 8; i++) {      // a cluster at the top, where the crown begins
    vc.getPointAt(1, tmpV);
    bloomPos.push(tmpV.x + (r() - 0.5) * 0.5, tmpV.y + 0.1 + r() * 0.35, tmpV.z + (r() - 0.5) * 0.5);
    bloomT.push(0.9 + i * 0.015);
    bloomSize.push(0.4 + r() * 0.12);
  }
  var bloomGeo = new THREE.BufferGeometry();
  bloomGeo.setAttribute('position', new THREE.Float32BufferAttribute(bloomPos, 3));
  bloomGeo.setAttribute('aT', new THREE.Float32BufferAttribute(bloomT, 1));
  bloomGeo.setAttribute('aSize', new THREE.Float32BufferAttribute(bloomSize, 1));
  var bloomU = { uGrow: { value: 0 }, uScale: { value: 700 }, uTime: { value: 0 }, uPetal: { value: new THREE.Color('#ffd8e2') },
                 uHeart: { value: new THREE.Color('#ffc060') }, uLight: { value: 1 } };
  var blooms = new THREE.Points(bloomGeo, new THREE.ShaderMaterial({ uniforms: bloomU, vertexShader: BLOOM_VS, fragmentShader: BLOOM_FS,
    transparent: true, depthWrite: false }));
  blooms.frustumCulled = false;
  tree.add(blooms);
  // Rings spreading on the water behind the tree as it goes.
  var ringTex = canvasTexture(128, 128, function (x) {
    x.strokeStyle = 'rgba(255,255,255,0.9)';
    x.lineWidth = 5;
    if ('filter' in x) x.filter = 'blur(2px)';
    x.beginPath(); x.arc(64, 64, 52, 0, Math.PI * 2); x.stroke();
  });
  var rings = [];
  for (i = 0; i < 8; i++) {
    var rm = new THREE.Mesh(new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ map: ringTex, color: '#9fb0d0',
      transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending }));
    rm.position.y = 0.03;
    rm.userData.t0 = -99;
    rings.push(rm);
    world.add(rm);
  }
  var ringIdx = 0, ringClock = 0, lastTree = new THREE.Vector3().copy(TREE_BASE);

  // ── Motes that become boats ──
  // Each comes from one of the four things he names, gathers at the window,
  // goes down to the water as a little boat and sails for the isles.
  var NB = small ? 20 : 30;
  var SOURCES = [
    { p: new THREE.Vector3(FIRE.x - 0.1, F + 0.2, FIRE.z), s: 0.25, c: '#ffa458' },          // the log's embers, the ash
    { p: new THREE.Vector3(CUP.x, CUP.y + 0.12, CUP.z), s: 0.08, c: '#e8dcff' },             // aromas
    { p: new THREE.Vector3(-1.0, F + 0.15, 3.0), s: 0.5, c: '#b8ccff' },                      // light
    { p: new THREE.Vector3(CANDLE.x - 0.05, CANDLE.y + 0.2, 2.8), s: 0.6, c: '#ffd88a' }       // metals
  ];
  var boats = [], bAttrPos = new Float32Array(NB * 3), bAttrCol = new Float32Array(NB * 3), bAttrA = new Float32Array(NB), bAttrSize = new Float32Array(NB);
  var mirPos = new Float32Array(NB * 3);
  for (i = 0; i < NB; i++) {
    var src = SOURCES[i % 4], col = new THREE.Color(src.c);
    var S = src.p.clone().add(new THREE.Vector3((r() - 0.5) * src.s, r() * 0.25, (r() - 0.5) * src.s));
    var W = new THREE.Vector3(lerp(WIN.x0 + 0.2, WIN.x1 - 0.2, r()), lerp(WIN.y0 + 0.3, WIN.y1 - 0.25, r()), -WALL - 0.2);
    var ang = (r() - 0.5) * 1.1, dist = 13 + Math.pow(r(), 0.8) * 15;
    var B = new THREE.Vector3(Math.sin(ang) * dist * 0.9 + 1.5, 0, -dist);
    var head = new THREE.Vector3(-B.x / 700, 0, -1).normalize();
    boats.push({ S: S, W: W, B: B, head: head, col: col, delay: r() * 0.4, thr: 0.04 + (i / NB) * 0.82 + r() * 0.05,
                 ph: r() * 6.28, yaw: Math.atan2(-head.x, -head.z), speed: 26 + r() * 14 });
    bAttrCol.set([col.r, col.g, col.b], i * 3);
  }
  var boatGeo = boatGeometry(), glowAttr = new THREE.InstancedBufferAttribute(new Float32Array(NB * 3), 3);
  boatGeo.setAttribute('aGlow', glowAttr);
  var boatMat = new THREE.MeshLambertMaterial({ color: '#8a8890', side: THREE.DoubleSide });
  boatMat.onBeforeCompile = function (sh) {
    sh.vertexShader = 'attribute vec3 aGlow; varying vec3 vGlow;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n vGlow = aGlow * (0.45 + 0.55 * smoothstep(0.5, 0.05, position.y));');
    sh.fragmentShader = 'varying vec3 vGlow;\n' + sh.fragmentShader.replace('#include <emissivemap_fragment>',
      '#include <emissivemap_fragment>\n totalEmissiveRadiance += vGlow;');
  };
  var boatMesh = new THREE.InstancedMesh(boatGeo, boatMat, NB);
  boatMesh.frustumCulled = false;
  world.add(boatMesh);
  var boatGlow = glowPoints(bAttrPos, bAttrCol, bAttrA, bAttrSize, 1);
  world.add(boatGlow);
  var boatMirror = glowPoints(mirPos, bAttrCol, bAttrA, bAttrSize, 0.55);
  boatMirror.material.uniforms.uMirror.value = 1;
  world.add(boatMirror);

  // ── Wind: mist, torn leaves, banners ──
  var puff = puffTexture(r), mists = [];
  for (i = 0; i < (small ? 26 : 40); i++) {
    var far = i % 3 !== 0, mx = (r() - 0.5) * (far ? 1600 : 260), mz = far ? -420 - r() * 480 : -45 - r() * 260;
    var ms = new THREE.Sprite(new THREE.SpriteMaterial({ map: puff, color: '#8090aa', transparent: true, depthWrite: false, opacity: 0, fog: false }));
    ms.position.set(mx, far ? 15 + r() * 40 : 3 + r() * 6, mz);
    ms.scale.set(far ? 380 + r() * 380 : 70 + r() * 70, far ? 70 + r() * 70 : 12 + r() * 10, 1);
    ms.userData = { x: mx, ph: r() * 6.28, far: far };
    world.add(ms);
    mists.push(ms);
  }
  var NL = small ? 240 : 480, LB = [36, 12, 36], leafPos = new Float32Array(NL * 3), leafSeed = new Float32Array(NL);
  for (i = 0; i < NL; i++) {
    leafPos[i * 3] = (r() - 0.5) * LB[0]; leafPos[i * 3 + 1] = r() * LB[1]; leafPos[i * 3 + 2] = (r() - 0.5) * LB[2];
    leafSeed[i] = r();
  }
  var leafGeo = new THREE.BufferGeometry();
  leafGeo.setAttribute('position', new THREE.BufferAttribute(leafPos, 3));
  leafGeo.setAttribute('aSeed', new THREE.BufferAttribute(leafSeed, 1));
  var leafU = { uAmt: { value: 0 }, uScale: { value: 700 }, uTime: { value: 0 }, uLight: { value: 1 } };
  var torn = new THREE.Points(leafGeo, new THREE.ShaderMaterial({ uniforms: leafU, vertexShader: LEAF_VS, fragmentShader: LEAF_FS,
    transparent: true, depthWrite: false }));
  torn.frustumCulled = false;
  world.add(torn);
  function wrap(v, c, size) { return c - size / 2 + ((((v - c + size / 2) % size) + size) % size); }

  var NBAN = small ? 10 : 16, banGeo = new THREE.PlaneGeometry(1, 1, 40, 1).translate(0.5, 0, 0);
  var aBan = new Float32Array(NBAN * 4), aLane = new Float32Array(NBAN * 4);
  for (i = 0; i < NBAN; i++) {
    aBan.set([7 + r() * 8, 0.45 + r() * 0.6, r() * 6.28, i % 5], i * 4);
    aLane.set([r() * 70, 1.6 + r() * 6.5, -23 + r() * 15, 0.7 + r() * 0.6], i * 4);
  }
  banGeo.setAttribute('aBan', new THREE.InstancedBufferAttribute(aBan, 4));
  banGeo.setAttribute('aLane', new THREE.InstancedBufferAttribute(aLane, 4));
  var banU = { uTime: { value: 0 }, uFlow: { value: 0 }, uAmt: { value: 0 }, uX0: { value: -32 }, uSpan: { value: 70 }, uLight: { value: 1 },
               uC0: { value: new THREE.Color('#b0262a') }, uC1: { value: new THREE.Color('#e09a30') }, uC2: { value: new THREE.Color('#3a4aa0') },
               uC3: { value: new THREE.Color('#e6dccb') }, uC4: { value: new THREE.Color('#2f7a58') } };
  var banners = new THREE.InstancedMesh(banGeo, new THREE.ShaderMaterial({ uniforms: banU, vertexShader: BANNER_VS, fragmentShader: BANNER_FS,
    transparent: true, depthWrite: false, side: THREE.DoubleSide }), NBAN);
  banners.frustumCulled = false;
  world.add(banners);
  var flow = 0;

  // ── Per frame ──
  var NIGHT = { top: new THREE.Color('#02040b'), mid: new THREE.Color('#0a1330'), hor: new THREE.Color('#1c2a50') };
  var DAY = { top: new THREE.Color('#121a40'), mid: new THREE.Color('#3c3060'), hor: new THREE.Color('#b0685a') };
  var MISTC = new THREE.Color('#3a4560'), boatTmp = new THREE.Color();
  var bez = new THREE.Vector3(), ctl = new THREE.Vector3();
  function quad(a, c, b, t, out) {                 // a quadratic Bezier from a to b through control c
    var s = 1 - t;
    return out.set(s * s * a.x + 2 * s * t * c.x + t * t * b.x, s * s * a.y + 2 * s * t * c.y + t * t * b.y, s * s * a.z + 2 * s * t * c.z + t * t * b.z);
  }
  var GO0 = new THREE.Vector3(TREE_BASE.x, 0, TREE_BASE.z), GO1 = new THREE.Vector3(ROCK.x - 6, 0, ROCK.z - 3.5), GO2 = new THREE.Vector3(ROCK.x - 15, 0, ROCK.z - 6.5);

  function frame(f) {
    var row = f.row, time = f.time, dt = f.dt;
    var fire = row[5], open = row[6], journey = row[7], dim = row[8], phase = row[9], gust = row[10], mist = row[11];
    var banAmt = row[12], lift = row[13], roots = row[14], go = row[15], dawn = row[16], flower = row[17];
    var shake = env.reduceMotion ? 0 : gust, afloat = smooth(-2, -8, row[2]);

    // ── Camera ──
    camera.position.set(row[0], row[1] + Math.sin(time * 0.6) * 0.05 * afloat, row[2]);
    camera.rotation.set(row[4] + (portrait ? row[19] : 0) - f.my * 0.05 + Math.sin(time * 2.3) * 0.008 * shake,
                        row[3] + (portrait ? row[18] : 0) - f.mx * 0.1 + Math.sin(time * 1.7) * 0.012 * shake,
                        Math.sin(time * 0.5) * 0.012 * afloat + Math.sin(time * 2.9) * 0.012 * shake, 'YXZ');
    sky.position.copy(camera.position);
    water.position.set(camera.position.x, 0, camera.position.z);

    // ── Sky, light and air: night to dawn ──
    var moonAmt = (0.45 + 0.55 * phase) * (1 - dawn);
    skyU.uTop.value.copy(NIGHT.top).lerp(DAY.top, dawn);
    skyU.uMid.value.copy(NIGHT.mid).lerp(DAY.mid, dawn);
    var hor = skyU.uHorizon.value.copy(NIGHT.hor).lerp(DAY.hor, dawn).lerp(MISTC, mist * 0.45 * (1 - dawn));
    skyU.uDawn.value = dawn;
    skyU.uMoonGlow.value = moonAmt * (1 - mist * 0.7);
    moonU.uPhase.value = phase;
    moonU.uAlpha.value = (1 - dawn * 0.85) * (1 - mist * 0.6);
    moonHalo.material.opacity = moonAmt * (1 - mist * 0.7) * 0.55;
    moon.lookAt(camera.position);
    stars.material.opacity = 0.85 * (1 - dawn) * (1 - mist * 0.8);
    world.fog.color.copy(hor);
    world.fog.density = 0.0045 + mist * 0.01;
    hemi.intensity = 0.45 + dawn * 0.45;
    hemi.color.set('#4f5f86').lerp(tmpC.set('#d0a898'), dawn);
    hemi.groundColor.set('#16130f').lerp(tmpC.set('#3a2a20'), dawn);
    moonLight.intensity = moonAmt * 1.4 * (1 - mist * 0.4);
    dawnLight.intensity = dawn * dawn * 1.5;
    waterU.uMoon.value = moonAmt * (1 - mist * 0.8);
    waterU.uChop.value = gust;
    waterU.uTime.value = time;
    waterU.uFogColor.value.copy(hor);
    waterU.uFogDensity.value = 0.0013 + mist * 0.002;
    isleU.uDawn.value = dawn;
    isleU.uMoon.value = moonAmt;
    isleU.uMist.value = mist;
    isleU.uHaze.value.copy(hor);
    lamps.material.uniforms.uAmt.value = (0.5 + dawn * 0.8) * (1 - mist);
    lamps.material.uniforms.uTime.value = time;
    lamps.material.uniforms.uScale.value = pxScale;
    for (i = 0; i < mists.length; i++) {
      var ms = mists[i], mu = ms.userData;
      ms.position.x = mu.x + Math.sin(time * 0.02 + mu.ph) * (mu.far ? 40 : 10) + (env.reduceMotion ? 0 : gust * time * 4 % 60);
      ms.material.opacity = mist * (mu.far ? 0.55 : 0.3) * (0.75 + 0.25 * Math.sin(time * 0.15 + mu.ph));
      ms.material.color.set('#8090aa').lerp(tmpC.set('#e8b8a0'), dawn);
      ms.visible = ms.material.opacity > 0.005;
    }

    // ── The fire ──
    var flick = 0.85 + 0.1 * Math.sin(time * 7.3) * Math.sin(time * 3.1 + 1) + 0.05 * Math.sin(time * 17.0);
    // Blazing: the fire at its height, flickering harder.
    var blazing = smooth(0.6, 1, fire), flick2 = 0.8 + 0.12 * Math.sin(time * 11.0) * Math.sin(time * 4.3 + 2) + 0.08 * Math.sin(time * 23.0);
    fireLight.intensity = (0.4 + fire * 5.5) * flick + blazing * 6 * flick2;
    spill.intensity = (2.5 + fire * fire * 10 + blazing * 30 * flick2) * flick;
    for (i = 0; i < flameU.length; i++) { flameU[i].uTime.value = time; flameU[i].uFire.value = fire; }
    fireGlow.material.opacity = (0.25 + fire * 0.6) * flick;
    logMat.emissiveIntensity = (0.35 + fire * 1.4) * flick;
    hearthSparks.uTime.value = time; hearthSparks.uAmt.value = fire; hearthSparks.uScale.value = pxScale;
    chimneySparks.uTime.value = time; chimneySparks.uAmt.value = smooth(0.6, 1, fire); chimneySparks.uScale.value = pxScale;
    chimneySparks.uWind.value = 0.15;
    winGlow.material.opacity = (0.08 + fire * 0.4 + blazing * 0.45 * flick2) * flick;
    winGlow.scale.set(3.6 + blazing * 2.4, 3.2 + blazing * 2.0, 1);
    blaze.material.uniforms.uAmt.value = blazing;
    blaze.material.uniforms.uTime.value = time;
    blaze.visible = blazing > 0.01 && camera.position.z < -0.6;

    // ── The room's moonlight, dust and steam ──
    var inRoom = smooth(-1.5, 0.5, camera.position.z);
    pool.material.opacity = moonAmt * 0.9;
    shaft.material.opacity = 0.01 * moonAmt;
    var da = dust.geometry.attributes.position.array;
    for (i = 0; i < ND; i++) {
      var dh = dustHome[i];
      da[i * 3] = dh.x + Math.sin(time * 0.11 + i) * 0.05;
      da[i * 3 + 1] = dh.y + Math.sin(time * 0.07 + i * 1.7) * 0.06;
      da[i * 3 + 2] = dh.z + Math.cos(time * 0.09 + i * 0.6) * 0.05;
    }
    dust.geometry.attributes.position.needsUpdate = true;
    dust.material.uniforms.uAmt.value = moonAmt * inRoom;
    dust.material.uniforms.uTime.value = time;
    dust.material.uniforms.uScale.value = pxScale;
    for (i = 0; i < steam.length; i++) {
      var sp = steam[i], st = (time * 0.12 + sp.userData.ph) % 1;
      sp.position.set(CUP.x + Math.sin(st * 6 + i) * 0.03 * st, CUP.y + 0.1 + st * 0.5, CUP.z + Math.cos(st * 5 + i) * 0.02);
      sp.scale.setScalar(0.06 + st * 0.22);
      sp.material.opacity = 0.16 * Math.sin(st * Math.PI) * inRoom;
    }

    // ── The casement ──
    var sw = open * open * (3 - 2 * open) * 1.75;
    leaves[0].rotation.y = sw;
    leaves[1].rotation.y = -sw;

    // ── Motes and boats ──
    var warm = smooth(0.4, 1, dawn) * 0.5;
    for (i = 0; i < NB; i++) {
      var b = boats[i], p = bez, scale = 0, lit = 1 - smooth(b.thr, b.thr + 0.07, dim), size = 0.05, yaw = b.yaw;
      if (journey <= 1) {
        var t1 = smooth(0, 1, clamp((journey - b.delay * 0.7) / 0.55, 0, 1));
        ctl.copy(b.S).lerp(b.W, 0.35).y += 0.9;
        quad(b.S, ctl, b.W, t1, p);
        p.x += Math.sin(time * 1.3 + b.ph) * 0.04 * (1 - t1) * t1 * 4;
        p.y += Math.sin(time * 1.7 + b.ph) * 0.03;
        size = 0.1;
      } else if (journey <= 2) {
        var t2 = smooth(0, 1, clamp((journey - 1 - b.delay * 0.5) / 0.7, 0, 1));
        ctl.copy(b.W).lerp(b.B, 0.3); ctl.y = b.W.y + 0.8;
        quad(b.W, ctl, b.B, t2, p);
        scale = smooth(0.75, 1, t2);
        size = lerp(0.1, 0.75, smooth(0.55, 1, t2));
      } else {
        p.copy(b.B).addScaledVector(b.head, (journey - 2) * b.speed);
        scale = 1;
        size = 0.75 + (journey - 2) * 0.35;
      }
      if (scale > 0) {
        p.y = Math.sin(time * 1.1 + b.ph) * 0.06 * scale + (1 - scale) * p.y;
        tmpQ.setFromEuler(tmpE.set(Math.sin(time * 0.9 + b.ph) * 0.06, yaw, Math.sin(time * 1.2 + b.ph) * 0.08));
        tmpM.compose(p, tmpQ, tmpV.setScalar(scale * 1.25));
      } else {
        tmpM.makeScale(0, 0, 0);
      }
      boatMesh.setMatrixAt(i, tmpM);
      boatTmp.copy(b.col).lerp(tmpC.set('#ffb860'), warm).multiplyScalar(lit * 1.5);
      glowAttr.setXYZ(i, boatTmp.r, boatTmp.g, boatTmp.b);
      bAttrPos[i * 3] = p.x; bAttrPos[i * 3 + 1] = p.y + 0.3 * scale; bAttrPos[i * 3 + 2] = p.z;
      mirPos[i * 3] = p.x; mirPos[i * 3 + 1] = 0.02; mirPos[i * 3 + 2] = p.z;
      bAttrA[i] = lit * (scale > 0 ? 1 : smooth(0, 0.08, journey - b.delay * 0.7) * (0.8 + 0.2 * Math.sin(time * 3 + b.ph)));
      bAttrSize[i] = size;
      bAttrCol[i * 3] = lerp(b.col.r, 1.0, warm); bAttrCol[i * 3 + 1] = lerp(b.col.g, 0.72, warm); bAttrCol[i * 3 + 2] = lerp(b.col.b, 0.38, warm);
    }
    boatMesh.instanceMatrix.needsUpdate = true;
    glowAttr.needsUpdate = true;
    ['position', 'aA', 'aSize', 'aCol'].forEach(function (n) { boatGlow.geometry.attributes[n].needsUpdate = true; });
    boatMirror.geometry.attributes.position.needsUpdate = true;
    boatGlow.visible = journey > 0.001 || dim < 1;
    [boatGlow, boatMirror].forEach(function (g) { g.material.uniforms.uTime.value = time; g.material.uniforms.uScale.value = pxScale; });
    boatMirror.material.uniforms.uAmt.value = 0.55 * smooth(1.6, 2, journey);

    // ── The tree: lift its arms, let go its roots, set off over the water ──
    var mt = treeMesh.morphTargetInfluences, lt = smooth(0, 1, lift), rt = smooth(0, 1, roots);
    mt[0] = lt; mt[1] = rt;
    var gt = smooth(0, 1, go);
    quad(GO0, GO1, GO2, gt, tree.position);
    tree.position.y = lerp(TREE_BASE.y + 1.5 * rt, 0.95, smooth(0, 0.45, go));
    var swayAmt = env.reduceMotion ? 0.3 : 1;
    tree.rotation.set(Math.sin(time * 0.6) * 0.01 * swayAmt + gust * 0.03 * Math.sin(time * 2.1),
                      gt * 0.7 + Math.sin(time * 0.4) * 0.03 * rt,
                      Math.sin(time * 0.5 + 1) * 0.012 * swayAmt + gust * 0.04 * (0.6 + 0.4 * Math.sin(time * 1.6)) + 0.05 * rt * Math.sin(time * 0.9));
    var moved = tmpV.copy(tree.position).sub(lastTree).length() / Math.max(dt, 1e-3);
    lastTree.copy(tree.position);
    ringClock += dt;
    if (moved > 0.3 && go > 0.05 && ringClock > 0.45) {
      ringClock = 0;
      var rm = rings[ringIdx++ % rings.length];
      rm.position.set(tree.position.x, 0.03, tree.position.z);
      rm.userData.t0 = time;
    }
    for (i = 0; i < rings.length; i++) {
      var age = time - rings[i].userData.t0;
      rings[i].visible = age < 4;
      if (!rings[i].visible) continue;
      rings[i].scale.setScalar(1.5 + age * 1.6);
      rings[i].material.opacity = 0.35 * (1 - age / 4) * (0.4 + 0.6 * moonAmt + dawn);
    }
    vineGrow.value = flower;
    vine.visible = flower > 0.001;
    bloomU.uGrow.value = flower;
    bloomU.uScale.value = pxScale;
    bloomU.uTime.value = time;
    bloomU.uLight.value = 1.1 + 0.6 * dawn;
    blooms.visible = flower > 0.05;

    // ── Wind: torn leaves and banners ──
    leafU.uAmt.value = clamp(gust * 0.9 + banAmt * 0.5, 0, 1);
    torn.visible = leafU.uAmt.value > 0.01;
    if (torn.visible) {
      var lw = (0.4 + gust * 1.6) * (env.reduceMotion ? 4 : 9), cp = camera.position;
      for (i = 0; i < NL; i++) {
        var lk = i * 3;
        leafPos[lk] = wrap(leafPos[lk] + (lw + Math.sin(time * 0.9 + i) * 0.8) * dt, cp.x, LB[0]);
        leafPos[lk + 1] = wrap(leafPos[lk + 1] - (0.3 + leafSeed[i] * 0.7 - gust * 0.5 * Math.sin(time * 1.3 + i)) * dt, cp.y + 2, LB[1]);
        leafPos[lk + 2] = wrap(leafPos[lk + 2] + Math.cos(time * 0.7 + i) * 0.5 * dt, cp.z, LB[2]);
      }
      leafGeo.attributes.position.needsUpdate = true;
      leafU.uTime.value = time;
      leafU.uScale.value = pxScale;
      leafU.uLight.value = 0.45 + 0.4 * moonAmt + dawn;
    }
    flow += dt * (env.reduceMotion ? 1.5 : 4) * (0.5 + banAmt);
    banU.uFlow.value = flow;
    banU.uTime.value = time;
    banU.uAmt.value = banAmt;
    banU.uLight.value = 0.5 + 0.5 * moonAmt + dawn;
    banners.visible = banAmt > 0.005;

    gl.render(world, camera);
  }

  function resize(w, h, dpr) {
    fitCamera(gl, camera, w, h, dpr, small);
    portrait = w < h;
    pxScale = h * gl.getPixelRatio() / (2 * Math.tan(camera.fov * Math.PI / 360));
  }

  return {
    resize: resize,
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('isles', {
  renderer: renderer3d,
  maxLines: 4,
  scrim: 0.6,
  accent: '#ffb36b',
  emphasis: /^fire\W*$/i,
  align: ['left', 'left', 'left', 'right', 'center', 'left', 'center', 'center', 'right', 'left', 'left', 'right', 'left', 'left'],
  // Panels: 0 I; 1-4 II (the window, the fire, "everything", the boats);
  // 5 III; 6 IV; 7-9 V (banners, roots, setting off); 10-13 VI.
  keys: function (T) {
    function at(i, d) { return T.start(Math.min(i, T.count - 1)) + d; }   // d units into panel i (0..1.6)
    var E = F + 1.5;
    //  unit          x      y         z      yaw    pitch  fire  open jour  dim  moon  gust  mist  ban   lift root go   dawn  flow  pyaw ppitch
    return [
      [0,             -0.35, E,        4.4,   -0.05, -0.16, 0.22, 0,   0.00, 0,   1.0,  0.0,  0.0,  0.0,  0,   0,   0,   0.0,  0,    0, 0],
      [0.7,           -0.35, E,        4.3,   -0.05, -0.16, 0.22, 0,   0.00, 0,   1.0,  0.0,  0.0,  0.0,  0,   0,   0,   0.0,  0,    0, 0],
      [at(0, 1.2),    -0.3,  E,        3.3,   -0.06, -0.1,  0.24, 0,   0.00, 0,   1.0,  0.0,  0.0,  0.0,  0,   0,   0,   0.0,  0,    0, 0],   // "I want you to know one thing"
      [at(1, 0.55),   -0.15, F + 1.3,  1.05,  -0.08,  0.16, 0.26, 0,   0.00, 0,   1.0,  0.0,  0.0,  0.0,  0,   0,   0,   0.0,  0,    0, 0],   // "the crystal moon, the red branch"
      [at(1, 1.2),    -0.12, F + 1.3,  1.0,   -0.1,   0.16, 0.28, 0,   0.00, 0,   1.0,  0.0,  0.0,  0.0,  0,   0,   0,   0.0,  0,    0, 0],
      [at(2, 0.4),     0.15, F + 1.3,  2.9,   -1.6,  -0.24, 0.55, 0,   0.00, 0,   1.0,  0.0,  0.0,  0.0,  0,   0,   0,   0.0,  0,    0, 0],   // "near the fire"
      [at(2, 1.1),     0.35, F + 1.25, 2.85,  -1.6,  -0.26, 0.8,  0,   0.00, 0,   1.0,  0.0,  0.0,  0.0,  0,   0,   0,   0.0,  0,    0, 0],   // "the impalpable ash"
      [at(3, 0.25),    0.35, F + 1.27, 2.8,   -1.55, -0.28, 0.75, 0,   0.15, 0,   1.0,  0.0,  0.0,  0.0,  0,   0,   0,   0.0,  0,    0, 0],   // "the wrinkled body of the log"
      [at(3, 0.65),    0.3,  F + 1.4,  2.5,   -0.15,  0.02, 0.7,  0.35, 0.55, 0,  1.0,  0.0,  0.0,  0.0,  0,   0,   0,   0.0,  0,    0.25, -0.12], // "everything carries me to you"
      [at(3, 1.35),    0.1,  E,        2.2,   -0.03,  0.04, 0.6,  1,   1.00, 0,   1.0,  0.0,  0.0,  0.0,  0,   0,   0,   0.0,  0,    0.05, -0.05], // "aromas, light, metals": out of the window
      [at(4, 0.1),     0.05, E,        1.6,    0.0,   0.02, 0.55, 1,   1.05, 0,   1.0,  0.0,  0.0,  0.0,  0,   0,   0,   0.0,  0,    0, 0],
      [at(4, 0.65),    0.05, E - 0.05, -0.7,   0.0,  -0.06, 0.5,  1,   1.6,  0,   1.0,  0.0,  0.0,  0.0,  0,   0,   0,   0.0,  0,    0, 0],   // "were little boats"
      [at(4, 1.4),     1.0,  3.0,      -9.5,  -0.02, -0.07, 0.45, 1,   2.05, 0,   1.0,  0.0,  0.0,  0.0,  0,   0,   0,   0.0,  0,    0, 0],   // "that sail toward those isles"
      [at(5, 0.25),    1.0,  2.5,      -15,    0.0,   0.06, 0.4,  1,   2.2,  0,   1.0,  0.0,  0.0,  0.0,  0,   0,   0,   0.0,  0,    0, 0],
      [at(5, 1.3),     0.8,  2.3,      -21,    0.02,  0.12, 0.25, 1,   2.55, 1,   0.25, 0.0,  0.05, 0.0,  0,   0,   0,   0.0,  0,    0, 0],   // "little by little"
      [at(6, 0.2),     0.9,  2.25,     -23,    0.02,  0.08, 0.15, 1,   2.62, 1,   0.2,  0.1,  0.1,  0.0,  0,   0,   0,   0.0,  0,    0, 0],
      [at(6, 0.45),    0.95, 2.2,      -24,    0.03,  0.01, 0.08, 1,   2.68, 1,   0.2,  1.0,  0.55, 0.0,  0,   0,   0,   0.0,  0,    0, 0],   // "If suddenly you forget me"
      [at(6, 1.25),    1.2,  2.2,      -25,    0.05,  0.02, 0.08, 1,   2.8,  1,   0.2,  0.75, 1.0,  0.0,  0,   0,   0,   0.0,  0,    0, 0],   // "forgotten"
      [at(7, 0.1),     2.2,  2.3,      -25.5,  1.4,   0.03, 0.08, 1,   2.85, 1,   0.2,  0.6,  0.85, 0.3,  0,   0,   0,   0.0,  0,    0, 0],
      [at(7, 0.6),     3.4,  2.5,      -25,    2.86,  0.05, 0.08, 1,   2.9,  1,   0.2,  0.5,  0.8,  1.0,  0,   0,   0,   0.0,  0,    0, 0],   // "the wind of banners"
      [at(7, 1.4),     0.6,  2.2,      -22,    2.8,   0.08, 0.08, 1,   2.93, 1,   0.2,  0.35, 0.7,  0.8,  0,   0,   0,   0.0,  0,    0, 0],
      [at(8, 0.7),    -2.5,  2.0,      -21.5,  2.76,  0.12, 0.08, 1,   2.97, 1,   0.2,  0.2,  0.6,  0.15, 0,   0,   0,   0.0,  0,    0, 0],   // "the shore of the heart where I have roots"
      [at(9, 0.45),   -2.5,  2.0,      -22,    2.76,  0.12, 0.08, 1,   3.0,  1,   0.2,  0.2,  0.55, 0.15, 0,   0,   0,   0.0,  0,    0, 0],
      [at(9, 0.78),   -2.5,  2.05,     -22.5,  2.76,  0.14, 0.08, 1,   3.0,  1,   0.2,  0.3,  0.5,  0.1,  1,   0,   0,   0.0,  0,    0, 0],   // "I shall lift my arms"
      [at(9, 1.0),    -2.5,  2.05,     -22.6,  2.72,  0.14, 0.08, 1,   3.0,  1,   0.2,  0.3,  0.5,  0.0,  1,   1,   0,   0.0,  0,    0, 0],   // "and my roots will set off"
      [at(10, 0.4),   -2.5,  2.1,      -22.6,  1.85,  0.1,  0.08, 1,   3.03, 1,   0.2,  0.2,  0.4,  0.0,  1,   1,   1,   0.1,  0,    0, 0],   // "to seek another land"
      [at(10, 0.95),  -2.5,  2.1,      -22.6,  1.8,   0.1,  0.08, 1,   3.05, 1,   0.2,  0.05, 0.3,  0.0,  0.6, 1,   1,   0.3,  0,    0,    0],     // "But if each day": it stops, and turns back
      [at(11, 0.25),  -3.4,  2.8,      -18.5,  2.45,  0.12, 0.08, 1,   3.07, 1,   0.2,  0.0,  0.22, 0.0,  0.35, 1,  0,   0.45, 0,    0.2,  0.1],   // "destined for me": it comes home
      [at(11, 0.5),   -3.8,  3.1,      -16.5,  2.48,  0.12, 0.08, 1,   3.08, 1,   0.2,  0.0,  0.2,  0.0,  0.25, 0,  0,   0.5,  0.15, 0.23, 0.12],  // its roots take hold again
      [at(11, 1.2),   -4.2,  3.4,      -15.0,  2.5,   0.12, 0.1,  1,   3.11, 1,   0.2,  0.0,  0.15, 0.0,  0.15, 0,  0,   0.6,  1.0,  0.23, 0.12],  // "a flower climbs up to your lips"
      [at(12, 0.55),  -1.5,  2.0,      -17.0,  3.05,  0.1,  1.0,  1,   3.15, 1,   0.2,  0.0,  0.1,  0.0,  0.1, 0,   0,   0.8,  1.0,  0.16, 0.2],   // "all that fire is repeated"
      [at(12, 1.3),   -1.2,  2.2,      -16.0,  3.06,  0.1,  1.0,  1,   3.18, 1,   0.2,  0.0,  0.05, 0.0,  0.1, 0,   0,   0.88, 1.0,  0.16, 0.2],
      [at(13, 0.6),    0.3,  4.2,      -5.5,   1.4,   0.03, 1.0,  1,   3.22, 0.5, 0.2,  0.0,  0.0,  0.0,  0.1, 0,   0,   0.95, 1.0,  0, 0],   // "my love feeds on your love"
      [at(13, 1.4),    0.05, F + 1.4,  1.0,    0.0,   0.22, 1.0,  1,   3.28, 0,   0.2,  0.0,  0.0,  0.0,  0.1, 0,   0,   1.0,  1.0,  0, 0],   // "without leaving mine"
      [T.total,        0.05, F + 1.4,  1.3,    0.0,   0.25, 1.0,  1,   3.4,  0,   0.2,  0.0,  0.0,  0.0,  0.1, 0,   0,   1.0,  1.0,  0, 0]
    ];
  },
  sound: {
    src: '/audio/ocean.mp3',
    label: 'Play the sea and the fire',
    // Muffled in the room, open over the water, loudest in the gust.
    volume: function (row) { return lerp(0.035, 0.1 + 0.28 * row[10], smooth(0.5, -1.5, row[2])) * (1 - 0.3 * row[16]); },
    cues: [{ stanza: 2, at: 0.3, play: crackle(7, 0.3) }, { stanza: 6, at: 0.35, play: gustSound },
           { stanza: 12, at: 0.45, play: crackle(9, 0.45) }]
  }
});
