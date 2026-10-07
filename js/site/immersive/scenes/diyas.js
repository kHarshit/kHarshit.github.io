/*
 * Scene for "Sanskrit Shlokas": a river at the ghats before dawn.
 *
 * Standing on the stone steps of a ghat, temples and akash-deep lanterns
 * along both banks, the river running away to the east.
 *
 * I    विद्यां ददाति विनयं: on the step in front of you one diya is lit.
 * II   "With knowledge comes humility ...": the flame passes on, lamp to lamp
 *      along the steps, each lit from one already burning.
 * ·    अयं निजः परो वेति: floating diyas are set on the water from both
 *      banks and drift downstream in two separate lines, mine and other.
 * III  "... see the whole world as their family": the two lines swing out
 *      into midstream and join, one river of light.
 * ·    उत्तिष्ठत जाग्रत: the temple bell; the sky pales over the water.
 * IV   "Arise, awake": the sun comes up at the end of the river and the sky
 *      turns saffron and gold.
 *
 * The engine ends a panel at each stanza's last line, so with maxLines 2
 * the panels are: shloka I, its translation, shloka II, its translation,
 * the third shloka, its translation. Lamps are shader point fields (each
 * lights at its own moment; the floating ones are moved on the GPU).
 * Columns: [unit, climb, -, -, wind, yaw, pitch, first, chain, float, join, dawn, sun]
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, softSprite, starField,
         oceanMaterial, oceanMesh, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout (metres): the river runs towards -z; you stand on the west bank ─
var RUN = 0.9, RISE = 0.42, STEPS = 16;
var CALM = [[1, 0.3, 0.35, 0.025, 0.7], [-0.4, 1, 0.55, 0.018, 0.9], [0.7, -0.8, 0.9, 0.01, 1.3]];
var SUN_AZ = 0.2;                       // radians right of straight down the river
var FIRST = new THREE.Vector3(-6 * RUN - 0.3, -0.6 + 7 * RISE, -1.6);
function nearB(z) { return z > -100 ? 0 : (z + 100) * 0.12; }
function farB(z) { return z > -100 ? 96 : 96 - (z + 100) * 0.2; }
function stepTop(k) { return -0.6 + (k + 1) * RISE; }
function stepY(dx) { return stepTop(Math.min(Math.floor(dx / RUN), STEPS - 1)); }

// ── Sounds: a brass temple bell and a conch, synthesised ──────────────────
function bell(freq, gain) {
  return function (ac, out) {
    var t = ac.currentTime;
    // Hum, prime (doubled and detuned so it beats), tierce, quint, nominal, upper partials.
    [[0.5, 0.35, 7], [1, 0.5, 5.5], [1.004, 0.4, 5.5], [1.19, 0.28, 4], [1.5, 0.22, 3.2],
     [2, 0.4, 3], [2.51, 0.16, 2.2], [3.02, 0.12, 1.7], [4.11, 0.08, 1.1], [5.4, 0.05, 0.7]].forEach(function (p) {
      var o = ac.createOscillator(), g = ac.createGain();
      o.type = 'sine';
      o.frequency.value = freq * p[0];
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(gain * p[1], t + 0.006);
      g.gain.exponentialRampToValueAtTime(0.0001, t + p[2]);
      o.connect(g);
      g.connect(out);
      o.start(t);
      o.stop(t + p[2] + 0.05);
    });
  };
}
function bells(freq, gain, n, gap) {
  var one = bell(freq, gain);
  return function (ac, out) {
    for (var i = 0; i < n; i++) setTimeout(function () { one(ac, out); }, i * gap * 1000);
  };
}
// A shankh: a buzzing reed tone through vowel-like formants, swelling and falling.
function conch(ac, out) {
  var t = ac.currentTime, dur = 3.6;
  var o = ac.createOscillator(), lfo = ac.createOscillator(), lfoGain = ac.createGain(), amp = ac.createGain();
  o.type = 'sawtooth';
  o.frequency.setValueAtTime(228, t);
  o.frequency.linearRampToValueAtTime(240, t + 0.6);
  o.frequency.setValueAtTime(240, t + dur - 0.8);
  o.frequency.exponentialRampToValueAtTime(196, t + dur);
  lfo.frequency.value = 5.2;
  lfoGain.gain.value = 2.2;
  lfo.connect(lfoGain);
  lfoGain.connect(o.frequency);
  amp.gain.setValueAtTime(0.0001, t);
  amp.gain.exponentialRampToValueAtTime(0.22, t + 0.7);
  amp.gain.setValueAtTime(0.22, t + dur - 0.9);
  amp.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  [[720, 7, 1], [1150, 9, 0.6], [2500, 12, 0.25]].forEach(function (fm) {
    var bp = ac.createBiquadFilter(), g = ac.createGain();
    bp.type = 'bandpass';
    bp.frequency.value = fm[0];
    bp.Q.value = fm[1];
    g.gain.value = fm[2];
    o.connect(bp);
    bp.connect(g);
    g.connect(amp);
  });
  amp.connect(out);
  o.start(t);
  lfo.start(t);
  o.stop(t + dur + 0.1);
  lfo.stop(t + dur + 0.1);
}

// ── Geometry helpers ─────────────────────────────────────────────────────
function box(geos, x0, x1, y0, y1, z0, z1, color) {
  geos.push(tinted(new THREE.BoxGeometry(Math.abs(x1 - x0), y1 - y0, Math.abs(z1 - z0))
    .translate((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), color));
}

// A curvilinear Nagara spire, faceted so it reads as ribbed.
function spire(geos, x, y, z, rad, h, color) {
  var pts = [];
  for (var i = 0; i <= 10; i++) {
    var t = i / 10 * 0.92;
    pts.push(new THREE.Vector2(rad * Math.pow(Math.cos(t * Math.PI / 2), 0.75), t * h));
  }
  geos.push(tinted(new THREE.LatheGeometry(pts, 8).translate(x, y, z), color));
  var top = y + h * 0.92;
  geos.push(tinted(new THREE.CylinderGeometry(rad * 0.34, rad * 0.34, h * 0.05, 12).translate(x, top + h * 0.02, z), color));
  geos.push(tinted(new THREE.ConeGeometry(rad * 0.12, h * 0.14, 6).translate(x, top + h * 0.11, z), color));
}

// A temple: plinth, sanctum with its tall spire and four small ones, porch.
function temple(geos, x, y, z, s, dir, color) {
  box(geos, x - 4 * s, x + 4 * s, y, y + 1.6 * s, z - 4 * s, z + 4 * s, color);
  box(geos, x - 2.6 * s, x + 2.6 * s, y + 1.6 * s, y + 5.6 * s, z - 2.6 * s, z + 2.6 * s, color);
  spire(geos, x, y + 5.6 * s, z, 2.6 * s, 9 * s, color);
  [[-1, -1], [1, -1], [-1, 1], [1, 1]].forEach(function (c) {
    spire(geos, x + c[0] * 2.1 * s, y + 5.6 * s, z + c[1] * 2.1 * s, 0.9 * s, 3.6 * s, color);
  });
  var px = x - dir * 4.6 * s;
  box(geos, px - 1.8 * s, px + 1.8 * s, y + 1.6 * s, y + 4.2 * s, z - 2 * s, z + 2 * s, color);
  geos.push(tinted(new THREE.ConeGeometry(2.8 * s, 2.4 * s, 4).rotateY(Math.PI / 4).translate(px, y + 5.4 * s, z), color));
}

// A ghat: flights of steps down to the water in segments, a terrace, and a
// row of houses and temples behind. dir is the inland direction (-1 or +1).
function bank(geos, r, o) {
  var z = o.z0, terraceAt = [];
  while (z > o.z1) {
    var len = o.first && z === o.z0 ? o.firstLen : 22 + r() * 38;
    var bx = o.bankX(z - len), steps = o.first && z === o.z0 ? STEPS : 11 + Math.floor(r() * 7);
    for (var k = 0; k < steps; k++) {
      var tone = 0.42 + r() * 0.06 + k * 0.004, wet = smooth(1.2, -0.4, stepTop(k));
      var col = new THREE.Color().setRGB(tone * 1.12, tone * 0.93, tone * 0.72).lerp(new THREE.Color('#2a2622'), wet * 0.7);
      box(geos, bx + o.dir * k * RUN, bx + o.dir * (k + 1) * RUN, -4, stepTop(k), z - len, z, col);
    }
    var top = stepTop(steps - 1), xin = bx + o.dir * steps * RUN;
    box(geos, xin, xin + o.dir * 90, -4, top, z - len, z, '#4c4034');
    terraceAt.push({ x: xin, y: top, z0: z - len, z1: z });
    // Houses and temples along the terrace, a taller row behind.
    for (var row = 0; row < 2; row++) {
      var bz = z;
      while (bz > z - len + 1) {
        var w = Math.min(5 + r() * 10, bz - (z - len)), back = 3 + row * 16 + r() * 4, d = 9 + r() * 10;
        var h = (7 + r() * 16) * o.scale + row * 8, x0 = xin + o.dir * back, x1 = x0 + o.dir * d;
        var tone2 = 0.3 + r() * 0.12, bcol = new THREE.Color().setRGB(tone2 * 1.1, tone2 * 0.92, tone2 * 0.78);
        if (row === 0 && r() < o.temples) {
          temple(geos, (x0 + x1) / 2, top, bz - w / 2, Math.min(w, d) / 8.5 * o.scale * 1.2, o.dir, bcol);
        } else {
          box(geos, x0, x1, top, top + h, bz - w, bz, bcol);
          var dark = bcol.clone().multiplyScalar(0.45), trim = bcol.clone().multiplyScalar(1.25);
          // Cornice, a plinth band, sometimes a balcony or a domed roof kiosk.
          box(geos, x0 - o.dir * 0.35, x0 + o.dir * 0.4, top + h - 0.5, top + h, bz - w, bz, trim);
          box(geos, x0 - o.dir * 0.2, x0 + o.dir * 0.4, top, top + 1.2, bz - w, bz, dark);
          if (h > 9 && r() < 0.5) box(geos, x0 - o.dir * 1.1, x0, top + 5.2, top + 5.45, bz - w + 0.6, bz - 0.6, trim);
          if (r() < 0.3) {
            var kx = x0 + o.dir * 2.5, kz = bz - w / 2;
            box(geos, kx - 1.2, kx + 1.2, top + h, top + h + 1.8, kz - 1.2, kz + 1.2, bcol);
            geos.push(tinted(new THREE.SphereGeometry(1.3, 8, 4, 0, Math.PI * 2, 0, Math.PI / 2).translate(kx, top + h + 1.8, kz), bcol));
          } else if (r() < 0.25) spire(geos, (x0 + x1) / 2, top + h, bz - w / 2, Math.min(w, d) * 0.22, Math.min(w, d) * 0.7, bcol);
          // Windows on the river face, a few of them lit.
          if (row === 0) {
            for (var wy = top + 2.4; wy < top + h - 1.5; wy += 3.1) {
              for (var wz = bz - 1.4; wz > bz - w + 0.8; wz -= 2.4) {
                if (r() < 0.3) continue;
                if (r() < 0.1) o.windows.push(x0 - o.dir * 0.12, wy, wz);
                else box(geos, x0 - o.dir * 0.08, x0 + o.dir * 0.05, wy - 0.7, wy + 0.7, wz - 0.4, wz + 0.4, dark);
              }
            }
          }
        }
        bz -= w;
      }
    }
    z -= len;
  }
  return terraceAt;
}

// A clay diya: a shallow bowl with a pinched lip, rim lit warm from within.
function diyaGeometry() {
  var pts = [[0, 0], [0.05, 0.004], [0.075, 0.025], [0.085, 0.05], [0.07, 0.05], [0.05, 0.03], [0, 0.03]]
    .map(function (p) { return new THREE.Vector2(p[0], p[1]); });
  var geo = new THREE.LatheGeometry(pts, 12).toNonIndexed(), pos = geo.attributes.position, cols = [];
  for (var i = 0; i < pos.count; i++) {
    var y = pos.getY(i), rr = Math.hypot(pos.getX(i), pos.getZ(i)), inner = rr < 0.072 && y > 0.026;
    var c = inner ? new THREE.Color('#e08a3c') : new THREE.Color('#6a3a22').lerp(new THREE.Color('#a85a2c'), y / 0.05);
    cols.push(c.r, c.g, c.b);
  }
  geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
  return geo;
}

// A wooden boat: the lower half of a stretched sphere, open at the top.
function boatGeometry() {
  var g = new THREE.SphereGeometry(1, 14, 6, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2).scale(3.2, 0.75, 0.95);
  var deck = new THREE.CircleGeometry(1, 16).rotateX(-Math.PI / 2).scale(3.1, 1, 0.9).translate(0, -0.12, 0);
  return merge([tinted(g, '#2a1c14'), tinted(deck, '#1c120c'), tinted(new THREE.BoxGeometry(0.2, 0.06, 1.7).translate(0.6, -0.06, 0), '#3a2a1c')]);
}

// ── Lamp shaders ─────────────────────────────────────────────────────────
// Every lamp is a point: a flame, a halo and a streak of reflection on the
// water. lampPos() gives each one's position and how lit it is.
var LAMP_HEAD =
  'uniform float uTime; uniform float uScale; uniform float uFloat; uniform float uJoin; uniform float uFlow;\n' +
  'uniform float uChain; uniform float uFirst; uniform float uFogD; uniform float uDay;\n' +
  'attribute vec4 aLamp; attribute vec3 aFrom;\n' +
  'varying float vA; varying float vPx; varying float vSeed; varying float vHeat;\n';

// Floating diyas. aLamp = (bank 0 near | 1 far, phase, lane -1..1, seed).
// Each is launched near the camera, pushed out from its bank, and carried
// downstream; uJoin swings both lines into one stream in midstream, and
// uFloat is how far down the river the procession has reached.
var RIVER_POS =
  'float nearB(float z){ return z > -100.0 ? 0.0 : (z + 100.0) * 0.12; }\n' +
  'float farB(float z){ return z > -100.0 ? 96.0 : 96.0 - (z + 100.0) * 0.2; }\n' +
  'vec4 lampPos(){\n' +
  ' float s = fract(aLamp.y + uTime * uFlow);\n' +
  ' float z = -4.0 - aLamp.w * 46.0 - s * 640.0;\n' +
  ' float nb = nearB(z), fb = farB(z), lane = aLamp.z * 0.5 + 0.5;\n' +
  ' float push = 0.9 + smoothstep(0.0, 0.03, s) * (1.2 + lane * 6.0);\n' +
  ' float band = mix(nb + push, fb - push, aLamp.x);\n' +
  ' float mid = (nb + fb) * 0.5 + sin(z * 0.011 + 1.3) * 5.0 + aLamp.z * (1.6 + s * 4.5);\n' +
  ' float x = mix(band, mid, uJoin * smoothstep(0.015, 0.16, s)) + sin(uTime * 0.23 + aLamp.w * 40.0) * 0.35;\n' +
  ' float vis = smoothstep(0.0, 0.004, s) * (1.0 - smoothstep(uFloat - 0.05, uFloat, s)) * step(0.002, uFloat);\n' +
  ' vSeed = aLamp.w; vHeat = 0.0;\n' +
  ' return vec4(x, 0.07 + sin(uTime * 1.4 + aLamp.w * 30.0) * 0.02, z, vis);\n' +
  '}\n';

// Lamps on the steps. aLamp.x = when it lights (0..1 of the chain; < 0 for
// the first diya, which uFirst lights), aLamp.w = seed, aFrom = the lamp it
// is lit from.
var STEP_POS =
  'vec4 lampPos(){\n' +
  ' float lit = aLamp.x < 0.0 ? uFirst : smoothstep(aLamp.x, aLamp.x + 0.012, uChain);\n' +
  ' vSeed = aLamp.w;\n' +
  ' vHeat = aLamp.x < 0.0 ? uFirst * (1.0 - uFirst) * 4.0 : exp(-pow((uChain - aLamp.x - 0.01) * 50.0, 2.0));\n' +
  ' return vec4(position + vec3(0.0, 0.05, 0.0), lit);\n' +
  '}\n';

// A spark that flies from aFrom to the lamp just before it lights.
var SPARK_POS =
  'vec4 lampPos(){\n' +
  ' float k = clamp((uChain - aLamp.x + 0.03) / 0.03, 0.0, 1.0);\n' +
  ' vSeed = aLamp.w; vHeat = 1.0;\n' +
  ' vec3 p = mix(aFrom, position, k) + vec3(0.0, 0.12 + sin(k * 3.14159) * (0.25 + 0.15 * distance(aFrom, position)), 0.0);\n' +
  ' return vec4(p, step(0.0, aLamp.x) * step(0.001, k) * step(k, 0.999));\n' +
  '}\n';

var FLAME_V =
  'void main(){ vec4 L = lampPos();\n' +
  ' vec4 mv = modelViewMatrix * vec4(L.xyz + vec3(0.0, 0.07, 0.0), 1.0); gl_Position = projectionMatrix * mv;\n' +
  ' float flick = 0.85 + 0.15 * sin(uTime * 13.0 + vSeed * 60.0) * sin(uTime * 7.3 + vSeed * 21.0);\n' +
  ' vA = L.w * flick * exp(-pow(uFogD * -mv.z, 2.0)) * (1.0 + vHeat * 0.5);\n' +
  ' vPx = SIZE * uScale / -mv.z * (1.0 + step(aLamp.x, -0.5) * 0.7); gl_PointSize = max(vPx * (1.0 + vHeat * 0.3), MINPX); }';

var REFLECT_V =
  'void main(){ vec4 L = lampPos();\n' +
  ' vec3 m = vec3(L.x, -L.y - 0.1, L.z), c = cameraPosition;\n' +
  ' vec3 q = c + (m - c) * (c.y / (c.y - m.y)); q.y = 0.03;\n' +
  ' vec4 mv = modelViewMatrix * vec4(q, 1.0); gl_Position = projectionMatrix * mv;\n' +
  ' float d = distance(m, c);\n' +
  ' vA = L.w * exp(-pow(uFogD * d, 2.0)) * (1.0 - uDay * 0.6);\n' +
  ' vPx = SIZE * uScale / d; gl_PointSize = max(vPx, MINPX); }';

var FLAME_F =
  'varying float vA; varying float vPx; varying float vSeed; varying float vHeat;\n' +
  'void main(){ vec2 p = gl_PointCoord - 0.5; p.y = -p.y;\n' +
  ' float sx = mix(0.15, 0.035, clamp(p.y + 0.35, 0.0, 1.0));\n' +
  ' float f = exp(-(p.x * p.x) / (sx * sx) - pow((p.y + 0.12) / 0.26, 2.0));\n' +
  ' f = mix(exp(-dot(p, p) * 18.0), f, smoothstep(3.0, 9.0, vPx));\n' +
  ' vec3 col = mix(vec3(1.0, 0.42, 0.08), vec3(1.0, 0.93, 0.72), smoothstep(0.35, 0.9, f));\n' +
  ' float a = f * vA; if (a < 0.004) discard;\n' +
  ' gl_FragColor = vec4(col, min(a, 1.0));\n #include <colorspace_fragment>\n }';

var GLOW_F =
  'varying float vA; varying float vPx; varying float vSeed; varying float vHeat;\n' +
  'void main(){ vec2 p = gl_PointCoord - 0.5; float d2 = dot(p, p);\n' +
  ' float a = (exp(-d2 * 14.0) * 0.4 + exp(-d2 * 70.0) * 0.35) * vA; if (a < 0.003) discard;\n' +
  ' gl_FragColor = vec4(1.0, 0.6, 0.25, min(a, 1.0));\n #include <colorspace_fragment>\n }';

var REFLECT_F =
  'uniform float uTime; varying float vA; varying float vPx; varying float vSeed; varying float vHeat;\n' +
  'void main(){ vec2 p = gl_PointCoord - 0.5;\n' +
  ' float a = exp(-p.x * p.x * 90.0 - (p.y - 0.12) * (p.y - 0.12) * 7.0) * (0.6 + 0.4 * sin(p.y * 40.0 + uTime * 3.0 + vSeed * 50.0));\n' +
  ' a *= vA * 0.7; if (a < 0.003) discard;\n' +
  ' gl_FragColor = vec4(1.0, 0.62, 0.28, min(a, 1.0));\n #include <colorspace_fragment>\n }';

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(108);
  var gl = makeRenderer(canvas, { clear: '#0a0d20' });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#2a2a48', 0.0026);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.05, 3000);
  camera.rotation.order = 'YXZ';

  // ── Sky: a dawn gradient with a warm band under the sun, stars, the sun ─
  var sky = new THREE.Group();
  world.add(sky);
  var skyU = {
    uTop: { value: new THREE.Color() }, uMid: { value: new THREE.Color() }, uHor: { value: new THREE.Color() },
    uGlow: { value: new THREE.Color() }, uSunCol: { value: new THREE.Color() }, uSun: { value: new THREE.Vector3(0, -0.1, -1) }
  };
  sky.add(new THREE.Mesh(new THREE.SphereGeometry(1500, 32, 16), new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false, uniforms: skyU,
    vertexShader: 'varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform vec3 uTop; uniform vec3 uMid; uniform vec3 uHor; uniform vec3 uGlow; uniform vec3 uSunCol; uniform vec3 uSun; varying vec3 vP;\n' +
      'void main(){ vec3 d = normalize(vP); float h = max(d.y, 0.0);\n' +
      ' vec3 c = mix(uHor, uMid, smoothstep(0.0, 0.16, h)); c = mix(c, uTop, smoothstep(0.12, 0.65, h));\n' +
      ' float az = 0.5 + 0.5 * dot(normalize(d.xz + 1e-5), normalize(uSun.xz + 1e-5));\n' +
      ' c += uGlow * pow(az, 5.0) * exp(-h * 6.0);\n' +
      ' float s = max(dot(d, normalize(uSun)), 0.0);\n' +
      ' c += uSunCol * (pow(s, 8.0) * 0.45 + pow(s, 60.0) * 1.1 + pow(s, 600.0) * 3.0);\n' +
      ' gl_FragColor = vec4(c, 1.0);\n #include <tonemapping_fragment>\n #include <colorspace_fragment>\n }'
  })));
  var stars = starField(r, small ? 1200 : 2400, 1300, 0.06, 1.3);
  sky.add(stars);
  // The sun's disc, cut off at the horizon so it can rise out of the river.
  var sunMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, fog: false, blending: THREE.AdditiveBlending, uniforms: { uCol: { value: new THREE.Color('#ffe6a0') }, uA: { value: 0 } },
    vertexShader: 'varying vec3 vW; varying vec2 vUv; void main(){ vUv = uv; vW = (modelMatrix * vec4(position, 1.0)).xyz; gl_Position = projectionMatrix * viewMatrix * vec4(vW, 1.0); }',
    fragmentShader: 'uniform vec3 uCol; uniform float uA; varying vec3 vW; varying vec2 vUv;\n' +
      'void main(){ if (vW.y - cameraPosition.y < 0.0) discard; float d = length(vUv - 0.5) * 2.0;\n' +
      ' float core = smoothstep(0.22, 0.2, d), halo = exp(-d * 6.0) * 0.55 * (1.0 - core);\n' +
      ' gl_FragColor = vec4(mix(vec3(1.0, 0.62, 0.2), uCol * 1.3, core), (core + halo) * uA);\n #include <colorspace_fragment>\n }'
  });
  var sunDisc = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), sunMat);
  sunDisc.scale.setScalar(160);
  sky.add(sunDisc);

  var hemi = new THREE.HemisphereLight('#5a6aa0', '#1e1812', 0.6);
  var sunLight = new THREE.DirectionalLight('#ffc27a', 0);
  world.add(hemi, sunLight, sunLight.target);

  // ── The ghats ──────────────────────────────────────────────────────────
  var geos = [], windows = [];
  var nearTerr = bank(geos, r, { dir: -1, bankX: nearB, z0: 30, z1: -720, first: true, firstLen: 130, scale: 1, temples: 0.3, windows: windows });
  bank(geos, r, { dir: 1, bankX: farB, z0: 20, z1: -720, scale: 0.85, temples: 0.25, windows: windows });
  // Towers (burj) jutting out between some of the near flights.
  [-128, -205, -300].forEach(function (z) {
    var bx = nearB(z);
    box(geos, bx - 8, bx + 2, -4, 9, z - 4, z + 4, '#55473a');
    spire(geos, bx - 3, 9, z, 2.6, 6, '#55473a');
  });
  var stone = new THREE.Mesh(merge(geos), new THREE.MeshLambertMaterial({ vertexColors: true }));
  world.add(stone);

  // Round straw umbrellas on the lower steps, as at Dashashwamedh.
  var umbGeo = merge([tinted(new THREE.ConeGeometry(1.7, 0.5, 12).translate(0, 2.5, 0), '#6a5236'),
                      tinted(new THREE.CylinderGeometry(0.04, 0.04, 2.5, 5).translate(0, 1.25, 0), '#3a2a1c'),
                      tinted(new THREE.BoxGeometry(2.2, 0.3, 2.2).translate(0, 0.15, 0), '#5a4a3a')]);
  var umbrellas = new THREE.InstancedMesh(umbGeo, new THREE.MeshLambertMaterial({ vertexColors: true }), 8), up = new THREE.Vector3(0, 1, 0);
  var umbZ = [6, -10, -26, -44, -60, -78, -94];
  umbZ.forEach(function (z, i) {
    var k = i % 2 ? 2 : 1, m = new THREE.Matrix4().compose(new THREE.Vector3(-k * RUN - 0.45, stepTop(k), z),
      new THREE.Quaternion().setFromAxisAngle(up, r() * 3), new THREE.Vector3(1, 1, 1));
    umbrellas.setMatrixAt(i, m);
  });
  umbrellas.count = umbZ.length;
  world.add(umbrellas);

  // Akash deep: lanterns raised on tall bamboo poles along the terrace.
  var poleGeos = [], lanternTex = softSprite('rgba(255,200,130,1)', 'rgba(255,150,70,0)'), poleLights = [];
  var t0 = nearTerr[0];
  [8, -8, -24, -40, -58, -76, -96, -116].forEach(function (z) {
    var x = t0.x - 1.5, h = 9 + r() * 2;
    poleGeos.push(tinted(new THREE.CylinderGeometry(0.05, 0.08, h, 5).translate(x, t0.y + h / 2, z), '#2a2016'));
    poleGeos.push(tinted(new THREE.BoxGeometry(0.35, 0.45, 0.35).translate(x + 0.3, t0.y + h - 0.6, z), '#e0904a'));
    poleLights.push(x + 0.3, t0.y + h - 0.6, z);
  });
  world.add(new THREE.Mesh(merge(poleGeos), new THREE.MeshLambertMaterial({ vertexColors: true, emissive: '#3a1e08' })));
  var smallLights = new THREE.BufferGeometry();
  smallLights.setAttribute('position', new THREE.Float32BufferAttribute(windows.concat(poleLights), 3));
  var winMat = new THREE.PointsMaterial({ color: '#ffb46a', size: 0.9, map: lanternTex, transparent: true, depthWrite: false,
                                          blending: THREE.AdditiveBlending });
  world.add(new THREE.Points(smallLights, winMat));

  // Boats moored at the steps, and two out on the water.
  var boatGeo = boatGeometry(), boatMat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
  var boats = [[3.2, -16, 0.1], [3.4, -23, -0.12], [4.6, -52, 0.05], [92.4, -30, 0.08], [38, -150, 0.5], [60, -230, -0.3]].map(function (b) {
    var m = new THREE.Mesh(boatGeo, boatMat);
    m.position.set(b[0], 0.4, b[1]);
    m.rotation.y = Math.PI / 2 + b[2];
    m.userData.ph = r() * 6;
    world.add(m);
    return m;
  });

  // ── The river ──────────────────────────────────────────────────────────
  var waterMat = oceanMaterial({ color: '#0b1424', specular: '#ffd09a', shininess: 120, waves: CALM });
  var water = oceanMesh(waterMat, 1600, small ? 150 : 220);
  world.add(water);
  var wu = waterMat.userData.uniforms;

  // Shared uniforms for every lamp material.
  var LU = { uTime: { value: 0 }, uScale: { value: 400 }, uFloat: { value: 0 }, uJoin: { value: 0 }, uFlow: { value: 0.0024 },
             uChain: { value: 0 }, uFirst: { value: 0 }, uFogD: { value: 0.0016 }, uDay: { value: 0 } };
  function lampMat(pos, vert, frag, size, minPx) {
    return new THREE.ShaderMaterial({
      uniforms: LU, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      vertexShader: '#define SIZE ' + size.toFixed(3) + '\n#define MINPX ' + minPx.toFixed(2) + '\n' + LAMP_HEAD + pos + vert,
      fragmentShader: frag
    });
  }
  function lampPoints(geo, pos, layers) {
    layers.forEach(function (l) {
      var p = new THREE.Points(geo, lampMat(pos, l[0], l[1], l[2], l[3]));
      p.frustumCulled = false;
      p.renderOrder = l[4] || 0;
      world.add(p);
    });
  }

  // Lamps on the steps, lit outwards from the first, each from its nearest
  // lit neighbour; the spreading quickens as more flames pass it on.
  var steps = [FIRST.clone()];
  [1, 3, 5, 8, 12].forEach(function (k) {
    for (var z = 9; z > -122; z -= 1.1 + r() * 1.2) {
      if (r() < 0.22) continue;
      var near = umbZ.some(function (u) { return Math.abs(u - z) < 1.6; }) && k < 3;
      if (near) continue;
      var p = new THREE.Vector3(-k * RUN - 0.22 - r() * 0.1, stepTop(k), z);
      if (p.distanceTo(FIRST) < 0.8) continue;
      steps.push(p);
    }
  });
  var NS = steps.length, lit = new Float32Array(NS), dist = steps.map(function (p) {
    // Along the steps counts for less than climbing them.
    return Math.hypot((p.x - FIRST.x) * 1.6, (p.y - FIRST.y) * 1.6, p.z - FIRST.z);
  });
  var dMax = Math.max.apply(null, dist);
  for (var i = 0; i < NS; i++) lit[i] = i === 0 ? -1 : 0.04 + 0.94 * Math.log(1 + dist[i] / 2.5) / Math.log(1 + dMax / 2.5) + r() * 0.015;
  var sPos = new Float32Array(NS * 3), sLamp = new Float32Array(NS * 4), sFrom = new Float32Array(NS * 3);
  for (i = 0; i < NS; i++) {
    var best = 0, bd = 1e9;
    for (var j = 0; j < NS; j++) {
      if (j === i || !(lit[j] < lit[i] - 0.004)) continue;
      var dd = steps[j].distanceToSquared(steps[i]);
      if (dd < bd) { bd = dd; best = j; }
    }
    sPos.set([steps[i].x, steps[i].y, steps[i].z], i * 3);
    sFrom.set([steps[best].x, steps[best].y + 0.05, steps[best].z], i * 3);
    sLamp.set([lit[i], 0, 0, r()], i * 4);
  }
  var stepGeo = new THREE.BufferGeometry();
  stepGeo.setAttribute('position', new THREE.BufferAttribute(sPos, 3));
  stepGeo.setAttribute('aLamp', new THREE.BufferAttribute(sLamp, 4));
  stepGeo.setAttribute('aFrom', new THREE.BufferAttribute(sFrom, 3));
  lampPoints(stepGeo, STEP_POS, [[REFLECT_V, REFLECT_F, 1.6, 2.0], [FLAME_V, GLOW_F, 1.3, 3.0, 1], [FLAME_V, FLAME_F, 0.2, 1.6, 2]]);
  lampPoints(stepGeo, SPARK_POS, [[FLAME_V, GLOW_F, 0.3, 3.0, 1], [FLAME_V, FLAME_F, 0.07, 2.0, 2]]);
  var bowlGeo = diyaGeometry(), bowlMat = new THREE.MeshLambertMaterial({ vertexColors: true });
  var stepBowls = new THREE.InstancedMesh(bowlGeo, bowlMat, NS), mtx = new THREE.Matrix4();
  for (i = 0; i < NS; i++) stepBowls.setMatrixAt(i, mtx.makeTranslation(steps[i].x, steps[i].y, steps[i].z));
  world.add(stepBowls);

  // Floating diyas, moved on the GPU.
  var NR = small ? 600 : 1300, rLamp = new Float32Array(NR * 4);
  for (i = 0; i < NR; i++) rLamp.set([i % 2, r(), r() * 2 - 1, r()], i * 4);
  var riverGeo = new THREE.BufferGeometry();
  riverGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NR * 3), 3));
  riverGeo.setAttribute('aLamp', new THREE.BufferAttribute(rLamp, 4));
  lampPoints(riverGeo, RIVER_POS, [[REFLECT_V, REFLECT_F, 1.7, 1.8], [FLAME_V, GLOW_F, 1.3, 2.6, 1], [FLAME_V, FLAME_F, 0.24, 1.6, 2]]);
  // Their bowls, for the near ones.
  var floatBowls = new THREE.InstancedMesh(bowlGeo, new THREE.MeshLambertMaterial({ vertexColors: true, emissive: '#2a1206' }), NR);
  floatBowls.geometry = bowlGeo.clone();
  floatBowls.geometry.setAttribute('aLamp', new THREE.InstancedBufferAttribute(rLamp, 4));
  floatBowls.frustumCulled = false;
  floatBowls.material.onBeforeCompile = function (sh) {
    Object.assign(sh.uniforms, LU);
    sh.vertexShader = LAMP_HEAD.replace('attribute vec4 aLamp; attribute vec3 aFrom;\n', 'attribute vec4 aLamp;\n') + RIVER_POS +
      sh.vertexShader.replace('#include <begin_vertex>', 'vec4 L = lampPos(); vec3 transformed = position * step(0.05, L.w) + L.xyz - vec3(0.0, 0.035, 0.0);');
  };
  world.add(floatBowls);

  // Lamplight on the steps: the first diya, then the spreading flames.
  var firstLight = new THREE.PointLight('#ffa050', 0, 7, 1.6);
  firstLight.position.copy(FIRST).y += 0.7;
  var stepLights = [-4, -22, -48].map(function (z) {
    var l = new THREE.PointLight('#ff9a48', 0, 24, 1.5);
    l.position.set(-3, 2.6, z);
    l.userData.at = lit.reduce(function (m, v, k) { return Math.abs(steps[k].z - z) < 6 && v > 0 ? Math.min(m, v) : m; }, 1);
    world.add(l);
    return l;
  });
  world.add(firstLight);

  // Low mist on the water, warmed by the dawn.
  var mistTex = softSprite('rgba(255,255,255,0.6)', 'rgba(255,255,255,0)'), mist = [];
  for (i = 0; i < 16; i++) {
    var ms = new THREE.Sprite(new THREE.SpriteMaterial({ map: mistTex, transparent: true, depthWrite: false, opacity: 0.12 }));
    ms.position.set(10 + r() * 80, 1.5 + r() * 2, -30 - r() * 260);
    ms.scale.set(40 + r() * 40, 5 + r() * 3, 1);
    world.add(ms);
    mist.push(ms);
  }

  // ── Colour ramps through the dawn: night, first light, saffron, sunrise ─
  var RAMP = {
    top: ['#05081a', '#141c44', '#2a3a78', '#30508e'],
    mid: ['#0e1434', '#2c2c5c', '#a8504a', '#e8783a'],
    hor: ['#262446', '#7a4a5e', '#f0782e', '#ff9a2e'],
    glow: ['#1a1226', '#6a3040', '#ff5a14', '#ff8a10']
  };
  var cA = new THREE.Color(), cB = new THREE.Color(), tmp = new THREE.Color(), sunDir = new THREE.Vector3(), H = 800;
  function ramp(list, t, out) {
    var x = clamp(t, 0, 1) * (list.length - 1), i0 = Math.min(Math.floor(x), list.length - 2);
    return out.set(list[i0]).lerp(cB.set(list[i0 + 1]), x - i0);
  }

  function frame(f) {
    var row = f.row, time = f.time, climb = f.cam, first = row[6], chain = row[7], float = row[8], join = row[9];
    var dawn = row[10], rise = row[11], tone = dawn * 0.75 + rise * 0.25;

    // Up the steps a little as the river fills with light.
    var cx = lerp(-6.6, -12.4, climb);
    camera.position.set(cx, stepY(-cx) + 1.55, lerp(2, 6, climb));
    // On a portrait screen the verse sits mid-frame: keep the first diya
    // below it, and the sunrise above it, and turn less.
    var yaw = row[4], pitch = row[5];
    if (camera.aspect < 1) { yaw = lerp(yaw, 0.38, 0.4); pitch += lerp(0.14, -0.12, smooth(0.3, 0.9, row[10])); }
    camera.rotation.set(pitch - f.my * 0.05, -(yaw + f.mx * 0.12), 0);
    sky.position.copy(camera.position);
    water.userData.follow(camera.position);

    // Sky, sun and light.
    var el = lerp(-0.07, 0.09, rise);
    sunDir.set(Math.sin(SUN_AZ) * Math.cos(el), Math.sin(el), -Math.cos(SUN_AZ) * Math.cos(el));
    skyU.uSun.value.copy(sunDir);
    ramp(RAMP.top, tone, skyU.uTop.value);
    ramp(RAMP.mid, tone, skyU.uMid.value);
    var hor = ramp(RAMP.hor, tone, skyU.uHor.value);
    ramp(RAMP.glow, tone, skyU.uGlow.value);
    skyU.uSunCol.value.set('#ffa83a').multiplyScalar(smooth(0.3, 1, dawn) * (0.5 + rise * 1.2));
    sunDisc.position.copy(sunDir).multiplyScalar(1100);
    sunDisc.lookAt(camera.position.x, camera.position.y, camera.position.z);
    sunMat.uniforms.uA.value = smooth(0.0, 0.25, rise);
    stars.material.opacity = 0.8 * (1 - smooth(0.1, 0.7, dawn));
    world.fog.color.copy(hor).multiplyScalar(lerp(1, 0.85, tone));
    world.fog.density = lerp(0.0026, 0.0015, tone);
    gl.setClearColor(hor);

    sunLight.position.copy(camera.position).addScaledVector(sunDir, 300);
    sunLight.target.position.copy(camera.position);
    sunLight.intensity = smooth(0.05, 0.8, rise) * 2.0 + dawn * 0.3;
    hemi.intensity = 0.5 + dawn * 0.5;
    hemi.color.set('#5a6aa0').lerp(tmp.set('#ffc890'), smooth(0.4, 1, tone));
    hemi.groundColor.set('#1e1812').lerp(tmp.set('#5a3a28'), tone);
    wu.uTime.value = time;
    wu.uAmp.value = 1;
    wu.uSky.value.copy(hor).multiplyScalar(lerp(0.55, 0.35, tone));
    winMat.opacity = 1 - smooth(0.5, 1, dawn) * 0.7;
    mist.forEach(function (m, k) {
      m.material.color.copy(hor);
      m.material.opacity = 0.1 + 0.12 * dawn;
      m.position.x += f.dt * (0.4 + (k % 3) * 0.2);
      if (m.position.x > 110) m.position.x -= 120;
    });
    boats.forEach(function (b) { b.position.y = 0.4 + Math.sin(time * 0.9 + b.userData.ph) * 0.04; b.rotation.z = Math.sin(time * 0.7 + b.userData.ph) * 0.03; });

    // The lamps.
    LU.uTime.value = time;
    LU.uScale.value = H / (2 * Math.tan(camera.fov * Math.PI / 360));
    LU.uFirst.value = first;
    LU.uChain.value = chain;
    LU.uFloat.value = float * 1.06;
    LU.uJoin.value = join;
    LU.uFlow.value = env.reduceMotion ? 0.0009 : 0.0024;
    LU.uDay.value = smooth(0.2, 1, rise);
    var flick = 0.85 + 0.15 * Math.sin(time * 11) * Math.sin(time * 6.7);
    firstLight.intensity = first * 1.5 * flick;
    stepLights.forEach(function (l) { l.intensity = smooth(l.userData.at, l.userData.at + 0.25, chain) * 5 * flick * (1 - rise * 0.6); });

    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) {
      fitCamera(gl, camera, w, h, dpr, small);
      H = h * gl.getPixelRatio();
    },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('diyas', {
  renderer: renderer3d,
  maxLines: 2,
  scrim: 0.62,
  align: ['left', 'right', 'left', 'left', 'right', 'left'],
  // Panels: 0 shloka I, 1 its translation, 2 shloka II, 3 its translation,
  // 4 उत्तिष्ठत जाग्रत, 5 "Arise, awake".
  keys: function (T) {
    var S = T.start, E = T.end;
    //          unit         climb -  -  wind  yaw    pitch  first chain float join dawn sun
    return [
      [0,                   0.00, 0, 0, 0.1, 0.10, -0.10, 0, 0.00, 0.00, 0, 0.00, 0],
      [0.7,                 0.00, 0, 0, 0.1, 0.16, -0.18, 0, 0.00, 0.00, 0, 0.00, 0],
      [S(0) + 0.35,         0.00, 0, 0, 0.1, 0.12, -0.44, 0, 0.00, 0.00, 0, 0.00, 0],
      [S(0) + 0.95,         0.00, 0, 0, 0.1, 0.12, -0.44, 1, 0.00, 0.00, 0, 0.00, 0],   // the first diya is lit
      [E(0),                0.04, 0, 0, 0.1, 0.20, -0.36, 1, 0.02, 0.00, 0, 0.00, 0],
      [S(1) + 0.5,          0.10, 0, 0, 0.1, 0.32, -0.24, 1, 0.30, 0.00, 0, 0.00, 0],   // lamp lit from lamp
      [E(1) - 0.1,          0.18, 0, 0, 0.1, 0.30, -0.17, 1, 1.00, 0.00, 0, 0.02, 0],   // "... comes happiness and bliss"
      [S(2) + 0.35,         0.40, 0, 0, 0.1, 0.26, -0.16, 1, 1.00, 0.06, 0, 0.04, 0],   // diyas set out from both banks
      [E(2),                0.65, 0, 0, 0.1, 0.30, -0.16, 1, 1.00, 0.40, 0, 0.06, 0],   // two lines, mine and other
      [S(3) + 0.55,         0.80, 0, 0, 0.1, 0.28, -0.15, 1, 1.00, 0.70, 0, 0.09, 0],   // "Whereas ..."
      [S(3) + 1.25,         0.95, 0, 0, 0.1, 0.26, -0.14, 1, 1.00, 0.95, 1, 0.15, 0],   // "the whole world as their family"
      [E(3),                1.00, 0, 0, 0.1, 0.25, -0.13, 1, 1.00, 1.00, 1, 0.18, 0],
      [S(4) + 0.6,          1.00, 0, 0, 0.1, 0.21, -0.07, 1, 1.00, 1.00, 1, 0.55, 0],   // उत्तिष्ठत: first light
      [E(4),                1.00, 0, 0, 0.1, 0.20, -0.04, 1, 1.00, 1.00, 1, 0.80, 0.10],
      [S(5) + 0.6,          1.00, 0, 0, 0.1, 0.20, 0.00, 1, 1.00, 1.00, 1, 0.95, 0.50], // "Arise, awake": the sun
      [E(5),                1.00, 0, 0, 0.1, 0.10, 0.06, 1, 1.00, 1.00, 1, 1.00, 0.80],
      [T.total,             1.00, 0, 0, 0.1, -0.08, 0.10, 1, 1.00, 1.00, 1, 1.00, 1.00]
    ];
  },
  sound: {
    src: '/audio/birds.mp3',
    label: 'Play the ghats at dawn: birds, temple bells and a conch',
    volume: function (row) { return 0.02 + 0.3 * smooth(0.3, 1, row[10]); },
    cues: [
      { stanza: 0, at: 0.6, play: bell(880, 0.12) },
      { stanza: 2, at: 0.35, play: bell(660, 0.14) },
      { stanza: 3, at: 1.1, play: bell(740, 0.14) },
      { stanza: 4, at: 0.3, play: bells(392, 0.2, 3, 1.4) },
      { stanza: 5, at: 0.35, play: conch }
    ]
  }
});
