/*
 * Scene for "Silentium!" (Fyodor Tyutchev): keep still, and look inward.
 *
 * I   A perfectly still night lake that mirrors "stars in crystal skies".
 *     The heavens wheel slowly; the stars, and the evening star with them,
 *     sink and "set before the night is blurred" as mist gathers on the
 *     water. "Delight in them and speak no word."
 * II  "Drink at the source": a spring in a ring of mossy rocks at the
 *     water's edge, mirror-still and full of stars. On "a thought, once
 *     uttered, is untrue" a drop falls from the lip of rock and its rings
 *     break up the reflection; more drops stir up the silt ("dimmed is the
 *     fountainhead when stirred"), then it settles and stills.
 * III "Live in your inner self alone": you sink through the spring into a
 *     luminous inner world, glowing motes like a sky of their own and
 *     fronds of light, while "the outer light", the day, is a muffled
 *     brightness in the window of the surface far above. On "take in
 *     their song" the lights pulse together.
 *
 * The water is one plane at y = 0 that samples a mirror render of the
 * world (a reflection pass from below the surface) and bends it with
 * analytic ripple rings; seen from below it shows Snell's window.
 * Columns:
 *   [unit, x, y, z, yaw, pitch, wheel, mist, drop, stir, spring, glow, day, song, phone]
 * where "drop" lets a drop fall each time it passes a whole number, and
 * "phone" turns the view on portrait screens, whose text sits mid-screen, to
 * bring the subject held to one side back towards the centre.
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, softSprite, skyDome, terrain,
         scatter, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout (metres; the lake lies north, -z, of where you stand) ─────────
function shoreR(a) { return 200 + 26 * Math.sin(a * 3 + 0.4) + 14 * Math.sin(a * 7 + 1.3); }
var LAKE = { x: 0, z: 29 - shoreR(Math.PI / 2) };        // the near shore crosses z = 29
var BASIN = new THREE.Vector2(-5, 23.5), BR = 1.3;       // the spring: centre and inner radius
var DIR = new THREE.Vector2(-Math.sin(0.3), -Math.cos(0.3));  // from the viewpoint across the spring
var LIP = new THREE.Vector3(BASIN.x + DIR.x * 0.75, 0.8, BASIN.y + DIR.y * 0.75);   // where the drops fall from
var POLE = new THREE.Vector3(0, 0.7, -0.71).normalize();  // the celestial pole, north
var WHEEL_END = -0.62;

function ground(x, z) {
  var dx = x - LAKE.x, dz = z - LAKE.z, d = Math.hypot(dx, dz), a = Math.atan2(dz, dx), R = shoreR(a);
  var bed = -9 * smooth(R, R - 14, d) - 9 * smooth(R - 14, R - 110, d);
  var hills = smooth(R + 10, R + 150, d) * (40 + 24 * Math.sin(a * 5 + 2) + 10 * Math.sin(x * 0.03) * Math.cos(z * 0.025));
  return bed + 0.55 * smooth(R - 2, R + 5, d) + hills + 0.25 * Math.sin(x * 0.4) * Math.cos(z * 0.33);
}

// ── Shaders ──────────────────────────────────────────────────────────────
var GLSL_NOISE =
  'float hash2(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }\n' +
  'float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);\n' +
  '  return mix(mix(hash2(i), hash2(i + vec2(1.0, 0.0)), f.x), mix(hash2(i + vec2(0.0, 1.0)), hash2(i + vec2(1.0, 1.0)), f.x), f.y); }\n' +
  'float fbm2(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = p * 2.03 + vec2(1.7, 9.2); a *= 0.5; } return s; }\n';

var STAR_VS = 'attribute vec3 star; uniform float uTime; uniform float uScale;\n' +
  'varying float vA; varying float vWarm;\n' +
  'void main(){ gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);\n' +
  ' float tw = 0.72 + 0.28 * sin(uTime * (1.2 + star.y * 2.5) + star.y * 40.0);\n' +
  ' vA = tw; vWarm = star.z;\n' +
  ' gl_PointSize = star.x * uScale; }';
var STAR_FS = 'varying float vA; varying float vWarm;\n' +
  'void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;\n' +
  ' float a = smoothstep(0.5, 0.0, d) * vA;\n' +
  ' vec3 col = mix(vec3(0.8, 0.86, 1.0), vec3(1.0, 0.86, 0.68), vWarm);\n' +
  ' gl_FragColor = vec4(col * a, a);\n #include <colorspace_fragment>\n }';

var BAND_FS = 'uniform vec3 uN; uniform float uAmt; varying vec3 vP;\n' +
  'float hash(vec3 p){ p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }\n' +
  'float noise(vec3 x){ vec3 i = floor(x); vec3 f = fract(x); f = f * f * (3.0 - 2.0 * f);\n' +
  ' return mix(mix(mix(hash(i), hash(i + vec3(1,0,0)), f.x), mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),\n' +
  '            mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x), mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z); }\n' +
  'void main(){ vec3 d = normalize(vP); float b = dot(d, uN);\n' +
  ' float n = noise(d * 7.0) * 0.55 + noise(d * 15.0) * 0.3 + noise(d * 31.0) * 0.15;\n' +
  ' float band = exp(-b * b / 0.006) * (0.15 + 1.3 * n * n) - exp(-b * b / 0.0008) * 0.5 * n;\n' +
  ' gl_FragColor = vec4(vec3(0.42, 0.47, 0.66) * max(band, 0.0) * uAmt, 1.0); }';

// The water. From above: dark water under a mirror of the sky, bent by
// the rings of falling drops, with a faint inner glow and stirred silt in
// the spring. From below: Snell's window, the muffled outer light in a
// circle overhead, and the dim mirror of the depths around it.
function waterMaterial(tRefl) {
  var rip = [];
  for (var i = 0; i < 8; i++) rip.push(new THREE.Vector4(0, 0, -99, 0));
  var uniforms = {
    tRefl: { value: tRefl }, uTexMat: { value: new THREE.Matrix4() }, uHasRefl: { value: 0 },
    uTime: { value: 0 }, uRip: { value: rip }, uBasin: { value: BASIN },
    uStir: { value: 0 }, uSpring: { value: 0 }, uDay: { value: 0 },
    uDeep: { value: new THREE.Color('#020407') }, uSilt: { value: new THREE.Color('#4a453a') },
    uGlowC: { value: new THREE.Color('#1f6f78') }, uNight: { value: new THREE.Color('#14243a') },
    uWin: { value: new THREE.Color('#f2ecd6') }, uMurk: { value: new THREE.Color('#05161b') },
    uFogColor: { value: new THREE.Color() }, uFogDensity: { value: 0.002 }
  };
  return new THREE.ShaderMaterial({
    uniforms: uniforms, side: THREE.DoubleSide,
    vertexShader: 'uniform mat4 uTexMat; varying vec3 vW; varying vec4 vR;\n' +
      'void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; vR = uTexMat * w; gl_Position = projectionMatrix * viewMatrix * w; }',
    fragmentShader:
      'uniform sampler2D tRefl; uniform float uHasRefl; uniform float uTime; uniform vec4 uRip[8]; uniform vec2 uBasin;\n' +
      'uniform float uStir; uniform float uSpring; uniform float uDay;\n' +
      'uniform vec3 uDeep; uniform vec3 uSilt; uniform vec3 uGlowC; uniform vec3 uNight; uniform vec3 uWin; uniform vec3 uMurk;\n' +
      'uniform vec3 uFogColor; uniform float uFogDensity;\n' +
      'varying vec3 vW; varying vec4 vR;\n' + GLSL_NOISE +
      // Rings from each drop: a short train of waves behind a front that
      // spreads at ~0.28 m/s, fading with age and distance, held in by the rocks.
      'vec2 ripples(vec2 p){ vec2 g = vec2(0.0);\n' +
      ' for (int i = 0; i < 8; i++) { vec4 R = uRip[i]; float a = uTime - R.z;\n' +
      '  if (a <= 0.0 || a > 8.0 || R.w <= 0.0) continue;\n' +
      '  vec2 d = p - R.xy; float r = length(d) + 1e-4; float s = 0.28 * a + 0.015 - r;\n' +
      '  if (s < 0.0) continue;\n' +
      '  float env = exp(-s * 3.6) * exp(-a * 0.45) * R.w / (1.0 + r * 2.5) * smoothstep(0.0, 0.025, s);\n' +
      '  g += d / r * (-cos(s * 36.0) * 0.22 * env); }\n' +
      ' return g * (1.0 - smoothstep(' + (BR - 0.15).toFixed(2) + ', ' + (BR + 0.1).toFixed(2) + ', length(p - uBasin))); }\n' +
      'void main(){\n' +
      ' vec3 toCam = cameraPosition - vW; float dist = length(toCam); vec3 V = toCam / dist;\n' +
      ' float fine = 1.0 - smoothstep(15.0, 220.0, dist);\n' +
      ' vec2 g = ripples(vW.xz);\n' +
      // The lake breathes only very faintly.
      ' g += (vec2(vnoise(vW.xz * 0.3 + uTime * 0.04), vnoise(vW.xz * 0.3 - uTime * 0.035 + 3.0)) - 0.5) * 0.006 * fine;\n' +
      ' vec3 N = normalize(vec3(-g.x, 1.0, -g.y));\n' +
      ' float bd = length(vW.xz - uBasin), bm = 1.0 - smoothstep(' + (BR - 0.2).toFixed(2) + ', ' + BR.toFixed(2) + ', bd);\n' +
      ' vec3 col;\n' +
      ' if (gl_FrontFacing) {\n' +
      '   float cosv = max(dot(N, V), 0.0), fres = 0.02 + 0.98 * pow(1.0 - cosv, 5.0);\n' +
      '   vec4 rc = vR; rc.xy += g * 0.45 * rc.w;\n' +
      '   vec3 refl = uHasRefl > 0.5 ? texture2DProj(tRefl, rc).rgb : uNight;\n' +
      '   col = mix(uDeep, refl, 0.6 + 0.4 * fres);\n' +
      // The spring: a faint light from within, and silt clouding it when stirred.
      '   col += uGlowC * uSpring * bm * (0.22 + 0.1 * vnoise(vW.xz * 2.5 + uTime * 0.2)) * (1.0 - 0.5 * uStir);\n' +
      '   col += vec3(0.5, 0.62, 0.8) * min(length(g) * 1.8, 0.4) * bm;\n' +
      '   float silt = uStir * bm * smoothstep(0.35, 0.75, fbm2(vW.xz * 2.2 + vec2(uTime * 0.12, -uTime * 0.09)) + 0.15 * (1.0 - bd / ' + BR.toFixed(2) + '));\n' +
      '   col = mix(col, uSilt * (0.6 + 0.6 * length(refl)), clamp(silt * 1.1, 0.0, 0.9));\n' +
      ' } else {\n' +
      // From below: the window of the sky overhead, wobbling with the ripples.
      '   vec2 gu = g + (vec2(vnoise(vW.xz * 0.22 + uTime * 0.18), vnoise(vW.xz * 0.22 - uTime * 0.15 + 5.0)) - 0.5) * 0.1;\n' +
      '   float up = -V.y + dot(gu, vec2(0.7));\n' +
      '   float e = clamp((up - 0.66) / 0.34, 0.0, 1.0), win = smoothstep(0.61, 0.69, up);\n' +
      '   vec2 cp = vW.xz * 0.3 + gu * 6.0 + vec2(uTime * 0.11, uTime * 0.08);\n' +
      '   float caus = pow(1.0 - abs(vnoise(cp) * 2.0 - 1.0), 6.0) + pow(1.0 - abs(vnoise(cp * 1.7 + 4.0) * 2.0 - 1.0), 6.0);\n' +
      '   vec3 lightDay = uWin * (0.15 + 0.85 * pow(e, 1.6)) * (0.85 + 0.25 * caus);\n' +
      '   vec3 sky = mix(uNight * (0.45 + 0.55 * e), lightDay, uDay);\n' +
      '   float rim = smoothstep(0.6, 0.66, up) * (1.0 - smoothstep(0.66, 0.75, up));\n' +
      '   vec3 tir = uMurk * (0.7 + 0.5 * vnoise(vW.xz * 0.4 + uTime * 0.15)) + uGlowC * 0.05 * uSpring;\n' +
      '   col = mix(tir, sky, win) + rim * mix(uNight, uWin, uDay) * 0.35;\n' +
      ' }\n' +
      ' gl_FragColor = vec4(col, 1.0);\n' +
      ' #include <tonemapping_fragment>\n #include <colorspace_fragment>\n' +
      ' float fogF = 1.0 - exp(-uFogDensity * uFogDensity * dist * dist);\n' +
      ' gl_FragColor.rgb = mix(gl_FragColor.rgb, uFogColor, fogF);\n}'
  });
}

// Inner-world motes: drifting, twinkling, and pulsing together with "song".
var MOTE_VS = 'attribute vec4 mote; uniform float uTime; uniform float uScale; uniform float uGlow; uniform float uSong;\n' +
  'varying float vA; varying vec3 vC;\n' +
  'void main(){ vec3 p = position;\n' +
  ' p.x += sin(uTime * 0.13 + mote.y * 30.0) * 0.6; p.y += sin(uTime * 0.11 + mote.y * 17.0) * 0.4; p.z += cos(uTime * 0.09 + mote.y * 23.0) * 0.6;\n' +
  ' vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_Position = projectionMatrix * mv;\n' +
  ' float tw = 0.6 + 0.4 * sin(uTime * (0.8 + mote.y * 1.5) + mote.y * 50.0);\n' +
  ' float wave = 0.5 + 0.5 * sin(uTime * 1.6 - length(p.xz) * 0.25);\n' +
  ' vA = uGlow * mix(tw, 0.35 + 1.1 * wave, uSong);\n' +
  ' vC = mote.z < 0.33 ? vec3(0.45, 0.95, 1.0) : mote.z < 0.72 ? vec3(1.0, 0.82, 0.5) : vec3(0.85, 0.6, 1.0);\n' +
  ' gl_PointSize = clamp(mote.x * uScale / -mv.z, 1.0, 28.0); }';
var MOTE_FS = 'varying float vA; varying vec3 vC;\n' +
  'void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;\n' +
  ' float a = smoothstep(0.5, 0.0, d); a = a * a * vA;\n' +
  ' gl_FragColor = vec4(vC * a, 1.0);\n #include <colorspace_fragment>\n }';

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
    for (var i = 0; i < 28; i++) {
      var a = r() * 6.28, d = r() * 24, cx = 64 + Math.cos(a) * d, cy = 64 + Math.sin(a) * d * 0.7, rad = 16 + r() * 22;
      var g = x.createRadialGradient(cx, cy, 0, cx, cy, rad);
      g.addColorStop(0, 'rgba(255,255,255,0.2)');
      g.addColorStop(1, 'rgba(255,255,255,0)');
      x.fillStyle = g;
      x.fillRect(0, 0, 128, 128);
    }
  });
}

// A rough rock: a jittered icosphere, mossy where it faces up above the
// water, wet and dark below. `geo` is placed in world space.
function rockGeometry(r, rad, sx, sy, sz, x, y, z, rotY, lean) {
  var geo = new THREE.IcosahedronGeometry(rad, 3), p = geo.attributes.position, nrm = geo.attributes.normal, v = new THREE.Vector3();
  var k1 = r() * 6, k2 = r() * 6;
  for (var i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i).normalize();
    nrm.setXYZ(i, v.x, v.y, v.z);              // smooth: the normal of the sphere before it is roughened
    var n = rad * (1 + 0.14 * Math.sin(v.x * 4.1 + v.y * 2.3 + k1) * Math.cos(v.z * 3.7 - v.y * 2.1 + k2));
    p.setXYZ(i, v.x * n, v.y * n, v.z * n);
  }
  geo.scale(sx, sy, sz).rotateZ(lean || 0).rotateY(rotY).translate(x, y, z);
  var pos = geo.attributes.position, nor = geo.attributes.normal, cols = new Float32Array(pos.count * 3);
  var moss = new THREE.Color(), stone = new THREE.Color('#77766e'), wet = new THREE.Color('#2a2b2a'), c = new THREE.Color();
  for (i = 0; i < pos.count; i++) {
    var y0 = pos.getY(i), ny = nor.getY(i), m = smooth(0.2, 0.75, ny + 0.25 * Math.sin(pos.getX(i) * 9 + pos.getZ(i) * 7));
    moss.setHSL(0.24 + 0.04 * Math.sin(pos.getX(i) * 5), 0.42, 0.22 + 0.06 * Math.sin(pos.getZ(i) * 11));
    c.copy(stone).lerp(moss, m * smooth(-0.05, 0.15, y0)).lerp(wet, smooth(0.12, -0.25, y0));
    cols[i * 3] = c.r; cols[i * 3 + 1] = c.g; cols[i * 3 + 2] = c.b;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  geo.deleteAttribute('uv');
  return geo;
}

// A drop striking water: a high plink falling in pitch, then a soft bloop.
function plink(ac, out) {
  var t = ac.currentTime + 0.42, o = ac.createOscillator(), g = ac.createGain();
  o.type = 'sine';
  o.frequency.setValueAtTime(1500, t);
  o.frequency.exponentialRampToValueAtTime(560, t + 0.09);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.22, t + 0.006);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 0.3);
  o.connect(g); g.connect(out);
  o.start(t); o.stop(t + 0.35);
  var o2 = ac.createOscillator(), g2 = ac.createGain();
  o2.type = 'sine';
  o2.frequency.setValueAtTime(320, t + 0.03);
  o2.frequency.exponentialRampToValueAtTime(700, t + 0.16);
  g2.gain.setValueAtTime(0.0001, t + 0.03);
  g2.gain.exponentialRampToValueAtTime(0.08, t + 0.05);
  g2.gain.exponentialRampToValueAtTime(0.0001, t + 0.22);
  o2.connect(g2); g2.connect(out);
  o2.start(t + 0.03); o2.stop(t + 0.25);
}

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(1830);
  var gl = makeRenderer(canvas, { clear: '#070b16' });
  var world = new THREE.Scene();
  var AIR_FOG = new THREE.Color('#0a1020'), WATER_FOG = new THREE.Color('#041a20');
  world.fog = new THREE.FogExp2(AIR_FOG.clone(), 0.0022);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.05, 4000);
  var pxScale = 600, portrait = false;

  // ── Sky: a dome, stars and the Milky Way, which wheel round the pole ──
  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#02030a', mid: '#0a1026', horizon: '#26325a' }, 1500);
  sky.add(dome.mesh);
  var heavens = new THREE.Group();
  sky.add(heavens);
  var BAND_N = new THREE.Vector3(0.25, -0.41, -0.86).normalize();
  var bandMat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false, transparent: true, blending: THREE.AdditiveBlending,
    uniforms: { uN: { value: BAND_N }, uAmt: { value: 0.55 } },
    vertexShader: 'varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: BAND_FS
  });
  heavens.add(new THREE.Mesh(new THREE.SphereGeometry(1400, 48, 24), bandMat));
  var SN = small ? 3500 : 7000, MW = small ? 2500 : 5500, sPos = [], sAttr = [], v = new THREE.Vector3();
  var bandU = new THREE.Vector3().crossVectors(BAND_N, new THREE.Vector3(0, 1, 0)).normalize();
  var bandV = new THREE.Vector3().crossVectors(BAND_N, bandU).normalize();
  for (var i = 0; i < SN + MW; i++) {
    if (i < SN) {
      var y = -0.4 + r() * 1.4, s = Math.sqrt(1 - y * y), th = r() * Math.PI * 2;
      v.set(s * Math.cos(th), y, s * Math.sin(th));
    } else {
      var a = r() * Math.PI * 2, off = (r() + r() + r() - 1.5) * 0.12;
      v.copy(bandU).multiplyScalar(Math.cos(a)).addScaledVector(bandV, Math.sin(a)).addScaledVector(BAND_N, off).normalize();
    }
    sPos.push(v.x * 1300, v.y * 1300, v.z * 1300);
    sAttr.push(0.9 + Math.pow(r(), i < SN ? 4 : 7) * 3.4, r(), r() < 0.25 ? r() : 0);
  }
  var starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.Float32BufferAttribute(sPos, 3));
  starGeo.setAttribute('star', new THREE.Float32BufferAttribute(sAttr, 3));
  var starMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
    uniforms: { uTime: { value: 0 }, uScale: { value: 1 } }, vertexShader: STAR_VS, fragmentShader: STAR_FS
  });
  var stars = new THREE.Points(starGeo, starMat);
  stars.frustumCulled = false;
  heavens.add(stars);
  // The evening star, placed so that the wheel brings it down behind the
  // far hills, a little right of where you look.
  var venusDir = new THREE.Vector3(-Math.sin(0.7), 0.0, -Math.cos(0.7)).normalize().applyAxisAngle(POLE, -WHEEL_END);
  var venus = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,250,235,1)', 'rgba(220,230,255,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
  venus.position.copy(venusDir).multiplyScalar(1250);
  venus.scale.setScalar(26);
  heavens.add(venus);

  var hemi = new THREE.HemisphereLight('#4a5a88', '#0b0d10', 1.0);
  var starlight = new THREE.DirectionalLight('#9aa8d8', 0.55);
  starlight.position.set(-0.3, 1, -0.4);
  world.add(hemi, starlight);

  // ── Land: the far shore's wooded hills, the near bank, the lake bed ───
  var gc = new THREE.Color(), sand = new THREE.Color('#3a3a30'), grass = new THREE.Color('#1c2418'), bedC = new THREE.Color('#2d3a30');
  world.add(terrain(2400, small ? 200 : 300, 0, LAKE.z, ground, new THREE.MeshLambertMaterial({ vertexColors: true }), function (x, z, h) {
    var n = 0.5 + 0.5 * Math.sin(x * 0.21 + Math.sin(z * 0.13) * 2);
    if (h < -0.3) return gc.copy(bedC).multiplyScalar(0.7 + 0.5 * n);
    return gc.copy(sand).lerp(grass, smooth(0.3, 1.5, h)).multiplyScalar(0.8 + 0.4 * n);
  }));

  var conifer = merge([tinted(new THREE.CylinderGeometry(0.15, 0.25, 2, 5).translate(0, 1, 0), '#2a2018'),
                       tinted(new THREE.ConeGeometry(1.9, 5.5, 7).translate(0, 4.2, 0), '#ffffff'),
                       tinted(new THREE.ConeGeometry(1.4, 4.5, 7).translate(0, 7.2, 0), '#ffffff'),
                       tinted(new THREE.ConeGeometry(0.9, 3.6, 7).translate(0, 9.8, 0), '#ffffff')]);
  var trees = new THREE.InstancedMesh(conifer, new THREE.MeshLambertMaterial({ vertexColors: true }), small ? 1400 : 3000), up = new THREE.Vector3(0, 1, 0);
  scatter(trees, 30000, function (n, p, q, sc, c) {
    var ang = r() * Math.PI * 2, d = shoreR(ang) + 6 + Math.pow(r(), 1.6) * 260;
    var x = LAKE.x + Math.cos(ang) * d, z = LAKE.z + Math.sin(ang) * d;
    if (z > 15 && Math.abs(x) < 120) return false;         // keep the near bank open
    p.set(x, ground(x, z) - 0.3, z);
    q.setFromAxisAngle(up, r() * 6.28);
    sc.set(1, 0.8 + r() * 0.9, 1).multiplyScalar(0.9 + r() * 0.8);
    c.setHSL(0.36, 0.25, 0.05 + r() * 0.04);
  });
  world.add(trees);

  // Reeds along the near bank, some standing in the shallows.
  var reedGeo = merge([tinted(new THREE.ConeGeometry(0.012, 1.6, 3).translate(0, 0.8, 0), '#ffffff'),
                       tinted(new THREE.ConeGeometry(0.01, 1.3, 3).translate(0, 0.65, 0).rotateZ(0.12), '#ffffff')]);
  var reedClock = { value: 0 }, reedMat = new THREE.MeshLambertMaterial({ vertexColors: true });
  reedMat.onBeforeCompile = function (sh) {
    sh.uniforms.uClock = reedClock;
    sh.vertexShader = 'uniform float uClock;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n transformed.x += sin(uClock * 0.7 + instanceMatrix[3][0] * 0.8 + instanceMatrix[3][2]) * 0.015 * position.y * position.y;');
  };
  var reeds = new THREE.InstancedMesh(reedGeo, reedMat, small ? 500 : 1100);
  scatter(reeds, 30000, function (n, p, q, sc, c) {
    var x = -40 + r() * 70, z = 24 + r() * 10, h = ground(x, z);
    if (h < -0.5 || h > 0.45 || Math.hypot(x - BASIN.x, z - BASIN.y) < 3.2 || Math.hypot(x, z - 33) < 2.5) return false;
    if (x > -8 && x < 6 && z > 27 && r() < 0.8) return false;   // open ground at your feet
    p.set(x, Math.max(h, -0.5) - 0.05, z);
    q.setFromAxisAngle(up, r() * 6.28);
    sc.setScalar(0.6 + r() * 0.8);
    c.setHSL(0.17, 0.3, 0.07 + r() * 0.06);
  });
  world.add(reeds);

  // ── The spring: a ring of mossy rocks rising from deep water ──────────
  var rockParts = [];
  for (i = 0; i < 9; i++) {
    var ra = i / 9 * Math.PI * 2 + (r() - 0.5) * 0.3, rr = BR + 0.38 + r() * 0.12, top = 0.12 + r() * 0.35;
    if (Math.abs(((ra - Math.atan2(DIR.y, DIR.x)) + 9.42) % 6.283 - 3.14) < 0.45) continue;   // the lip rock goes here
    rockParts.push(rockGeometry(r, 0.5, 1, 4.2, 0.9, BASIN.x + Math.cos(ra) * rr, top - 2.0, BASIN.y + Math.sin(ra) * rr, r() * 6, (r() - 0.5) * 0.15));
  }
  // The lip: a taller rock leaning over the spring, where the drops gather.
  rockParts.push(rockGeometry(r, 0.6, 1.0, 2.6, 0.95, BASIN.x + DIR.x * 1.55, -0.5, BASIN.y + DIR.y * 1.55, Math.PI / 2 - Math.atan2(DIR.x, -DIR.y), 0.5));
  // A flat stone under your feet at the water's edge, and a few boulders.
  rockParts.push(rockGeometry(r, 1.0, 1.3, 0.35, 1.0, -4.0, 0.05, 26.5, 0.4, 0));
  [[-11, 28.5, 1.1], [6, 30.5, 0.9], [-1.5, 27.8, 0.5], [9.5, 27, 0.7]].forEach(function (b) {
    rockParts.push(rockGeometry(r, b[2], 1.2, 0.8, 1, b[0], ground(b[0], b[1]) + b[2] * 0.25, b[1], r() * 6, 0));
  });
  // The stone shaft of the spring goes on down, into the deep.
  for (i = 0; i < 7; i++) {
    var sa = i / 7 * Math.PI * 2;
    rockParts.push(rockGeometry(r, 0.6, 1, 3.5, 1, BASIN.x + Math.cos(sa) * (BR + 0.6), -6.5, BASIN.y + Math.sin(sa) * (BR + 0.6), r() * 6, 0));
  }
  // A faint light from deep in the spring, as if the source glowed.
  var springLight = new THREE.PointLight('#4fd8d0', 0, 5, 1.4);
  springLight.position.set(BASIN.x, 0.25, BASIN.y);
  world.add(springLight);
  var rocks = new THREE.Mesh(merge(rockParts), new THREE.MeshLambertMaterial({ vertexColors: true }));
  world.add(rocks);

  // The water.
  var mirrorRT = new THREE.WebGLRenderTarget(16, 16, { type: THREE.HalfFloatType });
  var waterMat = waterMaterial(mirrorRT.texture), WU = waterMat.uniforms;
  var water = new THREE.Mesh(new THREE.PlaneGeometry(3000, 3000).rotateX(-Math.PI / 2), waterMat);
  water.position.set(0, 0, LAKE.z);
  water.frustumCulled = false;
  world.add(water);

  // Drops and their splashes.
  var drops = [], dropMat = new THREE.MeshBasicMaterial({ color: '#e6f2ff' }), dropGeo = new THREE.SphereGeometry(0.018, 8, 6).scale(1, 1.5, 1);
  var glint = softSprite('rgba(220,235,255,1)', 'rgba(200,220,255,0)');
  for (i = 0; i < 6; i++) {
    var dm = new THREE.Mesh(dropGeo, dropMat), gs = new THREE.Sprite(new THREE.SpriteMaterial({ map: glint, blending: THREE.AdditiveBlending,
      depthWrite: false, transparent: true, opacity: 0.7 }));
    gs.scale.setScalar(0.12);
    dm.add(gs);
    dm.visible = false;
    world.add(dm);
    drops.push({ mesh: dm, t0: 0, live: false, x: 0, z: 0, amp: 1, delay: 0 });
  }
  var SPL = 48, splPos = new Float32Array(SPL * 3), splVel = new Float32Array(SPL * 3), splLife = new Float32Array(SPL);
  var splGeo = new THREE.BufferGeometry();
  splGeo.setAttribute('position', new THREE.BufferAttribute(splPos, 3));
  var splash = new THREE.Points(splGeo, new THREE.PointsMaterial({ color: '#dceaff', size: 0.012, transparent: true, depthWrite: false, opacity: 0.9 }));
  splash.frustumCulled = false;
  world.add(splash);
  for (i = 0; i < SPL; i++) splPos[i * 3 + 1] = -99;
  var ripIdx = 0, lastDrop = 0;

  // Mist that gathers on the lake as the night blurs.
  var puff = puffTexture(r), mists = [];
  for (i = 0; i < 26; i++) {
    var mx = -170 + r() * 230, mz = 18 - r() * 190;
    var ms = new THREE.Sprite(new THREE.SpriteMaterial({ map: puff, color: '#8a98c0', transparent: true, depthWrite: false, opacity: 0 }));
    ms.position.set(mx, 0.8 + r() * 2.5, mz);
    ms.scale.set(40 + r() * 50, 4 + r() * 5, 1);
    ms.userData = { x: mx, ph: r() * 6.28 };
    world.add(ms);
    mists.push(ms);
  }

  // ── The inner world, beneath the surface ──────────────────────────────
  var inner = new THREE.Group();
  world.add(inner);
  var MN = small ? 2200 : 4800, mPos = [], mAttr = [];
  for (i = 0; i < MN; i++) {
    var x = -40 + r() * 70, z = -60 + r() * 92, yb = Math.max(ground(x, z), -18);
    mPos.push(x, yb + 0.3 + r() * (-0.8 - yb), z);
    mAttr.push(0.05 + Math.pow(r(), 3) * 0.2, r(), r(), 0);
  }
  var moteGeo = new THREE.BufferGeometry();
  moteGeo.setAttribute('position', new THREE.Float32BufferAttribute(mPos, 3));
  moteGeo.setAttribute('mote', new THREE.Float32BufferAttribute(mAttr, 4));
  var moteMat = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
    uniforms: { uTime: { value: 0 }, uScale: { value: 600 }, uGlow: { value: 0 }, uSong: { value: 0 } },
    vertexShader: MOTE_VS, fragmentShader: MOTE_FS });
  var motes = new THREE.Points(moteGeo, moteMat);
  motes.frustumCulled = false;
  inner.add(motes);

  // Fronds of light growing from the bed, swaying in the slow water.
  var frondGeo = new THREE.PlaneGeometry(0.05, 1, 1, 10).translate(0, 0.5, 0), fp = frondGeo.attributes.position, fc = [];
  for (i = 0; i < fp.count; i++) { var hy = fp.getY(i), lum = 0.03 + 0.9 * Math.pow(hy, 3); fc.push(lum, lum, lum); }
  frondGeo.setAttribute('color', new THREE.Float32BufferAttribute(fc, 3));
  var frondClock = { value: 0 }, frondGlow = { value: 0 };
  var frondMat = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, side: THREE.DoubleSide,
                                               blending: THREE.AdditiveBlending, fog: false });
  frondMat.onBeforeCompile = function (sh) {
    sh.uniforms.uClock = frondClock;
    sh.uniforms.uGlowF = frondGlow;
    sh.vertexShader = 'uniform float uClock; varying float vNear;\n' + sh.vertexShader.replace('#include <project_vertex>',
      '#include <project_vertex>\n vNear = smoothstep(2.5, 9.0, -mvPosition.z);').replace('#include <begin_vertex>',
      '#include <begin_vertex>\n float fph = instanceMatrix[3][0] * 0.7 + instanceMatrix[3][2] * 0.5;\n' +
      ' transformed.x += sin(uClock * 0.6 + fph + position.y * 1.8) * 0.35 * position.y * position.y;\n' +
      ' transformed.z += cos(uClock * 0.5 + fph + position.y * 1.3) * 0.2 * position.y * position.y;');
    sh.fragmentShader = 'uniform float uGlowF; varying float vNear;\n' + sh.fragmentShader.replace('#include <color_fragment>',
      '#include <color_fragment>\n diffuseColor.rgb *= uGlowF * vNear * 0.7;');
  };
  var fronds = new THREE.InstancedMesh(frondGeo, frondMat, small ? 260 : 560);
  var frondCols = ['#8ae6dc', '#94bcff', '#c4a8ff', '#ffd8a0'].map(function (cc) { return new THREE.Color(cc); });
  scatter(fronds, 20000, function (n, p, q, sc, c) {
    var x = -35 + r() * 60, z = -55 + r() * 80, h = ground(x, z);
    if (h > -4 || Math.hypot(x - BASIN.x, z - BASIN.y) < 3) return false;
    p.set(x, h - 0.1, z);
    q.setFromAxisAngle(up, r() * 6.28);
    sc.set(1, 1.5 + r() * 4, 1);
    c.copy(frondCols[Math.floor(r() * frondCols.length)]).multiplyScalar(0.5 + r() * 0.5);
  });
  inner.add(fronds);

  // Shafts of the muffled outer light, falling from the surface.
  var shaftTex = canvasTexture(64, 256, function (x, w, h) {
    var g = x.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, 'rgba(255,255,255,0.9)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    x.fillStyle = g;
    x.fillRect(0, 0, w, h);
    var gx = x.createLinearGradient(0, 0, w, 0);
    gx.addColorStop(0, 'rgba(0,0,0,1)'); gx.addColorStop(0.5, 'rgba(0,0,0,0)'); gx.addColorStop(1, 'rgba(0,0,0,1)');
    x.globalCompositeOperation = 'destination-out';
    x.fillStyle = gx;
    x.fillRect(0, 0, w, h);
  });
  var shafts = [];
  for (i = 0; i < 12; i++) {
    var sh = new THREE.Mesh(new THREE.PlaneGeometry(2.5 + r() * 3, 22).translate(0, -11, 0), new THREE.MeshBasicMaterial({
      map: shaftTex, color: '#bfe6e0', transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide, fog: false, opacity: 0 }));
    sh.userData = { x: -25 + r() * 40, z: -40 + r() * 60, tilt: (r() - 0.5) * 0.35, ph: r() * 6 };
    inner.add(sh);
    shafts.push(sh);
  }

  // ── Mirror: the world rendered from below the surface ─────────────────
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
    WU.uTexMat.value.copy(bias).multiply(mirrorCam.projectionMatrix).multiply(mirrorCam.matrixWorldInverse);
    // Oblique near plane at the surface, so nothing beneath it is reflected.
    mPlane.setFromNormalAndCoplanarPoint(UP_N, ORIGIN).applyMatrix4(mirrorCam.matrixWorldInverse);
    clipV.set(mPlane.normal.x, mPlane.normal.y, mPlane.normal.z, mPlane.constant);
    var e = mirrorCam.projectionMatrix.elements;
    qv.set((Math.sign(clipV.x) + e[8]) / e[0], (Math.sign(clipV.y) + e[9]) / e[5], -1, (1 + e[10]) / e[14]);
    clipV.multiplyScalar(2 / clipV.dot(qv));
    e[2] = clipV.x; e[6] = clipV.y; e[10] = clipV.z + 1 - 0.003; e[14] = clipV.w;
    mirrorCam.projectionMatrixInverse.copy(mirrorCam.projectionMatrix).invert();

    water.visible = false;
    inner.visible = false;
    gl.setRenderTarget(mirrorRT);
    gl.render(world, mirrorCam);
    gl.setRenderTarget(null);
    water.visible = true;
  }

  // ── Frame ──────────────────────────────────────────────────────────────
  var tmpC = new THREE.Color();

  function spawnDrop(amp, delay) {
    for (var k = 0; k < drops.length; k++) {
      var d = drops[k];
      if (d.live) continue;
      d.live = true;
      d.delay = delay;
      d.t0 = -1;
      d.amp = amp;
      d.x = LIP.x + (amp < 1 ? (Math.random() - 0.5) * 0.25 : 0);
      d.z = LIP.z + (amp < 1 ? (Math.random() - 0.5) * 0.25 : 0);
      return;
    }
  }

  function frame(f) {
    var row = f.row, time = f.time, dt = f.dt;
    var yaw = row[3], pitch = row[4], wheel = row[5], mist = row[6], drop = row[7], stir = row[8], spring = row[9],
        glow = row[10], day = row[11], song = row[12];
    camera.position.set(row[0], row[1] + Math.sin(time * 0.5) * 0.01, row[2]);
    camera.rotation.set(pitch - f.my * 0.05, yaw + (portrait ? row[13] : 0) - f.mx * 0.1, 0, 'YXZ');
    var under = camera.position.y < 0;
    sky.position.copy(camera.position);
    heavens.quaternion.setFromAxisAngle(POLE, wheel);
    reedClock.value = time;
    starMat.uniforms.uTime.value = time;
    starMat.uniforms.uScale.value = pxScale / 760;
    // The evening star dims as it sets into the haze.
    venus.material.opacity = 0.35 + 0.65 * smooth(WHEEL_END, -0.1, wheel);

    // Drops: one falls each time the column passes a whole number.
    var dn = Math.floor(drop + 1e-4);
    if (dn > lastDrop) {
      for (var k = 0; k < Math.min(dn - lastDrop, 3); k++) spawnDrop(lastDrop + k === 0 ? 1 : 0.75, k * 0.35);
    }
    lastDrop = dn;
    drops.forEach(function (d) {
      if (!d.live) { d.mesh.visible = false; return; }
      d.delay -= dt;
      if (d.delay > 0) return;
      if (d.t0 < 0) d.t0 = time;
      var a = time - d.t0, yy = LIP.y - 4.9 * a * a;
      if (yy <= 0) {
        d.live = false;
        d.mesh.visible = false;
        WU.uRip.value[ripIdx++ % 8].set(d.x, d.z, time, d.amp);
        for (var sp = 0; sp < 12; sp++) {
          var si = Math.floor(Math.random() * SPL), j = si * 3, an = Math.random() * 6.28;
          splPos[j] = d.x; splPos[j + 1] = 0.01; splPos[j + 2] = d.z;
          splVel[j] = Math.cos(an) * 0.25; splVel[j + 1] = 0.6 + Math.random() * 0.6; splVel[j + 2] = Math.sin(an) * 0.25;
          splLife[si] = 0.4;
        }
        return;
      }
      d.mesh.visible = true;
      d.mesh.position.set(d.x, yy, d.z);
    });
    for (var s = 0; s < SPL; s++) {
      var q = s * 3;
      if (splLife[s] <= 0) { splPos[q + 1] = -99; continue; }
      splLife[s] -= dt;
      splVel[q + 1] -= 9.8 * dt;
      splPos[q] += splVel[q] * dt; splPos[q + 1] += splVel[q + 1] * dt; splPos[q + 2] += splVel[q + 2] * dt;
      if (splPos[q + 1] < 0) splLife[s] = 0;
    }
    splGeo.attributes.position.needsUpdate = true;

    mists.forEach(function (ms) {
      var u = ms.userData;
      ms.position.x = u.x + Math.sin(time * 0.03 + u.ph) * 8;
      ms.material.opacity = mist * 0.16 * (0.7 + 0.3 * Math.sin(time * 0.2 + u.ph));
      ms.visible = mist > 0.01 && !under;
    });

    // Above the water: the night air. Below: the luminous inner world.
    world.fog.color.copy(under ? WATER_FOG : AIR_FOG);
    if (under) world.fog.color.lerp(tmpC.set('#0b3a40'), glow * 0.35 + day * 0.15);
    world.fog.density = under ? 0.055 - glow * 0.012 : 0.0022 + mist * 0.0015;
    gl.setClearColor(world.fog.color);
    hemi.intensity = under ? 0.35 + day * 0.4 : 1.0;
    hemi.color.set(under ? '#3f8c90' : '#4a5a88');
    WU.uTime.value = time;
    WU.uStir.value = stir;
    WU.uSpring.value = spring;
    springLight.intensity = spring * (0.8 + 0.15 * Math.sin(time * 1.3)) * (1 - stir * 0.6);
    WU.uDay.value = day;
    WU.uFogColor.value.copy(world.fog.color).convertLinearToSRGB();
    WU.uFogDensity.value = world.fog.density;

    frondClock.value = time;
    frondGlow.value = glow * (1 + song * 0.35 * (0.5 + 0.5 * Math.sin(time * 1.6)));
    moteMat.uniforms.uTime.value = time;
    moteMat.uniforms.uGlow.value = glow;
    moteMat.uniforms.uSong.value = song;
    moteMat.uniforms.uScale.value = pxScale;
    shafts.forEach(function (sh, k) {
      var u = sh.userData;
      sh.position.set(u.x, -0.05, u.z);
      sh.rotation.set(0, Math.atan2(camera.position.x - u.x, camera.position.z - u.z), u.tilt);
      sh.material.opacity = (0.015 + day * 0.07) * (0.7 + 0.3 * Math.sin(time * 0.5 + u.ph)) * (under ? 1 : 0);
    });

    if (!under) {
      renderMirror();
      WU.uHasRefl.value = 1;
    } else {
      WU.uHasRefl.value = 0;
    }
    inner.visible = camera.position.y < 0.6;
    gl.toneMappingExposure = under ? 1.1 : 1.05;
    gl.render(world, camera);
  }

  function resize(w, h, dpr) {
    fitCamera(gl, camera, w, h, dpr, small);
    portrait = w < h;
    var ratio = gl.getPixelRatio(), k = small ? 0.6 : 0.8;
    mirrorRT.setSize(Math.max(16, Math.round(w * ratio * k)), Math.max(16, Math.round(h * ratio * k)));
    pxScale = h * ratio / (2 * Math.tan(camera.fov * Math.PI / 360));
  }

  return {
    resize: resize,
    frame: frame,
    destroy: function () { mirrorRT.dispose(); disposeAll(world, gl); }
  };
}

// The viewpoint over the spring, looking down into it.
var SPRING_EYE = [BASIN.x - DIR.x * 2.6, 2.3, BASIN.y - DIR.y * 2.6];

PI.register('still-water', {
  renderer: renderer3d,
  align: ['left', 'right', 'center'],
  keys: function (T) {
    function at(i, frac) { i = Math.min(i, T.count - 1); return lerp(T.start(i), T.end(i), frac); }
    var E = SPRING_EYE;
    //       unit          x      y      z      yaw   pitch  wheel mist drop stir spring glow day  song phone
    return [
      [0,             -14.0,  1.8,   31.4,  0.92, 0.03,  0.00, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15],
      [0.7,           -14.0,  1.8,   31.2,  0.92, 0.04,  0.00, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15],
      [at(0, 0.2),    -14.0,  1.8,   31.0,  0.90, 0.05, -0.05, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15],  // "Speak not, lie hidden"
      [at(0, 0.45),   -14.0,  1.8,   30.8,  0.88, 0.07, -0.20, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15],  // "stars in crystal skies"
      [at(0, 0.7),    -14.0,  1.8,   30.6,  0.86, 0.05, -0.52, 0.5, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15],  // "that set before the night is blurred"
      [at(0, 0.95),   -14.0,  1.8,   30.4,  0.84, 0.04, WHEEL_END, 0.8, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15], // "delight in them"
      [at(1, 0.15),   -8.0,   1.7,   28.6,  0.40, -0.30, WHEEL_END, 0.6, 0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
      [at(1, 0.3),     E[0],  E[1],  E[2],  0.12, -0.74, WHEEL_END, 0.4, 0, 0.0, 0.35, 0.0, 0.0, 0.0, 0.22], // "How can a heart expression find?"
      [at(1, 0.55),    E[0],  E[1],  E[2],  0.12, -0.74, WHEEL_END, 0.4, 0, 0.0, 0.35, 0.0, 0.0, 0.0, 0.22],
      [at(1, 0.6),     E[0],  E[1],  E[2],  0.12, -0.74, WHEEL_END, 0.4, 1, 0.0, 0.35, 0.0, 0.0, 0.0, 0.22], // "A thought, once uttered, is untrue."
      [at(1, 0.68),    E[0],  E[1],  E[2],  0.12, -0.74, WHEEL_END, 0.4, 1, 0.1, 0.35, 0.0, 0.0, 0.0, 0.22],
      [at(1, 0.78),    E[0],  E[1],  E[2],  0.12, -0.74, WHEEL_END, 0.4, 4, 1.0, 0.3, 0.0, 0.0, 0.0, 0.22],   // "Dimmed is the fountainhead when stirred"
      [at(1, 0.88),    E[0],  E[1],  E[2],  0.12, -0.74, WHEEL_END, 0.4, 4, 0.7, 0.45, 0.0, 0.0, 0.0, 0.22],
      [at(1, 1.0),     E[0],  E[1],  E[2],  0.10, -0.72, WHEEL_END, 0.4, 4, 0.0, 0.7, 0.0, 0.0, 0.0, 0.22],   // "drink at the source"
      [at(2, 0.08),   BASIN.x - DIR.x * 0.4, 1.0, BASIN.y - DIR.y * 0.4, 0.30, -1.0, WHEEL_END, 0.4, 4, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0],
      [at(2, 0.22),   BASIN.x, -0.8, BASIN.y, 0.30, -0.75, WHEEL_END, 0.4, 4, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0],  // "Live in your inner self alone"
      [at(2, 0.42),   -5.6,  -4.6,  19.8,  0.30, -0.08, WHEEL_END, 0.4, 4, 0.0, 1.0, 1.0, 0.1, 0.0, 0.0],  // "within your soul a world has grown"
      [at(2, 0.62),   -6.6,  -7.8,  12.0,  0.32, 0.55, WHEEL_END, 0.4, 4, 0.0, 1.0, 1.0, 0.7, 0.0, 0.0],   // "blinded by the outer light"
      [at(2, 0.8),    -7.2,  -9.0,   7.5,  0.30, 0.48, WHEEL_END, 0.4, 4, 0.0, 1.0, 1.0, 1.0, 0.2, 0.0],   // "drowned in the noise of day"
      [at(2, 1.0),    -8.0,  -10.0,  2.0,  0.28, 0.20, WHEEL_END, 0.4, 4, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],   // "take in their song"
      [T.total,       -10.0, -11.0, -8.0,  0.26, 0.10, WHEEL_END, 0.4, 4, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    ];
  },
  sound: {
    src: '/audio/wind.mp3',
    label: 'Play the night air',
    // Low over the lake, and nearly nothing beneath the surface.
    volume: function (row) { return row[1] > 0 ? 0.08 : 0.02; },
    cues: [{ stanza: 1, at: 0.96, play: plink }]
  }
});
