/*
 * Scene for "Do Not Stand at My Grave and Weep" (Mary Elizabeth Frye): a
 * field with one headstone, and everything the voice says it is instead.
 *
 * The twelve lines are read in couplets (maxLines 2), six panels:
 * I    Dusk: a plain headstone in the grass by a lone tree. "I am not
 *      there": the view lifts, passes over it and leaves.
 * II   Low over the meadow with the wind, threads of air streaming past and
 *      gusts rippling the long grass ("a thousand winds that blow"); then the
 *      field whitens and the snow glints like diamonds.
 * III  The snow gives way to ripened wheat in low gold sunlight; then grey
 *      skies, orange trees and a gentle autumn rain.
 * IV   "The morning's hush": dawn mist lying in a still field, until a flock
 *      bursts up out of the grass, "the swift uplifting rush" ...
 * V    ... "of quiet birds in circled flight", wheeling against the sunrise.
 *      The sky deepens and the soft stars come out.
 * VI   Gliding home under the stars to the headstone; on "I did not die"
 *      the view lifts from it to the sky.
 *
 * The camera flies one closed loop over the same rolling fields. The season
 * is a palette index (column "look") blended between seven looks; grass and
 * wheat are GPU-instanced in tiles that wrap round the camera, so the fields
 * never end; the wind threads are ribbons computed in a vertex shader; the
 * birds are a small boids flock. Columns:
 *   [unit, path, lift, snowfall, wind, look, yaw, pitch, threads, rain, flock]
 */
import { THREE, isSmall, makeRenderer, fitCamera, broadleafGeometry, softSprite, skyDome,
         distanceTo, scatter, particleField, rainField, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout (metres) ──────────────────────────────────────────────────────
var CENTER = { x: -95, z: -55 };                 // middle of the loop
var STONE = new THREE.Vector3(1.0, 0, -4.8);     // right of the path, clear of left-hand text
var LONE = new THREE.Vector3(13, 0, -30);
// A closed loop: out from the grave towards -z, round, and home from behind.
var PATH = [[0, 0], [0, -40], [-12, -100], [-60, -150], [-130, -165], [-190, -130], [-210, -60],
            [-185, 10], [-130, 55], [-65, 70], [-20, 50], [0, 18]];
var curve = new THREE.CatmullRomCurve3(PATH.map(function (p) { return new THREE.Vector3(p[0], 0, p[1]); }), true);
var distToPath = distanceTo(curve.getSpacedPoints(400));
var P_HUSH = 0.445;                               // where the birds rise

function wrap01(v) { return v - Math.floor(v); }

// Rolling fields, the grave on a gentle rise, hills ringing the horizon.
// The first part is mirrored in GLSL below (the ring is beyond the grass).
function ground(x, z) {
  var h = 2.4 * Math.sin(x * 0.018 + 0.6) * Math.cos(z * 0.015 - 0.3) + 1.3 * Math.sin(x * 0.041 + z * 0.033) +
          0.45 * Math.sin(x * 0.13 - z * 0.11) + 1.4 * Math.exp(-(x * x + (z + 5) * (z + 5)) / 900);
  var d = Math.hypot(x - CENTER.x, z - CENTER.z), a = Math.atan2(z - CENTER.z, x - CENTER.x);
  return h + smooth(330, 620, d) * (45 + 22 * Math.sin(a * 5 + 1) + 9 * Math.sin(a * 13));
}

var GLSL = [
  'float groundH(vec2 p){',
  '  return 2.4 * sin(p.x * 0.018 + 0.6) * cos(p.y * 0.015 - 0.3) + 1.3 * sin(p.x * 0.041 + p.y * 0.033) +',
  '         0.45 * sin(p.x * 0.13 - p.y * 0.11) + 1.4 * exp(-(p.x * p.x + (p.y + 5.0) * (p.y + 5.0)) / 900.0); }',
  'float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }',
  'float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);',
  '  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y); }',
  // Patchy cover for an overall amount (snow settling, wheat ripening).
  'float cover(vec2 p, float amt, float sc){ float n = vnoise(p * sc) * 0.65 + vnoise(p * sc * 4.3 + 17.0) * 0.35;',
  '  return smoothstep(n - 0.12, n + 0.12, amt * 1.3 - 0.15); }'
].join('\n');

// Three's order: tone map, convert, then fog in display space.
var FRAG_END = '#include <tonemapping_fragment>\n#include <colorspace_fragment>\n' +
               ' gl_FragColor.rgb = mix(gl_FragColor.rgb, uFog, 1.0 - exp(-uFogD * uFogD * vDepth * vDepth));\n';

// ── The seasons: seven looks the "look" column blends between ────────────
// el/az place the sun relative to the direction of travel (az > 0 is right).
var LOOKS = [
  { // 0 dusk at the grave
    top: '#141b3a', mid: '#3e4374', horizon: '#e0956c', glow: '#ffb070', glowAmt: 0.8, el: 0.035, az: 0.3,
    sun: '#ffab6a', sunI: 1.6, hemiSky: '#8a90c0', hemiGnd: '#4a4030', hemiI: 1.45, fog: '#7a5e6e', fogD: 0.0045,
    ground: '#404c26', ground2: '#566232', grass: '#a0ac62', grassH: 0.5, grassOn: 1, wheat: '#d8a850', wheatOn: 0,
    tree: '#56663a', snow: 0, wet: 0, stars: 0.15, clouds: 0.55, cloud: '#c88a84', mist: 0.1, exp: 1 },
  { // 1 the winds: a bright, breezy afternoon
    top: '#2a5a9e', mid: '#86acd6', horizon: '#dfe9f0', glow: '#fff3da', glowAmt: 0.25, el: 0.45, az: -0.6,
    sun: '#fff1d6', sunI: 2.6, hemiSky: '#c8dcf8', hemiGnd: '#4a5a30', hemiI: 1.3, fog: '#bccbd8', fogD: 0.0032,
    ground: '#4a6a2c', ground2: '#6a8a3c', grass: '#9ab868', grassH: 1.3, grassOn: 1, wheat: '#d8a850', wheatOn: 0,
    tree: '#5f8a3c', snow: 0, wet: 0, stars: 0, clouds: 0.9, cloud: '#ffffff', mist: 0, exp: 1 },
  { // 2 diamond glints on snow
    top: '#2f5c9c', mid: '#9fbde0', horizon: '#eef3f8', glow: '#fff8ec', glowAmt: 0.35, el: 0.1, az: 0.3,
    sun: '#fff4e4', sunI: 2.4, hemiSky: '#c8daf8', hemiGnd: '#8a9ab8', hemiI: 1.2, fog: '#d6e2ee', fogD: 0.0036,
    ground: '#5a6a48', ground2: '#6a7a58', grass: '#c0ccb8', grassH: 1.0, grassOn: 1, wheat: '#d8a850', wheatOn: 0,
    tree: '#e6ecf4', snow: 1, wet: 0, stars: 0, clouds: 0.25, cloud: '#ffffff', mist: 0.1, exp: 0.95 },
  { // 3 sunlight on ripened grain
    top: '#34528a', mid: '#d4a87c', horizon: '#ffcf90', glow: '#ffc070', glowAmt: 1, el: 0.07, az: 0.12,
    sun: '#ffc77a', sunI: 2.8, hemiSky: '#ffd2a8', hemiGnd: '#5a4420', hemiI: 1.0, fog: '#e0b482', fogD: 0.0036,
    ground: '#7a5a28', ground2: '#9a7838', grass: '#9a9a50', grassH: 0.6, grassOn: 0, wheat: '#e8b860', wheatOn: 1,
    tree: '#7a7a34', snow: 0, wet: 0, stars: 0, clouds: 0.45, cloud: '#ffd0a0', mist: 0.05, exp: 1 },
  { // 4 the gentle autumn rain
    top: '#3a424e', mid: '#646c78', horizon: '#9298a0', glow: '#000000', glowAmt: 0, el: 0.6, az: 0,
    sun: '#c8d0dc', sunI: 0.7, hemiSky: '#a0aabb', hemiGnd: '#3a3226', hemiI: 1.7, fog: '#7c848c', fogD: 0.0075,
    ground: '#3e3424', ground2: '#4e4028', grass: '#7a7448', grassH: 0.6, grassOn: 0.3, wheat: '#b89050', wheatOn: 0.85,
    tree: '#e88038', snow: 0, wet: 1, stars: 0, clouds: 1, cloud: '#5e646e', mist: 0.25, exp: 1.05 },
  { // 5 the morning's hush
    top: '#27325e', mid: '#a8829e', horizon: '#ffc49c', glow: '#ffd0a0', glowAmt: 0.9, el: 0.015, az: -0.25,
    sun: '#ffc89a', sunI: 1.3, hemiSky: '#b8aed0', hemiGnd: '#2a3024', hemiI: 0.95, fog: '#c4a4ac', fogD: 0.0075,
    ground: '#34462a', ground2: '#4a5a34', grass: '#7c9258', grassH: 0.9, grassOn: 1, wheat: '#d8a850', wheatOn: 0,
    tree: '#405a34', snow: 0, wet: 0.2, stars: 0.1, clouds: 0.5, cloud: '#f0a8a0', mist: 1, exp: 1 },
  { // 6 soft stars at night, with a low moon off to the right lighting the field
    top: '#02040c', mid: '#08112c', horizon: '#1e2c58', glow: '#000000', glowAmt: 0, el: 0.4, az: 0.7,
    sun: '#a4b8ec', sunI: 1.0, hemiSky: '#7088cc', hemiGnd: '#222838', hemiI: 1.9, fog: '#101a3a', fogD: 0.0045,
    ground: '#26342a', ground2: '#2e3c2e', grass: '#6a8062', grassH: 0.75, grassOn: 1, wheat: '#d8a850', wheatOn: 0,
    tree: '#34483c', snow: 0, wet: 0, stars: 1, clouds: 0, cloud: '#000000', mist: 0.05, exp: 1.15 }
];
var COLOR_KEYS = ['top', 'mid', 'horizon', 'glow', 'sun', 'hemiSky', 'hemiGnd', 'fog', 'ground', 'ground2', 'grass', 'wheat', 'tree', 'cloud'];
var NUM_KEYS = ['glowAmt', 'el', 'az', 'sunI', 'hemiI', 'fogD', 'grassH', 'grassOn', 'wheatOn', 'snow', 'wet', 'stars', 'clouds', 'mist', 'exp'];
LOOKS.forEach(function (l) { COLOR_KEYS.forEach(function (k) { l[k] = new THREE.Color(l[k]); }); });

// A soft rush of wings: a few layers of filtered noise beating at wing rate.
function wings(ac, out) {
  var t = ac.currentTime, len = 2.2, sr = ac.sampleRate, b = ac.createBuffer(1, Math.floor(sr * len), sr), d = b.getChannelData(0);
  var rates = [8.5, 11, 13.5], phase = [0, 1.3, 2.1];
  for (var i = 0; i < d.length; i++) {
    var x = i / sr, env = Math.min(1, x * 10) * Math.pow(1 - x / len, 2.2), s = 0;
    for (var k = 0; k < 3; k++) { var fl = 0.5 + 0.5 * Math.sin(x * Math.PI * 2 * rates[k] * (1 - x * 0.2) + phase[k]); s += fl * fl * fl; }
    d[i] = (Math.random() * 2 - 1) * s * 0.33 * env;
  }
  var src = ac.createBufferSource(), bp = ac.createBiquadFilter(), g = ac.createGain();
  src.buffer = b;
  bp.type = 'bandpass';
  bp.frequency.value = 1100;
  bp.Q.value = 0.6;
  g.gain.value = 0.6;
  src.connect(bp); bp.connect(g); g.connect(out);
  src.start(t);
}

// ── Geometry helpers ─────────────────────────────────────────────────────
// A blade of grass, 1 tall (y is the 0..1 height along it), curving forward.
function bladeGeometry() {
  var pos = [], idx = [], L = 4;
  for (var k = 0; k < L; k++) {
    var y = k / L, w = 0.035 * Math.pow(1 - y, 0.7), z = 0.12 * y * y;
    pos.push(-w, y, z, w, y, z);
  }
  pos.push(0, 1, 0.12);
  for (k = 0; k < L - 1; k++) { var a = k * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
  idx.push((L - 1) * 2, (L - 1) * 2 + 1, L * 2);
  var g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  return g;
}

// A wheat stalk with its ear, 1 tall: a flat stem and two crossed diamonds.
function wheatGeometry() {
  var pos = [], idx = [];
  function quad(a, b, c, d) { var n = pos.length / 3; pos.push.apply(pos, a.concat(b, c, d)); idx.push(n, n + 1, n + 2, n, n + 2, n + 3); }
  quad([-0.006, 0, 0], [0.006, 0, 0], [0.004, 0.8, 0], [-0.004, 0.8, 0]);
  quad([0, 0.76, 0], [0.022, 0.86, 0], [0, 1, 0], [-0.022, 0.86, 0]);
  quad([0, 0.76, 0], [0, 0.86, 0.022], [0, 1, 0], [0, 0.86, -0.022]);
  var g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  return g;
}

// Instances scattered in a square tile; the shader repeats it round the camera.
function tiled(base, count, tile, r, hMin, hMax, tint) {
  var g = new THREE.InstancedBufferGeometry();
  g.index = base.index;
  g.setAttribute('position', base.attributes.position);
  var inst = new Float32Array(count * 4), col = new Float32Array(count * 3), c = new THREE.Color();
  for (var i = 0; i < count; i++) {
    inst[i * 4] = (r() - 0.5) * tile;
    inst[i * 4 + 1] = (r() - 0.5) * tile;
    inst[i * 4 + 2] = hMin + r() * (hMax - hMin);
    inst[i * 4 + 3] = r() * 6.283;
    tint(c, r);
    col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
  }
  g.setAttribute('aInst', new THREE.InstancedBufferAttribute(inst, 4));
  g.setAttribute('aTint', new THREE.InstancedBufferAttribute(col, 3));
  g.instanceCount = count;
  return g;
}

function stoneTexture() {
  var c = document.createElement('canvas');
  c.width = 128; c.height = 256;
  var x = c.getContext('2d'), r = rng(8);
  x.fillStyle = '#8e8c86';
  x.fillRect(0, 0, 128, 256);
  for (var i = 0; i < 2600; i++) {
    var v = Math.floor(110 + r() * 70);
    x.fillStyle = 'rgba(' + v + ',' + v + ',' + (v - 6) + ',0.35)';
    x.fillRect(r() * 128, r() * 256, 1 + r() * 2, 1 + r() * 2);
  }
  // Lichen low down and on the shoulders; damp at the foot.
  for (i = 0; i < 70; i++) {
    var lx = r() * 128, ly = r() < 0.5 ? 200 + r() * 56 : r() * 60;
    x.fillStyle = 'rgba(' + (150 + r() * 40) + ',' + (150 + r() * 30) + ',90,' + (0.25 + r() * 0.3) + ')';
    x.beginPath(); x.arc(lx, ly, 1 + r() * 4, 0, 6.3); x.fill();
  }
  var gr = x.createLinearGradient(0, 256, 0, 190);
  gr.addColorStop(0, 'rgba(40,44,30,0.55)'); gr.addColorStop(1, 'rgba(40,44,30,0)');
  x.fillStyle = gr;
  x.fillRect(0, 190, 128, 66);
  // Worn lettering, too weathered to read.
  x.fillStyle = 'rgba(60,58,54,0.5)';
  [[34, 60, 74], [44, 82, 40], [30, 104, 68], [40, 126, 48]].forEach(function (l) {
    for (var k = 0; k < l[2]; k += 4 + Math.floor(r() * 4)) x.fillRect(l[0] + k, 256 - l[1] - 150, 2 + r() * 3, 6);
  });
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

// ── Renderer ─────────────────────────────────────────────────────────────
function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(53);
  var gl = makeRenderer(canvas, { clear: '#7a5e6e' });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#7a5e6e', 0.0045);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.1, 3000);

  // Uniforms shared by every custom shader (values set once a frame).
  var U = {
    uTime: { value: 0 }, uCam: { value: new THREE.Vector3() }, uWindDir: { value: new THREE.Vector2(0, -1) }, uWind: { value: 0 },
    uSunDir: { value: new THREE.Vector3(0, 1, 0) }, uSunCol: { value: new THREE.Color() },
    uHemiSky: { value: new THREE.Color() }, uHemiGnd: { value: new THREE.Color() },
    uFog: { value: new THREE.Color() }, uFogD: { value: 0.005 }, uSnow: { value: 0 }, uWet: { value: 0 }
  };

  // ── Sky ──
  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#141b3a', mid: '#3e4374', horizon: '#e0956c', sun: '#ffb070' }, 1500);
  sky.add(dome.mesh);

  var NS = small ? 2600 : 5200, sp = [], sa = [];
  for (var i = 0; i < NS; i++) {
    var th = r() * Math.PI * 2, y = 0.03 + r() * 0.97, s = Math.sqrt(1 - y * y);
    sp.push(1300 * s * Math.cos(th), 1300 * y, 1300 * s * Math.sin(th));
    sa.push(0.7 + Math.pow(r(), 3) * 2.8, r() * 6.28);
  }
  // The Milky Way: a band of fainter, denser stars.
  var band = new THREE.Euler(0.25, 0.45, 1.25), v3 = new THREE.Vector3();
  for (i = 0; i < (small ? 4000 : 9000); i++) {
    var bt = r() * Math.PI * 2, by = (r() + r() + r() - 1.5) * 0.1;
    v3.set(Math.cos(bt), by, Math.sin(bt)).normalize().applyEuler(band);
    if (v3.y < 0.03) continue;
    sp.push(v3.x * 1280, v3.y * 1280, v3.z * 1280);
    sa.push(0.7 + r() * 0.9, r() * 6.28);
  }
  var starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.Float32BufferAttribute(sp, 3));
  starGeo.setAttribute('star', new THREE.Float32BufferAttribute(sa, 2));
  var starMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uTime: U.uTime, uAmt: { value: 0 }, uScale: { value: 1 } },
    vertexShader: 'attribute vec2 star; uniform float uTime; uniform float uAmt; uniform float uScale; varying float vA;\n' +
      'void main(){ gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);\n' +
      ' vA = uAmt * (0.7 + 0.3 * sin(uTime * (1.2 + fract(star.y * 3.7) * 2.0) + star.y));\n' +
      ' gl_PointSize = star.x * uScale; }',
    fragmentShader: 'varying float vA; void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;\n' +
      ' float a = smoothstep(0.5, 0.0, d) * vA; gl_FragColor = vec4(vec3(0.86, 0.9, 1.0) * a, a);\n #include <colorspace_fragment>\n }'
  });
  var stars = new THREE.Points(starGeo, starMat);
  stars.frustumCulled = false;
  sky.add(stars);

  var cloudTex = softSprite('rgba(255,255,255,0.9)', 'rgba(255,255,255,0)'), clouds = new THREE.Group(), cloudList = [];
  for (i = 0; i < 30; i++) {
    var ca = r() * Math.PI * 2, cd = 850 + r() * 250;
    var cl = new THREE.Sprite(new THREE.SpriteMaterial({ map: cloudTex, transparent: true, depthWrite: false, fog: false, opacity: 0 }));
    cl.position.set(Math.cos(ca) * cd, 90 + r() * 260, Math.sin(ca) * cd);
    cl.scale.set(260 + r() * 300, 70 + r() * 70, 1);
    cl.userData.base = 0.5 + r() * 0.5;
    clouds.add(cl);
    cloudList.push(cl);
  }
  sky.add(clouds);

  var hemi = new THREE.HemisphereLight('#8088b8', '#2a2a1a', 0.9);
  var sun = new THREE.DirectionalLight('#ffab6a', 1.6);
  world.add(hemi, sun, sun.target);

  // ── Ground ──
  var groundUniforms = Object.assign({ uGround: { value: new THREE.Color() }, uGround2: { value: new THREE.Color() },
                                       uSnowCol: { value: new THREE.Color('#eef4ff') } }, U);
  var terrainGeo = new THREE.PlaneGeometry(1500, 1500, small ? 150 : 220, small ? 150 : 220).rotateX(-Math.PI / 2).translate(CENTER.x, 0, CENTER.z);
  var tp = terrainGeo.attributes.position;
  for (i = 0; i < tp.count; i++) tp.setY(i, ground(tp.getX(i), tp.getZ(i)));
  terrainGeo.computeVertexNormals();
  var terrainMat = new THREE.ShaderMaterial({
    uniforms: groundUniforms,
    vertexShader: 'varying vec3 vW; varying vec3 vN; varying float vDepth;\n' +
      'void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; vN = normal; vec4 mv = viewMatrix * w; vDepth = -mv.z; gl_Position = projectionMatrix * mv; }',
    fragmentShader: 'uniform vec3 uGround; uniform vec3 uGround2; uniform vec3 uSnowCol; uniform float uSnow; uniform float uWet;\n' +
      'uniform vec3 uSunDir; uniform vec3 uSunCol; uniform vec3 uHemiSky; uniform vec3 uHemiGnd; uniform vec3 uFog; uniform float uFogD;\n' +
      'varying vec3 vW; varying vec3 vN; varying float vDepth;\n' + GLSL + '\n' +
      'void main(){ float n = vnoise(vW.xz * 0.07) * 0.6 + vnoise(vW.xz * 0.45) * 0.4;\n' +
      ' vec3 base = mix(uGround, uGround2, n);\n' +
      ' float sn = cover(vW.xz, uSnow, 0.05); base = mix(base, uSnowCol, sn);\n' +
      ' vec3 N = normalize(vN);\n' +
      ' vec3 c = base * (mix(uHemiGnd, uHemiSky, N.y * 0.5 + 0.5) + uSunCol * max(dot(N, uSunDir), 0.0)) / 3.14159;\n' +
      ' vec3 V = normalize(cameraPosition - vW); float fres = pow(1.0 - max(dot(N, V), 0.0), 4.0);\n' +
      ' c += uHemiSky * fres * (uWet * 0.12 + sn * 0.05);\n' +
      ' gl_FragColor = vec4(c, 1.0);\n' + FRAG_END + '}'
  });
  world.add(new THREE.Mesh(terrainGeo, terrainMat));

  // ── Grass and wheat, tiled round the camera ──
  function vegMaterial(stiff) {
    var u = Object.assign({ uTint: { value: new THREE.Color() }, uOn: { value: 1 }, uHeight: { value: 1 }, uTile: { value: 1 },
                            uStiff: { value: stiff }, uCoverScale: { value: 0.06 } }, U);
    return new THREE.ShaderMaterial({
      side: THREE.DoubleSide, uniforms: u,
      vertexShader: 'attribute vec4 aInst; attribute vec3 aTint;\n' +
        'uniform vec3 uCam; uniform float uTile; uniform float uTime; uniform float uWind; uniform vec2 uWindDir; uniform float uOn;\n' +
        'uniform float uHeight; uniform float uStiff; uniform float uCoverScale; uniform float uSnow;\n' +
        'uniform vec3 uSunDir; uniform vec3 uSunCol; uniform vec3 uHemiSky; uniform vec3 uHemiGnd; uniform vec3 uTint;\n' +
        'varying vec3 vCol; varying float vDepth;\n' + GLSL + '\n' +
        'void main(){\n' +
        ' vec2 w = aInst.xy + uTile * floor((uCam.xz - aInst.xy) / uTile + 0.5);\n' +
        ' float d = distance(w, uCam.xz), buried = cover(w, uSnow, 0.05);\n' +
        ' float s = aInst.z * uHeight * cover(w + 40.0, uOn, uCoverScale) * (1.0 - smoothstep(uTile * 0.3, uTile * 0.48, d)) * (1.0 - 0.8 * buried);\n' +
        ' float k = position.y, c = cos(aInst.w), sn = sin(aInst.w);\n' +
        ' vec3 p = vec3(c * position.x - sn * position.z, position.y, sn * position.x + c * position.z) * s;\n' +
        // Gusts: bands of wind running across the field, and a flutter.
        ' float g = 0.5 + 0.5 * sin(dot(w, uWindDir) * 0.16 - uTime * 2.4 + vnoise(w * 0.04) * 5.0); g *= g;\n' +
        ' float bend = (0.06 + uWind * (0.2 + 1.0 * g) + sin(uTime * 4.0 + aInst.w * 7.0 + w.x * 0.7) * 0.04 * (0.4 + uWind)) * uStiff;\n' +
        ' p.xz += uWindDir * bend * k * k * s;\n' +
        ' p.y -= bend * bend * 0.4 * k * k * s;\n' +
        ' vec3 wp = vec3(w.x, groundH(w), w.y) + p;\n' +
        ' vec4 mv = viewMatrix * vec4(wp, 1.0); gl_Position = projectionMatrix * mv; vDepth = -mv.z;\n' +
        // Sky light from above, and the sun through the blades when you face it.
        ' float back = pow(max(dot(normalize(wp - cameraPosition), uSunDir), 0.0), 3.0);\n' +
        ' vec3 lit = mix(uHemiGnd, uHemiSky, 0.3 + 0.7 * k) + uSunCol * (0.3 + 0.4 * k + back * k * 1.4);\n' +
        ' vec3 base = mix(aTint * uTint, vec3(0.9, 0.93, 1.0), buried * 0.7);\n' +
        ' vCol = base * lit * mix(0.3, 1.0, k) / 3.14159 + uSunCol * g * uWind * k * 0.025; }',
      fragmentShader: 'uniform vec3 uFog; uniform float uFogD; varying vec3 vCol; varying float vDepth;\n' +
        'void main(){ gl_FragColor = vec4(vCol, 1.0);\n' + FRAG_END + '}'
    });
  }
  var GRASS_TILE = small ? 56 : 70, WHEAT_TILE = small ? 50 : 60;
  var grassMat = vegMaterial(1), wheatMat = vegMaterial(0.45);
  grassMat.uniforms.uTile.value = GRASS_TILE;
  wheatMat.uniforms.uTile.value = WHEAT_TILE;
  wheatMat.uniforms.uCoverScale.value = 0.09;
  var grass = new THREE.Mesh(tiled(bladeGeometry(), small ? 12000 : 32000, GRASS_TILE, r, 0.5, 1.15, function (c, rr) {
    c.setHSL(0.22 + rr() * 0.08, 0.35 + rr() * 0.2, 0.42 + rr() * 0.22);
  }), grassMat);
  var wheat = new THREE.Mesh(tiled(wheatGeometry(), small ? 11000 : 30000, WHEAT_TILE, r, 0.85, 1.15, function (c, rr) {
    c.setHSL(0.1 + rr() * 0.03, 0.5, 0.45 + rr() * 0.15);
  }), wheatMat);
  grass.frustumCulled = wheat.frustumCulled = false;
  world.add(grass, wheat);

  // Diamond glints on the snow: points that flash as the view moves.
  var GL_TILE = 46, NG = small ? 4000 : 10000, gInst = new Float32Array(NG * 4);
  for (i = 0; i < NG; i++) { gInst[i * 4] = (r() - 0.5) * GL_TILE; gInst[i * 4 + 1] = (r() - 0.5) * GL_TILE; gInst[i * 4 + 2] = r(); gInst[i * 4 + 3] = r(); }
  var glintGeo = new THREE.BufferGeometry();
  glintGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(NG * 3), 3));
  glintGeo.setAttribute('aInst', new THREE.Float32BufferAttribute(gInst, 4));
  var glintMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: Object.assign({ uTile: { value: GL_TILE }, uScale: { value: 1 } }, U),
    vertexShader: 'attribute vec4 aInst; uniform vec3 uCam; uniform float uTile; uniform float uTime; uniform float uSnow; uniform float uScale;\n' +
      'varying float vA; varying float vWarm;\n' + GLSL + '\n' +
      'void main(){ vec2 w = aInst.xy + uTile * floor((uCam.xz - aInst.xy) / uTile + 0.5);\n' +
      ' vec3 wp = vec3(w.x, groundH(w) + 0.04, w.y); vec3 v = normalize(wp - cameraPosition);\n' +
      ' float ph = aInst.z * 80.0 + dot(v, vec3(41.0, 23.0, 37.0)) + uTime * (0.6 + aInst.w);\n' +
      ' float flash = pow(max(sin(ph), 0.0), 16.0);\n' +
      ' float d = distance(w, uCam.xz);\n' +
      ' vA = flash * cover(w, uSnow, 0.05) * smoothstep(uTile * 0.5, uTile * 0.25, d) * smoothstep(0.4, 0.9, uSnow);\n' +
      ' vWarm = aInst.w;\n' +
      ' vec4 mv = viewMatrix * vec4(wp, 1.0); gl_Position = projectionMatrix * mv;\n' +
      ' gl_PointSize = (2.0 + flash * 5.0) * uScale * (vA > 0.003 ? 1.0 : 0.0); }',
    fragmentShader: 'varying float vA; varying float vWarm; void main(){ vec2 c = gl_PointCoord - 0.5;\n' +
      ' float a = (smoothstep(0.5, 0.0, length(c)) + smoothstep(0.08, 0.0, abs(c.x)) * smoothstep(0.5, 0.0, abs(c.y)) + smoothstep(0.08, 0.0, abs(c.y)) * smoothstep(0.5, 0.0, abs(c.x))) * vA;\n' +
      ' vec3 col = mix(vec3(0.8, 0.92, 1.0), vec3(1.0, 0.95, 0.85), vWarm);\n' +
      ' gl_FragColor = vec4(col * a, a);\n #include <colorspace_fragment>\n }'
  });
  var glints = new THREE.Points(glintGeo, glintMat);
  glints.frustumCulled = false;
  world.add(glints);

  // ── Trees: groves and a lone tree over the grave ──
  var treeMat = new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true });
  var up = new THREE.Vector3(0, 1, 0), groves = [];
  for (i = 0; i < 16; i++) {
    var ga = r() * Math.PI * 2, gd = 40 + r() * 230;
    groves.push([CENTER.x + Math.cos(ga) * gd, CENTER.z + Math.sin(ga) * gd, 10 + r() * 26]);
  }
  [broadleafGeometry(rng(7), '#3e3026'), broadleafGeometry(rng(19), '#3e3026')].forEach(function (geo, gi) {
    var trees = new THREE.InstancedMesh(geo, treeMat, small ? 70 : 150);
    scatter(trees, 6000, function (k, p, q, s, c) {
      var gr = groves[Math.floor(r() * groves.length)], a = r() * 6.28, d = Math.sqrt(r()) * gr[2];
      var x = gr[0] + Math.cos(a) * d, z = gr[1] + Math.sin(a) * d;
      if (distToPath(x, z) < 20 || Math.hypot(x, z + 5) < 55) return false;
      p.set(x, ground(x, z) - 0.3, z);
      q.setFromAxisAngle(up, r() * 6.28);
      var sc = 1.1 + r() * 0.8;
      s.set(sc, sc * (0.85 + r() * 0.35), sc);
      c.setHSL(0, 0, 0.75 + r() * 0.25);
    });
    world.add(trees);
  });
  var lone = new THREE.Mesh(broadleafGeometry(rng(5), '#3e3026'), treeMat);
  lone.position.set(LONE.x, ground(LONE.x, LONE.z) - 0.3, LONE.z);
  lone.scale.set(2.3, 1.7, 2.3);
  lone.rotation.y = 0.6;
  world.add(lone);

  // ── The headstone ──
  var shape = new THREE.Shape(), hw = 0.3, hh = 0.6;
  shape.moveTo(-hw, -0.1);
  shape.lineTo(hw, -0.1);
  shape.lineTo(hw, hh);
  shape.absarc(0, hh, hw, 0, Math.PI, false);
  shape.lineTo(-hw, -0.1);
  var stoneGeo = new THREE.ExtrudeGeometry(shape, { depth: 0.1, bevelEnabled: true, bevelThickness: 0.018, bevelSize: 0.018,
                                                    bevelSegments: 2, curveSegments: 18 });
  stoneGeo.translate(0, 0, -0.05);
  var stoneTex = stoneTexture();
  stoneTex.repeat.set(1.6, 1.0);
  stoneTex.offset.set(0.5, 0.1);
  var stone = new THREE.Mesh(stoneGeo, new THREE.MeshLambertMaterial({ map: stoneTex, emissive: '#2a2622' }));
  stone.position.set(STONE.x, ground(STONE.x, STONE.z), STONE.z);
  stone.rotation.set(-0.04, -0.12, 0.025);
  world.add(stone);

  // ── Mist lying in the fields ──
  var mistTex = softSprite('rgba(255,255,255,0.7)', 'rgba(255,255,255,0)'), mist = [], MIST_TILE = 200;
  for (i = 0; i < (small ? 24 : 40); i++) {
    var m = new THREE.Sprite(new THREE.SpriteMaterial({ map: mistTex, transparent: true, depthWrite: false, fog: false, opacity: 0 }));
    m.userData = { x: (r() - 0.5) * MIST_TILE, z: (r() - 0.5) * MIST_TILE, h: 0.8 + r() * 3.5, base: 0.5 + r() * 0.5 };
    m.scale.set(36 + r() * 30, 5 + r() * 4, 1);
    world.add(m);
    mist.push(m);
  }

  // ── Particles: snowfall, gold motes, autumn leaves, rain, fireflies ──
  var snowfall = particleField({ count: small ? 700 : 1600, box: [40, 16, 40], fall: [0.6, 1.3], size: 0.07, color: '#ffffff',
                                 map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)'), sway: 0.5, windSpeed: 4 });
  var motes = particleField({ count: small ? 250 : 500, box: [30, 8, 30], fall: [-0.06, 0.06], size: 0.06, color: '#ffe0a0',
                              map: softSprite('rgba(255,235,190,1)', 'rgba(255,215,150,0)'), sway: 0.3, windSpeed: 1 });
  motes.points.material.blending = THREE.AdditiveBlending;
  var leaves = particleField({ count: small ? 120 : 240, box: [36, 14, 36], fall: [0.7, 1.4], size: 0.16, map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)'), alphaTest: 0.2,
                               colors: ['#d8742a', '#c0461e', '#e8a33a', '#a8521e'], sway: 1.0, windSpeed: 3 });
  var fireflies = particleField({ count: small ? 50 : 100, box: [36, 3, 36], fall: [-0.1, 0.1], size: 0.055, color: '#d8ff8a',
                                  map: softSprite('rgba(230,255,160,1)', 'rgba(210,255,120,0)'), sway: 0.5, windSpeed: 0.3 });
  fireflies.points.material.blending = THREE.AdditiveBlending;
  var rain = rainField({ count: small ? 900 : 2000, box: [30, 18, 40], color: '#b8c2d0', opacity: 0.32, speed: 11, windSpeed: 2 });
  world.add(snowfall.points, motes.points, leaves.points, fireflies.points, rain.lines);

  // ── Wind threads: ribbons streaming with the wind round the camera ──
  var TH = small ? 90 : 190, K = 16, SPAN = 90;
  var nv = TH * K * 2, aPt = new Float32Array(nv * 2), aSeed = new Float32Array(nv * 4), aShape = new Float32Array(nv * 4), tIdx = [];
  for (i = 0; i < TH; i++) {
    var seed = [(r() - 0.5) * 44, 0.3 + Math.pow(r(), 2) * 2.8, r() * SPAN, 10 + r() * 9];
    var shp = [8 + r() * 14, 0.2 + r() * 0.8, 0.08 + r() * 0.22, r() * 6.28];
    for (var j = 0; j < K; j++) {
      for (var sd = 0; sd < 2; sd++) {
        var vi = (i * K + j) * 2 + sd;
        aPt[vi * 2] = j / (K - 1);
        aPt[vi * 2 + 1] = sd ? 1 : -1;
        for (var q4 = 0; q4 < 4; q4++) { aSeed[vi * 4 + q4] = seed[q4]; aShape[vi * 4 + q4] = shp[q4]; }
      }
      if (j < K - 1) { var a0 = (i * K + j) * 2; tIdx.push(a0, a0 + 2, a0 + 1, a0 + 1, a0 + 2, a0 + 3); }
    }
  }
  var threadGeo = new THREE.BufferGeometry();
  threadGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(nv * 3), 3));
  threadGeo.setAttribute('aPt', new THREE.Float32BufferAttribute(aPt, 2));
  threadGeo.setAttribute('aSeed', new THREE.Float32BufferAttribute(aSeed, 4));
  threadGeo.setAttribute('aShape', new THREE.Float32BufferAttribute(aShape, 4));
  threadGeo.setIndex(tIdx);
  var threadMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    uniforms: { uAnchor: { value: new THREE.Vector3() }, uDir: U.uWindDir, uClock: { value: 0 }, uSpan: { value: SPAN },
                uAmt: { value: 0 }, uWidth: { value: 0.055 }, uColor: { value: new THREE.Color('#e8eeff') } },
    vertexShader: 'attribute vec2 aPt; attribute vec4 aSeed; attribute vec4 aShape;\n' +
      'uniform vec3 uAnchor; uniform vec2 uDir; uniform float uClock; uniform float uSpan; uniform float uAmt; uniform float uWidth;\n' +
      'varying float vA; varying float vE;\n' +
      'vec3 at(float s){ vec3 d = vec3(uDir.x, 0.0, uDir.y), sd = vec3(-uDir.y, 0.0, uDir.x);\n' +
      ' float wob = sin(aShape.z * s + aShape.w + uClock * 0.8);\n' +
      ' float lift = aSeed.y + aShape.y * 0.5 * sin(aShape.z * 0.6 * s + aShape.w * 1.7 + uClock * 0.5);\n' +
      ' return uAnchor + d * s + sd * (aSeed.x + aShape.y * wob) + vec3(0.0, lift, 0.0); }\n' +
      'void main(){ float head = mod(aSeed.z + uClock * aSeed.w, uSpan) - uSpan * 0.5;\n' +
      ' float s = head - aShape.x * (1.0 - aPt.x);\n' +
      ' vec3 p = at(s), t = normalize(at(s + 0.4) - p);\n' +
      ' vec3 side = normalize(cross(t, normalize(cameraPosition - p)));\n' +
      ' p += side * aPt.y * uWidth * (0.2 + 0.8 * sin(3.14159 * aPt.x));\n' +
      ' float fade = smoothstep(-0.5 * uSpan, -0.25 * uSpan, head - aShape.x) * (1.0 - smoothstep(0.25 * uSpan, 0.5 * uSpan, head));\n' +
      ' vA = pow(sin(3.14159 * aPt.x), 1.5) * (0.35 + 0.65 * aPt.x) * fade * uAmt * smoothstep(2.5, 8.0, distance(p, cameraPosition)); vE = aPt.y;\n' +
      ' gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0); }',
    fragmentShader: 'uniform vec3 uColor; varying float vA; varying float vE;\n' +
      'void main(){ float a = vA * (1.0 - vE * vE); gl_FragColor = vec4(uColor * a, a);\n #include <colorspace_fragment>\n }'
  });
  var threads = new THREE.Mesh(threadGeo, threadMat);
  threads.frustumCulled = false;
  world.add(threads);

  // ── The flock ──
  var NB = small ? 140 : 280;
  var birdGeo = new THREE.BufferGeometry();
  birdGeo.setAttribute('position', new THREE.Float32BufferAttribute([
    0, 0, 0.14, -0.2, 0.02, 0.03, 0, 0, -0.08,       -0.2, 0.02, 0.03, -0.46, 0, -0.1, -0.18, 0.01, -0.05,
    0, 0, 0.14, 0, 0, -0.08, 0.2, 0.02, 0.03,        0.2, 0.02, 0.03, 0.18, 0.01, -0.05, 0.46, 0, -0.1,
    0, 0.01, 0.2, -0.035, 0, -0.16, 0.035, 0, -0.16], 3));
  var phases = new Float32Array(NB);
  for (i = 0; i < NB; i++) phases[i] = r() * 6.28;
  birdGeo.setAttribute('aPh', new THREE.InstancedBufferAttribute(phases, 1));
  var birdClock = { value: 0 };
  var birdMat = new THREE.MeshBasicMaterial({ color: '#1d1a26', side: THREE.DoubleSide, transparent: true });
  birdMat.onBeforeCompile = function (sh) {
    sh.uniforms.uClock = birdClock;
    sh.vertexShader = 'attribute float aPh; uniform float uClock;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n float glide = smoothstep(-0.3, 0.4, sin(uClock * 0.7 + aPh * 3.0));\n' +
      ' transformed.y += sin(uClock * 13.0 + aPh) * abs(position.x) * 0.9 * glide;');
  };
  var birds = new THREE.InstancedMesh(birdGeo, birdMat, NB);
  birds.frustumCulled = false;
  birds.visible = false;
  world.add(birds);
  var bPos = new Float32Array(NB * 3), bVel = new Float32Array(NB * 3), bRest = new Float32Array(NB * 3), flying = false;
  var hushAt = curve.getPointAt(P_HUSH), hushTan = curve.getTangentAt(P_HUSH);
  // They rest in the grass ~28 m ahead, then wheel round a point further on and to the right.
  var nest = new THREE.Vector3(hushAt.x + hushTan.x * 28 - hushTan.z * 3, 0, hushAt.z + hushTan.z * 28 + hushTan.x * 3);
  var wheel = new THREE.Vector3(hushAt.x + hushTan.x * 62 - hushTan.z * 9, 0, hushAt.z + hushTan.z * 62 + hushTan.x * 9);
  wheel.y = ground(wheel.x, wheel.z);
  for (i = 0; i < NB; i++) {
    var na = r() * 6.28, nd = Math.sqrt(r()) * 9;
    bRest[i * 3] = nest.x + Math.cos(na) * nd;
    bRest[i * 3 + 2] = nest.z + Math.sin(na) * nd;
    bRest[i * 3 + 1] = ground(bRest[i * 3], bRest[i * 3 + 2]) + 0.2;
  }

  // Where the wind blows: along the meadow stretch of the path, angled right.
  var wt = curve.getTangentAt(0.09);
  U.uWindDir.value.set(wt.x * Math.cos(0.35) - wt.z * Math.sin(0.35), wt.z * Math.cos(0.35) + wt.x * Math.sin(0.35)).normalize();

  var cur = {};
  COLOR_KEYS.forEach(function (k) { cur[k] = new THREE.Color(); });
  function blend(L) {
    L = clamp(L, 0, LOOKS.length - 1);
    var i0 = Math.min(Math.floor(L), LOOKS.length - 2), t = L - i0, a = LOOKS[i0], b = LOOKS[i0 + 1];
    for (var k = 0; k < COLOR_KEYS.length; k++) cur[COLOR_KEYS[k]].copy(a[COLOR_KEYS[k]]).lerp(b[COLOR_KEYS[k]], t);
    for (k = 0; k < NUM_KEYS.length; k++) cur[NUM_KEYS[k]] = lerp(a[NUM_KEYS[k]], b[NUM_KEYS[k]], t);
  }

  var camPos = new THREE.Vector3(), look = new THREE.Vector3(), tan = new THREE.Vector3(), attract = new THREE.Vector3();
  var m4 = new THREE.Matrix4(), bq = new THREE.Quaternion(), bs = new THREE.Vector3(2.2, 2.2, 2.2), bp = new THREE.Vector3();
  var bv = new THREE.Vector3(), FWD = new THREE.Vector3(0, 0, 1);
  var pf = { snow: 0, wind: 0, dt: 0, time: 0 }, clock = 0, H = 800, portrait = false;

  function updateBirds(f, flock, dt) {
    if (flock < 0.03) {
      if (flying) { bPos.set(bRest); flying = false; }
      birds.visible = false;
      return;
    }
    if (!flying) {
      // Take off together: a burst up and away out of the grass.
      flying = true;
      bPos.set(bRest);
      for (var b = 0; b < NB; b++) {
        bVel[b * 3] = (Math.random() - 0.5) * 4 + hushTan.x * 3;
        bVel[b * 3 + 1] = 6 + Math.random() * 6;
        bVel[b * 3 + 2] = (Math.random() - 0.5) * 4 + hushTan.z * 3;
      }
    }
    birds.visible = true;
    birdMat.opacity = 1 - smooth(1.5, 2, flock);
    var rise = smooth(0, 1, flock), away = smooth(1, 2, flock), ang = clock * 0.5;
    var cxw = lerp(nest.x, wheel.x, rise), czw = lerp(nest.z, wheel.z, rise), rad = 3 + rise * 16;
    attract.set(cxw + Math.cos(ang) * rad + hushTan.x * away * 160, wheel.y + 2 + rise * 20 + Math.sin(ang * 2) * 3 + away * 70,
                czw + Math.sin(ang) * rad * 0.8 + hushTan.z * away * 160);
    for (var i = 0; i < NB; i++) {
      var o = i * 3, px = bPos[o], py = bPos[o + 1], pz = bPos[o + 2];
      var ax = attract.x - px, ay = attract.y - py, az = attract.z - pz, ad = Math.hypot(ax, ay, az) || 1, pull = 4 + Math.min(ad, 30) * 0.25;
      var fx = ax / ad * pull, fy = ay / ad * pull, fz = az / ad * pull;
      var sx = 0, sy = 0, sz = 0, vx = 0, vy = 0, vz = 0, cx = 0, cy = 0, cz = 0;
      for (var k = 1; k <= 10; k++) {
        var jo = ((i + k * 37) % NB) * 3, dx = bPos[jo] - px, dy = bPos[jo + 1] - py, dz = bPos[jo + 2] - pz, d2 = dx * dx + dy * dy + dz * dz + 0.01;
        if (d2 < 9) { sx -= dx / d2; sy -= dy / d2; sz -= dz / d2; }
        vx += bVel[jo]; vy += bVel[jo + 1]; vz += bVel[jo + 2];
        cx += dx; cy += dy; cz += dz;
      }
      fx += sx * 7 + (vx / 10 - bVel[o]) * 0.9 + cx / 10 * 0.15;
      fy += sy * 7 + (vy / 10 - bVel[o + 1]) * 0.9 + cy / 10 * 0.15;
      fz += sz * 7 + (vz / 10 - bVel[o + 2]) * 0.9 + cz / 10 * 0.15;
      var nvx = bVel[o] + fx * dt, nvy = bVel[o + 1] + fy * dt, nvz = bVel[o + 2] + fz * dt, sp2 = Math.hypot(nvx, nvy, nvz) || 1;
      var lim = clamp(sp2, 5, 12) / sp2;
      bVel[o] = nvx * lim; bVel[o + 1] = nvy * lim; bVel[o + 2] = nvz * lim;
      bPos[o] += bVel[o] * dt; bPos[o + 1] += bVel[o + 1] * dt; bPos[o + 2] += bVel[o + 2] * dt;
      var floor = ground(bPos[o], bPos[o + 2]) + 0.6;
      if (bPos[o + 1] < floor) { bPos[o + 1] = floor; bVel[o + 1] = Math.abs(bVel[o + 1]); }
      bp.set(bPos[o], bPos[o + 1], bPos[o + 2]);
      bv.set(bVel[o], bVel[o + 1] * 0.5, bVel[o + 2]).normalize();
      bq.setFromUnitVectors(FWD, bv);
      birds.setMatrixAt(i, m4.compose(bp, bq, bs));
    }
    birds.instanceMatrix.needsUpdate = true;
  }

  function frame(f) {
    var row = f.row, dt = f.dt, slow = env.reduceMotion;
    var lift = row[1], yaw = row[5], pitch = row[6], threadAmt = row[7], rainAmt = row[8], flock = row[9];
    clock += dt * (slow ? 0.5 : 1);
    blend(row[4]);

    // ── Camera on the loop ──
    var p = wrap01(f.cam);
    curve.getPointAt(p, camPos);
    curve.getPointAt(wrap01(p + 0.025), look);
    curve.getTangentAt(p, tan);
    camera.position.set(camPos.x, ground(camPos.x, camPos.z) + lift + Math.sin(f.time * 0.8) * 0.04, camPos.z);
    look.y = camera.position.y - 0.2;
    camera.lookAt(look);
    camera.rotateY(yaw - f.mx * 0.14);
    // On a phone the verse spans the width: tip up a little near the ground so the stone sits below it.
    camera.rotateX(pitch - f.my * 0.07 + (portrait ? 0.1 * (1 - smooth(2.5, 5, lift)) : 0));
    sky.position.copy(camera.position);

    // ── Light and sky for the season ──
    var ce = Math.cos(cur.el), hx = tan.x * Math.cos(cur.az) - tan.z * Math.sin(cur.az), hz = tan.z * Math.cos(cur.az) + tan.x * Math.sin(cur.az);
    U.uSunDir.value.set(hx * ce, Math.sin(cur.el), hz * ce).normalize();
    dome.uniforms.sunDir.value.copy(U.uSunDir.value);
    dome.uniforms.top.value.copy(cur.top);
    dome.uniforms.mid.value.copy(cur.mid);
    dome.uniforms.horizon.value.copy(cur.horizon);
    dome.uniforms.sunColor.value.copy(cur.glow).multiplyScalar(cur.glowAmt);
    sun.position.copy(camera.position).addScaledVector(U.uSunDir.value, 100);
    sun.target.position.copy(camera.position);
    sun.color.copy(cur.sun);
    sun.intensity = cur.sunI;
    hemi.color.copy(cur.hemiSky);
    hemi.groundColor.copy(cur.hemiGnd);
    hemi.intensity = cur.hemiI;
    U.uSunCol.value.copy(cur.sun).multiplyScalar(cur.sunI);
    U.uHemiSky.value.copy(cur.hemiSky).multiplyScalar(cur.hemiI);
    U.uHemiGnd.value.copy(cur.hemiGnd).multiplyScalar(cur.hemiI);
    world.fog.color.copy(cur.fog);
    world.fog.density = cur.fogD;
    U.uFog.value.copy(cur.fog);
    U.uFogD.value = cur.fogD;
    gl.setClearColor(cur.fog);
    gl.toneMappingExposure = cur.exp;

    starMat.uniforms.uAmt.value = cur.stars;
    starMat.uniforms.uScale.value = Math.min(window.devicePixelRatio || 1, 2) * (H / 800 * 0.6 + 0.5) * 1.3;
    stars.visible = cur.stars > 0.01;
    clouds.rotation.y += dt * (0.002 + f.wind * 0.01);
    for (var c = 0; c < cloudList.length; c++) {
      cloudList[c].material.color.copy(cur.cloud);
      cloudList[c].material.opacity = cur.clouds * cloudList[c].userData.base;
    }
    clouds.visible = cur.clouds > 0.01;

    // ── The fields ──
    U.uTime.value = clock;
    U.uCam.value.copy(camera.position);
    U.uWind.value = f.wind;
    U.uSnow.value = cur.snow;
    U.uWet.value = cur.wet;
    groundUniforms.uGround.value.copy(cur.ground);
    groundUniforms.uGround2.value.copy(cur.ground2);
    grassMat.uniforms.uTint.value.copy(cur.grass);
    grassMat.uniforms.uOn.value = cur.grassOn;
    grassMat.uniforms.uHeight.value = cur.grassH;
    grass.visible = cur.grassOn > 0.01;
    wheatMat.uniforms.uTint.value.copy(cur.wheat);
    wheatMat.uniforms.uOn.value = cur.wheatOn;
    wheat.visible = cur.wheatOn > 0.01;
    glintMat.uniforms.uScale.value = Math.min(window.devicePixelRatio || 1, 2);
    glints.visible = cur.snow > 0.4;
    treeMat.color.copy(cur.tree);

    for (var m = 0; m < mist.length; m++) {
      var md = mist[m], mu = md.userData;
      var wx = mu.x + MIST_TILE * Math.floor((camera.position.x - mu.x) / MIST_TILE + 0.5);
      var wz = mu.z + MIST_TILE * Math.floor((camera.position.z - mu.z) / MIST_TILE + 0.5);
      md.position.set(wx, ground(wx, wz) + mu.h, wz);
      var dd = Math.hypot(wx - camera.position.x, wz - camera.position.z);
      md.material.opacity = cur.mist * 0.32 * mu.base * smooth(6, 22, dd) * (1 - smooth(70, 100, dd));
      md.material.color.copy(cur.fog).lerp(cur.horizon, 0.5);
      md.visible = md.material.opacity > 0.003;
    }

    // ── Weather ──
    pf.wind = f.wind; pf.dt = dt; pf.time = f.time;
    pf.snow = f.snow;
    snowfall.update(pf, camera.position, slow);
    snowfall.points.visible = f.snow > 0.01;
    pf.snow = cur.wheatOn * (1 - cur.wet);
    motes.update(pf, camera.position, slow);
    motes.points.visible = pf.snow > 0.01;
    pf.snow = rainAmt * 0.8;
    leaves.update(pf, camera.position, slow);
    leaves.points.visible = pf.snow > 0.01;
    rain.update(f, camera.position, rainAmt, slow);
    rain.lines.visible = rainAmt > 0.01;
    pf.snow = cur.stars * (1 - smooth(3, 8, lift));
    look.set(camera.position.x, ground(camera.position.x, camera.position.z) + 0.6, camera.position.z);
    fireflies.update(pf, look, slow);
    fireflies.points.visible = pf.snow > 0.01;

    // ── Wind threads ──
    threadMat.uniforms.uClock.value = clock;
    threadMat.uniforms.uAmt.value = threadAmt;
    threadMat.uniforms.uAnchor.value.set(camera.position.x, ground(camera.position.x, camera.position.z), camera.position.z);
    threadMat.uniforms.uColor.value.copy(cur.hemiSky).lerp(cur.sun, 0.4).multiplyScalar(0.34);
    threads.visible = threadAmt > 0.01;

    // ── Birds ──
    birdClock.value = clock;
    updateBirds(f, flock, Math.min(dt, 0.04) * (slow ? 0.6 : 1));

    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) { H = h; portrait = w < h; fitCamera(gl, camera, w, h, dpr, small); },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('thousand-winds', {
  renderer: renderer3d,
  maxLines: 2,
  scrim: 0.62,
  align: ['left', 'right', 'left', 'right', 'left', 'left'],
  // Panels (couplets): 0 the grave, 1 winds | snow, 2 grain | rain,
  // 3 morning hush | the rush, 4 circled flight | stars, 5 the grave again.
  keys: function (T) {
    function at(i, d) { return T.start(i) + d; }   // d units into panel i (0..1.6)
    var H0 = P_HUSH;
    //   unit          path    lift  snow  wind  look  yaw    pitch  thr   rain  flock
    return [
      [0,              0.000,  1.6,  0.0,  0.20, 0.00,  0.00, -0.12, 0.00, 0.0, 0.0],
      [0.7,            0.000,  1.6,  0.0,  0.20, 0.00,  0.00, -0.12, 0.00, 0.0, 0.0],
      [at(0, 0.45),    0.000,  1.65, 0.0,  0.25, 0.00,  0.00, -0.13, 0.05, 0.0, 0.0],  // "Do not stand at my grave and weep"
      [at(0, 0.7),     0.000,  2.2,  0.0,  0.30, 0.00,  0.00, -0.22, 0.10, 0.0, 0.0],  // "I am not there": lifting
      [at(0, 1.15),    -0.003, 6.5,  0.0,  0.45, 0.10,  0.00, -0.70, 0.30, 0.0, 0.0],  // looking down on the stone
      [at(0, 1.5),     0.004,  8.5,  0.0,  0.60, 0.45,  0.00, -0.75, 0.60, 0.0, 0.0],  // passing over it
      [at(1, 0.2),     0.040,  3.4,  0.0,  1.00, 1.00,  0.00, -0.04, 1.00, 0.0, 0.0],  // "a thousand winds that blow"
      [at(1, 0.6),     0.100,  2.4,  0.0,  1.00, 1.00,  0.00, -0.03, 1.00, 0.0, 0.0],
      [at(1, 0.95),    0.140,  2.2,  0.7,  0.35, 2.00,  0.00, -0.07, 0.20, 0.0, 0.0],  // "the diamond glints on snow"
      [at(1, 1.3),     0.170,  2.0,  0.4,  0.15, 2.00,  0.00, -0.09, 0.00, 0.0, 0.0],
      [at(2, 0.15),    0.230,  2.1,  0.0,  0.30, 3.00,  0.00, -0.02, 0.00, 0.0, 0.0],  // "the sunlight on ripened grain"
      [at(2, 0.7),     0.280,  2.3,  0.0,  0.30, 3.00,  0.00, -0.02, 0.00, 0.0, 0.0],
      [at(2, 0.95),    0.320,  3.6,  0.0,  0.35, 4.00,  0.00, -0.04, 0.00, 0.8, 0.0],  // "the gentle autumn rain"
      [at(2, 1.4),     0.360,  4.5,  0.0,  0.35, 4.00,  0.00, -0.04, 0.00, 0.8, 0.0],
      [at(3, 0.2),     H0 - 0.02, 1.8, 0.0, 0.00, 5.00, 0.00, 0.00, 0.00, 0.0, 0.0],  // "the morning's hush"
      [at(3, 0.8),     H0 - 0.005, 1.8, 0.0, 0.00, 5.00, 0.00, 0.02, 0.00, 0.0, 0.0],
      [at(3, 0.95),    H0,     1.8,  0.0,  0.05, 5.00,  0.00,  0.10, 0.00, 0.0, 0.6],  // "the swift uplifting rush"
      [at(3, 1.4),     H0 + 0.003, 2.0, 0.0, 0.05, 5.00, -0.05, 0.28, 0.00, 0.0, 1.0],
      [at(4, 0.45),    H0 + 0.006, 2.2, 0.0, 0.05, 5.00, -0.08, 0.34, 0.00, 0.0, 1.0], // "quiet birds in circled flight"
      [at(4, 0.9),     H0 + 0.015, 4.0, 0.0, 0.10, 5.60, -0.05, 0.45, 0.00, 0.0, 1.4],
      [at(4, 1.35),    H0 + 0.05, 12.0, 0.0, 0.10, 6.00, 0.00,  0.55, 0.00, 0.0, 2.0],  // "the soft stars that shine at night"
      [at(5, 0.05),    0.80,   26.0, 0.0,  0.15, 6.00,  0.00,  0.20, 0.00, 0.0, 2.0],  // gliding home under the stars
      [at(5, 0.5),     0.985,  3.5,  0.0,  0.15, 6.00,  0.00, -0.12, 0.00, 0.0, 2.0],  // "Do not stand at my grave and cry"
      [at(5, 0.8),     1.000,  1.6,  0.0,  0.15, 6.00,  0.00, -0.13, 0.05, 0.0, 2.0],
      [at(5, 1.25),    1.000,  1.6,  0.0,  0.20, 6.00,  0.00,  0.24, 0.00, 0.0, 2.0],  // "I did not die": up to the stars
      [T.total,        1.000,  1.6,  0.0,  0.20, 6.00,  0.00,  0.36, 0.00, 0.0, 2.0]
    ];
  },
  sound: {
    src: '/audio/wind.mp3',
    label: 'Play the wind',
    volume: function (row) { return 0.05 + 0.3 * row[3] * (1 - 0.4 * row[8]); },
    cues: [{ stanza: 3, at: 0.95, play: wings }]
  }
});
