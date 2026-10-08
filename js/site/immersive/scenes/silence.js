/*
 * Scene for "Silence" (Edgar Lee Masters): one night walk through the
 * silences, from the sea to the graves.
 *
 * I    "The silence of the stars and of the sea": a calm sea under a low moon.
 *      Then the town "when it pauses", its main street asleep, a few windows
 *      still lit (a man and a maid, a sick room). "We cannot speak": they go
 *      dark one by one as you walk up the hushed street.
 * II   The grocery-store porch: a lamp, an empty chair, a crutch against the
 *      wall. "His mind flies away": out to the edge of town, where "the
 *      flashes of guns, the thunder of cannon" flicker in smoke beyond the
 *      ridge; you sink to the ground ("himself lying on the ground"); the
 *      world dims for "the long days in bed", and the smoke drifts away.
 * III  The great silences, out on the plain: two far farmhouse lights;
 *      "visions not to be uttered" in the Milky Way and falling stars; a
 *      window goes dark for "the dying whose hand suddenly grips yours".
 * IV   Over the ridge, "the vast silence that covers broken nations and
 *      vanquished leaders": a plain of campfires dying out. "Jeanne d'Arc
 *      amid the flames": a single distant pyre, which flares and sinks.
 *      "The silence of age": an old oak on the ridge.
 * V    "The silence of the dead": the graveyard under the oak, which you
 *      approach as the mist comes up.
 *
 * Long stanzas are split six lines a panel (14 panels), so keys are a
 * function of the timeline. Columns:
 *   [unit, path, dim, smoke, wind, heading, pitch, eye, hush, lamp, battle, sky, farm, fires, pyre, mist, phone]
 * where "phone" turns the view on portrait screens, whose text sits in the
 * middle, to bring the subject held off to one side back to the centre.
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, softSprite, skyDome, terrain, ribbon,
         scatter, broadleafGeometry, oceanMaterial, oceanMesh, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout (metres; north is -z, the sea lies east, +x) ──────────────────
var MOON = new THREE.Vector3(0.95, 0.1, 0.3).normalize();
var CALM = [[1, 0.25, 0.14, 0.05, 0.5], [0.4, 1, 0.22, 0.035, 0.7], [-0.7, 0.8, 0.35, 0.02, 1.0]];
var STREET = 7;                                   // shopfronts stand at x = ±7
var TOWN_Y = 0.2;
var PYRE = new THREE.Vector3(-105, 0, -520);
var OAK = new THREE.Vector3(51, 0, -318);
var YARD = { x0: 30, x1: 58, z0: -302, z1: -328, gate: -313.5 };
var FARM_A = new THREE.Vector3(-46, 0, -250), FARM_B = new THREE.Vector3(78, 0, -268);
var LAMP = new THREE.Vector3(-5.35, 0, -28.3);   // the porch lamp (y set from the awning)
// Waypoints of the walk; the "path" column counts them (2.0 = the third).
var PATH = [[3, 21], [2, 13], [0.5, 0.0], [-0.3, -15], [-0.8, -28.1], [0, -40], [0, -54], [0, -64], [-1, -100],
            [-3, -150], [-2, -200], [1, -248], [4, -318], [14, -313], [23, -311], [29, -313.5], [35.5, -315], [40, -316]];

function coast(z) { return 12 + Math.max(0, 12 - z) * 2; }

function land(x, z) {
  var plain = smooth(-55, -110, z);
  var roll = 1.5 * Math.sin(x * 0.012 + 0.7) * Math.cos(z * 0.009) + 0.6 * Math.sin(x * 0.027 - z * 0.019);
  var ridge = 26 * Math.exp(-Math.pow((z + 300) / 75, 2));
  var basin = -5 * smooth(-340, -460, z) * (1 - smooth(-800, -1100, z));
  var far = smooth(-750, -1350, z) * (40 + 18 * Math.sin(x * 0.005 + 1.3));
  var h = Math.max(TOWN_Y, plain * (3 + roll + ridge + basin + far));
  var c = coast(z);
  return lerp(h, -5, smooth(c - 4, c + 16, x));
}

// ── Textures and small pieces ────────────────────────────────────────────
function canvasTexture(w, h, paint) {
  var c = document.createElement('canvas');
  c.width = w; c.height = h;
  paint(c.getContext('2d'), w, h);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

// A billowing puff for smoke and mist: soft blobs heaped together.
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

function windowTexture() {
  return canvasTexture(64, 96, function (x, w, h) {
    x.fillStyle = '#100e0c';
    x.fillRect(0, 0, w, h);
    var g = x.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, '#cfcfcf');
    g.addColorStop(1, '#ffffff');
    x.fillStyle = g;
    x.fillRect(5, 5, w - 10, h - 10);
    x.fillStyle = '#100e0c';
    x.fillRect(w / 2 - 2, 5, 4, h - 10);
    x.fillRect(5, h / 2 - 3, w - 10, 6);
    x.fillRect(5, h * 0.25 - 1, w - 10, 2);
    x.fillRect(5, h * 0.75 - 1, w - 10, 2);
  });
}

function signTexture() {
  return canvasTexture(512, 112, function (x, w, h) {
    x.fillStyle = '#1d2a22';
    x.fillRect(0, 0, w, h);
    x.strokeStyle = '#b9ae8e';
    x.lineWidth = 4;
    x.strokeRect(8, 8, w - 16, h - 16);
    x.fillStyle = '#d8ccaa';
    x.font = 'bold 64px Georgia, serif';
    x.textAlign = 'center';
    x.textBaseline = 'middle';
    x.fillText('G R O C E R Y', w / 2, h / 2 + 3);
  });
}

function stoneTexture(r) {
  var t = canvasTexture(256, 128, function (x, w, h) {
    x.fillStyle = '#3a3a3e';
    x.fillRect(0, 0, w, h);
    for (var row = 0; row < 5; row++) {
      for (var cx = -20; cx < w + 20;) {
        var sw = 26 + r() * 30, l = 110 + r() * 60;
        x.fillStyle = 'rgb(' + Math.round(l) + ',' + Math.round(l * 0.98) + ',' + Math.round(l * 0.92) + ')';
        x.beginPath();
        x.ellipse(cx + sw / 2, row * 26 + 13 + (r() - 0.5) * 4, sw / 2 - 2, 11 + r() * 2, (r() - 0.5) * 0.2, 0, Math.PI * 2);
        x.fill();
        cx += sw;
      }
    }
  });
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

// A tall flame for the pyre.
function flameTexture() {
  return canvasTexture(64, 128, function (x, w, h) {
    var g = x.createRadialGradient(32, 100, 2, 32, 84, 60);
    g.addColorStop(0, 'rgba(255,248,220,1)');
    g.addColorStop(0.25, 'rgba(255,190,90,0.9)');
    g.addColorStop(0.6, 'rgba(230,80,20,0.45)');
    g.addColorStop(1, 'rgba(120,20,0,0)');
    x.fillStyle = g;
    x.beginPath();
    x.moveTo(32, 2);
    x.bezierCurveTo(46, 50, 62, 80, 52, 112);
    x.bezierCurveTo(44, 128, 20, 128, 12, 112);
    x.bezierCurveTo(2, 80, 18, 50, 32, 2);
    x.fill();
  });
}

// A gable roof: the triangle spans `w` along z, the ridge runs `d` along x.
function gable(w, rise, d) {
  var s = new THREE.Shape();
  s.moveTo(-w / 2, 0); s.lineTo(w / 2, 0); s.lineTo(0, rise); s.lineTo(-w / 2, 0);
  return new THREE.ExtrudeGeometry(s, { depth: d, bevelEnabled: false }).translate(0, 0, -d / 2).rotateY(Math.PI / 2);
}

function box(w, h, d, x, y, z, color) {
  return tinted(new THREE.BoxGeometry(w, h, d).translate(x, y, z), color);
}

// ── Sound cues ───────────────────────────────────────────────────────────
// Far-off cannon: a low thump with a long rolling tail.
function cannon(ac, out) {
  var t = ac.currentTime, len = 3.2, b = ac.createBuffer(1, Math.floor(ac.sampleRate * len), ac.sampleRate), d = b.getChannelData(0);
  for (var i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / d.length, 2.2);
  var src = ac.createBufferSource(), lp = ac.createBiquadFilter(), g = ac.createGain();
  src.buffer = b;
  lp.type = 'lowpass';
  lp.frequency.setValueAtTime(420, t);
  lp.frequency.exponentialRampToValueAtTime(90, t + 1.5);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.45, t + 0.04);
  g.gain.exponentialRampToValueAtTime(0.0001, t + len);
  src.connect(lp); lp.connect(g); g.connect(out);
  src.start(t);
}

// One distant toll of a church bell: inharmonic partials, long decay.
function toll(ac, out) {
  var t = ac.currentTime;
  [[0.5, 0.05, 7], [1, 0.07, 5], [1.19, 0.03, 4], [1.5, 0.025, 3.5], [2, 0.02, 3], [2.51, 0.012, 2.4], [3.01, 0.008, 2]].forEach(function (p) {
    var o = ac.createOscillator(), g = ac.createGain();
    o.type = 'sine';
    o.frequency.value = 196 * p[0];
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(p[1], t + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, t + p[2]);
    o.connect(g); g.connect(out);
    o.start(t); o.stop(t + p[2] + 0.1);
  });
}

// ── Sky: twinkling stars, the Milky Way and falling stars ────────────────
var STAR_VS = 'attribute vec3 star; uniform float uTime; uniform float uBoost; uniform float uScale;\n' +
  'varying float vA; varying float vWarm;\n' +
  'void main(){ gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);\n' +
  ' float tw = 0.7 + 0.3 * sin(uTime * (1.3 + star.y * 2.5) + star.y * 40.0);\n' +
  ' vA = tw * (0.75 + 0.5 * uBoost) * smoothstep(-0.02, 0.08, normalize(position).y); vWarm = star.z;\n' +
  ' gl_PointSize = star.x * uScale * (1.0 + 0.35 * uBoost); }';
var STAR_FS = 'varying float vA; varying float vWarm;\n' +
  'void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;\n' +
  ' float a = smoothstep(0.5, 0.0, d) * vA;\n' +
  ' vec3 col = mix(vec3(0.78, 0.85, 1.0), vec3(1.0, 0.86, 0.7), vWarm);\n' +
  ' gl_FragColor = vec4(col * a, a);\n #include <colorspace_fragment>\n }';

var BAND_FS = 'uniform vec3 uN; uniform float uAmt; varying vec3 vP;\n' +
  'float hash(vec3 p){ p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }\n' +
  'float noise(vec3 x){ vec3 i = floor(x); vec3 f = fract(x); f = f * f * (3.0 - 2.0 * f);\n' +
  ' return mix(mix(mix(hash(i), hash(i + vec3(1,0,0)), f.x), mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),\n' +
  '            mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x), mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z); }\n' +
  'void main(){ vec3 d = normalize(vP); float b = dot(d, uN);\n' +
  ' float n = noise(d * 7.0) * 0.55 + noise(d * 15.0) * 0.3 + noise(d * 31.0) * 0.15;\n' +
  ' float band = exp(-b * b / 0.006) * (0.15 + 1.3 * n * n) - exp(-b * b / 0.0008) * 0.5 * n;\n' +
  ' float a = max(band, 0.0) * smoothstep(0.0, 0.2, d.y) * uAmt;\n' +
  ' gl_FragColor = vec4(vec3(0.42, 0.47, 0.66) * a, 1.0); }';

// Campfires: each flickers and, as `uDie` passes its moment, sinks to an
// ember and goes out.
var FIRE_VS = 'attribute vec3 fire; uniform float uTime; uniform float uDie; uniform float uScale; uniform float uAlpha;\n' +
  'varying float vA; varying float vHot;\n' +
  'void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;\n' +
  ' float alive = 1.0 - smoothstep(fire.z, fire.z + 0.05, uDie);\n' +
  ' float ember = (1.0 - alive) * (1.0 - smoothstep(fire.z + 0.05, fire.z + 0.2, uDie));\n' +
  ' float fl = 0.72 + 0.28 * sin(uTime * (4.0 + fire.y * 3.0) + fire.y * 20.0) * sin(uTime * 2.3 + fire.y * 9.0);\n' +
  ' vA = (alive * fl + ember * 0.4) * uAlpha; vHot = alive;\n' +
  ' gl_PointSize = clamp(fire.x * uScale / -mv.z, 1.6, 14.0) * (0.55 + 0.45 * alive); }';
var FIRE_FS = 'varying float vA; varying float vHot;\n' +
  'void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;\n' +
  ' float a = smoothstep(0.5, 0.0, d); float core = smoothstep(0.2, 0.0, d);\n' +
  ' vec3 col = mix(vec3(0.9, 0.18, 0.04), vec3(1.0, 0.5, 0.18), vHot) * a * a + vec3(1.0, 0.82, 0.55) * core * vHot;\n' +
  ' gl_FragColor = vec4(col * vA, 1.0);\n #include <colorspace_fragment>\n }';

// Musket fire along the ridge: each point flashes at random moments.
var VOLLEY_VS = 'attribute float seed; uniform float uTime; uniform float uRate;\n' +
  'varying float vA;\n' +
  'void main(){ gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);\n' +
  ' float k = uTime * 7.0 + seed * 37.0; float s = floor(k);\n' +
  ' float h = fract(sin(s * 12.9898 + seed * 78.233) * 43758.5453);\n' +
  ' vA = step(1.0 - uRate * 0.07, h) * (1.0 - fract(k));\n' +
  ' gl_PointSize = 1.5 + 2.5 * vA; }';
var VOLLEY_FS = 'varying float vA;\n' +
  'void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5 || vA < 0.01) discard;\n' +
  ' gl_FragColor = vec4(vec3(1.0, 0.85, 0.6) * vA * smoothstep(0.5, 0.0, d), 1.0); }';

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(1915);
  var gl = makeRenderer(canvas, { clear: '#0a0f20' });
  var world = new THREE.Scene();
  var FOG = new THREE.Color('#0b1124');
  world.fog = new THREE.FogExp2(FOG.clone(), 0.003);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.08, 4000);
  var pxScale = 400, portrait = false;

  // ── Sky ────────────────────────────────────────────────────────────────
  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#02040b', mid: '#08102a', horizon: '#26335a', sun: '#5a6890' }, 1500);
  dome.uniforms.sunDir.value.copy(MOON);
  sky.add(dome.mesh);
  var BAND_N = new THREE.Vector3(-0.09, -0.83, -0.56).normalize();
  var bandMat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false, transparent: true, blending: THREE.AdditiveBlending,
    uniforms: { uN: { value: BAND_N }, uAmt: { value: 0.5 } },
    vertexShader: 'varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: BAND_FS
  });
  sky.add(new THREE.Mesh(new THREE.SphereGeometry(1400, 48, 24), bandMat));

  var SN = small ? 3500 : 7000, MW = small ? 2500 : 6000, sPos = [], sAttr = [], v = new THREE.Vector3();
  var bandU = new THREE.Vector3().crossVectors(BAND_N, new THREE.Vector3(0, 1, 0)).normalize();
  var bandV = new THREE.Vector3().crossVectors(BAND_N, bandU).normalize();
  for (var i = 0; i < SN + MW; i++) {
    if (i < SN) {
      var th = r() * Math.PI * 2, y = -0.05 + r() * 1.05, s = Math.sqrt(1 - y * y);
      v.set(s * Math.cos(th), y, s * Math.sin(th));
    } else {
      var a = r() * Math.PI * 2, off = (r() + r() + r() - 1.5) * 0.12;
      v.copy(bandU).multiplyScalar(Math.cos(a)).addScaledVector(bandV, Math.sin(a)).addScaledVector(BAND_N, off).normalize();
      if (v.y < -0.05) continue;
    }
    sPos.push(v.x * 1300, v.y * 1300, v.z * 1300);
    var bright = Math.pow(r(), i < SN ? 4 : 7);
    sAttr.push(0.9 + bright * 3.4, r(), r() < 0.25 ? r() : 0);
  }
  var starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.Float32BufferAttribute(sPos, 3));
  starGeo.setAttribute('star', new THREE.Float32BufferAttribute(sAttr, 3));
  var starMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
    uniforms: { uTime: { value: 0 }, uBoost: { value: 0 }, uScale: { value: 1 } },
    vertexShader: STAR_VS, fragmentShader: STAR_FS
  });
  var stars = new THREE.Points(starGeo, starMat);
  stars.frustumCulled = false;
  sky.add(stars);

  // The moon, low over the sea, and its glow.
  var moonDisc = new THREE.Mesh(new THREE.CircleGeometry(13, 48), new THREE.MeshBasicMaterial({ color: '#e6e9f6', fog: false }));
  moonDisc.position.copy(MOON).multiplyScalar(1100);
  moonDisc.lookAt(0, 0, 0);
  var moonGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(190,205,255,0.4)', 'rgba(120,140,220,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
  moonGlow.position.copy(MOON).multiplyScalar(1090);
  moonGlow.scale.setScalar(100);
  sky.add(moonDisc, moonGlow);

  // Falling stars: a pool of thin streaks, each a quad fading to its tail.
  var METEORS = 3, metPos = new Float32Array(METEORS * 12), metCol = new Float32Array(METEORS * 16), metIdx = [];
  for (i = 0; i < METEORS; i++) metIdx.push(i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 2, i * 4 + 1, i * 4 + 3);
  var metGeo = new THREE.BufferGeometry();
  metGeo.setAttribute('position', new THREE.BufferAttribute(metPos, 3));
  metGeo.setAttribute('color', new THREE.BufferAttribute(metCol, 4));
  metGeo.setIndex(metIdx);
  var meteorMesh = new THREE.Mesh(metGeo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, fog: false, side: THREE.DoubleSide }));
  meteorMesh.frustumCulled = false;
  sky.add(meteorMesh);
  var meteors = [];
  for (i = 0; i < METEORS; i++) meteors.push({ age: 99, life: 1, from: new THREE.Vector3(), dir: new THREE.Vector3() });
  var meteorClock = 0.5;

  // ── Light ──────────────────────────────────────────────────────────────
  var hemi = new THREE.HemisphereLight('#4a5888', '#141210', 1.1);
  var moonLight = new THREE.DirectionalLight('#aebce6', 1.4);
  world.add(hemi, moonLight, moonLight.target);

  // ── Land, sea and the road ─────────────────────────────────────────────
  var gcol = new THREE.Color(), dirt = new THREE.Color('#2b2722'), sand = new THREE.Color('#57524a'), grassA = new THREE.Color();
  world.add(terrain(2600, small ? 200 : 300, 0, -520, land, new THREE.MeshLambertMaterial({ vertexColors: true }), function (x, z, y) {
    var n = 0.5 + 0.5 * Math.sin(x * 0.11 + Math.sin(z * 0.07) * 2) * Math.sin(z * 0.09 - x * 0.03);
    grassA.setHSL(0.2 + n * 0.05, 0.26, 0.11 + n * 0.05);
    gcol.copy(dirt).lerp(grassA, smooth(-52, -75, z) + smooth(10, 30, Math.abs(x)) * (1 - smooth(-52, -75, z)));
    return gcol.lerp(sand, smooth(coast(z) - 7, coast(z) + 1, x));
  }));

  var seaMat = oceanMaterial({ color: '#050912', specular: '#dfe6ff', shininess: 150, waves: CALM, sky: '#18213e' });
  var sea = oceanMesh(seaMat, 1600, small ? 160 : 240);
  sea.position.y = -0.5;
  world.add(sea);
  var su = seaMat.userData.uniforms;

  var path = new THREE.CatmullRomCurve3(PATH.map(function (p) { return new THREE.Vector3(p[0], 0, p[1]); }));
  var roadCurve = new THREE.CatmullRomCurve3(PATH.slice(5, 13).map(function (p) { return new THREE.Vector3(p[0], 0, p[1]); }));
  var footCurve = new THREE.CatmullRomCurve3(PATH.slice(12, 16).map(function (p) { return new THREE.Vector3(p[0], 0, p[1]); }));
  var roadPts = roadCurve.getSpacedPoints(160);
  var roadMat = new THREE.MeshLambertMaterial({ color: '#2e2a24' });
  world.add(new THREE.Mesh(ribbon(roadPts, 0, 4.2, land, 0.05), roadMat));
  world.add(new THREE.Mesh(ribbon(footCurve.getSpacedPoints(40), 0, 1.2, land, 0.05), roadMat));
  function roadX(z) {                              // where the road is at a given z
    var best = 0, bd = 1e9;
    for (var k = 0; k < roadPts.length; k += 2) { var dz = Math.abs(roadPts[k].z - z); if (dz < bd) { bd = dz; best = roadPts[k].x; } }
    return best;
  }

  // The pier, out into the moon's glitter.
  var pierParts = [box(31, 0.22, 2.2, 22.5, 0.55, 24, '#2a241e')];
  for (var px = 9; px < 38; px += 3) {
    pierParts.push(box(0.22, 4, 0.22, px, -1.4, 22.95, '#1c1814'), box(0.22, 4, 0.22, px, -1.4, 25.05, '#1c1814'));
  }
  pierParts.push(box(0.3, 0.5, 0.3, 37.6, 0.85, 23.1, '#1c1814'), box(0.3, 0.5, 0.3, 37.6, 0.85, 24.9, '#1c1814'));
  world.add(new THREE.Mesh(merge(pierParts), new THREE.MeshLambertMaterial({ vertexColors: true })));

  // ── The town ───────────────────────────────────────────────────────────
  // Clapboard: board lines and grain painted in world space on any wall.
  var sidingMat = new THREE.MeshLambertMaterial({ vertexColors: true });
  sidingMat.onBeforeCompile = function (sh) {
    sh.vertexShader = 'varying vec3 vWPos; varying vec3 vWN;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n vWPos = (modelMatrix * vec4(transformed, 1.0)).xyz; vWN = normalize(mat3(modelMatrix) * objectNormal);');
    sh.fragmentShader = 'varying vec3 vWPos; varying vec3 vWN;\n' + sh.fragmentShader.replace('#include <color_fragment>',
      '#include <color_fragment>\n float vert = 1.0 - smoothstep(0.15, 0.45, abs(vWN.y));\n' +
      ' float fy = fract(vWPos.y * 4.2), row = floor(vWPos.y * 4.2);\n' +
      ' float seam = smoothstep(0.0, 0.1, fy) * smoothstep(1.0, 0.85, fy);\n' +
      ' float grain = fract(sin(row * 91.7 + floor((vWPos.x + vWPos.z) * 0.7) * 13.1) * 4375.5);\n' +
      ' diffuseColor.rgb *= mix(1.0, (0.72 + 0.28 * seam) * (0.86 + 0.18 * grain), vert);');
  };

  // [side, z0, z1, depth, height, false-front rise, colour, kind]
  var SHOPS = [
    [-1, 13, 4, 9, 6.4, 0, '#686670', 'house'],
    [-1, 2, -9, 12, 7.0, 1.6, '#5e4c40', 'shop'],
    [-1, -11, -18.5, 10, 5.0, 1.5, '#78705f', 'shop'],
    [-1, -21, -33, 14, 5.2, 2.0, '#4f5e52', 'grocery'],
    [-1, -36, -44, 10, 4.6, 1.4, '#6a5a52', 'shop'],
    [-1, -47, -56, 9, 6.0, 0, '#6c6a74', 'house'],
    [1, 4, -3, 10, 5.0, 1.6, '#6e6256', 'shop'],
    [1, -5.5, -17, 14, 8.0, 1.0, '#4e5560', 'shop'],
    [1, -19.5, -27.5, 10, 4.8, 1.8, '#7c6e5a', 'shop'],
    [1, -30, -41, 13, 6.6, 0, '#5a4c44', 'barn'],
    [1, -44, -53, 9, 6.0, 0, '#68666e', 'house']
  ];
  var townParts = [], wins = [], WOOD = '#3c3229', DARK = '#15120f';
  function win(side, z, y, w, h, lit) { wins.push({ x: side * (STREET - 0.04), y: TOWN_Y + y, z: z, w: w, h: h, side: side, lit: lit || 0 }); }
  SHOPS.forEach(function (s, n) {
    var side = s[0], z0 = s[1], z1 = s[2], d = s[3], h = s[4], ff = s[5], col = s[6], kind = s[7];
    var w = z0 - z1, zc = (z0 + z1) / 2, xc = side * (STREET + d / 2), xf = side * STREET, b = TOWN_Y;
    townParts.push(box(d, h, w, xc, b + h / 2, zc, col));
    if (kind === 'house' || kind === 'barn') {
      townParts.push(tinted(gable(w + 0.7, kind === 'barn' ? 3.4 : 2.8, d + 0.7).translate(xc, b + h, zc), '#26252a'));
    } else {
      townParts.push(tinted(gable(w * 0.96, 1.5, d - 0.6).translate(xc + side * 0.3, b + h, zc), '#26252a'));
      townParts.push(box(0.3, ff + 0.8, w, xf + side * 0.15, b + h + ff / 2 - 0.4, zc, col));
      townParts.push(box(0.55, 0.28, w + 0.3, xf + side * 0.1, b + h + ff + 0.1, zc, '#2a2622'));
      townParts.push(box(0.4, 0.16, w + 0.1, xf - side * 0.05, b + h - 1.2, zc, '#2a2622'));
    }
    if (kind === 'shop' || kind === 'grocery') {
      // Boardwalk, awning on posts, a door and two display windows.
      townParts.push(box(2.4, 0.26, w, xf - side * 1.2, b + 0.13, zc, WOOD));
      townParts.push(box(2.7, 0.12, w + 0.2, xf - side * 1.3, b + 3.15, zc, '#2a241e'));
      for (var pz = z0 - 0.3; pz >= z1 + 0.2; pz -= Math.max(2.8, (w - 0.5) / Math.round((w - 0.5) / 3.2))) {
        townParts.push(box(0.14, 2.9, 0.14, xf - side * 2.45, b + 1.6, pz, '#2e2620'));
      }
      townParts.push(box(0.08, 2.3, 1.1, xf - side * 0.03, b + 1.4, zc, DARK));
      win(side, zc + w * 0.27, 1.55, w * 0.32, 1.9, 0);
      win(side, zc - w * 0.27, 1.55, w * 0.32, 1.9, 0);
      if (h >= 6.5) for (var k = 0; k < 3; k++) win(side, zc + (k - 1) * w * 0.3, h - 1.9, 0.9, 1.4, 0);
    } else if (kind === 'house') {
      townParts.push(box(1.6, 0.4, 2.2, xf - side * 0.8, b + 0.2, zc, WOOD));
      townParts.push(box(0.08, 2.2, 1.0, xf - side * 0.03, b + 1.3, zc, DARK));
      win(side, zc + w * 0.28, 1.6, 0.9, 1.4, 0); win(side, zc - w * 0.28, 1.6, 0.9, 1.4, 0);
      win(side, zc + w * 0.28, 4.4, 0.9, 1.3, 0); win(side, zc - w * 0.28, 4.4, 0.9, 1.3, 0);
      win(side, zc, h + 1.0, 0.6, 0.8, 0);
    } else if (kind === 'barn') {
      townParts.push(box(0.1, 3.6, 3.4, xf - side * 0.04, b + 1.8, zc, DARK));
      townParts.push(box(0.1, 1.4, 1.4, xf - side * 0.04, b + h - 0.6, zc, DARK));
    }
  });
  // The church at the end of town: white boards, a steeple, pointed windows.
  var CH = { x: 15, z: -66 };
  townParts.push(box(14, 6, 9, CH.x + 1, TOWN_Y + 3, CH.z, '#a9aba6'));
  townParts.push(tinted(gable(9.7, 4, 14.6).translate(CH.x + 1, TOWN_Y + 6, CH.z), '#2e2d33'));
  townParts.push(box(3.2, 12, 3.2, CH.x - 6.4, TOWN_Y + 6, CH.z, '#b1b3ae'));
  townParts.push(box(3.6, 0.3, 3.6, CH.x - 6.4, TOWN_Y + 12.1, CH.z, '#38363a'));
  townParts.push(box(2.6, 3, 2.6, CH.x - 6.4, TOWN_Y + 13.7, CH.z, '#b1b3ae'));
  townParts.push(tinted(new THREE.ConeGeometry(2.1, 9, 4).rotateY(Math.PI / 4).translate(CH.x - 6.4, TOWN_Y + 19.7, CH.z), '#38363a'));
  townParts.push(box(0.1, 2.6, 1.4, CH.x - 8.05, TOWN_Y + 1.3, CH.z, DARK));
  [-4.6, -0.6, 3.4, 7.4].forEach(function (dx) {
    [-1, 1].forEach(function (sz) { townParts.push(box(0.9, 2.6, 0.1, CH.x + 1 + dx, TOWN_Y + 3, CH.z + sz * 4.55, '#1e2230')); });
  });
  // Hitching rails along the street.
  [[-1, -9], [-1, -35], [1, -24], [1, -1]].forEach(function (hr) {
    var hx = hr[0] * (STREET - 3.1);
    townParts.push(box(0.12, 1.0, 0.12, hx, TOWN_Y + 0.5, hr[1] + 1.4, WOOD), box(0.12, 1.0, 0.12, hx, TOWN_Y + 0.5, hr[1] - 1.4, WOOD),
                   box(0.09, 0.09, 3.0, hx, TOWN_Y + 0.95, hr[1], WOOD));
  });

  // The grocery porch: the lamp hangs from the awning; a chair stands empty
  // by the door with a crutch leant against the wall; barrels and a crate.
  var CHAIR = new THREE.Vector3(-6.25, TOWN_Y + 0.26, -28.5);
  var chairParts = [box(0.46, 0.05, 0.44, 0, 0.46, 0, '#5a4535')];
  [[-0.2, -0.19], [0.2, -0.19], [-0.2, 0.19], [0.2, 0.19]].forEach(function (p) { chairParts.push(box(0.04, 0.46, 0.04, p[0], 0.23, p[1], '#4a3829')); });
  chairParts.push(box(0.045, 0.62, 0.045, -0.2, 0.79, 0.19, '#4a3829'), box(0.045, 0.62, 0.045, 0.2, 0.79, 0.19, '#4a3829'));
  [0.66, 0.82, 0.98].forEach(function (y) { chairParts.push(box(0.4, 0.07, 0.025, 0, y, 0.19, '#5a4535')); });
  [0.12, 0.3].forEach(function (y) { chairParts.push(box(0.4, 0.025, 0.025, 0, y, -0.19, '#4a3829'), box(0.025, 0.025, 0.38, -0.2, y, 0, '#4a3829')); });
  var chairGeo = merge(chairParts).rotateY(-Math.PI / 2 - 0.35).translate(CHAIR.x, CHAIR.y, CHAIR.z);
  // The crutch: two rails meeting at the foot, a hand grip and an armpit pad.
  var crutch = [tinted(new THREE.CylinderGeometry(0.018, 0.022, 1.3, 6).translate(0, 0.65, 0).rotateZ(0.06).translate(-0.04, 0, 0), '#6a5038'),
                tinted(new THREE.CylinderGeometry(0.018, 0.022, 1.3, 6).translate(0, 0.65, 0).rotateZ(-0.06).translate(0.04, 0, 0), '#6a5038'),
                tinted(new THREE.CylinderGeometry(0.02, 0.02, 0.2, 6).rotateZ(Math.PI / 2).translate(0, 0.78, 0), '#6a5038'),
                tinted(new THREE.CylinderGeometry(0.035, 0.035, 0.22, 8).rotateZ(Math.PI / 2).translate(0, 1.31, 0), '#3a2e26'),
                tinted(new THREE.CylinderGeometry(0.016, 0.02, 0.3, 6).translate(0, -0.15, 0), '#2a221c')];
  var crutchGeo = merge(crutch).rotateX(0.22).rotateY(Math.PI / 2).translate(-6.62, TOWN_Y + 0.56, -29.35);
  var barrel = new THREE.CylinderGeometry(0.3, 0.27, 0.85, 12);
  var porchParts = [chairGeo, crutchGeo, tinted(barrel.clone().translate(-6.5, TOWN_Y + 0.68, -24.3), '#4a3a2c'),
                    tinted(barrel.clone().translate(-5.85, TOWN_Y + 0.68, -23.7), '#4a3a2c'),
                    box(0.6, 0.45, 0.6, -6.55, TOWN_Y + 0.48, -31.4, '#5a4a38'), box(0.5, 0.35, 0.5, -6.5, TOWN_Y + 0.88, -31.35, '#56463a')];
  townParts = townParts.concat(porchParts);
  world.add(new THREE.Mesh(merge(townParts), sidingMat));

  var sign = new THREE.Mesh(new THREE.PlaneGeometry(7.5, 1.4), new THREE.MeshLambertMaterial({ map: signTexture() }));
  sign.position.set(-STREET + 0.03, TOWN_Y + 4.35, -27);
  sign.rotation.y = Math.PI / 2;
  world.add(sign);

  LAMP.y = TOWN_Y + 2.62;
  var lantern = new THREE.Group();
  lantern.position.copy(LAMP);
  var lampGlass = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.24, 10), new THREE.MeshBasicMaterial({ color: '#ffd9a0' }));
  var lampCap = new THREE.Mesh(new THREE.ConeGeometry(0.15, 0.14, 10), new THREE.MeshLambertMaterial({ color: '#1e1a16' }));
  lampCap.position.y = 0.19;
  var lampWire = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.36, 4), lampCap.material);
  lampWire.position.y = 0.42;
  var lampGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,205,140,1)', 'rgba(255,150,80,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
  lampGlow.scale.setScalar(1.5);
  var lampLight = new THREE.PointLight('#ffb066', 14, 13, 1.5);
  lantern.add(lampGlass, lampCap, lampWire, lampGlow, lampLight);
  world.add(lantern);
  // Moths about the lamp.
  var MOTHS = 7, mothPos = new Float32Array(MOTHS * 3), mothGeo = new THREE.BufferGeometry();
  mothGeo.setAttribute('position', new THREE.BufferAttribute(mothPos, 3));
  var moths = new THREE.Points(mothGeo, new THREE.PointsMaterial({ color: '#ffe6c0', size: 0.035, transparent: true, depthWrite: false }));
  moths.frustumCulled = false;
  world.add(moths);

  // Two street lamps on posts.
  var glowTex = softSprite('rgba(255,200,130,1)', 'rgba(255,150,80,0)'), streetLamps = [];
  [[1, -9], [-1, -45.5]].forEach(function (sl) {
    var x = sl[0] * (STREET - 2.9), post = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.07, 3.4, 6), lampCap.material);
    post.position.set(x, TOWN_Y + 1.7, sl[1]);
    var head = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.36, 0.26), new THREE.MeshBasicMaterial({ color: '#ffd29a' }));
    head.position.set(x, TOWN_Y + 3.55, sl[1]);
    var g = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
    g.position.copy(head.position);
    g.scale.setScalar(2.2);
    var l = new THREE.PointLight('#ffb766', 9, 14, 1.6);
    l.position.copy(head.position);
    world.add(post, head, g, l);
    streetLamps.push({ glow: g, light: l, head: head });
  });

  // Windows: one instanced quad each. A few are lit and go out one by one.
  var winMesh = new THREE.InstancedMesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: windowTexture() }), wins.length);
  var litList = [], m4 = new THREE.Matrix4(), q4 = new THREE.Quaternion(), p4 = new THREE.Vector3(), s4 = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0);
  var darkGlass = new THREE.Color('#1c2335'), warm = new THREE.Color('#ffc27a'), tmpC = new THREE.Color();
  var winGlowMat = new THREE.SpriteMaterial({ map: glowTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true });
  wins.forEach(function (wd, n) {
    q4.setFromAxisAngle(up, wd.side < 0 ? Math.PI / 2 : -Math.PI / 2);
    winMesh.setMatrixAt(n, m4.compose(p4.set(wd.x, wd.y, wd.z), q4, s4.set(wd.w, wd.h, 1)));
    winMesh.setColorAt(n, darkGlass);
  });
  // Which windows are lit: upstairs of the tall shops, a house or two.
  // [building, window]; windows are counted per building in creation order.
  [[1, 3], [1, 4], [7, 3], [0, 3], [5, 2], [10, 3], [3, 1]].forEach(function (pick, k) {
    var found = -1, bStart = 0;
    SHOPS.forEach(function (s, bi) {
      var nw = s[7] === 'house' ? 5 : s[7] === 'barn' ? 0 : 2 + (s[4] >= 6.5 ? 3 : 0);
      if (bi === pick[0]) found = bStart + Math.min(pick[1], nw - 1);
      bStart += nw;
    });
    if (found < 0 || found >= wins.length) return;
    var glow = new THREE.Sprite(winGlowMat.clone());
    var wd = wins[found];
    glow.position.set(wd.x - wd.side * 0.3, wd.y, wd.z);
    glow.scale.setScalar(2.4);
    world.add(glow);
    // The first two (side by side) are the man and the maid; the third is the sick room.
    litList.push({ i: found, glow: glow, darkAt: [0.55, 0.6, 0.85, 0.15, 0.3, 0.45, 0.7][k], sick: k === 2, last: -1 });
  });
  world.add(winMesh);

  // Telegraph poles along the east side of the road, and their wires.
  var poleParts = [], wire = [], poleTops = [];
  for (var pzz = -4; pzz > -285; pzz -= 30) {
    var pxx = (pzz > -58 ? STREET - 2.2 : roadX(pzz) + 4.2), py = land(pxx, pzz);
    if (pzz < -58 && pzz > -76) continue;
    poleParts.push(box(0.2, 7.5, 0.2, pxx, py + 3.75, pzz, '#1c1814'), box(0.1, 0.12, 1.6, pxx, py + 7.0, pzz, '#1c1814'));
    poleTops.push(new THREE.Vector3(pxx, py + 7.1, pzz));
  }
  world.add(new THREE.Mesh(merge(poleParts), new THREE.MeshLambertMaterial({ vertexColors: true })));
  for (i = 0; i < poleTops.length - 1; i++) {
    [-0.65, 0.65].forEach(function (o) {
      var A = poleTops[i], B = poleTops[i + 1];
      for (var sgm = 0; sgm < 8; sgm++) {
        var t0 = sgm / 8, t1 = (sgm + 1) / 8;
        wire.push(lerp(A.x, B.x, t0), lerp(A.y, B.y, t0) - Math.sin(Math.PI * t0) * 0.7, lerp(A.z, B.z, t0) + o,
                  lerp(A.x, B.x, t1), lerp(A.y, B.y, t1) - Math.sin(Math.PI * t1) * 0.7, lerp(A.z, B.z, t1) + o);
      }
    });
  }
  var wireGeo = new THREE.BufferGeometry();
  wireGeo.setAttribute('position', new THREE.Float32BufferAttribute(wire, 3));
  world.add(new THREE.LineSegments(wireGeo, new THREE.LineBasicMaterial({ color: '#0c0d12', transparent: true, opacity: 0.8 })));

  // ── The plain: fences, grass, trees and two farms ──────────────────────
  var fenceParts = [];
  for (var fz = -72; fz > -272; fz -= 3.2) {
    if ((Math.floor(-fz / 3.2) % 11) === 4) continue;                // a gap here and there
    var fx = roadX(fz);
    [-1, 1].forEach(function (sd) {
      var x = fx + sd * 4.4, y = land(x, fz);
      fenceParts.push(box(0.14, 1.25, 0.14, x, y + 0.55, fz, '#2a241e'));
      fenceParts.push(box(0.07, 0.1, 3.3, x, y + 0.95, fz - 1.6, '#2e2822'), box(0.07, 0.1, 3.3, x, y + 0.5, fz - 1.6, '#2e2822'));
    });
  }
  world.add(new THREE.Mesh(merge(fenceParts), new THREE.MeshLambertMaterial({ vertexColors: true })));

  var clock = { value: 0 }, sway = { value: 0.3 };
  var grassMat = new THREE.MeshLambertMaterial({ vertexColors: true });
  grassMat.onBeforeCompile = function (sh) {
    sh.uniforms.uClock = clock;
    sh.uniforms.uSway = sway;
    sh.vertexShader = 'uniform float uClock; uniform float uSway;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n float gph = instanceMatrix[3][0] * 0.3 + instanceMatrix[3][2] * 0.2;\n' +
      ' transformed.x += sin(uClock * 1.3 + gph) * uSway * position.y * position.y;');
  };
  var tuft = merge([0, 1, 2].map(function (k) {
    return tinted(new THREE.ConeGeometry(0.035, 0.6, 3).translate(0, 0.3, 0).rotateZ((k - 1) * 0.3).rotateY(k * 2.1), '#ffffff');
  }));
  var grass = new THREE.InstancedMesh(tuft, grassMat, small ? 5000 : 14000);
  scatter(grass, 60000, function (n, p, q, s, c) {
    var z = -62 - r() * 240, x = roadX(z) + (r() - 0.5) * 60;
    if (Math.abs(x - roadX(z)) < 3) return false;
    if (r() < 0.3) { x = YARD.x0 - 6 + r() * 36; z = YARD.z0 + 6 - r() * 38; }
    p.set(x, land(x, z) - 0.05, z);
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar(0.45 + r() * 0.55);
    c.setHSL(0.18 + r() * 0.06, 0.28, 0.13 + r() * 0.08);
  });
  world.add(grass);

  var trees = new THREE.InstancedMesh(broadleafGeometry(r, '#16120e'), new THREE.MeshLambertMaterial({ vertexColors: true }), 46);
  scatter(trees, 400, function (n, p, q, s, c) {
    var z = -90 - r() * 260, x = (r() < 0.5 ? -1 : 1) * (22 + r() * 200);
    if (Math.hypot(x - YARD.x0 - 14, z - YARD.z0 + 13) < 30 || Math.hypot(x - FARM_A.x, z - FARM_A.z) < 14 ||
        Math.hypot(x - FARM_B.x, z - FARM_B.z) < 18) return false;
    p.set(x, land(x, z) - 0.2, z);
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar(1.1 + r() * 0.8);
    c.setHSL(0.25, 0.25, 0.07 + r() * 0.04);
  });
  world.add(trees);

  // Farmhouses: a house and a barn each, one window lit.
  var farmParts = [], farmLights = [];
  [FARM_A, FARM_B].forEach(function (F, k) {
    var y = land(F.x, F.z);
    farmParts.push(box(7, 4.6, 9, F.x, y + 2.3, F.z, k ? '#5a5a62' : '#6a6460'));
    farmParts.push(tinted(gable(9.6, 2.8, 7.6).rotateY(Math.PI / 2).translate(F.x, y + 4.6, F.z), '#232228'));
    farmParts.push(box(0.6, 2, 0.6, F.x + 1.5, y + 6, F.z - 2.5, '#2a2626'));
    var bx = F.x + (k ? 14 : -13), bz = F.z - 4, by = land(bx, bz);
    farmParts.push(box(10, 5.5, 8, bx, by + 2.75, bz, '#4a2c26'));
    farmParts.push(tinted(gable(8.8, 3.6, 10.6).translate(bx, by + 5.5, bz), '#26232a'));
    // The lit window faces the road.
    var wx = F.x + (k ? -3.52 : 3.52), wy = y + 1.7;
    var wm = new THREE.Mesh(new THREE.PlaneGeometry(1.0, 1.3), new THREE.MeshBasicMaterial({ map: winMesh.material.map, color: '#ffc27a' }));
    wm.position.set(wx, wy, F.z + 1.5);
    wm.rotation.y = k ? -Math.PI / 2 : Math.PI / 2;
    var g = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
    g.position.set(wx + (k ? -0.6 : 0.6), wy, F.z + 1.5);
    g.scale.setScalar(5);
    world.add(wm, g);
    farmLights.push({ win: wm, glow: g });
  });
  world.add(new THREE.Mesh(merge(farmParts), sidingMat));

  // ── The battle remembered: smoke beyond the ridge, lit by gunfire ──────
  var puff = puffTexture(r), batteries = [], smokes = [];
  for (i = 0; i < 16; i++) {
    var bxp = -260 + i * 34 + (r() - 0.5) * 20, onRidge = i % 3 === 0, bzp = onRidge ? -300 : -330 - r() * 80;
    var flash = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, blending: THREE.AdditiveBlending, depthWrite: false,
      transparent: true, opacity: 0, fog: false, color: '#ffe0b0' }));
    flash.position.set(bxp, land(bxp, bzp) + 1.5, bzp);
    flash.scale.setScalar(onRidge ? 14 : 20);
    world.add(flash);
    batteries.push({ pos: flash.position, sprite: flash, age: 9, wait: r() * 2, glow: 0 });
  }
  var smokeBase = new THREE.Color('#2c303c'), fireTint = new THREE.Color('#ff9a50');
  for (i = 0; i < (small ? 22 : 34); i++) {
    var sx = -320 + r() * 640, sz = i < 8 ? -90 - r() * 120 : -300 - r() * 120, near = i < 8;
    var sm = new THREE.Sprite(new THREE.SpriteMaterial({ map: puff, transparent: true, depthWrite: false, opacity: 0, fog: !near ? false : true }));
    if (near) sx = (r() - 0.5) * 120;
    sm.position.set(sx, land(sx, sz) + (near ? 2 + r() * 4 : 10 + r() * 38), sz);
    var sc = near ? 18 + r() * 16 : 45 + r() * 50;
    sm.scale.set(sc * 1.4, sc, 1);
    sm.userData = { base: sm.position.clone(), near: near, ph: r() * 6.28, w: [] };
    batteries.forEach(function (bt) {
      var dd = sm.position.distanceTo(bt.pos);
      sm.userData.w.push(1 / (1 + dd * dd / 1200));
    });
    world.add(sm);
    smokes.push(sm);
  }
  var vol = [], volSeed = [];
  for (i = 0; i < (small ? 160 : 320); i++) {
    var vx = -300 + r() * 600, vz = -296 - r() * 8;
    vol.push(vx, land(vx, vz) + 1.2, vz);
    volSeed.push(r());
  }
  var volGeo = new THREE.BufferGeometry();
  volGeo.setAttribute('position', new THREE.Float32BufferAttribute(vol, 3));
  volGeo.setAttribute('seed', new THREE.Float32BufferAttribute(volSeed, 1));
  var volMat = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
    uniforms: { uTime: { value: 0 }, uRate: { value: 0 } }, vertexShader: VOLLEY_VS, fragmentShader: VOLLEY_FS });
  var volley = new THREE.Points(volGeo, volMat);
  volley.frustumCulled = false;
  world.add(volley);

  // ── Beyond the ridge: campfires of a beaten army, and the pyre ─────────
  var fPos = [], fAttr = [], centres = [];
  for (i = 0; i < 34; i++) centres.push([-420 + r() * 760, -350 - r() * 380, 25 + r() * 45]);
  for (i = 0; i < (small ? 420 : 800); i++) {
    var cc = centres[i % centres.length], ga = r() * 6.28, gd = Math.sqrt(-2 * Math.log(1 - r() * 0.999)) * cc[2] * 0.5;
    var fxx = cc[0] + Math.cos(ga) * gd, fzz = cc[1] + Math.sin(ga) * gd;
    if (Math.hypot(fxx - PYRE.x, fzz - PYRE.z) < 40) continue;
    fPos.push(fxx, land(fxx, fzz) + 1.4, fzz);
    fAttr.push(4 + r() * 4, r(), 0.05 + r() * 0.85);
  }
  var fireGeo = new THREE.BufferGeometry();
  fireGeo.setAttribute('position', new THREE.Float32BufferAttribute(fPos, 3));
  fireGeo.setAttribute('fire', new THREE.Float32BufferAttribute(fAttr, 3));
  var fireMat = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
    uniforms: { uTime: { value: 0 }, uDie: { value: 0 }, uScale: { value: 400 }, uAlpha: { value: 0 } },
    vertexShader: FIRE_VS, fragmentShader: FIRE_FS });
  var campfires = new THREE.Points(fireGeo, fireMat);
  campfires.frustumCulled = false;
  world.add(campfires);
  // A low haze over the camps, warm from below.
  var haze = [];
  for (i = 0; i < 14; i++) {
    var hx = -500 + r() * 1000, hz = -460 - r() * 560;
    var hs = new THREE.Sprite(new THREE.SpriteMaterial({ map: puff, color: '#c06a3a', transparent: true, depthWrite: false,
      opacity: 0, fog: false, blending: THREE.AdditiveBlending }));
    hs.position.set(hx, land(hx, hz) + 14, hz);
    hs.scale.set(260, 60, 1);
    world.add(hs);
    haze.push(hs);
  }

  PYRE.y = land(PYRE.x, PYRE.z);
  var flameTex = flameTexture(), flames = [];
  for (i = 0; i < 6; i++) {
    var fl = new THREE.Sprite(new THREE.SpriteMaterial({ map: flameTex, blending: THREE.AdditiveBlending, depthWrite: false,
      transparent: true, fog: false, opacity: 0 }));
    fl.userData = { ph: r() * 6.28, dx: (r() - 0.5) * 4, h: 14 + r() * 10 };
    world.add(fl);
    flames.push(fl);
  }
  var pyreGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, blending: THREE.AdditiveBlending, depthWrite: false,
    transparent: true, fog: false, opacity: 0, color: '#ff8a40' }));
  pyreGlow.position.set(PYRE.x, PYRE.y + 10, PYRE.z);
  pyreGlow.scale.setScalar(150);
  var pyrePost = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.4, 9, 5), new THREE.MeshBasicMaterial({ color: '#050302', fog: false }));
  pyrePost.position.set(PYRE.x, PYRE.y + 4.5, PYRE.z);
  world.add(pyreGlow, pyrePost);
  var SPARKS = small ? 90 : 180, spPos = new Float32Array(SPARKS * 3), spVel = new Float32Array(SPARKS * 3), spLife = new Float32Array(SPARKS);
  var spGeo = new THREE.BufferGeometry();
  spGeo.setAttribute('position', new THREE.BufferAttribute(spPos, 3));
  var sparks = new THREE.Points(spGeo, new THREE.PointsMaterial({ color: '#ffb066', size: 2.2, sizeAttenuation: false, transparent: true,
    depthWrite: false, blending: THREE.AdditiveBlending, fog: false }));
  sparks.frustumCulled = false;
  world.add(sparks);
  for (i = 0; i < SPARKS; i++) spPos[i * 3 + 1] = -999;
  var sparkClock = 0;

  // ── The old oak and the graveyard ──────────────────────────────────────
  OAK.y = land(OAK.x, OAK.z) - 0.3;
  var limbs = [], axis = new THREE.Vector3();
  (function grow(start, dir, len, rad, level) {
    var p = start.clone(), d = dir.clone();
    for (var sg = 0; sg < 3; sg++) {
      limbs.push({ start: p.clone(), dir: d.clone(), len: len / 3 * 1.06, rad: rad * (1 - sg * 0.1) });
      p.addScaledVector(d, len / 3);
      d.x += (r() - 0.5) * 0.45; d.y += (r() - 0.5) * 0.25 - level * 0.035; d.z += (r() - 0.5) * 0.45;
      d.normalize();
    }
    if (level >= (small ? 5 : 6)) return;
    var kids = level === 0 ? 4 : level < 3 ? 3 : 2, a0 = r() * 6.28;
    for (var kk = 0; kk < kids; kk++) {
      var ang = a0 + kk / kids * 6.28 + (r() - 0.5) * 0.6;
      axis.set(Math.cos(ang), 0.15, Math.sin(ang)).normalize();
      var nd = d.clone().lerp(axis, level === 0 ? 0.75 : 0.35 + r() * 0.3).normalize();
      grow(p, nd, len * (level === 0 ? 0.85 : 0.64 + r() * 0.12), rad * (level === 0 ? 0.52 : 0.6), level + 1);
    }
  })(new THREE.Vector3(), new THREE.Vector3(0.05, 1, 0).normalize(), 5.5, 0.95, 0);
  var limbMesh = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.88, 1, 1, 7).translate(0, 0.5, 0),
    new THREE.MeshLambertMaterial({ color: '#1a1712' }), limbs.length + 1);
  limbs.forEach(function (lb, n) {
    q4.setFromUnitVectors(up, lb.dir);
    limbMesh.setMatrixAt(n, m4.compose(p4.copy(lb.start).add(OAK), q4, s4.set(lb.rad, lb.len, lb.rad)));
  });
  q4.identity();
  limbMesh.setMatrixAt(limbs.length, m4.compose(p4.copy(OAK).setY(OAK.y - 0.2), q4, s4.set(1.6, 1.2, 1.6)));   // root flare
  world.add(limbMesh);

  // Headstones: round-topped tablets, crosses and a few obelisks.
  function tabletShape(w, h) {
    var sh = new THREE.Shape();
    sh.moveTo(-w / 2, 0); sh.lineTo(-w / 2, h - w / 2); sh.absarc(0, h - w / 2, w / 2, Math.PI, 0, true); sh.lineTo(w / 2, 0); sh.lineTo(-w / 2, 0);
    return sh;
  }
  function crossShape() {
    var sh = new THREE.Shape(), a = 0.07, t = 1.05, y = 0.72, arm = 0.26;
    sh.moveTo(-a, 0); sh.lineTo(-a, y - a); sh.lineTo(-arm, y - a); sh.lineTo(-arm, y + a); sh.lineTo(-a, y + a); sh.lineTo(-a, t);
    sh.lineTo(a, t); sh.lineTo(a, y + a); sh.lineTo(arm, y + a); sh.lineTo(arm, y - a); sh.lineTo(a, y - a); sh.lineTo(a, 0); sh.lineTo(-a, 0);
    return sh;
  }
  var ext = { depth: 0.12, bevelEnabled: true, bevelThickness: 0.015, bevelSize: 0.015, bevelSegments: 1, curveSegments: 10 };
  var stoneGeos = [
    new THREE.ExtrudeGeometry(tabletShape(0.6, 0.85), ext).translate(0, 0, -0.06),
    new THREE.ExtrudeGeometry(crossShape(), ext).translate(0, 0, -0.06),
    merge([tinted(new THREE.BoxGeometry(0.62, 0.35, 0.62).translate(0, 0.17, 0), '#ffffff'),
           tinted(new THREE.CylinderGeometry(0.14, 0.22, 1.7, 4).rotateY(Math.PI / 4).translate(0, 1.2, 0), '#ffffff'),
           tinted(new THREE.ConeGeometry(0.16, 0.3, 4).rotateY(Math.PI / 4).translate(0, 2.2, 0), '#ffffff')])
  ];
  var stoneMat = new THREE.MeshStandardMaterial({ color: '#a5a49c', roughness: 0.95 });
  var graves = [];
  for (var gx = YARD.x0 + 4; gx < YARD.x1 - 3; gx += 2.6) {
    for (var gz = YARD.z0 - 2.5; gz > YARD.z1 + 2; gz -= 1.7) {
      if (Math.abs(gz - YARD.gate) < 1.4 || r() < 0.18 || Math.hypot(gx - OAK.x, gz - OAK.z) < 2.6) continue;
      graves.push([gx + (r() - 0.5) * 0.5, gz + (r() - 0.5) * 0.4, r() < 0.68 ? 0 : r() < 0.8 ? 1 : 2]);
    }
  }
  stoneGeos.forEach(function (geo, type) {
    var list = graves.filter(function (g) { return g[2] === type; });
    var im = new THREE.InstancedMesh(geo, stoneMat, Math.max(1, list.length)), e = new THREE.Euler();
    list.forEach(function (g, n) {
      var sc = type === 2 ? 0.8 + r() * 0.5 : 0.8 + r() * 0.6;
      e.set((r() - 0.5) * 0.16, -Math.PI / 2 + (r() - 0.5) * 0.25, (r() - 0.5) * 0.14);
      im.setMatrixAt(n, m4.compose(p4.set(g[0], land(g[0], g[1]) - 0.05, g[1]), q4.setFromEuler(e), s4.set(sc, sc, sc)));
      im.setColorAt(n, tmpC.setHSL(0.12, 0.05, 0.5 + r() * 0.35));
    });
    im.count = list.length;
    world.add(im);
  });

  // A low fieldstone wall round the yard, gate pillars, an open iron gate.
  var wallTex = stoneTexture(r), wallMat = new THREE.MeshLambertMaterial({ map: wallTex, emissive: '#2a2c34', emissiveMap: wallTex });
  // Each run of wall is bent to sit on the slope of the ridge.
  function wallSeg(x0, z0, x1, z1) {
    var len = Math.hypot(x1 - x0, z1 - z0), geo = new THREE.BoxGeometry(len, 1.0, 0.5, Math.ceil(len / 1.5), 1, 1);
    var uv = geo.attributes.uv, pos = geo.attributes.position, k;
    for (k = 0; k < uv.count; k++) uv.setXY(k, uv.getX(k) * len * 0.9, uv.getY(k) * 0.95);
    geo.rotateY(-Math.atan2(z1 - z0, x1 - x0)).translate((x0 + x1) / 2, 0, (z0 + z1) / 2);
    for (k = 0; k < pos.count; k++) pos.setY(k, pos.getY(k) + land(pos.getX(k), pos.getZ(k)) + 0.35);
    geo.computeVertexNormals();
    world.add(new THREE.Mesh(geo, wallMat));
  }
  wallSeg(YARD.x0, YARD.z0, YARD.x1, YARD.z0);
  wallSeg(YARD.x0, YARD.z1, YARD.x1, YARD.z1);
  wallSeg(YARD.x1, YARD.z0, YARD.x1, YARD.z1);
  wallSeg(YARD.x0, YARD.z0, YARD.x0, YARD.gate + 1.3);
  wallSeg(YARD.x0, YARD.gate - 1.3, YARD.x0, YARD.z1);
  var gateParts = [];
  [1.3, -1.3].forEach(function (o, k) {
    var gzz = YARD.gate + o, gy = land(YARD.x0, gzz);
    gateParts.push(box(0.6, 1.7, 0.6, YARD.x0, gy + 0.6, gzz, '#76767a'), box(0.75, 0.15, 0.75, YARD.x0, gy + 1.5, gzz, '#5a5a5e'));
    // Each leaf swings inward, hinged at its pillar.
    var leaf = [], sgn = k ? 1 : -1;
    for (var bar = 0; bar <= 7; bar++) leaf.push(box(0.03, 1.2, 0.03, 0, 0.6, sgn * (0.12 + bar * 0.15), '#121214'));
    leaf.push(box(0.04, 0.05, 1.15, 0, 1.15, sgn * 0.65, '#121214'), box(0.04, 0.05, 1.15, 0, 0.2, sgn * 0.65, '#121214'));
    gateParts.push(merge(leaf).rotateY(sgn * 1.1).translate(YARD.x0 + 0.1, gy, gzz - sgn * 0.3));
  });
  world.add(new THREE.Mesh(merge(gateParts), new THREE.MeshLambertMaterial({ vertexColors: true })));

  // Mist that gathers among the graves.
  var mists = [];
  for (i = 0; i < 16; i++) {
    var mx2 = YARD.x0 - 8 + r() * 40, mz2 = YARD.z0 + 6 - r() * 38;
    var ms = new THREE.Sprite(new THREE.SpriteMaterial({ map: puff, color: '#9aa8d0', transparent: true, depthWrite: false, opacity: 0 }));
    ms.position.set(mx2, land(mx2, mz2) + 0.5 + r() * 0.8, mz2);
    ms.scale.set(10 + r() * 10, 2.2 + r() * 1.5, 1);
    ms.userData = { x: mx2, ph: r() * 6.28 };
    world.add(ms);
    mists.push(ms);
  }

  // ── Frame ──────────────────────────────────────────────────────────────
  var horizon = new THREE.Color(), tmp = new THREE.Color(), HORIZON = new THREE.Color('#26335a'), side = new THREE.Vector3(),
      head = new THREE.Vector3(), tail = new THREE.Vector3(), lightDir = new THREE.Vector3(),
      HIGH_MOON = new THREE.Vector3(0.75, 0.5, 0.42).normalize();

  function frame(f) {
    var row = f.row, time = f.time, dt = f.dt, slow = env.reduceMotion;
    var dim = row[1], smoke = row[2], heading = row[4], pitch = row[5], eye = row[6], hush = row[7], lamp = row[8],
        battle = row[9], skyAmt = row[10], farm = row[11], fires = row[12], pyre = row[13], mist = row[14];
    clock.value += dt * (0.4 + f.wind);
    sway.value = 0.05 + f.wind * 0.25;

    path.getPoint(clamp(f.cam / (PATH.length - 1), 0, 1), camera.position);
    camera.position.y = land(camera.position.x, camera.position.z) + eye + Math.sin(time * 0.8) * 0.012;
    camera.rotation.set(pitch - f.my * 0.06, heading + (portrait ? row[15] : 0) - f.mx * 0.12, (1.7 - eye) * 0.08, 'YXZ');
    sky.position.copy(camera.position);
    sea.userData.follow(camera.position);
    sea.position.y = -0.5;
    lightDir.copy(MOON).lerp(HIGH_MOON, smooth(0.6, 2.8, f.cam)).normalize();
    moonLight.position.copy(camera.position).addScaledVector(lightDir, 300);
    moonLight.target.position.copy(camera.position);

    // Gunfire beyond the ridge: each battery flashes at its own moments,
    // lighting the smoke near it.
    var energy = 0;
    batteries.forEach(function (bt) {
      bt.age += dt;
      bt.wait -= dt * battle * (slow ? 0.5 : 1);
      if (battle > 0.02 && bt.wait <= 0) { bt.age = 0; bt.wait = 0.4 + Math.random() * 1.6; }
      var pop = Math.exp(-bt.age * 7);
      bt.glow = Math.exp(-bt.age * 3) * (battle > 0.02 ? 1 : 0);
      bt.sprite.material.opacity = pop * Math.min(1, battle * 2);
      energy += bt.glow;
    });
    smokes.forEach(function (sm) {
      var u = sm.userData, lit = 0;
      for (var k = 0; k < batteries.length; k++) lit += batteries[k].glow * u.w[k];
      sm.material.color.copy(smokeBase).lerp(fireTint, Math.min(lit * 1.6, 1));
      sm.material.opacity = smoke * (u.near ? 0.22 : 0.55) * (0.85 + 0.15 * Math.sin(time * 0.2 + u.ph));
      sm.visible = sm.material.opacity > 0.003;
      sm.position.x = u.base.x + Math.sin(time * 0.05 + u.ph) * 6 + time * f.wind * 0.8 % 40;
    });
    volMat.uniforms.uTime.value = time;
    volMat.uniforms.uRate.value = battle;
    volley.visible = battle > 0.01;

    // The sky: the moon's glow on the horizon, warmed by the guns.
    horizon.copy(HORIZON).lerp(tmp.set('#4a2a24'), Math.min(energy * 0.12, 0.6) * battle);
    dome.uniforms.horizon.value.copy(horizon);
    world.fog.color.copy(FOG).lerp(tmp.set('#1e1a1e'), smoke * 0.4);
    world.fog.density = 0.0028 + smoke * 0.0015 - smooth(9, 12, f.cam) * 0.001;
    gl.setClearColor(world.fog.color);
    su.uTime.value = time;
    su.uSky.value.copy(horizon);
    starMat.uniforms.uTime.value = time;
    starMat.uniforms.uBoost.value = skyAmt + mist * 0.2;
    starMat.uniforms.uScale.value = pxScale / 400;
    bandMat.uniforms.uAmt.value = 0.4 + skyAmt * 0.8 + mist * 0.1;
    hemi.intensity = 0.75 + Math.min(energy * 0.05, 0.3) * battle;

    // Falling stars while the sky is open.
    meteorClock -= dt * skyAmt * (slow ? 0.4 : 1);
    if (skyAmt > 0.3 && meteorClock <= 0) {
      meteorClock = 1.4 + Math.random() * 2.2;
      for (var mi = 0; mi < METEORS; mi++) {
        var mt = meteors[mi];
        if (mt.age < mt.life) continue;
        var az = heading + (Math.random() - 0.5) * 0.9, el = 0.5 + Math.random() * 0.45;
        mt.from.set(-Math.sin(az) * Math.cos(el), Math.sin(el), -Math.cos(az) * Math.cos(el));
        mt.dir.set(Math.random() - 0.5, -0.6 - Math.random() * 0.4, Math.random() - 0.5).normalize();
        mt.age = 0;
        mt.life = 0.7 + Math.random() * 0.5;
        break;
      }
    }
    meteors.forEach(function (mt, mi) {
      mt.age += dt;
      var p = clamp(mt.age / mt.life, 0, 1), on = mt.age < mt.life ? Math.sin(Math.PI * p) : 0;
      head.copy(mt.from).addScaledVector(mt.dir, p * 0.32).normalize().multiplyScalar(1000);
      tail.copy(mt.from).addScaledVector(mt.dir, Math.max(0, p * 0.32 - 0.1)).normalize().multiplyScalar(1000);
      side.copy(head).sub(tail).cross(head).normalize().multiplyScalar(1.6);
      var o = mi * 12;
      metPos[o] = head.x + side.x; metPos[o + 1] = head.y + side.y; metPos[o + 2] = head.z + side.z;
      metPos[o + 3] = head.x - side.x; metPos[o + 4] = head.y - side.y; metPos[o + 5] = head.z - side.z;
      metPos[o + 6] = tail.x + side.x * 0.2; metPos[o + 7] = tail.y + side.y * 0.2; metPos[o + 8] = tail.z + side.z * 0.2;
      metPos[o + 9] = tail.x - side.x * 0.2; metPos[o + 10] = tail.y - side.y * 0.2; metPos[o + 11] = tail.z - side.z * 0.2;
      var c = mi * 16;
      for (var k = 0; k < 2; k++) { metCol[c + k * 4] = 0.9; metCol[c + k * 4 + 1] = 0.94; metCol[c + k * 4 + 2] = 1; metCol[c + k * 4 + 3] = on; }
      for (k = 2; k < 4; k++) { metCol[c + k * 4] = 0.6; metCol[c + k * 4 + 1] = 0.7; metCol[c + k * 4 + 2] = 1; metCol[c + k * 4 + 3] = 0; }
    });
    metGeo.attributes.position.needsUpdate = true;
    metGeo.attributes.color.needsUpdate = true;

    // The town falls silent: lit windows go out one by one.
    {
      litList.forEach(function (l) {
        var on = 1 - smooth(l.darkAt, l.darkAt + 0.06, hush);
        if (l.sick) on *= 0.55 + 0.12 * Math.sin(time * 2.1) * Math.sin(time * 5.3);
        if (Math.abs(on - l.last) < 0.002) return;
        l.last = on;
        winMesh.setColorAt(l.i, tmpC.copy(darkGlass).lerp(warm, on));
        l.glow.material.opacity = on * 0.4;
        winMesh.instanceColor.needsUpdate = true;
      });
    }

    // The porch lamp, guttering when his mind flies away.
    var flick = 0.92 + 0.08 * Math.sin(time * 9) * Math.sin(time * 4.3);
    lampLight.intensity = 14 * lamp * flick;
    lampGlow.material.opacity = lamp * flick;
    lampGlass.material.color.set('#ffd9a0').multiplyScalar(0.3 + 0.7 * lamp);
    for (var mo = 0; mo < MOTHS; mo++) {
      var ph = mo * 1.7, rad = 0.35 + 0.25 * Math.sin(time * 0.7 + ph);
      mothPos[mo * 3] = LAMP.x + Math.cos(time * (2.2 + mo * 0.3) + ph) * rad;
      mothPos[mo * 3 + 1] = LAMP.y + Math.sin(time * (1.7 + mo * 0.2) + ph) * 0.25;
      mothPos[mo * 3 + 2] = LAMP.z + Math.sin(time * (2.2 + mo * 0.3) + ph) * rad;
    }
    mothGeo.attributes.position.needsUpdate = true;
    moths.material.opacity = lamp * 0.8;

    // Farm windows; the first goes dark.
    farmLights[0].win.material.color.set('#1c2335').lerp(warm, farm);
    farmLights[0].glow.material.opacity = farm * 0.6;
    farmLights[1].glow.material.opacity = 0.6;

    // The camps die out; the pyre flares, sends up its sparks and sinks.
    fireMat.uniforms.uTime.value = time;
    fireMat.uniforms.uDie.value = fires;
    fireMat.uniforms.uScale.value = pxScale;
    fireMat.uniforms.uAlpha.value = smooth(9.5, 11.5, f.cam) * 0.8;
    haze.forEach(function (hs) { hs.material.opacity = 0.05 * (1 - fires) * fireMat.uniforms.uAlpha.value; });
    var pyreOn = clamp(pyre, 0, 1.3);
    flames.forEach(function (fl, k) {
      var u = fl.userData, fk = 0.8 + 0.2 * Math.sin(time * 7 + u.ph) * Math.sin(time * 4.1 + u.ph * 2);
      var hgt = u.h * pyreOn * fk;
      fl.position.set(PYRE.x + u.dx + Math.sin(time * 3 + u.ph) * 0.6, PYRE.y + hgt * 0.45, PYRE.z);
      fl.scale.set(hgt * 0.55, hgt, 1);
      fl.material.opacity = Math.min(1, pyreOn) * 0.85;
      fl.visible = pyreOn > 0.01;
    });
    pyreGlow.material.opacity = Math.min(1, pyreOn) * 0.35;
    sparkClock += dt * pyreOn * 50;
    while (sparkClock > 1) {
      sparkClock -= 1;
      var si = Math.floor(Math.random() * SPARKS) * 3;
      spPos[si] = PYRE.x + (Math.random() - 0.5) * 4; spPos[si + 1] = PYRE.y + 4 + Math.random() * 6; spPos[si + 2] = PYRE.z;
      spVel[si] = (Math.random() - 0.5) * 2 + f.wind * 2; spVel[si + 1] = 5 + Math.random() * 8; spVel[si + 2] = (Math.random() - 0.5) * 2;
      spLife[si / 3] = 2 + Math.random() * 3;
    }
    for (var sp = 0; sp < SPARKS; sp++) {
      var j = sp * 3;
      if (spLife[sp] <= 0) { spPos[j + 1] = -999; continue; }
      spLife[sp] -= dt;
      spPos[j] += spVel[j] * dt; spPos[j + 1] += spVel[j + 1] * dt; spPos[j + 2] += spVel[j + 2] * dt;
      spVel[j + 1] *= 0.995;
    }
    spGeo.attributes.position.needsUpdate = true;
    sparks.material.opacity = Math.min(1, pyreOn + 0.2);

    // Mist rising among the graves.
    mists.forEach(function (ms) {
      var u = ms.userData;
      ms.position.x = u.x + Math.sin(time * 0.06 + u.ph) * 3;
      ms.material.opacity = mist * 0.2 * (0.7 + 0.3 * Math.sin(time * 0.3 + u.ph));
      ms.visible = mist > 0.01;
    });

    gl.toneMappingExposure = 1.05 * (1 - dim * 0.65);
    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) {
      fitCamera(gl, camera, w, h, dpr, small);
      portrait = w < h;
      pxScale = h * Math.min(dpr, small ? 1.5 : 1.75) / (2 * Math.tan(camera.fov * Math.PI / 360));
    },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('silence', {
  renderer: renderer3d,
  maxLines: 6,
  align: ['left', 'right', 'left', 'left', 'left', 'right', 'left', 'center', 'right', 'left', 'right', 'right', 'left', 'center'],
  // Panels: 0-1 stanza I; 2-5 the old soldier; 6-8 the great silences;
  // 9-11 nations, Jeanne d'Arc and age; 12-13 the dead.
  keys: function (T) {
    function at(i, frac) { i = Math.min(i, T.count - 1); return lerp(T.start(i), T.end(i), frac); }
    //       unit        path  dim  smoke wind heading pitch eye  hush lamp battle sky farm fires pyre mist phone
    return [
      [0,            0.00, 0.0, 0.0, 0.20, -1.52, 0.10, 1.7, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -0.12],
      [0.7,          0.04, 0.0, 0.0, 0.20, -1.50, 0.12, 1.7, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -0.12],
      [at(0, 0.3),   0.15, 0.0, 0.0, 0.20, -1.46, 0.15, 1.7, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -0.10],  // "the stars and ... the sea"
      [at(0, 0.6),   0.45, 0.0, 0.0, 0.15, -0.55, 0.04, 1.7, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  // "the city when it pauses"
      [at(0, 1.0),   0.95, 0.0, 0.0, 0.10, -0.06, 0.04, 1.7, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  // a man and a maid; the sick
      [at(1, 0.5),   1.90, 0.0, 0.0, 0.05, 0.04, 0.02, 1.7, 0.35, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], // "of what use is language?"
      [at(1, 1.0),   2.90, 0.0, 0.0, 0.00, 0.30, 0.0, 1.7, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],   // "we cannot speak"
      [at(2, 0.35),  3.85, 0.0, 0.0, 0.00, 1.55, -0.06, 1.7, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -0.28], // the grocery porch
      [at(2, 0.8),   4.00, 0.0, 0.0, 0.00, 1.62, -0.08, 1.65, 1.0, 0.6, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -0.28], // "his mind flies away"
      [at(3, 0.35),  4.04, 0.0, 0.0, 0.00, 1.64, -0.08, 1.65, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -0.28], // "A bear bit it off."
      [at(3, 0.62),  4.90, 0.0, 0.3, 0.05, 0.12, 0.02, 1.7, 1.0, 0.45, 0.15, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], // "lives over"
      [at(3, 1.0),   6.30, 0.0, 0.6, 0.10, 0.04, 0.05, 1.7, 1.0, 0.35, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
      [at(4, 0.35),  6.90, 0.0, 1.0, 0.10, 0.00, 0.07, 1.6, 1.0, 0.3, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  // "the flashes of guns"
      [at(4, 0.6),   7.00, 0.1, 1.0, 0.10, 0.00, 0.14, 0.45, 1.0, 0.3, 0.9, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], // "lying on the ground"
      [at(4, 0.85),  7.00, 0.5, 0.9, 0.10, 0.00, 0.14, 0.45, 1.0, 0.3, 0.3, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], // "the long days in bed"
      [at(4, 1.0),   7.02, 0.6, 0.85, 0.10, 0.00, 0.12, 0.55, 1.0, 0.3, 0.05, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
      [at(5, 0.35),  7.15, 0.1, 0.6, 0.15, 0.00, 0.04, 1.65, 1.0, 0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], // "if he could describe it all"
      [at(5, 1.0),   8.10, 0.0, 0.15, 0.20, -0.02, 0.03, 1.7, 1.0, 0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
      [at(6, 0.5),   8.70, 0.0, 0.0, 0.20, 0.00, 0.03, 1.7, 1.0, 0.3, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0],  // hatred, love, friendship
      [at(6, 1.0),   9.05, 0.0, 0.0, 0.20, 0.02, 0.10, 1.7, 1.0, 0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, 0.0],
      [at(7, 0.35),  9.25, 0.0, 0.0, 0.15, 0.10, 0.62, 1.7, 1.0, 0.3, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  // "visions not to be uttered"
      [at(7, 0.8),   9.50, 0.0, 0.0, 0.15, 0.14, 0.55, 1.7, 1.0, 0.3, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
      [at(8, 0.1),   9.75, 0.0, 0.0, 0.15, 0.30, 0.06, 1.7, 1.0, 0.3, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.0],  // "the silence of defeat"
      [at(8, 0.45),  10.00, 0.0, 0.0, 0.15, 0.36, 0.03, 1.7, 1.0, 0.3, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0],
      [at(8, 0.65),  10.15, 0.0, 0.0, 0.15, 0.38, 0.03, 1.7, 1.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "whose hand suddenly grips yours"
      [at(8, 1.0),   10.60, 0.0, 0.0, 0.20, 0.10, 0.02, 1.7, 1.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [at(9, 0.45),  11.80, 0.0, 0.0, 0.20, 0.05, -0.12, 1.7, 1.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // over the ridge: the camps
      [at(9, 1.0),   12.00, 0.0, 0.0, 0.20, 0.12, -0.12, 1.7, 1.0, 0.3, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0], // "broken nations and vanquished leaders"
      [at(10, 0.3),  12.05, 0.0, 0.0, 0.20, 0.22, -0.08, 1.7, 1.0, 0.3, 0.0, 0.0, 0.0, 0.4, 0.35, 0.0, 0.08], // Lincoln, Napoleon
      [at(10, 0.65), 12.10, 0.0, 0.0, 0.20, 0.20, -0.06, 1.7, 1.0, 0.3, 0.0, 0.0, 0.0, 0.6, 1.0, 0.0, 0.08],  // "amid the flames"
      [at(11, 0.3),  12.20, 0.0, 0.0, 0.20, 0.22, -0.04, 1.7, 1.0, 0.3, 0.0, 0.0, 0.0, 0.7, 1.25, 0.0, 0.08], // "all sorrows, all hope"
      [at(11, 0.7),  12.70, 0.0, 0.0, 0.20, -0.90, 0.06, 1.7, 1.0, 0.3, 0.0, 0.0, 0.0, 0.8, 0.4, 0.1, 0.0],  // "the silence of age"
      [at(11, 1.0),  13.30, 0.0, 0.0, 0.20, -1.60, 0.10, 1.7, 1.0, 0.3, 0.0, 0.1, 0.0, 0.85, 0.15, 0.3, 0.0],
      [at(12, 0.5),  14.20, 0.0, 0.0, 0.15, -1.40, 0.03, 1.7, 1.0, 0.3, 0.0, 0.1, 0.0, 0.9, 0.05, 0.7, 0.0], // "the silence of the dead"
      [at(12, 1.0),  14.80, 0.0, 0.0, 0.15, -1.40, 0.03, 1.7, 1.0, 0.3, 0.0, 0.1, 0.0, 0.95, 0.0, 0.85, 0.0],
      [at(13, 0.6),  15.70, 0.0, 0.0, 0.10, -1.42, 0.02, 1.65, 1.0, 0.3, 0.0, 0.2, 0.0, 1.0, 0.0, 1.0, 0.0], // "as we approach them"
      [T.total,      16.20, 0.0, 0.0, 0.10, -1.45, 0.10, 1.6, 1.0, 0.3, 0.0, 0.3, 0.0, 1.0, 0.0, 1.0, 0.0]
    ];
  },
  sound: {
    src: '/audio/wind.mp3',
    label: 'Play the night wind',
    volume: function (row) { return 0.04 + 0.1 * row[3]; },
    cues: [
      { stanza: 4, at: 0.2, play: cannon },
      { stanza: 4, at: 0.45, play: cannon },
      { stanza: 4, at: 0.6, play: cannon },
      { stanza: 4, at: 0.9, play: cannon },
      { stanza: 12, at: 0.35, play: toll }
    ]
  }
});
