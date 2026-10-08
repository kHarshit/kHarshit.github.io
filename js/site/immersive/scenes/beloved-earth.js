/*
 * Scene for "Late Fragment" (Raymond Carver), his last poem, written at
 * home on the Strait of Juan de Fuca.
 *
 * Read as its two questions and answers (maxLines 3):
 * I   A still Pacific Northwest shore at evening: dark firs to the water's
 *     edge, the strait like glass under a fading sky, Vancouver Island low
 *     across it, and a small house among the firs with its lights on. "I
 *     did.": its windows warm.
 * II  "To call myself beloved, to feel myself beloved on the earth": warmth
 *     suffuses the sky and the water, lights come on along both shores, and
 *     the view rises, slowly, out over the strait, until the curve of the
 *     earth shows: the whole earth, its twilight and its lit towns below.
 *
 * Everything is at true scale (metres) on a real-sized Earth, so the rise
 * can go from eye height to thousands of kilometres in one move: the local
 * shore bends with the planet's curvature in its vertex shaders and sits on
 * a procedural globe, drawn with a logarithmic depth buffer. Columns:
 *   [unit, altitude (log10 m), warmth, (unused), wind, yaw, pitch, drift, windows]
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, softSprite, skyDome, scatter,
         disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

var R = 6.371e6;                                     // the Earth, in metres
var CURVE = (0.5 / R).toExponential(6);              // drop per square metre from the origin
var SUN = new THREE.Vector3(0.35, -0.09, -0.94).normalize();    // just set, west-north-west
var EYE = new THREE.Vector3(5, 0, 0);                // at the end of a jetty, looking west (-z); land to the left (-x)

// ── The shore ────────────────────────────────────────────────────────────
// Land lies on the low-x side of shore(z); a wooded point juts out ahead.
function shore(z) { return 6 * Math.sin(z * 0.012) + 95 * Math.exp(-Math.pow((z + 700) / 260, 2)); }

function height(x, z) {
  var d = shore(z) - x, h;                          // > 0 inland
  if (d < 0) h = Math.max(d * 0.06, -25);
  else h = Math.min(d * 0.06, 2.2) + smooth(30, 1600, d) * (170 + 90 * Math.sin(z * 0.0021) + 50 * Math.sin(z * 0.0053 + x * 0.003));
  // The Olympics, far to the south; Vancouver Island across the strait.
  h += smooth(9000, 22000, d) * (1500 + 500 * Math.sin(z * 0.00031) + 300 * Math.sin(z * 0.0011 + 1)) * (0.7 + 0.3 * Math.abs(Math.sin(z * 0.0007)));
  var e = x - 21000 - 2500 * Math.sin(z * 0.00006);
  h += smooth(-1200, 1500, e) * (445 + 260 * Math.sin(z * 0.00023) + 120 * Math.sin(z * 0.0009));
  // Sink the patch's edge under the globe.
  return lerp(h, -600, smooth(42000, 56000, Math.hypot(x, z)));
}

// Bend a built-in material's vertices down with the Earth's curvature. A
// mirrored copy (the reflection in the still water) clips anything that
// would rise above the surface.
function curved(mat, mirror) {
  mat.onBeforeCompile = function (sh) {
    sh.vertexShader = (mirror ? 'varying float vMirY;\n' : '') + sh.vertexShader.replace('#include <project_vertex>',
      'vec4 mvPosition = vec4(transformed, 1.0);\n#ifdef USE_INSTANCING\n mvPosition = instanceMatrix * mvPosition;\n#endif\n' +
      'vec4 cwp = modelMatrix * mvPosition;\n' + (mirror ? 'vMirY = cwp.y;\n' : '') +
      'cwp.y -= dot(cwp.xz, cwp.xz) * ' + CURVE + ';\n' +
      'mvPosition = viewMatrix * cwp;\ngl_Position = projectionMatrix * mvPosition;');
    if (mirror) {
      sh.fragmentShader = 'varying float vMirY;\n' + sh.fragmentShader.replace('void main() {', 'void main() {\n if (vMirY > 0.05) discard;');
    }
  };
  return mat;
}

// GLSL shared by the custom shaders: curvature, noise.
var GLSL = [
  'vec4 bend(vec4 w){ w.y -= dot(w.xz, w.xz) * ' + CURVE + '; return w; }',
  'float h3(vec3 p){ p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }',
  'float n3(vec3 x){ vec3 i = floor(x), f = fract(x); f = f * f * (3.0 - 2.0 * f);',
  '  return mix(mix(mix(h3(i), h3(i + vec3(1,0,0)), f.x), mix(h3(i + vec3(0,1,0)), h3(i + vec3(1,1,0)), f.x), f.y),',
  '             mix(mix(h3(i + vec3(0,0,1)), h3(i + vec3(1,0,1)), f.x), mix(h3(i + vec3(0,1,1)), h3(i + vec3(1,1,1)), f.x), f.y), f.z); }',
  'float dots(vec3 p, float dens){ vec3 c = floor(p), j = vec3(h3(c + 1.3), h3(c + 2.7), h3(c + 5.1));',
  '  return step(1.0 - dens, h3(c)) * smoothstep(0.22, 0.05, length(fract(p) - j * 0.6 - 0.2)); }',
  'float fbm(vec3 p){ float s = 0.0, a = 0.5; for (int k = 0; k < 5; k++) { s += a * n3(p); p = p * 2.03 + 1.7; a *= 0.5; } return s; }'
].join('\n');

// A grid that is fine near the origin and coarse far out: u in -1..1 -> L|u|^p.
function warpedGrid(L, seg, p) {
  var g = new THREE.PlaneGeometry(2, 2, seg, seg).rotateX(-Math.PI / 2), a = g.attributes.position;
  for (var i = 0; i < a.count; i++) {
    var u = a.getX(i), v = a.getZ(i);
    a.setX(i, Math.sign(u) * L * Math.pow(Math.abs(u), p));
    a.setZ(i, Math.sign(v) * L * Math.pow(Math.abs(v), p));
  }
  return g;
}

// A Douglas fir about 1 m tall (scaled per instance): trunk and three tiers.
function firGeometry() {
  return merge([
    tinted(new THREE.CylinderGeometry(0.012, 0.02, 0.3, 5).translate(0, 0.15, 0), '#2a2018'),
    tinted(new THREE.ConeGeometry(0.17, 0.5, 7).translate(0, 0.38, 0), '#ffffff'),
    tinted(new THREE.ConeGeometry(0.13, 0.42, 7).translate(0, 0.6, 0), '#ffffff'),
    tinted(new THREE.ConeGeometry(0.085, 0.36, 7).translate(0, 0.82, 0), '#ffffff')
  ]);
}

// A soft rising chime, for the moment the view lifts.
function swell(ac, out) {
  var t = ac.currentTime;
  [261.6, 329.6, 392, 523.3].forEach(function (f, k) {
    var o = ac.createOscillator(), g = ac.createGain();
    o.type = 'sine';
    o.frequency.value = f;
    g.gain.setValueAtTime(0.0001, t + k * 0.35);
    g.gain.exponentialRampToValueAtTime(0.05, t + k * 0.35 + 0.9);
    g.gain.exponentialRampToValueAtTime(0.0001, t + k * 0.35 + 6);
    o.connect(g); g.connect(out);
    o.start(t + k * 0.35);
    o.stop(t + k * 0.35 + 6.2);
  });
}

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(17);
  var gl = makeRenderer(canvas, { clear: '#1b2140', logDepth: true });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#3a3a5a', 0.00012);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.2, 8e7);

  // ── Sky ──
  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#16204a', mid: '#4a5488', horizon: '#e0a888', sun: '#ffb27a' }, 1500);
  dome.mesh.material.depthTest = false;
  dome.mesh.renderOrder = -10;
  dome.uniforms.sunDir.value.copy(SUN);
  sky.add(dome.mesh);

  var starPos = [], starSize = [];
  for (var i = 0; i < (small ? 2500 : 5000); i++) {
    var th = r() * Math.PI * 2, y = r() * 2 - 1, s = Math.sqrt(1 - y * y);
    starPos.push(s * Math.cos(th), y, s * Math.sin(th));
    starSize.push(0.6 + Math.pow(r(), 3) * 2.4);
  }
  var starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.Float32BufferAttribute(starPos, 3));
  starGeo.setAttribute('aSize', new THREE.Float32BufferAttribute(starSize, 1));
  // Far beyond the Earth, so the globe hides the stars behind it.
  var starMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uAmt: { value: 0 }, uScale: { value: 1 }, uLow: { value: 0 } },
    vertexShader: '#include <common>\n#include <logdepthbuf_pars_vertex>\nattribute float aSize; uniform float uAmt; uniform float uScale; uniform float uLow; varying float vA;\n' +
      'void main(){ vec4 mv = modelViewMatrix * vec4(position * 3.0e7, 1.0); gl_Position = projectionMatrix * mv;\n' +
      ' vA = uAmt * mix(smoothstep(0.05, 0.5, position.y), 1.0, uLow); gl_PointSize = aSize * uScale;\n#include <logdepthbuf_vertex>\n}',
    fragmentShader: '#include <logdepthbuf_pars_fragment>\nvarying float vA; void main(){\n#include <logdepthbuf_fragment>\n' +
      ' float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard; float a = smoothstep(0.5, 0.0, d) * vA;\n' +
      ' gl_FragColor = vec4(vec3(0.88, 0.91, 1.0) * a, a); }'
  });
  var stars = new THREE.Points(starGeo, starMat);
  stars.frustumCulled = false;
  var starGroup = new THREE.Group();
  starGroup.add(stars);
  world.add(starGroup);

  var hemi = new THREE.HemisphereLight('#8a90c8', '#2a2430', 1.2);
  var glow = new THREE.DirectionalLight('#ffb080', 0.6);    // the afterglow from the west
  glow.position.set(0.35, 0.12, -0.94);
  world.add(hemi, glow);

  // ── Land: the near shore, the Olympics, Vancouver Island ──
  var land = warpedGrid(60000, small ? 180 : 260, 2.2), lp = land.attributes.position, cols = [];
  var cBeach = new THREE.Color('#6a6258'), cWood = new THREE.Color('#1c2a22'), cRock = new THREE.Color('#4a5060'),
      cSnow = new THREE.Color('#c8cede'), cFar = new THREE.Color('#2a3448'), c = new THREE.Color();
  for (i = 0; i < lp.count; i++) {
    var x = lp.getX(i), z = lp.getZ(i), h = height(x, z), d = shore(z) - x;
    lp.setY(i, h);
    c.copy(cBeach).lerp(cWood, smooth(4, 22, d));
    if (d < -2) c.set('#0a1220');
    c.lerp(cRock, smooth(900, 1800, h)).lerp(cSnow, smooth(1500, 1900, h));
    if (x > 15000) c.copy(cFar);
    cols.push(c.r, c.g, c.b);
  }
  land.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
  land.computeVertexNormals();
  var landMat = curved(new THREE.MeshLambertMaterial({ vertexColors: true }));
  var landMesh = new THREE.Mesh(land, landMat);
  world.add(landMesh);
  var landMirror = new THREE.Mesh(land, curved(new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }), true));
  landMirror.scale.y = -1;
  world.add(landMirror);

  // ── Firs down to the water, and their reflections ──
  var firGeo = firGeometry(), firMat = curved(new THREE.MeshLambertMaterial({ vertexColors: true }));
  var firs = new THREE.InstancedMesh(firGeo, firMat, small ? 3500 : 9000), up = new THREE.Vector3(0, 1, 0);
  scatter(firs, 60000, function (k, p, q, sc, col) {
    var fz = 600 - r() * 4200, fd = 14 + Math.pow(r(), 2.2) * 3000, fx = shore(fz) - fd;
    if (Math.abs(fz) < 50 && fd < 26) return false;          // a clearing round you on the beach
    if (Math.abs(fz + 90) < 14 && Math.abs(fd - 16) < 10) return false;    // the house
    var fh = height(fx, fz);
    if (fh > 500) return false;
    p.set(fx, fh - 0.5, fz);
    q.setFromAxisAngle(up, r() * 6.28);
    var t = 18 + r() * 26;
    sc.set(t * (0.85 + r() * 0.3), t, t * (0.85 + r() * 0.3));
    col.setHSL(0.38 + r() * 0.06, 0.25, 0.12 + r() * 0.06);
  });
  world.add(firs);
  var firMirror = new THREE.InstancedMesh(firGeo, curved(new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }), true), firs.count);
  firMirror.instanceMatrix = firs.instanceMatrix;
  firMirror.instanceColor = firs.instanceColor;
  firMirror.count = firs.count;
  var mirror = new THREE.Group();
  mirror.scale.y = -1;
  mirror.add(firMirror);
  world.add(mirror);

  // ── The house: dark boards, a gable roof, lit windows ──
  var HX = shore(-90) - 16, HZ = -90, HY = height(HX, HZ);
  var house = new THREE.Group(), houseM = new THREE.Group();
  var boards = curved(new THREE.MeshLambertMaterial({ color: '#3a3430' }));
  var roofMat = curved(new THREE.MeshLambertMaterial({ color: '#2a2a2e' }));
  var body = new THREE.Mesh(new THREE.BoxGeometry(9, 4.2, 6.5).translate(0, 2.1, 0), boards);
  var roofShape = new THREE.Shape();
  roofShape.moveTo(-5.2, 0); roofShape.lineTo(5.2, 0); roofShape.lineTo(0, 2.6); roofShape.lineTo(-5.2, 0);
  var roof = new THREE.Mesh(new THREE.ExtrudeGeometry(roofShape, { depth: 7.3, bevelEnabled: false }).translate(0, 4.2, -3.65), roofMat);
  var chimney = new THREE.Mesh(new THREE.BoxGeometry(0.8, 2.2, 0.8).translate(2.5, 6, 0), boards);
  var winMat = curved(new THREE.MeshBasicMaterial({ color: '#ffc27a' }));
  var wins = [[-2.6, 2.0, 3.26, 0], [0.4, 2.0, 3.26, 0], [2.9, 2.0, 3.26, 0], [4.51, 2.0, -1.2, 1], [4.51, 2.0, 1.4, 1]];
  wins.forEach(function (w) {
    var m = new THREE.Mesh(new THREE.PlaneGeometry(1.3, 1.1), winMat);
    m.position.set(w[0], w[1], w[2]);
    if (w[3]) m.rotation.y = Math.PI / 2;
    house.add(m);
  });
  house.add(body, roof, chimney);
  var porch = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,200,130,1)', 'rgba(255,160,90,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
  porch.scale.setScalar(9);
  porch.position.set(1, 2.4, 4.5);
  house.add(porch);
  var lamp = new THREE.PointLight('#ffb070', 0, 45, 1.4);
  lamp.position.set(1, 2.5, 6);
  house.add(lamp);
  house.position.set(HX, HY - 0.2, HZ);
  house.rotation.y = 0.5;
  world.add(house);
  // Its reflection: the same meshes with clipped, two-sided copies of the materials.
  var mirrorWins = [];
  house.children.forEach(function (m) {
    if (!m.isMesh) return;
    var mm = m.clone();
    mm.material = curved(m.material.clone(), true);
    mm.material.side = THREE.DoubleSide;
    if (m.material === winMat) mirrorWins.push(mm.material);
    houseM.add(mm);
  });
  houseM.position.copy(house.position);
  houseM.rotation.copy(house.rotation);
  mirror.add(houseM);

  // ── The strait: still water reflecting the sky and the shore ──
  var waterU = { uTop: dome.uniforms.top, uMid: dome.uniforms.mid, uHorizon: dome.uniforms.horizon, uGlow: dome.uniforms.sunColor,
                 uSun: { value: SUN }, uTime: { value: 0 }, uDeep: { value: new THREE.Color('#0a1222') }, uWarm: { value: 0 }, uFade: { value: 1 } };
  var water = new THREE.Mesh(warpedGrid(150000, small ? 120 : 160, 2.0), new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, uniforms: waterU,
    vertexShader: '#include <common>\n#include <logdepthbuf_pars_vertex>\n' + GLSL + '\nvarying vec3 vW; varying float vR;\n' +
      'void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; vR = length(w.xz); w = bend(w);\n' +
      ' gl_Position = projectionMatrix * viewMatrix * w;\n#include <logdepthbuf_vertex>\n}',
    fragmentShader: '#include <logdepthbuf_pars_fragment>\n' + GLSL + '\n' +
      'uniform vec3 uTop; uniform vec3 uMid; uniform vec3 uHorizon; uniform vec3 uGlow; uniform vec3 uSun; uniform vec3 uDeep; uniform float uTime; uniform float uWarm; uniform float uFade;\n' +
      'varying vec3 vW; varying float vR;\n' +
      'vec3 skyCol(vec3 d){ float h = clamp(d.y, 0.0, 1.0); vec3 c = mix(uHorizon, uMid, smoothstep(0.0, 0.22, h)); c = mix(c, uTop, smoothstep(0.22, 0.75, h));\n' +
      ' float s = max(dot(d, normalize(uSun)), 0.0); return c + uGlow * (pow(s, 600.0) * 4.0 + pow(s, 12.0) * 0.35); }\n' +
      'void main(){\n#include <logdepthbuf_fragment>\n' +
      ' vec3 V = normalize(vW - cameraPosition);\n' +
      // Faint ripples only; the strait is almost glass.
      ' vec2 q = vW.xz * vec2(0.12, 0.3); float e = 0.04;\n' +
      ' float a0 = n3(vec3(q, uTime * 0.25)), ax = n3(vec3(q + vec2(e, 0.0), uTime * 0.25)), az = n3(vec3(q + vec2(0.0, e), uTime * 0.25));\n' +
      ' float fadeN = 1.0 - smoothstep(30.0, 400.0, distance(vW, cameraPosition));\n' +
      ' vec3 N = normalize(vec3(-(ax - a0) / e * 0.025 * fadeN, 1.0, -(az - a0) / e * 0.025 * fadeN));\n' +
      ' vec3 Rf = reflect(V, N); Rf.y = abs(Rf.y);\n' +
      ' float fres = 0.04 + 0.96 * pow(1.0 - max(dot(-V, N), 0.0), 5.0);\n' +
      ' vec3 col = mix(uDeep, skyCol(Rf), fres);\n' +
      ' float alpha = mix(0.82, 0.97, fres) * (1.0 - smoothstep(110000.0, 150000.0, vR)) * uFade;\n' +
      ' gl_FragColor = vec4(col, alpha); }'
  }));
  water.frustumCulled = false;
  water.renderOrder = 1;
  world.add(water);

  // ── Lights coming on along both shores ──
  var LN = small ? 900 : 2000, lpos = [], lt = [], lsz = [];
  function addLight(x, z, lift, t) { lpos.push(x, height(x, z) + lift, z); lt.push(t); lsz.push(1.2 + r() * 1.8); }
  for (i = 0; i < LN; i++) {
    var kind = r();
    if (kind < 0.5) {            // Victoria and the villages across the strait, up on the slopes
      var vz = -26000 + r() * 30000 + (r() < 0.5 ? r() * 2500 : 0), vx = 21600 + 2500 * Math.sin(vz * 0.00006) + Math.pow(r(), 2) * 2200;
      addLight(vx, vz, 8, 0.15 + r() * 0.8);
    } else if (kind < 0.85) {    // houses along this shore and its hills
      var sz = 400 - r() * 9000, sd = 40 + Math.pow(r(), 1.5) * 2500;
      addLight(shore(sz) - sd, sz, 5, 0.1 + r() * 0.85);
    } else {                     // Port Angeles, behind you to the east
      var pz = 2500 + r() * 4000, pd = 30 + r() * 1800;
      addLight(shore(pz) - pd, pz, 6, 0.1 + r() * 0.7);
    }
  }
  var lightGeo = new THREE.BufferGeometry();
  lightGeo.setAttribute('position', new THREE.Float32BufferAttribute(lpos, 3));
  lightGeo.setAttribute('aT', new THREE.Float32BufferAttribute(lt, 1));
  lightGeo.setAttribute('aSize', new THREE.Float32BufferAttribute(lsz, 1));
  var lightMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uOn: { value: 0 }, uScale: { value: 1 }, uTime: { value: 0 }, uFade: { value: 1 } },
    vertexShader: '#include <common>\n#include <logdepthbuf_pars_vertex>\n' + GLSL + '\nattribute float aT; attribute float aSize; uniform float uOn; uniform float uScale; uniform float uTime; uniform float uFade; varying float vA;\n' +
      'void main(){ vec4 w = bend(modelMatrix * vec4(position, 1.0)); vec4 mv = viewMatrix * w; gl_Position = projectionMatrix * mv;\n' +
      ' vA = smoothstep(aT, aT + 0.06, uOn) * (0.85 + 0.15 * sin(uTime * 1.7 + aT * 61.0)) * uFade;\n' +
      ' gl_PointSize = aSize * uScale;\n#include <logdepthbuf_vertex>\n}',
    fragmentShader: '#include <logdepthbuf_pars_fragment>\nvarying float vA; void main(){\n#include <logdepthbuf_fragment>\n' +
      ' float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard; float a = smoothstep(0.5, 0.05, d) * vA;\n' +
      ' gl_FragColor = vec4(vec3(1.0, 0.78, 0.48) * a, a); }'
  });
  var shoreLights = new THREE.Points(lightGeo, lightMat);
  shoreLights.frustumCulled = false;
  world.add(shoreLights);

  // ── The globe beneath it all ──
  var globeU = { uSun: { value: SUN }, uWarm: { value: 0 }, uRim: { value: 0 }, uSkyCol: { value: new THREE.Color() } };
  var globe = new THREE.Mesh(new THREE.SphereGeometry(R - 150, small ? 160 : 256, small ? 80 : 128), new THREE.ShaderMaterial({
    uniforms: globeU,
    vertexShader: '#include <common>\n#include <logdepthbuf_pars_vertex>\nvarying vec3 vN; varying vec3 vW;\n' +
      'void main(){ vN = normalize(position); vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w;\n#include <logdepthbuf_vertex>\n}',
    fragmentShader: '#include <logdepthbuf_pars_fragment>\n' + GLSL + '\nuniform vec3 uSun; uniform float uWarm; uniform float uRim; uniform vec3 uSkyCol; varying vec3 vN; varying vec3 vW;\n' +
      'void main(){\n#include <logdepthbuf_fragment>\n' +
      ' vec3 n = normalize(vN), V = normalize(cameraPosition - vW);\n' +
      ' float e = fbm(n * 2.4 + vec3(3.1, 0.0, 1.7) + 0.4 * fbm(n * 6.0));\n' +
      // Where you stand (the top) is coast: land to the south, water to the north.
      ' float landM = smoothstep(0.47, 0.5, e);\n' +
      // Round about you, the coast runs north-south: the Pacific to the west.
      ' landM = mix(landM, smoothstep(-0.004, 0.004, n.z + 0.03 * (e - 0.5)), smoothstep(0.96, 0.985, n.y));\n' +
      ' float ice = smoothstep(0.82, 0.9, abs(n.y) + 0.05 * n3(n * 20.0)) * step(n.y, 0.0);\n' +
      ' float dry = smoothstep(0.45, 0.7, n3(n * 5.0 + 9.0)) * (1.0 - abs(n.y));\n' +
      ' vec3 ocean = mix(vec3(0.008, 0.025, 0.08), vec3(0.02, 0.07, 0.15), smoothstep(0.4, 0.5, e));\n' +
      ' vec3 ground = mix(vec3(0.045, 0.075, 0.04), vec3(0.24, 0.2, 0.13), dry);\n' +
      ' vec3 base = mix(mix(ocean, ground, landM), vec3(0.85, 0.88, 0.92), ice);\n' +
      ' float cl = smoothstep(0.56, 0.74, fbm(n * 14.0 + vec3(0.0, 4.0, 2.0) + 0.6 * fbm(n * 3.0)));\n' +
      ' float l = dot(n, normalize(uSun)), day = smoothstep(-0.04, 0.3, l), dusk = exp(-(l + 0.03) * (l + 0.03) * 400.0);\n' +
      ' vec3 sunC = mix(vec3(1.0, 0.55, 0.3), vec3(1.0, 0.96, 0.9), smoothstep(0.0, 0.3, l));\n' +
      ' vec3 c = mix(base, vec3(0.8), cl * 0.7) * sunC * day * 1.05;\n' +
      ' vec3 h = normalize(normalize(uSun) + V); c += vec3(1.0, 0.85, 0.6) * pow(max(dot(n, h), 0.0), 80.0) * (1.0 - landM) * (1.0 - cl) * day * 0.8;\n' +
      // Low down, the sea mirrors the evening sky, as the strait does.
      ' c += uSkyCol * pow(1.0 - max(dot(n, V), 0.0), 6.0) * (1.0 - landM) * (1.0 - uRim);\n' +
      // Twilight: a warm band along the terminator.
      ' c += vec3(1.0, 0.5, 0.36) * dusk * (0.02 + 0.03 * uWarm) * (1.0 - cl * 0.5);\n' +
      // Towns on the night side.
      ' float town = landM * (1.0 - ice) * smoothstep(0.45, 0.65, n3(n * 30.0)) * (dots(n * 900.0, 0.35) + 0.6 * dots(n * 3100.0, 0.3));\n' +
      ' c += vec3(1.0, 0.72, 0.4) * town * (1.0 - smoothstep(-0.1, 0.05, l)) * (1.8 + 1.0 * uWarm) * (1.0 - cl * 0.8);\n' +
      ' float rim = pow(1.0 - max(dot(n, V), 0.0), 5.0);\n' +
      ' c += mix(vec3(1.0, 0.5, 0.3), vec3(0.35, 0.6, 1.0), smoothstep(-0.05, 0.2, l)) * rim * smoothstep(-0.15, 0.25, l) * 0.8 * uRim;\n' +
      ' gl_FragColor = vec4(c, 1.0);\n#include <tonemapping_fragment>\n#include <colorspace_fragment>\n}'
  }));
  globe.position.set(0, -R, 0);
  globe.frustumCulled = false;
  world.add(globe);
  var atmo = new THREE.Mesh(new THREE.SphereGeometry(R * 1.009, 160, 80), new THREE.ShaderMaterial({
    side: THREE.BackSide, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uSun: { value: SUN }, uAmt: { value: 0 } },
    vertexShader: '#include <common>\n#include <logdepthbuf_pars_vertex>\nvarying vec3 vN; varying vec3 vW;\n' +
      'void main(){ vN = normalize(position); vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w;\n#include <logdepthbuf_vertex>\n}',
    fragmentShader: '#include <logdepthbuf_pars_fragment>\nuniform vec3 uSun; uniform float uAmt; varying vec3 vN; varying vec3 vW;\n' +
      'void main(){\n#include <logdepthbuf_fragment>\n vec3 n = normalize(vN), v = normalize(cameraPosition - vW);\n' +
      ' float rim = pow(1.0 - abs(dot(n, v)), 1.6), l = dot(n, normalize(uSun));\n' +
      ' vec3 col = mix(vec3(1.0, 0.5, 0.3), vec3(0.35, 0.6, 1.0), smoothstep(-0.05, 0.35, l));\n' +
      ' float a = rim * smoothstep(-0.35, 0.25, l) * uAmt;\n gl_FragColor = vec4(col * a * 1.3, a);\n#include <colorspace_fragment>\n}'
  }));
  atmo.position.copy(globe.position);
  atmo.frustumCulled = false;
  world.add(atmo);

  var tmp = new THREE.Color(), H = 800, dpr = 1, portrait = false;
  var TOP = new THREE.Color('#16204a'), MID = new THREE.Color('#4a5488'), HOR = new THREE.Color('#e0a888');
  var TOP_W = new THREE.Color('#2a2452'), MID_W = new THREE.Color('#9a6a8a'), HOR_W = new THREE.Color('#ffc890');

  function frame(f) {
    var row = f.row, alt = row[0], warm = row[1], yaw = row[4], pitch = row[5], drift = row[6], windows = row[7];
    var h = Math.pow(10, alt), high = smooth(3.5, 5.5, alt), space = smooth(4.3, 5.6, alt);

    // Straight up from the beach. High up, keep the horizon a little above centre.
    var dip = Math.acos(R / (R + h));
    camera.position.set(EYE.x, 0.4 + h + Math.sin(f.time * 0.5) * 0.02 * (1 - high), -drift);
    camera.rotation.set(0, 0, 0);
    // Portrait screens are narrow: turn a little towards the house.
    camera.rotateY(yaw + (portrait ? 0.14 : 0) * (1 - high) - f.mx * 0.1 * (1 - high));
    camera.rotateX(pitch - high * (dip + 0.1) - f.my * 0.05 * (1 - high));
    camera.near = clamp(h * 0.05, 0.2, 2e4);
    camera.updateProjectionMatrix();
    sky.position.copy(camera.position);
    sky.scale.setScalar(Math.max(1, camera.near * 12 / 1500));
    starGroup.position.copy(camera.position);
    mirror.visible = landMirror.visible = alt < 2.2;

    // Warmth suffuses the sky and water; high up the sky goes to space.
    dome.uniforms.top.value.copy(TOP).lerp(TOP_W, warm);
    dome.uniforms.mid.value.copy(MID).lerp(MID_W, warm);
    dome.uniforms.horizon.value.copy(HOR).lerp(HOR_W, warm);
    dome.uniforms.sunColor.value.set('#ffb27a').multiplyScalar(0.7 + warm * 0.6);
    dome.uniforms.dark.value = space;
    world.fog.color.copy(dome.uniforms.horizon.value).lerp(dome.uniforms.mid.value, 0.55).multiplyScalar(0.75);
    world.fog.density = 0.00011 * (1 - smooth(2.6, 4.3, alt));
    gl.setClearColor(tmp.copy(world.fog.color).multiplyScalar(1 - space));
    hemi.color.set('#8a90c8').lerp(tmp.set('#e8b0a0'), warm);
    hemi.intensity = 1.2 + warm * 0.5;
    glow.intensity = 0.6 + warm * 0.8;
    gl.toneMappingExposure = 1 + warm * 0.08;
    waterU.uTime.value = f.time;
    waterU.uFade.value = 1 - smooth(3.9, 4.3, alt);
    water.visible = landMesh.visible = firs.visible = house.visible = alt < 4.3;

    starMat.uniforms.uAmt.value = lerp(0.35 + warm * 0.1, 1, high);
    starMat.uniforms.uLow.value = space;
    starMat.uniforms.uScale.value = dpr * (H / 800 * 0.5 + 0.6);

    // Windows, porch and the lights along the shores.
    var flick = 0.95 + 0.05 * Math.sin(f.time * 5.3) * Math.sin(f.time * 2.1);
    winMat.color.set('#5a3a20').lerp(tmp.set('#ffd29a'), windows);
    for (var mw = 0; mw < mirrorWins.length; mw++) mirrorWins[mw].color.copy(winMat.color).multiplyScalar(0.7);
    porch.material.opacity = windows * flick * (0.6 + warm * 0.4);
    lamp.intensity = windows * (40 + warm * 30) * flick;
    lightMat.uniforms.uOn.value = warm;
    lightMat.uniforms.uTime.value = f.time;
    lightMat.uniforms.uScale.value = dpr;
    lightMat.uniforms.uFade.value = 1 - smooth(5.2, 6.0, alt);

    globe.visible = alt > 2.3;
    globeU.uWarm.value = warm;
    globeU.uRim.value = smooth(5.0, 6.0, alt);
    globeU.uSkyCol.value.copy(dome.uniforms.horizon.value).multiplyScalar(0.3 * (1 - space));
    atmo.material.uniforms.uAmt.value = smooth(5.0, 5.9, alt);
    atmo.visible = alt > 5.0;

    gl.render(world, camera);
  }

  return {
    resize: function (w, hh, d) { H = hh; portrait = w < hh; dpr = Math.min(d, small ? 1.5 : 1.75); fitCamera(gl, camera, w, hh, d, small); },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('beloved-earth', {
  renderer: renderer3d,
  maxLines: 3,
  scrim: 0.6,
  align: ['right', 'right'],
  // Panels: 0 "And did you get what you wanted ... I did.", 1 "And what did
  // you want? To call myself beloved ... on the earth."
  keys: function (T) {
    function at(i, d) { return T.start(i) + d; }   // d units into panel i (0..1.6)
    //   unit          alt   warm  -  wind  yaw    pitch  drift windows
    return [
      [0,              0.18, 0.00, 0, 0.05, -0.20, 0.03,  0,   0.25],
      [0.7,            0.18, 0.00, 0, 0.05, -0.20, 0.03,  2,   0.25],
      [at(0, 0.6),     0.18, 0.00, 0, 0.05, -0.17, 0.02,  7,   0.25],  // "did you get what you wanted from this life"
      [at(0, 0.95),    0.18, 0.00, 0, 0.05, -0.15, 0.02,  9,   0.30],
      [at(0, 1.2),     0.18, 0.05, 0, 0.05, -0.14, 0.02, 10,   1.00],  // "I did."
      [at(1, 0.3),     0.18, 0.10, 0, 0.05, -0.16, 0.03, 12,   1.00],  // "And what did you want?"
      [at(1, 0.8),     0.18, 0.55, 0, 0.05, -0.18, 0.05, 14,   1.00],  // "To call myself beloved"
      [at(1, 1.1),     0.40, 1.00, 0, 0.05, -0.20, 0.02, 15,   1.00],  // "beloved on the earth"
      [at(1, 1.6),     2.30, 1.00, 0, 0.05, -0.32, -0.30, 16,  1.00],
      [T.total - 0.8,  4.40, 1.00, 0, 0.05, -0.80, -0.15, 16,  1.00],  // the curve of the earth, turning north
      [T.total,        6.45, 1.00, 0, 0.05, -1.40, 0.00, 16,   1.00]   // the whole earth: day, dusk, lit towns
    ];
  },
  sound: {
    src: '/audio/ocean.mp3',
    label: 'Play the quiet shore',
    volume: function (row) { return 0.09 * (1 - smooth(1.5, 3.5, row[0])); },
    cues: [{ stanza: 1, at: 1.1, play: swell }]
  }
});
