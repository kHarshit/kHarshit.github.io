/*
 * Scene for "Phenomenal Woman" (Maya Angelou): a field of sunflowers at
 * dusk, and a golden light that walks through it like a small sun.
 *
 * I   The heads hang asleep under a violet dusk. A golden light comes
 *     striding through the field, and every head it passes lifts and turns
 *     to follow it ("the stride of my step"). On the refrain it flares and
 *     every head in sight swings round to face it.
 * II  "I walk into a room just as cool as you please": it glides close past
 *     you and "the fellows stand or fall down on their knees", some flowers
 *     stretching tall, some bowing to the ground; then "a hive of honey
 *     bees", golden motes swarm round it. "The fire in my eyes, and the
 *     flash of my teeth": sparks and a flare; it swings, and sparks spring
 *     up where it treads ("the joy in my feet").
 * III "My inner mystery ... they still can't see": a veil of mist rolls in
 *     and the light glows inside it, half hidden. "The sun of my smile": it
 *     rises in an arch like the sun, the mist burns off, the field turns gold.
 * IV  "Now you understand just why my head's not bowed": golden morning,
 *     every head in the field held high and turned to it, and you rise above
 *     the field.
 *
 * The sunflowers are one instanced mesh; a vertex shader turns each head to
 * the light, bows it, and lights the petals near it. Distant heads are a
 * shader point field. Stanza IV is always the last two panels (the text has
 * a stray stanza break in III, so the panel count may change).
 * Columns: [unit, camZ, camY, pollen, wind, yaw, pitch, orbX, orbY, orbZ,
 *           radius, all, bow, swarm, sparks, mist, dawn, flare, proud]
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, broadleafGeometry, softSprite, skyDome, starField,
         terrain, scatter, particleField, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

var STEM = 1.55;                       // stem height (m) before each plant's own scale
var FIELD = { x: 34, near: 22, far: -44 };   // the instanced part of the field

function ground(x, z) {
  var d = Math.hypot(x * 0.8, z + 60);
  return 0.18 * Math.sin(x * 0.13) * Math.cos(z * 0.09) + 0.1 * Math.sin(x * 0.41 + z * 0.27) +
         smooth(150, 430, d) * 42 * (0.55 + 0.45 * Math.sin(Math.atan2(z + 60, x) * 3 + 1));
}

// ── Geometry ─────────────────────────────────────────────────────────────
// Concatenate tinted geometries, tagging each with `part` (0 stem, 1 head).
function mergeParts(list) {
  var total = 0;
  list.forEach(function (p) { total += p.geo.attributes.position.count; });
  var out = new THREE.BufferGeometry(), part = new Float32Array(total), o = 0;
  ['position', 'normal', 'color'].forEach(function (name) {
    var arr = new Float32Array(total * 3), k = 0;
    list.forEach(function (p) { arr.set(p.geo.attributes[name].array, k); k += p.geo.attributes[name].array.length; });
    out.setAttribute(name, new THREE.BufferAttribute(arr, 3));
  });
  list.forEach(function (p) { var n = p.geo.attributes.position.count; part.fill(p.part, o, o + n); o += n; p.geo.dispose(); });
  out.setAttribute('aPart', new THREE.BufferAttribute(part, 1));
  return out;
}

// A star of petals facing +z, tips cupped forward.
function petalStar(rOut, rIn, points, cup) {
  var g = new THREE.CircleGeometry(rOut, points * 2), p = g.attributes.position;
  for (var i = 1; i < p.count; i++) {
    var tip = i % 2 === 1, s = tip ? 1 : rIn / rOut;
    p.setXYZ(i, p.getX(i) * s, p.getY(i) * s, tip ? cup : 0);
  }
  g.computeVertexNormals();
  return g;
}

// One sunflower, about 1.8 m: stem and leaves (part 0) and a head built
// round the origin facing +z (part 1), which the shader turns and sets on
// top of the stem.
function sunflowerGeometry(r) {
  var parts = [];
  function add(geo, color, part) { parts.push({ geo: tinted(geo, color), part: part }); }
  add(new THREE.CylinderGeometry(0.016, 0.027, STEM, 5, 1, true).translate(0, STEM / 2, 0), '#4c6b28', 0);
  [0.42, 0.78, 1.12].forEach(function (h, k) {
    add(new THREE.CircleGeometry(0.14, 6).scale(0.62, 1, 1).translate(0, 0.14, 0)
      .rotateX(Math.PI / 2 + 0.4).rotateY(k * 2.4 + r()).translate(0, h, 0), '#567a2c', 0);
  });
  add(petalStar(0.26, 0.1, 12, 0.05), '#f8c41c', 1);
  add(petalStar(0.23, 0.1, 12, 0.035).rotateZ(Math.PI / 12).translate(0, 0, -0.012), '#e79a12', 1);
  var disk = new THREE.CircleGeometry(0.115, 12);
  disk.attributes.position.setZ(0, 0.035);                 // a shallow dome of seeds
  disk.computeVertexNormals();
  add(disk.translate(0, 0, 0.012), '#5a3614', 1);
  add(new THREE.CircleGeometry(0.125, 10).rotateY(Math.PI).translate(0, 0, -0.024), '#4a6a26', 1);
  return mergeParts(parts);
}

// Thin radiant rays round the light, for its flares.
function raysTexture(r) {
  var c = document.createElement('canvas');
  c.width = c.height = 256;
  var x = c.getContext('2d');
  x.translate(128, 128);
  for (var i = 0; i < 28; i++) {
    var a = i / 28 * Math.PI * 2 + r() * 0.1, len = 70 + r() * 58, w = 0.012 + r() * 0.02;
    var g = x.createLinearGradient(0, 0, Math.cos(a) * len, Math.sin(a) * len);
    g.addColorStop(0, 'rgba(255,236,180,0.9)');
    g.addColorStop(1, 'rgba(255,200,110,0)');
    x.fillStyle = g;
    x.beginPath();
    x.moveTo(0, 0);
    x.lineTo(Math.cos(a - w) * len, Math.sin(a - w) * len);
    x.lineTo(Math.cos(a + w) * len, Math.sin(a + w) * len);
    x.fill();
  }
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// Soft, lumpy mist: overlapping blobs on a canvas.
function mistTexture(r) {
  var c = document.createElement('canvas');
  c.width = 256; c.height = 128;
  var x = c.getContext('2d');
  for (var i = 0; i < 40; i++) {
    var px = 40 + r() * 176, py = 40 + r() * 48, rad = 18 + r() * 34, g = x.createRadialGradient(px, py, 0, px, py, rad);
    g.addColorStop(0, 'rgba(255,255,255,0.22)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    x.fillStyle = g;
    x.fillRect(0, 0, 256, 128);
  }
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// ── Synthesised sound cues ───────────────────────────────────────────────
// A small swarm: detuned sawtooth drones, wobbling, through a band-pass.
function buzz(ac, out) {
  var t = ac.currentTime, g = ac.createGain(), bp = ac.createBiquadFilter();
  bp.type = 'bandpass';
  bp.frequency.value = 850;
  bp.Q.value = 1.4;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.06, t + 0.5);
  g.gain.setValueAtTime(0.06, t + 1.8);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 3);
  bp.connect(g);
  g.connect(out);
  [208, 231, 247].forEach(function (f, k) {
    var o = ac.createOscillator(), lfo = ac.createOscillator(), lg = ac.createGain();
    o.type = 'sawtooth';
    o.frequency.value = f;
    lfo.frequency.value = 5 + k * 2.3;
    lg.gain.value = 9 + k * 4;
    lfo.connect(lg);
    lg.connect(o.frequency);
    o.connect(bp);
    o.start(t); lfo.start(t);
    o.stop(t + 3.1); lfo.stop(t + 3.1);
  });
}

// A bright shimmer for a flare: high sine partials, staggered.
function shimmer(ac, out) {
  var now = ac.currentTime;
  [1568, 2093, 2637, 3136].forEach(function (f, k) {
    var t = now + k * 0.09, o = ac.createOscillator(), g = ac.createGain();
    o.type = 'sine';
    o.frequency.value = f;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.035, t + 0.05);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 2.2);
    o.connect(g); g.connect(out);
    o.start(t); o.stop(t + 2.3);
  });
}

// Sunrise: a warm major chord swelling and fading.
function sunrise(ac, out) {
  var t = ac.currentTime, lp = ac.createBiquadFilter();
  lp.type = 'lowpass';
  lp.frequency.value = 1400;
  lp.connect(out);
  [261.6, 329.6, 392, 523.3].forEach(function (f) {
    var o = ac.createOscillator(), g = ac.createGain();
    o.type = 'triangle';
    o.frequency.value = f;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.045, t + 2);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 6.5);
    o.connect(g); g.connect(lp);
    o.start(t); o.stop(t + 6.6);
  });
}

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(2019);
  var gl = makeRenderer(canvas, { clear: '#1a1630' });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#2a2040', 0.006);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.05, 3000);
  camera.rotation.order = 'YXZ';

  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#060920', mid: '#191a44', horizon: '#4a3050', sun: '#ffcf80' }, 1500);
  sky.add(dome.mesh);
  var stars = starField(r, small ? 1500 : 3000, 1300, 0.04, 1.5);
  sky.add(stars);

  var hemi = new THREE.HemisphereLight('#7a86c0', '#1a160c', 0.35);
  var sun = new THREE.DirectionalLight('#ffb060', 0);
  world.add(hemi, sun, sun.target);

  // Ground: dark soil between the rows, rolling up into hills and woods.
  var tmp = new THREE.Color(), soil = new THREE.Color('#2c2a16'), farSoil = new THREE.Color('#4a4a1c'), hillC = new THREE.Color('#3e4a22');
  world.add(terrain(1400, small ? 120 : 180, 0, -200, ground, new THREE.MeshLambertMaterial({ vertexColors: true }),
    function (x, z, y) { return tmp.copy(soil).lerp(farSoil, smooth(-30, -60, z)).lerp(hillC, smooth(2, 14, y)); }));

  var treeMat = new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true });
  var trees = new THREE.InstancedMesh(broadleafGeometry(rng(4), '#3a2e22'), treeMat, small ? 180 : 360), up = new THREE.Vector3(0, 1, 0);
  scatter(trees, 6000, function (i, p, q, s, c) {
    var a = -Math.PI * (0.1 + r() * 0.8), d = 230 + r() * 160, x = Math.cos(a) * d * 1.25, z = -60 + Math.sin(a) * d;
    if (r() > 0.5 + 0.5 * Math.sin(a * 7)) return false;   // in copses
    p.set(x, ground(x, z) - 0.5, z);
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar(2.2 + r() * 1.6);
    c.setHSL(0.24 + r() * 0.06, 0.35, 0.22 + r() * 0.1);
  });
  world.add(trees);

  // ── The sunflowers ─────────────────────────────────────────────────────
  var U = {
    uLight: { value: new THREE.Vector3() }, uRadius: { value: 0 }, uAll: { value: 0 }, uBow: { value: 0 },
    uProud: { value: 0 }, uTime: { value: 0 }, uWind: { value: 0.2 }, uGlow: { value: 1 },
    uGlowColor: { value: new THREE.Color('#ffc850') }, uBack: { value: 0 }
  };
  var flowerMat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
  flowerMat.onBeforeCompile = function (sh) {
    Object.assign(sh.uniforms, U);
    sh.vertexShader = 'attribute float aPart; attribute vec4 aSeed;\n' +
      'uniform vec3 uLight; uniform float uRadius; uniform float uAll; uniform float uBow; uniform float uProud;\n' +
      'uniform float uTime; uniform float uWind; uniform float uGlow; varying float vGlow;\n' +
      'mat3 rotY(float a){ float c = cos(a), s = sin(a); return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c); }\n' +
      'mat3 rotX(float a){ float c = cos(a), s = sin(a); return mat3(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c); }\n' +
      sh.vertexShader
      .replace('#include <beginnormal_vertex>',
        'vec3 objectNormal = vec3(normal);\n' +
        'float STEM = ' + STEM.toFixed(3) + ';\n' +
        'vec3 fIp = vec3(instanceMatrix[3][0], instanceMatrix[3][1], instanceMatrix[3][2]);\n' +
        'float fSc = length(vec3(instanceMatrix[0][0], instanceMatrix[0][1], instanceMatrix[0][2]));\n' +
        'vec3 fToL = uLight - (fIp + vec3(0.0, STEM * fSc, 0.0));\n' +
        'float fFlat = length(uLight.xz - fIp.xz);\n' +
        // Awake inside the light's widening reach (each a little late), or all at once.
        'float fAwake = max(uAll, clamp(1.0 - (fFlat - uRadius + aSeed.w * 4.0) / 5.0, 0.0, 1.0) * step(0.01, uRadius));\n' +
        'float fYawL = atan(fToL.x, fToL.z);\n' +
        'float fPitchL = clamp(atan(fToL.y, length(fToL.xz)), -0.5, 1.15);\n' +
        'float fDy = mod(fYawL - aSeed.z + 3.14159, 6.28318) - 3.14159;\n' +
        'float fYaw = aSeed.z + fDy * fAwake;\n' +
        // Near the light some fall to their knees, the rest stand tall.
        'float fNear = smoothstep(10.0, 3.0, fFlat) * uBow;\n' +
        'float fKneel = step(0.45, aSeed.x) * fNear, fStand = (1.0 - step(0.45, aSeed.x)) * fNear;\n' +
        'float fPitch = mix(-1.0 + aSeed.y * 0.3, fPitchL, fAwake) - fKneel * 1.2 + fStand * 0.2 + uProud * 0.18;\n' +
        'vec2 fDir = fToL.xz / max(length(fToL.xz), 0.001);\n' +
        'float fSway = sin(uTime * 1.3 + aSeed.y * 6.28) * (0.015 + 0.05 * uWind);\n' +
        'vec2 fBend = fDir * fKneel * 0.5 + vec2(fSway, fSway * 0.6);\n' +
        'float fLift = 1.0 + fStand * 0.1 + uProud * 0.05;\n' +
        'mat3 fRot = rotY(fYaw) * rotX(-fPitch);\n' +
        'if (aPart > 0.5) objectNormal = fRot * objectNormal;\n' +
        'float fD2 = dot(fToL, fToL);\n' +
        'vGlow = uGlow * (0.15 + 0.85 * fAwake) / (1.0 + fD2 * 0.035);\n')
      .replace('#include <begin_vertex>',
        'vec3 transformed = vec3(position);\n' +
        'float fDrop = dot(fBend, fBend) * STEM * 0.5;\n' +
        'if (aPart > 0.5) {\n' +
        '  transformed = vec3(fBend.x * STEM, STEM * fLift - fDrop, fBend.y * STEM) + fRot * (position + vec3(0.0, 0.0, 0.05));\n' +
        '} else {\n' +
        '  float fh = position.y / STEM;\n' +
        '  transformed.y = position.y * fLift - fDrop * fh * fh;\n' +
        '  transformed.xz += fBend * STEM * fh * fh;\n' +
        '}\n');
    // Petals and leaves seen from behind glow with the light coming through.
    sh.fragmentShader = 'uniform vec3 uGlowColor; uniform float uBack; varying float vGlow;\n' + sh.fragmentShader
      .replace('#include <emissivemap_fragment>',
        '#include <emissivemap_fragment>\n totalEmissiveRadiance += diffuseColor.rgb * uGlowColor * vGlow;\n' +
        ' if (!gl_FrontFacing) totalEmissiveRadiance += diffuseColor.rgb * uBack;');
  };

  var flowerGeo = sunflowerGeometry(rng(11));
  var NF = small ? 2600 : 6500, seeds = new Float32Array(NF * 4);
  var flowers = new THREE.InstancedMesh(flowerGeo, flowerMat, NF);
  var xw = small ? 20 : FIELD.x;
  scatter(flowers, NF * 6, function (i, p, q, s, c) {
    // Rows running away from you, 0.9 m apart, thinning with distance.
    var row = Math.floor((r() - 0.5) * 2 * xw / 0.9), x = (row + 0.5) * 0.9 + (r() - 0.5) * 0.25;
    var z = FIELD.far + r() * (FIELD.near - FIELD.far), d = Math.hypot(x, z - 14);
    if (r() > 1.2 - d / 95 * (small ? 1.3 : 1)) return false;
    if (Math.abs(x) < 1.1 && z > 6) return false;            // your own lane
    p.set(x, ground(x, z), z);
    q.identity();                                   // the shader needs the plants unrotated
    s.setScalar(0.82 + r() * 0.38);
    c.setRGB(0.85 + r() * 0.15, 0.8 + r() * 0.2, 0.75 + r() * 0.25);
    seeds[i * 4] = r(); seeds[i * 4 + 1] = r(); seeds[i * 4 + 2] = 1.4 + (r() - 0.5) * 1.2; seeds[i * 4 + 3] = r();
  });
  flowerGeo.setAttribute('aSeed', new THREE.InstancedBufferAttribute(seeds, 4));
  flowers.frustumCulled = false;
  world.add(flowers);

  // The rest of the field, out to the hills: heads as points, golden where
  // they face you and the light reaches them.
  var far = [], farSize = [];
  for (var k = 0; k < (small ? 18000 : 45000); k++) {
    var fz = 22 - Math.pow(r(), 0.8) * 430, half = 40 + Math.max(0, -fz) * 1.1, fx = (r() - 0.5) * 2 * half;
    if (Math.abs(fx) < xw - 1 && fz > FIELD.far + 1) continue;
    if (fz > FIELD.far - 12 && r() > (FIELD.far + 1 - fz) / 13 + 0.15) continue;   // feather the seam
    var gy = ground(fx, fz);
    if (gy > 9) continue;
    far.push(fx, gy + 1.6 + r() * 0.4, fz);
    farSize.push(0.45 + r() * 0.25);
  }
  var farGeo = new THREE.BufferGeometry();
  farGeo.setAttribute('position', new THREE.Float32BufferAttribute(far, 3));
  farGeo.setAttribute('aSize', new THREE.Float32BufferAttribute(farSize, 1));
  var farMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false,
    uniforms: { uLight: U.uLight, uAll: U.uAll, uRadius: U.uRadius, uDawn: { value: 0 }, uScale: { value: 400 },
                uFog: { value: new THREE.Color() }, uFogD: { value: 0.006 }, uGlow: U.uGlow },
    vertexShader: 'attribute float aSize; uniform vec3 uLight; uniform float uAll; uniform float uRadius; uniform float uDawn;\n' +
      'uniform float uScale; uniform float uGlow; varying vec3 vCol; varying float vDepth;\n' +
      'void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;\n' +
      ' gl_PointSize = max(aSize * uScale / -mv.z, 1.0);\n' +
      ' vec3 toL = normalize(uLight - position), toC = normalize(cameraPosition - position);\n' +
      ' float awake = max(uAll, step(length(uLight.xz - position.xz), uRadius));\n' +
      ' float face = mix(0.2, 1.0, awake * smoothstep(-0.3, 0.7, dot(toL.xz, toC.xz)));\n' +
      ' float dl = length(uLight - position);\n' +
      ' float lit = clamp(uDawn * 0.75 + uGlow * 25.0 / (30.0 + dl * dl * 0.05), 0.0, 1.0);\n' +
      ' vCol = mix(vec3(0.03, 0.035, 0.03), mix(vec3(0.16, 0.2, 0.06), vec3(0.85, 0.5, 0.07), face), lit);\n' +
      ' vDepth = -mv.z; }',
    fragmentShader: 'uniform vec3 uFog; uniform float uFogD; varying vec3 vCol; varying float vDepth;\n' +
      'void main(){ vec2 c = gl_PointCoord - 0.5; float d = length(c); if (d > 0.5) discard;\n' +
      ' float fog = 1.0 - exp(-uFogD * uFogD * vDepth * vDepth);\n' +
      ' vec3 col = mix(vCol * mix(0.35, 1.0, smoothstep(0.12, 0.3, d)), uFog, fog);\n' +
      ' gl_FragColor = vec4(col, smoothstep(0.5, 0.35, d));\n #include <colorspace_fragment>\n }'
  });
  var farPts = new THREE.Points(farGeo, farMat);
  farPts.frustumCulled = false;
  world.add(farPts);

  // ── The light ──────────────────────────────────────────────────────────
  var orb = new THREE.Group();
  var glowTex = softSprite('rgba(255,236,190,1)', 'rgba(255,170,60,0)');
  function sprite(map, scale, opacity) {
    var s = new THREE.Sprite(new THREE.SpriteMaterial({ map: map, blending: THREE.AdditiveBlending, depthWrite: false,
                                                        transparent: true, opacity: opacity, fog: false }));
    s.scale.setScalar(scale);
    return s;
  }
  var core = sprite(softSprite('rgba(255,252,236,1)', 'rgba(255,210,120,0)'), 1.1, 1);
  var halo = sprite(glowTex, 6, 0.5);
  var rays = sprite(raysTexture(r), 8, 0);
  var streak = sprite(glowTex, 1, 0);
  var orbLight = new THREE.PointLight('#ffc070', 0, 0, 1.3);
  orb.add(halo, rays, streak, core, orbLight);
  world.add(orb);

  // Honey bees: golden motes orbiting the light, each on its own tilted ring.
  var NB = small ? 220 : 420, bee = [];
  for (k = 0; k < NB; k++) bee.push(0.6 + r() * 2.6, (0.6 + r() * 1.6) * (r() < 0.5 ? -1 : 1), r() * 6.28, r() * 6.28);
  var beeGeo = new THREE.BufferGeometry();
  beeGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(NB * 3), 3));
  beeGeo.setAttribute('aOrb', new THREE.Float32BufferAttribute(bee, 4));
  var beeMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uLight: U.uLight, uTime: U.uTime, uSwarm: { value: 0 }, uScale: { value: 400 } },
    vertexShader: 'attribute vec4 aOrb; uniform vec3 uLight; uniform float uTime; uniform float uSwarm; uniform float uScale; varying float vA;\n' +
      'void main(){ float a = uTime * aOrb.y + aOrb.z, rad = aOrb.x * (0.4 + 0.8 * uSwarm);\n' +
      ' vec3 o = vec3(cos(a) * rad, sin(a * 1.7 + aOrb.w) * rad * 0.45, sin(a) * rad);\n' +
      ' o.xy = mat2(cos(aOrb.w), sin(aOrb.w), -sin(aOrb.w), cos(aOrb.w)) * o.xy * 0.6 + o.xy * 0.4;\n' +
      ' o += 0.06 * vec3(sin(uTime * 23.0 + aOrb.z * 9.0), sin(uTime * 19.0 + aOrb.w * 7.0), cos(uTime * 21.0 + aOrb.z * 5.0));\n' +
      ' vec4 mv = modelViewMatrix * vec4(uLight + o, 1.0); gl_Position = projectionMatrix * mv;\n' +
      ' vA = uSwarm * (0.6 + 0.4 * sin(uTime * 13.0 + aOrb.z * 11.0));\n' +
      ' gl_PointSize = uScale * 0.11 / -mv.z; }',
    fragmentShader: 'varying float vA;\n' +
      'void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;\n' +
      ' float a = smoothstep(0.5, 0.0, d) * vA; gl_FragColor = vec4(vec3(1.0, 0.8, 0.35) * a, a);\n #include <colorspace_fragment>\n }'
  });
  var bees = new THREE.Points(beeGeo, beeMat);
  bees.frustumCulled = false;
  world.add(bees);

  // Sparks: thrown from the light, and springing up from the ground below it.
  var SP = small ? 160 : 320, spPos = new Float32Array(SP * 3), spVel = new Float32Array(SP * 3), spCol = new Float32Array(SP * 3),
      spLife = new Float32Array(SP), spMax = new Float32Array(SP).fill(1), nextSpark = 0, sparkClock = 0;
  for (k = 0; k < SP; k++) spPos[k * 3 + 1] = -50;
  var spGeo = new THREE.BufferGeometry();
  spGeo.setAttribute('position', new THREE.BufferAttribute(spPos, 3));
  spGeo.setAttribute('color', new THREE.BufferAttribute(spCol, 3));
  var sparks = new THREE.Points(spGeo, new THREE.PointsMaterial({ size: 0.12, vertexColors: true, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, map: softSprite('rgba(255,240,200,1)', 'rgba(255,170,60,0)') }));
  sparks.frustumCulled = false;
  world.add(sparks);

  // The veil: banks of mist that glow gold near the light.
  var mistTex = mistTexture(r), mists = [];
  for (k = 0; k < (small ? 16 : 28); k++) {
    var m = new THREE.Sprite(new THREE.SpriteMaterial({ map: mistTex, transparent: true, depthWrite: false, opacity: 0, fog: false }));
    m.userData = { x: (r() - 0.5) * 70, z: -46 + r() * 56, y: 1.2 + r() * 2.2, ph: r() * 6 };
    m.scale.set(18 + r() * 16, 6 + r() * 4, 1);
    world.add(m);
    mists.push(m);
  }

  // Pollen drifting in the morning light.
  var pollen = particleField({ count: small ? 250 : 600, box: [30, 10, 30], fall: [-0.06, 0.06], size: 0.06, color: '#ffe6a0',
                               map: softSprite('rgba(255,240,190,1)', 'rgba(255,230,160,0)'), sway: 0.3, windSpeed: 1 });
  pollen.points.material.blending = THREE.AdditiveBlending;
  world.add(pollen.points);

  // Sky colours: night, sunrise, golden morning.
  var SKY = {
    top: ['#070a22', '#2a3a78', '#bfe0ff'], mid: ['#22204c', '#c47a6a', '#ffe2b0'], horizon: ['#7a4660', '#ff9a5a', '#fff0c8']
  };
  Object.keys(SKY).forEach(function (key) { SKY[key] = SKY[key].map(function (c) { return new THREE.Color(c); }); });
  function sky3(list, d, out) { return out.copy(list[0]).lerp(list[1], smooth(0, 0.5, d)).lerp(list[2], smooth(0.5, 1, d)); }

  var tmp2 = new THREE.Color(), dir = new THREE.Vector3(), H = 800, portrait = false;
  var cold = new THREE.Color('#5a5f80'), gold = new THREE.Color('#ffc870');

  function frame(f) {
    var row = f.row, time = f.time, dt = f.dt;
    var orbX = row[6], orbY = row[7], orbZ = row[8], swarm = row[12], sparkAmt = row[13], mist = row[14], dawn = row[15],
        flare = row[16], proud = row[17];

    // Standing just above the heads; at the end, risen over the field.
    camera.position.set(0, row[1], row[0]);
    // On a phone the verse is centred: look up a little at night so the field and
    // the light sit below it, and draw the light in towards the narrow view.
    camera.rotation.set(row[5] - f.my * 0.06 + (portrait ? 0.2 * (1 - dawn) - 0.05 * dawn : 0), row[4] - f.mx * 0.14, 0);
    sky.position.copy(camera.position);

    // The light walks: a stride in its bob, and a swing when it dances.
    var stride = Math.abs(Math.sin(time * 2.2)) * 0.12 * (1 - dawn);
    orb.position.set((portrait ? orbX * 0.3 : orbX) + Math.sin(time * 0.7) * 0.3, orbY + stride, orbZ);
    U.uLight.value.copy(orb.position);
    U.uRadius.value = row[9];
    U.uAll.value = row[10];
    U.uBow.value = row[11];
    U.uProud.value = proud;
    U.uTime.value = time;
    U.uWind.value = f.wind;

    // Glow: hidden in the mist, then a sun.
    var veil = 1 - mist * 0.7, sunny = smooth(0.2, 0.9, dawn);
    var pulse = 1 + 0.06 * Math.sin(time * 2.6);
    core.scale.setScalar((1.0 + flare * 0.9 + sunny * 3.2) * pulse);
    core.material.opacity = veil;
    halo.scale.setScalar((5 + flare * 4 + mist * 14 + sunny * 10) * pulse);
    halo.material.opacity = 0.45 + mist * 0.5 + sunny * 0.2;
    rays.scale.setScalar(6 + flare * 6 + sunny * 13);
    rays.material.opacity = Math.min(1, flare * 0.9 + sunny * 0.4) * veil;
    rays.material.rotation = time * 0.05;
    streak.scale.set(4 + flare * 22 + sunny * 14, 0.3 + flare * 0.25, 1);
    streak.material.opacity = (flare * 0.7 + sunny * 0.15) * veil;
    U.uBack.value = sunny * 0.45;
    orbLight.intensity = (14 + flare * 30) * (1 - sunny * 0.5) * (0.85 + 0.15 * veil);
    U.uGlow.value = (1 + flare * 1.5) * (1 - sunny * 0.6) * (0.8 + mist * 0.6);

    // Dusk to golden morning.
    var top = sky3(SKY.top, dawn, dome.uniforms.top.value);
    sky3(SKY.mid, dawn, dome.uniforms.mid.value);
    var horizon = sky3(SKY.horizon, dawn, dome.uniforms.horizon.value);
    dir.copy(orb.position).sub(camera.position).normalize();
    dome.uniforms.sunDir.value.copy(dir);
    dome.uniforms.sunColor.value.set('#ffc070').multiplyScalar(sunny * 0.9);
    stars.material.opacity = 0.85 * (1 - smooth(0.1, 0.6, dawn));
    world.fog.color.copy(horizon).lerp(cold, mist * 0.5 * (1 - dawn)).multiplyScalar(0.45 + dawn * 0.45 + mist * 0.3 * (1 - dawn));
    world.fog.density = lerp(0.0055, 0.0032, dawn) + mist * 0.012;
    gl.setClearColor(world.fog.color);
    farMat.uniforms.uFog.value.copy(world.fog.color);
    farMat.uniforms.uFogD.value = world.fog.density;
    farMat.uniforms.uDawn.value = sunny;
    hemi.intensity = 1.0 + dawn * 0.6;
    hemi.color.set('#6a74b8').lerp(tmp.set('#ffe6c0'), dawn);
    hemi.groundColor.set('#1a160c').lerp(tmp.set('#5a4a20'), dawn);
    sun.intensity = sunny * 2.6;
    sun.color.set('#ff9a50').lerp(tmp.set('#fff0d0'), smooth(0.5, 1, dawn));
    // Daylight fill from over your shoulder, so the far faces read in the morning.
    sun.position.set(camera.position.x + 10, camera.position.y + 40, camera.position.z + 30);
    sun.target.position.set(0, 0, -30);
    gl.toneMappingExposure = 1 + dawn * 0.08;

    // Mist banks drift, and glow where the light is.
    for (var i = 0; i < mists.length; i++) {
      var m = mists[i], u = m.userData;
      m.position.set(u.x + Math.sin(time * 0.05 + u.ph) * 4, u.y, u.z);
      var near = 1 / (1 + m.position.distanceToSquared(orb.position) / 220);
      m.material.color.copy(cold).lerp(tmp2.copy(top).lerp(horizon, 0.7), dawn).lerp(gold, Math.min(1, near * 1.6));
      m.material.opacity = mist * (0.55 + 0.25 * Math.sin(time * 0.3 + u.ph));
    }

    // Bees: the swarm closes round the light.
    beeMat.uniforms.uSwarm.value = swarm;
    bees.visible = swarm > 0.01;

    // Sparks: from the light ("the fire in my eyes") and from its feet.
    sparkClock += dt * sparkAmt * (env.reduceMotion ? 60 : 160);
    while (sparkClock > 1) {
      sparkClock -= 1;
      var s = nextSpark++ % SP, j = s * 3, fromFeet = Math.random() < 0.35;
      var th = Math.random() * 6.28, ph = Math.random() * 2 - 1, sp = 1.5 + Math.random() * 3, rr = Math.sqrt(1 - ph * ph);
      if (fromFeet) {
        spPos[j] = orb.position.x + (Math.random() - 0.5) * 2; spPos[j + 1] = ground(orb.position.x, orb.position.z) + 0.3;
        spPos[j + 2] = orb.position.z + (Math.random() - 0.5) * 2;
        spVel[j] = (Math.random() - 0.5) * 0.8; spVel[j + 1] = 2 + Math.random() * 2.5; spVel[j + 2] = (Math.random() - 0.5) * 0.8;
      } else {
        spPos[j] = orb.position.x; spPos[j + 1] = orb.position.y; spPos[j + 2] = orb.position.z;
        spVel[j] = rr * Math.cos(th) * sp; spVel[j + 1] = ph * sp + 1; spVel[j + 2] = rr * Math.sin(th) * sp;
      }
      spLife[s] = spMax[s] = 0.6 + Math.random() * 0.7;
    }
    for (i = 0; i < SP; i++) {
      j = i * 3;
      if (spLife[i] <= 0) { spCol[j] = spCol[j + 1] = spCol[j + 2] = 0; continue; }
      spLife[i] -= dt;
      spVel[j + 1] -= 4 * dt;
      spPos[j] += spVel[j] * dt; spPos[j + 1] += spVel[j + 1] * dt; spPos[j + 2] += spVel[j + 2] * dt;
      var fade = Math.max(spLife[i] / spMax[i], 0);
      spCol[j] = fade; spCol[j + 1] = 0.75 * fade; spCol[j + 2] = 0.35 * fade;
    }
    spGeo.attributes.position.needsUpdate = true;
    spGeo.attributes.color.needsUpdate = true;

    pollen.update({ snow: row[2], wind: f.wind * 0.4, dt: dt, time: time }, camera.position, env.reduceMotion);

    var px = H / (2 * Math.tan(camera.fov * Math.PI / 360));
    farMat.uniforms.uScale.value = px;
    beeMat.uniforms.uScale.value = px;
    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) { H = h * Math.min(dpr, small ? 1.5 : 1.75); portrait = w < h; fitCamera(gl, camera, w, h, dpr, small); },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

var ALIGN = ['left', 'left', 'right', 'right', 'left', 'left', 'right', 'left', 'right'];

PI.register('sunflowers', {
  renderer: renderer3d,
  maxLines: 9,
  align: ALIGN,
  scrim: 0.6,
  accent: '#ffc94a',
  emphasis: /^\W*phenomenal/i,
  // Panels (9 lines at most): I splits before the refrain, II after "a hive
  // of honey bees", III after "they still can't see", IV after "proud".
  // A stray break in III leaves its refrain as a panel of its own, so
  // stanza IV is found from the end.
  keys: function (T) {
    var n = T.count, iv = n - 2, frag = n >= 9 ? 6 : -1;
    function at(i, u) { return T.start(i) + u; }
    //            camZ  camY pollen wind yaw   pitch  orbX orbY  orbZ  rad all bow swarm spark mist dawn flare proud
    var k = [
      [0,          16.0, 2.5, 0.0, 0.20, 0.00, -0.05, 20,  1.9,  -80,  0,  0,  0,  0.0, 0.0, 0.0, 0.00, 0.0, 0],
      [0.7,        15.6, 2.5, 0.0, 0.20, 0.00, -0.05, 18,  1.9,  -70,  3,  0,  0,  0.0, 0.0, 0.0, 0.00, 0.0, 0],
      [at(0, 0.6), 14.8, 2.5, 0.0, 0.20, 0.03, -0.06, 13,  1.9,  -46,  9,  0,  0,  0.0, 0.0, 0.0, 0.00, 0.0, 0],   // "where my secret lies"
      [at(0, 1.3), 13.8, 2.5, 0.0, 0.20, 0.05, -0.07, 10,  1.9,  -27,  14, 0,  0,  0.0, 0.0, 0.0, 0.00, 0.1, 0],   // "the stride of my step"
      [at(1, 0.3), 13.2, 2.5, 0.0, 0.20, 0.05, -0.07, 9,   2.0,  -22,  16, 0,  0,  0.0, 0.0, 0.0, 0.00, 0.1, 0],
      [at(1, 0.6), 13.0, 2.5, 0.0, 0.20, 0.05, -0.07, 8.5, 2.3,  -20,  18, 1,  0,  0.0, 0.0, 0.0, 0.00, 1.0, 0],   // "Phenomenal woman": all heads turn
      [at(1, 1.3), 12.6, 2.5, 0.0, 0.20, 0.04, -0.07, 8,   2.0,  -16,  18, 1,  0,  0.1, 0.0, 0.0, 0.00, 0.2, 0],
      [at(2, 0.45),11.4, 2.5, 0.0, 0.20, 0.00, -0.09, 2.5, 1.9,  -1,   18, 1,  0.3,0.1, 0.0, 0.0, 0.00, 0.1, 0],   // "I walk into a room"
      [at(2, 0.8), 11.2, 2.5, 0.0, 0.20, -0.04,-0.12, -1,  1.9,  4,    18, 1,  1,  0.2, 0.0, 0.0, 0.00, 0.1, 0],   // "fall down on their knees"
      [at(2, 1.15),11.0, 2.5, 0.0, 0.20, -0.06,-0.09, -4.5,2.1,  3,    18, 1,  0.6,1.0, 0.0, 0.0, 0.00, 0.2, 0],   // "a hive of honey bees"
      [at(3, 0.25),11.0, 2.5, 0.0, 0.20, -0.06,-0.08, -5,  2.2,  2,    18, 1,  0.1,0.8, 0.3, 0.0, 0.00, 0.3, 0],
      [at(3, 0.42),11.0, 2.5, 0.0, 0.20, -0.06,-0.08, -5,  2.4,  2,    18, 1,  0,  0.7, 1.0, 0.0, 0.00, 1.0, 0],   // "the fire in my eyes ... the flash of my teeth"
      [at(3, 0.6), 11.0, 2.5, 0.0, 0.20, -0.06,-0.08, -2.5,2.0,  1.5,  18, 1,  0,  0.6, 0.8, 0.0, 0.00, 0.3, 0],   // "the swing in my waist"
      [at(3, 0.78),11.0, 2.5, 0.0, 0.20, -0.06,-0.09, -6,  1.6,  2,    18, 1,  0,  0.6, 1.0, 0.0, 0.00, 0.3, 0],   // "the joy in my feet"
      [at(3, 1.1), 11.0, 2.5, 0.0, 0.20, -0.05,-0.08, -5,  2.2,  1,    18, 1,  0,  0.5, 0.4, 0.0, 0.00, 0.9, 0],   // the refrain
      [at(4, 0.4), 10.5, 2.5, 0.0, 0.20, 0.00, -0.07, 2,   2.2,  -14,  18, 1,  0,  0.2, 0.0, 0.6, 0.00, 0.1, 0],   // "what they see in me"
      [at(4, 0.9), 10.0, 2.5, 0.0, 0.20, 0.03, -0.07, 6,   2.2,  -18,  18, 1,  0,  0.1, 0.0, 1.0, 0.00, 0.0, 0],   // "my inner mystery"
      [at(5, 0.2), 9.8,  2.5, 0.0, 0.20, 0.03, -0.06, 6,   2.6,  -20,  18, 1,  0,  0.1, 0.0, 1.0, 0.05, 0.0, 0],
      [at(5, 0.55),9.5,  2.6, 0.2, 0.20, 0.02, 0.02,  4,   8,    -28,  18, 1,  0,  0.0, 0.0, 0.5, 0.50, 0.2, 0],   // "the sun of my smile"
      [at(5, 1.3), 9.0,  2.8, 0.4, 0.20, 0.00, 0.08,  2,   13,   -34,  18, 1,  0,  0.0, 0.0, 0.1, 0.85, 0.3, 0]    // "the grace of my style"
    ];
    if (frag > 0) k.push([at(frag, 0.6), 9.5, 3.0, 0.5, 0.2, 0.00, 0.10, 1, 18, -36, 18, 1, 0, 0, 0, 0, 1, 0.8, 0.3]);
    k.push([at(iv, 0.5),     11.0, 3.6, 0.6, 0.2, 0.00, 0.09, 0, 24, -36, 18, 1, 0, 0, 0, 0, 1, 0.3, 1]);   // "just why my head's not bowed"
    k.push([at(iv + 1, 0.6), 13.0, 4.8, 0.7, 0.2, 0.00, 0.08, 0, 26, -36, 18, 1, 0, 0, 0, 0, 1, 0.3, 1]);
    k.push([T.total,         14.0, 5.2, 0.7, 0.2, 0.00, 0.08, 0, 26, -36, 18, 1, 0, 0, 0, 0, 1, 0.5, 1]);
    return k;
  },
  sound: {
    src: '/audio/birds.mp3',
    label: 'Play the field at daybreak',
    volume: function (row) { return 0.03 + 0.35 * row[15]; },
    cues: [
      { stanza: 1, at: 0.6, play: shimmer },
      { stanza: 2, at: 1.0, play: buzz },
      { stanza: 3, at: 0.42, play: shimmer },
      { stanza: 5, at: 0.5, play: sunrise }
    ]
  }
});
