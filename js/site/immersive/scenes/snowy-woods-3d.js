/*
 * Scene for "Stopping by Woods on a Snowy Evening" in real 3D (three.js).
 *
 * The camera rides a sleigh track at eye height: along the edge of the woods
 * with the village across a meadow on the left, past a frozen lake on the
 * right, then into deep woods where it slowly goes dark. Terrain, a few
 * thousand instanced pines, the lake, the village and the snowfall are all
 * generated here; nothing is downloaded beyond three.js itself.
 *
 * Loaded by js/site/immersive/engine.js as a module (`?scene=snowy-woods-3d`
 * or `immersive: snowy-woods-3d`). The engine owns the timeline, text and
 * sound; this file only renders. Keyframe columns:
 *   [unit, path progress 0..1, darkness, snowfall, wind, look yaw (radians, + is left)]
 */
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── World layout (metres; the camera looks down -z) ─────────────────────
var PATH = [[0, 48], [0, 30], [2, 12], [4, -10], [8, -32], [10, -50], [8, -70],
            [4, -92], [0, -115], [-2, -140], [0, -165], [0, -190]];
var LAKE = { x: 30, z: -62, r: 18, y: -0.45 };
var VILLAGE = { x: -36, z: -40 };
var DEEP_Z = -100;   // beyond this the woods close in

var curve = new THREE.CatmullRomCurve3(PATH.map(function (p) { return new THREE.Vector3(p[0], 0, p[1]); }));
var pathPts = curve.getSpacedPoints(160);

function distToPath(x, z) {
  var best = 1e9;
  for (var i = 0; i < pathPts.length; i++) {
    var dx = pathPts[i].x - x, dz = pathPts[i].z - z, d = dx * dx + dz * dz;
    if (d < best) best = d;
  }
  return Math.sqrt(best);
}

// Rolling snowfields, flattened along the track, a basin for the lake and a
// ring of mountains far out where the fog turns them into silhouettes.
function height(x, z, dPath) {
  if (dPath === undefined) dPath = distToPath(x, z);
  var h = 1.4 * Math.sin(x * 0.045 + 1.3) * Math.cos(z * 0.038) +
          0.7 * Math.sin(x * 0.11 + z * 0.07) + 0.35 * Math.sin(x * 0.23 - z * 0.19);
  h *= smooth(3, 22, dPath);
  var cx = x, cz = z + 60, rr = Math.sqrt(cx * cx + cz * cz), ang = Math.atan2(cz, cx);
  // 1 - |sin| gives sharp summits rather than rounded humps.
  var ridge = 0.3 + 0.45 * (1 - Math.abs(Math.sin(ang * 4 + 1))) + 0.25 * (1 - Math.abs(Math.sin(ang * 9 + 0.3))) +
              0.08 * Math.sin(ang * 31) + 0.05 * Math.sin(rr * 0.08 + ang * 5);
  h += smooth(170, 340, rr) * 95 * ridge;
  var dl = Math.hypot(x - LAKE.x, z - LAKE.z);
  if (dl < LAKE.r + 8) h = lerp(h, LAKE.y - 0.3, smooth(LAKE.r + 8, LAKE.r - 1, dl));
  return h;
}

// ── Geometry helpers ─────────────────────────────────────────────────────
function tinted(geo, color) {
  geo = geo.index ? geo.toNonIndexed() : geo;
  var c = new THREE.Color(color), n = geo.attributes.position.count, a = new Float32Array(n * 3);
  for (var i = 0; i < n; i++) { a[i * 3] = c.r; a[i * 3 + 1] = c.g; a[i * 3 + 2] = c.b; }
  geo.setAttribute('color', new THREE.BufferAttribute(a, 3));
  return geo;
}

function merge(geos) {
  var total = 0;
  geos.forEach(function (g) { total += g.attributes.position.count; });
  var out = new THREE.BufferGeometry();
  ['position', 'normal', 'color'].forEach(function (name) {
    var arr = new Float32Array(total * 3), o = 0;
    geos.forEach(function (g) { arr.set(g.attributes[name].array, o); o += g.attributes[name].array.length; });
    out.setAttribute(name, new THREE.BufferAttribute(arr, 3));
  });
  geos.forEach(function (g) { g.dispose(); });
  return out;
}

// One low-poly pine, base at the origin, about 5.7 m tall: a trunk, four
// tiers of foliage and a snow cap poking out of the top of each tier.
function pineGeometry() {
  var parts = [tinted(new THREE.CylinderGeometry(0.12, 0.2, 1.2, 5).translate(0, 0.6, 0), '#2a2226')];
  for (var k = 0; k < 4; k++) {
    var r = 1.6 * (1 - k * 0.22), h = 2.2 * (1 - k * 0.12), y = 0.8 + k * 1.15;
    parts.push(tinted(new THREE.ConeGeometry(r, h, 7, 1, true).translate(0, y + h / 2, 0), '#1c2c2c'));
    parts.push(tinted(new THREE.ConeGeometry(r * 0.62, h * 0.42, 7, 1, true).translate(0, y + h * 0.58 + h * 0.21 + 0.02, 0), '#e4ecfa'));
  }
  return merge(parts);
}

function softSprite(inner, outer) {
  var c = document.createElement('canvas');
  c.width = c.height = 64;
  var x = c.getContext('2d'), g = x.createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0, inner);
  g.addColorStop(0.4, inner.replace(/[\d.]+\)$/, '0.55)'));
  g.addColorStop(1, outer);
  x.fillStyle = g;
  x.fillRect(0, 0, 64, 64);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// ── Renderer ─────────────────────────────────────────────────────────────
function renderer3d(canvas, scene, env) {
  // env.capture: rendering stills for the image-layer scene (transparent
  // background, full quality whatever the device).
  var capture = !!env.capture;
  var small = !capture && (window.innerWidth < 800 || !window.matchMedia('(pointer: fine)').matches);
  var r = rng(17);

  var gl = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, powerPreference: 'high-performance',
                                     alpha: capture, preserveDrawingBuffer: capture });
  gl.setClearColor('#03060f', capture ? 0 : 1);
  gl.toneMapping = THREE.ACESFilmicToneMapping;
  gl.outputColorSpace = THREE.SRGBColorSpace;
  gl.shadowMap.enabled = !small;
  gl.shadowMap.type = THREE.PCFSoftShadowMap;

  var world = new THREE.Scene();
  var FOG = new THREE.Color('#25365c'), NIGHT = new THREE.Color('#04060c');
  world.fog = new THREE.Fog(FOG.clone(), 14, 280);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.1, 2000);

  // Sky dome + stars + moon follow the camera so they read as infinitely far.
  var sky = new THREE.Group();
  world.add(sky);
  var skyMat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false,
    uniforms: { top: { value: new THREE.Color('#02040b') }, mid: { value: new THREE.Color('#0d1a3a') },
                horizon: { value: FOG.clone() }, dark: { value: 0 } },
    vertexShader: 'varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
    fragmentShader: 'uniform vec3 top; uniform vec3 mid; uniform vec3 horizon; uniform float dark; varying vec3 vP;' +
      'void main(){ float h = clamp(vP.y, 0.0, 1.0);' +
      ' vec3 c = mix(horizon, mid, smoothstep(0.0, 0.22, h)); c = mix(c, top, smoothstep(0.22, 0.75, h));' +
      ' gl_FragColor = vec4(mix(c, vec3(0.012,0.016,0.03), dark), 1.0); }'
  });
  sky.add(new THREE.Mesh(new THREE.SphereGeometry(1200, 32, 16), skyMat));

  var starPos = [];
  for (var i = 0; i < 1600; i++) {
    var th = r() * Math.PI * 2, ph = Math.acos(0.08 + r() * 0.92), R = 1100;
    starPos.push(R * Math.sin(ph) * Math.cos(th), R * Math.cos(ph), R * Math.sin(ph) * Math.sin(th));
  }
  var starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.Float32BufferAttribute(starPos, 3));
  var starMat = new THREE.PointsMaterial({ color: '#dfe7fa', size: 1.6, sizeAttenuation: false, fog: false, transparent: true, opacity: 0.85, depthWrite: false });
  sky.add(new THREE.Points(starGeo, starMat));

  var moonDir = new THREE.Vector3(0.38, 0.3, -1).normalize();
  var moonDisc = new THREE.Mesh(new THREE.CircleGeometry(16, 48), new THREE.MeshBasicMaterial({ color: '#eef3ff', fog: false }));
  moonDisc.position.copy(moonDir).multiplyScalar(1000);
  moonDisc.lookAt(0, 0, 0);
  sky.add(moonDisc);
  var moonGlow = new THREE.Sprite(new THREE.SpriteMaterial({
    map: softSprite('rgba(190,208,255,0.55)', 'rgba(120,140,200,0)'), fog: false,
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true
  }));
  moonGlow.position.copy(moonDir).multiplyScalar(990);
  moonGlow.scale.setScalar(260);
  sky.add(moonGlow);

  // Moonlight (with shadows that follow the camera) and a cold sky fill.
  var hemi = new THREE.HemisphereLight('#5b70a8', '#1a2238', 1.1);
  world.add(hemi);
  var moon = new THREE.DirectionalLight('#c3d2ff', 2.1);
  moon.castShadow = !small;
  moon.shadow.mapSize.set(2048, 2048);
  moon.shadow.camera.left = moon.shadow.camera.bottom = -45;
  moon.shadow.camera.right = moon.shadow.camera.top = 45;
  moon.shadow.camera.near = 1;
  moon.shadow.camera.far = 260;
  moon.shadow.bias = -0.0006;
  moon.shadow.normalBias = 0.04;
  world.add(moon, moon.target);

  // Terrain.
  var SIZE = 780, SEG = small ? 150 : 220;
  var ground = new THREE.PlaneGeometry(SIZE, SIZE, SEG, SEG).rotateX(-Math.PI / 2).translate(0, 0, -60);
  var gp = ground.attributes.position;
  for (i = 0; i < gp.count; i++) gp.setY(i, height(gp.getX(i), gp.getZ(i)));
  ground.computeVertexNormals();
  var snowMat = new THREE.MeshLambertMaterial({ color: '#d4def2' });
  var groundMesh = new THREE.Mesh(ground, snowMat);
  groundMesh.receiveShadow = true;
  groundMesh.name = 'ground';
  world.add(groundMesh);

  // Frozen lake: dark ice that catches the moon.
  var lake = new THREE.Mesh(new THREE.CircleGeometry(LAKE.r, 72).rotateX(-Math.PI / 2),
    new THREE.MeshPhongMaterial({ color: '#2b3c64', specular: '#b9c9f2', shininess: 140 }));
  lake.position.set(LAKE.x, LAKE.y, LAKE.z);
  lake.receiveShadow = true;
  lake.name = 'ground';
  world.add(lake);

  // Sleigh-runner tracks along the path: two slightly darker grooves.
  var trackPos = [], pts = curve.getSpacedPoints(600);
  [-0.55, 0.55].forEach(function (off) {
    for (var j = 0; j < pts.length - 1; j++) {
      var a = pts[j], b = pts[j + 1], dx = b.x - a.x, dz = b.z - a.z, len = Math.hypot(dx, dz) || 1;
      var nx = -dz / len, nz = dx / len, w = 0.045;
      var ax0 = a.x + nx * (off - w), az0 = a.z + nz * (off - w), ax1 = a.x + nx * (off + w), az1 = a.z + nz * (off + w);
      var bx0 = b.x + nx * (off - w), bz0 = b.z + nz * (off - w), bx1 = b.x + nx * (off + w), bz1 = b.z + nz * (off + w);
      var ya = height(a.x, a.z, 0) + 0.03, yb = height(b.x, b.z, 0) + 0.03;
      trackPos.push(ax0, ya, az0, bx0, yb, bz0, ax1, ya, az1, ax1, ya, az1, bx0, yb, bz0, bx1, yb, bz1);
    }
  });
  var trackGeo = new THREE.BufferGeometry();
  trackGeo.setAttribute('position', new THREE.Float32BufferAttribute(trackPos, 3));
  trackGeo.computeVertexNormals();
  var tracks = new THREE.Mesh(trackGeo, new THREE.MeshLambertMaterial({ color: '#a9b7dc', side: THREE.DoubleSide }));
  tracks.receiveShadow = true;
  tracks.name = 'ground';
  world.add(tracks);

  // Pines, placed by density: open meadow towards the village, a clear
  // shore round the lake, a cleared track, and dense woods past DEEP_Z.
  function density(x, z, d) {
    if (Math.hypot(x - LAKE.x, z - LAKE.z) < LAKE.r + 12) return 0;
    if (Math.hypot(x - VILLAGE.x, z - VILLAGE.z) < 20) return 0;
    var p = 0.5;
    if (x < -4 && x > -85 && z > -64 && z < 40) p = 0.015;     // meadow
    else if (z > 26) p = 0.05;                                 // open field at the start
    else if (z < DEEP_Z) p = 1;                                // deep woods
    var clear = z < DEEP_Z ? 2.6 : 3.6;
    if (d < clear) return 0;
    if (d < clear + 4) p *= 0.3;
    return p;
  }
  var treeGeo = pineGeometry();
  var treeMat = new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true });
  var cap = small ? 3200 : 6500, trees = new THREE.InstancedMesh(treeGeo, treeMat, cap);
  trees.castShadow = trees.receiveShadow = true;
  var m = new THREE.Matrix4(), q = new THREE.Quaternion(), sc = new THREE.Vector3(), pos = new THREE.Vector3();
  var tint = new THREE.Color(), up = new THREE.Vector3(0, 1, 0), count = 0;
  for (var tries = 0; tries < cap * 4 && count < cap; tries++) {
    var x = -170 + r() * 340, z = -270 + r() * 340, d = distToPath(x, z);
    if (r() > density(x, z, d)) continue;
    var s = (z < DEEP_Z ? 1.3 : 0.75) + r() * (z < DEEP_Z ? 1.0 : 1.1);
    pos.set(x, height(x, z, d) - 0.25, z);
    q.setFromAxisAngle(up, r() * Math.PI * 2);
    sc.set(s * (0.85 + r() * 0.3), s, s * (0.85 + r() * 0.3));
    trees.setMatrixAt(count, m.compose(pos, q, sc));
    var v = 0.8 + r() * 0.35;
    trees.setColorAt(count, tint.setRGB(v, v, v * (0.95 + r() * 0.1)));
    count++;
  }
  trees.count = count;
  trees.instanceMatrix.needsUpdate = true;
  if (trees.instanceColor) trees.instanceColor.needsUpdate = true;
  world.add(trees);

  // The village: "His house is in the village though".
  var village = new THREE.Group(), windowMat = new THREE.MeshBasicMaterial({ color: '#ffcf87', fog: false });
  var wallMat = new THREE.MeshLambertMaterial({ color: '#3b3542' }), roofMat = new THREE.MeshLambertMaterial({ color: '#dce5f6' });
  var glowTex = softSprite('rgba(255,196,120,0.9)', 'rgba(255,160,80,0)');
  [[0, 0, 0], [7, -3, 0.4], [-6, -5, -0.3], [3, -11, 0.2], [-2, 7, 0.1]].forEach(function (hdef) {
    var house = new THREE.Group(), w = 4 + r() * 1.5, dpt = 5 + r() * 1.5, hh = 2.8 + r();
    var walls = new THREE.Mesh(new THREE.BoxGeometry(w, hh, dpt), wallMat);
    walls.position.y = hh / 2;
    var roof = new THREE.Mesh(new THREE.ConeGeometry(Math.max(w, dpt) * 0.75, 2.4, 4).rotateY(Math.PI / 4), roofMat);
    roof.scale.set(w / Math.max(w, dpt), 1, dpt / Math.max(w, dpt));
    roof.position.y = hh + 1.2;
    walls.castShadow = roof.castShadow = walls.receiveShadow = roof.receiveShadow = true;
    house.add(walls, roof);
    [-w * 0.25, w * 0.25].forEach(function (wx) {
      var win = new THREE.Mesh(new THREE.PlaneGeometry(0.7, 0.8), windowMat);
      win.position.set(wx, hh * 0.5, dpt / 2 + 0.01);
      var glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0.5, fog: false }));
      glow.position.set(wx, hh * 0.5, dpt / 2 + 0.3);
      glow.scale.setScalar(3.2);
      house.add(win, glow);
    });
    var hx = VILLAGE.x + hdef[0], hz = VILLAGE.z + hdef[1];
    house.position.set(hx, height(hx, hz) - 0.1, hz);
    house.rotation.y = 0.35 + hdef[2];
    village.add(house);
  });
  var hearth = new THREE.PointLight('#ffb36b', 40, 30, 1.6);
  hearth.position.set(VILLAGE.x, 3, VILLAGE.z + 4);
  village.add(hearth);
  world.add(village);

  // Snowfall in a box that travels with the camera, wrapped per axis.
  var N = small ? 3500 : 9000, BX = 40, BY = 22, BZ = 44;
  var flakes = new Float32Array(N * 3), speed = new Float32Array(N), phase = new Float32Array(N);
  for (i = 0; i < N; i++) {
    flakes[i * 3] = (Math.random() - 0.5) * BX;
    flakes[i * 3 + 1] = Math.random() * BY;
    flakes[i * 3 + 2] = (Math.random() - 0.5) * BZ;
    speed[i] = 0.7 + Math.random() * 0.8;
    phase[i] = Math.random() * 6.28;
  }
  var snowGeo = new THREE.BufferGeometry();
  snowGeo.setAttribute('position', new THREE.BufferAttribute(flakes, 3));
  var snowPts = new THREE.Points(snowGeo, new THREE.PointsMaterial({
    color: '#f2f6ff', size: 0.14, map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)'),
    transparent: true, depthWrite: false, sizeAttenuation: true
  }));
  snowPts.frustumCulled = false;
  world.add(snowPts);

  // ── Per-frame ──────────────────────────────────────────────────────────
  var look = new THREE.Vector3(), tangent = new THREE.Vector3(), tmp = new THREE.Color();

  function wrap(v, c, size) { return c - size / 2 + ((((v - c + size / 2) % size) + size) % size); }

  function frame(f) {
    var t = clamp(f.cam, 0, 1), yaw = f.row[4] || 0, dark = f.dark;

    // Ride the track at eye height, looking a little way ahead.
    curve.getPointAt(t, camera.position);
    camera.position.y = height(camera.position.x, camera.position.z, 0) + 1.75 + Math.sin(f.time * 1.1) * 0.025;
    var ahead = t + 0.035;
    if (ahead <= 1) curve.getPointAt(ahead, look);
    else { curve.getTangentAt(1, tangent); curve.getPointAt(1, look).addScaledVector(tangent, (ahead - 1) * curve.getLength()); }
    look.y = height(look.x, look.z, 0) + 1.5;
    camera.lookAt(look);
    camera.rotateY(yaw - f.mx * 0.16);
    camera.rotateX(-f.my * 0.07);

    sky.position.copy(camera.position);
    moon.position.copy(camera.position).addScaledVector(moonDir, 120);
    curve.getTangentAt(t, tangent);
    moon.target.position.copy(camera.position).addScaledVector(tangent, 25);

    // Darkness: fog closes in and goes black, the moon and sky dim.
    world.fog.color.copy(FOG).lerp(NIGHT, dark);
    world.fog.far = lerp(280, 110, dark);
    skyMat.uniforms.horizon.value.copy(world.fog.color);
    skyMat.uniforms.dark.value = dark * 0.85;
    starMat.opacity = 0.85 * (1 - dark * 0.6);
    moonGlow.material.opacity = moonDisc.material.opacity = 1 - dark * 0.6;
    moonDisc.material.transparent = true;
    moon.intensity = 2.1 * (1 - dark * 0.75);
    hemi.intensity = 1.1 * (1 - dark * 0.8);
    gl.toneMappingExposure = 1 - dark * 0.55;

    // Snow.
    var cx = camera.position.x, cy = camera.position.y, cz = camera.position.z;
    var fall = env.reduceMotion ? 0.5 : 1, wx = f.wind * 5.5;
    for (i = 0; i < N; i++) {
      var k = i * 3;
      flakes[k] += (wx + Math.sin(f.time * 0.9 + phase[i]) * 0.35) * f.dt * fall;
      flakes[k + 1] -= speed[i] * f.dt * fall;
      flakes[k + 2] += Math.cos(f.time * 0.7 + phase[i]) * 0.2 * f.dt * fall;
      flakes[k] = wrap(flakes[k], cx, BX);
      flakes[k + 1] = wrap(flakes[k + 1], cy + BY * 0.3, BY);
      flakes[k + 2] = wrap(flakes[k + 2], cz, BZ);
    }
    snowGeo.attributes.position.needsUpdate = true;
    snowGeo.setDrawRange(0, Math.floor(N * f.snow));
    snowPts.material.opacity = 0.95 * (1 - dark * 0.45);

    gl.render(world, camera);
  }

  function resize(w, h, dpr) {
    gl.setPixelRatio(Math.min(dpr, small ? 1.5 : 1.75));
    gl.setSize(w, h, false);
    camera.aspect = w / h;
    camera.fov = w / h < 1 ? 70 : 55;
    camera.updateProjectionMatrix();
  }

  function destroy() {
    world.traverse(function (o) {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        if (o.material.map) o.material.map.dispose();
        o.material.dispose();
      }
    });
    gl.dispose();
    gl.forceContextLoss();
  }

  // `parts` lets scripts/immersive/capture-layers.js reuse this world to
  // render the image-layer version of the scene.
  return {
    resize: resize, frame: frame, destroy: destroy,
    parts: { gl: gl, world: world, camera: camera, sky: sky, snow: snowPts, height: height, curve: curve }
  };
}

PI.register('snowy-woods-3d', {
  renderer: renderer3d,
  align: ['right', 'left', 'left', 'center'],
  keys: [
    [0.0, 0.000, 0.00, 0.45, 0.10, 0.00],
    [0.7, 0.010, 0.00, 0.50, 0.10, 0.00],
    [1.3, 0.085, 0.00, 0.60, 0.12, 0.28],   // village across the meadow
    [2.3, 0.140, 0.04, 0.95, 0.12, 0.18],   // "...fill up with snow"
    [2.9, 0.300, 0.08, 0.70, 0.10, -0.25],
    [3.9, 0.360, 0.12, 0.70, 0.10, -0.55],  // the frozen lake
    [4.5, 0.450, 0.14, 0.80, 0.35, -0.20],
    [5.0, 0.490, 0.15, 1.00, 1.00, 0.00],   // "the sweep of easy wind"
    [5.5, 0.530, 0.20, 0.90, 0.50, 0.00],
    [6.1, 0.680, 0.35, 0.80, 0.20, 0.00],
    [7.1, 0.800, 0.55, 0.70, 0.12, 0.00],   // "lovely, dark and deep"
    [8.6, 0.900, 0.90, 0.50, 0.05, 0.00]
  ],
  sound: {
    src: '/audio/wind.mp3',
    label: 'Play wind and sleigh bells',
    cues: [{ stanza: 2, at: 0.35, play: PI.sounds.sleighBells }]
  }
});
