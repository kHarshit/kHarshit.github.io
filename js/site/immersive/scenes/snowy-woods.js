/*
 * Scene for "Stopping by Woods on a Snowy Evening" in real 3D (three.js).
 *
 * The camera rides a sleigh track at eye height: along the edge of the woods
 * with the village across a meadow on the left, past a frozen lake on the
 * right, then into deep woods where it slowly goes dark. Terrain, a few
 * thousand instanced pines, the lake, the village and the snowfall are all
 * generated here; nothing is downloaded beyond three.js itself.
 *
 * Loaded by js/site/immersive/engine.js as a module for
 * `immersive: snowy-woods`. The engine owns the timeline, text and sound;
 * this file only renders. Keyframe columns:
 *   [unit, path progress 0..1, darkness, snowfall, wind, look yaw (radians, + is left)]
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, softSprite, skyDome, starField, distanceTo,
         terrain, ribbon, scatter, particleField, followPath, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── World layout (metres; the camera looks down -z) ─────────────────────
var PATH = [[0, 48], [0, 30], [2, 12], [4, -10], [8, -32], [10, -50], [8, -70],
            [4, -92], [0, -115], [-2, -140], [0, -165], [0, -190]];
var LAKE = { x: 30, z: -62, r: 18, y: -0.45 };
var VILLAGE = { x: -36, z: -40 };
var DEEP_Z = -100;   // beyond this the woods close in

var curve = new THREE.CatmullRomCurve3(PATH.map(function (p) { return new THREE.Vector3(p[0], 0, p[1]); }));
var distToPath = distanceTo(curve.getSpacedPoints(160));

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

// ── Renderer ─────────────────────────────────────────────────────────────
function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(17);
  var gl = makeRenderer(canvas, { shadows: !small, clear: '#03060f' });

  var world = new THREE.Scene();
  var FOG = new THREE.Color('#25365c'), NIGHT = new THREE.Color('#04060c');
  world.fog = new THREE.Fog(FOG.clone(), 14, 280);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.1, 2000);

  // Sky dome + stars + moon follow the camera so they read as infinitely far.
  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#02040b', mid: '#0d1a3a', horizon: '#25365c' });
  sky.add(dome.mesh);
  var stars = starField(r, 1600, 1100, 0.08);
  var starMat = stars.material;
  sky.add(stars);

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

  world.add(terrain(780, small ? 150 : 220, 0, -60, height, new THREE.MeshLambertMaterial({ color: '#d4def2' })));

  // Frozen lake: dark ice that catches the moon.
  var lake = new THREE.Mesh(new THREE.CircleGeometry(LAKE.r, 72).rotateX(-Math.PI / 2),
    new THREE.MeshPhongMaterial({ color: '#2b3c64', specular: '#b9c9f2', shininess: 140 }));
  lake.position.set(LAKE.x, LAKE.y, LAKE.z);
  lake.receiveShadow = true;
  world.add(lake);

  // Sleigh-runner tracks along the path: two slightly darker grooves.
  var pts = curve.getSpacedPoints(600), onPath = function (x, z) { return height(x, z, 0); };
  var trackMat = new THREE.MeshLambertMaterial({ color: '#a9b7dc', side: THREE.DoubleSide });
  [-0.55, 0.55].forEach(function (off) {
    var tracks = new THREE.Mesh(ribbon(pts, off, 0.09, onPath, 0.03), trackMat);
    tracks.receiveShadow = true;
    world.add(tracks);
  });

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
  var up = new THREE.Vector3(0, 1, 0);
  scatter(trees, cap * 4, function (i, pos, q, sc, tint) {
    var x = -170 + r() * 340, z = -270 + r() * 340, d = distToPath(x, z);
    if (r() > density(x, z, d)) return false;
    var s = (z < DEEP_Z ? 1.3 : 0.75) + r() * (z < DEEP_Z ? 1.0 : 1.1);
    pos.set(x, height(x, z, d) - 0.25, z);
    q.setFromAxisAngle(up, r() * Math.PI * 2);
    sc.set(s * (0.85 + r() * 0.3), s, s * (0.85 + r() * 0.3));
    var v = 0.8 + r() * 0.35;
    tint.setRGB(v, v, v * (0.95 + r() * 0.1));
  });
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

  // Snowfall in a box that travels with the camera.
  var snow = particleField({
    count: small ? 3500 : 9000, box: [40, 22, 44], fall: [0.7, 1.5], size: 0.14, color: '#f2f6ff',
    map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)')
  });
  world.add(snow.points);

  // ── Per-frame ──────────────────────────────────────────────────────────
  var tangent = new THREE.Vector3();

  function frame(f) {
    var t = clamp(f.cam, 0, 1), yaw = f.row[4] || 0, dark = f.dark;

    // Ride the track at eye height, looking a little way ahead.
    followPath(camera, curve, onPath, t, { eye: 1.75, yaw: yaw, mx: f.mx, my: f.my, time: f.time });

    sky.position.copy(camera.position);
    moon.position.copy(camera.position).addScaledVector(moonDir, 120);
    curve.getTangentAt(t, tangent);
    moon.target.position.copy(camera.position).addScaledVector(tangent, 25);

    // Darkness: fog closes in and goes black, the moon and sky dim.
    world.fog.color.copy(FOG).lerp(NIGHT, dark);
    world.fog.far = lerp(280, 110, dark);
    dome.uniforms.horizon.value.copy(world.fog.color);
    dome.uniforms.dark.value = dark * 0.85;
    starMat.opacity = 0.85 * (1 - dark * 0.6);
    moonGlow.material.opacity = moonDisc.material.opacity = 1 - dark * 0.6;
    moonDisc.material.transparent = true;
    moon.intensity = 2.1 * (1 - dark * 0.75);
    hemi.intensity = 1.1 * (1 - dark * 0.8);
    gl.toneMappingExposure = 1 - dark * 0.55;

    snow.update(f, camera.position, env.reduceMotion);
    snow.points.material.opacity = 0.95 * (1 - dark * 0.45);

    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) { fitCamera(gl, camera, w, h, dpr, small); },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('snowy-woods', {
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
