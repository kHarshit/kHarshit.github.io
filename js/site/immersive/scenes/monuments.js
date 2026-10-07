/*
 * Scene for Sonnet 55, "Not marble, nor the gilded monuments": stone falls,
 * the verse (and the one it praises) outlasts it.
 *
 * I   A plaza of marble columns round a gilded obelisk at dusk, dusty with
 *     "sluttish time"; at its heart floats one warm light, you, shining
 *     "more bright in these contents".
 * II  "Wasteful war shall statues overturn": fire and embers, smoke, the
 *     columns topple one by one; the light does not burn.
 * III "Shall you pace forth": the light moves forward as days and nights race
 *     past, the ruins sinking into the sand "to the ending doom".
 * IV  The couplet under the stars: only the light, steady.
 *
 * Columns: [unit, travel, smoke, embers, wind, yaw, pitch, topple, fire, ages, glow, sink]
 */
import { THREE, isSmall, makeRenderer, fitCamera, softSprite, skyDome, starField, terrain,
         particleField, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

function ground(x, z) {
  return 0.25 * Math.sin(x * 0.11) * Math.cos(z * 0.09) + 0.12 * Math.sin(x * 0.37 + z * 0.23) +
         smooth(60, 200, Math.hypot(x, z)) * (14 + 8 * Math.sin(Math.atan2(z, x) * 3));
}

function column(marble, broken, r) {
  var g = new THREE.Group(), h = broken ? 2 + r() * 3 : 7;
  var base = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.4, 1.5), marble);
  base.position.y = 0.2;
  var shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.58, h, 20, 1), marble);
  shaft.position.y = 0.4 + h / 2;
  g.add(base, shaft);
  if (!broken) {
    var cap = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.45, 1.4), marble);
    cap.position.y = 0.4 + h + 0.22;
    g.add(cap);
  }
  g.children.forEach(function (m) { m.castShadow = m.receiveShadow = true; });
  return g;
}

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(55);
  var gl = makeRenderer(canvas, { shadows: !small, clear: '#2a2030' });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#8a6a5a', 0.012);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.1, 3000);

  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#1a2240', mid: '#6a5a6a', horizon: '#e09a6a', sun: '#ffb070' }, 1500);
  sky.add(dome.mesh);
  var stars = starField(r, small ? 1500 : 3000, 1300, 0.03, 1.5);
  sky.add(stars);

  var hemi = new THREE.HemisphereLight('#c8b0a0', '#3a2a20', 1.0);
  var sun = new THREE.DirectionalLight('#ffcf9a', 2.4);
  sun.castShadow = !small;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = sun.shadow.camera.bottom = -40;
  sun.shadow.camera.right = sun.shadow.camera.top = 40;
  sun.shadow.camera.far = 300;
  sun.shadow.bias = -0.0008;
  world.add(hemi, sun, sun.target);

  world.add(terrain(800, small ? 120 : 180, 0, 0, ground, new THREE.MeshStandardMaterial({ color: '#8f7d62', roughness: 0.95 })));
  var drift = new THREE.Mesh(new THREE.CircleGeometry(30, 48).rotateX(-Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: '#9a876a', roughness: 1 }));
  world.add(drift);                               // sand that rises over the ruins with the ages

  // The colonnade: a ring of columns round a plinth, some already broken.
  var marble = new THREE.MeshStandardMaterial({ color: '#d9d2c3', roughness: 0.55 });
  var columns = [];
  for (var i = 0; i < 14; i++) {
    var a = i / 14 * Math.PI * 2, broken = i % 5 === 2;
    var col = column(marble, broken, r);
    col.position.set(Math.cos(a) * 13, ground(Math.cos(a) * 13, Math.sin(a) * 13) - 0.1, Math.sin(a) * 13);
    // Each falls outward, at its own moment in the war.
    col.userData = { a: a, at: r() * 0.7, dir: a + (r() - 0.5) * 0.6, home: col.position.clone() };
    world.add(col);
    columns.push(col);
  }
  var plinth = new THREE.Mesh(new THREE.BoxGeometry(5, 1.2, 5), marble);
  plinth.position.y = 0.6;
  plinth.castShadow = plinth.receiveShadow = true;
  world.add(plinth);
  var gold = new THREE.MeshPhongMaterial({ color: '#a8802a', specular: '#ffe2a0', shininess: 90, emissive: '#2a1c06' });
  var obelisk = new THREE.Group();
  var shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 1.1, 11, 4).rotateY(Math.PI / 4), gold);
  shaft.position.y = 6.7;
  var tip = new THREE.Mesh(new THREE.ConeGeometry(0.78, 1.4, 4).rotateY(Math.PI / 4), gold);
  tip.position.y = 12.9;
  obelisk.add(shaft, tip);
  obelisk.children.forEach(function (m) { m.castShadow = true; });
  obelisk.position.set(0, 0, -26);
  world.add(obelisk);

  // You: one warm light that outshines the stone.
  var heart = new THREE.Group();
  var glowTex = softSprite('rgba(255,224,170,1)', 'rgba(255,180,100,0)');
  var core = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
  var halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0.5 }));
  halo.scale.setScalar(9);
  var heartLight = new THREE.PointLight('#ffcf8a', 0, 40, 1.5);
  heart.add(core, halo, heartLight);
  heart.position.set(0, 3.2, 0);
  world.add(heart);

  // War: fires among the columns, embers and smoke.
  var fires = [];
  for (i = 0; i < 5; i++) {
    var fa = r() * Math.PI * 2, fr = 8 + r() * 10;
    var fl = new THREE.PointLight('#ff6a20', 0, 22, 1.6);
    fl.position.set(Math.cos(fa) * fr, 1, Math.sin(fa) * fr);
    var flame = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,150,60,1)', 'rgba(255,60,10,0)'),
      blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0 }));
    flame.position.copy(fl.position);
    flame.scale.set(3, 4, 1);
    world.add(fl, flame);
    fires.push({ light: fl, flame: flame, ph: r() * 6 });
  }
  var embers = particleField({ count: small ? 400 : 900, box: [50, 20, 50], fall: [-1.6, -0.6], size: 0.12, color: '#ff9a40',
                               map: softSprite('rgba(255,200,120,1)', 'rgba(255,120,40,0)'), sway: 0.8, windSpeed: 2 });
  embers.points.material.blending = THREE.AdditiveBlending;
  world.add(embers.points);
  var smokeTex = softSprite('rgba(60,50,50,0.8)', 'rgba(60,50,50,0)'), smoke = [];
  for (i = 0; i < 16; i++) {
    var sm = new THREE.Sprite(new THREE.SpriteMaterial({ map: smokeTex, transparent: true, depthWrite: false, opacity: 0 }));
    sm.userData = { x: (r() - 0.5) * 40, z: (r() - 0.5) * 40 - 10, ph: r() };
    sm.scale.setScalar(14 + r() * 10);
    world.add(sm);
    smoke.push(sm);
  }
  var dust = particleField({ count: small ? 300 : 700, box: [40, 10, 40], fall: [-0.05, 0.05], size: 0.06, color: '#e8d8b8',
                             map: softSprite('rgba(255,245,220,1)', 'rgba(255,245,220,0)'), sway: 0.3 });
  world.add(dust.points);

  var tmp = new THREE.Color(), tmp2 = new THREE.Color(), sunDir = new THREE.Vector3(), q = new THREE.Quaternion(), axis = new THREE.Vector3();

  function frame(f) {
    var row = f.row, time = f.time, dt = f.dt, smokeAmt = row[1], topple = row[6], fire = row[7], ages = row[8], glow = row[9], sink = row[10];

    // Walk slowly into the plaza, then follow the light as it paces forth.
    var t = f.cam;
    camera.position.set(Math.sin(t * 2.2) * 7, 2.2 + t * 3, 30 - t * 34);
    camera.lookAt(heart.position.x, 3 + t * 1.5, heart.position.z - 4);
    camera.rotateY(row[4] - f.mx * 0.14);
    camera.rotateX(row[5] - f.my * 0.07);
    sky.position.copy(camera.position);

    // The sun: low at dusk, then racing round as the ages pass, then gone.
    var ang = 0.12 + ages * Math.PI * 2 * 3.2;
    sunDir.set(Math.cos(ang) * 0.9, Math.sin(ang), -0.45).normalize();
    var dayAmt = clamp(sunDir.y * 3 + 0.3, 0, 1) * (1 - smooth(0.9, 1, ages));
    dome.uniforms.sunDir.value.copy(sunDir);
    dome.uniforms.top.value.set('#06081a').lerp(tmp.set('#2f4f86'), dayAmt * 0.8);
    dome.uniforms.mid.value.set('#141a36').lerp(tmp.set('#a88a8a'), dayAmt);
    var horizon = tmp2.set('#2a2440').lerp(tmp.set('#e09a6a'), dayAmt).lerp(tmp.set('#8a2a14'), fire * 0.8);
    dome.uniforms.horizon.value.copy(horizon);
    dome.uniforms.sunColor.value.set('#ffb070').multiplyScalar(dayAmt);
    stars.material.opacity = 0.9 * (1 - dayAmt) * (1 - smokeAmt * 0.7);
    world.fog.color.copy(horizon).multiplyScalar(0.7);
    world.fog.density = 0.008 + smokeAmt * 0.02;
    gl.setClearColor(world.fog.color);
    sun.position.copy(sunDir).multiplyScalar(120);
    sun.intensity = 2.4 * dayAmt;
    hemi.intensity = 0.25 + 0.8 * dayAmt + fire * 0.3;

    // Toppling: each column swings down about its base, away from the centre.
    columns.forEach(function (c) {
      var u = c.userData, p = smooth(u.at, u.at + 0.3, topple), fall = p * p * Math.PI / 2;
      axis.set(-Math.sin(u.dir), 0, Math.cos(u.dir));
      c.quaternion.copy(q.setFromAxisAngle(axis, -fall));
      c.position.copy(u.home);
      c.position.y = u.home.y - sink * 6;
    });
    obelisk.rotation.z = smooth(0.4, 1, topple) * 0.35;
    obelisk.position.y = -sink * 9;
    plinth.position.y = 0.6 - sink * 2.5;
    drift.position.y = -1 + sink * 1.8;

    // The light: brighter than the stone, paces forth through the ages.
    heart.position.set(0, 3.2 + Math.sin(time * 0.8) * 0.15, -smooth(0, 1, ages) * 18);
    var b = 0.4 + glow * 0.9;
    core.scale.setScalar(1.6 + glow * 1.2 + Math.sin(time * 2) * 0.08);
    core.material.opacity = Math.min(1, b);
    halo.material.opacity = 0.25 + glow * 0.45;
    heartLight.intensity = 6 + glow * 40;

    fires.forEach(function (fi) {
      var flick = 0.7 + 0.3 * Math.sin(time * 9 + fi.ph) * Math.sin(time * 5.3 + fi.ph * 2);
      fi.light.intensity = fire * 60 * flick;
      fi.flame.material.opacity = fire * flick;
      fi.flame.position.y = 1.4 - sink * 4;
    });
    smoke.forEach(function (sm) {
      var u = sm.userData, rise = (time * 0.05 + u.ph) % 1;
      sm.position.set(u.x + rise * 6, 2 + rise * 18, u.z);
      sm.material.opacity = smokeAmt * 0.55 * Math.sin(rise * Math.PI);
    });
    embers.update({ snow: fire, wind: f.wind, dt: dt, time: time }, camera.position, env.reduceMotion);
    dust.update({ snow: 1 - fire * 0.5, wind: f.wind * 0.3, dt: dt, time: time }, camera.position, env.reduceMotion);

    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) { fitCamera(gl, camera, w, h, dpr, small); },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('monuments', {
  renderer: renderer3d,
  align: ['left', 'right', 'left', 'center'],
  keys: [
    //  unit travel smoke embers wind yaw   pitch topple fire ages glow sink
    [0.0, 0.00, 0.00, 0.0, 0.2, 0.00, 0.02, 0.0, 0.0, 0.00, 0.2, 0.0],
    [1.3, 0.08, 0.00, 0.0, 0.2, 0.10, 0.02, 0.0, 0.0, 0.00, 0.5, 0.0],  // "Not marble, nor the gilded monuments"
    [2.3, 0.16, 0.05, 0.0, 0.2, 0.05, 0.04, 0.0, 0.0, 0.00, 0.9, 0.0],  // "you shall shine more bright"
    [2.9, 0.22, 0.40, 0.6, 0.4, 0.00, 0.06, 0.15, 0.8, 0.00, 0.9, 0.0], // "wasteful war shall statues overturn"
    [3.9, 0.30, 0.80, 1.0, 0.6, -0.05, 0.08, 0.95, 1.0, 0.00, 1.0, 0.0], // "war's quick fire shall burn"
    [4.5, 0.38, 0.40, 0.4, 0.4, 0.00, 0.06, 1.0, 0.3, 0.05, 1.0, 0.0],
    [5.5, 0.55, 0.10, 0.0, 0.3, 0.00, 0.04, 1.0, 0.0, 0.75, 1.0, 0.7], // "shall you pace forth ... to the ending doom"
    [6.1, 0.62, 0.00, 0.0, 0.2, 0.00, 0.10, 1.0, 0.0, 0.95, 1.0, 1.0],
    [7.4, 0.68, 0.00, 0.0, 0.2, 0.00, 0.14, 1.0, 0.0, 1.00, 1.0, 1.0]  // "you live in this, and dwell in lovers' eyes"
  ],
  sound: {
    src: '/audio/wind.mp3',
    label: 'Play the wind over the ruins',
    volume: function (row) { return 0.08 + 0.4 * row[7]; }
  }
});
