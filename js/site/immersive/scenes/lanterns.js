/*
 * Scene for "Have You Ever Met Someone" (Jason J. Beaton): a hilltop strung
 * with lights above a valley town.
 *
 * I   "Every night and every day": days and nights spin past.
 * II  "Someone you want to talk forever": two small lights meet and circle
 *     each other slowly in the dark.
 * III "Dance all night ... and together watch the rising sun": they whirl
 *     among the swaying lanterns until the sun comes up.
 * IV  "Pause the time so it doesn't end": everything stops mid-air. Then
 *     "this someone is you": the two lights rise together and the sky fills
 *     with lanterns.
 *
 * The scene keeps its own clock, which stops while time is paused.
 * Columns: [unit, travel, (unused), (unused), wind, yaw, pitch, cycle, pair, dance, pause, reveal]
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, softSprite, skyDome, starField, terrain,
         scatter, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

function hill(x, z) {
  var d = Math.hypot(x, z + 4);
  return 6 * Math.exp(-d * d / 900) - smooth(30, 140, d) * 22 + 0.4 * Math.sin(x * 0.2) * Math.cos(z * 0.17) +
         smooth(160, 320, Math.hypot(x, z + 120)) * 50 * (0.6 + 0.4 * Math.sin(Math.atan2(z + 120, x) * 4));
}

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(77);
  var gl = makeRenderer(canvas, { clear: '#1a1830' });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#3a3048', 0.006);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.1, 3000);

  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#2a3a6a', mid: '#c88a7a', horizon: '#f2b07a', sun: '#ffcf8a' }, 1500);
  sky.add(dome.mesh);
  var stars = starField(r, small ? 1500 : 3000, 1300, 0.03, 1.5);
  sky.add(stars);

  var hemi = new THREE.HemisphereLight('#c8c0e0', '#2a3a20', 1);
  var sun = new THREE.DirectionalLight('#ffd8a8', 2);
  world.add(hemi, sun, sun.target);

  world.add(terrain(900, small ? 120 : 180, 0, -60, hill, new THREE.MeshLambertMaterial({ color: '#3f5a2c' })));

  // Grass swaying on the clock that stops when time is paused.
  var clock = { value: 0 };
  var grassMat = new THREE.MeshLambertMaterial({ vertexColors: true });
  grassMat.onBeforeCompile = function (sh) {
    sh.uniforms.uClock = clock;
    sh.vertexShader = 'uniform float uClock;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n float gph = instanceMatrix[3][0] * 0.5 + instanceMatrix[3][2] * 0.3;\n' +
      ' transformed.x += sin(uClock * 1.8 + gph) * 0.12 * position.y * position.y;');
  };
  var blade = merge([tinted(new THREE.ConeGeometry(0.03, 0.5, 3).translate(0, 0.25, 0), '#5f8a3a')]);
  var grass = new THREE.InstancedMesh(blade, grassMat, small ? 5000 : 12000), up = new THREE.Vector3(0, 1, 0);
  scatter(grass, 40000, function (i, p, q, s, c) {
    var x = (r() - 0.5) * 60, z = -4 + (r() - 0.5) * 60;
    if (Math.hypot(x, z + 4) > 28) return false;
    p.set(x, hill(x, z), z);
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar(0.7 + r() * 0.9);
    c.setHSL(0.24 + r() * 0.06, 0.45, 0.32 + r() * 0.15);
  });
  world.add(grass);

  // Town lights in the valley below.
  var town = [];
  for (var i = 0; i < (small ? 500 : 1100); i++) {
    var tx = (r() - 0.5) * 260, tz = -110 - r() * 120;
    town.push(tx, hill(tx, tz) + 0.5, tz);
  }
  var townGeo = new THREE.BufferGeometry();
  townGeo.setAttribute('position', new THREE.Float32BufferAttribute(town, 3));
  var townMat = new THREE.PointsMaterial({ color: '#ffc77a', size: 0.9, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, map: softSprite('rgba(255,220,160,1)', 'rgba(255,200,120,0)') });
  world.add(new THREE.Points(townGeo, townMat));

  // String lights swung between posts in an arc, with paper lanterns.
  var wood = new THREE.MeshLambertMaterial({ color: '#3a2a20' }), posts = [];
  for (i = 0; i < 6; i++) {
    var a = -1.1 + i * 0.44, px = Math.sin(a) * 9, pz = -4 - Math.cos(a) * 9;
    var post = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.08, 3.2, 6), wood);
    post.position.set(px, hill(px, pz) + 1.6, pz);
    world.add(post);
    posts.push(new THREE.Vector3(px, hill(px, pz) + 3.1, pz));
  }
  var bulbTex = softSprite('rgba(255,215,150,1)', 'rgba(255,180,100,0)'), bulbs = [];
  var lanternMat = new THREE.MeshBasicMaterial({ color: '#ffb36a' });
  for (i = 0; i < posts.length - 1; i++) {
    for (var k = 1; k < 10; k++) {
      var t = k / 10, b = new THREE.Sprite(new THREE.SpriteMaterial({ map: bulbTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
      b.userData = { a: posts[i], b: posts[i + 1], t: t, sag: 0.9, lantern: k === 5 };
      b.scale.setScalar(k === 5 ? 1.4 : 0.5);
      if (k === 5) {
        var lan = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.42, 12), lanternMat);
        b.add(lan);
        lan.scale.setScalar(1 / 1.4);
      }
      world.add(b);
      bulbs.push(b);
    }
  }

  // The two lights.
  var pair = [0, 1].map(function (k) {
    var g = new THREE.Group();
    // One rose, one gold, so they read apart from the string lights.
    var core = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite(k ? 'rgba(255,150,190,1)' : 'rgba(255,225,140,1)', k ? 'rgba(255,110,160,0)' : 'rgba(255,190,90,0)'),
      blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
    core.scale.setScalar(1.3);
    var light = new THREE.PointLight(k ? '#ff9ab8' : '#ffd890', 0, 8, 1.6);
    g.add(core, light);
    g.userData = { core: core, light: light, trail: [] };
    world.add(g);
    return g;
  });

  // Sky lanterns, launched on "this someone is you".
  var SL = small ? 60 : 130, sky0 = [], slMat = new THREE.MeshBasicMaterial({ color: '#ffb060' }), skyLanterns = [];
  var slGeo = new THREE.CylinderGeometry(0.28, 0.2, 0.5, 8);
  for (i = 0; i < SL; i++) {
    var lg = new THREE.Group(), lm = new THREE.Mesh(slGeo, slMat);
    var gs = new THREE.Sprite(new THREE.SpriteMaterial({ map: bulbTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0.6 }));
    gs.scale.setScalar(1.6);
    lg.add(lm, gs);
    var sx = (r() - 0.5) * 50, sz = -4 + (r() - 0.5) * 40;
    lg.userData = { x: sx, z: sz, y0: hill(sx, sz) + 0.5, start: r() * 0.6, drift: (r() - 0.5) * 20, ph: r() * 6 };
    lg.visible = false;
    world.add(lg);
    skyLanterns.push(lg);
  }

  var tmp = new THREE.Color(), tmp2 = new THREE.Color(), sunDir = new THREE.Vector3(), mid = new THREE.Vector3();

  function frame(f) {
    var row = f.row, cycle = row[6], together = row[7], dance = row[8], pause = row[9], reveal = row[10];
    var dt = f.dt * (1 - pause);
    clock.value += dt;
    var ct = clock.value;

    camera.position.set(Math.sin(f.cam * 1.5) * 2, hill(0, 6) + 1.6, 6 - f.cam * 3);
    camera.lookAt(0, hill(0, -8) + 2.6, -16);
    camera.rotateY(row[4] - f.mx * 0.14);
    camera.rotateX(row[5] - f.my * 0.07);
    sky.position.copy(camera.position);

    // Days and nights: the sun sets in the west, rises in the east.
    var th = Math.PI + cycle * Math.PI * 2;
    sunDir.set(Math.cos(th) * 0.9, Math.sin(th), -0.5).normalize();
    var day = smooth(-0.15, 0.25, sunDir.y), dusk = Math.exp(-sunDir.y * sunDir.y * 40);
    dome.uniforms.sunDir.value.copy(sunDir);
    dome.uniforms.top.value.set('#070a1e').lerp(tmp.set('#3a64a8'), day);
    dome.uniforms.mid.value.set('#141838').lerp(tmp.set('#86a8d8'), day).lerp(tmp.set('#c87a7a'), dusk * 0.8);
    var horizon = tmp2.set('#2a2848').lerp(tmp.set('#cfe0f0'), day).lerp(tmp.set('#f5a86a'), dusk);
    dome.uniforms.horizon.value.copy(horizon);
    dome.uniforms.sunColor.value.set('#ffcf8a').multiplyScalar(Math.max(day, dusk) * 0.9);
    stars.material.opacity = 0.9 * (1 - day);
    world.fog.color.copy(horizon).multiplyScalar(0.8);
    gl.setClearColor(world.fog.color);
    sun.position.copy(sunDir).multiplyScalar(150);
    sun.intensity = 2.2 * day + dusk * 0.8;
    sun.color.set('#fff2dc').lerp(tmp.set('#ff9a5a'), dusk);
    // Paused time goes faintly cold and still.
    hemi.intensity = 0.25 + 0.9 * day;
    hemi.color.set('#c8c0e0').lerp(tmp.set('#9ab4ff'), pause * 0.6);
    gl.toneMappingExposure = 1 - pause * 0.12;
    townMat.opacity = 1 - day * 0.85;

    // String lights sway (on the scene clock) and glow at night.
    var lit = 0.35 + 0.65 * (1 - day);
    bulbs.forEach(function (b, n) {
      var u = b.userData, sway = Math.sin(ct * 1.3 + n * 0.4) * 0.12 * (1 + dance);
      b.position.lerpVectors(u.a, u.b, u.t);
      b.position.y -= Math.sin(Math.PI * u.t) * u.sag;
      b.position.z += sway * Math.sin(Math.PI * u.t);
      b.material.opacity = lit * (0.85 + 0.15 * Math.sin(ct * 3 + n));
    });

    // The two lights: drift in, circle while they talk, whirl as they dance.
    // Left of centre, clear of the stanza text on the right.
    mid.set(-2.6, hill(-2.6, -6) + 3.4, -6);
    pair.forEach(function (p, k) {
      var u = p.userData, phase = k * Math.PI;
      var radius = lerp(4, 0.7, together) + dance * 1.2, speed = 0.4 + dance * 2.2;
      var ang = ct * speed + phase, bob = Math.sin(ct * (1.2 + dance * 3) + phase) * (0.2 + dance * 0.8);
      p.position.set(mid.x + Math.cos(ang) * radius, mid.y + bob + reveal * 7, mid.z + Math.sin(ang) * radius * 0.6);
      // On "this someone is you" they close in and rise as one.
      p.position.lerp(new THREE.Vector3(mid.x + (k ? 0.18 : -0.18), mid.y + reveal * 7, mid.z), smooth(0, 0.5, reveal));
      u.core.material.opacity = smooth(0, 0.3, together);
      u.core.scale.setScalar(1.2 + 0.15 * Math.sin(ct * 4 + k));
      u.light.intensity = together * 6;
    });

    // Sky lanterns rise into the morning.
    skyLanterns.forEach(function (lg) {
      var u = lg.userData, p = smooth(u.start, u.start + 0.5, reveal);
      lg.visible = p > 0.001;
      if (!lg.visible) return;
      lg.position.set(u.x + u.drift * p + Math.sin(ct * 0.5 + u.ph) * 0.4, u.y0 + p * 45, u.z - p * 30);
      lg.rotation.y = ct * 0.3 + u.ph;
    });

    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) { fitCamera(gl, camera, w, h, dpr, small); },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('lanterns', {
  renderer: renderer3d,
  align: ['left', 'right', 'left', 'center'],
  keys: [
    //  unit travel -  -  wind yaw   pitch cycle pair dance pause reveal
    [0.0, 0.00, 0, 0, 0.2, 0.00, 0.02, 0.00, 0.0, 0.0, 0.0, 0.0],
    [1.3, 0.05, 0, 0, 0.2, 0.05, 0.04, 0.15, 0.0, 0.0, 0.0, 0.0],  // "every night"
    [2.3, 0.12, 0, 0, 0.2, 0.00, 0.06, 0.95, 0.0, 0.0, 0.0, 0.0],  // "and every day"
    [2.9, 0.18, 0, 0, 0.2, 0.00, 0.06, 1.18, 0.6, 0.0, 0.0, 0.0],  // two lights meet
    [3.9, 0.26, 0, 0, 0.2, 0.00, 0.05, 1.26, 1.0, 0.0, 0.0, 0.0],  // "talk forever"
    [4.5, 0.34, 0, 0, 0.3, 0.00, 0.05, 1.32, 1.0, 1.0, 0.0, 0.0],  // "dance all night"
    [5.5, 0.44, 0, 0, 0.3, -0.08, 0.04, 1.50, 1.0, 0.7, 0.0, 0.0], // "the rising sun"
    [6.0, 0.48, 0, 0, 0.2, -0.08, 0.04, 1.53, 1.0, 0.5, 1.0, 0.0], // "pause the time so it doesn't end"
    [6.35, 0.51, 0, 0, 0.2, -0.04, 0.08, 1.54, 1.0, 0.4, 1.0, 0.0],
    [6.6, 0.54, 0, 0, 0.2, 0.00, 0.14, 1.55, 1.0, 0.2, 0.0, 0.25], // "this someone is you"
    [8.6, 0.62, 0, 0, 0.2, 0.00, 0.30, 1.58, 1.0, 0.1, 0.0, 1.0]
  ],
  sound: {
    src: '/audio/birds.mp3',
    label: 'Play the hilltop',
    volume: function (row) { return 0.06 + 0.3 * Math.max(0, Math.sin(Math.PI + row[6] * Math.PI * 2)); }
  }
});
