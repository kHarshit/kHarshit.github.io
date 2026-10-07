/*
 * Scene for "I Wandered Lonely as a Cloud" (The Daffodils), after Glencoyne
 * Bay on Ullswater.
 *
 * I   You drift high over the fells among the clouds, then sink towards the
 *     lake and the "host of golden daffodils" beside it.
 * II  Low along the margin of the bay, a never-ending line of them.
 * III You turn to the lake, the waves dancing beside the flowers.
 * IV  On the couch, in memory: the day fades to a warm dusk and the
 *     daffodils glow on in the inward eye.
 *
 * The flowers are one instanced mesh swaying in a vertex shader, so there
 * can be thousands of them. Keyframe columns:
 *   [unit, path progress, reverie, pollen, wind, yaw, pitch]
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, broadleafGeometry, softSprite, skyDome,
         distanceTo, terrain, scatter, particleField, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout (metres) ──────────────────────────────────────────────────────
// The lake lies to the west (x < shore(z)); the shore curves into a bay.
function shore(z) { return -10 + 9 * Math.sin(z * 0.022) + 6 * Math.sin(z * 0.061 + 1); }

function height(x, z) {
  var d = x - shore(z);                       // >0 on land, <0 in the water
  var land = smooth(-4, 6, d) * (0.6 + 0.35 * Math.sin(z * 0.09) * Math.cos(x * 0.07));
  // Fells: 1 - |sin| ridges give sharp crests and gullies, not smooth domes.
  var ridge = 0.45 + 0.35 * (1 - Math.abs(Math.sin(z * 0.018 + x * 0.006))) + 0.2 * (1 - Math.abs(Math.sin(z * 0.047 - x * 0.021)));
  var fells = smooth(30, 220, d) * 105 * ridge * (0.7 + 0.3 * Math.cos(x * 0.009));
  var far = smooth(-200, -330, d) * 130 * (0.5 + 0.35 * (1 - Math.abs(Math.sin(z * 0.016 + 2))) + 0.15 * Math.sin(z * 0.05));
  return land - 1.6 * smooth(0, -8, d) + fells + far + 0.4 * Math.sin(x * 0.21 + z * 0.17) * smooth(8, 30, d);
}

// Flight path: from high over the eastern fells down to the shore, then
// along it. Points carry their own height; on the ground the camera keeps
// at least eye height above the terrain.
var PATH = [[95, 78, 230], [70, 64, 160], [40, 42, 95], [18, 16, 45], [6, 3, 10],
            [0, 0, -20], [2, 0, -55], [-2, 0, -90], [0, 0, -125], [-4, 0, -160]];
var curve = new THREE.CatmullRomCurve3(PATH.map(function (p) {
  var x = p[2] > 15 ? p[0] : shore(p[2]) + 7 + p[0];
  return new THREE.Vector3(x, p[1], p[2]);
}));
var distToPath = distanceTo(curve.getSpacedPoints(300));

// One daffodil, ~0.4 m and ~25 triangles so there can be tens of
// thousands: stem, leaf, a six-pointed star of petals and a trumpet,
// nodding forwards.
function daffodilGeometry() {
  var star = new THREE.CircleGeometry(0.075, 12), sp = star.attributes.position;
  for (var i = 2; i < sp.count; i += 2) sp.setXYZ(i, sp.getX(i) * 0.42, sp.getY(i) * 0.42, 0);
  var parts = [
    tinted(new THREE.CylinderGeometry(0.007, 0.011, 0.38, 3).translate(0, 0.19, 0), '#4f7a2a'),
    tinted(new THREE.ConeGeometry(0.025, 0.3, 3).scale(1, 1, 0.25).rotateZ(0.18).translate(0.03, 0.15, 0), '#5f8a32'),
    tinted(star.rotateX(-0.5).translate(0, 0.39, 0.015), '#f8d84a'),
    tinted(new THREE.CylinderGeometry(0.026, 0.016, 0.055, 5, 1, true).rotateX(-1.05).translate(0, 0.395, 0.04), '#f2a51e')
  ];
  return merge(parts);
}

// Rippled bump texture for the water; scrolling it makes the glints move.
function rippleTexture() {
  var c = document.createElement('canvas');
  c.width = c.height = 256;
  var x = c.getContext('2d'), r = rng(31);
  x.fillStyle = '#808080';
  x.fillRect(0, 0, 256, 256);
  for (var i = 0; i < 900; i++) {
    var px = r() * 256, py = r() * 256, w = 6 + r() * 22, h = 1 + r() * 3, v = Math.floor(90 + r() * 120);
    x.fillStyle = 'rgba(' + v + ',' + v + ',' + v + ',0.5)';
    x.beginPath();
    x.ellipse(px, py, w, h, 0, 0, Math.PI * 2);
    x.fill();
  }
  var t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(60, 60);
  return t;
}

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(41);
  var gl = makeRenderer(canvas, { shadows: !small, clear: '#cfe0ef' });

  var world = new THREE.Scene();
  var DAY = new THREE.Color('#c9dbea'), DUSK = new THREE.Color('#e9b07a');
  world.fog = new THREE.Fog(DAY.clone(), 60, 950);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.1, 3000);

  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#4f86c6', mid: '#8fb7dd', horizon: '#d7e4ee', sun: '#fff1cf' }, 2000);
  sky.add(dome.mesh);
  var sunDir = new THREE.Vector3(0.25, 0.55, -0.8).normalize();
  dome.uniforms.sunDir.value.copy(sunDir);

  var hemi = new THREE.HemisphereLight('#dbe9ff', '#5a7a3a', 1.6);
  world.add(hemi);
  var sun = new THREE.DirectionalLight('#fff4dc', 3.2);
  sun.castShadow = !small;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = sun.shadow.camera.bottom = -40;
  sun.shadow.camera.right = sun.shadow.camera.top = 40;
  sun.shadow.camera.far = 300;
  sun.shadow.bias = -0.0005;
  sun.shadow.normalBias = 0.04;
  world.add(sun, sun.target);

  // Land: grass by the water, browner up on the fells.
  var tmp = new THREE.Color(), grass = new THREE.Color('#6f9a3e'), fell = new THREE.Color('#8a7f52'),
      rock = new THREE.Color('#7d7a72'), sand = new THREE.Color('#b8ab86');
  world.add(terrain(1400, small ? 180 : 260, -80, -40, height,
    new THREE.MeshLambertMaterial({ vertexColors: true }),
    function (x, z, y) {
      var d = x - shore(z);
      if (d < 1.5 && d > -6) return tmp.copy(sand);
      tmp.copy(grass).lerp(fell, smooth(4, 40, y));
      return tmp.lerp(rock, smooth(45, 90, y));
    }));

  // The lake, with rippled glints.
  var ripples = rippleTexture();
  var water = new THREE.Mesh(new THREE.PlaneGeometry(1600, 1600).rotateX(-Math.PI / 2),
    new THREE.MeshPhongMaterial({ color: '#3f6f98', specular: '#ffffff', shininess: 90, bumpMap: ripples, bumpScale: 0.6,
                                  transparent: true, opacity: 0.94 }));
  water.position.set(-200, -0.15, -40);
  world.add(water);

  // Trees along the shore, behind the flowers ("beneath the trees").
  var treeMat = new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true });
  var greens = ['#7fa64a', '#93b552', '#6c9440', '#a6c060', '#5f8a3c'], up = new THREE.Vector3(0, 1, 0);
  [broadleafGeometry(rng(7), '#5a4636'), broadleafGeometry(rng(13), '#5a4636')].forEach(function (geo) {
    var trees = new THREE.InstancedMesh(geo, treeMat, small ? 500 : 1100);
    trees.castShadow = trees.receiveShadow = true;
    scatter(trees, 15000, function (i, p, q, s, c) {
      // Woods along the shore; the fells above stay mostly bare.
      var z = -320 + r() * 520, d = 13 + Math.pow(r(), 2.2) * 110, x = shore(z) + d;
      if (r() > 0.9 - d * 0.007 || distToPath(x, z) < 7) return false;
      p.set(x, height(x, z) - 0.2, z);
      q.setFromAxisAngle(up, r() * 6.28);
      var sc = 1.1 + r() * 0.9;
      s.set(sc, sc * (0.9 + r() * 0.3), sc);
      c.set(greens[Math.floor(r() * greens.length)]);
    });
    world.add(trees);
  });

  // Ten thousand daffodils along the margin of the bay, swaying in a
  // vertex shader: each one bends by its own phase, more at the head.
  var flowerMat = new THREE.MeshLambertMaterial({ vertexColors: true, emissive: '#000000' });
  var uniforms = { uTime: { value: 0 }, uWind: { value: 0.3 } };
  flowerMat.onBeforeCompile = function (shader) {
    shader.uniforms.uTime = uniforms.uTime;
    shader.uniforms.uWind = uniforms.uWind;
    shader.vertexShader = 'uniform float uTime; uniform float uWind;\n' + shader.vertexShader.replace(
      '#include <begin_vertex>',
      '#include <begin_vertex>\n' +
      'vec3 ip = vec3(instanceMatrix[3][0], 0.0, instanceMatrix[3][2]);\n' +
      'float ph = ip.x * 0.37 + ip.z * 0.23;\n' +
      'float bend = (sin(uTime * 2.1 + ph) + 0.5 * sin(uTime * 3.7 + ph * 1.7)) * (0.05 + uWind * 0.12);\n' +
      'transformed.x += bend * position.y * position.y * 4.0;\n' +
      'transformed.z += bend * 0.5 * position.y * position.y * 4.0;');
  };
  var flowers = new THREE.InstancedMesh(daffodilGeometry(), flowerMat, small ? 12000 : 30000);
  var flowerCols = ['#ffffff', '#fff6d8', '#ffeaa8'];
  scatter(flowers, 200000, function (i, p, q, s, c) {
    var z = -185 + r() * 215, band = r(), d = 0.5 + band * 11, x = shore(z) + d;
    // Thickest in drifts a few metres from the water, thinning towards the trees.
    var drift = 0.6 + 0.4 * Math.sin(z * 0.11 + Math.sin(z * 0.03) * 3);
    if (r() > drift * (1 - band * 0.65)) return false;
    p.set(x, height(x, z) - 0.02, z);
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar(0.9 + r() * 0.6);
    c.set(flowerCols[Math.floor(r() * flowerCols.length)]);
  });
  world.add(flowers);

  // Clouds to drift through in stanza I.
  var cloudTex = softSprite('rgba(255,255,255,0.95)', 'rgba(255,255,255,0)');
  var clouds = new THREE.Group();
  for (var i = 0; i < 70; i++) {
    var cl = new THREE.Sprite(new THREE.SpriteMaterial({ map: cloudTex, transparent: true, depthWrite: false, opacity: 0.75, fog: false }));
    var cz = -150 + r() * 420, cx = -120 + r() * 260;
    cl.position.set(cx, 55 + r() * 40, cz);
    cl.scale.set(40 + r() * 60, 18 + r() * 20, 1);
    clouds.add(cl);
  }
  world.add(clouds);

  // Pollen and seeds catching the light by the water.
  var pollen = particleField({
    count: small ? 300 : 700, box: [30, 8, 30], fall: [-0.05, 0.08], size: 0.05, color: '#fff3b0',
    map: softSprite('rgba(255,245,190,1)', 'rgba(255,245,190,0)'), sway: 0.3, windSpeed: 1.5
  });
  pollen.points.material.blending = THREE.AdditiveBlending;
  world.add(pollen.points);

  var look = new THREE.Vector3(), dir = new THREE.Vector3(), duskSun = new THREE.Vector3(-0.6, 0.12, -0.75).normalize();
  var glow = new THREE.Color('#ffc23a');

  function frame(f) {
    var row = f.row, t = clamp(f.cam, 0, 1), reverie = f.dark;

    // Fly the path; on the ground keep eye height above the terrain.
    curve.getPointAt(t, camera.position);
    var groundY = height(camera.position.x, camera.position.z) + 1.2;
    camera.position.y = Math.max(camera.position.y, groundY) + Math.sin(f.time * 0.7) * (0.05 + camera.position.y * 0.004);
    curve.getPointAt(Math.min(t + 0.03, 1), look);
    if (t + 0.03 > 1) { curve.getTangentAt(1, dir); look.addScaledVector(dir, 10); }
    look.y = Math.max(look.y, height(look.x, look.z) + 0.9);
    camera.lookAt(look);
    camera.rotateY(row[4] - f.mx * 0.16);
    camera.rotateX(row[5] - f.my * 0.07);
    sky.position.copy(camera.position);

    // Day -> remembered dusk; the flowers keep their light.
    dir.copy(sunDir).lerp(duskSun, reverie).normalize();
    dome.uniforms.sunDir.value.copy(dir);
    sun.position.copy(camera.position).addScaledVector(dir, 150);
    sun.target.position.copy(camera.position);
    sun.color.set('#fff4dc').lerp(tmp.set('#ffb070'), reverie);
    sun.intensity = 3.2 - reverie * 1.6;
    hemi.intensity = 1.6 - reverie * 0.9;
    world.fog.color.copy(DAY).lerp(DUSK, reverie);
    dome.uniforms.horizon.value.copy(world.fog.color);
    dome.uniforms.mid.value.set('#8fb7dd').lerp(tmp.set('#d98a6a'), reverie);
    dome.uniforms.top.value.set('#4f86c6').lerp(tmp.set('#3a3a6a'), reverie);
    gl.setClearColor(world.fog.color);
    flowerMat.emissive.copy(glow).multiplyScalar(reverie * 0.55);

    uniforms.uTime.value = f.time;
    uniforms.uWind.value = f.wind;
    ripples.offset.set(f.time * 0.012, f.time * 0.02);
    clouds.children.forEach(function (c) { c.position.x += f.dt * 0.6; });

    pollen.update(f, camera.position, env.reduceMotion);
    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) { fitCamera(gl, camera, w, h, dpr, small); },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('daffodils', {
  renderer: renderer3d,
  align: ['left', 'right', 'right', 'center'],
  scrim: 0.62,
  keys: [
    [0.0, 0.00, 0.00, 0.20, 0.25, 0.00, -0.10],
    [0.7, 0.03, 0.00, 0.20, 0.25, 0.00, -0.12],   // lonely as a cloud, high over the fells
    [1.3, 0.12, 0.00, 0.30, 0.30, 0.10, -0.20],
    [2.3, 0.40, 0.00, 0.60, 0.45, -0.15, -0.12],  // "all at once I saw a crowd"
    [2.9, 0.48, 0.00, 0.70, 0.40, -0.10, -0.06],
    [3.9, 0.62, 0.00, 0.70, 0.50, 0.15, -0.08],   // along the margin of a bay
    [4.5, 0.68, 0.02, 0.70, 0.60, 0.65, -0.04],
    [5.5, 0.74, 0.05, 0.70, 0.70, 0.95, -0.02],   // the waves beside them danced
    [6.1, 0.79, 0.30, 0.60, 0.40, 0.30, -0.10],
    [7.1, 0.84, 0.70, 0.50, 0.30, 0.00, -0.16],   // "they flash upon that inward eye"
    [8.6, 0.88, 0.95, 0.40, 0.25, 0.00, -0.20]
  ],
  sound: {
    src: '/audio/birds.mp3',
    label: 'Play birdsong by the lake'
  }
});
