/*
 * Scene for "i carry your heart with me" (E. E. Cummings).
 *
 * I   A small warm light, beating softly, carried in front of you through a
 *     meadow at blue hour: "anywhere i go you go".
 * II  "Whatever a moon has always meant and whatever a sun will always
 *     sing": the moon and the sun stand in the sky together.
 * III "The root of the root and the bud of the bud and the sky of the sky
 *     of a tree called life": down to glowing roots, up a tree that buds and
 *     grows "higher than soul can hope", its crown holding the stars apart.
 * IV  "i carry your heart": back to the little light, beating.
 *
 * Columns: [unit, camZ, (unused), fireflies, wind, yaw, camY, lookY, grow, roots, buds, sun&moon, giant, heart]
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, softSprite, skyDome, starField, terrain,
         scatter, particleField, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

var TREE = new THREE.Vector3(0, 0, -30);
function meadow(x, z) {
  return 0.5 * Math.sin(x * 0.07) * Math.cos(z * 0.05) + 0.3 * Math.sin(x * 0.19 + z * 0.13) +
         smooth(80, 260, Math.hypot(x, z + 30)) * 30 * (0.6 + 0.4 * Math.sin(Math.atan2(z + 30, x) * 5));
}

// Branches as segments: { start, dir, len, rad, level }. `down` grows roots.
function branches(r, levels, len, rad, down) {
  var out = [], up = new THREE.Vector3(0, down ? -1 : 1, 0);
  function grow(start, dir, l, rd, level) {
    out.push({ start: start.clone(), dir: dir.clone(), len: l, rad: rd, level: level });
    if (level >= levels) return;
    var end = start.clone().addScaledVector(dir, l), kids = level < 2 ? 3 : 2;
    for (var k = 0; k < kids; k++) {
      var axis = new THREE.Vector3(r() - 0.5, 0, r() - 0.5).normalize();
      var nd = dir.clone().applyAxisAngle(axis, 0.35 + r() * 0.45).lerp(up, down ? 0.1 : 0.18).normalize();
      if (down) nd.y = Math.min(nd.y, -0.15);
      grow(end, nd, l * (0.68 + r() * 0.12), rd * 0.62, level + 1);
    }
  }
  grow(new THREE.Vector3(), up.clone(), len, rad, 0);
  return out;
}

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(62);
  var gl = makeRenderer(canvas, { clear: '#141c3a' });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#2a3258', 0.009);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.05, 3000);
  world.add(camera);

  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#0d1430', mid: '#2e3a6a', horizon: '#7a6a9a', sun: '#ffc890' }, 1500);
  dome.uniforms.sunDir.value.set(0.7, 0.12, -0.7).normalize();
  sky.add(dome.mesh);
  var stars = starField(r, small ? 1800 : 3500, 1300, 0.05, 1.5);
  sky.add(stars);
  var moon = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(240,244,255,1)', 'rgba(170,190,240,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false, opacity: 0 }));
  moon.position.set(-650, 380, -900);
  moon.scale.setScalar(150);
  var sunSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,230,180,1)', 'rgba(255,170,90,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false, opacity: 0 }));
  sunSprite.position.set(760, 160, -820);
  sunSprite.scale.setScalar(260);
  sky.add(moon, sunSprite);

  var hemi = new THREE.HemisphereLight('#8a96d0', '#1a2a1a', 0.9);
  var warm = new THREE.DirectionalLight('#ffcf9a', 0);
  warm.position.set(0.7, 0.2, -0.7);
  world.add(hemi, warm);

  world.add(terrain(900, small ? 120 : 180, 0, -30, meadow, new THREE.MeshLambertMaterial({ color: '#2f4a2a' })));
  var blade = merge([tinted(new THREE.ConeGeometry(0.035, 0.7, 3).translate(0, 0.35, 0), '#4e7a36')]);
  var clock = { value: 0 }, grassMat = new THREE.MeshLambertMaterial({ vertexColors: true });
  grassMat.onBeforeCompile = function (sh) {
    sh.uniforms.uClock = clock;
    sh.vertexShader = 'uniform float uClock;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n float gph = instanceMatrix[3][0] * 0.4 + instanceMatrix[3][2] * 0.3;\n' +
      ' transformed.x += sin(uClock * 1.4 + gph) * 0.1 * position.y * position.y;');
  };
  var grass = new THREE.InstancedMesh(blade, grassMat, small ? 6000 : 14000), up = new THREE.Vector3(0, 1, 0);
  scatter(grass, 40000, function (i, p, q, s, c) {
    var x = (r() - 0.5) * 50, z = 30 - r() * 80;
    p.set(x, meadow(x, z), z);
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar(0.7 + r());
    c.setHSL(0.27 + r() * 0.05, 0.4, 0.28 + r() * 0.12);
  });
  world.add(grass);

  // The tree, instanced segments that grow level by level.
  var cyl = new THREE.CylinderGeometry(0.75, 1, 1, 8).translate(0, 0.5, 0);
  var limbs = branches(r, 6, 8, 0.9, false), roots = branches(r, 4, 3, 0.5, true);
  var bark = new THREE.MeshStandardMaterial({ color: '#3a2a22', roughness: 0.85, emissive: '#1a0e08' });
  var treeMesh = new THREE.InstancedMesh(cyl, bark, limbs.length);
  var rootMat = new THREE.MeshBasicMaterial({ color: '#ffcf7a', transparent: true, depthTest: false, depthWrite: false,
                                              blending: THREE.AdditiveBlending, opacity: 0 });
  var rootMesh = new THREE.InstancedMesh(cyl, rootMat, roots.length);
  rootMesh.renderOrder = 5;
  var tree = new THREE.Group();
  tree.position.copy(TREE);
  tree.position.y = meadow(TREE.x, TREE.z) - 0.2;
  tree.add(treeMesh, rootMesh);
  world.add(tree);

  // Buds and blossoms at the twig tips.
  var tips = limbs.filter(function (b) { return b.level === 6; }), budPos = [];
  tips.forEach(function (b) {
    var e = b.start.clone().addScaledVector(b.dir, b.len);
    for (var k = 0; k < 3; k++) budPos.push(e.x + (r() - 0.5) * 0.8, e.y + (r() - 0.5) * 0.8, e.z + (r() - 0.5) * 0.8);
  });
  var budGeo = new THREE.BufferGeometry();
  budGeo.setAttribute('position', new THREE.Float32BufferAttribute(budPos, 3));
  var buds = new THREE.Points(budGeo, new THREE.PointsMaterial({ color: '#ffd8e8', size: 0.6, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, opacity: 0, map: softSprite('rgba(255,235,245,1)', 'rgba(255,190,220,0)') }));
  tree.add(buds);

  // Bright stars round the crown, pushed apart as the tree grows.
  var crown = new THREE.Vector3(0, 30, 0), sp = [], spHome = [];
  for (var i = 0; i < 220; i++) {
    var v = new THREE.Vector3(r() - 0.5, r() * 0.8 + 0.1, r() - 0.5).normalize().multiplyScalar(14 + r() * 26);
    spHome.push(v);
    sp.push(0, 0, 0);
  }
  var starsApartGeo = new THREE.BufferGeometry();
  starsApartGeo.setAttribute('position', new THREE.Float32BufferAttribute(sp, 3));
  var starsApart = new THREE.Points(starsApartGeo, new THREE.PointsMaterial({ color: '#eef2ff', size: 1.1, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, opacity: 0, fog: false, map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)') }));
  starsApart.frustumCulled = false;
  tree.add(starsApart);

  // The heart: a small warm light carried just in front of you, beating.
  var heartTex = softSprite('rgba(255,170,140,1)', 'rgba(255,100,90,0)');
  var heart = new THREE.Sprite(new THREE.SpriteMaterial({ map: heartTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
  var heartGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: heartTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0.35 }));
  var heartLight = new THREE.PointLight('#ff9a7a', 0, 6, 1.6);
  var held = new THREE.Group();
  held.add(heart, heartGlow, heartLight);
  held.position.set(0, -0.32, -1.5);
  camera.add(held);

  var flies = particleField({ count: small ? 120 : 260, box: [24, 6, 24], fall: [-0.08, 0.08], size: 0.12, color: '#e8ff9a',
                              map: softSprite('rgba(240,255,170,1)', 'rgba(220,255,120,0)'), sway: 0.5, windSpeed: 0.5 });
  flies.points.material.blending = THREE.AdditiveBlending;
  world.add(flies.points);

  var m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s3 = new THREE.Vector3(), Y = new THREE.Vector3(0, 1, 0), look = new THREE.Vector3();

  function place(mesh, list, g, levels) {
    for (var k = 0; k < list.length; k++) {
      var b = list[k], p = clamp(g * (levels + 1) - b.level, 0, 1);
      q.setFromUnitVectors(Y, b.dir);
      s3.set(b.rad, Math.max(b.len * p, 0.0001), b.rad);
      mesh.setMatrixAt(k, m4.compose(b.start, q, s3));
    }
    mesh.instanceMatrix.needsUpdate = true;
  }
  var lastGrow = -1, lastRoots = -1;

  function frame(f) {
    var row = f.row, time = f.time, grow = row[7], rootsAmt = row[8], budAmt = row[9], both = row[10], giant = row[11], heartAmt = row[12];
    clock.value = time;

    camera.position.set(Math.sin(time * 0.15) * 0.4, meadow(0, f.cam) + row[5], f.cam);
    look.set(TREE.x, row[6], TREE.z);
    camera.lookAt(look);
    camera.rotateY(row[4] - f.mx * 0.14);
    camera.rotateX(-f.my * 0.07);
    sky.position.copy(camera.position);

    if (Math.abs(grow - lastGrow) > 0.0005) { place(treeMesh, limbs, grow, 6); lastGrow = grow; }
    if (Math.abs(rootsAmt - lastRoots) > 0.0005) { place(rootMesh, roots, Math.max(rootsAmt, 0.01), 4); lastRoots = rootsAmt; }
    rootMat.opacity = rootsAmt * 0.85;
    rootMesh.visible = rootsAmt > 0.01;
    buds.material.opacity = budAmt * (0.8 + 0.2 * Math.sin(time * 2));
    var size = 1 + smooth(0, 1, giant) * 3;
    tree.scale.setScalar(size);
    buds.material.size = 0.6 * size;          // points don't scale with their parent
    starsApart.material.size = 1.1 * size;
    bark.emissive.set('#1a0e08').lerp(new THREE.Color('#5a3a18'), budAmt * 0.5);

    for (var k = 0; k < spHome.length; k++) {
      var h = spHome[k], spread = 1 + giant * 1.4;
      starsApartGeo.attributes.position.setXYZ(k, crown.x + h.x * spread, crown.y + h.y * spread, crown.z + h.z * spread);
    }
    starsApartGeo.attributes.position.needsUpdate = true;
    starsApart.material.opacity = smooth(0.2, 1, giant);

    // A moon and a sun at once.
    moon.material.opacity = both;
    sunSprite.material.opacity = both;
    dome.uniforms.sunColor.value.set('#ffc890').multiplyScalar(both * 0.8);
    dome.uniforms.horizon.value.set('#7a6a9a').lerp(new THREE.Color('#d89a7a'), both * 0.6);
    warm.intensity = both * 1.4;
    stars.material.opacity = 0.85 * (1 - both * 0.4);

    // Heartbeat: a double pulse, about once a second.
    var ph = (time * 1.1) % 1, beat = Math.exp(-Math.pow((ph - 0.1) * 14, 2)) + 0.6 * Math.exp(-Math.pow((ph - 0.3) * 14, 2));
    heart.scale.setScalar((0.22 + beat * 0.08) * (0.4 + heartAmt * 0.6));
    heartGlow.scale.setScalar((0.9 + beat * 0.3) * (0.4 + heartAmt * 0.6));
    heart.material.opacity = heartAmt;
    heartGlow.material.opacity = heartAmt * 0.35;
    heartLight.intensity = heartAmt * (1.5 + beat * 1.5);

    flies.update(f, camera.position, env.reduceMotion);
    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) { fitCamera(gl, camera, w, h, dpr, small); },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('heart-tree', {
  renderer: renderer3d,
  maxLines: 5,
  align: ['left', 'right', 'left', 'center'],
  // Panels: 0-1 the first stanza (split after "my darling)"), 2 "here is
  // the deepest secret", 3 "i carry your heart".
  keys: [
    //   unit camZ    -  flies wind yaw  camY  lookY grow roots buds both giant heart
    [0.0,  24,   0, 0.3, 0.2, 0.00, 1.7,  3.0, 0.30, 0.0, 0.0, 0.0, 0.0, 1.0],
    [1.3,  20,   0, 0.6, 0.2, 0.05, 1.7,  3.0, 0.32, 0.0, 0.0, 0.0, 0.0, 1.0],  // "i carry your heart with me"
    [2.4,  14,   0, 0.8, 0.2, 0.00, 1.8,  3.5, 0.36, 0.0, 0.0, 0.0, 0.0, 1.0],  // "anywhere i go you go"
    [2.9,  12,   0, 0.6, 0.2, 0.00, 2.6, 12.0, 0.40, 0.0, 0.0, 0.5, 0.0, 0.7],  // "i fear no fate"
    [3.9,  10,   0, 0.5, 0.2, 0.00, 3.0, 16.0, 0.45, 0.0, 0.0, 1.0, 0.0, 0.6],  // "whatever a moon ... a sun"
    [4.4, -14,   0, 0.3, 0.2, 0.00, 1.4, -1.0, 0.50, 1.0, 0.0, 0.3, 0.0, 0.2],  // "the root of the root"
    [4.9, -16,   0, 0.3, 0.2, 0.00, 9.0, 18.0, 0.80, 0.6, 0.7, 0.1, 0.0, 0.2],  // "the bud of the bud"
    [5.3, -12,   0, 0.3, 0.2, 0.00, 22.0, 34.0, 1.00, 0.3, 1.0, 0.0, 0.3, 0.2], // "the sky of the sky of a tree called life"
    [5.6,  40,   0, 0.3, 0.2, 0.00, 6.0, 55.0, 1.00, 0.1, 1.0, 0.0, 1.0, 0.2],  // "higher than soul can hope"
    [5.8,  44,   0, 0.3, 0.2, 0.00, 6.0, 60.0, 1.00, 0.0, 1.0, 0.0, 1.0, 0.2],  // "keeping the stars apart"
    [6.3,  28,   0, 0.5, 0.2, 0.00, 1.8,  9.0, 1.00, 0.0, 1.0, 0.0, 0.9, 1.0],  // "i carry your heart"
    [7.4,  26,   0, 0.5, 0.2, 0.00, 1.7,  8.0, 1.00, 0.0, 1.0, 0.0, 0.9, 1.0],
    [8.6,  25,   0, 0.4, 0.2, 0.00, 1.7,  8.0, 1.00, 0.0, 1.0, 0.0, 0.9, 1.0]
  ],
  sound: {
    src: '/audio/birds.mp3',
    label: 'Play the meadow',
    volume: function () { return 0.12; }
  }
});
