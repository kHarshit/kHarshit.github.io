/*
 * Scene for "The Road Not Taken": a yellow wood on an autumn morning where
 * the road forks.
 *
 * I   You walk up to the fork and stand looking down the left-hand road to
 *     where it bends into the undergrowth.
 * II  You turn and take the right-hand one, grassier and less worn.
 * III Both lie under fallen leaves; you glance back at the first.
 * IV  "Ages and ages hence": the view lifts above the canopy and turns back
 *     to show the two roads parting, as the light goes to evening gold and
 *     the roads glow faintly, the taken one brighter.
 *
 * Keyframe columns:
 *   [unit, path progress, dusk, falling leaves, wind, yaw, pitch, vista]
 * where vista blends the camera from the path up to a fixed view over the fork.
 */
import { THREE, isSmall, makeRenderer, fitCamera, broadleafGeometry, softSprite, skyDome, distanceTo,
         terrain, ribbon, scatter, particleField, followPath, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout (metres; you walk towards -z) ─────────────────────────────────
var FORK = new THREE.Vector3(0, 0, 0);
function curveOf(pts) { return new THREE.CatmullRomCurve3(pts.map(function (p) { return new THREE.Vector3(p[0], 0, p[1]); })); }
var TRUNK = [[0, 46], [-1, 30], [1, 14], [0, 0]];
// Road taken (right): grassy, wanting wear. Road not taken (left): worn,
// bending away into the undergrowth.
var TAKEN = curveOf(TRUNK.concat([[5, -14], [11, -30], [10, -48], [5, -66], [8, -86], [6, -110]]));
var OTHER = curveOf([[0, 0], [-5, -13], [-13, -26], [-24, -33], [-34, -36]]);
var takenPts = TAKEN.getSpacedPoints(400), otherPts = OTHER.getSpacedPoints(160);
var VISTA = { from: new THREE.Vector3(16, 56, -70), to: new THREE.Vector3(-8, 0, -14) };
var dTaken = distanceTo(takenPts), dOther = distanceTo(otherPts);

function height(x, z) {
  var h = 1.1 * Math.sin(x * 0.05 + 0.4) * Math.cos(z * 0.043) + 0.5 * Math.sin(x * 0.13 - z * 0.09) +
          0.25 * Math.sin(x * 0.31 + z * 0.23);
  var d = Math.min(dTaken(x, z), dOther(x, z));
  h *= smooth(2, 14, d);
  // Hills all round, so the wood sits in a shallow valley.
  var rr = Math.hypot(x, z + 30);
  return h + smooth(120, 280, rr) * 55 * (0.6 + 0.4 * Math.sin(Math.atan2(z + 30, x) * 5));
}

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(23);
  var gl = makeRenderer(canvas, { shadows: !small, clear: '#d9c79c' });

  var world = new THREE.Scene();
  var MORNING = new THREE.Color('#d9c79c'), EVENING = new THREE.Color('#b0703f');
  world.fog = new THREE.Fog(MORNING.clone(), 10, 210);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.1, 2000);

  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#7f9fc4', mid: '#c9cdb8', horizon: '#e6cf9c', sun: '#ffd9a0' });
  sky.add(dome.mesh);

  // Low autumn sun off to the left; it sinks and reddens for stanza IV.
  var sunDir = new THREE.Vector3(-0.7, 0.32, -0.45).normalize();
  dome.uniforms.sunDir.value.copy(sunDir);
  var hemi = new THREE.HemisphereLight('#fff1cf', '#8a6a3c', 2.3);
  world.add(hemi);
  var sun = new THREE.DirectionalLight('#ffd7a1', 3.4);
  sun.castShadow = !small;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = sun.shadow.camera.bottom = -50;
  sun.shadow.camera.right = sun.shadow.camera.top = 50;
  sun.shadow.camera.far = 300;
  sun.shadow.bias = -0.0006;
  sun.shadow.normalBias = 0.05;
  world.add(sun, sun.target);

  // Forest floor: earth and fallen leaves in patches.
  var floor = [new THREE.Color('#7b5a32'), new THREE.Color('#a8772f'), new THREE.Color('#6a5a30'), new THREE.Color('#8d4a26')];
  var tmp = new THREE.Color();
  world.add(terrain(620, small ? 140 : 210, 0, -30, height,
    new THREE.MeshLambertMaterial({ vertexColors: true }),
    function (x, z) {
      var n = 0.5 + 0.5 * Math.sin(x * 0.21 + Math.sin(z * 0.17) * 2) * Math.cos(z * 0.19 - x * 0.07);
      var k = Math.floor(n * 3.999);
      return tmp.copy(floor[k]).lerp(floor[(k + 1) % 4], n * 4 - k);
    }));

  // The two roads: the taken one greener, the other bare and worn.
  var road = function (x, z) { return height(x, z); };
  var taken = new THREE.Mesh(ribbon(takenPts.slice(Math.floor(takenPts.length * 0.29)), 0, 2.6, road, 0.03),
    new THREE.MeshLambertMaterial({ color: '#a3a85a' }));
  var trunkRoad = new THREE.Mesh(ribbon(takenPts.slice(0, Math.ceil(takenPts.length * 0.3)), 0, 2.8, road, 0.03),
    new THREE.MeshLambertMaterial({ color: '#b08a55' }));
  var other = new THREE.Mesh(ribbon(otherPts, 0, 2.6, road, 0.035), new THREE.MeshLambertMaterial({ color: '#b08a55' }));
  [taken, trunkRoad, other].forEach(function (m) { m.receiveShadow = true; world.add(m); });
  var glow = new THREE.Color('#ffcf73');

  // A signpost at the fork, one arm down each road.
  var wood = new THREE.MeshLambertMaterial({ color: '#5a4030' }), sign = new THREE.Group();
  var post = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.07, 2.1, 6).translate(0, 1.05, 0), wood);
  sign.add(post);
  [[-0.75, 1.75], [0.55, 1.5]].forEach(function (arm) {
    var plank = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.16, 0.04).translate(0.42, 0, 0), wood);
    plank.position.y = arm[1];
    plank.rotation.y = arm[0] + Math.PI / 2;
    plank.castShadow = true;
    sign.add(plank);
  });
  post.castShadow = true;
  sign.position.set(FORK.x + 2.2, height(FORK.x + 2.2, FORK.z + 1.5), FORK.z + 1.5);
  world.add(sign);

  // Leaves lying on both roads, "no step had trodden black".
  var leafGeo = new THREE.CircleGeometry(0.06, 5).scale(1.4, 0.8, 1).rotateX(-Math.PI / 2);
  var leaves = new THREE.InstancedMesh(leafGeo, new THREE.MeshLambertMaterial({ side: THREE.DoubleSide }), small ? 4000 : 12000);
  leaves.receiveShadow = true;
  var leafCols = ['#e0a52b', '#d2702a', '#b6402a', '#e8c547', '#9c5a2a'], up = new THREE.Vector3(0, 1, 0);
  scatter(leaves, 30000, function (i, p, q, s, c) {
    var src = r() < 0.6 ? takenPts : otherPts, a = src[Math.floor(r() * src.length)];
    var x = a.x + (r() - 0.5) * 3.2, z = a.z + (r() - 0.5) * 3.2;
    p.set(x, height(x, z) + 0.045 + r() * 0.01, z);
    q.setFromAxisAngle(up, r() * Math.PI * 2);
    s.setScalar(0.6 + r() * 0.7);
    c.set(leafCols[Math.floor(r() * leafCols.length)]);
  });
  world.add(leaves);

  // The yellow wood: broadleaf trees in golds and rusts, a few late greens.
  var treeMat = new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true });
  var crowns = ['#f2c23a', '#eab02c', '#f0d050', '#e08a2a', '#c95b2a', '#d9a03a', '#9aa040'];
  var shapes = [broadleafGeometry(rng(3)), broadleafGeometry(rng(5)), broadleafGeometry(rng(9))];
  shapes.forEach(function (geo, gi) {
    var mesh = new THREE.InstancedMesh(geo, treeMat, small ? 900 : 1900);
    mesh.castShadow = mesh.receiveShadow = true;
    scatter(mesh, 20000, function (i, p, q, s, c) {
      var x = -150 + r() * 300, z = -200 + r() * 260, d = Math.min(dTaken(x, z), dOther(x, z));
      // Wider clearing round the fork so it reads from above in stanza IV.
      var clear = Math.hypot(x, z + 10) < 34 ? 7.5 : 3.4;
      if (d < clear || (d < clear + 3.5 && r() < 0.6)) return false;
      p.set(x, height(x, z) - 0.2, z);
      q.setFromAxisAngle(up, r() * Math.PI * 2);
      var sc = 0.9 + r() * 0.9;
      s.set(sc * (0.9 + r() * 0.2), sc * (0.9 + r() * 0.35), sc * (0.9 + r() * 0.2));
      c.set(crowns[Math.floor(r() * crowns.length)]);
    });
    world.add(mesh);
  });

  // Undergrowth where the other road bends out of sight.
  var bushGeo = new THREE.IcosahedronGeometry(0.8, 0).scale(1, 0.7, 1).translate(0, 0.4, 0);
  var bushes = new THREE.InstancedMesh(bushGeo, new THREE.MeshLambertMaterial({ flatShading: true }), small ? 700 : 1500);
  bushes.castShadow = bushes.receiveShadow = true;
  var bushCols = ['#8a6a2a', '#a4532a', '#6f6a2c', '#b98a34'];
  scatter(bushes, 12000, function (i, p, q, s, c) {
    var nearBend = r() < 0.55, x, z;
    if (nearBend) { var a = otherPts[Math.floor(otherPts.length * (0.55 + r() * 0.45))]; x = a.x + (r() - 0.5) * 14; z = a.z + (r() - 0.5) * 14; }
    else { x = -120 + r() * 240; z = -180 + r() * 230; }
    if (Math.min(dTaken(x, z), dOther(x, z)) < 1.8) return false;
    p.set(x, height(x, z), z);
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar(0.6 + r() * 1.1);
    c.set(bushCols[Math.floor(r() * bushCols.length)]);
  });
  world.add(bushes);

  // Falling leaves.
  var leafTex = (function () {
    var cv = document.createElement('canvas');
    cv.width = cv.height = 32;
    var x = cv.getContext('2d');
    x.fillStyle = '#fff';
    x.beginPath();
    x.ellipse(16, 16, 13, 7, 0.6, 0, Math.PI * 2);
    x.fill();
    var t = new THREE.CanvasTexture(cv);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  })();
  var falling = particleField({
    count: small ? 900 : 2200, box: [36, 18, 40], fall: [0.35, 0.8], size: 0.2, map: leafTex,
    colors: leafCols, sway: 0.9, windSpeed: 4, alphaTest: 0.5
  });
  world.add(falling.points);

  // Morning haze motes in the low sunlight.
  var motes = particleField({
    count: small ? 300 : 700, box: [30, 10, 30], fall: [-0.05, 0.05], size: 0.05, color: '#fff2cc',
    map: softSprite('rgba(255,240,200,1)', 'rgba(255,240,200,0)'), sway: 0.15
  });
  motes.points.material.blending = THREE.AdditiveBlending;
  world.add(motes.points);

  var sunCol = new THREE.Color(), lowSun = new THREE.Vector3(-0.85, 0.12, -0.5).normalize(), dir = new THREE.Vector3();
  var vistaCam = new THREE.Object3D(), pathQ = new THREE.Quaternion();
  vistaCam.position.copy(VISTA.from);
  vistaCam.up.set(0, 1, 0);
  vistaCam.lookAt(VISTA.to);
  // Object3D.lookAt points +z at the target; cameras look down -z, so flip.
  vistaCam.rotateY(Math.PI);

  function frame(f) {
    var row = f.row, t = clamp(f.cam, 0, 1), dusk = f.dark;

    followPath(camera, TAKEN, road, t, {
      eye: 1.7, ahead: 0.03, yaw: row[4], pitch: row[5], mx: f.mx, my: f.my, time: f.time
    });
    var vista = row[6];
    if (vista > 0) {
      var e = vista * vista * (3 - 2 * vista);
      pathQ.copy(camera.quaternion);
      camera.position.lerp(VISTA.from, e);
      camera.quaternion.copy(pathQ).slerp(vistaCam.quaternion, e);
      camera.rotateY(-f.mx * 0.08 * e);
    }
    sky.position.copy(camera.position);

    // Light: morning gold -> low evening amber.
    dir.copy(sunDir).lerp(lowSun, dusk).normalize();
    dome.uniforms.sunDir.value.copy(dir);
    sun.position.copy(camera.position).addScaledVector(dir, 140);
    sun.target.position.copy(camera.position);
    sun.color.copy(sunCol.set('#ffd7a1').lerp(tmp.set('#ff9a52'), dusk));
    sun.intensity = 3.4 - dusk * 1.4;
    hemi.intensity = 2.3 - dusk * 1.1;
    world.fog.color.copy(MORNING).lerp(EVENING, dusk);
    world.fog.far = 210 + vista * 120;
    taken.material.emissive.copy(glow).multiplyScalar(0.75 * vista);
    trunkRoad.material.emissive.copy(glow).multiplyScalar(0.6 * vista);
    other.material.emissive.copy(glow).multiplyScalar(0.3 * vista);
    dome.uniforms.horizon.value.copy(world.fog.color);
    dome.uniforms.top.value.set('#7f9fc4').lerp(tmp.set('#3d3f66'), dusk);
    dome.uniforms.mid.value.set('#c9cdb8').lerp(tmp.set('#c98a5a'), dusk);
    gl.setClearColor(world.fog.color);
    gl.toneMappingExposure = 1.15 - dusk * 0.3;

    falling.update(f, camera.position, env.reduceMotion);
    motes.update({ snow: 1 - dusk * 0.5, wind: f.wind * 0.2, dt: f.dt, time: f.time }, camera.position, env.reduceMotion);

    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) { fitCamera(gl, camera, w, h, dpr, small); },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

// The fork is at path progress ~0.3 (the trunk is 46 m of ~165 m).
PI.register('yellow-wood', {
  renderer: renderer3d,
  align: ['right', 'left', 'right', 'center'],
  keys: [
    [0.0, 0.00, 0.00, 0.35, 0.10, 0.00, 0.00, 0],
    [0.7, 0.02, 0.00, 0.40, 0.10, 0.00, 0.00, 0],
    [1.3, 0.22, 0.00, 0.50, 0.15, 0.25, 0.00, 0],     // up to the fork
    [2.3, 0.26, 0.00, 0.50, 0.15, 0.72, -0.02, 0],    // "looked down one as far as I could"
    [2.9, 0.31, 0.02, 0.60, 0.20, 0.05, 0.00, 0],     // "then took the other"
    [3.9, 0.42, 0.05, 0.60, 0.20, -0.05, -0.05, 0],   // "grassy and wanted wear"
    [4.5, 0.49, 0.08, 0.80, 0.45, 0.00, -0.32, 0],    // "in leaves no step had trodden black"
    [5.0, 0.53, 0.10, 0.80, 0.35, 0.00, -0.18, 0],
    [5.5, 0.56, 0.12, 0.70, 0.25, 2.55, -0.05, 0],    // "I kept the first for another day!"
    [6.1, 0.58, 0.30, 0.50, 0.15, 2.70, -0.10, 0.55], // "ages and ages hence"
    [7.1, 0.60, 0.60, 0.40, 0.12, 2.80, -0.10, 1],    // both roads seen from above
    [8.6, 0.60, 0.85, 0.30, 0.10, 2.80, -0.10, 1]
  ],
  sound: {
    src: '/audio/wind.mp3',
    label: 'Play wind in the trees'
  }
});
