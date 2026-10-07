/*
 * Scene for "She Walks in Beauty" (Byron): "like the night of cloudless
 * climes and starry skies; and all that's best of dark and bright".
 *
 * I   A still, cloudless night sea under dense stars; a low moon lays a
 *     glitter path to you, "that tender light which heaven to gaudy day
 *     denies".
 * II  "One shade the more, one ray the less": a long ribbon of dark silk
 *     scattered with spangles of light, the raven tress, drifts across the
 *     stars, turning its dark and bright sides.
 * III "The smiles that win, the tints that glow": the horizon warms to rose
 *     and small lights rise off the water, a heart at peace.
 *
 * Keyframe columns: [unit, travel, (unused), spangles, wind, yaw, pitch, ribbon, glow]
 */
import { THREE, isSmall, makeRenderer, fitCamera, softSprite, skyDome, starField, terrain,
         oceanMaterial, oceanMesh, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

var CALM = [[1, 0.2, 0.12, 0.06, 0.4], [0.3, 1, 0.2, 0.04, 0.6], [-0.6, 0.8, 0.33, 0.02, 0.9]];
var MOON = new THREE.Vector3(0.32, 0.15, -1).normalize();   // off to the right, clear of the text

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(29);
  var gl = makeRenderer(canvas, { clear: '#020309' });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#0c1430', 0.0022);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.1, 3000);

  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#020309', mid: '#081026', horizon: '#1b2648', sun: '#cfd8ff' }, 1500);
  dome.uniforms.sunDir.value.copy(MOON);
  sky.add(dome.mesh);
  sky.add(starField(r, small ? 3000 : 6000, 1300, 0.02, 1.4));
  // The Milky Way: a dense band of faint stars.
  var mw = [], v = new THREE.Vector3(), tilt = new THREE.Euler(1.1, 0.4, 0.5);
  for (var i = 0; i < (small ? 3000 : 8000); i++) {
    var th = r() * Math.PI * 2, y = (r() + r() + r() - 1.5) * 0.1;
    v.set(Math.cos(th), y, Math.sin(th)).normalize().applyEuler(tilt);
    if (v.y < 0.02) continue;
    mw.push(v.x * 1250, v.y * 1250, v.z * 1250);
  }
  var mwGeo = new THREE.BufferGeometry();
  mwGeo.setAttribute('position', new THREE.Float32BufferAttribute(mw, 3));
  sky.add(new THREE.Points(mwGeo, new THREE.PointsMaterial({ color: '#c3cdee', size: 1, sizeAttenuation: false, transparent: true,
                                                             opacity: 0.5, depthWrite: false, fog: false })));
  var moonDisc = new THREE.Mesh(new THREE.CircleGeometry(18, 48), new THREE.MeshBasicMaterial({ color: '#f2f4ff', fog: false }));
  moonDisc.position.copy(MOON).multiplyScalar(1100);
  moonDisc.lookAt(0, 0, 0);
  var moonGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(200,212,255,0.6)', 'rgba(120,140,220,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
  moonGlow.position.copy(MOON).multiplyScalar(1090);
  moonGlow.scale.setScalar(240);
  sky.add(moonDisc, moonGlow);

  world.add(new THREE.HemisphereLight('#3a4a7a', '#05070d', 0.6));
  var moonLight = new THREE.DirectionalLight('#dfe6ff', 1.6);
  world.add(moonLight, moonLight.target);

  // Far hills on either hand, a dark coast under the stars.
  world.add(terrain(2400, small ? 120 : 180, 0, -400, function (x, z) {
    var rr = Math.hypot(x, z + 400), a = Math.atan2(z + 400, x);
    var away = smooth(380, 620, rr) * (0.5 + 0.5 * Math.cos(a * 2 + 2.6));
    return -6 + away * (40 + 25 * Math.sin(a * 7) + 12 * Math.sin(a * 17));
  }, new THREE.MeshLambertMaterial({ color: '#070a14' })));

  var seaMat = oceanMaterial({ color: '#04070f', specular: '#e8eeff', shininess: 140, waves: CALM });
  var sea = oceanMesh(seaMat, 1600, small ? 160 : 240);
  world.add(sea);
  var su = seaMat.userData.uniforms;
  su.uAmp.value = 1;

  // The ribbon: a long strip of dark silk, flowing, with spangles on it.
  var SEG = 260, LEN = 90, ribGeo = new THREE.PlaneGeometry(1, 1, SEG, 1), rp = ribGeo.attributes.position;
  var ribMat = new THREE.MeshPhongMaterial({ color: '#0d1430', specular: '#a9bcff', shininess: 50, side: THREE.DoubleSide,
                                             transparent: true, opacity: 0 });
  var ribbon = new THREE.Mesh(ribGeo, ribMat);
  ribbon.frustumCulled = false;
  world.add(ribbon);
  var SP = small ? 500 : 1100, spU = new Float32Array(SP), spV = new Float32Array(SP), spPh = new Float32Array(SP), spPos = new Float32Array(SP * 3);
  for (i = 0; i < SP; i++) { spU[i] = r(); spV[i] = r() - 0.5; spPh[i] = r() * 6.28; }
  var spGeo = new THREE.BufferGeometry();
  spGeo.setAttribute('position', new THREE.BufferAttribute(spPos, 3));
  var spangles = new THREE.Points(spGeo, new THREE.PointsMaterial({ color: '#f4f0ff', size: 0.55, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)') }));
  spangles.frustumCulled = false;
  world.add(spangles);

  // Where the ribbon is at length s (0..1) and time t, and its side vector.
  var c = new THREE.Vector3(), sideV = new THREE.Vector3();
  function ribbonAt(s, t, base, out, side) {
    var x = (s - 0.5) * LEN, wave = Math.sin(s * 9 - t * 0.6), wave2 = Math.sin(s * 4.3 + t * 0.35);
    out.set(base.x + x, base.y + wave * 3 + wave2 * 4 + Math.sin(s * 2) * 6, base.z + Math.cos(s * 5 - t * 0.4) * 6);
    var twist = s * 7 - t * 0.5;
    side.set(0, Math.cos(twist), Math.sin(twist));
    return out;
  }

  // Warm motes rising off the water at the end.
  var MN = small ? 120 : 260, mHome = [], mPos = new Float32Array(MN * 3);
  for (i = 0; i < MN; i++) mHome.push(new THREE.Vector3((r() - 0.5) * 60, r(), -10 - r() * 70));
  var motesGeo = new THREE.BufferGeometry();
  motesGeo.setAttribute('position', new THREE.BufferAttribute(mPos, 3));
  var motes = new THREE.Points(motesGeo, new THREE.PointsMaterial({ color: '#ffd6b0', size: 0.6, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, map: softSprite('rgba(255,220,190,1)', 'rgba(255,180,150,0)') }));
  motes.frustumCulled = false;
  world.add(motes);

  var tmp = new THREE.Color(), base = new THREE.Vector3();

  function frame(f) {
    var row = f.row, time = f.time, show = row[6], glow = row[7];
    var z = -f.cam * 120;
    camera.position.set(0, 2.4 + Math.sin(time * 0.4) * 0.05, z);
    camera.rotation.set(0, 0, 0);
    camera.lookAt(0, 2.6, z - 40);
    camera.rotateY(row[4] - f.mx * 0.14);
    camera.rotateX(row[5] - f.my * 0.07);
    sky.position.copy(camera.position);
    sea.userData.follow(camera.position);
    moonLight.position.copy(camera.position).addScaledVector(MOON, 200);
    moonLight.target.position.copy(camera.position);

    // The tints that glow: rose creeps into the horizon at the end.
    var horizon = tmp.set('#1b2648').lerp(new THREE.Color('#5a3550'), glow);
    dome.uniforms.horizon.value.copy(horizon);
    dome.uniforms.mid.value.set('#081026').lerp(new THREE.Color('#2a1e44'), glow * 0.7);
    world.fog.color.copy(horizon).multiplyScalar(0.6);
    su.uTime.value = time;
    su.uSky.value.copy(horizon);

    // The ribbon flows high across the stars ahead.
    ribMat.opacity = smooth(0, 0.4, show);
    ribbon.visible = show > 0.01;
    if (ribbon.visible) {
      base.set(Math.sin(time * 0.05) * 6 - 6, 15 + (1 - show) * 30, z - 42);
      for (var k = 0; k <= SEG; k++) {
        var s = k / SEG;
        ribbonAt(s, time, base, c, sideV);
        var w = 3.2 * Math.sin(Math.PI * s) + 0.3;
        rp.setXYZ(k, c.x + sideV.x * w, c.y + sideV.y * w, c.z + sideV.z * w);
        rp.setXYZ(k + SEG + 1, c.x - sideV.x * w, c.y - sideV.y * w, c.z - sideV.z * w);
      }
      rp.needsUpdate = true;
      ribGeo.computeVertexNormals();
      for (var p = 0; p < SP; p++) {
        ribbonAt(spU[p], time, base, c, sideV);
        var ww = (3.2 * Math.sin(Math.PI * spU[p]) + 0.3) * spV[p] * 2;
        spPos[p * 3] = c.x + sideV.x * ww; spPos[p * 3 + 1] = c.y + sideV.y * ww; spPos[p * 3 + 2] = c.z + sideV.z * ww;
      }
      spGeo.attributes.position.needsUpdate = true;
    }
    spangles.visible = ribbon.visible;
    spangles.material.opacity = smooth(0, 0.5, show) * clamp(f.snow, 0, 1) * (0.75 + 0.25 * Math.sin(time * 3));

    for (var m = 0; m < MN; m++) {
      var h = mHome[m], rise = ((time * 0.25 + h.y * 7) % 1);
      mPos[m * 3] = h.x + Math.sin(time * 0.6 + m) * 0.6;
      mPos[m * 3 + 1] = 0.3 + rise * 9;
      mPos[m * 3 + 2] = z + h.z;
    }
    motesGeo.attributes.position.needsUpdate = true;
    motes.material.opacity = glow * 0.85;
    motes.visible = glow > 0.01;

    gl.toneMappingExposure = 1 + glow * 0.15;
    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) { fitCamera(gl, camera, w, h, dpr, small); },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('night-beauty', {
  renderer: renderer3d,
  align: ['left', 'right', 'center'],
  //  unit travel  -   spangles wind  yaw   pitch ribbon glow
  keys: [
    [0.0, 0.00, 0, 0.0, 0.1, 0.00, 0.06, 0.0, 0.0],
    [1.3, 0.10, 0, 0.0, 0.1, 0.05, 0.10, 0.0, 0.0],   // "like the night of cloudless climes"
    [2.3, 0.22, 0, 0.3, 0.1, 0.05, 0.16, 0.3, 0.0],   // "all that's best of dark and bright"
    [2.9, 0.30, 0, 0.8, 0.1, 0.10, 0.24, 0.9, 0.0],   // "one shade the more, one ray the less"
    [3.9, 0.42, 0, 1.0, 0.1, 0.12, 0.26, 1.0, 0.0],   // "which waves in every raven tress"
    [4.5, 0.50, 0, 0.7, 0.1, 0.18, 0.16, 0.5, 0.2],
    [5.5, 0.62, 0, 0.3, 0.1, 0.24, 0.08, 0.0, 0.8],   // "the tints that glow"
    [6.6, 0.72, 0, 0.2, 0.1, 0.26, 0.06, 0.0, 1.0]    // "a heart whose love is innocent"
  ],
  sound: {
    src: '/audio/ocean.mp3',
    label: 'Play the quiet sea',
    volume: function () { return 0.14; }
  }
});
