/*
 * Scene for Sonnet 116, "Let me not to the marriage of true minds": love as
 * "an ever-fixed mark that looks on tempests and is never shaken ... the
 * star to every wand'ring bark".
 *
 * I   A small boat on a dusk sea, heading for a lighthouse; clouds gather.
 * II  The tempest: waves rise, rain and lightning, the boat is thrown about
 *     while the lighthouse beam keeps its steady sweep. The clouds part on
 *     one fixed star.
 * III "Love's not Time's fool": the sky wheels round that star and Time's
 *     sickle moon arcs over, while the star stays put.
 * IV  The couplet, at dawn on a calm sea.
 *
 * Keyframe columns:
 *   [unit, travel, storm, rain, wind, yaw, pitch, star, time wheel, dawn]
 */
import { THREE, isSmall, makeRenderer, fitCamera, softSprite, skyDome, starField, rainField,
         waveHeight, oceanMaterial, oceanMesh, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// Swell running roughly towards the camera, plus cross-seas and chop.
var WAVES = [[0.3, 1, 0.11, 1.0, 1.1], [-0.5, 0.8, 0.19, 0.55, 1.5], [0.8, 0.4, 0.33, 0.3, 2.0], [0.1, 1, 0.75, 0.1, 3.0]];
var LIGHTHOUSE = new THREE.Vector3(34, 0, -235);
var POLE = new THREE.Vector3(0.12, 0.62, -0.78).normalize();   // the fixed star

// ── Sound ────────────────────────────────────────────────────────────────
function thunder(ac, out) {
  var t = ac.currentTime, len = 3.5, b = ac.createBuffer(1, ac.sampleRate * len, ac.sampleRate), d = b.getChannelData(0);
  for (var i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / d.length, 2);
  var src = ac.createBufferSource(), lp = ac.createBiquadFilter(), g = ac.createGain();
  src.buffer = b;
  lp.type = 'lowpass';
  lp.frequency.setValueAtTime(900, t);
  lp.frequency.exponentialRampToValueAtTime(120, t + 1.2);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.9, t + 0.05);
  g.gain.exponentialRampToValueAtTime(0.0001, t + len);
  src.connect(lp); lp.connect(g); g.connect(out);
  src.start(t);
}

// ── Pieces ───────────────────────────────────────────────────────────────
function canvasTex(w, h, paint) {
  var c = document.createElement('canvas');
  c.width = w; c.height = h;
  paint(c.getContext('2d'), w, h);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function boat() {
  var g = new THREE.Group();
  var hullGeo = new THREE.BoxGeometry(1.3, 0.55, 3.4, 4, 1, 8), hp = hullGeo.attributes.position;
  for (var i = 0; i < hp.count; i++) {
    var z = hp.getZ(i), y = hp.getY(i);
    var taper = z < -0.6 ? 1 - (-0.6 - z) / 1.25 * 0.92 : 1;      // pointed bow towards -z
    hp.setX(i, hp.getX(i) * taper * (y < 0 ? 0.7 : 1));
  }
  hullGeo.computeVertexNormals();
  var hull = new THREE.Mesh(hullGeo, new THREE.MeshStandardMaterial({ color: '#3a2a20', roughness: 0.7 }));
  hull.position.y = 0.15;
  var mast = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.05, 3.6, 6), new THREE.MeshStandardMaterial({ color: '#2a2018' }));
  mast.position.set(0, 2.1, -0.2);
  var shape = new THREE.Shape();
  shape.moveTo(0, 0); shape.lineTo(0, 3.1); shape.lineTo(1.6, 0.1); shape.lineTo(0, 0);
  var sail = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshStandardMaterial({ color: '#d9d2bf', side: THREE.DoubleSide, roughness: 0.9 }));
  sail.position.set(0.03, 0.55, -0.15);
  sail.rotation.y = -Math.PI / 2 + 0.75;   // boom out, so the sail shows from astern
  var lamp = new THREE.Mesh(new THREE.SphereGeometry(0.08, 8, 6), new THREE.MeshBasicMaterial({ color: '#ffcf8a' }));
  lamp.position.set(0, 0.9, 1.4);
  var lampLight = new THREE.PointLight('#ffb760', 4, 9, 1.5);
  lampLight.position.copy(lamp.position);
  g.add(hull, mast, sail, lamp, lampLight);
  return g;
}

function lighthouse(r) {
  var g = new THREE.Group(), rock = new THREE.MeshStandardMaterial({ color: '#1b1d22', roughness: 0.95, flatShading: true });
  for (var i = 0; i < 9; i++) {
    var m = new THREE.Mesh(new THREE.DodecahedronGeometry(4 + r() * 6, 0), rock);
    m.position.set((r() - 0.5) * 22, -2 + r() * 2, (r() - 0.5) * 16);
    m.rotation.set(r() * 3, r() * 3, r() * 3);
    g.add(m);
  }
  var stripes = canvasTex(64, 256, function (x, w, h) {
    for (var i = 0; i < 8; i++) { x.fillStyle = i % 2 ? '#e8e2d6' : '#8a2a22'; x.fillRect(0, i * h / 8, w, h / 8); }
  });
  var tower = new THREE.Mesh(new THREE.CylinderGeometry(1.7, 2.6, 20, 20), new THREE.MeshStandardMaterial({ map: stripes, roughness: 0.6 }));
  tower.position.y = 12;
  var gallery = new THREE.Mesh(new THREE.CylinderGeometry(2.4, 2.4, 0.4, 20), new THREE.MeshStandardMaterial({ color: '#202226' }));
  gallery.position.y = 22.2;
  var lantern = new THREE.Mesh(new THREE.CylinderGeometry(1.3, 1.3, 2.2, 16), new THREE.MeshBasicMaterial({ color: '#fff1c4' }));
  lantern.position.y = 23.5;
  var roof = new THREE.Mesh(new THREE.ConeGeometry(1.8, 1.8, 16), new THREE.MeshStandardMaterial({ color: '#5a1a16' }));
  roof.position.y = 25.5;
  g.add(tower, gallery, lantern, roof);

  // The beam: two long cones with a fade along their length, turning.
  var fade = canvasTex(8, 256, function (x, w, h) {
    var gr = x.createLinearGradient(0, 0, 0, h);
    gr.addColorStop(0, 'rgba(255,240,200,0.9)');
    gr.addColorStop(0.35, 'rgba(255,240,200,0.25)');
    gr.addColorStop(1, 'rgba(255,240,200,0)');
    x.fillStyle = gr;
    x.fillRect(0, 0, w, h);
  });
  var beamMat = new THREE.MeshBasicMaterial({ map: fade, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
                                              side: THREE.DoubleSide, fog: false });
  var beam = new THREE.Group();
  beam.position.y = 23.5;
  [1, -1].forEach(function (s) {
    var cone = new THREE.Mesh(new THREE.ConeGeometry(10, 150, 24, 1, true).translate(0, -75, 0).rotateZ(s * Math.PI / 2), beamMat);
    beam.add(cone);
  });
  var sweep = new THREE.SpotLight('#fff0c8', 400, 260, 0.09, 0.5, 1.2);
  sweep.target.position.set(100, -23, 0);
  beam.add(sweep, sweep.target);
  var glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,240,200,1)', 'rgba(255,220,160,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
  glow.position.y = 23.5;
  glow.scale.setScalar(16);
  g.add(beam, glow);
  g.userData.beam = beam;
  g.userData.beamMat = beamMat;
  return g;
}

function starTexture() {
  return canvasTex(128, 128, function (x) {
    var g = x.createRadialGradient(64, 64, 0, 64, 64, 64);
    g.addColorStop(0, 'rgba(255,255,255,1)');
    g.addColorStop(0.12, 'rgba(220,235,255,0.9)');
    g.addColorStop(0.4, 'rgba(160,190,255,0.15)');
    g.addColorStop(1, 'rgba(160,190,255,0)');
    x.fillStyle = g;
    x.fillRect(0, 0, 128, 128);
    x.strokeStyle = 'rgba(230,240,255,0.7)';
    x.lineWidth = 1.5;
    x.beginPath(); x.moveTo(64, 4); x.lineTo(64, 124); x.moveTo(4, 64); x.lineTo(124, 64); x.stroke();
  });
}

function sickleTexture() {
  return canvasTex(128, 128, function (x) {
    x.fillStyle = '#f4ecd6';
    x.beginPath(); x.arc(64, 64, 40, 0, Math.PI * 2); x.fill();
    x.globalCompositeOperation = 'destination-out';
    x.beginPath(); x.arc(82, 54, 38, 0, Math.PI * 2); x.fill();
  });
}

// ── Renderer ─────────────────────────────────────────────────────────────
function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(13);
  var gl = makeRenderer(canvas, { clear: '#0a0f1c' });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#2a2f45', 0.006);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.1, 3000);

  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#141c3a', mid: '#3a3f63', horizon: '#c27a5a', sun: '#ffb070' }, 1500);
  sky.add(dome.mesh);
  var heavens = new THREE.Group();                 // stars that wheel round the pole
  heavens.add(starField(r, small ? 1500 : 3000, 1300, -0.05, 1.5));
  sky.add(heavens);
  var starMat = heavens.children[0].material;
  var pole = new THREE.Sprite(new THREE.SpriteMaterial({ map: starTexture(), transparent: true, depthWrite: false, fog: false,
                                                          blending: THREE.AdditiveBlending }));
  pole.position.copy(POLE).multiplyScalar(1250);
  pole.scale.setScalar(70);
  sky.add(pole);
  var moon = new THREE.Sprite(new THREE.SpriteMaterial({ map: sickleTexture(), transparent: true, depthWrite: false, fog: false }));
  moon.scale.setScalar(80);
  sky.add(moon);

  // Storm clouds: dark masses low in the sky that part around the star.
  var cloudTex = softSprite('rgba(255,255,255,0.9)', 'rgba(255,255,255,0)'), clouds = [];
  for (var i = 0; i < 46; i++) {
    var c = new THREE.Sprite(new THREE.SpriteMaterial({ map: cloudTex, color: '#2a2f3a', transparent: true, depthWrite: false, fog: false }));
    var a = r() * Math.PI * 2, el = 0.08 + r() * 0.5;
    c.userData.dir = new THREE.Vector3(Math.cos(a) * Math.cos(el), Math.sin(el), Math.sin(a) * Math.cos(el));
    c.scale.set(420 + r() * 380, 200 + r() * 160, 1);
    clouds.push(c);
    sky.add(c);
  }
  var flashSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: cloudTex, color: '#cfd8ff', transparent: true, opacity: 0,
                                                                 blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
  flashSprite.scale.set(1400, 700, 1);
  sky.add(flashSprite);

  var hemi = new THREE.HemisphereLight('#7a86b0', '#0a1018', 0.9);
  var sun = new THREE.DirectionalLight('#ffb080', 1.4);
  world.add(hemi, sun, sun.target);

  var oceanMat = oceanMaterial({ color: '#10243a', specular: '#9ab4d4', shininess: 70, foam: '#d6e0ea', waves: WAVES });
  var ocean = oceanMesh(oceanMat, 700, small ? 160 : 256);
  world.add(ocean);
  var u = oceanMat.userData.uniforms;

  var bark = boat();
  world.add(bark);
  var lh = lighthouse(r);
  lh.position.copy(LIGHTHOUSE);
  world.add(lh);

  var rain = rainField({ count: small ? 1200 : 3000, box: [20, 16, 30], speed: 14, windSpeed: 6, opacity: 0.3 });
  world.add(rain.lines);

  // ── Per frame ─────────────────────────────────────────────────────────
  var C = {
    dusk: ['#141c3a', '#3a3f63', '#c27a5a'], storm: ['#07090e', '#151a22', '#232a36'],
    night: ['#030612', '#0a1430', '#1b2a4a'], dawn: ['#2a3a6a', '#8a7a9a', '#f2b585']
  };
  var tmp = new THREE.Color(), tmp2 = new THREE.Color(), look = new THREE.Vector3(), q = new THREE.Quaternion();
  var e = new THREE.Euler(), flash = 0, sunDir = new THREE.Vector3();

  function mixSky(k, star, storm, dawn) {
    tmp.set(C.dusk[k]).lerp(tmp2.set(C.night[k]), star);
    tmp.lerp(tmp2.set(C.storm[k]), storm);
    return tmp.lerp(tmp2.set(C.dawn[k]), dawn);
  }

  function frame(f) {
    var row = f.row, storm = f.row[1], star = row[6], wheel = row[7], dawn = row[8], dt = f.dt, time = f.time;
    var amp = lerp(0.22, 1.9, storm);
    u.uTime.value = time;
    u.uAmp.value = amp;
    u.uFoamAmt.value = smooth(0.3, 1, storm);

    // The boat rides the swell towards the lighthouse.
    var bz = -f.cam * 230, wv = waveHeight(WAVES, 0, bz, time, amp);
    bark.position.set(0, wv.h - 0.1, bz);
    e.set(-Math.atan(wv.dz) * 0.9, 0, Math.atan(wv.dx) * 0.9);
    bark.quaternion.slerp(q.setFromEuler(e), 1 - Math.exp(-dt * 4));

    var cv = waveHeight(WAVES, 2.2, bz + 8.5, time, amp);
    camera.position.set(2.2, 3 + Math.max(cv.h, -0.5) * 0.7, bz + 8.5);
    look.set(0, 1.6, bz - 25);
    camera.lookAt(look);
    camera.rotateY(row[4] - f.mx * 0.16);
    camera.rotateX(row[5] - f.my * 0.07);
    camera.rotateZ(Math.atan(cv.dx) * 0.25);
    sky.position.copy(camera.position);
    ocean.userData.follow(camera.position);

    // Sky: dusk -> storm -> clear night -> dawn.
    dome.uniforms.top.value.copy(mixSky(0, star, storm, dawn));
    dome.uniforms.mid.value.copy(mixSky(1, star, storm, dawn));
    var horizon = mixSky(2, star, storm, dawn);
    dome.uniforms.horizon.value.copy(horizon);
    u.uSky.value.copy(horizon).lerp(tmp2.set('#ffffff'), flash * 0.4);
    world.fog.color.copy(horizon).multiplyScalar(0.8);
    world.fog.density = lerp(0.005, 0.017, storm);
    gl.setClearColor(world.fog.color);
    sunDir.set(-0.9, 0.06, -0.3).lerp(tmp2.set(0.9, 0.08, -0.4), dawn).normalize();
    dome.uniforms.sunDir.value.copy(sunDir);
    dome.uniforms.sunColor.value.set('#ffb070').multiplyScalar((1 - star) * (1 - storm) * 0.8 + dawn);
    sun.position.copy(camera.position).addScaledVector(sunDir, 200);
    sun.target.position.copy(camera.position);
    sun.intensity = 1.4 * (1 - star) * (1 - storm) + 1.8 * dawn;

    // Stars show once it clears; the heavens turn round the pole.
    starMat.opacity = 0.9 * star * (1 - storm) * (1 - dawn * 0.8);
    heavens.quaternion.setFromAxisAngle(POLE, wheel * Math.PI * 2.4);
    pole.material.opacity = star * (1 - storm * 0.9) * (1 - dawn * 0.5);
    pole.scale.setScalar(70 + Math.sin(time * 2.3) * 4);
    var ma = Math.PI * (0.1 + wheel * 0.8);
    moon.position.set(Math.cos(ma) * 900, Math.sin(ma) * 500 + 80, -700);
    moon.material.opacity = smooth(0, 0.15, wheel) * (1 - smooth(0.85, 1, wheel)) * (1 - dawn);

    // Clouds gather with the storm, then draw back from the star.
    clouds.forEach(function (c, k) {
      var d = c.userData.dir, away = Math.max(0, 1 - d.distanceTo(POLE) / 0.9);
      c.position.copy(d).addScaledVector(d.clone().sub(POLE).normalize(), star * away * 0.9).normalize().multiplyScalar(1150);
      c.material.opacity = clamp(storm * 1.1 + 0.25 * (1 - star) - star * away * 0.9 - dawn, 0, 0.95);
    });

    // Lightning in the worst of it.
    if (storm > 0.65 && Math.random() < dt * 0.9 * storm) {
      flash = 1;
      flashSprite.position.set((Math.random() - 0.5) * 1400, 300 + Math.random() * 300, -900);
    }
    flash *= Math.exp(-dt * 7);
    flashSprite.material.opacity = flash;
    hemi.intensity = lerp(0.9, 0.5, storm) + flash * 6 + dawn * 0.8;
    hemi.color.set('#7a86b0').lerp(tmp2.set('#ffd0a8'), dawn);

    // The beam turns at the same pace whatever the weather.
    lh.userData.beam.rotation.y = time * 0.55;
    lh.userData.beamMat.opacity = 1 - dawn * 0.8;

    rain.update(f, camera.position, f.snow, env.reduceMotion);
    gl.toneMappingExposure = 1 + dawn * 0.15;
    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) { fitCamera(gl, camera, w, h, dpr, small); },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('fixed-mark', {
  renderer: renderer3d,
  align: ['left', 'right', 'left', 'center'],
  keys: [
    //  unit travel storm rain  wind  yaw    pitch star  wheel dawn
    [0.0, 0.00, 0.05, 0.00, 0.20, 0.22, 0.02, 0.00, 0.00, 0.00],
    [1.3, 0.08, 0.15, 0.00, 0.25, 0.12, 0.00, 0.00, 0.00, 0.00],   // "Let me not to the marriage of true minds"
    [2.3, 0.16, 0.35, 0.10, 0.35, 0.05, 0.00, 0.00, 0.00, 0.00],   // clouds gather
    [2.9, 0.22, 0.85, 0.75, 0.85, 0.00, 0.02, 0.00, 0.00, 0.00],   // "looks on tempests"
    [3.4, 0.27, 1.00, 1.00, 1.00, -0.12, 0.05, 0.00, 0.00, 0.00],  // "and is never shaken"
    [3.9, 0.33, 0.55, 0.30, 0.60, -0.02, 0.36, 0.95, 0.00, 0.00],  // "the star to every wand'ring bark"
    [4.5, 0.39, 0.25, 0.05, 0.40, 0.00, 0.42, 1.00, 0.15, 0.00],
    [5.1, 0.45, 0.12, 0.00, 0.30, 0.00, 0.46, 1.00, 0.65, 0.00],   // "Time's ... bending sickle"
    [5.6, 0.50, 0.08, 0.00, 0.25, 0.00, 0.40, 1.00, 1.00, 0.05],   // "even to the edge of doom"
    [6.1, 0.56, 0.05, 0.00, 0.20, 0.00, 0.18, 1.00, 1.00, 0.40],   // the couplet, at dawn
    [7.1, 0.62, 0.02, 0.00, 0.15, 0.00, 0.06, 1.00, 1.00, 0.85],
    [8.6, 0.66, 0.00, 0.00, 0.10, 0.00, 0.05, 1.00, 1.00, 1.00]
  ],
  sound: {
    src: '/audio/ocean.mp3',
    label: 'Play the sea and the storm',
    volume: function (row) { return 0.18 + 0.55 * row[1]; },
    cues: [{ stanza: 1, at: 0.25, play: thunder }, { stanza: 1, at: 0.7, play: thunder }]
  }
});
