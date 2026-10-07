/*
 * Scene for "A Dream Within a Dream" (Poe): two worlds, one inside the other.
 *
 * The dream: gilt frames float over still black water, each showing the
 * same stormy shore. A warm glow for "Take this kiss upon the brow", then
 * lights rise and vanish as "hope has flown away", and you drift into the
 * largest frame.
 * The shore: "I stand amid the roar of a surf-tormented shore". Golden
 * grains of sand slip from your hand and blow to the sea ("how few! yet how
 * they creep ... to the deep"), a "pitiless wave" surges up the beach, and
 * on the last question you are pulled back out through the frame: a dream
 * within a dream.
 *
 * The shore is rendered into a texture every frame so the frames can show
 * it live, and the same texture crossfades over the screen on the way in
 * and out. Columns:
 *   [unit, travel, shore, surf, wind, yaw, pitch, grains, surge, hope, glow]
 */
import { THREE, isSmall, makeRenderer, softSprite, skyDome, starField, particleField,
         waveHeight, oceanMaterial, oceanMesh, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// Breakers rolling in towards the beach (+z); negative speed = shoreward.
var SURF = [[0.06, 1, 0.16, 0.55, -1.7], [0.3, 1, 0.27, 0.28, -2.3], [-0.4, 1, 0.46, 0.13, -2.9], [0.12, 1, 0.9, 0.05, -3.6]];
var SHORELINE = -12;
var FRAME = { z: -20, y: 2.4, w: 4.8, h: 3.2 };

function sand(x, z) {
  var h = (z - SHORELINE) * 0.07;
  h += 0.25 * Math.sin(x * 0.3) * Math.cos(z * 0.4) * smooth(SHORELINE + 2, SHORELINE + 10, z);
  return Math.max(h, -2.5);
}

function crash(ac, out) {
  var t = ac.currentTime, len = 2.8, b = ac.createBuffer(1, ac.sampleRate * len, ac.sampleRate), d = b.getChannelData(0);
  for (var i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / d.length, 1.5);
  var src = ac.createBufferSource(), lp = ac.createBiquadFilter(), g = ac.createGain();
  src.buffer = b;
  lp.type = 'lowpass';
  lp.frequency.setValueAtTime(2400, t);
  lp.frequency.exponentialRampToValueAtTime(300, t + 2);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.8, t + 0.25);
  g.gain.exponentialRampToValueAtTime(0.0001, t + len);
  src.connect(lp); lp.connect(g); g.connect(out);
  src.start(t);
}

function foamTexture(r) {
  var c = document.createElement('canvas');
  c.width = 512; c.height = 64;
  var x = c.getContext('2d');
  for (var i = 0; i < 700; i++) {
    x.fillStyle = 'rgba(235,240,245,' + (0.15 + r() * 0.5) + ')';
    x.beginPath();
    x.ellipse(r() * 512, 10 + r() * 50 * r(), 4 + r() * 22, 1 + r() * 3, 0, 0, Math.PI * 2);
    x.fill();
  }
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = THREE.RepeatWrapping;
  t.repeat.set(6, 1);
  return t;
}

// ── The shore ────────────────────────────────────────────────────────────
function buildShore(small, r) {
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#141a26', 0.018);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.05, 2000);

  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#05070d', mid: '#111827', horizon: '#2a3242' }, 1500);
  sky.add(dome.mesh);
  var moon = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(235,240,255,1)', 'rgba(150,170,220,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
  moon.position.set(-420, 380, -900);
  moon.scale.setScalar(260);
  sky.add(moon);
  var cloudTex = softSprite('rgba(255,255,255,0.9)', 'rgba(255,255,255,0)'), clouds = [];
  for (var i = 0; i < 30; i++) {
    var cl = new THREE.Sprite(new THREE.SpriteMaterial({ map: cloudTex, color: '#202836', transparent: true, depthWrite: false,
                                                          opacity: 0.85, fog: false }));
    cl.position.set(-1200 + r() * 2400, 150 + r() * 400, -900 - r() * 300);
    cl.scale.set(400 + r() * 400, 160 + r() * 140, 1);
    sky.add(cl);
    clouds.push(cl);
  }

  world.add(new THREE.HemisphereLight('#6a7aa0', '#1a1612', 1.1));
  var moonLight = new THREE.DirectionalLight('#c8d4ff', 1.6);
  moonLight.position.set(-0.4, 0.4, -0.9);
  world.add(moonLight);

  // Beach: pale sand, darker and wetter towards the water.
  var cols = [], geo = new THREE.PlaneGeometry(200, 60, 120, 60).rotateX(-Math.PI / 2).translate(0, 0, -6);
  var p = geo.attributes.position, dry = new THREE.Color('#b9a77c'), wet = new THREE.Color('#5a5140'), c = new THREE.Color();
  for (i = 0; i < p.count; i++) {
    var x = p.getX(i), z = p.getZ(i);
    p.setY(i, sand(x, z));
    c.copy(wet).lerp(dry, smooth(SHORELINE, SHORELINE + 6, z));
    cols.push(c.r, c.g, c.b);
  }
  geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
  geo.computeVertexNormals();
  world.add(new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9 })));

  // Rocks framing the bay.
  var rock = new THREE.MeshStandardMaterial({ color: '#15171c', roughness: 0.95, flatShading: true });
  [[-1, 1], [1, 1]].forEach(function (s) {
    for (var k = 0; k < 7; k++) {
      var m = new THREE.Mesh(new THREE.DodecahedronGeometry(3 + r() * 5, 0), rock);
      m.position.set(s[0] * (16 + r() * 18), -1 + r() * 3, -6 - r() * 26);
      m.rotation.set(r() * 3, r() * 3, r() * 3);
      world.add(m);
    }
  });

  var oceanMat = oceanMaterial({ color: '#0d1c2a', specular: '#b8c8e0', shininess: 80, foam: '#e6ecf2', waves: SURF });
  var ocean = oceanMesh(oceanMat, 500, small ? 140 : 220);
  world.add(ocean);

  // Swash: two sheets of foam sliding up and down the sand.
  var foamTex = foamTexture(r), swash = [0, 1.9].map(function (offset) {
    var m = new THREE.Mesh(new THREE.PlaneGeometry(120, 5), new THREE.MeshBasicMaterial({ map: foamTex, transparent: true, depthWrite: false }));
    m.rotation.x = -Math.PI / 2 + Math.atan(0.07);
    m.userData.offset = offset;
    world.add(m);
    return m;
  });

  // Spray off the breakers, and wind-blown spume.
  var SPRAY = 400, sprayPos = new Float32Array(SPRAY * 3), sprayVel = new Float32Array(SPRAY * 3), sprayLife = new Float32Array(SPRAY);
  var sprayGeo = new THREE.BufferGeometry();
  sprayGeo.setAttribute('position', new THREE.BufferAttribute(sprayPos, 3));
  var spray = new THREE.Points(sprayGeo, new THREE.PointsMaterial({ color: '#dfe6ee', size: 0.09, transparent: true, depthWrite: false, opacity: 0.8,
    map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)') }));
  spray.frustumCulled = false;
  world.add(spray);
  var spume = particleField({ count: small ? 300 : 700, box: [30, 8, 30], fall: [0.1, 0.4], size: 0.05, color: '#cdd6e2',
                              map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)'), sway: 0.6, windSpeed: 9 });
  world.add(spume.points);

  // Golden grains slipping from an unseen hand just below and right of view.
  var GR = 500, grPos = new Float32Array(GR * 3), grVel = new Float32Array(GR * 3), grLife = new Float32Array(GR), nextGrain = 0;
  var grGeo = new THREE.BufferGeometry();
  grGeo.setAttribute('position', new THREE.BufferAttribute(grPos, 3));
  var grains = new THREE.Points(grGeo, new THREE.PointsMaterial({ color: '#ffcf63', size: 0.035, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, map: softSprite('rgba(255,230,160,1)', 'rgba(255,200,90,0)') }));
  grains.frustumCulled = false;
  world.add(grains);
  for (i = 0; i < GR; i++) grPos[i * 3 + 1] = -99;
  var hand = new THREE.Vector3(), emitClock = 0, sprayClock = 0;

  function update(f, row) {
    var surf = row[2], surge = row[7], dt = f.dt, time = f.time, amp = lerp(0.35, 1.25, surf) + surge * 0.9;
    var u = oceanMat.userData.uniforms;
    u.uTime.value = time;
    u.uAmp.value = amp;
    u.uFoamAmt.value = 1.2 + surf * 0.8;
    u.uSky.value.set('#2a3242');

    // Standing on the beach facing the sea; the wave shakes you.
    camera.position.set(0, sand(0, 2) + 1.7, 2);
    camera.rotation.set(0, 0, 0);
    camera.lookAt(0, sand(0, 2) + 1.2, -30);
    camera.rotateY(row[4] - f.mx * 0.14);
    camera.rotateX(row[5] - f.my * 0.07);
    if (surge > 0.05) camera.position.add(new THREE.Vector3((Math.random() - 0.5), (Math.random() - 0.5), 0).multiplyScalar(surge * 0.06));
    sky.position.copy(camera.position);
    ocean.userData.follow(camera.position);
    clouds.forEach(function (cl) { cl.position.x += dt * f.wind * 14; if (cl.position.x > 1300) cl.position.x -= 2600; });

    // The swash runs up and drains back; the pitiless wave runs right up.
    swash.forEach(function (m) {
      var ph = time * 0.75 + m.userData.offset, run = 0.5 + 0.5 * Math.sin(ph);
      var z = SHORELINE - 1.5 + run * (2.5 + surf * 2) + surge * 13;
      m.position.set(0, sand(0, z) + 0.04, z);
      m.material.opacity = (0.35 + 0.5 * Math.cos(ph) * 0.5 + 0.25) * (0.6 + surf * 0.4) + surge * 0.3;
    });

    // Spray bursts along the break, more with the surf and the surge.
    sprayClock += dt * (3 + surf * 9 + surge * 40);
    while (sprayClock > 1) {
      sprayClock -= 1;
      var bx = (Math.random() - 0.5) * 50, bz = SHORELINE - 4 - Math.random() * 6 + surge * 10;
      for (var n = 0; n < 12; n++) {
        var k = Math.floor(Math.random() * SPRAY), j = k * 3;
        sprayPos[j] = bx + (Math.random() - 0.5) * 3; sprayPos[j + 1] = 0.3; sprayPos[j + 2] = bz;
        sprayVel[j] = (Math.random() - 0.5) * 2 + f.wind * 2; sprayVel[j + 1] = 2 + Math.random() * 4 * (1 + surge); sprayVel[j + 2] = 1 + Math.random() * 2;
        sprayLife[k] = 1.4;
      }
    }
    for (var s = 0; s < SPRAY; s++) {
      var q = s * 3;
      if (sprayLife[s] <= 0) { sprayPos[q + 1] = -99; continue; }
      sprayLife[s] -= dt;
      sprayVel[q + 1] -= 6 * dt;
      sprayPos[q] += sprayVel[q] * dt; sprayPos[q + 1] += sprayVel[q + 1] * dt; sprayPos[q + 2] += sprayVel[q + 2] * dt;
    }
    sprayGeo.attributes.position.needsUpdate = true;
    spume.update({ snow: 0.4 + surf * 0.6, wind: f.wind, dt: dt, time: time }, camera.position, false);

    // Grains: emitted from the hand, blown down the beach towards the sea.
    camera.updateMatrixWorld();
    hand.set(0.34, -0.42, -1.1);
    camera.localToWorld(hand);
    emitClock += dt * row[6] * 90;
    while (emitClock > 1) {
      emitClock -= 1;
      var g = (nextGrain++ % GR) * 3;
      grPos[g] = hand.x + (Math.random() - 0.5) * 0.06; grPos[g + 1] = hand.y; grPos[g + 2] = hand.z + (Math.random() - 0.5) * 0.06;
      grVel[g] = (Math.random() - 0.3) * 0.3; grVel[g + 1] = -0.2 - Math.random() * 0.3; grVel[g + 2] = -0.4 - Math.random() * 0.5;
      grLife[g / 3] = 3.5;
    }
    for (var a = 0; a < GR; a++) {
      var w = a * 3;
      if (grLife[a] <= 0) { grPos[w + 1] = -99; continue; }
      grLife[a] -= dt;
      grVel[w] += f.wind * 0.6 * dt; grVel[w + 1] -= 1.6 * dt; grVel[w + 2] -= 0.8 * f.wind * dt;
      grPos[w] += grVel[w] * dt; grPos[w + 1] += grVel[w + 1] * dt; grPos[w + 2] += grVel[w + 2] * dt;
      var floor = sand(grPos[w], grPos[w + 2]);
      if (grPos[w + 1] < floor) { grPos[w + 1] = floor; grVel[w] *= 0.3; grVel[w + 2] *= 0.3; grVel[w + 1] = 0; grLife[a] = Math.min(grLife[a], 0.6); }
    }
    grGeo.attributes.position.needsUpdate = true;
  }

  return { world: world, camera: camera, update: update };
}

// ── The dream ────────────────────────────────────────────────────────────
function buildDream(small, r, portal) {
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#120c26', 0.028);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.05, 2000);

  var sky = new THREE.Group();
  world.add(sky);
  sky.add(skyDome({ top: '#05030f', mid: '#140d2e', horizon: '#2e2152' }, 1500).mesh);
  sky.add(starField(r, small ? 700 : 1400, 1300, 0.02, 1.3));
  var halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(200,190,255,0.5)', 'rgba(120,100,200,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
  halo.position.set(0, 220, -1000);
  halo.scale.setScalar(900);
  sky.add(halo);

  world.add(new THREE.AmbientLight('#6a5a9a', 0.7));
  var key = new THREE.DirectionalLight('#c8c0ff', 1.3);
  key.position.set(0.3, 1, 0.6);
  world.add(key);
  var warm = new THREE.PointLight('#ffcf8a', 0, 30, 1.2);
  world.add(warm);

  // Gilt frames, each showing the shore. The big one is the way in.
  var gold = new THREE.MeshPhongMaterial({ color: '#7a5a22', specular: '#ffdc8a', shininess: 70, emissive: '#1a1206' });
  var screen = new THREE.MeshBasicMaterial({ map: portal });
  function frameMesh(w, h) {
    var g = new THREE.Group(), b = Math.max(0.16, h * 0.08), d = 0.16;
    [[0, h / 2 + b / 2, w + 2 * b, b], [0, -h / 2 - b / 2, w + 2 * b, b], [-w / 2 - b / 2, 0, b, h], [w / 2 + b / 2, 0, b, h]].forEach(function (s) {
      var m = new THREE.Mesh(new THREE.BoxGeometry(s[2], s[3], d), gold);
      m.position.set(s[0], s[1], 0);
      g.add(m);
    });
    g.add(new THREE.Mesh(new THREE.PlaneGeometry(w, h), screen));
    return g;
  }
  var frames = [], reflections = [];
  var layout = [[0, FRAME.y, FRAME.z, FRAME.w, FRAME.h, 0],
    [-9, 3.4, -28, 2.6, 1.8, 0.35], [8.5, 2.2, -26, 2.2, 3.0, -0.3], [-5, 1.6, -38, 3.2, 2.2, 0.15],
    [6, 4.6, -40, 2.8, 2.0, -0.2], [-13, 2.8, -46, 2.4, 1.7, 0.4], [13, 3.2, -50, 3.0, 2.0, -0.35]];
  layout.forEach(function (l, k) {
    var fm = frameMesh(l[3], l[4]);
    fm.userData = { home: new THREE.Vector3(l[0], l[1], l[2]), yaw: l[5], phase: r() * 6.28, main: k === 0 };
    world.add(fm);
    frames.push(fm);
    var rf = fm.clone();                    // mirrored below the water line
    world.add(rf);
    reflections.push(rf);
  });

  // Still black water, half-hiding the mirrored frames beneath it.
  var water = new THREE.Mesh(new THREE.PlaneGeometry(600, 600).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({ color: '#0a0718', transparent: true, opacity: 0.78 }));
  world.add(water);

  // Mist along the water.
  var mistTex = softSprite('rgba(200,190,240,0.5)', 'rgba(200,190,240,0)'), mist = [];
  for (var i = 0; i < 22; i++) {
    var m = new THREE.Sprite(new THREE.SpriteMaterial({ map: mistTex, transparent: true, depthWrite: false, opacity: 0.18 }));
    m.position.set(-30 + r() * 60, 0.6 + r() * 0.8, -60 + r() * 70);
    m.scale.set(14 + r() * 10, 2.5, 1);
    world.add(m);
    mist.push(m);
  }

  // Hope: warm motes that hover, then rise and scatter as it flies away.
  var HN = 160, homePos = [], flyDir = [], hpos = new Float32Array(HN * 3);
  for (i = 0; i < HN; i++) {
    homePos.push(new THREE.Vector3((r() - 0.5) * 8, 1 + r() * 3, 6 - r() * 14));
    flyDir.push(new THREE.Vector3((r() - 0.5) * 2, 1.2 + r(), -r()).normalize().multiplyScalar(20 + r() * 25));
  }
  var hopeGeo = new THREE.BufferGeometry();
  hopeGeo.setAttribute('position', new THREE.BufferAttribute(hpos, 3));
  var hope = new THREE.Points(hopeGeo, new THREE.PointsMaterial({ color: '#ffe0a8', size: 0.12, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, map: softSprite('rgba(255,235,190,1)', 'rgba(255,210,140,0)') }));
  hope.frustumCulled = false;
  world.add(hope);

  // The kiss: a soft warm glow just ahead.
  var kiss = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,210,160,0.9)', 'rgba(255,170,120,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0 }));
  kiss.scale.setScalar(3.2);
  camera.add(kiss);
  kiss.position.set(0, 0.25, -2.2);
  world.add(camera);

  var tmp = new THREE.Vector3();

  function update(f, row) {
    var travel = row[0], hopeAmt = row[8], glow = row[9], time = f.time, dt = f.dt;

    // Drift from the shoreless water into the big frame.
    var z = lerp(14, FRAME.z + 3.0, smooth(0, 1, travel));
    camera.position.set(Math.sin(time * 0.2) * 0.15 * (1 - travel), lerp(1.9, FRAME.y, travel), z);
    camera.lookAt(0, lerp(2.1, FRAME.y, travel), FRAME.z);
    camera.rotateY(-f.mx * 0.12 * (1 - travel));
    camera.rotateX(-f.my * 0.06 * (1 - travel));
    sky.position.copy(camera.position);

    frames.forEach(function (fm, k) {
      var u = fm.userData;
      fm.position.copy(u.home);
      if (!u.main) {
        fm.position.y += Math.sin(time * 0.5 + u.phase) * 0.25;
        fm.rotation.set(Math.sin(time * 0.3 + u.phase) * 0.06, u.yaw + Math.sin(time * 0.2 + u.phase) * 0.15, Math.sin(time * 0.25 + u.phase) * 0.04);
      }
      var rf = reflections[k];
      rf.position.set(fm.position.x, -fm.position.y, fm.position.z);
      rf.rotation.set(-fm.rotation.x, fm.rotation.y, -fm.rotation.z);
      rf.scale.set(1, -1, 1);
    });
    mist.forEach(function (m, k) { m.position.x += dt * (0.3 + (k % 3) * 0.15); if (m.position.x > 34) m.position.x -= 68; });

    for (var i = 0; i < HN; i++) {
      var e = Math.pow(smooth(0, 1, clamp(hopeAmt * 1.3 - (i % 7) * 0.04, 0, 1)), 1.6);
      tmp.copy(homePos[i]).addScaledVector(flyDir[i], e);
      tmp.y += Math.sin(time * 1.3 + i) * 0.08;
      hpos[i * 3] = tmp.x; hpos[i * 3 + 1] = tmp.y; hpos[i * 3 + 2] = tmp.z;
    }
    hopeGeo.attributes.position.needsUpdate = true;
    hope.material.opacity = 0.9 * (1 - smooth(0.55, 1, hopeAmt));

    kiss.material.opacity = glow * 0.55;
    warm.intensity = glow * 25;
    warm.position.copy(camera.position).add(tmp.set(0, 0.3, -2));
  }

  return { world: world, camera: camera, update: update };
}

// ── Renderer ─────────────────────────────────────────────────────────────
function renderer3d(canvas, scene, env) {
  var small = isSmall();
  var gl = makeRenderer(canvas, { clear: '#05030f' });
  gl.autoClear = true;
  var target = new THREE.WebGLRenderTarget(16, 16, { type: THREE.HalfFloatType });
  var shore = buildShore(small, rng(21)), dream = buildDream(small, rng(4), target.texture);

  // A full-screen quad for crossfading the shore over the dream.
  var overlayScene = new THREE.Scene(), overlayCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  var overlay = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: target.texture, transparent: true,
                                                                                            depthTest: false, depthWrite: false }));
  overlayScene.add(overlay);

  function frame(f) {
    var row = f.row, mix = row[1];
    shore.update(f, row);
    if (mix >= 0.999) {
      gl.setRenderTarget(null);
      gl.render(shore.world, shore.camera);
      return;
    }
    dream.update(f, row);
    gl.setRenderTarget(target);
    gl.render(shore.world, shore.camera);
    gl.setRenderTarget(null);
    gl.render(dream.world, dream.camera);
    if (mix > 0.001) {
      gl.autoClear = false;
      overlay.material.opacity = mix;
      gl.render(overlayScene, overlayCam);
      gl.autoClear = true;
    }
  }

  function resize(w, h, dpr) {
    var ratio = Math.min(dpr, small ? 1.5 : 1.75);
    gl.setPixelRatio(ratio);
    gl.setSize(w, h, false);
    target.setSize(Math.round(w * ratio * 0.6), Math.round(h * ratio * 0.6));
    [shore.camera, dream.camera].forEach(function (c) {
      c.aspect = w / h;
      c.fov = w / h < 1 ? 70 : 55;
      c.updateProjectionMatrix();
    });
  }

  return {
    resize: resize,
    frame: frame,
    destroy: function () { target.dispose(); disposeAll(overlayScene); disposeAll(dream.world); disposeAll(shore.world, gl); }
  };
}

PI.register('dream-shore', {
  renderer: renderer3d,
  maxLines: 6,
  align: ['center', 'center', 'left', 'left', 'center'],
  // Panels (stanzas split at their pauses): 0-1 the first stanza, in the
  // dream; 2 "I stand amid the roar", 3 "How few!", 4 "O God! can I not save".
  keys: function (T) {
    var n = T.count;
    function at(i, frac) { i = Math.min(Math.round(i * (n - 1) / 4), n - 1); return lerp(T.start(i), T.end(i), frac); }
    //       unit           travel shore surf  wind  yaw    pitch  grains surge hope glow
    return [
      [0,                    0.00, 0, 0.50, 0.30, 0.00, 0.00, 0.0, 0.0, 0.0, 0.0],
      [0.7,                  0.03, 0, 0.50, 0.30, 0.00, 0.00, 0.0, 0.0, 0.0, 0.2],
      [at(0, 0.3),           0.10, 0, 0.50, 0.30, 0.00, 0.00, 0.0, 0.0, 0.0, 1.0],   // "Take this kiss upon the brow!"
      [at(0, 1.0),           0.24, 0, 0.50, 0.30, 0.00, 0.00, 0.0, 0.0, 0.1, 0.3],
      [at(1, 0.3),           0.40, 0, 0.50, 0.30, 0.00, 0.00, 0.0, 0.0, 0.5, 0.0],   // "Yet if hope has flown away"
      [at(1, 0.7),           0.70, 0, 0.55, 0.35, 0.00, 0.00, 0.0, 0.0, 1.0, 0.0],
      [at(1, 1.0),           0.88, 0, 0.60, 0.40, 0.00, 0.00, 0.0, 0.0, 1.0, 0.0],   // "a dream within a dream"
      [at(2, 0.1),           1.00, 1, 0.65, 0.50, 0.00, -0.02, 0.0, 0.0, 1.0, 0.0],  // through the frame
      [at(2, 0.55),          1.00, 1, 0.80, 0.60, 0.10, -0.06, 0.0, 0.0, 1.0, 0.0],  // "I stand amid the roar"
      [at(2, 1.0),           1.00, 1, 0.80, 0.60, 0.06, -0.20, 1.0, 0.0, 1.0, 0.0],  // "Grains of the golden sand"
      [at(3, 0.5),           1.00, 1, 0.85, 0.65, 0.00, -0.14, 0.45, 0.0, 1.0, 0.0], // "How few! yet how they creep"
      [at(3, 1.0),           1.00, 1, 0.90, 0.70, 0.00, -0.08, 0.15, 0.0, 1.0, 0.0], // "O God! Can I not grasp"
      [at(4, 0.35),          1.00, 1, 1.00, 0.80, 0.00, 0.02, 0.03, 1.0, 1.0, 0.0],  // "One from the pitiless wave?"
      [at(4, 0.6),           1.00, 1, 0.80, 0.60, 0.00, 0.00, 0.00, 0.3, 1.0, 0.0],
      [at(4, 0.95),          0.96, 0, 0.60, 0.50, 0.00, 0.00, 0.00, 0.0, 1.0, 0.0],  // pulled back out of the frame
      [T.total,              0.40, 0, 0.50, 0.40, 0.00, 0.00, 0.00, 0.0, 1.0, 0.0]
    ];
  },
  sound: {
    src: '/audio/ocean.mp3',
    label: 'Play the surf',
    volume: function (row) { return 0.05 + row[1] * (0.25 + 0.45 * row[2]); },
    cues: [{ stanza: 4, at: 0.55, play: crash }]
  }
});
