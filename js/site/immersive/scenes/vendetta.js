/*
 * Scene for V's speech (V for Vendetta): a rain-slick London alley at night,
 * played as a stage.
 *
 * "Voilà!"        a theatre spotlight snaps on.
 * The introduction  the camera walks down the alley past gas lamps and
 *                   posters; every v-word lights up red as it is read.
 * [carves V]        a red V is slashed into the brick, two strokes, sparks.
 * The verdict       the rain hardens and the light turns red.
 * [giggles]         rose petals drift down; a rose lands on the cobbles.
 * "call me V"       out onto the embankment as fireworks break over the
 *                   clock tower: the fifth of November.
 *
 * Keyframes come from the panel list (prose chunks and stage directions).
 * Columns: [unit, path, gloom, rain, wind, yaw, spotlight, carve, fireworks, petals]
 */
import { THREE, isSmall, makeRenderer, fitCamera, softSprite, skyDome, particleField, rainField,
         followPath, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout (metres; the alley runs towards -z, walls at x = ±HALF) ──────
var HALF = 3.2, ALLEY_END = -60, WALL_H = 12;
var V_AT = { x: HALF - 0.02, y: 1.75, z: -27 };          // where V carves
var TOWER = new THREE.Vector3(18, 0, -330);
// Keeps to the left side past the V, so the carving is seen whole from ~4 m.
var curve = new THREE.CatmullRomCurve3([[0, 8], [0, -5], [-0.4, -16], [-0.9, -24], [-0.9, -32], [0, -46], [0, -62], [0, -74], [0, -80]]
  .map(function (p) { return new THREE.Vector3(p[0], 0, p[1]); }));
function flat() { return 0; }

// ── Textures painted on canvases ─────────────────────────────────────────
function canvasTexture(w, h, paint, repeat) {
  var c = document.createElement('canvas');
  c.width = w; c.height = h;
  paint(c.getContext('2d'), w, h);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  if (repeat) t.repeat.set(repeat[0], repeat[1]);
  t.anisotropy = 4;
  return t;
}

function bricks(r) {
  return function (x, w, h) {
    x.fillStyle = '#2b1d1a';
    x.fillRect(0, 0, w, h);
    var bh = 32, bw = 64;
    for (var row = 0; row < h / bh; row++) {
      for (var col = -1; col < w / bw + 1; col++) {
        var off = row % 2 ? bw / 2 : 0, l = 0.55 + r() * 0.45;
        x.fillStyle = 'rgb(' + Math.round(110 * l) + ',' + Math.round(52 * l) + ',' + Math.round(40 * l) + ')';
        x.fillRect(col * bw + off + 2, row * bh + 2, bw - 4, bh - 4);
      }
    }
    // Soot and damp running down from the top.
    var g = x.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, 'rgba(10,8,10,0.45)');
    g.addColorStop(1, 'rgba(10,8,10,0)');
    x.fillStyle = g;
    x.fillRect(0, 0, w, h);
  };
}

function cobbles(r) {
  return function (x, w, h) {
    x.fillStyle = '#0d0e12';
    x.fillRect(0, 0, w, h);
    for (var i = 0; i < 260; i++) {
      var cx = r() * w, cy = r() * h, rw = 10 + r() * 12, l = 30 + r() * 30;
      x.fillStyle = 'rgb(' + l + ',' + l + ',' + (l + 6) + ')';
      x.beginPath();
      x.ellipse(cx, cy, rw, rw * (0.7 + r() * 0.3), r() * 3, 0, Math.PI * 2);
      x.fill();
    }
  };
}

function poster(r, text) {
  return function (x, w, h) {
    x.fillStyle = '#c9bc9c';
    x.fillRect(0, 0, w, h);
    x.fillStyle = '#7a1c16';
    x.fillRect(0, h * 0.62, w, h * 0.38);
    x.fillStyle = '#1c1a18';
    x.font = 'bold ' + Math.round(w * 0.16) + 'px Georgia, serif';
    x.textAlign = 'center';
    text.forEach(function (line, i) { x.fillText(line, w / 2, h * (0.22 + i * 0.17)); });
    // Weathering.
    for (var i = 0; i < 90; i++) {
      x.fillStyle = 'rgba(30,24,20,' + (r() * 0.25) + ')';
      x.fillRect(r() * w, r() * h, 2 + r() * 30, 1 + r() * 6);
    }
  };
}

// ── Synthesised sound cues ───────────────────────────────────────────────
function noiseBuffer(ac, secs) {
  var b = ac.createBuffer(1, Math.floor(ac.sampleRate * secs), ac.sampleRate), d = b.getChannelData(0);
  for (var i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  return b;
}

// Two quick blade scrapes: band-passed noise sweeping down.
function carveSound(ac, out) {
  [0, 0.55].forEach(function (delay) {
    var t = ac.currentTime + delay, src = ac.createBufferSource(), bp = ac.createBiquadFilter(), g = ac.createGain();
    src.buffer = noiseBuffer(ac, 0.5);
    bp.type = 'bandpass';
    bp.Q.value = 6;
    bp.frequency.setValueAtTime(4200, t);
    bp.frequency.exponentialRampToValueAtTime(1300, t + 0.4);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.5, t + 0.03);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.45);
    src.connect(bp); bp.connect(g); g.connect(out);
    src.start(t);
  });
}

// A distant firework: a low thump and a crackle.
function boom(ac, out) {
  var t = ac.currentTime, o = ac.createOscillator(), g = ac.createGain();
  o.frequency.setValueAtTime(90, t);
  o.frequency.exponentialRampToValueAtTime(35, t + 0.6);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.7, t + 0.02);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 0.9);
  o.connect(g); g.connect(out);
  o.start(t); o.stop(t + 1);
  var n = ac.createBufferSource(), hp = ac.createBiquadFilter(), ng = ac.createGain();
  n.buffer = noiseBuffer(ac, 1.2);
  hp.type = 'highpass';
  hp.frequency.value = 2500;
  ng.gain.setValueAtTime(0.0001, t + 0.25);
  ng.gain.exponentialRampToValueAtTime(0.12, t + 0.35);
  ng.gain.exponentialRampToValueAtTime(0.0001, t + 1.4);
  n.connect(hp); hp.connect(ng); ng.connect(out);
  n.start(t + 0.25);
}

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(5);
  var gl = makeRenderer(canvas, { shadows: false, clear: '#05060b' });

  var world = new THREE.Scene();
  var NIGHT = new THREE.Color('#0d0f18'), BLOOD = new THREE.Color('#2a0a0c');
  world.fog = new THREE.Fog(NIGHT.clone(), 6, 150);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.05, 2000);

  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#03040a', mid: '#0b0d1a', horizon: '#2b2230' }, 1500);
  sky.add(dome.mesh);

  var hemi = new THREE.HemisphereLight('#3a4466', '#120c0c', 0.5);
  world.add(hemi);

  // Alley: brick walls, wet cobbles.
  // One texture tile is 2 m square: bricks about 25 x 12 cm.
  var brickTex = canvasTexture(512, 512, bricks(r), [(8 - ALLEY_END) / 2, WALL_H / 2]);
  var wallMat = new THREE.MeshStandardMaterial({ map: brickTex, roughness: 0.85, metalness: 0 });
  var wallLen = 8 - ALLEY_END;
  [-1, 1].forEach(function (side) {
    var wall = new THREE.Mesh(new THREE.PlaneGeometry(wallLen, WALL_H), wallMat);
    wall.rotation.y = -side * Math.PI / 2;
    wall.position.set(side * HALF, WALL_H / 2, (8 + ALLEY_END) / 2);
    world.add(wall);
  });
  var cobbleTex = canvasTexture(512, 512, cobbles(r), [4, 30]);
  var ground = new THREE.Mesh(new THREE.PlaneGeometry(2 * HALF + 0.2, wallLen).rotateX(-Math.PI / 2),
    new THREE.MeshStandardMaterial({ map: cobbleTex, roughness: 0.28, metalness: 0.15 }));
  ground.position.set(0, 0, (8 + ALLEY_END) / 2);
  world.add(ground);

  // Embankment beyond the alley, the river and the far bank.
  var pave = new THREE.Mesh(new THREE.PlaneGeometry(400, 24).rotateX(-Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: '#16171d', roughness: 0.4, metalness: 0.1 }));
  pave.position.set(0, 0, ALLEY_END - 12);
  world.add(pave);
  // Buildings either side of the alley mouth.
  var blockMat = new THREE.MeshStandardMaterial({ map: brickTex, roughness: 0.9 });
  [[-HALF - 30, 60], [HALF + 30, 60]].forEach(function (b) {
    var m = new THREE.Mesh(new THREE.BoxGeometry(b[1], WALL_H + 4, 4), blockMat);
    m.position.set(b[0], (WALL_H + 4) / 2, ALLEY_END + 2);
    world.add(m);
  });
  var rail = new THREE.Mesh(new THREE.BoxGeometry(400, 1.0, 0.4), new THREE.MeshStandardMaterial({ color: '#2a2b30', roughness: 0.6 }));
  rail.position.set(0, 0.5, ALLEY_END - 23);
  world.add(rail);
  var river = new THREE.Mesh(new THREE.PlaneGeometry(900, 250).rotateX(-Math.PI / 2),
    new THREE.MeshPhongMaterial({ color: '#05070c', specular: '#8a8fb0', shininess: 120 }));
  river.position.set(0, -3, ALLEY_END - 150);
  world.add(river);

  // The far bank: a long gothic block, spires and the clock tower, all lit
  // from below like the real thing, outside the fog so they stay crisp.
  // Floodlit from below, as Westminster is at night: warm gold stone.
  var stone = new THREE.MeshLambertMaterial({ color: '#b39060', emissive: '#2a1c0c', fog: false });
  var bank = new THREE.Group();
  var palace = new THREE.Mesh(new THREE.BoxGeometry(360, 30, 26), stone);
  palace.position.set(-90, 15, TOWER.z + 8);
  bank.add(palace);
  for (var i = 0; i < 26; i++) {
    var sp = new THREE.Mesh(new THREE.ConeGeometry(2, 12 + r() * 8, 4), stone);
    sp.position.set(-260 + i * 14 + r() * 4, 36 + r() * 3, TOWER.z + 8);
    bank.add(sp);
  }
  var shaft = new THREE.Mesh(new THREE.BoxGeometry(16, 92, 16), stone);
  shaft.position.set(TOWER.x, 46, TOWER.z);
  var belfry = new THREE.Mesh(new THREE.BoxGeometry(19, 22, 19), stone);
  belfry.position.set(TOWER.x, 102, TOWER.z);
  var spire = new THREE.Mesh(new THREE.ConeGeometry(13, 34, 4).rotateY(Math.PI / 4), stone);
  spire.position.set(TOWER.x, 130, TOWER.z);
  bank.add(shaft, belfry, spire);
  var faceMat = new THREE.MeshBasicMaterial({ color: '#ffd9a0', fog: false });
  var face = new THREE.Mesh(new THREE.CircleGeometry(6.2, 40), faceMat);
  face.position.set(TOWER.x, 102, TOWER.z + 9.55);
  bank.add(face);
  var faceGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,210,150,0.8)', 'rgba(255,180,100,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
  faceGlow.position.set(TOWER.x, 102, TOWER.z + 11);
  faceGlow.scale.setScalar(44);
  bank.add(faceGlow);
  world.add(bank);
  // Floodlights at the foot of the buildings, so they brighten towards the base.
  [[TOWER.x, TOWER.z + 30, 3200], [-60, TOWER.z + 40, 4800], [-190, TOWER.z + 40, 3600]].forEach(function (fl) {
    var light = new THREE.PointLight('#ffc27a', fl[2], 260, 1.4);
    light.position.set(fl[0], 2, fl[1]);
    world.add(light);
  });

  // Windows high on the alley walls, a few still lit.
  var winGeo = new THREE.PlaneGeometry(1.1, 1.6), lit = new THREE.MeshBasicMaterial({ color: '#e8a860' }),
      dark = new THREE.MeshStandardMaterial({ color: '#0a0b10', roughness: 0.2, metalness: 0.4 });
  for (var z = 2; z > ALLEY_END + 3; z -= 4.2) {
    [-1, 1].forEach(function (side) {
      [4.5, 8].forEach(function (y) {
        var w = new THREE.Mesh(winGeo, r() < 0.18 ? lit : dark);
        w.position.set(side * (HALF - 0.02), y, z + r() * 1.5);
        w.rotation.y = -side * Math.PI / 2;
        world.add(w);
      });
    });
  }

  // Posters on the left wall.
  [[-6, ['REMEMBER,', 'REMEMBER']], [-15, ['THE FIFTH', 'OF NOVEMBER']], [-36, ['REMEMBER,', 'REMEMBER']]].forEach(function (p) {
    var m = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.7),
      new THREE.MeshStandardMaterial({ map: canvasTexture(256, 360, poster(r, p[1])), roughness: 0.9 }));
    m.position.set(-HALF + 0.03, 1.9, p[0]);
    m.rotation.y = Math.PI / 2;
    m.rotation.z = (r() - 0.5) * 0.06;
    world.add(m);
  });

  // Gas lamps on brackets.
  var lampGlow = softSprite('rgba(255,190,110,0.9)', 'rgba(255,160,80,0)');
  var lampHead = new THREE.MeshBasicMaterial({ color: '#ffd29a' }), iron = new THREE.MeshStandardMaterial({ color: '#1a1a1e', roughness: 0.5 });
  [[-1, -6], [1, -17], [-1, -36], [1, -50]].forEach(function (l) {
    var x = l[0] * (HALF - 0.55);
    var arm = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.06, 0.06), iron);
    arm.position.set(l[0] * (HALF - 0.3), 3.7, l[1]);
    var head = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.4, 0.28), lampHead);
    head.position.set(x, 3.4, l[1]);
    var glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: lampGlow, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
    glow.position.copy(head.position);
    glow.scale.setScalar(3);
    var light = new THREE.PointLight('#ffb766', 22, 16, 1.6);
    light.position.set(x, 3.2, l[1]);
    world.add(arm, head, glow, light);
  });

  // The stage spotlight for "Voilà!".
  var spot = new THREE.SpotLight('#fff1dc', 0, 18, 0.42, 0.45, 1.2);
  spot.position.set(0.4, 8, V_AT.z + 1.5);
  spot.target.position.set(V_AT.x - 0.5, 1.2, V_AT.z);
  world.add(spot, spot.target);
  var beamMat = new THREE.MeshBasicMaterial({ color: '#fff1dc', transparent: true, opacity: 0, depthWrite: false,
                                              blending: THREE.AdditiveBlending, side: THREE.DoubleSide });
  var beam = new THREE.Mesh(new THREE.ConeGeometry(2.6, 7.8, 32, 1, true).translate(0, -3.9, 0), beamMat);
  beam.position.copy(spot.position);
  beam.lookAt(spot.target.position);
  beam.rotateX(-Math.PI / 2);
  world.add(beam);

  // The V: two strokes that grow from their start point, plus a glow at the
  // blade's tip and sparks while it cuts.
  var vMat = new THREE.MeshBasicMaterial({ color: '#ff2418' });
  var strokes = [[[2.55, -27.75], [0.95, -27.0]], [[0.95, -27.0], [2.55, -26.25]]].map(function (s) {
    var a = s[0], b = s[1], len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    var m = new THREE.Mesh(new THREE.PlaneGeometry(len, 0.11).translate(len / 2, 0, 0), vMat);
    m.position.set(V_AT.x, a[0], a[1]);
    m.rotation.y = -Math.PI / 2;
    // On the wall, the plane's x runs along world +z and its y is height.
    m.rotateZ(Math.atan2(b[0] - a[0], b[1] - a[1]));
    m.scale.x = 0.0001;
    world.add(m);
    return { mesh: m, a: a, b: b };
  });
  var tip = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,80,40,1)', 'rgba(255,40,20,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0 }));
  tip.scale.setScalar(0.7);
  world.add(tip);
  var vLight = new THREE.PointLight('#ff3020', 0, 6, 1.5);
  vLight.position.set(V_AT.x - 0.6, V_AT.y, V_AT.z);
  world.add(vLight);

  var SPARKS = 160, sparkPos = new Float32Array(SPARKS * 3), sparkVel = new Float32Array(SPARKS * 3), sparkLife = new Float32Array(SPARKS);
  var sparkGeo = new THREE.BufferGeometry();
  sparkGeo.setAttribute('position', new THREE.BufferAttribute(sparkPos, 3));
  var sparks = new THREE.Points(sparkGeo, new THREE.PointsMaterial({ color: '#ffb070', size: 0.05, transparent: true,
    blending: THREE.AdditiveBlending, depthWrite: false }));
  sparks.frustumCulled = false;
  world.add(sparks);
  var nextSpark = 0;

  // A rose on the cobbles below the V, and petals for the giggle.
  var rose = new THREE.Group();
  var stem = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.5, 4).rotateZ(Math.PI / 2), new THREE.MeshStandardMaterial({ color: '#1f4a1c' }));
  var petalsMat = new THREE.MeshStandardMaterial({ color: '#b3101a', roughness: 0.5, flatShading: true });
  var bloom = new THREE.Mesh(new THREE.IcosahedronGeometry(0.07, 1), petalsMat);
  bloom.position.x = 0.27;
  rose.add(stem, bloom);
  rose.position.set(V_AT.x - 0.9, 0.02, V_AT.z - 0.4);
  rose.rotation.y = 0.6;
  rose.scale.setScalar(0.0001);
  world.add(rose);
  var petalTex = canvasTexture(32, 32, function (x) { x.fillStyle = '#fff'; x.beginPath(); x.ellipse(16, 16, 12, 8, 0.5, 0, Math.PI * 2); x.fill(); });
  var petals = particleField({ count: small ? 120 : 260, box: [6, 6, 8], fall: [0.25, 0.5], size: 0.07, map: petalTex,
                               colors: ['#b3101a', '#8f0c14', '#d0202a'], sway: 0.7, windSpeed: 1.5, alphaTest: 0.5 });
  world.add(petals.points);

  var rain = rainField({ count: small ? 1300 : 3200 });
  world.add(rain.lines);

  // Fireworks over the far bank: a pool of bursts, each a shell of points.
  var BURSTS = 14, PER = small ? 140 : 220, fwN = BURSTS * PER;
  var fwPos = new Float32Array(fwN * 3), fwVel = new Float32Array(fwN * 3), fwCol = new Float32Array(fwN * 3), fwBase = new Float32Array(fwN * 3);
  var burstAge = new Float32Array(BURSTS).fill(99), nextBurst = 0, burstClock = 0;
  var fwGeo = new THREE.BufferGeometry();
  fwGeo.setAttribute('position', new THREE.BufferAttribute(fwPos, 3));
  fwGeo.setAttribute('color', new THREE.BufferAttribute(fwCol, 3));
  var fireworks = new THREE.Points(fwGeo, new THREE.PointsMaterial({ size: 3.4, vertexColors: true, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, fog: false, map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)') }));
  fireworks.frustumCulled = false;
  world.add(fireworks);
  var flash = new THREE.PointLight('#ffb070', 0, 900, 1);
  world.add(flash);
  var palette = ['#ff3a2a', '#ffd27a', '#fff4e0', '#ff7a2a', '#ffb0a0'].map(function (c) { return new THREE.Color(c); });

  function launch() {
    var b = nextBurst++ % BURSTS, c = palette[Math.floor(Math.random() * palette.length)];
    var cx = TOWER.x - 140 + Math.random() * 220, cy = 95 + Math.random() * 70, cz = TOWER.z + 20 - Math.random() * 40;
    var speed = 34 + Math.random() * 18;
    for (var k = b * PER; k < (b + 1) * PER; k++) {
      var u = Math.random() * 2 - 1, th = Math.random() * 6.283, s = Math.sqrt(1 - u * u), sp = speed * (0.85 + Math.random() * 0.3);
      fwPos[k * 3] = cx; fwPos[k * 3 + 1] = cy; fwPos[k * 3 + 2] = cz;
      fwVel[k * 3] = s * Math.cos(th) * sp; fwVel[k * 3 + 1] = u * sp; fwVel[k * 3 + 2] = s * Math.sin(th) * sp;
      fwBase[k * 3] = c.r; fwBase[k * 3 + 1] = c.g; fwBase[k * 3 + 2] = c.b;
    }
    burstAge[b] = 0;
    flash.position.set(cx, cy, cz);
    flash.color.copy(c);
    flash.intensity = 700;
  }

  var tmp = new THREE.Color();

  function frame(f) {
    var row = f.row, t = clamp(f.cam, 0, 1), gloom = f.dark, dt = f.dt;
    var spotOn = row[5], carve = row[6], fw = row[7], petalAmt = row[8];

    // Look up towards the tower as the fireworks start.
    followPath(camera, curve, flat, t, { eye: 1.7, ahead: 0.03, yaw: row[4], pitch: smooth(0, 1, fw) * 0.3,
                                          mx: f.mx, my: f.my, time: f.time });
    sky.position.copy(camera.position);

    // Gloom: fog and fill light go towards blood red for the verdict.
    world.fog.color.copy(NIGHT).lerp(BLOOD, gloom);
    dome.uniforms.horizon.value.set('#2b2230').lerp(tmp.set('#4a1015'), gloom);
    hemi.color.set('#3a4466').lerp(tmp.set('#6a1a1a'), gloom);
    gl.setClearColor(world.fog.color);

    // Spotlight with a little theatrical flicker as it catches.
    spot.intensity = spotOn * 90 * (0.94 + 0.06 * Math.sin(f.time * 23));
    beamMat.opacity = spotOn * 0.05;

    // Carving: stroke 1 over the first half, stroke 2 over the second.
    var tipAt = null;
    strokes.forEach(function (s, k) {
      var p = clamp(carve * 2 - k, 0, 1);
      s.mesh.scale.x = Math.max(p, 0.0001);
      if (p > 0 && p < 1) tipAt = [lerp(s.a[0], s.b[0], p), lerp(s.a[1], s.b[1], p)];
    });
    vLight.intensity = carve * 2.2 + (tipAt ? 3 : 0);
    tip.material.opacity = tipAt ? 1 : 0;
    if (tipAt) {
      tip.position.set(V_AT.x - 0.05, tipAt[0], tipAt[1]);
      for (var n = 0; n < 4; n++) {
        var k3 = (nextSpark++ % SPARKS) * 3;
        sparkPos[k3] = V_AT.x - 0.05; sparkPos[k3 + 1] = tipAt[0]; sparkPos[k3 + 2] = tipAt[1];
        sparkVel[k3] = -1 - Math.random() * 2.5; sparkVel[k3 + 1] = Math.random() * 2.5; sparkVel[k3 + 2] = (Math.random() - 0.5) * 3;
        sparkLife[k3 / 3] = 0.5 + Math.random() * 0.4;
      }
    }
    for (var si = 0; si < SPARKS; si++) {
      var j = si * 3;
      if (sparkLife[si] <= 0) { sparkPos[j + 1] = -50; continue; }
      sparkLife[si] -= dt;
      sparkVel[j + 1] -= 9.8 * dt;
      sparkPos[j] += sparkVel[j] * dt; sparkPos[j + 1] += sparkVel[j + 1] * dt; sparkPos[j + 2] += sparkVel[j + 2] * dt;
    }
    sparkGeo.attributes.position.needsUpdate = true;

    // The rose appears with the giggle; petals fall around it.
    rose.scale.setScalar(Math.max(smooth(0, 0.4, petalAmt), 0.0001));
    petals.update({ snow: petalAmt, wind: f.wind * 0.3, dt: dt, time: f.time }, camera.position, env.reduceMotion);

    rain.update(f, camera.position, f.snow, env.reduceMotion);

    // Fireworks.
    burstClock += dt * fw * (env.reduceMotion ? 1 : 2.6);
    while (burstClock > 1) { burstClock -= 1 + Math.random() * 0.6; launch(); }
    for (var b = 0; b < BURSTS; b++) {
      var age = burstAge[b] += dt, fade = clamp(1 - age / 2.8, 0, 1), sparkle = 0.75 + 0.25 * Math.sin(age * 40 + b);
      for (var p2 = b * PER; p2 < (b + 1) * PER; p2++) {
        var k = p2 * 3;
        if (fade <= 0) { fwCol[k] = fwCol[k + 1] = fwCol[k + 2] = 0; continue; }
        fwVel[k] *= 0.985; fwVel[k + 1] = fwVel[k + 1] * 0.985 - 5.5 * dt; fwVel[k + 2] *= 0.985;
        fwPos[k] += fwVel[k] * dt; fwPos[k + 1] += fwVel[k + 1] * dt; fwPos[k + 2] += fwVel[k + 2] * dt;
        fwCol[k] = fwBase[k] * fade * sparkle; fwCol[k + 1] = fwBase[k + 1] * fade * sparkle; fwCol[k + 2] = fwBase[k + 2] * fade * sparkle;
      }
    }
    fwGeo.attributes.position.needsUpdate = true;
    fwGeo.attributes.color.needsUpdate = true;
    flash.intensity *= Math.exp(-dt * 5);

    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) { fitCamera(gl, camera, w, h, dpr, small); },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('vendetta', {
  renderer: renderer3d,
  emphasis: /^\W*v/i,
  accent: '#ff4436',
  align: ['center', 'left', 'left', 'left', 'left', 'left', 'center'],
  // Panels: 0 "Voilà!", 1-2 the introduction, 3 [carves V into wall],
  // 4 the verdict, 5 [giggles], 6 "Verily ... call me V."
  keys: function (T) {
    var n = T.count;
    function at(i, frac) { i = Math.min(Math.round(i * (n - 1) / 6), n - 1); return lerp(T.start(i), T.end(i), frac); }
    //   unit        path  gloom rain wind  yaw    spot carve fw   petals
    return [
      [0,            0.00, 0.00, 0.60, 0.30, 0.00, 0, 0, 0, 0],
      [0.7,          0.03, 0.00, 0.60, 0.30, 0.00, 0, 0, 0, 0],
      [at(0, 0.15),  0.05, 0.00, 0.55, 0.30, 0.00, 0, 0, 0, 0],
      [at(0, 0.3),   0.06, 0.00, 0.55, 0.30, 0.00, 1, 0, 0, 0],        // "Voilà!"
      [at(1, 0.5),   0.20, 0.00, 0.55, 0.30, -0.12, 1, 0, 0, 0],
      [at(2, 1.0),   0.37, 0.00, 0.60, 0.30, -0.70, 1, 0, 0, 0],
      // Path 0.40 is level with the V; hold there, facing it, through
      // the carving, the verdict and the giggle.
      [at(3, 0.15),  0.400, 0.00, 0.60, 0.30, -1.52, 1, 0, 0, 0],      // [carves V into wall]
      [at(3, 0.95),  0.402, 0.05, 0.60, 0.30, -1.55, 1, 1, 0, 0],
      [at(4, 0.5),   0.404, 0.45, 1.00, 0.65, -1.56, 1, 1, 0, 0],      // "The only verdict is vengeance"
      [at(4, 1.0),   0.406, 0.35, 0.90, 0.55, -1.54, 1, 1, 0, 0],
      [at(5, 0.1),   0.408, 0.15, 0.60, 0.30, -1.48, 1, 1, 0, 0],
      [at(5, 1.0),   0.410, 0.05, 0.50, 0.30, -1.40, 1, 1, 0, 1],      // [giggles]
      [at(6, 0.1),   0.52, 0.00, 0.40, 0.25, -0.50, 0.6, 1, 0.1, 0.6],
      [at(6, 0.6),   0.80, 0.00, 0.30, 0.20, 0.05, 0.2, 1, 0.7, 0.2], // out onto the embankment
      [at(6, 1.0),   0.92, 0.00, 0.20, 0.15, 0.06, 0, 1, 1, 0],        // "you may call me V"
      [T.total,      0.97, 0.00, 0.15, 0.15, 0.06, 0, 1, 1, 0]
    ];
  },
  sound: {
    src: '/audio/rain.mp3',
    label: 'Play rain, the blade and the fireworks',
    cues: [
      { stanza: 3, at: 0.15, play: carveSound },
      { stanza: 6, at: 0.75, play: boom },
      { stanza: 6, at: 1.05, play: boom },
      { stanza: 6, at: 1.4, play: boom }
    ]
  }
});
