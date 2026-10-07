/*
 * Scene for "Invictus" (W. E. Henley): a lone ship on a black sea, and the
 * one lantern on her stern that does not go out.
 *
 * I   "Out of the night that covers me, black as the pit": near-total dark,
 *     close by an unlit lantern; on "I thank whatever gods may be" the flame
 *     catches, and the view draws back over the deck, the sails and the sea.
 * II  "The fell clutch of circumstance ... the bludgeonings of chance": a
 *     storm, rain lashing, heavy seas over the bow, lightning; the flame
 *     gutters but holds, "bloody, but unbowed".
 * III "Looms but the Horror of the shade": the storm thins into fog and a
 *     vast dark wall of cliffs looms out of it; the lantern burns steady,
 *     "unafraid".
 * IV  "How strait the gate": the ship threads a narrow strait between the
 *     cliffs into open water as dawn breaks ahead; "master of my fate ...
 *     captain of my soul" under the rising sun.
 *
 * Columns: [unit, ship z, dark, rain, wind, storm, fog, dawn, flame, back, up, side, lookY]
 */
import { THREE, isSmall, makeRenderer, fitCamera, softSprite, skyDome, rainField,
         waveHeight, oceanMaterial, oceanMesh, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// Swell from ahead and abeam, so she pitches and rolls; chop on top.
var WAVES = [[0.35, -1, 0.09, 1.0, 1.0], [-0.8, -0.5, 0.16, 0.5, 1.4], [0.9, -0.3, 0.29, 0.28, 1.9], [0.2, -1, 0.7, 0.09, 2.8]];
var GATE_Z = -600;                                        // the middle of the strait
var SUN = new THREE.Vector3(0.16, 0.02, -1).normalize();  // dawn, ahead through the gate
var BOLTS = [];                                           // timeline points for scripted lightning

// ── Noise for the rock ───────────────────────────────────────────────────
function hash(x, y) { var h = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return h - Math.floor(h); }
function noise(x, y) {
  var ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy;
  fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
  return lerp(lerp(hash(ix, iy), hash(ix + 1, iy), fx), lerp(hash(ix, iy + 1), hash(ix + 1, iy + 1), fx), fy) * 2 - 1;
}
function fbm(x, y) { return noise(x, y) * 0.55 + noise(x * 2.1 + 5.2, y * 2.1 + 1.3) * 0.28 + noise(x * 4.4 + 9.1, y * 4.4 + 7.7) * 0.14; }

// ── Sound ────────────────────────────────────────────────────────────────
// Thunder: a crack, then a long low roll, a beat after the flash.
function thunder(ac, out) {
  var t = ac.currentTime + 0.35, len = 4.5, b = ac.createBuffer(1, ac.sampleRate * len, ac.sampleRate), d = b.getChannelData(0);
  for (var i = 0, last = 0; i < d.length; i++) {
    last = last * 0.96 + (Math.random() * 2 - 1) * 0.04;                 // brown-ish rumble
    var k = i / d.length;
    d[i] = (last * 6 + (k < 0.03 ? (Math.random() * 2 - 1) * (1 - k / 0.03) : 0)) * Math.pow(1 - k, 1.6);
  }
  var src = ac.createBufferSource(), lp = ac.createBiquadFilter(), g = ac.createGain();
  src.buffer = b;
  lp.type = 'lowpass';
  lp.frequency.setValueAtTime(1400, t);
  lp.frequency.exponentialRampToValueAtTime(160, t + 1.5);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(1.0, t + 0.04);
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

function flameTexture() {
  return canvasTex(64, 128, function (x, w, h) {
    x.save();
    x.translate(32, 84);
    x.scale(1, 2.2);
    var g = x.createRadialGradient(0, 0, 0, 0, 0, 26);
    g.addColorStop(0, 'rgba(255,255,235,1)');
    g.addColorStop(0.25, 'rgba(255,214,120,0.95)');
    g.addColorStop(0.6, 'rgba(255,120,30,0.35)');
    g.addColorStop(1, 'rgba(255,80,10,0)');
    x.fillStyle = g;
    x.beginPath(); x.arc(0, 0, 26, 0, Math.PI * 2); x.fill();
    x.restore();
  });
}

// Foam trailing from the stern: streaks that fan out and fade.
function wakeTexture(r) {
  return canvasTex(128, 512, function (x, w, h) {
    for (var i = 0; i < 900; i++) {
      var v = r(), spread = 0.12 + v * 0.88, u = 0.5 + (r() - 0.5) * spread;
      x.fillStyle = 'rgba(230,236,240,' + (0.5 * (1 - v) * (0.3 + r() * 0.7)).toFixed(3) + ')';
      x.beginPath();
      x.ellipse(u * w, v * h, 1 + r() * 3, 3 + r() * 14, 0, 0, Math.PI * 2);
      x.fill();
    }
  });
}

// A small brig, about 15 m long, bow towards -z. Returns the group and the
// pieces the frame loop animates.
function brig(small) {
  var g = new THREE.Group();
  var hullMat = new THREE.MeshStandardMaterial({ color: '#2e2119', roughness: 0.7 });
  var deckMat = new THREE.MeshStandardMaterial({ color: '#6e5a45', roughness: 0.85 });
  var iron = new THREE.MeshStandardMaterial({ color: '#141210', roughness: 0.6, metalness: 0.4 });
  var spar = new THREE.MeshStandardMaterial({ color: '#2a1f17', roughness: 0.8 });
  var cloth = new THREE.MeshStandardMaterial({ color: '#cfc2a4', roughness: 0.95, side: THREE.DoubleSide });
  var FREE = 1.3;                                       // deck height above the water

  function sheer(z) { return Math.max(0, -z - 2) * 0.09 + Math.max(0, z - 4) * 0.06; }
  var half = [[0, -8.4], [1.05, -6.6], [1.8, -4.2], [2.15, -1.2], [2.2, 2.2], [2.05, 4.8], [1.6, 6.5], [0.0, 7.0]];
  function outline(scale) {
    var s = new THREE.Shape(), pts = [];
    half.forEach(function (p) { pts.push(new THREE.Vector2(p[0] * scale, p[1] * (0.6 + 0.4 * scale))); });
    for (var i = half.length - 2; i > 0; i--) pts.push(new THREE.Vector2(-half[i][0] * scale, half[i][1] * (0.6 + 0.4 * scale)));
    s.moveTo(pts[0].x, pts[0].y);
    s.splineThru(pts.slice(1).concat([pts[0]]));
    return s;
  }
  // Hull: the deck outline extruded down, narrowing to the keel.
  var hullGeo = new THREE.ExtrudeGeometry(outline(1), { depth: 2.6, bevelEnabled: false, curveSegments: 6 }).rotateX(Math.PI / 2);
  var hp = hullGeo.attributes.position;
  for (var i = 0; i < hp.count; i++) {
    var y = hp.getY(i), z = hp.getZ(i), t = -y / 2.6;
    hp.setX(i, hp.getX(i) * (1 - 0.6 * t * t * t));
    hp.setY(i, y + FREE + sheer(z));
  }
  hullGeo.computeVertexNormals();
  var hull = new THREE.Mesh(hullGeo, [deckMat, hullMat]);
  // Bulwarks: a ring of planking round the deck edge.
  var ring = outline(1);
  ring.holes.push(new THREE.Path(outline(0.9).getPoints(10).reverse()));
  var bulGeo = new THREE.ExtrudeGeometry(ring, { depth: 0.6, bevelEnabled: false, curveSegments: 6 }).rotateX(Math.PI / 2).translate(0, 0.6, 0);
  var bp = bulGeo.attributes.position;
  for (i = 0; i < bp.count; i++) bp.setY(i, bp.getY(i) + FREE + sheer(bp.getZ(i)));
  bulGeo.computeVertexNormals();
  var bulwark = new THREE.Mesh(bulGeo, hullMat);
  // A painted wale just below the rail.
  var band = outline(1.012);
  band.holes.push(new THREE.Path(outline(0.95).getPoints(10).reverse()));
  var waleGeo = new THREE.ExtrudeGeometry(band, { depth: 0.16, bevelEnabled: false, curveSegments: 6 }).rotateX(Math.PI / 2);
  var wv = waleGeo.attributes.position;
  for (i = 0; i < wv.count; i++) wv.setY(i, wv.getY(i) + FREE - 0.12 + sheer(wv.getZ(i)));
  waleGeo.computeVertexNormals();
  var wale = new THREE.Mesh(waleGeo, new THREE.MeshStandardMaterial({ color: '#8a6a3a', roughness: 0.7 }));
  g.add(hull, bulwark, wale);

  // Deckhouse aft, and the wheel just behind it, facing the stern.
  var house = new THREE.Mesh(new THREE.BoxGeometry(2.4, 0.9, 2.6), hullMat);
  house.position.set(0, FREE + 0.45, 2.6);
  var roof = new THREE.Mesh(new THREE.BoxGeometry(2.6, 0.08, 2.8), deckMat);
  roof.position.set(0, FREE + 0.94, 2.6);
  var wheel = new THREE.Group();
  wheel.add(new THREE.Mesh(new THREE.TorusGeometry(0.55, 0.035, 6, 24), spar));
  for (i = 0; i < 4; i++) {
    var spoke = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 1.45, 4), spar);
    spoke.rotation.z = i * Math.PI / 4;
    wheel.add(spoke);
  }
  wheel.position.set(0, FREE + 1.0, 4.6);
  var binnacle = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.1, 0.9, 6), spar);
  binnacle.position.set(0, FREE + 0.45, 4.5);
  g.add(house, roof, wheel, binnacle);

  // Masts, yards and square sails bellied forward.
  var masts = [{ z: -3.2, h: 14 }, { z: 1.0, h: 15.5 }], sails = [];
  masts.forEach(function (m) {
    var mast = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.17, m.h, 8), spar);
    mast.position.set(0, FREE + m.h / 2, m.z);
    g.add(mast);
    [[4.2, 5.2, 0.32], [3.5, 8.9, 0.27], [2.6, 12.0, 0.22]].forEach(function (s, k) {
      var w = s[0] * 2, top = Math.min(s[1] + (m.h - 14) * 0.6, m.h - 0.4), hgt = k === 0 ? 3.2 : 3.0;
      var yard = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, w + 0.6, 6).rotateZ(Math.PI / 2), spar);
      yard.position.set(0, FREE + top, m.z - 0.2);
      g.add(yard);
      var geo = new THREE.PlaneGeometry(w, hgt, 8, 6), gp = geo.attributes.position;
      for (var v = 0; v < gp.count; v++) gp.setX(v, gp.getX(v) * (0.9 - 0.1 * gp.getY(v) / hgt * 2));   // narrower at the head
      var base = new Float32Array(gp.array);
      var sail = new THREE.Mesh(geo, cloth);
      sail.position.set(0, FREE + top - hgt / 2 - 0.05, m.z - 0.3);
      sail.userData = { base: base, w: w, h: hgt, belly: s[2] * 4, phase: k + m.z };
      g.add(sail);
      sails.push(sail);
    });
  });
  // Bowsprit and a jib.
  var sprit = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.12, 6, 6), spar);
  sprit.rotation.x = -(Math.PI / 2 - 0.22);
  sprit.position.set(0, FREE + 1.0, -10.6);
  g.add(sprit);
  var jibShape = new THREE.Shape();
  jibShape.moveTo(0, 0); jibShape.lineTo(5.6, 0); jibShape.lineTo(0, 9.5); jibShape.lineTo(0, 0);
  var jib = new THREE.Mesh(new THREE.ShapeGeometry(jibShape), cloth);
  jib.rotation.y = Math.PI / 2;
  jib.position.set(0, FREE + 1.6, -6.0);
  g.add(jib);

  // Rigging: shrouds to the rails, stays fore and aft.
  var rig = [];
  function line(a, b) { rig.push(a[0], a[1], a[2], b[0], b[1], b[2]); }
  masts.forEach(function (m) {
    var top = FREE + m.h - 0.3;
    for (var s = -1; s <= 1; s += 2) for (var k = 0; k < 3; k++) line([0, top - 2, m.z], [s * 2.05, FREE + 0.6, m.z + 0.6 + k * 0.7]);
  });
  line([0, FREE + 13.6, -3.2], [0, FREE + 2.1, -13.3]);
  line([0, FREE + 15.1, 1.0], [0, FREE + 13.5, -3.2]);
  line([0, FREE + 15.1, 1.0], [0, FREE + 1.6, 6.6]);
  line([0, FREE + 9.5, -3.2], [0, FREE + 1.6, -8.3]);
  var rigGeo = new THREE.BufferGeometry();
  rigGeo.setAttribute('position', new THREE.Float32BufferAttribute(rig, 3));
  g.add(new THREE.LineSegments(rigGeo, new THREE.LineBasicMaterial({ color: '#0e0b09', transparent: true, opacity: 0.85 })));

  // The lantern: an iron post on the taffrail and a glass lamp hung from it.
  var LY = FREE + sheer(6.7);
  var post = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.045, 1.9, 6), iron);
  post.position.set(0, LY + 1.2, 6.7);
  var arm = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.7, 5).rotateX(Math.PI / 2), iron);
  arm.position.set(0, LY + 2.1, 7.0);
  g.add(post, arm);
  var lamp = new THREE.Group();                          // pivots at the hook, out over the stern
  lamp.position.set(0, LY + 2.08, 7.3);
  var body = new THREE.Group();
  body.position.y = -0.36;
  var glass = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.32, 0.24),
    new THREE.MeshBasicMaterial({ color: '#ffc477', transparent: true, opacity: 0.35, depthWrite: false }));
  var cap = new THREE.Mesh(new THREE.ConeGeometry(0.21, 0.18, 4).rotateY(Math.PI / 4), iron);
  cap.position.y = 0.25;
  var base = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.05, 0.28), iron);
  base.position.y = -0.185;
  body.add(glass, cap, base);
  for (i = 0; i < 4; i++) {
    var bar = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.34, 0.02), iron);
    bar.position.set((i & 1 ? 1 : -1) * 0.12, 0, (i & 2 ? 1 : -1) * 0.12);
    body.add(bar);
  }
  var hook = new THREE.Mesh(new THREE.TorusGeometry(0.05, 0.01, 4, 10), iron);
  hook.position.y = 0.36;
  body.add(hook);
  var flame = new THREE.Sprite(new THREE.SpriteMaterial({ map: flameTexture(), blending: THREE.AdditiveBlending,
                                                           depthWrite: false, transparent: true }));
  flame.scale.set(0.11, 0.22, 1);
  flame.position.y = -0.02;
  var glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,190,110,1)', 'rgba(255,140,60,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
  glow.scale.setScalar(2.4);
  var halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,200,140,0.5)', 'rgba(255,160,90,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0 }));
  halo.scale.setScalar(14);
  var light = new THREE.PointLight('#ffb15c', 0, 34, 1.4);
  light.castShadow = !small;
  light.shadow.mapSize.set(512, 512);
  light.shadow.bias = -0.004;
  light.shadow.normalBias = 0.03;
  body.add(flame, glow, halo, light);
  lamp.add(body);
  g.add(lamp);

  g.traverse(function (o) { if (o.isMesh && o.material !== glass.material) { o.castShadow = !small; o.receiveShadow = !small; } });
  glass.castShadow = false;
  return { group: g, sails: sails, cloth: cloth, lamp: lamp, flame: flame, glow: glow, halo: halo, light: light, glass: glass };
}

// The coast: two masses of rock with a narrow strait between them. Each
// is a box bent into shape: the inner face follows the channel, the top
// is a ragged ridge, the seaward faces are broken cliffs.
function cliffs(r) {
  var mat = new THREE.MeshStandardMaterial({ color: '#2a2c30', roughness: 0.95, flatShading: true, vertexColors: true });
  var g = new THREE.Group(), c = new THREE.Color(), DEPTH = 130;
  function halfWidth(z) { var d = z - GATE_Z; return 12 + 0.0055 * d * d; }
  [-1, 1].forEach(function (side) {
    var geo = new THREE.BoxGeometry(1, 1, 1, 56, 18, 34), p = geo.attributes.position, cols = [];
    for (var i = 0; i < p.count; i++) {
      var a = side > 0 ? p.getX(i) + 0.5 : 0.5 - p.getX(i), b = p.getY(i) + 0.5, w = p.getZ(i) + 0.5;
      var z = GATE_Z + (w - 0.5) * DEPTH;
      var x = side * (halfWidth(z) + Math.pow(a, 2.4) * 1500);
      var top = 82 + 30 * fbm(x * 0.008 + side * 3, z * 0.02) - smooth(150, 1400, Math.abs(x)) * 38 + 20 * (1 - smooth(10, 120, Math.abs(x)));
      var y = lerp(-14, top, b);
      // Rough faces: the channel wall, the seaward cliffs, a ragged crest.
      x += side * 7 * fbm(y * 0.05 + side, z * 0.045) * (1 - a);
      z += (w > 0.5 ? 1 : -1) * 9 * fbm(x * 0.03, y * 0.05) * Math.abs(w - 0.5) * 2;
      y += b > 0.99 ? 6 * fbm(x * 0.05, z * 0.05) : 0;
      p.setXYZ(i, x, y, z);
      var shade = 0.75 + 0.25 * fbm(x * 0.1, y * 0.13) + 0.1 * smooth(0, 60, y);
      c.setRGB(0.17 * shade, 0.18 * shade, 0.2 * shade);
      cols.push(c.r, c.g, c.b);
    }
    geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
    geo.computeVertexNormals();
    var m = new THREE.Mesh(geo, mat);
    m.frustumCulled = false;
    g.add(m);
  });
  // Fallen rocks at the feet of the cliffs.
  var rock = new THREE.MeshStandardMaterial({ color: '#1d1f22', roughness: 0.95, flatShading: true });
  for (var k = 0; k < 26; k++) {
    var s = r() < 0.5 ? -1 : 1, z = GATE_Z + (r() - 0.5) * 160, m2 = new THREE.Mesh(new THREE.DodecahedronGeometry(1.5 + r() * 4, 0), rock);
    m2.position.set(s * (halfWidth(z) + r() * 3), -0.8 + r() * 1.2, z);
    m2.rotation.set(r() * 3, r() * 3, r() * 3);
    g.add(m2);
  }
  return g;
}

// ── Renderer ─────────────────────────────────────────────────────────────
function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(17);
  var gl = makeRenderer(canvas, { clear: '#000000', shadows: !small });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#020304', 0.004);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.1, 4000);

  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#000000', mid: '#010203', horizon: '#030406', sun: '#ffb36b' }, 2000);
  dome.uniforms.sunDir.value.copy(SUN);
  sky.add(dome.mesh);

  // Storm clouds: a low, heavy ceiling that lightning lights from within.
  var cloudTex = softSprite('rgba(255,255,255,0.9)', 'rgba(255,255,255,0)'), clouds = [];
  for (var i = 0; i < 40; i++) {
    var cl = new THREE.Sprite(new THREE.SpriteMaterial({ map: cloudTex, color: '#20242c', transparent: true, depthWrite: false, fog: false, opacity: 0 }));
    var a = r() * Math.PI * 2, el = 0.05 + r() * 0.45;
    cl.position.set(Math.cos(a) * Math.cos(el) * 1500, Math.sin(el) * 1500, Math.sin(a) * Math.cos(el) * 1500);
    cl.scale.set(600 + r() * 500, 260 + r() * 200, 1);
    sky.add(cl);
    clouds.push(cl);
  }
  var flashSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: cloudTex, color: '#c9d4ff', transparent: true, opacity: 0,
                                                                 blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
  flashSprite.scale.set(1600, 800, 1);
  sky.add(flashSprite);

  // A forked bolt, redrawn for every strike.
  var BOLT_SEG = 90, boltPos = new Float32Array(BOLT_SEG * 6), boltGeo = new THREE.BufferGeometry();
  boltGeo.setAttribute('position', new THREE.BufferAttribute(boltPos, 3));
  var bolt = new THREE.LineSegments(boltGeo, new THREE.LineBasicMaterial({ color: '#eef2ff', transparent: true, opacity: 0,
                                                                          blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
  bolt.frustumCulled = false;
  world.add(bolt);
  function strike(origin) {
    var n = 0, x = origin.x + (Math.random() - 0.5) * 500, z = origin.z - 260 - Math.random() * 300, y = 260;
    flashSprite.position.set(x - origin.x, 330, z - origin.z);
    function branch(x0, y0, z0, steps, spread) {
      for (var s = 0; s < steps && n < BOLT_SEG; s++) {
        var x1 = x0 + (Math.random() - 0.5) * spread, y1 = y0 - 8 - Math.random() * 14, z1 = z0 + (Math.random() - 0.5) * spread * 0.5;
        if (y1 < 0) y1 = 0;
        boltPos.set([x0, y0, z0, x1, y1, z1], n * 6);
        n++;
        if (Math.random() < 0.12 && steps > 6) branch(x1, y1, z1, Math.floor(steps * 0.35), spread * 0.8);
        x0 = x1; y0 = y1; z0 = z1;
        if (y0 <= 0) break;
      }
    }
    branch(x, y, z, 26, 16);
    for (var k = n; k < BOLT_SEG; k++) boltPos.fill(0, k * 6, k * 6 + 6);
    boltGeo.attributes.position.needsUpdate = true;
    flash = 1;
  }

  var hemi = new THREE.HemisphereLight('#7a86a8', '#0a0d12', 0.05);
  var sun = new THREE.DirectionalLight('#ffc08a', 0);
  world.add(hemi, sun, sun.target);

  var oceanMat = oceanMaterial({ color: '#0a1520', specular: '#a4b2c4', shininess: 70, foam: '#cdd6de', waves: WAVES });
  var ocean = oceanMesh(oceanMat, 900, small ? 170 : 260);
  ocean.receiveShadow = !small;
  world.add(ocean);
  var su = oceanMat.userData.uniforms;

  var ship = brig(small);
  world.add(ship.group);
  var coast = cliffs(r);
  world.add(coast);

  var wakeGeo = new THREE.PlaneGeometry(1, 1, 6, 16).rotateX(-Math.PI / 2), wp = wakeGeo.attributes.position;
  var wakeBase = new Float32Array(wp.array);
  var wake = new THREE.Mesh(wakeGeo, new THREE.MeshLambertMaterial({ map: wakeTexture(r), transparent: true, depthWrite: false, opacity: 0.5 }));
  wake.frustumCulled = false;
  world.add(wake);

  var rain = rainField({ count: small ? 1400 : 3200, box: [26, 18, 34], speed: 16, windSpeed: 7, opacity: 0.32, color: '#a9b6cc' });
  world.add(rain.lines);

  // Spray thrown back over the bow when she buries it in a sea.
  var SPRAY = small ? 300 : 600, sprayPos = new Float32Array(SPRAY * 3), sprayVel = new Float32Array(SPRAY * 3), sprayLife = new Float32Array(SPRAY);
  var sprayGeo = new THREE.BufferGeometry();
  sprayGeo.setAttribute('position', new THREE.BufferAttribute(sprayPos, 3));
  var spray = new THREE.Points(sprayGeo, new THREE.PointsMaterial({ color: '#c8d2dc', size: 0.22, transparent: true, depthWrite: false, opacity: 0.55,
    map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)') }));
  spray.frustumCulled = false;
  world.add(spray);
  for (i = 0; i < SPRAY; i++) sprayPos[i * 3 + 1] = -99;

  // Low banks of fog drifting across the water.
  var fogTex = softSprite('rgba(200,206,214,0.6)', 'rgba(200,206,214,0)'), banks = [];
  for (i = 0; i < (small ? 14 : 26); i++) {
    var fb = new THREE.Sprite(new THREE.SpriteMaterial({ map: fogTex, transparent: true, depthWrite: false, opacity: 0, color: '#8a94a0' }));
    fb.userData = { x: (r() - 0.5) * 160, z: -20 - r() * 140, y: 2 + r() * 8, s: 40 + r() * 50 };
    world.add(fb);
    banks.push(fb);
  }

  // ── Per frame ─────────────────────────────────────────────────────────
  var C = {
    pit: ['#000000', '#010203', '#030406'], storm: ['#05070a', '#0c1016', '#171d26'],
    fog: ['#353d47', '#3c4550', '#434c57'], dawn: ['#3a5684', '#9a95a8', '#f3b27c']
  };
  var tmp = new THREE.Color(), tmp2 = new THREE.Color(), horizon = new THREE.Color();
  var e = new THREE.Euler(), q = new THREE.Quaternion(), look = new THREE.Vector3(), v = new THREE.Vector3();
  var aspect = 1.6, flash = 0, sprayClock = 0, lastU = 0, camY = 3.5, roll = 0, gust = 0;

  function mixSky(k, storm, fog, dawn, dark) {
    tmp.set(C.pit[k]).lerp(tmp2.set(C.storm[k]), clamp(storm * 1.4, 0, 1) * (1 - fog));
    tmp.lerp(tmp2.set(C.fog[k]), fog);
    tmp.lerp(tmp2.set(C.dawn[k]), dawn);
    return tmp.multiplyScalar(1 - dark * 0.7);
  }

  function frame(f) {
    var row = f.row, time = f.time, dt = f.dt;
    var shipZ = row[0], dark = row[1], storm = row[4], fog = row[5], dawn = row[6], flameAmt = row[7];
    var amp = lerp(0.3, 2.3, storm);
    su.uTime.value = time;
    su.uAmp.value = amp;
    su.uFoamAmt.value = smooth(0.2, 1, storm) * 1.4;

    // The ship rides the sea: heave, pitch and roll from four points.
    var hb = waveHeight(WAVES, 0, shipZ - 7, time, amp).h, hs = waveHeight(WAVES, 0, shipZ + 6, time, amp).h;
    var hp = waveHeight(WAVES, -2, shipZ, time, amp).h, hst = waveHeight(WAVES, 2, shipZ, time, amp).h;
    ship.group.position.set(0, (hb + hs + hp + hst) * 0.25 - 0.6, shipZ);
    e.set(Math.atan((hb - hs) / 13) * 0.85, 0, Math.atan((hp - hst) / 4) * 0.55);
    ship.group.quaternion.slerp(q.setFromEuler(e), 1 - Math.exp(-dt * 3));

    // Sails fill and, in the gale, shiver.
    ship.sails.forEach(function (s) {
      var ud = s.userData, pa = s.geometry.attributes.position, flap = storm * 0.25;
      for (var k = 0; k < pa.count; k++) {
        var bx = ud.base[k * 3], by = ud.base[k * 3 + 1], nx = bx / ud.w * 2, ny = by / ud.h + 0.5;
        var belly = (1 - nx * nx) * (0.35 + 0.65 * Math.sin(ny * Math.PI * 0.9 + 0.2)) * ud.belly;
        pa.setZ(k, -belly * (1 + flap * Math.sin(time * 9 + ud.phase + bx * 1.3 + by)));
      }
      pa.needsUpdate = true;
      s.geometry.computeVertexNormals();
    });

    // Camera: in the ship's frame, but only half-following her roll.
    var side = row[10] * (aspect < 1 ? lerp(0.4, 0.55, smooth(2, 10, Math.abs(row[10]))) : 1);             // portrait: keep the lantern in shot
    var cv = waveHeight(WAVES, side, shipZ + row[8], time, amp).h;
    camY += ((ship.group.position.y + 0.6) * 0.7 + cv * 0.3 + row[9] - camY) * (1 - Math.exp(-dt * 4));
    camera.position.set(side, camY, shipZ + row[8]);
    look.set(side * lerp(0.25, 1, smooth(2, 10, Math.abs(side))), ship.group.position.y + row[11], shipZ + row[8] - 40);
    camera.lookAt(look);
    camera.rotateY(-f.mx * 0.14);
    camera.rotateX(-f.my * 0.06);
    roll += (Math.atan((hp - hst) / 4) * 0.3 - roll) * (1 - Math.exp(-dt * 3));
    camera.rotateZ(roll);
    sky.position.copy(camera.position);
    ocean.userData.follow(camera.position);

    // Sky: the pit, the storm, the fog, the dawn.
    dome.uniforms.top.value.copy(mixSky(0, storm, fog, dawn, dark));
    dome.uniforms.mid.value.copy(mixSky(1, storm, fog, dawn, dark));
    horizon.copy(mixSky(2, storm, fog, dawn, dark));
    dome.uniforms.horizon.value.copy(horizon).lerp(tmp2.set('#7d88a8'), flash * 0.5);
    dome.uniforms.sunColor.value.set('#ff9a50').multiplyScalar(dawn * 1.2);
    var sunUp = smooth(0.45, 1, dawn);
    v.copy(SUN).setY(lerp(-0.04, 0.07, sunUp)).normalize();
    dome.uniforms.sunDir.value.copy(v);
    world.fog.color.copy(horizon).multiplyScalar(lerp(0.85, 1, fog)).lerp(tmp2.copy(dome.uniforms.mid.value).multiplyScalar(0.75), dawn * 0.7);
    coast.visible = shipZ < -125;
    world.fog.density = lerp(0.0028, 0.0105, storm * (1 - fog)) + fog * 0.0036 - dawn * 0.0012;
    gl.setClearColor(world.fog.color);
    su.uSky.value.copy(horizon).lerp(dome.uniforms.mid.value, dawn * 0.85).multiplyScalar(lerp(0.6, 0.38, dawn)).lerp(tmp2.set('#c0c8ff'), flash * 0.4);

    sun.position.copy(camera.position).addScaledVector(v, 300);
    sun.target.position.copy(camera.position);
    sun.intensity = dawn * 2.4;
    hemi.intensity = (0.05 + storm * 0.45 + fog * 0.55 + dawn * 0.55) * (1 - dark * 0.8) + flash * 5;
    ship.cloth.emissive.set('#ff9a50').multiplyScalar(dawn * 0.22);       // canvas glowing with the sun behind it
    hemi.color.set('#7a86a8').lerp(tmp2.set('#ffd2a8'), dawn);

    // Lightning: random in the worst of it, plus the scripted strikes the
    // thunder cues are timed to.
    for (var b = 0; b < BOLTS.length; b++) if (lastU < BOLTS[b] && f.u >= BOLTS[b]) strike(camera.position);
    lastU = f.u;
    if (storm > 0.7 && Math.random() < dt * 0.5 * storm) strike(camera.position);
    flash *= Math.exp(-dt * (flash > 0.6 ? 9 : 4));
    var flicker = flash > 0.15 ? (0.6 + 0.4 * Math.sin(time * 90)) : 1;
    bolt.material.opacity = smooth(0.3, 0.8, flash) * flicker;
    flashSprite.material.opacity = flash * 0.8 * flicker;
    clouds.forEach(function (cl) {
      cl.material.opacity = clamp(storm * 1.2 - fog * 0.8 - dawn, 0, 0.95);
      cl.material.color.set('#1a1e25').lerp(tmp2.set('#8a94b4'), flash * 0.7);
      cl.position.x += dt * (4 + storm * 30); if (cl.position.x > 1500) cl.position.x -= 3000;
    });

    // The lantern: dead, then lit; guttering in the gale, never out.
    gust += ((noise(time * 1.7, 3.1) * 0.5 + 0.5) - gust) * (1 - Math.exp(-dt * 6));
    var gutter = storm * smooth(0.35, 0.9, gust) * 0.65;
    var flick = (0.88 + 0.12 * Math.sin(time * 11) * Math.sin(time * 4.3)) * (1 - gutter) * (0.85 + 0.15 * noise(time * 14, 0.5));
    var lit = flameAmt * flick;
    ship.flame.scale.set(0.11 * (0.4 + 0.6 * lit), 0.22 * (0.3 + 0.7 * lit) * (1 + storm * 0.2 * Math.sin(time * 23)), 1);
    ship.flame.material.opacity = clamp(0.25 + lit, 0, 1);
    ship.flame.material.color.set('#ff6a2a').lerp(tmp2.set('#ffffff'), smooth(0.1, 0.7, flameAmt));
    ship.glow.material.opacity = 0.08 + lit * 0.75;
    ship.glow.scale.setScalar(1.2 + lit * 1.6);
    ship.halo.material.opacity = lit * (0.12 + fog * 0.35 + storm * 0.08) * (1 - dawn * 0.7);
    ship.light.intensity = 0.2 + lit * 14;
    ship.glass.material.opacity = 0.18 + lit * 0.4;
    ship.lamp.rotation.set(-ship.group.rotation.x * 0.8 + Math.sin(time * 1.3) * 0.04 * (1 + storm * 3),
                           0, -ship.group.rotation.z * 0.8 + Math.sin(time * 1.1 + 1) * 0.05 * (1 + storm * 3));

    // Wake: a fan of foam from the stern, laid on the waves.
    for (var k = 0; k < wp.count; k++) {
      var lz = wakeBase[k * 3 + 2] + 0.5, lx = wakeBase[k * 3];
      var wx = lx * (3 + lz * 22), wz = shipZ + 6.5 + lz * 70;
      wp.setXYZ(k, wx, waveHeight(WAVES, wx, wz, time, amp).h + 0.06, wz);
    }
    wp.needsUpdate = true;
    wake.material.opacity = 0.35 * (1 - storm * 0.5) + 0.1;

    // Spray over the bow, more when she pitches into a sea.
    sprayClock += dt * storm * (6 + Math.max(0, -e.x) * 120);
    ship.group.updateMatrixWorld();
    while (sprayClock > 1) {
      sprayClock -= 1;
      v.set((Math.random() - 0.5) * 3, 1.2, -7.5).applyMatrix4(ship.group.matrixWorld);
      for (var n = 0; n < 14; n++) {
        var s = Math.floor(Math.random() * SPRAY), j = s * 3;
        sprayPos[j] = v.x + (Math.random() - 0.5) * 2; sprayPos[j + 1] = v.y; sprayPos[j + 2] = v.z;
        sprayVel[j] = (Math.random() - 0.5) * 4 + f.wind * 2; sprayVel[j + 1] = 3 + Math.random() * 5; sprayVel[j + 2] = 4 + Math.random() * 7;
        sprayLife[s] = 1.6;
      }
    }
    for (var p = 0; p < SPRAY; p++) {
      var w = p * 3;
      if (sprayLife[p] <= 0) { sprayPos[w + 1] = -99; continue; }
      sprayLife[p] -= dt;
      sprayVel[w + 1] -= 9 * dt;
      sprayPos[w] += sprayVel[w] * dt; sprayPos[w + 1] += sprayVel[w + 1] * dt; sprayPos[w + 2] += sprayVel[w + 2] * dt;
    }
    sprayGeo.attributes.position.needsUpdate = true;

    banks.forEach(function (fb, k) {
      var ud = fb.userData, x = ud.x + ((time * (1.5 + k % 3)) % 160);
      fb.position.set(x > 80 ? x - 160 : x, ud.y, shipZ + ud.z);
      fb.scale.set(ud.s * 2.2, ud.s * 0.35, 1);
      fb.material.opacity = fog * 0.32;
    });

    rain.update(f, camera.position, f.snow, env.reduceMotion);
    gl.toneMappingExposure = 1 + dawn * 0.1;
    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) { aspect = w / h; fitCamera(gl, camera, w, h, dpr, small); },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('captain', {
  renderer: renderer3d,
  align: ['left', 'right', 'left', 'left'],
  scrim: 0.6,
  keys: function (T) {
    function at(i, frac) { i = Math.min(i, T.count - 1); return lerp(T.start(i), T.end(i), frac); }
    BOLTS = [at(1, 0.22), at(1, 0.5), at(1, 0.78)];
    //  unit         shipZ  dark rain wind storm fog  dawn flame back  up    side  lookY
    return [
      [0,            0,     1.0, 0.0, 0.10, 0.05, 0.0, 0.0, 0.05, 9.9,  3.45, -0.85, 1.6],
      [0.7,          -2,    1.0, 0.0, 0.10, 0.05, 0.0, 0.0, 0.05, 9.8,  3.45, -0.85, 1.6],
      [at(0, 0.45),  -8,    1.0, 0.0, 0.10, 0.08, 0.0, 0.0, 0.08, 9.7,  3.45, -0.85, 1.6],   // "black as the pit"
      [at(0, 0.62),  -12,   0.9, 0.0, 0.12, 0.10, 0.0, 0.0, 1.00, 9.8,  3.50, -0.85, 1.6],   // "I thank whatever gods may be"
      [at(0, 0.95),  -20,   0.7, 0.0, 0.20, 0.14, 0.0, 0.0, 1.00, 14.0, 4.40, -0.40, 2.2],   // "my unconquerable soul"
      [at(1, 0.08),  -32,   0.5, 0.2, 0.40, 0.40, 0.0, 0.0, 1.00, 19.0, 6.00, 1.40, 3.4],
      [at(1, 0.35),  -55,   0.4, 0.9, 0.85, 0.90, 0.0, 0.0, 0.85, 18.0, 6.40, 2.60, 4.2],   // "the fell clutch of circumstance"
      [at(1, 0.6),   -80,   0.4, 1.0, 1.00, 1.00, 0.0, 0.0, 0.70, 18.0, 6.40, 2.60, 4.2],   // "the bludgeonings of chance"
      [at(1, 0.85),  -105,  0.4, 0.9, 0.90, 0.95, 0.0, 0.0, 0.95, 18.0, 6.20, 2.20, 4.2],   // "bloody, but unbowed"
      [at(2, 0.12),  -150,  0.4, 0.2, 0.45, 0.45, 0.45, 0.0, 1.00, 19.0, 5.00, 1.20, 4.5],
      [at(2, 0.4),   -230,  0.3, 0.0, 0.25, 0.20, 1.00, 0.0, 1.00, 17.0, 4.00, 0.70, 7.0],   // "Looms but the Horror of the shade"
      [at(2, 0.75),  -330,  0.3, 0.0, 0.20, 0.15, 0.90, 0.0, 1.00, 16.0, 3.80, 0.50, 9.0],   // "the menace of the years"
      [at(2, 1.0),   -420,  0.3, 0.0, 0.20, 0.12, 0.65, 0.06, 1.00, 16.0, 3.80, 0.40, 9.0],  // "unafraid"
      [at(3, 0.3),   -560,  0.2, 0.0, 0.20, 0.10, 0.25, 0.30, 1.00, 16.0, 4.00, 0.00, 7.0],  // "how strait the gate"
      [at(3, 0.6),   -640,  0.1, 0.0, 0.20, 0.08, 0.05, 0.70, 1.00, 18.0, 4.60, 0.00, 5.0],  // "the master of my fate"
      [at(3, 0.95),  -700,  0.0, 0.0, 0.20, 0.06, 0.00, 1.00, 1.00, 28.0, 8.00, -4.00, 7.0],  // "the captain of my soul"
      [T.total,      -780,  0.0, 0.0, 0.20, 0.05, 0.00, 1.00, 1.00, 46.0, 14.0, -14.0, 14.0]
    ];
  },
  sound: {
    src: '/audio/ocean.mp3',
    label: 'Play the sea and the storm',
    volume: function (row) { return 0.1 + 0.6 * row[4] + 0.06 * row[6]; },
    cues: [{ stanza: 1, at: 0.22 * 1.6, play: thunder }, { stanza: 1, at: 0.5 * 1.6, play: thunder }, { stanza: 1, at: 0.78 * 1.6, play: thunder }]
  }
});
