/*
 * Scene for Sonnet 18, "Shall I compare thee to a summer's day?"
 * (Shakespeare): an English orchard opening onto a meadow, through one
 * summer and then out of time.
 *
 * I   A May morning down an orchard aisle in blossom. "Rough winds do shake
 *     the darling buds of May": the trees toss, the petals stream away, and
 *     the blossom is gone almost as soon as it came, "all too short a date".
 * II  "Sometime too hot the eye of heaven shines": you look up into a white
 *     blazing sun and the air shimmers; "his gold complexion dimm'd": clouds
 *     cross it; "every fair from fair sometime declines": the meadow yellows,
 *     the leaves turn and fall, and winter comes in grey and bare.
 * III "But thy eternal summer shall not fade": summer returns at once,
 *     golden, and "in eternal lines" faint golden script writes itself
 *     across the meadow in ribbons of light.
 * IV  "So long as men can breathe or eyes can see": a still, golden evening
 *     that does not end; the lines rise into the sky and stay.
 *
 * Columns: [unit, path, cloud, petals, wind, yaw, pitch, heat, blossom,
 *           autumn, winter, eternal, lines, evening, sun]
 * (sun: 0 morning, 1 noon, 2 afternoon, 3 evening)
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, softSprite, skyDome, terrain,
         scatter, particleField, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout (metres; you walk towards -z) ─────────────────────────────────
// The orchard runs from z = 14 to z = -14 in rows either side of an aisle at
// x = 0; beyond it the meadow dips into a shallow vale and the far hills
// rise in a patchwork of hedged fields.
function land(x, z) {
  return 0.25 * Math.sin(x * 0.09) * Math.cos(z * 0.07) + 0.12 * Math.sin(x * 0.27 + z * 0.19) -
         smooth(-12, -70, z) * 2.5 +
         smooth(-90, -330, z) * (34 + 14 * Math.sin(x * 0.011 + 0.6) + 7 * Math.sin(x * 0.031)) +
         smooth(70, 260, Math.abs(x)) * 22;
}
var curve = new THREE.CatmullRomCurve3([[0.2, 13], [-0.3, 6], [0.3, -1], [-0.2, -8], [0, -14], [0, -19]]
  .map(function (p) { return new THREE.Vector3(p[0], 0, p[1]); }));

// Where the sun stands through the day: [azimuth right of ahead, elevation].
var SUN = [[-0.55, 0.36], [-0.32, 0.72], [0.5, 0.4], [0.52, 0.05]];

function hash2(a, b) { var s = Math.sin(a * 127.1 + b * 311.7) * 43758.5453; return s - Math.floor(s); }

// ── Shader snippets ──────────────────────────────────────────────────────
var NOISE = 'float h31(vec3 p){ p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }\n' +
  'float vnoise(vec3 x){ vec3 i = floor(x), f = fract(x); f = f * f * (3.0 - 2.0 * f);\n' +
  ' return mix(mix(mix(h31(i), h31(i + vec3(1,0,0)), f.x), mix(h31(i + vec3(0,1,0)), h31(i + vec3(1,1,0)), f.x), f.y),\n' +
  '            mix(mix(h31(i + vec3(0,0,1)), h31(i + vec3(1,0,1)), f.x), mix(h31(i + vec3(0,1,1)), h31(i + vec3(1,1,1)), f.x), f.y), f.z); }\n';

// Grass, fields and hedges dry to hay, frost over and turn golden.
var SEASON = 'uniform float uDry; uniform float uFrost; uniform float uGold;\n' +
  'vec3 season(vec3 c){ float l = dot(c, vec3(0.299, 0.587, 0.114));\n' +
  ' c = mix(c, vec3(l * 1.5, l * 1.16, l * 0.5), uDry);\n' +
  ' c = mix(c, vec3(0.64, 0.68, 0.76) * (0.6 + l * 1.2), uFrost);\n' +
  ' return c * mix(vec3(1.0), vec3(1.12, 1.03, 0.72), uGold); }\n';

// ── Pieces ───────────────────────────────────────────────────────────────
// Push a (non-indexed) geometry in or out along its normals by a smooth
// function of position, so shared corners move together and stay closed.
function lumpy(geo, amp, seed) {
  var p = geo.attributes.position, v = new THREE.Vector3();
  for (var i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    var k = 1 + amp * (Math.sin(v.x * 3.1 + seed) * Math.cos(v.y * 2.7 - seed) + 0.5 * Math.sin(v.z * 4.3 + v.x * 1.3));
    p.setXYZ(i, v.x * k, v.y * k, v.z * k);
  }
  geo.computeVertexNormals();
  return geo;
}

// A tree: trunk, limbs and twigs (bark) and a crown of lumpy clumps (white,
// coloured by the season in the shader). `o` sets the shape: an apple tree
// is low and broad, an oak tall.
function treeGeometry(r, o) {
  var bark = [], crown = [], clumps = [], Y = new THREE.Vector3(0, 1, 0), q = new THREE.Quaternion();
  var top = new THREE.Vector3(0, o.trunk, 0);
  bark.push(tinted(new THREE.CylinderGeometry(o.girth * 0.65, o.girth, o.trunk + 0.1, 6).translate(0, o.trunk / 2, 0), '#6a5848'));
  function limb(from, dir, len, rad) {
    var g = new THREE.CylinderGeometry(rad * 0.55, rad, len, 5).translate(0, len / 2, 0);
    g.applyQuaternion(q.setFromUnitVectors(Y, dir));
    bark.push(tinted(g.translate(from.x, from.y, from.z), '#6a5848'));
    return from.clone().addScaledVector(dir, len);
  }
  for (var k = 0; k < o.limbs; k++) {
    var a = k / o.limbs * Math.PI * 2 + r() * 0.9, tilt = o.spread + r() * 0.3;
    var dir = new THREE.Vector3(Math.cos(a) * Math.sin(tilt), Math.cos(tilt), Math.sin(a) * Math.sin(tilt));
    var end = limb(top, dir, o.reach * (0.8 + r() * 0.4), o.girth * 0.55);
    for (var t = 0; t < 3; t++) {
      var tw = dir.clone().add(new THREE.Vector3(r() - 0.5, r() * 0.6, r() - 0.5)).normalize();
      limb(end, tw, o.reach * (0.45 + r() * 0.3), o.girth * 0.22);
    }
  }
  for (k = 0; k < o.clumps; k++) {
    var ca = r() * Math.PI * 2, cd = k === 0 ? 0 : o.width * (0.45 + r() * 0.5), rad = o.clump * (0.8 + r() * 0.45);
    var c = new THREE.Vector3(Math.cos(ca) * cd, o.crownY + r() * o.crownH - cd * 0.2, Math.sin(ca) * cd);
    clumps.push({ c: c, r: rad });
  }
  return { bark: merge(bark), crown: crownGeometry(r, clumps, o), clumps: clumps };
}

// A crown: a dark lumpy core per clump (so it reads as solid) wrapped in
// leaf cards, alpha-cut in the shader. `card` is a per-card random number
// (cores get -1) that picks blossom or leaf and the moment it falls.
function crownGeometry(r, clumps, o) {
  var pos = [], nor = [], uv = [], card = [], v = new THREE.Vector3(), n = new THREE.Vector3();
  var a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  clumps.forEach(function (cl, k) {
    if (o.core) {
      var core = lumpy(new THREE.IcosahedronGeometry(cl.r * o.core, 1), 0.12, k * 1.7).scale(1, 0.8, 1).translate(cl.c.x, cl.c.y, cl.c.z);
      var cp = core.attributes.position, cn = core.attributes.normal;
      for (var i = 0; i < cp.count; i++) {
        pos.push(cp.getX(i), cp.getY(i), cp.getZ(i));
        nor.push(cn.getX(i), cn.getY(i), cn.getZ(i));
        uv.push(0.5, 0.5);
        card.push(-1);
      }
      core.dispose();
    }
    for (var j = 0; j < o.cards; j++) {
      n.set(r() - 0.5, r() - 0.3, r() - 0.5).normalize();
      c.copy(n).multiplyScalar(cl.r * (0.7 + r() * 0.42));
      c.y *= 0.8;
      c.add(cl.c);
      // A card roughly facing outwards, at a random twist.
      v.set(r() - 0.5, r() - 0.5, r() - 0.5).multiplyScalar(1.4).add(n).normalize();
      a.set(0, 1, 0).cross(v);
      if (a.lengthSq() < 0.01) a.set(1, 0, 0);
      a.normalize().applyAxisAngle(v, r() * 6.28);
      b.copy(v).cross(a).normalize();
      var s = o.card * (0.75 + r() * 0.5) / 2, id = r();
      [[-1, -1], [1, -1], [1, 1], [-1, -1], [1, 1], [-1, 1]].forEach(function (q) {
        pos.push(c.x + (a.x * q[0] + b.x * q[1]) * s, c.y + (a.y * q[0] + b.y * q[1]) * s, c.z + (a.z * q[0] + b.z * q[1]) * s);
        nor.push(n.x * 0.8, n.y * 0.8 + 0.45, n.z * 0.8);
        uv.push((q[0] + 1) / 2, (q[1] + 1) / 2);
        card.push(id);
      });
    }
  });
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  geo.setAttribute('card', new THREE.Float32BufferAttribute(card, 1));
  return geo;
}

// The leaf card: red = a spray of leaves, green = apple blossom (five-petal
// flowers), blue = the flowers' hearts. Alpha covers both, for shadows.
function cardTexture(r) {
  var c = document.createElement('canvas');
  c.width = c.height = 128;
  var x = c.getContext('2d');
  x.globalCompositeOperation = 'lighter';
  x.fillStyle = 'rgb(255,0,0)';
  for (var i = 0; i < 9; i++) {
    var ang = i / 9 * Math.PI * 2 + r() * 0.4, d = 18 + r() * 26;
    x.save();
    x.translate(64 + Math.cos(ang) * d * 0.6, 64 + Math.sin(ang) * d * 0.6);
    x.rotate(ang + Math.PI / 2);
    x.beginPath();
    x.ellipse(0, 0, 8 + r() * 4, 20 + r() * 8, 0, 0, Math.PI * 2);
    x.fill();
    x.restore();
  }
  x.beginPath(); x.arc(64, 64, 14, 0, Math.PI * 2); x.fill();
  for (i = 0; i < 7; i++) {
    var fx = i ? 64 + (r() - 0.5) * 78 : 64, fy = i ? 64 + (r() - 0.5) * 78 : 64, fr = 9 + r() * 6;
    x.fillStyle = 'rgb(0,255,0)';
    for (var p = 0; p < 5; p++) {
      var pa = p / 5 * Math.PI * 2 + i;
      x.beginPath();
      x.arc(fx + Math.cos(pa) * fr * 0.62, fy + Math.sin(pa) * fr * 0.62, fr * 0.55, 0, Math.PI * 2);
      x.fill();
    }
    x.fillStyle = 'rgb(0,0,255)';
    x.beginPath(); x.arc(fx, fy, fr * 0.28, 0, Math.PI * 2); x.fill();
  }
  var t = new THREE.CanvasTexture(c);
  t.generateMipmaps = false;
  t.minFilter = THREE.LinearFilter;
  return t;
}

// A tuft of thin blades, dark at the root and light at the tip, lit as if
// facing up so the meadow shades evenly.
function tuftGeometry(r) {
  var pos = [], nor = [], col = [], root = new THREE.Color('#2c4a1a'), tip = new THREE.Color('#a9cc64');
  for (var i = 0; i < 7; i++) {
    var a = r() * 6.28, w = 0.012 + r() * 0.012, h = 0.28 + r() * 0.3, lean = 0.05 + r() * 0.14;
    var ox = Math.cos(a) * 0.05, oz = Math.sin(a) * 0.05, px = -Math.sin(a) * w, pz = Math.cos(a) * w;
    pos.push(ox - px, 0, oz - pz, ox + px, 0, oz + pz, ox + Math.cos(a) * lean, h, oz + Math.sin(a) * lean);
    nor.push(0, 1, 0, 0, 1, 0, 0, 1, 0);
    col.push(root.r, root.g, root.b, root.r, root.g, root.b, tip.r, tip.g, tip.b);
  }
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  return geo;
}

// Hand-written lines that aren't words: a pen looping along a baseline,
// with tall loops, descenders, small e-loops and humps, broken into words.
function scriptTexture(r) {
  var c = document.createElement('canvas');
  c.width = 2048;
  c.height = 128;
  var x = c.getContext('2d'), path = new Path2D(), px = 16, base = 76;
  while (px < 1990) {
    var letters = 2 + Math.floor(r() * 7);
    path.moveTo(px, base);
    for (var l = 0; l < letters && px < 2020; l++) {
      var k = r(), w = 14 + r() * 10, h, a;
      if (k < 0.2) { h = 56; a = w * 0.42; }            // tall loop, an l
      else if (k < 0.32) { h = -34; a = w * 0.4; }      // descender, a g
      else if (k < 0.6) { h = 24; a = w * 0.32; }       // small loop, an e
      else { h = 22 + r() * 6; a = -w * 0.1; }          // hump, an n
      for (var s = 1; s <= 18; s++) {
        var th = s / 18 * Math.PI * 2;
        path.lineTo(px + w * th / (Math.PI * 2) + a * Math.sin(th), base - h * (1 - Math.cos(th)) / 2);
      }
      px += w;
    }
    if (r() < 0.2) {                                    // a flourish under the word
      path.quadraticCurveTo(px + 20, base + 26, px - 60 - r() * 40, base + 16);
    }
    px += 22 + r() * 26;
  }
  x.lineCap = x.lineJoin = 'round';
  x.strokeStyle = 'rgba(255,255,255,0.25)';
  x.lineWidth = 14;
  x.stroke(path);
  x.strokeStyle = 'rgba(255,255,255,1)';
  x.lineWidth = 4.5;
  x.stroke(path);
  var t = new THREE.CanvasTexture(c);
  t.wrapS = THREE.RepeatWrapping;
  t.anisotropy = 4;
  return t;
}

// A ribbon along a curve, upright (its width runs up), with u along it.
function ribbonGeometry(points, width) {
  var cv = new THREE.CatmullRomCurve3(points), len = cv.getLength(), segs = Math.ceil(len * 4);
  var pos = [], uv = [], idx = [], p = new THREE.Vector3(), tg = new THREE.Vector3(), side = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0);
  for (var i = 0; i <= segs; i++) {
    var u = i / segs;
    cv.getPointAt(u, p);
    cv.getTangentAt(u, tg);
    side.copy(up).addScaledVector(tg, -tg.dot(up)).normalize().multiplyScalar(width / 2);
    pos.push(p.x - side.x, p.y - side.y, p.z - side.z, p.x + side.x, p.y + side.y, p.z + side.z);
    uv.push(u, 0, u, 1);
    if (i < segs) idx.push(i * 2, i * 2 + 2, i * 2 + 1, i * 2 + 1, i * 2 + 2, i * 2 + 3);
  }
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  geo.setIndex(idx);
  return { geo: geo, len: len };
}

// A cumulus: overlapping puffs, flat underneath, greyer at the base.
function cloudTexture(r) {
  var c = document.createElement('canvas');
  c.width = 256;
  c.height = 128;
  var x = c.getContext('2d');
  for (var i = 0; i < 46; i++) {
    var u = r(), px = 30 + u * 196, top = 30 + Math.pow(Math.abs(u - 0.5) * 2, 1.6) * 52;
    var py = top + r() * (96 - top), rad = 14 + r() * 26 * (1 - Math.abs(u - 0.5));
    var g = x.createRadialGradient(px, py, 0, px, py, rad), shade = Math.round(255 - (py / 128) * 50);
    g.addColorStop(0, 'rgba(' + shade + ',' + shade + ',' + (shade + 4) + ',0.5)');
    g.addColorStop(1, 'rgba(' + shade + ',' + shade + ',' + (shade + 4) + ',0)');
    x.fillStyle = g;
    x.fillRect(px - rad, py - rad, rad * 2, rad * 2);
  }
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// Sun rays: thin soft wedges round a centre.
function raysTexture(r) {
  var c = document.createElement('canvas');
  c.width = c.height = 256;
  var x = c.getContext('2d');
  x.translate(128, 128);
  x.globalCompositeOperation = 'lighter';
  for (var i = 0; i < 36; i++) {
    x.rotate(Math.PI * 2 / 36 + r() * 0.1);
    var len = 70 + r() * 58, g = x.createLinearGradient(0, 0, len, 0);
    g.addColorStop(0, 'rgba(255,250,235,0.22)');
    g.addColorStop(1, 'rgba(255,250,235,0)');
    x.fillStyle = g;
    x.beginPath();
    x.moveTo(0, 0);
    x.lineTo(len, -2 - r() * 3);
    x.lineTo(len, 2 + r() * 3);
    x.fill();
  }
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// A glow that falls off fast from a bright core, so the sun is not a disc.
function glowTexture() {
  var c = document.createElement('canvas');
  c.width = c.height = 256;
  var x = c.getContext('2d'), g = x.createRadialGradient(128, 128, 0, 128, 128, 128);
  [[0, 1], [0.06, 0.6], [0.15, 0.25], [0.35, 0.08], [0.7, 0.02], [1, 0]].forEach(function (s) {
    g.addColorStop(s[0], 'rgba(255,246,222,' + s[1] + ')');
  });
  x.fillStyle = g;
  x.fillRect(0, 0, 256, 256);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function shapeTexture(draw) {
  var c = document.createElement('canvas');
  c.width = c.height = 32;
  var x = c.getContext('2d');
  x.fillStyle = '#fff';
  draw(x);
  return new THREE.CanvasTexture(c);
}

// ── Synthesised sound cues ───────────────────────────────────────────────
// A rough gust: noise swelling through a sweeping band-pass.
function gust(ac, out) {
  var t = ac.currentTime, len = 3, b = ac.createBuffer(1, ac.sampleRate * len, ac.sampleRate), d = b.getChannelData(0);
  for (var i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  var src = ac.createBufferSource(), bp = ac.createBiquadFilter(), g = ac.createGain();
  src.buffer = b;
  bp.type = 'bandpass';
  bp.Q.value = 0.8;
  bp.frequency.setValueAtTime(300, t);
  bp.frequency.linearRampToValueAtTime(900, t + 1.2);
  bp.frequency.linearRampToValueAtTime(350, t + len);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.35, t + 1.0);
  g.gain.exponentialRampToValueAtTime(0.0001, t + len);
  src.connect(bp); bp.connect(g); g.connect(out);
  src.start(t);
}

// The eternal lines: a soft rising run of glassy notes.
function shimmer(ac, out) {
  var now = ac.currentTime;
  [523.25, 659.25, 783.99, 987.77, 1174.66, 1567.98].forEach(function (f, i) {
    var t = now + i * 0.16, o = ac.createOscillator(), o2 = ac.createOscillator(), g = ac.createGain();
    o.type = 'sine';
    o.frequency.value = f;
    o2.type = 'sine';
    o2.frequency.value = f * 2.01;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.07, t + 0.03);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 2.6);
    o.connect(g); o2.connect(g); g.connect(out);
    o.start(t); o2.start(t);
    o.stop(t + 2.7); o2.stop(t + 2.7);
  });
}

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(18);
  var gl = makeRenderer(canvas, { shadows: !small, clear: '#d8e4ee' });
  var world = new THREE.Scene();
  world.fog = new THREE.Fog('#dfe8ee', 70, 900);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.1, 3000);

  var U = { uClock: { value: 0 }, uWind: { value: 0.2 }, uDry: { value: 0 }, uFrost: { value: 0 }, uGold: { value: 0 },
            uBlossom: { value: 1 }, uAutumn: { value: 0 }, uBare: { value: 0 },
            uLeaf: { value: new THREE.Color('#5c8e36') }, uBloom: { value: new THREE.Color('#fff0f2') },
            uAutA: { value: new THREE.Color('#e09a2a') }, uAutB: { value: new THREE.Color('#a8401a') } };
  function seasonal(mat) {
    mat.onBeforeCompile = function (sh) {
      ['uDry', 'uFrost', 'uGold'].forEach(function (k) { sh.uniforms[k] = U[k]; });
      sh.fragmentShader = SEASON + sh.fragmentShader.replace('#include <color_fragment>',
        '#include <color_fragment>\n diffuseColor.rgb = season(diffuseColor.rgb);');
    };
    return mat;
  }

  // ── Sky ──
  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#3f78c0', mid: '#8db8e4', horizon: '#e6eef2', sun: '#fff4dc' }, 1500);
  sky.add(dome.mesh);
  var sunDisc = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,255,250,1)', 'rgba(255,250,230,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
  var sunGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture(),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
  var sunRays = new THREE.Sprite(new THREE.SpriteMaterial({ map: raysTexture(r), blending: THREE.AdditiveBlending,
    depthWrite: false, transparent: true, fog: false }));
  sky.add(sunGlow, sunRays, sunDisc);

  // Fair-weather clouds round the sky, and a bank that crosses the sun.
  var cloudTex = cloudTexture(r), clouds = [], cover = [];
  for (var i = 0; i < 16; i++) {
    var cl = new THREE.Sprite(new THREE.SpriteMaterial({ map: cloudTex, transparent: true, depthWrite: false, fog: false }));
    var ca = -1.6 + r() * 3.2, ce = 0.06 + r() * 0.22;
    cl.position.set(Math.sin(ca) * 1050, Math.sin(ce) * 1050 + 30, -Math.cos(ca) * 1050);
    cl.scale.set(260 + r() * 260, 110 + r() * 80, 1);
    cl.userData.a = ca;
    sky.add(cl);
    clouds.push(cl);
  }
  for (i = 0; i < 7; i++) {
    var cc = new THREE.Sprite(new THREE.SpriteMaterial({ map: cloudTex, transparent: true, depthWrite: false, fog: false, opacity: 0 }));
    cc.scale.set(300 + r() * 220, 150 + r() * 70, 1);
    cc.userData = { dx: (i - 3) * 120 + r() * 60, dy: (r() - 0.5) * 120 };
    sky.add(cc);
    cover.push(cc);
  }

  // ── Light ──
  var hemi = new THREE.HemisphereLight('#dbe8ff', '#5a6a3a', 1.2);
  var sun = new THREE.DirectionalLight('#fff3dc', 2.6);
  sun.castShadow = !small;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = sun.shadow.camera.bottom = -30;
  sun.shadow.camera.right = sun.shadow.camera.top = 30;
  sun.shadow.camera.far = 200;
  sun.shadow.bias = -0.0006;
  sun.shadow.normalBias = 0.05;
  world.add(hemi, sun, sun.target);

  // ── Land: the meadow, and hedged fields on the far hills ──
  var meadowCols = ['#4f7c2c', '#58852f', '#4b772a'].map(function (c) { return new THREE.Color(c); });
  var fieldCols = ['#5f8a3a', '#6e9641', '#7e9c48', '#9aa458', '#58803a', '#86a24c', '#a9a860'].map(function (c) { return new THREE.Color(c); });
  var tmp = new THREE.Color();
  function fieldUV(x, z) { return [(x * 0.85 + z * 0.35) / 44, (z * 0.9 - x * 0.3) / 34]; }
  var ground = terrain(1800, small ? 160 : 240, 0, -420, land, seasonal(new THREE.MeshLambertMaterial({ vertexColors: true })),
    function (x, z) {
      var f = fieldUV(x, z), field = fieldCols[Math.floor(hash2(Math.floor(f[0]), Math.floor(f[1])) * fieldCols.length)];
      var near = meadowCols[Math.floor(hash2(Math.floor(x / 9), Math.floor(z / 9)) * 3)];
      return tmp.copy(near).lerp(field, smooth(-50, -110, z) + smooth(40, 90, Math.abs(x)) * (1 - smooth(-50, -110, z)));
    });
  world.add(ground);

  // Hedgerows along the field edges, and hedgerow oaks.
  // Walk every field edge, setting a bush every couple of metres so the
  // hedges read as continuous dark lines over the hills.
  var hedgeAt = [], oakSpots = [], up = new THREE.Vector3(0, 1, 0), step = small ? 2.6 : 1.7;
  function fieldXZ(u, v) { return [(0.9 * 44 * u - 0.35 * 34 * v) / 0.87, (0.3 * 44 * u + 0.85 * 34 * v) / 0.87]; }
  for (var line = -20; line <= 10; line++) {
    for (var along = -700; along <= 450; along += step) {
      [fieldXZ(line, along / 34), fieldXZ(along / 44, line)].forEach(function (h) {
        var x = h[0] + (r() - 0.5) * 0.6, z = h[1] + (r() - 0.5) * 0.6;
        if (z > -95 || z < -560 || Math.abs(x) > 460) return;
        if (r() < 0.012) oakSpots.push([x, z]);
        hedgeAt.push(x, z);
      });
    }
  }
  var bush = lumpy(new THREE.IcosahedronGeometry(1, 0), 0.15, 2).scale(1.5, 1.1, 1.5);
  var hedges = new THREE.InstancedMesh(bush, seasonal(new THREE.MeshLambertMaterial({ flatShading: true })), hedgeAt.length / 2);
  scatter(hedges, hedgeAt.length / 2, function (n, p, q, s, c) {
    var x = hedgeAt[n * 2], z = hedgeAt[n * 2 + 1];
    p.set(x, land(x, z) + 0.4, z);
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar(0.9 + r() * 0.6);
    c.setHSL(0.27 + r() * 0.04, 0.38, 0.13 + r() * 0.06);
  });
  hedges.receiveShadow = true;
  world.add(hedges);

  // A church tower on the far hill.
  var stone = seasonal(new THREE.MeshLambertMaterial({ color: '#9a9284' })), church = new THREE.Group();
  var tower = new THREE.Mesh(new THREE.BoxGeometry(5, 16, 5).translate(0, 8, 0), stone);
  var nave = new THREE.Mesh(new THREE.BoxGeometry(7, 7, 18).translate(0, 3.5, 11), stone);
  var roof = new THREE.Mesh(new THREE.CylinderGeometry(0.01, 5, 18, 3, 1).rotateX(Math.PI / 2).rotateZ(Math.PI).translate(0, 8.4, 11),
    new THREE.MeshLambertMaterial({ color: '#5a4e48' }));
  church.add(tower, nave, roof);
  for (i = 0; i < 4; i++) {
    var pin = new THREE.Mesh(new THREE.BoxGeometry(0.8, 1.6, 0.8), stone);
    pin.position.set(i % 2 ? 2.1 : -2.1, 16.8, i < 2 ? 2.1 : -2.1);
    church.add(pin);
  }
  church.position.set(-95, land(-95, -260) - 0.5, -260);
  church.rotation.y = 0.4;
  world.add(church);

  // ── Trees: the orchard rows, and oaks on the hills ──
  var barkMat = seasonal(new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true }));
  var cardTex = cardTexture(rng(12));
  var crownMat = new THREE.MeshLambertMaterial({ map: cardTex, side: THREE.DoubleSide });
  crownMat.onBeforeCompile = function (sh) {
    Object.keys(U).forEach(function (k) { sh.uniforms[k] = U[k]; });
    sh.vertexShader = 'uniform float uClock; uniform float uWind; attribute float card; varying vec3 vWP; varying float vSeed; varying float vCard;\n' +
      sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n float tph = instanceMatrix[3][0] * 0.31 + instanceMatrix[3][2] * 0.17;\n' +
      ' float sway = (sin(uClock * (1.2 + uWind * 3.0) + tph) + 0.45 * sin(uClock * (3.1 + uWind * 4.0) + tph * 2.0 + card * 6.0)) * (0.015 + uWind * 0.1) * max(position.y - 1.0, 0.0);\n' +
      ' transformed.x += sway; transformed.z += sway * 0.6;\n' +
      ' vWP = (modelMatrix * instanceMatrix * vec4(transformed, 1.0)).xyz;\n' +
      ' vSeed = fract(sin(tph * 12.9898) * 43758.5453); vCard = card;');
    sh.fragmentShader = 'uniform float uBlossom; uniform float uAutumn; uniform float uBare; uniform float uGold;\n' +
      'uniform vec3 uLeaf; uniform vec3 uBloom; uniform vec3 uAutA; uniform vec3 uAutB; varying vec3 vWP; varying float vSeed; varying float vCard;\n' + NOISE +
      sh.fragmentShader.replace('#include <map_fragment>',
        ' vec4 cardT = texture2D(map, vMapUv);\n' +
        ' float isCard = step(0.0, vCard), bloomCard = uBlossom * step(0.22, fract(vCard * 7.13)) * isCard;\n' +
        ' if (isCard > 0.5 && mix(cardT.r, cardT.g, bloomCard) < 0.5) discard;\n' +
        ' float n = vnoise(vWP * 1.6) * 0.7 + vnoise(vWP * 4.7) * 0.3, k = mix(n, fract(vCard * 3.1), isCard);\n' +
        // Leaves fall card by card; the cores thin out first.
        ' if (mix(n * 0.6 + 0.04, fract(vCard * 13.7), isCard) < uBare * 1.05) discard;\n' +
        ' vec3 c = mix(uLeaf * (0.7 + 0.5 * k), mix(uAutA, uAutB, fract(vSeed * 3.7 + k * 0.9)), uAutumn);\n' +
        ' c = mix(c, mix(uBloom, vec3(1.0, 0.85, 0.45), step(0.5, cardT.b)), bloomCard);\n' +
        ' c *= mix(0.62, 1.0, isCard) * mix(vec3(1.0), vec3(1.16, 1.02, 0.66), uGold);\n' +
        ' diffuseColor.rgb *= c;')
      .replace('#include <normal_fragment_begin>', '#include <normal_fragment_begin>\n#ifdef DOUBLE_SIDED\n normal *= faceDirection;\n#endif');
  };
  var crownDepth = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: cardTex, alphaTest: 0.5, side: THREE.DoubleSide });

  var orchard = [], m4 = new THREE.Matrix4(), v3 = new THREE.Vector3();
  var appleKinds = [0, 1].map(function (k) {
    return treeGeometry(rng(30 + k), { trunk: 1.3, girth: 0.17, limbs: 4, spread: 0.75, reach: 1.2, clumps: 7, width: 1.6,
                                       clump: 0.9, crownY: 2.4, crownH: 0.8, core: 0.62, cards: small ? 45 : 75, card: 0.6 });
  });
  [-16, -10.4, -4.6, 4.6, 10.4, 16].forEach(function (x) {
    for (var z = 12; z > -16; z -= 6.5) orchard.push([x + (r() - 0.5) * 0.8, z + (r() - 0.5) * 0.8]);
  });
  var crowns = [];
  appleKinds.forEach(function (kind, k) {
    var mine = orchard.filter(function (t, n) { return n % 2 === k; });
    var trunks = new THREE.InstancedMesh(kind.bark, barkMat, mine.length), crown = new THREE.InstancedMesh(kind.crown, crownMat, mine.length);
    mine.forEach(function (t, n) {
      var sc = 0.9 + r() * 0.3;
      m4.compose(v3.set(t[0], land(t[0], t[1]) - 0.05, t[1]), new THREE.Quaternion().setFromAxisAngle(up, r() * 6.28), new THREE.Vector3(sc, sc, sc));
      trunks.setMatrixAt(n, m4);
      crown.setMatrixAt(n, m4);
      crown.setColorAt(n, tmp.setScalar(0.9 + r() * 0.15));
    });
    crown.customDepthMaterial = crownDepth;
    trunks.castShadow = crown.castShadow = true;
    trunks.receiveShadow = crown.receiveShadow = true;
    world.add(trunks, crown);
    crowns.push(crown);
  });

  var oakKind = treeGeometry(rng(44), { trunk: 2.6, girth: 0.35, limbs: 5, spread: 0.55, reach: 2.2, clumps: 8, width: 3.0,
                                        clump: 1.7, crownY: 4.8, crownH: 2.2, core: 0.8, cards: small ? 16 : 30, card: 1.5 });
  // A few oaks at the meadow's edge, the rest in the hedgerows.
  [[-24, -30], [27, -44], [-38, -58], [19, -78], [44, -26]].forEach(function (p) { oakSpots.unshift(p); });
  var oaks = oakSpots.slice(0, small ? 90 : 200);
  var oakTrunks = new THREE.InstancedMesh(oakKind.bark, barkMat, oaks.length), oakCrowns = new THREE.InstancedMesh(oakKind.crown, crownMat, oaks.length);
  oaks.forEach(function (p, n) {
    var sc = 0.9 + r() * 0.6;
    m4.compose(v3.set(p[0], land(p[0], p[1]) - 0.2, p[1]), new THREE.Quaternion().setFromAxisAngle(up, r() * 6.28), new THREE.Vector3(sc, sc * (0.9 + r() * 0.25), sc));
    oakTrunks.setMatrixAt(n, m4);
    oakCrowns.setMatrixAt(n, m4);
    oakCrowns.setColorAt(n, tmp.setScalar(0.75 + r() * 0.2));
  });
  oakCrowns.customDepthMaterial = crownDepth;
  oakTrunks.castShadow = oakCrowns.castShadow = true;
  world.add(oakTrunks, oakCrowns);
  crowns.push(oakCrowns);

  // ── The meadow grass and its flowers ──
  var grassMat = seasonal(new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }));
  var grassCompile = grassMat.onBeforeCompile;
  grassMat.onBeforeCompile = function (sh) {
    grassCompile(sh);
    sh.uniforms.uClock = U.uClock;
    sh.uniforms.uWind = U.uWind;
    sh.vertexShader = 'uniform float uClock; uniform float uWind;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n float gph = instanceMatrix[3][0] * 0.4 + instanceMatrix[3][2] * 0.3;\n' +
      ' float gb = (sin(uClock * (1.4 + uWind * 3.0) + gph) * 0.6 + 0.4 + uWind * 0.8) * (0.06 + uWind * 0.3) * position.y * position.y;\n' +
      ' transformed.x += gb; transformed.z += gb * 0.3;');
  };
  var grass = new THREE.InstancedMesh(tuftGeometry(rng(3)), grassMat, small ? 10000 : 30000);
  scatter(grass, 120000, function (n, p, q, s, c) {
    var x = (r() - 0.5) * (r() < 0.6 ? 18 : 54), z = 16 - r() * 66;
    if (Math.abs(x) < 0.5 + (z + 50) * 0.002 && r() < 0.6) return false;      // a trodden line down the aisle
    p.set(x, land(x, z), z);
    q.setFromAxisAngle(up, r() * 6.28);
    s.set(1.2, 0.8 + r() * 0.8, 1.2);
    c.setHSL(0.2 + r() * 0.08, 0.3, 0.75 + r() * 0.25);
  });
  grass.receiveShadow = true;
  world.add(grass);

  var flowerPos = [], flowerCol = [], fc = ['#ffd21f', '#ffd21f', '#fbfbf2', '#fbfbf2', '#e86a9a', '#f2efe0'].map(function (c) { return new THREE.Color(c); });
  for (i = 0; i < (small ? 2500 : 6000); i++) {
    var fx = (r() - 0.5) * 56, fz = 16 - r() * 70;
    flowerPos.push(fx, land(fx, fz) + 0.3 + r() * 0.35, fz);
    var col = fc[Math.floor(r() * fc.length)];
    flowerCol.push(col.r, col.g, col.b);
  }
  var flowerGeo = new THREE.BufferGeometry();
  flowerGeo.setAttribute('position', new THREE.Float32BufferAttribute(flowerPos, 3));
  flowerGeo.setAttribute('color', new THREE.Float32BufferAttribute(flowerCol, 3));
  var flowers = new THREE.Points(flowerGeo, new THREE.PointsMaterial({ size: 0.08, vertexColors: true, transparent: true, depthWrite: false,
    map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)') }));
  world.add(flowers);

  // ── The eternal lines: ribbons of golden script ──
  var script = scriptTexture(rng(9)), lines = [];
  [
    { pts: [[-6, 1.2, -27], [0, 1.7, -28.5], [6, 1.2, -27.5], [12, 1.8, -29], [18, 1.3, -28]], w: 0.9, at: 0.0 },
    { pts: [[2, 2.4, -35], [10, 3.2, -32.5], [19, 2.6, -35.5], [28, 3.3, -33]], w: 1.2, at: 0.15 },
    { pts: [[5, 1.8, -30], [11, 3.6, -33], [15, 6.5, -36], [16, 10, -40], [13, 13.5, -45]], w: 1.1, at: 0.3 },
    { pts: [[-4, 1.5, -29], [-7, 4, -32], [-5, 7, -35], [0, 10, -39], [5, 12.5, -44]], w: 1.0, at: 0.45 }
  ].forEach(function (L) {
    var rb = ribbonGeometry(L.pts.map(function (p) { return new THREE.Vector3(p[0], land(p[0], p[2]) + p[1], p[2]); }), L.w);
    var mat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
      uniforms: { tScript: { value: script }, uGrow: { value: 0 }, uAlpha: { value: 0 }, uRepeat: { value: rb.len / (L.w * 16) }, uOff: { value: r() },
                  uTime: { value: 0 }, uColor: { value: new THREE.Color('#f0b850') } },
      vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
      fragmentShader: 'uniform sampler2D tScript; uniform float uGrow; uniform float uAlpha; uniform float uRepeat; uniform float uTime; uniform float uOff; uniform vec3 uColor; varying vec2 vUv;\n' +
        'void main(){ float head = uGrow * 1.12 - 0.06;\n' +
        ' float shown = 1.0 - smoothstep(head - 0.025, head, vUv.x);\n' +
        ' float ends = smoothstep(0.0, 0.08, vUv.x) * smoothstep(1.0, 0.9, vUv.x);\n' +
        ' float ink = texture2D(tScript, vec2(vUv.x * uRepeat + uOff, vUv.y)).a;\n' +
        ' float tip = exp(-pow((vUv.x - head) * 70.0, 2.0)) * exp(-pow((vUv.y - 0.45) * 4.0, 2.0)) * step(0.001, uGrow) * step(uGrow, 0.999);\n' +
        ' float a = (ink * shown * (0.7 + 0.3 * sin(vUv.x * 90.0 - uTime * 1.7)) + tip * 1.6) * ends * uAlpha;\n' +
        ' gl_FragColor = vec4(uColor * a, 1.0); }'
    });
    var mesh = new THREE.Mesh(rb.geo, mat);
    mesh.frustumCulled = false;
    mesh.userData = { at: L.at, y: 0 };
    world.add(mesh);
    lines.push(mesh);
  });

  // ── Things in the air ──
  var petalTex = shapeTexture(function (x) { x.beginPath(); x.ellipse(16, 16, 13, 7, 0.6, 0, Math.PI * 2); x.fill(); });
  var petals = particleField({ count: small ? 500 : 1200, box: [34, 12, 34], fall: [0.3, 0.8], size: 0.15, map: petalTex, alphaTest: 0.4,
                               colors: ['#fff6f8', '#ffe2ea', '#fbd0dc', '#ffffff'], sway: 0.9, windSpeed: 7 });
  var leafTex = shapeTexture(function (x) { x.beginPath(); x.moveTo(16, 2); x.quadraticCurveTo(30, 16, 16, 30); x.quadraticCurveTo(2, 16, 16, 2); x.fill(); });
  var leaves = particleField({ count: small ? 300 : 700, box: [34, 12, 34], fall: [0.6, 1.3], size: 0.16, map: leafTex, alphaTest: 0.4,
                               colors: ['#d98a26', '#b8501c', '#e6b13a', '#8a3a18'], sway: 1.1, windSpeed: 4 });
  var snow = particleField({ count: small ? 600 : 1500, box: [34, 14, 34], fall: [0.6, 1.1], size: 0.07,
                             map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)'), sway: 0.4, windSpeed: 2 });
  var motes = particleField({ count: small ? 250 : 600, box: [30, 8, 30], fall: [-0.06, 0.05], size: 0.05, color: '#ffe0a0',
                              map: softSprite('rgba(255,236,190,1)', 'rgba(255,220,150,0)'), sway: 0.25, windSpeed: 0.4 });
  motes.points.material.blending = THREE.AdditiveBlending;
  world.add(petals.points, leaves.points, snow.points, motes.points);
  var pf = { snow: 0, wind: 0, dt: 0, time: 0 };

  // ── Heat shimmer: a post pass that only runs while the sun blazes ──
  var rt = new THREE.WebGLRenderTarget(4, 4, { type: THREE.HalfFloatType, samples: small ? 0 : 4 });
  var postScene = new THREE.Scene(), postCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  var postMat = new THREE.ShaderMaterial({
    depthTest: false, depthWrite: false,
    uniforms: { tScene: { value: rt.texture }, uHeat: { value: 0 }, uTime: { value: 0 } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
    fragmentShader: 'uniform sampler2D tScene; uniform float uHeat; uniform float uTime; varying vec2 vUv;\n' +
      'void main(){ float m = uHeat * (0.35 + 0.65 * smoothstep(0.75, 0.1, vUv.y));\n' +
      ' float w = sin(vUv.y * 140.0 + uTime * 7.0 + sin(vUv.x * 9.0 + uTime) * 2.0) * 0.6 + sin(vUv.y * 61.0 - uTime * 4.6 + vUv.x * 23.0) * 0.4;\n' +
      ' vec4 c = texture2D(tScene, vUv + vec2(w * 0.0022, w * 0.0012) * m);\n' +
      ' float l = dot(c.rgb, vec3(0.299, 0.587, 0.114));\n' +
      ' c.rgb = mix(c.rgb, vec3(l) * vec3(1.05, 1.03, 0.98) + 0.03, uHeat * 0.08);\n' +
      ' gl_FragColor = c;\n #include <tonemapping_fragment>\n #include <colorspace_fragment>\n }'
  });
  postScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), postMat));
  var bufSize = new THREE.Vector2();

  // ── Frame ──
  var sunDir = new THREE.Vector3(), right = new THREE.Vector3(), fwd = new THREE.Vector3(), look = new THREE.Vector3();
  var horizon = new THREE.Color(), portraitTilt = 0;
  function sunAt(t, out) {
    var k = clamp(Math.floor(t), 0, SUN.length - 2), u = clamp(t - k, 0, 1);
    var az = lerp(SUN[k][0], SUN[k + 1][0], u), el = lerp(SUN[k][1], SUN[k + 1][1], u);
    return out.set(Math.sin(az) * Math.cos(el), Math.sin(el), -Math.cos(az) * Math.cos(el));
  }

  function frame(f) {
    var row = f.row, time = f.time, dt = f.dt;
    var cloud = row[1], wind = row[3], heat = row[6], bloom = row[7], autumn = row[8], winter = row[9];
    var eternal = row[10], lineAmt = row[11], evening = row[12];

    // Walk down the aisle.
    curve.getPointAt(clamp(f.cam, 0, 1), camera.position);
    camera.position.y = land(camera.position.x, camera.position.z) + 1.65 + Math.sin(time * 0.9) * 0.02;
    look.set(camera.position.x, camera.position.y - 0.1, camera.position.z - 10);
    camera.lookAt(look);
    camera.rotateY(row[4] - f.mx * 0.14);
    camera.rotateX(row[5] + portraitTilt - f.my * 0.07);
    sky.position.copy(camera.position);

    // The sun through the day, blazing, clouded, and lost in winter.
    sunAt(row[13], sunDir);
    var veil = 1 - cloud * 0.85, lost = 1 - winter * 0.92;
    dome.uniforms.sunDir.value.copy(sunDir);
    dome.uniforms.sunColor.value.set('#fff4dc').lerp(tmp.set('#ffb062'), evening).multiplyScalar((0.18 + heat * 0.22 + evening * 0.3) * veil * lost);
    sunDisc.position.copy(sunDir).multiplyScalar(1000);
    sunGlow.position.copy(sunDisc.position);
    sunRays.position.copy(sunDisc.position);
    sunDisc.scale.setScalar(26 + heat * 22 + evening * 16);
    sunGlow.scale.setScalar(420 + heat * 380 + evening * 380);
    sunRays.scale.setScalar(380 + heat * 600);
    sunRays.material.rotation = time * 0.01;
    sunDisc.material.opacity = veil * lost;
    sunGlow.material.opacity = (0.8 + heat * 0.2) * veil * lost;
    sunRays.material.opacity = (0.16 + heat * 0.34) * veil * lost * (1 - evening * 0.6);
    sunDisc.material.color.set('#ffffff').lerp(tmp.set('#ffc070'), evening);
    sunGlow.material.color.set('#ffffff').lerp(tmp.set('#ff9a4a'), evening);

    // Sky: May morning, bleached at noon, dimmed, hazy autumn, grey winter,
    // eternal gold and the evening that stays.
    dome.uniforms.top.value.set('#3f78c0').lerp(tmp.set('#4f88c8'), heat).lerp(tmp.set('#5d7c9c'), cloud * 0.7)
      .lerp(tmp.set('#4c6a92'), autumn).lerp(tmp.set('#5c6977'), winter).lerp(tmp.set('#3b6db6'), eternal).lerp(tmp.set('#2d3f7a'), evening);
    dome.uniforms.mid.value.set('#8db8e4').lerp(tmp.set('#a4c8ea'), heat).lerp(tmp.set('#98abc0'), cloud * 0.7)
      .lerp(tmp.set('#a7adb8'), autumn).lerp(tmp.set('#8f99a6'), winter).lerp(tmp.set('#a6c6e2'), eternal).lerp(tmp.set('#d08e74'), evening);
    horizon.set('#e6eef2').lerp(tmp.set('#eef3f4'), heat).lerp(tmp.set('#c9d2d8'), cloud * 0.7)
      .lerp(tmp.set('#e4ccaa'), autumn).lerp(tmp.set('#bcc3ca'), winter).lerp(tmp.set('#f4e4bc'), eternal).lerp(tmp.set('#ffc07a'), evening);
    dome.uniforms.horizon.value.copy(horizon);
    world.fog.color.copy(horizon);
    world.fog.near = 70 - winter * 40;
    world.fog.far = 900 - winter * 450 - heat * 200;
    gl.setClearColor(horizon);
    gl.toneMappingExposure = 1 + heat * 0.3 - winter * 0.05;

    clouds.forEach(function (c) {
      c.userData.a += dt * 0.0012;
      if (c.userData.a > 1.7) c.userData.a -= 3.4;
      c.position.x = Math.sin(c.userData.a) * 1050;
      c.position.z = -Math.cos(c.userData.a) * 1050;
      c.material.opacity = 0.75 - heat * 0.5 + winter * 0.25;
      c.material.color.set('#ffffff').lerp(tmp.set('#8f98a3'), winter).lerp(tmp.set('#ffc49a'), evening);
    });
    // The bank that dims the sun slides in from the right across it.
    right.set(-sunDir.z, 0, sunDir.x).normalize();
    cover.forEach(function (c, k) {
      var u = c.userData;
      c.position.copy(sunDir).multiplyScalar(980).addScaledVector(right, u.dx + (1 - cloud) * 900 - winter * 300);
      c.position.y += u.dy;
      c.material.opacity = Math.max(cloud, winter * 0.8) * 0.95;
      c.material.color.set('#f2f4f8').lerp(tmp.set('#9aa3ae'), Math.max(winter, cloud * 0.35));
    });

    sun.position.copy(camera.position).addScaledVector(sunDir, 90);
    fwd.set(0, 0, -1).applyQuaternion(camera.quaternion).setY(0).normalize();
    sun.target.position.copy(camera.position).addScaledVector(fwd, 14);
    sun.intensity = (2.6 + heat * 1.2) * veil * (1 - winter * 0.8) * (1 - evening * 0.3);
    sun.color.set('#fff3dc').lerp(tmp.set('#ffb36a'), evening);
    hemi.intensity = 1.15 + heat * 0.3 - winter * 0.15 - evening * 0.35;
    hemi.color.set('#dbe8ff').lerp(tmp.set('#b8c2cc'), winter).lerp(tmp.set('#ffd2a0'), evening);
    hemi.groundColor.set('#5a6a3a').lerp(tmp.set('#6a6a62'), winter).lerp(tmp.set('#6a4a2a'), evening);

    // The year: blossom, high summer, autumn, the bare winter, eternal gold.
    var dry = clamp(heat * 0.25 + autumn * 0.75 + winter * 0.5, 0, 1) * (1 - eternal);
    U.uClock.value = time;
    U.uWind.value = wind;
    U.uDry.value = dry;
    U.uFrost.value = winter * (1 - eternal) * 0.8;
    U.uGold.value = Math.max(eternal * 0.6, evening * 0.9);
    U.uBlossom.value = bloom;
    U.uAutumn.value = autumn * (1 - eternal);
    U.uBare.value = smooth(0.3, 1, winter) * (1 - eternal);
    crowns.forEach(function (c) { c.castShadow = U.uBare.value < 0.3; });
    flowers.material.opacity = clamp(1 - autumn * 0.8 - winter, 0, 1) * (1 - eternal) + eternal;

    // Golden script writes itself; at evening the lines lift into the sky.
    lines.forEach(function (m, k) {
      var u = m.userData, uni = m.material.uniforms, g = smooth(u.at, u.at + 0.5, lineAmt);
      uni.uGrow.value = g;
      uni.uAlpha.value = smooth(0, 0.08, g) * (0.95 - evening * 0.5);
      uni.uTime.value = time;
      m.position.y = evening * (3 + k * 1.2) + Math.sin(time * 0.3 + k) * 0.15;
    });

    pf.dt = dt; pf.time = time;
    pf.snow = row[2]; pf.wind = wind;
    petals.update(pf, camera.position, env.reduceMotion);
    pf.snow = autumn * (1 - winter) * 0.9; pf.wind = wind * 0.7;
    leaves.update(pf, camera.position, env.reduceMotion);
    pf.snow = smooth(0.4, 1, winter) * (1 - eternal) * 0.8; pf.wind = 0.15;
    snow.update(pf, camera.position, env.reduceMotion);
    pf.snow = eternal * 0.5 + evening * 0.5; pf.wind = 0.05;
    motes.update(pf, camera.position, env.reduceMotion);

    if (heat < 0.01) {
      gl.setRenderTarget(null);
      gl.render(world, camera);
      return;
    }
    postMat.uniforms.uHeat.value = heat * (env.reduceMotion ? 0.4 : 1);
    postMat.uniforms.uTime.value = time;
    gl.getDrawingBufferSize(bufSize);
    if (rt.width !== bufSize.x || rt.height !== bufSize.y) rt.setSize(bufSize.x, bufSize.y);
    gl.setRenderTarget(rt);
    gl.render(world, camera);
    gl.setRenderTarget(null);
    gl.render(postScene, postCam);
  }

  return {
    resize: function (w, h, dpr) {
      fitCamera(gl, camera, w, h, dpr, small);
      portraitTilt = w < h ? 0.1 : 0;          // less foreground grass on a phone
    },
    frame: frame,
    destroy: function () { rt.dispose(); disposeAll(postScene); disposeAll(world, gl); }
  };
}

PI.register('summer-meadow', {
  renderer: renderer3d,
  align: ['right', 'right', 'left', 'center'],
  scrim: 0.6,
  keys: function (T) {
    var n = T.count;
    function at(i, frac) { i = Math.min(i, n - 1); return lerp(T.start(i), T.end(i), frac); }
    //  unit          path  cloud petal wind  yaw    pitch heat blos  aut  wint etern lines eve  sun
    return [
      [0,             0.00, 0.0, 0.10, 0.12, 0.00, 0.04, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.7,           0.03, 0.0, 0.12, 0.15, 0.00, 0.04, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [at(0, 0.25),   0.10, 0.0, 0.30, 0.40, 0.04, 0.06, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],  // "shall I compare thee"
      [at(0, 0.55),   0.20, 0.0, 1.00, 1.00, 0.10, 0.08, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],  // "rough winds do shake the darling buds"
      [at(0, 0.85),   0.28, 0.0, 0.60, 0.55, 0.06, 0.08, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], // "summer's lease ... all too short a date"
      [at(1, 0.1),    0.34, 0.0, 0.10, 0.20, 0.00, 0.30, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95],
      [at(1, 0.3),    0.38, 0.0, 0.00, 0.12, -0.04, 0.42, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], // "too hot the eye of heaven shines"
      [at(1, 0.47),   0.42, 1.0, 0.00, 0.20, -0.04, 0.40, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],// "his gold complexion dimm'd"
      [at(1, 0.62),   0.47, 0.7, 0.00, 0.35, 0.00, 0.10, 0.0, 0.0, 0.55, 0.0, 0.0, 0.0, 0.0, 1.1], // "every fair from fair sometime declines"
      [at(1, 0.8),    0.52, 0.6, 0.00, 0.55, 0.00, 0.06, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.0, 1.2],
      [at(1, 1.0),    0.56, 0.8, 0.00, 0.35, 0.00, 0.06, 0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 1.3], // winter, "untrimm'd"
      [at(2, 0.18),   0.60, 0.8, 0.00, 0.30, 0.00, 0.06, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.4],
      [at(2, 0.38),   0.66, 0.0, 0.00, 0.15, 0.00, 0.06, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0], // "but thy eternal summer shall not fade"
      [at(2, 0.6),    0.74, 0.0, 0.00, 0.15, 0.02, 0.08, 0.0, 0.0, 0.0, 0.0, 1.0, 0.35, 0.0, 2.0],
      [at(2, 0.95),   0.84, 0.0, 0.00, 0.12, 0.02, 0.10, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.1], // "when in eternal lines to time thou grow'st"
      [at(3, 0.25),   0.90, 0.0, 0.00, 0.10, 0.00, 0.12, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 2.6],
      [at(3, 0.65),   0.95, 0.0, 0.00, 0.08, 0.00, 0.14, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 3.0], // "so long as men can breathe or eyes can see"
      [T.total,       0.98, 0.0, 0.00, 0.06, 0.00, 0.16, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 3.0]
    ];
  },
  sound: {
    src: '/audio/birds.mp3',
    label: 'Play the orchard birdsong',
    // Birdsong through the year; quiet in winter, softer at evening.
    volume: function (row) { return (0.12 + row[3] * 0.05) * (1 - row[9] * 0.85) * (1 - row[12] * 0.35); },
    cues: [
      { stanza: 0, at: 0.6, play: gust },
      { stanza: 2, at: 1.05, play: shimmer }
    ]
  }
});
