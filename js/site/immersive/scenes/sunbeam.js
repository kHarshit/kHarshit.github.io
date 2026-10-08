/*
 * Scene for "You Are" (u/Poem_for_your_sprog): a bright summer morning in a
 * little orchard garden, which turns out to be smiling.
 *
 * I   A round lawn among apple trees. "My comfort and my pleasure": a
 *     garden bench with a gingham blanket, a steaming cup and two balloons
 *     tied to its arm; "the apple of my eye": one red apple glows on a low
 *     branch; "precious as a treasure": a little sweet tin glints in the
 *     grass; "the sunbeam in the sky": a shaft of sun breaks through the
 *     leaves, full of dust motes.
 * II  "The essence of emotion": the flowers round the lawn open; "a
 *     wholesome honeybee": a bee buzzes from flower to flower; "the spirit
 *     of devotion": the sunflowers wake and turn their faces to the sun;
 *     "wonderful to me": petals and light swirl round you.
 * III "It's you I'll hold above me": the red balloon slips free and rises,
 *     and you rise with it into the blue, the garden dropping away.
 * IV  "But I think I like you more": high in the morning sky the sun
 *     swells, and the yellow balloon comes up after the red one, nudges it
 *     and settles a little higher.
 * V   ":)": you look down, and the garden is a face: the round lawn ringed
 *     with flowers, two apple trees for eyes and the row of sunflowers for
 *     a mouth, every head turned up to you. It lies on its side at first,
 *     like the ":)" on the page, and turns upright in the outro.
 *
 * Columns: [unit, camX, camY, camZ, wind, yaw, pitch, sunEl, beam, apple,
 *           glint, open, bee, devote, swirl, liftR, liftY, swell, smile,
 *           phYaw, phPitch]
 * (the keys aim the camera at a point and work out yaw/pitch from it;
 * phYaw/phPitch turn the view on a phone, where the text is centred)
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, softSprite, skyDome, terrain,
         scatter, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout (metres; north is -z) ─────────────────────────────────────────
// The face: a round lawn (R 10.2) in a gravel ring, flowers just inside it,
// eye trees at (±4, -4) and a mouth of sunflowers on an arc round (0, -1).
// The bench sits between the eyes, turned a little to the south-west; the
// morning sun is east-north-east, so the sky beat looks east and tips
// straight down into the sideways ":)".
var LAWN = 10.2, PATH = 10.95, BAND = [8.95, 10.1];
var EYES = [[-4, -4], [4, -4]];
var MOUTH = { cx: 0, cz: -1, r: 7, a0: 0.44, a1: Math.PI - 0.44 };
var BENCH = { x: 0, z: -3.6, turn: -0.52 };
var TIN = [1.25, -1.95];
var HERO = [2.55, 2.25, -3.05];        // the apple of my eye, under the right eye's crown
var SUN_AZ = 1.35;                     // radians east of north

function land(x, z) {
  var r = Math.hypot(x, z);
  return smooth(45, 90, r) * (0.6 * Math.sin(x * 0.05) * Math.cos(z * 0.045) + 0.3 * Math.sin(x * 0.13 + z * 0.09)) +
         smooth(150, 560, r) * (30 + 14 * Math.sin(Math.atan2(z, x) * 3 + 0.7) + 8 * Math.sin(x * 0.012 - z * 0.01));
}
function mouthAt(a, dr, out) {
  return out.set(MOUTH.cx + Math.cos(a) * (MOUTH.r + dr), 0, MOUTH.cz + Math.sin(a) * (MOUTH.r + dr));
}
// The balloons, tied (lift 0) and high in the sky (lift 1).
// Tied to the bench's west arm; they end side by side, the yellow one just
// north of the red and higher.
function redAt(l, out) {
  out = out || [];
  out[0] = -0.55 + 5.55 * l; out[1] = 2.15 + 30 * l; out[2] = -3.9 - 0.1 * l;
  return out;
}
function yellowAt(l, out) {
  out = out || [];
  out[0] = -0.95 + 5.95 * l; out[1] = 1.95 + 30.65 * l;
  out[2] = -4.15 - 0.47 * l + 14 * l * (1 - l);   // rising wide of the line in IV, closing in at the top
  return out;
}

function hash2(a, b) { var s = Math.sin(a * 127.1 + b * 311.7) * 43758.5453; return s - Math.floor(s); }

// ── Textures ─────────────────────────────────────────────────────────────
function canvasTex(w, h, paint, srgb) {
  var c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  paint(c.getContext('2d'), w, h);
  var t = new THREE.CanvasTexture(c);
  if (srgb !== false) t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// The mown lawn: broad stripes, speckle and a few daisies.
function lawnTexture(r) {
  return canvasTex(1024, 1024, function (x, w, h) {
    x.fillStyle = '#7fb447';
    x.fillRect(0, 0, w, h);
    for (var s = 0; s < 12; s++) {
      x.fillStyle = s % 2 ? 'rgba(255,255,220,0.07)' : 'rgba(20,60,10,0.06)';
      x.fillRect(s * w / 12, 0, w / 12, h);
    }
    for (var i = 0; i < 26000; i++) {
      var l = r();
      x.fillStyle = l < 0.5 ? 'rgba(40,90,20,0.22)' : 'rgba(200,235,140,0.2)';
      x.fillRect(r() * w, r() * h, 2 + r() * 3, 2 + r() * 3);
    }
    for (i = 0; i < 700; i++) {
      x.fillStyle = r() < 0.8 ? 'rgba(255,255,250,0.9)' : 'rgba(255,220,90,0.9)';
      x.beginPath();
      x.arc(r() * w, r() * h, 1.6 + r() * 1.4, 0, Math.PI * 2);
      x.fill();
    }
  });
}

function speckleTexture(r, base, dark, light, n) {
  var t = canvasTex(256, 256, function (x, w, h) {
    x.fillStyle = base;
    x.fillRect(0, 0, w, h);
    for (var i = 0; i < n; i++) {
      x.fillStyle = r() < 0.5 ? dark : light;
      x.fillRect(r() * w, r() * h, 1 + r() * 3, 1 + r() * 3);
    }
  });
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = 4;
  return t;
}

// Red gingham for the blanket.
function ginghamTexture() {
  var t = canvasTex(128, 128, function (x, w, h) {
    x.fillStyle = '#fbf6ee';
    x.fillRect(0, 0, w, h);
    x.fillStyle = 'rgba(206,36,44,0.55)';
    for (var i = 0; i < 4; i++) {
      x.fillRect(i * 32, 0, 16, h);
      x.fillRect(0, i * 32, w, 16);
    }
  });
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(4, 4);
  return t;
}

// A spray of leaves on a transparent card, white so vertex colour tints it.
function leafTexture(r) {
  var t = canvasTex(128, 128, function (x) {
    x.fillStyle = '#ffffff';
    for (var i = 0; i < 10; i++) {
      var ang = i / 10 * Math.PI * 2 + r() * 0.4, d = 18 + r() * 26;
      x.save();
      x.translate(64 + Math.cos(ang) * d * 0.6, 64 + Math.sin(ang) * d * 0.6);
      x.rotate(ang + Math.PI / 2);
      x.beginPath();
      x.ellipse(0, 0, 8 + r() * 4, 19 + r() * 8, 0, 0, Math.PI * 2);
      x.fill();
      x.restore();
    }
    x.beginPath();
    x.arc(64, 64, 15, 0, Math.PI * 2);
    x.fill();
  }, false);
  t.generateMipmaps = false;
  t.minFilter = THREE.LinearFilter;
  return t;
}

// The honeybee's stripes, head end at the top.
function stripeTexture() {
  return canvasTex(32, 128, function (x, w, h) {
    [[0, 34, '#f5bd22'], [34, 50, '#2a1d12'], [50, 70, '#f7c52a'], [70, 86, '#2a1d12'], [86, 102, '#f5bd22'], [102, 128, '#2a1d12']]
      .forEach(function (b) { x.fillStyle = b[2]; x.fillRect(0, b[0], w, b[1] - b[0]); });
  });
}

// A cumulus: overlapping puffs, flat underneath, greyer at the base.
function cloudTexture(r) {
  return canvasTex(256, 128, function (x) {
    for (var i = 0; i < 46; i++) {
      var u = r(), px = 30 + u * 196, top = 30 + Math.pow(Math.abs(u - 0.5) * 2, 1.6) * 52;
      var py = top + r() * (96 - top), rad = 14 + r() * 26 * (1 - Math.abs(u - 0.5));
      var g = x.createRadialGradient(px, py, 0, px, py, rad), shade = Math.round(255 - (py / 128) * 40);
      g.addColorStop(0, 'rgba(' + shade + ',' + shade + ',' + (shade + 6) + ',0.5)');
      g.addColorStop(1, 'rgba(' + shade + ',' + shade + ',' + (shade + 6) + ',0)');
      x.fillStyle = g;
      x.fillRect(px - rad, py - rad, rad * 2, rad * 2);
    }
  });
}

// The sun: a glow that falls off fast, and thin soft rays.
function glowTexture() {
  return canvasTex(256, 256, function (x) {
    var g = x.createRadialGradient(128, 128, 0, 128, 128, 128);
    [[0, 1], [0.06, 0.6], [0.15, 0.25], [0.35, 0.08], [0.7, 0.02], [1, 0]].forEach(function (s) {
      g.addColorStop(s[0], 'rgba(255,246,222,' + s[1] + ')');
    });
    x.fillStyle = g;
    x.fillRect(0, 0, 256, 256);
  });
}
function raysTexture(r) {
  return canvasTex(256, 256, function (x) {
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
  });
}
// A four-point twinkle for glints.
function twinkleTexture() {
  return canvasTex(128, 128, function (x) {
    x.translate(64, 64);
    x.globalCompositeOperation = 'lighter';
    [[0, 62, 3.2], [Math.PI / 2, 62, 3.2], [Math.PI / 4, 30, 1.8], [-Math.PI / 4, 30, 1.8]].forEach(function (s) {
      x.save();
      x.rotate(s[0]);
      var g = x.createLinearGradient(-s[1], 0, s[1], 0);
      g.addColorStop(0, 'rgba(255,250,230,0)');
      g.addColorStop(0.5, 'rgba(255,252,240,1)');
      g.addColorStop(1, 'rgba(255,250,230,0)');
      x.fillStyle = g;
      x.beginPath();
      x.moveTo(-s[1], 0); x.lineTo(0, -s[2]); x.lineTo(s[1], 0); x.lineTo(0, s[2]);
      x.fill();
      x.restore();
    });
    var g = x.createRadialGradient(0, 0, 0, 0, 0, 22);
    g.addColorStop(0, 'rgba(255,255,255,1)');
    g.addColorStop(1, 'rgba(255,240,200,0)');
    x.fillStyle = g;
    x.fillRect(-22, -22, 44, 44);
  });
}

// ── Geometry ─────────────────────────────────────────────────────────────
function box(w, h, d, col, x, y, z, rx) {
  var g = new THREE.BoxGeometry(w, h, d);
  if (rx) g.rotateX(rx);
  return tinted(g.translate(x, y, z), col);
}

// Push a geometry in or out along its normals by a smooth function of position.
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

// An apple tree: a short trunk and limbs (bark) and a crown of lumpy dark
// cores wrapped in leaf cards (vertex coloured, alpha-cut by the texture).
function appleTree(r, cards) {
  var bark = [], clumps = [], Y = new THREE.Vector3(0, 1, 0), q = new THREE.Quaternion();
  var top = new THREE.Vector3(0, 1.35, 0);
  bark.push(tinted(new THREE.CylinderGeometry(0.12, 0.19, 1.45, 7).translate(0, 0.72, 0), '#836a58'));
  function limb(from, dir, len, rad) {
    var g = new THREE.CylinderGeometry(rad * 0.55, rad, len, 5).translate(0, len / 2, 0);
    g.applyQuaternion(q.setFromUnitVectors(Y, dir));
    bark.push(tinted(g.translate(from.x, from.y, from.z), '#836a58'));
    return from.clone().addScaledVector(dir, len);
  }
  for (var k = 0; k < 4; k++) {
    var a = k / 4 * Math.PI * 2 + r() * 0.9, tilt = 0.75 + r() * 0.3;
    var dir = new THREE.Vector3(Math.cos(a) * Math.sin(tilt), Math.cos(tilt), Math.sin(a) * Math.sin(tilt));
    var end = limb(top, dir, 1.2 * (0.8 + r() * 0.4), 0.09);
    for (var t = 0; t < 2; t++) limb(end, dir.clone().add(new THREE.Vector3(r() - 0.5, r() * 0.6, r() - 0.5)).normalize(), 0.7, 0.035);
  }
  for (k = 0; k < 8; k++) {
    var ca = k / 7 * Math.PI * 2 + r() * 0.6, cd = k === 0 ? 0 : 1.05 + r() * 0.45, rad = 0.95 * (0.85 + r() * 0.35);
    clumps.push({ c: new THREE.Vector3(Math.cos(ca) * cd, 3.05 + r() * 0.7 - cd * 0.25 + (k === 0 ? 0.55 : 0), Math.sin(ca) * cd), r: rad });
  }
  var pos = [], nor = [], uv = [], col = [], n = new THREE.Vector3(), v = new THREE.Vector3();
  var e1 = new THREE.Vector3(), e2 = new THREE.Vector3(), c = new THREE.Vector3(), cc = new THREE.Color();
  clumps.forEach(function (cl, j) {
    var core = lumpy(new THREE.IcosahedronGeometry(cl.r * 0.66, 1), 0.12, j * 1.7).scale(1, 0.82, 1).translate(cl.c.x, cl.c.y, cl.c.z);
    var cp = core.attributes.position, cn = core.attributes.normal;
    for (var i = 0; i < cp.count; i++) {
      pos.push(cp.getX(i), cp.getY(i), cp.getZ(i));
      nor.push(cn.getX(i), cn.getY(i), cn.getZ(i));
      uv.push(0.5, 0.5);
      col.push(0.16, 0.3, 0.1);
    }
    core.dispose();
    for (var m = 0; m < cards; m++) {
      n.set(r() - 0.5, r() - 0.3, r() - 0.5).normalize();
      c.copy(n).multiplyScalar(cl.r * (0.7 + r() * 0.42));
      c.y *= 0.82;
      c.add(cl.c);
      v.set(r() - 0.5, r() - 0.5, r() - 0.5).multiplyScalar(1.4).add(n).normalize();
      e1.set(0, 1, 0).cross(v);
      if (e1.lengthSq() < 0.01) e1.set(1, 0, 0);
      e1.normalize().applyAxisAngle(v, r() * 6.28);
      e2.copy(v).cross(e1).normalize();
      var s = 0.62 * (0.75 + r() * 0.5) / 2;
      cc.setHSL(0.23 + r() * 0.06, 0.55 + r() * 0.15, 0.2 + r() * 0.12 + Math.max(n.y, 0) * 0.03);
      [[-1, -1], [1, -1], [1, 1], [-1, -1], [1, 1], [-1, 1]].forEach(function (qd) {
        pos.push(c.x + (e1.x * qd[0] + e2.x * qd[1]) * s, c.y + (e1.y * qd[0] + e2.y * qd[1]) * s, c.z + (e1.z * qd[0] + e2.z * qd[1]) * s);
        nor.push(n.x * 0.8, n.y * 0.8 + 0.45, n.z * 0.8);
        uv.push((qd[0] + 1) / 2, (qd[1] + 1) / 2);
        col.push(cc.r, cc.g, cc.b);
      });
    }
  });
  var crown = new THREE.BufferGeometry();
  crown.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  crown.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  crown.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  crown.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  return { bark: merge(bark), crown: crown, clumps: clumps };
}

// A cottage-garden flower about 0.4 m tall: a stem, a yellow eye and seven
// petals that the shader swings open about their hinges (aPetal marks the
// petals, which alone take the instance colour).
function flowerGeometry() {
  var pos = [], nor = [], col = [], petal = [], hinge = [], axis = [];
  function push(geo, c, isPetal, h, ax) {
    geo = geo.index ? geo.toNonIndexed() : geo;
    var p = geo.attributes.position, nn = geo.attributes.normal;
    for (var i = 0; i < p.count; i++) {
      pos.push(p.getX(i), p.getY(i), p.getZ(i));
      nor.push(nn.getX(i), nn.getY(i), nn.getZ(i));
      var k = isPetal ? 0.78 + 0.22 * clamp((p.getY(i) - h[1]) / 0.08, 0, 1) : 1;
      col.push(c[0] * k, c[1] * k, c[2] * k);
      petal.push(isPetal ? 1 : 0);
      hinge.push(h ? h[0] : 0, h ? h[1] : 0, h ? h[2] : 0);
      axis.push(ax ? ax[0] : 0, ax ? ax[1] : 0, ax ? ax[2] : 0);
    }
    geo.dispose();
  }
  var H = 0.36;
  push(new THREE.CylinderGeometry(0.005, 0.008, H, 4, 1, true).translate(0, H / 2, 0), [0.27, 0.45, 0.16], false);
  push(new THREE.CircleGeometry(0.06, 5).scale(0.5, 1, 1).rotateX(-1.1).translate(0.02, 0.12, 0.02), [0.3, 0.52, 0.18], false);
  var eye = new THREE.SphereGeometry(0.022, 8, 4, 0, Math.PI * 2, 0, Math.PI / 2).scale(1, 0.55, 1).translate(0, H, 0);
  push(eye, [0.98, 0.76, 0.16], false);
  for (var k = 0; k < 7; k++) {
    var a = k / 7 * Math.PI * 2, rx = Math.cos(a), rz = Math.sin(a), h = [rx * 0.014, H, rz * 0.014];
    // Closed: upright, facing out. Built as a rounded blade from the hinge up.
    var shape = new THREE.Shape();
    shape.moveTo(-0.008, 0);
    shape.quadraticCurveTo(-0.034, 0.05, -0.012, 0.088);
    shape.quadraticCurveTo(0, 0.096, 0.012, 0.088);
    shape.quadraticCurveTo(0.034, 0.05, 0.008, 0);
    var g = new THREE.ShapeGeometry(shape, 3);
    g.rotateY(Math.PI / 2 - a);               // face outward (+z of the shape points along the radius)
    g.translate(h[0], h[1], h[2]);
    push(g, [1, 1, 1], true, h, [Math.sin(a), 0, -Math.cos(a)]);
  }
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  geo.setAttribute('aPetal', new THREE.Float32BufferAttribute(petal, 1));
  geo.setAttribute('aHinge', new THREE.Float32BufferAttribute(hinge, 3));
  geo.setAttribute('aAxis', new THREE.Float32BufferAttribute(axis, 3));
  return geo;
}

// A star of petals facing +z, tips cupped forward.
function petalStar(rOut, rIn, points, cup) {
  var g = new THREE.CircleGeometry(rOut, points * 2), p = g.attributes.position;
  for (var i = 1; i < p.count; i++) {
    var tip = i % 2 === 1, s = tip ? 1 : rIn / rOut;
    p.setXYZ(i, p.getX(i) * s, p.getY(i) * s, tip ? cup : 0);
  }
  g.computeVertexNormals();
  return g;
}

// A sunflower head facing +z: two rings of petals, a domed seed disk and a
// green back.
function sunflowerHead() {
  var disk = new THREE.CircleGeometry(0.13, 16);
  disk.attributes.position.setZ(0, 0.03);
  disk.computeVertexNormals();
  return merge([
    tinted(petalStar(0.31, 0.12, 14, 0.06), '#ffc61a'),
    tinted(petalStar(0.27, 0.12, 14, 0.04).rotateZ(Math.PI / 14).translate(0, 0, -0.012), '#f29a12'),
    tinted(disk.translate(0, 0, 0.014), '#5b3412'),
    tinted(new THREE.CircleGeometry(0.15, 12).rotateY(Math.PI).translate(0, 0, -0.025), '#4f7a28'),
    tinted(new THREE.CylinderGeometry(0.03, 0.05, 0.08, 6).rotateX(Math.PI / 2).translate(0, 0, -0.06), '#4f7a28')
  ]);
}
// Its stem (1 m, scaled per plant) with three leaves.
function sunflowerStem(r) {
  var parts = [tinted(new THREE.CylinderGeometry(0.018, 0.03, 1, 5, 1, true).translate(0, 0.5, 0), '#4c7228')];
  [0.32, 0.55, 0.76].forEach(function (h, k) {
    var leaf = new THREE.Shape();
    leaf.moveTo(0, 0);
    leaf.quadraticCurveTo(0.12, 0.1, 0, 0.26);
    leaf.quadraticCurveTo(-0.12, 0.1, 0, 0);
    parts.push(tinted(new THREE.ShapeGeometry(leaf, 3).rotateX(Math.PI / 2 - 0.5).rotateY(k * 2.3 + r()).translate(0, h, 0), '#5b8a2e'));
  });
  return merge(parts);
}

// A balloon: a sphere drawn into a soft teardrop, with a knot.
function balloonGeometry() {
  var g = new THREE.SphereGeometry(0.3, 28, 20), p = g.attributes.position;
  for (var i = 0; i < p.count; i++) {
    var y = p.getY(i) / 0.3, k = y < 0 ? 1 - 0.32 * y * y : 1 + 0.04 * (1 - y * y);
    p.setXYZ(i, p.getX(i) * k, p.getY(i) * 1.16, p.getZ(i) * k);
  }
  g.computeVertexNormals();
  var knot = new THREE.CylinderGeometry(0.012, 0.03, 0.045, 8).translate(0, -0.37, 0);
  var parts = [g, knot].map(function (x) { return tinted(x, '#ffffff'); });
  return merge(parts);
}

// ── Synthesised sound cues ───────────────────────────────────────────────
function bell(ac, out, f, t, dur, gain) {
  [[1, 1], [2.01, 0.35], [3.02, 0.16]].forEach(function (pt) {
    var o = ac.createOscillator(), g = ac.createGain();
    o.type = 'sine';
    o.frequency.value = f * pt[0];
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain * pt[1], t + 0.008);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur / pt[0]);
    o.connect(g); g.connect(out);
    o.start(t); o.stop(t + dur + 0.05);
  });
}

// One honeybee: a buzzing drone that wanders in pitch, settles on a flower
// (quieter), lifts again and drifts across.
function honeybee(ac, out) {
  var t = ac.currentTime, len = 3.6, g = ac.createGain(), bp = ac.createBiquadFilter();
  var pan = ac.createStereoPanner ? ac.createStereoPanner() : null;
  bp.type = 'bandpass';
  bp.frequency.value = 950;
  bp.Q.value = 1.1;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.07, t + 0.45);
  g.gain.exponentialRampToValueAtTime(0.025, t + 1.3);
  g.gain.exponentialRampToValueAtTime(0.06, t + 2.0);
  g.gain.exponentialRampToValueAtTime(0.0001, t + len);
  bp.connect(g);
  if (pan) {
    pan.pan.setValueAtTime(0.7, t);
    pan.pan.linearRampToValueAtTime(0.1, t + 1.3);
    pan.pan.linearRampToValueAtTime(-0.6, t + len);
    g.connect(pan); pan.connect(out);
  } else {
    g.connect(out);
  }
  var vib = ac.createOscillator(), vg = ac.createGain();
  vib.frequency.value = 24;
  vg.gain.value = 5;
  vib.connect(vg);
  [['sawtooth', 1], ['square', 2.004]].forEach(function (w) {
    var o = ac.createOscillator(), og = ac.createGain();
    o.type = w[0];
    o.frequency.setValueAtTime(232 * w[1], t);
    o.frequency.linearRampToValueAtTime(248 * w[1], t + 0.5);
    o.frequency.linearRampToValueAtTime(214 * w[1], t + 1.3);
    o.frequency.linearRampToValueAtTime(240 * w[1], t + 2.0);
    o.frequency.linearRampToValueAtTime(220 * w[1], t + len);
    og.gain.value = w[1] > 1 ? 0.25 : 1;
    vg.connect(o.frequency);
    o.connect(og); og.connect(bp);
    o.start(t); o.stop(t + len + 0.05);
  });
  vib.start(t); vib.stop(t + len + 0.05);
}

// The balloon lets go: a soft little slide-whistle up.
function lift(ac, out) {
  var t = ac.currentTime, o = ac.createOscillator(), g = ac.createGain(), vib = ac.createOscillator(), vg = ac.createGain();
  o.type = 'sine';
  o.frequency.setValueAtTime(520, t);
  o.frequency.exponentialRampToValueAtTime(1180, t + 1.1);
  vib.frequency.value = 6;
  vg.gain.value = 9;
  vib.connect(vg); vg.connect(o.frequency);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.035, t + 0.15);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 1.4);
  o.connect(g); g.connect(out);
  o.start(t); vib.start(t);
  o.stop(t + 1.5); vib.stop(t + 1.5);
}

// The sun swells: a warm major chord.
function warmth(ac, out) {
  var t = ac.currentTime, lp = ac.createBiquadFilter();
  lp.type = 'lowpass';
  lp.frequency.value = 1500;
  lp.connect(out);
  [261.6, 329.6, 392, 523.3].forEach(function (f) {
    var o = ac.createOscillator(), g = ac.createGain();
    o.type = 'triangle';
    o.frequency.value = f;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.03, t + 1.6);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 5.5);
    o.connect(g); g.connect(lp);
    o.start(t); o.stop(t + 5.6);
  });
}

// ":)": a bright little chime, up the major chord and a twinkle on top.
function smileChime(ac, out) {
  var t = ac.currentTime + 0.02;
  [[1046.5, 0], [1318.5, 0.11], [1568, 0.22], [2093, 0.36]].forEach(function (n) { bell(ac, out, n[0], t + n[1], 1.9, 0.06); });
  bell(ac, out, 3136, t + 0.62, 0.9, 0.025);
}

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(1116);
  var gl = makeRenderer(canvas, { shadows: !small, clear: '#cfe2f2' });
  gl.toneMappingExposure = 1.05;
  var world = new THREE.Scene();
  world.fog = new THREE.Fog('#dde9f2', 120, 1100);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.05, 3000);
  camera.rotation.order = 'YXZ';
  var up = new THREE.Vector3(0, 1, 0), tmp = new THREE.Color(), m4 = new THREE.Matrix4(), q4 = new THREE.Quaternion();
  var v3 = new THREE.Vector3(), s3 = new THREE.Vector3();
  var U = { uClock: { value: 0 }, uWind: { value: 0.15 }, uOpen: { value: 0 } };

  // ── Sky ──
  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#1d5fca', mid: '#5a9be4', horizon: '#cfe2f1', sun: '#fff4dc' }, 1500);
  // Tone-map the dome like everything else, so the sky looks the same with
  // or without the light-shaft pass.
  dome.mesh.material.fragmentShader = dome.mesh.material.fragmentShader.replace(/}\s*$/,
    '\n#include <tonemapping_fragment>\n#include <colorspace_fragment>\n}');
  sky.add(dome.mesh);
  function additive(map, opacity) {
    return new THREE.Sprite(new THREE.SpriteMaterial({ map: map, blending: THREE.AdditiveBlending, depthWrite: false,
                                                       transparent: true, fog: false, opacity: opacity == null ? 1 : opacity }));
  }
  var sunDisc = additive(softSprite('rgba(255,255,250,1)', 'rgba(255,250,230,0)'));
  var sunGlow = additive(glowTexture());
  var sunRays = additive(raysTexture(r), 0.25);
  sky.add(sunGlow, sunRays, sunDisc);

  // Fair-weather clouds all round, a few higher ones for the sky beat.
  var cloudTex = cloudTexture(r), clouds = [];
  for (var i = 0; i < 22; i++) {
    var cl = new THREE.Sprite(new THREE.SpriteMaterial({ map: cloudTex, transparent: true, depthWrite: false, fog: false, opacity: 0.85 }));
    var ca = r() * Math.PI * 2, ce = i < 15 ? 0.04 + r() * 0.2 : 0.3 + r() * 0.35;
    if (i >= 15) ca = SUN_AZ - 0.9 + r() * 1.6;
    cl.position.set(Math.sin(ca) * Math.cos(ce) * 1100, Math.sin(ce) * 1100 + 20, -Math.cos(ca) * Math.cos(ce) * 1100);
    cl.scale.set(240 + r() * 240, 100 + r() * 70, 1);
    cl.userData.a = ca;
    cl.userData.e = ce;
    sky.add(cl);
    clouds.push(cl);
  }

  // ── Light ──
  var hemi = new THREE.HemisphereLight('#dcecff', '#7d8d50', 1.25);
  var sun = new THREE.DirectionalLight('#fff1d6', 2.7);
  sun.castShadow = !small;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = sun.shadow.camera.bottom = -30;
  sun.shadow.camera.right = sun.shadow.camera.top = 30;
  sun.shadow.camera.far = 160;
  sun.shadow.bias = -0.0005;
  sun.shadow.normalBias = 0.04;
  sun.target.position.set(0, 0, 0);
  world.add(hemi, sun, sun.target);

  // ── Land: meadow round the garden, fields on the far hills ──
  var fieldCols = ['#6a9a3e', '#7aa446', '#8aa64e', '#a8ac5e', '#5f8d3a', '#94aa52', '#b4ae64'].map(function (c) { return new THREE.Color(c); });
  var meadow = new THREE.Color('#6e9e3c');
  var ground = terrain(2400, small ? 160 : 220, 0, 0, land,
    new THREE.MeshLambertMaterial({ vertexColors: true, map: speckleTexture(r, '#ffffff', 'rgba(60,90,30,0.5)', 'rgba(255,255,200,0.4)', 2600) }),
    function (x, z) {
      var fx = (x * 0.85 + z * 0.35) / 60, fz = (z * 0.9 - x * 0.3) / 46;
      var field = fieldCols[Math.floor(hash2(Math.floor(fx), Math.floor(fz)) * fieldCols.length)];
      return tmp.copy(meadow).lerp(field, smooth(70, 160, Math.hypot(x, z)));
    });
  ground.material.map.repeat.set(380, 380);
  world.add(ground);

  // The face: the lawn and its gravel ring, a hair above the meadow.
  var lawn = new THREE.Mesh(new THREE.CircleGeometry(LAWN, 128).rotateX(-Math.PI / 2).translate(0, 0.012, 0),
                            new THREE.MeshLambertMaterial({ map: lawnTexture(rng(5)) }));
  lawn.material.map.anisotropy = 8;
  lawn.receiveShadow = true;
  var gravelTex = speckleTexture(r, '#e2d4b0', 'rgba(150,130,100,0.6)', 'rgba(255,255,250,0.7)', 5000);
  gravelTex.repeat.set(14, 14);
  var path = new THREE.Mesh(new THREE.RingGeometry(LAWN - 0.02, PATH, 160, 1).rotateX(-Math.PI / 2).translate(0, 0.014, 0),
                            new THREE.MeshLambertMaterial({ map: gravelTex }));
  path.receiveShadow = true;
  world.add(lawn, path);

  // ── Grass: short tufts on the lawn, longer in the meadow ──
  function tuft(blades, hgt, root, tipC) {
    var pos = [], nor = [], col = [], a0 = new THREE.Color(root), a1 = new THREE.Color(tipC), rr = rng(blades * 7);
    for (var b = 0; b < blades; b++) {
      var a = rr() * 6.28, w = 0.01 + rr() * 0.01, h = hgt * (0.6 + rr() * 0.6), lean = 0.04 + rr() * 0.1;
      var ox = Math.cos(a) * 0.04, oz = Math.sin(a) * 0.04, px = -Math.sin(a) * w, pz = Math.cos(a) * w;
      pos.push(ox - px, 0, oz - pz, ox + px, 0, oz + pz, ox + Math.cos(a) * lean, h, oz + Math.sin(a) * lean);
      nor.push(0, 1, 0, 0, 1, 0, 0, 1, 0);
      col.push(a0.r, a0.g, a0.b, a0.r, a0.g, a0.b, a1.r, a1.g, a1.b);
    }
    var g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    return g;
  }
  var grassMat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
  grassMat.onBeforeCompile = function (sh) {
    sh.uniforms.uClock = U.uClock;
    sh.uniforms.uWind = U.uWind;
    sh.vertexShader = 'uniform float uClock; uniform float uWind;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n float gph = instanceMatrix[3][0] * 0.4 + instanceMatrix[3][2] * 0.3;\n' +
      ' float gb = (sin(uClock * 1.6 + gph) * 0.6 + 0.4) * (0.08 + uWind * 0.3) * position.y * position.y * 4.0;\n' +
      ' transformed.x += gb; transformed.z += gb * 0.3;');
  };
  var lawnTufts = new THREE.InstancedMesh(tuft(5, 0.055, '#5f9436', '#b8e274'), grassMat, small ? 5000 : 14000);
  scatter(lawnTufts, 60000, function (n, p, q, s, c) {
    var a = r() * 6.28, d = Math.sqrt(r()) * (LAWN - 0.1);
    p.set(Math.cos(a) * d, 0.012, Math.sin(a) * d);
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar(0.8 + r() * 0.6);
    c.setHSL(0.24 + r() * 0.05, 0.4, 0.8 + r() * 0.2);
  });
  var meadowTufts = new THREE.InstancedMesh(tuft(8, 0.22, '#4d7c2e', '#bfda78'), grassMat, small ? 8000 : 22000);
  scatter(meadowTufts, 90000, function (n, p, q, s, c) {
    var a = r() * 6.28, d = PATH + 0.15 + Math.pow(r(), 1.4) * 30;
    p.set(Math.cos(a) * d, 0, Math.sin(a) * d);
    q.setFromAxisAngle(up, r() * 6.28);
    s.set(1.2, 0.7 + r() * 0.9, 1.2);
    c.setHSL(0.2 + r() * 0.08, 0.32, 0.75 + r() * 0.25);
  });
  lawnTufts.receiveShadow = meadowTufts.receiveShadow = true;
  world.add(lawnTufts, meadowTufts);

  // Wild flowers in the meadow: soft points.
  var wfPos = [], wfCol = [], wfc = ['#ffd84a', '#ffffff', '#ffffff', '#e67aa8', '#b9a4ec'].map(function (c) { return new THREE.Color(c); });
  for (i = 0; i < (small ? 2000 : 5000); i++) {
    var wa = r() * 6.28, wd = PATH + 0.6 + Math.pow(r(), 1.3) * 32;
    wfPos.push(Math.cos(wa) * wd, 0.18 + r() * 0.2, Math.sin(wa) * wd);
    var wc = wfc[Math.floor(r() * wfc.length)];
    wfCol.push(wc.r, wc.g, wc.b);
  }
  var wfGeo = new THREE.BufferGeometry();
  wfGeo.setAttribute('position', new THREE.Float32BufferAttribute(wfPos, 3));
  wfGeo.setAttribute('color', new THREE.Float32BufferAttribute(wfCol, 3));
  world.add(new THREE.Points(wfGeo, new THREE.PointsMaterial({ size: 0.07, vertexColors: true, transparent: true, depthWrite: false,
    map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)') })));

  // ── Trees: the two eyes, the orchard round the lawn, far woods ──
  var leafTex = leafTexture(rng(12));
  var barkMat = new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true, emissive: '#2a2018' });
  var crownMat = new THREE.MeshLambertMaterial({ map: leafTex, alphaTest: 0.5, vertexColors: true, side: THREE.DoubleSide });
  crownMat.onBeforeCompile = function (sh) {
    sh.uniforms.uClock = U.uClock;
    sh.uniforms.uWind = U.uWind;
    sh.vertexShader = 'uniform float uClock; uniform float uWind;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n float tph = instanceMatrix[3][0] * 0.31 + instanceMatrix[3][2] * 0.17;\n' +
      ' float sway = (sin(uClock * 1.3 + tph) + 0.4 * sin(uClock * 3.3 + tph * 2.0 + position.x * 3.0)) * (0.012 + uWind * 0.05) * max(position.y - 1.5, 0.0);\n' +
      ' transformed.x += sway; transformed.z += sway * 0.6;');
    sh.fragmentShader = sh.fragmentShader.replace('#include <normal_fragment_begin>',
      '#include <normal_fragment_begin>\n#ifdef DOUBLE_SIDED\n normal *= faceDirection;\n#endif');
  };
  var crownDepth = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: leafTex, alphaTest: 0.5, side: THREE.DoubleSide });

  var spots = [{ x: EYES[0][0], z: EYES[0][1], s: 1.08, a: 0.6 }, { x: EYES[1][0], z: EYES[1][1], s: 1.08, a: 2.4 }];
  for (var gx = -42; gx <= 42; gx += 6.4) {
    for (var gz = -42; gz <= 42; gz += 6.4) {
      var tx = gx + (Math.floor(gz / 6.4) % 2 ? 3.2 : 0) + (r() - 0.5) * 1.2, tz = gz + (r() - 0.5) * 1.2, td = Math.hypot(tx, tz);
      if (td < 14.6 || td > (small ? 34 : 44)) continue;
      spots.push({ x: tx, z: tz, s: 0.85 + r() * 0.35, a: r() * 6.28 });
    }
  }
  var kinds = [appleTree(rng(31), small ? 40 : 70), appleTree(rng(47), small ? 40 : 70)];
  var apples = [], beamFrom = new THREE.Vector3();
  kinds.forEach(function (kind, k) {
    var mine = spots.filter(function (t, n) { return n % 2 === k; });
    var trunks = new THREE.InstancedMesh(kind.bark, barkMat, mine.length), crowns = new THREE.InstancedMesh(kind.crown, crownMat, mine.length);
    mine.forEach(function (t, n) {
      m4.compose(v3.set(t.x, land(t.x, t.z) - 0.03, t.z), q4.setFromAxisAngle(up, t.a), s3.setScalar(t.s));
      trunks.setMatrixAt(n, m4);
      crowns.setMatrixAt(n, m4);
      // The eyes are a shade darker, so they read from the sky.
      crowns.setColorAt(n, tmp.setScalar(t === spots[0] || t === spots[1] ? 0.74 : 0.92 + r() * 0.16));
      // Apples round the crown's lower half.
      for (var a = 0; a < 9; a++) {
        var cl = kind.clumps[1 + Math.floor(r() * (kind.clumps.length - 1))];
        v3.set(r() - 0.5, -0.15 - r() * 0.6, r() - 0.5).normalize().multiplyScalar(cl.r * 0.9).add(cl.c).applyMatrix4(m4);
        apples.push(v3.x, v3.y, v3.z);
      }
      // The sunbeam starts in the right eye's crown.
      if (t === spots[1]) beamFrom.set(t.x, 2.9, t.z);
    });
    crowns.customDepthMaterial = crownDepth;
    trunks.castShadow = crowns.castShadow = !small;
    trunks.receiveShadow = crowns.receiveShadow = !small;
    world.add(trunks, crowns);
  });
  var appleMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.055, 8, 6), new THREE.MeshLambertMaterial(), apples.length / 3);
  var appleCols = ['#c8262a', '#d6402a', '#b42a26', '#e0552e', '#a8c040'].map(function (c) { return new THREE.Color(c); });
  scatter(appleMesh, apples.length / 3, function (n, p, q, s, c) {
    p.set(apples[n * 3], apples[n * 3 + 1], apples[n * 3 + 2]);
    q.identity();
    s.setScalar(0.85 + r() * 0.35);
    c.copy(appleCols[Math.floor(r() * appleCols.length)]);
  });
  world.add(appleMesh);

  // Field trees over the countryside: the same leafy crowns as the orchard,
  // with fewer cards, larger and without shadows.
  var farKind = appleTree(rng(58), small ? 14 : 22), NFT = small ? 160 : 380;
  var farTrunks = new THREE.InstancedMesh(farKind.bark, barkMat, NFT), farCrowns = new THREE.InstancedMesh(farKind.crown, crownMat, NFT);
  scatter(farCrowns, 6000, function (n, p, q, s, c) {
    var a = r() * 6.28, d = 50 + Math.pow(r(), 1.6) * 420, x = Math.cos(a) * d, z = Math.sin(a) * d;
    // In loose clumps and hedgerow lines.
    if (hash2(Math.floor(x / 40), Math.floor(z / 40)) < 0.45 && r() < 0.85) return false;
    p.set(x, land(x, z) - 0.1, z);
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar(1.3 + r() * 0.7);
    c.setScalar(0.85 + r() * 0.2);
    farTrunks.setMatrixAt(n, m4.compose(p, q, s));
  });
  farTrunks.count = farCrowns.count;
  farTrunks.instanceMatrix.needsUpdate = true;
  world.add(farTrunks, farCrowns);

  // The apple of my eye: bigger, glossy, glowing a little when named.
  var hero = new THREE.Group();
  var heroBody = new THREE.Mesh(new THREE.SphereGeometry(0.085, 20, 14).scale(1, 0.92, 1),
    new THREE.MeshPhongMaterial({ color: '#d61e25', specular: '#ffffff', shininess: 70, emissive: '#5a0000' }));
  // A few leaves on the twig it hangs from, so it is clearly on the branch.
  var twigLeaves = [];
  for (i = 0; i < 6; i++) {
    var tl = new THREE.Shape();
    tl.moveTo(0, 0); tl.quadraticCurveTo(0.05, 0.05, 0, 0.13); tl.quadraticCurveTo(-0.05, 0.05, 0, 0);
    twigLeaves.push(tinted(new THREE.ShapeGeometry(tl, 3).rotateX(-0.4 - r() * 0.8).rotateY(i * 1.05 + r() * 0.4).translate(0, 0.2 + r() * 0.06, 0), i % 2 ? '#4f8a2a' : '#6aa23a'));
  }
  twigLeaves.push(tinted(new THREE.CylinderGeometry(0.008, 0.012, 0.5, 5).rotateZ(0.5).translate(0.1, 0.36, 0), '#6b5444'));
  var stalk = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.007, 0.16, 5).translate(0, 0.1, 0), new THREE.MeshLambertMaterial({ color: '#5a4030' }));
  var leafShape = new THREE.Shape();
  leafShape.moveTo(0, 0); leafShape.quadraticCurveTo(0.04, 0.03, 0, 0.09); leafShape.quadraticCurveTo(-0.04, 0.03, 0, 0);
  var heroLeaf = new THREE.Mesh(new THREE.ShapeGeometry(leafShape, 3).rotateZ(-0.9).translate(0.005, 0.12, 0),
                                new THREE.MeshLambertMaterial({ color: '#5d9a2e', side: THREE.DoubleSide }));
  var heroGlow = additive(softSprite('rgba(255,140,110,1)', 'rgba(255,90,60,0)'), 0);
  heroGlow.scale.setScalar(0.5);
  hero.add(heroBody, stalk, heroLeaf, heroGlow,
           new THREE.Mesh(merge(twigLeaves), new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide })));
  hero.position.set(HERO[0], HERO[1], HERO[2]);
  heroBody.castShadow = !small;
  world.add(hero);

  // ── The bench, its blanket and cup, and the balloons ──
  // Built round the origin facing +z, then set between the eyes.
  var B = { x: 0, z: 0 }, wood = '#b9824e', dark = '#8c5f36', benchParts = [], benchGroup = new THREE.Group();
  benchGroup.position.set(BENCH.x, 0, BENCH.z);
  benchGroup.rotation.y = BENCH.turn;
  world.add(benchGroup);
  for (i = 0; i < 4; i++) benchParts.push(box(1.72, 0.03, 0.085, wood, B.x, 0.455, B.z - 0.17 + i * 0.115));
  for (i = 0; i < 3; i++) benchParts.push(box(1.72, 0.08, 0.022, wood, B.x, 0.6 + i * 0.12, B.z - 0.25 - i * 0.025, -0.2));
  [-0.84, 0.84].forEach(function (sx) {
    benchParts.push(box(0.06, 0.46, 0.06, dark, B.x + sx, 0.23, B.z + 0.18));
    benchParts.push(box(0.06, 0.92, 0.06, dark, B.x + sx, 0.46, B.z - 0.24, -0.08));
    benchParts.push(box(0.07, 0.04, 0.52, wood, B.x + sx + Math.sign(sx) * 0.01, 0.67, B.z - 0.02));
    benchParts.push(box(0.05, 0.2, 0.05, dark, B.x + sx, 0.56, B.z + 0.19));
    benchParts.push(box(0.05, 0.05, 0.4, dark, B.x + sx, 0.38, B.z - 0.03));
  });
  benchParts.push(box(1.62, 0.05, 0.04, dark, B.x, 0.405, B.z + 0.17));
  var bench = new THREE.Mesh(merge(benchParts), new THREE.MeshLambertMaterial({ vertexColors: true }));
  bench.castShadow = bench.receiveShadow = !small;
  benchGroup.add(bench);

  // The blanket: draped over the back, along the seat and over the front.
  var drape = new THREE.CatmullRomCurve3([[-0.31, 0.9], [-0.27, 0.66], [-0.22, 0.5], [-0.05, 0.488], [0.14, 0.488], [0.215, 0.43], [0.23, 0.2]]
    .map(function (p) { return new THREE.Vector3(0, p[1], B.z + p[0]); }));
  var blanketGeo = new THREE.PlaneGeometry(1, 1, 14, 30), bp = blanketGeo.attributes.position, dp = new THREE.Vector3();
  for (i = 0; i < bp.count; i++) {
    var bu = bp.getX(i) + 0.5, bv = 0.5 - bp.getY(i);
    drape.getPointAt(bv, dp);
    var bx = lerp(-0.02, 0.8, bu), wr = 0.012 * Math.sin(bu * 19 + bv * 5) * smooth(0.65, 1, bv);
    bp.setXYZ(i, B.x + bx + wr * 0.5, dp.y + 0.006 + Math.abs(wr) * 0.3, dp.z + wr);
  }
  blanketGeo.computeVertexNormals();
  var blanket = new THREE.Mesh(blanketGeo, new THREE.MeshLambertMaterial({ map: ginghamTexture(), side: THREE.DoubleSide }));
  blanket.castShadow = blanket.receiveShadow = !small;
  benchGroup.add(blanket);

  var cup = new THREE.Mesh(merge([
    tinted(new THREE.CylinderGeometry(0.046, 0.04, 0.095, 18).translate(0, 0.0475, 0), '#f7f2e8'),
    tinted(new THREE.TorusGeometry(0.026, 0.007, 6, 12, Math.PI).rotateZ(-Math.PI / 2).translate(0.046, 0.05, 0), '#f7f2e8'),
    tinted(new THREE.CircleGeometry(0.04, 16).rotateX(-Math.PI / 2).translate(0, 0.085, 0), '#6a3f22')
  ]), new THREE.MeshPhongMaterial({ vertexColors: true, shininess: 60, specular: '#666666' }));
  cup.position.set(B.x - 0.42, 0.47, B.z - 0.02);
  cup.rotation.y = 0.5;
  cup.castShadow = !small;
  benchGroup.add(cup);
  var steamTex = softSprite('rgba(255,255,255,0.55)', 'rgba(255,255,255,0)'), steam = [];
  for (i = 0; i < 5; i++) {
    var st = new THREE.Sprite(new THREE.SpriteMaterial({ map: steamTex, transparent: true, depthWrite: false, opacity: 0 }));
    st.userData.ph = i / 5;
    benchGroup.add(st);
    steam.push(st);
  }

  // The sweet tin, half in the grass, and its glint.
  var tin = new THREE.Group();
  tin.add(new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.075, 0.045, 24).translate(0, 0.0225, 0),
    new THREE.MeshPhongMaterial({ color: '#d8c38a', specular: '#fff6dc', shininess: 90 })));
  tin.add(new THREE.Mesh(new THREE.CylinderGeometry(0.079, 0.079, 0.016, 24).translate(0, 0.05, 0),
    new THREE.MeshPhongMaterial({ color: '#c9283a', specular: '#ffffff', shininess: 80 })));
  tin.add(new THREE.Mesh(new THREE.TorusGeometry(0.079, 0.004, 6, 32).rotateX(Math.PI / 2).translate(0, 0.058, 0),
    new THREE.MeshPhongMaterial({ color: '#e8c25a', specular: '#ffffff', shininess: 100 })));
  tin.position.set(TIN[0], 0.0, TIN[1]);
  tin.rotation.set(0.12, 0.7, -0.08);
  world.add(tin);
  var twinkleTex = twinkleTexture();
  var glint = additive(twinkleTex, 0);
  glint.position.set(TIN[0] + 0.05, 0.08, TIN[1] + 0.03);
  world.add(glint);
  // A twinkle in the right eye, for the last beat.
  var eyeTwinkle = additive(twinkleTex, 0);
  eyeTwinkle.position.set(EYES[1][0] + 0.4, 5.4, EYES[1][1] + 0.6);
  world.add(eyeTwinkle);

  var balloonGeo = balloonGeometry();
  function balloon(col) {
    var m = new THREE.Mesh(balloonGeo, new THREE.MeshPhongMaterial({ color: col, specular: '#ffffff', shininess: 110, emissive: col, emissiveIntensity: 0.12 }));
    m.castShadow = !small;
    m.userData.shadow = !small;
    var line = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: '#f2eee6', transparent: true, opacity: 0.9 }));
    line.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(16 * 3), 3));
    line.frustumCulled = false;
    world.add(m, line);
    return { mesh: m, line: line, pos: new THREE.Vector3() };
  }
  var red = balloon('#e3232c'), yellow = balloon('#ffc21f');
  benchGroup.updateMatrixWorld(true);
  var tie = benchGroup.localToWorld(new THREE.Vector3(-0.85, 0.69, -0.02));

  // ── Flowers: the ring round the lawn and the yellow band of the smile ──
  var flowerMat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
  flowerMat.onBeforeCompile = function (sh) {
    sh.uniforms.uOpen = U.uOpen;
    sh.uniforms.uClock = U.uClock;
    var rot = 'vec3 rotA(vec3 v, vec3 k, float a){ float c = cos(a), s = sin(a); return v * c + cross(k, v) * s + k * dot(k, v) * (1.0 - c); }\n';
    sh.vertexShader = 'uniform float uOpen; uniform float uClock; attribute float aPetal; attribute vec3 aHinge; attribute vec3 aAxis;\n' + rot +
      sh.vertexShader
        .replace('#include <beginnormal_vertex>',
          'float fph = fract(instanceMatrix[3][0] * 1.37 + instanceMatrix[3][2] * 2.11);\n' +
          ' float fo = clamp(uOpen * 1.7 - fph * 0.7, 0.0, 1.0); fo = fo * fo * (3.0 - 2.0 * fo);\n' +
          ' float fth = mix(0.1, 1.38, fo);\n' +
          '#include <beginnormal_vertex>\n objectNormal = mix(objectNormal, rotA(objectNormal, aAxis, fth), aPetal);')
        .replace('#include <begin_vertex>',
          '#include <begin_vertex>\n transformed = mix(transformed, aHinge + rotA(position - aHinge, aAxis, fth), aPetal);\n' +
          ' float fsw = sin(uClock * 1.7 + fph * 6.28) * 0.02 * position.y * position.y * 8.0; transformed.x += fsw; transformed.z += fsw * 0.5;')
        .replace('#include <color_vertex>', 'vColor = color * mix(vec3(1.0), instanceColor, aPetal);');
  };
  var NF = small ? 1100 : 2400, flowers = new THREE.InstancedMesh(flowerGeometry(), flowerMat, NF);
  var ringCols = ['#ff8fb4', '#ffffff', '#f6f0ff', '#c9a8ff', '#ff6f8e', '#ffd0e0', '#9cc4ff'].map(function (c) { return new THREE.Color(c); });
  var smileCols = ['#ffc21a', '#ffb012', '#ffd23a', '#ff9a1a'].map(function (c) { return new THREE.Color(c); });
  var mouthP = new THREE.Vector3(), flowerAt = [];
  scatter(flowers, NF * 4, function (n, p, q, s, c) {
    if (n < NF * 0.72) {
      var a = r() * 6.28, d = lerp(BAND[0], BAND[1], r());
      p.set(Math.cos(a) * d, 0.01, Math.sin(a) * d);
      c.copy(ringCols[Math.floor(r() * ringCols.length)]);
    } else {
      mouthAt(lerp(MOUTH.a0 - 0.03, MOUTH.a1 + 0.03, r()), (r() - 0.5) * 1.7, mouthP);
      p.copy(mouthP).setY(0.01);
      c.copy(smileCols[Math.floor(r() * smileCols.length)]);
    }
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar((small ? 1.25 : 1.05) * (0.85 + r() * 0.4));
    flowerAt.push(p.x, s.x * 0.36, p.z);
  });
  flowers.receiveShadow = !small;
  world.add(flowers);
  function nearestFlower(x, z, out) {
    var best = 0, bd = 1e9;
    for (var k = 0; k < flowerAt.length / 3; k++) {
      var d = Math.hypot(flowerAt[k * 3] - x, flowerAt[k * 3 + 2] - z);
      if (d < bd) { bd = d; best = k; }
    }
    return out.set(flowerAt[best * 3], flowerAt[best * 3 + 1] + 0.02, flowerAt[best * 3 + 2]);
  }

  // ── Sunflowers: the mouth ──
  var sfs = [];
  [[0.15, 1.15, 1.5, 0.52], [0.7, 1.55, 2.0, 0.52]].forEach(function (rowDef, k) {
    for (var a = MOUTH.a0 + k * 0.04; a <= MOUTH.a1; a += rowDef[3] / (MOUTH.r + rowDef[0])) {
      mouthAt(a, rowDef[0] + (r() - 0.5) * 0.12, v3);
      sfs.push({ x: v3.x, z: v3.z, h: lerp(rowDef[1], rowDef[2], r()), s: 0.9 + r() * 0.25, d: r() * 0.35 + (a - MOUTH.a0) / (MOUTH.a1 - MOUTH.a0) * 0.25,
                 sway: r() * 6.28, droop: new THREE.Vector3(r() - 0.5, -2.4, r() - 0.5).normalize() });
    }
  });
  // The hero, at the east tip, turns first.
  sfs.forEach(function (sf) { sf.d *= Math.min(1, Math.hypot(sf.x - 6.4, sf.z - 2) / 4); });
  var stems = new THREE.InstancedMesh(sunflowerStem(r), new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }), sfs.length);
  sfs.forEach(function (sf, n) {
    stems.setMatrixAt(n, m4.compose(v3.set(sf.x, 0, sf.z), q4.setFromAxisAngle(up, r() * 6.28), s3.set(1, sf.h, 1)));
  });
  stems.castShadow = !small;
  var heads = new THREE.InstancedMesh(sunflowerHead(), new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide,
                                                                                        emissive: '#ff9a10', emissiveIntensity: 0 }), sfs.length);
  heads.frustumCulled = false;
  heads.castShadow = !small;
  world.add(stems, heads);

  // ── The honeybee ──
  var bee = new THREE.Group(), beeBody = new THREE.Group();
  var dkMat = new THREE.MeshLambertMaterial({ color: '#2a1d12' });
  beeBody.add(new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14).rotateX(Math.PI / 2).scale(0.62, 0.6, 1),
                             new THREE.MeshLambertMaterial({ map: stripeTexture() })));
  var beeHead = new THREE.Mesh(new THREE.SphereGeometry(0.46, 14, 10).translate(0, 0.06, 1.08), dkMat);
  beeBody.add(beeHead);
  [-1, 1].forEach(function (sd) {
    beeBody.add(new THREE.Mesh(new THREE.SphereGeometry(0.17, 10, 8).translate(sd * 0.27, 0.16, 1.36),
      new THREE.MeshPhongMaterial({ color: '#0c0a08', specular: '#ffffff', shininess: 120 })));
    var ant = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.03, 0.62, 5).translate(0, 0.31, 0).rotateX(0.75).rotateZ(-sd * 0.35)
      .translate(sd * 0.14, 0.42, 1.25), dkMat);
    beeBody.add(ant);
    beeBody.add(new THREE.Mesh(new THREE.SphereGeometry(0.06, 6, 5).translate(sd * 0.14 + sd * 0.2, 0.42 + 0.42, 1.25 + 0.42), dkMat));
  });
  beeBody.add(new THREE.Mesh(new THREE.ConeGeometry(0.08, 0.32, 8).rotateX(-Math.PI / 2).translate(0, 0, -1.1), dkMat));
  var wingShape = new THREE.Shape();
  wingShape.absellipse(0.62, 0, 0.62, 0.3, 0, Math.PI * 2, false, 0);
  var wingGeo = new THREE.ShapeGeometry(wingShape, 12).rotateX(-Math.PI / 2).rotateY(0.35);
  var wingMat = new THREE.MeshBasicMaterial({ color: '#eef6ff', transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false });
  var wings = [-1, 1].map(function (sd) {
    var w = new THREE.Mesh(wingGeo, wingMat);
    w.position.set(sd * 0.12, 0.5, 0.25);
    w.scale.x = sd;
    beeBody.add(w);
    return w;
  });
  beeBody.scale.setScalar(0.05);
  bee.add(beeBody);
  bee.visible = false;
  world.add(bee);
  var beePath = new THREE.CatmullRomCurve3([
    new THREE.Vector3(10.6, 0.95, -1.4), new THREE.Vector3(9.6, 0.62, -2.6),
    nearestFlower(8.7, -3.2, new THREE.Vector3()).add(v3.set(0, 0.1, 0)), nearestFlower(8.7, -3.2, new THREE.Vector3()).add(v3.set(0, 0.055, 0)),
    nearestFlower(9.25, -3.5, new THREE.Vector3()).add(v3.set(0, 0.055, 0)),
    nearestFlower(8.95, -2.85, new THREE.Vector3()).add(v3.set(0, 0.055, 0)),
    new THREE.Vector3(8.2, 0.9, -1.6), new THREE.Vector3(7.0, 1.6, 0.8), new THREE.Vector3(6.2, 1.9, 2.0)
  ]);
  var beeAhead = new THREE.Vector3();

  // ── Light in the air: the sunbeam, its motes, the swirl, pollen ──
  var beamMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    uniforms: { uAmt: { value: 0 }, uTime: { value: 0 } },
    vertexShader: 'varying vec3 vN; varying vec3 vV; varying float vY; varying float vA;\n' +
      'void main(){ vN = normalize(normalMatrix * normal); vec4 mv = modelViewMatrix * vec4(position, 1.0); vV = normalize(-mv.xyz);\n' +
      ' vY = uv.y; vA = uv.x; gl_Position = projectionMatrix * mv; }',
    fragmentShader: 'uniform float uAmt; uniform float uTime; varying vec3 vN; varying vec3 vV; varying float vY; varying float vA;\n' +
      'void main(){ float core = pow(abs(dot(vN, vV)), 3.2);\n' +
      ' float ends = smoothstep(0.98, 0.72, vY) * smoothstep(0.0, 0.3, vY);\n' +
      ' float flick = 0.8 + 0.2 * sin(uTime * 0.9 + vA * 18.0);\n' +
      ' gl_FragColor = vec4(vec3(1.0, 0.84, 0.52) * core * ends * flick * uAmt * 0.3, 1.0); }'
  });
  // Shafts through gaps in the crown ([side, up, top radius, foot radius]),
  // each with a warm pool of light where it lands on the lawn.
  var beams = [], poolTex = softSprite('rgba(255,236,190,0.9)', 'rgba(255,220,160,0)');
  [[0, 0.1, 0.3, 0.5], [0.85, 0.35, 0.16, 0.3], [-0.75, 0.05, 0.14, 0.26], [0.35, -0.6, 0.12, 0.22], [-0.3, 0.75, 0.1, 0.2], [1.3, -0.25, 0.09, 0.17]]
    .forEach(function (b) {
      var mesh = new THREE.Mesh(new THREE.CylinderGeometry(b[2], b[3], 1, 20, 1, true).translate(0, -0.5, 0), beamMat);
      var pool = new THREE.Mesh(new THREE.CircleGeometry(1, 24).rotateX(-Math.PI / 2),
        new THREE.MeshBasicMaterial({ map: poolTex, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, opacity: 0 }));
      mesh.userData = { off: b, pool: pool };
      mesh.frustumCulled = false;
      world.add(mesh, pool);
      beams.push(mesh);
    });
  var beamDir = new THREE.Vector3(), beamSide = new THREE.Vector3(), beamUp = new THREE.Vector3();

  var NM = small ? 160 : 360, moteSeed = new Float32Array(NM * 4);
  for (i = 0; i < NM * 4; i++) moteSeed[i] = r();
  var moteGeo = new THREE.BufferGeometry();
  moteGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NM * 3), 3));
  moteGeo.setAttribute('aSeed', new THREE.BufferAttribute(moteSeed, 4));
  var moteMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uAmt: { value: 0 }, uTime: { value: 0 }, uFrom: { value: new THREE.Vector3() }, uDir: { value: new THREE.Vector3() },
                uSide: { value: new THREE.Vector3() }, uUp: { value: new THREE.Vector3() }, uScale: { value: 400 } },
    vertexShader: 'attribute vec4 aSeed; uniform float uTime; uniform vec3 uFrom; uniform vec3 uDir; uniform vec3 uSide; uniform vec3 uUp; uniform float uScale; uniform float uAmt; varying float vA;\n' +
      'void main(){ float t = fract(aSeed.x + uTime * 0.006 * (0.5 + aSeed.w));\n' +
      ' float ang = aSeed.y * 6.2832 + uTime * 0.15 * (aSeed.z - 0.5), rad = sqrt(aSeed.z) * (0.3 + t * 0.35);\n' +
      ' vec3 p = uFrom + uDir * (0.6 + t * 5.4) + (uSide * cos(ang) + uUp * sin(ang)) * rad;\n' +
      ' p += vec3(sin(uTime * 0.4 + aSeed.w * 9.0), sin(uTime * 0.3 + aSeed.x * 7.0), cos(uTime * 0.35 + aSeed.y * 8.0)) * 0.05;\n' +
      ' vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_Position = projectionMatrix * mv;\n' +
      ' gl_PointSize = (0.008 + aSeed.w * 0.01) * uScale / -mv.z;\n' +
      ' vA = uAmt * (0.5 + 0.5 * sin(uTime * (1.0 + aSeed.y * 2.0) + aSeed.x * 20.0)) * smoothstep(0.0, 0.15, t) * smoothstep(1.0, 0.8, t); }',
    fragmentShader: 'varying float vA; void main(){ vec2 q = gl_PointCoord - 0.5; float a = exp(-dot(q, q) * 22.0) * vA;\n' +
      ' gl_FragColor = vec4(vec3(1.0, 0.82, 0.5) * a * 0.8, 1.0); }'
  });
  var motes = new THREE.Points(moteGeo, moteMat);
  motes.frustumCulled = false;
  world.add(motes);

  var NS = small ? 260 : 600, swSeed = new Float32Array(NS * 4);
  for (i = 0; i < NS * 4; i++) swSeed[i] = r();
  var swGeo = new THREE.BufferGeometry();
  swGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NS * 3), 3));
  swGeo.setAttribute('aSeed', new THREE.BufferAttribute(swSeed, 4));
  var swMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false,
    uniforms: { uAmt: { value: 0 }, uTime: { value: 0 }, uCenter: { value: new THREE.Vector3() }, uScale: { value: 400 } },
    vertexShader: 'attribute vec4 aSeed; uniform float uTime; uniform float uAmt; uniform vec3 uCenter; uniform float uScale;\n' +
      'varying float vA; varying vec3 vC; varying float vRot; varying float vKind;\n' +
      'void main(){ float sp = 0.45 + aSeed.y * 0.6, a = aSeed.x * 6.2832 + uTime * sp * (0.5 + uAmt * 0.7);\n' +
      ' float h = fract(aSeed.z + uTime * 0.05 * (0.5 + aSeed.w));\n' +
      ' float rad = (0.5 + aSeed.y * 2.4) * (0.55 + 0.45 * uAmt) * (0.8 + h * 0.5);\n' +
      ' vec3 p = uCenter + vec3(cos(a) * rad, (h - 0.3) * 3.2 + sin(a * 2.0 + aSeed.w * 6.0) * 0.18, sin(a) * rad);\n' +
      ' vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_Position = projectionMatrix * mv;\n' +
      ' vKind = step(0.72, aSeed.w);\n' +
      ' gl_PointSize = mix(0.055, 0.028, vKind) * uScale / -mv.z;\n' +
      ' vA = uAmt * smoothstep(0.0, 0.12, h) * smoothstep(1.0, 0.75, h);\n' +
      ' vRot = a * 1.7 + aSeed.z * 6.0;\n' +
      ' vC = mix(mix(vec3(1.0, 0.72, 0.84), vec3(1.0, 0.97, 0.96), step(0.4, aSeed.w)), vec3(1.0, 0.86, 0.5), vKind); }',
    fragmentShader: 'varying float vA; varying vec3 vC; varying float vRot; varying float vKind;\n' +
      'void main(){ vec2 q = gl_PointCoord - 0.5; float c = cos(vRot), s = sin(vRot); q = mat2(c, -s, s, c) * q;\n' +
      ' float petal = 1.0 - smoothstep(0.38, 0.48, length(q * vec2(1.0, 1.9)));\n' +
      ' float mote = exp(-dot(q, q) * 26.0);\n' +
      ' float a = mix(petal * 0.95, mote, vKind) * vA; if (a < 0.01) discard;\n' +
      ' gl_FragColor = vec4(vC * (1.0 + vKind * 0.6), a); }'
  });
  var swirl = new THREE.Points(swGeo, swMat);
  swirl.frustumCulled = false;
  world.add(swirl);


  // ── Light shafts: a post pass, only while the sunbeam beat is on ──
  // Bright sky between the leaves is smeared out along the line from the
  // sun, so rays stream through the gaps in the crown.
  var rt = new THREE.WebGLRenderTarget(4, 4, { type: THREE.HalfFloatType, samples: small ? 0 : 4 });
  var postScene = new THREE.Scene(), postCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  var postMat = new THREE.ShaderMaterial({
    depthTest: false, depthWrite: false,
    uniforms: { tScene: { value: rt.texture }, uSun: { value: new THREE.Vector2(0.5, 0.8) }, uAmt: { value: 0 }, uAspect: { value: 1 } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
    fragmentShader: '#define N ' + (small ? 24 : 44) + '\n' +
      'uniform sampler2D tScene; uniform vec2 uSun; uniform float uAmt; uniform float uAspect; varying vec2 vUv;\n' +
      'void main(){ vec4 c = texture2D(tScene, vUv); vec2 d = (uSun - vUv) / float(N) * 0.85, p = vUv;\n' +
      ' vec3 acc = vec3(0.0); float decay = 1.0;\n' +
      ' for (int i = 0; i < N; i++) { p += d; vec3 s = texture2D(tScene, p).rgb;\n' +
      '  acc += s * smoothstep(0.75, 2.0, dot(s, vec3(0.3, 0.59, 0.11))) * decay; decay *= 0.965; }\n' +
      ' float fall = 1.0 - smoothstep(0.0, 1.1, length((vUv - uSun) * vec2(uAspect, 1.0)));\n' +
      ' c.rgb += acc / float(N) * vec3(1.0, 0.9, 0.7) * uAmt * 7.0 * fall;\n' +
      ' gl_FragColor = c;\n #include <tonemapping_fragment>\n #include <colorspace_fragment>\n }'
  });
  postScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), postMat));
  var bufSize = new THREE.Vector2(), sunScreen = new THREE.Vector3();

  // ── Frame ──
  var sunDir = new THREE.Vector3(), headDir = new THREE.Vector3(), faceDir = new THREE.Vector3(), Z = new THREE.Vector3(0, 0, 1);
  var fwd = new THREE.Vector3(), a3 = new THREE.Vector3(), b3 = new THREE.Vector3(), hTop = new THREE.Vector3();
  var portrait = false, H = 800, redP = [0, 0, 0], yellowP = [0, 0, 0];

  function placeBalloon(b, at, lift, time, k) {
    var free = smooth(0, 0.02, lift);
    b.pos.set(at[0], at[1], at[2]);
    b.pos.x += Math.sin(time * 0.7 + k * 2) * 0.05;
    b.pos.y += Math.sin(time * 0.9 + k) * 0.035;
    b.pos.z += Math.cos(time * 0.6 + k * 3) * 0.04;
    b.mesh.position.copy(b.pos);
    b.mesh.castShadow = b.mesh.userData.shadow && lift < 0.3;
    b.mesh.rotation.set(Math.sin(time * 0.8 + k) * 0.08 - (b.pos.z - at[2]) * 0.6, k * 1.3, (b.pos.x - at[0]) * -0.9);
    // The string: tied, a slack curve down to the bench arm; free, it trails below.
    var arr = b.line.geometry.attributes.position.array;
    a3.copy(b.pos).add(v3.set(0, -0.4, 0));
    b3.set(a3.x + Math.sin(time * 1.1 + k) * 0.12, a3.y - 1.3, a3.z + Math.cos(time * 0.9 + k) * 0.1);
    b3.lerpVectors(tie, b3, free);
    for (var j = 0; j < 16; j++) {
      var t = j / 15, sag = Math.sin(t * Math.PI) * (0.08 * (1 - free) + 0.05 * free);
      arr[j * 3] = lerp(a3.x, b3.x, t) + sag * 0.6 + Math.sin(t * 5 + time * 2 + k) * 0.012 * free;
      arr[j * 3 + 1] = lerp(a3.y, b3.y, t) - sag;
      arr[j * 3 + 2] = lerp(a3.z, b3.z, t) + sag * 0.3;
    }
    b.line.geometry.attributes.position.needsUpdate = true;
  }

  function frame(f) {
    var row = f.row, time = f.time, dt = f.dt;
    var wind = row[3], sunEl = row[6], beamAmt = row[7], appleAmt = row[8], glintAmt = row[9];
    var open = row[10], beeT = row[11], devote = row[12], swirlAmt = row[13], liftR = row[14], liftY = row[15];
    var swell = row[16], smile = row[17], slow = env.reduceMotion;

    // Camera: straight from the keys, a little breathing, and on a phone a
    // higher, offset view from the sky so the face clears the centred text.
    var high = smooth(6, 24, row[1]), topdown = smooth(-1.1, -1.5, row[5]);
    camera.position.set(row[0], row[1], row[2]);
    camera.position.y += Math.sin(time * 0.9) * (0.015 + high * 0.12);
    camera.position.x += Math.sin(time * 0.37) * high * 0.2;
    if (portrait) {
      camera.position.y += topdown * row[1] * 0.3;
      fwd.set(-Math.sin(row[4]), 0, -Math.cos(row[4]));
      camera.position.addScaledVector(fwd, -9 * topdown);
    }
    var look = 1 - topdown * 0.75;
    // A phone centres the text, so the keys carry a turn for it (phYaw/phPitch).
    var phYaw = portrait ? row[18] : 0, phPitch = portrait ? row[19] : 0;
    camera.rotation.set(row[5] + phPitch - f.my * 0.06 * look, row[4] + phYaw - f.mx * 0.13 * look, 0);
    sky.position.copy(camera.position);

    // Sun: east-north-east, climbing towards noon; it swells in stanza IV.
    sunDir.set(Math.sin(SUN_AZ) * Math.cos(sunEl), Math.sin(sunEl), -Math.cos(SUN_AZ) * Math.cos(sunEl));
    dome.uniforms.sunDir.value.copy(sunDir);
    dome.uniforms.sunColor.value.set('#fff2d4').multiplyScalar(0.22 + swell * 0.3);
    dome.uniforms.top.value.set('#1d5fca').lerp(tmp.set('#2a6ed2'), swell * 0.5);
    dome.uniforms.mid.value.set('#5a9be4').lerp(tmp.set('#80b6ec'), swell * 0.4);
    dome.uniforms.horizon.value.set('#cfe2f1').lerp(tmp.set('#f4e8d0'), swell * 0.5);
    world.fog.color.copy(dome.uniforms.horizon.value);
    gl.setClearColor(world.fog.color);
    sunDisc.position.copy(sunDir).multiplyScalar(1000);
    sunGlow.position.copy(sunDisc.position);
    sunRays.position.copy(sunDisc.position);
    sunDisc.scale.setScalar(30 + swell * 20);
    sunGlow.scale.setScalar(460 + swell * 700);
    sunRays.scale.setScalar(420 + swell * 500);
    sunRays.material.rotation = time * 0.012;
    sunRays.material.opacity = 0.22 + swell * 0.3;
    sunGlow.material.opacity = 0.85 + swell * 0.15;
    gl.toneMappingExposure = 1.05 + swell * 0.1;
    clouds.forEach(function (c) {
      c.userData.a += dt * 0.0015;
      c.position.x = Math.sin(c.userData.a) * Math.cos(c.userData.e) * 1100;
      c.position.z = -Math.cos(c.userData.a) * Math.cos(c.userData.e) * 1100;
      c.material.color.set('#ffffff').lerp(tmp.set('#fff1d8'), swell * 0.6);
    });
    sun.position.copy(sunDir).multiplyScalar(70);
    sun.intensity = (2.7 + swell * 0.4) * (1 - high * 0.15);
    hemi.intensity = 1.2 + swell * 0.15;

    U.uClock.value = slow ? time * 0.5 : time;
    U.uWind.value = wind;
    U.uOpen.value = open;

    // The bench things: steam off the cup, the glint, the glowing apple.
    for (var k = 0; k < steam.length; k++) {
      var sp = (time * 0.22 + steam[k].userData.ph) % 1;
      steam[k].position.set(cup.position.x + Math.sin(sp * 7 + k) * 0.02 * sp, cup.position.y + 0.1 + sp * 0.32, cup.position.z + Math.cos(sp * 5 + k) * 0.015);
      steam[k].scale.setScalar(0.04 + sp * 0.12);
      steam[k].material.opacity = Math.sin(sp * Math.PI) * 0.32;
    }
    var tw = 0.75 + 0.25 * Math.sin(time * 5.3) * Math.sin(time * 2.1);
    glint.material.opacity = glintAmt * tw;
    glint.scale.setScalar(0.1 + glintAmt * 0.16 * tw);
    glint.material.rotation = time * 0.3;
    heroBody.material.emissive.setRGB(0.2 + appleAmt * 0.45, appleAmt * 0.06, appleAmt * 0.04);
    heroGlow.material.opacity = appleAmt * (0.5 + 0.12 * Math.sin(time * 2.4));
    heroGlow.scale.setScalar(0.38 + appleAmt * 0.18);
    var et = smooth(0.75, 1, smile) * (0.65 + 0.35 * Math.sin(time * 3.1));
    eyeTwinkle.material.opacity = et;
    eyeTwinkle.scale.setScalar(1.6 + et * 2.6);
    eyeTwinkle.material.rotation = time * 0.2;

    // The sunbeam, from the right eye's crown along the light to the lawn.
    beamDir.copy(sunDir).negate();
    beamSide.crossVectors(beamDir, up).normalize();
    beamUp.crossVectors(beamSide, beamDir).normalize();
    beams.forEach(function (m) {
      var o = m.userData.off, pool = m.userData.pool, len = 0;
      m.position.copy(beamFrom).addScaledVector(beamSide, o[0]).addScaledVector(beamUp, o[1]);
      m.quaternion.setFromUnitVectors(up, sunDir);
      len = m.position.y / Math.max(sunDir.y, 0.1);
      m.scale.set(1, len, 1);
      pool.position.copy(m.position).addScaledVector(beamDir, len).setY(0.03);
      pool.rotation.y = Math.atan2(beamDir.x, beamDir.z);
      pool.scale.set(o[3] * 1.5, 1, o[3] * 1.5 / Math.max(sunDir.y, 0.2));
      pool.material.opacity = beamAmt * 0.55;
      m.visible = pool.visible = beamAmt > 0.01;
    });
    beamMat.uniforms.uAmt.value = beamAmt;
    beamMat.uniforms.uTime.value = time;
    moteMat.uniforms.uAmt.value = beamAmt;
    moteMat.uniforms.uTime.value = slow ? time * 0.5 : time;
    moteMat.uniforms.uFrom.value.copy(beamFrom);
    moteMat.uniforms.uDir.value.copy(beamDir);
    moteMat.uniforms.uSide.value.copy(beamSide);
    moteMat.uniforms.uUp.value.copy(beamUp);
    motes.visible = beamAmt > 0.01;

    // Sunflowers: bowed asleep, waking in a wave to face the sun, then all
    // turning up to look at you.
    faceDir.copy(sunDir);
    for (k = 0; k < sfs.length; k++) {
      var sf = sfs[k], dv = smooth(sf.d * 0.8, sf.d * 0.8 + 0.4, devote), sm = smooth(sf.d * 0.6, sf.d * 0.6 + 0.5, smile);
      headDir.copy(sf.droop).lerp(faceDir, dv).normalize().lerp(up, sm).normalize();
      headDir.x += Math.sin(time * 0.8 + sf.sway) * 0.03;
      headDir.z += Math.cos(time * 0.6 + sf.sway) * 0.03;
      headDir.normalize();
      q4.setFromUnitVectors(Z, headDir);
      hTop.set(sf.x, sf.h * 1.0, sf.z).addScaledVector(headDir, 0.07);
      heads.setMatrixAt(k, m4.compose(hTop, q4, s3.setScalar(sf.s * (1 + sm * 0.08))));
    }
    heads.instanceMatrix.needsUpdate = true;
    heads.material.emissiveIntensity = smile * 0.18;

    // The honeybee in stanza II.
    bee.visible = beeT > 0.002 && beeT < 0.998;
    if (bee.visible) {
      beePath.getPointAt(beeT, bee.position);
      beePath.getPointAt(Math.min(beeT + 0.02, 1), beeAhead);
      var hov = slow ? 0.4 : 1;
      bee.position.x += Math.sin(time * 7.3) * 0.012 * hov;
      bee.position.y += Math.sin(time * 9.1) * 0.01 * hov + 0.005;
      bee.position.z += Math.cos(time * 6.1) * 0.012 * hov;
      beeAhead.y = bee.position.y;
      if (beeAhead.distanceToSquared(bee.position) > 1e-6) bee.lookAt(beeAhead);
      bee.rotateZ(Math.sin(time * 3.3) * 0.15);
      var flap = Math.sin(time * (slow ? 20 : 58)) * 0.65;
      wings[0].rotation.z = -0.35 - flap;
      wings[1].rotation.z = 0.35 + flap;
    }

    // The balloons: tied to the bench arm, then up into the sky.
    placeBalloon(red, redAt(liftR, redP), liftR, time, 0);
    placeBalloon(yellow, yellowAt(liftY, yellowP), liftY, time, 1);
    // They nudge as the yellow one arrives.
    if (liftY > 0.9) {
      v3.subVectors(red.pos, yellow.pos);
      var gap = v3.length();
      if (gap < 0.62) { red.mesh.position.addScaledVector(v3.normalize(), (0.62 - gap) * 0.5); }
    }

    // Swirl of petals and light round you at "wonderful to me".
    fwd.set(0, 0, -1).applyQuaternion(camera.quaternion).setY(0).normalize();
    swMat.uniforms.uCenter.value.copy(camera.position).addScaledVector(fwd, 2.4).setY(camera.position.y - 0.6);
    swMat.uniforms.uAmt.value = swirlAmt;
    swMat.uniforms.uTime.value = slow ? time * 0.5 : time;
    swirl.visible = swirlAmt > 0.01;

    // Light shafts for the sunbeam, when the sun is in front of you.
    var rays = smooth(0.35, 1, beamAmt);
    sunScreen.copy(sunDir).multiplyScalar(500).add(camera.position).project(camera);
    if (rays < 0.01 || sunScreen.z > 1) {
      gl.setRenderTarget(null);
      gl.render(world, camera);
      return;
    }
    postMat.uniforms.uSun.value.set(sunScreen.x * 0.5 + 0.5, sunScreen.y * 0.5 + 0.5);
    postMat.uniforms.uAmt.value = rays;
    gl.getDrawingBufferSize(bufSize);
    postMat.uniforms.uAspect.value = bufSize.x / bufSize.y;
    if (rt.width !== bufSize.x || rt.height !== bufSize.y) rt.setSize(bufSize.x, bufSize.y);
    gl.setRenderTarget(rt);
    gl.render(world, camera);
    gl.setRenderTarget(null);
    gl.render(postScene, postCam);
  }

  return {
    resize: function (w, h, dpr) {
      fitCamera(gl, camera, w, h, dpr, small);
      portrait = w < h;
      H = h * Math.min(dpr, small ? 1.5 : 1.75);
      var scale = H / (2 * Math.tan(camera.fov * Math.PI / 360));
      moteMat.uniforms.uScale.value = scale;
      swMat.uniforms.uScale.value = scale;
    },
    frame: frame,
    destroy: function () { rt.dispose(); disposeAll(postScene); disposeAll(world, gl); }
  };
}

PI.register('sunbeam', {
  renderer: renderer3d,
  align: ['left', 'left', 'right', 'center', 'center'],
  scrim: 0.62,
  accent: '#ffd65a',
  emphasis: /^:\)$/,
  keys: function (T) {
    var last = T.count - 1, prevYaw = null;
    function at(i, u) { return T.start(Math.min(i, last)) + u; }
    // A row aiming the camera at a point (K), or with yaw and pitch given (A).
    function K(u, cam, target, rest) {
      var dx = target[0] - cam[0], dy = target[1] - cam[1], dz = target[2] - cam[2];
      return A(u, cam, Math.atan2(-dx, -dz), Math.atan2(dy, Math.hypot(dx, dz)), rest);
    }
    function A(u, cam, yaw, pitch, rest) {
      if (prevYaw != null) { while (yaw - prevYaw > Math.PI) yaw -= 2 * Math.PI; while (yaw - prevYaw < -Math.PI) yaw += 2 * Math.PI; }
      prevYaw = yaw;
      return [u, cam[0], cam[1], cam[2], rest[0], yaw, pitch].concat(rest.slice(1));
    }
    function off(p, dx, dy, dz) { return [p[0] + dx, p[1] + dy, p[2] + dz]; }
    var TINP = [TIN[0], 0.05, TIN[1]];
    // For the sunbeam: the right eye's crown with the sun just off to its right,
    // so the shafts slant down past you towards the bench.
    var BEAM = [-0.5, 1.35, 0.5];
    var SKY = off(redAt(1), 0, -1.4, -2.0);    // the pair sits up and right of the centred line
    var SIDE = -Math.PI / 2, DOWN = -Math.PI / 2;
    // wind sunEl beam apple glint open bee devo swirl liftR liftY swell smile phYaw phPitch
    var I = [0.15, 0.53, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    function W(base, set) { var o = base.slice(); Object.keys(set).forEach(function (k) { o[+k] = set[k]; }); return o; }
    var II = W(I, { 2: 0, 5: 1 }), III = W(II, { 0: 0.2, 6: 1, 7: 1 }), IV = W(III, { 9: 1, 10: 1 });
    //  (columns set per row: 1 sunEl, 2 beam, 3 apple, 4 glint, 5 open, 6 bee, 7 devote, 8 swirl,
    //   9 liftR, 10 liftY, 11 swell, 12 smile, 13 phYaw, 14 phPitch)
    return [
      K(0,             [-3.6, 1.65, 1.6],   [1.6, 2.1, -3.8],                 W(I, { 2: 0.25 })),
      K(0.7,           [-3.5, 1.62, 1.4],   [1.6, 2.0, -3.8],                 I),
      K(at(0, 0.3),    [-2.6, 1.45, 0.2],   [0.1, 0.7, -3.4],                 I),                                  // "my comfort and my pleasure"
      K(at(0, 0.52),   [0.4, 1.5, -1.3],    off(HERO, -0.25, 0.1, 0),         W(I, { 3: 1, 14: 0.2 })),            // "the apple of my eye"
      K(at(0, 0.74),   [-0.1, 1.2, -0.4],   TINP,                             W(I, { 3: 0.3, 4: 1, 14: 0.22 })),   // "precious as a treasure"
      K(at(0, 0.98),   BEAM,                [3.5, 2.7, -3.5],                 W(I, { 2: 1, 3: 0.2, 4: 0.2 })),     // "the sunbeam in the sky"
      K(at(1, 0.06),   [3.2, 1.4, -1.4],    [8.8, 0.6, -3.2],                 W(I, { 2: 0.4, 5: 0.1 })),
      K(at(1, 0.3),    [7.6, 1.0, -2.0],    [9.0, 0.3, -3.4],                 II),                                 // "the essence of emotion"
      K(at(1, 0.48),   [8.2, 0.7, -2.55],   [9.0, 0.36, -3.2],                W(II, { 6: 0.42 })),                 // "a wholesome honeybee"
      K(at(1, 0.6),    [8.25, 0.72, -2.5],  [9.05, 0.37, -3.15],              W(II, { 6: 0.72 })),
      K(at(1, 0.8),    [9.3, 1.5, 1.4],     [6.0, 1.45, 2.7],                 W(II, { 6: 1, 7: 1 })),              // "the spirit of devotion"
      K(at(1, 1.02),   [9.0, 1.6, 2.0],     [5.8, 2.0, 2.4],                  W(II, { 0: 0.25, 6: 1, 7: 1, 8: 1 })), // "and you're wonderful to me"
      K(at(2, 0.06),   [1.6, 1.5, -0.2],    off(redAt(0), 1.0, -0.3, 0),      W(III, { 1: 0.6, 8: 0.2, 13: 0.28, 14: -0.1 })),
      K(at(2, 0.3),    [1.4, 2.6, -0.6],    off(redAt(0.1), 1.3, -0.5, 0),    W(III, { 1: 0.64, 9: 0.1, 13: 0.3, 14: -0.2 })),    // "it's you I'll hold above me"
      K(at(2, 0.55),   [2.0, 10.5, -0.2],   off(redAt(0.35), 1.4, -0.5, 0),   W(III, { 1: 0.72, 9: 0.35, 13: 0.3, 14: -0.22 })),  // "you that I'll adore"
      K(at(2, 0.82),   [2.6, 19.6, 0.2],    off(redAt(0.65), 1.4, -0.5, 0),   W(III, { 1: 0.78, 9: 0.65, 13: 0.3, 14: -0.22 })),  // "you might just say you like me"
      K(at(3, 0.1),    [0.6, 27.0, -1.0],   off(redAt(0.9), 0, -1.6, -1.6),   W(III, { 1: 0.8, 9: 0.9, 10: 0.6, 11: 0.15, 13: -0.12 })),
      // The view stays on the sky for the whole line: the yellow balloon rises
      // into the frame below the red one, nudges it and settles higher.
      K(at(3, 0.42),   [0.0, 28.6, -2.8],   off(SKY, 0, -1.3, 0),             W(IV, { 1: 0.8, 10: 0.88, 11: 0.7, 13: -0.24 })),  // "but I think"
      K(at(3, 0.75),   [0.0, 28.8, -2.8],   SKY,                              W(IV, { 1: 0.8, 11: 1, 13: -0.24 })),               // "I like you more"
      K(at(3, 1.3),    [0.1, 29.0, -2.9],   SKY,                              W(IV, { 1: 0.82, 11: 0.9, 13: -0.24 })),
      A(at(4, 0.05),   [0.2, 29.6, -1.5],   -1.45, -0.75,                     W(IV, { 1: 0.95, 11: 0.3 })),
      A(at(4, 0.32),   [0, 30, 0.3],        SIDE, DOWN,                       W(IV, { 1: 1.1, 12: 0.2 })),
      A(at(4, 0.6),    [0, 30, 0.3],        SIDE, DOWN,                       W(IV, { 1: 1.2, 12: 1 })),                          // ":)"
      A(at(4, 1.2),    [0, 30.5, 0.3],      SIDE, DOWN,                       W(IV, { 1: 1.2, 12: 1 })),
      A(T.total - 0.35, [0, 34, 0.3],       0, DOWN,                          W(IV, { 1: 1.2, 12: 1 })),                          // and it turns upright
      A(T.total,       [0, 35, 0.3],        0, DOWN,                          W(IV, { 1: 1.2, 12: 1 }))
    ];
  },
  sound: {
    src: '/audio/birds.mp3',
    label: 'Play the garden birdsong',
    // Birdsong in the garden, fainter as you rise above it.
    volume: function (row) { return 0.16 - 0.08 * smooth(3, 25, row[1]); },
    cues: [
      { stanza: 1, at: 0.4, play: honeybee },
      { stanza: 2, at: 0.22, play: lift },
      { stanza: 3, at: 0.3, play: warmth },
      { stanza: 4, at: 0.55, play: smileChime }
    ]
  }
});
