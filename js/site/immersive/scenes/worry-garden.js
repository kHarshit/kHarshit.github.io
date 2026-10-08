/*
 * Scene for "I Worried" (Mary Oliver): a small hedged garden above a river,
 * from a grey dawn of worrying out into the morning.
 *
 * I   Grey dawn in the garden, everything muted under a low lid of cloud.
 *     "Will the garden grow": the flower beds, closed and colourless;
 *     "will the rivers flow": the river beyond the gate; "will the earth
 *     turn as it was taught": up into a slow, heavy sky that wheels over.
 * II  "Was I right, was I wrong": rain, and rings spreading in the puddles
 *     on the gravel path.
 * III "Even the sparrows can do it": a few sparrows on the top rail of the
 *     fence by the gate, hopping and chirping.
 * IV  "Is my eyesight fading": the garden blurs and greys and the fog
 *     closes in round you.
 * V   "Finally I saw that worrying had come to nothing. And gave it up":
 *     the fog lifts, colour seeps out from the brightening sun and the
 *     flowers open, and the sparrows burst up off the rail as the sun
 *     breaks through; "and went
 *     out into the morning, and sang": out through the gate and down the
 *     meadow to the bright river.
 *
 * The garden's colour, the blur and the haze are a post pass over the
 * whole frame, so worry and its lifting touch everything at once.
 *
 * Columns: [unit, path, gloom, rain, wind, yaw, pitch, fog, blur, colour,
 *           sun, open, flight, flood]
 * (colour is the grey world's saturation; flood spreads full colour out
 * from the sun across the frame)
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, broadleafGeometry, softSprite, terrain,
         scatter, particleField, rainField, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout (metres; you walk towards -z) ─────────────────────────────────
// The garden is flat, hedged at the sides (x = ±9.8) and fenced across at
// z = -12, with a gate on the path. Beyond it a meadow falls gently to a
// broad river running across the view at z ≈ -40; past that, fields and low hills.
var FENCE_Z = -12, GATE = 0.75, RAIL = 1.0;
function riverZ(x) { return -37.5 + 2.5 * Math.sin(x * 0.03) + 1.2 * Math.sin(x * 0.09 + 1); }
function land(x, z) {
  var d = Math.abs(z - riverZ(x)), out = smooth(-12.5, -16, z) + smooth(10.5, 16, Math.abs(x)) * (1 - smooth(-12.5, -16, z));
  return out * (0.22 * Math.sin(x * 0.13 + 1) * Math.cos(z * 0.11) + 0.12 * Math.sin(x * 0.31 + z * 0.23)) -
         smooth(-12.8, -21, z) * 1.6 - smooth(13.5, 9.5, d) * 1.3 +
         smooth(-58, -140, z) * (3 + 2 * Math.sin(x * 0.02)) +
         smooth(-160, -420, z) * (34 + 16 * Math.sin(x * 0.009 + 0.5) + 7 * Math.sin(x * 0.027)) +
         smooth(40, 220, Math.abs(x)) * 16 * (1 - smooth(-30, -60, z) * 0.4);
}
var WATER_Y = -2.1;

var curve = new THREE.CatmullRomCurve3([[0, 4.6], [0, 1], [0, -3], [0, -7], [0, -10], [0.05, -12.5], [0.5, -16], [1.4, -19.5],
                                        [2.4, -22.5], [3.0, -25]].map(function (p) { return new THREE.Vector3(p[0], 0, p[1]); }));
// Path progress where the walk reaches a given z.
var T_AT = (function () {
  var out = [], p = new THREE.Vector3();
  for (var i = 0; i <= 400; i++) { curve.getPointAt(i / 400, p); out.push(p.z); }
  return out;
})();
function tAt(z) {
  for (var i = 1; i < T_AT.length; i++) {
    if (T_AT[i] <= z) return (i - 1 + (T_AT[i - 1] - z) / (T_AT[i - 1] - T_AT[i])) / 400;
  }
  return 1;
}

// Raised beds: [x0, x1, z0, z1, kind] (kind 0 flowers, 1 vegetables).
var BEDS = [[1.0, 3.6, 5, -3.4, 0], [1.0, 3.6, -4.4, -11, 0], [-3.6, -1.0, 5, -3.4, 0], [-3.6, -1.0, -4.4, -11, 0],
            [4.7, 7.7, 5, -3.4, 1], [4.7, 7.7, -4.4, -11, 1], [-7.7, -4.7, 5, -3.4, 1]];
var PUDDLES = [[-0.1, -2.5, 0.62, 0.5], [0.3, -4.3, 0.42, 0.3], [-0.2, 1.9, 0.5, 0.34], [0.12, -8.0, 0.55, 0.36], [-0.25, -10.6, 0.36, 0.24]];
// Sparrows on the top rail: x along it (right of the gate mostly).
var PERCH = [1.35, 1.72, 2.3, 2.85, 3.5, -2.2, -3.1, 5.6, 6.3];
var SUN_DIR = new THREE.Vector3(0.55, 0.22, -0.8).normalize();

// Where the portrait tilt by the rail starts, peaks, eases and ends.
var T_RAIL = [tAt(-5), tAt(-8), tAt(-9.2), tAt(-10.6)];

function hash2(a, b) { var s = Math.sin(a * 127.1 + b * 311.7) * 43758.5453; return s - Math.floor(s); }

// ── Geometry with extra per-vertex data ──────────────────────────────────
// A flower part: colour, whether the instance colour tints it (petals) and
// how far it folds up into a bud when closed.
function part(geo, color, tint, fold) {
  geo = tinted(geo, color);
  var n = geo.attributes.position.count, t = new Float32Array(n), f = new Float32Array(n);
  t.fill(tint); f.fill(fold);
  geo.setAttribute('aTint', new THREE.BufferAttribute(t, 1));
  geo.setAttribute('aFold', new THREE.BufferAttribute(f, 1));
  return geo;
}
function mergeParts(geos) {
  var out = merge(geos.map(function (g) { return g.clone(); }));
  ['aTint', 'aFold'].forEach(function (name) {
    var total = 0, o = 0;
    geos.forEach(function (g) { total += g.attributes[name].count; });
    var arr = new Float32Array(total);
    geos.forEach(function (g) { arr.set(g.attributes[name].array, o); o += g.attributes[name].count; g.dispose(); });
    out.setAttribute(name, new THREE.BufferAttribute(arr, 1));
  });
  return out;
}

// A petal: a flat blade from the centre outwards along +x, tipped up by `cup`.
function petal(len, wid, cup, ang, h) {
  return new THREE.PlaneGeometry(len, wid, 2, 1).rotateX(-Math.PI / 2).translate(len / 2 + 0.006, 0, 0)
    .rotateZ(cup).rotateY(ang).translate(0, h, 0);
}

// Three garden flowers: a cosmos-like daisy, a poppy cup and a spire of
// bells (delphinium, foxglove). The head sits on the stem axis.
function daisyGeometry() {
  var H = 0.42, parts = [part(new THREE.CylinderGeometry(0.003, 0.0045, H, 3).translate(0, H / 2, 0), '#5c8636', 0, 0),
                         part(new THREE.ConeGeometry(0.02, 0.16, 3).scale(1, 1, 0.3).rotateZ(0.5).translate(0.05, 0.16, 0), '#5f8a34', 0, 0)];
  for (var i = 0; i < 11; i++) parts.push(part(petal(0.06, 0.024, 0.16, i / 11 * Math.PI * 2, H), '#ffffff', 1, 0.95));
  parts.push(part(new THREE.SphereGeometry(0.017, 8, 4).scale(1, 0.5, 1).translate(0, H + 0.006, 0), '#e8b828', 0, 0));
  return mergeParts(parts);
}
function poppyGeometry() {
  var H = 0.38, parts = [part(new THREE.CylinderGeometry(0.003, 0.004, H, 3).translate(0, H / 2, 0), '#628a3e', 0, 0)];
  for (var i = 0; i < 5; i++) parts.push(part(petal(0.056, 0.062, 0.62, i / 5 * Math.PI * 2 + 0.3, H), '#ffffff', 1, 0.6));
  parts.push(part(new THREE.SphereGeometry(0.014, 6, 4).translate(0, H + 0.012, 0), '#1c1c18', 0, 0));
  return mergeParts(parts);
}
function spireGeometry() {
  var H = 0.82, parts = [part(new THREE.CylinderGeometry(0.005, 0.008, H, 4).translate(0, H / 2, 0), '#557a32', 0, 0)];
  for (var k = 0; k < 4; k++) {
    parts.push(part(new THREE.ConeGeometry(0.035, 0.24, 3).scale(1, 1, 0.3).rotateZ(0.8).rotateY(k * 1.6).translate(0, 0.1, 0), '#557c32', 0, 0));
  }
  for (var i = 0; i < 16; i++) {
    var y = 0.4 + i * 0.026, a = i * 2.4, rad = 0.03 * (1 - i / 20);
    parts.push(part(new THREE.ConeGeometry(0.022 * (1 - i / 24), 0.036, 5).rotateZ(Math.PI + 0.9).rotateY(a)
      .translate(Math.cos(a) * rad, y, -Math.sin(a) * rad), '#ffffff', 1, 0));
  }
  return mergeParts(parts);
}

// Push a blob (centred on the origin) in or out by a smooth function of
// position, so shared corners move together and stay closed; normals point
// straight out, so it shades round rather than faceted.
function lumpy(geo, amp, seed) {
  var p = geo.attributes.position, v = new THREE.Vector3(), nor = new Float32Array(p.count * 3);
  for (var i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    var k = 1 + amp * (Math.sin(v.x * 3.1 + seed) * Math.cos(v.y * 2.7 - seed) + 0.5 * Math.sin(v.z * 4.3 + v.x * 1.3));
    p.setXYZ(i, v.x * k, v.y * k, v.z * k);
    v.normalize();
    nor[i * 3] = v.x; nor[i * 3 + 1] = v.y; nor[i * 3 + 2] = v.z;
  }
  geo.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  return geo;
}

// Foliage: dapple a material with world-space noise, so a blob reads as a
// mass of leaves (darker in its hollows) rather than a smooth shell.
var NOISE3 = 'float lh(vec3 p){ return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }\n' +
  'float ln(vec3 p){ vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);\n' +
  ' return mix(mix(mix(lh(i), lh(i + vec3(1.0, 0.0, 0.0)), f.x), mix(lh(i + vec3(0.0, 1.0, 0.0)), lh(i + vec3(1.0, 1.0, 0.0)), f.x), f.y),\n' +
  '            mix(mix(lh(i + vec3(0.0, 0.0, 1.0)), lh(i + vec3(1.0, 0.0, 1.0)), f.x), mix(lh(i + vec3(0.0, 1.0, 1.0)), lh(i + vec3(1.0, 1.0, 1.0)), f.x), f.y), f.z); }\n';
function leafy(mat, scale) {
  mat.onBeforeCompile = function (sh) {
    sh.vertexShader = 'varying vec3 vLeafP;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n#ifdef USE_INSTANCING\n vLeafP = (modelMatrix * instanceMatrix * vec4(transformed, 1.0)).xyz;\n' +
      '#else\n vLeafP = (modelMatrix * vec4(transformed, 1.0)).xyz;\n#endif');
    sh.fragmentShader = 'varying vec3 vLeafP;\n' + NOISE3 + sh.fragmentShader.replace('#include <color_fragment>',
      '#include <color_fragment>\n float lf = ln(vLeafP * ' + scale.toFixed(2) + ') * 0.55 + ln(vLeafP * ' + (scale * 2.9).toFixed(2) + ') * 0.3 + ln(vLeafP * ' + (scale * 7.3).toFixed(2) + ') * 0.15;\n' +
      ' diffuseColor.rgb *= 0.5 + 0.85 * smoothstep(0.2, 0.8, lf);');
  };
  mat.customProgramCacheKey = function () { return 'wg-leafy' + scale; };
  return mat;
}

// A tuft of thin blades, dark at the root and light at the tip.
function tuftGeometry(r) {
  var pos = [], nor = [], col = [], root = new THREE.Color('#2c4a1a'), tip = new THREE.Color('#9fc062');
  for (var i = 0; i < 7; i++) {
    var a = r() * 6.28, w = 0.012 + r() * 0.012, h = 0.22 + r() * 0.28, lean = 0.05 + r() * 0.14;
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

// A house sparrow about 15 cm long, facing +x, feet at the origin: brown
// streaked back, grey crown, chestnut nape, black bib, pale cheeks.
function sparrowGeometry() {
  return merge([
    tinted(new THREE.CylinderGeometry(0.0035, 0.0035, 0.03, 3).translate(0.004, 0.015, 0.011), '#8a6a58'),
    tinted(new THREE.CylinderGeometry(0.0035, 0.0035, 0.03, 3).translate(0.004, 0.015, -0.011), '#8a6a58'),
    tinted(new THREE.SphereGeometry(0.036, 10, 8).scale(1.35, 0.95, 1).translate(0, 0.056, 0), '#bdb2a0'),
    tinted(new THREE.SphereGeometry(0.036, 10, 8).scale(1.4, 0.8, 1.03).translate(-0.006, 0.066, 0), '#8a6440'),
    tinted(new THREE.SphereGeometry(0.03, 8, 6).scale(1.5, 0.6, 0.38).translate(-0.016, 0.068, 0.031), '#6a4528'),
    tinted(new THREE.SphereGeometry(0.03, 8, 6).scale(1.5, 0.6, 0.38).translate(-0.016, 0.068, -0.031), '#6a4528'),
    tinted(new THREE.SphereGeometry(0.012, 6, 4).scale(1.6, 0.4, 0.5).translate(-0.004, 0.06, 0.04), '#e4dccb'),
    tinted(new THREE.SphereGeometry(0.012, 6, 4).scale(1.6, 0.4, 0.5).translate(-0.004, 0.06, -0.04), '#e4dccb'),
    tinted(new THREE.SphereGeometry(0.025, 10, 8).translate(0.045, 0.098, 0), '#cfc7b6'),
    tinted(new THREE.SphereGeometry(0.024, 10, 8).scale(1.05, 0.7, 0.9).translate(0.041, 0.108, 0), '#7a7570'),
    tinted(new THREE.SphereGeometry(0.02, 8, 6).scale(0.8, 0.9, 1.15).translate(0.028, 0.1, 0), '#8c552c'),
    tinted(new THREE.SphereGeometry(0.017, 8, 6).scale(0.7, 1.3, 0.95).translate(0.058, 0.08, 0), '#1e1a18'),
    tinted(new THREE.ConeGeometry(0.008, 0.017, 5).rotateZ(-Math.PI / 2).translate(0.075, 0.097, 0), '#2a2420'),
    tinted(new THREE.SphereGeometry(0.0045, 5, 4).translate(0.06, 0.104, 0.016), '#050505'),
    tinted(new THREE.SphereGeometry(0.0045, 5, 4).translate(0.06, 0.104, -0.016), '#050505'),
    tinted(new THREE.BoxGeometry(0.055, 0.006, 0.028).rotateZ(0.32).translate(-0.072, 0.07, 0), '#4e3a2a')
  ]);
}

// A sparrow in flight, nose to -z: a plump body, short rounded wings (all
// vertices past |x| 0.02 flap in the shader) and a square tail.
function flyerGeometry() {
  var v = [], c = [], brown = new THREE.Color('#7a5638'), dark = new THREE.Color('#4e3826'), pale = new THREE.Color('#c4b8a4');
  function tri(a, b, d, col) { [a, b, d].forEach(function (p) { v.push(p[0], p[1], p[2]); c.push(col.r, col.g, col.b); }); }
  [1, -1].forEach(function (s) {
    var w = [[s * 0.018, 0, -0.03], [s * 0.06, 0.004, -0.042], [s * 0.11, 0.008, -0.03], [s * 0.135, 0.01, -0.004],
             [s * 0.12, 0.008, 0.022], [s * 0.07, 0.004, 0.034], [s * 0.018, 0, 0.022]];
    for (var i = 1; i < w.length - 1; i++) tri(w[0], w[i], w[i + 1], i < 3 ? brown : dark);
    tri([0, 0.004, 0.03], [s * 0.024, 0, 0.09], [0, 0, 0.095], dark);
  });
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(v, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(c, 3));
  geo.computeVertexNormals();
  return merge([geo,
    tinted(new THREE.SphereGeometry(0.03, 8, 6).scale(0.85, 0.8, 1.8).translate(0, 0, 0), '#8a6440'),
    tinted(new THREE.SphereGeometry(0.022, 8, 6).scale(0.9, 0.7, 1.4).translate(0, -0.008, 0.005), pale.getStyle()),
    tinted(new THREE.SphereGeometry(0.02, 8, 6).translate(0, 0.006, -0.055), '#7a7570'),
    tinted(new THREE.ConeGeometry(0.007, 0.016, 5).rotateX(-Math.PI / 2).translate(0, 0.004, -0.078), '#2a2420')
  ]);
}

// A tree: trunk and limbs (bark) and a crown of lumpy clumps; `o` sets the
// shape. Apples are dotted over the old apple tree's crown.
function treeGeometry(r, o) {
  var parts = [], Y = new THREE.Vector3(0, 1, 0), q = new THREE.Quaternion();
  var top = new THREE.Vector3(o.lean || 0, o.trunk, 0);
  var trunk = new THREE.CylinderGeometry(o.girth * 0.7, o.girth, o.trunk, 6).translate(0, o.trunk / 2, 0);
  trunk.applyQuaternion(q.setFromUnitVectors(Y, top.clone().normalize()));
  parts.push(tinted(trunk, o.bark));
  for (var k = 0; k < o.limbs; k++) {
    var a = k / o.limbs * Math.PI * 2 + r() * 0.9, tilt = o.spread + r() * 0.3;
    var dir = new THREE.Vector3(Math.cos(a) * Math.sin(tilt), Math.cos(tilt), Math.sin(a) * Math.sin(tilt)), len = o.reach * (0.8 + r() * 0.4);
    var g = new THREE.CylinderGeometry(o.girth * 0.25, o.girth * 0.5, len, 5).translate(0, len / 2, 0);
    g.applyQuaternion(q.setFromUnitVectors(Y, dir));
    parts.push(tinted(g.translate(top.x, top.y, top.z), o.bark));
  }
  for (k = 0; k < o.clumps; k++) {
    var ca = r() * Math.PI * 2, cd = k === 0 ? 0 : o.width * (0.45 + r() * 0.5), rad = o.clump * (0.8 + r() * 0.45);
    var cy = o.crownY + r() * o.crownH - cd * (o.droop || 0.2);
    var shade = new THREE.Color(o.leaf).multiplyScalar(0.8 + r() * 0.35);
    parts.push(tinted(lumpy(new THREE.IcosahedronGeometry(rad, 2), 0.14, k * 1.7 + o.trunk)
      .scale(1, o.squash || 0.8, 1).translate(top.x + Math.cos(ca) * cd, cy, Math.sin(ca) * cd), shade));
    for (var f = 0; f < (o.fruit || 0); f++) {
      var fa = r() * 6.28, fe = (r() - 0.3) * 1.4;
      parts.push(tinted(new THREE.SphereGeometry(0.055, 6, 4).translate(top.x + Math.cos(ca) * cd + Math.cos(fa) * Math.cos(fe) * rad * 0.92,
        cy + Math.sin(fe) * rad * 0.75, Math.sin(ca) * cd + Math.sin(fa) * Math.cos(fe) * rad * 0.92), r() < 0.7 ? '#b8321e' : '#c9a030'));
    }
  }
  return merge(parts);
}

function canvasTexture(w, h, paint) {
  var c = document.createElement('canvas');
  c.width = w; c.height = h;
  paint(c.getContext('2d'), w, h);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = 4;
  return t;
}

// Gravel, or a trodden earth track, with ragged grassy edges cut out.
function pathTexture(r, earth) {
  return canvasTexture(128, 512, function (x, w, h) {
    var i;
    x.fillStyle = earth ? '#6a5a44' : '#9a8e7a';
    x.fillRect(0, 0, w, h);
    for (i = 0; i < 4200; i++) {
      var v = Math.floor((earth ? 70 : 110) + r() * (earth ? 50 : 80)), s = 1 + r() * (earth ? 2 : 3);
      x.fillStyle = 'rgba(' + v + ',' + Math.floor(v * 0.94) + ',' + Math.floor(v * 0.84) + ',0.7)';
      x.beginPath(); x.ellipse(r() * w, r() * h, s, s * (0.6 + r() * 0.5), r() * 3, 0, Math.PI * 2); x.fill();
    }
    for (i = 0; i < (earth ? 60 : 14); i++) {
      x.fillStyle = r() < 0.5 ? 'rgba(78,104,46,0.75)' : 'rgba(96,120,56,0.55)';
      x.beginPath(); x.ellipse(r() * w, r() * h, 3 + r() * 9, 2 + r() * 6, r() * 3, 0, Math.PI * 2); x.fill();
    }
    x.globalCompositeOperation = 'destination-out';
    for (i = 0; i < 160; i++) {
      var side = r() < 0.5 ? 0 : w, rad = 4 + r() * (earth ? 20 : 9);
      x.beginPath(); x.ellipse(side + (side ? -1 : 1) * r() * (earth ? 20 : 6), r() * h, rad, rad * 1.6, 0, 0, Math.PI * 2); x.fill();
    }
    x.globalCompositeOperation = 'source-over';
  });
}

// A strip that follows the ground along `points`, uv = (across, along).
function stripGeometry(points, width, lift) {
  var pos = [], uv = [], idx = [], along = 0;
  for (var j = 0; j < points.length; j++) {
    var a = points[Math.max(j - 1, 0)], b = points[Math.min(j + 1, points.length - 1)], p = points[j];
    var dx = b.x - a.x, dz = b.z - a.z, len = Math.hypot(dx, dz) || 1, nx = -dz / len, nz = dx / len;
    if (j > 0) along += p.distanceTo(points[j - 1]);
    for (var s = 0; s <= 4; s++) {
      var o = (s / 4 - 0.5) * width, x = p.x + nx * o, z = p.z + nz * o;
      pos.push(x, land(x, z) + lift, z);
      uv.push(s / 4, along / 4);
    }
    if (j > 0) for (s = 0; s < 4; s++) { var k = (j - 1) * 5 + s; idx.push(k, k + 1, k + 5, k + 1, k + 6, k + 5); }
  }
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  return geo;
}

// Sun rays: thin soft wedges round a centre.
function raysTexture(r) {
  return canvasTexture(256, 256, function (x) {
    x.translate(128, 128);
    x.globalCompositeOperation = 'lighter';
    for (var i = 0; i < 40; i++) {
      x.rotate(Math.PI * 2 / 40 + r() * 0.1);
      var len = 60 + r() * 68, g = x.createLinearGradient(0, 0, len, 0);
      g.addColorStop(0, 'rgba(255,248,230,0.24)');
      g.addColorStop(1, 'rgba(255,248,230,0)');
      x.fillStyle = g;
      x.beginPath(); x.moveTo(0, 0); x.lineTo(len, -2 - r() * 4); x.lineTo(len, 2 + r() * 4); x.fill();
    }
  });
}

// A glow that falls off fast from a bright core.
function glowTexture() {
  return canvasTexture(256, 256, function (x) {
    var g = x.createRadialGradient(128, 128, 0, 128, 128, 128);
    [[0, 1], [0.05, 0.7], [0.14, 0.28], [0.35, 0.08], [0.7, 0.02], [1, 0]].forEach(function (s) {
      g.addColorStop(s[0], 'rgba(255,246,222,' + s[1] + ')');
    });
    x.fillStyle = g;
    x.fillRect(0, 0, 256, 256);
  });
}

// ── Shaders ──────────────────────────────────────────────────────────────
// Rings from raindrops (uRain sets how many) and, for the river, a current
// running along +x.
var WATER = [
  'float rh(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }',
  'vec2 rings(vec2 p){ vec2 g = vec2(0.0), b = floor(p);',
  ' for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {',
  '  vec2 c = b + vec2(float(i), float(j)); float k = rh(c);',
  '  if (k > uRain) continue;',
  '  vec2 o = c + 0.25 + 0.5 * vec2(rh(c + 1.7), rh(c + 4.3));',
  '  float t = fract(uTime * (0.6 + k * 0.5) + k * 7.0);',
  '  vec2 d = p - o; float r = length(d) + 0.0001; float x = (r - t * 1.2) * 9.0;',
  '  g += d / r * sin(x * 2.4) * exp(-x * x) * (1.0 - t) * (1.0 - t);',
  ' }',
  ' return g; }',
  'float cur(vec2 p){ return sin(p.x * 1.3 - uTime * 1.6 + sin(p.y * 1.9) * 1.4) * 0.5 + sin(p.x * 2.9 + p.y * 2.2 - uTime * 2.4) * 0.3 +',
  '  sin(p.x * 0.6 - p.y * 3.7 - uTime * 0.9) * 0.2; }'
].join('\n') + '\n';

// Water mirrors a simple sky (uSkyLow at the horizon, uSkyHigh overhead)
// along the reflected view ray, so rings and the current show as light.
function waterMaterial(color, U, flow, puddle) {
  var mat = new THREE.MeshPhongMaterial({ color: color, specular: '#a0a0a0', shininess: 160, transparent: !!puddle,
                                          polygonOffset: !!puddle, polygonOffsetFactor: -2, polygonOffsetUnits: -2, depthWrite: !puddle });
  mat.onBeforeCompile = function (sh) {
    sh.uniforms.uTime = U.uTime;
    sh.uniforms.uRain = U.uRain;
    sh.uniforms.uSkyLow = U.uSkyLow;
    sh.uniforms.uSkyHigh = U.uSkyHigh;
    sh.vertexShader = 'varying vec3 vWPos; varying vec2 vPuv;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n vWPos = (modelMatrix * vec4(position, 1.0)).xyz; vPuv = position.xz;');
    sh.fragmentShader = 'uniform float uTime; uniform float uRain; uniform vec3 uSkyLow; uniform vec3 uSkyHigh; varying vec3 vWPos; varying vec2 vPuv;\n' + WATER +
      sh.fragmentShader
      .replace('#include <alphamap_fragment>', '#include <alphamap_fragment>\n' +
        (puddle ? ' diffuseColor.a *= smoothstep(1.0, 0.72, length(vPuv));\n' : ''))
      .replace('#include <normal_fragment_maps>',
        '#include <normal_fragment_maps>\n vec2 rg = rings(vWPos.xz * ' + (puddle ? '3.2' : '1.6') + ') * ' + (puddle ? '1.0' : '0.6') + ';\n' +
        (flow ? ' vec2 fp = vWPos.xz * vec2(0.9, 1.4); float e = 0.05;\n' +
                ' rg += vec2(cur(fp + vec2(e, 0.0)) - cur(fp - vec2(e, 0.0)), cur(fp + vec2(0.0, e)) - cur(fp - vec2(0.0, e))) / (2.0 * e) * 0.07;\n' : '') +
        ' vec3 wn = normalize(vec3(rg.x * 0.6, 1.0, rg.y * 0.6));\n' +
        ' normal = normalize((viewMatrix * vec4(wn, 0.0)).xyz);\n' +
        ' vec3 wv = normalize(vWPos - cameraPosition), rv = reflect(wv, wn);\n' +
        ' float fres = 0.3 + 0.7 * pow(1.0 - max(-dot(wv, wn), 0.0), 4.0);\n' +
        ' totalEmissiveRadiance += mix(uSkyLow, uSkyHigh, smoothstep(0.0, 0.6, rv.y)) * fres;');
  };
  mat.customProgramCacheKey = function () { return 'wg-water' + !!flow + !!puddle; };
  return mat;
}

// A dome of slow cloud over a gradient sky. `uCover` runs from clear (0) to
// overcast (1); `uBreak` opens the cloud round the sun as it breaks through.
// Linear output: the post pass tone maps the whole frame.
function cloudDome(radius) {
  var u = {
    uTop: { value: new THREE.Color() }, uHorizon: { value: new THREE.Color() }, uCloud: { value: new THREE.Color() },
    uShade: { value: new THREE.Color() }, uSun: { value: new THREE.Color() }, uSunDir: { value: SUN_DIR.clone() },
    uCover: { value: 0.95 }, uBreak: { value: 0 }, uTime: { value: 0 }, uTurn: { value: 0 }
  };
  var mat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false, uniforms: u,
    vertexShader: 'varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: [
      'uniform vec3 uTop; uniform vec3 uHorizon; uniform vec3 uCloud; uniform vec3 uShade; uniform vec3 uSun; uniform vec3 uSunDir;',
      'uniform float uCover; uniform float uBreak; uniform float uTime; uniform float uTurn; varying vec3 vP;',
      'float hh(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }',
      'float nn(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);',
      ' return mix(mix(hh(i), hh(i + vec2(1.0, 0.0)), f.x), mix(hh(i + vec2(0.0, 1.0)), hh(i + vec2(1.0, 1.0)), f.x), f.y); }',
      'float fbm(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 5; i++) { s += a * nn(p); p = p * 2.07 + vec2(3.1, 1.7); a *= 0.5; } return s; }',
      'void main(){',
      ' vec3 d = normalize(vP); float h = max(d.y, 0.0);',
      ' vec3 c = mix(uHorizon, uTop, smoothstep(0.0, 0.5, h));',
      ' float s = max(dot(d, normalize(uSunDir)), 0.0);',
      ' c += uSun * (pow(s, 8.0) * 0.5 + pow(s, 60.0) * 1.2);',
      // The cloud deck wheels slowly round the zenith.
      ' float ca = cos(uTurn), sa = sin(uTurn); vec2 xz = mat2(ca, -sa, sa, ca) * d.xz;',
      ' vec2 p = xz / (h + 0.12) * 1.5 + vec2(uTime * 0.012, uTime * 0.005);',
      ' float f = fbm(p) * 0.65 + fbm(p * 2.3 + 7.0) * 0.35;',
      ' float cover = uCover - uBreak * 0.55 * smoothstep(0.82, 0.99, s);',
      ' float cov = smoothstep(1.0 - cover, 1.0 - cover + 0.32, f) * smoothstep(-0.02, 0.18, h);',
      ' vec3 cl = mix(uCloud, uShade, smoothstep(0.35, 0.9, f));',
      // Silver linings where the sun stands behind thin cloud.
      ' cl += uSun * pow(s, 6.0) * 0.9 * (1.0 - smoothstep(0.4, 0.8, f));',
      ' c = mix(c, cl, cov);',
      ' c = mix(c, uHorizon, 1.0 - smoothstep(0.0, 0.24, d.y));',
      ' gl_FragColor = vec4(c, 1.0);',
      '}'
    ].join('\n')
  });
  return { mesh: new THREE.Mesh(new THREE.SphereGeometry(radius, 32, 16), mat), uniforms: u };
}

// ── Synthesised sound cues ───────────────────────────────────────────────
function noiseBuffer(ac, sec) {
  var b = ac.createBuffer(1, Math.floor(ac.sampleRate * sec), ac.sampleRate), d = b.getChannelData(0);
  for (var i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  return b;
}

// Soft rain: a hiss that swells and fades, with a scatter of drips.
function rainPatter(ac, out) {
  var t = ac.currentTime, len = 9, src = ac.createBufferSource(), hp = ac.createBiquadFilter(), lp = ac.createBiquadFilter(), g = ac.createGain();
  src.buffer = noiseBuffer(ac, len);
  hp.type = 'highpass'; hp.frequency.value = 700;
  lp.type = 'lowpass'; lp.frequency.value = 5200;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.16, t + 2.5);
  g.gain.setValueAtTime(0.16, t + 5.5);
  g.gain.exponentialRampToValueAtTime(0.0001, t + len);
  src.connect(hp); hp.connect(lp); lp.connect(g); g.connect(out);
  src.start(t);
  for (var i = 0; i < 40; i++) {
    var dt = t + 0.8 + Math.random() * 7, o = ac.createOscillator(), dg = ac.createGain(), f = 900 + Math.random() * 1600;
    o.frequency.setValueAtTime(f, dt);
    o.frequency.exponentialRampToValueAtTime(f * 1.6, dt + 0.03);
    dg.gain.setValueAtTime(0.0001, dt);
    dg.gain.exponentialRampToValueAtTime(0.025, dt + 0.003);
    dg.gain.exponentialRampToValueAtTime(0.0001, dt + 0.05);
    o.connect(dg); dg.connect(out);
    o.start(dt); o.stop(dt + 0.06);
  }
}

// One sparrow's "cheep": a short bright note that dips and lifts.
function cheep(ac, out, t, f, vol) {
  var o = ac.createOscillator(), o2 = ac.createOscillator(), g = ac.createGain(), g2 = ac.createGain(), d = 0.07 + Math.random() * 0.05;
  o.type = 'sine';
  o.frequency.setValueAtTime(f, t);
  o.frequency.exponentialRampToValueAtTime(f * 0.78, t + d * 0.45);
  o.frequency.exponentialRampToValueAtTime(f * 0.95, t + d);
  o2.type = 'triangle';
  o2.frequency.setValueAtTime(f * 2, t);
  o2.frequency.exponentialRampToValueAtTime(f * 1.56, t + d * 0.45);
  o2.frequency.exponentialRampToValueAtTime(f * 1.9, t + d);
  g2.gain.value = 0.3;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(vol, t + 0.008);
  g.gain.exponentialRampToValueAtTime(0.0001, t + d);
  o.connect(g); o2.connect(g2); g2.connect(g); g.connect(out);
  o.start(t); o2.start(t); o.stop(t + d + 0.02); o2.stop(t + d + 0.02);
}

// A few sparrows chattering on the rail.
function sparrows(ac, out) {
  var t = ac.currentTime;
  for (var i = 0; i < 14; i++) {
    t += 0.09 + Math.random() * (i % 4 === 3 ? 0.5 : 0.14);
    cheep(ac, out, t, 3300 + Math.random() * 1100, 0.045);
  }
}

// Wings: a flurry of soft whirrs, then excited chirping as they go.
function flight(ac, out) {
  var t = ac.currentTime;
  for (var k = 0; k < 9; k++) {
    var s = t + k * 0.06 + Math.random() * 0.08, src = ac.createBufferSource(), bp = ac.createBiquadFilter(), g = ac.createGain();
    src.buffer = noiseBuffer(ac, 0.6);
    bp.type = 'bandpass'; bp.frequency.value = 900 + Math.random() * 900; bp.Q.value = 1.2;
    g.gain.setValueAtTime(0.0001, s);
    for (var b = 0; b < 8; b++) {
      g.gain.linearRampToValueAtTime(0.09 * (1 - b / 9), s + b * 0.055 + 0.02);
      g.gain.linearRampToValueAtTime(0.004, s + b * 0.055 + 0.05);
    }
    src.connect(bp); bp.connect(g); g.connect(out);
    src.start(s); src.stop(s + 0.6);
  }
  for (var i = 0; i < 18; i++) cheep(ac, out, t + 0.3 + Math.random() * 1.8, 3400 + Math.random() * 1400, 0.035);
}

// "And sang": a bright chorus of cheeps and a rising run of whistles.
function chorus(ac, out) {
  var t = ac.currentTime;
  for (var i = 0; i < 34; i++) cheep(ac, out, t + Math.random() * 3.2, 3200 + Math.random() * 1600, 0.03 + Math.random() * 0.02);
  [[2600, 3600], [3000, 4200], [3400, 3900], [2800, 4400], [3900, 4600]].forEach(function (n, k) {
    var s = t + 0.6 + k * 0.32, o = ac.createOscillator(), g = ac.createGain();
    o.type = 'sine';
    o.frequency.setValueAtTime(n[0], s);
    o.frequency.exponentialRampToValueAtTime(n[1], s + 0.18);
    g.gain.setValueAtTime(0.0001, s);
    g.gain.exponentialRampToValueAtTime(0.05, s + 0.03);
    g.gain.exponentialRampToValueAtTime(0.0001, s + 0.24);
    o.connect(g); g.connect(out);
    o.start(s); o.stop(s + 0.26);
  });
}

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(57), i, k;
  var gl = makeRenderer(canvas, { shadows: !small, clear: '#aab1b4' });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#aab1b4', 0.02);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.08, 3000);
  var up = new THREE.Vector3(0, 1, 0), tmp = new THREE.Color();
  var U = { uTime: { value: 0 }, uRain: { value: 0 }, uSkyLow: { value: new THREE.Color() }, uSkyHigh: { value: new THREE.Color() }, uWind: { value: 0.2 }, uOpen: { value: 0 } };

  // ── Sky and sun ──
  var sky = new THREE.Group();
  world.add(sky);
  var dome = cloudDome(1500);
  sky.add(dome.mesh);
  var glowMat = new THREE.SpriteMaterial({ map: glowTexture(), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false });
  var raysMat = new THREE.SpriteMaterial({ map: raysTexture(r), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false });
  var discMat = new THREE.SpriteMaterial({ map: softSprite('rgba(255,255,250,1)', 'rgba(255,250,230,0)'), blending: THREE.AdditiveBlending,
                                           depthWrite: false, transparent: true, fog: false });
  var sunGlow = new THREE.Sprite(glowMat), sunRays = new THREE.Sprite(raysMat), sunDisc = new THREE.Sprite(discMat);
  [sunGlow, sunRays, sunDisc].forEach(function (s) { s.position.copy(SUN_DIR).multiplyScalar(1000); s.renderOrder = 1; sky.add(s); });
  sunGlow.scale.setScalar(700);
  sunRays.scale.setScalar(900);
  sunDisc.scale.setScalar(34);

  var hemi = new THREE.HemisphereLight('#c4ccd2', '#4a5040', 1.6);
  var sun = new THREE.DirectionalLight('#ffe2b8', 0.3);
  sun.castShadow = !small;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = sun.shadow.camera.bottom = -24;
  sun.shadow.camera.right = sun.shadow.camera.top = 24;
  sun.shadow.camera.far = 160;
  sun.shadow.bias = -0.0006;
  sun.shadow.normalBias = 0.04;
  world.add(hemi, sun, sun.target);

  // ── Land ──
  var lawn = ['#4f6c32', '#56743a', '#4a6830'].map(function (c) { return new THREE.Color(c); });
  var fields = ['#5f8a3a', '#6e9641', '#7e9c48', '#9aa458', '#58803a', '#86a24c', '#a8a464'].map(function (c) { return new THREE.Color(c); });
  var mud = new THREE.Color('#4e4636');
  var ground = terrain(900, small ? 260 : 420, 0, -280, land, new THREE.MeshLambertMaterial({ vertexColors: true }), function (x, z, y) {
    var near = lawn[Math.floor(hash2(Math.floor(x / 5), Math.floor(z / 5)) * 3)];
    var fu = Math.floor((x * 0.85 + z * 0.35) / 46), fv = Math.floor((z * 0.9 - x * 0.3) / 34);
    tmp.copy(near).lerp(fields[Math.floor(hash2(fu, fv) * fields.length)], smooth(-55, -110, z));
    return tmp.lerp(mud, smooth(WATER_Y + 0.7, WATER_Y + 0.1, y));
  });
  world.add(ground);

  // The river, and puddles on the path.
  var river = new THREE.Mesh(new THREE.PlaneGeometry(900, 80, 1, 1).rotateX(-Math.PI / 2), waterMaterial('#2a3c46', U, true, false));
  river.position.set(0, WATER_Y, -38);
  world.add(river);
  var puddleMat = waterMaterial('#1a1e1e', U, false, true);
  PUDDLES.forEach(function (p, n) {
    // A wobbly oval: radius 1 in local xz before scaling (the alpha edge).
    var g = new THREE.CircleGeometry(1, 20).rotateX(-Math.PI / 2), pp = g.attributes.position;
    for (var j = 1; j < pp.count; j++) {
      var a = Math.atan2(pp.getZ(j), pp.getX(j)), w = 1 + 0.16 * Math.sin(a * 3 + n) + 0.08 * Math.sin(a * 5 + n * 2);
      pp.setXYZ(j, pp.getX(j) * w, 0, pp.getZ(j) * w);
    }
    var m = new THREE.Mesh(g, puddleMat);
    m.position.set(p[0], land(p[0], p[1]) + 0.045, p[1]);
    m.scale.set(p[2], 1, p[3]);
    m.rotation.y = n * 0.7;
    m.receiveShadow = true;
    world.add(m);
  });

  // The gravel path in the garden, then a trodden track to the river.
  var pts = curve.getSpacedPoints(80).filter(function (p) { return p.z <= 5.5; });
  var garden = pts.filter(function (p) { return p.z >= FENCE_Z - 0.6; }), track = pts.filter(function (p) { return p.z <= FENCE_Z + 0.2; });
  var gravelMat = new THREE.MeshLambertMaterial({ map: pathTexture(rng(5), false), alphaTest: 0.5 });
  gravelMat.map.repeat.set(1, 1);
  var gravel = new THREE.Mesh(stripGeometry(garden, 1.45, 0.03), gravelMat);
  var trackMesh = new THREE.Mesh(stripGeometry(track, 1.1, 0.05), new THREE.MeshLambertMaterial({ map: pathTexture(rng(6), true), alphaTest: 0.5 }));
  gravel.receiveShadow = trackMesh.receiveShadow = true;
  world.add(gravel, trackMesh);

  // ── Raised beds, the fence and gate, a watering can ──
  var wood = [], soil = [];
  BEDS.forEach(function (b, n) {
    var w = b[1] - b[0], d = b[2] - b[3], cx = (b[0] + b[1]) / 2, cz = (b[2] + b[3]) / 2, plank = n % 2 ? '#6e5e4c' : '#665646';
    wood.push(tinted(new THREE.BoxGeometry(w + 0.1, 0.24, 0.05).translate(cx, 0.12, b[2]), plank),
              tinted(new THREE.BoxGeometry(w + 0.1, 0.24, 0.05).translate(cx, 0.12, b[3]), plank),
              tinted(new THREE.BoxGeometry(0.05, 0.24, d).translate(b[0], 0.12, cz), plank),
              tinted(new THREE.BoxGeometry(0.05, 0.24, d).translate(b[1], 0.12, cz), plank));
    var top = new THREE.PlaneGeometry(w, d, Math.ceil(w * 4), Math.ceil(d * 4)).rotateX(-Math.PI / 2), tp = top.attributes.position;
    for (var j = 0; j < tp.count; j++) {
      var x = tp.getX(j), z = tp.getZ(j);
      tp.setY(j, 0.19 + 0.025 * Math.sin(x * 9 + n) * Math.cos(z * 7) + 0.02 * Math.sin(z * 2.3 + x));
    }
    soil.push(tinted(top.translate(cx, 0, cz), '#3a2a20'));
  });
  var bedWood = new THREE.Mesh(merge(wood), new THREE.MeshLambertMaterial({ vertexColors: true }));
  var bedSoil = new THREE.Mesh(merge(soil), new THREE.MeshLambertMaterial({ vertexColors: true }));
  bedWood.castShadow = bedWood.receiveShadow = bedSoil.receiveShadow = true;
  world.add(bedWood, bedSoil);

  // Post-and-rail fence, weathered grey, with the gate swung open.
  var fence = [], greyWood = '#837a6c';
  function railY(x) { return land(x, FENCE_Z) + RAIL; }
  var posts = [-GATE, GATE];
  for (var px = 2.6; px <= 14.5; px += 1.9) posts.push(px, -px);
  posts.forEach(function (x) {
    var hgt = Math.abs(x) === GATE ? 1.3 : 1.12;
    fence.push(tinted(new THREE.BoxGeometry(0.11, hgt, 0.11).translate(x, land(x, FENCE_Z) + hgt / 2 - 0.05, FENCE_Z), Math.abs(x) === GATE ? '#6e6658' : greyWood));
  });
  [1, -1].forEach(function (s) {
    [0.5, RAIL].forEach(function (y) {
      fence.push(tinted(new THREE.BoxGeometry(14.5 - GATE, 0.07, 0.05).translate(s * (GATE + 14.5) / 2, land(0, FENCE_Z) + y, FENCE_Z + 0.07), greyWood));
    });
  });
  var gateParts = [];
  [0.06, 1.4].forEach(function (x) { gateParts.push(tinted(new THREE.BoxGeometry(0.07, 1.05, 0.05).translate(x, 0.6, 0), '#8a8274')); });
  [0.2, 0.62, 1.04].forEach(function (y) { gateParts.push(tinted(new THREE.BoxGeometry(1.4, 0.07, 0.04).translate(0.73, y, 0), '#8a8274')); });
  gateParts.push(tinted(new THREE.BoxGeometry(1.5, 0.06, 0.035).rotateZ(Math.atan2(0.84, 1.34)).translate(0.73, 0.62, 0.01), '#7e7668'));
  var gateGeo = merge(gateParts).rotateY(1.95).translate(-GATE + 0.05, land(0, FENCE_Z), FENCE_Z - 0.05);
  fence.push(gateGeo);
  // A watering can left at the edge of the path.
  var canM = new THREE.Matrix4().compose(new THREE.Vector3(0.98, 0.02, 2.6), new THREE.Quaternion().setFromAxisAngle(up, 2.3), new THREE.Vector3(1, 1, 1));
  fence.push(merge([
    tinted(new THREE.CylinderGeometry(0.13, 0.14, 0.26, 12).translate(0, 0.13, 0), '#5c7a78'),
    tinted(new THREE.CylinderGeometry(0.012, 0.022, 0.42, 5).rotateZ(-0.95).translate(0.27, 0.24, 0), '#56716f'),
    tinted(new THREE.CylinderGeometry(0.035, 0.012, 0.05, 6).rotateZ(-0.95).translate(0.45, 0.37, 0), '#4e6866'),
    tinted(new THREE.TorusGeometry(0.11, 0.012, 4, 10, Math.PI).translate(0, 0.27, 0), '#56716f')
  ]).applyMatrix4(canM));
  var fenceMesh = new THREE.Mesh(merge(fence), new THREE.MeshLambertMaterial({ vertexColors: true }));
  fenceMesh.castShadow = fenceMesh.receiveShadow = true;
  world.add(fenceMesh);

  // ── Plants ──
  // Flowers open and sway in the vertex shader; petals take the instance
  // colour, stems and hearts keep their own.
  var flowerMat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
  flowerMat.onBeforeCompile = function (sh) {
    sh.uniforms.uTime = U.uTime; sh.uniforms.uWind = U.uWind; sh.uniforms.uOpen = U.uOpen;
    sh.vertexShader = 'uniform float uTime; uniform float uWind; uniform float uOpen; attribute float aTint; attribute float aFold;\n' +
      sh.vertexShader.replace('#include <color_vertex>',
        'vColor = vec3(1.0); vColor *= color; vColor *= mix(vec3(1.0), instanceColor.xyz, aTint);')
      .replace('#include <begin_vertex>',
        '#include <begin_vertex>\n float ph = instanceMatrix[3][0] * 1.7 + instanceMatrix[3][2] * 1.3;\n' +
        ' float own = fract(sin(ph) * 437.5);\n' +
        ' float k = mix(0.22, 1.0, clamp(uOpen * 1.4 - own * 0.4, 0.0, 1.0)), rr = length(transformed.xz);\n' +
        ' transformed.xz *= mix(1.0, k, aTint);\n' +
        ' transformed.y += aTint * aFold * rr * (1.0 - k) * 1.1;\n' +
        ' float bend = (sin(uTime * (1.3 + uWind * 2.0) + ph) * 0.6 + 0.4 * sin(uTime * 2.9 + ph * 1.9)) * (0.03 + uWind * 0.12);\n' +
        ' transformed.x += bend * transformed.y * transformed.y; transformed.z += bend * 0.5 * transformed.y * transformed.y;');
  };
  flowerMat.customProgramCacheKey = function () { return 'wg-flower'; };
  var palettes = [
    ['#f4f1ea', '#f29ab8', '#d8487c', '#f6d24a', '#ffffff', '#e86aa0'],   // daisies: white, pinks, butter yellow
    ['#e2321e', '#ef5a1c', '#d42a22', '#f08a2a'],                         // poppies: scarlet, orange
    ['#5a62d8', '#7c58c8', '#3e5ad0', '#c070c8', '#f0eef6']               // spires: blues, violet, white
  ];
  // Wild flowers in the meadow past the gate: ox-eye daisies, field poppies
  // and a few spires of purple loosestrife.
  var wild = [['#fbfaf2', '#fbfaf2', '#f6e27a'], ['#e0281a', '#d8261c', '#e8401e'], ['#b24ab8', '#9a48c0']];
  var GARDEN_N = small ? [600, 380, 200] : [1300, 800, 460], MEADOW_N = small ? [700, 650, 60] : [1700, 1500, 140];
  var flowerBeds = BEDS.filter(function (b) { return !b[4]; });
  [daisyGeometry(), poppyGeometry(), spireGeometry()].forEach(function (geo, kind) {
    var mesh = new THREE.InstancedMesh(geo, flowerMat, GARDEN_N[kind] + MEADOW_N[kind]);
    mesh.castShadow = kind === 2;
    scatter(mesh, 80000, function (n, p, q, s, c) {
      var x, z, sc = 0.8 + r() * 0.45;
      if (n < GARDEN_N[kind]) {
        var b = flowerBeds[Math.floor(r() * flowerBeds.length)], u = r();
        // Spires stand at the back of each bed, away from the path.
        if (kind === 2) u = 0.5 + u * 0.46; else if (kind === 0 && r() < 0.5) u = u * 0.7;
        x = lerp(b[0] + 0.1, b[1] - 0.1, b[0] > 0 ? u : 1 - u);
        z = lerp(b[3] + 0.12, b[2] - 0.12, r());
        if (kind === 1 && hash2(Math.floor(x * 1.3), Math.floor(z * 0.8)) < 0.45) return false;    // poppies in drifts
        p.set(x, 0.19, z);
        c.set(palettes[kind][Math.floor(r() * palettes[kind].length)]);
      } else {
        x = (r() - 0.5) * (r() < 0.6 ? 26 : 60);
        z = FENCE_Z - 0.8 - r() * 17;
        var y = land(x, z);
        if (y < WATER_Y + 0.2 || Math.abs(x - curve.getPointAt(tAt(z)).x) < 0.6) return false;
        if (hash2(Math.floor(x / 3 + kind * 7), Math.floor(z / 3)) < (kind === 1 ? 0.5 : 0.3)) return false;   // in drifts
        p.set(x, y - 0.02, z);
        sc *= 1.15;
        c.set(wild[kind][Math.floor(r() * wild[kind].length)]);
      }
      q.setFromAxisAngle(up, r() * 6.28);
      s.set(sc, sc * (0.85 + r() * 0.35), sc);
    });
    world.add(mesh);
  });

  // Leafy clumps under the flowers, and lettuces and cabbages in the
  // vegetable beds.
  var leafMat = leafy(new THREE.MeshLambertMaterial(), 9);
  var clumps = new THREE.InstancedMesh(lumpy(new THREE.IcosahedronGeometry(0.24, 1), 0.2, 3).scale(1, 0.85, 1), leafMat, small ? 1400 : 2800);
  scatter(clumps, 20000, function (n, p, q, s, c) {
    var b = BEDS[Math.floor(r() * BEDS.length)], veg = b[4];
    var x = lerp(b[0] + 0.2, b[1] - 0.2, r()), z = lerp(b[3] + 0.2, b[2] - 0.2, r());
    if (veg) { x = lerp(b[0] + 0.45, b[1] - 0.45, Math.round(r() * 3) / 3); z = Math.round(z / 0.55) * 0.55; }
    if (veg && hash2(Math.round(x * 3), Math.round(z)) < 0.15) return false;
    p.set(x, 0.17, z);
    q.setFromAxisAngle(up, r() * 6.28);
    var sc = veg ? 1.1 + r() * 0.5 : 0.9 + r() * 0.7;
    s.set(sc, sc * (veg ? 1.1 : 0.9), sc);
    if (veg && r() < 0.5) c.setHSL(0.36 + r() * 0.06, 0.22, 0.38 + r() * 0.08);         // blue-green cabbages
    else c.setHSL(0.25 + r() * 0.06, 0.45, (veg ? 0.3 : 0.2) + r() * 0.08);
  });
  clumps.castShadow = clumps.receiveShadow = true;
  world.add(clumps);

  // Bean poles: wigwams of canes with leaves climbing them.
  var canes = [], beans = [];
  [[6.2, -1], [6.2, -7.5], [-6.2, 1.5]].forEach(function (w) {
    for (var c = 0; c < 6; c++) {
      var a = c / 6 * Math.PI * 2, bx = w[0] + Math.cos(a) * 0.5, bz = w[1] + Math.sin(a) * 0.5;
      var base = new THREE.Vector3(bx, 0.15, bz), apex = new THREE.Vector3(w[0], 2.15, w[1]), dir = apex.clone().sub(base);
      var g = new THREE.CylinderGeometry(0.012, 0.016, dir.length() + 0.2, 4).translate(0, dir.length() / 2, 0);
      g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(up, dir.clone().normalize()));
      canes.push(tinted(g.translate(bx, 0.15, bz), '#8a7a52'));
      for (var l = 0; l < 7; l++) {
        var t = 0.08 + l / 7 * 0.8, lp = base.clone().lerp(apex, t);
        beans.push(tinted(lumpy(new THREE.IcosahedronGeometry(0.13 + r() * 0.06, 0), 0.2, l + c)
          .translate(lp.x + (r() - 0.5) * 0.12, lp.y, lp.z + (r() - 0.5) * 0.12), new THREE.Color('#ffffff').multiplyScalar(0.8 + r() * 0.3)));
      }
    }
  });
  var caneMesh = new THREE.Mesh(merge(canes), new THREE.MeshLambertMaterial({ vertexColors: true }));
  var beanMat = leafy(new THREE.MeshLambertMaterial({ vertexColors: true, color: '#4f7a2c' }), 8);
  var beanMesh = new THREE.Mesh(merge(beans), beanMat);
  caneMesh.castShadow = beanMesh.castShadow = true;
  world.add(caneMesh, beanMesh);

  // Grass: the lawn round the beds and the meadow down to the river.
  var grassMat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
  grassMat.onBeforeCompile = function (sh) {
    sh.uniforms.uTime = U.uTime; sh.uniforms.uWind = U.uWind;
    sh.vertexShader = 'uniform float uTime; uniform float uWind;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n float gph = instanceMatrix[3][0] * 0.4 + instanceMatrix[3][2] * 0.3;\n' +
      ' float gb = (sin(uTime * (1.4 + uWind * 3.0) + gph) * 0.6 + 0.4 + uWind * 0.6) * (0.05 + uWind * 0.25) * position.y * position.y;\n' +
      ' transformed.x += gb; transformed.z += gb * 0.3;');
  };
  grassMat.customProgramCacheKey = function () { return 'wg-grass'; };
  function inBed(x, z, m) {
    for (var j = 0; j < BEDS.length; j++) {
      var b = BEDS[j];
      if (x > Math.min(b[0], b[1]) - m && x < Math.max(b[0], b[1]) + m && z < b[2] + m && z > b[3] - m) return true;
    }
    return false;
  }
  var grass = new THREE.InstancedMesh(tuftGeometry(rng(3)), grassMat, small ? 11000 : 32000);
  scatter(grass, 200000, function (n, p, q, s, c) {
    var x = (r() - 0.5) * (r() < 0.55 ? 22 : 64), z = 6 - r() * 33;
    var y = land(x, z);
    if (inBed(x, z, 0.03) || y < WATER_Y + 0.15) return false;
    if (z > FENCE_Z && Math.abs(x) < 0.78) return false;                       // the gravel path
    if (z < FENCE_Z && Math.abs(x - curve.getPointAt(tAt(z)).x) < 0.45 && r() < 0.8) return false;
    p.set(x, y, z);
    q.setFromAxisAngle(up, r() * 6.28);
    var tall = z < FENCE_Z - 3 ? 0.9 + r() * 0.6 : 0.35 + r() * 0.3;           // the lawn and the strip below the fence are mown
    s.set(z < FENCE_Z ? 0.85 : 1.2, tall, z < FENCE_Z ? 0.85 : 1.2);
    c.setHSL(0.2 + r() * 0.08, 0.3, 0.72 + r() * 0.28);
  });
  grass.receiveShadow = true;
  world.add(grass);

  // Meadow flowers past the gate: buttercups, clover, a few cornflowers.
  var mf = [], mc = [], mcols = ['#f6d21e', '#f6d21e', '#fbfaf0', '#f2efe2', '#6a7ae0'].map(function (c) { return new THREE.Color(c); });
  for (i = 0; i < (small ? 1800 : 4500); i++) {
    var fx = (r() - 0.5) * 60, fz = FENCE_Z - 0.6 - r() * 16;
    var fy = land(fx, fz);
    if (fy < WATER_Y + 0.4) continue;
    mf.push(fx, fy + 0.2 + r() * 0.3, fz);
    var col = mcols[Math.floor(r() * mcols.length)];
    mc.push(col.r, col.g, col.b);
  }
  var mfGeo = new THREE.BufferGeometry();
  mfGeo.setAttribute('position', new THREE.Float32BufferAttribute(mf, 3));
  mfGeo.setAttribute('color', new THREE.Float32BufferAttribute(mc, 3));
  world.add(new THREE.Points(mfGeo, new THREE.PointsMaterial({ size: 0.07, vertexColors: true, transparent: true, depthWrite: false,
    map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)') })));

  // Hedges round the garden, and hedgerows over the far fields.
  var hedgeAt = [];
  [-9.8, 9.8].forEach(function (x) { for (var z = 7; z > FENCE_Z - 0.5; z -= 0.75) hedgeAt.push(x + (r() - 0.5) * 0.4, z); });
  for (var line = -4; line <= 6; line++) {
    for (var along = -400; along <= 400; along += small ? 2.6 : 1.8) {
      [[along, -52 - line * 34 + along * 0.3], [line * 52 + along * 0.08, -60 - Math.abs(along)]].forEach(function (h) {
        var x = h[0] + (r() - 0.5) * 0.6, z = h[1] + (r() - 0.5) * 0.6;
        if (z > -66 || z < -380 || Math.abs(x) > 380) return;
        hedgeAt.push(x, z);
      });
    }
  }
  var hedges = new THREE.InstancedMesh(lumpy(new THREE.IcosahedronGeometry(1, 2), 0.16, 2).scale(1.0, 1.15, 1.0),
    leafy(new THREE.MeshLambertMaterial(), 3.2), hedgeAt.length / 2);
  scatter(hedges, hedgeAt.length / 2, function (n, p, q, s, c) {
    var x = hedgeAt[n * 2], z = hedgeAt[n * 2 + 1], garden = z > FENCE_Z - 1;
    p.set(x, land(x, z) + (garden ? 0.75 : 0.4), z);
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar(garden ? 0.95 + r() * 0.3 : 1.1 + r() * 0.6);
    c.setHSL(0.27 + r() * 0.05, 0.4, 0.15 + r() * 0.07);
  });
  hedges.castShadow = hedges.receiveShadow = true;
  world.add(hedges);

  // The old apple tree in the lawn, trees along the river and over the
  // far fields.
  var treeMat = leafy(new THREE.MeshLambertMaterial({ vertexColors: true }), 3.4);
  var apple = new THREE.Mesh(treeGeometry(rng(21), { trunk: 1.6, girth: 0.2, lean: 0.35, limbs: 5, spread: 0.85, reach: 1.6, clumps: 9,
    width: 2.0, clump: 1.0, crownY: 2.7, crownH: 0.9, leaf: '#5a7e34', bark: '#5a4a3c', fruit: 3 }), treeMat);
  apple.position.set(-6.4, 0, -8.2);
  apple.castShadow = apple.receiveShadow = true;
  world.add(apple);

  // Alders and ash along the far bank, kept clear of the line to the sun.
  var BANK = [[-34, -53, 1.15], [-29, -56, 0.8], [-17, -54, 1.35], [-12, -58, 0.75], [40, -53, 1.3],
              [46, -57, 0.85], [58, -54, 1.0], [-50, -54, 1.2], [-58, -59, 0.9], [70, -55, 1.1], [-15, -23.5, 1.25], [-25, -19, 1.05]];
  var bankKinds = [0, 1].map(function (k) {
    return treeGeometry(rng(70 + k), { trunk: 3.0, girth: 0.38, limbs: 5, spread: 0.6, reach: 2.4, clumps: 11, width: 3.0, clump: 1.9,
                                        crownY: 5.4, crownH: 2.8, leaf: '#ffffff', bark: '#8a7a68', squash: 0.85 });
  });
  bankKinds.forEach(function (geo, kind) {
    var mine = BANK.filter(function (b, n) { return n % 2 === kind; });
    var trees = new THREE.InstancedMesh(geo, treeMat, mine.length);
    scatter(trees, mine.length, function (n, p, q, s, c) {
      var w = mine[n];
      p.set(w[0], land(w[0], w[1]) - 0.2, w[1]);
      q.setFromAxisAngle(up, r() * 6.28);
      s.set(w[2] * 1.3, w[2] * (1.2 + r() * 0.3), w[2] * 1.3);
      c.setHSL(0.24 + r() * 0.05, 0.42, 0.42 + r() * 0.1);
    });
    trees.castShadow = true;
    world.add(trees);
  });

  var far = new THREE.InstancedMesh(broadleafGeometry(rng(13), '#4a3a2e'), new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true }),
                                    small ? 260 : 520);
  scatter(far, 30000, function (n, p, q, s, c) {
    var x = (r() - 0.5) * 700, z = -62 - Math.pow(r(), 0.8) * 320;
    if (z > -120 && x > 5 && x < 90) return false;                              // the way to the sun stays open
    var fu = Math.floor((x * 0.85 + z * 0.35) / 46), fv = Math.floor((z * 0.9 - x * 0.3) / 34);
    if (hash2(fu, fv) > 0.3 && r() < 0.85) return false;                        // copses, not a carpet
    p.set(x, land(x, z) - 1.4, z);
    q.setFromAxisAngle(up, r() * 6.28);
    var sc = 1.7 + r() * 1.2;
    s.set(sc, sc * (0.9 + r() * 0.3), sc);
    c.setHSL(0.24 + r() * 0.06, 0.38, 0.28 + r() * 0.1);
  });
  far.castShadow = true;
  world.add(far);

  // ── Sparrows: perched on the rail, then flying ──
  var perched = new THREE.InstancedMesh(sparrowGeometry(), new THREE.MeshLambertMaterial({ vertexColors: true }), PERCH.length);
  perched.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  perched.castShadow = true;
  world.add(perched);
  var flyMat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
  flyMat.onBeforeCompile = function (sh) {
    sh.uniforms.uTime = U.uTime;
    sh.vertexShader = 'uniform float uTime;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n float fph = float(gl_InstanceID) * 1.37;\n' +
      // Sparrows flap in bursts and glide between.
      ' float beat = sin(uTime * 26.0 + fph) * (0.25 + 0.75 * smoothstep(-0.4, 0.2, sin(uTime * 2.1 + fph)));\n' +
      ' transformed.y += beat * max(abs(position.x) - 0.018, 0.0) * 1.6;');
  };
  flyMat.customProgramCacheKey = function () { return 'wg-flyer'; };
  var NF = small ? 12 : 16;
  var flyers = new THREE.InstancedMesh(flyerGeometry(), flyMat, NF);
  flyers.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  flyers.frustumCulled = false;
  world.add(flyers);
  var perch = PERCH.map(function (x) { return new THREE.Vector3(x, railY(x) + 0.035, FENCE_Z + 0.07); });
  var sp = PERCH.map(function (x, n) {
    var face = (n % 3 === 0 ? 0.3 : n % 3 === 1 ? 2.9 : -1.4) + (r() - 0.5) * 0.6;
    return { base: face, yaw: face, aim: face, next: 0, hop: 0 };
  });
  // Each flyer: where it starts (a perch, or the hedge), its delay, and the
  // loop it settles into over the meadow.
  var birds = [];
  for (i = 0; i < NF; i++) {
    var from = i < PERCH.length ? perch[i].clone() : new THREE.Vector3((r() < 0.5 ? -1 : 1) * (6 + r() * 6), 1.0 + r() * 0.8, FENCE_Z + 0.3 + r() * 1.5);
    birds.push({ from: from, delay: i < PERCH.length ? r() * 0.35 : 0.2 + r() * 0.7, R: 4 + r() * 7, w: (0.5 + r() * 0.4) * (r() < 0.5 ? -1 : 1),
                 ph: r() * 6.28, cy: 6 + r() * 5, sq: 0.45 + r() * 0.35, bob: 0.6 + r() * 1.2 });
  }
  var FLOCK = new THREE.Vector3(5, 0, -36), launched = -1;

  // ── Weather: mist, rain, and light in the air ──
  var mistTex = softSprite('rgba(255,255,255,0.85)', 'rgba(255,255,255,0)'), mist = [];
  for (i = 0; i < (small ? 26 : 40); i++) {
    var ms = new THREE.Sprite(new THREE.SpriteMaterial({ map: mistTex, transparent: true, depthWrite: false, opacity: 0 }));
    var mx = (r() - 0.5) * 90, mz = i % 2 ? riverZ(mx) + (r() - 0.5) * 22 : 4 - r() * 30;     // half of it lies on the river
    ms.position.set(mx, Math.max(land(mx, mz), WATER_Y) + 0.6 + r() * 2.2, mz);
    ms.scale.set(10 + r() * 14, 2.5 + r() * 3, 1);
    ms.userData.y = ms.position.y;
    world.add(ms);
    mist.push(ms);
  }
  var rain = rainField({ count: small ? 2000 : 4500, box: [14, 12, 20], speed: 10, windSpeed: 2, color: '#dfe4e8', opacity: 0.34 });
  world.add(rain.lines);
  var motes = particleField({ count: small ? 220 : 520, box: [26, 8, 26], fall: [-0.06, 0.04], size: 0.035, color: '#ffe2a8',
                              map: softSprite('rgba(255,236,190,1)', 'rgba(255,220,150,0)'), sway: 0.25, windSpeed: 0.3 });
  motes.points.material.blending = THREE.AdditiveBlending;
  world.add(motes.points);

  // ── Post pass: colour, blur and vignette for the whole frame ──
  var rt = new THREE.WebGLRenderTarget(4, 4, { type: THREE.HalfFloatType, samples: small ? 0 : 4 });
  var postScene = new THREE.Scene(), postCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  var postMat = new THREE.ShaderMaterial({
    depthTest: false, depthWrite: false,
    uniforms: { tScene: { value: rt.texture }, uRes: { value: new THREE.Vector2(1, 1) }, uSat: { value: 0.3 }, uBlur: { value: 0 },
                uVig: { value: 0.4 }, uWarm: { value: 0 }, uFlood: { value: 0 }, uSunUv: { value: new THREE.Vector2(0.7, 0.6) } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
    fragmentShader: [
      'uniform sampler2D tScene; uniform vec2 uRes; uniform float uSat; uniform float uBlur; uniform float uVig; uniform float uWarm;',
      'uniform float uFlood; uniform vec2 uSunUv; varying vec2 vUv;',
      'void main(){',
      ' vec4 c = texture2D(tScene, vUv); vec2 q = vUv - 0.5;',
      ' if (uBlur > 0.002) {',
      // Failing sight: a soft disc blur, worse towards the edges.
      '  float rad = uBlur * 0.016 * (0.55 + 1.4 * length(q)); vec3 acc = c.rgb;',
      '  for (int i = 0; i < 24; i++) { float fi = float(i) + 0.5, a = fi * 2.39996;',
      '   acc += texture2D(tScene, vUv + vec2(cos(a), sin(a)) * sqrt(fi / 24.0) * rad * vec2(uRes.y / uRes.x, 1.0)).rgb; }',
      '  c.rgb = acc / 25.0;',
      ' }',
      ' float l = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));',
      // Colour floods outwards from the sun as it breaks through.
      ' vec2 a = vec2(uRes.x / uRes.y, 1.0); float R = uFlood * 2.6;',
      ' float sat = mix(uSat, 1.22, 1.0 - smoothstep(R - 0.5, R, length((vUv - uSunUv) * a)));',
      ' c.rgb = max(mix(vec3(l), c.rgb, sat), 0.0);',
      ' c.rgb *= mix(vec3(0.97, 1.0, 1.04), vec3(1.07, 1.01, 0.9), uWarm);',
      ' c.rgb *= 1.0 - uVig * dot(q, q) * 1.8;',
      ' gl_FragColor = c;',
      ' #include <tonemapping_fragment>',
      ' #include <colorspace_fragment>',
      '}'
    ].join('\n')
  });
  postScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), postMat));
  var bufSize = new THREE.Vector2();

  // ── Frame ──
  var look = new THREE.Vector3(), tan = new THREE.Vector3(), fwd = new THREE.Vector3();
  var m4 = new THREE.Matrix4(), q4 = new THREE.Quaternion(), s3 = new THREE.Vector3(), p3 = new THREE.Vector3(), t3 = new THREE.Vector3(),
      b3 = new THREE.Vector3(), c3 = new THREE.Vector3();
  var C = {
    top: [new THREE.Color('#7c848b'), new THREE.Color('#4a82c6')], hor: [new THREE.Color('#aeb4b6'), new THREE.Color('#e6e2d2')],
    cloud: [new THREE.Color('#a4aaae'), new THREE.Color('#fbf6ec')], shade: [new THREE.Color('#50575d'), new THREE.Color('#b2b8c6')],
    fog: [new THREE.Color('#a7aeb1'), new THREE.Color('#b6c7d4')], hemiSky: [new THREE.Color('#c4ccd2'), new THREE.Color('#d2e2f6')],
    hemiGround: [new THREE.Color('#4a5040'), new THREE.Color('#5d6b3a')], blind: new THREE.Color('#b9bfc1'), sunCol: new THREE.Color('#fff0d8')
  };
  var pf = { snow: 0, wind: 0, dt: 0, time: 0 }, portrait = false;

  // Where a launched sparrow is `a` seconds after take-off: a rising burst
  // away from the rail that eases into a wide loop over the meadow.
  function flyerAt(b, a, out) {
    var spin = a * b.w * (env.reduceMotion ? 0.4 : 1) + b.ph;
    out.set(FLOCK.x + Math.cos(spin) * b.R, b.cy + Math.sin(spin * 2) * b.bob, FLOCK.z + Math.sin(spin) * b.R * b.sq);
    var e = smooth(0, 2.6, a);
    if (e < 1) {
      // From the perch: up and out, a quick arc.
      c3.copy(b.from).lerp(out, e);
      c3.y += Math.sin(e * Math.PI) * 2.2;
      out.copy(c3);
    }
    return out;
  }

  function frame(f) {
    var row = f.row, time = f.time, dt = f.dt;
    var gloom = row[1], rainAmt = row[2], wind = row[3], fog = row[6], blur = row[7], colour = row[8], sunAmt = row[9],
        open = row[10], flight = row[11], clear = 1 - gloom;

    // Walk the path: along its direction, then turn by yaw and pitch. On a
    // portrait screen the verse sits mid-frame, so by the rail look down a
    // little to keep the sparrows above it and the flowers below.
    var t = clamp(f.cam, 0, 1);
    curve.getPointAt(t, camera.position);
    curve.getTangentAt(t, tan);
    camera.position.y = land(camera.position.x, camera.position.z) + 1.6 + Math.sin(time * 1.1) * 0.012;
    look.copy(camera.position).addScaledVector(tan.setY(0).normalize(), 10);
    camera.lookAt(look);
    camera.rotateY(row[4] - f.mx * 0.14);
    camera.rotateX(row[5] + (portrait ? 0.05 - 0.24 * smooth(T_RAIL[0], T_RAIL[1], t) * (1 - smooth(T_RAIL[2], T_RAIL[3], t)) : 0) - f.my * 0.06 + (env.reduceMotion ? 0 : Math.sin(time * 0.7) * 0.008 * blur));
    sky.position.copy(camera.position);

    // Sky and light from grey dawn to the bright morning.
    var du = dome.uniforms;
    du.uTop.value.copy(C.top[0]).lerp(C.top[1], clear);
    du.uHorizon.value.copy(C.hor[0]).lerp(C.hor[1], clear);
    du.uCloud.value.copy(C.cloud[0]).lerp(C.cloud[1], clear);
    du.uShade.value.copy(C.shade[0]).lerp(C.shade[1], clear);
    du.uSun.value.copy(C.sunCol).multiplyScalar(0.06 + sunAmt * 0.9);
    du.uCover.value = lerp(0.97, 0.42, sunAmt);
    du.uBreak.value = smooth(0, 0.6, sunAmt);
    du.uTime.value = time;
    du.uTurn.value = time * 0.006 + f.cam * 0.6;

    world.fog.color.copy(C.fog[0]).lerp(C.fog[1], clear).lerp(C.blind, fog * 0.6);
    world.fog.density = 0.0018 + gloom * 0.0097 + fog * 0.06;
    du.uHorizon.value.copy(world.fog.color);
    gl.setClearColor(world.fog.color);

    hemi.color.copy(C.hemiSky[0]).lerp(C.hemiSky[1], clear);
    hemi.groundColor.copy(C.hemiGround[0]).lerp(C.hemiGround[1], clear);
    hemi.intensity = lerp(1.7, 1.25, sunAmt);
    sun.intensity = 0.25 + sunAmt * 3.0;
    sun.color.copy(C.sunCol);
    sun.position.copy(camera.position).addScaledVector(SUN_DIR, 80);
    fwd.set(0, 0, -1).applyQuaternion(camera.quaternion).setY(0).normalize();
    sun.target.position.copy(camera.position).addScaledVector(fwd, 10);

    var veil = smooth(0.1, 0.7, sunAmt);
    sunGlow.material.opacity = 0.15 + veil * 0.85;
    sunDisc.material.opacity = veil;
    sunRays.material.opacity = veil * 0.5;
    sunRays.material.rotation = time * 0.008;
    sunGlow.scale.setScalar(500 + veil * 500);

    U.uTime.value = time;
    U.uWind.value = wind;
    U.uRain.value = Math.max(rainAmt, 0.04);
    U.uOpen.value = open;
    U.uSkyLow.value.copy(du.uHorizon.value).multiplyScalar(0.7);
    U.uSkyHigh.value.copy(du.uCloud.value).lerp(du.uTop.value, 1 - du.uCover.value).multiplyScalar(0.75);

    // Mist hangs round you, closes in for the failing eyes, then rises
    // and thins in the sun.
    for (k = 0; k < mist.length; k++) {
      var m = mist[k];
      m.position.y = m.userData.y + sunAmt * 5;
      m.material.opacity = (0.18 + fog * 0.4) * gloom * (1 - sunAmt * 0.9);
      m.material.color.copy(world.fog.color);
    }

    // Sparrows on the rail: quick turns and hops, until they go.
    if (flight > 0.5 && launched < 0) launched = time;
    if (flight < 0.3) launched = -1;
    var age0 = launched < 0 ? -1 : time - launched;
    for (k = 0; k < perch.length; k++) {
      var s = sp[k];
      if (age0 >= birds[k].delay) { perched.setMatrixAt(k, m4.makeScale(0, 0, 0)); continue; }
      if (time > s.next) {
        s.aim = s.base + (Math.random() - 0.5) * 1.6;
        s.next = time + 0.6 + Math.random() * 2.2;
        if (Math.random() < 0.35) s.hop = 1;
      }
      s.yaw += (s.aim - s.yaw) * (1 - Math.exp(-dt * 16));
      s.hop = Math.max(0, s.hop - dt * 4);
      q4.setFromAxisAngle(up, s.yaw);
      p3.copy(perch[k]);
      p3.y += Math.sin(s.hop * Math.PI) * 0.05;
      s3.set(1.7, 1.7 * (1 + 0.03 * Math.sin(time * 5 + k)), 1.7);
      perched.setMatrixAt(k, m4.compose(p3, q4, s3));
    }
    perched.instanceMatrix.needsUpdate = true;

    for (k = 0; k < birds.length; k++) {
      var b = birds[k], a = age0 - b.delay;
      if (a < 0) { flyers.setMatrixAt(k, m4.makeScale(0, 0, 0)); continue; }
      flyerAt(b, a, p3);
      flyerAt(b, a + 0.05, t3);
      b3.set(0, 1, 0);
      m4.lookAt(p3, t3, b3);
      s3.setScalar(1.6);
      m4.scale(s3);
      m4.setPosition(p3);
      flyers.setMatrixAt(k, m4);
    }
    flyers.instanceMatrix.needsUpdate = true;

    pf.dt = dt; pf.time = time; pf.wind = wind * 0.5;
    rain.update(f, camera.position, rainAmt, env.reduceMotion);
    pf.snow = sunAmt * 0.7;
    motes.update(pf, camera.position, env.reduceMotion);

    // Post: grey and blurred while worrying, colour flooding back after.
    postMat.uniforms.uSat.value = colour;
    postMat.uniforms.uFlood.value = row[12];
    p3.copy(SUN_DIR).multiplyScalar(500).add(camera.position).project(camera);
    postMat.uniforms.uSunUv.value.set(p3.x * 0.5 + 0.5, p3.y * 0.5 + 0.5);
    postMat.uniforms.uBlur.value = blur;
    postMat.uniforms.uVig.value = 0.35 + gloom * 0.25 + blur * 0.6;
    postMat.uniforms.uWarm.value = sunAmt;
    gl.toneMappingExposure = lerp(0.95, 1.08, sunAmt);
    gl.getDrawingBufferSize(bufSize);
    if (rt.width !== bufSize.x || rt.height !== bufSize.y) rt.setSize(bufSize.x, bufSize.y);
    postMat.uniforms.uRes.value.copy(bufSize);
    gl.setRenderTarget(rt);
    gl.render(world, camera);
    gl.setRenderTarget(null);
    gl.render(postScene, postCam);
  }

  return {
    resize: function (w, h, dpr) {
      fitCamera(gl, camera, w, h, dpr, small);
      portrait = w < h;
    },
    frame: frame,
    destroy: function () { rt.dispose(); disposeAll(postScene); disposeAll(world, gl); }
  };
}

PI.register('worry-garden', {
  renderer: renderer3d,
  align: ['left', 'right', 'left', 'center', 'left'],
  scrim: 0.6,
  keys: function (T) {
    var n = T.count;
    function at(i, frac) { i = Math.min(i, n - 1); return lerp(T.start(i), T.end(i), frac); }
    //  unit          path        gloom rain  wind  yaw    pitch  fog   blur  colour sun  open  flight flood
    return [
      [0,             tAt(4.4),   1.0, 0.0, 0.15, 0.00,  0.03,  0.30, 0.0, 0.42, 0.0, 0.30, 0, 0],
      [0.7,           tAt(4.2),   1.0, 0.0, 0.15, 0.00,  0.03,  0.30, 0.0, 0.42, 0.0, 0.30, 0, 0],
      [at(0, 0.3),    tAt(3.5),   1.0, 0.0, 0.20, 0.50, -0.30,  0.25, 0.0, 0.42, 0.0, 0.30, 0, 0],   // "will the garden grow"
      [at(0, 0.58),   tAt(2.7),   1.0, 0.0, 0.25, 0.00, -0.02,  0.00, 0.0, 0.42, 0.0, 0.30, 0, 0],   // "will the rivers flow"
      [at(0, 0.9),    tAt(2.0),   1.0, 0.0, 0.30, -0.08, 0.45,  0.15, 0.0, 0.40, 0.0, 0.30, 0, 0],   // "will the earth turn as it was taught"
      [at(1, 0.2),    tAt(1.2),   1.0, 0.5, 0.35, 0.00, -0.05,  0.30, 0.0, 0.38, 0.0, 0.30, 0, 0],
      [at(1, 0.5),    tAt(0.4),   1.0, 1.0, 0.40, -0.08, -0.52,  0.35, 0.0, 0.38, 0.0, 0.30, 0, 0],   // "was I right, was I wrong": puddles
      [at(1, 0.85),   tAt(-0.8),  1.0, 0.9, 0.35, 0.00, -0.30,  0.35, 0.0, 0.38, 0.0, 0.30, 0, 0],
      [at(2, 0.15),   tAt(-3.2),  1.0, 0.5, 0.30, -0.15, -0.10, 0.30, 0.0, 0.40, 0.0, 0.30, 0, 0],
      [at(2, 0.5),    tAt(-8.2),  1.0, 0.2, 0.20, -0.55, -0.20, 0.25, 0.0, 0.42, 0.0, 0.30, 0, 0],   // "even the sparrows can do it"
      [at(2, 0.85),   tAt(-8.5),  1.0, 0.1, 0.20, -0.52, -0.18, 0.25, 0.0, 0.42, 0.0, 0.30, 0, 0],
      [at(3, 0.2),    tAt(-8.7),  1.0, 0.0, 0.15, -0.46, -0.14, 0.75, 0.5, 0.22, 0.0, 0.30, 0, 0],
      [at(3, 0.55),   tAt(-8.8),  1.0, 0.0, 0.10, -0.42, -0.12, 1.00, 1.0, 0.10, 0.0, 0.30, 0, 0],   // "is my eyesight fading"
      [at(3, 0.92),   tAt(-8.8),  1.0, 0.0, 0.10, -0.42, -0.12, 1.00, 1.0, 0.10, 0.0, 0.30, 0, 0],
      [at(4, 0.14),   tAt(-8.9),  0.85, 0.0, 0.15, -0.44, -0.16, 0.45, 0.25, 0.40, 0.05, 0.30, 0, 0],  // "worrying had come to nothing"
      [at(4, 0.28),   tAt(-9.0),  0.6, 0.0, 0.15, -0.42, -0.15, 0.15, 0.0, 0.45, 0.35, 0.55, 0, 0.35],  // colour spreads over the bed
      [at(4, 0.42),   tAt(-9.3),  0.4, 0.0, 0.20, -0.34,  0.08, 0.06, 0.0, 0.50, 0.65, 0.90, 1, 0.7],   // "and gave it up": the sparrows go
      [at(4, 0.62),   tAt(-11.2), 0.15, 0.0, 0.25, -0.12, 0.05, 0.03, 0.0, 0.50, 0.9, 1.00, 1, 1],     // "went out into the morning"
      [at(4, 0.92),   tAt(-14),   0.0, 0.0, 0.25, -0.06,  0.03, 0.02, 0.0, 0.50, 1.0, 1.00, 1, 1],     // "and sang"
      [T.total,       tAt(-16.5), 0.0, 0.0, 0.20, -0.1,   0.06, 0.02, 0.0, 0.50, 1.0, 1.00, 1, 1]
    ];
  },
  sound: {
    src: '/audio/birds.mp3',
    label: 'Play the garden birdsong',
    // Faint in the grey, hushed by the rain, swelling into the morning.
    volume: function (row) { return (0.05 + row[9] * 0.6) * (1 - row[2] * 0.6) * (1 - row[7] * 0.5); },
    cues: [
      { stanza: 1, at: 0.2, play: rainPatter },
      { stanza: 2, at: 0.5, play: sparrows },
      { stanza: 4, at: 0.56, play: flight },
      { stanza: 4, at: 1.35, play: chorus }
    ]
  }
});
