/*
 * Scene for "Veins of Marble" (u/OlivesBark): a sculpture garden by moonlight.
 *
 * The three stanzas are 4, 8 and 3 lines; maxLines 6 splits the middle one
 * after "lips of marble?", so there are four panels.
 * Opening  the gate of a formal garden at night: a gravel allée between
 *     clipped box hedges and black cypresses, marble urns on plinths, and at
 *     its far end, under the moon, one pale group on a plinth.
 * I   "If lovers turned to stone": walking down the allée towards it.
 * II  "Mere statues in your passing": past the urns, into the round garden,
 *     and the group is two veiled forms leaning together, fused at the brow
 *     and at the hem; we stop and look up to where they meet, "lips of
 *     marble". Rose petals begin to drift down round them.
 * III "The frozen folds of a flowing dress": close on the carved cloth,
 *     swept sideways as if caught mid-step. "Cold against your fingertips":
 *     frost climbs the stone and spreads across the lawn, the light goes
 *     blue, and "better petrified": time stops, the petals hang in the air,
 *     the cypresses stop stirring and the clouds stop crossing the moon.
 * IV  "This beating heart is only half of a home": one warm light beside
 *     the cold pair, beating, alone, warming the stone it cannot thaw.
 * Outro the view rises over the frozen garden; the small light still beats.
 *
 * The marble is procedural (veins from fbm-warped sine bands in world
 * space, a cloudy body, a faint rim of subsurface glow that the heart's
 * light also bleeds into), and the frost is the same shader, rising from
 * the ground up the stone and outward over the lawn and hedges.
 * Columns (see COLS): [unit, camZ, cold, petals, wind, camX, camY, lookX,
 *   lookY, lookZ, frost, still, heart]
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, softSprite, skyDome, starField,
         scatter, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

var COLS = ['camZ', 'cold', 'petals', 'wind', 'camX', 'camY', 'lookX', 'lookY', 'lookZ', 'frost', 'still', 'heart'];
var K = {};
COLS.forEach(function (c, i) { K[c] = i; });

// ── Layout (metres): the allée runs north (-z) to a round garden ─────────
var CENTER = new THREE.Vector3(0, 0, -20);      // the lovers' plinth
var PLINTH_TOP = 1.4;
var RING = 7.6;                                 // the low hedge round the garden
var HEART = new THREE.Vector3(-1.65, 2.3, -18.5);

// ── Shared GLSL: value noise, fbm, and the marble and frost ──────────────
var NOISE = [
  'float mgHash(vec3 p){ p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }',
  'float mgNoise(vec3 x){ vec3 i = floor(x), f = fract(x); f = f * f * (3.0 - 2.0 * f);',
  ' return mix(mix(mix(mgHash(i), mgHash(i + vec3(1,0,0)), f.x), mix(mgHash(i + vec3(0,1,0)), mgHash(i + vec3(1,1,0)), f.x), f.y),',
  '            mix(mix(mgHash(i + vec3(0,0,1)), mgHash(i + vec3(1,0,1)), f.x), mix(mgHash(i + vec3(0,1,1)), mgHash(i + vec3(1,1,1)), f.x), f.y), f.z); }',
  'float mgFbm(vec3 p){ float a = 0.5, s = 0.0; for (int i = 0; i < 5; i++){ s += a * mgNoise(p); p = p * 2.07 + vec3(1.7, 9.2, 3.1); a *= 0.5; } return s; }',
  // Frost: 1 where it has reached. It climbs from the ground near the
  // lovers and spreads outward over the garden as uFrost goes 0 -> 1.
  'float mgFrost(vec3 w, float top){ float n = mgFbm(w * 1.7);',
  ' float reach = uFrost * 26.0 - length(w.xz - uCenter.xz) + (n - 0.5) * 3.0;',
  ' float climb = uFrost * top - w.y + (n - 0.5) * 0.9;',
  ' return smoothstep(0.0, 0.5, climb) * smoothstep(0.0, 2.0, reach); }'
].join('\n');

function worldPosVertex(sh) {
  sh.vertexShader = 'varying vec3 vMgW;\n' + sh.vertexShader.replace('#include <project_vertex>',
    '#include <project_vertex>\n vec4 mgW = vec4(transformed, 1.0);\n' +
    '#ifdef USE_INSTANCING\n mgW = instanceMatrix * mgW;\n#endif\n vMgW = (modelMatrix * mgW).xyz;');
}

// Marble: veined, cloudy, a little translucent at the rim; frosts over.
function marbleMaterial(U, o) {
  var mat = new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.34, metalness: 0 });
  mat.onBeforeCompile = function (sh) {
    Object.assign(sh.uniforms, U);
    sh.uniforms.uVein = { value: o.vein };
    sh.uniforms.uGlow = { value: o.glow };
    worldPosVertex(sh);
    sh.fragmentShader = [
      'uniform float uFrost; uniform float uCold; uniform vec3 uCenter; uniform vec3 uHeartPos; uniform vec3 uHeartCol;',
      'uniform float uVein; uniform float uGlow; varying vec3 vMgW;', NOISE
    ].join('\n') + '\n' + sh.fragmentShader
      .replace('#include <color_fragment>', [
        '#include <color_fragment>',
        ' vec3 mp = vMgW * uVein;',
        ' float warp = mgFbm(mp * 0.9);',
        // Two families of veins: broad drifting grey bands and fine cracks.
        ' float b1 = abs(sin(mp.x * 1.1 + mp.y * 1.9 - mp.z * 0.6 + warp * 7.0));',
        ' float b2 = abs(sin(mp.z * 2.6 - mp.y * 1.3 + mp.x * 0.8 + mgFbm(mp * 2.3 + 7.0) * 6.0));',
        ' float vein = 1.0 - smoothstep(0.0, 0.07, b1);',
        ' float halo = 1.0 - smoothstep(0.0, 0.35, b1);',
        ' float fine = 1.0 - smoothstep(0.0, 0.03, b2);',
        // Fade veins that would be thinner than a pixel, which only shimmer.
        ' float fw = length(fwidth(mp));',
        ' vein *= 1.0 - smoothstep(0.08, 0.35, fw); fine *= 1.0 - smoothstep(0.03, 0.15, fw);',
        ' vec3 body = mix(vec3(0.95, 0.93, 0.89), vec3(0.84, 0.84, 0.85), smoothstep(0.35, 0.75, mgFbm(mp * 0.5 + 3.0)));',
        ' body = mix(body, vec3(0.74, 0.74, 0.77), halo * 0.35);',
        ' body = mix(body, vec3(0.36, 0.38, 0.44), vein * 0.8);',
        ' body = mix(body, vec3(0.55, 0.53, 0.52), fine * 0.45);',
        ' float fr = mgFrost(vMgW, 4.2);',
        ' float feather = 1.0 - abs(mgNoise(vMgW * 38.0) * 2.0 - 1.0);',
        ' body = mix(body, mix(vec3(0.78, 0.86, 0.96), vec3(0.95, 0.98, 1.0), feather), fr * 0.85);',
        ' diffuseColor.rgb *= mix(body, body * vec3(0.86, 0.92, 1.05), uCold);'
      ].join('\n'))
      .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\n roughnessFactor = mix(roughnessFactor, 0.75, mgFrost(vMgW, 4.2));')
      .replace('#include <emissivemap_fragment>', [
        '#include <emissivemap_fragment>',
        ' vec3 mgV = normalize(vViewPosition);',
        ' float rim = pow(1.0 - clamp(dot(normal, mgV), 0.0, 1.0), 2.5);',
        // Light sinking into the stone and coming back out, cool by moonlight...
        ' totalEmissiveRadiance += mix(vec3(0.10, 0.12, 0.17), vec3(0.06, 0.09, 0.16), uCold) * uGlow * (0.35 + rim);',
        // ...and warm on the side the heart is on (wrapped, so it reaches round).
        ' vec3 hl = uHeartPos - (-vViewPosition); float hd = length(hl);',
        ' float wrap = clamp((dot(normal, hl / hd) + 0.7) / 1.7, 0.0, 1.0);',
        ' totalEmissiveRadiance += uHeartCol * wrap * wrap * uGlow / (1.0 + hd * hd * 0.9);',
        // Frost sparkles: tiny facets that catch the light as you move.
        ' float fz = mgFrost(vMgW, 4.2);',
        ' float spark = smoothstep(0.86, 0.97, mgNoise(vMgW * 140.0 + mgV * 6.0)) * (1.0 - smoothstep(0.02, 0.06, length(fwidth(vMgW))));',
        ' totalEmissiveRadiance += vec3(0.7, 0.8, 1.0) * spark * fz * 0.2;'
      ].join('\n'));
  };
  return mat;
}

// Frost on everything else (lawn, gravel, hedges, cypresses): a pale rime.
function frosty(mat, U, amount, top) {
  mat.onBeforeCompile = function (sh) {
    Object.assign(sh.uniforms, U);
    worldPosVertex(sh);
    if (mat.userData.sway) mat.userData.sway(sh);
    sh.fragmentShader = 'uniform float uFrost; uniform float uCold; uniform vec3 uCenter; varying vec3 vMgW;\n' + NOISE + '\n' +
      sh.fragmentShader.replace('#include <color_fragment>', [
        '#include <color_fragment>',
        ' float mgF = mgFrost(vMgW, ' + top.toFixed(2) + ') * (0.55 + 0.45 * mgNoise(vMgW * 21.0));',
        ' diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.62, 0.7, 0.82), mgF * ' + amount.toFixed(2) + ');',
        ' diffuseColor.rgb *= mix(vec3(1.0), vec3(0.85, 0.92, 1.08), uCold);'
      ].join('\n'));
  };
  return mat;
}

// ── Textures ─────────────────────────────────────────────────────────────
function canvasTex(size, draw, repeat) {
  var c = document.createElement('canvas');
  c.width = c.height = size;
  draw(c.getContext('2d'), size);
  var t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  if (repeat) t.repeat.set(repeat[0], repeat[1]);
  return t;
}

function speckle(r, base, shades, count, rmin, rmax) {
  return function (x, s) {
    x.fillStyle = base;
    x.fillRect(0, 0, s, s);
    for (var i = 0; i < count; i++) {
      x.fillStyle = shades[Math.floor(r() * shades.length)];
      var px = r() * s, py = r() * s, rad = rmin + r() * (rmax - rmin), a = r() * Math.PI;
      // Draw wrapped so the tile repeats seamlessly.
      for (var dx = -1; dx <= 1; dx++) for (var dy = -1; dy <= 1; dy++) {
        x.beginPath();
        x.ellipse(px + dx * s, py + dy * s, rad, rad * 0.6, a, 0, Math.PI * 2);
        x.fill();
      }
    }
  };
}

// ── Geometry ─────────────────────────────────────────────────────────────
// A clipped hedge block whose UVs are in metres (so the leaf texture keeps
// its scale), as non-indexed geometry with position, normal and uv.
function hedgeBox(w, h, d, x, z, ry) {
  var g = new THREE.BoxGeometry(w, h, d).toNonIndexed();
  var uv = g.attributes.uv, dims = [[d, h], [d, h], [w, d], [w, d], [w, h], [w, h]];
  for (var i = 0; i < uv.count; i++) {
    var f = dims[Math.floor(i / 6)];
    uv.setXY(i, uv.getX(i) * f[0], uv.getY(i) * f[1]);
  }
  g.rotateY(ry || 0);
  g.translate(x, h / 2, z);
  return g;
}

// A curved hedge round the garden centre, open towards +z (up the allée)
// for `gap` radians either side. Angle 0 points along +z.
function ringHedge(rIn, rOut, h, gap, segs) {
  var P = [], N = [], T = [];
  function pt(rad, a, y) { return [CENTER.x + Math.sin(a) * rad, y, CENTER.z + Math.cos(a) * rad]; }
  function quad(v, n, uv) {
    [0, 1, 2, 0, 2, 3].forEach(function (k) { P.push.apply(P, v[k]); N.push.apply(N, n[k]); T.push.apply(T, uv[k]); });
  }
  var a0 = gap, span = Math.PI * 2 - gap * 2, up = [0, 1, 0];
  for (var k = 0; k < segs; k++) {
    var a = a0 + span * k / segs, b = a0 + span * (k + 1) / segs;
    var na = [Math.sin(a), 0, Math.cos(a)], nb = [Math.sin(b), 0, Math.cos(b)];
    var ia = [-na[0], 0, -na[2]], ib = [-nb[0], 0, -nb[2]];
    var ua = a * rOut, ub = b * rOut;
    quad([pt(rOut, a, 0), pt(rOut, b, 0), pt(rOut, b, h), pt(rOut, a, h)], [na, nb, nb, na], [[ua, 0], [ub, 0], [ub, h], [ua, h]]);
    quad([pt(rIn, b, 0), pt(rIn, a, 0), pt(rIn, a, h), pt(rIn, b, h)], [ib, ia, ia, ib], [[ub, 0], [ua, 0], [ua, h], [ub, h]]);
    quad([pt(rIn, a, h), pt(rOut, a, h), pt(rOut, b, h), pt(rIn, b, h)], [up, up, up, up], [[ua, 0], [ua, rOut - rIn], [ub, rOut - rIn], [ub, 0]]);
  }
  // Square ends at the opening.
  var e = a0 + span, t0 = [-Math.cos(a0), 0, Math.sin(a0)], t1 = [Math.cos(e), 0, -Math.sin(e)], w = rOut - rIn;
  quad([pt(rIn, a0, 0), pt(rOut, a0, 0), pt(rOut, a0, h), pt(rIn, a0, h)], [t0, t0, t0, t0], [[0, 0], [w, 0], [w, h], [0, h]]);
  quad([pt(rOut, e, 0), pt(rIn, e, 0), pt(rIn, e, h), pt(rOut, e, h)], [t1, t1, t1, t1], [[0, 0], [w, 0], [w, h], [0, h]]);
  var g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(P, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(N, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(T, 2));
  return g;
}

function mergeUV(geos) {
  var total = 0;
  geos.forEach(function (g) { total += g.attributes.position.count; });
  var out = new THREE.BufferGeometry();
  [['position', 3], ['normal', 3], ['uv', 2]].forEach(function (a) {
    var arr = new Float32Array(total * a[1]), o = 0;
    geos.forEach(function (g) { arr.set(g.attributes[a[0]].array, o); o += g.attributes[a[0]].array.length; });
    out.setAttribute(a[0], new THREE.BufferAttribute(arr, a[1]));
  });
  geos.forEach(function (g) { g.dispose(); });
  return out;
}

// A value noise in JS, for lumpy foliage.
function noise3(r) {
  var P = new Float32Array(512);
  for (var i = 0; i < 512; i++) P[i] = r();
  function h(x, y, z) { return P[((x * 73856093) ^ (y * 19349663) ^ (z * 83492791)) & 511]; }
  return function (x, y, z) {
    var ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z), fx = x - ix, fy = y - iy, fz = z - iz;
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz);
    function l(a, b, t) { return a + (b - a) * t; }
    return l(l(l(h(ix, iy, iz), h(ix + 1, iy, iz), fx), l(h(ix, iy + 1, iz), h(ix + 1, iy + 1, iz), fx), fy),
             l(l(h(ix, iy, iz + 1), h(ix + 1, iy, iz + 1), fx), l(h(ix, iy + 1, iz + 1), h(ix + 1, iy + 1, iz + 1), fx), fy), fz);
  };
}

// An Italian cypress about 1 m tall (scaled per instance), a lumpy flame.
function cypressGeometry(n) {
  var prof = [[0.001, 0], [0.05, 0.0], [0.05, 0.05], [0.075, 0.06], [0.095, 0.16], [0.1, 0.36], [0.088, 0.58], [0.064, 0.78],
              [0.034, 0.92], [0.012, 0.985], [0.001, 1]];
  var pts = prof.map(function (p) { return new THREE.Vector2(p[0], p[1]); });
  // Sampled along a smooth curve so the outline isn't faceted.
  var g = new THREE.LatheGeometry(new THREE.SplineCurve(pts).getSpacedPoints(36), 16);
  var p = g.attributes.position, cols = new Float32Array(p.count * 3), c = new THREE.Color();
  for (var i = 0; i < p.count; i++) {
    var x = p.getX(i), y = p.getY(i), z = p.getZ(i), rr = Math.hypot(x, z);
    if (rr > 0.02 && y > 0.055) {
      var k = 1 + 0.28 * (n(x * 40, y * 26, z * 40) - 0.5) + 0.14 * (n(x * 110, y * 70, z * 110) - 0.5);
      p.setX(i, x * k);
      p.setZ(i, z * k);
    }
    var shade = y < 0.055 ? 0.35 : 0.55 + 0.45 * smooth(0, 0.8, y);
    c.set(y < 0.055 ? '#3a3026' : '#ffffff').multiplyScalar(shade);
    cols[i * 3] = c.r; cols[i * 3 + 1] = c.g; cols[i * 3 + 2] = c.b;
  }
  g.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  g.computeVertexNormals();
  return g;
}

// A stone pine for the skyline: tall bare trunk, flat umbrella crown.
function pineGeometry(r) {
  var parts = [tinted(new THREE.CylinderGeometry(0.18, 0.32, 9, 6).translate(0, 4.5, 0), '#2a241e')];
  for (var k = 0; k < 9; k++) {
    var a = r() * 6.28, d = k ? 1.4 + r() * 2.2 : 0;
    parts.push(tinted(new THREE.IcosahedronGeometry(1.6 + r() * 1.1, 1).scale(1, 0.42, 1)
      .translate(Math.cos(a) * d, 9.2 + r() * 0.8, Math.sin(a) * d), '#ffffff'));
  }
  return merge(parts);
}

// A plinth: moulded base, die and cap. Returns its geometries (tinted, for
// merging) and the height of its top.
function plinthParts(w, h, d, x, z, out) {
  out.push(tinted(new THREE.BoxGeometry(w * 1.2, 0.22, d * 1.2).translate(x, 0.11, z), '#ffffff'));
  out.push(tinted(new THREE.BoxGeometry(w * 1.08, 0.08, d * 1.08).translate(x, 0.26, z), '#ffffff'));
  out.push(tinted(new THREE.BoxGeometry(w, h, d).translate(x, 0.3 + h / 2, z), '#ffffff'));
  out.push(tinted(new THREE.BoxGeometry(w * 1.1, 0.07, d * 1.1).translate(x, 0.3 + h + 0.035, z), '#ffffff'));
  out.push(tinted(new THREE.BoxGeometry(w * 1.18, 0.1, d * 1.18).translate(x, 0.3 + h + 0.12, z), '#ffffff'));
  return 0.3 + h + 0.17;
}

// A garden urn on a short foot, by lathe.
function urnGeometry(s) {
  var prof = [[0.001, 0], [0.16, 0], [0.16, 0.05], [0.09, 0.08], [0.07, 0.16], [0.12, 0.2], [0.24, 0.3], [0.3, 0.44], [0.28, 0.58],
              [0.2, 0.66], [0.17, 0.7], [0.24, 0.74], [0.25, 0.78], [0.2, 0.79], [0.18, 0.72], [0.001, 0.72]];
  var pts = new THREE.SplineCurve(prof.map(function (p) { return new THREE.Vector2(p[0] * s, p[1] * s); })).getSpacedPoints(60);
  return new THREE.LatheGeometry(pts, 28);
}

// One of the lovers: an abstract veiled form, its cloth falling in folds
// that sweep sideways near the hem as if the dress were caught mid-flow,
// leaning in (side = +1 is the right-hand form, leaning to -x).
function veiledForm(H, scale, side, seed, rows, cols) {
  var r = rng(seed);
  var prof = [[0.5, 0.0], [0.47, 0.04], [0.4, 0.17], [0.33, 0.33], [0.285, 0.48], [0.272, 0.6], [0.285, 0.69], [0.255, 0.765],
              [0.185, 0.815], [0.17, 0.855], [0.172, 0.905], [0.14, 0.96], [0.07, 0.993], [0.0, 1.0]];
  var curve = new THREE.CatmullRomCurve3(prof.map(function (p) { return new THREE.Vector3(p[0] * scale, p[1], 0); }));
  var pt = new THREE.Vector3(), ph = [r() * 6.28, r() * 6.28, r() * 6.28, r() * 6.28];
  var pos = new Float32Array((rows * cols + 1) * 3), idx = [], v = 0;
  var awayA = side > 0 ? 0 : Math.PI;        // the outside, away from the partner
  for (var j = 0; j < rows; j++) {
    curve.getPointAt(j / rows, pt);
    var t = pt.y, base = pt.x, low = Math.pow(1 - t, 1.6);
    for (var i = 0; i < cols; i++) {
      var th = i / cols * Math.PI * 2;
      // Folds run down the cloth, deepening and swirling towards the hem.
      var fl = th + low * 1.1 * side + 0.25 * Math.sin(t * 9 + ph[3]);
      var fold = 0.55 * Math.sin(8 * fl + ph[0] + t * 2) + 0.3 * Math.sin(15 * fl + ph[1] - t * 6) + 0.15 * Math.sin(27 * fl + ph[2]);
      fold = (fold < 0 ? -1 : 1) * Math.pow(Math.abs(fold), 0.75);
      var amp = 0.008 + 0.095 * low * smooth(0.0, 0.08, t + 0.02);
      var rad = base + amp * fold;
      // A veil over head and shoulders, ending in a wavy hem at the waist.
      var hemT = 0.6 + 0.035 * Math.sin(3 * th + ph[1]) + 0.018 * Math.sin(7 * th + ph[2]);
      rad += 0.026 * scale * smooth(hemT - 0.012, hemT + 0.006, t) * (1 - smooth(0.9, 1.0, t));
      rad += 0.012 * Math.sin(11 * th + ph[0]) * smooth(hemT, hemT + 0.1, t) * (1 - smooth(0.78, 0.86, t));
      // The skirt sweeps out on the outside, frozen mid-swirl.
      var away = Math.max(0, Math.cos(th - awayA));
      rad *= 1 + 0.42 * Math.pow(1 - t, 2.6) * away * away;
      var x = Math.cos(th) * rad * 0.8, z = Math.sin(th) * rad * 1.05;
      x += -side * 0.3 * Math.pow(t, 2.2) + side * 0.1 * Math.pow(1 - t, 3);
      pos[v++] = x; pos[v++] = t * H; pos[v++] = z;
      if (j < rows - 1) {
        var a = j * cols + i, b = j * cols + (i + 1) % cols, c = a + cols, d = b + cols;
        idx.push(a, c, b, b, c, d);
      } else {
        idx.push(j * cols + i, rows * cols, j * cols + (i + 1) % cols);
      }
    }
  }
  pos[v++] = -side * 0.3; pos[v++] = H; pos[v++] = 0;
  var g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

// A rose petal: a small cupped oval.
function petalGeometry() {
  var g = new THREE.CircleGeometry(0.032, 10).scale(1.25, 1, 1);
  var p = g.attributes.position;
  for (var i = 0; i < p.count; i++) {
    var x = p.getX(i), y = p.getY(i);
    p.setZ(i, (x * x + y * y * 0.6) * 9);
  }
  g.computeVertexNormals();
  return g;
}

function moonTexture(r) {
  return canvasTex(256, function (x, s) {
    x.clearRect(0, 0, s, s);
    x.save();
    x.beginPath();
    x.arc(s / 2, s / 2, s * 0.3, 0, Math.PI * 2);
    x.clip();
    x.fillStyle = '#f4f2ea';
    x.fillRect(0, 0, s, s);
    for (var i = 0; i < 26; i++) {           // maria
      x.fillStyle = 'rgba(150,155,170,' + (0.08 + r() * 0.16) + ')';
      x.beginPath();
      x.arc(s * (0.32 + r() * 0.36), s * (0.3 + r() * 0.4), s * (0.02 + r() * 0.07), 0, Math.PI * 2);
      x.fill();
    }
    x.restore();
    var g = x.createRadialGradient(s / 2, s / 2, s * 0.29, s / 2, s / 2, s / 2);
    g.addColorStop(0, 'rgba(220,228,255,0.55)');
    g.addColorStop(0.25, 'rgba(200,210,250,0.12)');
    g.addColorStop(1, 'rgba(200,210,250,0)');
    x.fillStyle = g;
    x.fillRect(0, 0, s, s);
  });
}

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(1720), n3 = noise3(r);
  var gl = makeRenderer(canvas, { shadows: !small, clear: '#0b1222' });
  gl.toneMappingExposure = 1.3;
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#101a30', 0.017);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.1, 3000);

  // Uniforms shared by every material that frosts over.
  var U = {
    uFrost: { value: 0 }, uCold: { value: 0 }, uCenter: { value: CENTER },
    uHeartPos: { value: new THREE.Vector3() }, uHeartCol: { value: new THREE.Color(0, 0, 0) }
  };

  // ── Sky: deep blue night, stars, a large moon with drifting cloud ──────
  var sky = new THREE.Group();
  world.add(sky);
  var MOON_DIR = new THREE.Vector3(-0.16, 0.33, -0.93).normalize();
  var dome = skyDome({ top: '#040814', mid: '#0c1630', horizon: '#26365a', sun: '#2c3a60' }, 1500);
  dome.uniforms.sunDir.value.copy(MOON_DIR);
  sky.add(dome.mesh);
  var stars = starField(r, small ? 1400 : 2800, 1300, 0.04, 1.4);
  sky.add(stars);
  var moon = new THREE.Sprite(new THREE.SpriteMaterial({ map: moonTexture(r), transparent: true, depthWrite: false, fog: false }));
  moon.position.copy(MOON_DIR).multiplyScalar(1000);
  moon.scale.setScalar(150);
  var moonHalo = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(170,190,240,0.5)', 'rgba(120,140,200,0)'),
    blending: THREE.AdditiveBlending, transparent: true, depthWrite: false, fog: false, opacity: 0.45 }));
  moonHalo.position.copy(moon.position);
  moonHalo.scale.setScalar(620);
  sky.add(moonHalo, moon);
  var cloudTex = softSprite('rgba(120,135,170,0.75)', 'rgba(120,135,170,0)'), clouds = [];
  for (var i = 0; i < 9; i++) {
    var cl = new THREE.Sprite(new THREE.SpriteMaterial({ map: cloudTex, transparent: true, depthWrite: false, fog: false, opacity: 0.35 + r() * 0.25 }));
    cl.userData = { a: -0.9 + r() * 1.2, y: 260 + r() * 260, s: 0.004 + r() * 0.004 };
    cl.scale.set(380 + r() * 340, 60 + r() * 40, 1);
    sky.add(cl);
    clouds.push(cl);
  }

  // ── Light: moon from high on the left, a blue fill, the heart ──────────
  var hemi = new THREE.HemisphereLight('#5c6f9e', '#182030', 0.9);
  var moonLight = new THREE.DirectionalLight('#c4d2ff', 2.1);
  moonLight.position.set(-16, 17, -4).add(CENTER);
  moonLight.target.position.copy(CENTER).setZ(-6);
  moonLight.castShadow = !small;
  moonLight.shadow.mapSize.set(2048, 2048);
  moonLight.shadow.camera.left = -26; moonLight.shadow.camera.right = 26;
  moonLight.shadow.camera.bottom = -30; moonLight.shadow.camera.top = 30;
  moonLight.shadow.camera.far = 90;
  moonLight.shadow.bias = -0.0006;
  moonLight.shadow.normalBias = 0.03;
  world.add(hemi, moonLight, moonLight.target);

  // ── Ground: lawn, gravel allée and the round garden's gravel ───────────
  var lawnTex = canvasTex(256, speckle(r, '#24361f', ['#2c4226', '#1c2c18', '#33492b', '#1a2616'], 2600, 0.6, 2.2), [110, 110]);
  var lawnMat = frosty(new THREE.MeshStandardMaterial({ map: lawnTex, color: '#7c8a74', roughness: 1 }), U, 0.9, 1.0);
  var lawn = new THREE.Mesh(new THREE.PlaneGeometry(400, 400).rotateX(-Math.PI / 2), lawnMat);
  lawn.receiveShadow = true;
  world.add(lawn);
  var gravelTex = canvasTex(256, speckle(r, '#8d8a82', ['#a29e94', '#77736b', '#b4b0a6', '#6a675f', '#99958b'], 4200, 0.6, 1.7), [1, 1]);
  var gravelMat = frosty(new THREE.MeshStandardMaterial({ map: gravelTex, color: '#a2a2a6', roughness: 0.95 }), U, 0.6, 1.0);
  var allee = new THREE.PlaneGeometry(3.6, 42).rotateX(-Math.PI / 2).translate(0, 0.012, 7.4);
  var uvA = allee.attributes.uv;
  for (i = 0; i < uvA.count; i++) uvA.setXY(i, uvA.getX(i) * 3.6 / 2.5, uvA.getY(i) * 42 / 2.5);
  var walk = new THREE.Mesh(allee, gravelMat);
  walk.receiveShadow = true;
  var disc = new THREE.CircleGeometry(RING - 0.5, 64).rotateX(-Math.PI / 2).translate(CENTER.x, 0.022, CENTER.z);
  var uvD = disc.attributes.uv;
  for (i = 0; i < uvD.count; i++) uvD.setXY(i, uvD.getX(i) * 5.6, uvD.getY(i) * 5.6);
  var round = new THREE.Mesh(disc, gravelMat);
  round.receiveShadow = true;
  world.add(walk, round);

  // ── Hedges: the allée's low box hedges, the ring, the tall enclosure ───
  var leafTex = canvasTex(256, speckle(r, '#1d2e1a', ['#2c4527', '#16241a', '#355430', '#22381f', '#3d5e36'], 3400, 1.2, 3.6), [1, 1]);
  var hedgeMat = frosty(new THREE.MeshStandardMaterial({ map: leafTex, bumpMap: leafTex, bumpScale: 3, color: '#9aa894', roughness: 0.95 }), U, 0.8, 1.6);
  var hedges = [], PLINTHS_Z = [20, 13, 6, -1, -8];
  [-1, 1].forEach(function (sd) {
    var zs = [27].concat(PLINTHS_Z).concat([-12.8]);
    for (var k = 0; k < zs.length - 1; k++) {
      var z0 = zs[k] - (k ? 0.75 : 0), z1 = zs[k + 1] + (k + 1 < zs.length - 1 ? 0.75 : 0), len = z0 - z1;
      hedges.push(hedgeBox(0.8, 0.95, len, sd * 2.4, (z0 + z1) / 2, 0));
    }
    hedges.push(hedgeBox(1.4, 2.6, 42, sd * 8.2, 6.5, 0));     // the tall enclosure along the allée
  });
  hedges.push(ringHedge(RING - 0.4, RING + 0.4, 0.95, 0.27, 72));
  hedges.push(ringHedge(12.5, 13.9, 2.8, 0.62, 72));
  var hedgeMesh = new THREE.Mesh(mergeUV(hedges), hedgeMat);
  hedgeMesh.castShadow = hedgeMesh.receiveShadow = !small;
  world.add(hedgeMesh);

  // ── Cypresses along the allée and round the garden, swaying a little ───
  var clock = { value: 0 };
  var cypMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.92, color: '#ffffff' });
  cypMat.userData.sway = function (sh) {
    sh.uniforms.uClock = clock;
    sh.vertexShader = 'uniform float uClock;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n float cph = instanceMatrix[3][0] * 0.7 + instanceMatrix[3][2] * 0.4;\n' +
      ' transformed.x += sin(uClock * 0.9 + cph) * 0.012 * position.y * position.y;');
  };
  frosty(cypMat, U, 0.5, 1.4);
  var spots = [];
  for (var z = 25; z > -11; z -= 4.5) { spots.push([-4.4, z], [4.4, z]); }
  for (i = 0; i < 15; i++) {
    var ca = (i + 0.5) / 15 * Math.PI * 2;
    if (Math.abs(Math.atan2(Math.sin(ca), Math.cos(ca))) < 0.5) continue;
    spots.push([CENTER.x + Math.sin(ca) * 10.2, CENTER.z + Math.cos(ca) * 10.2]);
  }
  var cypress = new THREE.InstancedMesh(cypressGeometry(n3), cypMat, spots.length), up = new THREE.Vector3(0, 1, 0);
  scatter(cypress, spots.length, function (k, p, q, s, c) {
    p.set(spots[k][0] + (r() - 0.5) * 0.3, 0, spots[k][1] + (r() - 0.5) * 0.3);
    q.setFromAxisAngle(up, r() * 6.28);
    var hgt = 7.5 + r() * 2.5;
    s.set(hgt * (0.85 + r() * 0.3), hgt, hgt * (0.85 + r() * 0.3));
    c.setHSL(0.3 + r() * 0.04, 0.32, 0.13 + r() * 0.04);
  });
  cypress.castShadow = !small;
  world.add(cypress);

  // Stone pines on the skyline beyond the walls.
  var pines = new THREE.InstancedMesh(pineGeometry(r), new THREE.MeshLambertMaterial({ vertexColors: true }), small ? 14 : 24);
  scatter(pines, 400, function (k, p, q, s, c) {
    var a = r() * Math.PI * 2, d = 24 + r() * 26;
    p.set(Math.sin(a) * d, 0, -14 + Math.cos(a) * d * 1.4);
    if (Math.abs(p.x) < 12 && p.z > -30) return false;
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar(0.85 + r() * 0.5);
    c.setHSL(0.3, 0.3, 0.045 + r() * 0.025);
  });
  world.add(pines);

  // ── Marble: plinths with urns and spheres, and the lovers' plinth ──────
  var marbleGarden = marbleMaterial(U, { vein: 1.4, glow: 0.7 });
  var stone = [];
  PLINTHS_Z.forEach(function (pz, k) {
    [-1, 1].forEach(function (sd) {
      var top = plinthParts(0.62, 0.95, 0.62, sd * 2.4, pz, stone);
      if ((k + (sd > 0 ? 1 : 0)) % 2 === 0) {
        stone.push(tinted(urnGeometry(1.05).translate(sd * 2.4, top, pz), '#ffffff'));
      } else {
        stone.push(tinted(new THREE.CylinderGeometry(0.12, 0.18, 0.12, 20).translate(sd * 2.4, top + 0.06, pz), '#ffffff'));
        stone.push(tinted(new THREE.SphereGeometry(0.3, 28, 18).translate(sd * 2.4, top + 0.42, pz), '#ffffff'));
      }
    });
  });
  // Two urns stand behind the lovers, on the diagonals.
  [2.3, -2.3].forEach(function (a) {
    var ux = CENTER.x + Math.sin(a) * 5.4, uz = CENTER.z + Math.cos(a) * 5.4;
    var top = plinthParts(0.58, 0.8, 0.58, ux, uz, stone);
    stone.push(tinted(urnGeometry(0.95).translate(ux, top, uz), '#ffffff'));
  });
  stone.push(tinted(new THREE.BoxGeometry(3.2, 0.24, 2.2).translate(CENTER.x, 0.12, CENTER.z), '#ffffff'));
  stone.push(tinted(new THREE.BoxGeometry(2.9, 0.1, 1.95).translate(CENTER.x, 0.29, CENTER.z), '#ffffff'));
  stone.push(tinted(new THREE.BoxGeometry(2.5, 0.88, 1.6).translate(CENTER.x, 0.78, CENTER.z), '#ffffff'));
  stone.push(tinted(new THREE.BoxGeometry(2.7, 0.08, 1.75).translate(CENTER.x, 1.26, CENTER.z), '#ffffff'));
  stone.push(tinted(new THREE.BoxGeometry(2.85, 0.1, 1.85).translate(CENTER.x, 1.35, CENTER.z), '#ffffff'));
  var stoneMesh = new THREE.Mesh(merge(stone), marbleGarden);
  stoneMesh.castShadow = stoneMesh.receiveShadow = !small;
  world.add(stoneMesh);

  // The lovers: two veiled forms on one plinth, leaning in until they fuse.
  var marbleLovers = marbleMaterial(U, { vein: 1.9, glow: 1.0 });
  var rows = small ? 70 : 120, cols = small ? 72 : 128;
  var lovers = new THREE.Group();
  var right = new THREE.Mesh(veiledForm(1.95, 1.0, 1, 7, rows, cols), marbleLovers);
  right.position.set(0.4, 0, 0);
  var left = new THREE.Mesh(veiledForm(1.8, 0.94, -1, 11, rows, cols), marbleLovers);
  left.position.set(-0.4, 0, 0.04);
  lovers.add(right, left);
  lovers.position.set(CENTER.x, PLINTH_TOP, CENTER.z);
  lovers.rotation.y = 0.12;
  lovers.children.forEach(function (m) { m.castShadow = m.receiveShadow = !small; });
  world.add(lovers);

  // ── Rose petals round the lovers, which drift, then hang frozen ────────
  var NP = small ? 90 : 180, petalMat = new THREE.MeshStandardMaterial({ color: '#c8304c', roughness: 0.6, side: THREE.DoubleSide,
                                                                          emissive: '#4a0a16' });
  var petals = new THREE.InstancedMesh(petalGeometry(), petalMat, NP);
  petals.frustumCulled = false;
  var pp = new Float32Array(NP * 3), pAxis = [], pSpin = new Float32Array(NP), pPh = new Float32Array(NP), pFall = new Float32Array(NP);
  var BOX = [9, 5.2, 9];
  for (i = 0; i < NP; i++) {
    pp[i * 3] = (r() - 0.5) * BOX[0];
    pp[i * 3 + 1] = 0.3 + r() * BOX[1];
    pp[i * 3 + 2] = (r() - 0.5) * BOX[2];
    pAxis.push(new THREE.Vector3(r() - 0.5, r() - 0.5, r() - 0.5).normalize());
    pSpin[i] = 1 + r() * 2.5;
    pPh[i] = r() * 6.28;
    pFall[i] = 0.25 + r() * 0.3;
  }
  world.add(petals);

  // ── The heart: one warm light that beats ───────────────────────────────
  var heartTex = softSprite('rgba(255,196,150,1)', 'rgba(255,110,70,0)');
  var heartCore = new THREE.Sprite(new THREE.SpriteMaterial({ map: heartTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
  var heartHalo = new THREE.Sprite(new THREE.SpriteMaterial({ map: heartTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
  var heartAura = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,150,100,0.6)', 'rgba(255,90,60,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
  var heartLight = new THREE.PointLight('#ff9a66', 0, 7, 1.8);
  var heart = new THREE.Group();
  heart.add(heartAura, heartHalo, heartCore, heartLight);
  heart.position.copy(HEART);
  world.add(heart);

  // ── Per-frame temps ────────────────────────────────────────────────────
  var tmp = new THREE.Color(), tmp2 = new THREE.Color(), look = new THREE.Vector3(), hv = new THREE.Vector3();
  var m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s3 = new THREE.Vector3(), p3 = new THREE.Vector3();
  var flowTime = 0, portrait = false, focus = new THREE.Vector3();
  var focusHeart = new THREE.Vector3((HEART.x + CENTER.x) / 2, 2.4, (HEART.z + CENTER.z) / 2);

  function wrap(v, lo, size) { return lo + ((((v - lo) % size) + size) % size); }

  function frame(f) {
    var row = f.row, time = f.time, dt = f.dt;
    var cold = row[K.cold], frost = row[K.frost], still = row[K.still], heartAmt = row[K.heart], petalAmt = row[K.petals];
    var flow = (1 - still) * (env.reduceMotion ? 0.4 : 1);
    flowTime += dt * flow;                  // the garden's own time, which stops
    clock.value = env.reduceMotion ? 0 : flowTime;

    // ── Camera: authored position and target, a little pointer look ─────
    camera.position.set(row[K.camX], row[K.camY] + Math.sin(time * 0.9) * 0.015 * flow, row[K.camZ]);
    look.set(row[K.lookX], row[K.lookY], row[K.lookZ]);
    if (portrait) {
      // Portrait text sits mid-screen: turn a subject held to one side
      // (the lovers, then the lovers and the heart) back into the middle,
      // but not once the view rises over the garden.
      focus.set(CENTER.x, 2.4, CENTER.z).lerp(focusHeart, heartAmt);
      look.lerp(focus, (0.55 + 0.45 * heartAmt) * (1 - smooth(2.4, 6, row[K.camY])));
      look.y += 0.3 + 0.5 * heartAmt;
    }
    camera.lookAt(look);
    camera.rotateY(-f.mx * 0.1);
    camera.rotateX(-f.my * 0.05);
    sky.position.copy(camera.position);

    // ── Night grade: deep blue, colder as the frost comes ──────────────
    dome.uniforms.top.value.set('#040814').lerp(tmp.set('#030612'), cold);
    dome.uniforms.mid.value.set('#0c1630').lerp(tmp.set('#0a1836'), cold);
    var horizon = tmp2.set('#26365a').lerp(tmp.set('#2a4470'), cold);
    dome.uniforms.horizon.value.copy(horizon);
    world.fog.color.copy(horizon).multiplyScalar(0.48);
    gl.setClearColor(world.fog.color);
    hemi.color.set('#5c6f9e').lerp(tmp.set('#6a88c4'), cold);
    moonLight.color.set('#c4d2ff').lerp(tmp.set('#b4ccff'), cold);
    U.uFrost.value = frost;
    U.uCold.value = cold;

    // Clouds cross the moon, until time stops.
    for (var c = 0; c < clouds.length; c++) {
      var cu = clouds[c].userData, a = cu.a + flowTime * cu.s;
      a = -1.3 + ((a + 1.3) % 1.6 + 1.6) % 1.6;
      clouds[c].position.set(Math.sin(a - 0.4) * 1000, cu.y, -Math.cos(a - 0.4) * 1000);
    }

    // ── Petals: drifting round the lovers, then hung in the air ────────
    var np = Math.floor(NP * clamp(petalAmt, 0, 1));
    for (var i = 0; i < np; i++) {
      var k = i * 3;
      pp[k] += (0.22 + Math.sin(flowTime * 0.7 + pPh[i]) * 0.3) * dt * flow;
      pp[k + 1] -= pFall[i] * dt * flow;
      pp[k + 2] += Math.cos(flowTime * 0.5 + pPh[i]) * 0.2 * dt * flow;
      pp[k] = wrap(pp[k], -BOX[0] / 2, BOX[0]);
      pp[k + 1] = wrap(pp[k + 1], 0.15, BOX[1]);
      pp[k + 2] = wrap(pp[k + 2], -BOX[2] / 2, BOX[2]);
      p3.set(CENTER.x + 0.6 + pp[k], pp[k + 1], CENTER.z + 2 + pp[k + 2]);
      q.setFromAxisAngle(pAxis[i], pPh[i] + flowTime * pSpin[i]);
      // Thin out the petals right in front of the lens, which read as blots.
      var near = p3.distanceToSquared(camera.position);
      // And keep the heart clear, where a petal would show as a black blot.
      var byHeart = p3.distanceToSquared(HEART) < 0.6;
      s3.setScalar(near < 2.5 || byHeart ? 0.0001 : 1.1);
      petals.setMatrixAt(i, m4.compose(p3, q, s3));
    }
    petals.count = np;
    petals.instanceMatrix.needsUpdate = true;
    petalMat.color.set('#c8304c').lerp(tmp.set('#a8788c'), frost * 0.55);
    petalMat.emissive.set('#4a0a16').lerp(tmp.set('#141c30'), frost);

    // ── The heart: a double beat, about once a second ──────────────────
    var ph = (time * 1.05) % 1;
    var beat = Math.exp(-Math.pow((ph - 0.08) * 13, 2)) + 0.65 * Math.exp(-Math.pow((ph - 0.3) * 13, 2));
    var hs = heartAmt * (0.85 + 0.25 * beat);
    heart.visible = heartAmt > 0.002;
    heart.position.set(HEART.x, HEART.y + Math.sin(time * 0.6) * 0.04, HEART.z);
    heartCore.scale.setScalar(0.5 * hs + 0.001);
    heartHalo.scale.setScalar(1.9 * hs + 0.001);
    heartAura.scale.setScalar(5.5 * hs + 0.001);
    heartCore.material.opacity = Math.min(1, heartAmt * 1.2);
    heartHalo.material.opacity = heartAmt * (0.75 + 0.25 * beat);
    heartAura.material.opacity = heartAmt * (0.2 + 0.2 * beat);
    heartLight.intensity = heartAmt * (0.8 + 1.2 * beat);
    hv.copy(heart.position).applyMatrix4(camera.matrixWorldInverse);
    U.uHeartPos.value.copy(hv);
    U.uHeartCol.value.set('#ff8a50').multiplyScalar(heartAmt * (0.08 + 0.1 * beat));

    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) { portrait = w / h < 1; fitCamera(gl, camera, w, h, dpr, small); },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

// A soft heartbeat: low thumps in pairs, lub-dub, slowing and fading.
function heartbeat(ac, out) {
  var now = ac.currentTime;
  for (var b = 0; b < 6; b++) {
    var t0 = now + b * 0.95, amp = 0.55 * (1 - b * 0.13);
    [0, 0.27].forEach(function (d, k) {
      var t = t0 + d, o = ac.createOscillator(), g = ac.createGain(), lp = ac.createBiquadFilter();
      o.type = 'sine';
      o.frequency.setValueAtTime(k ? 64 : 56, t);
      o.frequency.exponentialRampToValueAtTime(34, t + 0.16);
      lp.type = 'lowpass';
      lp.frequency.value = 180;
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(amp * (k ? 0.7 : 1), t + 0.014);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.24);
      o.connect(lp);
      lp.connect(g);
      g.connect(out);
      o.start(t);
      o.stop(t + 0.26);
    });
  }
}

PI.register('marble-garden', {
  renderer: renderer3d,
  maxLines: 6,
  scrim: 0.6,
  align: ['left', 'right', 'left', 'right'],
  // Panels: 0 "If lovers turned to stone / would you still envy them",
  // 1 "Mere statues ... lips of marble?", 2 "And the frozen folds ...
  // better petrified", 3 "This beating heart ... made of stone".
  keys: function (T) {
    var n = T.count;
    function at(i, frac) { i = Math.min(i, n - 1); return lerp(T.start(i), T.end(i), frac); }
    //   unit         camZ   cold petals wind camX  camY  lookX lookY lookZ  frost still heart
    return [
      [0,             31.0,  0.0, 0.0,  0.3, 0.0,  1.7, -0.8, 4.6, -20.0, 0.0, 0.0, 0.0],
      [0.7,           30.0,  0.0, 0.0,  0.3, 0.0,  1.7, -0.8, 4.2, -20.0, 0.0, 0.0, 0.0],
      [at(0, 0.15),   25.0,  0.0, 0.0,  0.3, -0.2, 1.7, -1.6, 3.0, -20.0, 0.0, 0.0, 0.0],  // "If lovers turned to stone"
      [at(0, 0.95),   12.0,  0.0, 0.1,  0.3, -0.3, 1.7, -1.8, 2.8, -20.0, 0.0, 0.0, 0.0],  // "the grandeur of their solid smirks"
      [at(1, 0.2),     5.0,  0.0, 0.3,  0.3, 0.5,  1.7, -1.4, 1.9, -1.0, 0.0, 0.0, 0.0],   // "mere statues in your passing"
      [at(1, 0.5),    -4.5,  0.0, 0.6,  0.3, 0.0,  1.7,  1.2, 2.8, -20.0, 0.0, 0.0, 0.0],  // "would you stop and wonder"
      [at(1, 0.85),  -14.0,  0.0, 0.85, 0.25, -1.1, 1.6, 1.4, 3.2, -20.0, 0.0, 0.0, 0.0],  // "lips of marble?"
      [at(1, 1.0),   -14.3,  0.0, 0.9,  0.25, -1.1, 1.6, 1.4, 3.2, -20.0, 0.0, 0.0, 0.0],
      [at(2, 0.2),   -17.7,  0.1, 1.0,  0.25, 1.3,  2.3, -0.6, 2.0, -20.0, 0.08, 0.05, 0.0], // "the frozen folds of a flowing dress"
      [at(2, 0.5),   -18.0,  0.6, 1.0,  0.15, 1.15, 2.25, -0.6, 2.05, -20.0, 0.5, 0.5, 0.0], // "cold against your fingertips"
      [at(2, 0.85),  -18.1,  1.0, 1.0,  0.0, 1.1,  2.3, -0.55, 2.1, -20.0, 0.85, 1.0, 0.0], // "better petrified"
      [at(3, 0.12),  -14.2,  1.0, 1.0,  0.0, -2.3, 2.1,  3.2, 2.6, -20.0, 0.9, 1.0, 0.7],  // "this beating heart"
      [at(3, 0.9),   -13.6,  1.0, 1.0,  0.0, -2.5, 2.15, 3.2, 2.65, -20.0, 0.92, 1.0, 1.0], // "without a lover made of stone"
      [T.total,       -2.0,  1.0, 1.0,  0.0, 0.8,  6.5,  0.6, 6.2, -20.0, 1.0, 1.0, 1.0]
    ];
  },
  sound: {
    src: '/audio/wind.mp3',
    label: 'Play the night wind and a heartbeat',
    volume: function (row) { return 0.05 + 0.2 * row[K.wind]; },
    cues: [
      { stanza: 3, at: 0.2, play: heartbeat },
      { stanza: 3, at: 1.45, play: heartbeat }
    ]
  }
});
