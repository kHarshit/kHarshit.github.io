/*
 * Scene for "Faults" (Sara Teasdale): kintsugi. A single porcelain bowl
 * turns slowly under a warm, low lamp on a linen-covered table in a dark room.
 *
 * The six lines have no stanza breaks. maxLines 2 splits them into three
 * panels of two (maxLines 3 would split them 2 + 4, at the semicolon), one
 * for each turn of the poem.
 * Opening  the lamp comes up on a flawless bowl.
 * I    "They came to tell your faults to me, / They named them over one by
 *      one": hairline cracks run through the glaze one at a time, each with
 *      a faint tick, and a chip opens at the rim; the light cools a little.
 * II   "I laughed aloud when they were done": a warm lift of lamplight.
 *      "I knew them all so well before": close in over the rim, where the
 *      cracks are traced in faint amber, known by heart.
 * III  "Oh, they were blind, too blind to see / Your faults had made me love
 *      you more": molten gold runs down every crack from the rim, glowing
 *      at its front, then cools into raised, gleaming seams, and the bowl is
 *      lovelier than it was.
 * Outro the camera rises as the bowl turns, the gold catching the lamp.
 *
 * The cracks are a procedural network painted once into two canvas textures
 * over the lathe's UVs: a distance ridge (R; B marks the chip) and a timing
 * map (R: when it cracks, G: when the gold reaches it). The glaze shader
 * cuts the ridge thin for a hairline crack, or wide and domed for a seam.
 * Columns (see COLS): [unit, dist, gloom, motes, wind, az, el, ly, side,
 *   lamp, lift, crack, know, gold, heat, spin]
 */
import { THREE, isSmall, makeRenderer, fitCamera, softSprite, particleField, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

var COLS = ['dist', 'gloom', 'motes', 'wind', 'az', 'el', 'ly', 'side', 'lamp', 'lift', 'crack', 'know', 'gold', 'heat', 'spin'];
var K = {};
COLS.forEach(function (c, i) { K[c] = i; });

// ── The bowl (metres-ish: rim radius 1) ──────────────────────────────────
// Profile from the centre of the foot, out and up the outer wall, over the
// rounded lip and down the inside to the centre of the well.
var PROFILE = [[0, 0.036], [0.2, 0.037], [0.33, 0.031], [0.362, 0.012], [0.385, 0.0], [0.43, 0.0], [0.452, 0.016],
  [0.462, 0.062], [0.49, 0.09], [0.6, 0.142], [0.75, 0.245], [0.87, 0.38], [0.95, 0.53], [0.988, 0.66], [1.0, 0.745],
  [0.995, 0.772], [0.978, 0.781], [0.962, 0.768], [0.955, 0.73], [0.944, 0.62], [0.895, 0.485], [0.815, 0.355],
  [0.68, 0.228], [0.5, 0.142], [0.3, 0.104], [0.12, 0.095], [0, 0.094]];

var NCRACK = 7;            // six cracks and a chip, named one by one
var CRACK_AT = 0.32;       // into panel I, when the first one runs
var CRACK_GAP = 0.165;     // between them
var VIEW_AZ = 0.2;         // the camera's bearing while they are named
var HW = 0.026;            // half-width of the painted distance ridge
var TC = 0.86;             // ridge level of a hairline crack's edge
var TG = 0.45;             // ridge level of a gold seam's edge

function buildProfile(n) {
  var curve = new THREE.CatmullRomCurve3(PROFILE.map(function (p) { return new THREE.Vector3(p[0], p[1], 0); }), false, 'centripetal');
  var pts = curve.getSpacedPoints(n - 1).map(function (p) { return new THREE.Vector2(Math.max(p.x, 0), p.y); });
  pts[0].x = 0; pts[n - 1].x = 0;
  var len = curve.getLength(), rim = 0, foot = 0, well = n - 1;
  for (var j = 0; j < n; j++) if (pts[j].y > pts[rim].y) rim = j;
  // The glaze stops a little above the foot; the well is where the inner wall flattens.
  for (j = rim; j > 0; j--) if (pts[j].y < 0.075 && pts[j].x > 0.44) { foot = j; break; }
  for (j = rim; j < n; j++) if (pts[j].x < 0.24) { well = j; break; }
  return { pts: pts, len: len, n: n, rimV: rim / (n - 1), footV: foot / (n - 1), wellV: well / (n - 1) };
}

// A hand-thrown bowl: a little out of round, the lip rising and dipping.
function bowlGeometry(prof, seg) {
  var geo = new THREE.LatheGeometry(prof.pts, seg);
  var pos = geo.attributes.position, P = prof.n, v = new THREE.Vector3();
  for (var i = 0; i <= seg; i++) {
    var phi = i / seg * Math.PI * 2;
    var round = 0.006 * Math.sin(2 * phi + 0.7) + 0.004 * Math.sin(3 * phi + 2.1);
    var lip = 0.007 * Math.sin(3 * phi + 1.3) + 0.004 * Math.sin(5 * phi + 0.4);
    for (var j = 0; j < P; j++) {
      var k = i * P + j;
      v.fromBufferAttribute(pos, k);
      var up = smooth(0.1, 0.78, v.y);
      v.x *= 1 + round * up;
      v.z *= 1 + round * up;
      v.y += lip * smooth(0.55, 0.78, v.y);
      pos.setXYZ(k, v.x, v.y, v.z);
    }
  }
  geo.computeVertexNormals();
  // Weld the seam's normals, and point the two centres straight down / up.
  var nrm = geo.attributes.normal, a = new THREE.Vector3(), b = new THREE.Vector3();
  for (j = 0; j < P; j++) {
    a.fromBufferAttribute(nrm, j);
    b.fromBufferAttribute(nrm, seg * P + j);
    a.add(b).normalize();
    nrm.setXYZ(j, a.x, a.y, a.z);
    nrm.setXYZ(seg * P + j, a.x, a.y, a.z);
  }
  for (i = 0; i <= seg; i++) { nrm.setXYZ(i * P, 0, -1, 0); nrm.setXYZ(i * P + P - 1, 0, 1, 0); }
  // dP/du and dP/dv as duals (P_u / |P_u|^2), so the shader can turn a
  // height field painted over the UVs into a surface gradient.
  var du = new Float32Array(pos.count * 3), dv = new Float32Array(pos.count * 3), c = new THREE.Vector3();
  for (i = 0; i <= seg; i++) {
    var ip = i === seg ? 1 : i + 1, im = i === 0 ? seg - 1 : i - 1;
    for (j = 0; j < P; j++) {
      k = i * P + j;
      a.fromBufferAttribute(pos, ip * P + j);
      b.fromBufferAttribute(pos, im * P + j);
      c.subVectors(a, b).multiplyScalar(seg / 2);
      var l2 = c.lengthSq();
      if (l2 > 1e-5) c.divideScalar(l2); else c.set(0, 0, 0);
      du[k * 3] = c.x; du[k * 3 + 1] = c.y; du[k * 3 + 2] = c.z;
      var jp = Math.min(j + 1, P - 1), jm = Math.max(j - 1, 0);
      a.fromBufferAttribute(pos, i * P + jp);
      b.fromBufferAttribute(pos, i * P + jm);
      c.subVectors(a, b).multiplyScalar((P - 1) / (jp - jm));
      l2 = c.lengthSq();
      if (l2 > 1e-5) c.divideScalar(l2); else c.set(0, 0, 0);
      dv[k * 3] = c.x; dv[k * 3 + 1] = c.y; dv[k * 3 + 2] = c.z;
    }
  }
  geo.setAttribute('aPu', new THREE.BufferAttribute(du, 3));
  geo.setAttribute('aPv', new THREE.BufferAttribute(dv, 3));
  return geo;
}

// ── The faults ───────────────────────────────────────────────────────────
// Each crack starts at the lip and runs both ways, down the outside and
// down into the well, meandering, kinking and now and then branching. Paths
// are lists of { phi, s, al (distance from the lip), w (taper) }.
function makeCracks(r, prof) {
  var L = prof.len, sRim = prof.rimV * L, sFoot = prof.footV * L + 0.03, sWell = prof.wellV * L;
  function radius(s) {
    var f = clamp(s / L, 0, 1) * (prof.n - 1), j = Math.min(Math.floor(f), prof.n - 2);
    return Math.max(lerp(prof.pts[j].x, prof.pts[j + 1].x, f - j), 0.2);
  }
  var paths = [];
  function walk(k, phi, s, dir, len, psi, al, w, depth) {
    var pts = [{ phi: phi, s: s, al: al, w: w }], aim = psi, step = 0.007, kids = 0;
    for (var d = 0; d < len; d += step) {
      psi += (r() - 0.5) * 0.5;
      if (r() < 0.05) psi += (r() < 0.5 ? -1 : 1) * (0.35 + r() * 0.5);      // a sharp kink
      psi = psi * 0.9 + aim * 0.1;
      phi += Math.sin(psi) * step / radius(s);
      s += dir * Math.cos(psi) * step;
      if (s < sFoot || s > sWell) break;
      al += step;
      pts.push({ phi: phi, s: s, al: al, w: w * (1 - 0.6 * Math.pow(d / len, 1.8)) });
      if (depth < 2 && kids < 3 - depth && d > 0.06 && r() < 0.022) {
        kids++;
        walk(k, phi, s, dir, len * (0.25 + r() * 0.3), psi + (r() < 0.5 ? -1 : 1) * (0.5 + r() * 0.6), al,
             pts[pts.length - 1].w * 0.8, depth + 1);
      }
    }
    paths.push({ k: k, pts: pts });
  }
  // [bearing from the camera while they are named, depth outside, depth inside]
  var plan = [[0.38, 0.62, 0.3], [Math.PI - 0.5, 0.25, 0.95], [-0.72, 0.5, 0.2], [Math.PI + 0.75, 0.2, 0.6],
              [-0.12, 0, 0], [1.2, 0.42, 0.25], [Math.PI - 1.35, 0.18, 0.5]];
  plan.forEach(function (p, k) {
    var phi = VIEW_AZ + p[0];
    if (p[1] === 0) return;                                  // the chip, below
    var lean = (r() - 0.5) * 0.8;
    walk(k, phi, sRim, -1, p[1], lean, 0, 1, 0);
    walk(k, phi, sRim, 1, p[2], -lean * 0.6 + (r() - 0.5) * 0.4, 0, 0.9, 0);
  });
  // The chip: an irregular bite out of the lip, glaze gone to the body.
  var chipPhi = VIEW_AZ + plan[4][0], chip = [];
  for (var q = 0; q < 14; q++) {
    var t = q / 14 * Math.PI * 2, rr = 0.85 + r() * 0.3;
    chip.push({ phi: chipPhi + Math.cos(t) * 0.05 * rr, s: sRim + Math.sin(t) * 0.05 * rr - 0.012 });
  }
  // A short crack runs on from the chip down the outside.
  walk(4, chipPhi + 0.02, sRim - 0.06, -1, 0.3, 0.3, 0.05, 0.8, 1);
  return { paths: paths, chip: { k: 4, pts: chip }, radius: radius };
}

// Paint the ridge (distance to the nearest crack, as a cone) and the timing
// map. Strokes are drawn in surface units: x scaled by 2πr, y by the profile.
function paintCracks(cracks, prof, W, H, W2, H2) {
  var L = prof.len, TAU = Math.PI * 2;
  var ridge = document.createElement('canvas'), times = document.createElement('canvas');
  ridge.width = W; ridge.height = H; times.width = W2; times.height = H2;
  var x = ridge.getContext('2d'), y = times.getContext('2d');
  x.fillStyle = '#000'; x.fillRect(0, 0, W, H);
  y.fillStyle = '#fff'; y.fillRect(0, 0, W2, H2);
  x.lineCap = y.lineCap = 'round';
  x.lineJoin = y.lineJoin = 'round';
  x.globalCompositeOperation = 'lighten';

  var lens = [];
  cracks.paths.forEach(function (p) { lens[p.k] = Math.max(lens[p.k] || 0.1, p.pts[p.pts.length - 1].al); });
  function crackT(k, al) { return (k + 0.62 * clamp(al / lens[k], 0, 1)) / NCRACK; }
  function goldT(k, al) { return clamp(0.02 + 0.025 * k + 0.82 * al / 1.0, 0, 0.97); }

  // Map a surface point to canvas pixels and set a transform so that one
  // surface unit is drawn true-to-size around it (with copies across the seam).
  function frameAt(ctx, w, h, phi, s, rad, dx) {
    var px = ((phi / TAU) % 1 + 1) % 1 * w + dx;
    ctx.setTransform(w / (TAU * rad), 0, 0, -h / L, px, (1 - s / L) * h);
  }
  function each(ctx, w, phi, draw) {
    var px = ((phi / TAU) % 1 + 1) % 1 * w;
    draw(0);
    if (px < w * 0.1) draw(w);
    if (px > w * 0.9) draw(-w);
  }

  var LEVELS = 14;
  function seg(ctx, w, h, a, b, style, width) {
    var rd = cracks.radius(a.s);
    each(ctx, w, a.phi, function (dx) {
      frameAt(ctx, w, h, a.phi, a.s, rd, dx);
      ctx.strokeStyle = style;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo((b.phi - a.phi) * rd, b.s - a.s);
      ctx.stroke();
    });
  }
  // First a wide pass in B, the earliest gold time nearby, for the molten
  // glow that spills onto the glaze round each seam.
  y.globalCompositeOperation = 'darken';
  cracks.paths.forEach(function (p) {
    for (var s = 0; s < p.pts.length - 1; s++) {
      seg(y, W2, H2, p.pts[s], p.pts[s + 1], 'rgb(255,255,' + Math.round(goldT(p.k, p.pts[s].al) * 255) + ')', 0.2);
    }
  });
  y.globalCompositeOperation = 'source-over';
  // Draw latest first, so where cracks cross the earlier one's timing wins.
  cracks.paths.slice().sort(function (a, b) { return b.k - a.k; }).forEach(function (p) {
    var pts = p.pts;
    for (var c = 0; c < pts.length - 1; c += 6) {
      var e = Math.min(c + 6, pts.length - 1), p0 = pts[c], rad = cracks.radius(pts[(c + e) >> 1].s), wv = pts[(c + e) >> 1].w;
      each(x, W, p0.phi, function (dx) {
        frameAt(x, W, H, p0.phi, p0.s, rad, dx);
        for (var lv = 0; lv < LEVELS; lv++) {
          var v = 0.04 + lv / LEVELS * 0.94;
          x.strokeStyle = 'rgb(' + Math.round(v * 255) + ',0,0)';
          x.lineWidth = 2 * HW * wv * (1 - v);
          x.beginPath();
          x.moveTo(0, 0);
          for (var m = c + 1; m <= e; m++) x.lineTo((pts[m].phi - p0.phi) * rad, pts[m].s - p0.s);
          x.stroke();
        }
      });
    }
    for (var s = 0; s < pts.length - 1; s++) {
      var g = Math.round(goldT(p.k, pts[s].al) * 255);
      seg(y, W2, H2, pts[s], pts[s + 1], 'rgb(' + Math.round(crackT(p.k, pts[s].al) * 255) + ',' + g + ',' + g + ')',
          2 * HW * 1.6 * Math.max(pts[s].w, 0.6));
    }
  });

  // The chip: filled in B, its outline a ridge like a crack's.
  var ch = cracks.chip.pts, c0 = ch[0], crad = cracks.radius(c0.s);
  function chipPath(ctx) {
    ctx.beginPath();
    ctx.moveTo(0, 0);
    for (var m = 1; m <= ch.length; m++) { var q = ch[m % ch.length]; ctx.lineTo((q.phi - c0.phi) * crad, q.s - c0.s); }
    ctx.closePath();
  }
  each(x, W, c0.phi, function (dx) {
    frameAt(x, W, H, c0.phi, c0.s, crad, dx);
    x.fillStyle = 'rgb(0,0,255)';
    chipPath(x);
    x.fill();
    for (var lv = 0; lv < LEVELS; lv++) {
      var v = 0.04 + lv / LEVELS * 0.94;
      x.strokeStyle = 'rgb(' + Math.round(v * 255) + ',0,0)';
      x.lineWidth = 2 * HW * 0.85 * (1 - v);
      chipPath(x);
      x.stroke();
    }
  });
  each(y, W2, c0.phi, function (dx) {
    frameAt(y, W2, H2, c0.phi, c0.s, crad, dx);
    var g = Math.round(goldT(4, 0) * 255);
    y.fillStyle = y.strokeStyle = 'rgb(' + Math.round(crackT(4, 0) * 255 + 3) + ',' + g + ',' + g + ')';
    y.lineWidth = 2 * HW * 1.6;
    chipPath(y);
    y.fill();
    y.stroke();
  });

  var tr = new THREE.CanvasTexture(ridge), tt = new THREE.CanvasTexture(times);
  tr.wrapS = tt.wrapS = THREE.RepeatWrapping;
  tr.anisotropy = 8;
  tt.generateMipmaps = false;
  tt.minFilter = THREE.LinearFilter;
  return { ridge: tr, times: tt };
}

// ── The glaze: porcelain with cracks, gold seams and a little light let through
function glazeMaterial(prof, tex, W, H) {
  var mat = new THREE.MeshPhysicalMaterial({ color: '#f4f1ea', roughness: 0.14, metalness: 0, clearcoat: 1,
    clearcoatRoughness: 0.05, specularIntensity: 0.7 });
  var u = {
    uRidge: { value: tex.ridge }, uTimes: { value: tex.times }, uTexel: { value: new THREE.Vector2(1.5 / W, 1.5 / H) },
    uCrack: { value: 0 }, uGold: { value: 0 }, uTime: { value: 0 }, uHeat: { value: 0 }, uKnow: { value: 0 }, uBead: { value: 0.0045 },
    uKeyDir: { value: new THREE.Vector3(0, 1, 0) }, uKeyI: { value: 1 }, uSss: { value: new THREE.Color('#ffc49a') },
    uRimV: { value: prof.rimV }, uFootV: { value: prof.footV }, uWellV: { value: prof.wellV }
  };
  mat.onBeforeCompile = function (sh) {
    Object.assign(sh.uniforms, u);
    sh.vertexShader = 'attribute vec3 aPu; attribute vec3 aPv; varying vec2 vKUv; varying vec3 vGu; varying vec3 vGv; varying vec3 vObj;\n' +
      sh.vertexShader.replace('#include <uv_vertex>',
        '#include <uv_vertex>\n vKUv = uv; vGu = normalMatrix * aPu; vGv = normalMatrix * aPv; vObj = position;');
    sh.fragmentShader = [
      'uniform sampler2D uRidge; uniform sampler2D uTimes; uniform vec2 uTexel; uniform float uCrack; uniform float uGold; uniform float uTime;',
      'uniform float uHeat; uniform float uKnow; uniform float uBead; uniform vec3 uKeyDir; uniform float uKeyI; uniform vec3 uSss;',
      'uniform float uRimV; uniform float uFootV; uniform float uWellV;',
      'varying vec2 vKUv; varying vec3 vGu; varying vec3 vGv; varying vec3 vObj;',
      'float kHash(vec3 p){ p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }',
      'float kNoise(vec3 x){ vec3 i = floor(x), f = fract(x); f = f * f * (3.0 - 2.0 * f);',
      ' return mix(mix(mix(kHash(i), kHash(i + vec3(1,0,0)), f.x), mix(kHash(i + vec3(0,1,0)), kHash(i + vec3(1,1,0)), f.x), f.y),',
      '  mix(mix(kHash(i + vec3(0,0,1)), kHash(i + vec3(1,0,1)), f.x), mix(kHash(i + vec3(0,1,1)), kHash(i + vec3(1,1,1)), f.x), f.y), f.z); }',
      // A line cut from the ridge at `edge`: crisp, and never thinner than a pixel (it fades instead).
      'float kLine(float R, float edge){ float fw = max(fwidth(R), 1e-4), hw = 1.0 - edge, e = max(hw, fw * 0.8);',
      ' return smoothstep(1.0 - e - fw * 0.5, 1.0 - e + fw * 0.5, R) * (hw / e) * (1.0 - smoothstep(0.35, 0.6, fw)); }',
      // The gold bead's height: a dome across the seam.
      'float kBeadH(vec2 uv){ vec4 t = texture2D(uRidge, uv); float x = clamp((1.0 - max(t.r, t.b)) / (1.0 - ' + TG.toFixed(3) + '), 0.0, 1.0); return 1.0 - x * x; }',
      sh.fragmentShader
    ].join('\n')
      .replace('#include <color_fragment>', [
        '#include <color_fragment>',
        'vec4 kRid = texture2D(uRidge, vKUv); vec3 kTm = texture2D(uTimes, vKUv).rgb;',
        'float kR = kRid.r, kChipA = kRid.b;',
        'float kOn = smoothstep(kTm.x, kTm.x + 0.004, uCrack);',
        'float kGoldOn = smoothstep(kTm.y, kTm.y + 0.006, uGold);',
        'float kCrack = kLine(kR, ' + TC.toFixed(3) + ') * kOn;',
        'float kGold = kLine(max(kR, kChipA), ' + TG.toFixed(3) + ') * kGoldOn;',
        'float kPhi = atan(vObj.x, vObj.z);',
        // Unglazed foot below a wavering drip line; a pale celadon pool in the well.
        'float kDrip = uFootV + 0.004 * sin(kPhi * 7.0) + 0.003 * sin(kPhi * 13.0 + 1.0);',
        'float kBisc = 1.0 - smoothstep(kDrip - 0.0015, kDrip + 0.0015, vKUv.y);',
        'float kN = kNoise(vObj * 9.0) * 0.6 + kNoise(vObj * 31.0) * 0.4;',
        'diffuseColor.rgb *= 0.965 + 0.05 * kN;',
        'diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.62, 0.70, 0.62), smoothstep(uWellV - 0.06, 1.0, vKUv.y) * 0.55);',
        'diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.52, 0.40, 0.28) * (0.85 + 0.3 * kN), kBisc);',
        // The chip shows the body; the cracks are dark, with a faint stain about them.
        'float kChip = smoothstep(0.4, 0.6, kChipA) * kOn;',
        'diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.56, 0.47, 0.38) * (0.8 + 0.3 * kN), kChip * (1.0 - kGold));',
        'diffuseColor.rgb *= 1.0 - 0.14 * smoothstep(' + (TC - 0.3).toFixed(3) + ', ' + TC.toFixed(3) + ', kR) * kOn * (1.0 - kGold);',
        'vec3 kCrackCol = mix(vec3(0.035, 0.028, 0.024), vec3(0.42, 0.24, 0.08), uKnow * 0.55);',
        'diffuseColor.rgb = mix(diffuseColor.rgb, kCrackCol, kCrack * (1.0 - kGold));',
        'diffuseColor.rgb = mix(diffuseColor.rgb, vec3(1.0, 0.78, 0.36) * (0.92 + 0.12 * kN), kGold);'
      ].join('\n'))
      .replace('#include <roughnessmap_fragment>', [
        '#include <roughnessmap_fragment>',
        'roughnessFactor = mix(roughnessFactor * (0.8 + 0.5 * kN), 0.8, max(kBisc, kChip));',
        'roughnessFactor = mix(roughnessFactor, 0.6, kCrack * (1.0 - kGold));',
        'roughnessFactor = mix(roughnessFactor, 0.26 + 0.12 * kN, kGold);'
      ].join('\n'))
      .replace('#include <metalnessmap_fragment>', '#include <metalnessmap_fragment>\n metalnessFactor = mix(metalnessFactor, 1.0, kGold);')
      .replace('#include <normal_fragment_maps>', [
        '#include <normal_fragment_maps>',
        'if (kGoldOn > 0.0 && max(kR, kChipA) > ' + (TG - 0.05).toFixed(3) + ') {',
        ' float hu = (kBeadH(vKUv + vec2(uTexel.x, 0.0)) - kBeadH(vKUv - vec2(uTexel.x, 0.0))) / (2.0 * uTexel.x);',
        ' float hv = (kBeadH(vKUv + vec2(0.0, uTexel.y)) - kBeadH(vKUv - vec2(0.0, uTexel.y))) / (2.0 * uTexel.y);',
        ' normal = normalize(normal - (hu * vGu + hv * vGv) * uBead * kGoldOn * faceDirection);',
        '}',
        // Smoothed normals just past the lip can face away at grazing angles
        // and shade as dark specks; tip them back towards the eye.
        'vec3 kEye = normalize(vViewPosition); float kNe = dot(normal, kEye);',
        'if (kNe < 0.08) normal = normalize(normal + kEye * (0.08 - kNe));',
        'kNe = dot(nonPerturbedNormal, kEye);',
        'if (kNe < 0.08) nonPerturbedNormal = normalize(nonPerturbedNormal + kEye * (0.08 - kNe));'
      ].join('\n'))
      .replace('#include <emissivemap_fragment>', [
        '#include <emissivemap_fragment>',
        // Molten at the front, cooling behind it; a glow bleeds onto the glaze.
        'float kSince = max(uGold - kTm.y, 0.0), kFront = exp(-kSince * 9.0);',
        'float kHot = uHeat * kGoldOn * (0.3 + 0.7 * kFront);',
        'totalEmissiveRadiance += mix(vec3(1.0, 0.3, 0.05), vec3(1.0, 0.86, 0.55), kFront) * kHot * kGold * (2.2 + 3.5 * kFront);',
        'float kHaloOn = smoothstep(kTm.z, kTm.z + 0.02, uGold) * (0.35 + 0.65 * exp(-max(uGold - kTm.z, 0.0) * 6.0));',
        'float kBlur = clamp(texture2D(uRidge, vKUv, 4.0).r * 3.0, 0.0, 1.0);',
        'totalEmissiveRadiance += vec3(1.0, 0.4, 0.08) * (kBlur * kBlur * 0.5 + pow(max(kR, kChipA), 3.0)) * uHeat * kHaloOn * 0.7 * (1.0 - kGold);',
        // Settled gold keeps a little of the room's warmth even where it faces the dark.
        'float kNv = 1.0 - abs(dot(normal, normalize(vViewPosition)));',
        'float kGlint = pow(0.5 + 0.5 * sin(dot(vObj, vec3(5.0, 3.0, 4.0)) - uTime * 1.3), 10.0);',
        'totalEmissiveRadiance += vec3(0.62, 0.44, 0.17) * kGold * uKeyI * (0.32 + 0.5 * kNv * kNv + 0.55 * kGlint) * (1.0 - kHot * 0.5);',
        'totalEmissiveRadiance += vec3(0.55, 0.3, 0.08) * kCrack * (1.0 - kGold) * uKnow * 0.12;'
      ].join('\n'))
      .replace('#include <lights_physical_fragment>', '#include <lights_physical_fragment>\n' +
        '#ifdef USE_CLEARCOAT\n material.clearcoat *= (1.0 - kGold) * (1.0 - max(kBisc, kChip)) * (1.0 - kCrack * 0.8);\n#endif')
      .replace('#include <lights_fragment_end>', [
        '#include <lights_fragment_end>',
        // Porcelain lets light through: a warm wrap past the terminator, and a
        // glow on the far side of the thin walls near the lip.
        'float kNdl = dot(normal, uKeyDir);',
        'float kWrap = max((kNdl + 0.5) / 1.5, 0.0) - max(kNdl, 0.0);',
        'float kThin = exp(-abs(vKUv.y - uRimV) * 7.0);',
        'float kThru = pow(max(-kNdl, 0.0), 1.2) * kThin;',
        'reflectedLight.indirectDiffuse += uSss * diffuseColor.rgb * (kWrap * 0.3 + kThru * 0.4) * uKeyI * (1.0 - kGold) * (1.0 - kBisc);',
        'reflectedLight.indirectSpecular *= 1.0 + kGold * 3.5;'
      ].join('\n'));
  };
  mat.userData.uniforms = u;
  return mat;
}

// A small studio for reflections: a dark room, a warm softbox where the lamp
// is, a tall cool strip behind, a dim fill, and the table's warm bounce.
function studioEnvironment(gl) {
  var room = new THREE.Scene(), geos = [], mats = [];
  function add(geo, rgb, at, look, side) {
    var m = new THREE.MeshBasicMaterial({ color: new THREE.Color(rgb[0], rgb[1], rgb[2]), side: side || THREE.DoubleSide });
    var mesh = new THREE.Mesh(geo, m);
    mesh.position.set(at[0], at[1], at[2]);
    if (look) mesh.lookAt(0, 0.4, 0);
    room.add(mesh);
    geos.push(geo); mats.push(m);
  }
  add(new THREE.BoxGeometry(14, 9, 14).translate(0, 4.5, 0), [0.03, 0.022, 0.018], [0, -0.01, 0], false, THREE.BackSide);
  add(new THREE.PlaneGeometry(14, 14).rotateX(-Math.PI / 2), [0.13, 0.09, 0.065], [0, 0, 0]);
  add(new THREE.PlaneGeometry(3.6, 2.6), [7, 5.4, 3.9], [-3.6, 5.2, 3.0], true);
  add(new THREE.PlaneGeometry(5, 5).rotateX(Math.PI / 2), [0.5, 0.42, 0.34], [0, 7.5, 0]);
  add(new THREE.PlaneGeometry(0.6, 4.2), [1.1, 1.25, 1.5], [4.8, 2.6, -3.4], true);
  add(new THREE.PlaneGeometry(3.4, 1.2), [0.45, 0.36, 0.28], [3.0, 3.2, 4.6], true);
  add(new THREE.PlaneGeometry(1.4, 1.4), [1.6, 1.2, 0.9], [-1.5, 6.5, -2.5], true);
  var pm = new THREE.PMREMGenerator(gl), rt = pm.fromScene(room, 0.03);
  pm.dispose();
  geos.forEach(function (g) { g.dispose(); });
  mats.forEach(function (m) { m.dispose(); });
  return rt;
}

// Linen: a fine, uneven weave in warm grey-brown.
function linenTexture(r) {
  var c = document.createElement('canvas');
  c.width = c.height = 256;
  var x = c.getContext('2d');
  x.fillStyle = '#6e6058';
  x.fillRect(0, 0, 256, 256);
  for (var i = 0; i < 256; i += 2) {
    x.fillStyle = 'rgba(255,240,225,' + (0.03 + r() * 0.07).toFixed(3) + ')';
    x.fillRect(0, i, 256, 1);
    x.fillStyle = 'rgba(20,12,8,' + (0.04 + r() * 0.08).toFixed(3) + ')';
    x.fillRect(i, 0, 1, 256);
  }
  for (i = 0; i < 40; i++) {                       // slubs
    x.fillStyle = 'rgba(255,240,225,0.08)';
    x.fillRect(r() * 256, Math.floor(r() * 128) * 2, 6 + r() * 30, 1);
  }
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(26, 26);
  t.anisotropy = 8;
  return t;
}

// ── Sound ────────────────────────────────────────────────────────────────
// A crack running through the glaze: a tiny, dry porcelain tick.
function tick(ac, out) {
  var t = ac.currentTime;
  [3150, 4870, 7420].forEach(function (f, j) {
    var o = ac.createOscillator(), g = ac.createGain();
    o.type = 'sine';
    o.frequency.value = f * (0.96 + Math.random() * 0.08);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.016 / (j + 1), t + 0.003);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.16 / (j + 1));
    o.connect(g); g.connect(out);
    o.start(t); o.stop(t + 0.2);
  });
}

// The gold flowing: soft porcelain-bowl tones, rising slowly.
function goldChime(ac, out) {
  var t = ac.currentTime;
  [523.3, 659.3, 784, 987.8, 1046.5].forEach(function (f, k) {
    var at = t + k * 0.55 + Math.random() * 0.06;
    [1, 2.71, 5.1].forEach(function (m, j) {
      var o = ac.createOscillator(), g = ac.createGain();
      o.type = 'sine';
      o.frequency.value = f * m;
      g.gain.setValueAtTime(0.0001, at);
      g.gain.exponentialRampToValueAtTime(0.034 / (j * 1.6 + 1) / (1 + k * 0.12), at + 0.012);
      g.gain.exponentialRampToValueAtTime(0.0001, at + 4.2 / (j + 1));
      o.connect(g); g.connect(out);
      o.start(at); o.stop(at + 4.3);
    });
  });
}

// "made me love you more": one low, warm tone, held.
function settleTone(ac, out) {
  var t = ac.currentTime;
  [261.6, 392, 523.3].forEach(function (f, k) {
    var o = ac.createOscillator(), g = ac.createGain();
    o.type = 'sine';
    o.frequency.value = f;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.03 / (1 + k * 0.5), t + 0.4);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 6);
    o.connect(g); g.connect(out);
    o.start(t); o.stop(t + 6.1);
  });
}

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(1919);
  var gl = makeRenderer(canvas, { clear: '#070504', shadows: !small });
  gl.toneMappingExposure = 1.12;
  var world = new THREE.Scene();
  world.fog = new THREE.Fog('#160f0b', 7, 22);
  var camera = new THREE.PerspectiveCamera(35, 1, 0.05, 120);
  var envRT = studioEnvironment(gl);
  world.environment = envRT.texture;
  world.environmentIntensity = 0.6;

  // ── The room: a dark dome, and a linen table running off into the dark ──
  var domeMat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false,
    uniforms: { uHorizon: { value: new THREE.Color('#160f0b') }, uTop: { value: new THREE.Color('#040303') },
                uGlow: { value: new THREE.Color('#2a1a10') }, uGlowDir: { value: new THREE.Vector3(-0.6, 0.25, -0.75).normalize() } },
    vertexShader: 'varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform vec3 uHorizon; uniform vec3 uTop; uniform vec3 uGlow; uniform vec3 uGlowDir; varying vec3 vP;\n' +
      'void main(){ vec3 c = mix(uHorizon, uTop, smoothstep(-0.02, 0.45, vP.y));\n' +
      ' c += uGlow * pow(max(dot(vP, uGlowDir), 0.0), 3.0) * smoothstep(0.0, 0.3, vP.y);\n' +
      ' gl_FragColor = vec4(c, 1.0);\n #include <colorspace_fragment>\n }'
  });
  var dome = new THREE.Mesh(new THREE.SphereGeometry(60, 32, 16), domeMat);
  world.add(dome);

  var tableMat = new THREE.MeshStandardMaterial({ map: linenTexture(r), color: '#a08f84', roughness: 0.95, envMapIntensity: 0.25 });
  var table = new THREE.Mesh(new THREE.PlaneGeometry(120, 120).rotateX(-Math.PI / 2), tableMat);
  table.receiveShadow = true;
  world.add(table);
  // A soft contact shadow where the foot meets the cloth.
  var contact = new THREE.Mesh(new THREE.PlaneGeometry(2.0, 2.0).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({ map: softSprite('rgba(0,0,0,1)', 'rgba(0,0,0,0)'), transparent: true, depthWrite: false, opacity: 0.75 }));
  contact.position.y = 0.002;
  world.add(contact);

  // ── Light: a warm lamp up to the left, a faint cool rim, a low fill ──
  var key = new THREE.SpotLight('#ffe0c0', 0, 0, 0.42, 0.85, 2);
  key.position.set(-3.4, 5.0, 2.8);
  key.target.position.set(0, 0.3, 0);
  if (!small) {
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 3;
    key.shadow.camera.far = 12;
    key.shadow.bias = -0.0008;
    key.shadow.normalBias = 0.04;
    key.shadow.radius = 4;
  }
  var rim = new THREE.SpotLight('#b4c2dc', 0, 0, 0.35, 0.9, 2);
  rim.position.set(3.2, 2.6, -4.2);
  rim.target.position.set(0, 0.4, 0);
  var fill = new THREE.HemisphereLight('#4a3a2e', '#0c0806', 0.35);
  var molten = new THREE.PointLight('#ffa04a', 0, 5, 2);
  molten.position.set(0, 0.55, 0);
  world.add(key, key.target, rim, rim.target, fill, molten);

  // ── The bowl ──
  var prof = buildProfile(small ? 150 : 230);
  var cracks = makeCracks(r, prof);
  var TW = small ? 2048 : 3072, TH = Math.round(TW * prof.len / (Math.PI * 2));
  var tex = paintCracks(cracks, prof, TW, TH, Math.round(TW / 3), Math.round(TH / 3));
  var glaze = glazeMaterial(prof, tex, TW, TH), gu = glaze.userData.uniforms;
  var bowl = new THREE.Mesh(bowlGeometry(prof, small ? 112 : 180), glaze);
  bowl.castShadow = bowl.receiveShadow = !small;
  world.add(bowl);

  // Dust in the lamplight.
  var motes = particleField({ count: small ? 120 : 260, box: [6, 3.2, 6], fall: [-0.02, 0.03], size: 0.018,
    color: '#ffd9a8', map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)'), sway: 0.04, windSpeed: 0.2 });
  motes.points.material.blending = THREE.AdditiveBlending;
  world.add(motes.points);

  // ── Per-frame state (no allocations in frame) ──
  var W = 1, H = 1, portrait = false;
  var target = new THREE.Vector3(), dir = new THREE.Vector3(), c1 = new THREE.Color(), moteC = new THREE.Vector3(0, 0.6, 0);
  var LAMP = new THREE.Color('#ffe4c8'), LAMP_COOL = new THREE.Color('#e2e0dc'), LAMP_WARM = new THREE.Color('#ffc890');
  var HOR = new THREE.Color('#160f0b'), HOR_WARM = new THREE.Color('#2a190e');
  var pf = { snow: 0, wind: 0, dt: 0, time: 0 };

  function frame(f) {
    var row = f.row, time = f.time, slow = env.reduceMotion;
    var lamp = row[K.lamp], lift = row[K.lift], gloom = row[K.gloom], heat = row[K.heat], gold = row[K.gold];

    // ── Camera: an orbit round the bowl, the bowl set beside the verse ──
    var dist = row[K.dist] * (portrait ? 1.42 + Math.max(0, 6 - row[K.dist]) * 0.12 : 1), az = row[K.az] - f.mx * 0.06, el = row[K.el] + f.my * 0.03;
    // On a phone the verse sits mid-screen, so the bowl needs less pushing down.
    target.set(0, portrait ? 0.36 + (row[K.ly] - 0.36) * 0.55 : row[K.ly], 0);
    camera.position.set(Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)).multiplyScalar(dist).add(target);
    camera.lookAt(target);
    if (portrait) camera.setViewOffset(W, H, 0, -H * 0.2, W, H);
    else camera.setViewOffset(W, H, -row[K.side] * W, 0, W, H);

    // The bowl turns with the reading, breathing a little on its own.
    bowl.rotation.y = row[K.spin] + (slow ? 0 : Math.sin(time * 0.23) * 0.025);

    // ── Light: cooler as the faults are named, a warm lift at the laugh ──
    key.color.copy(LAMP).lerp(LAMP_COOL, gloom).lerp(LAMP_WARM, lift * 0.8);
    key.intensity = 52 * lamp * (1 - gloom * 0.25 + lift * 0.3);
    rim.intensity = 14 * lamp * (1 - lift * 0.3);
    fill.intensity = 0.3 + lift * 0.35;
    world.environmentIntensity = (0.4 - gloom * 0.1 + lift * 0.12) * (0.25 + 0.75 * lamp);
    molten.intensity = heat * smooth(0, 0.15, gold) * 0.6 * (0.9 + (slow ? 0 : 0.1 * Math.sin(time * 7.3)));
    c1.copy(HOR).lerp(HOR_WARM, lift + heat * 0.3).multiplyScalar((0.35 + 0.65 * lamp) * (1 - gloom * 0.35));
    domeMat.uniforms.uHorizon.value.copy(c1);
    world.fog.color.copy(c1);
    domeMat.uniforms.uGlow.value.set('#2a1a10').multiplyScalar((0.5 + lift * 2.2) * lamp * (1 - gloom * 0.5));

    // For the glaze's light-through: the lamp's direction in view space.
    camera.updateMatrixWorld();
    dir.subVectors(key.position, key.target.position).normalize().transformDirection(camera.matrixWorldInverse);
    gu.uKeyDir.value.copy(dir);
    gu.uKeyI.value = key.intensity / 52;
    gu.uCrack.value = row[K.crack];
    gu.uKnow.value = row[K.know];
    gu.uGold.value = gold;
    gu.uHeat.value = heat;
    gu.uTime.value = slow ? 0 : time;

    // ── Dust drifting through the lamplight ──
    pf.snow = row[K.motes]; pf.wind = row[K.wind]; pf.dt = f.dt; pf.time = time;
    motes.update(pf, moteC, slow);
    motes.points.material.opacity = (0.3 + lift * 0.6) * lamp;

    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) {
      W = w; H = h; portrait = w < h;
      fitCamera(gl, camera, w, h, dpr, small);
      camera.fov = portrait ? 46 : 35;
      camera.updateProjectionMatrix();
    },
    frame: frame,
    destroy: function () { envRT.dispose(); tex.ridge.dispose(); tex.times.dispose(); disposeAll(world, gl); }
  };
}

PI.register('kintsugi', {
  renderer: renderer3d,
  maxLines: 2,
  scrim: 0.5,
  accent: '#e9c27c',
  emphasis: /^(faults|love)\W*$/i,
  align: ['left', 'right', 'left'],
  // Three panels of two lines; key() carries every column forward, so each
  // beat lists only what changes.
  keys: function (T) {
    function at(i, d) { return T.start(i) + d; }     // d units into panel i (0..1.6)
    var rows = [], cur = {};
    function key(u, ch) {
      for (var c in ch) cur[c] = ch[c];
      rows.push([u].concat(COLS.map(function (c) { return cur[c]; })));
    }
    key(0, { dist: 9.6, gloom: 0, motes: 0.3, wind: 0, az: 0.62, el: 0.4, ly: 2.5, side: 0, lamp: 0.12, lift: 0,
             crack: 0, know: 0, gold: 0, heat: 0, spin: -0.45 });
    key(0.75, { lamp: 0.5, dist: 9.2, az: 0.5, ly: 2.3, spin: -0.32 });                                   // the lamp comes up
    key(at(0, 0.25), { dist: 7.0, az: VIEW_AZ, el: 0.6, ly: 0.36, side: 0.2, lamp: 1, spin: 0, motes: 0.5 });
    // "They named them over one by one": a crack (or the chip) each step.
    for (var k = 0; k <= NCRACK; k++) {
      key(at(0, CRACK_AT + k * CRACK_GAP), { crack: k / NCRACK, dist: 7.0 - 0.5 * k / NCRACK, gloom: 0.4 * k / NCRACK });
    }
    // "I laughed aloud when they were done": a warm lift of light.
    key(at(1, 0.12), { lift: 0, gloom: 0.4, dist: 6.0, el: 0.5, az: -0.05, side: -0.24, spin: 0.2 });
    key(at(1, 0.55), { lift: 1, gloom: 0, dist: 5.7, spin: 0.27, motes: 0.8 });
    // "I knew them all so well before": close over the lip, the cracks known by heart.
    key(at(1, 0.8), { know: 0 });
    key(at(1, 1.35), { know: 1, dist: 4.9, el: 0.72, side: -0.25, az: -0.22, ly: 0.48, spin: 0.42 });
    // "Oh, they were blind": the gold runs.
    key(at(2, 0.12), { gold: 0, heat: 0, dist: 6.4, el: 0.6, az: 0.08, ly: 0.36, side: 0.2, spin: 0.5, lift: 0.85 });
    key(at(2, 0.3), { heat: 1 });
    key(at(2, 1.0), { gold: 1, heat: 0.85, spin: 0.68, know: 0.4 });
    // "Your faults had made me love you more": it cools and gleams.
    key(at(2, 1.5), { heat: 0.12, know: 0, spin: 0.8, dist: 6.6, lift: 1.1 });
    key(T.total - 0.55, { heat: 0, dist: 6.3, el: 0.6, side: 0, spin: 0.98, az: 0.26, ly: 1.85, motes: 0.7 });
    key(T.total, { spin: 1.06, dist: 6.5 });
    return rows;
  },
  sound: {
    src: '/audio/wind.mp3',
    label: 'Play the room and the chimes',
    volume: function (row) { return 0.022 + 0.018 * row[K.lift]; },
    cues: [0, 1, 2, 3, 4, 5, 6].map(function (k) {
      return { stanza: 0, at: CRACK_AT + k * CRACK_GAP + 0.04, play: tick };
    }).concat([
      { stanza: 2, at: 0.2, play: goldChime },
      { stanza: 2, at: 1.25, play: settleTone }
    ])
  }
});
