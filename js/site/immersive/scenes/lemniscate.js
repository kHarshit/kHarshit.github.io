/*
 * Scene for "Love and Tensor Algebra" (Stanisław Lem, The Cyberiad): a love
 * poem in pure mathematics, read as a flight through one dark-blue space of
 * glowing lines. Two lights, a cool one and a gold one, are the lovers; they
 * travel with you from figure to figure and are part of every one.
 *
 * Intro "Hasten to a higher plane": you rise off a ruled floor plane into
 *       the dark, the two lights circling above.
 * I    "Dyads tread the fairy fields of Venn": two circles drawn round the
 *       two lights, overlapping; around them a Markov chain of nodes,
 *       indexed one to n, a pulse walking it at random.
 * II   "Every frustum longs to be a cone": a wire frustum closes to its
 *       apex. "Every vector dreams of matrices": one arrow becomes a
 *       bracketed grid of arrows, swaying in "the gradient of the breeze".
 * III  A curved sheet of grid (Riemann, Hilbert, Banach space); two waves
 *       come into phase, the lights ride them in from either end and meet
 *       "face to face", denting the space where they meet.
 * IV   "Random access to my heart": a heart curve drawn from its cusp, a
 *       "bound partition" down its middle with a light in each half, a
 *       lattice of memory cells flickering behind, the constants around it.
 * V    "Thy supernal sinusoidal spell": Fourier epicycles, harmonic on
 *       harmonic, the gold light at the pen spelling out a square wave.
 * VI   "Cancel me not": everything falls in to a torus and a node, a root
 *       or two on an abscissa; the lights part and dim to "a null domain".
 * VII  "Ellipse of bliss, converge": scattered ellipses converge into one,
 *       the lights at its foci; it rounds to a circle and the foci meet.
 *       Below, a haversine "cuts capers".
 * VIII "a² cos 2ψ": the lemniscate of Bernoulli is drawn from its node by
 *       the two lights, one lobe each, over a polar grid; in the outro the
 *       camera centres it and they keep tracing it, meeting at the node.
 *
 * The lines are camera-facing ribbons with a soft glow profile (glowLines),
 * updated in place. Columns:
 *   [unit, station, dark, dust, wind, yaw, pitch, venn, cone, field, phase,
 *    heart, fourier, collapse, ellipse, lem, glow]
 */
import { THREE, isSmall, makeRenderer, fitCamera, softSprite, skyDome, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout: one figure per station along -z, 40 units apart ─────────────
var FLOOR = -4.6, FOG = 0.02;
var FOCUS = [[0, 5.8, 0], [0.6, 1.0, -40], [-0.6, 0.8, -80], [0.4, 0.8, -120], [-0.4, 0.8, -160],
             [0.4, 0.8, -200], [0, 0.8, -240], [0.4, 0.8, -280], [-0.4, 1.0, -320]]
  .map(function (p) { return new THREE.Vector3(p[0], p[1], p[2]); });
// Which side of the screen each station's figure sits on (the text takes
// the other): +1 right, -1 left, 0 centre. Index 9 is the outro.
var SIDE = [0, 1, -1, 1, -1, 1, -1, 1, -1, 0];

var COOL = '#7fb4ff', PALE = '#d4e8ff', GOLD = '#ffcf86', INK = '#3b62b8', VIOLET = '#9a94ff';

// ── Glowing lines ────────────────────────────────────────────────────────
// Each polyline is a ribbon two vertices wide, turned to face the camera in
// the vertex shader; the fragment fades across it, so a line has a bright
// core and a soft halo. Width is in world units (with a floor in pixels).
// Lines fog out a little sooner than the floor, so the next figure is only a hint.
var SHARED = { res: { value: new THREE.Vector2(1, 1) }, fogD: { value: FOG * 1.35 } };

var LINE_VS = [
  'attribute vec3 prev; attribute vec3 next; attribute float side; attribute float lu; attribute vec3 tint;',
  'uniform vec2 uRes; uniform float uFogD; uniform float uWidth; uniform float uMinPx;',
  'varying float vSide; varying float vU; varying vec3 vCol;',
  'vec2 screen(vec4 c, float aspect){ vec2 s = c.xy / max(c.w, 1e-4); s.x *= aspect; return s; }',
  'void main(){',
  '  vec4 mv = modelViewMatrix * vec4(position, 1.0);',
  '  vec4 c = projectionMatrix * mv;',
  '  float aspect = uRes.x / uRes.y;',
  '  vec2 sc = screen(c, aspect);',
  '  vec2 d1 = screen(projectionMatrix * (modelViewMatrix * vec4(next, 1.0)), aspect) - sc;',
  '  vec2 d2 = sc - screen(projectionMatrix * (modelViewMatrix * vec4(prev, 1.0)), aspect);',
  '  float l1 = length(d1), l2 = length(d2);',
  '  vec2 dir = l1 > 1e-6 ? d1 / l1 : (l2 > 1e-6 ? d2 / l2 : vec2(1.0, 0.0));',
  '  if (l1 > 1e-6 && l2 > 1e-6) { vec2 m = d1 / l1 + d2 / l2; if (length(m) > 1e-3) dir = normalize(m); }',
  '  vec2 nrm = vec2(-dir.y, dir.x);',
  '  float px = uWidth * projectionMatrix[1][1] * 0.5 * uRes.y / max(c.w, 1e-4);',
  '  float w = max(px, uMinPx);',
  '  vec2 off = nrm * side * w / uRes.y;',
  '  off.x /= aspect;',
  '  c.xy += off * c.w;',
  '  gl_Position = c;',
  '  vSide = side; vU = lu;',
  '  vCol = tint * exp(-uFogD * uFogD * dot(mv.xyz, mv.xyz)) * min(px / uMinPx, 1.0);',
  '}'
].join('\n');

var LINE_FS = [
  'uniform float uDraw; uniform float uOpacity; uniform vec3 uColor; uniform float uCore;',
  'varying float vSide; varying float vU; varying vec3 vCol;',
  'void main(){',
  '  if (vU > uDraw) discard;',
  '  float s = vSide * vSide;',
  '  float a = exp(-s * 4.5) * 0.5 + exp(-s * 38.0) * uCore;',
  '  gl_FragColor = vec4(uColor * vCol * a * uOpacity, 1.0);',
  '  #include <colorspace_fragment>',
  '}'
].join('\n');

function lineMat(color, width, opacity, core) {
  return new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uRes: SHARED.res, uFogD: SHARED.fogD, uWidth: { value: width }, uMinPx: { value: 1.6 },
                uDraw: { value: 1 }, uOpacity: { value: opacity == null ? 1 : opacity },
                uColor: { value: new THREE.Color(color || '#ffffff') }, uCore: { value: core == null ? 0.9 : core } },
    vertexShader: LINE_VS, fragmentShader: LINE_FS
  });
}

// A set of polylines (counts = points per strip) sharing one material.
// set() writes a point, tint() a colour, commit() rebuilds the ribbon.
function glowLines(counts, mat) {
  var N = 0;
  counts.forEach(function (c) { N += c; });
  var pts = new Float32Array(N * 3), pos = new Float32Array(N * 6), prv = new Float32Array(N * 6), nxt = new Float32Array(N * 6);
  var side = new Float32Array(N * 2), lu = new Float32Array(N * 2), col = new Float32Array(N * 6).fill(1);
  var offs = [], idx = [], o = 0;
  counts.forEach(function (c) {
    offs.push(o);
    for (var j = 0; j < c; j++) {
      var v = (o + j) * 2;
      side[v] = -1; side[v + 1] = 1;
      lu[v] = lu[v + 1] = c > 1 ? j / (c - 1) : 0;
      if (j < c - 1) idx.push(v, v + 2, v + 1, v + 2, v + 3, v + 1);
    }
    o += c;
  });
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('prev', new THREE.BufferAttribute(prv, 3));
  geo.setAttribute('next', new THREE.BufferAttribute(nxt, 3));
  geo.setAttribute('side', new THREE.BufferAttribute(side, 1));
  geo.setAttribute('lu', new THREE.BufferAttribute(lu, 1));
  geo.setAttribute('tint', new THREE.BufferAttribute(col, 3));
  geo.setIndex(idx);
  var mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;
  var tc = new THREE.Color();
  return {
    mesh: mesh, mat: mat,
    set: function (s, j, x, y, z) { var k = (offs[s] + j) * 3; pts[k] = x; pts[k + 1] = y; pts[k + 2] = z; },
    tint: function (s, j, color, k) {
      tc.set(color);
      var v = (offs[s] + j) * 6, m = k == null ? 1 : k;
      col[v] = col[v + 3] = tc.r * m; col[v + 1] = col[v + 4] = tc.g * m; col[v + 2] = col[v + 5] = tc.b * m;
    },
    tintStrip: function (s, color, k) { for (var j = 0; j < counts[s]; j++) this.tint(s, j, color, k); },
    tinted: function () { geo.attributes.tint.needsUpdate = true; },
    commit: function () {
      for (var s = 0; s < counts.length; s++) {
        var o2 = offs[s], c = counts[s];
        for (var j = 0; j < c; j++) {
          var a = (o2 + j) * 3, p = (o2 + Math.max(j - 1, 0)) * 3, n = (o2 + Math.min(j + 1, c - 1)) * 3, v = (o2 + j) * 6;
          for (var e = 0; e < 3; e++) {
            pos[v + e] = pos[v + 3 + e] = pts[a + e];
            prv[v + e] = prv[v + 3 + e] = pts[p + e];
            nxt[v + e] = nxt[v + 3 + e] = pts[n + e];
          }
        }
      }
      geo.attributes.position.needsUpdate = true;
      geo.attributes.prev.needsUpdate = true;
      geo.attributes.next.needsUpdate = true;
    }
  };
}

// A glyph (π, e, n ...) or a line of text in italic serif on a canvas.
function glyphTexture(text, w, h, px) {
  var c = document.createElement('canvas');
  c.width = w; c.height = h;
  var x = c.getContext('2d');
  x.fillStyle = '#ffffff';
  x.font = 'italic ' + px + 'px Georgia, "Times New Roman", serif';
  x.textAlign = 'center';
  x.textBaseline = 'middle';
  x.shadowColor = 'rgba(255,255,255,0.8)';
  x.shadowBlur = px * 0.12;
  x.fillText(text, w / 2, h / 2);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// ── Synthesised sound cues: soft sine tones ──────────────────────────────
function tone(ac, out, f, delay, dur, gain, f2) {
  var t = ac.currentTime + delay, o = ac.createOscillator(), g = ac.createGain();
  o.type = 'sine';
  o.frequency.setValueAtTime(f, t);
  if (f2) o.frequency.exponentialRampToValueAtTime(f2, t + dur * 0.7);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(gain, t + Math.min(0.5, dur * 0.3));
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g);
  g.connect(out);
  o.start(t);
  o.stop(t + dur + 0.05);
}
// The two lights: a fifth.
function dyadSound(ac, out) { tone(ac, out, 440, 0, 3.5, 0.05); tone(ac, out, 659.25, 0.3, 3.5, 0.04); }
// The frustum closing to its apex: a soft upward glide.
function coneSound(ac, out) { tone(ac, out, 392, 0, 3, 0.04, 523.25); }
// Out of phase into phase: two tones beating slower and slower to unison.
function phaseSound(ac, out) { tone(ac, out, 330, 0, 5, 0.045); tone(ac, out, 352, 0, 5, 0.045, 330); }
// Random access: blips at random addresses on a pentatonic scale.
function accessSound(ac, out) {
  var scale = [523.25, 587.33, 659.25, 783.99, 880, 1046.5];
  for (var i = 0; i < 7; i++) tone(ac, out, scale[Math.floor(Math.random() * scale.length)], i * 0.16 + Math.random() * 0.05, 0.4, 0.025);
}
// A square wave spelled out harmonic by harmonic.
function fourierSound(ac, out) {
  [1, 3, 5, 7].forEach(function (n, k) { tone(ac, out, 130.81 * n, k * 0.55, 5 - k * 0.55, 0.06 / n); });
}
// "Cancel me not": a tone sinking away.
function cancelSound(ac, out) { tone(ac, out, 220, 0, 4.5, 0.04, 110); }
// Ellipses converging: two tones gliding into one, then a fifth above.
function convergeSound(ac, out) {
  tone(ac, out, 415.3, 0, 4.5, 0.035, 440); tone(ac, out, 466.16, 0, 4.5, 0.035, 440); tone(ac, out, 659.25, 2.2, 3.5, 0.03);
}
// The lemniscate closes: a slow, open chord.
function lemniscateSound(ac, out) {
  [220, 329.63, 440, 554.37, 659.25].forEach(function (f, k) { tone(ac, out, f, k * 0.22, 7 - k * 0.3, 0.032); });
}

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(1964), slow = env.reduceMotion ? 0.4 : 1;
  var gl = makeRenderer(canvas, { clear: '#030817' });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#050c22', FOG);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.05, 2500);

  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#01040e', mid: '#040b22', horizon: '#0a1838' }, 1500);
  sky.add(dome.mesh);

  // Every glow material, with its base opacity, so the dark can dim them.
  var lit = [];
  function line(counts, color, width, opacity, core) {
    var g = glowLines(counts, lineMat(color, width, opacity, core));
    lit.push({ mat: g.mat, base: opacity == null ? 1 : opacity, k: 1 });
    world.add(g.mesh);
    return g;
  }
  function litOf(g) { for (var i = 0; i < lit.length; i++) if (lit[i].mat === g.mat) return lit[i]; return null; }

  var glowTex = softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)');
  function glow(color, size, opacity) {
    var s = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: color, blending: THREE.AdditiveBlending,
      depthWrite: false, transparent: true, opacity: opacity == null ? 1 : opacity }));
    s.scale.setScalar(size);
    world.add(s);
    return s;
  }

  // ── The floor plane: a ruled grid, minor and major lines ───────────────
  var X0 = -90, X1 = 90, Z0 = 40, Z1 = -380;
  function grid(step, color, opacity) {
    var p = [];
    for (var x = X0; x <= X1; x += step) p.push(x, FLOOR, Z0, x, FLOOR, Z1);
    for (var z = Z0; z >= Z1; z -= step) p.push(X0, FLOOR, z, X1, FLOOR, z);
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(p, 3));
    var m = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: opacity,
      blending: THREE.AdditiveBlending, depthWrite: false }));
    world.add(m);
    return m.material;
  }
  var floorMinor = grid(2, '#2a4a96', 0.22), floorMajor = grid(10, '#4c74d0', 0.4);

  // ── Dust of the space: twinkling points drifting on the breeze ─────────
  var DN = small ? 1400 : 3000, dpos = [], dattr = [];
  for (var i = 0; i < DN; i++) {
    dpos.push((r() - 0.5) * 90, FLOOR + r() * 22, 30 - r() * 400);
    dattr.push(0.6 + Math.pow(r(), 4) * 3.4, r() * 6.28, r() < 0.18 ? 1 : 0);
  }
  var dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute('position', new THREE.Float32BufferAttribute(dpos, 3));
  dustGeo.setAttribute('mote', new THREE.Float32BufferAttribute(dattr, 3));
  var dustMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uTime: { value: 0 }, uDrift: { value: 0 }, uAlpha: { value: 1 }, uScale: { value: 1 }, uFogD: SHARED.fogD },
    vertexShader: 'attribute vec3 mote; uniform float uTime; uniform float uDrift; uniform float uScale; uniform float uFogD;\n' +
      'varying float vA; varying float vGold;\n' +
      'void main(){ vec3 p = position; p.x = mod(p.x + uDrift + 45.0, 90.0) - 45.0;\n' +
      ' p.y += sin(uTime * 0.3 + mote.y) * 0.3;\n' +
      ' vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_Position = projectionMatrix * mv;\n' +
      ' float d = length(mv.xyz); vA = (0.6 + 0.4 * sin(uTime * 1.3 + mote.y * 3.0)) * exp(-uFogD * uFogD * d * d * 0.6);\n' +
      ' vGold = mote.z; gl_PointSize = clamp(mote.x * uScale / max(d, 0.5), 1.0, 9.0); }',
    fragmentShader: 'uniform float uAlpha; varying float vA; varying float vGold;\n' +
      'void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;\n' +
      ' float a = smoothstep(0.5, 0.0, d) * vA * uAlpha;\n' +
      ' vec3 c = mix(vec3(0.62, 0.78, 1.0), vec3(1.0, 0.82, 0.55), vGold);\n' +
      ' gl_FragColor = vec4(c * a, 1.0);\n #include <colorspace_fragment>\n }'
  });
  var dust = new THREE.Points(dustGeo, dustMat);
  dust.frustumCulled = false;
  world.add(dust);

  // Symbols floating in the space, faint as watermarks.
  var GLYPHS = ['π', 'e', 'φ', '∞', '∑', '∂', '∫', 'ψ', 'λ', 'i', 'n', '∇', 'θ', '0', '1', 'ε'];
  var glyphTex = {};
  GLYPHS.forEach(function (g) { glyphTex[g] = glyphTexture(g, 64, 64, 46); });
  var floaters = [], FN = small ? 34 : 70;
  for (i = 0; i < FN; i++) {
    var gs = new THREE.Sprite(new THREE.SpriteMaterial({ map: glyphTex[GLYPHS[i % GLYPHS.length]], color: r() < 0.25 ? GOLD : COOL,
      blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0.0 }));
    var sx = (r() < 0.5 ? -1 : 1) * (9 + r() * 22);
    gs.position.set(sx, FLOOR + 2 + r() * 14, 20 - r() * 360);
    gs.scale.setScalar(0.8 + r() * 1.2);
    gs.userData = { base: 0.12 + r() * 0.16, y: gs.position.y, ph: r() * 6.28 };
    world.add(gs);
    floaters.push(gs);
  }

  // ── I  Venn circles round the two lights, and a Markov chain ───────────
  var F1 = FOCUS[1], VR = 2.3, VD = 1.3;
  var venn = line([129, 129], '#ffffff', 0.09, 1, 1.0);
  for (i = 0; i < 129; i++) {
    var va = i / 128 * Math.PI * 2 + Math.PI / 2;
    venn.set(0, i, F1.x - VD + Math.cos(va) * VR, F1.y + Math.sin(va) * VR, F1.z);
    venn.set(1, i, F1.x + VD - Math.cos(va) * VR, F1.y + Math.sin(va) * VR, F1.z);
  }
  venn.tintStrip(0, COOL); venn.tintStrip(1, GOLD); venn.tinted(); venn.commit();
  var discs = [COOL, GOLD].map(function (c, k) {
    var m = new THREE.Mesh(new THREE.CircleGeometry(VR, 64), new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
    m.position.set(F1.x + (k ? VD : -VD), F1.y, F1.z - 0.02);
    world.add(m);
    return m;
  });
  var NODES = [[-4.1, -1.9, -1.2], [-1.9, -3.3, 0.6], [0.4, -3.1, -1.6], [2.6, -3.3, 0.4], [4.3, -1.6, -1.4],
               [4.1, 1.6, -2.4], [2.4, 3.6, -2.2], [-0.2, 3.9, -2.8], [-3.5, 3.0, -2.0]]
    .map(function (p) { return new THREE.Vector3(F1.x + p[0], F1.y + p[1], F1.z + p[2]); });
  var EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 0]];
  var EP = 24, markov = line(EDGES.map(function () { return EP; }), COOL, 0.04, 0.75, 0.8);
  var edgeCurves = EDGES.map(function (e, k) {
    var a = NODES[e[0]], b = NODES[e[1]], mid = a.clone().add(b).multiplyScalar(0.5);
    var ctrl = mid.clone().sub(F1).multiplyScalar(0.12).add(mid).add(new THREE.Vector3(0, 0, 0.3));
    var cv = new THREE.QuadraticBezierCurve3(a, ctrl, b), p = new THREE.Vector3();
    for (var j = 0; j < EP; j++) { cv.getPoint(j / (EP - 1), p); markov.set(k, j, p.x, p.y, p.z); }
    return cv;
  });
  markov.commit();
  var nodeGlows = NODES.map(function (p) { var s = glow(PALE, 0.5, 0); s.position.copy(p); return s; });
  var nodeLabels = NODES.map(function (p, k) {
    var t = k === NODES.length - 1 ? glyphTexture('n', 64, 64, 46) : glyphTexture(String(k + 1), 64, 64, 46);
    var s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, color: PALE, blending: THREE.AdditiveBlending, depthWrite: false,
      transparent: true, opacity: 0 }));
    s.position.set(p.x + 0.38, p.y + 0.38, p.z);
    s.scale.setScalar(0.42);
    world.add(s);
    return s;
  });
  // A pulse walking the chain: at each node it takes a random edge.
  var walk = { edge: 0, fwd: true, t: 0 }, walker = glow(GOLD, 0.55, 0);
  function nextEdge(node) {
    var opts = [];
    EDGES.forEach(function (e, k) { if (e[0] === node || e[1] === node) opts.push(k); });
    var k = opts[Math.floor(Math.random() * opts.length)];
    walk.edge = k; walk.fwd = EDGES[k][0] === node; walk.t = 0;
  }

  // ── II  Frustum to cone; a vector becomes a matrix of vectors ──────────
  var F2 = FOCUS[2], GEN = 12;
  var coneCounts = [65, 65];
  for (i = 0; i < GEN; i++) coneCounts.push(2);
  var cone = line(coneCounts, COOL, 0.05, 0.9, 0.9);
  cone.mesh.position.set(F2.x + 1.7, F2.y - 0.2, F2.z + 0.8);
  cone.mesh.rotation.x = 0.22;
  var AC = 4, AR = 5, ASP = 0.78, ACEN = new THREE.Vector3(F2.x - 2.3, F2.y + 0.2, F2.z - 1.0);
  var arrowCounts = [];
  for (i = 0; i < AC * AR; i++) arrowCounts.push(2, 3);
  arrowCounts.push(4, 4);
  var arrows = line(arrowCounts, PALE, 0.045, 0.95, 0.9);
  var BX = (AC - 1) / 2 * ASP + 0.6, BY = (AR - 1) / 2 * ASP + 0.55;
  [[-1, 0], [1, 1]].forEach(function (b) {
    var sgn = b[0], s = AC * AR * 2 + b[1];
    arrows.set(s, 0, ACEN.x + sgn * (BX - 0.25), ACEN.y + BY, ACEN.z);
    arrows.set(s, 1, ACEN.x + sgn * BX, ACEN.y + BY, ACEN.z);
    arrows.set(s, 2, ACEN.x + sgn * BX, ACEN.y - BY, ACEN.z);
    arrows.set(s, 3, ACEN.x + sgn * (BX - 0.25), ACEN.y - BY, ACEN.z);
  });

  // ── III  Curved space; two waves come into phase ──────────────────────
  var F3 = FOCUS[3], SV = 11, SH = 9, SVP = 28, SHP = 40;
  var sheetCounts = [];
  for (i = 0; i < SV; i++) sheetCounts.push(SVP);
  for (i = 0; i < SH; i++) sheetCounts.push(SHP);
  var sheet = line(sheetCounts, INK, 0.035, 0.85, 0.7);
  sheet.mesh.position.set(F3.x, F3.y, F3.z - 3.4);
  var WP = 140, WX = 4.8;
  var waves = line([WP, WP], '#ffffff', 0.075, 1, 1.0);
  waves.tintStrip(0, COOL); waves.tintStrip(1, GOLD); waves.tinted();
  var meetFlash = glow('#ffe9c8', 1.5, 0);
  meetFlash.position.set(F3.x, F3.y, F3.z + 0.1);
  function waveY(x, ph, k) { return 1.05 * Math.sin(1.25 * x + (k ? -1 : 1) * ph * Math.PI / 2) * (1 - 0.35 * smooth(3.4, 4.8, Math.abs(x))); }

  // ── IV  Heart curve, bound partition, random-access lattice ────────────
  var F4 = FOCUS[4], HP = 120, HS = 0.15;
  var heart = line([HP, HP, 2], GOLD, 0.085, 1, 1.0);
  for (i = 0; i < HP; i++) {
    var ht = i / (HP - 1) * Math.PI, sx3 = Math.pow(Math.sin(ht), 3);
    var hy = 13 * Math.cos(ht) - 5 * Math.cos(2 * ht) - 2 * Math.cos(3 * ht) - Math.cos(4 * ht);
    heart.set(0, i, F4.x - 16 * sx3 * HS, F4.y + (hy + 2.6) * HS, F4.z);
    heart.set(1, i, F4.x + 16 * sx3 * HS, F4.y + (hy + 2.6) * HS, F4.z);
  }
  heart.set(2, 0, F4.x, F4.y + 7.6 * HS, F4.z);
  heart.set(2, 1, F4.x, F4.y - 14.4 * HS, F4.z);
  heart.tintStrip(2, PALE, 0.55); heart.tinted(); heart.commit();
  var LC = 13, LR = 9, LS = 0.6, cells = [], cellLit = new Float32Array(LC * LR);
  for (var cy = 0; cy < LR; cy++) for (var cx = 0; cx < LC; cx++) {
    cells.push(F4.x + (cx - (LC - 1) / 2) * LS, F4.y + (cy - (LR - 1) / 2) * LS, F4.z - 3.4);
  }
  var cellGeo = new THREE.BufferGeometry();
  cellGeo.setAttribute('position', new THREE.Float32BufferAttribute(cells, 3));
  cellGeo.setAttribute('lit', new THREE.BufferAttribute(cellLit, 1));
  var cellMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uScale: { value: 1 }, uAlpha: { value: 0 } },
    vertexShader: 'attribute float lit; uniform float uScale; varying float vLit;\n' +
      'void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv; vLit = lit; gl_PointSize = uScale / -mv.z; }',
    fragmentShader: 'uniform float uAlpha; varying float vLit;\n' +
      'void main(){ vec2 c = abs(gl_PointCoord - 0.5); float m = max(c.x, c.y);\n' +
      ' float edge = smoothstep(0.24, 0.27, m) * (1.0 - smoothstep(0.29, 0.32, m));\n' +
      ' float fill = (1.0 - smoothstep(0.18, 0.24, m)) * vLit;\n' +
      ' vec3 col = vec3(0.25, 0.42, 0.85) * edge * 0.3 + vec3(1.0, 0.78, 0.46) * fill * 0.8;\n' +
      ' gl_FragColor = vec4(col * uAlpha, 1.0);\n #include <colorspace_fragment>\n }'
  });
  var cellPts = new THREE.Points(cellGeo, cellMat);
  cellPts.frustumCulled = false;
  world.add(cellPts);
  var cellClock = 0;
  var constants = [['π', -3.2, 1.9], ['e', 3.3, 1.7], ['φ', -3.5, -1.1], ['γ', 3.0, -1.6], ['i', 0.1, 2.6], ['c', 2.0, -2.9]]
    .map(function (c) {
      var s = new THREE.Sprite(new THREE.SpriteMaterial({ map: glyphTexture(c[0], 64, 64, 46), color: GOLD,
        blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0 }));
      s.position.set(F4.x + c[1], F4.y + c[2], F4.z + 0.4);
      s.scale.setScalar(0.62);
      s.userData.y = s.position.y;
      world.add(s);
      return s;
    });

  // ── V  Fourier epicycles spelling a square wave ────────────────────────
  var F5 = FOCUS[5], HARM = [1, 3, 5, 7, 9], AMP = 1.15, CP = 64, FW = 150, FLEN = 4.9;
  var FC = new THREE.Vector3(F5.x - 2.4, F5.y, F5.z), WAVE0 = F5.x + 0.15;
  var fCounts = HARM.map(function () { return CP; });
  fCounts.push(HARM.length + 1, 2);
  var epi = line(fCounts, COOL, 0.04, 0.85, 0.85);
  epi.tintStrip(HARM.length, PALE); epi.tintStrip(HARM.length + 1, PALE, 0.4); epi.tinted();
  var fwave = line([FW], GOLD, 0.08, 1, 1.0);
  var harmonics = line([FW, FW, FW], COOL, 0.03, 0.35, 0.6);
  var pen = new THREE.Vector3();

  // ── VI  A torus and a node, a root or two ──────────────────────────────
  var F6 = FOCUS[6], TM = 16, TPn = 8, TR = 2.0, Tr = 0.78;
  var tCounts = [];
  for (i = 0; i < TM; i++) tCounts.push(33);
  for (i = 0; i < TPn; i++) tCounts.push(65);
  var torus = line(tCounts, VIOLET, 0.035, 0.8, 0.8);
  for (i = 0; i < TM; i++) {
    var tu = i / TM * Math.PI * 2;
    for (var j = 0; j < 33; j++) {
      var tv = j / 32 * Math.PI * 2;
      torus.set(i, j, (TR + Tr * Math.cos(tv)) * Math.cos(tu), Tr * Math.sin(tv), (TR + Tr * Math.cos(tv)) * Math.sin(tu));
    }
  }
  for (i = 0; i < TPn; i++) {
    var tv2 = i / TPn * Math.PI * 2;
    for (j = 0; j < 65; j++) {
      var tu2 = j / 64 * Math.PI * 2;
      torus.set(TM + i, j, (TR + Tr * Math.cos(tv2)) * Math.cos(tu2), Tr * Math.sin(tv2), (TR + Tr * Math.cos(tv2)) * Math.sin(tu2));
    }
  }
  torus.commit();
  torus.mesh.position.set(F6.x, F6.y + 0.5, F6.z);
  var node = glow('#e6e2ff', 0.9, 0);
  node.position.set(F6.x, F6.y + 0.5, F6.z);
  var roots = line([2, 60], INK, 0.03, 0.8, 0.7);
  var RY = F6.y - 2.5;
  roots.set(0, 0, F6.x - 3.6, RY, F6.z); roots.set(0, 1, F6.x + 3.6, RY, F6.z);
  for (i = 0; i < 60; i++) { var rx = -2.4 + i / 59 * 4.8; roots.set(1, i, F6.x + rx, RY + 0.42 * (rx * rx - 1.69), F6.z); }
  roots.tintStrip(1, VIOLET, 0.8); roots.tinted(); roots.commit();
  var rootDots = [-1.3, 1.3].map(function (x) { var s = glow('#c8c4ff', 0.4, 0); s.position.set(F6.x + x, RY, F6.z); return s; });

  // ── VII  Ellipses converging; a haversine dancing ──────────────────────
  var F7 = FOCUS[7], EN = 7, EPN = 96, EC = new THREE.Vector3(F7.x, F7.y + 0.55, F7.z);
  var ell = line(Array(EN).fill(EPN), '#ffffff', 0.05, 0.55, 0.9);
  var ellK = [];
  for (i = 0; i < EN; i++) {
    ellK.push({ a: 1.5 + r() * 1.7, b: 0.5 + r() * 1.0, rot: r() * Math.PI, spin: (r() - 0.5) * 0.5 });
    ell.tintStrip(i, i % 2 ? GOLD : COOL);
  }
  ell.tinted();
  var hav = line([160], PALE, 0.06, 1, 1.0);
  var foci = [new THREE.Vector3(), new THREE.Vector3()];

  // ── VIII  The lemniscate of Bernoulli, r² = a² cos 2θ ─────────────────
  var F8 = FOCUS[8], LA = 3.3, LP = 200;
  function lemPoint(t, out) {
    var s = Math.sin(t), d = 1 + s * s;
    return out.set(F8.x + LA * Math.cos(t) / d, F8.y + LA * s * Math.cos(t) / d, F8.z);
  }
  var lemCore = line([LP, LP], '#fff4e2', 0.07, 1, 1.0), lemHalo = line([LP, LP], GOLD, 0.5, 0.0, 0.0);
  var lp = new THREE.Vector3();
  [lemCore, lemHalo].forEach(function (g) {
    for (var j2 = 0; j2 < LP; j2++) {
      var u2 = j2 / (LP - 1) * Math.PI;
      lemPoint(Math.PI / 2 - u2, lp); g.set(0, j2, lp.x, lp.y, lp.z);
      lemPoint(Math.PI / 2 + u2, lp); g.set(1, j2, lp.x, lp.y, lp.z);
    }
    g.commit();
  });
  var polarCounts = [97, 97, 97, 97];
  for (i = 0; i < 12; i++) polarCounts.push(2);
  var polar = line(polarCounts, INK, 0.03, 0.0, 0.6);
  for (i = 0; i < 4; i++) for (j = 0; j < 97; j++) {
    var pa = j / 96 * Math.PI * 2;
    polar.set(i, j, F8.x + Math.cos(pa) * (i + 1), F8.y + Math.sin(pa) * (i + 1), F8.z - 0.3);
  }
  for (i = 0; i < 12; i++) {
    var ra = i / 12 * Math.PI * 2;
    polar.set(4 + i, 0, F8.x + Math.cos(ra) * 0.4, F8.y + Math.sin(ra) * 0.4, F8.z - 0.3);
    polar.set(4 + i, 1, F8.x + Math.cos(ra) * 4.3, F8.y + Math.sin(ra) * 4.3, F8.z - 0.3);
  }
  polar.commit();
  var lemNode = glow('#fff1d8', 1.2, 0);
  lemNode.position.copy(F8);
  var formula = new THREE.Sprite(new THREE.SpriteMaterial({ map: glyphTexture('r² = a² cos 2θ', 512, 96, 54), color: GOLD,
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0 }));
  formula.scale.set(3.4, 0.64, 1);
  formula.position.set(F8.x, F8.y - 3.3, F8.z);
  world.add(formula);

  // ── The two lights and their trails ────────────────────────────────────
  var TRN = 44, trails = line([TRN, TRN], '#ffffff', 0.06, 0.9, 0.8);
  for (i = 0; i < TRN; i++) {
    var fade = Math.pow(1 - i / (TRN - 1), 1.6);
    trails.tint(0, i, COOL, fade); trails.tint(1, i, GOLD, fade);
  }
  trails.tinted();
  var lights = [COOL, GOLD].map(function (c) {
    return { core: glow(c === COOL ? '#e4f0ff' : '#fff1d6', 0.32, 1), halo: glow(c, 1.6, 0.7),
             pos: new THREE.Vector3(), hist: new Float32Array(TRN * 3), fresh: true };
  });
  var dA = new THREE.Vector3(), dB = new THREE.Vector3(), eA = new THREE.Vector3(), eB = new THREE.Vector3();

  // Where the lights are at station k (they belong to its figure).
  function dyadAt(k, row, time, A, B) {
    var F = FOCUS[Math.min(k, 8)], t = time * slow;
    switch (k) {
      case 0: {
        var ang = t * 0.7;
        A.set(F.x + Math.cos(ang) * 1.3, F.y + Math.sin(ang * 2) * 0.15, F.z + Math.sin(ang) * 0.7);
        B.set(F.x - Math.cos(ang) * 1.3, F.y - Math.sin(ang * 2) * 0.15, F.z - Math.sin(ang) * 0.7);
        break;
      }
      case 1:
        A.set(F.x - VD, F.y + Math.sin(t * 1.1) * 0.08, F.z + 0.05);
        B.set(F.x + VD, F.y + Math.sin(t * 1.1 + 2) * 0.08, F.z + 0.05);
        break;
      case 2: {
        var cp = cone.mesh.position, apex = 1.7 + 0.3 * row[7];
        A.set(cp.x, cp.y + apex + 0.15, cp.z);
        B.set(ACEN.x + Math.cos(t * 0.5) * 0.3, ACEN.y + Math.sin(t * 0.5) * 0.3, ACEN.z + 0.1);
        break;
      }
      case 3: {
        var ph = row[9], xa = -WX * 0.92 * smooth(0, 1, ph);
        A.set(F3.x + xa, F3.y + waveY(xa, ph, 0), F3.z + 0.05);
        B.set(F3.x - xa, F3.y + waveY(-xa, ph, 1), F3.z + 0.05);
        break;
      }
      case 4:
        A.set(F4.x - 1.05, F4.y + 0.35 + Math.sin(t) * 0.06, F4.z + 0.1);
        B.set(F4.x + 1.05, F4.y + 0.35 + Math.sin(t + 1.5) * 0.06, F4.z + 0.1);
        break;
      case 5:
        A.copy(FC);
        B.copy(pen);
        break;
      case 6: {
        var apart = smooth(0, 1, row[12]);
        A.set(F.x - 1.0 - apart * 2.8, F.y + 1.6 + apart * 0.6, F.z - apart * 2);
        B.set(F.x + 1.0 + apart * 1.4, F.y - 0.6 - apart * 1.4, F.z - apart * 2);
        break;
      }
      case 7:
        A.copy(foci[0]);
        B.copy(foci[1]);
        break;
      default: {
        // Tracing the lemniscate, one lobe each, out from the node and back.
        var u = clamp(row[14], 0, 1);
        if (row[15] > 0) u = (u + Math.max(0, t - lemStart) * 0.07 * row[15]) % 1;
        lemPoint(Math.PI / 2 - u * Math.PI, A);
        lemPoint(Math.PI / 2 + u * Math.PI, B);
        A.z += 0.05; B.z += 0.05;
      }
    }
  }
  var lemStart = 0;

  // ── Camera path through the stations ──────────────────────────────────
  var camCurve = null, lookCurve = null, portrait = null;
  function buildPath(p) {
    portrait = p;
    var cams = [], looks = [];
    for (var k = 0; k < 10; k++) {
      var F = FOCUS[Math.min(k, 8)], s = SIDE[k];
      if (k === 0) {
        cams.push(new THREE.Vector3(0, -3.3, p ? 22 : 18));
        looks.push(new THREE.Vector3(0, p ? -3.4 : -1.2, 0));
      } else if (k === 9) {
        cams.push(new THREE.Vector3(F.x, F.y - 0.2, F.z + (p ? 22 : 15.5)));
        looks.push(new THREE.Vector3(F.x, F.y - (p ? 4.5 : 2.7), F.z));
      } else if (p) {
        // Portrait: the figure sits above the text, a little further off.
        cams.push(new THREE.Vector3(F.x, F.y - 0.6, F.z + 20));
        looks.push(new THREE.Vector3(F.x, F.y - 6.4, F.z));
      } else {
        cams.push(new THREE.Vector3(F.x - s * 4.8, F.y + 0.9, F.z + 13.5));
        looks.push(new THREE.Vector3(F.x - s * 4.8, F.y + 0.25, F.z));
      }
    }
    camCurve = new THREE.CatmullRomCurve3(cams);
    lookCurve = new THREE.CatmullRomCurve3(looks);
  }

  var look = new THREE.Vector3(), tmp = new THREE.Color(), tmp2 = new THREE.Color(), H = 800, P11 = 1;
  var NIGHT = new THREE.Color('#050c22'), VOID = new THREE.Color('#010205');

  function frame(f) {
    var row = f.row, time = f.time, dt = f.dt, s = clamp(row[0], 0, 9), dark = f.dark;
    var vennA = row[6], coneA = row[7], field = row[8], phase = row[9], heartA = row[10], four = row[11];
    var coll = row[12], ellA = row[13], lem = row[14], glowA = row[15];
    var light = 1 - dark * 0.92;

    // Camera.
    camCurve.getPoint(s / 9, camera.position);
    lookCurve.getPoint(s / 9, look);
    camera.position.y += Math.sin(time * 0.5) * 0.06;
    camera.lookAt(look);
    camera.rotateY(row[4] - f.mx * 0.07);
    camera.rotateX(row[5] - f.my * 0.04);
    sky.position.copy(camera.position);

    // The space: darkens for the null domain.
    dome.uniforms.dark.value = dark * 0.9;
    world.fog.color.copy(NIGHT).lerp(VOID, dark);
    gl.setClearColor(world.fog.color);
    floorMinor.opacity = 0.22 * (1 - dark * 0.95);
    floorMajor.opacity = 0.4 * (1 - dark * 0.95);
    dustMat.uniforms.uTime.value = time;
    dustMat.uniforms.uDrift.value += dt * (0.15 + f.wind * 1.6) * slow;
    dustMat.uniforms.uAlpha.value = f.snow * (1 - dark * 0.85);
    for (var g = 0; g < floaters.length; g++) {
      var fl = floaters[g];
      // Symbols close to the camera fade, so none looms out of focus.
      fl.material.opacity = fl.userData.base * (1 - dark) * smooth(9, 16, fl.position.distanceTo(camera.position));
      fl.position.y = fl.userData.y + Math.sin(time * 0.25 + fl.userData.ph) * 0.4;
    }
    lit.forEach(function (l) { l.mat.uniforms.uOpacity.value = l.base * l.k * light; });

    // I  Venn and the chain.
    venn.mat.uniforms.uDraw.value = smooth(0, 0.75, vennA);
    discs.forEach(function (d) { d.material.opacity = smooth(0.55, 1, vennA) * 0.09 * light; });
    var mk = smooth(0.4, 1, vennA);
    markov.mat.uniforms.uDraw.value = mk;
    nodeGlows.forEach(function (n, k) { n.material.opacity = smooth(0.4 + k * 0.05, 0.6 + k * 0.05, vennA) * 0.8 * light; });
    nodeLabels.forEach(function (n, k) { n.material.opacity = smooth(0.5 + k * 0.05, 0.7 + k * 0.05, vennA) * 0.55 * light; });
    walk.t += dt * 0.9 * slow;
    if (walk.t >= 1) nextEdge(EDGES[walk.edge][walk.fwd ? 1 : 0]);
    edgeCurves[walk.edge].getPoint(walk.fwd ? walk.t : 1 - walk.t, walker.position);
    walker.material.opacity = smooth(0.85, 1, vennA) * light;

    // II  The frustum closes to a cone; the arrows sway in the breeze.
    var topR = 0.95 * (1 - smooth(0, 1, coneA)), topY = 1.7 + 0.3 * coneA;
    for (var j = 0; j < 65; j++) {
      var a = j / 64 * Math.PI * 2, ca = Math.cos(a), sa = Math.sin(a);
      cone.set(0, j, ca * 1.7, -1.7, sa * 1.7);
      cone.set(1, j, ca * topR, topY, sa * topR);
    }
    for (j = 0; j < GEN; j++) {
      var ga = j / GEN * Math.PI * 2;
      cone.set(2 + j, 0, Math.cos(ga) * 1.7, -1.7, Math.sin(ga) * 1.7);
      cone.set(2 + j, 1, Math.cos(ga) * topR, topY, Math.sin(ga) * topR);
    }
    cone.commit();
    cone.mesh.rotation.y = time * 0.22 * slow;
    var breeze = f.wind, tw = time * slow;
    for (var ai = 0; ai < AC * AR; ai++) {
      var col = ai % AC, rw = Math.floor(ai / AC);
      var ax = ACEN.x + (col - (AC - 1) / 2) * ASP, ay = ACEN.y + (rw - (AR - 1) / 2) * ASP;
      // The gradient of a slowly moving field, leaning with the breeze.
      var gx = 0.9 * Math.cos(0.9 * ax + tw * 0.5) + breeze * 1.4, gy = -1.1 * Math.sin(1.1 * ay - tw * 0.35) * 0.8 + breeze * 0.3;
      var gl2 = Math.hypot(gx, gy) || 1, ux = gx / gl2, uy = gy / gl2, L = 0.27;
      var tx = ax + ux * L, ty = ay + uy * L;
      arrows.set(ai * 2, 0, ax - ux * L, ay - uy * L, ACEN.z);
      arrows.set(ai * 2, 1, tx, ty, ACEN.z);
      arrows.set(ai * 2 + 1, 0, tx - ux * 0.16 - uy * 0.1, ty - uy * 0.16 + ux * 0.1, ACEN.z);
      arrows.set(ai * 2 + 1, 1, tx, ty, ACEN.z);
      arrows.set(ai * 2 + 1, 2, tx - ux * 0.16 + uy * 0.1, ty - uy * 0.16 - ux * 0.1, ACEN.z);
      // One vector first (the centre), then the matrix fills in around it.
      var dn = Math.hypot(col - (AC - 1) / 2, rw - (AR - 1) / 2) / 2.5, al = ai === 9 ? 1 : clamp((field * 1.5 - dn) * 3, 0, 1);
      arrows.tintStrip(ai * 2, ai === 9 ? GOLD : PALE, al);
      arrows.tintStrip(ai * 2 + 1, ai === 9 ? GOLD : PALE, al);
    }
    arrows.tintStrip(AC * AR * 2, PALE, smooth(0.6, 1, field));
    arrows.tintStrip(AC * AR * 2 + 1, PALE, smooth(0.6, 1, field));
    arrows.tinted();
    arrows.commit();

    // III  The sheet dents where the waves meet.
    var meet = 1 - smooth(0, 0.25, phase);
    for (var v = 0; v < SV; v++) for (j = 0; j < SVP; j++) {
      var x = -5.5 + v * 1.1, y = -3.6 + j / (SVP - 1) * 7.2;
      sheet.set(v, j, x, y, 0.06 * x * x - 0.04 * y * y - 1.6 * meet * Math.exp(-(x * x + y * y) / 3.5));
    }
    for (var hh = 0; hh < SH; hh++) for (j = 0; j < SHP; j++) {
      var x2 = -5.5 + j / (SHP - 1) * 11, y2 = -3.6 + hh * 0.9;
      sheet.set(SV + hh, j, x2, y2, 0.06 * x2 * x2 - 0.04 * y2 * y2 - 1.6 * meet * Math.exp(-(x2 * x2 + y2 * y2) / 3.5));
    }
    sheet.commit();
    for (j = 0; j < WP; j++) {
      var wx = -WX + j / (WP - 1) * 2 * WX;
      waves.set(0, j, F3.x + wx, F3.y + waveY(wx, phase, 0), F3.z);
      waves.set(1, j, F3.x + wx, F3.y + waveY(wx, phase, 1), F3.z);
    }
    waves.commit();
    meetFlash.material.opacity = meet * (0.55 + 0.15 * Math.sin(time * 2)) * light;

    // IV  The heart is drawn from its cusp; memory flickers behind it.
    heart.mat.uniforms.uDraw.value = smooth(0, 0.8, heartA);
    cellMat.uniforms.uAlpha.value = smooth(0, 0.5, heartA) * light;
    cellClock += dt;
    if (cellClock > 0.09) {
      cellClock = 0;
      cellLit[Math.floor(Math.random() * cellLit.length)] = 1;
    }
    for (j = 0; j < cellLit.length; j++) cellLit[j] *= Math.exp(-dt * 2.6);
    cellGeo.attributes.lit.needsUpdate = true;
    constants.forEach(function (c, k) {
      c.material.opacity = smooth(0.5 + k * 0.07, 0.75 + k * 0.07, heartA) * 0.7 * light;
      c.position.y = c.userData.y + Math.sin(time * 0.6 + k) * 0.08;
    });

    // V  Epicycles: each harmonic switches on in turn.
    var th = time * 0.9 * slow;
    pen.copy(FC);
    for (var hi = 0; hi < HARM.length; hi++) {
      var n = HARM[hi], rad = 4 * AMP / (n * Math.PI) * smooth(hi * 0.18, hi * 0.18 + 0.2, four);
      for (j = 0; j < CP; j++) {
        var ea = j / (CP - 1) * Math.PI * 2;
        epi.set(hi, j, pen.x + Math.cos(ea) * rad, pen.y + Math.sin(ea) * rad, pen.z);
      }
      epi.set(HARM.length, hi, pen.x, pen.y, pen.z);
      pen.x += Math.cos(n * th) * rad;
      pen.y += Math.sin(n * th) * rad;
    }
    epi.set(HARM.length, HARM.length, pen.x, pen.y, pen.z);
    epi.set(HARM.length + 1, 0, pen.x, pen.y, pen.z);
    epi.set(HARM.length + 1, 1, WAVE0, pen.y, pen.z);
    epi.commit();
    // The trace is the partial sum at earlier times, so no history is kept.
    for (j = 0; j < FW; j++) {
      var fx = j / (FW - 1) * FLEN, phs = th - fx * 1.25, fy = 0;
      for (hi = 0; hi < HARM.length; hi++) {
        fy += 4 * AMP / (HARM[hi] * Math.PI) * smooth(hi * 0.18, hi * 0.18 + 0.2, four) * Math.sin(HARM[hi] * phs);
      }
      fwave.set(0, j, WAVE0 + fx, FC.y + fy, FC.z);
      for (var hk = 0; hk < 3; hk++) {
        var hn = HARM[hk + 1];
        harmonics.set(hk, j, WAVE0 + fx, FC.y + 2.3 + hk * 0.42 + 0.45 / hn * Math.sin(hn * phs), FC.z - 0.8);
      }
    }
    fwave.commit();
    harmonics.commit();
    litOf(harmonics).k = smooth(0.3, 0.9, four);

    // VI  Things fall in to a torus and a node; then the null domain.
    var gather = smooth(0, 0.45, coll), ghost = 1 - 0.7 * smooth(0.75, 1, coll);
    torus.mesh.scale.setScalar(lerp(3.2, 1, gather));
    torus.mesh.rotation.set(1.1 + Math.sin(time * 0.2) * 0.1, time * 0.18 * slow, 0.3);
    var torusLit = litOf(torus);
    torusLit.k = gather * ghost / Math.max(light, 0.08);
    litOf(roots).k = smooth(0.25, 0.6, coll) * ghost / Math.max(light, 0.08);
    node.material.opacity = smooth(0.3, 0.55, coll) * (0.6 + 0.4 * Math.sin(time * 2.2)) * (0.4 + 0.6 * ghost);
    rootDots.forEach(function (d) { d.material.opacity = smooth(0.35, 0.6, coll) * 0.8 * ghost; });

    // VII  Ellipses converge to one; it rounds to a circle, the foci meet.
    var e1 = smooth(0, 0.7, ellA), e2 = smooth(0.7, 1, ellA);
    for (var ei = 0; ei < EN; ei++) {
      var K = ellK[ei];
      var spread = (ei - (EN - 1) / 2) * 0.012;   // a bundle, not one line
      var ea2 = lerp(lerp(K.a, 2.6 + spread, e1), 1.95 + spread, e2), eb = lerp(lerp(K.b, 1.35 + spread, e1), 1.95 + spread, e2);
      var rot = lerp(K.rot + time * K.spin * slow, spread, e1), cr = Math.cos(rot), sr = Math.sin(rot);
      for (j = 0; j < EPN; j++) {
        var et = j / (EPN - 1) * Math.PI * 2, px = Math.cos(et) * ea2, py = Math.sin(et) * eb;
        ell.set(ei, j, EC.x + px * cr - py * sr, EC.y + px * sr + py * cr, EC.z);
      }
    }
    ell.commit();
    litOf(ell).k = 0.75 - 0.3 * e1;   // seven lines laid on one: keep the sum soft
    var fa = lerp(2.6, 1.95, e2), fb = lerp(1.35, 1.95, e2), fc = Math.sqrt(Math.max(fa * fa - fb * fb, 0)) * e1 + 1.6 * (1 - e1);
    foci[0].set(EC.x - fc, EC.y, EC.z + 0.05);
    foci[1].set(EC.x + fc, EC.y, EC.z + 0.05);
    var caper = 0.75 + 0.25 * Math.sin(time * 2.7 * slow);
    for (j = 0; j < 160; j++) {
      var hx = -4.4 + j / 159 * 8.8, hth = hx * 1.9 - time * 2.4 * slow;
      // Each hump hops to its own beat: hav θ = (1 - cos θ) / 2.
      var hump = Math.floor(hth / (Math.PI * 2)), skip = caper * (0.45 + 0.55 * Math.abs(Math.sin(time * 2.2 * slow + hump * 1.9)));
      hav.set(0, j, EC.x + hx, F7.y - 2.85 + skip * 1.5 * (1 - Math.cos(hth)) / 2 * (1 - 0.7 * smooth(3.4, 4.4, Math.abs(hx))), EC.z);
    }
    hav.commit();

    // VIII  The lemniscate drawn from its node, then glowing.
    if (glowA <= 0) lemStart = time;
    lemCore.mat.uniforms.uDraw.value = lem;
    lemHalo.mat.uniforms.uDraw.value = lem;
    litOf(lemHalo).base = 0.14 + glowA * 0.2;
    litOf(lemCore).k = 1 + glowA * 0.4;
    litOf(polar).base = smooth(0, 0.5, lem) * 0.45 * (1 - 0.35 * glowA);
    lemNode.material.opacity = (smooth(0.97, 1, lem) * 0.25 + glowA * 0.2) * (0.85 + 0.15 * Math.sin(time * 1.7));
    formula.material.opacity = glowA * 0.75;

    // The two lights: at the figure of the station you are at, flying on
    // to the next as you travel.
    var k0 = Math.floor(s), fr = s - k0, w = smooth(0.05, 0.75, fr);
    dyadAt(k0, row, time, dA, dB);
    if (w > 0 && k0 < 8) {
      dyadAt(k0 + 1, row, time, eA, eB);
      var arc = Math.sin(Math.PI * w);
      dA.lerp(eA, w); dB.lerp(eB, w);
      dA.x -= arc * 1.2; dA.y += arc * 1.4;
      dB.x += arc * 1.2; dB.y += arc * 0.6;
    }
    var bright = (1 - 0.85 * smooth(0.35, 1, dark)) * (1 - 0.5 * smooth(0.3, 1, coll) * (1 - smooth(0, 0.3, ellA)));
    [dA, dB].forEach(function (p, k) {
      var L2 = lights[k], hist = L2.hist;
      if (L2.fresh || L2.pos.distanceToSquared(p) > 64) {
        for (var q = 0; q < TRN; q++) { hist[q * 3] = p.x; hist[q * 3 + 1] = p.y; hist[q * 3 + 2] = p.z; }
        L2.fresh = false;
      } else {
        hist.copyWithin(3, 0, (TRN - 1) * 3);
        hist[0] = p.x; hist[1] = p.y; hist[2] = p.z;
      }
      L2.pos.copy(p);
      for (q = 0; q < TRN; q++) trails.set(k, q, hist[q * 3], hist[q * 3 + 1], hist[q * 3 + 2]);
      L2.core.position.copy(p);
      L2.halo.position.copy(p);
      L2.core.material.opacity = bright;
      L2.halo.material.opacity = 0.65 * bright * (0.9 + 0.1 * Math.sin(time * 2 + k));
    });
    trails.commit();
    litOf(trails).k = bright / Math.max(light, 0.08);

    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) {
      H = h;
      fitCamera(gl, camera, w, h, dpr, small);
      gl.getDrawingBufferSize(SHARED.res.value);
      P11 = camera.projectionMatrix.elements[5];
      var bufH = SHARED.res.value.y;
      dustMat.uniforms.uScale.value = bufH / 800 * 9;
      cellMat.uniforms.uScale.value = 0.5 * P11 * bufH / 2;
      if (portrait !== (w / h < 1)) buildPath(w / h < 1);
    },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('lemniscate', {
  renderer: renderer3d,
  accent: '#ffcf86',
  // The heart and love of stanza IV, and Bernoulli's curve at the end.
  emphasis: /^\W*(heart|love)|^(a\^2|cos|2|ψ\W*)$/i,
  align: ['left', 'right', 'left', 'right', 'left', 'right', 'left', 'right'],
  // One panel per stanza (8). Each figure is drawn while its stanza reads.
  keys: function (T) {
    function at(i, d) { return T.start(i) + d; }
    //  unit          stn   dark  dust wind  yaw   pitch venn cone field phase heart four  coll  ell   lem   glow
    return [
      [0,             0.00, 0.00, 0.8, 0.15, 0.00, 0.00, 0,   0,   0,    1,    0,    0,    0,    0,    0,    0],
      [0.7,           0.04, 0.00, 0.8, 0.15, 0.00, 0.00, 0,   0,   0,    1,    0,    0,    0,    0,    0,    0],
      [at(0, 0.3),    0.97, 0.00, 0.8, 0.15, 0.00, 0.00, 0.1, 0,   0,    1,    0,    0,    0,    0,    0,    0],
      [at(0, 0.8),    1.00, 0.00, 0.8, 0.15, 0.00, 0.00, 0.75, 0,  0,    1,    0,    0,    0,    0,    0,    0],  // "fairy fields of Venn"
      [at(0, 1.25),   1.03, 0.00, 0.8, 0.20, 0.00, 0.00, 1,   0,   0,    1,    0,    0,    0,    0,    0,    0],  // "an endless Markov chain"
      [at(1, 0.3),    1.97, 0.00, 0.8, 0.25, 0.00, 0.00, 1,   0,   0,    1,    0,    0,    0,    0,    0,    0],
      [at(1, 0.6),    2.00, 0.00, 0.8, 0.25, 0.00, 0.00, 1,   1,   0,    1,    0,    0,    0,    0,    0,    0],  // "longs to be a cone"
      [at(1, 0.95),   2.02, 0.00, 0.8, 0.35, 0.00, 0.00, 1,   1,   1,    1,    0,    0,    0,    0,    0,    0],  // "dreams of matrices"
      [at(1, 1.3),    2.04, 0.00, 0.9, 0.85, 0.00, 0.00, 1,   1,   1,    1,    0,    0,    0,    0,    0,    0],  // "gradient of the breeze"
      [at(2, 0.3),    2.97, 0.00, 0.8, 0.30, 0.00, 0.00, 1,   1,   1,    1,    0,    0,    0,    0,    0,    0],
      [at(2, 0.55),   2.99, 0.00, 0.8, 0.20, 0.00, 0.00, 1,   1,   1,    0.85, 0,   0,    0,    0,    0,    0],
      [at(2, 1.15),   3.02, 0.00, 0.8, 0.20, 0.00, 0.00, 1,   1,   1,    0,    0,    0,    0,    0,    0,    0],  // "face to face"
      [at(3, 0.3),    3.97, 0.00, 0.8, 0.20, 0.00, 0.00, 1,   1,   1,    0,    0,    0,    0,    0,    0,    0],
      [at(3, 0.95),   4.00, 0.00, 0.8, 0.20, 0.00, 0.00, 1,   1,   1,    0,    1,    0,    0,    0,    0,    0],  // "random access to my heart"
      [at(3, 1.3),    4.03, 0.00, 0.8, 0.20, 0.00, 0.00, 1,   1,   1,    0,    1,    0,    0,    0,    0,    0],
      [at(4, 0.3),    4.97, 0.00, 0.8, 0.20, 0.00, 0.00, 1,   1,   1,    0,    1,    0.05, 0,    0,    0,    0],
      [at(4, 1.2),    5.02, 0.00, 0.8, 0.20, 0.00, 0.00, 1,   1,   1,    0,    1,    1,    0,    0,    0,    0],  // "sinusoidal spell"
      [at(5, 0.3),    5.97, 0.30, 0.6, 0.10, 0.00, 0.00, 1,   1,   1,    0,    1,    1,    0.1,  0,    0,    0],  // "cancel me not"
      [at(5, 0.75),   6.00, 0.55, 0.5, 0.05, 0.00, 0.00, 1,   1,   1,    0,    1,    1,    0.6,  0,    0,    0],  // "a torus and a node"
      [at(5, 1.25),   6.02, 0.95, 0.3, 0.05, 0.00, 0.00, 1,   1,   1,    0,    1,    1,    1,    0,    0,    0],  // "a null domain"
      [at(6, 0.3),    6.97, 0.10, 0.8, 0.20, 0.00, 0.00, 1,   1,   1,    0,    1,    1,    1,    0.1,  0,    0],
      [at(6, 0.85),   7.00, 0.00, 0.8, 0.25, 0.00, 0.00, 1,   1,   1,    0,    1,    1,    1,    0.75, 0,    0],  // "ellipse of bliss, converge"
      [at(6, 1.2),    7.03, 0.00, 0.8, 0.30, 0.00, 0.00, 1,   1,   1,    0,    1,    1,    1,    1,    0,    0],  // "O lips divine"
      [at(7, 0.3),    7.97, 0.00, 0.8, 0.20, 0.00, 0.00, 1,   1,   1,    0,    1,    1,    1,    1,    0,    0],
      [at(7, 0.45),   7.99, 0.00, 0.8, 0.20, 0.00, 0.00, 1,   1,   1,    0,    1,    1,    1,    1,    0.03, 0],
      [at(7, 1.15),   8.03, 0.00, 0.9, 0.20, 0.00, 0.00, 1,   1,   1,    0,    1,    1,    1,    1,    1,    0],  // "a² cos 2ψ"
      [at(7, 1.6),    8.60, 0.00, 1.0, 0.20, 0.00, 0.00, 1,   1,   1,    0,    1,    1,    1,    1,    1,    0.6],
      [T.total,       9.00, 0.00, 1.0, 0.20, 0.00, 0.00, 1,   1,   1,    0,    1,    1,    1,    1,    1,    1]
    ];
  },
  sound: {
    src: '/audio/wind.mp3',
    label: 'Play a faint wind and soft tones',
    volume: function (row) { return 0.03 + 0.05 * row[3] * (1 - row[1] * 0.5); },
    cues: [
      { stanza: 0, at: 0.6, play: dyadSound },
      { stanza: 1, at: 0.5, play: coneSound },
      { stanza: 2, at: 0.5, play: phaseSound },
      { stanza: 3, at: 0.5, play: accessSound },
      { stanza: 4, at: 0.4, play: fourierSound },
      { stanza: 5, at: 0.45, play: cancelSound },
      { stanza: 6, at: 0.45, play: convergeSound },
      { stanza: 7, at: 1.1, play: lemniscateSound }
    ]
  }
});
