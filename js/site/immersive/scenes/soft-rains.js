/*
 * Scene for "There Will Come Soft Rains" (Sara Teasdale): a small town
 * years after the last people left it, softly taken back by spring.
 *
 * I   Soft rain on the old road into town: wet ground, puddles ringing,
 *     swallows wheeling low over the grass.
 * II  Night by the pools at the roadside: rings on the water where the
 *     frogs are, wild plum trees in trembling white, fireflies.
 * III Morning: robins on a low fence wire, their breasts lit by the sun.
 * IV  "Not one will know of the war": a rusted tank in the field, sunk in
 *     grass and flowers, a robin perched on its gun.
 * V   "Neither bird nor tree": empty, roofless house shells at dusk, ivy
 *     climbing them, trees growing up through them.
 * VI  Spring wakes at dawn: up over the town in the mist, almost gone under
 *     green, as the sun comes up.
 *
 * Columns: [unit, path, light, rain, wind, yaw, pitch, lift, green, mist, petals]
 * `light` runs through the times of day in LIGHT (0 rain, 1 night, 2 morning,
 * 3 drizzle, 4 dusk, 5 dawn); `green` is how far ivy and grass have taken
 * the town.
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, broadleafGeometry, softSprite, starField,
         distanceTo, terrain, scatter, particleField, rainField, followPath, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout (metres; the old road runs into town towards -z) ──────────────
var ROAD = [[0, 66], [0.6, 46], [-0.7, 30], [0.4, 14], [1.0, 0], [0.4, -16], [-0.4, -34], [0.2, -50],
            [0, -66], [0.6, -84], [0, -104], [0, -126]];
var curve = new THREE.CatmullRomCurve3(ROAD.map(function (p) { return new THREE.Vector3(p[0], 0, p[1]); }));
var roadPts = curve.getSpacedPoints(150), distRoad = distanceTo(roadPts);
var POOLS = [[-8.4, 26, 4.2], [-14.8, 20.5, 3.0], [-10.2, 16.4, 2.0]];   // x, z, radius
var FENCE = { x: 3.5, z0: 14, z1: -10 };
var ROBINS = [[5.0, 2.7], [4.35, 3.5], [3.6, 2.4], [2.9, 3.1]];           // z on the wire, facing
var TANK = { x: -8.6, z: -17, rot: 0.75 };
// x, z, width, depth, height, turn, roof remnant, tree inside
var HOUSES = [[10.5, -31, 7.5, 8, 5.4, 0.04, 0, 1], [-10.5, -38, 8, 7.5, 4.8, -0.06, 1, 0],
              [11, -48, 6.5, 7, 4.6, 0.08, 1, 0], [-11, -55, 7, 8, 5.8, 0.03, 0, 1],
              [10.5, -64, 8, 7.5, 5.0, -0.05, 0, 0], [-10.5, -71, 6.5, 7, 4.6, 0.1, 1, 1],
              [11, -82, 7.5, 8, 5.6, 0.0, 0, 1], [-11.5, -89, 7.5, 7.5, 5.0, -0.08, 0, 0],
              [10.5, -99, 7, 7.5, 4.8, 0.06, 1, 0], [-2.5, -119, 9, 8, 6.2, 0.12, 0, 1]];
var PLUMS = [[-15.5, 29], [-19.5, 17.5], [-12, 33.5], [-13, 11.5], [-5.2, 19], [-21, 25], [7.5, 22],
             [16, -42], [-6.5, -62], [16, 6], [-18, -6], [7, -76]];

// Path progress where the road reaches a given z (the road runs down -z).
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

function height(x, z) {
  var h = (0.4 * Math.sin(x * 0.09 + 1) * Math.cos(z * 0.07) + 0.18 * Math.sin(x * 0.23 + z * 0.17)) *
          smooth(2.5, 10, distRoad(x, z));
  for (var i = 0; i < POOLS.length; i++) {
    var p = POOLS[i];
    h -= smooth(p[2] + 1.6, p[2] - 0.8, Math.hypot(x - p[0], z - p[1]));
  }
  var rr = Math.hypot(x * 0.9, z + 30);
  return h + smooth(95, 250, rr) * 42 * (0.55 + 0.45 * Math.sin(Math.atan2(z + 30, x) * 4 + 1));
}
function onRoad(x, z) { return height(x, z); }

// ── Light through the day, in the order the `light` column visits it ─────
var LIGHT = [
  // soft rain, afternoon
  { top: '#7f8c98', hor: '#c3cac6', cloud: '#c9cdcd', shade: '#848a90', fog: '#aeb6b2', sun: '#fff2dc', sunI: 0.7,
    sky: '#dfe6e8', ground: '#4c5c38', hemiI: 1.9, dir: [0.3, 0.6, -0.5], glow: 0.0, cover: 0.97, fogD: 0.011, exp: 1.0 },
  // night, a moon over the pools
  { top: '#060c20', hor: '#24345a', cloud: '#2e3b5a', shade: '#0d1426', fog: '#18233f', sun: '#a8bbee', sunI: 1.3,
    sky: '#5a6ca0', ground: '#101a16', hemiI: 1.05, dir: [-0.72, 0.2, -0.3], glow: 0.0, cover: 0.4, fogD: 0.008, exp: 1.0 },
  // clear morning
  { top: '#4c82c2', hor: '#ecdcc0', cloud: '#fff6ea', shade: '#aab3c4', fog: '#cfd2c4', sun: '#ffdcae', sunI: 2.9,
    sky: '#dbe8ff', ground: '#5a6a3a', hemiI: 1.5, dir: [-0.62, 0.32, 0.32], glow: 0.5, cover: 0.32, fogD: 0.006, exp: 1.0 },
  // drizzle
  { top: '#7c8994', hor: '#bec6c0', cloud: '#c3c8c6', shade: '#7c8388', fog: '#a7b0aa', sun: '#fff2dc', sunI: 0.8,
    sky: '#dce4e2', ground: '#4c5c36', hemiI: 1.9, dir: [0.2, 0.6, -0.5], glow: 0.0, cover: 0.9, fogD: 0.01, exp: 1.0 },
  // dusk
  { top: '#2a3466', hor: '#eba878', cloud: '#eab096', shade: '#5e5070', fog: '#a08a84', sun: '#ffb47a', sunI: 2.1,
    sky: '#c8a8bc', ground: '#3a3a2a', hemiI: 1.0, dir: [0.75, 0.1, -0.65], glow: 1.0, cover: 0.5, fogD: 0.007, exp: 1.05 },
  // dawn, mist in the hollows
  { top: '#46679e', hor: '#ffd6aa', cloud: '#ffe4cc', shade: '#a090ac', fog: '#d6cbbd', sun: '#ffd6a0', sunI: 2.3,
    sky: '#e2e6ff', ground: '#5a7040', hemiI: 1.5, dir: [-0.45, 0.08, -0.9], glow: 1.0, cover: 0.3, fogD: 0.0065, exp: 1.05 }
].map(function (L) {
  var o = { dir: new THREE.Vector3().fromArray(L.dir).normalize() };
  ['top', 'hor', 'cloud', 'shade', 'fog', 'sun', 'sky', 'ground'].forEach(function (k) { o[k] = new THREE.Color(L[k]); });
  ['sunI', 'hemiI', 'glow', 'cover', 'fogD', 'exp'].forEach(function (k) { o[k] = L[k]; });
  return o;
});
var LC = ['top', 'hor', 'cloud', 'shade', 'fog', 'sun', 'sky', 'ground'], LN = ['sunI', 'hemiI', 'glow', 'cover', 'fogD', 'exp'];

// ── Sound cues ───────────────────────────────────────────────────────────
// Swallows: quick bright twitters.
function twitter(ac, out) {
  var now = ac.currentTime;
  for (var i = 0; i < 9; i++) {
    var t = now + i * 0.1 + Math.random() * 0.06, o = ac.createOscillator(), g = ac.createGain(), f = 3600 + Math.random() * 1800;
    o.type = 'sine';
    o.frequency.setValueAtTime(f, t);
    o.frequency.exponentialRampToValueAtTime(f * (0.7 + Math.random() * 0.6), t + 0.06);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.05, t + 0.008);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.08);
    o.connect(g); g.connect(out);
    o.start(t); o.stop(t + 0.1);
  }
}

// Frogs: low croaks, a buzzy tone chopped by a fast tremolo.
function frogs(ac, out) {
  var now = ac.currentTime;
  for (var i = 0; i < 6; i++) {
    var t = now + i * 0.5 + Math.random() * 0.25, f = 150 + Math.random() * 120;
    var o = ac.createOscillator(), lfo = ac.createOscillator(), depth = ac.createGain(), trem = ac.createGain(),
        bp = ac.createBiquadFilter(), env = ac.createGain();
    o.type = 'sawtooth';
    o.frequency.setValueAtTime(f, t);
    o.frequency.linearRampToValueAtTime(f * 0.88, t + 0.35);
    lfo.frequency.value = 24 + Math.random() * 12;
    depth.gain.value = 0.5;
    trem.gain.value = 0.5;
    bp.type = 'bandpass';
    bp.frequency.value = f * 3;
    bp.Q.value = 3;
    env.gain.setValueAtTime(0.0001, t);
    env.gain.exponentialRampToValueAtTime(0.3, t + 0.04);
    env.gain.exponentialRampToValueAtTime(0.0001, t + 0.4);
    lfo.connect(depth); depth.connect(trem.gain);
    o.connect(trem); trem.connect(bp); bp.connect(env); env.connect(out);
    o.start(t); lfo.start(t); o.stop(t + 0.45); lfo.stop(t + 0.45);
  }
}

// A robin's phrase: clear whistled notes that slide up and down.
function robinSong(ac, out) {
  var now = ac.currentTime;
  [[2600, 3400], [3800, 3000], [2900, 2950], [3300, 4100], [2500, 3100], [3600, 3300]].forEach(function (n, i) {
    var t = now + i * 0.22 + Math.random() * 0.04, o = ac.createOscillator(), g = ac.createGain();
    o.type = 'sine';
    o.frequency.setValueAtTime(n[0], t);
    o.frequency.exponentialRampToValueAtTime(n[1], t + 0.14);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.06, t + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.17);
    o.connect(g); g.connect(out);
    o.start(t); o.stop(t + 0.2);
  });
}

// ── Shaders ──────────────────────────────────────────────────────────────
var NOISE3 = [
  'float vh(vec3 p){ return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }',
  'float vn(vec3 p){ vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);',
  ' return mix(mix(mix(vh(i), vh(i + vec3(1.0, 0.0, 0.0)), f.x), mix(vh(i + vec3(0.0, 1.0, 0.0)), vh(i + vec3(1.0, 1.0, 0.0)), f.x), f.y),',
  '            mix(mix(vh(i + vec3(0.0, 0.0, 1.0)), vh(i + vec3(1.0, 0.0, 1.0)), f.x), mix(vh(i + vec3(0.0, 1.0, 1.0)), vh(i + vec3(1.0, 1.0, 1.0)), f.x), f.y), f.z); }'
].join('\n') + '\n';

// Weathered stone or rusted steel that ivy climbs as `uGreen` rises; `reach`
// is how high (m) the ivy gets at uGreen = 1.
function overgrown(green, reach, rust, spread) {
  var mat = new THREE.MeshLambertMaterial({ vertexColors: true });
  mat.onBeforeCompile = function (sh) {
    sh.uniforms.uGreen = green;
    sh.vertexShader = 'varying vec3 vWPos;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n vWPos = (modelMatrix * vec4(position, 1.0)).xyz;');
    sh.fragmentShader = 'uniform float uGreen; varying vec3 vWPos;\n' + NOISE3 + sh.fragmentShader.replace('#include <color_fragment>',
      '#include <color_fragment>\n' +
      ' float n1 = vn(vWPos * 1.6), n2 = vn(vec3(vWPos.x * 2.6, vWPos.y * 0.3, vWPos.z * 2.6)), n3 = vn(vWPos * 7.0);\n' +
      (rust ? ' diffuseColor.rgb *= mix(vec3(0.55, 0.42, 0.36), vec3(1.25, 0.95, 0.7), n1) * (0.85 + 0.3 * n3);\n'
            : ' diffuseColor.rgb *= 0.78 + 0.34 * n1 - 0.12 * n3;\n') +
      ' diffuseColor.rgb *= mix(0.55, 1.0, smoothstep(0.0, 1.4, vWPos.y));\n' +              // damp at the foot
      ' float top = uGreen * ' + reach.toFixed(2) + ' + (n2 - 0.5) * ' + (spread || 3.2).toFixed(2) + ' + (n1 - 0.5) * 0.9 - 0.4;\n' +
      ' float ivy = smoothstep(top + 0.2, top - 0.2, vWPos.y) * smoothstep(0.02, 0.12, uGreen);\n' +
      ' vec3 leaf = mix(vec3(0.09, 0.18, 0.04), vec3(0.16, 0.27, 0.055), n3 * 0.7 + n1 * 0.3);\n' +
      ' diffuseColor.rgb = mix(diffuseColor.rgb, leaf, ivy);');
  };
  // Same source, different constants: keep the compiled programs apart.
  mat.customProgramCacheKey = function () { return 'overgrown' + reach + rust + spread; };
  return mat;
}

// Water that rings with drops: `uRain` sets how many rings, `uSky` is
// mirrored at grazing angles.
var RIPPLES = [
  'float rh(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }',
  'vec2 ripples(vec2 p){ vec2 g = vec2(0.0), b = floor(p);',
  ' for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {',
  '  vec2 c = b + vec2(float(i), float(j)); float k = rh(c);',
  '  if (k > uRain) continue;',
  '  vec2 o = c + 0.25 + 0.5 * vec2(rh(c + 1.7), rh(c + 4.3));',
  '  float t = fract(uTime * (0.55 + k * 0.5) + k * 7.0);',
  '  vec2 d = p - o; float r = length(d) + 0.0001; float x = (r - t * 1.3) * 9.0;',
  '  g += d / r * sin(x * 2.4) * exp(-x * x) * (1.0 - t) * (1.0 - t);',
  ' }',
  ' return g; }'
].join('\n') + '\n';

function waterMaterial(color, u) {
  var mat = new THREE.MeshPhongMaterial({ color: color, specular: '#7a7a7a', shininess: 160 });
  mat.onBeforeCompile = function (sh) {
    sh.uniforms.uTime = u.time;
    sh.uniforms.uRain = u.rings;
    sh.uniforms.uSky = u.sky;
    sh.vertexShader = 'varying vec3 vWPos;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n vWPos = (modelMatrix * vec4(position, 1.0)).xyz;');
    sh.fragmentShader = 'uniform float uTime; uniform float uRain; uniform vec3 uSky; varying vec3 vWPos;\n' + RIPPLES +
      sh.fragmentShader.replace('#include <normal_fragment_maps>',
        '#include <normal_fragment_maps>\n vec2 rg = ripples(vWPos.xz * 2.2);\n' +
        ' normal = normalize(normal + (viewMatrix * vec4(rg.x, 0.0, rg.y, 0.0)).xyz * 0.7);\n' +
        ' float fres = pow(1.0 - max(dot(normal, normalize(vViewPosition)), 0.0), 3.0);\n' +
        ' totalEmissiveRadiance += uSky * (0.12 + fres * 0.5);');
  };
  return mat;
}

// A dome of slow clouds over a gradient sky with a sun. `uCover` runs from
// clear (0) to overcast (1). Colours are as displayed (no tone mapping), so
// the horizon can match the fog.
function cloudDome(radius) {
  var u = {
    uTop: { value: new THREE.Color() }, uHorizon: { value: new THREE.Color() }, uCloud: { value: new THREE.Color() },
    uShade: { value: new THREE.Color() }, uSun: { value: new THREE.Color() }, uSunDir: { value: new THREE.Vector3(0, 1, 0) },
    uCover: { value: 0.8 }, uTime: { value: 0 }
  };
  var mat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false, uniforms: u,
    vertexShader: 'varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: [
      'uniform vec3 uTop; uniform vec3 uHorizon; uniform vec3 uCloud; uniform vec3 uShade; uniform vec3 uSun; uniform vec3 uSunDir;',
      'uniform float uCover; uniform float uTime; varying vec3 vP;',
      'float hh(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }',
      'float nn(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);',
      ' return mix(mix(hh(i), hh(i + vec2(1.0, 0.0)), f.x), mix(hh(i + vec2(0.0, 1.0)), hh(i + vec2(1.0, 1.0)), f.x), f.y); }',
      'float fbm(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 5; i++) { s += a * nn(p); p = p * 2.07 + vec2(3.1, 1.7); a *= 0.5; } return s; }',
      'void main(){',
      ' vec3 d = normalize(vP); float h = max(d.y, 0.0);',
      ' vec3 c = mix(uHorizon, uTop, smoothstep(0.0, 0.55, h));',
      ' float s = max(dot(d, normalize(uSunDir)), 0.0);',
      ' c += uSun * pow(s, 10.0) * 0.3;',
      ' c = mix(c, vec3(1.0, 0.97, 0.9), clamp(pow(s, 1800.0) * 2.5, 0.0, 1.0) * min(length(uSun), 1.0));',
      ' vec2 p = d.xz / (h + 0.1) * 0.8 + vec2(uTime * 0.01, uTime * 0.004);',
      ' float f = fbm(p) * 0.65 + fbm(p * 2.3 + 7.0) * 0.35;',
      ' float cov = smoothstep(1.0 - uCover, 1.0 - uCover + 0.3, f) * smoothstep(0.0, 0.2, h);',
      ' vec3 cl = mix(uCloud, uShade, smoothstep(0.4, 0.85, f));',
      ' cl += uSun * pow(s, 5.0) * 0.4 * (1.0 - smoothstep(0.45, 0.85, f));',
      ' c = mix(c, cl, cov);',
      ' c = mix(c, uHorizon, 1.0 - smoothstep(0.0, 0.07, d.y));',
      ' gl_FragColor = vec4(c, 1.0);',
      ' #include <colorspace_fragment>',
      '}'
    ].join('\n')
  });
  return { mesh: new THREE.Mesh(new THREE.SphereGeometry(radius, 32, 16), mat), uniforms: u };
}

// ── Pieces ───────────────────────────────────────────────────────────────
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

// Old asphalt: cracks with grass in them, a ghost of the centre line, and
// ragged edges where the verge has grown over (cut out with alpha).
function roadTexture(r) {
  return canvasTexture(256, 1024, function (x, w, h) {
    var i, k;
    x.fillStyle = '#56585a';
    x.fillRect(0, 0, w, h);
    for (i = 0; i < 5000; i++) {
      var v = Math.floor(60 + r() * 50);
      x.fillStyle = 'rgba(' + v + ',' + v + ',' + (v + 2) + ',0.4)';
      x.fillRect(r() * w, r() * h, 2, 2);
    }
    x.fillStyle = 'rgba(214,208,176,0.22)';
    for (k = 0; k < h; k += 170) x.fillRect(123 + r() * 3, k + r() * 20, 9, 70 + r() * 30);
    for (i = 0; i < 46; i++) {
      var cx = r() * w, cy = r() * h, pts = [[cx, cy]];
      for (k = 0; k < 8; k++) { cx += (r() - 0.5) * 40; cy += (r() - 0.3) * 36; pts.push([cx, cy]); }
      x.lineWidth = 2;
      x.strokeStyle = '#2c2d2b';
      x.beginPath(); pts.forEach(function (p, j) { if (j) x.lineTo(p[0], p[1]); else x.moveTo(p[0], p[1]); }); x.stroke();
      if (r() < 0.6) {
        x.lineWidth = 5;
        x.strokeStyle = 'rgba(86,118,48,0.85)';
        x.stroke();
      }
    }
    for (i = 0; i < 70; i++) {
      x.fillStyle = r() < 0.5 ? 'rgba(78,108,44,0.75)' : 'rgba(104,128,58,0.6)';
      x.beginPath(); x.ellipse(r() * w, r() * h, 6 + r() * 22, 4 + r() * 14, r() * 3, 0, Math.PI * 2); x.fill();
    }
    // The verge growing in from both sides.
    x.globalCompositeOperation = 'destination-out';
    for (i = 0; i < 260; i++) {
      var side = r() < 0.5 ? 0 : w, rad = 10 + r() * 34;
      x.beginPath(); x.ellipse(side + (side ? -1 : 1) * r() * 30, r() * h, rad, rad * 1.6, 0, 0, Math.PI * 2); x.fill();
    }
    x.globalCompositeOperation = 'source-over';
  });
}

// A strip that follows the ground, with uv (across, along) for the texture.
function roadGeometry(points, width, lift) {
  var pos = [], uv = [], idx = [], along = 0;
  for (var j = 0; j < points.length; j++) {
    var a = points[Math.max(j - 1, 0)], b = points[Math.min(j + 1, points.length - 1)], p = points[j];
    var dx = b.x - a.x, dz = b.z - a.z, len = Math.hypot(dx, dz) || 1, nx = -dz / len, nz = dx / len;
    if (j > 0) along += p.distanceTo(points[j - 1]);
    for (var s = 0; s <= 4; s++) {
      var o = (s / 4 - 0.5) * width, x = p.x + nx * o, z = p.z + nz * o;
      pos.push(x, height(x, z) + lift, z);
      uv.push(s / 4, along / 9);
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

// A roofless house shell: four walls with window and door holes, broken
// along the top, sometimes a chimney and a few rafters or a slab of roof.
var STONES = ['#b9ad9a', '#a99c8a', '#c4b8a4', '#9d9284', '#b3a08a'];
function houseGeometry(r, def) {
  var W = def[2], D = def[3], H = def[4], T = 0.28, parts = [], col = STONES[Math.floor(r() * STONES.length)];
  var gable = 2.0 + r() * 0.6, timber = '#3e3128';

  function wall(len, top, holes) {
    var s = new THREE.Shape();
    s.moveTo(-len / 2, 0);
    s.lineTo(len / 2, 0);
    for (var i = 0; i < top.length; i++) s.lineTo(top[i][0], top[i][1]);
    s.lineTo(-len / 2, 0);
    holes.forEach(function (o) {
      var hp = new THREE.Path();
      hp.moveTo(o[0], o[1]); hp.lineTo(o[0], o[1] + o[3]); hp.lineTo(o[0] + o[2], o[1] + o[3]); hp.lineTo(o[0] + o[2], o[1]); hp.lineTo(o[0], o[1]);
      s.holes.push(hp);
    });
    return tinted(new THREE.ExtrudeGeometry(s, { depth: T, bevelEnabled: false, curveSegments: 1 }), col);
  }
  // The top edge from +len/2 back to -len/2; `peak` raises a gable and
  // `gap` [x, half-width, y] is a stretch that has fallen in.
  function brokenTop(len, peak, gap) {
    var pts = [], n = Math.max(5, Math.round(len / 0.7));
    for (var i = 0; i <= n; i++) {
      var x = len / 2 - i * len / n, y = H + peak * (1 - Math.abs(x) / (len / 2)) - r() * 0.45;
      if (gap && Math.abs(x - gap[0]) < gap[1]) y = Math.min(y, gap[2] + r() * 0.5);
      pts.push([x, y]);
    }
    return pts;
  }
  function openings(len, door, gap) {
    var out = [], n = Math.max(1, Math.floor(len / 2.6)), step = len / n;
    for (var i = 0; i < n; i++) {
      var cx = -len / 2 + step * (i + 0.5), isDoor = door && i === Math.floor(n / 2);
      out.push(isDoor ? [cx - 0.5, 0.25, 1.0, 2.1] : [cx - 0.45, 1.15, 0.9, 1.1]);
      var up = [cx - 0.45, H * 0.58 + 0.2, 0.9, 1.0];
      if (up[1] + up[3] < H - 0.6 && !(gap && Math.abs(cx - gap[0]) < gap[1] + 0.6)) out.push(up);
    }
    return out;
  }
  function gapFor(len) { return r() < 0.6 ? [(r() - 0.5) * len * 0.6, 0.8 + r() * 1.2, H * 0.62] : null; }

  var gF = gapFor(W), gB = gapFor(W);
  parts.push(wall(W, brokenTop(W, 0, gF), openings(W, true, gF)).translate(0, 0, D / 2 - T));
  parts.push(wall(W, brokenTop(W, 0, gB), openings(W, false, gB)).translate(0, 0, -D / 2));
  [-1, 1].forEach(function (side) {
    var len = D - 2 * T, gS = r() < 0.5 ? [(r() - 0.5) * len * 0.4, 1 + r(), H + 0.3] : null;
    parts.push(wall(len, brokenTop(len, gable, gS), openings(len, false, null)).rotateY(Math.PI / 2)
      .translate(side < 0 ? -W / 2 : W / 2 - T, 0, 0));
  });
  if (r() < 0.6) parts.push(tinted(new THREE.BoxGeometry(0.9, H + gable + 0.6, 0.75).translate(W / 2 + 0.3, (H + gable + 0.6) / 2, -D * 0.2), col));

  var slope = Math.hypot(D / 2, gable), ang = Math.atan2(gable, D / 2);
  if (def[6]) {
    // A slab of the back slope still hanging on, and a bare ridge beam.
    parts.push(tinted(new THREE.BoxGeometry(W * 0.55, 0.14, slope + 0.3).rotateX(-ang - 0.06).translate(-W * 0.2, H + gable / 2 - 0.1, -D / 4), '#4c4a4c'));
    parts.push(tinted(new THREE.BoxGeometry(W, 0.16, 0.16).translate(0, H + gable - 0.15, 0), timber));
  }
  for (var k = 0; k < 4; k++) {
    if (r() < 0.45) continue;
    parts.push(tinted(new THREE.BoxGeometry(0.12, 0.14, slope).rotateX(ang).translate(-W / 2 + 0.6 + k * (W - 1.2) / 3, H + gable / 2 - 0.1, D / 4), timber));
  }

  var turn = (Math.abs(def[0]) < 5 ? 0 : def[0] > 0 ? -Math.PI / 2 : Math.PI / 2) + def[5];
  var m = new THREE.Matrix4().makeRotationY(turn).setPosition(def[0], height(def[0], def[1]) - 0.25, def[1]);
  parts.forEach(function (g) { g.applyMatrix4(m); });
  return parts;
}

// A tank, long rusted: hull, tracks with road wheels, turret and a gun
// drooping towards the grass. Its gun points along +x.
function tankGeometry() {
  var parts = [], hull = new THREE.Shape();
  [[-3.3, 0.55], [-2.9, 0.15], [2.8, 0.15], [3.4, 0.7], [3.0, 1.35], [-3.1, 1.35]].forEach(function (p, i) {
    if (i) hull.lineTo(p[0], p[1]); else hull.moveTo(p[0], p[1]);
  });
  parts.push(tinted(new THREE.ExtrudeGeometry(hull, { depth: 2.5, bevelEnabled: false }).translate(0, 0, -1.25), '#8a5232'));
  [-1, 1].forEach(function (side) {
    var t = new THREE.Shape();
    t.moveTo(-2.9, 0); t.lineTo(2.9, 0);
    t.absarc(2.9, 0.45, 0.45, -Math.PI / 2, Math.PI / 2, false);
    t.lineTo(-2.9, 0.9);
    t.absarc(-2.9, 0.45, 0.45, Math.PI / 2, Math.PI * 1.5, false);
    parts.push(tinted(new THREE.ExtrudeGeometry(t, { depth: 0.55, bevelEnabled: false, curveSegments: 6 }).translate(0, 0, side > 0 ? 1.25 : -1.8), '#4a3324'));
    for (var i = 0; i < 6; i++) {
      parts.push(tinted(new THREE.CylinderGeometry(0.32, 0.32, 0.12, 10).rotateX(Math.PI / 2).translate(-2.4 + i * 0.96, 0.42, side * 1.86), '#3e2a1e'));
    }
  });
  parts.push(tinted(new THREE.CylinderGeometry(1.15, 1.35, 0.75, 12).scale(1.1, 1, 0.95).translate(-0.4, 1.72, 0), '#6e4228'));
  parts.push(tinted(new THREE.CylinderGeometry(0.35, 0.35, 0.14, 10).translate(-0.7, 2.15, 0.35), '#5a3620'));
  parts.push(tinted(new THREE.BoxGeometry(0.6, 0.45, 0.5).translate(0.8, 1.72, 0), '#6e4228'));
  parts.push(tinted(new THREE.CylinderGeometry(0.085, 0.11, 4.0, 8).rotateZ(-Math.PI / 2).translate(2.0, 0, 0).rotateZ(-0.2).translate(0.75, 1.75, 0), '#6a3e24'));
  parts.push(tinted(new THREE.CylinderGeometry(0.16, 0.16, 0.35, 8).rotateZ(-Math.PI / 2).translate(4.05, 0, 0).rotateZ(-0.2).translate(0.75, 1.75, 0), '#5a341e'));
  return merge(parts);
}

// A robin about 18 cm long facing +x, feet at the origin: olive-brown back,
// orange-red breast and face, pale belly.
function robinGeometry() {
  var back = '#6f5c48', fire = '#e2622a';
  return merge([
    tinted(new THREE.CylinderGeometry(0.004, 0.004, 0.035, 3).translate(0.005, 0.017, 0.012), '#5a4032'),
    tinted(new THREE.CylinderGeometry(0.004, 0.004, 0.035, 3).translate(0.005, 0.017, -0.012), '#5a4032'),
    tinted(new THREE.SphereGeometry(0.045, 10, 8).scale(1.35, 1, 1).translate(0, 0.068, 0), back),
    tinted(new THREE.SphereGeometry(0.038, 10, 8).scale(1, 1.1, 1.08).translate(0.03, 0.064, 0), fire),
    tinted(new THREE.SphereGeometry(0.03, 8, 6).translate(0.004, 0.046, 0), '#e6ddcc'),
    tinted(new THREE.SphereGeometry(0.029, 10, 8).translate(0.052, 0.112, 0), back),
    tinted(new THREE.SphereGeometry(0.022, 8, 6).translate(0.064, 0.105, 0), fire),
    tinted(new THREE.ConeGeometry(0.006, 0.02, 4).rotateZ(-Math.PI / 2).translate(0.088, 0.108, 0), '#2e241e'),
    tinted(new THREE.SphereGeometry(0.005, 5, 4).translate(0.071, 0.116, 0.019), '#050505'),
    tinted(new THREE.SphereGeometry(0.005, 5, 4).translate(0.071, 0.116, -0.019), '#050505'),
    tinted(new THREE.SphereGeometry(0.035, 8, 6).scale(1.4, 0.6, 0.35).translate(-0.012, 0.074, 0.036), '#5c4a3a'),
    tinted(new THREE.SphereGeometry(0.035, 8, 6).scale(1.4, 0.6, 0.35).translate(-0.012, 0.074, -0.036), '#5c4a3a'),
    tinted(new THREE.BoxGeometry(0.055, 0.007, 0.03).rotateZ(0.4).translate(-0.078, 0.082, 0), '#54443a')
  ]);
}

// A swallow seen from below: pointed swept wings and a forked tail, nose to -z.
function swallowGeometry() {
  var v = [];
  function tri(a, b, c) { [a, b, c].forEach(function (p) { v.push(p[0], p[1], p[2]); }); }
  [1, -1].forEach(function (s) {
    tri([0, 0.012, -0.1], [s * 0.022, 0, -0.01], [0, 0.01, 0.06]);
    tri([s * 0.015, 0, -0.035], [s * 0.1, 0.004, -0.025], [s * 0.015, 0, 0.025]);
    tri([s * 0.1, 0.004, -0.025], [s * 0.19, 0.008, 0.065], [s * 0.015, 0, 0.025]);
    tri([s * 0.006, 0, 0.05], [s * 0.05, 0, 0.17], [0, 0, 0.075]);
  });
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(v, 3));
  geo.computeVertexNormals();
  return geo;
}

function tuftGeometry() {
  var parts = [];
  for (var k = 0; k < 3; k++) {
    var a = k * 2.1;
    parts.push(tinted(new THREE.ConeGeometry(0.028, 0.6, 3, 1, true).translate(0, 0.3, 0).rotateZ((k - 1) * 0.28).rotateY(a)
      .translate(Math.cos(a) * 0.05, 0, Math.sin(a) * 0.05), '#ffffff'));
  }
  return merge(parts);
}

function flowerGeometry() {
  return merge([
    tinted(new THREE.CylinderGeometry(0.004, 0.006, 0.34, 3).translate(0, 0.17, 0), '#86a070'),
    tinted(new THREE.CircleGeometry(0.05, 5).rotateX(-Math.PI / 2 + 0.5).translate(0, 0.34, 0), '#ffffff'),
    tinted(new THREE.CircleGeometry(0.016, 5).rotateX(-Math.PI / 2 + 0.5).translate(0, 0.343, 0.002), '#f4d060')
  ]);
}

// ── Renderer ─────────────────────────────────────────────────────────────
function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(73);
  var gl = makeRenderer(canvas, { shadows: !small, clear: '#aeb6b2' });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#aeb6b2', 0.011);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.1, 3000);

  var U = { time: { value: 0 }, wind: { value: 0.3 }, green: { value: 0.3 }, rings: { value: 0.5 }, sky: { value: new THREE.Color() } };

  var sky = new THREE.Group();
  world.add(sky);
  var dome = cloudDome(1500);
  sky.add(dome.mesh);
  var stars = starField(r, small ? 1200 : 2400, 1300, 0.06, 1.4);
  sky.add(stars);
  var moonDir = LIGHT[1].dir;
  var moon = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(246,248,255,1)', 'rgba(190,205,250,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
  moon.position.copy(moonDir).multiplyScalar(1000);
  moon.scale.setScalar(64);
  var moonHalo = new THREE.Sprite(moon.material.clone());
  moonHalo.position.copy(moon.position);
  moonHalo.scale.setScalar(320);
  sky.add(moon, moonHalo);

  var hemi = new THREE.HemisphereLight('#dfe6e8', '#4c5c38', 1.9);
  var sun = new THREE.DirectionalLight('#fff2dc', 0.7);
  sun.castShadow = !small;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = sun.shadow.camera.bottom = -34;
  sun.shadow.camera.right = sun.shadow.camera.top = 34;
  sun.shadow.camera.far = 260;
  sun.shadow.bias = -0.0006;
  sun.shadow.normalBias = 0.04;
  world.add(hemi, sun, sun.target);

  // Ground: spring grass, mud at the pool banks, darker woods on the hills.
  var tmp = new THREE.Color(), gA = new THREE.Color('#4d7a30'), gB = new THREE.Color('#6f9a3e'), mud = new THREE.Color('#4a4430'),
      hill = new THREE.Color('#3a5e2c');
  world.add(terrain(560, small ? 170 : 260, 0, -40, height, new THREE.MeshLambertMaterial({ vertexColors: true }), function (x, z, y) {
    var n = 0.5 + 0.5 * Math.sin(x * 0.31 + Math.sin(z * 0.23) * 2) * Math.cos(z * 0.27 - x * 0.11);
    tmp.copy(gA).lerp(gB, n);
    if (y < -0.3) tmp.lerp(mud, smooth(-0.3, -0.75, y));
    return tmp.lerp(hill, smooth(3, 26, y));
  }));

  var road = new THREE.Mesh(roadGeometry(curve.getSpacedPoints(360), 5.2, 0.05),
    new THREE.MeshPhongMaterial({ map: roadTexture(r), alphaTest: 0.5, shininess: 40, specular: '#2a2c2e' }));
  road.receiveShadow = true;
  world.add(road);

  // Puddles on the road and the pools beside it, ringing with the rain.
  var water = waterMaterial('#1a242a', U), pt = new THREE.Vector3(), tan = new THREE.Vector3();
  for (var i = 0; i < 16; i++) {
    var tt = 0.02 + i * 0.033 + r() * 0.015;
    curve.getPointAt(tt, pt);
    curve.getTangentAt(tt, tan);
    var off = (r() - 0.5) * 3, px = pt.x - tan.z * off, pz = pt.z + tan.x * off;
    var pud = new THREE.Mesh(new THREE.CircleGeometry(1, 20).rotateX(-Math.PI / 2), water);
    pud.scale.set(0.7 + r() * 1.3, 1, 0.5 + r() * 0.8);
    pud.rotation.y = r() * 3;
    pud.position.set(px, height(px, pz) + 0.075, pz);
    world.add(pud);
  }
  var poolMat = waterMaterial('#1a2830', U);
  POOLS.forEach(function (p) {
    var m = new THREE.Mesh(new THREE.CircleGeometry(p[2] + 1.1, 40).rotateX(-Math.PI / 2), poolMat);
    m.position.set(p[0], -0.42, p[1]);
    world.add(m);
  });
  // Lily pads, and reeds round the banks.
  var pads = new THREE.InstancedMesh(new THREE.CircleGeometry(0.22, 9, 0.3, 5.9).rotateX(-Math.PI / 2),
    new THREE.MeshLambertMaterial({ color: '#4f7a34' }), 60), up = new THREE.Vector3(0, 1, 0);
  scatter(pads, 400, function (k, p, q, s, c) {
    var pl = POOLS[k % POOLS.length], a = r() * 6.28, d = pl[2] * (0.35 + r() * 0.6);
    p.set(pl[0] + Math.cos(a) * d, -0.405, pl[1] + Math.sin(a) * d);
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar(0.7 + r() * 0.8);
    c.setHSL(0.27 + r() * 0.05, 0.45, 0.3 + r() * 0.1);
  });
  world.add(pads);
  var reeds = new THREE.InstancedMesh(merge([tinted(new THREE.ConeGeometry(0.02, 1.4, 3, 1, true).translate(0, 0.7, 0), '#5a7038'),
    tinted(new THREE.CylinderGeometry(0.03, 0.03, 0.2, 5).translate(0, 1.15, 0), '#5a3c26')]),
    new THREE.MeshLambertMaterial({ vertexColors: true }), small ? 200 : 420);
  scatter(reeds, 3000, function (k, p, q, s, c) {
    var pl = POOLS[k % POOLS.length], a = r() * 6.28, d = pl[2] + 0.2 + r() * 1.3;
    var x = pl[0] + Math.cos(a) * d, z = pl[1] + Math.sin(a) * d;
    if (distRoad(x, z) < 3.4 || r() < 0.3) return false;
    p.set(x, height(x, z) - 0.05, z);
    q.setFromAxisAngle(up, r() * 6.28);
    s.set(1, 0.6 + r() * 0.7, 1);
  });
  world.add(reeds);

  // Grass, taller as the town goes back to green, swaying in the wind.
  var grassMat = new THREE.MeshLambertMaterial({ vertexColors: true });
  grassMat.onBeforeCompile = function (sh) {
    sh.uniforms.uTime = U.time; sh.uniforms.uWind = U.wind; sh.uniforms.uGreen = U.green;
    sh.vertexShader = 'uniform float uTime; uniform float uWind; uniform float uGreen;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n transformed.y *= 0.65 + 0.6 * uGreen;\n' +
      ' float gph = instanceMatrix[3][0] * 0.4 + instanceMatrix[3][2] * 0.3;\n' +
      ' transformed.x += sin(uTime * 1.3 + gph) * (0.04 + uWind * 0.12) * transformed.y * transformed.y;');
  };
  var grass = new THREE.InstancedMesh(tuftGeometry(), grassMat, small ? 9000 : 22000);
  grass.receiveShadow = true;
  var near = function (x, z) { return Math.hypot(x - TANK.x, z - TANK.z); };
  scatter(grass, 90000, function (k, p, q, s, c) {
    var z = 70 - r() * 185, x = (r() - 0.5) * 44, dr = distRoad(x, z);
    if (dr < 2.2 && r() < 0.93) return false;
    if (r() > (near(x, z) < 6 ? 0.8 : 0.55)) return false;
    var y = height(x, z);
    if (y < -0.5) return false;
    p.set(x, y - 0.02, z);
    q.setFromAxisAngle(up, r() * 6.28);
    var sc = (0.7 + r() * 0.7) * (near(x, z) < 5 ? 1.15 : 1);
    s.set(sc, sc * (0.8 + r() * 0.5), sc);
    c.setHSL(0.24 + r() * 0.06, 0.45, 0.25 + r() * 0.13);
  });
  world.add(grass);

  // Wildflowers in the verges, the field and on the tank.
  var tankM = new THREE.Matrix4().compose(new THREE.Vector3(TANK.x, height(TANK.x, TANK.z) - 0.4, TANK.z),
    new THREE.Quaternion().setFromEuler(new THREE.Euler(0.05, TANK.rot, 0.07)), new THREE.Vector3(1, 1, 1));
  var flowerCols = ['#ffffff', '#fff6e0', '#f4e070', '#c8a8e8', '#f2b0c8', '#ffffff'];
  var flowers = new THREE.InstancedMesh(flowerGeometry(), new THREE.MeshLambertMaterial({ vertexColors: true }), small ? 4000 : 9000);
  var deck = new THREE.Vector3();
  scatter(flowers, 60000, function (k, p, q, s, c) {
    if (k < 70) {
      // On the tank's deck and turret.
      deck.set(-2.8 + r() * 5.4, k < 50 ? 1.36 : 2.1, (r() - 0.5) * (k < 50 ? 2.3 : 1.6));
      if (k >= 50) deck.x = -0.4 + (r() - 0.5) * 1.8;
      p.copy(deck).applyMatrix4(tankM);
    } else {
      var z = 70 - r() * 185, x = (r() - 0.5) * 50, y = height(x, z);
      if (distRoad(x, z) < 2.8 || y < -0.45) return false;
      var drift = 0.5 + 0.5 * Math.sin(x * 0.21 + z * 0.13) * Math.sin(z * 0.09 - x * 0.05);
      if (r() > drift * (near(x, z) < 9 ? 1 : 0.6)) return false;
      p.set(x, y - 0.02, z);
    }
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar(0.8 + r() * 0.7);
    c.set(flowerCols[Math.floor(r() * flowerCols.length)]);
  });
  world.add(flowers);

  // The town: shells of houses that ivy climbs as `green` rises.
  var shells = [];
  HOUSES.forEach(function (def) { shells = shells.concat(houseGeometry(r, def)); });
  var houses = new THREE.Mesh(merge(shells), overgrown(U.green, 6.2, false));
  houses.castShadow = houses.receiveShadow = true;
  world.add(houses);

  var tank = new THREE.Mesh(tankGeometry(), overgrown(U.green, 1.5, true, 1.0));
  tank.matrixAutoUpdate = false;
  tank.matrix.copy(tankM);
  tank.castShadow = tank.receiveShadow = true;
  world.add(tank);

  // Fence: weathered posts and two strands of wire along the right verge.
  var fenceParts = [], wireY = function (z, base) {
    var f = ((FENCE.z0 - z) / 2.6) % 1;
    return base - 0.05 * Math.sin(Math.PI * f);
  };
  for (var fz = FENCE.z0; fz >= FENCE.z1; fz -= 2.6) {
    fenceParts.push(tinted(new THREE.BoxGeometry(0.12, 1.35, 0.12).rotateZ((r() - 0.5) * 0.12).translate(FENCE.x, height(FENCE.x, fz) + 0.55, fz), '#7a6e60'));
  }
  [1.05, 0.66].forEach(function (base) {
    for (var wz = FENCE.z0; wz > FENCE.z1; wz -= 0.4) {
      var a = new THREE.Vector3(FENCE.x, wireY(wz, base), wz), b = new THREE.Vector3(FENCE.x, wireY(wz - 0.4, base), wz - 0.4);
      fenceParts.push(tinted(new THREE.CylinderGeometry(0.007, 0.007, a.distanceTo(b), 3)
        .applyMatrix4(new THREE.Matrix4().lookAt(a, b, up).multiply(new THREE.Matrix4().makeRotationX(Math.PI / 2)))
        .translate((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2), '#4c4038'));
    }
  });
  var fence = new THREE.Mesh(merge(fenceParts), new THREE.MeshLambertMaterial({ vertexColors: true }));
  fence.castShadow = true;
  world.add(fence);

  // Robins on the top wire, and one on the tank's gun.
  var robins = new THREE.InstancedMesh(robinGeometry(), new THREE.MeshLambertMaterial({ vertexColors: true }), ROBINS.length + 1);
  robins.castShadow = true;
  robins.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  var perch = ROBINS.map(function (rb) { return new THREE.Vector3(FENCE.x, wireY(rb[0], 1.05) + 0.004, rb[0]); });
  perch.push(new THREE.Vector3(3.6, 1.29, 0).applyMatrix4(tankM));
  var rbBase = ROBINS.map(function (rb) { return rb[1]; }).concat([TANK.rot + 0.4]);
  var rbYaw = rbBase.slice(), rbAim = rbBase.slice(), rbNext = rbBase.map(function () { return 0; }), rbHop = rbBase.map(function () { return 0; });
  world.add(robins);

  // Trees: fresh spring greens round the town, a few through the houses;
  // wild plums in white by the pools.
  var treeMat = new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true });
  var greens = ['#7fae4a', '#93c052', '#6c9a40', '#a8c862', '#5f8e3c'];
  var housesNear = function (x, z) {
    for (var k = 0; k < HOUSES.length; k++) if (Math.hypot(x - HOUSES[k][0], z - HOUSES[k][1]) < 7) return true;
    return false;
  };
  [broadleafGeometry(rng(7), '#4a3a2e'), broadleafGeometry(rng(11), '#4a3a2e')].forEach(function (geo, gi) {
    var trees = new THREE.InstancedMesh(geo, treeMat, small ? 380 : 800), inside = HOUSES.filter(function (h) { return h[7]; });
    trees.castShadow = trees.receiveShadow = true;
    scatter(trees, 20000, function (k, p, q, s, c) {
      var x, z, sc;
      if (gi === 0 && k < inside.length) {
        x = inside[k][0] + (r() - 0.5); z = inside[k][1] + (r() - 0.5); sc = 1.35 + r() * 0.3;
      } else {
        x = (r() - 0.5) * 300; z = 60 - r() * 260;
        var dr = distRoad(x, z);
        if (dr < 8 || housesNear(x, z) || near(x, z) < 8) return false;
        if (POOLS.some(function (pl) { return Math.hypot(x - pl[0], z - pl[1]) < pl[2] + 3; })) return false;
        if (x > 1.5 && x < 6 && z < FENCE.z0 + 2 && z > FENCE.z1 - 2) return false;
        if (r() > 0.18 + smooth(10, 60, dr) * 0.7) return false;
        sc = 0.9 + r() * 0.9;
      }
      p.set(x, height(x, z) - 0.2, z);
      q.setFromAxisAngle(up, r() * 6.28);
      s.set(sc, sc * (0.9 + r() * 0.3), sc);
      c.set(greens[Math.floor(r() * greens.length)]);
    });
    world.add(trees);
  });

  var plumMat = new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true, emissive: '#2a2224' });
  var plums = new THREE.InstancedMesh(broadleafGeometry(rng(19), '#3a2a26'), plumMat, PLUMS.length), blossom = [];
  plums.castShadow = true;
  scatter(plums, PLUMS.length, function (k, p, q, s, c) {
    var pl = PLUMS[k], sc = 0.75 + r() * 0.3;
    p.set(pl[0], height(pl[0], pl[1]) - 0.2, pl[1]);
    q.setFromAxisAngle(up, r() * 6.28);
    s.set(sc, sc * 0.85, sc);
    c.set('#f4e2e8');
    for (var b = 0; b < (small ? 260 : 600); b++) {
      var a = r() * 6.28, rad = (0.6 + 0.4 * Math.sqrt(r())) * 2.2 * sc, hgt = 3.9 + (r() - 0.3) * 2.6;
      blossom.push(pl[0] + Math.cos(a) * rad, p.y + hgt * sc * 0.85, pl[1] + Math.sin(a) * rad);
    }
  });
  world.add(plums);
  var bloomGeo = new THREE.BufferGeometry();
  bloomGeo.setAttribute('position', new THREE.Float32BufferAttribute(blossom, 3));
  var bloomMat = new THREE.PointsMaterial({ color: '#ffffff', size: 0.1, map: softSprite('rgba(255,255,255,1)', 'rgba(255,240,246,0)'),
    transparent: true, depthWrite: false, alphaTest: 0.05 });
  bloomMat.onBeforeCompile = function (sh) {
    sh.uniforms.uTime = U.time;
    sh.vertexShader = 'uniform float uTime;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n float bph = position.x * 7.1 + position.z * 3.7 + position.y * 5.3;\n' +
      ' transformed += vec3(sin(uTime * 7.0 + bph), sin(uTime * 6.1 + bph * 1.3), cos(uTime * 6.6 + bph)) * 0.03;');
  };
  world.add(new THREE.Points(bloomGeo, bloomMat));

  // Swallows wheeling in loose flocks over the grass and the streets.
  var FLOCKS = [[-5, 4, 50], [7, 5, 32], [-6, 3.5, 12], [5, 6, -6], [-5, 4.5, -26], [7, 5, -44], [0, 7, -60], [-3, 11, -52]];
  var swallowMat = new THREE.MeshLambertMaterial({ color: '#262c3a', side: THREE.DoubleSide });
  swallowMat.onBeforeCompile = function (sh) {
    sh.uniforms.uTime = U.time;
    sh.vertexShader = 'uniform float uTime;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n float sph = float(gl_InstanceID) * 1.37;\n' +
      ' float beat = sin(uTime * 15.0 + sph) * (0.35 + 0.65 * smoothstep(-0.3, 0.4, sin(uTime * 0.9 + sph * 0.7)));\n' +
      ' transformed.y += beat * max(abs(position.x) - 0.02, 0.0) * 0.9;');
  };
  var NS = small ? 20 : 34, swallows = new THREE.InstancedMesh(swallowGeometry(), swallowMat, NS), birds = [];
  swallows.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  swallows.frustumCulled = false;
  for (i = 0; i < NS; i++) {
    birds.push({ f: FLOCKS[i % FLOCKS.length], R: 3 + r() * 6, w: (0.55 + r() * 0.5) * (r() < 0.5 ? -1 : 1), ph: r() * 6.28,
                 v: 0.8 + r() * 1.4, sq: 0.5 + r() * 0.4 });
  }
  world.add(swallows);

  // Low mist that gathers for the dawn.
  var mistTex = softSprite('rgba(255,255,255,0.8)', 'rgba(255,255,255,0)'), mist = [];
  for (i = 0; i < 30; i++) {
    var ms = new THREE.Sprite(new THREE.SpriteMaterial({ map: mistTex, transparent: true, depthWrite: false, opacity: 0 }));
    var mx = (r() - 0.5) * 110, mz = 40 - r() * 170;
    ms.position.set(mx, height(mx, mz) + 0.8 + r() * 2.5, mz);
    ms.scale.set(22 + r() * 22, 5 + r() * 5, 1);
    world.add(ms);
    mist.push(ms);
  }

  var rain = rainField({ count: small ? 1500 : 3600, box: [18, 14, 26], speed: 9, windSpeed: 3, color: '#d2dae2', opacity: 0.24 });
  world.add(rain.lines);
  var flies = particleField({ count: small ? 120 : 260, box: [26, 4, 26], fall: [-0.1, 0.1], size: 0.09, color: '#e6ff9a',
                              map: softSprite('rgba(240,255,170,1)', 'rgba(220,255,120,0)'), sway: 0.6, windSpeed: 0.3 });
  flies.points.material.blending = THREE.AdditiveBlending;
  world.add(flies.points);
  var petalTex = canvasTexture(32, 32, function (x) { x.fillStyle = '#fff'; x.beginPath(); x.ellipse(16, 16, 12, 7, 0.6, 0, Math.PI * 2); x.fill(); });
  var petals = particleField({ count: small ? 160 : 340, box: [22, 9, 22], fall: [0.25, 0.55], size: 0.06, map: petalTex,
                               colors: ['#ffffff', '#fbeef2', '#f6e2ea'], sway: 0.8, windSpeed: 1.2, alphaTest: 0.4 });
  world.add(petals.points);

  // ── Per frame ─────────────────────────────────────────────────────────
  var L = { dir: new THREE.Vector3() };
  LC.forEach(function (k) { L[k] = new THREE.Color(); });
  function mixLight(v) {
    var k = clamp(Math.floor(v), 0, LIGHT.length - 2), t = clamp(v - k, 0, 1), a = LIGHT[k], b = LIGHT[k + 1], n;
    for (n = 0; n < LC.length; n++) L[LC[n]].copy(a[LC[n]]).lerp(b[LC[n]], t);
    for (n = 0; n < LN.length; n++) L[LN[n]] = lerp(a[LN[n]], b[LN[n]], t);
    L.dir.copy(a.dir).lerp(b.dir, t).normalize();
  }
  var m4 = new THREE.Matrix4(), q4 = new THREE.Quaternion(), s3 = new THREE.Vector3(), p3 = new THREE.Vector3(), t3 = new THREE.Vector3(),
      b3 = new THREE.Vector3(), low = new THREE.Vector3(), white = new THREE.Color('#ffffff'), moonlit = new THREE.Color('#7a86aa');
  var fx = { snow: 0, wind: 0, dt: 0, time: 0 }, portrait = false;

  function birdAt(b, a, out) {
    out.set(b.f[0] + Math.cos(a) * b.R, b.f[1] + Math.sin(a * 2) * b.v * 0.5 + Math.sin(a * 0.5) * 0.6, b.f[2] + Math.sin(a) * b.R * b.sq);
    return out;
  }

  function frame(f) {
    var row = f.row, time = f.time, dt = f.dt, light = row[1], rainAmt = row[2], lift = row[6], green = row[7],
        mistAmt = row[8], petalAmt = row[9];
    var night = clamp(1 - Math.abs(light - 1), 0, 1), day = 1 - night;

    // On a portrait screen the verse sits mid-frame, so look up a little at
    // the robins and the tank to bring them in below it.
    var tilt = portrait ? 0.21 * smooth(1.4, 2, light) * (1 - smooth(3.4, 4, light)) : 0;
    followPath(camera, curve, onRoad, clamp(f.cam, 0, 1), { eye: 1.65, lift: lift, yaw: row[4], pitch: row[5] + tilt,
                                                           mx: f.mx, my: f.my, time: time });
    sky.position.copy(camera.position);

    // Light for the time of day.
    mixLight(light);
    var du = dome.uniforms;
    du.uTop.value.copy(L.top); du.uHorizon.value.copy(L.hor); du.uCloud.value.copy(L.cloud); du.uShade.value.copy(L.shade);
    du.uSunDir.value.copy(L.dir);
    du.uSun.value.copy(L.sun).multiplyScalar(L.glow);
    du.uCover.value = L.cover;
    du.uTime.value = time;
    world.fog.color.copy(L.fog);
    world.fog.density = L.fogD + mistAmt * 0.006;
    gl.setClearColor(L.fog);
    hemi.color.copy(L.sky); hemi.groundColor.copy(L.ground); hemi.intensity = L.hemiI;
    sun.color.copy(L.sun); sun.intensity = L.sunI;
    sun.position.copy(camera.position).addScaledVector(L.dir, 120);
    sun.target.position.copy(camera.position);
    gl.toneMappingExposure = L.exp;
    U.sky.value.copy(L.hor).multiplyScalar(0.3);
    stars.material.opacity = 0.85 * night * (1 - L.cover * 0.6);
    moon.material.opacity = night;
    moonHalo.material.opacity = night * 0.25;

    U.time.value = time;
    U.wind.value = f.wind;
    U.green.value = green;
    U.rings.value = Math.max(rainAmt, 0.12 + night * 0.25);
    bloomMat.color.copy(white).lerp(moonlit, night).multiplyScalar(0.75 + 0.25 * day);
    plumMat.emissive.setRGB(0.16, 0.13, 0.14).multiplyScalar(0.3 + 0.7 * day);

    // Swallows by day.
    for (var k = 0; k < birds.length; k++) {
      var b = birds[k], a = time * b.w * (env.reduceMotion ? 0.4 : 1) + b.ph;
      birdAt(b, a, p3);
      birdAt(b, a + 0.06 * Math.sign(b.w), t3);
      b3.set(b.f[0] - p3.x, 0, b.f[2] - p3.z).normalize().multiplyScalar(0.5);
      b3.y = 1;
      m4.lookAt(p3, t3, b3);
      s3.setScalar(1.5 * day + 0.0001);
      m4.scale(s3);
      m4.setPosition(p3);
      swallows.setMatrixAt(k, m4);
    }
    swallows.instanceMatrix.needsUpdate = true;

    // Robins: quick turns of the body and a little hop now and then.
    for (k = 0; k < perch.length; k++) {
      if (time > rbNext[k]) {
        rbAim[k] = rbBase[k] + (Math.random() - 0.5) * 1.3;
        rbNext[k] = time + 0.8 + Math.random() * 2.6;
        if (Math.random() < 0.35) rbHop[k] = 1;
      }
      rbYaw[k] += (rbAim[k] - rbYaw[k]) * (1 - Math.exp(-dt * 16));
      rbHop[k] = Math.max(0, rbHop[k] - dt * 4);
      q4.setFromAxisAngle(up, rbYaw[k]);
      p3.copy(perch[k]);
      p3.y += Math.sin(rbHop[k] * Math.PI) * 0.05;
      s3.set(1.8, 1.8 * (1 + 0.025 * Math.sin(time * 5 + k)), 1.8);
      robins.setMatrixAt(k, m4.compose(p3, q4, s3));
    }
    robins.instanceMatrix.needsUpdate = true;

    for (k = 0; k < mist.length; k++) {
      mist[k].material.opacity = mistAmt * 0.42;
      mist[k].material.color.copy(L.fog).lerp(white, 0.25);
    }

    rain.update(f, camera.position, rainAmt, env.reduceMotion);
    fx.wind = f.wind; fx.dt = dt; fx.time = time;
    fx.snow = night;
    low.set(camera.position.x, onRoad(camera.position.x, camera.position.z) - 0.4, camera.position.z);
    flies.update(fx, low, env.reduceMotion);
    fx.snow = petalAmt;
    petals.update(fx, camera.position, env.reduceMotion);
    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) { portrait = w < h; fitCamera(gl, camera, w, h, dpr, small); },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('soft-rains', {
  renderer: renderer3d,
  align: ['left', 'right', 'left', 'right', 'left', 'center'],
  scrim: 0.62,
  keys: function (T) {
    function at(i, frac) { return lerp(T.start(i), T.end(i), frac); }
    //  unit          path       light rain  wind  yaw    pitch  lift green mist petals
    return [
      [0,             tAt(64),   0.0, 0.70, 0.25, 0.00, 0.03, 0, 0.30, 0.25, 0.0],
      [0.7,           tAt(62),   0.0, 0.70, 0.25, 0.00, 0.04, 0, 0.30, 0.25, 0.0],
      [at(0, 0.55),   tAt(52),   0.0, 0.85, 0.30, 0.06, 0.12, 0, 0.32, 0.25, 0.0],   // soft rains; swallows calling
      [at(0, 1.0),    tAt(44),   0.3, 0.50, 0.25, 0.30, 0.03, 0, 0.34, 0.20, 0.1],
      [at(1, 0.35),   tAt(35),   1.0, 0.10, 0.10, 0.78, -0.10, 0, 0.36, 0.12, 0.5],  // frogs in the pools at night; plum blossom
      [at(1, 0.9),    tAt(32),   1.0, 0.05, 0.10, 0.88, -0.05, 0, 0.38, 0.12, 0.7],
      [at(2, 0.3),    tAt(7.4),  2.0, 0.00, 0.10, -0.55, -0.13, 0, 0.42, 0.08, 0.15], // robins on a low fence-wire
      [at(2, 0.9),    tAt(7.0),  2.0, 0.00, 0.10, -0.6, -0.12, 0, 0.44, 0.08, 0.1],
      [at(3, 0.3),    tAt(-7),   3.0, 0.45, 0.25, 0.52, -0.07, 0, 0.52, 0.12, 0.0],  // not one will know of the war
      [at(3, 0.9),    tAt(-9),   3.0, 0.45, 0.25, 0.46, -0.05, 0, 0.56, 0.12, 0.0],
      [at(4, 0.35),   tAt(-17),  4.0, 0.12, 0.20, -0.36, 0.03, 0, 0.66, 0.18, 0.0],  // neither bird nor tree
      [at(4, 0.95),   tAt(-24),  4.4, 0.04, 0.15, -0.15, 0.00, 1.5, 0.76, 0.40, 0.1],
      [at(5, 0.4),    tAt(-30),  5.0, 0.00, 0.15, 0.00, -0.17, 10, 0.92, 0.85, 0.4],  // Spring woke at dawn
      [at(5, 1.0),    tAt(-34),  5.0, 0.00, 0.15, 0.00, -0.2, 13, 1.00, 0.80, 0.5],
      [T.total,       tAt(-38),  5.0, 0.00, 0.15, 0.00, -0.16, 15, 1.00, 0.70, 0.5]
    ];
  },
  sound: {
    src: '/audio/rain.mp3',
    label: 'Play the soft rain, the frogs and the robins',
    volume: function (row) { return 0.05 + 0.55 * row[2]; },
    cues: [
      { stanza: 0, at: 0.55, play: twitter },
      { stanza: 1, at: 0.4, play: frogs },
      { stanza: 2, at: 0.35, play: robinSong },
      { stanza: 5, at: 0.5, play: robinSong }
    ]
  }
});
