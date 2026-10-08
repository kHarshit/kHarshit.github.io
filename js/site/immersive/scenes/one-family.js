/*
 * Scene for "Vasudhaiva Kutumbakam" (Maha Upanishad 6.71–75): the night
 * side of the Earth, seen from low orbit.
 *
 * Title  Night over a continent, its cities glowing amber, lightning now
 *        and then in the clouds out at sea, the thin green line of airglow
 *        over the curve of the world.
 * I      अयं निजः परो वेति ...: faint lines of light draw themselves over the
 *        land, the borders, and the city lights sit in separate clusters
 *        with dark between them.
 * II     "The world is a family": a single arc of light leaves a city on the
 *        near coast and arches over the horizon to a city on the far shore.
 * III    "One is a relative, the other stranger, say the small minded": the
 *        borders flare hard and cold. "The entire world is a family": they
 *        break up and dissolve, the dark gaps between the clusters fill in,
 *        and arcs leap between cities across the oceans into one web.
 * IV     "Be detached, be magnanimous, lift up your mind": the view lifts
 *        away; the sun rises over the limb and dawn sweeps across the
 *        planet, lighting it as one, the web glowing gold across it.
 *
 * The globe has radius 1. Continents come from 3D value noise baked into an
 * equirectangular texture (poles along x, so the orbit runs along the
 * equator), with finer coastline noise and clouds added in the shader.
 * Nations are Voronoi cells of seeds on land; the shader draws their
 * borders, and the same cells decide which city lights wait in the border
 * gaps until the borders dissolve. Columns:
 *   [unit, orbit, -, -, wind, altitude, look down, yaw, borders, flare, dissolve, web, sun, first arc]
 */
import { THREE, isSmall, makeRenderer, softSprite, starField, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

var THR = 0.55;                                 // land where elev() > THR
var SUN_X = -0.2;                               // the sun rises a little left of ahead

// ── Continents ───────────────────────────────────────────────────────────
function noise3(seed) {
  function h(i, j, k) {
    var n = Math.imul(i, 374761393) ^ Math.imul(j, 668265263) ^ Math.imul(k, 1274126177) ^ seed;
    n = Math.imul(n ^ (n >>> 13), 1274126177);
    return ((n ^ (n >>> 16)) >>> 0) / 4294967295;
  }
  function f(t) { return t * t * (3 - 2 * t); }
  return function (x, y, z) {
    var i = Math.floor(x), j = Math.floor(y), k = Math.floor(z), u = f(x - i), v = f(y - j), w = f(z - k);
    var a = lerp(lerp(h(i, j, k), h(i + 1, j, k), u), lerp(h(i, j + 1, k), h(i + 1, j + 1, k), u), v);
    var b = lerp(lerp(h(i, j, k + 1), h(i + 1, j, k + 1), u), lerp(h(i, j + 1, k + 1), h(i + 1, j + 1, k + 1), u), v);
    return lerp(a, b, w);
  };
}
function fbm(n, x, y, z, oct) {
  var s = 0, a = 0.5, fr = 1;
  for (var o = 0; o < oct; o++) { s += a * n(x * fr, y * fr, z * fr); a *= 0.5; fr *= 2.03; }
  return s;
}
function gauss(v, s) { return Math.exp(-(v / s) * (v / s)); }

var nLand = noise3(7), nWarp = noise3(11), nPop = noise3(23);
// Height of the land on the unit sphere. The orbit runs from (0,1,0)
// towards -z (angle th along it): a home continent under you at the start,
// an ocean ahead, and a far shore beyond it.
function elev(x, y, z) {
  var w = 0.6 * (nWarp(x * 2.1 + 5, y * 2.1, z * 2.1) - 0.5);
  var e = fbm(nLand, x * 1.6 + w, y * 1.6 - w, z * 1.6 + 3, 6);
  var th = Math.atan2(-z, y);
  return e + 0.16 * gauss(th + 0.08, 0.3) * gauss(x, 0.55) - 0.22 * gauss(th - 0.34, 0.09) * gauss(x, 0.7) +
         0.3 * gauss(th - 0.62, 0.2) * gauss(x - 0.1, 0.45);
}

// Borders wander a little (as rivers and ridgelines do): the same wobble in
// JS and GLSL, so lights can keep clear of them.
function wobble(p, out) {
  out.set(p.x + 0.012 * Math.sin(p.y * 23 + p.z * 7) + 0.004 * Math.sin(p.z * 71 + p.x * 13),
          p.y + 0.012 * Math.sin(p.z * 19 + p.x * 11 + 1.3) + 0.004 * Math.sin(p.x * 67 + p.y * 17 + 0.7),
          p.z + 0.012 * Math.sin(p.x * 29 + p.y * 5 + 2.1) + 0.004 * Math.sin(p.y * 73 + p.z * 11 + 2.9));
  return out.normalize();
}
var WOBBLE_GLSL =
  'vec3 wob(vec3 p){ return normalize(p + 0.012 * vec3(sin(p.y * 23.0 + p.z * 7.0), sin(p.z * 19.0 + p.x * 11.0 + 1.3), sin(p.x * 29.0 + p.y * 5.0 + 2.1))' +
  ' + 0.004 * vec3(sin(p.z * 71.0 + p.x * 13.0), sin(p.x * 67.0 + p.y * 17.0 + 0.7), sin(p.y * 73.0 + p.z * 11.0 + 2.9))); }\n';

// Shared GLSL noise (shader-only detail: coastlines, clouds, breakup).
var NOISE_GLSL = [
  'float h3(vec3 p){ p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }',
  'float n3(vec3 x){ vec3 i = floor(x), f = fract(x); f = f * f * (3.0 - 2.0 * f);',
  '  return mix(mix(mix(h3(i), h3(i + vec3(1,0,0)), f.x), mix(h3(i + vec3(0,1,0)), h3(i + vec3(1,1,0)), f.x), f.y),',
  '             mix(mix(h3(i + vec3(0,0,1)), h3(i + vec3(1,0,1)), f.x), mix(h3(i + vec3(0,1,1)), h3(i + vec3(1,1,1)), f.x), f.y), f.z); }',
  'float fbm(vec3 p){ float s = 0.0, a = 0.5; for (int k = 0; k < 5; k++) { s += a * n3(p); p = p * 2.03 + 1.7; a *= 0.5; } return s; }',
  'float clouds(vec3 n){ vec3 p = n * vec3(8.0, 5.0, 5.0) + vec3(2.0, 0.0, 7.0); p += 0.9 * vec3(fbm(n * 2.5 + 4.0), fbm(n * 2.5 + 9.0), fbm(n * 2.5 + 1.0));\n' +
  '  float c = fbm(p) + (fbm(n * 23.0 + 3.0) - 0.5) * 0.35 + (n3(n * 90.0) - 0.5) * 0.1; return smoothstep(0.5, 0.74, c); }'
].join('\n') + '\n';

// ── Sounds: a tanpura drone and soft chimes, synthesised ─────────────────
// One plucked tanpura string: a bright saw, its buzz (jawari) a swept band
// of overtones, dying away slowly.
function pluck(ac, out, f, t, gain) {
  var o = ac.createOscillator(), lp = ac.createBiquadFilter(), bp = ac.createBiquadFilter(), g = ac.createGain(), gb = ac.createGain();
  o.type = 'sawtooth';
  o.frequency.value = f;
  lp.type = 'lowpass';
  lp.frequency.setValueAtTime(2400, t);
  lp.frequency.exponentialRampToValueAtTime(500, t + 6);
  bp.type = 'bandpass';
  bp.Q.value = 9;
  bp.frequency.setValueAtTime(f * 5, t);
  bp.frequency.exponentialRampToValueAtTime(f * 15, t + 4.5);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(gain, t + 0.02);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 7);
  gb.gain.value = 1.6;
  o.connect(lp); lp.connect(g);
  o.connect(bp); bp.connect(gb); gb.connect(g);
  g.connect(out);
  o.start(t);
  o.stop(t + 7.1);
}
function tanpura(ac, out) {
  var t = ac.currentTime, sa = 138.6;
  // Pa, Sa, Sa, low Sa, twice round.
  [[sa * 0.749, 0], [sa, 1.2], [sa, 2.0], [sa / 2, 2.9], [sa * 0.749, 4.6], [sa, 5.8], [sa, 6.6], [sa / 2, 7.5]].forEach(function (p) {
    pluck(ac, out, p[0], t + p[1], 0.022);
  });
}
// A soft glassy chime: one note or a few rising ones.
function chime(notes, gap, gain) {
  return function (ac, out) {
    var t0 = ac.currentTime;
    notes.forEach(function (f, k) {
      var t = t0 + k * gap;
      [[1, 1, 3.5], [2.01, 0.35, 2.2], [3.0, 0.12, 1.4]].forEach(function (p) {
        var o = ac.createOscillator(), g = ac.createGain();
        o.type = 'sine';
        o.frequency.value = f * p[0];
        g.gain.setValueAtTime(0.0001, t);
        g.gain.exponentialRampToValueAtTime(gain * p[1], t + 0.015);
        g.gain.exponentialRampToValueAtTime(0.0001, t + p[2]);
        o.connect(g); g.connect(out);
        o.start(t);
        o.stop(t + p[2] + 0.05);
      });
    });
  };
}

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(29);
  var gl = makeRenderer(canvas, { clear: '#000000' });
  var world = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(55, 1, 0.005, 600);
  var v = new THREE.Vector3(), w = new THREE.Vector3(), q = new THREE.Vector3(), i, k;

  // ── The continents, baked: R = height, G = the glow of the cities ──
  var TW = small ? 768 : 1280, TH = TW / 2, height = new Float32Array(TW * TH);
  var glow = new Float32Array(TW * TH);
  function toSphere(lat, lon, out) { var c = Math.cos(lat); return out.set(Math.sin(lat), c * Math.cos(lon), -c * Math.sin(lon)); }
  for (var ty = 0; ty < TH; ty++) {
    var lat = ((ty + 0.5) / TH - 0.5) * Math.PI;
    for (var tx = 0; tx < TW; tx++) {
      toSphere(lat, ((tx + 0.5) / TW - 0.5) * Math.PI * 2, v);
      height[ty * TW + tx] = elev(v.x, v.y, v.z);
    }
  }
  function isLand(p, m) { return elev(p.x, p.y, p.z) > THR + (m || 0) && Math.abs(p.x) < 0.88; }

  // ── Nations: Voronoi seeds on land ──
  // Smaller nations where you can see them, along the orbit.
  var seeds = [];
  for (k = 0; k < 9000 && seeds.length < 60; k++) {
    var near = k < 3000;
    if (near) { var th0 = -0.5 + r() * 1.4; v.set((r() - 0.5) * 1.0, Math.cos(th0), -Math.sin(th0)).normalize(); }
    else randomDir(v);
    if (!isLand(v, 0.02)) continue;
    var ok = true, sep = Math.cos(near ? 0.12 : 0.2);
    for (i = 0; i < seeds.length; i++) if (v.dot(seeds[i]) > sep) { ok = false; break; }
    if (ok) seeds.push(v.clone());
  }
  function randomDir(out) {
    var y = r() * 2 - 1, a = r() * Math.PI * 2, s = Math.sqrt(1 - y * y);
    return out.set(s * Math.cos(a), y, s * Math.sin(a));
  }
  var nationOf = 0;
  // Distance from p to the nearest border (and which nation p is in).
  function borderDist(p) {
    wobble(p, q);
    var best = -2, a = 0, d;
    for (var s = 0; s < seeds.length; s++) { d = q.dot(seeds[s]); if (d > best) { best = d; a = s; } }
    var A = seeds[a], bd = 9;
    for (s = 0; s < seeds.length; s++) {
      if (s === a) continue;
      var B = seeds[s], dx = A.x - B.x, dy = A.y - B.y, dz = A.z - B.z, l = Math.sqrt(dx * dx + dy * dy + dz * dz);
      bd = Math.min(bd, (q.x * dx + q.y * dy + q.z * dz) / l);
    }
    nationOf = a;
    return bd;
  }

  // ── Cities: each nation has a capital and a few towns, mostly coastal ──
  var cities = [];
  function addCity(p, size, nation) { cities.push({ p: p.clone(), s: size, n: nation }); }
  seeds.forEach(function (sd, n) {
    addCity(sd, 0.7 + r() * 0.5, n);
    var want = 2 + Math.floor(r() * 4), got = 0;
    for (var t = 0; t < 200 && got < want; t++) {
      v.copy(sd).addScaledVector(randomDir(w), 0.06 + r() * 0.2).normalize();
      if (!isLand(v, 0.015)) continue;
      var bd = borderDist(v);
      if (nationOf !== n || bd < 0.02) continue;
      var coastal = elev(v.x, v.y, v.z) < THR + 0.05;
      if (!coastal && r() < 0.6) continue;
      addCity(v, 0.25 + Math.pow(r(), 2) * 0.8, n);
      got++;
    }
  });
  // Two cities the first arc joins: on the coast below you and on the far shore.
  // The land nearest a point (th along the orbit, x to the side), well inland of the coast by m.
  function landNear(th, x, m) {
    var best = null, bd = 9, tgt = new THREE.Vector3(x, Math.cos(th), -Math.sin(th)).normalize();
    for (var a = 0; a < 60; a++) {
      for (var b = 0; b < 60; b++) {
        v.set(x + (b - 30) * 0.006, Math.cos(th + (a - 30) * 0.006), -Math.sin(th + (a - 30) * 0.006)).normalize();
        if (!isLand(v, m)) continue;
        var d = v.angleTo(tgt);
        if (d < bd) { bd = d; best = v.clone(); }
      }
    }
    return best;
  }
  var heroA = landNear(0.18, -0.1, 0.02), heroB = landNear(0.52, 0.12, 0.02);
  if (heroA) { borderDist(heroA); addCity(heroA, 0.9, nationOf); heroA = cities[cities.length - 1]; }
  if (heroB) { borderDist(heroB); addCity(heroB, 0.9, nationOf); heroB = cities[cities.length - 1]; }

  // ── City lights: clustered points, towns, and roads between them ──
  // aInfo: size, joins (waits for the borders to dissolve), tone, when it joins.
  // Land is looked up in the baked heights here (it's much faster).
  function texel(p) {
    var la = Math.asin(clamp(p.x, -1, 1)), lo = Math.atan2(-p.z, p.y);
    var ty = clamp(Math.floor((la / Math.PI + 0.5) * TH), 0, TH - 1), tx = Math.floor((lo / (Math.PI * 2) + 0.5) * TW) % TW;
    return ty * TW + tx;
  }
  var LP = [], LI = [], scale = small ? 0.45 : 0.8, margin = 0.007;
  function light(p, size, tone, mustJoin) {
    var t = texel(p);
    if (height[t] < THR + 0.01 || Math.abs(p.x) > 0.86) return false;
    var bd = borderDist(p), join = mustJoin || bd < margin ? 1 : 0;
    LP.push(p.x * 1.0004, p.y * 1.0004, p.z * 1.0004);
    LI.push(size, join, tone, r() * 0.8);
    glow[t] += size * 400;
    return true;
  }
  // A clump of lights round p: a city's sprawl, or a small town.
  function clump(p, n, sig, big) {
    for (var j = 0; j < n; j++) {
      var rad = sig * Math.sqrt(-2 * Math.log(1 - r() * 0.999)) * 0.7, core = Math.exp(-rad / sig * 1.6);
      v.copy(p).addScaledVector(randomDir(w), rad).normalize();
      light(v, (0.00045 + core * 0.0011 * big + r() * 0.0004), r() < 0.1 + core * 0.4 ? 1 : r() * 0.45, false);
    }
  }
  var lobe = new THREE.Vector3();
  cities.forEach(function (c) {
    var N = Math.round((80 + 800 * c.s * c.s) * scale), sig = 0.003 + 0.008 * c.s;
    // A few lobes, so the sprawl isn't a perfect disc, and suburbs round it.
    for (k = 0; k < 4; k++) {
      lobe.copy(c.p).addScaledVector(randomDir(w), sig * (k ? 1.6 : 0) * r()).normalize();
      clump(lobe, Math.round(N / 4), sig * (k ? 0.7 : 1), 1);
    }
    for (k = 0; k < 6; k++) {
      lobe.copy(c.p).addScaledVector(randomDir(w), sig * (2.5 + r() * 3)).normalize();
      clump(lobe, Math.round((10 + r() * 30) * scale), sig * 0.3, 0.4);
    }
    c.sig = sig;
  });
  // Towns: small clumps where the population noise is high, thickest along
  // the coasts; most of them near the orbit, where they are seen close.
  var towns = Math.round(6500 * scale), placed = 0, e;
  for (k = 0; k < towns * 12 && placed < towns; k++) {
    if (k % 3) {
      var th = -0.7 + r() * 2.1, sx = (r() - 0.5) * 1.3;
      v.set(sx, Math.cos(th), -Math.sin(th)).normalize();
    } else randomDir(v);
    e = height[texel(v)];
    if (e < THR + 0.01 || Math.abs(v.x) > 0.8) continue;
    var dens = smooth(0.42, 0.72, nPop(v.x * 11, v.y * 11, v.z * 11)) * 0.55 + (e < THR + 0.035 ? 0.55 : 0);
    if (r() > dens) continue;
    clump(v, 2 + Math.floor(Math.pow(r(), 2) * 22 * Math.max(scale, 0.6)), 0.0012 + r() * 0.0015, 0.35);
    placed++;
  }
  // Roads: threads of lights between nearby cities, within a nation, and
  // across borders (those wait for the borders to go).
  for (i = 0; i < cities.length; i++) {
    for (k = i + 1; k < cities.length; k++) {
      var A = cities[i], B = cities[k], ang = A.p.angleTo(B.p);
      if (ang > 0.2 || ang < 0.03) continue;
      var cross = A.n !== B.n;
      if (cross && r() < 0.3) continue;
      var steps = Math.round(ang / 0.002 * scale), wig = r() * 6.28, amp = 0.002 + r() * 0.006;
      w.crossVectors(A.p, B.p).normalize();
      for (var s2 = 0; s2 < steps; s2++) {
        var t = r();
        v.copy(A.p).lerp(B.p, t).normalize();
        v.addScaledVector(w, Math.sin(t * 9 + wig) * amp + (r() - 0.5) * 0.0012).normalize();
        light(v, 0.00035 + r() * 0.0004, r() * 0.3, cross);
      }
    }
  }
  var lightGeo = new THREE.BufferGeometry();
  lightGeo.setAttribute('position', new THREE.Float32BufferAttribute(LP, 3));
  lightGeo.setAttribute('aInfo', new THREE.Float32BufferAttribute(LI, 4));

  // The lights' glow, blurred, into the texture's G channel.
  function blur(a, rad, horiz) {
    var out = new Float32Array(a.length), n = rad * 2 + 1;
    for (var yy = 0; yy < TH; yy++) {
      for (var xx = 0; xx < TW; xx++) {
        var sum = 0;
        for (var d = -rad; d <= rad; d++) {
          sum += horiz ? a[yy * TW + (xx + d + TW) % TW] : a[clamp(yy + d, 0, TH - 1) * TW + xx];
        }
        out[yy * TW + xx] = sum / n;
      }
    }
    return out;
  }
  glow = blur(blur(blur(blur(glow, 2, true), 2, false), 1, true), 1, false);
  var tex = new Uint8Array(TW * TH * 4);
  for (k = 0; k < TW * TH; k++) {
    tex[k * 4] = clamp((height[k] - 0.2) / 0.7, 0, 1) * 255;
    tex[k * 4 + 1] = clamp(Math.sqrt(glow[k]) * (small ? 0.36 : 0.5), 0, 1) * 255;
    tex[k * 4 + 3] = 255;
  }
  var mapTex = new THREE.DataTexture(tex, TW, TH, THREE.RGBAFormat);
  mapTex.wrapS = THREE.RepeatWrapping;
  mapTex.magFilter = mapTex.minFilter = THREE.LinearFilter;
  mapTex.needsUpdate = true;

  // ── Sky: stars, then the sun ──
  var stars = starField(r, small ? 2200 : 4500, 300, -1, 1.3);
  var brightStars = starField(r, small ? 250 : 500, 300, -1, 2.4);
  world.add(stars, brightStars);
  var sunDir = new THREE.Vector3(), moonDir = new THREE.Vector3(0.5, 0.75, 0.42).normalize();
  var sun = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,250,235,1)', 'rgba(255,215,160,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
  sun.scale.setScalar(9);
  var rays = new THREE.Sprite(new THREE.SpriteMaterial({ map: starburst(), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
  rays.scale.setScalar(80);
  world.add(sun, rays);

  // ── The globe ──
  var globe = new THREE.Group();
  world.add(globe);
  var gU = {
    uMap: { value: mapTex }, uSeeds: { value: seeds }, uSun: { value: sunDir }, uMoon: { value: moonDir },
    uBorder: { value: 0 }, uFlare: { value: 0 }, uDissolve: { value: 0 }, uTime: { value: 0 }
  };
  var earth = new THREE.Mesh(new THREE.SphereGeometry(1, small ? 192 : 288, small ? 96 : 144), new THREE.ShaderMaterial({
    uniforms: gU,
    vertexShader: 'varying vec3 vL; varying vec3 vW;\n' +
      'void main(){ vL = position; vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }',
    fragmentShader: '#define NS ' + seeds.length + '\n' + NOISE_GLSL + WOBBLE_GLSL +
      'uniform sampler2D uMap; uniform vec3 uSeeds[NS]; uniform vec3 uSun; uniform vec3 uMoon;\n' +
      'uniform float uBorder; uniform float uFlare; uniform float uDissolve; uniform float uTime;\n' +
      'varying vec3 vL; varying vec3 vW;\n' +
      'float border(vec3 n){ vec3 q = wob(n), A = uSeeds[0]; float best = -2.0;\n' +
      ' for (int i = 0; i < NS; i++) { float d = dot(q, uSeeds[i]); if (d > best) { best = d; A = uSeeds[i]; } }\n' +
      ' float bd = 9.0; for (int i = 0; i < NS; i++) { vec3 D = A - uSeeds[i]; float l = length(D); if (l > 1e-4) bd = min(bd, dot(q, D) / l); }\n' +
      ' return bd; }\n' +
      'void main(){\n' +
      ' vec3 n = normalize(vL), N = normalize(vW), V = normalize(cameraPosition - vW);\n' +
      ' float lat = asin(clamp(n.x, -1.0, 1.0)), lon = atan(-n.z, n.y);\n' +
      ' vec4 m = texture2D(uMap, vec2(lon / 6.2831853 + 0.5, lat / 3.1415927 + 0.5));\n' +
      ' float e = m.r * 0.7 + 0.2 + (n3(n * 38.0) - 0.5) * 0.05 + (n3(n * 140.0) - 0.5) * 0.024 + (n3(n * 420.0) - 0.5) * 0.01;\n' +
      ' float land = smoothstep(' + (THR - 0.003).toFixed(3) + ', ' + (THR + 0.003).toFixed(3) + ', e);\n' +
      ' float ice = smoothstep(0.86, 0.9, abs(n.x) + 0.04 * n3(n * 20.0));\n' +
      ' float cl = clouds(n) * (1.0 - 0.5 * ice);\n' +
      ' float l = dot(N, uSun), day = smoothstep(-0.06, 0.22, l), night = 1.0 - smoothstep(-0.12, 0.04, l);\n' +
      ' float mu = max(dot(N, uMoon), 0.0) * 0.8 + 0.2;\n' +
      // Day: oceans, forests, deserts, ice and white cloud.
      ' float dry = smoothstep(0.42, 0.72, fbm(n * 3.0 + 9.0)) * (1.0 - abs(n.x) * 1.5);\n' +
      ' float tex = fbm(n * 22.0 + 5.0) * 1.3 - 0.15;\n' +
      ' vec3 ocean = mix(vec3(0.004, 0.016, 0.05), vec3(0.01, 0.05, 0.09), smoothstep(' + (THR - 0.025).toFixed(3) + ', ' + THR.toFixed(3) + ', e));\n' +
      ' vec3 ground = mix(vec3(0.035, 0.06, 0.025), vec3(0.11, 0.1, 0.05), smoothstep(0.3, 0.7, tex));\n' +
      ' ground = mix(ground, vec3(0.32, 0.24, 0.14) * (0.8 + 0.4 * tex), dry);\n' +
      ' ground = mix(ground, vec3(0.2, 0.18, 0.16), smoothstep(0.66, 0.78, e) * 0.6);\n' +
      ' vec3 base = mix(mix(ocean, ground, land), vec3(0.75, 0.8, 0.86), ice);\n' +
      ' vec3 sunC = mix(vec3(1.0, 0.45, 0.2), vec3(1.0, 0.96, 0.9), smoothstep(0.0, 0.35, l));\n' +
      ' float dayC = smoothstep(-0.05, 0.1, l);\n' +
      // Cloud tops catch the dawn before the ground does.
      ' vec3 c = mix(base * day, vec3(0.78, 0.8, 0.84) * dayC, cl * 0.9) * sunC;\n' +
      ' vec3 hv = normalize(uSun + V); float gl = max(dot(N, hv), 0.0); c += vec3(1.0, 0.82, 0.6) * (pow(gl, 500.0) * 0.35 + pow(gl, 30.0) * 0.025) * (1.0 - land) * (1.0 - cl) * day;\n' +
      ' c += vec3(1.0, 0.4, 0.2) * exp(-(l + 0.01) * (l + 0.01) * 900.0) * 0.025 * (1.0 + cl);\n' +
      // Night: moonlit cloud tops and faint land, the cities\' glow under them.
      ' vec3 nb = mix(vec3(0.0015, 0.003, 0.009), vec3(0.008, 0.009, 0.010) * (0.7 + 0.6 * tex), land);\n' +
      ' nb = mix(nb, vec3(0.022, 0.026, 0.036), cl * 0.75) * mu;\n' +
      ' c += nb * night;\n' +
      ' c += vec3(1.0, 0.5, 0.18) * m.g * m.g * 0.12 * night * (1.0 - cl * 0.6);\n' +
      // Borders: a thin line and a soft glow, only on land, at night.
      ' if (uBorder > 0.001) {\n' +
      '  float bd = border(n), aa = length(fwidth(vW)) * 0.6;\n' +
      '  float core = 1.0 - smoothstep(aa * 0.5, aa * (1.6 + uFlare * 0.4), bd);\n' +
      '  float halo = exp(-bd / max(aa * (5.0 + uFlare * 4.0), 1e-6));\n' +
      '  float k = n3(n * 90.0) * 0.6 + n3(n * 260.0) * 0.4, keep = smoothstep(uDissolve * 1.2 - 0.12, uDissolve * 1.2, k);\n' +
      '  float spark = keep * (1.0 - keep) * 4.0 * step(0.01, uDissolve);\n' +
      '  vec3 bc = mix(vec3(0.42, 0.7, 1.0), vec3(0.55, 0.8, 1.0), uFlare);\n' +
      '  float shimmer = 0.85 + 0.15 * sin(uTime * 2.0 + bd * 900.0);\n' +
      '  c += bc * (core * (0.6 + uFlare * 0.7) * keep + halo * (0.07 + uFlare * 0.16) * (1.0 - smoothstep(0.0, 0.7, uDissolve))) * shimmer * uBorder * land * (1.0 - ice) * (1.0 - day);\n' +
      '  c += vec3(1.0, 0.8, 0.5) * spark * core * 1.5 * uBorder * land * (1.0 - day);\n' +
      ' }\n' +
      // Haze at the limb: blue by day, a warm band at dawn, airglow at night.
      ' float rim = pow(1.0 - max(dot(N, V), 0.0), 4.0);\n' +
      ' c += vec3(0.3, 0.55, 1.0) * rim * day * 0.45 + vec3(0.04, 0.08, 0.1) * rim * night;\n' +
      ' gl_FragColor = vec4(c, 1.0);\n#include <tonemapping_fragment>\n#include <colorspace_fragment>\n}'
  }));
  globe.add(earth);

  // City lights as points, dimmed under clouds, by day, and at grazing angles.
  var lightMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uSun: gU.uSun, uJoin: { value: 0 }, uPx: { value: 500 }, uTime: gU.uTime, uWarm: { value: 0 } },
    vertexShader: NOISE_GLSL + 'attribute vec4 aInfo; uniform vec3 uSun; uniform float uJoin; uniform float uPx; uniform float uTime;\n' +
      'varying float vA; varying float vTone;\n' +
      'void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vec4 mv = viewMatrix * w; gl_Position = projectionMatrix * mv;\n' +
      ' vec3 N = normalize(w.xyz), V = normalize(cameraPosition - w.xyz);\n' +
      ' float night = 1.0 - smoothstep(-0.1, 0.06, dot(N, uSun));\n' +
      ' float join = mix(1.0, smoothstep(aInfo.w, aInfo.w + 0.2, uJoin), aInfo.y);\n' +
      ' float graze = smoothstep(0.0, 0.3, dot(N, V));\n' +
      ' float cl = clouds(normalize(position));\n' +
      ' float px = aInfo.x * uPx / -mv.z;\n' +
      ' vA = night * join * graze * (1.0 - cl * 0.75) * min(px / 1.4, 1.0) * (0.9 + 0.1 * sin(uTime * 3.0 + aInfo.w * 40.0));\n' +
      ' vTone = aInfo.z;\n' +
      ' gl_PointSize = clamp(px, 1.4, 9.0); }',
    fragmentShader: 'uniform float uWarm; varying float vA; varying float vTone;\n' +
      'void main(){ vec2 c = gl_PointCoord - 0.5; float d = dot(c, c) * 4.0; if (d > 1.0) discard;\n' +
      ' float a = exp(-d * 3.0) * vA;\n' +
      ' vec3 col = mix(vec3(1.0, 0.62, 0.26), vec3(1.0, 0.9, 0.72), vTone);\n' +
      ' col = mix(col, vec3(1.0, 0.8, 0.45), uWarm * 0.5);\n' +
      ' gl_FragColor = vec4(col * a * 1.3, a);\n#include <colorspace_fragment>\n}'
  });
  var cityLights = new THREE.Points(lightGeo, lightMat);
  cityLights.frustumCulled = false;
  globe.add(cityLights);

  // Lightning, now and then, in the storm clouds out over the sea.
  var LZ = [];
  for (k = 0; k < 400 && LZ.length < 3 * 14; k++) {
    var lth = 0.15 + r() * 0.55, lx = (r() - 0.5) * 0.7;
    v.set(lx, Math.cos(lth), -Math.sin(lth)).normalize();
    if (elev(v.x, v.y, v.z) > THR - 0.02) continue;
    v.multiplyScalar(1.003);
    LZ.push(v.x, v.y, v.z);
  }
  var boltGeo = new THREE.BufferGeometry();
  boltGeo.setAttribute('position', new THREE.Float32BufferAttribute(LZ, 3));
  var boltMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uTime: gU.uTime, uSun: gU.uSun, uPx: lightMat.uniforms.uPx, uOn: { value: 1 } },
    vertexShader: 'uniform float uTime; uniform vec3 uSun; uniform float uPx; uniform float uOn; varying float vA;\n' +
      'void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vec4 mv = viewMatrix * w; gl_Position = projectionMatrix * mv;\n' +
      ' float id = float(gl_VertexID), t = uTime * (0.35 + fract(id * 0.618) * 0.3) + id * 1.7;\n' +
      ' float k = fract(t), burst = step(0.82, fract(sin(floor(t) * 91.7 + id * 13.1) * 437.5));\n' +
      ' vA = burst * (exp(-k * 40.0) + 0.6 * exp(-abs(k - 0.06) * 60.0)) * (1.0 - smoothstep(-0.1, 0.05, dot(normalize(w.xyz), uSun))) * uOn;\n' +
      ' gl_PointSize = clamp(0.03 * uPx / -mv.z, 4.0, 60.0); }',
    fragmentShader: 'varying float vA; void main(){ vec2 c = gl_PointCoord - 0.5; float d = dot(c, c) * 4.0; if (d > 1.0) discard;\n' +
      ' float a = exp(-d * 5.0) * vA * 0.7; gl_FragColor = vec4(vec3(0.75, 0.82, 1.0) * a, a);\n#include <colorspace_fragment>\n}'
  });
  var bolts = new THREE.Points(boltGeo, boltMat);
  bolts.frustumCulled = false;
  globe.add(bolts);

  // ── Arcs: great circles lifted off the surface, drawn in as tubes ──
  var p0 = new THREE.Vector3(), p1 = new THREE.Vector3(), tan = new THREE.Vector3(), nrm = new THREE.Vector3(), bin = new THREE.Vector3();
  function arcPoint(a, b, ang, lift, t, out) {
    var s = Math.sin(ang), wa = Math.sin((1 - t) * ang) / s, wb = Math.sin(t * ang) / s;
    return out.copy(a).multiplyScalar(wa).addScaledVector(b, wb).multiplyScalar(1.0006 + lift * Math.sin(Math.PI * t));
  }
  var majors = cities.filter(function (c) { return c.s > 0.6; }), pairs = [];
  function water(a, b) {
    var n = 0;
    for (var s = 1; s < 8; s++) { v.copy(a).lerp(b, s / 8).normalize(); if (elev(v.x, v.y, v.z) < THR) n++; }
    return n;
  }
  majors.forEach(function (a) {
    var cand = [];
    majors.forEach(function (b) {
      if (a === b || a.n === b.n) return;
      var ang = a.p.angleTo(b.p);
      if (ang < 0.12 || ang > 1.1) return;
      var wn = water(a.p, b.p);
      cand.push({ b: b, score: ang - wn * 0.06 });
    });
    cand.sort(function (x, y) { return x.score - y.score; });
    for (var c = 0; c < Math.min(2, cand.length); c++) {
      var b = cand[c].b;
      if (!pairs.some(function (p) { return (p[0] === a && p[1] === b) || (p[0] === b && p[1] === a); })) pairs.push([a, b]);
    }
  });
  // Nearest the view first, so the web visibly grows from where you are.
  var view0 = new THREE.Vector3(0, Math.cos(0.4), -Math.sin(0.4));
  pairs.sort(function (x, y) { return y[0].p.dot(view0) + y[1].p.dot(view0) - x[0].p.dot(view0) - x[1].p.dot(view0); });
  // Skip any that pass close beneath the orbit: seen from just above, they
  // would streak past the camera instead of arching across the view.
  pairs = pairs.filter(function (pr) {
    var a = pr[0].p, b = pr[1].p, ang = a.angleTo(b);
    for (var s = 0; s <= 20; s++) {
      arcPoint(a, b, ang, 0.006 + 0.075 * ang, s / 20, p0);
      var th = Math.atan2(-p0.z, p0.y);
      if (Math.abs(p0.x) < 0.2 && th > -0.5 && th < 0.14) return false;
    }
    return true;
  }).slice(0, small ? 60 : 90);
  if (heroA && heroB) pairs.unshift([heroA, heroB]);

  var AP = [], AN = [], AT = [], AA = [], idx = [], base = 0, SEG = 48, RAD = 6, ep = [], ea = [];
  pairs.forEach(function (pr, pi) {
    var a = pr[0].p, b = pr[1].p, ang = a.angleTo(b), lift = 0.006 + 0.075 * ang, hero = pi === 0 && heroA ? 1 : 0;
    var start = hero ? 0 : 0.6 * (pi - 1) / pairs.length + r() * 0.08, dur = 0.22 + 0.25 * ang, rad = hero ? 0.0011 : 0.0007;
    for (var s = 0; s <= SEG; s++) {
      var t = s / SEG;
      arcPoint(a, b, ang, lift, t, p0);
      arcPoint(a, b, ang, lift, Math.min(t + 0.01, 1), p1);
      if (t + 0.01 > 1) { arcPoint(a, b, ang, lift, t - 0.01, p1); tan.subVectors(p0, p1).normalize(); }
      else tan.subVectors(p1, p0).normalize();
      nrm.copy(p0).normalize().cross(tan).normalize();
      bin.crossVectors(tan, nrm);
      for (var j = 0; j < RAD; j++) {
        var ag = j / RAD * Math.PI * 2, cx = Math.cos(ag), cy = Math.sin(ag);
        w.copy(nrm).multiplyScalar(cx).addScaledVector(bin, cy);
        AP.push(p0.x + w.x * rad, p0.y + w.y * rad, p0.z + w.z * rad);
        AN.push(w.x, w.y, w.z);
        AT.push(t);
        AA.push(start, dur, hero, pi * 0.37);
      }
    }
    for (s = 0; s < SEG; s++) {
      for (var j2 = 0; j2 < RAD; j2++) {
        var aI = base + s * RAD + j2, bI = base + s * RAD + (j2 + 1) % RAD, cI = aI + RAD, dI = bI + RAD;
        idx.push(aI, cI, bI, bI, cI, dI);
      }
    }
    base += (SEG + 1) * RAD;
    // Endpoints flare as the arc leaves and lands.
    ep.push(a.x * 1.001, a.y * 1.001, a.z * 1.001, b.x * 1.001, b.y * 1.001, b.z * 1.001);
    ea.push(start, hero, 0, start + dur, hero, 1);
  });
  var arcGeo = new THREE.BufferGeometry();
  arcGeo.setAttribute('position', new THREE.Float32BufferAttribute(AP, 3));
  arcGeo.setAttribute('normal', new THREE.Float32BufferAttribute(AN, 3));
  arcGeo.setAttribute('aT', new THREE.Float32BufferAttribute(AT, 1));
  arcGeo.setAttribute('aArc', new THREE.Float32BufferAttribute(AA, 4));
  arcGeo.setIndex(idx);
  var arcU = { uWeb: { value: 0 }, uHero: { value: 0 }, uTime: gU.uTime, uSun: gU.uSun, uDay: { value: 0 }, uThick: { value: 0 } };
  var arcMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, uniforms: arcU,
    vertexShader: 'attribute float aT; attribute vec4 aArc; uniform float uWeb; uniform float uHero; uniform float uThick;\n' +
      'varying float vT; varying float vP; varying float vSeed; varying vec3 vN; varying vec3 vW;\n' +
      'void main(){ vT = aT; vSeed = aArc.w; vP = aArc.z > 0.5 ? uHero : clamp((uWeb - aArc.x) / aArc.y, 0.0, 1.0);\n' +
      ' vN = normalize(mat3(modelMatrix) * normal); vec4 w = modelMatrix * vec4(position + normal * uThick, 1.0); vW = w.xyz;\n' +
      ' gl_Position = projectionMatrix * viewMatrix * w; }',
    fragmentShader: 'uniform float uTime; uniform float uDay; varying float vT; varying float vP; varying float vSeed; varying vec3 vN; varying vec3 vW;\n' +
      'void main(){ if (vT > vP + 0.002 || vP <= 0.0) discard;\n' +
      ' vec3 V = normalize(cameraPosition - vW); float core = pow(abs(dot(normalize(vN), V)), 1.5);\n' +
      ' float head = exp(-pow((vP - vT) * 30.0, 2.0)) * step(vP, 0.999);\n' +
      ' float pulse = exp(-pow((fract(vT * 0.8 - uTime * 0.18 + vSeed) - 0.5) * 9.0, 2.0)) * step(0.999, vP);\n' +
      ' float a = core * (0.55 + 1.6 * head + 0.6 * pulse) * (1.0 + uDay * 0.15);\n' +
      ' vec3 col = mix(vec3(1.0, 0.72, 0.35), vec3(1.0, 0.95, 0.85), (head + core * 0.3) * (1.0 - uDay * 0.6));\n' +
      ' gl_FragColor = vec4(col * a, a);\n#include <colorspace_fragment>\n}'
  });
  var arcs = new THREE.Mesh(arcGeo, arcMat);
  arcs.frustumCulled = false;
  globe.add(arcs);

  var endGeo = new THREE.BufferGeometry();
  endGeo.setAttribute('position', new THREE.Float32BufferAttribute(ep, 3));
  endGeo.setAttribute('aEnd', new THREE.Float32BufferAttribute(ea, 3));
  var endMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uWeb: arcU.uWeb, uHero: arcU.uHero, uPx: lightMat.uniforms.uPx, uDay: arcU.uDay },
    vertexShader: 'attribute vec3 aEnd; uniform float uWeb; uniform float uHero; uniform float uPx; varying float vA;\n' +
      'void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;\n' +
      ' float p = aEnd.y > 0.5 ? uHero : uWeb;\n' +
      ' float at = aEnd.y > 0.5 ? (aEnd.z > 0.5 ? 0.98 : 0.0) : aEnd.x;\n' +
      ' float on = smoothstep(at, at + 0.02, p), flash = on * exp(-max(p - at, 0.0) * 12.0);\n' +
      ' vA = on * 0.5 + flash * 1.5;\n' +
      ' gl_PointSize = clamp(0.012 * uPx / -mv.z, 3.0, 26.0) * (1.0 + flash); }',
    fragmentShader: 'uniform float uDay; varying float vA; void main(){ vec2 c = gl_PointCoord - 0.5; float d = dot(c, c) * 4.0; if (d > 1.0) discard;\n' +
      ' float a = (exp(-d * 6.0) + exp(-d * 30.0)) * vA * (1.0 - uDay * 0.5);\n' +
      ' gl_FragColor = vec4(vec3(1.0, 0.85, 0.6) * a, a);\n#include <colorspace_fragment>\n}'
  });
  var ends = new THREE.Points(endGeo, endMat);
  ends.frustumCulled = false;
  globe.add(ends);

  // ── The atmosphere: a shell whose colour comes from each view ray's
  // closest approach to the ground (blue by day, a red-gold band at the
  // terminator, the thin green airglow line at night) ──
  var atmoU = { uSun: gU.uSun, uAmt: { value: 1 } };
  var atmo = new THREE.Mesh(new THREE.SphereGeometry(1.06, 128, 64), new THREE.ShaderMaterial({
    side: THREE.BackSide, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, uniforms: atmoU,
    vertexShader: 'varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }',
    fragmentShader: 'uniform vec3 uSun; uniform float uAmt; varying vec3 vW;\n' +
      'void main(){ vec3 C = cameraPosition, V = normalize(vW - C);\n' +
      ' float t0 = max(-dot(C, V), 0.0); vec3 P = C + V * t0; float d = length(P);\n' +
      ' if (d < 1.0) discard;\n' +
      ' float x = (d - 1.0) / 0.022, s = dot(P / d, uSun);\n' +
      ' float dens = exp(-x * 3.2) * step(x, 1.6);\n' +
      ' float day = smoothstep(-0.18, 0.2, s), twi = exp(-pow((s + 0.02) / 0.13, 2.0));\n' +
      ' vec3 col = vec3(0.22, 0.48, 1.0) * dens * day * 1.0;\n' +
      ' col += vec3(1.0, 0.42, 0.14) * exp(-x * 9.0) * twi * 1.6 + vec3(0.55, 0.3, 0.6) * exp(-x * 2.5) * twi * 0.25;\n' +
      ' col += vec3(0.3, 0.9, 0.48) * exp(-pow((x - 0.72) / 0.045, 2.0)) * (1.0 - day) * 0.085;\n' +
      ' col += vec3(1.0, 0.8, 0.55) * pow(max(dot(V, uSun), 0.0), 30.0) * dens * 2.5;\n' +
      ' col *= uAmt; gl_FragColor = vec4(col, 1.0);\n#include <colorspace_fragment>\n}'
  }));
  world.add(atmo);

  var H = 800, dpr = 1, drift = 0, portrait = false, look = new THREE.Vector3();

  function frame(f) {
    var row = f.row, orbit = row[0], alt = row[4], down = row[5], yaw = row[6];
    if (!env.reduceMotion) drift += f.dt * 0.0012;

    globe.rotation.x = orbit + drift;
    gU.uTime.value = f.time;
    gU.uBorder.value = row[7];
    gU.uFlare.value = row[8];
    gU.uDissolve.value = row[9];
    lightMat.uniforms.uJoin.value = row[9];
    arcU.uWeb.value = row[10];
    arcU.uHero.value = row[12];

    // The sun: below the far limb, rising, then up over it.
    var beta = row[11];
    sunDir.set(SUN_X, Math.sin(beta), -Math.cos(beta)).normalize();
    var dayAmt = smooth(-0.9, 0.2, beta);
    arcU.uDay.value = dayAmt;
    arcU.uThick.value = 0.0006 * smooth(0.1, 0.55, alt);
    lightMat.uniforms.uWarm.value = row[10];
    boltMat.uniforms.uOn.value = env.reduceMotion ? 0 : 1;

    // In orbit over the nadir, looking ahead past the horizon.
    var dip = Math.acos(1 / (1 + alt)), el = -dip - down - (portrait ? 0.12 * (1 - smooth(0.2, 0.5, alt)) : 0) - f.my * 0.03, az = yaw - f.mx * 0.05;
    camera.position.set(0, 1 + alt, 0);
    look.set(Math.sin(az) * Math.cos(el), Math.sin(el), -Math.cos(az) * Math.cos(el)).add(camera.position);
    camera.up.set(0, 1, 0);
    camera.lookAt(look);
    camera.near = Math.max(alt * 0.2, 0.004);
    camera.updateProjectionMatrix();

    sun.position.copy(sunDir).multiplyScalar(250);
    rays.position.copy(sun.position);
    rays.material.rotation = f.time * 0.01;
    var rise = smooth(-dip - 0.08, -dip + 0.05, beta);
    sun.material.opacity = rise;
    rays.material.opacity = rise * (0.35 + 0.65 * Math.exp(-Math.pow((beta + dip) / 0.12, 2)));
    rays.scale.setScalar(60 + 60 * Math.exp(-Math.pow((beta + dip) / 0.12, 2)));

    var starAmt = 1 - smooth(-0.75, -0.2, beta);
    stars.material.opacity = 0.75 * starAmt;
    brightStars.material.opacity = 0.9 * starAmt;
    gl.toneMappingExposure = 1.0;
    lightMat.uniforms.uPx.value = H * dpr / (2 * Math.tan(camera.fov * Math.PI / 360));

    gl.render(world, camera);
  }

  return {
    resize: function (wd, hh, d) {
      H = hh;
      portrait = wd < hh;
      dpr = Math.min(d, small ? 1.5 : 1.75);
      gl.setPixelRatio(dpr);
      gl.setSize(wd, hh, false);
      camera.aspect = wd / hh;
      camera.fov = wd / hh < 1 ? 72 : 55;
      camera.updateProjectionMatrix();
    },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

// A soft cross of light for the sun on the limb.
function starburst() {
  var c = document.createElement('canvas');
  c.width = c.height = 256;
  var x = c.getContext('2d');
  x.translate(128, 128);
  x.globalCompositeOperation = 'lighter';
  for (var k = 0; k < 12; k++) {
    x.save();
    x.rotate(k / 12 * Math.PI * 2 + (k % 2) * 0.1);
    var len = k % 3 === 0 ? 128 : 70, g = x.createLinearGradient(0, 0, len, 0);
    g.addColorStop(0, 'rgba(255,240,215,0.55)');
    g.addColorStop(1, 'rgba(255,200,150,0)');
    x.fillStyle = g;
    x.beginPath();
    x.moveTo(0, -2.2); x.lineTo(len, 0); x.lineTo(0, 2.2);
    x.fill();
    x.restore();
  }
  var rg = x.createRadialGradient(0, 0, 0, 0, 0, 128);
  rg.addColorStop(0, 'rgba(255,235,200,0.6)');
  rg.addColorStop(0.15, 'rgba(255,190,130,0.18)');
  rg.addColorStop(1, 'rgba(255,160,100,0)');
  x.fillStyle = rg;
  x.fillRect(-128, -128, 256, 256);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

PI.register('one-family', {
  renderer: renderer3d,
  align: ['center', 'center', 'right', 'right'],
  scrim: 0.6,
  accent: '#ffcf7a',
  emphasis: /family|कुटुम्बकम्/,
  // Panels: 0 the Sanskrit couplet, 1 "The world is a family", 2 "One is a
  // relative ... live the magnanimous.", 3 "Be detached ... freedom."
  keys: function (T) {
    function at(i, d) { return T.start(i) + d; }    // d units into panel i (0..1.6)
    //   unit           orbit  -  -  wind  alt   down  yaw    border flare dissolve web   sun    first
    return [
      [0,              -0.20, 0, 0, 0.3, 0.14, 0.17, 0.00,  0.00, 0.00, 0.00, 0.00, -1.05, 0.00],
      [0.7,            -0.19, 0, 0, 0.3, 0.14, 0.17, 0.00,  0.00, 0.00, 0.00, 0.00, -1.05, 0.00],
      [at(0, 0.3),     -0.16, 0, 0, 0.3, 0.14, 0.17, 0.00,  0.00, 0.00, 0.00, 0.00, -1.05, 0.00],
      [at(0, 1.2),     -0.08, 0, 0, 0.3, 0.14, 0.18, 0.00,  0.75, 0.00, 0.00, 0.00, -1.05, 0.00],  // borders drawn
      [at(1, 0.35),    -0.07, 0, 0, 0.3, 0.14, 0.18, 0.00,  0.75, 0.00, 0.00, 0.00, -1.05, 0.00],
      [at(1, 1.15),    -0.04, 0, 0, 0.3, 0.14, 0.18, 0.00,  0.75, 0.00, 0.00, 0.00, -1.05, 1.00],  // "The world is a family": the first arc
      [at(2, 0.2),     -0.03, 0, 0, 0.3, 0.14, 0.24, 0.00,  0.75, 0.00, 0.00, 0.00, -1.05, 1.00],
      [at(2, 0.6),     -0.01, 0, 0, 0.3, 0.14, 0.27, 0.00,  1.00, 1.00, 0.00, 0.00, -1.05, 1.00],  // "say the small minded"
      [at(2, 0.8),      0.00, 0, 0, 0.3, 0.14, 0.25, 0.00,  1.00, 0.70, 0.00, 0.00, -1.05, 1.00],
      [at(2, 1.35),     0.04, 0, 0, 0.3, 0.15, 0.19, 0.00,  1.00, 0.00, 1.00, 0.75, -1.05, 1.00],  // "The entire world is a family": the web
      [at(3, 0.2),      0.07, 0, 0, 0.3, 0.16, 0.17, 0.00,  0.00, 0.00, 1.00, 1.00, -1.00, 1.00],
      [at(3, 0.65),     0.10, 0, 0, 0.3, 0.24, 0.18, 0.00,  0.00, 0.00, 1.00, 1.00, -0.68, 1.00],  // "lift up your mind": sunrise
      [at(3, 1.2),      0.13, 0, 0, 0.3, 0.36, 0.20, 0.00,  0.00, 0.00, 1.00, 1.00, -0.28, 1.00],  // dawn sweeps towards you
      [at(3, 1.6),      0.15, 0, 0, 0.3, 0.45, 0.24, 0.00,  0.00, 0.00, 1.00, 1.00, -0.02, 1.00],
      [T.total,         0.18, 0, 0, 0.3, 0.55, 0.30, 0.00,  0.00, 0.00, 1.00, 1.00,  0.15, 1.00]   // one lit world
    ];
  },
  sound: {
    src: '/audio/wind.mp3',
    label: 'Play the quiet of orbit: wind, a tanpura and chimes',
    volume: function () { return 0.05; },
    cues: [
      { stanza: 1, at: 0.4, play: chime([554.4], 0, 0.05) },
      { stanza: 1, at: 1.1, play: chime([830.6], 0, 0.05) },
      { stanza: 2, at: 0.85, play: tanpura },
      { stanza: 2, at: 1.0, play: chime([554.4, 622.3, 698.5, 830.6, 932.3], 0.32, 0.035) },
      { stanza: 3, at: 0.7, play: chime([277.2, 415.3, 554.4, 830.6], 0.45, 0.04) }
    ]
  }
});
