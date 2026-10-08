/*
 * Scene for "Reach into the thoughts of friends" (Dru Mims): a walk by night
 * through an empty chapel, one small disillusionment to each couplet, and
 * out of the west doors onto the moor.
 *
 * I    (seven couplets, split by maxLines) "Reach into the thoughts of
 *      friends": votive lights under a marble roll of names by the door.
 *      The names blur and fade to blank stone, the votives gutter out one
 *      by one, and their sparks drift away into the dark.
 * ·    "Squeeze the teddy bear too tight": a worn bear left on a pew bursts,
 *      and its feathers drift down through the moonbeams.
 * ·    "Touch the stained glass with your cheek": close up against a lancet
 *      of leaded glass; the moonlight hardens, the colours chill and frost
 *      creeps over the glass and out across the stone floor.
 * ·    "Hold a candle to the night": one candle on an iron stand; the moon
 *      goes in, the dark closes round it and bends the flame over.
 * ·    "Tear the mask of peace from God": the veil over the altar, its dove
 *      and its gold, tears in two and falls; red light breaks through cracks
 *      in the wall and floor, embers rise and the whole floor trembles.
 * ·    "Pluck a rose in name of love": a single rose on the altar; its petals
 *      curl, darken and fall one by one.
 * ·    "Lean upon the western wind": turning back, the west doors swing open
 *      on an empty moor under running cloud; the wind pours down the nave,
 *      and the walk ends out on the moor, alone.
 *
 * The nave runs east along -z from the west doors at z = 0; the moon is in
 * the south (+x) and its light is traced back through the glass in the
 * stone shaders, so the coloured patches fall where the beams do.
 *
 * Columns: [unit, camZ, dark, (unused), wind, camX, camY, yaw, pitch,
 *           names, burst, frost, bend, hell, tear, wilt, doors, moon,
 *           yawP, pitchP]  (the last two aim portrait screens straight at
 *           the subject, clear of the centred verse)
 */
import { THREE, isSmall, makeRenderer, fitCamera, softSprite, terrain, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Plan (metres) ────────────────────────────────────────────────────────
var W = 6, WT = 1, WALL_H = 9, VAULT_RISE = 7.5, EAST = -37;
var WIN_Z = [-9, -16, -23];                              // the lancets, both walls
var LH = 1, LS = 5, LRISE = 1.6, SILL = 1.7, LTOP = LS + LRISE;
var LXC = (LRISE * LRISE - LH * LH) / (2 * LH), LR = LXC + LH;
var GX = W + WT / 2;                                     // the plane of the south glass
var PIERS = [-4.5, -12.5, -19.5, -26.5];
var STEP_Z = -29, STEP_H = 0.3, ALTAR_Z = -34, ALTAR_TOP = STEP_H + 1.05;
var DOOR = { h: 1.7, st: 4.2, rise: 2.0 };
var TABLET = new THREE.Vector3(W - 0.03, 2.2, -2.3);
var BEAR = new THREE.Vector3(-1.04, 0.01, -9.55);   // slumped against a pew end
var CANDLE = new THREE.Vector3(-0.9, 0, -20.4), FLAME_Y = 1.58;
var ROSE = new THREE.Vector3(0.2, ALTAR_TOP + 0.44, -33.75);
var MOON_D = new THREE.Vector3(-1, -0.72, -0.06).normalize();   // the way the moonlight travels
var PEW_ROWS = [];
for (var pz = -5; pz > -27; pz -= 1.5) PEW_ROWS.push(pz);

// ── Noise (JS, for the moor) ─────────────────────────────────────────────
function hash(x, y) { var h = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return h - Math.floor(h); }
function noise(x, y) {
  var ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy;
  fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
  return lerp(lerp(hash(ix, iy), hash(ix + 1, iy), fx), lerp(hash(ix, iy + 1), hash(ix + 1, iy + 1), fx), fy) * 2 - 1;
}
function fbm(x, y) { return noise(x, y) * 0.55 + noise(x * 2.1 + 5.2, y * 2.1 + 1.3) * 0.28 + noise(x * 4.4 + 9.1, y * 4.4 + 7.7) * 0.14; }

// The moor beyond the west doors: flat at the threshold, rolling away to
// a long low ridge.
function moor(x, z) {
  var d = Math.max(0, z - 1.5);
  return -0.04 + smooth(0, 40, d) * (0.5 + fbm(x * 0.025, z * 0.025) * 2.5) +
         smooth(80, 330, d) * (16 + fbm(x * 0.006 + 3, 1.7) * 26) + fbm(x * 0.35, z * 0.35) * 0.12 * smooth(0, 4, d);
}

// ── Pointed arches ───────────────────────────────────────────────────────
// The right-hand side of a pointed arch of half-span h springing at `st`:
// from (h, st) up to the apex (0, st + rise).
function archRight(h, st, rise, n) {
  var xc = (rise * rise - h * h) / (2 * h), R = xc + h, a1 = Math.acos(xc / R), out = [];
  for (var i = 0; i <= n; i++) { var a = a1 * i / n; out.push([-xc + R * Math.cos(a), st + R * Math.sin(a)]); }
  return out;
}
// A whole pointed opening from `base` up, centred on cx.
function archOutline(h, st, rise, base, n, cx) {
  var right = archRight(h, st, rise, n), pts = [[cx - h, base], [cx + h, base]];
  right.forEach(function (p) { pts.push([cx + p[0], p[1]]); });
  for (var i = right.length - 2; i >= 0; i--) pts.push([cx - right[i][0], right[i][1]]);
  return pts;
}
function v2(pts) { return pts.map(function (p) { return new THREE.Vector2(p[0], p[1]); }); }

// ── GLSL shared by the stone, glass, motes and sky ───────────────────────
function g(v) { return v.toFixed(4); }
var NOISE_GLSL =
  'float piHash(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }\n' +
  'float piNoise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);\n' +
  ' return mix(mix(piHash(i), piHash(i + vec2(1.0, 0.0)), f.x), mix(piHash(i + vec2(0.0, 1.0)), piHash(i + vec2(1.0, 1.0)), f.x), f.y); }\n' +
  'float piFbm(vec2 p){ return piNoise(p) * 0.5 + piNoise(p * 2.03 + 1.7) * 0.27 + piNoise(p * 4.07 + 3.1) * 0.14 + piNoise(p * 8.3 + 5.3) * 0.09; }\n';
// Window-local (s across from the centre line, t up from the sill): inside the lancet?
var MOON_GLSL = 'uniform vec3 uMoonD;\n' +
  'float piLancet(float s, float t){ if (t < 0.0 || abs(s) > ' + g(LH) + ') return 0.0; if (t < ' + g(LS) + ') return 1.0;\n' +
  ' return step(length(vec2(abs(s) + ' + g(LXC) + ', t - ' + g(LS) + ')), ' + g(LR) + '); }\n' +
  // Follow a point back up the moonlight to the south glass: where it crosses, and whether that is glass.
  'vec2 piMoonUV(vec3 p, out float inside){ inside = 0.0; if (p.x > ' + g(GX) + ') return vec2(0.0);\n' +
  ' vec3 q = p - uMoonD * ((' + g(GX) + ' - p.x) / -uMoonD.x); float t = q.y - ' + g(SILL) + ';\n' +
  WIN_Z.map(function (z) {
    return ' { float s = q.z - (' + g(z) + '); float k = piLancet(s, t); if (k > 0.0) { inside = k; return vec2(s / ' + g(2 * LH) + ' + 0.5, t / ' + g(LTOP) + '); } }\n';
  }).join('') +
  ' return vec2(0.0); }\n';

// Stone, wood and cloth: add the coloured light that falls through the
// south glass, and frost that creeps out from the window's foot.
var MOONLIT_FS =
  '{ float pIn; vec2 wuv = piMoonUV(vPiW, pIn);\n' +
  ' vec3 N = normalize(vPiN) * faceDirection;\n' +
  ' float nl = max(dot(N, -uMoonD), 0.0);\n' +
  ' vec3 glassC = pIn > 0.0 ? texture2D(uGlass, wuv).rgb : vec3(0.0);\n' +
  ' float glum = dot(glassC, vec3(0.3, 0.5, 0.2));\n' +
  ' glassC = mix(glassC, vec3(glum) * vec3(0.6, 0.78, 1.15), uFrost * 0.55);\n' +
  ' outgoingLight += diffuseColor.rgb * glassC * nl * uMoon * uMoonCol;\n' +
  ' float up = smoothstep(0.55, 0.9, N.y);\n' +
  ' float dw = length(vec2(vPiW.x - 6.0, (vPiW.z + 16.0) * 0.55));\n' +
  ' float rime = up * smoothstep(0.0, 1.6, uFrost * 6.5 - dw + (piFbm(vPiW.xz * 1.7) - 0.5) * 4.0) * smoothstep(0.0, 0.15, uFrost);\n' +
  ' vec2 sc = floor(vPiW.xz * 55.0);\n' +
  ' float spark = step(0.996, fract(sin(dot(sc, vec2(12.9898, 78.233))) * 43758.5453)) * rime * (0.5 + 0.5 * pIn);\n' +
  ' outgoingLight = mix(outgoingLight, vec3(0.2, 0.25, 0.34) * (0.12 + glum * pIn * nl * uMoon * 1.2), rime * 0.55) + spark * vec3(0.5, 0.6, 0.8); }\n';

function moonlit(mat, U) {
  mat.onBeforeCompile = function (sh) {
    ['uMoonD', 'uMoon', 'uMoonCol', 'uGlass', 'uFrost'].forEach(function (k) { sh.uniforms[k] = U[k]; });
    sh.vertexShader = 'varying vec3 vPiW;\nvarying vec3 vPiN;\n' + sh.vertexShader.replace('#include <project_vertex>',
      '#include <project_vertex>\n#ifdef USE_INSTANCING\n mat4 piM = modelMatrix * instanceMatrix;\n#else\n mat4 piM = modelMatrix;\n#endif\n' +
      ' vPiW = (piM * vec4(transformed, 1.0)).xyz; vPiN = normalize(mat3(piM) * objectNormal);');
    sh.fragmentShader = 'varying vec3 vPiW;\nvarying vec3 vPiN;\nuniform float uMoon; uniform vec3 uMoonCol; uniform sampler2D uGlass; uniform float uFrost;\n' +
      NOISE_GLSL + MOON_GLSL + sh.fragmentShader.replace('#include <opaque_fragment>', MOONLIT_FS + '#include <opaque_fragment>');
  };
  return mat;
}

// ── Canvas textures ──────────────────────────────────────────────────────
function canvasTex(w, h, paint, repeat) {
  var c = document.createElement('canvas');
  c.width = w; c.height = h;
  paint(c.getContext('2d'), w, h);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  if (repeat) t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}
function speckle(x, w, h, r, n, a) {
  for (var i = 0; i < n; i++) {
    x.fillStyle = (r() < 0.5 ? 'rgba(0,0,0,' : 'rgba(255,255,255,') + (r() * a).toFixed(3) + ')';
    x.fillRect(r() * w, r() * h, 1 + r() * 2, 1 + r() * 2);
  }
}

// Ashlar: courses of dressed blocks, 4 m to the tile (UVs are in metres).
function ashlarTexture(r) {
  var t = canvasTex(512, 512, function (x, w, h) {
    x.fillStyle = '#3c3833';
    x.fillRect(0, 0, w, h);
    x.fillStyle = '#5a554d';
    x.fillRect(0, 0, w, h);
    for (var j = 0; j < 7; j++) {
      var ch = 512 / 7, top = Math.round(j * ch);
      for (var cx = -r() * 120; cx < w;) {
        var bw = 100 + r() * 110, l = 122 + r() * 30, warm = r() * 8;
        x.fillStyle = 'rgb(' + Math.round(l + warm) + ',' + Math.round(l + warm * 0.5) + ',' + Math.round(l - 6) + ')';
        x.fillRect(cx + 1.5, top + 1.5, bw - 3, ch - 3);
        var gr = x.createLinearGradient(0, top, 0, top + ch);
        gr.addColorStop(0, 'rgba(255,255,255,0.05)'); gr.addColorStop(1, 'rgba(0,0,0,0.1)');
        x.fillStyle = gr;
        x.fillRect(cx + 1.5, top + 1.5, bw - 3, ch - 3);
        for (var k = 0; k < 3; k++) {
          var bx = cx + r() * bw, by = top + r() * ch, bg = x.createRadialGradient(bx, by, 0, bx, by, 14 + r() * 26);
          bg.addColorStop(0, 'rgba(0,0,0,' + (r() * 0.1).toFixed(3) + ')'); bg.addColorStop(1, 'rgba(0,0,0,0)');
          x.fillStyle = bg; x.fillRect(cx, top, bw, ch);
        }
        cx += bw;
      }
    }
    speckle(x, w, h, r, 9000, 0.1);
  }, true);
  t.repeat.set(0.25, 0.25);
  return t;
}

// Worn floor slabs, also 4 m to the tile.
function slabTexture(r) {
  return canvasTex(512, 512, function (x, w, h) {
    x.fillStyle = '#26231f';
    x.fillRect(0, 0, w, h);
    for (var j = 0; j < 4; j++) {
      for (var i = 0; i < 3; i++) {
        var l = 92 + r() * 30, ox = (j % 2) * 85;
        x.fillStyle = 'rgb(' + Math.round(l) + ',' + Math.round(l * 0.97) + ',' + Math.round(l * 0.92) + ')';
        x.fillRect(((i * 171 + ox) % 512) + 2, j * 128 + 2, 167, 124);
        if (((i * 171 + ox) % 512) + 171 > 512) x.fillRect(0, j * 128 + 2, ((i * 171 + ox) % 512) + 169 - 512, 124);
      }
    }
    for (var k = 0; k < 40; k++) {
      var gx = r() * w, gy = r() * h, gr = x.createRadialGradient(gx, gy, 0, gx, gy, 30 + r() * 50);
      gr.addColorStop(0, 'rgba(0,0,0,0.12)'); gr.addColorStop(1, 'rgba(0,0,0,0)');
      x.fillStyle = gr;
      x.fillRect(0, 0, w, h);
    }
    speckle(x, w, h, r, 9000, 0.1);
  }, true);
}

// The leaded glass of a lancet, 2 m by 6.6 m: diamond quarries in cobalt,
// a border of ruby and amber, three medallions and a trefoil in the head.
function glassTexture(r) {
  var Wp = 320, Hp = 1056;
  return canvasTex(Wp * 2, Hp * 2, function (x) {
    x.scale(2, 2);
    x.fillStyle = '#060507';
    x.fillRect(0, 0, Wp, Hp);
    function jit(hex, a) { var c = new THREE.Color(hex); c.offsetHSL((r() - 0.5) * 0.03, 0, (r() - 0.5) * a); return '#' + c.getHexString(); }
    var q = 46;
    for (var j = -2; j < Hp / q * 2 + 2; j++) {
      for (var i = -1; i < Wp / q + 2; i++) {
        var cx = i * q + (j & 1 ? q / 2 : 0), cy = j * q / 2;
        x.beginPath();
        x.moveTo(cx, cy - q / 2 + 2.5); x.lineTo(cx + q / 2 - 2.5, cy); x.lineTo(cx, cy + q / 2 - 2.5); x.lineTo(cx - q / 2 + 2.5, cy);
        x.closePath();
        var p = r();
        x.fillStyle = p < 0.07 ? jit('#8a1428', 0.08) : p < 0.11 ? jit('#c88a1a', 0.08) : jit(p < 0.55 ? '#1a3a9e' : '#2552b0', 0.12);
        x.fill();
      }
    }
    // Border: 24 px strips down both sides and along the foot.
    x.fillStyle = '#060507';
    x.fillRect(0, 0, 30, Hp); x.fillRect(Wp - 30, 0, 30, Hp); x.fillRect(0, Hp - 30, Wp, 30);
    for (var y = 0; y < Hp; y += 42) {
      [3, Wp - 27].forEach(function (bx, k) {
        x.fillStyle = jit(((y / 42 + k) & 1) ? '#9c1026' : '#d2961c', 0.1);
        x.fillRect(bx, y + 3, 24, 37);
      });
    }
    for (var bx2 = 33; bx2 < Wp - 30; bx2 += 42) { x.fillStyle = jit('#2e7a3e', 0.1); x.fillRect(bx2, Hp - 27, 37, 24); }
    // Medallions.
    function medallion(cx, cy, R, field, petal, heart) {
      x.fillStyle = '#060507';
      x.beginPath(); x.arc(cx, cy, R + 4, 0, Math.PI * 2); x.fill();
      for (var s = 0; s < 12; s++) {
        x.fillStyle = jit('#a3122a', 0.12);
        x.beginPath(); x.arc(cx, cy, R, s * Math.PI / 6 + 0.03, (s + 1) * Math.PI / 6 - 0.03); x.arc(cx, cy, R - 15, (s + 1) * Math.PI / 6 - 0.04, s * Math.PI / 6 + 0.04, true); x.closePath(); x.fill();
      }
      x.fillStyle = jit(field, 0.1);
      x.beginPath(); x.arc(cx, cy, R - 19, 0, Math.PI * 2); x.fill();
      for (var k = 0; k < 8; k++) {
        var a = k * Math.PI / 4;
        x.save(); x.translate(cx + Math.cos(a) * (R - 52), cy + Math.sin(a) * (R - 52)); x.rotate(a);
        x.fillStyle = '#060507'; x.beginPath(); x.ellipse(0, 0, 30, 16, 0, 0, Math.PI * 2); x.fill();
        x.fillStyle = jit(k & 1 ? petal : heart, 0.12); x.beginPath(); x.ellipse(0, 0, 27, 13, 0, 0, Math.PI * 2); x.fill();
        x.restore();
      }
      x.fillStyle = '#060507'; x.beginPath(); x.arc(cx, cy, 22, 0, Math.PI * 2); x.fill();
      x.fillStyle = jit('#f2d68a', 0.06); x.beginPath(); x.arc(cx, cy, 18, 0, Math.PI * 2); x.fill();
    }
    medallion(160, 405, 112, '#1c6a4a', '#d8a020', '#7a2a8a');
    medallion(160, 660, 112, '#5a2380', '#2e8a5a', '#d06a1a');
    medallion(160, 915, 104, '#a8661a', '#1e48a8', '#a3122a');
    // The head: a trefoil.
    [[160, 120, 34], [126, 178, 34], [194, 178, 34]].forEach(function (c, k) {
      x.fillStyle = '#060507'; x.beginPath(); x.arc(c[0], c[1], c[2] + 4, 0, Math.PI * 2); x.fill();
      x.fillStyle = jit(['#d8a020', '#a3122a', '#2e8a5a'][k], 0.1); x.beginPath(); x.arc(c[0], c[1], c[2], 0, Math.PI * 2); x.fill();
    });
    x.fillStyle = '#060507'; x.beginPath(); x.arc(160, 158, 16, 0, Math.PI * 2); x.fill();
    x.fillStyle = '#f4e4b0'; x.beginPath(); x.arc(160, 158, 12, 0, Math.PI * 2); x.fill();
    // Streaks and seeds in the glass; overlay leaves the lead black.
    x.globalCompositeOperation = 'overlay';
    for (var s = 0; s < 500; s++) {
      x.strokeStyle = r() < 0.5 ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.14)';
      x.lineWidth = 1 + r() * 4;
      var sx = r() * Wp, sy = r() * Hp;
      x.beginPath(); x.moveTo(sx, sy); x.lineTo(sx + (r() - 0.5) * 6, sy + 20 + r() * 70); x.stroke();
    }
    x.globalCompositeOperation = 'source-over';
  });
}

// A marble roll of names (a blank tablet, and the names on their own so
// they can blur and fade).
var NAMES = ['ELEANOR VANE', 'THOMAS ASHBY', 'MARGARET HOLT', 'JAMES ORWELL', 'CLARA WEST', 'HENRY MARCH',
             'ALICE FENWICK', 'ROBERT HALE', 'EDITH CROWE', 'SAMUEL REED', 'LUCY BRAND', 'ARTHUR GREY'];
function marbleTexture(r) {
  return canvasTex(512, 352, function (x, w, h) {
    x.fillStyle = '#cfcac0';
    x.fillRect(0, 0, w, h);
    for (var i = 0; i < 26; i++) {
      x.strokeStyle = 'rgba(90,86,80,' + (0.05 + r() * 0.12).toFixed(3) + ')';
      x.lineWidth = 0.6 + r() * 2.2;
      x.beginPath();
      var sx = r() * w, sy = r() * h;
      x.moveTo(sx, sy);
      x.bezierCurveTo(sx + (r() - 0.3) * 260, sy + (r() - 0.5) * 120, sx + (r() - 0.3) * 260, sy + (r() - 0.5) * 160, sx + (r() - 0.2) * 420, sy + (r() - 0.5) * 200);
      x.stroke();
    }
    speckle(x, w, h, r, 3000, 0.08);
    x.strokeStyle = 'rgba(70,64,56,0.55)'; x.lineWidth = 3; x.strokeRect(16, 16, w - 32, h - 32);
    x.strokeStyle = 'rgba(255,255,255,0.5)'; x.lineWidth = 1.5; x.strokeRect(19, 19, w - 38, h - 38);
  });
}
function namesTexture(blur) {
  return canvasTex(512, 352, function (x, w) {
    if (blur) x.filter = 'blur(' + blur + 'px)';
    x.textAlign = 'center';
    x.textBaseline = 'middle';
    function engrave(text, px, py, size) {
      x.font = size + 'px Georgia, "Times New Roman", serif';
      x.fillStyle = 'rgba(255,255,255,0.55)'; x.fillText(text, px + 1, py + 1.5);
      x.fillStyle = 'rgba(52,44,30,0.95)'; x.fillText(text, px, py);
    }
    engrave('R E M E M B E R E D', w / 2, 52, 26);
    x.fillStyle = 'rgba(52,44,30,0.6)'; x.fillRect(w / 2 - 70, 74, 140, 2);
    NAMES.forEach(function (n, i) { engrave(n, i < 6 ? 140 : 372, 108 + (i % 6) * 38, 21); });
  });
}

function featherTexture() {
  return canvasTex(64, 128, function (x) {
    x.lineCap = 'round';
    for (var i = 0; i < 46; i++) {
      var t = i / 46, y = 116 - t * 104, half = Math.sin(Math.min(t * 1.25, 1) * Math.PI * 0.92) * 21 + 4;
      x.strokeStyle = 'rgba(242,238,230,' + (0.55 + t * 0.4).toFixed(2) + ')';
      x.lineWidth = 1.7;
      [-1, 1].forEach(function (s) { x.beginPath(); x.moveTo(32, y); x.quadraticCurveTo(32 + s * half * 0.6, y - 4, 32 + s * half, y - 10 - t * 5); x.stroke(); });
    }
    x.strokeStyle = 'rgba(255,255,255,1)'; x.lineWidth = 2.2;
    x.beginPath(); x.moveTo(32, 126); x.lineTo(32, 8); x.stroke();
  });
}

function petalTexture() {
  return canvasTex(64, 64, function (x) {
    var gr = x.createLinearGradient(0, 64, 0, 0);
    gr.addColorStop(0, '#3a0610'); gr.addColorStop(0.35, '#8a0d22'); gr.addColorStop(0.85, '#c42438'); gr.addColorStop(1, '#9c1428');
    x.fillStyle = gr;
    x.beginPath();
    x.moveTo(32, 64); x.bezierCurveTo(6, 54, -2, 22, 9, 7); x.quadraticCurveTo(21, -2, 32, 5); x.quadraticCurveTo(43, -2, 55, 7); x.bezierCurveTo(66, 22, 58, 54, 32, 64);
    x.fill();
  });
}

// The veil over the altar: linen, a gold border and sunburst, a white dove.
function veilTexture(r) {
  return canvasTex(256, 330, function (x, w, h) {
    x.fillStyle = '#d8d0bc';
    x.fillRect(0, 0, w, h);
    for (var i = 0; i < h; i += 2) { x.fillStyle = 'rgba(0,0,0,' + (0.02 + r() * 0.03).toFixed(3) + ')'; x.fillRect(0, i, w, 1); }
    var cx = w / 2, cy = 150;
    for (var k = 0; k < 32; k++) {
      var a = k * Math.PI / 16, l = k & 1 ? 70 : 104;
      x.fillStyle = 'rgba(176,132,48,0.55)';
      x.beginPath(); x.moveTo(cx + Math.cos(a - 0.05) * 30, cy + Math.sin(a - 0.05) * 30);
      x.lineTo(cx + Math.cos(a) * l, cy + Math.sin(a) * l); x.lineTo(cx + Math.cos(a + 0.05) * 30, cy + Math.sin(a + 0.05) * 30); x.fill();
    }
    var gl = x.createRadialGradient(cx, cy, 0, cx, cy, 46);
    gl.addColorStop(0, 'rgba(240,206,120,0.9)'); gl.addColorStop(1, 'rgba(240,206,120,0)');
    x.fillStyle = gl; x.beginPath(); x.arc(cx, cy, 46, 0, Math.PI * 2); x.fill();
    // The dove, wings raised.
    x.fillStyle = '#fbf8f0';
    x.beginPath(); x.ellipse(cx, cy + 6, 26, 10, -0.15, 0, Math.PI * 2); x.fill();
    x.beginPath(); x.arc(cx + 24, cy - 2, 7, 0, Math.PI * 2); x.fill();
    x.beginPath(); x.moveTo(cx + 30, cy - 3); x.lineTo(cx + 38, cy - 1); x.lineTo(cx + 30, cy + 1); x.fill();
    x.beginPath(); x.moveTo(cx - 22, cy + 8); x.lineTo(cx - 44, cy + 2); x.lineTo(cx - 40, cy + 14); x.fill();
    [-1, 1].forEach(function (s) {
      x.beginPath(); x.moveTo(cx - 6, cy + 2); x.quadraticCurveTo(cx - 10 + s * 6, cy - 40, cx - 34 + s * 14, cy - 58);
      x.quadraticCurveTo(cx + 2 + s * 6, cy - 34, cx + 10, cy + 2); x.fill();
    });
    x.strokeStyle = '#a8823a'; x.lineWidth = 4; x.strokeRect(10, 10, w - 20, h - 20);
    x.lineWidth = 1.5; x.strokeRect(17, 17, w - 34, h - 34);
    for (var f = 12; f < w - 10; f += 6) { x.fillStyle = '#a8823a'; x.fillRect(f, h - 9, 2, 9); }
  });
}
// Complementary alpha masks for the two halves, split along a ragged tear.
function tearMasks(r) {
  var jag = [];
  for (var y = 0; y <= 330; y += 10) jag.push(128 + (r() - 0.5) * 18);
  function mask(left) {
    return canvasTex(256, 330, function (x, w, h) {
      x.fillStyle = '#000'; x.fillRect(0, 0, w, h);
      x.fillStyle = '#fff'; x.beginPath();
      x.moveTo(left ? 0 : w, 0);
      jag.forEach(function (jx, k) { x.lineTo(jx + (left ? 3 : -3), k * 10); });
      x.lineTo(left ? 0 : w, h); x.closePath(); x.fill();
    });
  }
  return [mask(true), mask(false)];
}

function plankTexture(r) {
  return canvasTex(128, 448, function (x, w, h) {
    x.fillStyle = '#120c08'; x.fillRect(0, 0, w, h);
    for (var i = 0; i < 5; i++) {
      var l = 52 + r() * 16;
      x.fillStyle = 'rgb(' + Math.round(l) + ',' + Math.round(l * 0.72) + ',' + Math.round(l * 0.5) + ')';
      x.fillRect(i * 25.6 + 1, 0, 23.6, h);
      for (var k = 0; k < 14; k++) {
        x.strokeStyle = 'rgba(0,0,0,' + (0.1 + r() * 0.15).toFixed(2) + ')'; x.lineWidth = 1;
        var gx = i * 25.6 + 2 + r() * 21; x.beginPath(); x.moveTo(gx, 0); x.lineTo(gx + (r() - 0.5) * 6, h); x.stroke();
      }
    }
    [70, 230, 380].forEach(function (y) {
      x.fillStyle = '#16120f'; x.fillRect(0, y, w, 12);
      x.fillStyle = '#3a342e'; for (var k = 6; k < w; k += 18) { x.beginPath(); x.arc(k, y + 6, 2.2, 0, 7); x.fill(); }
    });
  });
}

// Glowing cracks for the hell beneath: branching lines on black (additive).
function crackTexture(r) {
  return canvasTex(512, 512, function (x, w, h) {
    x.fillStyle = '#000'; x.fillRect(0, 0, w, h);
    x.lineCap = 'round';
    function crack(px, py, a, len, wd, depth) {
      for (var s = 0; s < len; s++) {
        var nx = px + Math.cos(a) * 9, ny = py + Math.sin(a) * 9;
        x.lineWidth = wd * (1 - s / len * 0.6);
        x.beginPath(); x.moveTo(px, py); x.lineTo(nx, ny); x.stroke();
        px = nx; py = ny; a += (r() - 0.5) * 0.9;
        if (depth < 3 && r() < 0.12) crack(px, py, a + (r() < 0.5 ? 1 : -1) * (0.6 + r() * 0.6), Math.floor(len * 0.5), wd * 0.6, depth + 1);
      }
    }
    var pass = [['rgba(255,60,20,0.35)', 16, 18], ['rgba(255,110,40,0.9)', 5, 0], ['rgba(255,230,170,1)', 1.6, 0]];
    var seeds = [];
    for (var i = 0; i < 7; i++) seeds.push([256 + (r() - 0.5) * 200, 256 + (r() - 0.5) * 200, r() * 6.28, 18 + Math.floor(r() * 18)]);
    pass.forEach(function (p) {
      var save = r;
      r = rng(77);           // the same cracks for every pass
      x.strokeStyle = p[0]; x.shadowColor = 'rgba(255,40,0,1)'; x.shadowBlur = p[2];
      seeds.forEach(function (s) { crack(s[0], s[1], s[2], s[3], p[1], 0); crack(s[0], s[1], s[2] + Math.PI, Math.floor(s[3] * 0.7), p[1], 0); });
      r = save;
    });
  });
}

// ── Sound cues ───────────────────────────────────────────────────────────
function noiseBuffer(ac, len, shape) {
  var b = ac.createBuffer(1, Math.floor(ac.sampleRate * len), ac.sampleRate), d = b.getChannelData(0);
  for (var i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * shape(i / d.length);
  return b;
}
// One distant toll of the bell, as the doors close behind you.
function toll(ac, out) {
  var t = ac.currentTime;
  [[0.5, 0.04, 7], [1, 0.06, 5], [1.19, 0.025, 4], [1.5, 0.02, 3.5], [2, 0.016, 3], [2.51, 0.01, 2.4]].forEach(function (p) {
    var o = ac.createOscillator(), gn = ac.createGain();
    o.type = 'sine';
    o.frequency.value = 174.6 * p[0];
    gn.gain.setValueAtTime(0.0001, t);
    gn.gain.exponentialRampToValueAtTime(p[1], t + 0.02);
    gn.gain.exponentialRampToValueAtTime(0.0001, t + p[2]);
    o.connect(gn); gn.connect(out);
    o.start(t); o.stop(t + p[2] + 0.1);
  });
}
// The seams going: a soft tearing puff.
function burstSound(ac, out) {
  var t = ac.currentTime, src = ac.createBufferSource(), bp = ac.createBiquadFilter(), gn = ac.createGain();
  src.buffer = noiseBuffer(ac, 0.9, function (k) { return Math.pow(1 - k, 3) * (0.6 + 0.4 * Math.random()); });
  bp.type = 'bandpass'; bp.frequency.setValueAtTime(1800, t); bp.frequency.exponentialRampToValueAtTime(500, t + 0.6); bp.Q.value = 0.8;
  gn.gain.value = 0.5;
  src.connect(bp); bp.connect(gn); gn.connect(out);
  src.start(t);
}
// Cold glass: a high, thin shimmer.
function glassSound(ac, out) {
  var t = ac.currentTime;
  [2093, 2637, 3136, 3951].forEach(function (f, i) {
    var o = ac.createOscillator(), gn = ac.createGain(), s = t + i * 0.09;
    o.type = 'sine'; o.frequency.value = f;
    gn.gain.setValueAtTime(0.0001, s);
    gn.gain.exponentialRampToValueAtTime(0.022, s + 0.01);
    gn.gain.exponentialRampToValueAtTime(0.0001, s + 2.4);
    o.connect(gn); gn.connect(out); o.start(s); o.stop(s + 2.5);
  });
}
// The roar beneath: a long low rumble that swells and dies.
function rumble(ac, out) {
  var t = ac.currentTime, len = 4.5, src = ac.createBufferSource(), lp = ac.createBiquadFilter(), gn = ac.createGain();
  src.buffer = noiseBuffer(ac, len, function (k) { return Math.sin(Math.PI * Math.min(k * 2.2, 1)) * Math.pow(1 - k, 0.8); });
  lp.type = 'lowpass'; lp.frequency.value = 140;
  gn.gain.value = 0.9;
  src.connect(lp); lp.connect(gn); gn.connect(out);
  src.start(t);
  var o = ac.createOscillator(), og = ac.createGain();
  o.type = 'sine'; o.frequency.setValueAtTime(46, t); o.frequency.linearRampToValueAtTime(38, t + len);
  og.gain.setValueAtTime(0.0001, t); og.gain.exponentialRampToValueAtTime(0.12, t + 1.2); og.gain.exponentialRampToValueAtTime(0.0001, t + len);
  o.connect(og); og.connect(out); o.start(t); o.stop(t + len);
}
// The west doors: a slow iron-hinged creak.
function creak(ac, out) {
  var t = ac.currentTime, o = ac.createOscillator(), bp = ac.createBiquadFilter(), gn = ac.createGain();
  o.type = 'sawtooth';
  o.frequency.setValueAtTime(70, t);
  for (var k = 1; k < 18; k++) o.frequency.linearRampToValueAtTime(70 + Math.random() * 60 + k * 2, t + k * 0.09);
  bp.type = 'bandpass'; bp.frequency.value = 900; bp.Q.value = 4;
  gn.gain.setValueAtTime(0.0001, t); gn.gain.exponentialRampToValueAtTime(0.06, t + 0.15); gn.gain.exponentialRampToValueAtTime(0.0001, t + 1.7);
  o.connect(bp); bp.connect(gn); gn.connect(out); o.start(t); o.stop(t + 1.8);
}

// ── Renderer ─────────────────────────────────────────────────────────────
function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(31), still = env.reduceMotion;
  var gl = makeRenderer(canvas, { clear: '#04050a' });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#080a12', 0.02);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.04, 2400);
  var clock = { value: 0 }, pxScale = { value: 800 };
  var tmpC = new THREE.Color(), tmpC2 = new THREE.Color(), tv = new THREE.Vector3(), tv2 = new THREE.Vector3();
  var m4 = new THREE.Matrix4(), q4 = new THREE.Quaternion(), s3 = new THREE.Vector3(), e3 = new THREE.Euler();

  var glassTex = glassTexture(r);
  var U = { uMoonD: { value: MOON_D }, uMoon: { value: 1 }, uMoonCol: { value: new THREE.Color('#cfdcff') },
            uGlass: { value: glassTex }, uFrost: { value: 0 } };

  // ── Light ───────────────────────────────────────────────────────────────
  var hemi = new THREE.HemisphereLight('#44527c', '#100d12', 1.0);
  var moonLight = new THREE.DirectionalLight('#9fb3ff', 0.35);
  moonLight.position.set(10, 7.2, -14);
  moonLight.target.position.set(0, 0, -14);
  var outside = new THREE.HemisphereLight('#8e9ab6', '#2a2620', 0);
  world.add(hemi, moonLight, moonLight.target, outside);

  // ── Materials ───────────────────────────────────────────────────────────
  var stoneMat = moonlit(new THREE.MeshStandardMaterial({ map: ashlarTexture(r), color: '#a29a8e', roughness: 0.93 }), U);
  var trimMat = moonlit(new THREE.MeshStandardMaterial({ color: '#7f786e', roughness: 0.9 }), U);
  var floorTex = slabTexture(r);
  var floorMat = moonlit(new THREE.MeshStandardMaterial({ map: floorTex, color: '#8a847a', roughness: 0.55 }), U);
  var woodMat = moonlit(new THREE.MeshStandardMaterial({ color: '#4a3424', roughness: 0.6 }), U);
  var iron = new THREE.MeshStandardMaterial({ color: '#1c1a18', roughness: 0.5, metalness: 0.6 });

  // ── The chapel: walls with lancets, a pointed vault, piers and ribs ─────
  function lancetPath(cx) { return new THREE.Path(v2(archOutline(LH, SILL + LS, LRISE, SILL, 10, cx))); }
  var side = new THREE.Shape(v2([[-1, -0.5], [38, -0.5], [38, WALL_H + 0.3], [-1, WALL_H + 0.3]]));
  WIN_Z.forEach(function (z) { side.holes.push(lancetPath(-z)); });
  var sideGeo = new THREE.ExtrudeGeometry(side, { depth: WT, bevelEnabled: false, curveSegments: 4 });
  [W, -W - WT].forEach(function (x) {
    var m = new THREE.Mesh(sideGeo, stoneMat);
    m.rotation.y = Math.PI / 2;
    m.position.x = x;
    world.add(m);
  });
  function gable(holes) {
    var right = archRight(W + 0.6, WALL_H, VAULT_RISE + 0.6, 12), pts = [[-W - WT, -0.5], [W + WT, -0.5], [W + WT, WALL_H]];
    right.forEach(function (p) { pts.push(p); });
    for (var i = right.length - 2; i >= 0; i--) pts.push([-right[i][0], right[i][1]]);
    pts.push([-W - WT, WALL_H]);
    var s = new THREE.Shape(v2(pts));
    (holes || []).forEach(function (h) { s.holes.push(h); });
    return new THREE.ExtrudeGeometry(s, { depth: WT, bevelEnabled: false, curveSegments: 4 });
  }
  var west = new THREE.Mesh(gable([new THREE.Path(v2(archOutline(DOOR.h, DOOR.st, DOOR.rise, -0.5, 14, 0)))]), stoneMat);
  var east = new THREE.Mesh(gable(), stoneMat);
  east.position.z = EAST - WT;
  world.add(west, east);

  // The vault: a pointed barrel from wall-head to wall-head.
  var vr = archRight(W, WALL_H, VAULT_RISE, 12), prof = [];
  for (var i = 0; i < vr.length; i++) prof.push([-vr[i][0], vr[i][1]]);
  for (i = vr.length - 2; i >= 0; i--) prof.push(vr[i]);
  (function () {
    var pos = [], uv = [], idx = [], NZ = 24, along = [0];
    for (var k = 1; k < prof.length; k++) along.push(along[k - 1] + Math.hypot(prof[k][0] - prof[k - 1][0], prof[k][1] - prof[k - 1][1]));
    for (var j = 0; j <= NZ; j++) {
      var z = 1 - (39 * j / NZ);
      prof.forEach(function (p, k) { pos.push(p[0], p[1], z); uv.push(along[k], z); });
    }
    var n = prof.length;
    for (j = 0; j < NZ; j++) for (var k2 = 0; k2 < n - 1; k2++) {
      var a = j * n + k2, b = a + 1, c = a + n, d = c + 1;
      idx.push(a, c, b, b, c, d);
    }
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    geo.setIndex(idx);
    geo.computeVertexNormals();
    var vault = new THREE.Mesh(geo, stoneMat.clone());
    vault.material.side = THREE.DoubleSide;
    vault.material.onBeforeCompile = stoneMat.onBeforeCompile;
    world.add(vault);
  })();

  // Piers on the walls and transverse ribs over the vault.
  var trims = [];
  PIERS.concat([STEP_Z + 0.2]).forEach(function (z) {
    [-1, 1].forEach(function (s) {
      trims.push(new THREE.BoxGeometry(0.42, WALL_H, 0.7).translate(s * (W - 0.2), WALL_H / 2, z));
      trims.push(new THREE.BoxGeometry(0.6, 0.3, 0.9).translate(s * (W - 0.28), WALL_H - 0.1, z));
      trims.push(new THREE.BoxGeometry(0.6, 0.35, 0.9).translate(s * (W - 0.28), 0.17, z));
    });
    var curve = new THREE.CatmullRomCurve3(prof.map(function (p) { return new THREE.Vector3(p[0] * 0.985, p[1] - 0.12, z); }));
    trims.push(new THREE.TubeGeometry(curve, 40, 0.17, 6, false));
  });
  trims.push(new THREE.CylinderGeometry(0.14, 0.14, 39, 6).rotateX(Math.PI / 2).translate(0, WALL_H + VAULT_RISE - 0.15, -18.5));
  // A string course under the windows.
  [-1, 1].forEach(function (s) { trims.push(new THREE.BoxGeometry(0.16, 0.14, 37).translate(s * (W - 0.08), SILL - 0.1, -18.5)); });
  trims.forEach(function (t) { var m = new THREE.Mesh(t, trimMat); world.add(m); });

  // Floor, and the chancel raised one step.
  var floorGeo = new THREE.PlaneGeometry(2 * W, 38.4).rotateX(-Math.PI / 2).translate(0, 0, -18.2);
  floorTex.repeat.set(2 * W / 4, 38.4 / 4);
  world.add(new THREE.Mesh(floorGeo, floorMat));
  var chancel = new THREE.Mesh(new THREE.BoxGeometry(2 * W, STEP_H, STEP_Z - EAST), trimMat);
  chancel.position.set(0, STEP_H / 2, (STEP_Z + EAST) / 2);
  var chancelTop = new THREE.Mesh(new THREE.PlaneGeometry(2 * W, STEP_Z - EAST).rotateX(-Math.PI / 2), floorMat);
  chancelTop.position.set(0, STEP_H + 0.003, (STEP_Z + EAST) / 2);
  world.add(chancel, chancelTop);

  // ── The glass: three moonlit lancets on the south wall, three dim ones north
  var lancetGeo = new THREE.ShapeGeometry(new THREE.Shape(v2(archOutline(LH, LS, LRISE, 0, 14, 0))), 4);
  var glassMat = function (light, tint) {
    return new THREE.ShaderMaterial({
      uniforms: { uMap: { value: glassTex }, uLight: { value: light }, uTint: { value: new THREE.Color(tint) }, uFrost: U.uFrost, uTime: clock },
      vertexShader: 'varying vec2 vUv; void main(){ vUv = vec2(uv.x / ' + g(2 * LH) + ' + 0.5, uv.y / ' + g(LTOP) + ');\n' +
        ' gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
      fragmentShader: 'uniform sampler2D uMap; uniform float uLight; uniform vec3 uTint; uniform float uFrost; varying vec2 vUv;\n' + NOISE_GLSL +
        'void main(){ vec3 c = texture2D(uMap, vUv).rgb;\n' +
        ' float lum = dot(c, vec3(0.3, 0.5, 0.2));\n' +
        ' c = mix(c, vec3(lum) * vec3(0.62, 0.8, 1.2), uFrost * 0.6);\n' +
        // Frost: feathery rime growing in from the edges and the foot (in metres).
        ' float edge = min(min(vUv.x, 1.0 - vUv.x) * ' + g(2 * LH) + ', vUv.y * ' + g(LTOP) + ');\n' +
        ' float fr = piFbm(vUv * vec2(10.0, 33.0)) - 0.5;\n' +
        ' float reach = uFrost * 1.0;\n' +
        ' float frost = smoothstep(reach, reach - 0.3, edge + fr * 0.45) * step(0.001, uFrost);\n' +
        ' float fern = pow(piNoise(vUv * vec2(70.0, 230.0) + fr * 3.0), 2.0) * 0.6 + 0.4;\n' +
        ' vec3 col = c * uTint * uLight;\n' +
        ' col = mix(col, vec3(0.62, 0.72, 0.88) * (0.35 + uLight * 0.24), frost * fern * 0.9);\n' +
        ' gl_FragColor = vec4(col, 1.0);\n #include <tonemapping_fragment>\n #include <colorspace_fragment>\n }'
    });
  };
  var southGlass = glassMat(2.4, '#ffffff'), northGlass = glassMat(0.08, '#7088ff');
  northGlass.uniforms.uFrost = { value: 0 };
  WIN_Z.forEach(function (z) {
    var sg = new THREE.Mesh(lancetGeo, southGlass);
    sg.rotation.y = -Math.PI / 2;
    sg.position.set(GX, SILL, z);
    var ng = new THREE.Mesh(lancetGeo, northGlass);
    ng.rotation.y = Math.PI / 2;
    ng.position.set(-GX, SILL, z);
    world.add(sg, ng);
  });

  // Shafts of moonlight from each south lancet down to the floor: the
  // lancet's outline swept along the light; faces seen square-on glow most.
  (function () {
    var out = archOutline(LH, LS, LRISE, 0, 10, 0), pos = [], al = [];
    WIN_Z.forEach(function (zc) {
      for (var k = 0; k < out.length; k++) {
        var a = out[k], b = out[(k + 1) % out.length];
        var A = new THREE.Vector3(W, SILL + a[1], zc + a[0]), B = new THREE.Vector3(W, SILL + b[1], zc + b[0]);
        var A2 = A.clone().addScaledVector(MOON_D, A.y / -MOON_D.y), B2 = B.clone().addScaledVector(MOON_D, B.y / -MOON_D.y);
        pos.push(A.x, A.y, A.z, B.x, B.y, B.z, A2.x, A2.y, A2.z, B.x, B.y, B.z, B2.x, B2.y, B2.z, A2.x, A2.y, A2.z);
        al.push(0, 0, 1, 0, 1, 1);
      }
    });
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('aAlong', new THREE.Float32BufferAttribute(al, 1));
    geo.computeVertexNormals();
    var mat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
      uniforms: { uMoon: U.uMoon, uTime: clock, uCol: { value: new THREE.Color('#7f95d8') } },
      vertexShader: 'attribute float aAlong; varying float vAl; varying vec3 vN; varying vec3 vV; varying vec3 vW;\n' +
        'void main(){ vAl = aAlong; vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; vec4 mv = viewMatrix * w;\n' +
        ' vN = normalize(normalMatrix * normal); vV = -mv.xyz; gl_Position = projectionMatrix * mv; }',
      fragmentShader: 'uniform float uMoon; uniform float uTime; uniform vec3 uCol; varying float vAl; varying vec3 vN; varying vec3 vV; varying vec3 vW;\n' + NOISE_GLSL +
        'void main(){ float face = abs(dot(normalize(vN), normalize(vV)));\n' +
        ' float dust = 0.65 + 0.35 * piFbm(vW.yz * 0.8 + vec2(uTime * 0.05, 0.0));\n' +
        ' float a = pow(face, 1.4) * (1.0 - vAl * 0.65) * dust * uMoon * 0.08;\n' +
        ' gl_FragColor = vec4(uCol * a, a);\n #include <colorspace_fragment>\n }'
    });
    var shafts = new THREE.Mesh(geo, mat);
    shafts.renderOrder = 2;
    world.add(shafts);
  })();

  // Dust in the air, which shows only where the moonlight falls.
  var DN = small ? 900 : 1800, dPos = new Float32Array(DN * 3), dSeed = new Float32Array(DN);
  for (i = 0; i < DN; i++) {
    dPos[i * 3] = (r() - 0.5) * 2 * W; dPos[i * 3 + 1] = r() * 8; dPos[i * 3 + 2] = -3 - r() * 24;
    dSeed[i] = r();
  }
  var dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute('position', new THREE.BufferAttribute(dPos, 3));
  dustGeo.setAttribute('aSeed', new THREE.BufferAttribute(dSeed, 1));
  var dustMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uMoonD: U.uMoonD, uMoon: U.uMoon, uTime: clock, uPx: pxScale, uWind: { value: 0 } },
    vertexShader: 'attribute float aSeed; uniform float uMoon; uniform float uTime; uniform float uPx; uniform float uWind; varying float vA;\n' + MOON_GLSL +
      'void main(){ vec3 p = position + vec3(sin(uTime * 0.11 + aSeed * 40.0), sin(uTime * 0.07 + aSeed * 17.0) * 0.6, cos(uTime * 0.09 + aSeed * 23.0)) * 0.35;\n' +
      ' p.z -= mod(uTime * uWind * 3.0 + aSeed * 24.0, 24.0) * step(0.01, uWind);\n' +
      ' float inside; piMoonUV(p, inside);\n' +
      ' vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_Position = projectionMatrix * mv;\n' +
      ' vA = inside * uMoon * (0.35 + 0.65 * fract(aSeed * 13.7)) * (0.6 + 0.4 * sin(uTime * 1.3 + aSeed * 60.0));\n' +
      ' gl_PointSize = max(1.0, 0.012 * uPx / -mv.z); }',
    fragmentShader: 'varying float vA; void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;\n' +
      ' float a = smoothstep(0.5, 0.0, d) * vA * 0.7; gl_FragColor = vec4(vec3(0.75, 0.82, 1.0) * a, a);\n #include <colorspace_fragment>\n }'
  });
  var dust = new THREE.Points(dustGeo, dustMat);
  dust.frustumCulled = false;
  world.add(dust);

  // ── Pews ────────────────────────────────────────────────────────────────
  (function () {
    var parts = [
      new THREE.BoxGeometry(3.8, 0.06, 0.44).translate(0, 0.45, 0),
      new THREE.BoxGeometry(3.8, 0.5, 0.05).translate(0, 0.78, 0.22),
      new THREE.BoxGeometry(3.8, 0.05, 0.08).translate(0, 1.04, 0.22),
      new THREE.BoxGeometry(3.6, 0.05, 0.14).translate(0, 0.12, -0.42),
      new THREE.BoxGeometry(0.07, 1.0, 0.56).translate(-1.9, 0.5, 0.0),
      new THREE.BoxGeometry(0.07, 1.0, 0.56).translate(1.9, 0.5, 0.0)
    ].map(function (gq) { return gq.toNonIndexed(); });
    var n = 0;
    parts.forEach(function (p) { n += p.attributes.position.count; });
    var pos = new Float32Array(n * 3), nor = new Float32Array(n * 3), o = 0;
    parts.forEach(function (p) { pos.set(p.attributes.position.array, o); nor.set(p.attributes.normal.array, o); o += p.attributes.position.array.length; p.dispose(); });
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    var pews = new THREE.InstancedMesh(geo, woodMat, PEW_ROWS.length * 2), c = 0;
    PEW_ROWS.forEach(function (z) {
      [-3.1, 3.1].forEach(function (x) { pews.setMatrixAt(c++, m4.makeTranslation(x, 0, z)); });
    });
    world.add(pews);
  })();
  // Where a falling feather comes to rest: a pew seat, or the floor.
  function restY(x, z) {
    var ax = Math.abs(x);
    if (ax > 1.2 && ax < 5.0) for (var k = 0; k < PEW_ROWS.length; k++) if (Math.abs(z - PEW_ROWS[k]) < 0.22) return 0.49;
    return 0.012;
  }

  // ── I: the roll of names and the votive lights ──────────────────────────
  var tabletGroup = new THREE.Group();
  tabletGroup.position.copy(TABLET);
  tabletGroup.rotation.y = -Math.PI / 2;
  var marble = new THREE.Mesh(new THREE.BoxGeometry(1.6, 1.1, 0.06), new THREE.MeshStandardMaterial({ map: marbleTexture(r), roughness: 0.4 }));
  var namesSharp = new THREE.Mesh(new THREE.PlaneGeometry(1.6, 1.1), new THREE.MeshStandardMaterial({ map: namesTexture(0), transparent: true, roughness: 0.5, depthWrite: false }));
  var namesBlur = new THREE.Mesh(new THREE.PlaneGeometry(1.6, 1.1), new THREE.MeshStandardMaterial({ map: namesTexture(5), transparent: true, roughness: 0.5, depthWrite: false, opacity: 0 }));
  namesSharp.position.z = namesBlur.position.z = 0.032;
  var corbel = new THREE.Mesh(new THREE.BoxGeometry(1.75, 0.08, 0.12), trimMat);
  corbel.position.set(0, -0.6, 0.03);
  tabletGroup.add(marble, namesSharp, namesBlur, corbel);
  world.add(tabletGroup);

  // The votive stand: three stepped iron shelves of tea-lights.
  var votives = [], flameTex = canvasTex(32, 64, function (x) {
    x.translate(16, 42); x.scale(1, 2.2);
    var gr = x.createRadialGradient(0, 0, 0, 0, 0, 13);
    gr.addColorStop(0, 'rgba(255,255,235,1)'); gr.addColorStop(0.3, 'rgba(255,214,120,0.95)'); gr.addColorStop(0.65, 'rgba(255,120,30,0.35)'); gr.addColorStop(1, 'rgba(255,80,10,0)');
    x.fillStyle = gr; x.beginPath(); x.arc(0, 0, 13, 0, Math.PI * 2); x.fill();
  });
  var warmTex = softSprite('rgba(255,200,130,1)', 'rgba(255,150,70,0)');
  var stand = new THREE.Group();
  stand.position.set(W - 0.45, 0, TABLET.z);
  stand.add(new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.95, 0.06).translate(0, 0.475, 0), iron));
  stand.add(new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.02, 0.4).translate(0, 0.01, 0), iron));
  var waxMat = new THREE.MeshStandardMaterial({ color: '#e8e0d0', roughness: 0.6, emissive: '#ff9a40', emissiveIntensity: 0.25 });
  for (var tier = 0; tier < 3; tier++) {
    var ty = 0.95 + tier * 0.12, tx = -0.12 + tier * 0.12;
    stand.add(new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.015, 0.9).translate(tx, ty, 0), iron));
    for (var vk = 0; vk < 3; vk++) {
      var vz = (vk - 1) * 0.28 + (tier === 1 ? 0.07 : 0), cup = new THREE.Mesh(new THREE.CylinderGeometry(0.032, 0.03, 0.035, 10), waxMat);
      cup.position.set(tx, ty + 0.025, vz);
      var fl = new THREE.Sprite(new THREE.SpriteMaterial({ map: flameTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
      fl.position.set(tx, ty + 0.075, vz);
      stand.add(cup, fl);
      votives.push({ flame: fl, at: 0.15 + r() * 0.75, phase: r() * 6.28 });
    }
  }
  votives.sort(function (a, b) { return b.at - a.at; });
  votives[votives.length - 1].at = -1;         // one is left burning
  var votiveGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: warmTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0.5 }));
  votiveGlow.position.set(0.05, 1.12, 0);
  votiveGlow.scale.setScalar(1.6);
  stand.add(votiveGlow);
  var votiveLight = new THREE.PointLight('#ffa95a', 3, 7, 1.3);
  votiveLight.position.set(-0.3, 1.25, 0);
  stand.add(votiveLight);
  world.add(stand);

  // Sparks of thought around the names, which drift off as the names go.
  var MN = 40, mPos = new Float32Array(MN * 3), mBase = [], mDir = [];
  for (i = 0; i < MN; i++) {
    mBase.push(new THREE.Vector3(W - 0.3 - r() * 0.9, 1.1 + r() * 1.9, TABLET.z + (r() - 0.5) * 1.8));
    mDir.push(new THREE.Vector3(-0.5 - r() * 1.5, 0.6 + r() * 1.6, (r() - 0.5) * 3));
  }
  var moteGeo = new THREE.BufferGeometry();
  moteGeo.setAttribute('position', new THREE.BufferAttribute(mPos, 3));
  var moteMat = new THREE.PointsMaterial({ map: warmTex, color: '#ffcf8a', size: 0.07, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  var motes = new THREE.Points(moteGeo, moteMat);
  motes.frustumCulled = false;
  world.add(motes);

  // ── II: the bear, and its feathers ──────────────────────────────────────
  var bearMat = new THREE.MeshStandardMaterial({ color: '#a07850', roughness: 1, emissive: '#1a120a' });
  var bearDark = new THREE.MeshStandardMaterial({ color: '#141010', roughness: 0.4 });
  var bearPale = new THREE.MeshStandardMaterial({ color: '#c8ae8a', roughness: 1, emissive: '#1a140e' });
  var bear = new THREE.Group(), bearBody = new THREE.Group(), bearHead = new THREE.Group();
  function blob(grp, rad, sx, sy, sz, x, y, z, mat) {
    var m = new THREE.Mesh(new THREE.SphereGeometry(rad, 16, 12), mat);
    m.scale.set(sx, sy, sz); m.position.set(x, y, z); grp.add(m); return m;
  }
  blob(bearBody, 0.13, 1, 1.15, 0.9, 0, 0.14, 0, bearMat);
  blob(bearBody, 0.08, 1, 1.1, 0.5, 0, 0.12, 0.08, bearPale);
  var armL = blob(bearBody, 0.045, 1, 1.7, 1, -0.135, 0.17, 0.03, bearMat), armR = blob(bearBody, 0.045, 1, 1.7, 1, 0.135, 0.17, 0.03, bearMat);
  armL.rotation.z = -0.4; armR.rotation.z = 0.4;
  blob(bearBody, 0.055, 1, 1, 1.5, -0.075, 0.04, 0.1, bearMat);
  blob(bearBody, 0.055, 1, 1, 1.5, 0.075, 0.04, 0.1, bearMat);
  bearHead.position.set(0, 0.3, 0);
  blob(bearHead, 0.1, 1, 0.95, 0.95, 0, 0.04, 0, bearMat);
  blob(bearHead, 0.036, 1, 1, 0.5, -0.075, 0.125, 0, bearMat);
  blob(bearHead, 0.036, 1, 1, 0.5, 0.075, 0.125, 0, bearMat);
  blob(bearHead, 0.042, 1, 0.8, 0.9, 0, 0.015, 0.075, bearPale);
  blob(bearHead, 0.012, 1, 1, 1, -0.036, 0.06, 0.088, bearDark);
  blob(bearHead, 0.012, 1, 1, 1, 0.036, 0.06, 0.088, bearDark);
  blob(bearHead, 0.014, 1.2, 0.8, 1, 0, 0.03, 0.115, bearDark);
  bearBody.add(bearHead);
  bear.add(bearBody);
  bear.position.copy(BEAR);
  bear.rotation.y = 0.95;
  world.add(bear);

  var FN = small ? 130 : 240, feathers = new THREE.InstancedMesh(new THREE.PlaneGeometry(0.06, 0.13),
    new THREE.MeshLambertMaterial({ map: featherTexture(), alphaTest: 0.35, side: THREE.DoubleSide, color: '#f4f1ea', emissive: '#5a5e68' }), FN);
  feathers.frustumCulled = false;
  world.add(feathers);
  var fData = [];
  for (i = 0; i < FN; i++) {
    var th = r() * Math.PI * 2, el = 0.25 + r() * 1.1;
    fData.push({ dir: new THREE.Vector3(Math.cos(th) * Math.cos(el), Math.sin(el), Math.sin(th) * Math.cos(el)), v: 1.2 + r() * 3.2,
                 fall: 0.22 + r() * 0.26, ph: r() * 6.28, spin: (r() - 0.5) * 4, sc: 0.6 + r() * 0.7, sway: 0.15 + r() * 0.35 });
  }
  var fPos = new THREE.Vector3(), fOrigin = new THREE.Vector3(BEAR.x, BEAR.y + 0.18, BEAR.z);

  // ── IV: the candle on its iron stand ────────────────────────────────────
  var candle = new THREE.Group();
  candle.position.copy(CANDLE);
  candle.add(new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.024, 1.18, 8).translate(0, 0.62, 0), iron));
  for (var leg = 0; leg < 3; leg++) {
    var lgeo = new THREE.BoxGeometry(0.03, 0.03, 0.34).translate(0, 0.03, 0.15);
    lgeo.rotateY(leg * Math.PI * 2 / 3);
    candle.add(new THREE.Mesh(lgeo, iron));
  }
  candle.add(new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.09, 0.03, 16).translate(0, 1.22, 0), iron));
  var candleWax = new THREE.MeshStandardMaterial({ color: '#efe6d2', roughness: 0.55, emissive: '#ff8a30', emissiveIntensity: 0.12 });
  candle.add(new THREE.Mesh(new THREE.CylinderGeometry(0.036, 0.04, 0.32, 14).translate(0, 1.395, 0), candleWax));
  [[0.025, 0.04, 0.03], [-0.03, 0.06, 0.02]].forEach(function (d) {
    candle.add(new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.01, d[1], 6).translate(d[0], 1.53 - d[1] / 2, d[2]), candleWax));
  });
  world.add(candle);
  var flameMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uTime: clock, uBend: { value: 0 }, uLife: { value: 1 } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform float uTime; uniform float uBend; uniform float uLife; varying vec2 vUv;\n' +
      // The quad is 0.4 m by 0.26 m; the flame is 0.2 m tall at full life.
      'void main(){ float y = vUv.y * 0.26 / (0.2 * max(uLife, 0.05));\n' +
      ' float sway = uBend * pow(clamp(y, 0.0, 1.0), 1.5) * 0.11 + (sin(uTime * 13.0 + y * 5.0) * 0.006 + sin(uTime * 7.3) * 0.005) * y * (1.0 + uBend);\n' +
      ' float x = (vUv.x - 0.5) * 0.4 - sway;\n' +
      ' float w = 0.024 * pow(max(sin(3.14159 * pow(clamp(y, 0.0, 1.0), 0.55)), 0.0), 1.1) * (1.0 - uBend * 0.25);\n' +
      ' float d = abs(x) / max(w, 0.002);\n' +
      ' float body = smoothstep(1.0, 0.25, d) * step(y, 1.0);\n' +
      ' float core = smoothstep(0.7, 0.0, d) * smoothstep(0.95, 0.3, y);\n' +
      ' vec3 col = mix(vec3(1.0, 0.36, 0.06), vec3(1.0, 0.88, 0.58), core * (1.0 - uBend * 0.4));\n' +
      ' col = mix(col, vec3(0.3, 0.45, 1.0), smoothstep(0.18, 0.0, y) * 0.6);\n' +
      ' float a = body * (0.7 + core * 0.6);\n' +
      ' gl_FragColor = vec4(col * a * 2.2, a);\n #include <tonemapping_fragment>\n #include <colorspace_fragment>\n }'
  });
  var flame = new THREE.Mesh(new THREE.PlaneGeometry(0.4, 0.26).translate(0, 0.13, 0), flameMat);
  flame.position.set(CANDLE.x, FLAME_Y - 0.03, CANDLE.z);
  flame.renderOrder = 4;
  var candleGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: warmTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0.6 }));
  candleGlow.position.set(CANDLE.x, FLAME_Y + 0.05, CANDLE.z);
  var candleHalo = new THREE.Sprite(new THREE.SpriteMaterial({ map: warmTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0.15 }));
  candleHalo.position.copy(candleGlow.position);
  candleHalo.scale.setScalar(3.2);
  var candleLight = new THREE.PointLight('#ffad5a', 4, 9, 1.4);
  candleLight.position.set(CANDLE.x + 0.05, FLAME_Y + 0.12, CANDLE.z + 0.1);
  world.add(flame, candleGlow, candleHalo, candleLight);

  // The dark that closes round the flame: soft black smoke (normal blend).
  var SN = 46, sPos = new Float32Array(SN * 3), sAttr = new Float32Array(SN * 2), sData = [];
  for (i = 0; i < SN; i++) sData.push({ a: r() * 6.28, d: 0.4 + r() * 1.3, y: (r() - 0.5) * 1.6, sp: 0.1 + r() * 0.3, sz: 0.5 + r() * 0.8 });
  var smokeGeo = new THREE.BufferGeometry();
  smokeGeo.setAttribute('position', new THREE.BufferAttribute(sPos, 3));
  smokeGeo.setAttribute('aS', new THREE.BufferAttribute(sAttr, 2));
  var smokeMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false,
    uniforms: { uPx: pxScale },
    vertexShader: 'attribute vec2 aS; uniform float uPx; varying float vA;\n' +
      'void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv; vA = aS.y; gl_PointSize = aS.x * uPx / -mv.z; }',
    fragmentShader: 'varying float vA; void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;\n' +
      ' gl_FragColor = vec4(0.0, 0.0, 0.0, pow(smoothstep(0.5, 0.0, d), 1.5) * vA); }'
  });
  var smoke = new THREE.Points(smokeGeo, smokeMat);
  smoke.frustumCulled = false;
  smoke.renderOrder = 1;
  world.add(smoke);

  // ── V: the altar, the veil, and what is under it ────────────────────────
  var altar = new THREE.Mesh(new THREE.BoxGeometry(2.8, 1.05, 1.0), trimMat);
  altar.position.set(0, STEP_H + 0.525, ALTAR_Z);
  var linenMat = moonlit(new THREE.MeshStandardMaterial({ color: '#d6d0c4', roughness: 0.9 }), U);
  var cloth = new THREE.Mesh(new THREE.BoxGeometry(2.95, 0.02, 1.08), linenMat);
  cloth.position.set(0, ALTAR_TOP + 0.01, ALTAR_Z);
  var frontal = new THREE.Mesh(new THREE.BoxGeometry(2.95, 0.3, 0.012), linenMat);
  frontal.position.set(0, ALTAR_TOP - 0.15, ALTAR_Z + 0.545);
  world.add(altar, cloth, frontal);
  // Two candles on the altar light the veil, until it tears.
  var brass = new THREE.MeshStandardMaterial({ color: '#8a6a32', roughness: 0.35, metalness: 0.8 }), altarFlames = [];
  [-1.3, 1.3].forEach(function (x) {
    var cs = new THREE.Group();
    cs.position.set(x, ALTAR_TOP + 0.02, ALTAR_Z - 0.15);
    cs.add(new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.09, 0.05, 14).translate(0, 0.025, 0), brass));
    cs.add(new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.022, 0.32, 10).translate(0, 0.21, 0), brass));
    cs.add(new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.035, 0.03, 14).translate(0, 0.38, 0), brass));
    cs.add(new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.03, 0.42, 12).translate(0, 0.6, 0), new THREE.MeshStandardMaterial({ color: '#ece4d2', roughness: 0.6 })));
    var fl = new THREE.Sprite(new THREE.SpriteMaterial({ map: flameTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
    fl.position.set(0, 0.86, 0);
    var gw = new THREE.Sprite(new THREE.SpriteMaterial({ map: warmTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0.35 }));
    gw.position.set(0, 0.86, 0);
    gw.scale.setScalar(0.7);
    cs.add(fl, gw);
    world.add(cs);
    altarFlames.push({ flame: fl, glow: gw, stick: cs, s: Math.sign(x) });
  });
  var altarLight = new THREE.PointLight('#ffb466', 5, 14, 1.2);
  altarLight.position.set(0, ALTAR_TOP + 1.2, ALTAR_Z + 0.4);
  world.add(altarLight);

  var VEIL = { w: 4.2, h: 5.4, top: 8.0, z: EAST + 0.12 };
  var veilTex = veilTexture(r), masks = tearMasks(r), halves = [];
  var rod = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, VEIL.w + 0.5, 8).rotateZ(Math.PI / 2), iron);
  rod.position.set(0, VEIL.top + 0.03, VEIL.z + 0.02);
  world.add(rod);
  [-1, 1].forEach(function (s, k) {
    // Each half runs a little past the middle, so the ragged tear in the
    // alpha masks has cloth on both sides of it.
    var hw = VEIL.w / 2 + 0.3, geo = new THREE.PlaneGeometry(hw, VEIL.h, 9, 12).translate(-s * hw / 2, -VEIL.h / 2, 0);
    var uv = geo.attributes.uv, vp = geo.attributes.position;
    for (var j = 0; j < vp.count; j++) {
      var gx = s * VEIL.w / 2 + vp.getX(j), hang = -vp.getY(j) / VEIL.h;
      uv.setX(j, gx / VEIL.w + 0.5);
      // Soft vertical folds, deeper towards the hem.
      vp.setZ(j, (Math.sin(gx * 6.2 + 0.7) * 0.05 + Math.sin(gx * 2.3) * 0.03) * (0.35 + hang));
    }
    geo.computeVertexNormals();
    var mat = new THREE.MeshStandardMaterial({ map: veilTex, alphaMap: masks[k], alphaTest: 0.5, side: THREE.DoubleSide, roughness: 0.95,
                                                emissive: '#ffffff', emissiveMap: veilTex, emissiveIntensity: 0.07, transparent: true });
    var pivot = new THREE.Group();
    pivot.position.set(s * VEIL.w / 2, VEIL.top, VEIL.z);
    pivot.add(new THREE.Mesh(geo, mat));
    world.add(pivot);
    halves.push({ pivot: pivot, mat: mat, s: s });
  });

  var crackTex = crackTexture(r);
  function crackPlane(w, h, mat) { var m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat); world.add(m); return m; }
  var crackWallMat = new THREE.MeshBasicMaterial({ map: crackTex, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, opacity: 0, fog: false });
  var crackFloorMat = crackWallMat.clone();
  var wallCrack = crackPlane(7, 7, crackWallMat);
  wallCrack.position.set(0, 4.6, EAST + 0.02);
  var floorCrack = crackPlane(10, 7.6, crackFloorMat);
  floorCrack.rotation.x = -Math.PI / 2;
  floorCrack.position.set(0, STEP_H + 0.012, (STEP_Z + EAST) / 2);
  var hellLight = new THREE.PointLight('#ff3410', 0, 22, 1.2);
  hellLight.position.set(0, STEP_H + 0.5, -32.5);
  var hellWall = new THREE.PointLight('#ff2a08', 0, 16, 1.3);
  hellWall.position.set(0, 4.5, EAST + 1.6);
  world.add(hellLight, hellWall);

  // Embers rising out of the cracks.
  var EN = small ? 120 : 220, eAttr = new Float32Array(EN * 4), ePos = new Float32Array(EN * 3);
  for (i = 0; i < EN; i++) {
    ePos[i * 3] = (r() - 0.5) * 9; ePos[i * 3 + 1] = STEP_H; ePos[i * 3 + 2] = STEP_Z - 0.3 - r() * 7.4;
    eAttr[i * 4] = 0.4 + r() * 1.1; eAttr[i * 4 + 1] = r(); eAttr[i * 4 + 2] = r() * 6.28; eAttr[i * 4 + 3] = 0.5 + r();
  }
  var emberGeo = new THREE.BufferGeometry();
  emberGeo.setAttribute('position', new THREE.BufferAttribute(ePos, 3));
  emberGeo.setAttribute('aE', new THREE.BufferAttribute(eAttr, 4));
  var emberMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uTime: clock, uHell: { value: 0 }, uPx: pxScale },
    vertexShader: 'attribute vec4 aE; uniform float uTime; uniform float uHell; uniform float uPx; varying float vA;\n' +
      'void main(){ float life = fract(uTime * aE.x * 0.18 + aE.y); vec3 p = position;\n' +
      ' p.y += life * 6.0; p.x += sin(uTime * 1.3 + aE.z) * 0.4 * life; p.z += cos(uTime * 0.9 + aE.z) * 0.3 * life;\n' +
      ' vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_Position = projectionMatrix * mv;\n' +
      ' vA = uHell * smoothstep(0.0, 0.08, life) * (1.0 - life) * (0.6 + 0.4 * sin(uTime * 9.0 + aE.z * 5.0));\n' +
      ' gl_PointSize = max(1.5, 0.03 * aE.w * uPx / -mv.z); }',
    fragmentShader: 'varying float vA; void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;\n' +
      ' float a = smoothstep(0.5, 0.0, d) * vA; gl_FragColor = vec4(vec3(1.0, 0.45, 0.12) * a * 1.5, a);\n #include <colorspace_fragment>\n }'
  });
  var embers = new THREE.Points(emberGeo, emberMat);
  embers.frustumCulled = false;
  world.add(embers);

  // ── VI: the rose ────────────────────────────────────────────────────────
  var vaseMat = new THREE.MeshStandardMaterial({ color: '#a8bccc', roughness: 0.08, metalness: 0.1, transparent: true, opacity: 0.32, depthWrite: false });
  var vase = new THREE.Mesh(new THREE.LatheGeometry(v2([[0, 0], [0.045, 0], [0.05, 0.02], [0.03, 0.12], [0.018, 0.22], [0.028, 0.26]]), 18), vaseMat);
  vase.position.set(ROSE.x, ALTAR_TOP + 0.02, ROSE.z);
  var stemMat = new THREE.MeshStandardMaterial({ color: '#2a3d1e', roughness: 0.8 });
  var stem = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.006, ROSE.y - ALTAR_TOP - 0.02, 6), stemMat);
  stem.position.set(ROSE.x, (ROSE.y + ALTAR_TOP + 0.02) / 2, ROSE.z);
  var leafGeo = new THREE.SphereGeometry(0.03, 8, 6).scale(0.5, 0.12, 1.2);
  [[0.16, 0.6], [0.26, -0.9]].forEach(function (lf) {
    var leaf = new THREE.Mesh(leafGeo, stemMat);
    leaf.position.set(ROSE.x + Math.sin(lf[1]) * 0.025, ALTAR_TOP + lf[0] + 0.12, ROSE.z + Math.cos(lf[1]) * 0.025);
    leaf.rotation.set(0.5, lf[1], 0);
    world.add(leaf);
  });
  world.add(vase, stem);
  var head = new THREE.Group();
  head.position.copy(ROSE);
  var sepal = new THREE.Mesh(new THREE.ConeGeometry(0.02, 0.03, 8).rotateX(Math.PI), stemMat);
  sepal.position.y = -0.004;
  head.add(sepal);
  world.add(head);
  var petalGeo = new THREE.PlaneGeometry(0.1, 0.1, 6, 6).translate(0, 0.05, 0), pp = petalGeo.attributes.position;
  for (i = 0; i < pp.count; i++) { var px = pp.getX(i), py = pp.getY(i); pp.setZ(i, -7 * px * px + 0.8 * py * py); }
  petalGeo.computeVertexNormals();
  var curl = { value: 0 };
  var petalMat = moonlit(new THREE.MeshStandardMaterial({ map: petalTexture(), alphaTest: 0.5, side: THREE.DoubleSide, roughness: 0.65 }), U);
  var baseCompile = petalMat.onBeforeCompile;
  petalMat.onBeforeCompile = function (sh) {
    baseCompile(sh);
    sh.uniforms.uCurl = curl;
    sh.vertexShader = 'uniform float uCurl;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n float tip = max(uv.y - 0.4, 0.0);\n' +
      ' transformed.z += uCurl * (tip * tip * 0.22 + abs(position.x) * uv.y * 0.35);\n transformed.y -= uCurl * tip * tip * 0.05;');
  };
  // Petals in a spiral, tight and upright in the bud, opening outwards.
  var petals = [], PN = 24;
  head.scale.setScalar(1.35);
  for (var pk = 0; pk < PN; pk++) {
    var t = pk / (PN - 1), pivot = new THREE.Object3D(), holder = new THREE.Object3D();
    pivot.rotation.y = pk * 2.4;
    holder.position.set(0, (1 - t) * 0.022 - t * 0.008, 0.002 + t * 0.016);
    holder.scale.setScalar((0.42 + t * 0.62) * (0.94 + r() * 0.12));
    pivot.add(holder);
    head.add(pivot);
    var mesh = new THREE.Mesh(petalGeo, petalMat), rest = new THREE.Vector3();
    do { rest.set(ROSE.x + (r() - 0.55) * 0.75, ALTAR_TOP + 0.03, ROSE.z + (r() - 0.35) * 0.55); } while (Math.hypot(rest.x - ROSE.x, rest.z - ROSE.z) < 0.09);
    world.add(mesh);
    petals.push({ holder: holder, mesh: mesh, tilt: 0.04 + Math.pow(t, 1.6) * 0.8, ring: t * 2,
                  fallAt: pk < 8 ? 2 : 0.3 + (1 - t) * 0.5 + r() * 0.08, rest: rest,
                  // Fallen petals lie cupped side down.
                  restQ: new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2 + (r() - 0.5) * 0.3, r() * 6.28, 0, 'YXZ')),
                  ph: r() * 6.28 });
  }
  var pa = new THREE.Vector3(), qa = new THREE.Quaternion(), sa = new THREE.Vector3();
  var roseLight = new THREE.PointLight('#c8d2f2', 0, 5, 1.4);
  roseLight.position.set(ROSE.x - 1.0, ROSE.y + 1.1, ROSE.z + 0.9);
  world.add(roseLight);

  // ── VII: the west doors and the moor beyond ─────────────────────────────
  var plank = plankTexture(r);
  plank.repeat.set(1 / DOOR.h, 1 / (DOOR.st + DOOR.rise));
  var doorMat = new THREE.MeshStandardMaterial({ map: plank, roughness: 0.8, color: '#9a8a78', side: THREE.DoubleSide });
  var right = archRight(DOOR.h, DOOR.st, DOOR.rise, 10), leafPts = [[0, -0.02], [DOOR.h - 0.02, -0.02]];
  right.forEach(function (p) { leafPts.push([Math.min(p[0], DOOR.h - 0.02), p[1] - 0.02]); });
  var leafShape = new THREE.Shape(v2(leafPts));
  var doors = [-1, 1].map(function (s) {
    var geo = new THREE.ExtrudeGeometry(leafShape, { depth: 0.1, bevelEnabled: false, curveSegments: 4 });
    geo.translate(-DOOR.h, 0, -0.05);
    if (s < 0) geo.scale(-1, 1, 1);
    var hinge = new THREE.Group();
    hinge.position.set(s * DOOR.h, 0, 0.5);
    hinge.add(new THREE.Mesh(geo, doorMat));
    world.add(hinge);
    return { hinge: hinge, s: s };
  });

  // The moor: rolling ground, wind-combed grass and running cloud.
  var moorMesh = terrain(700, small ? 110 : 170, 0, 352, moor, new THREE.MeshLambertMaterial({ vertexColors: true }), function (x, z, y) {
    var n = fbm(x * 0.05, z * 0.05) * 0.5 + 0.5, hz = fbm(x * 0.013 + 8, z * 0.013) * 0.5 + 0.5;
    var big = fbm(x * 0.006 + 2, z * 0.006) * 0.5 + 0.5;
    return tmpC.set('#7a7454').lerp(tmpC2.set('#a49a72'), n).lerp(tmpC2.set('#4e3c46'), smooth(0.5, 0.75, hz) * 0.8)
      .lerp(tmpC2.set('#3a3a30'), smooth(0.45, 0.7, big) * 0.5).clone();
  });
  world.add(moorMesh);

  var wind = { value: 0 };
  var tuft = (function () {
    var pos = [], nor = [];
    for (var b = 0; b < 9; b++) {
      var a = b * 2.4 + 0.3, h = 0.28 + ((b * 7) % 5) * 0.08, lean = 0.08 + (b % 3) * 0.07, ox = Math.cos(a) * 0.05, oz = Math.sin(a) * 0.05;
      var px = Math.cos(a + 1.57) * 0.011, pz = Math.sin(a + 1.57) * 0.011;
      pos.push(ox - px, 0, oz - pz, ox + px, 0, oz + pz, ox + Math.cos(a) * lean, h, oz + Math.sin(a) * lean);
      nor.push(0, 1, 0, 0, 1, 0, 0, 1, 0);
    }
    var gq = new THREE.BufferGeometry();
    gq.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    gq.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
    return gq;
  })();
  var grassMat = new THREE.MeshLambertMaterial({ side: THREE.DoubleSide });
  grassMat.onBeforeCompile = function (sh) {
    sh.uniforms.uTime = clock; sh.uniforms.uWind = wind;
    sh.vertexShader = 'uniform float uTime; uniform float uWind;\n' + sh.vertexShader.replace('#include <project_vertex>',
      'vec4 mvPosition = vec4(transformed, 1.0);\n#ifdef USE_INSTANCING\n mvPosition = instanceMatrix * mvPosition;\n#endif\n' +
      ' float bend = position.y * position.y * 2.2;\n' +
      ' float gust = 0.55 + 0.45 * sin(uTime * 2.1 + mvPosition.z * 0.35 + mvPosition.x * 0.15) + 0.12 * sin(uTime * 9.0 + mvPosition.x * 3.0);\n' +
      ' mvPosition.z -= bend * uWind * gust * 0.6; mvPosition.x -= bend * uWind * gust * 0.18; mvPosition.y -= bend * uWind * gust * 0.12;\n' +
      ' mvPosition = modelViewMatrix * mvPosition;\n gl_Position = projectionMatrix * mvPosition;');
  };
  var GN = small ? 7000 : 16000, grass = new THREE.InstancedMesh(tuft, grassMat, GN), gn = 0;
  for (i = 0; i < GN * 2 && gn < GN; i++) {
    var gz = 1.6 + Math.pow(r(), 2.0) * 100, gx = (r() - 0.5) * (14 + gz * 1.4);
    if (r() < 0.3 && gz > 20) continue;
    var sc = 0.45 + r() * 0.45;
    q4.setFromAxisAngle(tv.set(0, 1, 0), r() * 6.28);
    m4.compose(tv2.set(gx, moor(gx, gz) - 0.02, gz), q4, s3.set(sc, sc * (0.8 + r() * 0.5), sc));
    grass.setMatrixAt(gn, m4);
    var gk = r();
    grass.setColorAt(gn, tmpC.set(gk < 0.22 ? '#5e4652' : gk < 0.4 ? '#8a6a48' : '#8a8458').lerp(tmpC2.set('#b8aa7c'), r() * 0.5));
    gn++;
  }
  grass.count = gn;
  grass.frustumCulled = false;
  world.add(grass);

  // A lone thorn on the moor, bare and bent over by the wind: branching
  // limbs swept to the lee side, merged into one mesh.
  (function () {
    var pos = [], nor = [], up = new THREE.Vector3(0, 1, 0), lee = new THREE.Vector3(-1, 0.15, -0.3).normalize();
    var qq = new THREE.Quaternion(), dir = new THREE.Vector3(), end = new THREE.Vector3();
    function limb(p, d, len, rad, depth) {
      end.copy(p).addScaledVector(d, len);
      var geo = new THREE.CylinderGeometry(rad * 0.62, rad, len, 5, 1, true).translate(0, len / 2, 0);
      geo.applyQuaternion(qq.setFromUnitVectors(up, d)).translate(p.x, p.y, p.z);
      geo = geo.toNonIndexed();
      for (var k = 0; k < geo.attributes.position.array.length; k++) { pos.push(geo.attributes.position.array[k]); nor.push(geo.attributes.normal.array[k]); }
      geo.dispose();
      if (depth >= 5) return;
      var tip = end.clone(), n = depth < 1 ? 3 : 2 + (r() < 0.45 ? 1 : 0);
      for (var c = 0; c < n; c++) {
        dir.set(r() - 0.5, r() * 0.7, r() - 0.5).multiplyScalar(2.1).add(d).addScaledVector(lee, 0.45 + depth * 0.1).normalize();
        limb(tip, dir.clone(), len * (0.62 + r() * 0.18), rad * 0.62, depth + 1);
      }
    }
    limb(new THREE.Vector3(0, -0.3, 0), new THREE.Vector3(-0.42, 1, 0.05).normalize(), 2.4, 0.26, 0);
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
    var tree = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ color: '#1a1712' }));
    tree.scale.setScalar(1.25);
    tree.position.set(15, moor(15, 46) - 0.1, 46);
    world.add(tree);
  })();

  // Sky over the moor: overcast night, the moon behind running cloud.
  var skyMat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false,
    uniforms: { uTime: clock, uWind: wind, uMoonP: { value: MOON_D.clone().negate() }, uBright: { value: 1 } },
    vertexShader: 'varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform float uTime; uniform float uWind; uniform vec3 uMoonP; uniform float uBright; varying vec3 vP;\n' + NOISE_GLSL +
      'void main(){ vec3 d = normalize(vP); float h = d.y;\n' +
      ' vec3 col = mix(vec3(0.17, 0.19, 0.25), vec3(0.04, 0.05, 0.075), smoothstep(0.0, 0.7, h));\n' +
      ' float m = max(dot(d, normalize(uMoonP)), 0.0);\n' +
      ' vec2 cp = d.xz / (h + 0.12) * 1.3 + vec2(0.12, -1.0) * uTime * (0.012 + uWind * 0.035);\n' +
      ' float c = piFbm(cp), c2 = piFbm(cp * 2.4 + 7.0);\n' +
      ' float cov = smoothstep(0.36, 0.7, c);\n' +
      ' vec3 cloud = mix(vec3(0.07, 0.075, 0.095), vec3(0.34, 0.36, 0.42), smoothstep(0.35, 0.95, c2) * 0.6 + m * m * 0.35);\n' +
      ' col += vec3(0.6, 0.66, 0.8) * (pow(m, 40.0) * 0.8 + pow(m, 6.0) * 0.12) * (1.0 - cov * 0.85);\n' +
      ' col = mix(col, cloud, cov * smoothstep(-0.01, 0.1, h) * 0.92);\n' +
      ' col = mix(vec3(0.03, 0.03, 0.035), col, smoothstep(-0.12, 0.0, h));\n' +
      ' gl_FragColor = vec4(col * uBright, 1.0);\n #include <tonemapping_fragment>\n #include <colorspace_fragment>\n }'
  });
  var sky = new THREE.Mesh(new THREE.SphereGeometry(1000, 32, 16), skyMat);
  world.add(sky);

  // Cold light through the open doorway, and debris on the wind.
  var doorSpot = new THREE.SpotLight('#9aa8c6', 0, 45, 0.55, 1, 1.1);
  doorSpot.position.set(0, 4.5, 4);
  doorSpot.target.position.set(0, 0, -14);
  world.add(doorSpot, doorSpot.target);
  var spill = new THREE.Mesh(new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2), new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uAmt: { value: 0 } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform float uAmt; varying vec2 vUv;\n' +
      'void main(){ float w = 0.5 * mix(1.0, 2.6, vUv.y); float x = abs(vUv.x - 0.5) * 2.6;\n' +
      ' float a = uAmt * smoothstep(w, w * 0.6, x) * pow(1.0 - vUv.y, 1.2) * 0.32; gl_FragColor = vec4(vec3(0.62, 0.68, 0.82) * a, a);\n #include <colorspace_fragment>\n }'
  }));
  spill.scale.set(6, 1, 16);
  spill.position.set(0, 0.015, -8);
  world.add(spill);

  var BN = small ? 160 : 320, bPos = new Float32Array(BN * 3), bSeed = [];
  for (i = 0; i < BN; i++) bSeed.push([r(), r(), r(), 0.6 + r() * 0.8]);
  var blownGeo = new THREE.BufferGeometry();
  blownGeo.setAttribute('position', new THREE.BufferAttribute(bPos, 3));
  var blownMat = new THREE.PointsMaterial({ map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)'), color: '#8a8270', size: 0.06, transparent: true, opacity: 0, depthWrite: false });
  var blown = new THREE.Points(blownGeo, blownMat);
  blown.frustumCulled = false;
  world.add(blown);

  // ── Frame ───────────────────────────────────────────────────────────────
  var gust = 0, layout = { portrait: false };
  function frame(f) {
    var row = f.row, time = f.time, dt = f.dt;
    var names = row[8], burst = row[9], frost = row[10], bend = row[11], hell = row[12], tear = row[13], wilt = row[14], doorOpen = row[15], moon = row[16];
    clock.value = time;
    var out = smooth(-2, 6, f.cam);                  // how far out of the chapel

    // Camera, with a tremor while hell roars beneath.
    var quake = still ? 0 : hell * smooth(0.35, 0.8, hell);
    var qx = (Math.sin(time * 41) * Math.sin(time * 13.7) + Math.sin(time * 27.3) * 0.5) * quake;
    var qy = (Math.sin(time * 37.1) * Math.sin(time * 9.3) + Math.sin(time * 23.9) * 0.5) * quake;
    camera.position.set(row[4] + qx * 0.025, row[5] + qy * 0.02 + Math.sin(time * 0.9) * 0.01, f.cam);
    camera.rotation.set(0, 0, 0);
    camera.rotateY((layout.portrait ? row[17] : row[6]) - f.mx * 0.08 + qx * 0.004);
    camera.rotateX((layout.portrait ? row[18] : row[7]) - f.my * 0.045 + qy * 0.004);
    camera.rotateZ(qx * 0.003);
    sky.position.copy(camera.position);

    // Light, darkness and the air.
    U.uMoon.value = moon * 3.6;
    U.uFrost.value = frost;
    southGlass.uniforms.uLight.value = 0.35 + moon * 2.2;
    moonLight.intensity = 0.1 + moon * 0.25;
    var dark = row[1];
    hemi.intensity = (1.0 + doorOpen * 0.3) * (1 - dark * 0.7);
    outside.intensity = out * 2.4;
    world.fog.color.set('#080a12').lerp(tmpC.set('#2e3442'), smooth(0.2, 1, doorOpen) * 0.35 + out * 0.65);
    world.fog.density = lerp(0.018 + dark * 0.04, 0.0075, out);
    wind.value = f.wind;
    dustMat.uniforms.uWind.value = doorOpen * 0.4;

    // I: names blur and fade; the votives go out one by one.
    var blurA = smooth(0.85, 0.45, names);
    namesSharp.material.opacity = 1 - blurA;
    namesBlur.material.opacity = blurA * smooth(0.0, 0.3, names);
    var alive = 0;
    votives.forEach(function (v, k) {
      var on = smooth(v.at - 0.05, v.at + 0.03, names), fl = 0.85 + 0.15 * Math.sin(time * 9 + v.phase) * Math.sin(time * 4.1 + v.phase);
      v.flame.scale.set(0.03 * on * fl, 0.065 * on * (0.6 + 0.4 * fl), 1);
      v.flame.visible = on > 0.01;
      alive += on;
    });
    alive /= votives.length;
    votiveGlow.material.opacity = 0.45 * alive;
    votiveLight.intensity = 3.2 * alive * (0.92 + 0.08 * Math.sin(time * 7.3));
    waxMat.emissiveIntensity = 0.3 * alive;
    var away = 1 - names;
    for (var k = 0; k < MN; k++) {
      var b = mBase[k], d = mDir[k], ph = k * 1.7;
      mPos[k * 3] = b.x + d.x * away * 1.6 + Math.sin(time * 0.4 + ph) * 0.06;
      mPos[k * 3 + 1] = b.y + d.y * away * 1.6 + Math.sin(time * 0.6 + ph * 1.3) * 0.08;
      mPos[k * 3 + 2] = b.z + d.z * away * 1.6 + Math.cos(time * 0.5 + ph) * 0.06;
    }
    moteGeo.attributes.position.needsUpdate = true;
    moteMat.opacity = (0.15 + 0.85 * smooth(0.0, 0.35, names) * (1 - smooth(0.3, 1, away) * 0.9)) * (1 - smooth(0.6, 1, away));
    motes.visible = moteMat.opacity > 0.01;

    // II: the bear bursts and the feathers drift down.
    var torn = smooth(0.0, 0.06, burst);
    bearBody.scale.set(1 + torn * 0.12, 1 - torn * 0.28, 1 + torn * 0.08);
    bearHead.rotation.set(torn * 0.5, 0, torn * 0.45);
    bearHead.position.set(torn * 0.03, 0.3 - torn * 0.07, torn * 0.02);
    var tau = burst * 9;
    for (k = 0; k < FN; k++) {
      var fd = fData[k];
      if (tau <= 0) { feathers.setMatrixAt(k, m4.makeScale(0, 0, 0)); continue; }
      var shot = (1 - Math.exp(-3.2 * tau)) / 3.2 * fd.v;
      fPos.copy(fOrigin).addScaledVector(fd.dir, shot);
      var peak = fPos.y, sink = Math.max(0, tau - 0.45) * fd.fall;
      var ry = restY(fPos.x, fPos.z);
      var y = Math.max(ry, peak - sink), air = smooth(ry, ry + 0.25, y);
      fPos.x += Math.sin(tau * 1.4 + fd.ph) * fd.sway * air * smooth(0, 0.8, tau);
      fPos.z += Math.cos(tau * 1.1 + fd.ph) * fd.sway * 0.7 * air * smooth(0, 0.8, tau);
      fPos.y = y;
      e3.set(lerp(-Math.PI / 2, fd.ph + Math.sin(time * 1.7 + fd.ph) * 0.4, air), fd.ph * 3 + tau * fd.spin * air, lerp(0, Math.sin(tau * 2 + fd.ph) * 0.8, air), 'YXZ');
      q4.setFromEuler(e3);
      feathers.setMatrixAt(k, m4.compose(fPos, q4, s3.setScalar(fd.sc * smooth(0, 0.15, tau))));
    }
    feathers.instanceMatrix.needsUpdate = true;

    // IV: the candle; the dark presses in and bends the flame over.
    gust += ((noise(time * 1.6, 2.3) * 0.5 + 0.5) - gust) * (1 - Math.exp(-dt * 5));
    var lean = bend * (0.75 + 0.25 * gust) + doorOpen * 0.3;
    flameMat.uniforms.uBend.value = lean;
    flameMat.uniforms.uLife.value = 1 - bend * 0.2;
    flame.quaternion.copy(camera.quaternion);
    var flick = 0.9 + 0.1 * Math.sin(time * 11) * Math.sin(time * 4.3);
    candleLight.intensity = 3.5 * flick * (1 - bend * 0.55);
    candleLight.distance = 9 - bend * 5;
    candleGlow.scale.setScalar(0.5 * (1 - bend * 0.35) * flick);
    candleGlow.position.set(CANDLE.x + lean * 0.03, FLAME_Y + 0.05, CANDLE.z);
    candleHalo.material.opacity = 0.12 * (1 - bend * 0.6);
    // The smoke comes from the camera's right, the side the flame flees.
    tv.set(1, 0, 0).applyQuaternion(camera.quaternion);
    tv2.set(0, 0, 1).applyQuaternion(camera.quaternion);
    for (k = 0; k < SN; k++) {
      var sd = sData[k], ang = sd.a + time * sd.sp, rad = lerp(3.2, 0.5, bend) * sd.d + 0.35;
      var cx = Math.cos(ang) * 0.6 + 0.8, cz = Math.sin(ang);
      sPos[k * 3] = CANDLE.x + (tv.x * cx + tv2.x * cz * 0.6) * rad;
      sPos[k * 3 + 1] = FLAME_Y + sd.y * rad * 0.5 + Math.sin(time * 0.3 + sd.a) * 0.1;
      sPos[k * 3 + 2] = CANDLE.z + (tv.z * cx + tv2.z * cz * 0.6) * rad;
      sAttr[k * 2] = sd.sz * (0.7 + bend * 0.5);
      sAttr[k * 2 + 1] = bend * 0.45 * smooth(0.4, 0.7, dark);
    }
    smokeGeo.attributes.position.needsUpdate = true;
    smokeGeo.attributes.aS.needsUpdate = true;
    smoke.visible = bend > 0.01 && dark > 0.4;

    // V: the veil tears and falls; the red beneath, and embers.
    halves.forEach(function (hv) {
      // Torn in two, each half sags from its outer corner, then drops.
      var swing = smooth(0, 0.35, tear), drop = smooth(0.3, 0.85, tear);
      hv.pivot.rotation.set(drop * 0.5, 0, hv.s * (swing * 0.2 + Math.sin(time * 1.3 + hv.s) * 0.015 * swing));
      hv.pivot.position.set(hv.s * (VEIL.w / 2 + swing * 0.1), VEIL.top - drop * drop * 7, VEIL.z + drop * 0.9);
      hv.pivot.scale.y = 1 - drop * 0.45;
      hv.mat.opacity = 1 - smooth(0.55, 0.85, tear);
      hv.pivot.visible = hv.mat.opacity > 0.01;
    });
    var douse = 1 - smooth(0.02, 0.2, tear);
    altarFlames.forEach(function (af, n) {
      var fl = 0.9 + 0.1 * Math.sin(time * 10 + n * 2);
      af.flame.scale.set(0.035 * douse * fl, 0.08 * douse, 1);
      af.flame.visible = douse > 0.01;
      af.glow.material.opacity = 0.35 * douse;
      // The tremor knocks them over the ends of the altar.
      var topple = smooth(0.35, 0.75, tear);
      af.stick.rotation.z = -af.s * topple * 1.5;
      af.stick.position.y = ALTAR_TOP + 0.02 - topple * 0.05;
    });
    altarLight.intensity = 5 * douse * (0.93 + 0.07 * Math.sin(time * 8.1));
    var pulse = 0.8 + 0.2 * Math.sin(time * 2.3) * Math.sin(time * 0.7 + 1);
    crackWallMat.opacity = hell * pulse * smooth(0.1, 0.5, tear);
    crackFloorMat.opacity = hell * pulse;
    hellLight.intensity = hell * 16 * pulse;
    hellWall.intensity = hell * 14 * pulse * smooth(0.1, 0.5, tear);
    emberMat.uniforms.uHell.value = hell;
    embers.visible = hell > 0.01;

    // VI: the rose wilts and drops its petals.
    curl.value = smooth(0.05, 0.85, wilt);
    petalMat.color.set('#ffffff').lerp(tmpC.set('#b08a7a'), smooth(0.2, 1, wilt));
    head.rotation.set(wilt * 0.35, 0, -wilt * 0.45);
    head.position.set(ROSE.x - wilt * 0.02, ROSE.y - wilt * 0.03, ROSE.z);
    head.updateMatrixWorld(true);
    petals.forEach(function (p) {
      p.holder.rotation.x = p.tilt + wilt * (0.35 + p.ring * 0.25);
      p.holder.updateMatrixWorld(true);
      p.holder.matrixWorld.decompose(pa, qa, sa);
      var fall = smooth(p.fallAt, p.fallAt + 0.16, wilt);
      if (fall > 0) {
        p.mesh.position.lerpVectors(pa, p.rest, fall);
        p.mesh.position.y += Math.sin(fall * Math.PI) * 0.06;
        p.mesh.position.x += Math.sin(fall * Math.PI * 2 + p.ph) * 0.04 * (1 - fall);
        p.mesh.quaternion.slerpQuaternions(qa, p.restQ, smooth(0, 1, fall));
      } else {
        p.mesh.position.copy(pa);
        p.mesh.quaternion.copy(qa);
      }
      p.mesh.scale.copy(sa);
    });
    roseLight.intensity = 3.2 * smooth(-31, -32.4, f.cam) * (1 - doorOpen);

    // VII: the doors swing in on the moor; the wind comes down the nave.
    doors.forEach(function (d) { d.hinge.rotation.y = -d.s * doorOpen * 1.45 * (1 + Math.sin(time * 1.7 + d.s) * 0.015 * doorOpen); });
    doorSpot.intensity = doorOpen * 60;
    spill.material.uniforms.uAmt.value = doorOpen * (1 - out);
    skyMat.uniforms.uBright.value = 0.75 + out * 0.4;
    blownMat.opacity = doorOpen * 0.85;
    blown.visible = doorOpen > 0.01;
    if (blown.visible) {
      for (k = 0; k < BN; k++) {
        var bs = bSeed[k], span = 34;
        var z = 8 - ((time * (5 + bs[3] * 5) * f.wind + bs[0] * span) % span);
        var bx = (bs[1] - 0.5) * (z > 0 ? 16 : 3.2) + Math.sin(time * 2.3 + k) * 0.3;
        var by = 0.1 + bs[2] * (z > 0 ? 2.5 : 3.8) + Math.sin(time * 3.1 + k * 0.7) * 0.25;
        bPos[k * 3] = bx; bPos[k * 3 + 1] = by; bPos[k * 3 + 2] = z;
      }
      blownGeo.attributes.position.needsUpdate = true;
    }

    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) {
      fitCamera(gl, camera, w, h, dpr, small);
      pxScale.value = h * gl.getPixelRatio() / (2 * Math.tan(camera.fov * Math.PI / 360));
      moteMat.size = w / h < 1 ? 0.09 : 0.07;
      layout.portrait = w / h < 1;
    },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('stained-glass', {
  renderer: renderer3d,
  maxLines: 2,
  scrim: 0.68,
  accent: '#a9bfe8',
  emphasis: /^alone/i,
  align: ['left', 'right', 'left', 'right', 'left', 'left', 'center'],
  keys: function (T) {
    var rows = [], st = { dark: 0.1, wind: 0.08, names: 1, burst: 0, frost: 0, bend: 0, hell: 0, tear: 0, wilt: 0, doors: 0, moon: 1 }, prev = 0, prevP = 0;
    function S(i, frac) { i = Math.min(i, T.count - 1); return T.start(i) + frac * 1.6; }
    // A key: where the camera stands, what it looks at, and what changes.
    // `subj` (default: look) is where a portrait screen aims instead.
    function aim(cam, look, last) {
      var dx = look[0] - cam[0], dy = look[1] - cam[1], dz = look[2] - cam[2];
      var yaw = Math.atan2(-dx, -dz);
      while (yaw - last > Math.PI) yaw -= Math.PI * 2;
      while (last - yaw > Math.PI) yaw += Math.PI * 2;
      return [yaw, Math.atan2(dy, Math.hypot(dx, dz))];
    }
    function K(u, cam, look, set, subj) {
      for (var k in set) st[k] = set[k];
      var a = aim(cam, look, prev), b = aim(cam, subj || look, prevP);
      prev = a[0]; prevP = b[0];
      rows.push([u, cam[2], st.dark, 0, st.wind, cam[0], cam[1], a[0], a[1],
                 st.names, st.burst, st.frost, st.bend, st.hell, st.tear, st.wilt, st.doors, st.moon, b[0], b[1]]);
    }
    var DOORS = [0, 9, 0];
    K(0,          [0.2, 2.5, -0.5],      [0, 2.4, -30], {});
    K(0.7,        [0.5, 2.3, -0.8],      [0.2, 2.6, -30], {});
    // I: the roll of names by the door.
    K(S(0, 0.2),  [1.6, 1.68, -1.2],     [6, 1.95, -3.4], {});
    K(S(0, 0.3),  [1.8, 1.68, -1.25],    [6, 1.95, -3.35], { names: 1 });
    K(S(0, 0.85), [2.4, 1.68, -1.4],     [6, 1.95, -3.2], { names: 0 });
    // II: the bear on the pew, and the feathers.
    K(S(1, 0.18), [0.45, 1.4, -7.9],     [-0.45, 0.3, -9.6], {}, [-1.0, 0.75, -9.55]);
    K(S(1, 0.36), [0.4, 1.35, -8.1],     [-0.5, 0.32, -9.6], { burst: 0 }, [-1.0, 0.75, -9.55]);
    K(S(1, 0.9),  [0.55, 1.45, -7.7],    [-0.5, 1.0, -9.8], { burst: 0.55 }, [-1.0, 1.15, -9.6]);
    // III: up against the glass; frost and the chill.
    K(S(2, 0.2),  [3.0, 2.2, -15.2],     [6.5, 3.8, -17.0], { burst: 0.85 }, [6.5, 2.0, -16.2]);
    K(S(2, 0.5),  [4.3, 2.75, -16.5],    [6.5, 3.3, -17.15], { burst: 1, frost: 0.05, moon: 1.15 }, [6.5, 1.0, -16.1]);
    K(S(2, 0.93), [4.45, 2.75, -16.55],  [6.5, 3.3, -17.15], { frost: 1, moon: 1.3 }, [6.5, 1.0, -16.1]);
    // IV: the candle in the dark.
    K(S(3, 0.2),  [0.9, 1.62, -18.1],    [-0.25, 1.45, -20.6], { moon: 0.9, frost: 0.85 });
    K(S(3, 0.45), [0.75, 1.6, -18.5],    [-0.3, 1.45, -20.6], { bend: 0.2, moon: 0.55, dark: 0.3 });
    K(S(3, 0.93), [0.6, 1.58, -18.9],    [-0.35, 1.48, -20.6], { bend: 1, moon: 0.2, dark: 0.7, frost: 0.6 });
    // V: the veil torn from the altar; hell beneath.
    K(S(4, 0.2),  [0.25, 1.7, -24.2],    [0.6, 4.9, -37], { moon: 0.35, dark: 0.45 });
    K(S(4, 0.42), [0.2, 1.7, -24.8],     [0.6, 4.9, -37], { tear: 0, hell: 0 });
    K(S(4, 0.62), [0.15, 1.7, -25.2],    [0.6, 4.4, -37], { tear: 0.5, hell: 0.7, frost: 0.2 });
    K(S(4, 0.92), [0.1, 1.7, -25.6],     [0.6, 3.8, -37], { tear: 1, hell: 1, frost: 0 });
    // VI: the rose on the altar.
    K(S(5, 0.2),  [-0.35, 1.88, -32.85], [0.0, 1.74, -33.85], { hell: 0.08, dark: 0.45, moon: 0.45 }, [0.2, 1.6, -33.75]);
    K(S(5, 0.35), [-0.32, 1.87, -32.92], [0.0, 1.74, -33.85], { wilt: 0 }, [0.2, 1.6, -33.75]);
    K(S(5, 0.93), [-0.3, 1.85, -32.98],  [0.02, 1.7, -33.85], { wilt: 1, hell: 0.06 }, [0.22, 1.52, -33.8]);
    // VII: turning back to the west doors; out onto the moor.
    K(S(6, 0.02), [-0.3, 1.84, -32.9],   [0.02, 1.7, -33.85], { hell: 0.03 }, [0.22, 1.55, -33.8]);
    K(S(6, 0.3),  [-0.1, 1.72, -31.4],   DOORS, { hell: 0, doors: 0.05, wind: 0.2, moon: 0.8, dark: 0.3 });
    K(S(6, 0.75), [0, 1.72, -30.2],      DOORS, { doors: 1, wind: 0.65, moon: 0.9 });
    K(S(6, 1.0),  [0, 1.72, -25],        [0, 5, 0], { wind: 0.75 });
    K(T.total - 0.55, [0, 1.75, 0.6],    [0, 2.1, 30], { wind: 0.9, dark: 0.2 });
    K(T.total,    [0, 1.75, 9],          [0, 3.2, 60], { wind: 1 });
    return rows;
  },
  sound: {
    src: '/audio/wind.mp3',
    label: 'Play the wind and the chapel',
    volume: function (row) { return 0.04 + 0.5 * row[3] * row[3]; },
    cues: [
      { stanza: 0, at: 0.1, play: toll },
      { stanza: 1, at: 0.58, play: burstSound },
      { stanza: 2, at: 0.75, play: glassSound },
      { stanza: 4, at: 0.7, play: rumble },
      { stanza: 6, at: 0.45, play: creak }
    ]
  }
});
