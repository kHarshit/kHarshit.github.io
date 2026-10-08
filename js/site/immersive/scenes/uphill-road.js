/*
 * Scene for "Don't Quit" (John Greenleaf Whittier): a walled lane climbing
 * a green fellside in the Dales, under a low grey sky in the drizzle, to
 * the top of the hill and the sun.
 *
 * The poem is one stanza of eighteen lines; maxLines 6 makes three panels.
 * I    "When the road you're trudging seems all up hill": the lane runs on
 *      and up between its dry-stone walls, past the barns and the sheep, into
 *      the cloud. "Rest if you must": a wooden bench and an old milestone
 *      ("1 mile") by the wall, and the view turns to them for a moment.
 * II   "Life is strange with its twists and turns": through the intake wall
 *      the lane turns back on itself in hairpins up the steep rough pasture.
 *      At the first bend the view looks out over the dale below; at the next,
 *      up at the lane still winding on above, sheep idling on it: "the pace
 *      seems slow". The drizzle thins.
 * III  "The silver tint of the clouds of doubt": the view lifts to the sky
 *      and the cloud near the hidden sun takes a silver edge. "It may be near
 *      when it seems so far": through the gate in the top wall the hill is
 *      crested, and a golden dale opens below with the lane running down into
 *      it, and sunbeams falling through gaps in the cloud. "When things seem
 *      worst": a last cloud slides over, then the sun breaks out of it, a
 *      skylark goes up, and a beam of light lands where you stand.
 *
 * The cloud deck is one shader plane; the same cloud density is projected
 * along the sun onto everything below, so its gaps light the land. Columns:
 *   [unit, path, gloom, drizzle, wind, yaw, pitch, silver, sun, gold,
 *    beams, lift, out, rest, sky]
 * where silver opens a gap round the sun, sun slides it over the sun's
 * line from the crest, and gold breaks up the whole sky; out, rest and sky
 * turn the view from the lane to the dale below, the bench, or the sun.
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, softSprite, skyDome, broadleafGeometry,
         scatter, rainField, particleField, followPath, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout (metres; you climb towards -z, where the sun is) ──────────────
var CLOUD_Y = 230, CREST_Z = -205;
var SUN_DIR = new THREE.Vector3(-0.3, 0.27, -0.91).normalize();

// The fellside rises from the dale floor to a rounded crest, then falls
// away into the far dale; flat-topped fells beyond, and the far side of
// the home dale behind you.
function fell(x, z) {
  return 52 * smooth(60, CREST_Z, z) - 88 * smooth(CREST_Z - 10, -520, z) +
         (58 + 22 * Math.sin(x * 0.0031 + 1.2)) * smooth(-700, -1150, z) +
         70 * smooth(-1500, -2300, z) * (0.8 + 0.2 * Math.sin(x * 0.002)) +
         75 * smooth(230, 620, z) +
         9 * Math.sin(x * 0.009 + 0.5) * Math.cos(z * 0.006) + 4 * Math.sin(x * 0.027 - z * 0.019) +
         1.1 * Math.sin(x * 0.08 + z * 0.06);
}

// The lane: straight up the lower fields, hairpins up the steep intake,
// over the crest and down into the far dale.
var ROAD_XZ = [[3, 70], [1, 48], [-2, 26], [0, 4], [3, -18], [2, -40], [-3, -60],
  [-16, -73], [-34, -81], [-47, -88], [-53, -95], [-47, -101], [-33, -104],
  [-8, -110], [16, -116], [29, -121], [35, -128], [29, -134], [14, -137],
  [-10, -143], [-29, -150], [-39, -157], [-36, -164], [-23, -168],
  [-6, -175], [2, -186], [1, -198], [0, -212], [-1, -228], [-4, -246],
  [-9, -275], [-20, -310], [-29, -350], [-27, -400], [-14, -450]];
var ROAD = new THREE.CatmullRomCurve3(ROAD_XZ.map(function (p) { return new THREE.Vector3(p[0], 0, p[1]); }), false, 'centripetal');
var NR = 900, roadPts = ROAD.getSpacedPoints(NR), roadY = new Float32Array(NR + 1);

// Road heights follow the fell, smoothed along the lane.
(function () {
  var raw = roadPts.map(function (p) { return fell(p.x, p.z); });
  for (var i = 0; i <= NR; i++) {
    var s = 0, n = 0;
    for (var k = -4; k <= 4; k++) { var j = i + k; if (j >= 0 && j <= NR) { s += raw[j]; n++; } }
    roadY[i] = s / n;
  }
})();

// A coarse grid over the road points, for the distance to the lane.
var CELL = 8, GRID = {};
roadPts.forEach(function (p, i) {
  var key = (Math.floor(p.x / CELL) + 500) * 4096 + Math.floor(p.z / CELL) + 500;
  (GRID[key] || (GRID[key] = [])).push(i);
});
var RN = { d: 0, y: 0, i: 0 };
function roadNear(x, z, out) {
  var cx = Math.floor(x / CELL) + 500, cz = Math.floor(z / CELL) + 500, best = 1e9, wy = 0, ws = 0, bi = -1;
  for (var a = -2; a <= 2; a++) for (var b = -2; b <= 2; b++) {
    var list = GRID[(cx + a) * 4096 + cz + b];
    if (!list) continue;
    for (var k = 0; k < list.length; k++) {
      var p = roadPts[list[k]], dx = p.x - x, dz = p.z - z, d2 = dx * dx + dz * dz;
      if (d2 < best) { best = d2; bi = list[k]; }
      var w = Math.exp(-d2 / 18);
      wy += w * roadY[list[k]];
      ws += w;
    }
  }
  out.d = Math.sqrt(best);
  out.y = ws > 1e-6 ? wy / ws : (bi >= 0 ? roadY[bi] : 0);
  out.i = bi;
  return out;
}

// The ground: the fell, cut level across the lane.
function ground(x, z) {
  var h = fell(x, z);
  roadNear(x, z, RN);
  if (RN.d > 10.5) return h;
  return lerp(RN.y, h, smooth(2.3, 10.5, RN.d));
}

// Path progress nearest a point, for the keyframes.
function tAt(x, z) {
  var best = 1e9, bi = 0;
  roadPts.forEach(function (p, i) { var d = (p.x - x) * (p.x - x) + (p.z - z) * (p.z - z); if (d < best) { best = d; bi = i; } });
  return bi / NR;
}
var T_BENCH = tAt(2, -36), T_H1 = tAt(-53, -95), T_H2 = tAt(35, -128), T_H3 = tAt(-39, -157);
var T_GATE = tAt(1, -194), T_TOP = tAt(-1, -226);

// The walls that parcel up the fields. HEAD is the top wall of the home
// fellside (with the gate), FARTOP the top wall of the far dale; between
// them is open moor.
function HEAD(x) { return -191 + 7 * Math.sin(x * 0.013 + 0.4); }
function FARTOP(x) { return -294 + 8 * Math.sin(x * 0.011 + 2); }
var ACROSS_NEAR = [330, 270, 210, 150, 100, 58, 18, -24, -66, -108, -150];
var UP_NEAR = [-640, -520, -410, -305, -220, -145, -80, 72, 140, 215, 300, 395, 500, 615];
var ACROSS_FAR = [-352, -418, -492, -572, -652, -736, -812, -880];
var UP_FAR = [-900, -770, -640, -520, -400, -290, -180, -70, 40, 150, 262, 380, 500, 620, 750, 880];
// Walls in the far dale (j, k >= 20) run straighter.
function acrossZ(z0, j, x) {
  return j < 20 ? z0 + 10 * Math.sin(x * 0.011 + j * 1.7) + 4 * Math.sin(x * 0.035 + j)
                : z0 + 6 * Math.sin(x * 0.005 + j * 1.7) + 1.5 * Math.sin(x * 0.03 + j);
}
function upX(x0, k, z) {
  return k < 20 ? x0 + 9 * Math.sin(z * 0.012 + k * 2.1) + 3 * Math.sin(z * 0.05 + k)
                : x0 + 5 * Math.sin(z * 0.007 + k * 2.1) + 1.5 * Math.sin(z * 0.04 + k);
}
var BECK_Z = function (x) { return -606 + 34 * Math.sin(x * 0.004 + 1) + 12 * Math.sin(x * 0.013); };

function hash2(a, b) { var s = Math.sin(a * 127.1 + b * 311.7) * 43758.5453; return s - Math.floor(s); }

// Which walled field a point is in (or -1 on the moor).
function fieldOf(x, z) {
  var j = 0, k = 0, far = z < -240;
  if (z < HEAD(x) && z > FARTOP(x)) return -1;
  if (far) {
    if (fell(x, z) > 22) return -1;
    ACROSS_FAR.forEach(function (z0, n) { if (z < acrossZ(z0, n + 20, x)) j++; });
    UP_FAR.forEach(function (x0, n) { if (x > upX(x0, n + 20, z)) k++; });
    return 1000 + j * 40 + k;
  }
  if (fell(x, z) > 46 && z > 200) return -1;
  ACROSS_NEAR.forEach(function (z0, n) { if (z < acrossZ(z0, n, x)) j++; });
  UP_NEAR.forEach(function (x0, n) { if (x > upX(x0, n, z)) k++; });
  // One big rough pasture round the hairpins.
  if (z < -66 && z > -191 && Math.abs(x + 10) < 85) return 777;
  return j * 40 + k;
}

// ── Shader snippets ──────────────────────────────────────────────────────
// Cloud density over the plane y = CLOUD_Y: fbm thinned by `uCover`, with
// a gap that opens round uGapC and three holes for the far sunbeams.
// `oct` octaves of noise (fewer on phones); the deck and the ground must
// use the same.
function cloudGLSL(oct) { return '' +
  'uniform float uTime; uniform float uCover; uniform vec2 uGapC; uniform float uGapR; uniform float uGap;\n' +
  'uniform vec4 uHoles[3]; uniform float uHoleAmt; uniform vec3 uSunDir;\n' +
  'float ch(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }\n' +
  'float cn(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);\n' +
  ' return mix(mix(ch(i), ch(i + vec2(1.0, 0.0)), f.x), mix(ch(i + vec2(0.0, 1.0)), ch(i + vec2(1.0, 1.0)), f.x), f.y); }\n' +
  'float cfbm(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < ' + oct + '; i++){ s += a * cn(p); p = mat2(1.6, 1.2, -1.2, 1.6) * p + 7.1; a *= 0.5; } return s; }\n' +
  'float cloudD(vec2 p){\n' +
  ' vec2 q = p / 380.0 + vec2(uTime * 0.006, uTime * 0.0025);\n' +
  ' float n = cfbm(q);\n' +
  ' float d = (n - 0.5) * 2.4 + uCover;\n' +
  ' vec2 gq = p - uGapC; gq.x *= 0.7;\n' +
  ' float g = length(gq) + (n - 0.5) * 420.0;\n' +
  ' d -= uGap * (1.0 - smoothstep(uGapR * 0.35, uGapR, g)) * 1.6;\n' +
  ' for (int k = 0; k < 3; k++){ float hd = distance(p, uHoles[k].xy) + (n - 0.5) * 200.0;\n' +
  '  d -= uHoleAmt * (1.0 - smoothstep(uHoles[k].z * 0.3, uHoles[k].z, hd)) * 1.4; }\n' +
  ' return clamp(d, 0.0, 1.3); }\n' +
  // How much sun reaches a point: look up along the sun to the deck.
  'float cloudLit(vec3 wp){ vec2 p = wp.xz + uSunDir.xz * ((' + CLOUD_Y.toFixed(1) + ' - wp.y) / uSunDir.y);\n' +
  ' return 1.0 - smoothstep(0.1, 0.42, cloudD(p)); }\n';
}

// Patch a Lambert material so the sun reaches it only through the cloud
// gaps. A wet one (the lane) also takes a sheen of sky and a sun glint.
function sunlit(mat, U, glsl, wet) {
  mat.onBeforeCompile = function (sh) {
    Object.keys(U).forEach(function (k) { sh.uniforms[k] = U[k]; });
    sh.vertexShader = 'varying vec3 vSWP;\n' + sh.vertexShader.replace('#include <project_vertex>',
      '#include <project_vertex>\n vec4 sWP = vec4(transformed, 1.0);\n' +
      '#ifdef USE_INSTANCING\n sWP = instanceMatrix * sWP;\n#endif\n vSWP = (modelMatrix * sWP).xyz;');
    sh.fragmentShader = 'varying vec3 vSWP; uniform vec3 uSunV; uniform vec3 uSunCol; uniform vec3 uSheen; uniform float uWet;\n' +
      glsl + sh.fragmentShader.replace('#include <lights_fragment_end>',
      '#include <lights_fragment_end>\n float sLit = cloudLit(vSWP);\n' +
      ' reflectedLight.directDiffuse += diffuseColor.rgb * uSunCol * max(dot(normal, uSunV), 0.0) * sLit;\n' +
      (wet ? ' vec3 sV = normalize(vViewPosition);\n' +
             ' float fr = pow(1.0 - max(dot(normal, sV), 0.0), 3.0);\n' +
             ' reflectedLight.directDiffuse += uSheen * (0.03 + fr * 0.22) * uWet;\n' +
             ' float sp = pow(max(dot(normal, normalize(uSunV + sV)), 0.0), 24.0);\n' +
             ' reflectedLight.directDiffuse += uSunCol * sp * sLit * 0.22 * uWet;\n' : ''));
  };
  mat.customProgramCacheKey = function () { return 'uphill-sunlit' + (wet ? '-wet' : ''); };
  return mat;
}

// ── Textures painted on canvases ─────────────────────────────────────────
function canvasTex(w, h, paint, srgb) {
  var c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  paint(c.getContext('2d'), w, h);
  var t = new THREE.CanvasTexture(c);
  if (srgb !== false) t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = 4;
  return t;
}

// Dry-stone walling: thin courses of flat grey limestone, laid long, with
// dark gaps and lichen, damp and mossy at the foot, and a row of upright
// coping stones along the top (the top 14%). 512 px is 2 m of wall.
function stoneTexture(r) {
  return canvasTex(512, 256, function (x, w, h) {
    x.fillStyle = '#2f2e2a';
    x.fillRect(0, 0, w, h);
    function stone(px, py, sw, sh, l) {
      var g = Math.round(l), warm = Math.round((r() - 0.45) * 10), k = Math.min(sw, sh) * 0.3;
      x.fillStyle = 'rgb(' + (g + warm) + ',' + (g + 1) + ',' + (g - warm - 3) + ')';
      x.beginPath();
      x.moveTo(px + k, py + r() * 2);
      x.lineTo(px + sw - k, py + r() * 2);
      x.lineTo(px + sw, py + k + r() * 2);
      x.lineTo(px + sw - r() * 2, py + sh - k);
      x.lineTo(px + sw - k, py + sh - r() * 1.5);
      x.lineTo(px + k, py + sh - r() * 1.5);
      x.lineTo(px + r() * 2, py + sh - k);
      x.lineTo(px, py + k);
      x.fill();
      x.fillStyle = 'rgba(255,255,248,0.13)';            // weathered top edge
      x.fillRect(px + k, py + 1, sw - k * 2, Math.max(1.5, sh * 0.22));
      x.fillStyle = 'rgba(0,0,0,0.16)';                  // shadowed underside
      x.fillRect(px + k, py + sh - Math.max(1.5, sh * 0.2), sw - k * 2, Math.max(1.5, sh * 0.2));
      if (r() < 0.35) {
        x.fillStyle = r() < 0.6 ? 'rgba(214,208,150,0.5)' : 'rgba(236,236,226,0.45)';
        x.beginPath();
        x.ellipse(px + r() * sw, py + r() * sh, 1.5 + r() * 4, 1 + r() * 2.5, 0, 0, 6.28);
        x.fill();
      }
    }
    var cap = Math.round(h * 0.14);
    for (var px = 0; px < w;) {                         // coping, upright
      var cw = 16 + r() * 16;
      stone(px + 1, 1 + r() * 5, cw - 2, cap - 3, 118 + r() * 50);
      px += cw;
    }
    for (var py = cap; py < h;) {                       // courses
      var ch = 9 + r() * 11;
      for (var qx = -r() * 40; qx < w;) {
        var sw = 26 + r() * 74;
        if (r() < 0.12) sw = 12 + r() * 10;             // a pinning stone
        stone(qx + 1, py + 1 + (r() - 0.5) * 2, sw - 2, ch - 2, 120 + r() * 62);
        qx += sw;
      }
      py += ch;
    }
    var g = x.createLinearGradient(0, h * 0.55, 0, h);
    g.addColorStop(0, 'rgba(40,52,24,0)');
    g.addColorStop(1, 'rgba(40,52,24,0.55)');
    x.fillStyle = g;
    x.fillRect(0, h * 0.55, w, h * 0.45);
  });
}

// Coursed rubble for the barns: squarer blocks, mortar-pale joints.
function rubbleTexture(r) {
  return canvasTex(256, 256, function (x, w, h) {
    x.fillStyle = '#7d7a70';
    x.fillRect(0, 0, w, h);
    for (var py = 0; py < h;) {
      var ch = 14 + r() * 14;
      for (var px = -r() * 30; px < w;) {
        var sw = 22 + r() * 40, g = Math.round(150 + r() * 70);
        x.fillStyle = 'rgb(' + (g + 4) + ',' + g + ',' + (g - 8) + ')';
        x.fillRect(px + 1.5, py + 1.5, sw - 3, ch - 3);
        x.fillStyle = 'rgba(0,0,0,0.12)';
        x.fillRect(px + 1.5, py + ch - 4, sw - 3, 2.5);
        px += sw;
      }
      py += ch;
    }
  });
}

// A narrow tarmac lane, worn paler in the wheel tracks, gravelly at the
// edges with grass creeping in. u runs across, v along.
function roadTexture(r) {
  return canvasTex(128, 512, function (x, w, h) {
    x.fillStyle = '#4f5152';
    x.fillRect(0, 0, w, h);
    for (var i = 0; i < 9000; i++) {
      var g = 70 + r() * 60;
      x.fillStyle = 'rgba(' + g + ',' + g + ',' + (g + 2) + ',0.5)';
      x.fillRect(r() * w, r() * h, 1 + r() * 2, 1 + r() * 2);
    }
    [0.3, 0.7].forEach(function (u) {
      var gr = x.createLinearGradient(u * w - 14, 0, u * w + 14, 0);
      gr.addColorStop(0, 'rgba(150,150,148,0)');
      gr.addColorStop(0.5, 'rgba(150,150,148,0.22)');
      gr.addColorStop(1, 'rgba(150,150,148,0)');
      x.fillStyle = gr;
      x.fillRect(u * w - 14, 0, 28, h);
    });
    // Patches and dark wet hollows.
    for (i = 0; i < 40; i++) {
      x.fillStyle = r() < 0.4 ? 'rgba(44,46,48,0.22)' : 'rgba(112,112,108,0.18)';
      x.beginPath();
      x.ellipse(16 + r() * 96, r() * h, 3 + r() * 8, 4 + r() * 14, r() * 3, 0, 6.28);
      x.fill();
    }
    // Gravel and grass at the edges.
    [[0, 1], [w, -1]].forEach(function (e) {
      for (var k = 0; k < 1600; k++) {
        var d = Math.pow(r(), 2) * 22, y = r() * h;
        x.fillStyle = r() < 0.55 ? 'rgba(' + (60 + r() * 30) + ',' + (80 + r() * 40) + ',40,0.8)' : 'rgba(140,134,120,0.7)';
        x.fillRect(e[0] + e[1] * d - 1, y, 2, 2 + r() * 4);
      }
    });
  });
}

// Fine grain for the grass: grey, multiplied into the field colours.
function grainTexture(r) {
  return canvasTex(256, 256, function (x, w, h) {
    x.fillStyle = '#dcdcdc';
    x.fillRect(0, 0, w, h);
    for (var i = 0; i < 7000; i++) {
      var g = Math.round(170 + r() * 85);
      x.fillStyle = 'rgb(' + g + ',' + g + ',' + g + ')';
      x.fillRect(r() * w, r() * h, 1, 2 + r() * 4);
    }
  });
}

// The milestone's face: a carved arrow up the hill and the distance left.
function milestoneTexture() {
  return canvasTex(128, 192, function (x, w, h) {
    x.fillStyle = '#a4a091';
    x.fillRect(0, 0, w, h);
    for (var i = 0; i < 900; i++) {
      x.fillStyle = Math.random() < 0.5 ? 'rgba(70,70,60,0.18)' : 'rgba(230,228,210,0.2)';
      x.fillRect(Math.random() * w, Math.random() * h, 2, 2);
    }
    x.fillStyle = 'rgba(52,50,44,0.85)';
    x.beginPath();
    x.moveTo(64, 26); x.lineTo(84, 52); x.lineTo(70, 52); x.lineTo(70, 80); x.lineTo(58, 80); x.lineTo(58, 52); x.lineTo(44, 52);
    x.fill();
    x.textAlign = 'center';
    x.font = 'bold 44px Georgia, serif';
    x.fillText('1', 64, 128);
    x.font = 'bold 22px Georgia, serif';
    x.fillText('MILE', 64, 158);
    x.fillStyle = 'rgba(200,205,140,0.5)';
    for (i = 0; i < 14; i++) { x.beginPath(); x.arc(Math.random() * w, 150 + Math.random() * 40, 2 + Math.random() * 5, 0, 6.28); x.fill(); }
  });
}

// A glow that falls off fast from a bright core.
function glowTexture() {
  return canvasTex(256, 256, function (x) {
    var g = x.createRadialGradient(128, 128, 0, 128, 128, 128);
    [[0, 1], [0.05, 0.7], [0.14, 0.3], [0.35, 0.09], [0.7, 0.02], [1, 0]].forEach(function (s) {
      g.addColorStop(s[0], 'rgba(255,248,230,' + s[1] + ')');
    });
    x.fillStyle = g;
    x.fillRect(0, 0, 256, 256);
  });
}

// ── Geometry ─────────────────────────────────────────────────────────────
// A rectangle of terrain [x0, x1] x [z0, z1] with vertex colours.
function terrainRect(x0, x1, z0, z1, cell, height, color, sink) {
  var nx = Math.round((x1 - x0) / cell), nz = Math.round((z1 - z0) / cell);
  var geo = new THREE.PlaneGeometry(x1 - x0, z1 - z0, nx, nz).rotateX(-Math.PI / 2).translate((x0 + x1) / 2, 0, (z0 + z1) / 2);
  var p = geo.attributes.position, uv = geo.attributes.uv, cols = new Float32Array(p.count * 3);
  for (var i = 0; i < p.count; i++) {
    var x = p.getX(i), z = p.getZ(i), y = height(x, z), c = color(x, z, y);
    p.setY(i, y - (sink ? sink(x, z) : 0));
    uv.setXY(i, x / 9, z / 9);
    cols[i * 3] = c.r; cols[i * 3 + 1] = c.g; cols[i * 3 + 2] = c.b;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  geo.computeVertexNormals();
  return geo;
}

// The lane surface: a strip along the road points, level across.
function roadGeometry(width) {
  var pos = [], uv = [], idx = [], along = 0;
  for (var i = 0; i <= NR; i++) {
    var p = roadPts[i], q = roadPts[Math.min(i + 1, NR)], o = roadPts[Math.max(i - 1, 0)];
    var dx = q.x - o.x, dz = q.z - o.z, l = Math.hypot(dx, dz) || 1, nx = -dz / l, nz = dx / l;
    if (i) along += p.distanceTo(roadPts[i - 1]);
    var y = ground(p.x, p.z) + 0.05;
    pos.push(p.x - nx * width / 2, y, p.z - nz * width / 2, p.x + nx * width / 2, y, p.z + nz * width / 2);
    uv.push(0, along / 7, 1, along / 7);
    if (i < NR) idx.push(i * 2, i * 2 + 1, i * 2 + 2, i * 2 + 1, i * 2 + 3, i * 2 + 2);
  }
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  // Make every normal face up (the strip winds both ways round hairpins).
  var n = geo.attributes.normal;
  for (i = 0; i < n.count; i++) if (n.getY(i) < 0) n.setXYZ(i, -n.getX(i), -n.getY(i), -n.getZ(i));
  return geo;
}

// Dry-stone walls: a battered section (wide at the foot, narrow at the
// coping) swept along runs of [x, y, z, jitter] samples.
function WallBuilder() {
  var pos = [], nor = [], uv = [], col = [], c = new THREE.Color();
  function v(x, y, z, nx, ny, nz, u, w) { pos.push(x, y, z); nor.push(nx, ny, nz); uv.push(u, w); col.push(c.r, c.g, c.b); }
  // Corners a, b, c, d; `flip` reverses the winding so the face looks out.
  function quad(a, b, cc, d, n, uvs, flip) {
    if (flip) { quad(b, a, d, cc, n, [uvs[2], uvs[3], uvs[0], uvs[1], uvs[6], uvs[7], uvs[4], uvs[5]]); return; }
    v(a[0], a[1], a[2], n[0], n[1], n[2], uvs[0], uvs[1]); v(b[0], b[1], b[2], n[0], n[1], n[2], uvs[2], uvs[3]);
    v(cc[0], cc[1], cc[2], n[0], n[1], n[2], uvs[4], uvs[5]);
    v(a[0], a[1], a[2], n[0], n[1], n[2], uvs[0], uvs[1]); v(cc[0], cc[1], cc[2], n[0], n[1], n[2], uvs[4], uvs[5]);
    v(d[0], d[1], d[2], n[0], n[1], n[2], uvs[6], uvs[7]);
  }
  this.run = function (s, H, tint) {
    if (s.length < 2) return;
    var WB = 0.4, WT = 0.2, along = 0;
    for (var i = 0; i < s.length - 1; i++) {
      var a = s[i], b = s[i + 1], dx = b[0] - a[0], dz = b[2] - a[2], len = Math.hypot(dx, dz) || 1;
      var nx = -dz / len, nz = dx / len, u0 = along / 2, u1 = (along + len) / 2;
      var ta = a[1] + H + a[3], tb = b[1] + H + b[3], ba = a[1] - 0.35, bb = b[1] - 0.35;
      c.copy(tint).multiplyScalar(0.88 + hash2(a[0], a[2]) * 0.24);
      [1, -1].forEach(function (sd) {
        var n = [nx * sd * 0.99, 0.14, nz * sd * 0.99];
        quad([a[0] + nx * WB * sd, ba, a[2] + nz * WB * sd], [b[0] + nx * WB * sd, bb, b[2] + nz * WB * sd],
             [b[0] + nx * WT * sd, tb, b[2] + nz * WT * sd], [a[0] + nx * WT * sd, ta, a[2] + nz * WT * sd], n,
             [u0, 0, u1, 0, u1, 0.86, u0, 0.86], sd < 0);
      });
      quad([a[0] - nx * WT, ta, a[2] - nz * WT], [b[0] - nx * WT, tb, b[2] - nz * WT],
           [b[0] + nx * WT, tb + 0.06, b[2] + nz * WT], [a[0] + nx * WT, ta + 0.06, a[2] + nz * WT], [0, 1, 0],
           [u0, 0.9, u1, 0.9, u1, 0.98, u0, 0.98], true);
      along += len;
    }
    // Ends: a squared-off wall head.
    [[s[0], s[1]], [s[s.length - 1], s[s.length - 2]]].forEach(function (e) {
      var p = e[0], q = e[1], dx = p[0] - q[0], dz = p[2] - q[2], l = Math.hypot(dx, dz) || 1, ex = dx / l, ez = dz / l;
      var nx = -ez, nz = ex, top = p[1] + H + p[3], bot = p[1] - 0.35;
      quad([p[0] - nx * WB, bot, p[2] - nz * WB], [p[0] + nx * WB, bot, p[2] + nz * WB],
           [p[0] + nx * WT, top, p[2] + nz * WT], [p[0] - nx * WT, top, p[2] - nz * WT], [ex, 0.1, ez],
           [0, 0, 0.3, 0, 0.3, 0.86, 0, 0.86], true);
    });
  };
  this.build = function () {
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    return geo;
  };
}

// A ewe: fleece, a dark face and legs; `grazing` lowers the head.
function sheepGeometry(grazing) {
  var dark = '#2b2623', parts = [
    tinted(new THREE.IcosahedronGeometry(0.5, 2).scale(0.74, 0.7, 1.15).translate(0, 0.74, 0), '#ffffff'),
    tinted(new THREE.IcosahedronGeometry(0.2, 1).scale(1, 1, 1).translate(0, 0.98, 0.42), '#ffffff')
  ];
  var head = new THREE.BoxGeometry(0.19, 0.22, 0.34);
  if (grazing) parts.push(tinted(head.rotateX(1.05).translate(0, 0.4, 0.7), dark));
  else parts.push(tinted(head.rotateX(0.35).translate(0, 0.95, 0.68), dark));
  [-1, 1].forEach(function (s) {
    parts.push(tinted(new THREE.BoxGeometry(0.16, 0.05, 0.08).translate(s * 0.16, grazing ? 0.52 : 1.05, grazing ? 0.58 : 0.6), dark));
  });
  [[-0.2, 0.36], [0.2, 0.36], [-0.2, -0.36], [0.2, -0.36]].forEach(function (l) {
    parts.push(tinted(new THREE.CylinderGeometry(0.045, 0.04, 0.5, 5).translate(l[0], 0.25, l[1]), dark));
  });
  return merge(parts);
}

// Like kit merge(), but keeping the uvs (for the barns' stone texture).
function mergeUV(geos) {
  var uv = [];
  geos.forEach(function (g) { Array.prototype.push.apply(uv, g.attributes.uv.array); });
  var out = merge(geos);
  out.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  return out;
}

// A stone field barn with a slate roof and a dark door on its long side.
function barnGeometry() {
  var shape = new THREE.Shape([new THREE.Vector2(-2.6, -0.5), new THREE.Vector2(2.6, -0.5), new THREE.Vector2(2.6, 4.2),
                               new THREE.Vector2(0, 6), new THREE.Vector2(-2.6, 4.2)]);
  var body = new THREE.ExtrudeGeometry(shape, { depth: 8, bevelEnabled: false }).translate(0, 0, -4).rotateY(Math.PI / 2);
  var parts = [tinted(body, '#ffffff')];
  [-1, 1].forEach(function (s) {
    parts.push(tinted(new THREE.BoxGeometry(8.6, 0.16, 3.4).rotateX(s * 0.605).translate(0, 5.12, s * 1.36), '#4a4c50'));
  });
  parts.push(tinted(new THREE.BoxGeometry(2.2, 2.6, 0.1).translate(0.6, 0.8, 2.62), '#2a2826'));
  parts.push(tinted(new THREE.BoxGeometry(0.5, 0.5, 0.1).translate(-2.4, 2.6, 2.62), '#2a2826'));
  return mergeUV(parts);
}

// A tuft of grass or rushes: thin blades, darker at the root.
function tuftGeometry(r, blades, hgt) {
  var pos = [], nor = [], col = [], root = new THREE.Color('#56653a'), tip = new THREE.Color('#d8dcb0');
  for (var i = 0; i < blades; i++) {
    var a = r() * 6.28, w = 0.012 + r() * 0.012, h = hgt * (0.6 + r() * 0.6), lean = 0.04 + r() * 0.14;
    var ox = Math.cos(a) * 0.05, oz = Math.sin(a) * 0.05, px = -Math.sin(a) * w, pz = Math.cos(a) * w;
    // Both windings, so neither side is lit as a back face.
    pos.push(ox - px, 0, oz - pz, ox + px, 0, oz + pz, ox + Math.cos(a) * lean, h, oz + Math.sin(a) * lean,
             ox + px, 0, oz + pz, ox - px, 0, oz - pz, ox + Math.cos(a) * lean, h, oz + Math.sin(a) * lean);
    nor.push(0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0);
    col.push(root.r, root.g, root.b, root.r, root.g, root.b, tip.r, tip.g, tip.b, root.r, root.g, root.b, root.r, root.g, root.b, tip.r, tip.g, tip.b);
  }
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  return geo;
}

// Sunbeams: soft ribbons from the cloud down along the light to a point on
// the ground, turned in the shader to face the camera.
function beamGeometry(targets) {
  var start = [], len = [], wid = [], corner = [], str = [], idx = [], n = 0;
  targets.forEach(function (t) {
    var tY = fell(t[0], t[1]), s = (CLOUD_Y - tY) / SUN_DIR.y;
    var sx = t[0] + SUN_DIR.x * s, sy = CLOUD_Y, sz = t[1] + SUN_DIR.z * s;
    [[0, -1], [0, 1], [1, -1], [1, 1]].forEach(function (c) {
      start.push(sx, sy, sz); len.push(s); wid.push(t[2]); corner.push(c[0], c[1]); str.push(t[3]);
    });
    idx.push(n, n + 2, n + 1, n + 1, n + 2, n + 3);
    n += 4;
  });
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(start, 3));
  geo.setAttribute('aLen', new THREE.Float32BufferAttribute(len, 1));
  geo.setAttribute('aWidth', new THREE.Float32BufferAttribute(wid, 1));
  geo.setAttribute('corner', new THREE.Float32BufferAttribute(corner, 2));
  geo.setAttribute('aStr', new THREE.Float32BufferAttribute(str, 1));
  geo.setIndex(idx);
  return geo;
}

// ── Sound: a skylark goes up when the sun comes out ─────────────────────
function larkSong(ac, out) {
  var now = ac.currentTime, t = now + 0.1;
  for (var i = 0; i < 46; i++) {
    var o = ac.createOscillator(), g = ac.createGain(), f = 2600 + Math.random() * 2600, d = 0.03 + Math.random() * 0.06;
    o.type = 'sine';
    o.frequency.setValueAtTime(f, t);
    o.frequency.exponentialRampToValueAtTime(f * (Math.random() < 0.5 ? 1.35 : 0.72), t + d);
    var v = 0.022 * (0.5 + 0.5 * Math.sin(i / 46 * Math.PI));
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(v, t + 0.008);
    g.gain.exponentialRampToValueAtTime(0.0001, t + d);
    o.connect(g);
    g.connect(out);
    o.start(t);
    o.stop(t + d + 0.02);
    t += d + 0.015 + Math.random() * (i % 7 === 6 ? 0.25 : 0.04);
  }
}

// ── Renderer ─────────────────────────────────────────────────────────────
function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(41);
  var GREY = new THREE.Color('#a4aaae'), GOLDEN = new THREE.Color('#e2bd84');
  var gl = makeRenderer(canvas, { clear: GREY });
  var world = new THREE.Scene();
  world.fog = new THREE.Fog(GREY.clone(), 10, 300);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.1, 9000);

  // Shared by every sun-lit material and the cloud deck.
  var U = {
    uTime: { value: 0 }, uCover: { value: 1 }, uGapC: { value: new THREE.Vector2() }, uGapR: { value: 200 },
    uGap: { value: 0 }, uHoleAmt: { value: 0 }, uSunDir: { value: SUN_DIR.clone() },
    uHoles: { value: [new THREE.Vector4(), new THREE.Vector4(), new THREE.Vector4()] },
    uSunV: { value: new THREE.Vector3() }, uSunCol: { value: new THREE.Color() },
    uSheen: { value: new THREE.Color() }, uWet: { value: 1 }
  };
  var CLOUDS = cloudGLSL(small ? 4 : 5);

  // ── Sky ──
  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#7b838b', mid: '#8f969c', horizon: '#a9aeb2', sun: '#fff0d0' }, 5000);
  dome.uniforms.sunDir.value.copy(SUN_DIR);
  // Tone-map and encode the dome like the lit world, so it meets the fog.
  dome.mesh.material.fragmentShader = dome.mesh.material.fragmentShader.replace(/\}\s*$/,
    '\n#include <tonemapping_fragment>\n#include <colorspace_fragment>\n}');
  sky.add(dome.mesh);

  // The cloud deck: dark and heavy where thick, pale where thin, edged
  // with silver near the sun; it fades into the haze towards the horizon.
  var deckMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, side: THREE.DoubleSide, fog: false,
    uniforms: Object.assign({
      uDark: { value: new THREE.Color() }, uLight: { value: new THREE.Color() }, uSilver: { value: new THREE.Color('#fbfaf4') },
      uSilverAmt: { value: 0 }, uHorizon: { value: new THREE.Color() }, uHaze: { value: new THREE.Vector2(300, 2500) }
    }, U),
    vertexShader: 'varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }',
    fragmentShader: CLOUDS +
      'uniform vec3 uDark; uniform vec3 uLight; uniform vec3 uSilver; uniform float uSilverAmt; uniform vec3 uHorizon; uniform vec2 uHaze; varying vec3 vW;\n' +
      'void main(){\n' +
      ' float d = cloudD(vW.xz);\n' +
      ' float d2 = cloudD(vW.xz + uSunDir.xz * 60.0);\n' +
      ' vec3 V = normalize(vW - cameraPosition);\n' +
      ' float s = max(dot(V, uSunDir), 0.0);\n' +
      ' vec3 col = mix(uLight, uDark, smoothstep(0.25, 1.0, d));\n' +
      ' col *= 1.0 - clamp((d2 - d) * 0.9, -0.25, 0.3);\n' +
      ' float edge = smoothstep(0.02, 0.22, d) * (1.0 - smoothstep(0.25, 0.8, d));\n' +
      ' col += uSilver * edge * (pow(s, 5.0) * 1.6 + pow(s, 60.0) * 3.0) * uSilverAmt;\n' +
      ' col += uSilver * pow(s, 20.0) * (1.0 - smoothstep(0.3, 1.1, d)) * 0.6 * uSilverAmt;\n' +
      ' float dist = length(vW.xz - cameraPosition.xz);\n' +
      ' float haze = smoothstep(uHaze.x, uHaze.y, dist);\n' +
      ' col = mix(col, uHorizon, haze);\n' +
      ' float a = smoothstep(0.04, 0.3, d);\n' +
      ' a = mix(a, 1.0, haze * 0.6) * (1.0 - smoothstep(3200.0, 4400.0, dist));\n' +
      ' gl_FragColor = vec4(col, a);\n' +
      ' #include <tonemapping_fragment>\n #include <colorspace_fragment>\n' +
      '}'
  });
  var deck = new THREE.Mesh(new THREE.PlaneGeometry(9000, 9000).rotateX(Math.PI / 2), deckMat);
  deck.frustumCulled = false;
  deck.renderOrder = 1;
  world.add(deck);

  // The sun's glare, over the cloud: a silver glow while it is hidden,
  // the full sun when it is out.
  var glare = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture(), color: '#fff6e4', transparent: true,
    blending: THREE.AdditiveBlending, depthWrite: false, fog: false, opacity: 0 }));
  glare.renderOrder = 3;
  world.add(glare);

  // ── Light ──
  var hemi = new THREE.HemisphereLight('#c8ced2', '#56583e', 2.2);
  world.add(hemi);

  // ── The land ──
  var tmp = new THREE.Color(), tmp2 = new THREE.Color();
  var PASTURE = ['#56782c', '#638530', '#4f6e2a', '#6e8c38', '#5c7d33', '#7e9442', '#4a6a2e', '#688a3a']
    .map(function (h) { return new THREE.Color(h); });
  var MOOR = [new THREE.Color('#596634'), new THREE.Color('#6a6b3b'), new THREE.Color('#52602f'), new THREE.Color('#574a45')];
  var ROUGH = new THREE.Color('#5f7034'), LIME = new THREE.Color('#b9b7aa'), VERGE = new THREE.Color('#4e6328');
  function landColor(x, z, y) {
    var f = fieldOf(x, z), n = 0.5 + 0.5 * Math.sin(x * 0.07 + Math.sin(z * 0.05) * 2) * Math.cos(z * 0.06 - x * 0.03);
    if (f < 0) {
      var k = Math.floor(n * 2.999);
      tmp.copy(MOOR[k]).lerp(MOOR[k + 1], n * 3 - k);
      // Heather in patches, and pale limestone scars on the far fell edges.
      if (Math.sin(x * 0.013 + z * 0.021) * Math.sin(x * 0.031 - z * 0.017) > 0.45) tmp.lerp(MOOR[3], 0.6);
      var scar = smooth(3, 9, y) * (1 - smooth(14, 22, y)) * (z < -700 ? 1 : 0) * smooth(0.2, 0.7, Math.sin(x * 0.011) * 0.5 + 0.5 + Math.sin(x * 0.047) * 0.3);
      tmp.lerp(LIME, scar * 0.85);
    } else if (f === 777) {
      tmp.copy(ROUGH).lerp(MOOR[0], 0.25 + n * 0.25);
    } else {
      tmp.copy(PASTURE[Math.floor(hash2(f, 3.7) * PASTURE.length)]);
      tmp.multiplyScalar(0.9 + n * 0.18);
      if (Math.abs(z - BECK_Z(x)) < 60) tmp.lerp(tmp2.set('#4a7a2a'), 0.4);
    }
    roadNear(x, z, RN);
    if (RN.d < 5) tmp.lerp(VERGE, 1 - smooth(2.2, 5, RN.d));
    return tmp;
  }
  var groundMat = sunlit(new THREE.MeshLambertMaterial({ vertexColors: true, map: grainTexture(r) }), U, CLOUDS);
  var NEAR = { x0: -270, x1: 250, z0: -480, z1: 180 };
  var near = new THREE.Mesh(terrainRect(NEAR.x0, NEAR.x1, NEAR.z0, NEAR.z1, small ? 3 : 2, ground, landColor), groundMat);
  world.add(near);
  // The wider land, coarser, sunk out of sight under the near patch.
  var farMat = sunlit(new THREE.MeshLambertMaterial({ vertexColors: true, map: groundMat.map,
    polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 2 }), U, CLOUDS);
  world.add(new THREE.Mesh(terrainRect(-2510, 2490, -3380, 1620, 20, fell, landColor, function (x, z) {
    return x > NEAR.x0 + 1 && x < NEAR.x1 - 1 && z > NEAR.z0 + 1 && z < NEAR.z1 - 1 ? 4 : 0;
  }), farMat));

  // The lane.
  var roadTex = roadTexture(r);
  var roadMat = sunlit(new THREE.MeshLambertMaterial({ map: roadTex, color: '#d8d8d8' }), U, CLOUDS, true);
  world.add(new THREE.Mesh(roadGeometry(3.4), roadMat));

  // A beck down the far dale, catching the sky.
  var beckPts = [];
  for (var bx = -2000; bx <= 2000; bx += 8) beckPts.push(new THREE.Vector3(bx, 0, BECK_Z(bx)));
  var beckGeo = (function () {
    var p = [], idx = [];
    beckPts.forEach(function (q, i) {
      var y = fell(q.x, q.z) + 0.9;
      p.push(q.x, y, q.z - 7, q.x, y, q.z + 7);
      if (i < beckPts.length - 1) idx.push(i * 2, i * 2 + 1, i * 2 + 2, i * 2 + 1, i * 2 + 3, i * 2 + 2);
    });
    var g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(p, 3));
    g.setIndex(idx);
    return g;
  })();
  var beckMat = new THREE.MeshBasicMaterial({ color: '#a9b0b4' });
  world.add(new THREE.Mesh(beckGeo, beckMat));

  // ── Dry-stone walls ──
  var wb = new WallBuilder(), STONE = new THREE.Color('#d6d3c8'), WALL_H = 1.25;
  var nearStep = small ? 2 : 1.2, farStep = small ? 6 : 3.5;
  function stepAt(x, z) { return Math.abs(x + 10) < 330 && z > -520 && z < 260 ? nearStep : farStep; }
  // Sample a wall along a line function, breaking it where the lane passes
  // (it butts into the lane walls) and outside the land.
  function wallAlong(fn, from, to, gap) {
    var run = [], s = from, dir = to > from ? 1 : -1;
    while ((to - s) * dir > 0) {
      var p = fn(s), x = p[0], z = p[1];
      roadNear(x, z, RN);
      if (RN.d < (gap || 3.35) || p[2] === false) { wb.run(run, WALL_H, STONE); run = []; }
      else run.push([x, ground(x, z), z, (hash2(x * 3.1, z * 1.7) - 0.5) * 0.12]);
      s += dir * stepAt(x, z) * 0.98;
    }
    wb.run(run, WALL_H, STONE);
  }
  ACROSS_NEAR.forEach(function (z0, j) {
    wallAlong(function (x) {
      var z = acrossZ(z0, j, x);
      return [x, z, !(z0 <= -100 && Math.abs(x + 10) < 85) && !(z > 200 && fell(x, z) > 46)];
    }, -1100, 1100);
  });
  UP_NEAR.forEach(function (x0, k) {
    wallAlong(function (z) { var x = upX(x0, k, z); return [x, z, z > HEAD(x) && !(z > 200 && fell(x, z) > 46)]; }, 420, -200);
  });
  wallAlong(function (x) { return [x, HEAD(x)]; }, -1100, 1100, 2.35);
  wallAlong(function (x) { return [x, FARTOP(x)]; }, -1300, 1300, 2.35);
  ACROSS_FAR.forEach(function (z0, j) {
    wallAlong(function (x) { var z = acrossZ(z0, j + 20, x); return [x, z, fell(x, z) < 22]; }, -1300, 1300);
  });
  UP_FAR.forEach(function (x0, k) {
    wallAlong(function (z) { var x = upX(x0, k + 20, z); return [x, z, z < FARTOP(x) && fell(x, z) < 22]; }, -280, -960);
  });
  // The lane's own walls, both sides, up to the gate and again below the
  // far top wall; open road over the moor between.
  [1, -1].forEach(function (sd) {
    var run = [];
    for (var i = 0; i <= NR; i++) {
      var p = roadPts[i], q = roadPts[Math.min(i + 1, NR)], o = roadPts[Math.max(i - 1, 0)];
      var dx = q.x - o.x, dz = q.z - o.z, l = Math.hypot(dx, dz) || 1;
      var x = p.x - dz / l * 3.1 * sd, z = p.z + dx / l * 3.1 * sd;
      roadNear(x, z, RN);
      var ok = RN.d > 2.9 && (z > acrossZ(-66, 8, x) + 0.5 || z < FARTOP(x) - 0.5);
      if (!ok) { wb.run(run, 1.15, STONE); run = []; continue; }
      if (i % (small ? 2 : 1) === 0) run.push([x, ground(x, z), z, (hash2(x * 3.1, z * 1.7) - 0.5) * 0.1]);
    }
    wb.run(run, 1.15, STONE);
  });
  var stoneTex = stoneTexture(r);
  stoneTex.wrapT = THREE.ClampToEdgeWrapping;
  var wallMat = sunlit(new THREE.MeshLambertMaterial({ map: stoneTex, vertexColors: true }), U, CLOUDS);
  world.add(new THREE.Mesh(wb.build(), wallMat));

  // ── Field barns, sheep, thorn trees, grass and rushes ──
  var up = new THREE.Vector3(0, 1, 0), m4 = new THREE.Matrix4(), q4 = new THREE.Quaternion(), s4 = new THREE.Vector3(), p4 = new THREE.Vector3();
  var BARNS = [[-34, 4, 0.05], [44, -44, 1.6], [-120, 60, 0.1], [118, 24, 1.5], [-70, -48, 1.62], [180, -96, 0.08],
               [-200, -10, 1.55], [260, 120, 0.02], [-300, 150, 1.6], [130, -380, 0.1], [-240, -430, 1.5], [330, -470, 0.05],
               [-80, -540, 1.58]];
  var village = [[150, -655], [172, -660], [190, -650], [160, -675], [205, -668], [138, -640], [182, -684], [221, -656]];
  var barnTex = rubbleTexture(r);
  barnTex.repeat.set(0.45, 0.45);
  var barns = new THREE.InstancedMesh(barnGeometry(), sunlit(new THREE.MeshLambertMaterial({ vertexColors: true, map: barnTex }), U, CLOUDS),
                                      BARNS.length + village.length);
  BARNS.concat(village.map(function (v, i) { return [v[0], v[1], 0.3 * Math.sin(i * 2.3), 0.8]; })).forEach(function (b, i) {
    p4.set(b[0], fell(b[0], b[1]) - 0.2, b[1]);
    q4.setFromAxisAngle(up, b[2]);
    s4.setScalar(b[3] || 1);
    barns.setMatrixAt(i, m4.compose(p4, q4, s4));
    barns.setColorAt(i, tmp.set(i % 3 ? '#a8a291' : '#9b9688').multiplyScalar(0.92 + hash2(i, 1) * 0.16));
  });
  world.add(barns);
  // The village church tower.
  var tower = new THREE.Mesh(new THREE.BoxGeometry(5, 15, 5).translate(0, 7, 0), barns.material);
  tower.position.set(176, fell(176, -638), -638);
  world.add(tower);

  function inPasture(x, z) { var f = fieldOf(x, z); return f >= 0; }
  var sheepMat = sunlit(new THREE.MeshLambertMaterial({ vertexColors: true }), U, CLOUDS);
  var SHEEP = small ? 180 : 460;
  [sheepGeometry(false), sheepGeometry(true)].forEach(function (geo, gi) {
    var flock = new THREE.InstancedMesh(geo, sheepMat, SHEEP / 2);
    scatter(flock, 20000, function (i, p, q, s, c) {
      var x, z;
      if (i < 8 && gi === 0) {                          // a few idling on the lane above the second hairpin
        var ai = Math.round(T_H2 * NR) + 22 + i * 4, a = roadPts[ai], an = roadPts[ai + 1], al = a.distanceTo(an), sd = i % 2 ? 1 : -1;
        var off = sd * (2.0 + r() * 0.7); x = a.x - (an.z - a.z) / al * off; z = a.z + (an.x - a.x) / al * off;
      } else if (r() < 0.6) {                           // most near the lane
        var b = roadPts[Math.floor(r() * NR * 0.7)]; x = b.x + (r() - 0.5) * 160; z = b.z + (r() - 0.5) * 120;
      } else { x = (r() - 0.5) * 1300; z = 300 - r() * 1060; }
      if (!(i < 8 && gi === 0)) {
        roadNear(x, z, RN);
        if (RN.d < 4 || !inPasture(x, z)) return false;
      }
      p.set(x, ground(x, z) - 0.04, z);
      q.setFromAxisAngle(up, r() * 6.28);
      s.setScalar(0.9 + r() * 0.25);
      c.set(r() < 0.15 ? '#cfc6ae' : '#eee9dc').multiplyScalar(0.9 + r() * 0.12);
    });
    world.add(flock);
  });

  var treeMat = sunlit(new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true }), U, CLOUDS);
  var thorn = new THREE.InstancedMesh(broadleafGeometry(rng(7), '#3e342c'), treeMat, small ? 220 : 480);
  var lean = new THREE.Quaternion(), LEAN_AXIS = new THREE.Vector3(0.3, 0, 1).normalize(), CROWN = ['#3d5428', '#4a5f2c', '#35482a', '#53662f'];
  scatter(thorn, 6000, function (i, p, q, s, c) {
    var x, z, sc;
    if (i < 40) {                                       // thorns by the walls near the lane
      var k = Math.floor(r() * UP_NEAR.length), zz = 120 - r() * 300; z = zz; x = upX(UP_NEAR[k], k, zz) + (r() < 0.5 ? 1.4 : -1.4);
      if (Math.abs(x) > 400 || z < HEAD(x) + 4) return false;
      sc = 0.5 + r() * 0.3;
    } else if (i < 60) {                                // by the barns
      var b = BARNS[Math.floor(r() * BARNS.length)], ba = r() * 6.28, bd = 8 + r() * 10; x = b[0] + Math.cos(ba) * bd; z = b[1] + Math.sin(ba) * bd;
      sc = 0.8 + r() * 0.5;
    } else if (r() < 0.65) {                            // along the beck in the far dale
      x = (r() - 0.5) * 2400; z = BECK_Z(x) + (r() - 0.5) * 40;
      sc = 0.9 + r() * 0.6;
    } else {                                            // field trees down the far dale
      x = (r() - 0.5) * 1600; z = -300 - r() * 560;
      if (fieldOf(x, z) < 0) return false;
      sc = 0.9 + r() * 0.7;
    }
    roadNear(x, z, RN);
    if (RN.d < 5) return false;
    p.set(x, ground(x, z) - 0.2, z);
    q.setFromAxisAngle(up, r() * 6.28);
    if (i < 40) q.premultiply(lean.setFromAxisAngle(LEAN_AXIS, -0.25));
    s.set(sc, sc * (0.8 + r() * 0.3), sc);
    c.set(CROWN[Math.floor(r() * CROWN.length)]);
  });
  world.add(thorn);

  // Grass on the verges, rushes in the rough pasture and on the moor.
  var tuftMat = sunlit(new THREE.MeshLambertMaterial({ vertexColors: true }), U, CLOUDS);
  var grass = new THREE.InstancedMesh(tuftGeometry(rng(3), 8, 0.32), tuftMat, small ? 3500 : 9000);
  scatter(grass, 40000, function (i, p, q, s, c) {
    var a = roadPts[Math.floor(r() * NR * 0.6)], side = r() < 0.5 ? 1 : -1, x = a.x + side * (1.9 + r() * 0.7) + (r() - 0.5) * 0.6, z = a.z + (r() - 0.5) * 2;
    if (r() < 0.35) { x = a.x + (r() - 0.5) * 40; z = a.z + (r() - 0.5) * 40; if (fieldOf(x, z) !== 777) return false; }
    roadNear(x, z, RN);
    if (RN.d < 1.85 || Math.abs(RN.i - T_BENCH * NR) < 12) return false;
    p.set(x, ground(x, z) - 0.02, z);
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar(0.8 + r() * 0.8);
    c.set(fieldOf(x, z) === 777 ? '#9c9e62' : '#93b25c').multiplyScalar(0.85 + r() * 0.3);
  });
  world.add(grass);
  var rushes = new THREE.InstancedMesh(tuftGeometry(rng(5), 14, 0.75), tuftMat, small ? 1600 : 4200);
  scatter(rushes, 30000, function (i, p, q, s, c) {
    var x = -95 + r() * 170, z = -66 - r() * 260;
    if (i % 3 === 0) { x = -40 + r() * 80; z = -195 - r() * 70; }
    var f = fieldOf(x, z);
    if (f !== 777 && f !== -1) return false;
    roadNear(x, z, RN);
    if (RN.d < 2.4) return false;
    p.set(x, ground(x, z) - 0.03, z);
    q.setFromAxisAngle(up, r() * 6.28);
    s.set(0.8 + r() * 0.7, 0.7 + r() * 0.7, 0.8 + r() * 0.7);
    c.set('#748a44').multiplyScalar(0.8 + r() * 0.4);
  });
  world.add(rushes);

  // Grey limestone boulders out on the moor.
  var rockGeo = new THREE.IcosahedronGeometry(0.7, 1);
  (function () {
    var p = rockGeo.attributes.position, v = new THREE.Vector3();
    for (var i = 0; i < p.count; i++) {
      v.fromBufferAttribute(p, i);
      var k = 1 + 0.22 * Math.sin(v.x * 5.1 + 1) * Math.cos(v.z * 4.3) + 0.12 * Math.sin(v.y * 7.7);
      p.setXYZ(i, v.x * k * 1.3, v.y * k * 0.62, v.z * k);
    }
    rockGeo.computeVertexNormals();
  })();
  var rocks = new THREE.InstancedMesh(rockGeo, sunlit(new THREE.MeshLambertMaterial({ flatShading: true }), U, CLOUDS), small ? 50 : 110);
  scatter(rocks, 4000, function (i, p, q, s, c) {
    var x = -160 + r() * 300, z = -190 - r() * 100;
    if (fieldOf(x, z) !== -1) return false;
    roadNear(x, z, RN);
    if (RN.d < 4) return false;
    p.set(x, ground(x, z) - 0.15, z);
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar(0.35 + Math.pow(r(), 2.5) * 1.3);
    c.set('#8e8c85').multiplyScalar(0.85 + r() * 0.25);
  });
  world.add(rocks);

  // ── "Rest if you must": a bench and a milestone by the wall ──
  var stoneMat = sunlit(new THREE.MeshLambertMaterial({ color: '#9d998d' }), U, CLOUDS);
  var woodMat = sunlit(new THREE.MeshLambertMaterial({ color: '#8c765e' }), U, CLOUDS);
  function onVerge(t, off) {
    var i = Math.round(t * NR), p = roadPts[i], q = roadPts[Math.min(i + 2, NR)], dx = q.x - p.x, dz = q.z - p.z, l = Math.hypot(dx, dz) || 1;
    var x = p.x - dz / l * off, z = p.z + dx / l * off;
    return { x: x, z: z, y: ground(x, z), ang: Math.atan2(dx, dz) };
  }
  var bench = new THREE.Group(), bp = onVerge(T_BENCH, 2.45);
  [-0.75, 0.75].forEach(function (s) {
    bench.add(new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.75, 0.4).translate(s, 0.07, 0), stoneMat));
    bench.add(new THREE.Mesh(new THREE.BoxGeometry(0.08, 1.25, 0.08).translate(s, 0.32, -0.2), woodMat));
  });
  bench.add(new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.07, 0.42).translate(0, 0.47, 0), woodMat));
  bench.add(new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.14, 0.04).translate(0, 0.78, -0.21), woodMat));
  bench.add(new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.1, 0.04).translate(0, 0.62, -0.21), woodMat));
  bench.position.set(bp.x, bp.y, bp.z);
  bench.rotation.y = bp.ang + Math.PI / 2;
  world.add(bench);
  var mp = onVerge(T_BENCH + 0.005, 2.3);
  var faceMat = sunlit(new THREE.MeshLambertMaterial({ map: milestoneTexture() }), U, CLOUDS);
  var mile = new THREE.Group();
  mile.add(new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.66, 0.24).translate(0, 0.33, 0), [stoneMat, stoneMat, stoneMat, stoneMat, faceMat, stoneMat]));
  mile.add(new THREE.Mesh(new THREE.CylinderGeometry(0.23, 0.23, 0.24, 12, 1, false, 0, Math.PI).rotateX(Math.PI / 2).rotateZ(Math.PI / 2).translate(0, 0.66, 0), stoneMat));
  mile.position.set(mp.x, mp.y - 0.05, mp.z);
  mile.rotation.y = mp.ang + Math.PI - 0.7;
  world.add(mile);

  // ── The gate in the top wall, swung open ──
  var gp = onVerge(T_GATE, 0), gate = new THREE.Group();
  [-2.1, 2.1].forEach(function (s) {
    gate.add(new THREE.Mesh(new THREE.BoxGeometry(0.34, 1.6, 0.34).translate(s, 0.8, 0), stoneMat));
  });
  var leaf = new THREE.Group();
  for (var bar = 0; bar < 5; bar++) leaf.add(new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.08, 0.05).translate(1.7, 0.3 + bar * 0.24, 0), woodMat));
  [0.05, 1.7, 3.35].forEach(function (s) { leaf.add(new THREE.Mesh(new THREE.BoxGeometry(0.08, 1.12, 0.06).translate(s, 0.82, 0), woodMat)); });
  leaf.add(new THREE.Mesh(new THREE.BoxGeometry(3.6, 0.07, 0.05).rotateZ(0.3).translate(1.7, 0.8, 0), woodMat));
  leaf.position.set(-2.0, 0, 0);
  leaf.rotation.y = Math.PI * 0.5 + 0.12;
  gate.add(leaf);
  gate.position.set(gp.x, gp.y, gp.z);
  gate.rotation.y = gp.ang - Math.PI / 2;
  world.add(gate);

  // ── Sunbeams through the gaps: [x, z, width, strength] on the ground ──
  var BEAM_T = [[90, -420, 26, 1], [-150, -560, 30, 0.9], [-20, -700, 34, 1]];
  for (var bi = 0; bi < 15; bi++) {
    BEAM_T.push([-480 + bi * 60 + (r() - 0.5) * 40, -300 - r() * 600, 7 + r() * 16, 0.45 + r() * 0.55]);
  }
  var beamMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false, side: THREE.DoubleSide,
    uniforms: { uSunDir: U.uSunDir, uAmt: { value: 0 }, uCol: { value: new THREE.Color('#ffd89a') } },
    vertexShader: 'attribute float aLen; attribute float aWidth; attribute vec2 corner; attribute float aStr;\n' +
      'uniform vec3 uSunDir; varying vec2 vC; varying float vS;\n' +
      'void main(){ vec3 A = -uSunDir; vec3 c = position + A * corner.x * aLen;\n' +
      ' vec3 side = normalize(cross(A, normalize(cameraPosition - c)));\n' +
      ' vec3 p = c + side * corner.y * aWidth * (1.0 + corner.x * 0.5);\n' +
      ' vC = corner; vS = aStr * (0.35 + 0.65 * pow(max(dot(normalize(c - cameraPosition), uSunDir), 0.0), 3.0));\n' +
      ' gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0); }',
    fragmentShader: 'uniform float uAmt; uniform vec3 uCol; varying vec2 vC; varying float vS;\n' +
      'void main(){ float e = 1.0 - vC.y * vC.y; e *= e;\n' +
      ' float a = e * smoothstep(0.0, 0.15, vC.x) * (1.0 - smoothstep(0.55, 1.0, vC.x)) * vS * uAmt * 0.17;\n' +
      ' gl_FragColor = vec4(uCol, a);\n #include <tonemapping_fragment>\n #include <colorspace_fragment>\n}'
  });
  var beams = new THREE.Mesh(beamGeometry(BEAM_T), beamMat);
  beams.frustumCulled = false;
  beams.renderOrder = 2;
  world.add(beams);
  // Holes in the cloud above three of the far beams.
  [0, 1, 2].forEach(function (b, k) {
    var t = BEAM_T[b], s = (CLOUD_Y - fell(t[0], t[1])) / SUN_DIR.y;
    U.uHoles.value[k].set(t[0] + SUN_DIR.x * s, t[1] + SUN_DIR.z * s, 170 + k * 40, 0);
  });

  // ── Drizzle, and motes in the sunbeam at the end ──
  var rain = rainField({ count: small ? 900 : 2200, box: [16, 12, 22], color: '#d5dbe0', opacity: 0.3, speed: 6.5, windSpeed: 2.2 });
  world.add(rain.lines);
  var motes = particleField({ count: small ? 250 : 600, box: [22, 9, 22], fall: [-0.06, 0.08], size: 0.06, color: '#fff0c8',
    map: softSprite('rgba(255,244,214,1)', 'rgba(255,244,214,0)'), sway: 0.2 });
  motes.points.material.blending = THREE.AdditiveBlending;
  world.add(motes.points);
  var mf = { snow: 0, wind: 0, dt: 0, time: 0 };

  // Where the sun's line from the crest meets the cloud: the gap opens to
  // one side of it and slides across.
  var CREST = new THREE.Vector3(0, fell(0, -215) + 1.7, -215);
  var GAP0 = new THREE.Vector2(CREST.x + SUN_DIR.x * (CLOUD_Y - CREST.y) / SUN_DIR.y, CREST.z + SUN_DIR.z * (CLOUD_Y - CREST.y) / SUN_DIR.y);
  var GAP_SIDE = new THREE.Vector2(-SUN_DIR.z, SUN_DIR.x).normalize();

  var C = {
    darkO: new THREE.Color('#4c5258'), darkG: new THREE.Color('#75655c'), lightO: new THREE.Color('#9ba2a8'), lightG: new THREE.Color('#f5d29a'),
    topO: new THREE.Color('#7b838b'), topG: new THREE.Color('#4a7cb6'), midO: new THREE.Color('#8f969c'), midG: new THREE.Color('#a8c2d8'),
    skyO: new THREE.Color('#c8ced2'), skyG: new THREE.Color('#ffdcaa'), sunW: new THREE.Color('#fff2da'), sunG: new THREE.Color('#ffcf82'),
    sheenO: new THREE.Color('#9aa3aa'), sheenG: new THREE.Color('#f4d9a8'), beckO: new THREE.Color('#9da6ac'), beckG: new THREE.Color('#ffe2a8')
  };
  // Object3D.lookAt points +z at the target; cameras look down -z, so the
  // aim is turned round after. `turn` and `tilt` move the subject off centre.
  var aim = new THREE.Object3D(), sunAt = new THREE.Vector3(), portrait = false;
  var OUT_AT = new THREE.Vector3(-8, fell(-8, 10) + 4, 10), OUT2_AT = new THREE.Vector3(-26, fell(-26, -162) + 1, -162), BENCH_AT = new THREE.Vector3(bp.x, bp.y + 1.1, bp.z);
  function aimAt(target, w, turn, tilt, f) {
    if (w <= 0) return;
    aim.position.copy(camera.position);
    aim.lookAt(target);
    aim.rotateY(Math.PI + turn - f.mx * 0.12);
    aim.rotateX(tilt - f.my * 0.05);
    camera.quaternion.slerp(aim.quaternion, w * w * (3 - 2 * w));
  }

  function frame(f) {
    var row = f.row, t = clamp(f.cam, 0, 1), gloom = f.dark, drizzle = f.snow;
    var silver = row[6], sun = row[7], gold = row[8], beamAmt = row[9], lift = row[10];
    var clear = Math.max(sun * 0.8, gold);

    followPath(camera, ROAD, ground, t, {
      eye: 1.65 + lift, ahead: 0.024, yaw: row[4], pitch: row[5], mx: f.mx, my: f.my, time: f.time
    });
    // Turn from the lane to the bench, out over the dale or up the lane at
    // the hairpins, or up to the sky by the sun. On a portrait screen the
    // verse sits mid-screen: keep the bench in shot, lift the sun above it.
    aimAt(BENCH_AT, row[12], portrait ? 0.12 : 0.32, 0, f);
    aimAt(t < (T_H1 + T_H2) / 2 ? OUT_AT : OUT2_AT, row[11], 0, 0, f);
    aimAt(sunAt.copy(camera.position).addScaledVector(SUN_DIR, 1000), row[13], portrait ? 0.15 : -0.2, portrait ? -0.3 : 0.16, f);
    camera.updateMatrixWorld();
    sky.position.copy(camera.position);
    deck.position.set(camera.position.x, CLOUD_Y, camera.position.z);
    glare.position.copy(camera.position).addScaledVector(SUN_DIR, 2000);
    glare.scale.setScalar(420 + sun * 180);
    glare.material.opacity = silver * 0.3 * (1 - sun) + sun * 0.75 * (1 - gloom);

    // The cloud: heavy, then a silver-edged gap by the sun that slides
    // over it, then the whole sky breaking up.
    U.uTime.value = env.reduceMotion ? f.time * 0.3 : f.time;
    U.uCover.value = 0.95 - 0.12 * silver - 0.62 * gold + 0.35 * gloom * silver;
    U.uGap.value = silver;
    U.uGapR.value = 150 + 170 * sun;
    U.uGapC.value.copy(GAP0).addScaledVector(GAP_SIDE, -170 * (1 - sun));
    U.uHoleAmt.value = Math.max(gold, beamAmt * 0.8);
    deckMat.uniforms.uSilverAmt.value = silver;

    // Light and colour: overcast grey to warm gold.
    var sunI = (0.35 * silver + 1.0 * sun + 0.8 * gold) * (1 - gloom * 0.6);
    U.uSunCol.value.copy(C.sunW).lerp(C.sunG, gold).multiplyScalar(sunI * 1.5);
    U.uSunV.value.copy(SUN_DIR).transformDirection(camera.matrixWorldInverse);
    U.uSheen.value.copy(C.sheenO).lerp(C.sheenG, clear);
    U.uWet.value = 0.5 + drizzle * 0.5;
    hemi.color.copy(C.skyO).lerp(C.skyG, clear);
    hemi.intensity = 2.3 - clear * 0.7;
    world.fog.color.copy(GREY).lerp(GOLDEN, clear);
    world.fog.near = lerp(12, 80, clear);
    world.fog.far = lerp(lerp(700, 260, drizzle), 3300, clear);
    gl.setClearColor(world.fog.color);
    dome.uniforms.horizon.value.copy(world.fog.color);
    dome.uniforms.mid.value.copy(C.midO).lerp(C.midG, clear);
    dome.uniforms.top.value.copy(C.topO).lerp(C.topG, clear);
    dome.uniforms.sunColor.value.copy(C.sunW).multiplyScalar(0.3 + silver * 0.7);
    deckMat.uniforms.uDark.value.copy(C.darkO).lerp(C.darkG, gold);
    deckMat.uniforms.uLight.value.copy(C.lightO).lerp(C.lightG, gold);
    deckMat.uniforms.uHorizon.value.copy(world.fog.color);
    deckMat.uniforms.uHaze.value.set(world.fog.far * 0.5, world.fog.far * 2.2);
    beckMat.color.copy(C.beckO).lerp(C.beckG, clear);
    beamMat.uniforms.uAmt.value = beamAmt * (1 - gloom * 0.7);
    gl.toneMappingExposure = 1.0 - gloom * 0.35 + clear * 0.08;

    rain.update(f, camera.position, drizzle, env.reduceMotion);
    rain.lines.material.color.copy(tmp.set('#d5dbe0').lerp(C.sunG, sun));
    mf.dt = f.dt; mf.time = f.time; mf.wind = 0.1; mf.snow = sun * (1 - gloom);
    motes.update(mf, camera.position, env.reduceMotion);

    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) { fitCamera(gl, camera, w, h, dpr, small); portrait = w < h; },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('uphill-road', {
  renderer: renderer3d,
  align: ['left', 'right', 'right'],
  maxLines: 6,
  scrim: 0.62,
  keys: function (T) {
    var s = T.start, B = T_BENCH;
    //  unit          path           gloom drz   wind  yaw   pitch silv sun   gold  beam  lift out   rest  sky
    return [
      [0,             0.000,         0.30, 0.80, 0.35, 0.00, 0.05, 0,   0,    0,    0,    0,   0,    0,    0],
      [0.7,           0.004,         0.30, 0.80, 0.35, 0.00, 0.05, 0,   0,    0,    0,    0,   0,    0,    0],
      [s(0) + 0.25,   0.030,         0.35, 0.90, 0.45, 0.00, 0.07, 0,   0,    0,    0,    0,   0,    0,    0],
      [s(0) + 0.65,   B - 0.024,     0.40, 1.00, 0.50, 0.00, 0.09, 0,   0,    0,    0,    0,   0,    0,    0],  // "seems all up hill"
      [s(0) + 1.05,   B - 0.0045,    0.40, 0.85, 0.40, 0.00, 0.00, 0,   0,    0,    0,    0,   0,    1,    0],  // "rest if you must"
      [s(0) + 1.45,   B - 0.0035,    0.38, 0.80, 0.40, 0.00, 0.00, 0,   0,    0,    0,    0,   0,    1,    0],
      [s(1) + 0.15,   B + 0.030,     0.35, 0.75, 0.40, 0.00, 0.02, 0,   0,    0,    0,    0,   0,    0,    0],
      [s(1) + 0.45,   T_H1 - 0.012,  0.33, 0.65, 0.45, 0.00, 0.00, 0,   0,    0,    0,    0.3, 0.2,  0,    0],
      [s(1) + 0.62,   T_H1 - 0.004,  0.32, 0.60, 0.45, 0.00, 0.00, 0,   0,    0,    0,    0.6, 1,    0,    0],  // "twists and turns": the dale below
      [s(1) + 0.85,   T_H1 + 0.002,  0.31, 0.55, 0.40, 0.00, 0.00, 0,   0,    0,    0,    0.6, 1,    0,    0],
      [s(1) + 1.08,   T_H2 - 0.028,  0.30, 0.50, 0.40, 0.00, 0.02, 0,   0,    0,    0,    0.3, 0,    0,    0],
      [s(1) + 1.30,   T_H2 - 0.006,  0.30, 0.45, 0.35, 0.00, 0.00, 0,   0,    0,    0,    0.6, 0.85, 0,    0],  // the road winding below
      [s(1) + 1.50,   T_H2 + 0.002,  0.30, 0.40, 0.35, 0.00, 0.00, 0,   0,    0,    0,    0.6, 0.85, 0,    0],  // "the pace seems slow"
      [s(2) + 0.08,   T_H3 - 0.004,  0.28, 0.35, 0.30, 0.00, 0.04, 0.1, 0,    0,    0,    0.3, 0,    0,    0.3],
      [s(2) + 0.45,   T_H3 + 0.030,  0.22, 0.25, 0.25, 0.00, 0.04, 1,   0,    0.05, 0,    0.1, 0,    0,    1],  // "the silver tint of the clouds"
      [s(2) + 0.62,   T_H3 + 0.050,  0.18, 0.18, 0.22, -0.45, 0.04, 1,   0.05, 0.15, 0.1,  0.1, 0,    0,    0.8],
      [s(2) + 0.88,   T_TOP - 0.008, 0.10, 0.10, 0.20, -0.30, 0.06, 1,   0.25, 0.60, 0.50, 0.5, 0,    0,    0],  // "it may be near"
      [s(2) + 1.08,   T_TOP,         0.55, 0.05, 0.25, 0.00, 0.08, 1,   0.25, 0.50, 0.30, 0.5, 0,    0,    0],  // "when things seem worst"
      [s(2) + 1.40,   T_TOP + 0.006, 0.00, 0.00, 0.15, 0.00, 0.09, 1,   1,    0.85, 1,    0.5, 0,    0,    0],  // "you must not quit"
      [T.total,       T_TOP + 0.012, 0.00, 0.00, 0.12, 0.00, 0.09, 1,   1,    1,    1,    0.5, 0,    0,    0]
    ];
  },
  sound: {
    src: '/audio/rain.mp3',
    label: 'Play the drizzle and a skylark',
    volume: function (row) { return 0.05 + row[2] * 0.5; },
    cues: [{ stanza: 2, at: 1.35, play: larkSong }]
  }
});
