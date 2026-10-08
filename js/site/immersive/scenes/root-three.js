/*
 * Scene for "The Square Root of Three" (David Feinberg): a lonely √3 in a
 * grey-blue void over a floor of still glass, and the root that waltzes in.
 *
 * The one long stanza is split into five panels of four lines (maxLines 4).
 * I    "A lonely number like root three": one glowing 3, small in the void,
 *      under a heavy radical sign. It brightens on "all that's good and
 *      right", then the sign settles lower ("keep out of sight").
 * II   "Beneath a vicious square root sign" the radical presses down and the
 *      3 squashes; "I wish instead I were a nine": a daydream rises above it,
 *      a ghostly √9, and "quick arithmetic" writes "= 3" beside it.
 * III  "I'll never see the sun, as 1.7321": the digits 1.7320508… run off
 *      to the horizon, towards a dim far sun they never reach ("a sad
 *      irrationality"). "When hark!": a second light far off to the left.
 * IV   The other √3 "has quietly come waltzing by", looping in; they turn
 *      round each other (the endless digits going out behind them, near to
 *      far) and "multiply": they meet in a flash and become one bright 3.
 * V    "Our square root signs become unglued": the two radicals shake loose
 *      and fly off trailing sparks, and "love ... renewed" breaks as a burst
 *      of warm light while the sun comes up over the glass.
 *
 * The glyphs are canvas textures set in the page's serif; the radicals are
 * extruded, so they read as heavy. Columns (see COLS):
 *   [unit, cx, gloom, motes, wind, cy, cz, lx, ly, lz, side, glow, press,
 *    wish, arith, digits, sun, partner, spin, fuse, rational, unglue, warm,
 *    pyaw (a turn used only on portrait screens, to keep wide shots' subjects in)]
 */
import { THREE, isSmall, makeRenderer, fitCamera, softSprite, skyDome, particleField, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

var COLS = ['cx', 'gloom', 'motes', 'wind', 'cy', 'cz', 'lx', 'ly', 'lz', 'side', 'glow', 'press',
            'wish', 'arith', 'digits', 'sun', 'partner', 'spin', 'fuse', 'rational', 'unglue', 'warm', 'pyaw'];
var K = {};
COLS.forEach(function (c, i) { K[c] = i; });

// ── Layout (metres; one glyph is about 1 m tall) ─────────────────────────
var FLOOR = -1.3;
var MEET = new THREE.Vector3(-1.2, 0, 0);          // where the two roots meet
var ORBIT = 1.2;                                    // A starts at MEET + ORBIT, B at MEET - ORBIT
var X0 = 0.24;                                      // shifts each root so its group origin is its centre
// The sun lies far off to the east-north-east; the digits run straight at it.
var SUN_DIR = new THREE.Vector3(0.927, 0, -0.375);
var SUN = SUN_DIR.clone().multiplyScalar(390).setY(4);
var SQRT3 = '1.73205080756887729352744634150587236694280525381038062805580697945193301690880003708114618675724857567562614141540670302996994509499895247881165551209437364852809323190230558206797482010108467492326501531234326690332288665067225466892183797122704713166036786158801904998653737985938946765034750657605075661834812960610094760218719032508314582952395983299778982450828871446383291734722416398458785539766795806381835366611084317378089437831610208830552490167002352071114428869599095636579708716849807289949329648428302078640860398873869753758231731783139599298300783870287705391336956331210370726401924910676823119928837564114142201674275210237299427083105989845947598766428889779614783795839022885485';

// The radical sign as strokes [x0, y0, x1, y1, width0, width1], drawn to fit
// over a 3 whose ink spans y -0.43..0.43: a small tick, a heavy down-stroke,
// a thin up-stroke and the bar, with a little drop at its end.
var STROKES = [
  [-0.98, 0.02, -0.84, 0.10, 0.045, 0.06],
  [-0.86, 0.12, -0.64, -0.57, 0.15, 0.10],
  [-0.67, -0.57, -0.40, 0.66, 0.045, 0.04],
  [-0.42, 0.66, 0.50, 0.66, 0.075, 0.075],
  [0.48, 0.69, 0.48, 0.58, 0.03, 0.03]
];

// The other root's way in: from far off to the left, looping as it comes.
var waltz = new THREE.CatmullRomCurve3([[-24, 3.2, -44], [-15, 1.6, -24], [-9, 0.5, -10], [-5, 0.1, -3], [-2.4, 0, 0]]
  .map(function (p) { return new THREE.Vector3(p[0], p[1], p[2]); }));
// The digits' road: it sets off sideways from the root, so the first
// digits read across, then swings round and runs off towards the sun.
var road = (function () {
  var pts = [], x = 0.95, z = 0, d = 0, h0 = Math.atan2(SUN_DIR.x, -SUN_DIR.z);
  while (d < 380) {
    pts.push(new THREE.Vector3(x, Math.min(d * 0.012, 4), z));
    var step = d < 40 ? 1 : 12, h = h0 + 0.82 * Math.exp(-d / 9);
    x += Math.sin(h) * step;
    z -= Math.cos(h) * step;
    d += step;
  }
  return new THREE.CatmullRomCurve3(pts);
})();

// ── Glyphs painted on canvases ───────────────────────────────────────────
var FONT = '"Source Serif 4", Georgia, "Times New Roman", serif';

// A canvas texture whose painter re-runs once the web font has loaded.
function paintedTexture(w, h, paint, painters) {
  var c = document.createElement('canvas');
  c.width = w; c.height = h;
  var x = c.getContext('2d'), t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  function draw() { x.clearRect(0, 0, w, h); paint(x, w, h); t.needsUpdate = true; }
  draw();
  painters.push(draw);
  return t;
}

// Font size that makes `sample` `inkH` pixels tall.
function sizeFor(x, sample, inkH) {
  x.font = '400 100px ' + FONT;
  var m = x.measureText(sample);
  return 100 * inkH / Math.max(1, m.actualBoundingBoxAscent + m.actualBoundingBoxDescent);
}

// White text with a soft glow, its ink centred on (cx, cy).
function glowText(x, text, cx, cy, size, blur) {
  x.font = '400 ' + size.toFixed(1) + 'px ' + FONT;
  x.textAlign = 'left';
  x.textBaseline = 'alphabetic';
  var m = x.measureText(text);
  var px = cx - (m.actualBoundingBoxRight - m.actualBoundingBoxLeft) / 2;
  var py = cy + (m.actualBoundingBoxAscent - m.actualBoundingBoxDescent) / 2;
  x.fillStyle = '#ffffff';
  x.shadowColor = 'rgba(255,255,255,0.9)';
  x.shadowBlur = blur;
  x.fillText(text, px, py);
  x.fillText(text, px, py);
  x.shadowBlur = 0;
  x.fillText(text, px, py);
}

// The radical's strokes on a canvas, `s` pixels to a unit, origin (ox, oy).
function paintRadical(x, ox, oy, s, blur) {
  x.fillStyle = '#ffffff';
  x.shadowColor = 'rgba(255,255,255,0.9)';
  x.shadowBlur = blur;
  STROKES.forEach(function (k) {
    var dx = k[2] - k[0], dy = k[3] - k[1], l = Math.hypot(dx, dy), nx = -dy / l, ny = dx / l;
    x.beginPath();
    x.moveTo(ox + (k[0] + nx * k[4] / 2) * s, oy - (k[1] + ny * k[4] / 2) * s);
    x.lineTo(ox + (k[2] + nx * k[5] / 2) * s, oy - (k[3] + ny * k[5] / 2) * s);
    x.lineTo(ox + (k[2] - nx * k[5] / 2) * s, oy - (k[3] - ny * k[5] / 2) * s);
    x.lineTo(ox + (k[0] - nx * k[4] / 2) * s, oy - (k[1] - ny * k[4] / 2) * s);
    x.closePath();
    x.fill();
  });
  x.shadowBlur = 0;
}

// The heavy radical, extruded from the same strokes.
function radicalGeometry() {
  var shapes = STROKES.map(function (k) {
    var dx = k[2] - k[0], dy = k[3] - k[1], l = Math.hypot(dx, dy), nx = -dy / l, ny = dx / l, s = new THREE.Shape();
    s.moveTo(k[0] + nx * k[4] / 2, k[1] + ny * k[4] / 2);
    s.lineTo(k[2] + nx * k[5] / 2, k[3] + ny * k[5] / 2);
    s.lineTo(k[2] - nx * k[5] / 2, k[3] - ny * k[5] / 2);
    s.lineTo(k[0] - nx * k[4] / 2, k[1] - ny * k[4] / 2);
    s.closePath();
    return s;
  });
  return new THREE.ExtrudeGeometry(shapes, { depth: 0.1, curveSegments: 1, bevelEnabled: true,
    bevelThickness: 0.025, bevelSize: 0.014, bevelSegments: 2 }).translate(0, 0, -0.05);
}

// Long soft rays for the risen sun.
function raysTexture(r) {
  var c = document.createElement('canvas');
  c.width = c.height = 512;
  var x = c.getContext('2d');
  x.translate(256, 256);
  for (var i = 0; i < 46; i++) {
    var a = r() * Math.PI * 2, w = 0.01 + r() * 0.04, len = 140 + r() * 110;
    var g = x.createLinearGradient(0, 0, Math.cos(a) * len, Math.sin(a) * len);
    g.addColorStop(0, 'rgba(255,236,200,0.5)');
    g.addColorStop(1, 'rgba(255,220,170,0)');
    x.fillStyle = g;
    x.beginPath();
    x.moveTo(0, 0);
    x.arc(0, 0, len, a - w, a + w);
    x.closePath();
    x.fill();
  }
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// ── Synthesised sound cues ───────────────────────────────────────────────
// Bell chimes when the two roots multiply: a bright major arpeggio.
function chimes(ac, out) {
  var t = ac.currentTime;
  [1046.5, 1318.5, 1568, 2093, 2637].forEach(function (f, k) {
    [1, 2.76, 5.4].forEach(function (m, j) {
      var o = ac.createOscillator(), g = ac.createGain(), at = t + k * 0.11;
      o.type = 'sine';
      o.frequency.value = f * m;
      g.gain.setValueAtTime(0.0001, at);
      g.gain.exponentialRampToValueAtTime(0.07 / (j + 1.5) / (1 + k * 0.15), at + 0.006);
      g.gain.exponentialRampToValueAtTime(0.0001, at + 2.6 / (j + 1));
      o.connect(g); g.connect(out);
      o.start(at); o.stop(at + 2.7);
    });
  });
}

// A wave of magic wands: a quick sparkling glissando up the pentatonic.
function wands(ac, out) {
  var t = ac.currentTime, scale = [1, 1.125, 1.25, 1.5, 1.667];
  for (var i = 0; i < 16; i++) {
    var f = 1568 * scale[i % 5] * Math.pow(2, Math.floor(i / 5)), at = t + i * 0.05 + Math.random() * 0.02;
    var o = ac.createOscillator(), g = ac.createGain();
    o.type = 'sine';
    o.frequency.value = f;
    g.gain.setValueAtTime(0.0001, at);
    g.gain.exponentialRampToValueAtTime(0.035, at + 0.004);
    g.gain.exponentialRampToValueAtTime(0.0001, at + 0.5);
    o.connect(g); g.connect(out);
    o.start(at); o.stop(at + 0.55);
  }
}

// "Love renewed": a soft, warm major chord swelling and fading.
function warmChord(ac, out) {
  var t = ac.currentTime;
  [261.6, 329.6, 392, 523.3].forEach(function (f, k) {
    var o = ac.createOscillator(), g = ac.createGain();
    o.type = 'triangle';
    o.frequency.value = f;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.045 / (1 + k * 0.3), t + 0.9);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 5.5);
    o.connect(g); g.connect(out);
    o.start(t); o.stop(t + 5.6);
  });
  chimes(ac, out);
}

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(1732), painters = [];
  var gl = makeRenderer(canvas, { clear: '#0b0f17' });
  gl.autoClear = false;
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#2a3442', 0.012);
  var camera = new THREE.PerspectiveCamera(50, 1, 0.05, 3000);

  // Everything that is mirrored in the glass floor lives in `stage`.
  var stage = new THREE.Group(), loose = new THREE.Group();
  world.add(stage, loose);

  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#0c111c', mid: '#1b2433', horizon: '#34404f', sun: '#000000' }, 1500);
  sky.add(dome.mesh);

  var hemi = new THREE.HemisphereLight('#8a9abb', '#121620', 0.7);
  var key = new THREE.DirectionalLight('#c8d4f0', 0.9);
  key.position.set(-3, 5, 6);
  var sunLight = new THREE.DirectionalLight('#ffcf98', 0);
  sunLight.position.copy(SUN).normalize();
  stage.add(hemi, key, sunLight);

  // ── The glass floor: faint graph-paper lines, pools of glyph light, and
  // the mirrored stage showing through it. ──
  var floorMat = new THREE.ShaderMaterial({
    transparent: true, fog: false,
    uniforms: { uBase: { value: new THREE.Color('#0d121b') }, uHorizon: { value: new THREE.Color('#34404f') },
                uCam: { value: new THREE.Vector3() }, uLine: { value: new THREE.Color('#5a6c88') },
                uPoolA: { value: new THREE.Vector3() }, uPoolB: { value: new THREE.Vector3() },
                uColA: { value: new THREE.Color() }, uColB: { value: new THREE.Color() } },
    vertexShader: 'varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }',
    fragmentShader: 'uniform vec3 uBase; uniform vec3 uHorizon; uniform vec3 uCam; uniform vec3 uLine; uniform vec3 uPoolA; uniform vec3 uPoolB; uniform vec3 uColA; uniform vec3 uColB; varying vec3 vW;\n' +
      'void main(){ float d = length(vW.xz - uCam.xz);\n' +
      ' vec2 q = vW.xz; vec2 g = abs(fract(q - 0.5) - 0.5) / fwidth(q);\n' +
      ' float line = (1.0 - min(min(g.x, g.y), 1.0)) * exp(-d * 0.07) * 0.22;\n' +
      ' vec2 a = vW.xz - uPoolA.xz, b = vW.xz - uPoolB.xz;\n' +
      ' vec3 c = uBase + uLine * line + uColA * exp(-dot(a, a) * 0.5) + uColB * exp(-dot(b, b) * 0.5);\n' +
      ' float fog = 1.0 - exp(-d * 0.03);\n' +
      ' c = mix(linearToOutputTexel(vec4(c, 1.0)).rgb, uHorizon, fog * 0.6);\n' +
      // Far off, the glass fades out onto the sky dome below it, so the
      // horizon has no seam.
      ' gl_FragColor = vec4(c, 0.74 * (1.0 - fog)); }'
  });
  var floor = new THREE.Mesh(new THREE.PlaneGeometry(1400, 1400).rotateX(-Math.PI / 2), floorMat);
  floor.position.y = FLOOR;
  floor.renderOrder = -1;          // drawn first, under the glyphs, which don't write depth
  floor.material.depthWrite = false;
  world.add(floor);

  // ── The two roots ──
  var threeTex = paintedTexture(512, 512, function (x, w, h) {
    glowText(x, '3', w / 2, h / 2, sizeFor(x, '3', h * 0.86 / 1.4), 26);
  }, painters);
  var radGeo = radicalGeometry(), halo = softSprite('rgba(200,220,255,1)', 'rgba(200,220,255,0)');

  function makeRoot(tint) {
    var g = new THREE.Group();
    var glyph = new THREE.Mesh(new THREE.PlaneGeometry(1.4, 1.4), new THREE.MeshBasicMaterial({
      map: threeTex, color: tint, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide, fog: false }));
    glyph.position.x = X0;
    var rad = new THREE.Mesh(radGeo, new THREE.MeshStandardMaterial({ color: '#76839c', metalness: 0.35, roughness: 0.36,
      emissive: '#000000', transparent: true }));
    rad.position.x = X0;
    var glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: halo, color: tint, transparent: true, depthWrite: false,
      blending: THREE.AdditiveBlending, fog: false }));
    glow.position.set(X0, 0, -0.15);
    glow.scale.setScalar(2.6);
    var light = new THREE.PointLight(tint, 2, 7, 1.6);
    light.position.set(X0, 0, 0.35);
    g.add(glow, glyph, rad, light);
    stage.add(g);
    return { group: g, glyph: glyph, rad: rad, glow: glow, light: light };
  }
  var A = makeRoot(new THREE.Color('#d4e2ff')), B = makeRoot(new THREE.Color('#ffd6e4'));

  // The one bright 3 they become.
  var one = new THREE.Mesh(new THREE.PlaneGeometry(1.4, 1.4), new THREE.MeshBasicMaterial({
    map: threeTex, color: '#ffe3a8', transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide, fog: false, opacity: 0 }));
  var oneGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,226,170,1)', 'rgba(255,200,140,0)'),
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false, opacity: 0 }));
  var oneLight = new THREE.PointLight('#ffd49a', 0, 9, 1.4);
  var flash = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,250,235,1)', 'rgba(255,230,190,0)'),
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false, opacity: 0 }));
  stage.add(oneGlow, one, oneLight, flash);

  // ── The daydream: a ghostly √9, then "= 3" ──
  var wishTex = paintedTexture(512, 320, function (x, w, h) {
    paintRadical(x, 175, 165, 150, 14);
    glowText(x, '9', 175, 165, sizeFor(x, '9', 130), 14);
  }, painters);
  var eqTex = paintedTexture(512, 320, function (x, w, h) {
    glowText(x, '= 3', w / 2, 165, sizeFor(x, '3', 130), 14);
  }, painters);
  function ghostPlane(tex) {
    return new THREE.Mesh(new THREE.PlaneGeometry(1.6, 1.0), new THREE.MeshBasicMaterial({ map: tex, color: '#c8c4ff',
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false, opacity: 0 }));
  }
  var wishRoot = ghostPlane(wishTex), wishEq = ghostPlane(eqTex);
  var WISH = new THREE.Vector3(-1.0, 1.55, -0.9);
  wishRoot.position.copy(WISH);
  wishEq.position.set(WISH.x + 0.98, WISH.y - 0.02, WISH.z);
  stage.add(wishRoot, wishEq);
  var bubbles = [];
  for (var i = 0; i < 3; i++) {
    var bub = new THREE.Sprite(new THREE.SpriteMaterial({ map: halo, color: '#c8c4ff', transparent: true, depthWrite: false,
      blending: THREE.AdditiveBlending, fog: false, opacity: 0 }));
    bub.position.set(lerp(0.0, -0.75, (i + 1) / 4), lerp(0.85, 1.2, (i + 1) / 4), lerp(0, -0.7, (i + 1) / 4));
    bub.scale.setScalar(0.16 + i * 0.07);
    stage.add(bub);
    bubbles.push(bub);
  }

  // ── The endless digits: camera-facing glyphs from one atlas, laid along
  // the road, growing slowly so the far ones still show. ──
  var CELL = 96;
  var atlas = paintedTexture(CELL * 11, 120, function (x, w, h) {
    var size = sizeFor(x, '8', 62);
    x.font = '400 ' + size.toFixed(1) + 'px ' + FONT;
    var m8 = x.measureText('8'), base = 60 + (m8.actualBoundingBoxAscent - m8.actualBoundingBoxDescent) / 2;
    '0123456789.'.split('').forEach(function (ch, k) {
      var m = x.measureText(ch);
      x.fillStyle = '#ffffff';
      x.shadowColor = 'rgba(255,255,255,0.9)';
      x.shadowBlur = 9;
      x.fillText(ch, k * CELL + CELL / 2 - m.width / 2, base);
      x.shadowBlur = 0;
      x.fillText(ch, k * CELL + CELL / 2 - m.width / 2, base);
    });
  }, painters);
  var ND = Math.min(small ? 380 : 620, SQRT3.length), dPos = new Float32Array(ND * 3), dInfo = new Float32Array(ND * 3);
  var roadLen = road.getLength(), s = 0, tmpV = new THREE.Vector3(), n = 0;
  for (i = 0; i < ND && s < roadLen; i++) {
    var ch = SQRT3[i], sc = 1 + i * 0.011, adv = (ch === '.' ? 0.17 : 0.31) * sc;
    road.getPointAt(Math.min((s + adv / 2) / roadLen, 1), tmpV);
    dPos[i * 3] = tmpV.x; dPos[i * 3 + 1] = tmpV.y; dPos[i * 3 + 2] = tmpV.z;
    dInfo[i * 3] = ch === '.' ? 10 : +ch;
    dInfo[i * 3 + 1] = i;
    dInfo[i * 3 + 2] = sc;
    s += adv;
    n++;
  }
  var cell = new THREE.PlaneGeometry(0.5, 0.625);
  var digitGeo = new THREE.InstancedBufferGeometry();
  digitGeo.index = cell.index;
  digitGeo.setAttribute('position', cell.attributes.position);
  digitGeo.setAttribute('uv', cell.attributes.uv);
  digitGeo.setAttribute('aPos', new THREE.InstancedBufferAttribute(dPos, 3));
  digitGeo.setAttribute('aInfo', new THREE.InstancedBufferAttribute(dInfo, 3));
  digitGeo.instanceCount = n;
  var digitMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
    uniforms: { uMap: { value: atlas }, uTime: { value: 0 }, uReveal: { value: 0 }, uGone: { value: 0 }, uFlip: { value: 1 },
                uColor: { value: new THREE.Color('#cfdcff') }, uFog: { value: 0.03 }, uDim: { value: 1 } },
    vertexShader: 'attribute vec3 aPos; attribute vec3 aInfo; uniform float uTime; uniform float uReveal; uniform float uGone; uniform float uFlip; uniform float uFog;\n' +
      'varying vec2 vUv; varying float vA;\n' +
      'void main(){ vec3 p = aPos; p.y += sin(uTime * 1.3 - aInfo.y * 0.35) * 0.035 * aInfo.z;\n' +
      ' vec4 mv = modelViewMatrix * vec4(p, 1.0);\n' +
      ' float on = clamp(uReveal - aInfo.y, 0.0, 1.0) * (1.0 - clamp(uGone - aInfo.y, 0.0, 1.0));\n' +
      ' mv.xy += vec2(position.x, position.y * uFlip) * aInfo.z * (0.6 + 0.4 * on);\n' +
      ' gl_Position = projectionMatrix * mv;\n' +
      ' vUv = vec2((aInfo.x + uv.x) / 11.0, uv.y);\n' +
      ' vA = on * exp(-pow(-mv.z * uFog, 1.5)); }',
    fragmentShader: 'uniform sampler2D uMap; uniform vec3 uColor; uniform float uDim; varying vec2 vUv; varying float vA;\n' +
      'void main(){ float a = texture2D(uMap, vUv).a * vA * uDim; if (a < 0.003) discard; gl_FragColor = vec4(uColor, a);\n #include <colorspace_fragment>\n }'
  });
  var digits = new THREE.Mesh(digitGeo, digitMat);
  digits.frustumCulled = false;
  stage.add(digits);

  // ── The far sun: a disc in the haze, and rays once it rises ──
  var sunDisc = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,244,220,1)', 'rgba(255,214,160,0)'),
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false, opacity: 0 }));
  var sunRays = new THREE.Sprite(new THREE.SpriteMaterial({ map: raysTexture(r), transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, fog: false, opacity: 0 }));
  loose.add(sunDisc, sunRays);

  // ── Sparks behind the flying radicals, and the burst of light ──
  var TRAIL = 46, spark = softSprite('rgba(255,236,190,1)', 'rgba(255,210,150,0)');
  function trail() {
    var pos = new Float32Array(TRAIL * 3), col = new Float32Array(TRAIL * 3), jit = new Float32Array(TRAIL * 3);
    for (var k = 0; k < TRAIL; k++) {
      var fade = Math.pow(1 - k / TRAIL, 1.6);
      col[k * 3] = fade; col[k * 3 + 1] = fade * 0.88; col[k * 3 + 2] = fade * 0.62;
      jit[k * 3] = (r() - 0.5); jit[k * 3 + 1] = (r() - 0.5); jit[k * 3 + 2] = (r() - 0.5);
    }
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    var pts = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.16, map: spark, vertexColors: true, transparent: true,
      depthWrite: false, blending: THREE.AdditiveBlending, fog: false }));
    pts.frustumCulled = false;
    stage.add(pts);
    return { pts: pts, pos: pos, jit: jit };
  }
  var trails = [trail(), trail()];

  var NB = small ? 260 : 600, bDir = [], bSeed = [];
  for (i = 0; i < NB; i++) {
    var th = r() * Math.PI * 2, y = r() * 1.6 - 0.6, rr = Math.sqrt(Math.max(0, 1 - y * y));
    bDir.push(rr * Math.cos(th), y, rr * Math.sin(th));
    bSeed.push(r());
  }
  var burstGeo = new THREE.BufferGeometry();
  burstGeo.setAttribute('position', new THREE.Float32BufferAttribute(bDir, 3));
  burstGeo.setAttribute('aSeed', new THREE.Float32BufferAttribute(bSeed, 1));
  var burstMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
    uniforms: { uT: { value: 0 }, uC: { value: MEET.clone() }, uTime: { value: 0 }, uScale: { value: 1 } },
    vertexShader: 'attribute float aSeed; uniform float uT; uniform vec3 uC; uniform float uTime; uniform float uScale; varying float vA; varying float vWarm;\n' +
      'void main(){ float e = 1.0 - pow(1.0 - uT, 3.0);\n' +
      ' vec3 p = uC + position * e * (1.2 + aSeed * 6.0) + vec3(0.0, uT * uT * (0.5 + aSeed * 1.5), 0.0);\n' +
      ' vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_Position = projectionMatrix * mv;\n' +
      ' float tw = 0.6 + 0.4 * sin(uTime * (3.0 + aSeed * 6.0) + aSeed * 40.0);\n' +
      ' vA = step(0.001, uT) * (1.0 - smoothstep(0.55, 1.0, uT)) * tw; vWarm = aSeed;\n' +
      ' gl_PointSize = uScale * (2.0 + aSeed * 5.0) * (6.0 / -mv.z); }',
    fragmentShader: 'varying float vA; varying float vWarm;\n' +
      'void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard; float a = smoothstep(0.5, 0.0, d) * vA;\n' +
      ' gl_FragColor = vec4(mix(vec3(1.0, 0.95, 0.85), vec3(1.0, 0.72, 0.42), vWarm), a);\n #include <colorspace_fragment>\n }'
  });
  var burst = new THREE.Points(burstGeo, burstMat);
  burst.frustumCulled = false;
  stage.add(burst);

  // Dust hanging in the void; it warms and rises at the end.
  var motes = particleField({ count: small ? 500 : 1100, box: [30, 9, 30], fall: [-0.06, 0.05], size: 0.06,
    color: '#b8c6e0', map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)'), sway: 0.08, windSpeed: 0.8 });
  motes.points.material.blending = THREE.AdditiveBlending;
  loose.add(motes.points);

  // Repaint the glyphs once the page's serif has loaded.
  if (document.fonts && document.fonts.load) {
    document.fonts.load('400 100px "Source Serif 4"').then(function () { painters.forEach(function (p) { p(); }); }, function () {});
  }

  // ── Per-frame state (no allocations in frame) ──
  var W = 1, H = 1, portrait = false;
  var cam = new THREE.Vector3(), look = new THREE.Vector3(), tmp = new THREE.Vector3(), sunNow = new THREE.Vector3();
  var c1 = new THREE.Color(), c2 = new THREE.Color();
  var VOID = { top: new THREE.Color('#121927'), mid: new THREE.Color('#252f3f'), horizon: new THREE.Color('#4c5a6e') };
  var GLOOM = { top: new THREE.Color('#080b12'), mid: new THREE.Color('#121822'), horizon: new THREE.Color('#29323f') };
  var WARM = { top: new THREE.Color('#25305a'), mid: new THREE.Color('#7a6680'), horizon: new THREE.Color('#e0a585') };
  var pf = { snow: 0, wind: 0, dt: 0, time: 0 };

  // Where a flying radical is, `t` (0..1) into its flight; sign -1 flies left.
  function flight(t, sign, out) {
    var up = portrait ? 0.4 : 1;     // on a phone they fly out sideways, clear of the verse
    return out.set(sign * (4.2 * t + 0.6 * t * t), up * (2.6 * t + 0.9 * Math.sin(Math.PI * t) + 1.2 * t * t),
                   -1.8 * t + sign * 0.7 * Math.sin(Math.PI * t));
  }

  function facing(p) { return Math.atan2(camera.position.x - p.x, camera.position.z - p.z); }

  function placeRadical(root, sign, ug, time, k) {
    var shake = smooth(0.02, 0.25, ug) * (1 - smooth(0.25, 0.35, ug)), t = smooth(0.25, 1, ug);
    flight(t, sign, root.rad.position);
    root.rad.position.x += X0;
    root.rad.position.y += shake * 0.05 * (1 + Math.sin(time * 37 + k));
    root.rad.rotation.set(t * 2.2 * sign, t * 3 * sign, sign * (0.04 * (1 - t) + t * 5.2) + Math.sin(time * 41 + k) * 0.05 * shake);
    root.rad.material.opacity = 1 - smooth(0.7, 1, t);
    root.rad.material.emissive.setRGB(0.9, 0.62, 0.3).multiplyScalar(smooth(0.1, 0.5, ug) * 0.6);
    return t;
  }

  function updateTrail(tr, root, sign, t, time) {
    var on = t > 0.001 && t < 0.999;
    tr.pts.visible = on;
    if (!on) return;
    var g = root.group, cy = Math.cos(g.rotation.y), sy = Math.sin(g.rotation.y);
    for (var k = 0; k < TRAIL; k++) {
      flight(Math.max(t - k * 0.012, 0), sign, tmp);
      var spread = 0.04 + k * 0.012, lx = X0 - 0.2 * sign + tmp.x;     // in the glyph's own frame
      tr.pos[k * 3] = g.position.x + lx * cy + tmp.z * sy + tr.jit[k * 3] * spread + Math.sin(time * 5 + k) * 0.02;
      tr.pos[k * 3 + 1] = g.position.y + 0.55 + tmp.y + tr.jit[k * 3 + 1] * spread - k * 0.004;
      tr.pos[k * 3 + 2] = g.position.z - lx * sy + tmp.z * cy + tr.jit[k * 3 + 2] * spread;
    }
    tr.pts.geometry.attributes.position.needsUpdate = true;
    tr.pts.material.opacity = 1 - smooth(0.8, 1, t);
  }

  function frame(f) {
    var row = f.row, time = f.time, slow = env.reduceMotion;
    var gloom = row[K.gloom], glow = row[K.glow], press = row[K.press], wish = row[K.wish], arith = row[K.arith];
    var partner = row[K.partner], spin = row[K.spin], fuse = row[K.fuse], ug = row[K.unglue], warm = row[K.warm];
    var bob = slow ? 0 : 1;

    // ── Camera ──
    cam.set(row[K.cx], row[K.cy], row[K.cz]);
    look.set(row[K.lx], row[K.ly], row[K.lz]);
    if (portrait) cam.sub(look).multiplyScalar(1.45).add(look);
    camera.position.copy(cam);
    camera.position.x += f.mx * 0.22;
    camera.position.y -= f.my * 0.1;
    camera.lookAt(look);
    if (portrait) camera.rotateY(row[K.pyaw]);
    // Put the subject beside the verse (desktop) or below it (phone).
    if (portrait) camera.setViewOffset(W, H, 0, -H * 0.2, W, H);
    else camera.setViewOffset(W, H, -row[K.side] * W, 0, W, H);
    sky.position.copy(camera.position);

    // ── The roots: A alone, then B waltzing in, the turn, and the meeting ──
    var close = smooth(0, 0.8, fuse), rad = ORBIT * (1 - close), ang = spin * Math.PI * 2.5 + close * Math.PI * 1.5;
    A.group.position.set(MEET.x + Math.cos(ang) * rad, MEET.y, MEET.z - Math.sin(ang) * rad);
    A.group.position.y += Math.sin(time * 0.9) * 0.04 * bob;
    if (partner < 1) {
      waltz.getPointAt(clamp(partner, 0, 1), B.group.position);
      var loop = partner * Math.PI * 6, lr = 0.9 * (1 - smooth(0.7, 1, partner));
      B.group.position.x += Math.cos(loop) * lr;
      B.group.position.z += Math.sin(loop) * lr;
      B.group.position.y += Math.abs(Math.sin(loop)) * 0.25 * lr;
      B.group.rotation.set(0, Math.sin(loop) * 0.55 * lr, Math.sin(loop * 0.5) * 0.08);
    } else {
      B.group.position.set(MEET.x - Math.cos(ang) * rad, MEET.y, MEET.z + Math.sin(ang) * rad);
      B.group.rotation.set(0, 0, 0);
    }
    B.group.position.y += Math.sin(time * 0.9 + 2) * 0.04 * bob;
    var sway = Math.sin(ang * 2) * 0.5 * (1 - close) * smooth(0, 0.1, spin);
    // The glyphs turn to face you wherever the camera goes.
    A.group.rotation.set(0, facing(A.group.position) + sway, -sway * 0.12);
    if (partner >= 1) B.group.rotation.set(0, facing(B.group.position) - sway, sway * 0.12);
    else B.group.rotation.y += facing(B.group.position);
    B.group.visible = partner > 0.001;

    // Pressed under the sign: the radical sinks, the 3 squashes and dims.
    var lit = glow * (1 - press * 0.35) * (0.94 + 0.06 * Math.sin(time * 1.7));
    A.glyph.scale.set(1 + press * 0.05, 1 - press * 0.13, 1);
    A.glyph.position.y = -press * 0.055;
    var sepA = 1 - smooth(0.78, 0.92, fuse);
    A.glyph.material.opacity = Math.min(1, 0.7 + lit * 0.3) * sepA;
    A.glow.material.opacity = 0.3 * lit * sepA;
    A.light.intensity = 2.4 * lit * sepA;
    B.glyph.material.opacity = smooth(0, 0.05, partner) * sepA;
    B.glow.material.opacity = 0.32 * smooth(0, 0.05, partner) * sepA;
    B.light.intensity = 2.4 * smooth(0.5, 1, partner) * sepA;

    // The radicals: pressing, then (after the meeting) stacked, then flying.
    var tA = placeRadical(A, 1, ug, time, 0), tB = placeRadical(B, -1, ug, time, 3);
    if (ug < 0.001) {
      A.rad.position.y = -press * 0.07;
      A.rad.rotation.set(press * 0.06, 0, 0.0);
      B.rad.rotation.set(0, 0, 0);
      if (fuse > 0.85) { A.rad.rotation.z = 0.035; B.rad.rotation.z = -0.035; B.rad.position.z = -0.08; B.rad.position.y = 0.04; }
    }
    A.rad.material.color.setRGB(0.46, 0.51, 0.61).multiplyScalar(1 - press * 0.25);
    updateTrail(trails[0], A, 1, tA, time);
    updateTrail(trails[1], B, -1, tB, time);

    // The one bright 3, and the flash where they met.
    var joined = smooth(0.8, 0.95, fuse), free = smooth(0.25, 0.9, ug);
    var hop = joined * (1 - free) * Math.abs(Math.sin(time * 2.6)) * 0.05 * bob;
    var fy = facing(MEET);
    one.position.set(MEET.x + X0 * Math.cos(fy), MEET.y + hop + free * 0.1, MEET.z - X0 * Math.sin(fy));
    one.rotation.y = fy;
    one.scale.setScalar(1 + joined * 0.12 + free * 0.08 + warm * 0.1);
    one.material.opacity = joined * (0.85 + 0.15 * Math.sin(time * 2)) * (1 - 0.3 * warm);
    one.material.color.set('#ffe3a8').lerp(c1.set('#fff3d6'), warm);
    oneGlow.position.copy(one.position);
    oneGlow.scale.setScalar(3 + warm * 2.5);
    oneGlow.material.opacity = joined * (0.4 - warm * 0.15);
    oneLight.position.set(one.position.x + Math.sin(fy) * 0.6, 0.2, one.position.z + Math.cos(fy) * 0.6);
    oneLight.intensity = joined * (3 + warm * 4);
    var fl = smooth(0.72, 0.86, fuse) * (1 - smooth(0.86, 1, fuse)) + smooth(0.1, 0.3, warm) * (1 - smooth(0.3, 0.8, warm)) * 0.25;
    flash.position.copy(one.position);
    flash.scale.setScalar(3 + fl * 5);
    flash.material.opacity = fl;
    flash.visible = fl > 0.002;

    // The daydream.
    var wf = wish * (0.8 + 0.2 * Math.sin(time * 2.3));
    wishRoot.material.opacity = wf * 0.75;
    wishEq.material.opacity = arith * wf * 0.9;
    wishRoot.position.x = WISH.x + (portrait ? 0.3 : 0);
    wishEq.position.x = wishRoot.position.x + 0.98;
    wishRoot.position.y = WISH.y - (portrait ? 0.3 : 0) + Math.sin(time * 0.8) * 0.05 * bob + (1 - wish) * -0.3;
    wishEq.position.y = wishRoot.position.y - 0.02;
    wishRoot.rotation.y = wishEq.rotation.y = facing(WISH);
    for (var b = 0; b < 3; b++) bubbles[b].material.opacity = smooth(b * 0.2, b * 0.2 + 0.3, wish) * 0.5;

    // The digits: revealed near to far, gone far after the meeting.
    digitMat.uniforms.uTime.value = slow ? 0 : time;
    digitMat.uniforms.uReveal.value = Math.pow(row[K.digits], 2.2) * (n + 10);
    digitMat.uniforms.uGone.value = Math.pow(row[K.rational], 1.6) * (n + 10);
    var dim = 1 - 0.5 * smooth(0.25, 0.8, partner);
    digitMat.uniforms.uDim.value = dim * 0.35;     // fainter in the glass
    digits.visible = row[K.digits] > 0.001 && row[K.rational] < 0.999;

    // ── Sky, sun and glass ──
    c1.copy(VOID.top).lerp(GLOOM.top, gloom).lerp(WARM.top, warm);
    dome.uniforms.top.value.copy(c1);
    c1.copy(VOID.mid).lerp(GLOOM.mid, gloom).lerp(WARM.mid, warm);
    dome.uniforms.mid.value.copy(c1);
    c2.copy(VOID.horizon).lerp(GLOOM.horizon, gloom).lerp(WARM.horizon, warm);
    dome.uniforms.horizon.value.copy(c2);
    world.fog.color.copy(c2);
    floorMat.uniforms.uHorizon.value.copy(c2);
    floorMat.uniforms.uBase.value.set('#0d121b').lerp(c1.set('#2a2026'), warm);
    floorMat.uniforms.uLine.value.set('#5a6c88').lerp(c1.set('#c8946a'), warm);

    var sun = row[K.sun];
    sunNow.set(SUN.x, lerp(SUN.y, 64, smooth(0, 1, warm)), SUN.z);
    sunDisc.position.copy(sunNow);
    sunDisc.scale.setScalar(40 + warm * 50);
    sunDisc.material.opacity = sun * (0.35 + warm * 0.45);
    sunDisc.material.color.set('#c8d2e0').lerp(c1.set('#ffe0b0'), warm);
    sunRays.position.copy(sunNow);
    sunRays.scale.setScalar(220 + warm * 80);
    sunRays.material.opacity = warm * 0.3;
    sunRays.material.rotation = slow ? 0 : time * 0.01;
    tmp.copy(sunNow).sub(camera.position).normalize();
    dome.uniforms.sunDir.value.copy(tmp);
    dome.uniforms.sunColor.value.set('#8a96aa').multiplyScalar(sun * 0.35).lerp(c1.set('#d8a070'), warm);
    sunLight.intensity = warm * 2.2;
    hemi.color.set('#8a9abb').lerp(c1.set('#ffd8b8'), warm);
    hemi.intensity = 0.7 - gloom * 0.3 + warm * 0.5;

    floorMat.uniforms.uCam.value.copy(camera.position);
    floorMat.uniforms.uPoolA.value.copy(A.group.position);
    floorMat.uniforms.uPoolB.value.copy(B.group.position);
    floorMat.uniforms.uColA.value.set('#3a4a6a').multiplyScalar(lit * 0.5 * sepA + joined * 0.4 + warm * 0.4);
    floorMat.uniforms.uColB.value.set('#6a4a5a').multiplyScalar(0.5 * smooth(0.6, 1, partner) * sepA);

    // The burst, and the motes warming and rising.
    burstMat.uniforms.uT.value = warm;
    burstMat.uniforms.uTime.value = time;
    burstMat.uniforms.uScale.value = Math.min(window.devicePixelRatio || 1, 1.75) * (H / 800);
    burst.visible = warm > 0.001 && warm < 0.999;
    motes.points.material.color.set('#b8c6e0').lerp(c1.set('#ffd7a0'), warm);
    motes.points.material.opacity = 0.55 + warm * 0.3;

    pf.snow = row[K.motes]; pf.wind = row[K.wind] - warm * 0.1; pf.dt = f.dt; pf.time = time;
    motes.update(pf, camera.position, slow);

    // ── Draw: first the stage mirrored under the glass, then the world ──
    gl.clear();
    floor.visible = false; loose.visible = false; sky.visible = true;
    stage.scale.y = -1; stage.position.y = 2 * FLOOR;
    digitMat.uniforms.uFlip.value = -1;
    gl.render(world, camera);
    gl.clearDepth();
    floor.visible = true; loose.visible = true; sky.visible = false;
    stage.scale.y = 1; stage.position.y = 0;
    digitMat.uniforms.uFlip.value = 1;
    digitMat.uniforms.uDim.value = dim;
    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) {
      W = w; H = h; portrait = w < h;
      fitCamera(gl, camera, w, h, dpr, small);
      camera.fov = portrait ? 66 : 50;
      camera.updateProjectionMatrix();
    },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('root-three', {
  renderer: renderer3d,
  maxLines: 4,
  accent: '#ffd89a',
  emphasis: /^(sun|multiply|integer|renewed)\W*$/i,
  align: ['right', 'left', 'left', 'right', 'left'],
  // Five panels of four lines. Each beat is pinned to the line it answers;
  // key() carries every column forward, so a beat lists only what changes.
  keys: function (T) {
    function at(i, d) { return T.start(i) + d; }     // d units into panel i (0..1.6)
    var rows = [], cur = {};
    function key(u, ch) {
      for (var c in ch) cur[c] = ch[c];
      rows.push([u].concat(COLS.map(function (c) { return cur[c]; })));
    }
    key(0, { cx: 0.3, cy: 0.9, cz: 11, lx: 0, ly: 1.4, lz: 0, side: 0, gloom: 0, motes: 0.6, wind: 0.05, glow: 0.75,
             press: 0, wish: 0, arith: 0, digits: 0, sun: 0, partner: 0, spin: 0, fuse: 0, rational: 0, unglue: 0, warm: 0, pyaw: 0 });
    key(0.7, { cz: 10.4 });
    key(at(0, 0.25), { cx: 0.5, cy: 0.6, cz: 6.4, lx: 0.3, ly: 0.45, side: -0.2, glow: 0.85 });     // "a lonely number like root three"
    key(at(0, 0.6), { cx: 0.6, cy: 0.45, cz: 5.0, ly: 0.32, glow: 0.9 });
    key(at(0, 0.85), { glow: 1.45 });                                                             // "all that's good and right"
    key(at(0, 1.2), { cz: 4.5, glow: 0.8, press: 0.4 });                                          // "keep out of sight"
    key(at(1, 0.15), { cx: -0.2, cy: 0.2, cz: 3.7, lx: 0, ly: 0.28, side: 0.2 });
    key(at(1, 0.4), { press: 1, gloom: 0.65, glow: 0.75, cy: 0.05, cz: 3.4 });                     // "beneath a vicious square root sign"
    key(at(1, 0.6), { wish: 0 });
    key(at(1, 0.8), { wish: 1, press: 0.85, cx: -0.4, cy: 0.5, cz: 4.4, ly: 0.75 });              // "I wish instead I were a nine"
    key(at(1, 1.05), { arith: 0 });
    key(at(1, 1.25), { arith: 1 });                                                               // "with just some quick arithmetic"
    key(at(1, 1.55), { wish: 0, arith: 0, press: 0.5, gloom: 0.35 });
    // Beside the root, looking along the road of digits to the far sun.
    key(at(2, 0.2), { cx: -3.6, cy: 1.1, cz: 4.0, lx: 11.3, ly: 0.9, lz: -9.9, side: 0.12, digits: 0.06, sun: 0.2, pyaw: -0.17 });
    key(at(2, 0.55), { cx: -3.35, cy: 1.0, cz: 3.7, digits: 0.3, sun: 0.45 });                     // "never see the sun, as 1.7321"
    key(at(2, 1.0), { cx: -3.2, digits: 1, sun: 0.5, gloom: 0.45, glow: 0.7 });                    // "a sad irrationality"
    key(at(2, 1.15), { partner: 0 });
    key(at(2, 1.5), { cx: 1.2, cy: 0.9, cz: 6.4, lx: -8, ly: 1.0, lz: -14, side: -0.1, partner: 0.3, gloom: 0.3, pyaw: 0 }); // "When hark!"
    key(at(3, 0.05), { rational: 0 });
    key(at(3, 0.35), { rational: 0.6, cx: -0.4, cy: 0.6, cz: 7.0, lx: -1.2, ly: 0.35, lz: 0, side: -0.24, partner: 1, glow: 0.9, gloom: 0.15 }); // "waltzing by"
    key(at(3, 0.4), { spin: 0 });
    key(at(3, 0.6), { rational: 1 });                                                             // the endless digits go out
    key(at(3, 0.8), { spin: 1, fuse: 0.45 });                                                     // "together now we multiply"
    key(at(3, 0.95), { fuse: 1, cz: 5.8 });                                                       // "to form a number we prefer"
    key(at(3, 1.45), { gloom: 0, sun: 0.6 });                                                     // "rejoicing as an integer"
    // Round to face the sun, with the one 3 just before it.
    key(at(4, 0.15), { cx: -6.3, cy: 0.5, cz: 3.6, lx: -1.2, ly: 0.5, lz: 0, side: 0.18, unglue: 0 });
    key(at(4, 0.45), { unglue: 0.25 });                                                           // "break free from our mortal bonds"
    key(at(4, 0.95), { unglue: 1, cy: 0.7, ly: 0.9 });                                            // "our square root signs become unglued"
    key(at(4, 1.05), { warm: 0 });
    key(at(4, 1.5), { warm: 0.75, sun: 1 });                                                      // "and love for me has been renewed"
    key(T.total - 0.7, { cx: -9.5, cy: 1.2, cz: 0, lx: 10.5, ly: 3.2, lz: -0.7, side: 0, warm: 0.95, motes: 0.9, pyaw: 0.12 }); // back, beside the risen sun
    key(T.total, { cx: -10.2, cy: 1.3, cz: 0.2, warm: 1 });
    return rows;
  },
  sound: {
    src: '/audio/wind.mp3',
    label: 'Play the wind and the chimes',
    volume: function (row) { return 0.05 + 0.05 * row[K.gloom] + 0.04 * row[K.warm]; },
    cues: [
      { stanza: 3, at: 0.85, play: chimes },
      { stanza: 4, at: 0.7, play: wands },
      { stanza: 4, at: 1.2, play: warmChord }
    ]
  }
});
