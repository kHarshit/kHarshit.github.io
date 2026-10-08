/*
 * Scene for "The Queen" (Pablo Neruda): an old cobbled street at blue hour,
 * and a queen no one else can see.
 *
 * I   "I have named you queen": among tall old houses and gas lamps, a
 *     circlet of crystal glints gathers in the air ahead, her crown; "but
 *     you are the queen": a carpet of red-gold light runs out from under
 *     your feet to where she stands.
 * II  "When you go through the streets no one recognizes you": she walks on
 *     and the carpet unrolls before her along the cobbles, shimmering, "the
 *     nonexistent carpet" only we can see, glints falling from the crown.
 * III "And when you appear all the rivers sound in my body, bells shake the
 *     sky": the street opens onto a river bridge; the bell towers across the
 *     water ring, rings of light ripple across the sky, birds lift, and "a
 *     hymn fills the world" in waves of gold.
 * IV  "Only you and I, my love, listen to it": a hush on the bridge, the
 *     first stars, the carpet glowing quietly and two warm lights side by side.
 *
 * Columns: [unit, path, (unused), glints, wind, yaw, pitch, crown, carpet, bells, hymn, hush]
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, softSprite, skyDome, starField,
         particleField, oceanMaterial, followPath, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout (metres; the street runs towards -z, facades at x = ±HALF) ────
var HALF = 5, CURB = 3.8, STREET_END = -62, BANK = -74, FAR_BANK = -152, WATER = -4.2;
var CATHEDRAL = new THREE.Vector3(17, 0, -205);
var BELFRY_Y = 31;
var START_Z = 8, END_Z = -124;
var path = new THREE.LineCurve3(new THREE.Vector3(-1, 0, START_Z), new THREE.Vector3(-1, 0, END_Z));
var PATH_LEN = START_Z - END_Z;
var CARPET_W = 1.3, CARPET_FROM = 11;          // the carpet starts behind you
var CREST = (BANK + FAR_BANK) / 2;               // where she stops, at the top of the bridge

// The walking surface: a flat street, then the gentle hump of the bridge.
function deck(x, z) {
  if (z > BANK || z < FAR_BANK) return 0;
  return 1.1 * Math.sin(Math.PI * (BANK - z) / (BANK - FAR_BANK));
}

// ── Painted textures ─────────────────────────────────────────────────────
function canvasTex(w, h, paint, rep) {
  var c = document.createElement('canvas');
  c.width = w; c.height = h;
  paint(c.getContext('2d'), w, h);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  if (rep) t.repeat.set(rep[0], rep[1]);
  t.anisotropy = 8;
  return t;
}

// Granite setts in staggered rows, each a little domed.
function cobbles(r) {
  return function (x, w, h) {
    x.fillStyle = '#121318';
    x.fillRect(0, 0, w, h);
    var rows = 14, rh = h / rows;
    function sett(gx, gy, ww, l) {
      var g = x.createRadialGradient(gx + ww * 0.4, gy + rh * 0.35, 1, gx + ww / 2, gy + rh / 2, ww * 0.7);
      g.addColorStop(0, 'rgb(' + Math.round(l * 1.25) + ',' + Math.round(l * 1.25) + ',' + Math.round(l * 1.35) + ')');
      g.addColorStop(1, 'rgb(' + Math.round(l * 0.6) + ',' + Math.round(l * 0.6) + ',' + Math.round(l * 0.68) + ')');
      x.fillStyle = g;
      x.beginPath();
      if (x.roundRect) x.roundRect(gx + 2, gy + 2, ww - 4, rh - 4, 7); else x.rect(gx + 2, gy + 2, ww - 4, rh - 4);
      x.fill();
    }
    for (var j = 0; j < rows; j++) {
      var cw = rh * (1.25 + r() * 0.2);
      for (var cx = -r() * cw; cx < w; cx += cw) {
        var ww = cw * (0.84 + r() * 0.1), l = 62 + r() * 46;
        sett(cx, j * rh, ww, l);
        if (cx < 0) sett(cx + w, j * rh, ww, l);          // wrap so the tile repeats
        if (cx + ww > w) sett(cx - w, j * rh, ww, l);
      }
    }
  };
}

function flags(r) {
  return function (x, w, h) {
    x.fillStyle = '#26262a';
    x.fillRect(0, 0, w, h);
    for (var j = 0; j < 4; j++) {
      for (var k = 0; k < 3; k++) {
        var l = 78 + r() * 30;
        x.fillStyle = 'rgb(' + l + ',' + (l - 2) + ',' + (l - 6) + ')';
        x.fillRect(k * w / 3 + 3, j * h / 4 + 3, w / 3 - 6, h / 4 - 6);
      }
    }
    for (var i = 0; i < 400; i++) {
      x.fillStyle = 'rgba(0,0,0,' + (r() * 0.15) + ')';
      x.fillRect(r() * w, r() * h, 2 + r() * 6, 2 + r() * 6);
    }
  };
}

// A window: warm glass behind a frame and a cross of glazing bars. Instance
// colours turn it into a lit room or a dark pane holding the sky.
function windowTex() {
  return canvasTex(64, 112, function (x, w, h) {
    var g = x.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, '#d8d0c0');
    g.addColorStop(1, '#ffffff');
    x.fillStyle = g;
    x.fillRect(0, 0, w, h);
    x.fillStyle = '#2a2420';
    x.fillRect(0, 0, w, 5); x.fillRect(0, h - 5, w, 5); x.fillRect(0, 0, 5, h); x.fillRect(w - 5, 0, 5, h);
    x.fillRect(w / 2 - 2, 0, 4, h);
    x.fillRect(0, h * 0.36, w, 4);
  });
}

// A four-rayed sparkle for the crystal glints.
function glintTex() {
  return canvasTex(64, 64, function (x, w, h) {
    var g = x.createRadialGradient(32, 32, 0, 32, 32, 14);
    g.addColorStop(0, 'rgba(255,255,255,1)');
    g.addColorStop(0.3, 'rgba(255,250,235,0.6)');
    g.addColorStop(1, 'rgba(255,240,210,0)');
    x.fillStyle = g;
    x.fillRect(0, 0, 64, 64);
    x.globalCompositeOperation = 'lighter';
    [[64, 3], [3, 64]].forEach(function (s) {
      var lg = x.createLinearGradient(32 - s[0] / 2, 32 - s[1] / 2, 32 + s[0] / 2, 32 + s[1] / 2);
      lg.addColorStop(0, 'rgba(255,255,255,0)');
      lg.addColorStop(0.5, 'rgba(255,255,255,0.95)');
      lg.addColorStop(1, 'rgba(255,255,255,0)');
      x.fillStyle = lg;
      x.fillRect(32 - s[0] / 2, 32 - s[1] / 2, s[0], s[1]);
    });
  });
}

// Broken streaks of lamplight on moving water.
function streakTex(r) {
  return canvasTex(32, 256, function (x, w, h) {
    for (var i = 0; i < 70; i++) {
      var y = r() * h, len = 2 + r() * 7, a = 0.25 + r() * 0.75, ww = w * (0.3 + r() * 0.7);
      x.fillStyle = 'rgba(255,214,150,' + a + ')';
      x.fillRect((w - ww) / 2, y, ww, len);
    }
  });
}

// ── Synthesised sound cues ───────────────────────────────────────────────
// A church bell: a hum, prime, minor-third tierce, quint and nominal, each
// with its own decay, struck with a short metallic clang.
function bell(ac, out, f, t, gain) {
  [[0.5, 0.5, 6], [1, 0.8, 4], [1.19, 0.5, 3], [1.5, 0.35, 2.4], [2, 0.6, 2.2], [2.5, 0.25, 1.4], [3, 0.2, 1.1], [4.2, 0.12, 0.7]].forEach(function (p) {
    var o = ac.createOscillator(), g = ac.createGain();
    o.type = 'sine';
    o.frequency.value = f * p[0];
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain * p[1], t + 0.006);
    g.gain.exponentialRampToValueAtTime(0.0001, t + p[2]);
    o.connect(g); g.connect(out);
    o.start(t); o.stop(t + p[2] + 0.1);
  });
}

// Two towers answering each other: rounds on four bells, then the tenor.
function peal(ac, out) {
  var now = ac.currentTime + 0.05, notes = [392, 349.2, 311.1, 261.6];
  for (var i = 0; i < 12; i++) bell(ac, out, notes[i % 4] * (i >= 8 ? 0.5 : 1) * (i % 8 >= 4 ? 1.122 : 1), now + i * 0.42, 0.11);
  bell(ac, out, 130.8, now + 5.2, 0.16);
}

// A hymn: a soft choir of detuned saws through a low-pass, I - IV - I.
function hymn(ac, out) {
  var t0 = ac.currentTime + 0.05, chords = [[130.8, 261.6, 329.6, 392], [130.8, 261.6, 349.2, 440], [130.8, 261.6, 329.6, 392, 523.3]];
  var lp = ac.createBiquadFilter(), master = ac.createGain();
  lp.type = 'lowpass';
  lp.frequency.value = 1100;
  master.gain.setValueAtTime(0.0001, t0);
  master.gain.exponentialRampToValueAtTime(0.09, t0 + 2.2);
  master.gain.setValueAtTime(0.09, t0 + 7.5);
  master.gain.exponentialRampToValueAtTime(0.0001, t0 + 12);
  lp.connect(master); master.connect(out);
  chords.forEach(function (chord, c) {
    var a = t0 + c * 3, b = a + (c === 2 ? 9 : 3.4);
    chord.forEach(function (f) {
      [-4, 4].forEach(function (cents) {
        var o = ac.createOscillator(), g = ac.createGain();
        o.type = 'sawtooth';
        o.frequency.value = f;
        o.detune.value = cents;
        g.gain.setValueAtTime(0.0001, a);
        g.gain.linearRampToValueAtTime(0.05, a + 0.8);
        g.gain.setValueAtTime(0.05, b - 0.6);
        g.gain.linearRampToValueAtTime(0.0001, b);
        o.connect(g); g.connect(lp);
        o.start(a); o.stop(b + 0.05);
      });
    });
  });
}

// One far bell in the hush.
function farBell(ac, out) { bell(ac, out, 196, ac.currentTime + 0.05, 0.06); }

// ── Renderer ─────────────────────────────────────────────────────────────
function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(17);
  var gl = makeRenderer(canvas, { clear: '#141e3c' });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#26355f', 0.0052);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.05, 3000);
  world.add(camera);

  // Blue hour: a deep blue dome, the last of the sunset low on the left.
  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#06102c', mid: '#1f3b7a', horizon: '#8a82b4', sun: '#ff9a66' }, 1500);
  dome.uniforms.sunDir.value.set(-0.85, 0.03, -0.5).normalize();
  sky.add(dome.mesh);
  var stars = starField(r, small ? 700 : 1500, 1300, 0.22, 1.4);
  stars.material.opacity = 0;
  sky.add(stars);

  // Rings of light rippling out from the belfries, and the hymn's gold.
  var shimmerU = { uTime: { value: 0 }, uRing: { value: 0 }, uHymn: { value: 0 }, uDir: { value: new THREE.Vector3(0, 0.3, -1) } };
  var shimmer = new THREE.Mesh(new THREE.SphereGeometry(1400, 48, 24), new THREE.ShaderMaterial({
    side: THREE.BackSide, transparent: true, depthWrite: false, fog: false, blending: THREE.AdditiveBlending, uniforms: shimmerU,
    vertexShader: 'varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform float uTime; uniform float uRing; uniform float uHymn; uniform vec3 uDir; varying vec3 vP;\n' +
      'void main(){ vec3 d = normalize(vP); float a = acos(clamp(dot(d, normalize(uDir)), -1.0, 1.0));\n' +
      ' float ring = pow(0.5 + 0.5 * sin(a * 30.0 - uTime * 2.4), 24.0) * exp(-a * 1.5) * smoothstep(0.06, 0.22, a) * uRing;\n' +
      ' float up = smoothstep(-0.02, 0.25, d.y);\n' +
      ' float bands = 0.5 + 0.5 * sin(d.x * 7.0 + d.y * 11.0 + uTime * 0.5 + 2.0 * sin(d.z * 5.0 - uTime * 0.3));\n' +
      ' float glow = uHymn * pow(bands, 3.0) * up * exp(-a * 1.1) * (1.0 - smoothstep(0.1, 0.6, d.y));\n' +
      ' vec3 c = vec3(1.0, 0.76, 0.42) * ring * up * 0.7 + vec3(1.0, 0.55, 0.18) * glow * 0.45;\n' +
      ' gl_FragColor = vec4(c, 1.0);\n #include <colorspace_fragment>\n }'
  }));
  shimmer.frustumCulled = false;
  sky.add(shimmer);

  var hemi = new THREE.HemisphereLight('#7088c8', '#2c2430', 1.15);
  var dusk = new THREE.DirectionalLight('#ffb088', 0.5);
  dusk.position.set(-0.85, 0.25, -0.5);
  world.add(hemi, dusk);

  // ── The street, the embankments, the river ─────────────────────────────
  var cobbleTex = canvasTex(512, 512, cobbles(r), [CURB * 2 / 3.4, 1]);
  var flagTex = canvasTex(256, 256, flags(r), [1, 1]);
  function ground(w, d, x, z, tex, rep, y) {
    var t = tex.clone();
    t.needsUpdate = true;
    t.repeat.set(rep[0], rep[1]);
    var m = new THREE.Mesh(new THREE.PlaneGeometry(w, d).rotateX(-Math.PI / 2),
      new THREE.MeshStandardMaterial({ map: t, bumpMap: t, bumpScale: 1.2, roughness: 0.62, metalness: 0.05 }));
    m.position.set(x, y || 0, z);
    world.add(m);
    return m;
  }
  var streetLen = 14 - STREET_END;
  ground(CURB * 2, streetLen, 0, (14 + STREET_END) / 2, cobbleTex, [CURB * 2 / 3.4, streetLen / 3.4]);
  [-1, 1].forEach(function (s) { ground(HALF - CURB, streetLen, s * (HALF + CURB) / 2, (14 + STREET_END) / 2, flagTex, [0.6, streetLen / 2.4], 0.12); });
  ground(800, STREET_END - BANK, 0, (STREET_END + BANK) / 2, cobbleTex, [800 / 3.4, (STREET_END - BANK) / 3.4]);
  ground(800, 260, 0, FAR_BANK - 130, cobbleTex, [800 / 3.4, 260 / 3.4]);

  // Static stone, plaster and roofs share one vertex-coloured mesh.
  var geos = [];
  // Kerbs and the river walls with their parapets (a gap for the bridge).
  [-1, 1].forEach(function (s) { geos.push(tinted(new THREE.BoxGeometry(0.2, 0.14, streetLen).translate(s * CURB, 0.07, (14 + STREET_END) / 2), '#6a6660')); });
  [[BANK - 0.5, -1], [FAR_BANK + 0.5, 1]].forEach(function (w) {
    geos.push(tinted(new THREE.BoxGeometry(800, 8, 1).translate(0, -4, w[0]), '#4e4a46'));
    [-1, 1].forEach(function (s) {
      geos.push(tinted(new THREE.BoxGeometry(396, 0.9, 0.55).translate(s * (4.4 + 198), 0.45, w[0] - w[1] * 0.2), '#77706a'));
    });
  });

  // ── Houses: rows of tall old facades with shutters, cornices and roofs ─
  var PLASTER = ['#c9a77a', '#b8714f', '#d8c9a8', '#8f9fb0', '#c48a5a', '#a35e48', '#d2b48c', '#9e8a6e', '#b49a8a'];
  var SHUTTER = ['#36574a', '#46566c', '#5a3a2e', '#2c4858', '#6a5a3a'];
  var windows = [];
  function at(geo, theta, x, y, z) { return geo.rotateY(theta).translate(x, y, z); }

  // A row along a line from (x0, z0); along = (cos θ, -sin θ), facing (sin θ, cos θ).
  function row(x0, z0, theta, length, depth, fMin, fMax, skip) {
    var ax = Math.cos(theta), az = -Math.sin(theta), nx = Math.sin(theta), nz = Math.cos(theta), s = 0, FH = 3.3;
    while (s < length - 1.5) {
      var w = Math.min(5.5 + r() * 5, length - s), floors = fMin + Math.floor(r() * (fMax - fMin + 1));
      var h = floors * FH + 0.9, cx = x0 + ax * (s + w / 2), cz = z0 + az * (s + w / 2);
      s += w;
      if (skip && skip(cx, cz)) continue;
      var col = new THREE.Color(PLASTER[Math.floor(r() * PLASTER.length)]).multiplyScalar(0.9 + r() * 0.2);
      var trim = col.clone().lerp(new THREE.Color('#ece4d4'), 0.45), base = col.clone().multiplyScalar(0.62);
      var shut = SHUTTER[Math.floor(r() * SHUTTER.length)];
      geos.push(tinted(at(new THREE.BoxGeometry(w, h, depth), theta, cx - nx * depth / 2, h / 2, cz - nz * depth / 2), col));
      geos.push(tinted(at(new THREE.BoxGeometry(w, 1.0, 0.14), theta, cx + nx * 0.06, 0.5, cz + nz * 0.06), base));
      geos.push(tinted(at(new THREE.BoxGeometry(w, 0.2, 0.26), theta, cx + nx * 0.12, FH, cz + nz * 0.12), trim));
      geos.push(tinted(at(new THREE.BoxGeometry(w + 0.25, 0.45, 0.6), theta, cx + nx * 0.25, h - 0.15, cz + nz * 0.25), trim));
      if (r() < 0.7) {
        var rs = depth * 0.7, roof = new THREE.BoxGeometry(w - 0.1, rs, rs).rotateX(Math.PI / 4).scale(1, 0.5, 1);
        geos.push(tinted(at(roof, theta, cx - nx * depth / 2, h, cz - nz * depth / 2), r() < 0.5 ? '#3a2c2c' : '#2c3038'));
        if (r() < 0.8) {
          var co = (r() - 0.5) * w * 0.7;
          geos.push(tinted(at(new THREE.BoxGeometry(0.7, 2.2, 0.7), theta, cx + ax * co - nx * depth * 0.4, h + 1.2, cz + az * co - nz * depth * 0.4), '#5a4a44'));
        }
      }
      var cols = Math.max(1, Math.floor(w / 2.4));
      for (var f = 0; f < floors; f++) {
        for (var k = 0; k < cols; k++) {
          var off = -w / 2 + (k + 0.5) * w / cols, low = f === 0;
          var ww = low ? Math.min(1.9, w / cols - 0.7) : 1.0, wh = low ? 2.2 : 1.75, wy = low ? 1.35 : f * FH + 1.55;
          var wx = cx + ax * off, wz = cz + az * off;
          windows.push({ x: wx + nx * 0.03, y: wy, z: wz + nz * 0.03, theta: theta, w: ww, h: wh, lit: r() < (low ? 0.55 : 0.33) });
          if (!low) {
            geos.push(tinted(at(new THREE.BoxGeometry(ww + 0.3, 0.12, 0.2), theta, wx + nx * 0.1, wy - wh / 2 - 0.06, wz + nz * 0.1), trim));
            if (r() < 0.6) {
              [-1, 1].forEach(function (sd) {
                geos.push(tinted(at(new THREE.PlaneGeometry(0.5, wh), theta, wx + ax * sd * (ww / 2 + 0.27) + nx * 0.04, wy, wz + az * sd * (ww / 2 + 0.27) + nz * 0.04), shut));
              });
            }
            if (f === 1 && r() < 0.35) {
              geos.push(tinted(at(new THREE.BoxGeometry(ww + 0.7, 0.12, 0.7), theta, wx + nx * 0.35, wy - wh / 2 - 0.1, wz + nz * 0.35), trim));
              geos.push(tinted(at(new THREE.BoxGeometry(ww + 0.7, 0.85, 0.04), theta, wx + nx * 0.7, wy - wh / 2 + 0.36, wz + nz * 0.7), '#1c1c20'));
            }
          }
        }
      }
    }
  }
  row(-HALF, 14, Math.PI / 2, 14 - STREET_END, 9, 3, 5);                  // left, facing +x
  row(HALF, STREET_END, -Math.PI / 2, 14 - STREET_END, 9, 3, 5);          // right, facing -x
  // Across the river, facing the water, with a square open for the cathedral.
  function openings(x) { return (x > -6 && x < 6) || (x > 3 && x < 32); }
  row(-230, -160, 0, 460, 12, 3, 6, function (x) { return openings(x); });
  row(-260, -235, 0, 520, 14, 5, 8);

  // The cathedral: a gabled nave between two bell towers with spires.
  var stoneCol = '#9a8670', C = CATHEDRAL;
  geos.push(tinted(new THREE.BoxGeometry(12, 22, 24).translate(C.x, 11, C.z - 12), stoneCol));
  geos.push(tinted(new THREE.CylinderGeometry(0.01, 9.2, 7, 4, 1).rotateY(Math.PI / 4).scale(1, 1, 0.3).translate(C.x, 25.5, C.z - 1), stoneCol));
  [-1, 1].forEach(function (s) {
    var tx = C.x + s * 9.5;
    geos.push(tinted(new THREE.BoxGeometry(7, 27, 7).translate(tx, 13.5, C.z - 3.5), stoneCol));
    geos.push(tinted(new THREE.BoxGeometry(7.6, 0.6, 7.6).translate(tx, 27.3, C.z - 3.5), '#d6c6a8'));
    geos.push(tinted(new THREE.BoxGeometry(6.4, 8, 6.4).translate(tx, BELFRY_Y + 0.4, C.z - 3.5), stoneCol));
    geos.push(tinted(new THREE.BoxGeometry(7.2, 0.6, 7.2).translate(tx, 35.6, C.z - 3.5), '#d6c6a8'));
    geos.push(tinted(new THREE.ConeGeometry(4.6, 13, 4).rotateY(Math.PI / 4).translate(tx, 42.4, C.z - 3.5), '#4a5058'));
  });
  // A dome further off, for the skyline.
  geos.push(tinted(new THREE.CylinderGeometry(7, 7, 8, 20).translate(-48, 22, -275), '#8c8070'));
  geos.push(tinted(new THREE.SphereGeometry(7.4, 20, 10, 0, Math.PI * 2, 0, Math.PI / 2).translate(-48, 26, -275), '#5c6670'));
  geos.push(tinted(new THREE.CylinderGeometry(0.6, 0.9, 4, 8).translate(-48, 35, -275), '#8c8070'));

  // The bridge: a stone body with three segmental arches, a cobbled deck,
  // balustrades.
  var profile = new THREE.Shape(), span = BANK - FAR_BANK;
  profile.moveTo(0, -8);
  profile.lineTo(span, -8);
  for (var u = span; u >= 0; u -= span / 40) profile.lineTo(u, deck(0, BANK - u) - 0.05);
  [[2, 24], [28, 50], [54, 76]].forEach(function (a) {
    var hole = new THREE.Path(), half = (a[1] - a[0]) / 2, mid = (a[0] + a[1]) / 2;
    hole.moveTo(a[0], -8);
    hole.lineTo(a[0], -4.6);
    for (var k = 1; k < 24; k++) { var th = Math.PI - k / 24 * Math.PI; hole.lineTo(mid + Math.cos(th) * half, -4.6 + Math.sin(th) * 3.2); }
    hole.lineTo(a[1], -4.6);
    hole.lineTo(a[1], -8);
    profile.holes.push(hole);
  });
  var body = new THREE.ExtrudeGeometry(profile, { depth: 8.4, bevelEnabled: false, curveSegments: 4 });
  body.rotateY(Math.PI / 2).translate(-4.2, 0, BANK);
  geos.push(tinted(body, '#6a645c'));
  var deckGeo = new THREE.PlaneGeometry(8, span, 1, 60).rotateX(-Math.PI / 2).translate(0, 0, (BANK + FAR_BANK) / 2);
  var dp = deckGeo.attributes.position;
  for (var i = 0; i < dp.count; i++) dp.setY(i, deck(0, dp.getZ(i)));
  deckGeo.computeVertexNormals();
  var deckTex = cobbleTex.clone();
  deckTex.needsUpdate = true;
  deckTex.repeat.set(8 / 3.4, span / 3.4);
  world.add(new THREE.Mesh(deckGeo, new THREE.MeshStandardMaterial({ map: deckTex, bumpMap: deckTex, bumpScale: 1.2, roughness: 0.62 })));
  var BAL = [];
  [-1, 1].forEach(function (s) {
    for (var z = BANK; z > FAR_BANK + 0.1; z -= 2) {
      var y0 = deck(0, z), y1 = deck(0, z - 2), len = Math.hypot(2, y1 - y0), tilt = Math.atan2(y1 - y0, 2);
      geos.push(tinted(new THREE.BoxGeometry(0.4, 0.16, len).rotateX(tilt).translate(s * 3.9, (y0 + y1) / 2 + 1.0, z - 1), '#8a8278'));
      geos.push(tinted(new THREE.BoxGeometry(0.46, 0.2, len).rotateX(tilt).translate(s * 3.9, (y0 + y1) / 2 + 0.1, z - 1), '#7a7268'));
      for (var b = 0; b < 5; b++) BAL.push(s * 3.9, deck(0, z - b * 0.4 - 0.2), z - b * 0.4 - 0.2);
    }
  });

  // ── Gas lamps: iron posts, lanterns, glows, pools of light ────────────
  var lampPos = [];                                        // [x, z, base height]
  [2, -12, -26, -40, -54].forEach(function (z) { lampPos.push([-CURB - 0.4, z, 0.12]); });
  [-5, -19, -33, -47, -60].forEach(function (z) { lampPos.push([CURB + 0.4, z, 0.12]); });
  [-84, -100, -116, -132, -148].forEach(function (z) { lampPos.push([-3.9, z, deck(0, z) + 1.1], [3.9, z, deck(0, z) + 1.1]); });
  var walkLamps = lampPos.length;                          // the ones along your way
  for (var lx = -200; lx <= 200; lx += 15) {
    if (Math.abs(lx) > 6) { lampPos.push([lx + 2, BANK + 1.2, 0]); lampPos.push([lx, FAR_BANK - 1.6, 0]); }
  }
  var iron = '#16161a', heads = [], glows = [];
  lampPos.forEach(function (l) {
    var y = l[2];
    geos.push(tinted(new THREE.CylinderGeometry(0.17, 0.22, 0.7, 8).translate(l[0], y + 0.35, l[1]), iron));
    geos.push(tinted(new THREE.CylinderGeometry(0.055, 0.075, 3.4, 6).translate(l[0], y + 2.2, l[1]), iron));
    geos.push(tinted(new THREE.ConeGeometry(0.3, 0.28, 4).rotateY(Math.PI / 4).translate(l[0], y + 4.5, l[1]), iron));
    geos.push(tinted(new THREE.CylinderGeometry(0.06, 0.13, 0.16, 4).rotateY(Math.PI / 4).translate(l[0], y + 3.9, l[1]), iron));
    heads.push(l[0], y + 4.17, l[1]);
    glows.push(new THREE.Vector3(l[0], y + 4.17, l[1]));
  });

  var stone = new THREE.Mesh(merge(geos), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.88 }));
  world.add(stone);

  var balusters = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.07, 0.09, 0.8, 6).translate(0, 0.6, 0),
    new THREE.MeshStandardMaterial({ color: '#8a8278', roughness: 0.85 }), BAL.length / 3);
  var m4 = new THREE.Matrix4();
  for (i = 0; i < BAL.length / 3; i++) balusters.setMatrixAt(i, m4.makeTranslation(BAL[i * 3], BAL[i * 3 + 1], BAL[i * 3 + 2]));
  world.add(balusters);

  // Windows: one instanced quad each, lit warm or holding the dusk.
  var winMesh = new THREE.InstancedMesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: windowTex() }), windows.length);
  var q = new THREE.Quaternion(), s3 = new THREE.Vector3(), p3 = new THREE.Vector3(), Y = new THREE.Vector3(0, 1, 0), tc = new THREE.Color();
  windows.forEach(function (w, k) {
    q.setFromAxisAngle(Y, w.theta);
    winMesh.setMatrixAt(k, m4.compose(p3.set(w.x, w.y, w.z), q, s3.set(w.w, w.h, 1)));
    if (w.lit) tc.setHSL(0.08 + r() * 0.04, 0.85, 0.55 + r() * 0.15).multiplyScalar(1.15);
    else tc.setRGB(0.07, 0.09, 0.16).multiplyScalar(0.8 + r() * 0.6);
    winMesh.setColorAt(k, tc);
  });
  world.add(winMesh);

  // Lantern glass, glows, and pools of light on the ground.
  var glass = new THREE.InstancedMesh(new THREE.BoxGeometry(0.3, 0.42, 0.3), new THREE.MeshBasicMaterial({ color: '#ffd39a' }), lampPos.length);
  glows.forEach(function (g, k) { glass.setMatrixAt(k, m4.makeTranslation(g.x, g.y, g.z)); });
  world.add(glass);
  var glowGeo = new THREE.BufferGeometry();
  glowGeo.setAttribute('position', new THREE.Float32BufferAttribute(heads, 3));
  var lampTex = softSprite('rgba(255,200,130,1)', 'rgba(255,160,80,0)');
  world.add(new THREE.Points(glowGeo, new THREE.PointsMaterial({ color: '#ffcf96', size: 2.4, map: lampTex, transparent: true,
    depthWrite: false, blending: THREE.AdditiveBlending })));
  var pools = new THREE.InstancedMesh(new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({
    map: softSprite('rgba(255,190,120,0.55)', 'rgba(255,170,100,0)'), transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, polygonOffset: true, polygonOffsetFactor: -2 }), lampPos.length);
  glows.forEach(function (g, k) { pools.setMatrixAt(k, m4.compose(p3.set(g.x, g.y - 4.12, g.z), q.identity(), s3.set(7, 1, 7))); });
  world.add(pools);
  // Real light only from the few lamps nearest you, moved along as you walk.
  var NL = small ? 3 : 6, lampLights = [];
  for (i = 0; i < NL; i++) {
    var pl = new THREE.PointLight('#ffb468', 0, 18, 1.6);
    world.add(pl);
    lampLights.push(pl);
  }
  var byZ = [];
  for (i = 0; i < walkLamps; i++) byZ.push(glows[i]);
  byZ.sort(function (a, b) { return b.z - a.z; });

  // The river: dark, gently moving water with streaks of lamplight on it.
  var riverMat = oceanMaterial({ color: '#05070e', specular: '#ffd8a0', shininess: 90, sky: '#3a4a7a',
    waves: [[1, 0.2, 0.5, 0.05, 1.1], [0.3, 1, 0.8, 0.03, 1.6], [-0.7, 0.6, 1.3, 0.015, 2.2]] });
  var river = new THREE.Mesh(new THREE.PlaneGeometry(800, BANK - FAR_BANK, small ? 160 : 260, 24).rotateX(-Math.PI / 2), riverMat);
  river.position.set(0, WATER, (BANK + FAR_BANK) / 2);
  world.add(river);
  var ru = riverMat.userData.uniforms;
  var streak = streakTex(r), streaks = [];
  glows.forEach(function (g) {
    if (Math.abs(g.z - (FAR_BANK - 1.6)) > 0.5) return;
    var sm = new THREE.Mesh(new THREE.PlaneGeometry(1.1, 30).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ map: streak.clone(),
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0.8 }));
    sm.material.map.needsUpdate = true;
    sm.position.set(g.x, WATER + 0.06, FAR_BANK + 15);
    world.add(sm);
    streaks.push(sm);
  });

  // ── The cathedral's belfries: warm arches, swinging bells ─────────────
  var archMat = new THREE.MeshBasicMaterial({ color: '#ffbf6e' }), bells = [], belfryGlows = [];
  var bellGeo = new THREE.LatheGeometry([[0, 0], [0.75, -0.05], [0.7, 0.15], [0.5, 0.45], [0.42, 0.95], [0.3, 1.15], [0, 1.2]]
    .map(function (p) { return new THREE.Vector2(p[0], p[1]); }), 14).translate(0, -1.1, 0);
  var bellMat = new THREE.MeshBasicMaterial({ color: '#1e140c' });
  [-1, 1].forEach(function (s) {
    [-1, 1].forEach(function (o) {
      var ax = C.x + s * 9.5 + o * 1.5, az = C.z - 0.25;
      var arch = new THREE.Mesh(new THREE.PlaneGeometry(1.7, 4.2), archMat);
      arch.position.set(ax, BELFRY_Y, az);
      world.add(arch);
      var cap = new THREE.Mesh(new THREE.CircleGeometry(0.85, 16, 0, Math.PI), archMat);
      cap.position.set(ax, BELFRY_Y + 2.1, az);
      world.add(cap);
      var bl = new THREE.Mesh(bellGeo, bellMat);
      bl.position.set(ax, BELFRY_Y + 1.6, az + 0.1);
      bl.userData.phase = r() * 6.28;
      world.add(bl);
      bells.push(bl);
    });
    var bg = new THREE.Sprite(new THREE.SpriteMaterial({ map: lampTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0 }));
    bg.position.set(C.x + s * 9.5, BELFRY_Y, C.z + 0.5);
    bg.scale.setScalar(14);
    world.add(bg);
    belfryGlows.push(bg);
  });
  // The rose window and the door, lit from within; light rising up the stone.
  var rose = new THREE.Mesh(new THREE.CircleGeometry(2.6, 32), new THREE.MeshBasicMaterial({ color: '#ffb46a' }));
  rose.position.set(C.x, 16, C.z + 0.05);
  var door = new THREE.Mesh(new THREE.PlaneGeometry(3, 5), new THREE.MeshBasicMaterial({ color: '#b86a30' }));
  door.position.set(C.x, 2.5, C.z + 0.05);
  world.add(rose, door);
  var flood = new THREE.PointLight('#ffb070', 420, 70, 1.5);
  flood.position.set(C.x, 3, C.z + 22);
  world.add(flood);

  // Birds lifting from the towers when the bells ring.
  var BIRDS = small ? 40 : 70, birdGeo = new THREE.BufferGeometry();
  birdGeo.setAttribute('position', new THREE.Float32BufferAttribute([-0.55, 0, -0.05, 0, 0, 0.12, 0, 0, -0.14, 0.55, 0, -0.05, 0, 0, -0.14, 0, 0, 0.12], 3));
  birdGeo.computeVertexNormals();
  var birdClock = { value: 0 }, birdMat = new THREE.MeshBasicMaterial({ color: '#1a1a26', side: THREE.DoubleSide });
  birdMat.onBeforeCompile = function (sh) {
    sh.uniforms.uClock = birdClock;
    sh.vertexShader = 'uniform float uClock;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n float bph = instanceMatrix[3][0] * 1.7 + instanceMatrix[3][2] * 0.9;\n' +
      ' transformed.y += abs(position.x) * sin(uClock * 11.0 + bph) * 0.9;');
  };
  var birds = new THREE.InstancedMesh(birdGeo, birdMat, BIRDS), birdData = [];
  for (i = 0; i < BIRDS; i++) birdData.push({ s: r() < 0.5 ? -1 : 1, a: r() * 6.28, dir: r() < 0.5 ? -1 : 1, at: r() * 0.5,
                                            rad: 8 + r() * 26, up: 6 + r() * 26, sp: 0.25 + r() * 0.3 });
  birds.frustumCulled = false;
  world.add(birds);

  // ── The queen no one sees: her crown, the carpet, the glints ──────────
  var gTex = glintTex();
  var CROWN = 12, crownPos = new Float32Array(CROWN * 3), crownCol = new Float32Array(CROWN * 3);
  var crownGeo = new THREE.BufferGeometry();
  crownGeo.setAttribute('position', new THREE.BufferAttribute(crownPos, 3));
  crownGeo.setAttribute('color', new THREE.BufferAttribute(crownCol, 3));
  var crownPts = new THREE.Points(crownGeo, new THREE.PointsMaterial({ size: 0.5, map: gTex, vertexColors: true, transparent: true,
    depthWrite: false, blending: THREE.AdditiveBlending }));
  crownPts.frustumCulled = false;
  world.add(crownPts);
  var halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,236,200,0.8)', 'rgba(255,200,140,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0 }));
  world.add(halo);
  var queenLight = new THREE.PointLight('#ffa860', 0, 11, 1.4);
  world.add(queenLight);

  // Glints falling from the crown as she walks.
  var FALL = 90, fallPos = new Float32Array(FALL * 3), fallVel = new Float32Array(FALL * 3), fallLife = new Float32Array(FALL), fallCol = new Float32Array(FALL * 3);
  var fallGeo = new THREE.BufferGeometry();
  fallGeo.setAttribute('position', new THREE.BufferAttribute(fallPos, 3));
  fallGeo.setAttribute('color', new THREE.BufferAttribute(fallCol, 3));
  var falling = new THREE.Points(fallGeo, new THREE.PointsMaterial({ size: 0.1, map: gTex, vertexColors: true, transparent: true,
    depthWrite: false, blending: THREE.AdditiveBlending }));
  falling.frustumCulled = false;
  world.add(falling);
  var nextFall = 0, fallClock = 0;

  // Glints hanging in the air all along the street.
  var air = particleField({ count: small ? 120 : 260, box: [12, 6, 30], fall: [-0.05, 0.08], size: 0.08, map: gTex,
                            colors: ['#fff4dc', '#ffd890', '#ffe8c0'], sway: 0.25, windSpeed: 0.6 });
  air.points.material.blending = THREE.AdditiveBlending;
  world.add(air.points);

  // The carpet: a ribbon along the street's crown, woven in the shader.
  var cps = [], cuv = [], cidx = [], STEP = 0.5, n = 0;
  for (var cz = CARPET_FROM; cz >= END_Z - 2; cz -= STEP) {
    var cy = deck(0, cz) + 0.035, sAlong = CARPET_FROM - cz;
    cps.push(-CARPET_W / 2, cy, cz, CARPET_W / 2, cy, cz);
    cuv.push(sAlong, 0, sAlong, 1);
    if (n > 0) { var a0 = (n - 1) * 2; cidx.push(a0, a0 + 1, a0 + 2, a0 + 1, a0 + 3, a0 + 2); }
    n++;
  }
  var carpetGeo = new THREE.BufferGeometry();
  carpetGeo.setAttribute('position', new THREE.Float32BufferAttribute(cps, 3));
  carpetGeo.setAttribute('uv', new THREE.Float32BufferAttribute(cuv, 2));
  carpetGeo.setIndex(cidx);
  var carpetU = { uTime: { value: 0 }, uFront: { value: 0 }, uQuiet: { value: 0 } };
  var carpet = new THREE.Mesh(carpetGeo, new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -4, uniforms: carpetU,
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform float uTime; uniform float uFront; uniform float uQuiet; varying vec2 vUv;\n' +
      'float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }\n' +
      'void main(){ float s = vUv.x, a = vUv.y, ahead = uFront - s; if (ahead < 0.0) discard;\n' +
      ' float edge = min(a, 1.0 - a);\n' +
      ' float border = smoothstep(0.035, 0.05, edge) * (1.0 - smoothstep(0.075, 0.09, edge)) + smoothstep(0.115, 0.125, edge) * (1.0 - smoothstep(0.135, 0.145, edge));\n' +
      ' vec2 g = vec2(s * 1.5, (a - 0.5) * 2.2);\n' +
      ' float lat = min(abs(fract(g.x + g.y) - 0.5), abs(fract(g.x - g.y) - 0.5));\n' +
      ' float lattice = (1.0 - smoothstep(0.012, 0.035, lat)) * step(0.15, edge);\n' +
      ' float flow = 0.82 + 0.18 * sin(s * 0.9 - uTime * 2.2 * (1.0 - uQuiet * 0.7));\n' +
      ' vec2 cp = vec2(s * 7.0, a * 6.0), cell = floor(cp); float h = hash(cell);\n' +
      ' float spk = 1.0 - smoothstep(0.03, 0.16, length(fract(cp) - 0.5 - (vec2(hash(cell + 3.1), hash(cell + 7.3)) - 0.5) * 0.5));\n' +
      ' float tw = pow(max(0.0, sin(uTime * (1.2 + 3.0 * h) + h * 40.0)), 12.0) * step(0.72, h) * spk;\n' +
      ' vec3 red = vec3(0.5, 0.025, 0.03), gold = vec3(1.0, 0.62, 0.2);\n' +
      ' vec3 c = red * flow * (0.9 + 0.2 * hash(floor(vec2(s * 40.0, a * 30.0))));\n' +
      ' c = mix(c, gold * 1.1, clamp(border + lattice * 0.6, 0.0, 1.0));\n' +
      ' c += vec3(1.0, 0.85, 0.6) * tw * 2.6 * (1.0 - uQuiet * 0.5);\n' +
      ' float lead = exp(-ahead * 1.2);\n' +
      ' c += gold * lead * 2.2 * (1.0 - uQuiet * 0.6);\n' +
      ' float fadeIn = smoothstep(0.0, 1.5, s) * smoothstep(0.0, 0.08, edge * 2.0);\n' +
      ' float alpha = fadeIn * (0.82 + 0.1 * flow) * mix(0.35, 1.0, smoothstep(0.0, 0.8, ahead));\n' +
      ' gl_FragColor = vec4(c, alpha);\n #include <colorspace_fragment>\n }'
  }));
  carpet.frustumCulled = false;
  world.add(carpet);

  // The two warm lights of the last stanza: hers, and yours joining it.
  var warmTex = softSprite('rgba(255,214,160,1)', 'rgba(255,160,90,0)');
  var pair = [0, 1].map(function () {
    var g = new THREE.Group();
    var core = new THREE.Sprite(new THREE.SpriteMaterial({ map: warmTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0 }));
    var glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: warmTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0 }));
    core.scale.setScalar(0.42);
    glow.scale.setScalar(2.2);
    g.add(core, glow);
    g.userData = { core: core, glow: glow };
    world.add(g);
    return g;
  });

  var tmp = new THREE.Color(), crownC = new THREE.Vector3(), front = new THREE.Vector3(), mine = new THREE.Vector3(), v = new THREE.Vector3();
  var belfry = new THREE.Vector3(C.x, BELFRY_Y, C.z), portrait = 0;

  function frame(f) {
    var row = f.row, time = f.time, dt = f.dt, t = clamp(f.cam, 0, 1);
    var crownAmt = row[6], carpetAmt = row[7], bellsAmt = row[8], hymnAmt = row[9], hush = row[10];

    // On a portrait screen, turn a little right so the carpet and the towers sit mid-frame.
    followPath(camera, path, deck, t, { eye: 1.65, ahead: 0.045, yaw: row[4] - portrait * 0.13, pitch: row[5], mx: f.mx, my: f.my, time: time });
    sky.position.copy(camera.position);
    var first = 0, cz0 = camera.position.z;
    while (first < byZ.length - NL && byZ[first].z > cz0 + 8) first++;
    lampLights.forEach(function (pl, k) {
      var lp = byZ[first + k], d = cz0 - lp.z;
      pl.position.copy(lp);
      pl.intensity = 26 * (1 - smooth(24, 38, d)) * (1 - smooth(4, 8, -d));
    });

    // Blue hour deepening towards night in the hush.
    dome.uniforms.top.value.set('#06102c').lerp(tmp.set('#030818'), hush);
    dome.uniforms.mid.value.set('#1f3b7a').lerp(tmp.set('#14275a'), hush);
    dome.uniforms.horizon.value.set('#8a82b4').lerp(tmp.set('#d0a080'), hymnAmt * 0.35).lerp(tmp.set('#4c4e86'), hush * 0.7);
    stars.material.opacity = 0.15 + hush * 0.7;
    v.copy(belfry).sub(camera.position).normalize();
    shimmerU.uDir.value.copy(v);
    shimmerU.uTime.value = time * (env.reduceMotion ? 0.4 : 1);
    shimmerU.uRing.value = bellsAmt;
    shimmerU.uHymn.value = hymnAmt;
    ru.uTime.value = time;
    ru.uAmp.value = 1 + bellsAmt * 0.8;
    streaks.forEach(function (sm, k) { sm.material.map.offset.y = (time * 0.07 + k * 0.37) % 1; sm.material.opacity = 0.65 + bellsAmt * 0.3; });

    // Bells swing and the belfries flare with each stroke.
    bells.forEach(function (b, k) {
      b.rotation.x = Math.sin(time * 2.4 + b.userData.phase) * 0.7 * bellsAmt;
    });
    belfryGlows.forEach(function (g, k) { g.material.opacity = bellsAmt * (0.35 + 0.35 * Math.max(0, Math.sin(time * 4.8 + k * 1.6))); });
    archMat.color.set('#ffbf6e').multiplyScalar(1 + bellsAmt * 0.6);

    // Birds: scattered up from the towers, wheeling over the square.
    birdClock.value = time;
    var shown = 0;
    for (var k = 0; k < BIRDS; k++) {
      var bd = birdData[k], p = clamp((Math.max(bellsAmt, hymnAmt) - bd.at) / 0.5, 0, 1);
      if (p <= 0) continue;
      var ang = bd.a + time * bd.sp * bd.dir, rad = 2 + bd.rad * smooth(0, 1, p);
      p3.set(C.x + bd.s * 9.5 + Math.cos(ang) * rad, BELFRY_Y + 2 + bd.up * smooth(0, 1, p) + Math.sin(time * 0.7 + k) * 1.5, C.z - 3 + Math.sin(ang) * rad * 0.7);
      q.setFromAxisAngle(Y, -ang * bd.dir);
      birds.setMatrixAt(shown++, m4.compose(p3, q, s3.setScalar(1.3)));
    }
    birds.count = shown;
    birds.instanceMatrix.needsUpdate = true;

    // Her crown, a few metres ahead along the carpet.
    var crownZ = Math.max(camera.position.z - 8.5, CREST);
    crownC.set(0, deck(0, crownZ) + 1.68 + Math.sin(time * 1.3) * 0.03, crownZ);
    var shine = smooth(0, 0.6, crownAmt) * (1 - hush);
    for (var c = 0; c < CROWN; c++) {
      var ca = c / CROWN * Math.PI * 2 + time * 0.35, gather = 1 + (1 - smooth(0, 1, crownAmt)) * (2.5 + c % 3);
      crownPos[c * 3] = crownC.x + Math.cos(ca) * 0.22 * gather;
      crownPos[c * 3 + 1] = crownC.y + (c % 2 ? 0.09 : 0) + (1 - smooth(0, 1, crownAmt)) * Math.sin(c * 2.3) * 0.4;
      crownPos[c * 3 + 2] = crownC.z + Math.sin(ca) * 0.22 * gather;
      var tw = shine * (0.55 + 0.45 * Math.sin(time * (2.5 + c * 0.37) + c * 1.7));
      crownCol[c * 3] = tw; crownCol[c * 3 + 1] = tw * 0.95; crownCol[c * 3 + 2] = tw * 0.85;
    }
    crownGeo.attributes.position.needsUpdate = true;
    crownGeo.attributes.color.needsUpdate = true;
    halo.position.copy(crownC);
    halo.scale.setScalar(1.6);
    halo.material.opacity = shine * 0.5;

    // The carpet rolls out from behind you to just beyond her feet.
    var crownS = CARPET_FROM - crownZ;
    carpetU.uFront.value = lerp(2, crownS + 1.4, smooth(0, 1, carpetAmt));
    carpetU.uTime.value = time;
    carpetU.uQuiet.value = hush;
    var frontZ = CARPET_FROM - carpetU.uFront.value;
    front.set(0, deck(0, frontZ) + 0.5, frontZ);
    queenLight.position.copy(front);
    queenLight.intensity = carpetAmt * (5 - hush * 2.5);

    // Glints fall from the crown while she walks.
    fallClock += dt * f.snow * 26;
    while (fallClock > 1) {
      fallClock -= 1;
      var j = (nextFall++ % FALL);
      fallPos[j * 3] = crownC.x + (Math.random() - 0.5) * 0.4; fallPos[j * 3 + 1] = crownC.y; fallPos[j * 3 + 2] = crownC.z + (Math.random() - 0.5) * 0.4;
      fallVel[j * 3] = (Math.random() - 0.5) * 0.2; fallVel[j * 3 + 1] = -0.25 - Math.random() * 0.3; fallVel[j * 3 + 2] = (Math.random() - 0.5) * 0.2;
      fallLife[j] = 2.5;
    }
    for (var fi = 0; fi < FALL; fi++) {
      var w3 = fi * 3;
      if (fallLife[fi] <= 0) { fallCol[w3] = fallCol[w3 + 1] = fallCol[w3 + 2] = 0; continue; }
      fallLife[fi] -= dt;
      fallPos[w3] += fallVel[w3] * dt; fallPos[w3 + 1] += fallVel[w3 + 1] * dt; fallPos[w3 + 2] += fallVel[w3 + 2] * dt;
      var fl = clamp(fallLife[fi] / 2.5, 0, 1) * (0.6 + 0.4 * Math.sin(time * 9 + fi));
      fallCol[w3] = fl; fallCol[w3 + 1] = fl * 0.85; fallCol[w3 + 2] = fl * 0.6;
    }
    fallGeo.attributes.position.needsUpdate = true;
    fallGeo.attributes.color.needsUpdate = true;
    air.update(f, camera.position, env.reduceMotion);

    // Only you and I: her light, and yours drifting up beside it.
    camera.updateMatrixWorld();
    mine.set(0.5, -0.45, -1.4);
    camera.localToWorld(mine);
    pair.forEach(function (g, k) {
      var u = g.userData, bob = Math.sin(time * 1.1 + k * 2.1) * 0.06;
      var rise = smooth(0.3, 1, hush) * 0.4;
      if (k === 0) g.position.set(crownC.x + 0.55, crownC.y + rise + bob, crownC.z);
      else g.position.lerpVectors(mine, v.set(crownC.x - 0.55, crownC.y + rise + bob, crownC.z), smooth(0.1, 1, hush));
      var o = smooth(0, 0.5, hush);
      u.core.material.opacity = o;
      u.glow.material.opacity = o * 0.45 * (0.85 + 0.15 * Math.sin(time * 2 + k));
    });

    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) { portrait = w < h ? 1 : 0; fitCamera(gl, camera, w, h, dpr, small); },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('red-gold-carpet', {
  renderer: renderer3d,
  align: ['left', 'right', 'left', 'left'],
  accent: '#ffb35a',
  keys: function (T) {
    var S = T.start, E = T.end;
    //    unit           path   -  glints wind  yaw    pitch  crown carpet bells hymn hush
    return [
      [0,                0.000, 0, 0.00, 0.15, 0.00, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.7,              0.004, 0, 0.10, 0.15, 0.04, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0],
      [S(0) + 0.45,      0.010, 0, 0.20, 0.15, 0.02, 0.06, 0.6, 0.0, 0.0, 0.0, 0.0],   // "I have named you queen"
      [S(0) + 0.95,      0.016, 0, 0.30, 0.15, -0.10, 0.30, 0.9, 0.0, 0.0, 0.0, 0.0],  // "there are taller than you"
      [E(0) - 0.15,      0.024, 0, 0.45, 0.15, 0.00, 0.05, 1.0, 1.0, 0.0, 0.0, 0.0],   // "but you are the queen"
      [S(1) + 0.4,       0.120, 0, 0.70, 0.15, 0.06, 0.02, 1.0, 1.0, 0.0, 0.0, 0.0],   // "when you go through the streets"
      [S(1) + 1.0,       0.300, 0, 0.90, 0.15, -0.05, 0.00, 1.0, 1.0, 0.0, 0.0, 0.0],  // "the carpet of red gold"
      [E(1) - 0.1,       0.430, 0, 0.80, 0.15, 0.00, 0.02, 1.0, 1.0, 0.0, 0.0, 0.0],   // "the nonexistent carpet"
      [S(2) + 0.35,      0.560, 0, 0.60, 0.20, 0.08, 0.10, 1.0, 1.0, 0.5, 0.1, 0.0],   // "and when you appear"
      [S(2) + 0.9,       0.660, 0, 0.60, 0.25, 0.10, 0.16, 1.0, 1.0, 1.0, 0.7, 0.0],   // "bells shake the sky"
      [E(2) - 0.1,       0.720, 0, 0.50, 0.25, 0.08, 0.14, 1.0, 1.0, 0.8, 1.0, 0.0],   // "a hymn fills the world"
      [S(3) + 0.45,      0.830, 0, 0.20, 0.10, 0.04, 0.08, 1.0, 1.0, 0.0, 0.35, 0.7],  // "only you and I"
      [E(3),             0.868, 0, 0.10, 0.10, 0.03, 0.06, 1.0, 1.0, 0.0, 0.2, 1.0],   // "listen to it"
      [T.total,          0.879, 0, 0.10, 0.10, 0.03, 0.06, 1.0, 1.0, 0.0, 0.15, 1.0]
    ];
  },
  sound: {
    src: '/audio/wind.mp3',
    label: 'Play the evening street and the bells',
    volume: function (row) { return 0.07 + 0.05 * row[8] - 0.03 * row[10]; },
    cues: [
      { stanza: 2, at: 0.3, play: peal },
      { stanza: 2, at: 0.75, play: hymn },
      { stanza: 3, at: 0.8, play: farBell }
    ]
  }
});
