/*
 * Scene for "Shakespeare Quotes": the Globe, empty, at night.
 *
 * Standing in the yard of the wooden O: three tiers of galleries, the
 * thatched ring, and the stage with its two painted pillars under the
 * heavens, open to the stars. The camera drifts round the empty house and
 * each quote is a lighting cue on the same stage.
 *
 * I     "Shall I compare thee to a summer's day?": the house fills with a
 *       warm golden light and the sky above the O goes to summer dusk.
 * II    "Love is not love which alters": one fixed star above the open roof.
 * III   "What's in a name?": a spot finds a single rose on the boards.
 * IV    "If music be the food of love": a lute plays; motes of light rise
 *       from the stage into the heavens.
 * V     "winged Cupid painted blind": a soft blindfold of mist across the eyes.
 * VI    "Doubt thou the stars are fire": the stars flare, burning warm.
 * VII   "The course of true love never did run smooth": a kinked, winding
 *       ribbon of light runs across the stage floor.
 * VIII  "Et tu, Brute?": a sudden red wash, and a dagger-sting chord.
 * IX    "I bear a charmed life": a ring of protective light round the rose.
 * X     "Stars, hide your fires": the stars go out one by one, and the
 *       fixed star with them.
 * XI    "loved not wisely but too well": dim, cold blue on the empty stage.
 * XII   "See how she leans her cheek": the balcony above the stage, softly
 *       lit and empty, its curtain stirring.
 * XIII  "How far that little candle throws its beams!": in the dark one
 *       candle is lit at the front of the stage and its light spreads until
 *       it fills the whole theatre.
 * XIV   The last quote: the house settles to moon and candlelight, a bell
 *       tolls, and in the outro the eye goes up to the fixed star again.
 *
 * The poem is fourteen separate lists, so fourteen panels; keys are a
 * function pinned to start(i). Columns:
 *   [unit, camX, dark, mist, wind, camZ, camY, lookX, lookY, lookZ, gold, star,
 *    spot, music, flare, ribbon, red, ring, out, blue, balcony, candle, spread, zoom,
 *    portraitLookX, portraitLookY, portraitLookZ]
 * Phones look at their own target where the centred text would cover the beat.
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, softSprite, skyDome,
         disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// ── Layout (metres): yard centre at the origin, the stage towards -z ─────
// Twenty bays; bay k is centred at k * 18 degrees (x = r sin, z = r cos),
// so bays 9-11 are the tiring house behind the stage.
var SIDES = 20, BAY = Math.PI * 2 / SIDES;
var R1 = 13.5, R2 = 17.1;                          // gallery front, back wall
var FLOORS = [0.75, 4.4, 7.6], TOP = 10.6;         // gallery floors, ceiling
var ZF = -R1 * Math.cos(1.5 * BAY), XF = R1 * Math.sin(1.5 * BAY);   // frons scenae
var STAGE_Y = 1.5, STAGE_FRONT = -3.8, STAGE_X = 6.0;
var HEAV_Y = 8.4, HEAV_FRONT = -5.3;
var PILLARS = [[-4.4, -5.75], [4.4, -5.75]];
var ROSE = new THREE.Vector3(1.0, STAGE_Y, -5.6);
var CANDLE = new THREE.Vector3(-1.1, STAGE_Y, -4.45);
var STAR_DIR = new THREE.Vector3(0.15, 1, -0.7).normalize();     // above the heavens' hut
function stageBay(k) { return k >= 9 && k <= 11; }

// ── Sounds: a lute (plucked strings), a dagger sting and a bell ──────────
// Karplus-Strong: a burst of noise recirculating through a short, damped
// delay line rings like a gut string.
function pluck(ac, out, freq, when, gain) {
  var sr = ac.sampleRate, len = Math.floor(sr * 2.4), buf = ac.createBuffer(1, len, sr), d = buf.getChannelData(0);
  var n = Math.max(2, Math.round(sr / freq)), line = new Float32Array(n), prev = 0;
  for (var i = 0; i < n; i++) { prev = prev * 0.5 + (Math.random() * 2 - 1) * 0.5; line[i] = prev; }
  for (var j = 0, p = 0; j < len; j++) {
    var q = (p + 1) % n, v = line[p];
    d[j] = v;
    line[p] = 0.4985 * (v + line[q]);
    p = q;
  }
  var src = ac.createBufferSource(), lp = ac.createBiquadFilter(), g = ac.createGain();
  src.buffer = buf;
  lp.type = 'lowpass';
  lp.frequency.value = 2600;
  g.gain.value = gain;
  src.connect(lp);
  lp.connect(g);
  g.connect(out);
  src.start(when);
}
function lute(notes, gain) {
  return function (ac, out) {
    var t = ac.currentTime + 0.05;
    notes.forEach(function (nt) { pluck(ac, out, nt[0], t + nt[1], gain * (nt[2] || 1)); });
  };
}
// A struck dissonant chord with a blade's ring over it.
function sting(ac, out) {
  var t = ac.currentTime;
  var lp = ac.createBiquadFilter(), amp = ac.createGain();
  lp.type = 'lowpass';
  lp.frequency.setValueAtTime(5000, t);
  lp.frequency.exponentialRampToValueAtTime(500, t + 1.6);
  amp.gain.setValueAtTime(0.0001, t);
  amp.gain.exponentialRampToValueAtTime(0.16, t + 0.012);
  amp.gain.exponentialRampToValueAtTime(0.0001, t + 2.4);
  [110, 116.5, 155.6, 233.1, 329.6].forEach(function (f) {
    var o = ac.createOscillator();
    o.type = 'sawtooth';
    o.frequency.value = f;
    o.connect(lp);
    o.start(t);
    o.stop(t + 2.5);
  });
  lp.connect(amp);
  amp.connect(out);
  [2410, 3170, 4420, 5980].forEach(function (f, k) {
    var o = ac.createOscillator(), g = ac.createGain();
    o.frequency.value = f;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.03 / (k + 1), t + 0.005);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 1.4 / (k * 0.4 + 1));
    o.connect(g);
    g.connect(out);
    o.start(t);
    o.stop(t + 1.5);
  });
}
// A distant church bell: hum, prime, tierce, quint and nominal partials.
function bell(freq, gain, n, gap) {
  function one(ac, out) {
    var t = ac.currentTime;
    [[0.5, 0.35, 7], [1, 0.5, 5.5], [1.003, 0.4, 5.5], [1.19, 0.28, 4], [1.5, 0.2, 3.2],
     [2, 0.35, 3], [2.51, 0.14, 2.2], [3.02, 0.1, 1.6]].forEach(function (p) {
      var o = ac.createOscillator(), g = ac.createGain();
      o.frequency.value = freq * p[0];
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(gain * p[1], t + 0.008);
      g.gain.exponentialRampToValueAtTime(0.0001, t + p[2]);
      o.connect(g);
      g.connect(out);
      o.start(t);
      o.stop(t + p[2] + 0.05);
    });
  }
  return function (ac, out) {
    for (var i = 0; i < n; i++) setTimeout(function () { one(ac, out); }, i * gap * 1000);
  };
}

// ── Textures painted on canvases ─────────────────────────────────────────
function canvasTexture(w, h, paint, repeat) {
  var c = document.createElement('canvas');
  c.width = w; c.height = h;
  paint(c.getContext('2d'), w, h);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  if (repeat) t.repeat.set(repeat[0], repeat[1]);
  t.anisotropy = 4;
  return t;
}
function rgb(r, g, b) { return 'rgb(' + Math.round(r) + ',' + Math.round(g) + ',' + Math.round(b) + ')'; }

// Oak boards: eight planks to a tile, grain and seams.
function boards(r) {
  return function (x, w, h) {
    var ph = h / 8;
    for (var k = 0; k < 8; k++) {
      var l = 0.75 + r() * 0.35;
      x.fillStyle = rgb(118 * l, 82 * l, 52 * l);
      x.fillRect(0, k * ph, w, ph);
      for (var g = 0; g < 14; g++) {
        var y = k * ph + r() * ph, a = r() * 6;
        x.strokeStyle = 'rgba(40,24,12,' + (0.12 + r() * 0.2) + ')';
        x.lineWidth = 0.6 + r() * 1.2;
        x.beginPath();
        x.moveTo(0, y);
        for (var s = 0; s <= w; s += 16) x.lineTo(s, y + Math.sin(s * 0.02 + a) * 2.5);
        x.stroke();
      }
      x.fillStyle = 'rgba(18,10,6,0.85)';
      x.fillRect(0, k * ph, w, 2);
      var joint = r() * w;
      x.fillRect(joint, k * ph, 2, ph);
    }
  };
}

// Lime plaster between dark oak framing; one bay per tile, three storeys.
function framing(r) {
  return function (x, w, h) {
    x.fillStyle = '#d6cbb2';
    x.fillRect(0, 0, w, h);
    for (var i = 0; i < 900; i++) {
      x.fillStyle = 'rgba(' + (r() < 0.5 ? '90,80,60' : '255,250,235') + ',' + (r() * 0.08) + ')';
      x.fillRect(r() * w, r() * h, 2 + r() * 10, 2 + r() * 10);
    }
    var oak = '#3a2618', storeys = [0, 0.37, 0.695, 1];
    x.fillStyle = oak;
    x.fillRect(0, 0, 12, h);
    x.fillRect(w - 12, 0, 12, h);
    x.fillRect(w / 2 - 6, 0, 12, h);
    for (var s = 0; s < 3; s++) {
      var y0 = h * (1 - storeys[s + 1]), y1 = h * (1 - storeys[s]);
      x.fillRect(0, y0, w, 12);
      x.fillRect(0, (y0 + y1) / 2, w, 8);
      // Braces and a dark doorway into the stair passage in each bay.
      x.lineWidth = 9;
      x.strokeStyle = oak;
      x.beginPath();
      x.moveTo(6, (y0 + y1) / 2); x.lineTo(w / 4, y0 + 6);
      x.moveTo(w - 6, (y0 + y1) / 2); x.lineTo(w * 3 / 4, y0 + 6);
      x.stroke();
      x.fillStyle = '#140c08';
      x.fillRect(w * 0.62, y1 - (y1 - y0) * 0.72, w * 0.16, (y1 - y0) * 0.72);
      x.fillStyle = oak;
    }
  };
}

// Reed thatch: thousands of short straws running down the slope.
function thatch(r) {
  return function (x, w, h) {
    x.fillStyle = '#6e5a38';
    x.fillRect(0, 0, w, h);
    for (var i = 0; i < 5000; i++) {
      var l = 0.6 + r() * 0.7, sx = r() * w, sy = r() * h;
      x.strokeStyle = rgb(150 * l, 122 * l, 78 * l);
      x.globalAlpha = 0.35 + r() * 0.4;
      x.lineWidth = 1 + r();
      x.beginPath();
      x.moveTo(sx, sy);
      x.lineTo(sx + (r() - 0.5) * 3, sy + 14 + r() * 26);
      x.stroke();
    }
    x.globalAlpha = 1;
    for (var c = 0; c < 6; c++) {
      x.fillStyle = 'rgba(30,22,10,0.22)';
      x.fillRect(0, c * h / 6, w, 5);
    }
  };
}

// The trodden yard: dark earth, grit and hazelnut shells.
function earth(r) {
  return function (x, w, h) {
    x.fillStyle = '#3e352b';
    x.fillRect(0, 0, w, h);
    for (var i = 0; i < 3500; i++) {
      var l = 0.5 + r() * 1.1;
      x.fillStyle = rgb(70 * l, 58 * l, 44 * l);
      x.globalAlpha = 0.3 + r() * 0.5;
      x.fillRect(r() * w, r() * h, 1 + r() * 3, 1 + r() * 3);
    }
    x.globalAlpha = 1;
  };
}

// Red marble painted on the pillars.
function marble(r) {
  return function (x, w, h) {
    var g = x.createLinearGradient(0, 0, w, 0);
    g.addColorStop(0, '#6a2a22'); g.addColorStop(0.5, '#8a3a2c'); g.addColorStop(1, '#6a2a22');
    x.fillStyle = g;
    x.fillRect(0, 0, w, h);
    for (var v = 0; v < 26; v++) {
      x.strokeStyle = r() < 0.6 ? 'rgba(235,205,180,0.35)' : 'rgba(40,10,8,0.4)';
      x.lineWidth = 0.7 + r() * 2;
      x.beginPath();
      var px = r() * w, py = 0;
      x.moveTo(px, py);
      while (py < h) { px += (r() - 0.5) * 30; py += 10 + r() * 30; x.lineTo(px, py); }
      x.stroke();
    }
  };
}

// The heavens: a blue ceiling in gilded coffers, stars, sun and moon.
function heavens(r) {
  return function (x, w, h) {
    var g = x.createRadialGradient(w / 2, h / 2, 20, w / 2, h / 2, w * 0.6);
    g.addColorStop(0, '#2a3f86'); g.addColorStop(1, '#0f1838');
    x.fillStyle = g;
    x.fillRect(0, 0, w, h);
    for (var i = 0; i < 260; i++) {
      var sx = r() * w, sy = r() * h, s = 1.5 + r() * 4;
      x.fillStyle = 'rgba(240,200,110,' + (0.6 + r() * 0.4) + ')';
      x.beginPath();
      for (var p = 0; p < 10; p++) {
        var a = p * Math.PI / 5, rr = p % 2 ? s * 0.4 : s;
        x.lineTo(sx + Math.cos(a) * rr, sy + Math.sin(a) * rr);
      }
      x.fill();
    }
    // Zodiac ring and the sun at its centre.
    x.strokeStyle = '#d8b25a';
    x.lineWidth = 6;
    x.beginPath(); x.arc(w / 2, h / 2, h * 0.36, 0, Math.PI * 2); x.stroke();
    x.lineWidth = 2;
    x.beginPath(); x.arc(w / 2, h / 2, h * 0.3, 0, Math.PI * 2); x.stroke();
    for (var z = 0; z < 12; z++) {
      var za = z * Math.PI / 6;
      x.beginPath();
      x.moveTo(w / 2 + Math.cos(za) * h * 0.3, h / 2 + Math.sin(za) * h * 0.3);
      x.lineTo(w / 2 + Math.cos(za) * h * 0.36, h / 2 + Math.sin(za) * h * 0.36);
      x.stroke();
      x.beginPath();
      x.arc(w / 2 + Math.cos(za + 0.26) * h * 0.33, h / 2 + Math.sin(za + 0.26) * h * 0.33, 6, 0, Math.PI * 2);
      x.stroke();
    }
    var sg = x.createRadialGradient(w / 2, h / 2, 4, w / 2, h / 2, h * 0.16);
    sg.addColorStop(0, '#d8b060'); sg.addColorStop(0.5, '#a87a34'); sg.addColorStop(1, 'rgba(168,122,52,0)');
    x.fillStyle = sg;
    x.beginPath(); x.arc(w / 2, h / 2, h * 0.16, 0, Math.PI * 2); x.fill();
    x.strokeStyle = '#b8903e';
    x.lineWidth = 3;
    for (var ry = 0; ry < 16; ry++) {
      var ra = ry * Math.PI / 8;
      x.beginPath();
      x.moveTo(w / 2 + Math.cos(ra) * h * 0.1, h / 2 + Math.sin(ra) * h * 0.1);
      x.lineTo(w / 2 + Math.cos(ra) * h * 0.22, h / 2 + Math.sin(ra) * h * 0.22);
      x.stroke();
    }
    // Crescent moon in a corner coffer.
    x.fillStyle = '#e9e2c8';
    x.beginPath(); x.arc(w * 0.13, h * 0.25, h * 0.07, 0, Math.PI * 2); x.fill();
    x.fillStyle = '#16224a';
    x.beginPath(); x.arc(w * 0.145, h * 0.23, h * 0.062, 0, Math.PI * 2); x.fill();
    // Gilded coffer frames.
    x.strokeStyle = '#c9a24a';
    x.lineWidth = 10;
    x.strokeRect(5, 5, w - 10, h - 10);
    x.lineWidth = 5;
    [w / 4, w * 3 / 4].forEach(function (cx) { x.beginPath(); x.moveTo(cx, 0); x.lineTo(cx, h); x.stroke(); });
  };
}

// The frons scenae, the painted back wall of the stage, from the stage top
// (y 1.5) to the gallery ceiling (y 10.6): two doors, the arras over the
// discovery space, marbled pilasters and gilt friezes. The balcony and the
// lords' rooms are cut out (alpha) so the rooms built behind show through.
function fronsPaint(r) {
  return function (x, w, h) {
    var sx = w / (2 * XF), sy = h / (TOP - STAGE_Y);
    function X(m) { return (m + XF) * sx; }
    function Y(m) { return (TOP - m) * sy; }
    function rect(x0, y0, x1, y1) { x.fillRect(X(x0), Y(y1), (x1 - x0) * sx, (y1 - y0) * sy); }
    function frame(x0, y0, x1, y1, col, lw) { x.strokeStyle = col; x.lineWidth = lw; x.strokeRect(X(x0), Y(y1), (x1 - x0) * sx, (y1 - y0) * sy); }
    function arch(x0, y0, x1, y1) {
      var rad = (x1 - x0) / 2, cx = (x0 + x1) / 2;
      x.beginPath();
      x.moveTo(X(x0), Y(y0));
      x.lineTo(X(x0), Y(y1 - rad));
      x.arc(X(cx), Y(y1 - rad), rad * sx, Math.PI, 0);
      x.lineTo(X(x1), Y(y0));
      x.closePath();
    }
    // Oxblood panelling.
    var g = x.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, '#2a140e'); g.addColorStop(1, '#3e1d12');
    x.fillStyle = g;
    x.fillRect(0, 0, w, h);
    [[1.5, 5.0], [5.35, 7.8], [8.1, 10.6]].forEach(function (lv) {
      for (var px = -XF + 0.2; px < XF - 0.4; px += 1.15) {
        x.fillStyle = 'rgba(110,30,22,0.55)';
        rect(px + 0.1, lv[0] + 0.25, px + 1.0, lv[1] - 0.25);
        frame(px + 0.1, lv[0] + 0.25, px + 1.0, lv[1] - 0.25, 'rgba(201,162,74,0.7)', 2);
      }
    });
    // Friezes.
    [[5.0, 5.35], [7.8, 8.1]].forEach(function (f) {
      x.fillStyle = '#8a6a2a';
      rect(-XF, f[0], XF, f[1]);
      x.fillStyle = '#3a2410';
      for (var fx = -XF; fx < XF; fx += 0.5) rect(fx + 0.15, f[0] + 0.08, fx + 0.35, f[1] - 0.08);
    });
    // Marbled pilasters with gilt capitals.
    [-5.5, -2.3, 2.3, 5.5].forEach(function (cx) {
      [[1.5, 5.0], [5.35, 7.8]].forEach(function (lv) {
        x.fillStyle = '#b8ac98';
        rect(cx - 0.22, lv[0], cx + 0.22, lv[1]);
        x.strokeStyle = 'rgba(90,70,60,0.5)';
        x.lineWidth = 1.5;
        for (var v = 0; v < 4; v++) {
          x.beginPath();
          x.moveTo(X(cx - 0.2 + r() * 0.4), Y(lv[0]));
          x.lineTo(X(cx - 0.2 + r() * 0.4), Y((lv[0] + lv[1]) / 2));
          x.lineTo(X(cx - 0.2 + r() * 0.4), Y(lv[1]));
          x.stroke();
        }
        x.fillStyle = '#d4ac4c';
        rect(cx - 0.3, lv[1] - 0.3, cx + 0.3, lv[1]);
        rect(cx - 0.3, lv[0], cx + 0.3, lv[0] + 0.2);
      });
    });
    // The two doors.
    [-3.85, 3.85].forEach(function (cx) {
      x.fillStyle = '#9c8a6c';
      arch(cx - 0.8, 1.5, cx + 0.8, 4.35); x.fill();
      x.fillStyle = '#21140c';
      arch(cx - 0.65, 1.5, cx + 0.65, 4.2); x.fill();
      x.strokeStyle = 'rgba(120,80,50,0.6)';
      x.lineWidth = 2;
      x.beginPath(); x.moveTo(X(cx), Y(1.5)); x.lineTo(X(cx), Y(3.55)); x.stroke();
      [2.1, 2.9].forEach(function (py) { frame(cx - 0.55, py, cx - 0.1, py + 0.6, 'rgba(120,80,50,0.6)', 2); frame(cx + 0.1, py, cx + 0.55, py + 0.6, 'rgba(120,80,50,0.6)', 2); });
    });
    // The arras over the discovery space.
    for (var ax = -1.45; ax < 1.45; ax += 0.08) {
      var fold = 0.5 + 0.5 * Math.sin(ax * 13);
      x.fillStyle = rgb(80 + fold * 50, 18 + fold * 10, 20 + fold * 8);
      rect(ax, 1.5, ax + 0.081, 4.55);
    }
    x.fillStyle = 'rgba(214,170,80,0.55)';
    for (var my = 1.9; my < 4.3; my += 0.6) {
      for (var mx = -1.15; mx < 1.3; mx += 0.58) {
        x.beginPath(); x.arc(X(mx), Y(my), 0.09 * sx, 0, Math.PI * 2); x.fill();
      }
    }
    frame(-1.5, 1.5, 1.5, 4.6, '#c9a24a', 5);
    // Cut out the balcony and the lords' rooms.
    x.globalCompositeOperation = 'destination-out';
    x.fillStyle = '#000';
    arch(-1.35, 5.35, 1.35, 7.62); x.fill();
    rect(-4.75, 5.45, -3.05, 7.4);
    rect(3.05, 5.45, 4.75, 7.4);
    x.globalCompositeOperation = 'source-over';
    x.strokeStyle = '#c9a24a';
    x.lineWidth = 4;
    arch(-1.4, 5.35, 1.4, 7.67); x.stroke();
  };
}

// A drape with soft vertical folds and a gold fringe.
function drape(x, w, h) {
  for (var i = 0; i < w; i++) {
    var f = 0.5 + 0.5 * Math.sin(i / w * Math.PI * 9);
    x.fillStyle = rgb(70 + f * 70, 14 + f * 14, 18 + f * 10);
    x.fillRect(i, 0, 1, h);
  }
  x.fillStyle = '#c9a24a';
  x.fillRect(0, h - 10, w, 10);
}

// The fixed star: a hot point with four long spikes.
function starSprite() {
  var c = document.createElement('canvas');
  c.width = c.height = 128;
  var x = c.getContext('2d'), g = x.createRadialGradient(64, 64, 0, 64, 64, 64);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(0.08, 'rgba(240,244,255,0.9)');
  g.addColorStop(0.25, 'rgba(170,190,255,0.25)');
  g.addColorStop(1, 'rgba(120,150,255,0)');
  x.fillStyle = g;
  x.fillRect(0, 0, 128, 128);
  x.globalCompositeOperation = 'lighter';
  [[128, 3], [3, 128]].forEach(function (s) {
    var lg = x.createRadialGradient(64, 64, 0, 64, 64, 64);
    lg.addColorStop(0, 'rgba(255,255,255,0.9)');
    lg.addColorStop(1, 'rgba(255,255,255,0)');
    x.fillStyle = lg;
    x.fillRect(64 - s[0] / 2, 64 - s[1] / 2, s[0], s[1]);
  });
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// A candle flame: a white core in an orange teardrop.
function flameSprite() {
  var c = document.createElement('canvas');
  c.width = 64; c.height = 128;
  var x = c.getContext('2d');
  x.translate(32, 92);
  x.scale(1, 2.6);
  var g = x.createRadialGradient(0, -6, 0, 0, -4, 26);
  g.addColorStop(0, 'rgba(255,255,240,1)');
  g.addColorStop(0.3, 'rgba(255,214,120,0.95)');
  g.addColorStop(0.65, 'rgba(255,130,40,0.5)');
  g.addColorStop(1, 'rgba(255,90,20,0)');
  x.fillStyle = g;
  x.beginPath(); x.arc(0, -4, 26, 0, Math.PI * 2); x.fill();
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// ── Geometry helpers ─────────────────────────────────────────────────────
function polar(r, a) { return [r * Math.sin(a), r * Math.cos(a)]; }
// A flat slab between y0 and y1 over a polygon of [x, z] points; its caps
// get world-space UVs (metres), so boards tile at a constant size.
function slab(pts, y0, y1) {
  var shape = new THREE.Shape(pts.map(function (p) { return new THREE.Vector2(p[0], p[1]); }));
  return new THREE.ExtrudeGeometry(shape, { depth: y1 - y0, bevelEnabled: false })
    .rotateX(Math.PI / 2).translate(0, y1, 0);
}
// The trapezoid of bay k between radii r0 and r1 (vertex radii).
function bayPts(k, r0, r1) {
  var a0 = (k - 0.5) * BAY, a1 = (k + 0.5) * BAY;
  return [polar(r0, a0), polar(r0, a1), polar(r1, a1), polar(r1, a0)];
}
// A box from a to b (each [x, z]) at height y0..y1, `t` thick.
function beam(a, b, y0, y1, t) {
  var dx = b[0] - a[0], dz = b[1] - a[1], len = Math.hypot(dx, dz);
  return new THREE.BoxGeometry(len, y1 - y0, t).rotateY(-Math.atan2(dz, dx))
    .translate((a[0] + b[0]) / 2, (y0 + y1) / 2, (a[1] + b[1]) / 2);
}
// Like kit merge(), but keeps UVs so the boards texture still maps.
function mergeUV(geos) {
  var total = 0;
  geos.forEach(function (g) { total += g.attributes.position.count; });
  var out = new THREE.BufferGeometry();
  [['position', 3], ['normal', 3], ['color', 3], ['uv', 2]].forEach(function (a) {
    var arr = new Float32Array(total * a[1]), o = 0;
    geos.forEach(function (g) {
      if (g.attributes[a[0]]) arr.set(g.attributes[a[0]].array, o);
      o += g.attributes.position.count * a[1];
    });
    out.setAttribute(a[0], new THREE.BufferAttribute(arr, a[1]));
  });
  geos.forEach(function (g) { g.dispose(); });
  return out;
}
// A turned baluster about 0.8 m tall.
function balusterGeometry() {
  var pts = [[0.05, 0], [0.05, 0.06], [0.03, 0.1], [0.055, 0.3], [0.03, 0.5], [0.025, 0.62], [0.045, 0.7], [0.045, 0.8]];
  return new THREE.LatheGeometry(pts.map(function (p) { return new THREE.Vector2(p[0], p[1]); }), 6);
}

// A rose lying on the boards: whorls of cupped petals on a short stem.
function roseGeometry() {
  var parts = [], red = new THREE.Color('#9a0f1c'), deep = new THREE.Color('#4a0610');
  for (var layer = 0; layer < 5; layer++) {
    var n = 3 + layer, rad = 0.035 + layer * 0.022, open = 0.25 + layer * 0.24;
    for (var p = 0; p < n; p++) {
      var g = new THREE.SphereGeometry(rad, 14, 8, 0, Math.PI * 0.9, 0, Math.PI * 0.55)
        .rotateX(-open).translate(0, 0, rad * 0.35 * layer / 4)
        .rotateY(p / n * Math.PI * 2 + layer * 0.7).translate(0, 0.06 - layer * 0.006, 0);
      parts.push(tinted(g, deep.clone().lerp(red, 0.4 + layer * 0.15)));
    }
  }
  parts.push(tinted(new THREE.SphereGeometry(0.06, 8, 5, 0, Math.PI * 2, Math.PI * 0.5, Math.PI * 0.5).translate(0, 0.04, 0), '#1e3a16'));
  parts.push(tinted(new THREE.CylinderGeometry(0.008, 0.01, 0.6, 5).translate(0, -0.26, 0), '#24401a'));
  [[-0.2, 0.6], [-0.36, -0.8]].forEach(function (lf) {
    parts.push(tinted(new THREE.SphereGeometry(0.05, 6, 4).scale(1, 0.15, 0.5).translate(0.05, 0, 0).rotateY(lf[1]).translate(0, lf[0], 0), '#2c5020'));
  });
  // Lay it down: stem along +x, bloom tilted up a little off the boards.
  return merge(parts).rotateZ(Math.PI / 2 + 0.25).translate(-0.05, 0.07, 0);
}

// ── Shaders ──────────────────────────────────────────────────────────────
var NOISE =
  'float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }\n' +
  'float noise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);\n' +
  ' return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y); }\n' +
  'float fbm(vec2 p){ return noise(p) * 0.55 + noise(p * 2.1 + 3.1) * 0.3 + noise(p * 4.3 + 7.7) * 0.15; }\n';

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(1599);
  var gl = makeRenderer(canvas, { clear: '#070a16', shadows: !small });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#0b1020', 0.012);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.05, 2500);
  world.add(camera);

  // ── Sky, stars and the fixed star ──────────────────────────────────────
  var dome = skyDome({ top: '#03050f', mid: '#0a1230', horizon: '#141c3c' }, 1200);
  world.add(dome.mesh);

  var N = small ? 2600 : 5200, pos = [], attr = [];
  for (var i = 0; i < N; i++) {
    var th = r() * Math.PI * 2, y = 0.25 + r() * 0.75, s = Math.sqrt(1 - y * y);
    pos.push(1000 * s * Math.cos(th), 1000 * y, 1000 * s * Math.sin(th));
    attr.push(0.8 + Math.pow(r(), 3) * 3.4, r(), r() * 6.28, r());   // size, dies at, phase, warmth
  }
  var starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  starGeo.setAttribute('star', new THREE.Float32BufferAttribute(attr, 4));
  var starMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
    uniforms: { uTime: { value: 0 }, uFlare: { value: 0 }, uOut: { value: 0 }, uDim: { value: 1 }, uScale: { value: 1 } },
    vertexShader: 'attribute vec4 star; uniform float uTime; uniform float uFlare; uniform float uOut; uniform float uDim; uniform float uScale;\n' +
      'varying float vA; varying float vWarm;\n' +
      'void main(){ gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);\n' +
      ' float alive = smoothstep(uOut * 1.08 - 0.06, uOut * 1.08 + 0.02, star.y);\n' +
      ' float tw = 0.72 + 0.28 * sin(uTime * (1.4 + star.w * 2.2) + star.z);\n' +
      ' float burn = uFlare * (0.75 + 0.25 * sin(uTime * 9.0 + star.z * 3.0));\n' +
      ' vWarm = uFlare * (0.4 + 0.6 * star.w);\n' +
      ' vA = alive * tw * uDim * (1.0 + burn * 1.4);\n' +
      ' gl_PointSize = star.x * uScale * (1.0 + burn * 2.2); }',
    fragmentShader: 'varying float vA; varying float vWarm;\n' +
      'void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;\n' +
      ' float a = smoothstep(0.5, 0.0, d) * vA;\n' +
      ' vec3 col = mix(vec3(0.82, 0.88, 1.0), vec3(1.0, 0.6, 0.28), vWarm);\n' +
      ' gl_FragColor = vec4(col * a, a);\n #include <colorspace_fragment>\n }'
  });
  var stars = new THREE.Points(starGeo, starMat);
  stars.frustumCulled = false;
  world.add(stars);

  var fixedStar = new THREE.Sprite(new THREE.SpriteMaterial({ map: starSprite(), transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, fog: false, opacity: 0 }));
  fixedStar.position.copy(STAR_DIR).multiplyScalar(900);
  fixedStar.scale.setScalar(70);
  world.add(fixedStar);

  // ── Light ──────────────────────────────────────────────────────────────
  var hemi = new THREE.HemisphereLight('#5a6ea8', '#1a120c', 0.6);
  var moon = new THREE.DirectionalLight('#a8bcf0', 1.2);
  moon.position.set(-14, 34, 16);
  var wash = new THREE.DirectionalLight('#ffc070', 0);      // the summer gold, and the red
  wash.position.set(18, 14, 20);
  world.add(hemi, moon, moon.target, wash);
  if (!small) {
    moon.castShadow = true;
    moon.shadow.mapSize.set(2048, 2048);
    var sc = moon.shadow.camera;
    sc.left = sc.bottom = -22; sc.right = sc.top = 22; sc.near = 5; sc.far = 90;
    moon.shadow.bias = -0.0006;
    moon.shadow.normalBias = 0.03;
  }

  // ── Materials ──────────────────────────────────────────────────────────
  var woodTex = canvasTexture(256, 256, boards(r), [0.5, 0.5]);
  var woodMat = new THREE.MeshLambertMaterial({ map: woodTex, vertexColors: true });
  var plasterMat = new THREE.MeshLambertMaterial({ map: canvasTexture(256, 512, framing(r), [17, 1]),
    color: '#8c8476', side: THREE.BackSide });
  var thatchTex = canvasTexture(256, 256, thatch(r), [1, 1]);
  var thatchMat = new THREE.MeshLambertMaterial({ map: thatchTex, side: THREE.DoubleSide });
  var yardMat = new THREE.MeshLambertMaterial({ map: canvasTexture(256, 256, earth(r), [7, 7]) });
  var marbleMat = new THREE.MeshLambertMaterial({ map: canvasTexture(128, 512, marble(r), [2, 1]) });
  var goldMat = new THREE.MeshLambertMaterial({ color: '#b08a3a', emissive: '#2a1c06' });

  function add(geo, mat, noShadow) {
    var m = new THREE.Mesh(geo, mat);
    if (!noShadow) { m.castShadow = true; m.receiveShadow = true; }
    world.add(m);
    return m;
  }

  // ── The yard ───────────────────────────────────────────────────────────
  add(new THREE.CircleGeometry(R1 + 0.6, 40).rotateX(-Math.PI / 2), yardMat).castShadow = false;

  // ── Galleries: floors, benches, posts, rails, balusters, back wall ─────
  var wood = [], rails = [], posts = {};
  var oak = new THREE.Color('#e8d4bc'), dim = new THREE.Color('#a89880'), bench = new THREE.Color('#b8a48a');
  for (var k = 0; k < SIDES; k++) {
    if (stageBay(k)) continue;
    wood.push(tinted(slab(bayPts(k, R1, R2), 0, FLOORS[0]), dim));
    wood.push(tinted(slab(bayPts(k, R1, R2), FLOORS[1] - 0.3, FLOORS[1]), oak));
    wood.push(tinted(slab(bayPts(k, R1, R2), FLOORS[2] - 0.3, FLOORS[2]), oak));
    wood.push(tinted(slab(bayPts(k, R1, R2), TOP - 0.3, TOP), dim));
    FLOORS.forEach(function (fy, lv) {
      for (var t = 0; t < 3; t++) {
        wood.push(tinted(slab(bayPts(k, R1 + 0.9 + t * 0.8, R2 - 0.05), fy, fy + 0.4 * (t + 1)), bench));
      }
      var a = polar(R1 + 0.08, (k - 0.5) * BAY), b = polar(R1 + 0.08, (k + 0.5) * BAY);
      rails.push(tinted(beam(a, b, fy + 0.86, fy + 0.98, 0.14), oak));
      rails.push(tinted(beam(a, b, fy + 0.04, fy + 0.12, 0.12), oak));
      if (lv > 0) rails.push(tinted(beam(polar(R1, (k - 0.5) * BAY), polar(R1, (k + 0.5) * BAY), fy - 0.42, fy, 0.22), '#c8a888'));
    });
    rails.push(tinted(beam(polar(R1, (k - 0.5) * BAY), polar(R1, (k + 0.5) * BAY), TOP - 0.45, TOP, 0.24), '#c8a888'));
    posts[k] = posts[k + 1] = true;
  }
  // Walls at each end of the tiring house close the galleries' sides.
  [8.5, 11.5].forEach(function (e) {
    wood.push(tinted(beam(polar(R1 + 0.1, e * BAY), polar(R2, e * BAY), 0, TOP, 0.2), '#a89070'));
  });
  Object.keys(posts).forEach(function (kk) {
    var p = polar(R1, (kk - 0.5) * BAY);
    wood.push(tinted(new THREE.BoxGeometry(0.3, TOP, 0.3).rotateY(-(kk - 0.5) * BAY).translate(p[0], TOP / 2, p[1]), '#d8bc9c'));
  });
  add(mergeUV(wood.concat(rails)), woodMat);

  var balusters = new THREE.InstancedMesh(balusterGeometry(), new THREE.MeshLambertMaterial({ color: '#8a6a4c' }), 17 * 3 * 11);
  var bm = new THREE.Matrix4(), bq = new THREE.Quaternion(), bs = new THREE.Vector3(1, 1, 1), bp = new THREE.Vector3(), bn = 0;
  for (k = 0; k < SIDES; k++) {
    if (stageBay(k)) continue;
    var a0 = polar(R1 + 0.08, (k - 0.5) * BAY), a1 = polar(R1 + 0.08, (k + 0.5) * BAY);
    for (var lv = 0; lv < 3; lv++) {
      for (var b = 1; b <= 11; b++) {
        var tt = b / 12;
        bp.set(lerp(a0[0], a1[0], tt), FLOORS[lv] + 0.1, lerp(a0[1], a1[1], tt));
        balusters.setMatrixAt(bn++, bm.compose(bp, bq, bs));
      }
    }
  }
  balusters.count = bn;
  balusters.castShadow = balusters.receiveShadow = true;
  world.add(balusters);

  add(new THREE.CylinderGeometry(R2, R2, TOP - FLOORS[0], 17, 1, true, 11.5 * BAY, 17 * BAY)
    .translate(0, (TOP + FLOORS[0]) / 2, 0), plasterMat).castShadow = false;

  // ── The thatched ring ──────────────────────────────────────────────────
  // Low at the yard edge, a ridge over the galleries, falling away outside.
  var RA = R1 - 0.9, YA = 10.3, RB = R2 - 1.8, YB = 13.4, RC = R2 + 0.9, YC = 11.2;
  var tp = [], tu = [];
  function quad(p0, p1, p2, p3, u0, u1, v0, v1) {
    tp.push(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], p0[0], p0[1], p0[2], p2[0], p2[1], p2[2], p3[0], p3[1], p3[2]);
    tu.push(u0, v0, u1, v0, u1, v1, u0, v0, u1, v1, u0, v1);
  }
  function pt(rr, yy, a) { var q = polar(rr, a); return [q[0], yy, q[1]]; }
  for (k = 0; k < SIDES; k++) {
    var c0 = (k - 0.5) * BAY, c1 = (k + 0.5) * BAY;
    quad(pt(RA, YA, c0), pt(RA, YA, c1), pt(RB, YB, c1), pt(RB, YB, c0), 0, 3, 0, 3.4);
    quad(pt(RB, YB, c0), pt(RB, YB, c1), pt(RC, YC, c1), pt(RC, YC, c0), 0, 3, 3.4, 6.4);
    quad(pt(RA, YA - 0.45, c0), pt(RA, YA - 0.45, c1), pt(RA, YA, c1), pt(RA, YA, c0), 0, 3, 0, 0.3);
    quad(pt(RA, YA - 0.45, c0), pt(RA, YA - 0.45, c1), pt(R1 + 0.2, TOP - 0.3, c1), pt(R1 + 0.2, TOP - 0.3, c0), 0, 3, 0, 0.6);
  }
  var thatchGeo = new THREE.BufferGeometry();
  thatchGeo.setAttribute('position', new THREE.Float32BufferAttribute(tp, 3));
  thatchGeo.setAttribute('uv', new THREE.Float32BufferAttribute(tu, 2));
  thatchGeo.computeVertexNormals();
  add(thatchGeo, thatchMat);

  // ── The stage, the tiring house and the heavens ────────────────────────
  var stage = [];
  stage.push(tinted(slab([[-STAGE_X, ZF - 0.6], [STAGE_X, ZF - 0.6], [STAGE_X, STAGE_FRONT], [-STAGE_X, STAGE_FRONT]], 0, STAGE_Y), '#e8d0b0'));
  // Cornices on the frons, the heavens slab and its gilt fascia.
  stage.push(tinted(new THREE.BoxGeometry(2 * XF, 0.3, 0.4).translate(0, 5.18, ZF + 0.15), '#ffd890'));
  stage.push(tinted(new THREE.BoxGeometry(2 * XF, 0.25, 0.35).translate(0, 7.95, ZF + 0.12), '#ffd890'));
  stage.push(tinted(slab([[-6.7, ZF - 0.5], [6.7, ZF - 0.5], [6.7, HEAV_FRONT], [-6.7, HEAV_FRONT]], HEAV_Y, HEAV_Y + 0.4), '#c8a888'));
  stage.push(tinted(new THREE.BoxGeometry(13.6, 0.5, 0.2).translate(0, HEAV_Y + 0.15, HEAV_FRONT + 0.05), '#ffcc70'));
  // The balcony: a platform on brackets, a rail and balusters.
  stage.push(tinted(slab([[-1.8, ZF - 2.2], [1.8, ZF - 2.2], [1.8, ZF + 0.75], [-1.8, ZF + 0.75]], 5.2, 5.38), '#d8c0a0'));
  stage.push(tinted(new THREE.BoxGeometry(3.6, 0.1, 0.12).translate(0, 6.25, ZF + 0.7), '#e8c8a0'));
  [-1, 1].forEach(function (sd) {
    stage.push(tinted(new THREE.BoxGeometry(0.1, 0.1, 0.75).translate(sd * 1.75, 6.25, ZF + 0.35), '#e8c8a0'));
    stage.push(tinted(new THREE.BoxGeometry(0.16, 0.9, 0.16).rotateX(0.6).translate(sd * 1.2, 4.85, ZF + 0.3), '#a88866'));
  });
  for (var bb = 0; bb < 13; bb++) {
    stage.push(tinted(balusterGeometry().scale(0.9, 1.05, 0.9).toNonIndexed().translate(-1.62 + bb * 0.27, 5.38, ZF + 0.7), '#e0c098'));
  }
  // A boarded edge under the stage lip.
  stage.push(tinted(new THREE.BoxGeometry(2 * STAGE_X + 0.2, 0.16, 0.2).translate(0, STAGE_Y - 0.08, STAGE_FRONT + 0.05), '#d8b890'));
  // Pillar plinths.
  PILLARS.forEach(function (p) {
    stage.push(tinted(new THREE.BoxGeometry(0.95, 0.7, 0.95).translate(p[0], STAGE_Y + 0.35, p[1]), '#d0c0a8'));
    stage.push(tinted(new THREE.BoxGeometry(1.0, 0.3, 1.0).translate(p[0], HEAV_Y - 0.15, p[1]), '#ffcc70'));
  });
  add(mergeUV(stage), woodMat);

  // Black-red cloth hung round the stage front and sides.
  var skirtTex = canvasTexture(512, 64, function (x, w, h) {
    for (var i = 0; i < w; i++) {
      var fo = 0.5 + 0.5 * Math.sin(i / w * Math.PI * 2 * 26);
      x.fillStyle = rgb(70 + fo * 60, 22 + fo * 16, 22 + fo * 12);
      x.fillRect(i, 0, 1, h);
    }
    x.fillStyle = '#8a6a30';
    x.fillRect(0, 0, w, 4);
  }, [3, 1]);
  var skirtMat = new THREE.MeshLambertMaterial({ map: skirtTex, side: THREE.DoubleSide });
  add(new THREE.PlaneGeometry(2 * STAGE_X + 0.2, STAGE_Y - 0.16).translate(0, (STAGE_Y - 0.16) / 2, STAGE_FRONT + 0.16), skirtMat);
  [-1, 1].forEach(function (sd) {
    add(new THREE.PlaneGeometry(STAGE_FRONT - ZF, STAGE_Y - 0.16).rotateY(Math.PI / 2)
      .translate(sd * (STAGE_X + 0.1), (STAGE_Y - 0.16) / 2, (STAGE_FRONT + ZF) / 2), skirtMat);
  });

  // The hut over the heavens: a gabled roof, thatched like the ring.
  var hz0 = ZF - 1.4, hz1 = HEAV_FRONT - 0.2, hw = 7.0, hy0 = HEAV_Y + 0.4, hy1 = 12.4;
  var hp = [], hu = [];
  function hq(p0, p1, p2, p3) {
    hp.push(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], p0[0], p0[1], p0[2], p2[0], p2[1], p2[2], p3[0], p3[1], p3[2]);
    hu.push(0, 0, 3, 0, 3, 3, 0, 0, 3, 3, 0, 3);
  }
  hq([-hw, hy0, hz1], [-hw, hy0, hz0], [0, hy1, hz0], [0, hy1, hz1]);
  hq([hw, hy0, hz0], [hw, hy0, hz1], [0, hy1, hz1], [0, hy1, hz0]);
  var hutGeo = new THREE.BufferGeometry();
  hutGeo.setAttribute('position', new THREE.Float32BufferAttribute(hp, 3));
  hutGeo.setAttribute('uv', new THREE.Float32BufferAttribute(hu, 2));
  hutGeo.computeVertexNormals();
  add(hutGeo, thatchMat);
  var gable = new THREE.Shape([new THREE.Vector2(-hw + 0.3, hy0), new THREE.Vector2(hw - 0.3, hy0), new THREE.Vector2(0, hy1 - 0.2)]);
  add(tinted(new THREE.ShapeGeometry(gable).translate(0, 0, hz1 + 0.05), '#c09070'), woodMat);

  // The heavens' painted underside.
  var heavensMesh = add(new THREE.PlaneGeometry(13.4, HEAV_FRONT - ZF + 0.5).rotateX(Math.PI / 2)
    .translate(0, HEAV_Y - 0.01, (HEAV_FRONT + ZF - 0.5) / 2),
    new THREE.MeshLambertMaterial({ map: canvasTexture(1024, 512, heavens(r)), emissive: '#080c22' }));
  heavensMesh.castShadow = false;

  // The frons scenae, with the balcony and lords' rooms cut out.
  var fronsTex = canvasTexture(1024, 760, fronsPaint(r));
  fronsTex.wrapS = fronsTex.wrapT = THREE.ClampToEdgeWrapping;
  var frons = add(new THREE.PlaneGeometry(2 * XF, TOP - STAGE_Y).translate(0, (TOP + STAGE_Y) / 2, ZF),
    new THREE.MeshLambertMaterial({ map: fronsTex, alphaTest: 0.5, side: THREE.DoubleSide }));
  frons.material.shadowSide = THREE.DoubleSide;
  // Wood behind it, so the cut-outs only open into the rooms.
  add(new THREE.PlaneGeometry(2 * XF + 1, TOP).translate(0, TOP / 2, ZF - 2.3), new THREE.MeshLambertMaterial({ color: '#1a100a' }), true);

  // Rooms behind the openings: boxes seen from inside.
  var roomTex = canvasTexture(256, 256, function (x, w, h) {
    var g = x.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, '#c9a070'); g.addColorStop(0.6, '#b08454'); g.addColorStop(0.62, '#4a2a18'); g.addColorStop(1, '#3a2012');
    x.fillStyle = g;
    x.fillRect(0, 0, w, h);
    x.fillStyle = 'rgba(70,40,20,0.4)';
    for (var px = 0; px < w; px += 32) x.fillRect(px, h * 0.62, 3, h * 0.38);
  });
  var roomMat = new THREE.MeshLambertMaterial({ map: roomTex, side: THREE.BackSide, emissive: '#000000' });
  var room = new THREE.Mesh(new THREE.BoxGeometry(3.0, 2.5, 2.2).translate(0, 5.35 + 1.25, ZF - 1.1), roomMat);
  room.receiveShadow = true;
  world.add(room);
  [-3.9, 3.9].forEach(function (cx) {
    world.add(new THREE.Mesh(new THREE.BoxGeometry(1.9, 2.1, 2.0).translate(cx, 5.45 + 1.05, ZF - 1.0),
      new THREE.MeshLambertMaterial({ color: '#3a2618', side: THREE.BackSide })));
  });

  // A leaded window at the back of the balcony room, faintly moonlit.
  var lattice = canvasTexture(128, 160, function (x, w, h) {
    x.fillStyle = '#1a2440';
    x.fillRect(0, 0, w, h);
    x.strokeStyle = '#120c08';
    x.lineWidth = 3;
    for (var d = -h; d < w + h; d += 18) {
      x.beginPath(); x.moveTo(d, 0); x.lineTo(d + h, h); x.stroke();
      x.beginPath(); x.moveTo(d, h); x.lineTo(d + h, 0); x.stroke();
    }
    x.lineWidth = 10;
    x.strokeRect(0, 0, w, h);
    x.fillStyle = '#120c08';
    x.fillRect(w / 2 - 4, 0, 8, h);
  });
  var windowMat = new THREE.MeshBasicMaterial({ map: lattice, color: '#7a8ab8' });
  var win = new THREE.Mesh(new THREE.PlaneGeometry(0.9, 1.15), windowMat);
  win.position.set(-0.1, 6.55, ZF - 2.18);
  world.add(win);

  // The balcony curtain, half drawn, and its lamp.
  var CW = 8, CH = 12;
  var curtainGeo = new THREE.PlaneGeometry(1.1, 2.15, CW, CH).translate(-0.75, 5.38 + 1.075, ZF - 0.25);
  var curtainBase = Float32Array.from(curtainGeo.attributes.position.array);
  var curtain = new THREE.Mesh(curtainGeo, new THREE.MeshLambertMaterial({ map: canvasTexture(128, 256, drape), side: THREE.DoubleSide }));
  world.add(curtain);
  var balcLight = new THREE.PointLight('#ffb870', 0, 7, 1.3);
  balcLight.position.set(0.6, 6.9, ZF - 1.2);
  var balcGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,200,130,1)', 'rgba(255,150,80,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0 }));
  balcGlow.position.copy(balcLight.position);
  balcGlow.scale.setScalar(1.2);
  world.add(balcLight, balcGlow);
  var lamp = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.09, 0.2, 8), new THREE.MeshBasicMaterial({ color: '#ffd8a0' }));
  lamp.position.copy(balcLight.position);
  world.add(lamp);
  world.add(new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.55, 3).translate(0.6, 7.3, ZF - 1.2), new THREE.MeshBasicMaterial({ color: '#1a120a' })));

  // The pillars: painted marble shafts with gilt capitals.
  var shaftPts = [[0.34, 0], [0.34, 0.1], [0.3, 0.18], [0.3, 0.3], [0.29, 1.5], [0.26, 5.0], [0.25, 5.4]]
    .map(function (p) { return new THREE.Vector2(p[0], p[1]); });
  var capPts = [[0.25, 0], [0.3, 0.05], [0.28, 0.15], [0.36, 0.35], [0.46, 0.5], [0.46, 0.58]]
    .map(function (p) { return new THREE.Vector2(p[0], p[1]); });
  PILLARS.forEach(function (p) {
    add(new THREE.LatheGeometry(shaftPts, 16).translate(p[0], STAGE_Y + 0.7, p[1]), marbleMat);
    add(new THREE.LatheGeometry(capPts, 16).translate(p[0], STAGE_Y + 6.1, p[1]), goldMat);
  });

  // ── Props and cues on the stage ────────────────────────────────────────
  var rose = new THREE.Mesh(roseGeometry(), new THREE.MeshLambertMaterial({ vertexColors: true, emissive: '#1a0204' }));
  rose.position.copy(ROSE);
  rose.rotation.y = 0.5;
  rose.scale.setScalar(1.6);
  rose.castShadow = true;
  world.add(rose);

  var spot = new THREE.SpotLight('#ffe2b8', 0, 16, 0.3, 0.7, 1.1);
  spot.position.set(ROSE.x - 0.6, HEAV_Y - 0.1, ROSE.z - 2.2);
  spot.target.position.copy(ROSE);
  world.add(spot, spot.target);

  // The candle: wax, a flame sprite and a light whose reach grows.
  var candle = new THREE.Group();
  candle.position.copy(CANDLE);
  candle.add(new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.18, 0.04, 14).translate(0, 0.02, 0), new THREE.MeshLambertMaterial({ color: '#6a5a3a' })));
  candle.add(new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.05, 0.3, 12).translate(0, 0.19, 0),
    new THREE.MeshLambertMaterial({ color: '#efe4c8', emissive: '#3a2410' })));
  var flame = new THREE.Sprite(new THREE.SpriteMaterial({ map: flameSprite(), blending: THREE.AdditiveBlending,
    depthWrite: false, transparent: true, opacity: 0 }));
  flame.position.set(0, 0.42, 0);
  flame.scale.set(0.07, 0.15, 1);
  var halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,190,110,1)', 'rgba(255,140,60,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0 }));
  halo.position.set(0, 0.42, 0);
  candle.add(flame, halo);
  var candleLight = new THREE.PointLight('#ffb064', 0, 6, 1.1);
  candleLight.position.set(0, 0.5, 0);
  candle.add(candleLight);
  world.add(candle);

  // Music: motes of light rising off the stage, moved on the GPU.
  var M = small ? 300 : 600, mp = [], ms = [];
  for (i = 0; i < M; i++) {
    mp.push(-5.5 + r() * 11, STAGE_Y, ZF + 0.6 + r() * (STAGE_FRONT - ZF - 0.8));
    ms.push(r(), 0.04 + r() * 0.07, r() * 6.28, 0.5 + Math.pow(r(), 2) * 1.6);   // phase, rise speed, sway phase, size
  }
  var moteGeo = new THREE.BufferGeometry();
  moteGeo.setAttribute('position', new THREE.Float32BufferAttribute(mp, 3));
  moteGeo.setAttribute('aMote', new THREE.Float32BufferAttribute(ms, 4));
  var moteMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uTime: { value: 0 }, uAmt: { value: 0 }, uScale: { value: 1 } },
    vertexShader: 'attribute vec4 aMote; uniform float uTime; uniform float uAmt; uniform float uScale; varying float vA;\n' +
      'void main(){ float life = fract(aMote.x + uTime * aMote.y);\n' +
      ' vec3 p = position + vec3(sin(uTime * 0.7 + aMote.z + life * 6.0) * 0.5, life * 10.0, cos(uTime * 0.5 + aMote.z) * 0.3);\n' +
      ' vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_Position = projectionMatrix * mv;\n' +
      ' vA = uAmt * smoothstep(0.0, 0.12, life) * (1.0 - smoothstep(0.6, 1.0, life)) * (0.6 + 0.4 * sin(uTime * 3.0 + aMote.z * 5.0));\n' +
      ' gl_PointSize = aMote.w * uScale * 190.0 / -mv.z; }',
    fragmentShader: 'varying float vA;\n' +
      'void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;\n' +
      ' float a = (pow(smoothstep(0.5, 0.0, d), 2.0) * 0.7 + smoothstep(0.12, 0.0, d)) * vA;\n' +
      ' gl_FragColor = vec4(vec3(1.0, 0.68, 0.3) * a, a);\n #include <colorspace_fragment>\n }'
  });
  var motes = new THREE.Points(moteGeo, moteMat);
  motes.frustumCulled = false;
  world.add(motes);

  // The course of true love: a ribbon of light on the boards that winds,
  // doubles back and kinks. aRib = (along 0..1, across -1..1).
  var path = new THREE.CatmullRomCurve3([[-4.8, -4.5], [-3.4, -5.0], [-3.7, -6.5], [-2.2, -7.3], [-1.0, -6.2], [-1.4, -5.0],
    [0.3, -4.6], [2.3, -5.1], [2.5, -6.7], [0.9, -7.6], [1.6, -9.2], [3.5, -8.9], [4.5, -10.0], [3.85, -11.6]]
    .map(function (p) { return new THREE.Vector3(p[0], STAGE_Y + 0.012, p[1]); }), false, 'centripetal');
  function ribbonGeo(width) {
    var SEG = 400, rp = [], ra = [], a = new THREE.Vector3(), tan = new THREE.Vector3();
    for (var s2 = 0; s2 <= SEG; s2++) {
      var t2 = s2 / SEG;
      path.getPointAt(t2, a);
      path.getTangentAt(t2, tan);
      // Small uneven jinks so it never runs smooth.
      var j = Math.sin(t2 * 97) * 0.035 + Math.sin(t2 * 41 + 1.3) * 0.05;
      var nx = -tan.z, nz = tan.x;
      rp.push(a.x + nx * (j - width), a.y, a.z + nz * (j - width), a.x + nx * (j + width), a.y, a.z + nz * (j + width));
      ra.push(t2, -1, t2, 1);
    }
    var idx = [];
    for (s2 = 0; s2 < SEG; s2++) { var q = s2 * 2; idx.push(q, q + 1, q + 2, q + 1, q + 3, q + 2); }
    var g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(rp, 3));
    g.setAttribute('aRib', new THREE.Float32BufferAttribute(ra, 2));
    g.setIndex(idx);
    return g;
  }
  var ribMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    uniforms: { uProg: { value: 0 }, uFade: { value: 1 }, uTime: { value: 0 } },
    vertexShader: 'attribute vec2 aRib; varying vec2 vR; void main(){ vR = aRib; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform float uProg; uniform float uFade; uniform float uTime; varying vec2 vR;\n' +
      'void main(){ float drawn = 1.0 - smoothstep(uProg - 0.01, uProg, vR.x);\n' +
      ' float core = exp(-vR.y * vR.y * 9.0), haze = exp(-vR.y * vR.y * 1.6) * 0.35;\n' +
      ' float pulse = 0.65 + 0.35 * sin(vR.x * 90.0 - uTime * 3.0);\n' +
      ' float head = exp(-pow((vR.x - uProg) * 40.0, 2.0)) * 1.5;\n' +
      ' float a = drawn * (core * pulse + haze + head * core) * uFade;\n' +
      ' gl_FragColor = vec4(vec3(1.0, 0.72, 0.55) * a, a);\n #include <colorspace_fragment>\n }'
  });
  var ribbon = new THREE.Mesh(ribbonGeo(0.24), ribMat);
  ribbon.frustumCulled = false;
  world.add(ribbon);
  var ribLight = new THREE.PointLight('#ffb48a', 0, 5, 1.4);
  world.add(ribLight);

  // A charmed life: a ring of light round the rose and a wall rising off it.
  var ringMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    uniforms: { uAmt: { value: 0 }, uTime: { value: 0 } },
    vertexShader: 'varying vec2 vP; void main(){ vP = position.xz; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform float uAmt; uniform float uTime; varying vec2 vP;\n' +
      'void main(){ float rr = length(vP), an = atan(vP.y, vP.x);\n' +
      ' float line = exp(-pow((rr - 1.1) / 0.035, 2.0)) + exp(-pow((rr - 0.92) / 0.02, 2.0)) * 0.6;\n' +
      ' float glow = exp(-pow((rr - 1.0) / 0.3, 2.0)) * 0.35;\n' +
      ' float runes = step(0.6, fract(an * 3.8197 + uTime * 0.05)) * exp(-pow((rr - 1.01) / 0.04, 2.0)) * 0.9;\n' +
      ' float a = (line + glow + runes) * uAmt;\n' +
      ' gl_FragColor = vec4(vec3(1.0, 0.93, 0.75) * a, a);\n #include <colorspace_fragment>\n }'
  });
  var ringDisk = new THREE.Mesh(new THREE.RingGeometry(0.6, 1.5, 96, 1).rotateX(-Math.PI / 2), ringMat);
  ringDisk.position.set(ROSE.x, STAGE_Y + 0.015, ROSE.z);
  var wallMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    uniforms: { uAmt: { value: 0 }, uTime: { value: 0 } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform float uAmt; uniform float uTime; varying vec2 vUv;\n' +
      'void main(){ float rise = pow(1.0 - vUv.y, 2.2);\n' +
      ' float streak = 0.55 + 0.45 * sin(vUv.x * 6.2832 * 18.0 + uTime * 1.2 + vUv.y * 5.0);\n' +
      ' float a = rise * streak * uAmt * 0.45;\n' +
      ' gl_FragColor = vec4(vec3(1.0, 0.92, 0.72) * a, a);\n #include <colorspace_fragment>\n }'
  });
  var ringWall = new THREE.Mesh(new THREE.CylinderGeometry(1.1, 1.1, 2.6, 64, 1, true).translate(0, 1.3, 0), wallMat);
  ringWall.position.copy(ringDisk.position);
  var ringLight = new THREE.PointLight('#fff0cc', 0, 7, 1.3);
  ringLight.position.set(ROSE.x, STAGE_Y + 1.2, ROSE.z);
  world.add(ringDisk, ringWall, ringLight);

  // Blind Cupid: a band of mist across the eyes, carried with the camera.
  var mistMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, depthTest: false, fog: false,
    uniforms: { uAmt: { value: 0 }, uTime: { value: 0 } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: NOISE + 'uniform float uAmt; uniform float uTime; varying vec2 vUv;\n' +
      'void main(){ vec2 p = (vUv - 0.5) * 4.0;\n' +
      ' float n = fbm(vec2(p.x * 1.6 - uTime * 0.12, p.y * 5.0 + sin(p.x * 0.8 + uTime * 0.1)));\n' +
      ' float band = exp(-pow((p.y + (n - 0.5) * 0.12) / 0.2, 2.0));\n' +
      ' float a = uAmt * band * (0.45 + 0.55 * n) * 0.78;\n' +
      ' gl_FragColor = vec4(vec3(0.62, 0.66, 0.78), a);\n #include <colorspace_fragment>\n }'
  });
  var mist = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), mistMat);
  mist.position.set(0, 0, -1);
  mist.renderOrder = 10;
  mist.frustumCulled = false;
  camera.add(mist);

  // ── Per-frame temps ────────────────────────────────────────────────────
  var tmp = new THREE.Color(), H = 800, portrait = false, baseFov = 55;
  var NIGHT_FOG = new THREE.Color('#0b1020'), GOLD_FOG = new THREE.Color('#3a2a18'), RED_FOG = new THREE.Color('#2a0606');
  var BLUE_FOG = new THREE.Color('#050a1c'), WARM_FOG = new THREE.Color('#2a1a0e');
  var ribPt = new THREE.Vector3();
  var curtainPos = curtainGeo.attributes.position;

  function frame(f) {
    var row = f.row, time = f.time, dt = f.dt;
    var dark = row[1], mistA = row[2], wind = row[3];
    var gold = row[9], starA = row[10], spotA = row[11], music = row[12], flare = row[13], rib = row[14];
    var red = row[15], ring = row[16], out = row[17], blue = row[18], balc = row[19], cand = row[20], spread = row[21];

    // Camera: a slow drift round the yard, eyes on the authored target.
    var drift = env.reduceMotion ? 0 : 1;
    camera.position.set(row[0] + Math.sin(time * 0.11) * 0.25 * drift, row[5] + Math.sin(time * 0.17) * 0.06 * drift, row[4]);
    if (portrait) camera.lookAt(row[23], row[24], row[25]);
    else camera.lookAt(row[6], row[7], row[8]);
    camera.rotateY(-f.mx * 0.08);
    camera.rotateX(-f.my * 0.05);
    var fov = baseFov * row[22];
    if (Math.abs(camera.fov - fov) > 0.01) { camera.fov = fov; camera.updateProjectionMatrix(); }

    // Sky: night, summer dusk, blood.
    var base = 1 - dark;
    dome.uniforms.top.value.set('#03050f').lerp(tmp.set('#3a5490'), gold * 0.8).lerp(tmp.set('#200306'), red * 0.8).lerp(tmp.set('#06102e'), blue);
    dome.uniforms.mid.value.set('#0a1230').lerp(tmp.set('#e09858'), gold * 0.85).lerp(tmp.set('#4a0808'), red * 0.8);
    dome.uniforms.dark.value = out * 0.6;
    starMat.uniforms.uTime.value = time;
    starMat.uniforms.uFlare.value = flare;
    starMat.uniforms.uOut.value = out;
    starMat.uniforms.uDim.value = (1 - gold * 0.85) * (1 - red * 0.6) * (1 - spread * 0.4);
    starMat.uniforms.uScale.value = Math.min(window.devicePixelRatio || 1, 2) * (H / 800 + 0.35);
    fixedStar.material.opacity = starA * (1 - out) * (1 - gold * 0.7) * (0.92 + 0.08 * Math.sin(time * 2.3));
    fixedStar.scale.setScalar(60 + starA * 30 + flare * 20);

    // Fill light and fog follow the cue.
    hemi.color.set('#6074b0').lerp(tmp.set('#ffcf90'), gold * 0.75).lerp(tmp.set('#d8281c'), red * 0.85).lerp(tmp.set('#4a6ae0'), blue);
    hemi.groundColor.set('#1a120c').lerp(tmp.set('#5a3a18'), gold * 0.6).lerp(tmp.set('#3a0606'), red);
    hemi.intensity = (3.8 - gold * 0.6 - red * 0.8 + blue * 1.2) * base * (1 - out * 0.45) + spread * 0.6;
    moon.intensity = (6.0 + blue * 2.0) * base * (1 - gold * 0.6) * (1 - out * 0.75) * (1 - red * 0.8);
    moon.color.set('#a8bcf0').lerp(tmp.set('#5a84ff'), blue);
    wash.color.set('#ffbe6a').lerp(tmp.set('#e0261a'), red / Math.max(red + gold, 0.001));
    wash.intensity = gold * 2.6 + red * 3.4;
    var fogC = world.fog.color.copy(NIGHT_FOG).lerp(GOLD_FOG, gold).lerp(RED_FOG, red).lerp(BLUE_FOG, blue).lerp(WARM_FOG, spread * 0.6);
    fogC.multiplyScalar(1 - dark * 0.5);
    world.fog.density = 0.012 + mistA * 0.035;
    gl.setClearColor(fogC);

    // The spot on the rose; cold and blue for the tragedy.
    spot.color.set('#ffe2b8').lerp(tmp.set('#6f8cff'), blue);
    spot.intensity = spotA * 60 + blue * 70;
    heavensMesh.material.emissive.set('#080c22').lerp(tmp.set('#2a0404'), red).multiplyScalar(1 - dark * 0.6);

    // Motes, ribbon, ring, mist.
    moteMat.uniforms.uTime.value = time;
    moteMat.uniforms.uAmt.value = music;
    moteMat.uniforms.uScale.value = Math.min(window.devicePixelRatio || 1, 2) * (H / 800);
    var prog = clamp(rib, 0, 1), ribFade = 1 - clamp(rib - 1, 0, 1);
    ribMat.uniforms.uProg.value = prog * 1.02;
    ribMat.uniforms.uFade.value = ribFade * (prog > 0 ? 1 : 0);
    ribMat.uniforms.uTime.value = time;
    path.getPointAt(clamp(prog, 0.001, 1), ribPt);
    ribLight.position.set(ribPt.x, ribPt.y + 0.5, ribPt.z);
    ribLight.intensity = (prog > 0 ? 4 : 0) * ribFade;
    ringMat.uniforms.uAmt.value = ring;
    ringMat.uniforms.uTime.value = time;
    wallMat.uniforms.uAmt.value = ring;
    wallMat.uniforms.uTime.value = time;
    ringLight.intensity = ring * 6;
    mistMat.uniforms.uAmt.value = mistA;
    mistMat.uniforms.uTime.value = time;
    mist.visible = mistA > 0.003;

    // The balcony lamp and the stir of its curtain.
    var flick = 0.92 + 0.08 * Math.sin(time * 7.3) * Math.sin(time * 3.1);
    balcLight.intensity = balc * 4 * flick;
    balcGlow.material.opacity = balc * 0.6 * flick;
    lamp.material.color.set('#3a2a1a').lerp(tmp.set('#ffd8a0'), clamp(balc * 1.5, 0, 1));
    roomMat.emissive.set('#000000').lerp(tmp.set('#24140a'), balc);
    windowMat.color.set('#7a8ab8').multiplyScalar(0.35 + 0.65 * (1 - out) * base);
    var cp = curtainBase, arr = curtainPos.array, sway = (0.04 + wind * 0.12) * drift;
    for (var v = 0; v < arr.length; v += 3) {
      var hang = clamp((5.38 + 2.15 - cp[v + 1]) / 2.15, 0, 1);
      arr[v + 2] = cp[v + 2] + Math.sin(time * 1.3 + cp[v] * 4 + cp[v + 1] * 1.5) * sway * hang;
    }
    curtainPos.needsUpdate = true;

    // The candle: its reach spreads until it lights the whole house.
    var cf = 0.9 + 0.1 * Math.sin(time * 11) * Math.sin(time * 4.7 + 1);
    flame.material.opacity = cand;
    flame.scale.set(0.075 * (0.95 + 0.05 * cf), 0.16 * cf, 1);
    halo.material.opacity = cand * (0.55 + spread * 0.3) * cf;
    halo.scale.setScalar(0.9 + spread * 1.6);
    candleLight.intensity = cand * (2.5 + spread * 30) * cf;
    candleLight.distance = lerp(5, 48, spread);

    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) {
      H = h;
      portrait = w / h < 1;
      fitCamera(gl, camera, w, h, dpr, small);
      baseFov = camera.fov;
    },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

// Column order after the unit, as the keys below name them.
var COLS = ['cx', 'dark', 'mist', 'wind', 'cz', 'cy', 'lx', 'ly', 'lz', 'gold', 'star', 'spot', 'music', 'flare',
            'ribbon', 'red', 'ring', 'out', 'blue', 'balc', 'candle', 'spread', 'zoom', 'plx', 'ply', 'plz'];

PI.register('globe-theatre', {
  renderer: renderer3d,
  align: ['left', 'right', 'left', 'right', 'left', 'right', 'left', 'center', 'right', 'left', 'right', 'left', 'right', 'center'],
  scrim: 0.6,
  // Each key changes only what it names; everything else carries on.
  keys: function (T) {
    var S = T.start, rows = [];
    var st = { cx: 0, dark: 0, mist: 0, wind: 0.2, cz: 10.5, cy: 1.6, lx: 0, ly: 5.5, lz: -10, gold: 0, star: 0.3, spot: 0,
               music: 0, flare: 0, ribbon: 0, red: 0, ring: 0, out: 0, blue: 0, balc: 0.12, candle: 0, spread: 0, zoom: 1, plx: 0, ply: 5.5, plz: -10 };
    // `p: [x, y, z]` is the portrait target; otherwise phones look where desktops do.
    function k(u, o) {
      for (var n in o) st[n] = o[n];
      if (o.p) { st.plx = o.p[0]; st.ply = o.p[1]; st.plz = o.p[2]; }
      else if ('lx' in o || 'ly' in o || 'lz' in o) { st.plx = st.lx; st.ply = st.ly; st.plz = st.lz; }
      rows.push([u].concat(COLS.map(function (c) { return st[c]; })));
    }
    // Look up into the sky from camera (x, y, z): `turn` radians off the
    // fixed star's bearing (+ puts the star right of centre), `el` up.
    var STAR_AZ = Math.atan2(STAR_DIR.x, STAR_DIR.z);
    function sky(x, y, z, turn, el) {
      var az = STAR_AZ + turn;
      return { cx: x, cy: y, cz: z, lx: x + Math.sin(az) * Math.cos(el) * 20, ly: y + Math.sin(el) * 20, lz: z + Math.cos(az) * Math.cos(el) * 20 };
    }
    function and(a, b) { for (var n in b) a[n] = b[n]; return a; }
    k(0, {});
    k(0.7, {});
    // I   a summer's day: gold floods the house
    k(S(0) + 0.15, { cx: 2.4, cz: 9.2, cy: 1.7, lx: -2.6, ly: 5.2 });
    k(S(0) + 0.6, { gold: 1, cx: 3.4, cz: 8.4, lx: -3.0, ly: 5.4 });
    k(S(0) + 1.2, { cx: 4.2, cz: 7.6 });
    // II  the fixed star above the open roof
    k(S(1) + 0.15, and(sky(0.4, 1.7, 7.2, -0.3, 0.32), { gold: 0.2, star: 0.5 }));
    k(S(1) + 0.6, and(sky(-1.4, 1.7, 6.8, -0.32, 0.68), { gold: 0, star: 1 }));
    k(S(1) + 1.2, sky(-2.2, 1.7, 6.4, -0.34, 0.7));
    // III a rose on the boards
    k(S(2) + 0.15, { star: 0.6, cx: 1.6, cz: 0.6, cy: 2.6, lx: -0.4, ly: 1.8, lz: -6, spot: 0.25, p: [0.9, 2.8, -5.8] });
    k(S(2) + 0.6, { cx: 0.4, cz: -1.6, cy: 2.9, lx: -0.5, ly: 1.4, lz: -5.8, spot: 1, p: [1.0, 2.6, -5.6] });
    k(S(2) + 1.2, { cx: 0.1, cz: -2.0 });
    // IV  music: motes rise into the heavens
    k(S(3) + 0.15, { cx: 1.8, cz: 2.4, cy: 2.0, lx: -1.6, ly: 3.4, lz: -7, spot: 0.7, music: 0.4 });
    k(S(3) + 0.6, { cx: 3.0, cz: 5.6, cy: 1.9, lx: -2.0, ly: 5.0, lz: -8, music: 1, spot: 0.45 });
    k(S(3) + 1.25, { cx: 2.8, cz: 6.4 });
    // V   blind Cupid: a blindfold of mist
    k(S(4) + 0.15, { music: 0.2, cx: -1.5, cz: 5.6, cy: 1.8, lx: 3.5, ly: 4.0, lz: -8, mist: 0.4, spot: 0.3, wind: 0.5 });
    k(S(4) + 0.6, { music: 0, mist: 1, cx: -3.8, cz: 5.0 });
    k(S(4) + 1.2, { cx: -4.8, cz: 4.2 });
    // VI  the stars are fire
    k(S(5) + 0.15, and(sky(-4.6, 1.7, 3.6, -0.9, 0.35), { mist: 0.2, wind: 0.2 }));
    k(S(5) + 0.6, and(sky(-4.4, 1.7, 3.2, -1.0, 0.8), { mist: 0, flare: 1 }));
    k(S(5) + 1.2, sky(-4.0, 1.7, 2.8, -1.05, 0.84));
    // VII the course of true love: a ribbon winds across the stage
    k(S(6) + 0.2, { flare: 0, cx: -3.4, cz: 1.6, cy: 4.4, lx: 0.6, ly: 1.6, lz: -7.6, ribbon: 0, p: [0.2, 3.4, -7.6] });
    k(S(6) + 1.2, { cx: -2.8, cz: 0.9, cy: 4.9, lx: 0.9, ly: 1.4, ribbon: 1, p: [0.4, 3.4, -7.6] });
    // VIII Et tu, Brute? the house goes red
    k(S(7) + 0.2, { cx: -0.6, cz: 4.4, cy: 2.0, lx: 0, ly: 4.0, lz: -10, ribbon: 1.5 });
    k(S(7) + 0.3, { red: 1, ribbon: 2 });
    k(S(7) + 1.2, { red: 0.85, cx: 0.2, cz: 4.0 });
    // IX  a charmed life: a ring of light round the rose
    k(S(8) + 0.2, { red: 0, cx: -1.8, cz: 0.4, cy: 2.8, lx: 2.6, ly: 2.0, lz: -6.2, spot: 0.15, p: [1.0, 3.6, -5.6] });
    k(S(8) + 0.65, { ring: 1, cx: -2.4, cz: -0.6 });
    k(S(8) + 1.2, { cx: -2.6, cz: -1.2 });
    // X   stars, hide your fires
    k(S(9) + 0.2, and(sky(-0.8, 1.7, 4.0, 0.3, 0.35), { ring: 0, spot: 0, star: 0.9, dark: 0.15 }));
    k(S(9) + 0.45, sky(-0.6, 1.7, 4.2, 0.3, 0.74));
    k(S(9) + 1.2, and(sky(-0.2, 1.7, 4.6, 0.32, 0.72), { out: 1, dark: 0.6 }));
    // XI  loved not wisely: dim tragic blue
    k(S(10) + 0.2, { cx: -1.8, cz: 0.2, cy: 2.5, lx: 2.4, ly: 2.0, lz: -7.2, dark: 0.15, blue: 1, balc: 0, p: [1.0, 3.7, -5.6] });
    k(S(10) + 1.2, { cx: -1.6, cz: -0.8, cy: 2.4 });
    // XII the balcony, softly lit and empty
    k(S(11) + 0.2, { blue: 0.4, out: 0.5, dark: 0.35, cx: -0.6, cz: 0.4, cy: 3.2, lx: -2.6, ly: 6.0, lz: -12, balc: 0.4, wind: 0.35, zoom: 0.8, p: [0, 4.0, -12] });
    k(S(11) + 0.6, { blue: 0, out: 0.3, balc: 1, dark: 0.2, cx: -0.9, cz: -0.4, cy: 3.5, zoom: 0.68 });
    k(S(11) + 1.2, { cx: -1.1, cz: -0.8, cy: 3.6, zoom: 0.66 });
    // XIII how far that little candle throws its beams
    k(S(12) + 0.15, { balc: 0.15, dark: 0.92, out: 0.6, cx: 0.6, cz: -1.0, cy: 2.2, lx: -2.6, ly: 2.2, lz: -6, candle: 0, wind: 0.15, zoom: 1 });
    k(S(12) + 0.35, { balc: 0, candle: 1, spread: 0 });
    k(S(12) + 1.25, { spread: 1, cx: 2.6, cz: 5.5, cy: 2.6, lx: -2, ly: 4.6, lz: -9, out: 0.4 });
    // XIV a quiet ending, and up to the fixed star
    k(S(13) + 0.2, { spread: 0.85, cx: 0.8, cz: 7.0, cy: 1.8, lx: 0, ly: 4.8, lz: -10, dark: 0.6, out: 0.2, star: 0.5 });
    k(S(13) + 1.2, { spread: 0.6, cx: 0, cz: 7.6, dark: 0.45, out: 0, star: 0.7 });
    k(T.total - 0.3, and(sky(0, 1.8, 7.4, 0, 0.62), { star: 1, spread: 0.45 }));
    k(T.total, {});
    return rows;
  },
  sound: {
    src: '/audio/wind.mp3',
    label: 'Play the night wind, a lute, a dagger and a bell',
    volume: function (row) { return (0.06 + row[3] * 0.08) * (1 - row[17] * 0.4); },
    cues: [
      { stanza: 3, at: 0.35, play: lute([[147, 0], [220, 0.14], [294, 0.28], [349, 0.42], [440, 0.56], [392, 1.1], [349, 1.4], [294, 1.75, 1.2]], 0.5) },
      { stanza: 7, at: 0.3, play: sting },
      { stanza: 11, at: 0.4, play: lute([[196, 0], [294, 0.2], [392, 0.4], [440, 0.9], [392, 1.3]], 0.3) },
      { stanza: 13, at: 0.4, play: bell(196, 0.28, 2, 3.2) }
    ]
  }
});
