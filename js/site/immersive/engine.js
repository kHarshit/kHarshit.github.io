/*
 * Immersive poem reading: the shared engine.
 *
 * A poem with `immersive: <scene>` in its front matter loads this file plus
 * js/site/immersive/scenes/<scene>.js (see _layouts/poem.html). The scene
 * file registers itself with PoemImmersive.register(); this file does the
 * rest: a tall section with a sticky full-screen stage above the article,
 * where a camera dollies through the scene's layers while each stanza fades
 * in and lights up word by word. The plain article stays below as the text
 * version, so the scene is decoration (aria-hidden) and the page still works
 * without JS.
 *
 * Rendering is one <canvas>: every layer is a set of Path2D shapes authored
 * in a 2000x1200 box with the horizon at y=720, placed at a depth `z`. A
 * layer at distance d = z - camZ is drawn at scale z/d around the vanishing
 * point, so near layers grow and slide past while far ones barely move.
 * Snow is a particle field in the same camera space, so it rushes towards
 * the reader when they scroll.
 *
 * A scene supplies:
 *   build(ctx)  -> layers [{ z, items: [{ path, fill | stroke, width }], fog?, glow? }]
 *   keys        camera keyframes [unit, depth, darkness, snowfall, wind]
 *   align       where each stanza sits: 'left' | 'right' | 'center'
 *   sky         gradient stops, top to horizon
 *   stars, moon optional sky dressing
 *   sound       optional { src, label, cues: [{ stanza, at, play(audioCtx, out) }] }
 */
(function () {
  'use strict';

  var SCENES = {};

  // ── Small helpers ─────────────────────────────────────────────────────
  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function smooth(a, b, v) { var t = clamp((v - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function rng(seed) {
    return function () {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  // ── Shape builders (layer space: 2000x1200, horizon at y=720) ─────────
  function ridgePath(r, base, amp, rough) {
    var p1 = r() * 10, p2 = r() * 10, p3 = r() * 10, n = 80;
    var top = [], d = 'M-300,3000 L-300,' + base;
    for (var i = 0; i <= n; i++) {
      var x = -300 + i * (2600 / n), t = x / 2000;
      var h = Math.sin(t * 3.1 + p1) * 0.5 + Math.sin(t * 7.3 + p2) * 0.3 +
              Math.sin(t * 17 + p3) * 0.15 * rough + (r() - 0.5) * 0.1 * rough;
      var y = base - (h * 0.5 + 0.5) * amp;
      top.push(x.toFixed(0) + ',' + y.toFixed(0));
      d += ' L' + top[top.length - 1];
    }
    return { fill: new Path2D(d + ' L2300,3000 Z'), line: new Path2D('M' + top.join(' L')) };
  }

  // A pine as stacked tiers with concave sides, plus snow caps on each tier.
  function pine(x, y, h, w, tree, snow) {
    var tiers = 4, step = h * 0.88 / tiers, tw0 = w * 0.5, trunk = w * 0.035;
    tree.push('M' + (x - trunk) + ',' + (y + 4) + 'h' + (2 * trunk) + 'v' + (-h * 0.14) + 'h' + (-2 * trunk) + 'Z');
    for (var k = 0; k < tiers; k++) {
      var by = y - h * 0.06 - k * step;
      var tw = tw0 * (1 - k / (tiers + 0.7));
      var top = k === tiers - 1 ? y - h : by - step * 1.7;
      var ty = by - (by - top) * 0.35, droop = h * 0.025;
      tree.push('M' + (x - tw) + ',' + by +
        ' Q' + (x - tw * 0.38) + ',' + ty + ' ' + x + ',' + top +
        ' Q' + (x + tw * 0.38) + ',' + ty + ' ' + (x + tw) + ',' + by +
        ' Q' + x + ',' + (by - droop * 2) + ' ' + (x - tw) + ',' + by + 'Z');
      var cy = by - (by - top) * 0.62, cw = tw * 0.3, wave = h * 0.012;
      snow.push('M' + (x - cw) + ',' + cy +
        ' Q' + (x - cw * 0.3) + ',' + (cy - (cy - top) * 0.4) + ' ' + x + ',' + top +
        ' Q' + (x + cw * 0.3) + ',' + (cy - (cy - top) * 0.4) + ' ' + (x + cw) + ',' + cy +
        ' Q' + (x + cw * 0.5) + ',' + (cy + wave * 2) + ' ' + (x + cw * 0.15) + ',' + (cy - wave) +
        ' Q' + (x - cw * 0.3) + ',' + (cy + wave * 2.2) + ' ' + (x - cw) + ',' + cy + 'Z');
    }
  }

  function groundPath(r, base, wave) {
    var d = 'M-400,3000 L-400,' + base, n = 30, p = r() * 6;
    for (var i = 0; i <= n; i++) {
      var x = -400 + i * (2800 / n);
      d += ' L' + x.toFixed(0) + ',' + (base + Math.sin(x / 260 + p) * wave + Math.sin(x / 90 + p * 2) * wave * 0.3).toFixed(0);
    }
    return new Path2D(d + ' L2400,3000 Z');
  }

  // A band of trees on its own strip of snow, with a clearing in the middle
  // so the camera can travel through it.
  function treeRow(ctx, r, o) {
    var tree = [], snow = [], x = -350;
    while (x < 2350) {
      x += o.gap[0] + r() * (o.gap[1] - o.gap[0]);
      var off = Math.abs(x - (o.cx || 1000));
      if (off < o.clear * (0.85 + r() * 0.3)) continue;
      var h = o.h[0] + r() * (o.h[1] - o.h[0]);
      // Trees grow a little taller away from the clearing.
      h *= 1 + Math.min(off / 2000, 0.35);
      pine(x, o.base + (r() - 0.5) * o.jitter, h, h * (0.42 + r() * 0.14), tree, snow);
    }
    var items = [];
    if (o.ground) items.push({ path: groundPath(r, o.base + 2, o.wave || 4), fill: vgrad(ctx, o.base, o.base + 500, o.ground) });
    items.push({ path: new Path2D(tree.join('')), fill: o.color });
    items.push({ path: new Path2D(snow.join('')), fill: o.snow });
    return items;
  }

  function vgrad(ctx, y0, y1, stops) {
    var g = ctx.createLinearGradient(0, y0, 0, y1);
    stops.forEach(function (c, i) { g.addColorStop(i / (stops.length - 1), c); });
    return g;
  }

  window.PoemImmersive = {
    register: function (name, scene) { SCENES[name] = scene; },
    util: { clamp: clamp, smooth: smooth, lerp: lerp, rng: rng },
    shapes: { ridgePath: ridgePath, pine: pine, groundPath: groundPath, treeRow: treeRow, vgrad: vgrad }
  };

  // Scene files are deferred scripts after this one, so they have all
  // registered by DOMContentLoaded.
  document.addEventListener('DOMContentLoaded', boot);

  function boot() {
    var article = document.querySelector('.poetry[data-immersive]');
    var root = document.documentElement;
    if (!article) { root.classList.remove('pi-pending'); return; }

    var sceneName = article.getAttribute('data-immersive');
    var toggle = document.getElementById('poem-immersive-toggle');
    var PREF_KEY = 'poem-immersive';
    var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // ── Timeline, in "units" of one screen of scroll ──────────────────────
    var INTRO = 1.0;      // title card; the camera starts moving at 0.7
    var STANZA = 1.6;     // 0.3 fade in, 1.0 reading, 0.3 fade out
    var OUTRO = 1.2;

    // ── DOM helpers ─────────────────────────────────────────────────────
    function el(tag, cls, html) {
      var n = document.createElement(tag);
      if (cls) n.className = cls;
      if (html != null) n.innerHTML = html;
      return n;
    }
    function escapeHtml(s) {
      return s.replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; });
    }
    function words(text) {
      return text.split(/\s+/).filter(Boolean).map(function (w) {
        return '<span class="pi-word">' + escapeHtml(w) + '</span>';
      }).join(' ');
    }
    function getPref() { try { return localStorage.getItem(PREF_KEY); } catch (e) { return null; } }
    function setPref(v) { try { localStorage.setItem(PREF_KEY, v); } catch (e) {} }

    // ── Read the verse out of the rendered article ────────────────────────
    // Lines are list items; kramdown wraps the last line of each stanza in <p>
    // because a blank line follows it, which is what marks the stanza break.
    function readStanzas() {
      var stanzas = [], current = [];
      var items = article.querySelectorAll('.poem-body > ul > li');
      Array.prototype.forEach.call(items, function (li) {
        current.push(li.textContent.trim());
        if (li.querySelector('p')) { stanzas.push(current); current = []; }
      });
      if (current.length) stanzas.push(current);
      return stanzas;
    }

    var scene = SCENES[sceneName];
    if (!scene) { root.classList.remove('pi-pending'); return; }

    var stanzas = readStanzas();
    if (!stanzas.length) { root.classList.remove('pi-pending'); return; }

    var TOTAL = INTRO + stanzas.length * STANZA + OUTRO;
    function stanzaStart(i) { return INTRO + i * STANZA; }

    // Camera keyframes are authored for the scene's own stanza count; if the
    // poem has a different number, stretch them over the actual timeline.
    var keyEnd = scene.keys[scene.keys.length - 1][0];
    var keys = scene.keys.map(function (k) { return [k[0] * TOTAL / keyEnd].concat(k.slice(1)); });

    function sample(u) {
      var out = keys[keys.length - 1].slice(1);
      for (var i = 0; i < keys.length - 1; i++) {
        var a = keys[i], b = keys[i + 1];
        if (u <= b[0]) {
          var t = smooth(a[0], b[0], u);
          for (var j = 1; j < a.length; j++) out[j - 1] = lerp(a[j], b[j], t);
          return out;
        }
      }
      return out;
    }

    // ── State shared by build / teardown ──────────────────────────────────
    var section = null, raf = 0, running = false, io = null;
    var sound = { on: false, audio: null, ctx: null, master: null, lastStanzaU: 0 };

    function buildDom() {
      var title = (article.querySelector('.poem-title') || {}).textContent || '';
      var author = (article.querySelector('.poem-author') || {}).textContent || '';

      section = el('section', 'pi');
      section.style.setProperty('--pi-units', (TOTAL + 1).toFixed(2));
      var stage = el('div', 'pi-stage');
      var canvas = el('canvas', 'pi-canvas');
      canvas.setAttribute('aria-hidden', 'true');
      var vignette = el('div', 'pi-vignette');
      vignette.setAttribute('aria-hidden', 'true');

      var text = el('div', 'pi-text');
      text.setAttribute('aria-hidden', 'true');

      var intro = el('div', 'pi-intro',
        '<p class="pi-kicker">' + escapeHtml(author) + '</p>' +
        '<p class="pi-title">' + escapeHtml(title) + '</p>' +
        '<p class="pi-cue"><span>Scroll to enter</span><i></i></p>');
      text.appendChild(intro);

      var panels = stanzas.map(function (lines, i) {
        var p = el('div', 'pi-panel');
        p.setAttribute('data-align', scene.align[i % scene.align.length]);
        p.innerHTML = '<p class="pi-num">' + ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII'][i] + '</p>' +
          lines.map(function (l) { return '<p class="pi-line">' + words(l) + '</p>'; }).join('');
        text.appendChild(p);
        return { el: p, words: Array.prototype.slice.call(p.querySelectorAll('.pi-word')), last: [] };
      });

      var outro = el('div', 'pi-outro',
        '<p class="pi-sign">&mdash; ' + escapeHtml(author) + '</p>');
      text.appendChild(outro);

      var rail = el('div', 'pi-rail', '<i></i>');
      rail.setAttribute('aria-hidden', 'true');

      var controls = el('div', 'pi-controls');
      if (scene.sound) {
        var soundBtn = el('button', 'pi-btn', '<i class="fa-solid fa-volume-xmark"></i><span>Sound</span>');
        soundBtn.type = 'button';
        soundBtn.setAttribute('aria-pressed', 'false');
        soundBtn.setAttribute('aria-label', scene.sound.label || 'Play sound');
        soundBtn.addEventListener('click', function () { setSound(!sound.on, soundBtn); });
        controls.appendChild(soundBtn);
      }
      var textBtn = el('button', 'pi-btn', '<i class="fa-solid fa-align-left"></i><span>Text view</span>');
      textBtn.type = 'button';
      textBtn.setAttribute('aria-label', 'Leave the immersive view and read the plain text');
      controls.appendChild(textBtn);

      stage.appendChild(canvas);
      stage.appendChild(vignette);
      stage.appendChild(text);
      stage.appendChild(rail);
      stage.appendChild(controls);
      section.appendChild(stage);

      var container = article.closest('.container');
      container.parentNode.insertBefore(section, container);

      textBtn.addEventListener('click', function () { setPref('off'); exit(true); });

      return { stage: stage, canvas: canvas, text: text, intro: intro, outro: outro, panels: panels, rail: rail.firstChild };
    }

    // ── Sound: the scene's ambient loop, plus one-off cues it synthesises ─
    function setSound(on, btn) {
      sound.on = on;
      btn.setAttribute('aria-pressed', on ? 'true' : 'false');
      btn.querySelector('i').className = on ? 'fa-solid fa-volume-high' : 'fa-solid fa-volume-xmark';
      if (on) {
        if (!sound.audio) {
          sound.audio = new Audio(scene.sound.src);
          sound.audio.loop = true;
          sound.audio.volume = 0;
        }
        sound.audio.play().catch(function () {});
        var AC = window.AudioContext || window.webkitAudioContext;
        if (AC && !sound.ctx) {
          sound.ctx = new AC();
          sound.master = sound.ctx.createGain();
          sound.master.gain.value = 0.5;
          sound.master.connect(sound.ctx.destination);
        }
        if (sound.ctx && sound.ctx.state === 'suspended') sound.ctx.resume();
      } else if (sound.audio) {
        sound.audio.pause();
      }
    }

    // ── Renderer ──────────────────────────────────────────────────────────
    function start() {
      var dom = buildDom();
      var canvas = dom.canvas, ctx = canvas.getContext('2d');
      var layers = scene.build(ctx);
      var W = 0, H = 0, dpr = 1, k = 1, vpx = 0, vpy = 0, skyGrad = null;
      var F = 800;   // focal length shared by layers and snow

      // Stars in normalised screen space, above the horizon.
      var sr = rng(5), stars = [];
      for (var i = 0; i < (scene.stars || 0); i++) stars.push([sr(), sr() * 0.6, sr() * 0.8 + 0.2, sr() * 6.28, 0.5 + sr() * 2]);

      // Soft snowflake sprite.
      var flake = document.createElement('canvas');
      flake.width = flake.height = 32;
      var fx = flake.getContext('2d'), fg = fx.createRadialGradient(16, 16, 0, 16, 16, 16);
      fg.addColorStop(0, 'rgba(255,255,255,1)');
      fg.addColorStop(0.35, 'rgba(240,246,255,0.8)');
      fg.addColorStop(1, 'rgba(230,240,255,0)');
      fx.fillStyle = fg;
      fx.fillRect(0, 0, 32, 32);

      var N = window.innerWidth < 700 ? 260 : 520, DEPTH = 2400;
      var PX = new Float32Array(N), PY = new Float32Array(N), PD = new Float32Array(N),
          PS = new Float32Array(N), PF = new Float32Array(N), PH = new Float32Array(N),
          P0 = new Float32Array(N), PA = new Float32Array(N);
      function spanX(d) { return (W / 2 + 200) / (k * F) * d; }
      function spanY(d) { return H / (k * F) * d; }
      // P0 is the depth a flake was placed at and PA its age, so new flakes
      // fade in rather than pop.
      function spawn(i, d) {
        PD[i] = P0[i] = d;
        PA[i] = 0;
        PX[i] = (Math.random() * 2 - 1) * spanX(d);
        PY[i] = (Math.random() * 2 - 1) * spanY(d);
      }

      function resize() {
        dpr = Math.min(window.devicePixelRatio || 1, 1.75);
        W = dom.stage.clientWidth;
        H = dom.stage.clientHeight;
        canvas.width = Math.round(W * dpr);
        canvas.height = Math.round(H * dpr);
        // Cover the stage like `slice`, but never show less than ~1000 units of
        // width, so portrait phones still see both sides of the clearing.
        k = Math.min(Math.max(W / 2000, H / 1200), W / 1000);
        vpx = W / 2;
        vpy = H * 0.62;
        skyGrad = ctx.createLinearGradient(0, 0, 0, vpy);
        scene.sky.forEach(function (c, i) { skyGrad.addColorStop(i / (scene.sky.length - 1), c); });
      }
      resize();
      for (i = 0; i < N; i++) {
        spawn(i, 20 + Math.random() * DEPTH);
        PS[i] = 1 + Math.random() * 1.3;
        PF[i] = 16 + Math.random() * 16;
        PH[i] = Math.random() * 6.28;
      }

      var u = progressTarget(), cam = sample(u)[0], last = performance.now(), time = 0;
      var mx = 0, my = 0, tmx = 0, tmy = 0, finePointer = window.matchMedia('(pointer: fine)').matches;

      function onPointer(e) {
        if (e.pointerType !== 'mouse') return;
        tmx = e.clientX / window.innerWidth * 2 - 1;
        tmy = e.clientY / window.innerHeight * 2 - 1;
      }
      window.addEventListener('pointermove', onPointer, { passive: true });
      window.addEventListener('resize', resize);

      // While the stage fills the viewport, tuck the site chrome away.
      var immersed = false;
      function progressTarget() {
        var rect = section.getBoundingClientRect();
        var range = rect.height - dom.stage.clientHeight;
        var inside = rect.top <= 1 && rect.bottom >= dom.stage.clientHeight - 1;
        if (inside !== immersed) { immersed = inside; root.classList.toggle('pi-immersed', inside); }
        return clamp(-rect.top / range, 0, 1) * TOTAL;
      }

      function drawLayer(L, d, ox, oy) {
        var rel = L.z / d, s = k * rel;
        var alpha = clamp((d - 30) / Math.min(200, (L.z - 30) * 0.7), 0, 1);
        if (alpha <= 0) return;
        ctx.globalAlpha = alpha;
        ctx.setTransform(s * dpr, 0, 0, s * dpr, (vpx + ox - 1000 * s) * dpr, (vpy + oy - 720 * s) * dpr);
        L.items.forEach(function (it) {
          if (it.fill) { ctx.fillStyle = it.fill; ctx.fill(it.path); }
          if (it.stroke) { ctx.strokeStyle = it.stroke; ctx.lineWidth = it.width; ctx.stroke(it.path); }
        });
        if (L.fog) {
          var f = L.fog, cx = 1000 + Math.sin(time / 30) * f.drift * 10 + time * f.drift;
          cx = ((cx % 3000) + 3000) % 3000 - 500;
          [cx - 1500, cx, cx + 1500].forEach(function (x) {
            ctx.save();
            ctx.translate(x, f.cy);
            ctx.scale(f.rx, f.ry);
            var g = ctx.createRadialGradient(0, 0, 0, 0, 0, 1);
            g.addColorStop(0, 'rgba(190,205,235,' + f.a + ')');
            g.addColorStop(1, 'rgba(190,205,235,0)');
            ctx.fillStyle = g;
            ctx.beginPath();
            ctx.arc(0, 0, 1, 0, 6.2832);
            ctx.fill();
            ctx.restore();
          });
        }
        if (L.glow) {
          ctx.globalCompositeOperation = 'lighter';
          L.glow.forEach(function (w, j) {
            var flick = 0.85 + 0.15 * Math.sin(time * 3 + j * 1.7) * Math.sin(time * 1.3 + j);
            var g = ctx.createRadialGradient(w[0], w[1], 0, w[0], w[1], w[2] * 6);
            g.addColorStop(0, 'rgba(255,205,130,' + 0.55 * flick + ')');
            g.addColorStop(1, 'rgba(255,170,90,0)');
            ctx.fillStyle = g;
            ctx.fillRect(w[0] - w[2] * 6, w[1] - w[2] * 6, w[2] * 12, w[2] * 12);
            ctx.fillStyle = 'rgba(255,214,150,' + flick + ')';
            ctx.fillRect(w[0] - w[2] / 2, w[1] - w[2] / 2, w[2], w[2] * 0.9);
          });
          ctx.globalCompositeOperation = 'source-over';
        }
      }

      function drawMoon(dark) {
        var mxp = scene.moon.x * W - mx * 10, myp = scene.moon.y * H - my * 6, mr = Math.min(W, H) * 0.045;
        var moonA = 1 - dark * 0.55;
        ctx.globalAlpha = moonA;
        var mg = ctx.createRadialGradient(mxp, myp, mr * 0.8, mxp, myp, mr * 9);
        mg.addColorStop(0, 'rgba(200,215,255,0.35)');
        mg.addColorStop(1, 'rgba(120,140,200,0)');
        ctx.fillStyle = mg;
        ctx.fillRect(0, 0, W, H);
        ctx.fillStyle = '#eef3ff';
        ctx.beginPath();
        ctx.arc(mxp, myp, mr, 0, 6.2832);
        ctx.fill();
        ctx.fillStyle = 'rgba(160,175,210,0.25)';
        ctx.beginPath();
        ctx.arc(mxp - mr * 0.3, myp - mr * 0.2, mr * 0.28, 0, 6.2832);
        ctx.arc(mxp + mr * 0.35, myp + mr * 0.3, mr * 0.18, 0, 6.2832);
        ctx.fill();
      }

      function frame(now) {
        raf = running ? requestAnimationFrame(frame) : 0;
        var dt = Math.min((now - last) / 1000, 0.05);
        last = now;
        time += dt;

        // Ease towards the scroll position, but jump outright on big leaps
        // (scroll restore, the text-view link) instead of replaying the poem.
        var ut = progressTarget(), leap = Math.abs(ut - u) > 1.2;
        u = leap ? ut : u + (ut - u) * (1 - Math.exp(-dt * 5));
        if (Math.abs(ut - u) < 0.0005) u = ut;
        var st = sample(u), camZ = st[0], dark = st[1], snow = st[2], wind = st[3];
        var dCam = leap ? 0 : camZ - cam;
        cam = camZ;
        if (leap) for (i = 0; i < N; i++) spawn(i, 20 + Math.random() * DEPTH);

        if (finePointer && !reduceMotion) {
          mx += (tmx - mx) * (1 - Math.exp(-dt * 3));
          my += (tmy - my) * (1 - Math.exp(-dt * 3));
        } else {
          mx = Math.sin(time * 0.13) * 0.25;
          my = Math.sin(time * 0.09) * 0.1;
        }
        var P = 22000 * W / 1440;

        // Sky, stars, moon (screen space).
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.globalAlpha = 1;
        ctx.fillStyle = skyGrad;
        ctx.fillRect(0, 0, W, H);
        ctx.fillStyle = scene.sky[scene.sky.length - 1];
        ctx.fillRect(0, vpy, W, H - vpy);
        ctx.fillStyle = '#dfe7fa';
        stars.forEach(function (s) {
          var a = s[2] * (0.6 + 0.4 * Math.sin(time * s[4] + s[3]));
          ctx.globalAlpha = a * (1 - s[1] / 0.75);
          ctx.fillRect(s[0] * W - mx * 4, s[1] * H - my * 3, 1.4, 1.4);
        });
        if (scene.moon) drawMoon(dark);

        // Layers, far to near.
        for (var li = 0; li < layers.length; li++) {
          var L = layers[li], d = L.z - camZ;
          if (d <= 30) continue;
          drawLayer(L, d, -mx * P / d, -my * P * 0.35 / d);
        }

        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.globalAlpha = 1;
        if (dark > 0) {
          ctx.fillStyle = 'rgba(2,3,9,' + dark + ')';
          ctx.fillRect(0, 0, W, H);
        }

        // Snow in camera space.
        var n = Math.floor(N * snow), sScale = k * F;
        var windX = wind * 140, fall = reduceMotion ? 0.5 : 1;
        for (i = 0; i < N; i++) {
          PD[i] -= dCam;
          PA[i] += dt;
          if (PD[i] < 12) spawn(i, PD[i] + DEPTH);
          // Backing up, flakes recede towards the vanishing point and leave the
          // edges bare; recycle them anywhere in the volume once they have
          // drifted well past where they started.
          else if (PD[i] > DEPTH + 20 || (dCam < 0 && PD[i] > P0[i] * 1.6 + 150)) spawn(i, 20 + Math.random() * DEPTH);
          PY[i] += PF[i] * dt * fall;
          PX[i] += (windX + Math.sin(time * 0.8 + PH[i]) * 9) * dt * fall;
          if (i >= n) continue;
          var dd = PD[i], s = sScale / dd;
          var sx = vpx - mx * P / dd + PX[i] * s, sy = vpy - my * P * 0.35 / dd + PY[i] * s;
          var rr = Math.min(PS[i] * s, 38);
          if (sy > H + rr + 10) { PY[i] -= (H + rr * 2 + 20) / s; continue; }
          if (sx < -rr - 60) { PX[i] += (W + rr * 2 + 120) / s; continue; }
          if (sx > W + rr + 60) { PX[i] -= (W + rr * 2 + 120) / s; continue; }
          if (rr < 0.5) rr = 0.5;
          ctx.globalAlpha = Math.min(1, dd / 90, PA[i] / 0.6) * clamp(1.35 - dd / DEPTH, 0, 1) * (1 - dark * 0.5) * 0.9;
          ctx.drawImage(flake, sx - rr * 1.6, sy - rr * 1.6, rr * 3.2, rr * 3.2);
        }
        ctx.globalAlpha = 1;

        updateText(u, dom);
        updateSound(u, wind, dark);
      }

      function updateText(u, dom) {
        dom.text.style.transform = 'translate3d(' + (-mx * 10).toFixed(1) + 'px,' + (-my * 6).toFixed(1) + 'px,0)';
        var oi = 1 - smooth(0.55, 0.95, u);
        dom.intro.style.opacity = oi.toFixed(3);
        dom.intro.style.transform = 'translate(-50%, calc(-50% - ' + ((1 - oi) * 40).toFixed(1) + 'px))';
        dom.intro.style.visibility = oi > 0 ? 'visible' : 'hidden';

        dom.panels.forEach(function (p, i) {
          var S = stanzaStart(i);
          var fin = smooth(S, S + 0.3, u), fout = smooth(S + 1.3, S + 1.6, u);
          var o = fin * (1 - fout);
          var y = (1 - fin) * 50 - fout * 50;
          p.el.style.opacity = o.toFixed(3);
          p.el.style.visibility = o > 0 ? 'visible' : 'hidden';
          p.el.style.setProperty('--y', y.toFixed(1) + 'px');
          p.el.style.filter = o < 0.99 ? 'blur(' + ((1 - o) * 8).toFixed(1) + 'px)' : 'none';
          if (o <= 0) return;
          var prog = clamp((u - (S + 0.15)) / 1.0, 0, 1), nw = p.words.length;
          for (var w = 0; w < nw; w++) {
            var lit = clamp((prog * (nw + 3) - w) / 3, 0, 1);
            var v = Math.round((0.14 + 0.86 * lit) * 100) / 100;
            if (p.last[w] !== v) { p.last[w] = v; p.words[w].style.opacity = v; }
          }
        });

        var oo = smooth(TOTAL - OUTRO + 0.2, TOTAL - OUTRO + 0.6, u);
        dom.outro.style.opacity = oo.toFixed(3);
        dom.outro.style.visibility = oo > 0 ? 'visible' : 'hidden';
        dom.rail.style.transform = 'scaleY(' + (u / TOTAL).toFixed(4) + ')';
      }

      function updateSound(u, wind, dark) {
        if (!sound.on) { sound.lastStanzaU = u; return; }
        if (sound.audio) {
          var target = (0.12 + 0.5 * wind) * (1 - dark * 0.6);
          sound.audio.volume = clamp(lerp(sound.audio.volume, target, 0.05), 0, 1);
        }
        // Cues fire when reading forwards past their point, not on the way back.
        (scene.sound.cues || []).forEach(function (c) {
          var at = stanzaStart(c.stanza) + c.at;
          if (sound.ctx && sound.lastStanzaU < at && u >= at) c.play(sound.ctx, sound.master);
        });
        sound.lastStanzaU = u;
      }

      function run(on) {
        if (on === running) return;
        running = on;
        if (on) { last = performance.now(); raf = requestAnimationFrame(frame); }
        else if (raf) { cancelAnimationFrame(raf); raf = 0; }
        if (sound.on && sound.audio) { if (on) sound.audio.play().catch(function () {}); else sound.audio.pause(); }
        if (!on) { immersed = false; root.classList.remove('pi-immersed'); }
      }

      io = new IntersectionObserver(function (entries) { run(entries[0].isIntersecting); });
      io.observe(section);

      section._cleanup = function () {
        root.classList.remove('pi-immersed');
        window.removeEventListener('pointermove', onPointer);
        window.removeEventListener('resize', resize);
      };
    }

    function exit(scrollToText) {
      if (!section) return;
      if (io) io.disconnect();
      running = false;
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
      if (sound.audio) sound.audio.pause();
      sound.on = false;
      section._cleanup();
      section.remove();
      section = null;
      article.classList.remove('is-immersive');
      updateToggle();
      if (scrollToText) article.scrollIntoView({ behavior: 'auto', block: 'start' });
    }

    function enter(scrollToScene) {
      if (section) return;
      article.classList.add('is-immersive');
      start();
      updateToggle();
      if (scrollToScene) window.scrollTo({ top: section.getBoundingClientRect().top + window.scrollY, behavior: 'auto' });
    }

    function updateToggle() {
      if (!toggle) return;
      toggle.hidden = false;
      toggle.querySelector('span').textContent = section ? 'Leave immersive view' : 'Immersive view';
    }

    if (toggle) {
      toggle.addEventListener('click', function () {
        if (section) { setPref('off'); exit(true); }
        else { setPref('on'); enter(true); }
      });
    }

    // Auto-start unless the reader turned it off, or asks for reduced motion
    // and hasn't opted in. The inline script in _layouts/poem.html applies the
    // same rule before first paint to hold space for the stage.
    var pref = getPref();
    if (pref === 'on' || (pref !== 'off' && !reduceMotion)) enter(false);
    else updateToggle();
    root.classList.remove('pi-pending');
  }
})();
