/*
 * Immersive poem reading: the shared engine.
 *
 * A poem with `immersive: <scene>` in its front matter loads this file,
 * which loads js/site/immersive/scenes/<scene>.js as a module. The scene
 * registers itself with PoemImmersive.register() and brings its own
 * renderer (three.js); this file owns everything else: a tall section with
 * a sticky full-screen stage above the article, the scroll timeline, each
 * stanza fading in and lighting up word by word, the controls and sound.
 * The plain article stays below as the text version, so the scene is
 * decoration (aria-hidden) and the page still works without JS or WebGL.
 *
 * A scene supplies:
 *   renderer(canvas, scene, env) -> { resize(w, h, dpr), frame(f), destroy? }
 *               frame(f) gets the sampled timeline: f.cam, f.dark, f.snow,
 *               f.wind, f.row (the whole keyframe row, for extra columns),
 *               f.mx/f.my pointer offsets in -1..1, f.dt and f.time
 *   keys        keyframes [unit, camera, darkness, snowfall, wind, ...];
 *               "camera" means whatever the renderer wants (e.g. path progress)
 *   align       where each stanza sits: 'left' | 'right' | 'center'
 *   sound       optional { src, label, volume(row)?, cues: [{ stanza, at, play(audioCtx, out) }] }
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

  // ── Synthesised sound cues scenes can share ───────────────────────────
  // A short shake of sleigh bells: bright inharmonic partials, fast decay.
  function sleighBells(ac, out) {
    var now = ac.currentTime;
    for (var i = 0; i < 9; i++) {
      var t = now + i * 0.065 + Math.random() * 0.035;
      var f = 2300 + Math.random() * 1100;
      [1, 2.76, 5.4].forEach(function (m, j) {
        var o = ac.createOscillator(), g = ac.createGain();
        o.type = 'sine';
        o.frequency.value = f * m;
        g.gain.setValueAtTime(0.0001, t);
        g.gain.exponentialRampToValueAtTime(0.07 / (j + 1), t + 0.004);
        g.gain.exponentialRampToValueAtTime(0.0001, t + 0.7 / (j + 1));
        o.connect(g);
        g.connect(out);
        o.start(t);
        o.stop(t + 0.75);
      });
    }
  }

  window.PoemImmersive = {
    register: function (name, scene) {
      SCENES[name] = scene;
      if (name === wanted) boot();
    },
    util: { clamp: clamp, smooth: smooth, lerp: lerp, rng: rng },
    sounds: { sleighBells: sleighBells }
  };

  // ── Loader ────────────────────────────────────────────────────────────
  // The scene file is a module (it imports three.js) and starts the engine
  // when it registers. If it fails to load, the page stays plain text.
  var article = document.querySelector('.poetry[data-immersive]');
  var root = document.documentElement;
  var wanted = null, booted = false;
  if (article) {
    wanted = article.getAttribute('data-immersive');
    var tag = document.createElement('script');
    tag.type = 'module';
    tag.src = '/js/site/immersive/scenes/' + wanted + '.js';
    tag.onerror = function () { root.classList.remove('pi-pending'); };
    document.head.appendChild(tag);
  } else {
    root.classList.remove('pi-pending');
  }

  function boot() {
    if (booted) return;
    booted = true;

    var sceneName = wanted;
    var toggle = document.getElementById('poem-immersive-toggle');
    var PREF_KEY = 'poem-immersive';
    var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // ── Timeline, in "units" of one screen of scroll ──────────────────────
    // Each panel (a stanza, or a chunk of prose) fades in over 0.3, reads
    // for `read` units (1 for a stanza, longer for long prose) and fades out
    // over 0.3.
    var INTRO = 1.0;      // title card; the camera starts moving at 0.7
    var FADE = 0.3;
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
    // A scene's `emphasis` regex picks words to colour with its accent
    // (V's speech lights every v-word red).
    function words(text) {
      return text.split(/\s+/).filter(Boolean).map(function (w) {
        var em = scene.emphasis && scene.emphasis.test(w) ? ' pi-em' : '';
        return '<span class="pi-word' + em + '">' + escapeHtml(w) + '</span>';
      }).join(' ');
    }
    function getPref() { try { return localStorage.getItem(PREF_KEY); } catch (e) { return null; } }
    function setPref(v) { try { localStorage.setItem(PREF_KEY, v); } catch (e) {} }

    // ── Read the text out of the rendered article ─────────────────────────
    // Verse: lines are list items; kramdown wraps the last line of each
    // stanza in <p> because a blank line follows it, which marks the break.
    // Prose (no list): each paragraph is split at sentence ends into chunks
    // of up to ~40 words, and longer chunks get longer to read. A paragraph
    // that is just an italic [stage direction] becomes a short caption.
    // Reading stops at the first <hr> (footnotes and extras follow it), and
    // skips explanation cards, since poem.js may have moved the text into
    // its explanations grid by now.
    function readPanels() {
      var panels = [], current = [], body = article.querySelector('.poem-body'), hr = body.querySelector('hr');
      function inCard(n) { return !!n.closest('.stanza-explain-card'); }
      function afterRule(n) { return hr && !(hr.compareDocumentPosition(n) & Node.DOCUMENT_POSITION_PRECEDING); }
      Array.prototype.forEach.call(body.querySelectorAll('ul > li'), function (li) {
        if (inCard(li) || afterRule(li)) return;
        current.push(li.textContent.trim());
        if (li.querySelector('p')) { panels.push({ lines: current, read: 1 }); current = []; }
      });
      if (current.length) panels.push({ lines: current, read: 1 });
      if (panels.length) return scene.maxLines ? splitLong(panels, scene.maxLines) : panels;

      var nodes = Array.prototype.slice.call(body.querySelectorAll('p, hr'));
      for (var k = 0; k < nodes.length && nodes[k].tagName !== 'HR'; k++) {
        var para = nodes[k], text = para.textContent.trim();
        if (inCard(para) || para.querySelector('img') || !text) continue;
        if (/^\[.*\]$/.test(text) && para.children.length === 1 && para.firstElementChild.tagName === 'EM') {
          panels.push({ lines: [text.slice(1, -1)], direction: true, read: 0.5 });
          continue;
        }
        readProse(text);
      }
      return panels;

      // A scene's `maxLines` splits taller stanzas into near-equal parts,
      // breaking after a line that ends in punctuation where it can. Only
      // the first part keeps the stanza's numeral.
      function splitLong(list, max) {
        var out = [];
        list.forEach(function (p) {
          var lines = p.lines, n = Math.ceil(lines.length / max);
          if (n < 2) { out.push(p); return; }
          var start = 0;
          for (var k = 1; k <= n; k++) {
            var end = lines.length;
            if (k < n) {
              var ideal = Math.round(k * lines.length / n), best = ideal;
              [0, -1, 1, -2, 2].some(function (d) {
                var e = ideal + d;
                if (e > start && e - start <= max && /[.!?;:)\u2014\u2013-]\s*$/.test(lines[e - 1])) { best = e; return true; }
              });
              end = best;
            }
            out.push({ lines: lines.slice(start, end), read: 1, cont: start > 0 });
            start = end;
          }
        });
        return out;
      }

      function readProse(text) {
        var sentences = text.match(/[^.!?]+[.!?]+["'\u2019\u201d)]*\s*|[^.!?]+$/g) || [];
        var chunk = '', n = 0;
        function flush() {
          if (!n) return;
          // A line of one to three words ("Voilà!") is set large, as a title.
          panels.push({ lines: [chunk.trim()], prose: n > 3, lead: n <= 3, read: clamp(n / 30, 1, 2.6) });
          chunk = '';
          n = 0;
        }
        sentences.forEach(function (sentence) {
          var w = sentence.split(/\s+/).filter(Boolean).length;
          if (n && n + w > 40) flush();
          chunk += sentence;
          n += w;
        });
        flush();
      }
    }

    var scene = SCENES[sceneName];
    if (!scene) { root.classList.remove('pi-pending'); return; }

    var stanzas = readPanels();
    if (!stanzas.length) { root.classList.remove('pi-pending'); return; }

    var starts = [], t0 = INTRO;
    stanzas.forEach(function (p) { starts.push(t0); t0 += FADE * 2 + p.read; });
    var TOTAL = t0 + OUTRO;
    function stanzaStart(i) { return starts[i]; }
    function stanzaEnd(i) { return starts[i] + FADE * 2 + stanzas[i].read; }

    // Keyframes are either a function of the timeline (for scenes whose
    // panel count depends on the text, like prose) or a fixed list authored
    // for a fixed stanza count, stretched if the poem differs.
    var keys;
    if (typeof scene.keys === 'function') {
      keys = scene.keys({ count: stanzas.length, total: TOTAL, intro: INTRO, start: stanzaStart, end: stanzaEnd });
    } else {
      var keyEnd = scene.keys[scene.keys.length - 1][0];
      keys = scene.keys.map(function (k) { return [k[0] * TOTAL / keyEnd].concat(k.slice(1)); });
    }

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
      // Daylight scenes can ask for a heavier shadow behind the text.
      if (scene.scrim != null) section.style.setProperty('--pi-scrim', scene.scrim);
      if (scene.accent) section.style.setProperty('--pi-accent', scene.accent);
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

      var stanzaNo = 0;
      var panels = stanzas.map(function (st, i) {
        var p = el('div', 'pi-panel' + (st.prose ? ' pi-prose' : '') + (st.lead ? ' pi-lead' : '') + (st.direction ? ' pi-direction' : ''));
        p.setAttribute('data-align', scene.align[i % scene.align.length]);
        if (!st.prose && !st.lead && !st.direction && !st.cont) stanzaNo++;
        p.innerHTML = (st.prose || st.lead || st.direction ? '' :
                       '<p class="pi-num">' + (st.cont ? '&middot;' : ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X'][stanzaNo - 1]) + '</p>') +
          st.lines.map(function (l) { return '<p class="pi-line">' + words(l) + '</p>'; }).join('');
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

    // ── Frame loop: timeline + pointer -> renderer, text and sound ───────
    function start() {
      var dom = buildDom(), view;
      try {
        view = scene.renderer(dom.canvas, scene, { reduceMotion: reduceMotion });
      } catch (e) {
        // No WebGL (old device, or disabled): drop the stage, keep the text.
        section.remove();
        section = null;
        return false;
      }

      function resize() {
        view.resize(dom.stage.clientWidth, dom.stage.clientHeight, Math.min(window.devicePixelRatio || 1, 1.75));
      }
      resize();

      var u = progressTarget(), last = performance.now(), time = 0;
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
        var st = sample(u);

        if (finePointer && !reduceMotion) {
          mx += (tmx - mx) * (1 - Math.exp(-dt * 3));
          my += (tmy - my) * (1 - Math.exp(-dt * 3));
        } else {
          mx = Math.sin(time * 0.13) * 0.25;
          my = Math.sin(time * 0.09) * 0.1;
        }

        view.frame({ u: u, cam: st[0], dark: st[1], snow: st[2], wind: st[3], row: st,
                     mx: mx, my: my, dt: dt, time: time });
        updateText(u, dom);
        updateSound(u, st);
      }

      function updateText(u, dom) {
        dom.text.style.transform = 'translate3d(' + (-mx * 10).toFixed(1) + 'px,' + (-my * 6).toFixed(1) + 'px,0)';
        var oi = 1 - smooth(0.55, 0.95, u);
        dom.intro.style.opacity = oi.toFixed(3);
        dom.intro.style.transform = 'translate(-50%, calc(-50% - ' + ((1 - oi) * 40).toFixed(1) + 'px))';
        dom.intro.style.visibility = oi > 0 ? 'visible' : 'hidden';

        dom.panels.forEach(function (p, i) {
          var S = stanzaStart(i), E = stanzaEnd(i), read = stanzas[i].read;
          var fin = smooth(S, S + FADE, u), fout = smooth(E - FADE, E, u);
          var o = fin * (1 - fout);
          var y = (1 - fin) * 50 - fout * 50;
          p.el.style.opacity = o.toFixed(3);
          p.el.style.visibility = o > 0 ? 'visible' : 'hidden';
          p.el.style.setProperty('--y', y.toFixed(1) + 'px');
          p.el.style.filter = o < 0.99 ? 'blur(' + ((1 - o) * 8).toFixed(1) + 'px)' : 'none';
          if (o <= 0) return;
          var prog = clamp((u - (S + 0.15)) / read, 0, 1), nw = p.words.length;
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

      // The loop follows the wind and fades in the dark, unless the scene
      // gives its own volume(row) (e.g. a sea that roars in a storm).
      function updateSound(u, st) {
        if (!sound.on) { sound.lastStanzaU = u; return; }
        if (sound.audio) {
          var target = scene.sound.volume ? scene.sound.volume(st) : (0.12 + 0.5 * st[3]) * (1 - st[1] * 0.6);
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
        if (view.destroy) view.destroy();
        window.removeEventListener('pointermove', onPointer);
        window.removeEventListener('resize', resize);
      };
      return true;
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
      updateToggle();
      if (scrollToText) article.scrollIntoView({ behavior: 'auto', block: 'start' });
    }

    function enter(scrollToScene) {
      if (section) return;
      if (!start()) {
        if (toggle) toggle.remove();
        toggle = null;
        return;
      }
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
