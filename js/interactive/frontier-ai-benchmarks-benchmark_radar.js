(function () {
  'use strict';

  var MODELS = [
    { name: 'Claude Fable 5',   color: '#f59e0b' },
    { name: 'GPT-5.5',          color: '#10b981' },
    { name: 'Claude Opus 4.8',  color: '#d97706' },
    { name: 'Gemini 3.1 Pro',   color: '#2563eb' },
    { name: 'Qwen 3.7 Max',     color: '#0891b2' },
    { name: 'DeepSeek-V4',      color: '#7c3aed' },
    { name: 'Grok 4',           color: '#dc2626' }
  ];

  var AXES = [
    { label: 'GPQA Diamond', key: 'gpqa' },
    { label: 'HLE',           key: 'hle' },
    { label: 'SWE-bench',     key: 'swe' },
    { label: 'AIME 2025',     key: 'aime' },
    { label: 'MMMU',          key: 'mmmu' },
    { label: 'Arena Elo',     key: 'arena' }
  ];

  var DATA = {
    'Claude Fable 5':  { gpqa: 94.1, hle: 59.0, swe: 80,   aime: 96, mmmu: 84, arena: 100  },
    'GPT-5.5':         { gpqa: 93.6, hle: 41.4, swe: 58.6, aime: 97, mmmu: 82, arena: 99.7 },
    'Claude Opus 4.8': { gpqa: 93.6, hle: 49.8, swe: 69,   aime: 94, mmmu: 83, arena: 99.7 },
    'Gemini 3.1 Pro':  { gpqa: 94.3, hle: 44.7, swe: 76,   aime: 94, mmmu: 86, arena: 99.7 },
    'Qwen 3.7 Max':    { gpqa: 92.4, hle: 36,   swe: 63,   aime: 93, mmmu: 80, arena: 98.4 },
    'DeepSeek-V4':     { gpqa: 88,   hle: 28,   swe: 65,   aime: 90, mmmu: 78, arena: 97.2 },
    'Grok 4':          { gpqa: 89,   hle: 32,   swe: 70,   aime: 90, mmmu: 79, arena: 99.1 }
  };

  var PADDING = 40;
  var RADIUS, CX, CY;
  var visibleModels = MODELS.map(function () { return true; });

  function getPixel(angle, frac) {
    return {
      x: CX + RADIUS * frac * Math.cos(angle - Math.PI / 2),
      y: CY + RADIUS * frac * Math.sin(angle - Math.PI / 2)
    };
  }

  function draw() {
    var canvas = document.getElementById('br-canvas');
    if (!canvas) return;
    var rect = canvas.getBoundingClientRect();
    var dpr = window.devicePixelRatio || 1;
    var w = Math.min(canvas.parentElement.clientWidth || 600, 600);
    var h = Math.round(w * 0.8);
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    var ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    CX = w / 2;
    CY = h / 2;
    RADIUS = Math.min(CX, CY) - PADDING;

    ctx.clearRect(0, 0, w, h);

    var isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    var gridColor = isDark ? '#374151' : '#d1d5db';
    var labelColor = isDark ? '#9ca3af' : '#6b7280';
    var textColor = isDark ? '#e5e7eb' : '#1f2937';

    // Draw grid circles
    for (var r = 0.2; r <= 1.0; r += 0.2) {
      ctx.beginPath();
      for (var a = 0; a <= 2 * Math.PI + 0.01; a += 0.02) {
        var p = getPixel(a, r);
        if (a === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }
      ctx.closePath();
      ctx.strokeStyle = gridColor;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    // Draw axes
    AXES.forEach(function (axis, i) {
      var angle = (2 * Math.PI * i) / AXES.length;
      var outer = getPixel(angle, 1);
      ctx.beginPath();
      ctx.moveTo(CX, CY);
      ctx.lineTo(outer.x, outer.y);
      ctx.strokeStyle = gridColor;
      ctx.lineWidth = 0.5;
      ctx.stroke();

      var labelPos = getPixel(angle, 1.15);
      ctx.fillStyle = labelColor;
      ctx.font = '13px system-ui, -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(axis.label, labelPos.x, labelPos.y);
    });

    // Draw data
    MODELS.forEach(function (model, mi) {
      if (!visibleModels[mi]) return;
      var data = DATA[model.name];
      ctx.beginPath();
      var pts = [];
      AXES.forEach(function (axis, ai) {
        var angle = (2 * Math.PI * ai) / AXES.length;
        var frac = data[axis.key] / 100;
        var p = getPixel(angle, frac);
        pts.push(p);
        if (ai === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      });
      ctx.closePath();
      ctx.fillStyle = model.color + '20';
      ctx.fill();
      ctx.strokeStyle = model.color;
      ctx.lineWidth = 2.5;
      ctx.stroke();

      // Draw points
      pts.forEach(function (p) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = model.color;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });
    });

    // Center label
    ctx.fillStyle = textColor;
    ctx.font = '12px system-ui, -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Score %', CX, CY);
  }

  function buildLegend() {
    var container = document.getElementById('br-legend');
    if (!container) return;
    container.innerHTML = '';
    MODELS.forEach(function (model, i) {
      var label = document.createElement('label');
      label.className = 'br-legend-item';
      var cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = visibleModels[i];
      cb.addEventListener('change', function () {
        visibleModels[i] = cb.checked;
        draw();
      });
      var swatch = document.createElement('span');
      swatch.className = 'br-legend-swatch';
      swatch.style.background = model.color;
      var name = document.createElement('span');
      name.textContent = model.name;
      label.appendChild(cb);
      label.appendChild(swatch);
      label.appendChild(name);
      container.appendChild(label);
    });
  }

  function init() {
    var el = document.getElementById('br-radar');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      buildLegend();
      draw();
      window.addEventListener('resize', draw);
      var observer = new MutationObserver(function () { draw(); });
      observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
