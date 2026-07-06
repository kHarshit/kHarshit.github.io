(function () {
  'use strict';

  function SpectrumExplorer(root) {
    this.root = root;
    this.slider = root.querySelector('.spec-slider');
    this.wlDisplay = root.querySelector('.spec-wl-val');
    this.swatch = root.querySelector('.spec-swatch');
    this.wlLabel = root.querySelector('.spec-wavelength-label');
    this.sVal = root.querySelector('.spec-s-val');
    this.mVal = root.querySelector('.spec-m-val');
    this.lVal = root.querySelector('.spec-l-val');
    this.sFill = root.querySelector('.spec-s-fill');
    this.mFill = root.querySelector('.spec-m-fill');
    this.lFill = root.querySelector('.spec-l-fill');
    this.domVal = root.querySelector('.spec-dom-val');
    this.canvas = root.querySelector('.spec-chart');
    this.ctx = this.canvas.getContext('2d');

    this.coneData = this.generateConeData();
    this.drawChart();

    var self = this;
    this.slider.addEventListener('input', function () { self.update(); });
    this.update();
  }

  SpectrumExplorer.prototype.generateConeData = function () {
    // Approximate cone sensitivity curves based on Stockman & Sharpe (2000) LMS fundamentals
    var data = [];
    for (var wl = 380; wl <= 780; wl++) {
      // S-cone: peak ~440nm, narrow
      var s = Math.exp(-0.5 * Math.pow((wl - 440) / 30, 2));
      // M-cone: peak ~540nm, medium
      var m = Math.exp(-0.5 * Math.pow((wl - 540) / 45, 2));
      // L-cone: peak ~570nm, medium-wide
      var l = Math.exp(-0.5 * Math.pow((wl - 570) / 50, 2));
      // Clip and normalize
      if (wl < 400) s *= (wl - 380) / 20;
      if (wl > 700) l *= (780 - wl) / 80;
      data.push({ wl: wl, s: s, m: m, l: l });
    }
    return data;
  };

  SpectrumExplorer.prototype.wavelengthToRgb = function (wl) {
    var r, g, b;
    if (wl >= 380 && wl < 440) {
      r = -(wl - 440) / (440 - 380);
      g = 0;
      b = 1;
    } else if (wl >= 440 && wl < 490) {
      r = 0;
      g = (wl - 440) / (490 - 440);
      b = 1;
    } else if (wl >= 490 && wl < 510) {
      r = 0;
      g = 1;
      b = -(wl - 510) / (510 - 490);
    } else if (wl >= 510 && wl < 580) {
      r = (wl - 510) / (580 - 510);
      g = 1;
      b = 0;
    } else if (wl >= 580 && wl < 645) {
      r = 1;
      g = -(wl - 645) / (645 - 580);
      b = 0;
    } else if (wl >= 645 && wl <= 780) {
      r = 1;
      g = 0;
      b = 0;
    } else {
      return { r: 0, g: 0, b: 0 };
    }
    // Intensity fall-off at edges
    var factor;
    if (wl >= 380 && wl < 420) {
      factor = 0.3 + 0.7 * (wl - 380) / (420 - 380);
    } else if (wl >= 420 && wl <= 700) {
      factor = 1.0;
    } else if (wl > 700 && wl <= 780) {
      factor = 0.3 + 0.7 * (780 - wl) / (780 - 700);
    } else {
      factor = 0;
    }
    return {
      r: Math.round(r * factor * 255),
      g: Math.round(g * factor * 255),
      b: Math.round(b * factor * 255)
    };
  };

  SpectrumExplorer.prototype.drawChart = function () {
    var ctx = this.ctx;
    var canvas = this.canvas;
    var w = canvas.width, h = canvas.height;
    var padL = 40, padR = 10, padT = 10, padB = 25;
    var plotW = w - padL - padR;
    var plotH = h - padT - padB;

    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, w, h);

    // Draw spectral gradient at bottom
    var gradH = 12;
    for (var px = 0; px < plotW; px++) {
      var wl = 380 + (px / plotW) * 400;
      var rgb = this.wavelengthToRgb(wl);
      ctx.fillStyle = 'rgb(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ')';
      ctx.fillRect(padL + px, h - padB + 2, 1, gradH);
    }

    // Draw curves
    var data = this.coneData;
    var colors = ['#3b82f6', '#22c55e', '#ef4444'];
    var labels = ['S-cone', 'M-cone', 'L-cone'];
    var channels = ['s', 'm', 'l'];

    for (var c = 0; c < 3; c++) {
      ctx.beginPath();
      ctx.strokeStyle = colors[c];
      ctx.lineWidth = 2.5;
      for (var i = 0; i < data.length; i++) {
        var x = padL + ((data[i].wl - 380) / 400) * plotW;
        var y = padT + plotH - data[i][channels[c]] * plotH * 0.95;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Label at max
      var maxIdx = 0;
      for (var i = 0; i < data.length; i++) {
        if (data[i][channels[c]] > data[maxIdx][channels[c]]) maxIdx = i;
      }
      var lx = padL + ((data[maxIdx].wl - 380) / 400) * plotW;
      var ly = padT + plotH - data[maxIdx][channels[c]] * plotH * 0.95;
      ctx.fillStyle = colors[c];
      ctx.font = '11px sans-serif';
      ctx.fillText(labels[c], lx - 15, ly - 6);
    }

    // Axes
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, h - padB);
    ctx.lineTo(w - padR, h - padB);
    ctx.stroke();

    // Tick labels
    ctx.fillStyle = '#888';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    for (var wl = 400; wl <= 780; wl += 50) {
      var tx = padL + ((wl - 380) / 400) * plotW;
      ctx.fillText(wl + 'nm', tx, h - padB + 18);
    }

    this.chartPadL = padL;
    this.chartPadT = padT;
    this.chartPlotW = plotW;
    this.chartPlotH = plotH;
  };

  SpectrumExplorer.prototype.update = function () {
    var wl = parseInt(this.slider.value);
    var rgb = this.wavelengthToRgb(wl);

    this.wlDisplay.textContent = wl + ' nm';
    this.swatch.style.background = 'rgb(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ')';
    this.wlLabel.textContent = wl + ' nm';

    // Find cone responses at this wavelength
    var idx = Math.round((wl - 380) / 400 * (this.coneData.length - 1));
    idx = Math.max(0, Math.min(this.coneData.length - 1, idx));
    var cone = this.coneData[idx];

    var sPct = Math.round(cone.s * 100);
    var mPct = Math.round(cone.m * 100);
    var lPct = Math.round(cone.l * 100);

    this.sVal.textContent = sPct + '%';
    this.mVal.textContent = mPct + '%';
    this.lVal.textContent = lPct + '%';
    this.sFill.style.width = sPct + '%';
    this.mFill.style.width = mPct + '%';
    this.lFill.style.width = lPct + '%';

    var dominant = 'L-cone';
    var maxVal = cone.l;
    if (cone.m > maxVal) { dominant = 'M-cone'; maxVal = cone.m; }
    if (cone.s > maxVal) { dominant = 'S-cone'; maxVal = cone.s; }
    this.domVal.textContent = dominant;

    // Draw indicator line on chart
    var ctx = this.ctx;
    var canvas = this.canvas;
    var w = canvas.width, h = canvas.height;

    // Redraw chart (clear/reset)
    this.drawChart();

    // Draw vertical line
    var x = this.chartPadL + ((wl - 380) / 400) * this.chartPlotW;
    ctx.strokeStyle = 'rgba(0,0,0,0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, this.chartPadT);
    ctx.lineTo(x, this.chartPadT + this.chartPlotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw current wavelength circle on each curve
    var channels = ['s', 'm', 'l'];
    var colors = ['#3b82f6', '#22c55e', '#ef4444'];
    for (var c = 0; c < 3; c++) {
      var y = this.chartPadT + this.chartPlotH - cone[channels[c]] * this.chartPlotH * 0.95;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = colors[c];
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  };

  function init() {
    var el = document.getElementById('spec-explorer');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new SpectrumExplorer(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
