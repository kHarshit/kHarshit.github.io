(function () {
  'use strict';

  function HSVExplorer(root) {
    this.root = root;
    this.modeTabs = root.querySelectorAll('.hsv-mode-tab');
    this.sliders = root.querySelectorAll('.hsv-slider');
    this.hDisplay = root.querySelector('.hsv-h-val');
    this.sDisplay = root.querySelector('.hsv-s-val');
    this.vDisplay = root.querySelector('.hsv-v-val');
    this.thirdLabel = root.querySelector('.hsv-third-label');
    this.swatch = root.querySelector('.hsv-swatch');
    this.rgbVal = root.querySelector('.hsv-rgb-val');
    this.planeCanvas = root.querySelector('.hsv-plane');
    this.planeLabel = root.querySelector('.hsv-plane-label');
    this.shapeCanvas = root.querySelector('.hsv-shape-canvas');
    this.shapeLabel = root.querySelector('.hsv-shape-name');
    this.mode = 'hsv';

    var self = this;
    this.sliders.forEach(function (sl) {
      sl.addEventListener('input', function () { self.update(); });
    });
    this.modeTabs.forEach(function (tab) {
      tab.addEventListener('click', function () {
        self.setMode(tab.getAttribute('data-mode'));
      });
    });
    this.update();
  }

  HSVExplorer.prototype.setMode = function (mode) {
    this.mode = mode;
    this.modeTabs.forEach(function (t) {
      t.classList.toggle('active', t.getAttribute('data-mode') === mode);
    });
    this.thirdLabel.textContent = mode === 'hsv' ? 'Value' : 'Lightness';
    this.update();
  };

  HSVExplorer.prototype.hsvToRgb = function (h, s, v) {
    h /= 360; s /= 100; v /= 100;
    var i = Math.floor(h * 6);
    var f = h * 6 - i;
    var p = v * (1 - s);
    var q = v * (1 - f * s);
    var t = v * (1 - (1 - f) * s);
    var r, g, b;
    switch (i % 6) {
      case 0: r = v; g = t; b = p; break;
      case 1: r = q; g = v; b = p; break;
      case 2: r = p; g = v; b = t; break;
      case 3: r = p; g = q; b = v; break;
      case 4: r = t; g = p; b = v; break;
      case 5: r = v; g = p; b = q; break;
    }
    return { r: Math.round(r * 255), g: Math.round(g * 255), b: Math.round(b * 255) };
  };

  HSVExplorer.prototype.hslToRgb = function (h, s, l) {
    h /= 360; s /= 100; l /= 100;
    if (s === 0) {
      var v = Math.round(l * 255);
      return { r: v, g: v, b: v };
    }
    var hue2rgb = function (p, q, t) {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1/6) return p + (q - p) * 6 * t;
      if (t < 1/2) return q;
      if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
      return p;
    };
    var q2 = l < 0.5 ? l * (1 + s) : l + s - l * s;
    var p2 = 2 * l - q2;
    return {
      r: Math.round(hue2rgb(p2, q2, h + 1/3) * 255),
      g: Math.round(hue2rgb(p2, q2, h) * 255),
      b: Math.round(hue2rgb(p2, q2, h - 1/3) * 255)
    };
  };

  HSVExplorer.prototype.renderPlane = function () {
    var canvas = this.planeCanvas;
    var ctx = canvas.getContext('2d');
    var w = canvas.width, h = canvas.height;
    var hue = parseInt(this.root.querySelector('.hsv-slider[data-channel="h"]').value);
    var sVal = parseInt(this.root.querySelector('.hsv-slider[data-channel="s"]').value);
    var vVal = parseInt(this.root.querySelector('.hsv-slider[data-channel="v"]').value);

    // Half-cylinder fills full card: S=0 at left edge, S=1 at right edge
    var pad = 10;
    var drawW = w - 2 * pad;
    var midX = pad; // center axis at left edge
    var rx = drawW;
    var ry = 25;
    var bodyTop = pad + ry;
    var bodyBottom = h - pad - ry;
    var bodyH = bodyBottom - bodyTop;

    ctx.clearRect(0, 0, w, h);

    // Fill body with S × V/L colors for current hue
    var imageData = ctx.createImageData(w, h);
    for (var iy = bodyTop; iy < bodyBottom; iy++) {
      for (var ix = pad; ix < pad + drawW; ix++) {
        var s = ((ix - pad) / drawW) * 100;
        var v = 100 - ((iy - bodyTop) / bodyH) * 100;
        var idx = (iy * w + ix) * 4;
        var rgb;
        if (this.mode === 'hsv') {
          rgb = this.hsvToRgb(hue, s, v);
        } else {
          rgb = this.hslToRgb(hue, s, v);
        }
        // Cylindrical shading: bright at center axis (left), dark at surface (right)
        var dx = (ix - pad) / drawW;
        var shade = 0.25 + 0.75 * Math.sqrt(Math.max(0, 1 - dx * dx));
        imageData.data[idx] = Math.round(rgb.r * shade);
        imageData.data[idx + 1] = Math.round(rgb.g * shade);
        imageData.data[idx + 2] = Math.round(rgb.b * shade);
        imageData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);

    // Body outline: left (center axis) and right (surface) sides
    ctx.strokeStyle = 'rgba(136,136,136,0.65)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(pad, bodyTop);
    ctx.lineTo(pad, bodyBottom);
    ctx.moveTo(pad + drawW, bodyTop);
    ctx.lineTo(pad + drawW, bodyBottom);
    ctx.stroke();

    // Top half-ellipse (right side from center axis)
    ctx.beginPath();
    ctx.ellipse(pad, bodyTop, rx, ry, 0, -Math.PI / 2, Math.PI / 2);
    ctx.stroke();

    // Bottom half-ellipse (right side from center axis)
    ctx.beginPath();
    ctx.ellipse(pad, bodyBottom, rx, ry, 0, -Math.PI / 2, Math.PI / 2);
    ctx.stroke();

    // Marker
    var sx = pad + (sVal / 100) * drawW;
    var sy = bodyTop + (1 - vVal / 100) * bodyH;
    ctx.beginPath();
    ctx.arc(sx, sy, 5, 0, 2 * Math.PI);
    ctx.fillStyle = '#fff';
    ctx.fill();
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(sx, sy, 3, 0, 2 * Math.PI);
    ctx.fillStyle = '#333';
    ctx.fill();

    // Vertical axis label
    ctx.save();
    ctx.fillStyle = '#888';
    ctx.font = '13px sans-serif';
    ctx.textAlign = 'left';
    ctx.translate(10, bodyTop + bodyH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(this.mode === 'hsv' ? 'Value' : 'Lightness', 0, 0);
    ctx.restore();

    // Horizontal axis label
    ctx.fillStyle = '#888';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Saturation', pad + drawW / 2, h - 3);
  };

  HSVExplorer.prototype.update = function () {
    var h = parseInt(this.root.querySelector('.hsv-slider[data-channel="h"]').value);
    var s = parseInt(this.root.querySelector('.hsv-slider[data-channel="s"]').value);
    var v = parseInt(this.root.querySelector('.hsv-slider[data-channel="v"]').value);

    this.hDisplay.textContent = h + '\u00b0';
    this.sDisplay.textContent = s + '%';
    this.vDisplay.textContent = v + '%';

    var rgb = this.mode === 'hsv' ? this.hsvToRgb(h, s, v) : this.hslToRgb(h, s, v);
    this.swatch.style.background = 'rgb(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ')';
    this.rgbVal.textContent = '(' + rgb.r + ', ' + rgb.g + ', ' + rgb.b + ')';

    this.renderPlane();
    this.renderShape();
  };

  HSVExplorer.prototype.renderShape = function () {
    var canvas = this.shapeCanvas;
    var ctx = canvas.getContext('2d');
    var w = canvas.width, h = canvas.height;
    var cx = w / 2, cy = h / 2;
    var radius = 105;

    var hue = parseInt(this.root.querySelector('.hsv-slider[data-channel="h"]').value);
    var sVal = parseInt(this.root.querySelector('.hsv-slider[data-channel="s"]').value) / 100;
    var vVal = parseInt(this.root.querySelector('.hsv-slider[data-channel="v"]').value) / 100;

    ctx.clearRect(0, 0, w, h);

    // Show current V/L as label
    var label = (this.mode === 'hsv' ? 'V=' : 'L=') + Math.round(vVal * 100) + '%';
    this.shapeLabel.textContent = 'Top-view ' + label;

    // Fill circle with all hues at current V/L, saturation as radius
    var imageData = ctx.createImageData(w, h);
    for (var iy = 0; iy < h; iy++) {
      for (var ix = 0; ix < w; ix++) {
        var dx = ix - cx;
        var dy = iy - cy;
        var dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > radius) continue;

        var sat = dist / radius;
        var angle = Math.atan2(-dy, dx);
        if (angle < 0) angle += 2 * Math.PI;
        var hDeg = angle * 180 / Math.PI;

        var idx = (iy * w + ix) * 4;
        var rgb;
        if (this.mode === 'hsv') {
          rgb = this.hsvToRgb(hDeg, sat * 100, vVal * 100);
        } else {
          rgb = this.hslToRgb(hDeg, sat * 100, vVal * 100);
        }
        imageData.data[idx] = rgb.r;
        imageData.data[idx + 1] = rgb.g;
        imageData.data[idx + 2] = rgb.b;
        imageData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);

    // Draw circle outline
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(136,136,136,0.5)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Gray center dot (S=0)
    ctx.beginPath();
    ctx.arc(cx, cy, 2, 0, 2 * Math.PI);
    ctx.fillStyle = '#999';
    ctx.fill();

    // Degree markings
    ctx.save();
    ctx.font = '10px sans-serif';
    ctx.fillStyle = '#888';
    ctx.strokeStyle = '#888';
    ctx.lineWidth = 1;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (var deg = 0; deg < 360; deg += 30) {
      var rad = deg * Math.PI / 180;
      var isCardinal = (deg % 90 === 0);
      var tickLen = isCardinal ? 8 : 5;
      var x1 = cx + radius * Math.cos(rad);
      var y1 = cy - radius * Math.sin(rad);
      var x2 = cx + (radius + tickLen) * Math.cos(rad);
      var y2 = cy - (radius + tickLen) * Math.sin(rad);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      if (deg % 45 === 0) {
        var labelR = radius + tickLen + 7;
        var lx = cx + labelR * Math.cos(rad);
        var ly = cy - labelR * Math.sin(rad);
        ctx.fillText(deg + '\u00b0', lx, ly);
      }
    }
    ctx.restore();

    // Marker at current (hue, saturation) position
    var markerAngle = hue * Math.PI / 180;
    var markerR = sVal * radius;
    var mx = cx + markerR * Math.cos(markerAngle);
    var my = cy - markerR * Math.sin(markerAngle);

    ctx.beginPath();
    ctx.arc(mx, my, 6, 0, 2 * Math.PI);
    ctx.fillStyle = '#fff';
    ctx.fill();
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(mx, my, 3, 0, 2 * Math.PI);
    ctx.fillStyle = '#333';
    ctx.fill();
  };

  function init() {
    var el = document.getElementById('hsv-explorer');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new HSVExplorer(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
