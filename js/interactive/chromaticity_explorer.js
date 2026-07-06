(function () {
  'use strict';

  function ChromaticityExplorer(root) {
    this.root = root;
    this.canvas = root.querySelector('.chroma-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.swatch = root.querySelector('.chroma-preview-swatch');
    this.rVal = root.querySelector('.chroma-r-val');
    this.gVal = root.querySelector('.chroma-g-val');
    this.bVal = root.querySelector('.chroma-b-val');
    this.brightnessSlider = root.querySelector('.chroma-brightness');
    this.brightnessVal = root.querySelector('.chroma-brightness-val');
    this.currentPoint = { r: 1, g: 0 };
    this.brightness = 128;

    var self = this;
    this.brightnessSlider.addEventListener('input', function () {
      self.brightness = parseInt(self.brightnessSlider.value);
      self.brightnessVal.textContent = self.brightness;
      self.renderCanvas();
      // Re-render the marker and preview with current point
      self.updatePreview(self.currentPoint.r, self.currentPoint.g);
    });

    var self = this;
    this.canvas.addEventListener('mousemove', function (e) {
      var rect = self.canvas.getBoundingClientRect();
      var x = e.clientX - rect.left;
      var y = e.clientY - rect.top;
      var pt = self.canvasToRg(x, y);
      if (pt) {
        self.currentPoint = pt;
        self.updatePreview(pt.r, pt.g);
        self.renderCanvas();
        self.drawMarker(pt.r, pt.g);
      }
    });

    this.canvas.addEventListener('mouseleave', function () {
      // Reset to white point
      self.currentPoint = { r: 1/3, g: 1/3 };
      self.renderCanvas();
      self.drawMarker(self.currentPoint.r, self.currentPoint.g);
      self.updatePreview(self.currentPoint.r, self.currentPoint.g);
    });

    // Initialize
    this.setupCanvas();
    this.renderCanvas();
    this.currentPoint = { r: 1/3, g: 1/3 };
    this.drawMarker(this.currentPoint.r, this.currentPoint.g);
    this.updatePreview(this.currentPoint.r, this.currentPoint.g);
  }

  ChromaticityExplorer.prototype.setupCanvas = function () {
    var canvas = this.canvas;
    var size = Math.min(canvas.width, canvas.height);
    var pad = 15;
    this.triSize = size - 2 * pad;
    this.triPad = pad;

    // Pre-compute triangle vertices (r,g coordinates of corners)
    // Red corner: r=1, g=0
    // Green corner: r=0, g=1
    // Blue corner: r=0, g=0
    // In the 2D projection, place them as an equilateral-ish triangle
    this.corners = {
      r: { x: pad + this.triSize * 0.95, y: pad + this.triSize * 0.95 },  // red (r=1,g=0) bottom right
      g: { x: pad + this.triSize * 0.05, y: pad + this.triSize * 0.95 },  // green (r=0,g=1) bottom left
      b: { x: pad + this.triSize * 0.5, y: pad + this.triSize * 0.05 }    // blue (r=0,g=0) top center
    };
  };

  ChromaticityExplorer.prototype.rgToCanvas = function (r, g) {
    // Convert (r,g) chromaticity to canvas coordinates via barycentric mapping
    var b = 1 - r - g;
    if (b < -0.01 || r < -0.01 || g < -0.01) return null;

    var cr = this.corners.r;
    var cg = this.corners.g;
    var cb = this.corners.b;

    // Barycentric: point = r * R + g * G + b * B   (where b = 1-r-g)
    // But this isn't quite right for the triangle shape.
    // Better: use the rg coordinates directly in a 2D projection
    // Red at (1,0), Green at (0,1), Blue at (0,0)
    // Map to a 2D grid where R=(1,0), G=(0,1), B=(0,0) -> we use a simple shear transform

    // Simple linear mapping: place in a 2D simplex
    // x = r * R.x + g * G.x + b * B.x
    // y = r * R.y + g * G.y + b * B.y
    var x = r * cr.x + g * cg.x + b * cb.x;
    var y = r * cr.y + g * cg.y + b * cb.y;
    return { x: x, y: y };
  };

  ChromaticityExplorer.prototype.canvasToRg = function (cx, cy) {
    // Inverse mapping from canvas coords to (r,g)
    // Use the same barycentric mapping but inverted
    var cr = this.corners.r;
    var cg = this.corners.g;
    var cb = this.corners.b;

    // Solve: [cr.x - cb.x, cg.x - cb.x] [r]   [cx - cb.x]
    //        [cr.y - cb.y, cg.y - cb.y] [g] = [cy - cb.y]
    var a11 = cr.x - cb.x;
    var a12 = cg.x - cb.x;
    var a21 = cr.y - cb.y;
    var a22 = cg.y - cb.y;
    var dx = cx - cb.x;
    var dy = cy - cb.y;

    var det = a11 * a22 - a12 * a21;
    if (Math.abs(det) < 1e-10) return null;

    var r = (dx * a22 - a12 * dy) / det;
    var g = (a11 * dy - dx * a21) / det;
    var b = 1 - r - g;

    if (r < -0.02 || g < -0.02 || b < -0.02) return null;

    return { r: Math.max(0, Math.min(1, r)), g: Math.max(0, Math.min(1, g)) };
  };

  ChromaticityExplorer.prototype.rgToColor = function (r, g, brightness) {
    var b = 1 - r - g;
    if (b < 0) b = 0;
    var maxC = Math.max(r, g, b);
    if (maxC === 0) return { r: 0, g: 0, b: 0 };
    // Scale to use full brightness range
    var scale = brightness / 255 / maxC;
    return {
      r: Math.round(r * scale * 255),
      g: Math.round(g * scale * 255),
      b: Math.round(b * scale * 255)
    };
  };

  ChromaticityExplorer.prototype.renderCanvas = function () {
    var ctx = this.ctx;
    var canvas = this.canvas;
    var w = canvas.width, h = canvas.height;
    var brightness = parseInt(this.brightnessSlider.value);

    ctx.clearRect(0, 0, w, h);

    // Draw triangle filled with colors
    var cr = this.corners.r;
    var cg = this.corners.g;
    var cb = this.corners.b;

    // Find bounding box of triangle
    var minX = Math.min(cr.x, cg.x, cb.x);
    var maxX = Math.max(cr.x, cg.x, cb.x);
    var minY = Math.min(cr.y, cg.y, cb.y);
    var maxY = Math.max(cr.y, cg.y, cb.y);

    var imageData = ctx.createImageData(w, h);
    var buf = new ArrayBuffer(imageData.data.length);
    var buf8 = new Uint8ClampedArray(buf);
    var data = new Uint32Array(buf);

    for (var y = Math.floor(minY); y < Math.ceil(maxY); y++) {
      for (var x = Math.floor(minX); x < Math.ceil(maxX); x++) {
        var pt = this.canvasToRg(x + 0.5, y + 0.5);
        if (!pt) continue;

        var b = 1 - pt.r - pt.g;
        if (b < 0 || pt.r < 0 || pt.g < 0) continue;

        var color = this.rgToColor(pt.r, pt.g, brightness);
        var idx = (y * w + x);
        data[idx] = (255 << 24) | (color.b << 16) | (color.g << 8) | color.r;
      }
    }

    imageData.data.set(new Uint8ClampedArray(buf));
    ctx.putImageData(imageData, 0, 0);

    // Draw triangle outline
    ctx.strokeStyle = 'rgba(0,0,0,0.25)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cr.x, cr.y);
    ctx.lineTo(cg.x, cg.y);
    ctx.lineTo(cb.x, cb.y);
    ctx.closePath();
    ctx.stroke();

    // Labels at corners
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillStyle = '#888';
    ctx.fillText('Red', cr.x - 20, cr.y + 18);
    ctx.fillText('Green', cg.x + 20, cg.y + 18);
    ctx.fillText('Blue', cb.x, cb.y - 10);

    // White point marker
    var wp = this.rgToCanvas(1/3, 1/3);
    if (wp) {
      ctx.beginPath();
      ctx.arc(wp.x, wp.y, 4, 0, 2 * Math.PI);
      ctx.fillStyle = '#888';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  };

  ChromaticityExplorer.prototype.drawMarker = function (r, g) {
    var pt = this.rgToCanvas(r, g);
    if (!pt) return;
    var ctx = this.ctx;
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 6, 0, 2 * Math.PI);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 5, 0, 2 * Math.PI);
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  };

  ChromaticityExplorer.prototype.updatePreview = function (r, g) {
    var brightness = parseInt(this.brightnessSlider.value);
    var color = this.rgToColor(r, g, brightness);
    var b = 1 - r - g;

    this.swatch.style.background = 'rgb(' + color.r + ',' + color.g + ',' + color.b + ')';
    this.rVal.textContent = r.toFixed(3);
    this.gVal.textContent = g.toFixed(3);
    this.bVal.textContent = Math.max(0, b).toFixed(3);
  };

  function init() {
    var el = document.getElementById('chroma-explorer');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new ChromaticityExplorer(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
