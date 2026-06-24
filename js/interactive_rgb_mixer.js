(function () {
  'use strict';

  function RGBMixer(root) {
    this.root = root;
    this.sliders = root.querySelectorAll('.rgb-slider');
    this.rDisplay = root.querySelector('.rgb-r-val');
    this.gDisplay = root.querySelector('.rgb-g-val');
    this.bDisplay = root.querySelector('.rgb-b-val');
    this.swatch = root.querySelector('.rgb-swatch');
    this.hexVal = root.querySelector('.rgb-hex-val');
    this.rgbVal = root.querySelector('.rgb-rgb-val');
    this.hsvVal = root.querySelector('.rgb-hsv-val');
    this.cmykVal = root.querySelector('.rgb-cmyk-val');
    this.presets = root.querySelectorAll('.rgb-preset-btn');

    var self = this;
    this.sliders.forEach(function (sl) {
      sl.addEventListener('input', function () { self.update(); });
    });
    this.presets.forEach(function (btn) {
      btn.addEventListener('click', function () {
        var r = parseInt(btn.getAttribute('data-r'));
        var g = parseInt(btn.getAttribute('data-g'));
        var b = parseInt(btn.getAttribute('data-b'));
        self.setValues(r, g, b);
      });
    });
    this.update();
  }

  RGBMixer.prototype.setValues = function (r, g, b) {
    this.root.querySelector('.rgb-slider[data-channel="r"]').value = r;
    this.root.querySelector('.rgb-slider[data-channel="g"]').value = g;
    this.root.querySelector('.rgb-slider[data-channel="b"]').value = b;
    this.update();
  };

  RGBMixer.prototype.rgbToHsv = function (r, g, b) {
    r /= 255; g /= 255; b /= 255;
    var max = Math.max(r, g, b), min = Math.min(r, g, b);
    var h, s, v = max;
    var d = max - min;
    s = max === 0 ? 0 : d / max;
    if (max === min) {
      h = 0;
    } else {
      if (max === r) { h = ((g - b) / d + (g < b ? 6 : 0)) / 6; }
      else if (max === g) { h = ((b - r) / d + 2) / 6; }
      else { h = ((r - g) / d + 4) / 6; }
    }
    return { h: Math.round(h * 360), s: Math.round(s * 100), v: Math.round(v * 100) };
  };

  RGBMixer.prototype.rgbToCmyk = function (r, g, b) {
    var cr = 1 - r / 255, cg = 1 - g / 255, cb = 1 - b / 255;
    var k = Math.min(cr, cg, cb);
    if (k === 1) return { c: 0, m: 0, y: 0, k: 100 };
    var c = Math.round(((cr - k) / (1 - k)) * 100);
    var m = Math.round(((cg - k) / (1 - k)) * 100);
    var y = Math.round(((cb - k) / (1 - k)) * 100);
    return { c: c, m: m, y: y, k: Math.round(k * 100) };
  };

  RGBMixer.prototype.componentToHex = function (c) {
    var hex = c.toString(16);
    return hex.length === 1 ? '0' + hex : hex;
  };

  RGBMixer.prototype.update = function () {
    var r = parseInt(this.root.querySelector('.rgb-slider[data-channel="r"]').value);
    var g = parseInt(this.root.querySelector('.rgb-slider[data-channel="g"]').value);
    var b = parseInt(this.root.querySelector('.rgb-slider[data-channel="b"]').value);

    this.rDisplay.textContent = r;
    this.gDisplay.textContent = g;
    this.bDisplay.textContent = b;

    this.swatch.style.background = 'rgb(' + r + ',' + g + ',' + b + ')';

    this.hexVal.textContent = '#' + this.componentToHex(r) + this.componentToHex(g) + this.componentToHex(b);
    this.rgbVal.textContent = r + ', ' + g + ', ' + b;

    var hsv = this.rgbToHsv(r, g, b);
    this.hsvVal.textContent = hsv.h + '\u00b0, ' + hsv.s + '%, ' + hsv.v + '%';

    var cmyk = this.rgbToCmyk(r, g, b);
    this.cmykVal.textContent = cmyk.c + '%, ' + cmyk.m + '%, ' + cmyk.y + '%, ' + cmyk.k + '%';
  };

  function init() {
    var el = document.getElementById('rgb-mixer');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new RGBMixer(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
