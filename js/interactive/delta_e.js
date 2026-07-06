(function () {
  'use strict';

  function DeltaECompare(root) {
    this.root = root;
    this.sliders = root.querySelectorAll('.de-slider');
    this.aSwatch = root.querySelector('.de-swatch-a');
    this.bSwatch = root.querySelector('.de-swatch-b');
    this.aHex = root.querySelector('.de-ahex-val');
    this.bHex = root.querySelector('.de-bhex-val');
    this.arVal = root.querySelector('.de-ar-val');
    this.agVal = root.querySelector('.de-ag-val');
    this.abVal = root.querySelector('.de-ab-val');
    this.brVal = root.querySelector('.de-br-val');
    this.bgVal = root.querySelector('.de-bg-val');
    this.bbVal = root.querySelector('.de-bb-val');
    this.cie76Val = root.querySelector('.de-cie76-val');
    this.cie94Val = root.querySelector('.de-cie94-val');
    this.cie00Val = root.querySelector('.de-cie00-val');
    this.perceptionVal = root.querySelector('.de-perception-val');
    this.presets = root.querySelectorAll('.de-preset-btn');

    var self = this;
    this.sliders.forEach(function (sl) {
      sl.addEventListener('input', function () { self.update(); });
    });
    this.presets.forEach(function (btn) {
      btn.addEventListener('click', function () {
        var a = btn.getAttribute('data-a').split(',').map(Number);
        var b = btn.getAttribute('data-b').split(',').map(Number);
        self.setColors(a, b);
      });
    });
    this.update();
  }

  DeltaECompare.prototype.setColors = function (a, b) {
    this.root.querySelector('.de-slider[data-side="a"][data-channel="r"]').value = a[0];
    this.root.querySelector('.de-slider[data-side="a"][data-channel="g"]').value = a[1];
    this.root.querySelector('.de-slider[data-side="a"][data-channel="b"]').value = a[2];
    this.root.querySelector('.de-slider[data-side="b"][data-channel="r"]').value = b[0];
    this.root.querySelector('.de-slider[data-side="b"][data-channel="g"]').value = b[1];
    this.root.querySelector('.de-slider[data-side="b"][data-channel="b"]').value = b[2];
    this.update();
  };

  DeltaECompare.prototype.componentToHex = function (c) {
    var hex = c.toString(16);
    return hex.length === 1 ? '0' + hex : hex;
  };

  DeltaECompare.prototype.getColor = function (side) {
    var r = parseInt(this.root.querySelector('.de-slider[data-side="' + side + '"][data-channel="r"]').value);
    var g = parseInt(this.root.querySelector('.de-slider[data-side="' + side + '"][data-channel="g"]').value);
    var b = parseInt(this.root.querySelector('.de-slider[data-side="' + side + '"][data-channel="b"]').value);
    return { r: r, g: g, b: b };
  };

  DeltaECompare.prototype.linearize = function (c) {
    c = c / 255;
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  };

  DeltaECompare.prototype.rgbToXyz = function (r, g, b) {
    var rl = this.linearize(r);
    var gl = this.linearize(g);
    var bl = this.linearize(b);
    var x = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375;
    var y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750;
    var z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041;
    return { x: x, y: y, z: z };
  };

  DeltaECompare.prototype.xyzToLab = function (x, y, z) {
    var xn = 0.95047, yn = 1.0, zn = 1.08883;
    var fx = this.labF(x / xn);
    var fy = this.labF(y / yn);
    var fz = this.labF(z / zn);
    return {
      l: 116 * fy - 16,
      a: 500 * (fx - fy),
      b: 200 * (fy - fz)
    };
  };

  DeltaECompare.prototype.labF = function (t) {
    return t > 0.008856 ? Math.pow(t, 1/3) : (7.787 * t + 16/116);
  };

  DeltaECompare.prototype.rgbToLab = function (r, g, b) {
    var xyz = this.rgbToXyz(r, g, b);
    return this.xyzToLab(xyz.x, xyz.y, xyz.z);
  };

  DeltaECompare.prototype.cie76 = function (l1, a1, b1, l2, a2, b2) {
    return Math.sqrt((l2 - l1) * (l2 - l1) + (a2 - a1) * (a2 - a1) + (b2 - b1) * (b2 - b1));
  };

  DeltaECompare.prototype.cie94 = function (l1, a1, b1, l2, a2, b2) {
    var kL = 1, kC = 1, kH = 1;
    var c1 = Math.sqrt(a1 * a1 + b1 * b1);
    var c2 = Math.sqrt(a2 * a2 + b2 * b2);
    var dc = c1 - c2;
    var dl = l1 - l2;
    var da = a1 - a2;
    var db = b1 - b2;
    var dh2 = da * da + db * db - dc * dc;
    var dh = Math.sqrt(Math.max(0, dh2));
    var cAb = Math.sqrt(c1 * c2);
    var sL = 1;
    var sC = 1 + 0.045 * cAb;
    var sH = 1 + 0.015 * cAb;
    return Math.sqrt(
      (dl / (kL * sL)) * (dl / (kL * sL)) +
      (dc / (kC * sC)) * (dc / (kC * sC)) +
      (dh / (kH * sH)) * (dh / (kH * sH))
    );
  };

  DeltaECompare.prototype.cie00 = function (l1, a1, b1, l2, a2, b2) {
    var kL = 1, kC = 1, kH = 1;
    var lBar = (l1 + l2) / 2;
    var c1 = Math.sqrt(a1 * a1 + b1 * b1);
    var c2 = Math.sqrt(a2 * a2 + b2 * b2);
    var cBar = (c1 + c2) / 2;

    var g = 0.5 * (1 - Math.sqrt(Math.pow(cBar, 7) / (Math.pow(cBar, 7) + Math.pow(25, 7))));
    var a1p = a1 * (1 + g);
    var a2p = a2 * (1 + g);
    var c1p = Math.sqrt(a1p * a1p + b1 * b1);
    var c2p = Math.sqrt(a2p * a2p + b2 * b2);
    var cBarP = (c1p + c2p) / 2;

    var h1p = 0, h2p = 0;
    if (c1p > 1e-12) { h1p = Math.atan2(b1, a1p) * 180 / Math.PI; if (h1p < 0) h1p += 360; }
    if (c2p > 1e-12) { h2p = Math.atan2(b2, a2p) * 180 / Math.PI; if (h2p < 0) h2p += 360; }

    var hDiff = h2p - h1p;
    var hBarP;
    if (c1p < 1e-12 || c2p < 1e-12) {
      hBarP = 0;
    } else if (Math.abs(hDiff) > 180) {
      hBarP = (h1p + h2p + 360) / 2;
    } else {
      hBarP = (h1p + h2p) / 2;
    }
    if (hBarP >= 360) hBarP -= 360;

    var t = 1 - 0.17 * Math.cos((hBarP - 30) * Math.PI / 180) +
      0.24 * Math.cos((2 * hBarP) * Math.PI / 180) +
      0.32 * Math.cos((3 * hBarP + 6) * Math.PI / 180) -
      0.20 * Math.cos((4 * hBarP - 63) * Math.PI / 180);

    var dHp;
    if (c1p < 1e-12 || c2p < 1e-12) {
      dHp = 0;
    } else if (Math.abs(hDiff) <= 180) {
      dHp = h2p - h1p;
    } else {
      dHp = h2p - h1p;
      if (dHp > 180) dHp -= 360;
      else dHp += 360;
    }

    var dLp = l2 - l1;
    var dCp = c2p - c1p;
    var dHpRad = dHp * Math.PI / 180;
    var dHpVal = 2 * Math.sqrt(c1p * c2p) * Math.sin(dHpRad / 2);

    var sl = 1 + (0.015 * (lBar - 50) * (lBar - 50)) / Math.sqrt(20 + (lBar - 50) * (lBar - 50));
    var sc = 1 + 0.045 * cBarP;
    var sh = 1 + 0.015 * cBarP * t;

    var dTheta = 30 * Math.exp(-((hBarP - 275) / 25) * ((hBarP - 275) / 25));
    var rc = 2 * Math.sqrt(Math.pow(cBarP, 7) / (Math.pow(cBarP, 7) + Math.pow(25, 7)));
    var rt = -rc * Math.sin(2 * dTheta * Math.PI / 180);

    return Math.sqrt(
      (dLp / (kL * sl)) * (dLp / (kL * sl)) +
      (dCp / (kC * sc)) * (dCp / (kC * sc)) +
      (dHpVal / (kH * sh)) * (dHpVal / (kH * sh)) +
      rt * (dCp / (kC * sc)) * (dHpVal / (kH * sh))
    );
  };

  DeltaECompare.prototype.getPerception = function (de) {
    if (de < 1) return 'Not perceptible';
    if (de < 2) return 'Perceptible on close observation';
    if (de < 10) return 'Perceptible at a glance';
    if (de < 50) return 'More similar than opposite';
    return 'Nearly opposite colors';
  };

  DeltaECompare.prototype.update = function () {
    var colorA = this.getColor('a');
    var colorB = this.getColor('b');

    var hexA = '#' + this.componentToHex(colorA.r) + this.componentToHex(colorA.g) + this.componentToHex(colorA.b);
    var hexB = '#' + this.componentToHex(colorB.r) + this.componentToHex(colorB.g) + this.componentToHex(colorB.b);

    this.aSwatch.style.background = hexA;
    this.bSwatch.style.background = hexB;
    this.aHex.textContent = hexA;
    this.bHex.textContent = hexB;
    this.arVal.textContent = colorA.r;
    this.agVal.textContent = colorA.g;
    this.abVal.textContent = colorA.b;
    this.brVal.textContent = colorB.r;
    this.bgVal.textContent = colorB.g;
    this.bbVal.textContent = colorB.b;

    var labA = this.rgbToLab(colorA.r, colorA.g, colorA.b);
    var labB = this.rgbToLab(colorB.r, colorB.g, colorB.b);

    var de76 = this.cie76(labA.l, labA.a, labA.b, labB.l, labB.a, labB.b);
    var de94 = this.cie94(labA.l, labA.a, labA.b, labB.l, labB.a, labB.b);
    var de00 = this.cie00(labA.l, labA.a, labA.b, labB.l, labB.a, labB.b);

    this.cie76Val.textContent = de76.toFixed(2);
    this.cie94Val.textContent = de94.toFixed(2);
    this.cie00Val.textContent = de00.toFixed(2);
    this.perceptionVal.textContent = this.getPerception(de00);
  };

  function init() {
    var el = document.getElementById('de-compare');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new DeltaECompare(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
