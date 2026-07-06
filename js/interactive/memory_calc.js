(function () {
  'use strict';

  function fmtGB(v) {
    if (v >= 1000) return (v / 1000).toFixed(1) + ' TB';
    if (v >= 1) return v.toFixed(1) + ' GB';
    return (v * 1024).toFixed(0) + ' MB';
  }

  function MemoryCalculator(root) {
    this.root = root;
    this.slider = root.querySelector('.mem-slider');
    this.precision = root.querySelector('.mem-precision');
    this.update();

    var self = this;
    this.slider.addEventListener('input', function () { self.update(); });
    this.precision.addEventListener('change', function () { self.update(); });
  }

  MemoryCalculator.prototype.getConfig = function () {
    var paramsB = parseFloat(this.slider.value);
    var prec = this.precision.value;
    var bytes, actMultiplier;
    if (prec === 'fp32') {
      bytes = 4;
      actMultiplier = 4;
    } else if (prec === 'bf16') {
      bytes = 2;
      actMultiplier = 2;
    } else {
      bytes = 1;
      actMultiplier = 1;
    }
    var params = paramsB * 1e9;
    var master = (prec === 'fp32') ? 0 : 4;
    var optMult = (prec === 'int8') ? 2 : 4;
    return {
      paramsB: paramsB,
      bytes: bytes,
      paramsMem: params * bytes / 1e9,
      gradsMem: params * bytes / 1e9,
      optMem: params * 2 * optMult / 1e9,
      masterMem: params * master / 1e9,
      actMem: params * actMultiplier / 1e9 * 0.3,
      tempMem: paramsB * 0.8,
      label: prec.toUpperCase()
    };
  };

  MemoryCalculator.prototype.update = function () {
    var c = this.getConfig();
    var total = c.paramsMem + c.gradsMem + c.optMem + c.masterMem + c.actMem + c.tempMem;

    function qs(el, sel) { try { return el.querySelector(sel); } catch(e) { return null; } }

    var pd = qs(this.root, '.mem-params-display');
    if (pd) pd.textContent = c.paramsB >= 1000 ? (c.paramsB / 1000).toFixed(1) + 'T' : c.paramsB + 'B';

    var pl = qs(this.root, '.mem-prec-label');
    if (pl) pl.textContent = c.label;

    var memValues = [c.paramsMem, c.gradsMem, c.optMem, c.masterMem, c.actMem, c.tempMem];
    var maxMem = 10000;

    var fills = this.root.querySelectorAll('.mem-calc-bar-fill');
    var vals = this.root.querySelectorAll('.mem-calc-bar-val');
    for (var i = 0; i < memValues.length; i++) {
      if (i < fills.length) {
        var pct = (memValues[i] / maxMem) * 100;
        fills[i].style.width = Math.max(pct, 0.5) + '%';
      }
      if (i < vals.length) {
        vals[i].textContent = memValues[i] > 0 ? fmtGB(memValues[i]) : '-';
      }
    }

    var tv = qs(this.root, '.mem-total-value');
    if (tv) tv.textContent = fmtGB(total);

    var sp = qs(this.root, '.mem-stat-params');
    if (sp) sp.textContent = c.paramsB >= 1000 ? (c.paramsB / 1000).toFixed(1) + 'T' : c.paramsB + 'B';

    var sg = qs(this.root, '.mem-stat-gpus');
    if (sg) sg.textContent = Math.ceil(total / 80) + ' × A100';

    var spr = qs(this.root, '.mem-stat-prec');
    if (spr) spr.textContent = c.label;
  };

  function init() {
    var el = document.getElementById('mem-calc');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new MemoryCalculator(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
