(function () {
  'use strict';

  function fmtGB(v) {
    if (v >= 1000) return (v / 1000).toFixed(1) + ' TB';
    if (v >= 1) return v.toFixed(1) + ' GB';
    return (v * 1024).toFixed(0) + ' MB';
  }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function ZeroComparison(root) {
    this.root = root;
    this.tabs = root.querySelectorAll('.zero-stage-tab');
    this.activeStage = 0;
    this.update();

    var self = this;
    this.tabs.forEach(function (tab) {
      tab.addEventListener('click', function () {
        self.tabs.forEach(function (t) { t.classList.remove('active'); });
        tab.classList.add('active');
        self.activeStage = parseInt(tab.getAttribute('data-stage'));
        self.update();
      });
    });
  }

  ZeroComparison.prototype.update = function () {
    var stage = this.activeStage;
    var totalParams = 10;
    var gpuCount = 8;
    var bytes = 2;

    var paramsMem = totalParams * bytes;
    var gradsMem = totalParams * bytes;
    var optMem = totalParams * 2 * 4;
    var masterMem = totalParams * 4;

    var paramsKept, gradsKept, optKept, masterKept, total;

    switch (stage) {
      case 0:
        paramsKept = paramsMem;
        gradsKept = gradsMem;
        optKept = optMem;
        masterKept = masterMem;
        break;
      case 1:
        paramsKept = paramsMem;
        gradsKept = gradsMem;
        optKept = optMem / gpuCount;
        masterKept = masterMem / gpuCount;
        break;
      case 2:
        paramsKept = paramsMem;
        gradsKept = gradsMem / gpuCount;
        optKept = optMem / gpuCount;
        masterKept = masterMem / gpuCount;
        break;
      case 3:
        paramsKept = paramsMem / gpuCount;
        gradsKept = gradsMem / gpuCount;
        optKept = optMem / gpuCount;
        masterKept = masterMem / gpuCount;
        break;
    }

    total = paramsKept + gradsKept + optKept + masterKept;
    var maxTotal = paramsMem + gradsMem + optMem + masterMem;

    var barStack = this.root.querySelector('.zero-bar-stack');
    var topVal = Math.max(maxTotal, 1);
    var pctH = clamp(total / topVal * 100, 2, 100);
    barStack.style.height = pctH + '%';

    var comps = [
      { cls: 'param', val: paramsKept, label: 'Params' },
      { cls: 'grad', val: gradsKept, label: 'Grads' },
      { cls: 'opt', val: optKept, label: 'Opt States' },
      { cls: 'master', val: masterKept, label: 'Master W' }
    ];

    barStack.innerHTML = '';
    var totalH = comps.reduce(function (s, c) { return s + c.val; }, 0);
    comps.forEach(function (c) {
      if (c.val <= 0) return;
      var seg = document.createElement('div');
      seg.className = 'zero-bar-segment ' + c.cls;
      var segPct = (c.val / totalH) * 100;
      seg.style.height = segPct + '%';
      seg.setAttribute('title', c.label + ': ' + fmtGB(c.val));
      if (segPct > 8) {
        var lbl = document.createElement('span');
        lbl.className = 'zero-bar-seg-label';
        lbl.textContent = fmtGB(c.val);
        seg.appendChild(lbl);
      }
      barStack.appendChild(seg);
    });

    this.root.querySelector('.zero-total-value').textContent = fmtGB(total);

    var names = ['Params', 'Grads', 'Opt States', 'Master W'];
    var vals = [paramsKept, gradsKept, optKept, masterKept];
    var items = this.root.querySelectorAll('.zero-savings-item');
    items.forEach(function (item, i) {
      if (i < names.length) {
        item.querySelector('.zero-savings-label').textContent = names[i];
        item.querySelector('.zero-savings-value').textContent = fmtGB(vals[i]);
      }
    });

    var modelBytes = totalParams * bytes;
    var N = gpuCount;
    var allreduceVol = 2 * (N - 1) / N * modelBytes;
    var reduceScatterVol = (N - 1) / N * modelBytes;
    var allgatherVol = (N - 1) / N * modelBytes;

    var commVol;
    switch (stage) {
      case 0: commVol = allreduceVol; break;
      case 1: commVol = reduceScatterVol + allgatherVol; break;
      case 2: commVol = reduceScatterVol + allgatherVol; break;
      case 3: commVol = 2 * allgatherVol + reduceScatterVol; break;
    }

    var commVols = [
      allreduceVol,
      reduceScatterVol + allgatherVol,
      reduceScatterVol + allgatherVol,
      2 * allgatherVol + reduceScatterVol
    ];
    var maxComm = Math.max.apply(null, commVols);

    var commFill = this.root.querySelector('.zero-comm-fill');
    var commValue = this.root.querySelector('.zero-comm-value');
    if (commFill) commFill.style.width = Math.max(commVol / maxComm * 100, 2) + '%';
    if (commValue) commValue.textContent = fmtGB(commVol);

    var ratio = commVol / commVols[0];
    var ratioFill = this.root.querySelector('.zero-comm-ratio');
    var ratioValue = this.root.querySelector('.zero-comm-ratio-value');
    if (ratioFill) ratioFill.style.width = Math.min(ratio / 2 * 100, 100) + '%';
    if (ratioValue) ratioValue.textContent = ratio.toFixed(1) + 'x';
  };

  function init() {
    var el = document.getElementById('zero-compare');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new ZeroComparison(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
