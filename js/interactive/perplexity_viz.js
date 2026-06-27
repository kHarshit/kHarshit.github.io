(function () {
  'use strict';

  function PerplexityViz(root) {
    this.root = root;
    this.sliders = root.querySelectorAll('.ppl-slider');
    this.bars = root.querySelectorAll('.ppl-bar-fill');
    this.labels = root.querySelectorAll('.ppl-prob-label');
    this.ceDisplay = root.querySelector('.ppl-ce-value');
    this.pplDisplay = root.querySelector('.ppl-ppl-value');
    this.intuitionDisplay = root.querySelector('.ppl-intuition');

    var self = this;
    this.sliders.forEach(function (sl) {
      sl.addEventListener('input', function () { self.update(); });
    });
    this.update();
  }

  PerplexityViz.prototype.update = function () {
    var values = [];
    this.sliders.forEach(function (sl) {
      values.push(parseFloat(sl.value));
    });

    var sum = values.reduce(function (a, b) { return a + b; }, 0) || 1;
    var probs = values.map(function (v) { return v / sum; });

    var self = this;
    probs.forEach(function (p, i) {
      if (i < self.bars.length) {
        self.bars[i].style.height = Math.max(p * 100, 0.5) + '%';
      }
      if (i < self.labels.length) {
        self.labels[i].textContent = p.toFixed(3);
      }
    });

    var ce = 0;
    probs.forEach(function (p) {
      if (p > 0) {
        ce -= p * Math.log2(p);
      }
    });

    var ppl = Math.pow(2, ce);

    if (this.ceDisplay) this.ceDisplay.textContent = ce.toFixed(3);
    if (this.pplDisplay) this.pplDisplay.textContent = ppl.toFixed(2);
    if (this.intuitionDisplay) this.intuitionDisplay.textContent = ppl.toFixed(1);
  };

  function init() {
    var el = document.getElementById('ppl-viz');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new PerplexityViz(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
