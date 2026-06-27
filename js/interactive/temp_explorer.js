(function () {
  'use strict';

  function TempExplorer(root) {
    this.root = root;
    this.tempSlider = root.querySelector('.temp-temp-slider');
    this.topkSlider = root.querySelector('.temp-topk-slider');
    this.toppSlider = root.querySelector('.temp-topp-slider');
    this.tempDisplay = root.querySelector('.temp-temp-display');
    this.topkDisplay = root.querySelector('.temp-topk-display');
    this.toppDisplay = root.querySelector('.temp-topp-display');
    this.origChart = root.querySelector('.temp-orig-chart');
    this.modChart = root.querySelector('.temp-mod-chart');
    this.entropyDisplay = root.querySelector('.temp-entropy-display');
    this.argmaxDisplay = root.querySelector('.temp-argmax-display');
    this.zerodDisplay = root.querySelector('.temp-zerod-display');

    this.tokens = [
      'the', 'and', 'that', 'have', 'with', 'this', 'will', 'from',
      'they', 'which', 'would', 'their', 'there', 'about', 'could',
      'should', 'after', 'while', 'because', 'through',
    ];

    var rawLogits = [
      3.2, 2.8, 2.5, 1.9, 1.7, 1.5, 1.3, 1.1,
      0.8, 0.6, 0.4, 0.2, 0.0, -0.3, -0.5,
      -0.8, -1.0, -1.3, -1.6, -2.0,
    ];
    this.baseLogits = rawLogits;

    var self = this;
    this.tempSlider.addEventListener('input', function () { self.update(); });
    this.topkSlider.addEventListener('input', function () { self.update(); });
    this.toppSlider.addEventListener('input', function () { self.update(); });
    this.update();
  }

  TempExplorer.prototype.softmax = function (logits) {
    var max = logits.reduce(function (a, b) { return Math.max(a, b); }, -Infinity);
    var exps = logits.map(function (l) { return Math.exp(l - max); });
    var sum = exps.reduce(function (a, b) { return a + b; }, 0);
    return exps.map(function (e) { return e / sum; });
  };

  TempExplorer.prototype.entropy = function (probs) {
    var e = 0;
    for (var i = 0; i < probs.length; i++) {
      if (probs[i] > 0) e -= probs[i] * Math.log2(probs[i]);
    }
    return e;
  };

  TempExplorer.prototype.renderBars = function (container, probs, tokens, highlightIdx, maxProb) {
    var n = probs.length;
    var html = '';
    for (var i = 0; i < n; i++) {
      var p = probs[i];
      var h = Math.max((p / maxProb) * 100, 0.5);
      var isZero = p < 0.0001;
      var color = isZero ? '#ddd' : (i === highlightIdx ? '#20B2AA' : '#3b82f6');
      var barColor = isZero ? '#e5e7eb' : color;
      html +=
        '<div style="display:flex;flex-direction:column;align-items:center;flex:1;min-width:0;gap:3px;">' +
        '<div style="width:100%;height:160px;background:#f1f5f9;border-radius:4px;position:relative;overflow:hidden;display:flex;align-items:flex-end;">' +
        '<div style="width:100%;height:' + h.toFixed(1) + '%;background:' + barColor + ';border-radius:4px 4px 0 0;transition:height 0.12s ease,background 0.12s;min-height:' + (isZero ? '0' : '2') + 'px;"></div>' +
        '</div>' +
        '<span style="font-size:0.65rem;font-weight:600;color:' + (isZero ? '#ccc' : '#555') + ';white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%;">' + tokens[i] + '</span>' +
        '<span style="font-size:0.6rem;font-weight:700;color:' + (isZero ? '#ddd' : '#20B2AA') + ';font-variant-numeric:tabular-nums;">' + (p < 0.001 ? '0' : (p * 100).toFixed(1)) + '%</span>' +
        '</div>';
    }
    container.innerHTML = html;
  };

  TempExplorer.prototype.update = function () {
    var tempVal = parseInt(this.tempSlider.value, 10);
    var temperature = tempVal / 100;
    var topk = parseInt(this.topkSlider.value, 10);
    var topp = parseInt(this.toppSlider.value, 10) / 100;

    this.tempDisplay.textContent = temperature.toFixed(2);
    this.topkDisplay.textContent = topk;
    this.toppDisplay.textContent = topp.toFixed(2);

    var logits = this.baseLogits.slice();

    var origProbs = this.softmax(logits);

    var scaledLogits = logits.map(function (l) { return l / temperature; });
    var modProbs = this.softmax(scaledLogits);

    var indexed = modProbs.map(function (p, i) { return { prob: p, idx: i }; });
    indexed.sort(function (a, b) { return b.prob - a.prob; });

    var kept = new Array(modProbs.length).fill(false);
    for (var i = 0; i < topk && i < indexed.length; i++) {
      kept[indexed[i].idx] = true;
    }

    var cumSum = 0;
    for (var j = 0; j < indexed.length; j++) {
      if (cumSum >= topp && kept[indexed[j].idx]) {
        var remaining = indexed.slice(j).filter(function (x) { return kept[x.idx]; });
        for (var r = 0; r < remaining.length; r++) {
          kept[remaining[r].idx] = false;
        }
      }
      if (kept[indexed[j].idx]) cumSum += indexed[j].prob;
    }

    for (var k = 0; k < modProbs.length; k++) {
      if (!kept[k]) modProbs[k] = 0;
    }

    var sum = modProbs.reduce(function (a, b) { return a + b; }, 0);
    if (sum > 0) {
      modProbs = modProbs.map(function (p) { return p / sum; });
    }

    var origHighlight = 0;
    var origMax = origProbs[0];
    for (var oi = 1; oi < origProbs.length; oi++) {
      if (origProbs[oi] > origMax) { origMax = origProbs[oi]; origHighlight = oi; }
    }

    var modHighlight = 0;
    var modMax = modProbs[0];
    for (var mi = 1; mi < modProbs.length; mi++) {
      if (modProbs[mi] > modMax) { modMax = modProbs[mi]; modHighlight = mi; }
    }

    var origMaxProb = origProbs.reduce(function (a, b) { return Math.max(a, b); }, 0);
    this.renderBars(this.origChart, origProbs, this.tokens, origHighlight, origMaxProb);

    var modMaxProb = Math.max(modProbs.reduce(function (a, b) { return Math.max(a, b); }, 0), 0.01);
    this.renderBars(this.modChart, modProbs, this.tokens, modHighlight, modMaxProb);

    var ent = this.entropy(modProbs);
    this.entropyDisplay.textContent = ent.toFixed(2);

    var zeroed = 0;
    for (var zi = 0; zi < modProbs.length; zi++) {
      if (modProbs[zi] < 0.0001) zeroed++;
    }
    this.zerodDisplay.textContent = zeroed;

    this.argmaxDisplay.textContent = this.tokens[modHighlight];
    this.argmaxDisplay.style.color = '#20B2AA';
  };

  function init() {
    var el = document.getElementById('temp-explorer');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new TempExplorer(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
