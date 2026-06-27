(function () {
  'use strict';

  function BleuCalc(root) {
    this.root = root;
    this.refInput = root.querySelector('.bleu-ref-input');
    this.candInput = root.querySelector('.bleu-cand-input');

    var self = this;
    function update() { self.update(); }
    this.refInput.addEventListener('input', update);
    this.candInput.addEventListener('input', update);
    this.update();
  }

  BleuCalc.prototype.tokenize = function (text) {
    return text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(Boolean);
  };

  BleuCalc.prototype.getNGrams = function (tokens, n) {
    var ngrams = {};
    for (var i = 0; i <= tokens.length - n; i++) {
      var key = tokens.slice(i, i + n).join(' ');
      ngrams[key] = (ngrams[key] || 0) + 1;
    }
    return ngrams;
  };

  BleuCalc.prototype.clippedPrecision = function (candNGrams, refNGrams, total) {
    if (total <= 0) return 0;
    var matches = 0;
    for (var ngram in candNGrams) {
      if (refNGrams[ngram]) {
        matches += Math.min(candNGrams[ngram], refNGrams[ngram]);
      }
    }
    return matches / total;
  };

  BleuCalc.prototype.update = function () {
    var refText = this.refInput.value || '';
    var candText = this.candInput.value || '';

    var refTokens = this.tokenize(refText);
    var candTokens = this.tokenize(candText);

    var candLenEl = this.root.querySelector('.bleu-cand-len');
    var refLenEl = this.root.querySelector('.bleu-ref-len');
    if (candLenEl) candLenEl.textContent = candTokens.length;
    if (refLenEl) refLenEl.textContent = refTokens.length;

    if (candTokens.length === 0 || refTokens.length === 0) {
      for (var i = 1; i <= 4; i++) {
        var f = this.root.querySelector('.bleu-p' + i + '-fill');
        var v = this.root.querySelector('.bleu-p' + i + '-value');
        if (f) f.style.width = '0%';
        if (v) v.textContent = '0.0%';
      }
      var bpEl = this.root.querySelector('.bleu-bp-value');
      if (bpEl) bpEl.textContent = '1.000';
      var scoreEl = this.root.querySelector('.bleu-score-value');
      if (scoreEl) scoreEl.textContent = '0.000';
      return;
    }

    var rawPrecisions = [];
    var smoothPrecisions = [];
    for (var n = 1; n <= 4; n++) {
      if (candTokens.length < n || refTokens.length < n) {
        rawPrecisions.push(0);
        smoothPrecisions.push(0);
        continue;
      }
      var candNGrams = this.getNGrams(candTokens, n);
      var refNGrams = this.getNGrams(refTokens, n);
      var total = candTokens.length - n + 1;
      var raw = this.clippedPrecision(candNGrams, refNGrams, total);
      rawPrecisions.push(raw);
      var matches = raw * total;
      smoothPrecisions.push((matches + 1) / (total + 1));
    }

    for (var i = 0; i < 4; i++) {
      var fill = this.root.querySelector('.bleu-p' + (i + 1) + '-fill');
      var val = this.root.querySelector('.bleu-p' + (i + 1) + '-value');
      var displayPct = smoothPrecisions[i] * 100;
      if (fill) fill.style.width = Math.max(displayPct, 0.5) + '%';
      if (val) {
        var rawPct = rawPrecisions[i] * 100;
        val.textContent = displayPct.toFixed(1) + '% (raw ' + rawPct.toFixed(1) + '%)';
      }
    }

    var c = candTokens.length;
    var r = refTokens.length;
    var bp = c > r ? 1 : Math.exp(1 - r / c);
    var bpEl = this.root.querySelector('.bleu-bp-value');
    if (bpEl) bpEl.textContent = bp.toFixed(3);

    var bleu = 0;
    var allNonZero = smoothPrecisions.every(function (p) { return p > 0; });
    if (allNonZero) {
      var logSum = 0;
      for (var i = 0; i < 4; i++) {
        logSum += Math.log(smoothPrecisions[i]);
      }
      var avgLog = logSum / 4;
      bleu = bp * Math.exp(avgLog);
    }

    var scoreEl = this.root.querySelector('.bleu-score-value');
    if (scoreEl) scoreEl.textContent = bleu.toFixed(3);
  };

  function init() {
    var el = document.getElementById('bleu-calc');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new BleuCalc(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
