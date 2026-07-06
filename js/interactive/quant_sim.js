(function () {
  'use strict';

  function QuantSim(root) {
    this.root = root;
    this.inputEl = root.querySelector('.quant-input');
    this.modeEl = root.querySelector('.quant-mode');
    this.bitsSlider = root.querySelector('.quant-bits-slider');
    this.bitsDisplay = root.querySelector('.quant-bits-display');
    this.tbody = root.querySelector('.quant-tbody');
    this.avgErrorEl = root.querySelector('.quant-avg-error');
    this.maxErrorEl = root.querySelector('.quant-max-error');
    this.scaleInfoEl = root.querySelector('.quant-scale-info');
    this.zpInfoEl = root.querySelector('.quant-zp-info');

    var self = this;
    this.inputEl.addEventListener('input', function () { self.update(); });
    this.modeEl.addEventListener('change', function () { self.update(); });
    this.bitsSlider.addEventListener('input', function () { self.update(); });
    this.update();
  }

  QuantSim.prototype.update = function () {
    var inputStr = this.inputEl.value;
    var nums = inputStr.split(',').map(function (s) {
      return parseFloat(s.trim());
    }).filter(function (v) { return !isNaN(v); });

    if (nums.length === 0) {
      this.tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;padding:20px;color:#888;">Enter comma-separated numbers above.</td></tr>';
      this.avgErrorEl.textContent = '—';
      this.maxErrorEl.textContent = '—';
      this.scaleInfoEl.textContent = '—';
      this.zpInfoEl.textContent = '—';
      return;
    }

    var mode = this.modeEl.value;
    var bits = parseInt(this.bitsSlider.value, 10);
    this.bitsDisplay.textContent = bits;

    var min = Math.min.apply(null, nums);
    var max = Math.max.apply(null, nums);
    var alpha, beta, alpha_q, beta_q;

    if (mode === 'asymmetric') {
      alpha_q = 0;
      beta_q = Math.pow(2, bits) - 1;
      alpha = min;
      beta = max;
    } else if (mode === 'symmetric-unsigned') {
      var absMax = Math.max(Math.abs(min), Math.abs(max));
      alpha = -absMax;
      beta = absMax;
      alpha_q = 0;
      beta_q = Math.pow(2, bits) - 1;
    } else if (mode === 'symmetric-signed') {
      var absMax2 = Math.max(Math.abs(min), Math.abs(max));
      alpha = -absMax2;
      beta = absMax2;
      alpha_q = -Math.pow(2, bits - 1);
      beta_q = Math.pow(2, bits - 1) - 1;
    } else { // restricted
      var absMax3 = Math.max(Math.abs(min), Math.abs(max));
      alpha = -absMax3;
      beta = absMax3;
      alpha_q = -(Math.pow(2, bits - 1) - 1);
      beta_q = Math.pow(2, bits - 1) - 1;
    }

    var scale = (beta - alpha) / (beta_q - alpha_q);
    var zeroPoint = mode === 'asymmetric'
      ? Math.round(alpha_q - alpha / scale)
      : 0;

    // clamp zeroPoint
    if (zeroPoint < alpha_q) zeroPoint = alpha_q;
    if (zeroPoint > beta_q) zeroPoint = beta_q;

    var totalError = 0;
    var maxError = 0;
    var rows = '';

    nums.forEach(function (v) {
      var clipped = Math.max(alpha, Math.min(beta, v));
      var quantized = Math.round(clipped / scale + zeroPoint);
      if (quantized < alpha_q) quantized = alpha_q;
      if (quantized > beta_q) quantized = beta_q;
      var dequant = (quantized - zeroPoint) * scale;
      var error = v - dequant;
      var absErr = Math.abs(error);
      totalError += absErr;
      if (absErr > maxError) maxError = absErr;

      var bgColor = absErr > scale * 0.5 ? '#fee2e2' : (absErr > scale * 0.25 ? '#fef9c3' : '#ecfdf5');
      rows += '<tr style="background:' + bgColor + ';">' +
        '<td style="padding:4px 8px;text-align:right;border-bottom:1px solid #e2e8f0;">' + v.toFixed(4) + '</td>' +
        '<td style="padding:4px 8px;text-align:right;border-bottom:1px solid #e2e8f0;">' + (clipped / scale + zeroPoint).toFixed(2) + '</td>' +
        '<td style="padding:4px 8px;text-align:right;border-bottom:1px solid #e2e8f0;font-weight:700;">' + quantized + '</td>' +
        '<td style="padding:4px 8px;text-align:right;border-bottom:1px solid #e2e8f0;">' + dequant.toFixed(4) + '</td>' +
        '<td style="padding:4px 8px;text-align:right;border-bottom:1px solid #e2e8f0;color:' + (absErr > scale * 0.5 ? '#dc2626' : (absErr > scale * 0.25 ? '#d97706' : '#16a34a')) + ';">' + error.toFixed(4) + '</td>' +
        '</tr>';
    });

    this.tbody.innerHTML = rows;
    this.avgErrorEl.textContent = (totalError / nums.length).toFixed(4);
    this.maxErrorEl.textContent = maxError.toFixed(4);
    this.scaleInfoEl.innerHTML = scale.toFixed(5) + '<span style="display:block;font-size:0.6rem;font-weight:400;color:#888;">(' + beta.toFixed(2) + '−' + alpha.toFixed(2) + ')/(' + beta_q + '−' + alpha_q + ')</span>';
    this.zpInfoEl.innerHTML = zeroPoint + '<span style="display:block;font-size:0.6rem;font-weight:400;color:#888;">round(' + alpha_q + ' − ' + alpha.toFixed(2) + '/' + scale.toFixed(5) + ')</span>';
  };

  function init() {
    var el = document.getElementById('quant-sim');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new QuantSim(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
