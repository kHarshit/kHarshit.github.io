(function () {
  'use strict';

  function MaskingViz(root) {
    this.root = root;
    this.tabs = root.querySelectorAll('.masking-tab');
    this.imageGrid = root.querySelector('.masking-image-grid');
    this.textTokens = root.querySelector('.masking-text-tokens');
    this.description = root.querySelector('.masking-description');

    this.tokens = ['A', 'dog', 'is', 'playing', 'in', 'the', 'park', '.'];

    this.modes = {
      mlm: {
        label: 'MLM (Masked Language Modeling)',
        desc: 'Random text tokens are masked with [MASK]. The model learns to predict the original token from surrounding context. Used in BERT and FLAVA\'s text encoder.',
        maskText: [2, 4],
        maskImage: [],
      },
      mim: {
        label: 'MIM (Masked Image Modeling)',
        desc: 'Random image patches are masked (grayed out). The model learns to reconstruct the missing visual content from remaining patches and text. Used in FLAVA\'s image encoder.',
        maskText: [],
        maskImage: [1, 5, 10, 14],
      },
      mmm: {
        label: 'MMM (Masked Multimodal Modeling)',
        desc: 'Both image patches and text tokens are masked simultaneously. The multimodal encoder learns cross-modal relationships by predicting missing content in both modalities. Used in FLAVA\'s multimodal encoder.',
        maskText: [1, 5],
        maskImage: [3, 7, 9, 12],
      },
    };

    var self = this;
    this.tabs.forEach(function (tab) {
      tab.addEventListener('click', function () {
        self.setMode(tab.getAttribute('data-mode'));
      });
    });

    this.setMode('mlm');
  }

  MaskingViz.prototype.setMode = function (mode) {
    this.tabs.forEach(function (tab) {
      var isActive = tab.getAttribute('data-mode') === mode;
      tab.style.background = isActive ? '#fff' : 'transparent';
      tab.style.color = isActive ? '#20B2AA' : '#888';
      tab.style.boxShadow = isActive ? '0 1px 4px rgba(0,0,0,0.1)' : 'none';
    });

    var config = this.modes[mode];
    this.description.innerHTML = '<strong>' + config.label + ':</strong> ' + config.desc;

    this.renderImageGrid(config.maskImage);
    this.renderTextTokens(config.maskText);
  };

  MaskingViz.prototype.renderImageGrid = function (maskedIndices) {
    var total = 16;
    var cols = 4;
    var html = '';
    for (var i = 0; i < total; i++) {
      var isMasked = maskedIndices.indexOf(i) >= 0;
      var row = Math.floor(i / cols);
      var col = i % cols;
      var hue = (i * 37 + 180) % 360;
      html +=
        '<div style="aspect-ratio:1;border-radius:4px;display:flex;align-items:center;justify-content:center;' +
        'font-size:0.85rem;font-weight:700;transition:all 0.25s;' +
        (isMasked
          ? 'background:#e5e7eb;border:1.5px solid #d1d5db;color:#9ca3af;'
          : 'background:hsl(' + hue + ',60%,75%);border:1.5px solid hsl(' + hue + ',50%,65%);color:#fff;') +
        '">' +
        (isMasked ? '[MASK]' : (row * cols + col + 1)) +
        '</div>';
    }
    this.imageGrid.innerHTML = html;
  };

  MaskingViz.prototype.renderTextTokens = function (maskedIndices) {
    var self = this;
    var html = '';
    this.tokens.forEach(function (token, i) {
      var isMasked = maskedIndices.indexOf(i) >= 0;
      html +=
        '<span style="padding:6px 10px;border-radius:6px;font-size:0.85rem;font-weight:600;transition:all 0.25s;' +
        (isMasked
          ? 'background:#e5e7eb;border:1.5px solid #d1d5db;color:#9ca3af;'
          : 'background:var(--bg-color,#fff);border:1.5px solid #20B2AA;color:var(--font-color,#333);') +
        '">' +
        (isMasked ? '[MASK]' : token) +
        '</span>';
    });
    this.textTokens.innerHTML = html;
  };

  function init() {
    var el = document.getElementById('masking-viz');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new MaskingViz(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
