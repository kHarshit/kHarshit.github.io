(function () {
  'use strict';

  function ClipMatrix(root) {
    this.root = root;
    this.tempSlider = root.querySelector('.clip-temp-slider');
    this.tempDisplay = root.querySelector('.clip-temp-display');
    this.svg = root.querySelector('.clip-matrix-svg');
    this.detailsEl = root.querySelector('.clip-details');
    this.imgLossEl = root.querySelector('.clip-img-loss');
    this.txtLossEl = root.querySelector('.clip-txt-loss');

    this.images = [
      { label: 'A dog in a park', emoji: '🐕' },
      { label: 'A red sports car', emoji: '🏎️' },
      { label: 'Sunset over mountains', emoji: '🌄' },
      { label: 'A cat on a couch', emoji: '🐱' },
      { label: 'A plate of pasta', emoji: '🍝' },
      { label: 'A city skyline at night', emoji: '🌃' },
    ];

    this.texts = [
      'a photo of a dog playing outside',
      'a sports car driving on a road',
      'a scenic mountain sunset view',
      'a cat sleeping on furniture',
      'a delicious pasta dish',
      'a city with tall buildings',
    ];

    this.rawSims = [
      [0.85, 0.12, 0.08, 0.72, 0.05, 0.10],
      [0.10, 0.88, 0.15, 0.08, 0.18, 0.55],
      [0.07, 0.14, 0.90, 0.06, 0.09, 0.20],
      [0.68, 0.06, 0.04, 0.82, 0.03, 0.07],
      [0.04, 0.20, 0.10, 0.05, 0.92, 0.15],
      [0.12, 0.50, 0.22, 0.10, 0.16, 0.86],
    ];

    var self = this;
    this.tempSlider.addEventListener('input', function () { self.update(); });
    this.update();
  }

  ClipMatrix.prototype.computeSoftmax = function (tau) {
    var N = this.images.length;
    var sims = this.rawSims;

    var expSims = sims.map(function (row) {
      return row.map(function (s) { return Math.exp(s / tau); });
    });

    var rowSum = expSims.map(function (row) {
      return row.reduce(function (a, b) { return a + b; }, 0);
    });
    var colSum = [];
    for (var j = 0; j < N; j++) {
      var s = 0;
      for (var i = 0; i < N; i++) s += expSims[i][j];
      colSum.push(s);
    }

    var softmaxRows = expSims.map(function (row, i) {
      return row.map(function (v) { return v / rowSum[i]; });
    });
    var softmaxCols = [];
    for (var i2 = 0; i2 < N; i2++) {
      softmaxCols.push(expSims[i2].map(function (v, j2) { return v / colSum[j2]; }));
    }

    return { softmaxRows: softmaxRows, softmaxCols: softmaxCols };
  };

  ClipMatrix.prototype.update = function () {
    var ns = 'http://www.w3.org/2000/svg';
    var tempVal = parseInt(this.tempSlider.value, 10);
    var tau = tempVal / 100;
    this.tempDisplay.textContent = tau.toFixed(2);

    var N = this.images.length;
    var result = this.computeSoftmax(tau);
    var softmaxRows = result.softmaxRows;
    var softmaxCols = result.softmaxCols;

    this.detailsEl.innerHTML = 'Hover over a cell to see details.';

    var imgLoss = 0, txtLoss = 0;
    for (var k = 0; k < N; k++) {
      imgLoss -= Math.log(Math.max(softmaxRows[k][k], 1e-10));
      txtLoss -= Math.log(Math.max(softmaxCols[k][k], 1e-10));
    }
    imgLoss /= N;
    txtLoss /= N;

    this.imgLossEl.textContent = imgLoss.toFixed(3);
    this.txtLossEl.textContent = txtLoss.toFixed(3);

    var cellSize = 38;
    var margin = { top: 34, left: 80, right: 30, bottom: 40 };
    var w = margin.left + N * cellSize + margin.right;
    var h = margin.top + N * cellSize + margin.bottom;

    this.svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
    this.svg.innerHTML = '';

    var bg = document.createElementNS(ns, 'rect');
    bg.setAttribute('width', '100%'); bg.setAttribute('height', '100%');
    bg.setAttribute('fill', 'var(--bg-color,#f8fafc)'); bg.setAttribute('rx', '8');
    this.svg.appendChild(bg);

    for (var ri = 0; ri < N; ri++) {
      for (var ci = 0; ci < N; ci++) {
        var prob = softmaxRows[ri][ci];
        var intensity = Math.max(0, Math.min(1, prob * 6));
        var r = Math.round(Math.max(0, 255 - intensity * 230));
        var g = Math.round(Math.max(0, 255 - intensity * 100));
        var b = Math.round(Math.max(0, 255 - intensity * 230));
        var x = margin.left + ci * cellSize;
        var y = margin.top + ri * cellSize;

        var rect = document.createElementNS(ns, 'rect');
        rect.setAttribute('x', x); rect.setAttribute('y', y);
        rect.setAttribute('width', cellSize - 2); rect.setAttribute('height', cellSize - 2);
        rect.setAttribute('rx', '4');
        rect.setAttribute('fill', 'rgb(' + r + ',' + g + ',' + b + ')');
        rect.setAttribute('stroke', (ri === ci) ? '#20B2AA' : '#e2e8f0');
        rect.setAttribute('stroke-width', (ri === ci) ? '2.5' : '1');
        rect.setAttribute('class', 'clip-cell');
        rect.setAttribute('data-row', ri);
        rect.setAttribute('data-col', ci);
        rect.style.cursor = 'pointer';

        var text = document.createElementNS(ns, 'text');
        text.setAttribute('x', x + (cellSize - 2) / 2);
        text.setAttribute('y', y + (cellSize - 2) / 2 + 4);
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('font-size', '9');
        text.setAttribute('font-weight', '700');
        text.setAttribute('fill', prob > 0.3 ? '#fff' : '#333');
        text.textContent = (prob * 100).toFixed(0) + '%';

        var g = document.createElementNS(ns, 'g');
        g.appendChild(rect);
        g.appendChild(text);

        var self = this;
        (function (r, c, p) {
          g.addEventListener('mouseenter', function () { self.showDetail(r, c, p); });
        })(ri, ci, prob);

        this.svg.appendChild(g);
      }
    }

    for (var ri2 = 0; ri2 < N; ri2++) {
      var yl = margin.top + ri2 * cellSize + (cellSize - 2) / 2 + 4;
      var label = document.createElementNS(ns, 'text');
      label.setAttribute('x', margin.left - 10); label.setAttribute('y', yl);
      label.setAttribute('text-anchor', 'end'); label.setAttribute('font-size', '22');
      label.setAttribute('fill', 'var(--font-color,#555)');
      label.textContent = this.images[ri2].emoji;
      this.svg.appendChild(label);

      var xl = margin.left + ri2 * cellSize + (cellSize - 2) / 2;
      var clabel = document.createElementNS(ns, 'text');
      clabel.setAttribute('x', xl); clabel.setAttribute('y', margin.top - 8);
      clabel.setAttribute('text-anchor', 'middle'); clabel.setAttribute('font-size', '9');
      clabel.setAttribute('fill', 'var(--font-color,#555)');
      clabel.textContent = 'T' + (ri2 + 1);
      this.svg.appendChild(clabel);
    }

    var xLabel = document.createElementNS(ns, 'text');
    xLabel.setAttribute('x', margin.left + N * cellSize / 2);
    xLabel.setAttribute('y', h - 8);
    xLabel.setAttribute('text-anchor', 'middle');
    xLabel.setAttribute('font-size', '10');
    xLabel.setAttribute('font-weight', '600');
    xLabel.setAttribute('fill', '#888');
    xLabel.textContent = 'Text Embeddings (T1–T6)';
    this.svg.appendChild(xLabel);

    var yLabel = document.createElementNS(ns, 'text');
    yLabel.setAttribute('x', 14);
    yLabel.setAttribute('y', margin.top + N * cellSize / 2);
    yLabel.setAttribute('text-anchor', 'middle');
    yLabel.setAttribute('font-size', '10');
    yLabel.setAttribute('font-weight', '600');
    yLabel.setAttribute('fill', '#888');
    yLabel.setAttribute('transform', 'rotate(-90, 14,' + (margin.top + N * cellSize / 2) + ')');
    yLabel.textContent = 'Image Embeddings';
    this.svg.appendChild(yLabel);
  };

  ClipMatrix.prototype.showDetail = function (row, col, prob) {
    var sim = this.rawSims[row][col];
    var isMatch = row === col;
    this.detailsEl.innerHTML =
      '<div style="margin-bottom:6px;"><strong>Row ' + row + ':</strong> ' + this.images[row].emoji + ' ' + this.images[row].label + '</div>' +
      '<div style="margin-bottom:6px;"><strong>Col ' + col + ':</strong> "' + this.texts[col] + '"</div>' +
      '<div style="margin-top:8px;padding-top:8px;border-top:1px solid #e2e8f0;">' +
      '<div>Raw similarity: <strong>' + (sim * 100).toFixed(0) + '%</strong></div>' +
      '<div>Softmax prob: <strong style="color:' + (isMatch ? '#20B2AA' : '#888') + ';">' + (prob * 100).toFixed(1) + '%</strong></div>' +
      '<div style="font-size:0.78rem;color:#888;margin-top:4px;">' +
      (isMatch
        ? '✓ Correct pair — loss penalizes this if prob < 100%'
        : '✗ Incorrect pair — loss pushes this toward 0%') +
      '</div></div>';
  };

  function init() {
    var el = document.getElementById('clip-matrix');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new ClipMatrix(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
