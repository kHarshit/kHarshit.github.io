(function () {
  'use strict';

  function CudaBlockMapper(root) {
    this.root = root;
    this.blockXEl = root.querySelector('.cbm-block-x');
    this.blockYEl = root.querySelector('.cbm-block-y');
    this.svg = root.querySelector('.cbm-grid-svg');
    this.infoEl = root.querySelector('.cbm-thread-info');

    this.matrixSize = 32;
    this.selectedRow = null;
    this.selectedCol = null;
    this.selectedBX = null;
    this.selectedBY = null;

    var self = this;
    this.blockXEl.addEventListener('change', function () { self.selectedRow = self.selectedCol = self.selectedBX = self.selectedBY = null; self.infoEl.innerHTML = 'Click a cell to see thread details.'; self.update(); });
    this.blockYEl.addEventListener('change', function () { self.selectedRow = self.selectedCol = self.selectedBX = self.selectedBY = null; self.infoEl.innerHTML = 'Click a cell to see thread details.'; self.update(); });
    this.update();
  }

  CudaBlockMapper.prototype.update = function () {
    var bx = parseInt(this.blockXEl.value, 10);
    var by = parseInt(this.blockYEl.value, 10);
    var N = this.matrixSize;
    var gx = Math.ceil(N / bx);
    var gy = Math.ceil(N / by);

    this.renderGrid(N, bx, by, gx, gy);
  };

  CudaBlockMapper.prototype.renderGrid = function (N, bx, by, gx, gy) {
    var ns = 'http://www.w3.org/2000/svg';
    var svg = this.svg;
    svg.innerHTML = '';

    var cellSize = Math.min(26, Math.floor(520 / N));
    var gap = 1;
    var total = N * cellSize + (N - 1) * gap;
    svg.setAttribute('viewBox', '0 0 ' + Math.max(total + 40, 300) + ' ' + Math.max(total + 50, 300));
    svg.setAttribute('height', Math.max(total + 50, 300));

    var w = parseInt(svg.getAttribute('viewBox').split(' ')[2]);
    var h = parseInt(svg.getAttribute('viewBox').split(' ')[3]);

    var self = this;

    for (var r = 0; r < N; r++) {
      for (var c = 0; c < N; c++) {
        var x = c * (cellSize + gap) + 20;
        var y = r * (cellSize + gap) + 20;

        var blockX = Math.floor(c / bx);
        var blockY = Math.floor(r / by);
        var tx = c % bx;
        var ty = r % by;

        var isHighlightBlock = self.selectedBX !== null && blockX === self.selectedBX && blockY === self.selectedBY;
        var isSelected = self.selectedRow === r && self.selectedCol === c;
        var hue = ((blockX + blockY * gx) * 37) % 360;

        var rect = document.createElementNS(ns, 'rect');
        rect.setAttribute('x', x); rect.setAttribute('y', y);
        rect.setAttribute('width', cellSize); rect.setAttribute('height', cellSize);
        rect.setAttribute('rx', '2');
        if (isSelected) {
          rect.setAttribute('fill', '#fbbf24');
          rect.setAttribute('fill-opacity', '0.9');
          rect.setAttribute('stroke', '#f59e0b');
          rect.setAttribute('stroke-width', '2.5');
        } else if (isHighlightBlock) {
          rect.setAttribute('fill', '#20B2AA');
          rect.setAttribute('fill-opacity', '0.6');
          rect.setAttribute('stroke', '#0d9488');
          rect.setAttribute('stroke-width', '1.5');
        } else {
          rect.setAttribute('fill', 'hsl(' + hue + ',40%,75%)');
          rect.setAttribute('fill-opacity', '0.3');
          rect.setAttribute('stroke', '#d1d5db');
          rect.setAttribute('stroke-width', '0.5');
        }
        rect.setAttribute('data-row', r);
        rect.setAttribute('data-col', c);
        rect.setAttribute('data-tx', tx);
        rect.setAttribute('data-ty', ty);
        rect.setAttribute('data-block-x', blockX);
        rect.setAttribute('data-block-y', blockY);
        rect.style.cursor = 'pointer';

        (function (r2, c2, tx2, ty2, bx2, by2) {
          rect.addEventListener('click', function () {
            self.selectedRow = r2;
            self.selectedCol = c2;
            self.selectedBX = bx2;
            self.selectedBY = by2;
            self.update();
            self.showInfo(r2, c2, tx2, ty2, bx2, by2, N, bx, by);
          });
        })(r, c, tx, ty, blockX, blockY);

        svg.appendChild(rect);

        if (cellSize >= 20 && isHighlightBlock) {
          var text = document.createElementNS(ns, 'text');
          text.setAttribute('x', x + cellSize / 2);
          text.setAttribute('y', y + cellSize / 2 + 3);
          text.setAttribute('text-anchor', 'middle');
          text.setAttribute('font-size', cellSize >= 24 ? '7' : '5');
          text.setAttribute('fill', isSelected ? '#333' : '#fff');
          text.setAttribute('font-weight', '600');
          text.textContent = '(' + tx + ',' + ty + ')';
          svg.appendChild(text);
        }
      }
    }

    if (self.selectedBX !== null) {
      var bx0 = self.selectedBX * bx * (cellSize + gap) + 20;
      var by0 = self.selectedBY * by * (cellSize + gap) + 20;
      var bw2 = bx * (cellSize + gap) - gap;
      var bh2 = by * (cellSize + gap) - gap;
      var border = document.createElementNS(ns, 'rect');
      border.setAttribute('x', bx0); border.setAttribute('y', by0);
      border.setAttribute('width', bw2); border.setAttribute('height', bh2);
      border.setAttribute('fill', 'none');
      border.setAttribute('stroke', '#20B2AA');
      border.setAttribute('stroke-width', '2.5');
      border.setAttribute('stroke-dasharray', '6,3');
      svg.appendChild(border);

      var blabel = document.createElementNS(ns, 'text');
      blabel.setAttribute('x', bx0 + bw2 / 2); blabel.setAttribute('y', by0 - 4);
      blabel.setAttribute('text-anchor', 'middle'); blabel.setAttribute('font-size', '8');
      blabel.setAttribute('fill', '#20B2AA'); blabel.setAttribute('font-weight', '700');
      blabel.textContent = 'blockIdx(' + self.selectedBX + ',' + self.selectedBY + ') — ' + bx + '×' + by + ' threads';
      svg.appendChild(blabel);
    }
    svg.appendChild(blabel);

    // Axis labels
    var xLabel = document.createElementNS(ns, 'text');
    xLabel.setAttribute('x', 20 + N * (cellSize + gap) / 2);
    xLabel.setAttribute('y', h - 6);
    xLabel.setAttribute('text-anchor', 'middle');
    xLabel.setAttribute('font-size', '9');
    xLabel.setAttribute('fill', '#888');
    xLabel.textContent = 'Output column (C)';
    svg.appendChild(xLabel);

    var yLabel = document.createElementNS(ns, 'text');
    yLabel.setAttribute('x', 8);
    yLabel.setAttribute('y', 20 + N * (cellSize + gap) / 2);
    yLabel.setAttribute('text-anchor', 'middle');
    yLabel.setAttribute('font-size', '9');
    yLabel.setAttribute('fill', '#888');
    yLabel.setAttribute('transform', 'rotate(-90, 8,' + (20 + N * (cellSize + gap) / 2) + ')');
    yLabel.textContent = 'Output row (C)';
    svg.appendChild(yLabel);
  };

  CudaBlockMapper.prototype.showInfo = function (row, col, tx, ty, bx, by, N, bdx, bdy) {
    var globalRow = by * bdy + ty;
    var globalCol = bx * bdx + tx;
    this.infoEl.innerHTML =
      '<div style="margin-bottom:6px;"><strong>Output element:</strong> C[' + row + '][' + col + ']</div>' +
      '<div style="margin-bottom:6px;"><strong>Block:</strong> (' + bx + ',' + by + ')</div>' +
      '<div style="margin-bottom:6px;"><strong>threadIdx:</strong> (' + tx + ', ' + ty + ')</div>' +
      '<div style="margin-bottom:6px;padding:6px 0;border-top:1px solid #e2e8f0;border-bottom:1px solid #e2e8f0;">' +
      '<code>row = ' + by + ' × ' + bdy + ' + ' + ty + ' = ' + globalRow + '</code><br>' +
      '<code>col = ' + bx + ' × ' + bdx + ' + ' + tx + ' = ' + globalCol + '</code>' +
      '</div>';
  };

  function init() {
    var el = document.getElementById('cuda-block-mapper');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new CudaBlockMapper(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
