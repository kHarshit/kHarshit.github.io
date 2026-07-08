(function () {
  'use strict';

  function fmtPct(v) { return (v * 100).toFixed(1) + '%'; }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function PipelineViz(root) {
    this.root = root;
    this.stagesSlider = root.querySelector('.pipeline-stages');
    this.microSlider = root.querySelector('.pipeline-micro');
    this.stagesDisplay = root.querySelector('.pipeline-stages-display');
    this.microDisplay = root.querySelector('.pipeline-micro-display');
    this.grid = root.querySelector('.pipeline-grid');
    this.fwdSteps = root.querySelector('.pipeline-fwd-steps');
    this.totalSteps = root.querySelector('.pipeline-total-steps');
    this.bubbleClass = root.querySelector('.pipeline-bubble-pct');

    var self = this;
    function update() { self.render(); }
    this.stagesSlider.addEventListener('input', update);
    this.microSlider.addEventListener('input', update);
    this.render();
  }

  PipelineViz.prototype.render = function () {
    var P = parseInt(this.stagesSlider.value);
    var M = parseInt(this.microSlider.value);
    this.stagesDisplay.textContent = P;
    this.microDisplay.textContent = M;

    var F = P + M - 1;
    var T = 2 * F;
    var activeCells = 2 * P * M;
    var totalCells = P * T;
    var idleCells = totalCells - activeCells;
    var ratio = clamp(idleCells / totalCells, 0, 1);

    this.bubbleClass.textContent = fmtPct(ratio);
    this.bubbleClass.className = 'pipeline-bubble-pct';
    if (ratio > 0.4) this.bubbleClass.classList.add('danger');
    else if (ratio > 0.2) this.bubbleClass.classList.add('warning');
    else this.bubbleClass.classList.add('success');

    this.fwdSteps.textContent = F;
    this.totalSteps.textContent = T;

    this.grid.style.gridTemplateColumns = '18px repeat(' + T + ', 32px)';
    this.grid.style.gridTemplateRows = '18px repeat(' + P + ', 32px)';
    this.grid.innerHTML = '';

    var corner = document.createElement('div');
    corner.style.gridRow = '1';
    corner.style.gridColumn = '1';
    this.grid.appendChild(corner);

    for (var t = 0; t < T; t++) {
      var lbl = document.createElement('div');
      lbl.style.fontSize = '0.6rem';
      lbl.style.color = '#888';
      lbl.style.textAlign = 'center';
      lbl.style.alignSelf = 'end';
      lbl.style.gridRow = '1';
      lbl.style.gridColumn = String(t + 2);
      lbl.textContent = 't' + t;
      this.grid.appendChild(lbl);
    }

    for (var s = 0; s < P; s++) {
      var rowLabel = document.createElement('div');
      rowLabel.style.fontSize = '0.6rem';
      rowLabel.style.color = '#888';
      rowLabel.style.display = 'flex';
      rowLabel.style.alignItems = 'center';
      rowLabel.style.justifyContent = 'center';
      rowLabel.style.gridRow = String(s + 2);
      rowLabel.style.gridColumn = '1';
      rowLabel.textContent = 'S' + s;
      this.grid.appendChild(rowLabel);

      for (var t = 0; t < T; t++) {
        var cell = document.createElement('div');
        cell.className = 'pipeline-cell';
        cell.style.gridRow = String(s + 2);
        cell.style.gridColumn = String(t + 2);

        if (t < F) {
          var mb = t - s;
          if (mb >= 0 && mb < M) {
            cell.classList.add('fwd');
            cell.textContent = 'F' + mb;
          } else {
            cell.classList.add('idle');
          }
        } else {
          var tBwd = t - F;
          var mbBwd = tBwd - (P - 1 - s);
          if (mbBwd >= 0 && mbBwd < M) {
            cell.classList.add('bwd');
            cell.textContent = 'B' + mbBwd;
          } else {
            cell.classList.add('idle');
          }
        }

        this.grid.appendChild(cell);
      }
    }
  };

  function init() {
    var el = document.getElementById('pipeline-viz');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new PipelineViz(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
