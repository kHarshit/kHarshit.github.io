(function () {
  'use strict';

  function CudaTiledAnim(root) {
    this.root = root;
    this.phaseEl = root.querySelector('.cta-phase');
    this.stepEl = root.querySelector('.cta-step');
    this.stepBar = root.querySelector('.cta-step-bar');
    this.svgA = root.querySelector('.cta-matrix-a');
    this.svgB = root.querySelector('.cta-matrix-b');
    this.svgC = root.querySelector('.cta-matrix-c');
    this.svgSharedA = root.querySelector('.cta-shared-a');
    this.svgSharedB = root.querySelector('.cta-shared-b');
    this.nextBtn = root.querySelector('.cta-next-btn');
    this.backBtn = root.querySelector('.cta-back-btn');
    this.resetBtn = root.querySelector('.cta-reset-btn');

    this.T = 4;
    this.M = 6; this.N = 6; this.K = 6;
    this.numTiles = Math.ceil(this.K / this.T);
    this.totalSteps = this.numTiles * 2 + 2;

    this.history = [];
    this.currentStep = -1;

    this.matA = this.randomMat(this.M, this.K);
    this.matB = this.randomMat(this.K, this.N);
    this.originalA = this.cloneMat(this.matA);
    this.originalB = this.cloneMat(this.matB);

    this.renderDots();

    var self = this;
    this.nextBtn.addEventListener('click', function () { self.next(); });
    this.backBtn.addEventListener('click', function () { self.back(); });
    this.resetBtn.addEventListener('click', function () { self.reset(); });

    this.reset();
  }

  CudaTiledAnim.prototype.renderDots = function () {
    this.stepBar.innerHTML = '';
    for (var i = 0; i < this.totalSteps; i++) {
      var dot = document.createElement('div');
      dot.className = 'ar-step-dot';
      dot.style.width = '10px';
      dot.style.height = '10px';
      dot.style.borderRadius = '50%';
      dot.style.background = '#e2e8f0';
      dot.style.transition = 'background 0.3s, transform 0.3s';
      dot.style.flexShrink = '0';
      this.stepBar.appendChild(dot);
    }
  };

  CudaTiledAnim.prototype.updateDots = function (step) {
    var dots = this.stepBar.querySelectorAll('.ar-step-dot');
    for (var i = 0; i < dots.length; i++) {
      dots[i].style.background = i < step ? '#20B2AA' : (i === step ? '#20B2AA' : '#e2e8f0');
      dots[i].style.transform = i === step ? 'scale(1.5)' : 'scale(1)';
      if (i === step) dots[i].style.boxShadow = '0 0 5px rgba(32,178,170,0.5)';
      else dots[i].style.boxShadow = 'none';
    }
  };

  CudaTiledAnim.prototype.cloneMat = function (m) {
    return m.map(function (r) { return r.slice(); });
  };

  CudaTiledAnim.prototype.randomMat = function (r, c) {
    var m = [];
    for (var i = 0; i < r; i++) {
      m[i] = [];
      for (var j = 0; j < c; j++) m[i][j] = Math.floor(Math.random() * 5) + 1;
    }
    return m;
  };

  CudaTiledAnim.prototype.zeroMat = function (r, c) {
    var m = [];
    for (var i = 0; i < r; i++) {
      m[i] = [];
      for (var j = 0; j < c; j++) m[i][j] = 0;
    }
    return m;
  };

  CudaTiledAnim.prototype.reset = function () {
    this.matA = this.cloneMat(this.originalA);
    this.matB = this.cloneMat(this.originalB);
    this.matC = this.zeroMat(this.M, this.N);
    this.tileA = this.zeroMat(this.T, this.T);
    this.tileB = this.zeroMat(this.T, this.T);
    this.tileIdx = 0;
    this.subStep = -1;
    this.history = [];
    this.currentStep = -1;

    this.saveSnapshot();

    this.backBtn.disabled = true;
    this.nextBtn.disabled = false;
    this.nextBtn.textContent = 'Next';
    this.renderFromSnapshot(this.history[0]);
    this.updateDots(0);
  };

  CudaTiledAnim.prototype.saveSnapshot = function () {
    this.history = this.history.slice(0, this.currentStep + 1);
    this.history.push({
      matC: this.cloneMat(this.matC),
      tileA: this.cloneMat(this.tileA),
      tileB: this.cloneMat(this.tileB),
      tileIdx: this.tileIdx !== undefined ? this.tileIdx : 0,
      subStep: this.subStep !== undefined ? this.subStep : 0
    });
    this.currentStep++;
  };

  CudaTiledAnim.prototype.restoreSnapshot = function (snap) {
    this.matC = this.cloneMat(snap.matC);
    this.tileA = this.cloneMat(snap.tileA);
    this.tileB = this.cloneMat(snap.tileB);
    this.tileIdx = snap.tileIdx;
    this.subStep = snap.subStep;
  };

  CudaTiledAnim.prototype.next = function () {
    var snap = this.history[this.currentStep];
    if (!snap) { this.nextStep(); return; }
    this.restoreSnapshot(snap);

    if (this.currentStep >= this.history.length - 1) {
      this.nextStep();
    } else {
      this.currentStep++;
      this.renderFromSnapshot(this.history[this.currentStep]);
    }

    this.updateDots(this.currentStep);
    this.backBtn.disabled = this.currentStep <= 0;
    this.updateNextButton();
  };

  CudaTiledAnim.prototype.updateNextButton = function () {
    var snap = this.history[this.currentStep];
    if (snap && snap.tileIdx >= this.numTiles) {
      this.nextBtn.disabled = true;
      this.nextBtn.textContent = 'Complete';
    } else {
      this.nextBtn.disabled = false;
      this.nextBtn.textContent = 'Next';
    }
  };

  CudaTiledAnim.prototype.back = function () {
    if (this.currentStep <= 0) return;
    this.currentStep--;
    this.renderFromSnapshot(this.history[this.currentStep]);

    this.updateDots(this.currentStep);
    this.backBtn.disabled = this.currentStep <= 0;
    this.nextBtn.disabled = false;
    this.nextBtn.textContent = 'Next';
  };

  CudaTiledAnim.prototype.renderFromSnapshot = function (snap) {
    this.restoreSnapshot(snap);
    var kStart = this.tileIdx * this.T;
    var numTiles = this.numTiles;
    var stepInTile = this.tileIdx * 2 + (this.subStep === 0 ? 0 : 1);

    if (this.subStep === -1) {
      this.phaseEl.textContent = 'Ready';
      this.stepEl.textContent = 'Initial state: shared memory tiles are empty. Press Next to begin.';
      this.renderMatrix(this.svgA, this.matA, 6, 6, '#10b981', null, null, null, null);
      this.renderMatrix(this.svgB, this.matB, 6, 6, '#6366f1', null, null, null, null);
      this.renderMatrix(this.svgC, this.matC, 6, 6, '#20B2AA', null, null, null, null);
      this.renderMatrix(this.svgSharedA, this.tileA, 4, 4, '#10b981', null, null, null, null);
      this.renderMatrix(this.svgSharedB, this.tileB, 4, 4, '#6366f1', null, null, null, null);
      return;
    }

    if (this.tileIdx >= numTiles) {
      this.phaseEl.textContent = 'Done! Csub = Σ A_tile × B_tile';
      this.stepEl.textContent = 'Only this block\u2019s 4\u00d74 tile region of C is computed (rows 0\u20133, cols 0\u20133). Other blocks handle the rest.';
      this.renderMatrix(this.svgA, this.matA, 6, 6, '#10b981', null, null, null, null);
      this.renderMatrix(this.svgB, this.matB, 6, 6, '#6366f1', null, null, null, null);
      this.renderMatrix(this.svgC, this.matC, 6, 6, '#20B2AA', null, null, null, null);
      this.renderMatrix(this.svgSharedA, this.tileA, 4, 4, '#10b981', null, null, null, null);
      this.renderMatrix(this.svgSharedB, this.tileB, 4, 4, '#6366f1', null, null, null, null);
      return;
    }

    if (this.subStep === 0) {
      this.phaseEl.textContent = '\uD83D\uDD04 Load tile ' + (this.tileIdx + 1) + '/' + numTiles;
      this.stepEl.textContent = 'Cooperative load: each thread copies one element to shared memory';
      this.renderMatrix(this.svgA, this.matA, 6, 6, '#10b981', null, null, 0, kStart);
      this.renderMatrix(this.svgB, this.matB, 6, 6, '#6366f1', null, null, kStart, 0);
      this.renderMatrix(this.svgC, this.matC, 6, 6, '#20B2AA', null, null, null, null);
      this.renderMatrix(this.svgSharedA, this.tileA, 4, 4, '#10b981', null, null, null, null);
      this.renderMatrix(this.svgSharedB, this.tileB, 4, 4, '#6366f1', null, null, null, null);
    } else {
      this.phaseEl.textContent = '\uD83E\uDDEE Compute tile ' + (this.tileIdx + 1) + '/' + numTiles;
      this.stepEl.textContent = '__syncthreads() done \u2192 Csub += A_tile \u00d7 B_tile (accumulate) \u2192 __syncthreads() for next load';
      this.renderMatrix(this.svgA, this.matA, 6, 6, '#10b981', null, null, 0, kStart);
      this.renderMatrix(this.svgB, this.matB, 6, 6, '#6366f1', null, null, kStart, 0);
      this.renderMatrix(this.svgC, this.matC, 6, 6, '#20B2AA', null, null, null, null);
      this.renderMatrix(this.svgSharedA, this.tileA, 4, 4, '#10b981', null, null, null, null);
      this.renderMatrix(this.svgSharedB, this.tileB, 4, 4, '#6366f1', null, null, null, null);
    }
  };

  CudaTiledAnim.prototype.loadTiles = function (kStart) {
    for (var i = 0; i < this.T; i++) {
      for (var j = 0; j < this.T; j++) {
        this.tileA[i][j] = (i < this.M && kStart + j < this.K) ? this.matA[i][kStart + j] : 0;
        this.tileB[i][j] = (kStart + i < this.K && j < this.N) ? this.matB[kStart + i][j] : 0;
      }
    }
  };

  CudaTiledAnim.prototype.computePartial = function (kStart) {
    for (var r2 = 0; r2 < Math.min(this.T, this.M); r2++) {
      for (var c2 = 0; c2 < Math.min(this.T, this.N); c2++) {
        for (var k2 = 0; k2 < this.T; k2++) {
          if (r2 < this.M && c2 < this.N && kStart + k2 < this.K) {
            this.matC[r2][c2] += this.tileA[r2][k2] * this.tileB[k2][c2];
          }
        }
      }
    }
  };

  CudaTiledAnim.prototype.nextStep = function () {
    var numTiles = this.numTiles;
    var kStart = this.tileIdx * this.T;

    if (this.tileIdx >= numTiles) return;

    if (this.subStep === -1) {
      this.loadTiles(kStart);
      this.subStep = 0;
      this.saveSnapshot();
      this.renderFromSnapshot(this.history[this.currentStep]);
    } else if (this.subStep === 0) {
      this.computePartial(kStart);
      this.subStep = 1;
      this.saveSnapshot();
      this.renderFromSnapshot(this.history[this.currentStep]);
    } else {
      this.tileIdx++;
      var nextK = this.tileIdx * this.T;
      if (this.tileIdx < numTiles) {
        this.loadTiles(nextK);
      }
      this.subStep = 0;
      this.saveSnapshot();
      this.renderFromSnapshot(this.history[this.currentStep]);
    }

    this.backBtn.disabled = this.currentStep <= 0;
    this.updateNextButton();
  };

  CudaTiledAnim.prototype.renderAll = function () {
    this.renderMatrix(this.svgA, this.matA, 6, 6, '#10b981', null, null, null, null);
    this.renderMatrix(this.svgB, this.matB, 6, 6, '#6366f1', null, null, null, null);
    this.renderMatrix(this.svgC, this.matC, 6, 6, '#20B2AA', null, null, null, null);
    this.renderMatrix(this.svgSharedA, this.tileA, 4, 4, '#10b981', null, null, null, null);
    this.renderMatrix(this.svgSharedB, this.tileB, 4, 4, '#6366f1', null, null, null, null);
  };

  CudaTiledAnim.prototype.renderMatrix = function (svg, data, rows, cols, color, highlightR, highlightC, tileR, tileC) {
    var ns = 'http://www.w3.org/2000/svg';
    svg.innerHTML = '';

    var cellSize = rows <= 4 ? 26 : 24;
    var gap = 2;
    var totalW = cols * (cellSize + gap);
    var totalH = rows * (cellSize + gap);
    svg.setAttribute('viewBox', '0 0 ' + totalW + ' ' + totalH);

    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        var x = c * (cellSize + gap);
        var y = r * (cellSize + gap);
        var val = data[r] !== undefined && data[r][c] !== undefined ? data[r][c] : null;

        var isTile = tileR !== null && r >= tileR && r < tileR + 4 && c >= tileC && c < tileC + 4;
        var isHL = highlightR !== null && r === highlightR && c === highlightC;

        var rect = document.createElementNS(ns, 'rect');
        rect.setAttribute('x', x); rect.setAttribute('y', y);
        rect.setAttribute('width', cellSize); rect.setAttribute('height', cellSize);
        rect.setAttribute('rx', '3');
        if (isHL) {
          rect.setAttribute('fill', '#fbbf24'); rect.setAttribute('fill-opacity', '0.8');
          rect.setAttribute('stroke', '#f59e0b'); rect.setAttribute('stroke-width', '2');
        } else if (isTile) {
          rect.setAttribute('fill', color); rect.setAttribute('fill-opacity', '0.4');
          rect.setAttribute('stroke', color); rect.setAttribute('stroke-width', '1.5');
        } else if (val === 0) {
          rect.setAttribute('fill', '#f1f5f9'); rect.setAttribute('fill-opacity', '1');
          rect.setAttribute('stroke', '#e2e8f0'); rect.setAttribute('stroke-width', '0.5');
        } else {
          rect.setAttribute('fill', color); rect.setAttribute('fill-opacity', '0.25');
          rect.setAttribute('stroke', color); rect.setAttribute('stroke-width', '0.5');
        }
        svg.appendChild(rect);

        if (val !== null) {
          var text = document.createElementNS(ns, 'text');
          text.setAttribute('x', x + cellSize / 2);
          text.setAttribute('y', y + cellSize / 2 + 3);
          text.setAttribute('text-anchor', 'middle');
          text.setAttribute('font-size', cellSize >= 24 ? '10' : '8');
          text.setAttribute('font-weight', '700');
          text.setAttribute('fill', isHL ? '#333' : (val === 0 ? '#ccc' : '#333'));
          text.textContent = val;
          svg.appendChild(text);
        }
      }
    }

    if (tileR !== null && rows >= 6) {
      var bx = tileC * (cellSize + gap);
      var by = tileR * (cellSize + gap);
      var bw = 4 * (cellSize + gap) - gap;
      var bh = 4 * (cellSize + gap) - gap;
      var border = document.createElementNS(ns, 'rect');
      border.setAttribute('x', bx); border.setAttribute('y', by);
      border.setAttribute('width', bw); border.setAttribute('height', bh);
      border.setAttribute('fill', 'none');
      border.setAttribute('stroke', '#fbbf24');
      border.setAttribute('stroke-width', '2');
      border.setAttribute('stroke-dasharray', '5,3');
      svg.appendChild(border);
    }
  };

  function init() {
    var el = document.getElementById('cuda-tiled-anim');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new CudaTiledAnim(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
