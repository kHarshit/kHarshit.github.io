(function () {
  'use strict';

  function AllReduceViz(root) {
    this.root = root;
    this.container = root.querySelector('.ar-container');
    this.slotsContainers = root.querySelectorAll('.ar-slots');
    this.nextBtn = root.querySelector('.ar-next-btn');
    this.backBtn = root.querySelector('.ar-back-btn');
    this.resetBtn = root.querySelector('.ar-reset-btn');
    this.stepLabel = root.querySelector('.ar-step-label');
    this.stepBar = root.querySelector('.ar-step-bar');
    this.step = 0;
    this.maxSteps = 7;

    this.states = [
      [[  1,   2,   3,   4],
       [ 10,  20,  30,  40],
       [100, 200, 300, 400],
       [1000,2000,3000,4000]],
      [[  1,   2,3003,   4],
       [ 10,  20,  30,  44],
       [110, 200, 300, 400],
       [1000,2200,3000,4000]],
      [[  1,2202,3003,   4],
       [ 10,  20,3033,  44],
       [110, 200, 300, 444],
       [1110,2200,3000,4000]],
      [[1111,2202,3003,   4],
       [ 10,2222,3033,  44],
       [110, 200,3333, 444],
       [1110,2200,3000,4444]],
      [[1111,2202,3003,4444],
       [1111,2222,3033,  44],
       [110, 2222,3333, 444],
       [1110,2200,3333,4444]],
      [[1111,2202,3333,4444],
       [1111,2222,3033,4444],
       [1111,2222,3333, 444],
       [1110,2222,3333,4444]],
      [[1111,2222,3333,4444],
       [1111,2222,3333,4444],
       [1111,2222,3333,4444],
       [1111,2222,3333,4444]],
      [[1111,2222,3333,4444],
       [1111,2222,3333,4444],
       [1111,2222,3333,4444],
       [1111,2222,3333,4444]]
    ];

    this.originGpu = [
      [0,0,0,0],
      [1,1,1,1],
      [2,2,2,2],
      [3,3,3,3]
    ];

    this.originals = {};
    for (var g = 0; g < 4; g++) {
      for (var c = 0; c < 4; c++) {
        this.originals[this.states[0][g][c]] = g;
      }
    }

    this.movements = {
      1: [[0,3,'4'],[1,0,'10'],[2,1,'200'],[3,2,'3000']],
      2: [[0,2,'3003'],[1,3,'44'],[2,0,'110'],[3,1,'2200']],
      3: [[0,1,'2202'],[1,2,'3033'],[2,3,'444'],[3,0,'1110']],
      4: [[0,0,'1111'],[1,1,'2222'],[2,2,'3333'],[3,3,'4444']],
      5: [[0,3,'4444'],[1,0,'1111'],[2,1,'2222'],[3,2,'3333']],
      6: [[0,2,'3333'],[1,3,'4444'],[2,0,'1111'],[3,1,'2222']]
    };

    this.labels = [
      'Step 0: Each GPU starts with its own vector of 4 numbers.',
      'Step 1: Reduce-Scatter round 1: send chunk 3 (GPU0), chunk 0 (GPU1), chunk 1 (GPU2), chunk 2 (GPU3) clockwise, accumulate.',
      'Step 2: Reduce-Scatter round 2: forward accumulated chunks clockwise, accumulate further.',
      'Step 3: Reduce-Scatter complete: each GPU now has one fully-reduced chunk.',
      'Step 4: All-Gather round 1: each GPU sends its reduced chunk counter-clockwise.',
      'Step 5: All-Gather round 2: forward received chunks counter-clockwise.',
      'Step 6: All-Gather round 3: forward again, all GPUs now have every reduced chunk.',
      '\u2713 Complete: all 4 GPUs have the same result [1111, 2222, 3333, 4444].'
    ];

    this.originColors = ['#6366f1','#8b5cf6','#ec4899','#f59e0b'];
    this.sumColor = '#10b981';
    this.accumColor = '#0891b2';

    this.buildDots();
    this.renderStep();

    var self = this;
    this.nextBtn.addEventListener('click', function () {
      if (self.step < self.maxSteps) { self.step++; self.renderStep(); }
    });
    this.backBtn.addEventListener('click', function () {
      if (self.step > 0) { self.step--; self.renderStep(); }
    });
    this.resetBtn.addEventListener('click', function () {
      self.step = 0; self.renderStep();
    });
  }

  AllReduceViz.prototype.buildDots = function () {
    this.stepBar.innerHTML = '';
    for (var i = 0; i <= this.maxSteps; i++) {
      var dot = document.createElement('span');
      dot.className = 'ar-step-dot';
      dot.setAttribute('data-idx', i);
      this.stepBar.appendChild(dot);
    }
    this.stepDots = this.stepBar.querySelectorAll('.ar-step-dot');
  };

  AllReduceViz.prototype.renderStep = function () {
    var container = this.container;

    this.stepLabel.textContent = this.labels[this.step] || this.labels[0];
    this.backBtn.disabled = (this.step <= 0);
    this.nextBtn.disabled = (this.step >= this.maxSteps);
    this.nextBtn.textContent = (this.step >= this.maxSteps) ? 'Done' : 'Next';

    if (this.stepDots) {
      for (var d = 0; d < this.stepDots.length; d++) {
        this.stepDots[d].classList.toggle('done', d < this.step);
        this.stepDots[d].classList.toggle('current', d === this.step);
      }
    }

    container.querySelectorAll('.ar-lines').forEach(function (el) { el.remove(); });

    this.renderCards();
    this.drawMovements();
  };

  AllReduceViz.prototype.renderCards = function () {
    var state = this.states[this.step];
    var self = this;
    var container = this.container;

    for (var gpuIdx = 0; gpuIdx < 4; gpuIdx++) {
      var slots = container.querySelector('.ar-card[data-gpu="' + gpuIdx + '"] .ar-slots');
      if (!slots) continue;
      slots.innerHTML = '';
      for (var c = 0; c < 4; c++) {
        var val = state[gpuIdx][c];
        var slot = document.createElement('span');
        slot.className = 'ar-slot';
        slot.textContent = val;

        var color = self.getValueColor(val, gpuIdx, c);
        slot.style.background = color;

        var moves = self.movements[self.step];
        if (moves) {
          for (var m = 0; m < moves.length; m++) {
            if (moves[m][0] === gpuIdx && moves[m][1] === c) {
              slot.classList.add('ar-slot-sending');
            }
          }
        }

        if (moves && (self.step === 1 || self.step === 2 || self.step === 3)) {
          for (var m = 0; m < moves.length; m++) {
            var sender = moves[m][0];
            var recvChunk = moves[m][1];
            if (gpuIdx === (sender + 1) % 4 && c === recvChunk) {
              slot.classList.add('ar-slot-receiving');
            }
          }
        }
        if (moves && (self.step >= 4 && self.step <= 6)) {
          for (var m = 0; m < moves.length; m++) {
            var sender = moves[m][0];
            var recvChunk = moves[m][1];
            if (gpuIdx === (sender + 3) % 4 && c === recvChunk) {
              slot.classList.add('ar-slot-receiving');
            }
          }
        }

        slots.appendChild(slot);
      }
    }
  };

  AllReduceViz.prototype.getValueColor = function (val, gpuIdx, chunkIdx) {
    if (val === 1111 || val === 2222 || val === 3333 || val === 4444) {
      return this.sumColor;
    }
    if (this.originals[val] !== undefined) {
      var initialState = this.states[0];
      if (initialState[gpuIdx][chunkIdx] === val) {
        return this.originColors[gpuIdx];
      }
      if (val === initialState[this.originals[val]][chunkIdx] && this.originals[val] === gpuIdx) {
      }
    }
    if (this.states[0][gpuIdx][chunkIdx] === val) {
      return this.originColors[gpuIdx];
    }
    return this.accumColor;
  };

  AllReduceViz.prototype.drawMovements = function () {
    var moves = this.movements[this.step];
    if (!moves) return;

    var container = this.container;
    if (!container.querySelectorAll('.ar-card').length) return;

    var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'ar-lines');
    svg.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:5;';

    var cr = container.getBoundingClientRect();
    var isGather = (this.step >= 4 && this.step <= 6);

    for (var m = 0; m < moves.length; m++) {
      var srcIdx = moves[m][0];
      var chunkIdx = moves[m][1];
      var label = moves[m][2];
      var tgtIdx = isGather ? (srcIdx + 3) % 4 : (srcIdx + 1) % 4;

      var srcCard = container.querySelector('.ar-card[data-gpu="' + srcIdx + '"]');
      var tgtCard = container.querySelector('.ar-card[data-gpu="' + tgtIdx + '"]');
      if (!srcCard || !tgtCard) continue;

      var srcR = srcCard.getBoundingClientRect();
      var tgtR = tgtCard.getBoundingClientRect();

      var srcSlots = srcCard.querySelector('.ar-slots');
      var tgtSlots = tgtCard.querySelector('.ar-slots');
      var srcSlotEls = srcSlots ? srcSlots.querySelectorAll('.ar-slot') : [];
      var tgtSlotEls = tgtSlots ? tgtSlots.querySelectorAll('.ar-slot') : [];
      var srcSlotR = srcSlotEls[chunkIdx] ? srcSlotEls[chunkIdx].getBoundingClientRect() : null;
      var tgtSlotR = tgtSlotEls[chunkIdx] ? tgtSlotEls[chunkIdx].getBoundingClientRect() : null;

      var srcCx = srcR.left + srcR.width / 2;
      var srcCy = srcR.top + srcR.height / 2;
      var tgtCx = tgtR.left + tgtR.width / 2;
      var tgtCy = tgtR.top + tgtR.height / 2;
      var dx = tgtCx - srcCx;
      var dy = tgtCy - srcCy;

      var x1, y1, x2, y2;
      if (Math.abs(dx) > Math.abs(dy)) {
        y1 = srcSlotR ? srcSlotR.top + srcSlotR.height / 2 - cr.top : srcR.top + srcR.height / 2 - cr.top;
        y2 = tgtSlotR ? tgtSlotR.top + tgtSlotR.height / 2 - cr.top : tgtR.top + tgtR.height / 2 - cr.top;
        if (dx > 0) {
          x1 = srcR.right - cr.left;
          x2 = tgtR.left - cr.left;
        } else {
          x1 = srcR.left - cr.left;
          x2 = tgtR.right - cr.left;
        }
      } else {
        x1 = srcSlotR ? srcSlotR.left + srcSlotR.width / 2 - cr.left : srcR.left + srcR.width / 2 - cr.left;
        x2 = tgtSlotR ? tgtSlotR.left + tgtSlotR.width / 2 - cr.left : tgtR.left + tgtR.width / 2 - cr.left;
        if (dy > 0) {
          y1 = srcR.bottom - cr.top;
          y2 = tgtR.top - cr.top;
        } else {
          y1 = srcR.top - cr.top;
          y2 = tgtR.bottom - cr.top;
        }
      }

      var line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', x1);
      line.setAttribute('y1', y1);
      line.setAttribute('x2', x2);
      line.setAttribute('y2', y2);
      line.setAttribute('stroke', '#20B2AA');
      line.setAttribute('stroke-width', '2');
      line.setAttribute('stroke-dasharray', '5,3');
      svg.appendChild(line);

      var dx = x2 - x1;
      var dy = y2 - y1;
      var len = Math.sqrt(dx * dx + dy * dy);
      if (len > 0) {
        var ux = dx / len;
        var uy = dy / len;
        var tipX = x2 - ux * 8;
        var tipY = y2 - uy * 8;
        var px = -uy * 5;
        var py = ux * 5;
        var arrow = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        arrow.setAttribute('points',
          (tipX) + ',' + (tipY) + ' ' +
          (tipX - ux * 10 + px) + ',' + (tipY - uy * 10 + py) + ' ' +
          (tipX - ux * 10 - px) + ',' + (tipY - uy * 10 - py)
        );
        arrow.setAttribute('fill', '#20B2AA');
        svg.appendChild(arrow);

        var lx = (x1 + x2) / 2;
        var ly = (y1 + y2) / 2 - 5;
        var labelEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        labelEl.setAttribute('x', lx);
        labelEl.setAttribute('y', ly);
        labelEl.setAttribute('text-anchor', 'middle');
        labelEl.setAttribute('fill', '#0d9488');
        labelEl.setAttribute('font-size', '10');
        labelEl.setAttribute('font-weight', '700');
        labelEl.textContent = label;
        svg.appendChild(labelEl);
      }
    }

    container.appendChild(svg);
  };

  function init() {
    var el = document.getElementById('ar-viz');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new AllReduceViz(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
