/* ===========================================================
   Interactive Components for Distributed Training Article
   Inspired by ciechanow.ski interactive visualizations
   =========================================================== */
(function () {
  'use strict';

  // ---- Helpers ----
  function fmtGB(v) {
    if (v >= 1000) return (v / 1000).toFixed(1) + ' TB';
    if (v >= 1) return v.toFixed(1) + ' GB';
    return (v * 1024).toFixed(0) + ' MB';
  }

  function fmtPct(v) { return (v * 100).toFixed(1) + '%'; }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  // ===========================================================
  // 1. MEMORY CALCULATOR
  //  Inspired by ciechanow.ski color picker sliders — real-time
  //  parameter adjustments with immediate visual feedback.
  // ===========================================================
  function MemoryCalculator(root) {
    this.root = root;
    this.slider = root.querySelector('.mem-slider');
    this.precision = root.querySelector('.mem-precision');
    this.update();

    var self = this;
    this.slider.addEventListener('input', function () { self.update(); });
    this.precision.addEventListener('change', function () { self.update(); });
  }

  MemoryCalculator.prototype.getConfig = function () {
    var paramsB = parseFloat(this.slider.value);
    var prec = this.precision.value;
    var bytes, actMultiplier;
    if (prec === 'fp32') {
      bytes = 4;
      actMultiplier = 4;
    } else if (prec === 'bf16') {
      bytes = 2;
      actMultiplier = 2;
    } else {
      bytes = 1;
      actMultiplier = 1;
    }
    var params = paramsB * 1e9;
    var master = (prec === 'fp32') ? 0 : 4;
    var optMult = (prec === 'int8') ? 2 : 4;
    return {
      paramsB: paramsB,
      bytes: bytes,
      paramsMem: params * bytes / 1e9,
      gradsMem: params * bytes / 1e9,
      optMem: params * 2 * optMult / 1e9,
      masterMem: params * master / 1e9,
      actMem: params * actMultiplier / 1e9 * 0.3,
      tempMem: paramsB * 0.8,
      label: prec.toUpperCase()
    };
  };

  MemoryCalculator.prototype.update = function () {
    var c = this.getConfig();
    var total = c.paramsMem + c.gradsMem + c.optMem + c.masterMem + c.actMem + c.tempMem;

    // Try both querySelector (for new code) and direct element reference (for robustness)
    function qs(el, sel) { try { return el.querySelector(sel); } catch(e) { return null; } }

    var pd = qs(this.root, '.mem-params-display');
    if (pd) pd.textContent = c.paramsB >= 1000 ? (c.paramsB / 1000).toFixed(1) + 'T' : c.paramsB + 'B';

    var pl = qs(this.root, '.mem-prec-label');
    if (pl) pl.textContent = c.label;

    var memValues = [c.paramsMem, c.gradsMem, c.optMem, c.masterMem, c.actMem, c.tempMem];
    var maxMem = 10000; // 10 TB fixed reference — consistent across all precision settings

    var fills = this.root.querySelectorAll('.mem-calc-bar-fill');
    var vals = this.root.querySelectorAll('.mem-calc-bar-val');
    for (var i = 0; i < memValues.length; i++) {
      if (i < fills.length) {
        var pct = (memValues[i] / maxMem) * 100;
        fills[i].style.width = Math.max(pct, 0.5) + '%';
      }
      if (i < vals.length) {
        vals[i].textContent = memValues[i] > 0 ? fmtGB(memValues[i]) : '-';
      }
    }

    var tv = qs(this.root, '.mem-total-value');
    if (tv) tv.textContent = fmtGB(total);

    var sp = qs(this.root, '.mem-stat-params');
    if (sp) sp.textContent = c.paramsB >= 1000 ? (c.paramsB / 1000).toFixed(1) + 'T' : c.paramsB + 'B';

    var sg = qs(this.root, '.mem-stat-gpus');
    if (sg) sg.textContent = Math.ceil(total / 80) + ' × A100';

    var spr = qs(this.root, '.mem-stat-prec');
    if (spr) spr.textContent = c.label;
  };

  // ===========================================================
  // 2. PIPELINE BUBBLE VISUALIZER
  //  Inspired by ciechanow.ski 3D cube visualization —
  //  interactive grid showing pipeline schedule and bubble ratio.
  // ===========================================================
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

    // Grid: 18px label col + T cols of 32px; 18px label row + P rows of 32px
    this.grid.style.gridTemplateColumns = '18px repeat(' + T + ', 32px)';
    this.grid.style.gridTemplateRows = '18px repeat(' + P + ', 32px)';
    this.grid.innerHTML = '';

    // Top-left corner spacer
    var corner = document.createElement('div');
    corner.style.gridRow = '1';
    corner.style.gridColumn = '1';
    this.grid.appendChild(corner);

    // Time-step labels (top row)
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

    // Stage labels + cells
    for (var s = 0; s < P; s++) {
      // Stage label
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
          // Forward pass: stage s processes micro-batch (t - s)
          var mb = t - s;
          if (mb >= 0 && mb < M) {
            cell.classList.add('fwd');
            cell.textContent = 'F' + mb;
          } else {
            cell.classList.add('idle');
          }
        } else {
          // Backward pass
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

  // ===========================================================
  // 3. ZERO STAGE COMPARISON
  //  Inspired by ciechanow.ski side-by-side color plates —
  //  toggle between ZeRO stages to see memory distribution.
  // ===========================================================
  function ZeroComparison(root) {
    this.root = root;
    this.tabs = root.querySelectorAll('.zero-stage-tab');
    this.activeStage = 0;
    this.update();

    var self = this;
    this.tabs.forEach(function (tab) {
      tab.addEventListener('click', function () {
        self.tabs.forEach(function (t) { t.classList.remove('active'); });
        tab.classList.add('active');
        self.activeStage = parseInt(tab.getAttribute('data-stage'));
        self.update();
      });
    });
  }

  ZeroComparison.prototype.update = function () {
    var stage = this.activeStage;
    var totalParams = 10; // 10B model
    var gpuCount = 8;
    var bytes = 2; // BF16

    // Memory components for a 10B model (in GB)
    var paramsMem = totalParams * bytes;
    var gradsMem = totalParams * bytes;
    var optMem = totalParams * 2 * 4; // Adam: momentum + variance in FP32
    var masterMem = totalParams * 4;

    var segments, paramsKept, gradsKept, optKept, masterKept, total;

    switch (stage) {
      case 0: // DDP
        paramsKept = paramsMem;
        gradsKept = gradsMem;
        optKept = optMem;
        masterKept = masterMem;
        break;
      case 1: // ZeRO-1
        paramsKept = paramsMem;
        gradsKept = gradsMem;
        optKept = optMem / gpuCount;
        masterKept = masterMem / gpuCount;
        break;
      case 2: // ZeRO-2
        paramsKept = paramsMem;
        gradsKept = gradsMem / gpuCount;
        optKept = optMem / gpuCount;
        masterKept = masterMem / gpuCount;
        break;
      case 3: // ZeRO-3
        paramsKept = paramsMem / gpuCount;
        gradsKept = gradsMem / gpuCount;
        optKept = optMem / gpuCount;
        masterKept = masterMem / gpuCount;
        break;
    }

    total = paramsKept + gradsKept + optKept + masterKept;
    var maxTotal = paramsMem + gradsMem + optMem + masterMem;
    var savings = 1 - total / maxTotal;

    // Build the bar
    var barStack = this.root.querySelector('.zero-bar-stack');
    var topVal = Math.max(maxTotal, 1);
    var pctH = clamp(total / topVal * 100, 2, 100);
    barStack.style.height = pctH + '%';

    var comps = [
      { cls: 'param', val: paramsKept, label: 'Params' },
      { cls: 'grad', val: gradsKept, label: 'Grads' },
      { cls: 'opt', val: optKept, label: 'Opt States' },
      { cls: 'master', val: masterKept, label: 'Master W' }
    ];

    barStack.innerHTML = '';
    var totalH = comps.reduce(function (s, c) { return s + c.val; }, 0);
    comps.forEach(function (c) {
      if (c.val <= 0) return;
      var seg = document.createElement('div');
      seg.className = 'zero-bar-segment ' + c.cls;
      var segPct = (c.val / totalH) * 100;
      seg.style.height = segPct + '%';
      seg.setAttribute('title', c.label + ': ' + fmtGB(c.val));
      // Show GB label inside segment if it's tall enough
      if (segPct > 8) {
        var lbl = document.createElement('span');
        lbl.className = 'zero-bar-seg-label';
        lbl.textContent = fmtGB(c.val);
        seg.appendChild(lbl);
      }
      barStack.appendChild(seg);
    });

    this.root.querySelector('.zero-total-value').textContent = fmtGB(total);

    // Update individual values
    var names = ['Params', 'Grads', 'Opt States', 'Master W'];
    var vals = [paramsKept, gradsKept, optKept, masterKept];
    var items = this.root.querySelectorAll('.zero-savings-item');
    items.forEach(function (item, i) {
      if (i < names.length) {
        item.querySelector('.zero-savings-label').textContent = names[i];
        item.querySelector('.zero-savings-value').textContent = fmtGB(vals[i]);
      }
    });

    // Communication volume (per GPU per step)
    var modelBytes = totalParams * bytes; // 20 GB for 10B BF16
    var N = gpuCount;
    var allreduceVol = 2 * (N - 1) / N * modelBytes;
    var reduceScatterVol = (N - 1) / N * modelBytes;
    var allgatherVol = (N - 1) / N * modelBytes;

    var commVol;
    switch (stage) {
      case 0: commVol = allreduceVol; break;                     // DDP: AllReduce grads
      case 1: commVol = reduceScatterVol + allgatherVol; break;  // ZeRO-1: ReduceScatter(grads) + AllGather(params)
      case 2: commVol = reduceScatterVol + allgatherVol; break;  // ZeRO-2: ReduceScatter(grads) + AllGather(params)
      case 3: commVol = 2 * allgatherVol + reduceScatterVol; break; // ZeRO-3: AllGather×2 + ReduceScatter
    }

    // Find max comm vol across all stages for bar scaling
    var commVols = [
      allreduceVol,                                            // DDP
      reduceScatterVol + allgatherVol,                         // ZeRO-1
      reduceScatterVol + allgatherVol,                         // ZeRO-2
      2 * allgatherVol + reduceScatterVol                      // ZeRO-3
    ];
    var maxComm = Math.max.apply(null, commVols);

    var commFill = this.root.querySelector('.zero-comm-fill');
    var commValue = this.root.querySelector('.zero-comm-value');
    if (commFill) commFill.style.width = Math.max(commVol / maxComm * 100, 2) + '%';
    if (commValue) commValue.textContent = fmtGB(commVol);

    // Relative ratio compared to DDP
    var ratio = commVol / commVols[0];
    var ratioFill = this.root.querySelector('.zero-comm-ratio');
    var ratioValue = this.root.querySelector('.zero-comm-ratio-value');
    if (ratioFill) ratioFill.style.width = Math.min(ratio / 2 * 100, 100) + '%';
    if (ratioValue) ratioValue.textContent = ratio.toFixed(1) + 'x';
  };

  // ===========================================================
  // 4. ALLREDUCE ANIMATION
  //  Manual step-through showing gradient shards moving
  //  across GPUs in a ring AllReduce algorithm.
  // ===========================================================
  // ===========================================================
  // 4. RING ALLREDUCE VISUALIZATION
  //  Concrete example: 4 GPUs × 4 values, 3+3 steps, N-1 chunks.
  //  Reduce-scatter: chunks circulate clockwise, accumulating.
  //  All-gather: fully-reduced chunks spread to all GPUs.
  // ===========================================================
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

    // Full state: [step][gpu][chunk]
    this.states = [
      // Step 0: initial
      [[  1,   2,   3,   4],
       [ 10,  20,  30,  40],
       [100, 200, 300, 400],
       [1000,2000,3000,4000]],

      // Step 1: RS round 0 — send diag (i-1 mod 4)
      [[  1,   2,3003,   4],
       [ 10,  20,  30,  44],
       [110, 200, 300, 400],
       [1000,2200,3000,4000]],

      // Step 2: RS round 1 — send diag (i-2 mod 4)
      [[  1,2202,3003,   4],
       [ 10,  20,3033,  44],
       [110, 200, 300, 444],
       [1110,2200,3000,4000]],

      // Step 3: RS round 2 — send diag (i-3 mod 4)
      [[1111,2202,3003,   4],
       [ 10,2222,3033,  44],
       [110, 200,3333, 444],
       [1110,2200,3000,4444]],

      // Step 4: AG round 0 — send own reduced chunk
      [[1111,2202,3003,4444],
       [1111,2222,3033,  44],
       [110, 2222,3333, 444],
       [1110,2200,3333,4444]],

      // Step 5: AG round 1 — forward received chunk
      [[1111,2202,3333,4444],
       [1111,2222,3033,4444],
       [1111,2222,3333, 444],
       [1110,2222,3333,4444]],

      // Step 6: AG round 2 — forward again
      [[1111,2222,3333,4444],
       [1111,2222,3333,4444],
       [1111,2222,3333,4444],
       [1111,2222,3333,4444]],

      // Step 7: done (same as step 6, just for display)
      [[1111,2222,3333,4444],
       [1111,2222,3333,4444],
       [1111,2222,3333,4444],
       [1111,2222,3333,4444]]
    ];

    // Origin GPU for each initial value: [gpu][chunk]
    // Used to color original (unaccumulated) values
    this.originGpu = [
      [0,0,0,0],
      [1,1,1,1],
      [2,2,2,2],
      [3,3,3,3]
    ];

    // Original values lookup
    this.originals = {};
    for (var g = 0; g < 4; g++) {
      for (var c = 0; c < 4; c++) {
        this.originals[this.states[0][g][c]] = g;
      }
    }

    // Movement groups per step: [srcGpu, chunkIdx, outgoingLabel]
    this.movements = {
      // RS rounds — send clockwise
      1: [[0,3,'4'],[1,0,'10'],[2,1,'200'],[3,2,'3000']],
      2: [[0,2,'3003'],[1,3,'44'],[2,0,'110'],[3,1,'2200']],
      3: [[0,1,'2202'],[1,2,'3033'],[2,3,'444'],[3,0,'1110']],
      // AG rounds — send counter-clockwise
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
      '✓ Complete: all 4 GPUs have the same result [1111, 2222, 3333, 4444].'
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

    // Update dots
    if (this.stepDots) {
      for (var d = 0; d < this.stepDots.length; d++) {
        this.stepDots[d].classList.toggle('done', d < this.step);
        this.stepDots[d].classList.toggle('current', d === this.step);
      }
    }

    // Remove old SVG
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

        // Determine color: original, accumulated, or fully-reduced sum
        var color = self.getValueColor(val, gpuIdx, c);
        slot.style.background = color;

        // Highlight chunk being sent
        var moves = self.movements[self.step];
        if (moves) {
          for (var m = 0; m < moves.length; m++) {
            if (moves[m][0] === gpuIdx && moves[m][1] === c) {
              slot.classList.add('ar-slot-sending');
            }
          }
        }

        // Highlight chunk being received
        if (moves && (self.step === 1 || self.step === 2 || self.step === 3)) {
          // In RS, receiving GPU is (gpuIdx + 3) % 4, at position = what was sent
          for (var m = 0; m < moves.length; m++) {
            var sender = moves[m][0];
            var recvChunk = moves[m][1];
            if (gpuIdx === (sender + 1) % 4 && c === recvChunk) {
              slot.classList.add('ar-slot-receiving');
            }
          }
        }
        if (moves && (self.step >= 4 && self.step <= 6)) {
          // In AG, receiving GPU is (gpuIdx + 1) % 4, at position = what was sent
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
    // Check if it's a fully-reduced sum
    if (val === 1111 || val === 2222 || val === 3333 || val === 4444) {
      // Check if this is the GPU's OWN reduced chunk (stayed in place) or a received one
      var fullSums = {0:1111, 1:2222, 2:3333, 3:4444};
      return this.sumColor;
    }
    // Check if it's an original value
    if (this.originals[val] !== undefined) {
      // But also check it's not accumulated — original values are only in step 0 and unchanged positions
      var initialState = this.states[0];
      if (initialState[gpuIdx][chunkIdx] === val) {
        return this.originColors[gpuIdx];
      }
      // But this value originated from another GPU and is still in its original form
      // (e.g., value 4 is from GPU0 pos3, sitting on GPU0 pos3 in step 1)
      // Check if it's original for the current position
      if (val === initialState[this.originals[val]][chunkIdx] && this.originals[val] === gpuIdx) {
        // Hmm, this is getting complex. Let me simplify.
      }
    }
    // Accumulated or non-original
    // If the value appears in the initial state at this position for this GPU, it's original
    if (this.states[0][gpuIdx][chunkIdx] === val) {
      return this.originColors[gpuIdx];
    }
    // Otherwise it's accumulated
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

      // Determine if connection is horizontal (same row) or vertical (same column)
      var srcCx = srcR.left + srcR.width / 2;
      var srcCy = srcR.top + srcR.height / 2;
      var tgtCx = tgtR.left + tgtR.width / 2;
      var tgtCy = tgtR.top + tgtR.height / 2;
      var dx = tgtCx - srcCx;
      var dy = tgtCy - srcCy;

      var x1, y1, x2, y2;
      if (Math.abs(dx) > Math.abs(dy)) {
        // Horizontal — exit/enter left/right edges, y follows chunk
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
        // Vertical — exit/enter top/bottom edges, x follows chunk
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

      // Line
      var line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', x1);
      line.setAttribute('y1', y1);
      line.setAttribute('x2', x2);
      line.setAttribute('y2', y2);
      line.setAttribute('stroke', '#20B2AA');
      line.setAttribute('stroke-width', '2');
      line.setAttribute('stroke-dasharray', '5,3');
      svg.appendChild(line);

      // Arrowhead
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

        // Label
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

  // ===========================================================
  // Initialization — handles both DCL and late script loads
  // ===========================================================
  function initWidgets() {
    // Debug: confirm script execution
    var debugEl = document.getElementById('dt-debug');
    if (debugEl) debugEl.textContent = 'interactive.js loaded OK';

    var widgets = [
      { id: 'mem-calc', ctor: MemoryCalculator },
      { id: 'pipeline-viz', ctor: PipelineViz },
      { id: 'zero-compare', ctor: ZeroComparison },
      { id: 'ar-viz', ctor: AllReduceViz }
    ];
    widgets.forEach(function (w) {
      var el = document.getElementById(w.id);
      if (el && !el.hasAttribute('data-dt-init')) {
        el.setAttribute('data-dt-init', '1');
        try {
          new w.ctor(el);
        } catch (e) {
          console.error('DT widget error [' + w.id + ']:', e);
        }
      }
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initWidgets);
  } else {
    initWidgets();
  }

})();
