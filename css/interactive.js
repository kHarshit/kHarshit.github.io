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
  //  Inspired by ciechanow.ski animated SVG diagrams —
  //  particle animation showing gradient synchronization.
  // ===========================================================
  function AllReduceViz(root) {
    this.root = root;
    this.container = root.querySelector('.ar-container');
    this.playBtn = root.querySelector('.ar-play-btn');
    this.resetBtn = root.querySelector('.ar-reset-btn');
    this.stepLabel = root.querySelector('.ar-step-label');
    this.animId = null;
    this.running = false;
    this.step = 0;
    this.maxSteps = 5;

    var self = this;
    this.playBtn.addEventListener('click', function () {
      if (self.running) {
        self.stop();
      } else {
        self.play();
      }
    });
    this.resetBtn.addEventListener('click', function () {
      self.stop();
      self.step = 0;
      self.renderStep();
    });
    this.renderStep();
  }

  AllReduceViz.prototype.getNodePos = function (node) {
    var rect = node.getBoundingClientRect();
    var cRect = this.container.getBoundingClientRect();
    return {
      x: rect.left + rect.width / 2 - cRect.left,
      y: rect.top + rect.height / 2 - cRect.top
    };
  };

  AllReduceViz.prototype.play = function () {
    this.running = true;
    this.playBtn.textContent = 'Pause';
    this.advance();
  };

  AllReduceViz.prototype.stop = function () {
    this.running = false;
    this.playBtn.textContent = 'Play';
    if (this.animId) {
      clearTimeout(this.animId);
      this.animId = null;
    }
  };

  AllReduceViz.prototype.advance = function () {
    var self = this;
    this.step++;
    if (this.step > this.maxSteps) {
      this.step = 0;
    }
    this.renderStep();
    if (this.running) {
      this.animId = setTimeout(function () { self.advance(); }, 1500);
    }
  };

  AllReduceViz.prototype.renderStep = function () {
    var nodes = this.container.querySelectorAll('.ar-node');
    var container = this.container;

    // Remove old particles
    container.querySelectorAll('.ar-particle').forEach(function (p) { p.remove(); });

    var self = this;
    var labels = [
      'Step ' + this.step + ': each GPU computes local gradients',
      'Step ' + this.step + ': GPUs exchange gradient shards',
      'Step ' + this.step + ': GPUs reduce received gradients',
      'Step ' + this.step + ': Scatter reduced shards back',
      'Step ' + this.step + ': All GPUs have synchronized gradients'
    ];

    var stepLabels = [
      'Local gradients computed on each GPU',
      'Gradient shards exchanged between neighbors',
      'Reduction (sum) of received shards',
      'Reduced shards scattered to all GPUs',
      'All GPUs have identical averaged gradients'
    ];

    var s = Math.min(this.step, this.maxSteps);
    this.stepLabel.textContent = stepLabels[s - 1] || 'Ready';

    // Highlight nodes based on step
    nodes.forEach(function (n) {
      n.classList.remove('active');
      if (self.step > 0) {
        n.classList.add('active');
      }
    });

    // Create particles between nodes
    if (this.step >= 2 && this.step <= 4) {
      var nodeList = Array.from(nodes);
      for (var i = 0; i < nodeList.length; i++) {
        var target = (i + 1) % nodeList.length;
        var p1 = this.getNodePos(nodeList[i]);
        var p2 = this.getNodePos(nodeList[target]);

        for (var k = 0; k < 3; k++) {
          var particle = document.createElement('div');
          particle.className = 'ar-particle visible';
          var frac = (k + 1) / 4;
          particle.style.left = (p1.x + (p2.x - p1.x) * frac - 4) + 'px';
          particle.style.top = (p1.y + (p2.y - p1.y) * frac - 4) + 'px';
          particle.style.animationDelay = (k * 0.3) + 's';
          container.appendChild(particle);
        }
      }
    }
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
        new w.ctor(el);
      }
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initWidgets);
  } else {
    initWidgets();
  }

})();
