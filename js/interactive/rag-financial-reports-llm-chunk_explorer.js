(function () {
  'use strict';

  function ChunkExplorer(root) {
    this.root = root;
    this.sizeSlider = root.querySelector('.chunk-size-slider');
    this.overlapSlider = root.querySelector('.chunk-overlap-slider');
    this.sizeDisplay = root.querySelector('.chunk-size-display');
    this.overlapDisplay = root.querySelector('.chunk-overlap-display');
    this.textDisplay = root.querySelector('.chunk-text-display');
    this.statsEl = root.querySelector('.chunk-stats');
    this.barsEl = root.querySelector('.chunk-bars');

    this.sampleText =
      'Tesla, Inc. (NASDAQ: TSLA) today announced financial results for the quarter ended September 30, 2023. ' +
      'Total revenue increased 9% year-over-year to $23.4 billion, driven by growth in vehicle deliveries ' +
      'and a 40% increase in energy generation and storage revenue. Automotive revenue reached $19.6 billion, up 5% ' +
      'compared to $18.7 billion in the prior-year period. The company delivered 435,000 vehicles during the quarter, ' +
      'despite planned downtime for factory upgrades at several facilities. Gross margin declined to 17.9% from 25.1% ' +
      'in Q3 2022, reflecting the impact of pricing reductions across the vehicle lineup and higher raw material costs. ' +
      'Operating income decreased to $1.8 billion, with operating margin falling to 7.6% from 17.2% in the same ' +
      'quarter last year. Net income attributable to common stockholders was $1.9 billion, or $0.53 per diluted share, ' +
      'compared to $3.3 billion, or $0.95 per diluted share, in the prior-year period. Cash and cash equivalents ' +
      'increased to $22.6 billion, providing strong liquidity for future capital investments. Free cash flow was ' +
      '$0.8 billion, impacted by higher capital expenditures and working capital investments. The company reaffirmed ' +
      'its full-year delivery guidance of approximately 1.8 million vehicles and highlighted the upcoming launch of ' +
      'the Cybertruck, with initial deliveries expected before the end of the year. Energy storage deployments grew ' +
      'to 4.0 GWh, with Megapack factory ramp progressing as planned. Tesla continued to invest in its global ' +
      'manufacturing footprint, with the Gigafactory in Mexico breaking ground during the quarter. Research and ' +
      'development expenses totaled $1.2 billion, reflecting investments in autonomous driving, artificial ' +
      'intelligence, and next-generation vehicle platforms. Automotive regulatory credit revenue was $554 million ' +
      'in the quarter. On the operational front, Tesla expanded its Supercharger network to over 55,000 global ' +
      'connectors and continued rolling out Full Self-Driving (FSD) Beta software to eligible customers in North ' +
      'America. The company ended the quarter with approximately 128,000 full-time employees. Capital expenditures ' +
      'for the quarter totaled $2.5 billion, allocated to capacity expansion at existing factories and tooling for ' +
      'the Cybertruck and next-generation vehicle programs. Management highlighted progress on cost reduction ' +
      'initiatives including 4680 battery cell production ramp at Gigafactory Texas and improvements in ' +
      'vehicle manufacturing efficiency. The company also reported that its insurance business grew to over ' +
      '300,000 policies in force. Tesla Energy gross margin improved sequentially, and management expressed ' +
      'confidence that the energy storage business would continue to grow at a faster rate than the automotive ' +
      'segment. Looking ahead, Tesla guided for production volume growth of approximately 50% in 2024, though ' +
      'management noted that the rate of growth would depend on macroeconomic conditions, supply chain stability, ' +
      'and the successful ramp of new production facilities. The company continues to evaluate additional ' +
      'factory locations to support long-term demand growth and reduce logistics costs across global markets.';

    this.colors = [
      '#20B2AA', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6',
      '#ec4899', '#06b6d4', '#d946ef', '#10b981', '#6366f1',
    ];

    var self = this;
    this.sizeSlider.addEventListener('input', function () { self.update(); });
    this.overlapSlider.addEventListener('input', function () { self.update(); });
    this.update();
  }

  ChunkExplorer.prototype.getChunks = function () {
    var size = parseInt(this.sizeSlider.value, 10);
    var overlap = parseInt(this.overlapSlider.value, 10);
    var step = Math.max(size - overlap, 1);
    var text = this.sampleText;
    var chunks = [];
    var start = 0;
    while (start < text.length) {
      var end = Math.min(start + size, text.length);
      chunks.push({ start: start, end: end, text: text.slice(start, end) });
      if (end >= text.length) break;
      start += step;
    }
    return chunks;
  };

  ChunkExplorer.prototype.hasHeaderSplit = function (chunks) {
    var patterns = [
      'Total revenue', 'Net income', 'Gross margin', 'Operating income',
      'Automotive revenue', 'Free cash flow', 'Research and development',
    ];
    for (var i = 0; i < chunks.length; i++) {
      var chunk = chunks[i];
      for (var p = 0; p < patterns.length; p++) {
        var pat = patterns[p];
        var idx = chunk.text.indexOf(pat);
        if (idx >= 0) {
          var globalIdx = chunk.start + idx;
          var nearStart = (globalIdx - chunk.start) < 8;
          var nearEnd = (globalIdx + pat.length) > (chunk.end - 8);
          if (nearStart || nearEnd) return true;
        }
      }
    }
    return false;
  };

  ChunkExplorer.prototype.update = function () {
    var size = parseInt(this.sizeSlider.value, 10);
    this.overlapSlider.max = size;
    var overlap = Math.min(parseInt(this.overlapSlider.value, 10), size);

    this.sizeDisplay.textContent = size;
    this.overlapDisplay.textContent = overlap;

    var chunks = this.getChunks();
    var text = this.sampleText;

    var charChunk = [];
    for (var i = 0; i < chunks.length; i++) {
      for (var j = chunks[i].start; j < chunks[i].end; j++) {
        charChunk[j] = i;
      }
    }

    var html = '';
    for (var k = 0; k < text.length; k++) {
      var ch = text[k];
      if (ch === '\n') { html += '<br>'; continue; }
      var ci = charChunk[k];
      var color = this.colors[(ci || 0) % this.colors.length];
      if (ch === ' ') ch = '\u00A0';
      html += '<span style="color:' + color + '">' + esc(ch) + '</span>';
    }
    this.textDisplay.innerHTML = html;

    this.statsEl.innerHTML = '';

    var maxW = text.length;
    var n = chunks.length;
    var barHtml =
      '<div style="height:320px;overflow-y:auto;overflow-x:hidden;border:1px solid #e2e8f0;border-radius:8px;box-sizing:border-box;scrollbar-width:thin;scrollbar-color:#cbd5e1 transparent;">' +
      '<div style="position:relative;height:' + (n * 32 + 8) + 'px;background:var(--bg-color,#f8fafc);padding:4px 0;">';

    for (var c = 0; c < n; c++) {
      var left = (chunks[c].start / maxW) * 100;
      var w = ((chunks[c].end - chunks[c].start) / maxW) * 100;
      var bc = this.colors[c % this.colors.length];
      var top = c * 32 + 4;

      barHtml +=
        '<div style="position:absolute;left:' + left.toFixed(1) + '%;top:' + top + 'px;' +
        'width:' + w.toFixed(1) + '%;height:24px;background:' + bc + ';border-radius:4px;opacity:0.85;' +
        'display:flex;align-items:center;justify-content:center;overflow:hidden;box-sizing:border-box;' +
        'transition:left 0.15s,width 0.15s;">' +
        '<span style="font-size:0.6rem;font-weight:700;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,0.35);white-space:nowrap;user-select:none;">' +
        'Chunk ' + (c + 1) +
        '</span></div>';

      if (c < n - 1) {
        var next = chunks[c + 1];
        var oStart = Math.max(chunks[c].start, next.start);
        var oEnd = Math.min(chunks[c].end, next.end);
        if (oStart < oEnd) {
          var ol = ((oStart - chunks[c].start) / (chunks[c].end - chunks[c].start)) * w;
          var ow = ((oEnd - oStart) / maxW) * 100;
          var oLeft = (oStart / maxW) * 100;
          barHtml +=
            '<div style="position:absolute;left:' + oLeft.toFixed(1) + '%;top:' + top + 'px;' +
            'width:' + ow.toFixed(1) + '%;height:24px;border-radius:4px;' +
            'background:repeating-linear-gradient(45deg,transparent,transparent 2px,rgba(255,255,255,0.25) 2px,rgba(255,255,255,0.25) 4px);' +
            'pointer-events:none;"></div>';
        }
      }
    }

    barHtml += '</div></div>';

    this.barsEl.innerHTML = barHtml;
  };

  function esc(ch) {
    if (ch === '&') return '&amp;';
    if (ch === '<') return '&lt;';
    if (ch === '>') return '&gt;';
    if (ch === '"') return '&quot;';
    return ch;
  }

  function init() {
    var el = document.getElementById('chunk-explorer');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new ChunkExplorer(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
