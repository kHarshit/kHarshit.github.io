(function () {
  'use strict';

  function VecSearchSim(root) {
    this.root = root;
    this.queryInput = root.querySelector('.vecsearch-query');
    this.searchBtn = root.querySelector('.vecsearch-btn');
    this.randomBtn = root.querySelector('.vecsearch-random-btn');
    this.kSlider = root.querySelector('.vecsearch-k-slider');
    this.kDisplay = root.querySelector('.vecsearch-k-display');
    this.svg = root.querySelector('.vecsearch-plot');
    this.resultsEl = root.querySelector('.vecsearch-results');

    this.generateData();
    this.renderPlot();

    var self = this;
    this.searchBtn.addEventListener('click', function () { self.search(self.queryInput.value); });
    this.randomBtn.addEventListener('click', function () { self.randomQuery(); });
    this.kSlider.addEventListener('input', function () {
      self.kDisplay.textContent = self.kSlider.value;
      if (self.lastQuery !== undefined) self.search(self.lastQuery);
    });
    this.queryInput.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') self.search(self.queryInput.value);
    });
    this.queryInput.addEventListener('focus', function () {
      this.style.borderColor = '#20B2AA';
    });
    this.queryInput.addEventListener('blur', function () {
      this.style.borderColor = '#e2e8f0';
    });

    this.randomQuery();
  }

  VecSearchSim.prototype.generateData = function () {
    var clusters = [
      { topic: 'Revenue', cx: 140, cy: 340, spread: 30, count: 8,
        words: ['revenue', 'sales', 'income', 'earnings', 'quarter', 'financial', 'total'],
        snippets: ['Total revenue reached $25.7B', 'Automotive revenue grew 22%', 'Quarterly revenue exceeded expectations',
                   'Revenue driven by vehicle sales', 'Energy revenue surged 35%', 'Services revenue increased',
                   'Total revenue breakdown by segment', 'Year-over-year revenue comparison'] },
      { topic: 'Deliveries', cx: 240, cy: 335, spread: 28, count: 7,
        words: ['deliveries', 'vehicles', 'units', 'shipped', 'volume', 'production', '468000'],
        snippets: ['468,000 vehicles delivered in Q1', 'Vehicle deliveries up 18% YoY', 'Exceeded delivery guidance',
                   'Delivery volume by model', 'Production output ramped up', 'Global deliveries expanded',
                   'Regional delivery breakdown'] },
      { topic: 'Margins', cx: 340, cy: 335, spread: 25, count: 6,
        words: ['margin', 'gross', 'profit', 'operating', 'net', 'profitability'],
        snippets: ['Gross margin improved to 19.8%', 'Operating margin expanded to 9.7%', 'Net income grew to $2.1B',
                   'Profitability improved YoY', 'Margin analysis by segment', 'Cost efficiencies drove margin expansion'] },
      { topic: 'Costs', cx: 90, cy: 190, spread: 25, count: 7,
        words: ['cost', 'expense', 'spending', 'capex', 'capital', 'battery', 'optimization'],
        snippets: ['Lower battery costs improved margins', 'Capital expenditures totaled $2.0B', 'Operating expenses managed tightly',
                   'Production cost optimization ongoing', 'Vertical integration reduced costs', 'R&D spending totaled $1.2B',
                   'Cost reduction initiatives progressing'] },
      { topic: 'Energy', cx: 390, cy: 190, spread: 28, count: 7,
        words: ['energy', 'battery', 'megapack', 'powerwall', 'storage', 'solar', 'grid'],
        snippets: ['Energy storage deployments doubled to 8.3 GWh', 'Megapack demand at record levels',
                   'Powerwall installations growing rapidly', 'Tesla Energy gross margin expected to exceed auto',
                   'Energy segment revenue surged 35%', 'Battery storage capacity expanding',
                   'Utility-scale energy projects pipeline'] },
      { topic: 'Cash', cx: 90, cy: 90, spread: 22, count: 6,
        words: ['cash', 'liquidity', 'balance', 'asset', 'free cash', 'capital', 'investment'],
        snippets: ['Cash reserves increased to $28.4B', 'Free cash flow reached $1.8B', 'Strong balance sheet position',
                   'Working capital management efficient', 'Cash from operations grew', 'Strategic investments funded by cash'] },
      { topic: 'R&D', cx: 390, cy: 90, spread: 25, count: 7,
        words: ['research', 'development', 'autonomous', 'fsd', 'ai', 'driving', 'neural', 'dojo', 'software'],
        snippets: ['R&D investment in autonomous driving', 'FSD v12 uses end-to-end neural networks',
                   'Dojo supercomputer for AI training', 'Next-gen vehicle platform development',
                   'Autonomous driving technology progress', 'AI infrastructure investments growing',
                   'Over-the-air software updates expanded'] },
      { topic: 'Production', cx: 240, cy: 90, spread: 28, count: 7,
        words: ['gigafactory', 'factory', 'manufacturing', 'cybertruck', 'capacity', 'ramp', 'line'],
        snippets: ['Cybertruck production run rate 150K/quarter', 'Gigafactories in Austin and Berlin ramping',
                   'Production capacity expanding globally', 'New affordable model in development for 2026',
                   'Manufacturing efficiency improvements', 'Factory output reached new records',
                   'Supply chain stability supports production'] },
      { topic: 'Regulatory', cx: 140, cy: 240, spread: 20, count: 5,
        words: ['regulatory', 'credit', 'compliance', 'emission', 'carbon'],
        snippets: ['Regulatory credit revenue was $620M', 'Emission credits sold to other automakers',
                   'Regulatory compliance maintained', 'Carbon credit market evolving',
                   'Zero-emission vehicle credits generated'] },
      { topic: 'Employment', cx: 340, cy: 240, spread: 20, count: 5,
        words: ['employee', 'workforce', 'hiring', 'staff', 'headcount', 'team'],
        snippets: ['145,000 full-time employees globally', 'Workforce expanded for production ramp',
                   'Engineering team grew significantly', 'Employee retention rates improved',
                   'New hiring across AI and energy teams'] },
    ];

    this.points = [];
    this.pointColors = ['#20B2AA','#3b82f6','#22c55e','#f59e0b','#8b5cf6','#06b6d4','#ec4899','#10b981','#f97316','#6366f1'];

    for (var c = 0; c < clusters.length; c++) {
      var cl = clusters[c];
      for (var i = 0; i < cl.count; i++) {
        var angle = Math.random() * 2 * Math.PI;
        var radius = Math.random() * cl.spread;
        var x = cl.cx + radius * Math.cos(angle);
        var y = cl.cy + radius * Math.sin(angle);
        this.points.push({
          x: x, y: y,
          topic: cl.topic,
          snippet: cl.snippets[i % cl.snippets.length],
          words: cl.words,
          color: this.pointColors[c % this.pointColors.length],
          clusterIdx: c,
        });
      }
    }

    this.topics = clusters.map(function (c) { return c.topic; });
  };

  VecSearchSim.prototype.renderPlot = function () {
    var svg = this.svg;
    var ns = 'http://www.w3.org/2000/svg';
    svg.innerHTML = '';

    var w = 480, h = 400, pad = 30;
    var xMin = 20, xMax = 460, yMin = 400 - 30, yMax = 30;

    function mapX(px) { return pad + (px / 460) * (w - 2 * pad); }
    function mapY(py) { return h - pad - (py / 460) * (h - 2 * pad); }

    var bg = document.createElementNS(ns, 'rect');
    bg.setAttribute('width', '100%'); bg.setAttribute('height', '100%');
    bg.setAttribute('fill', 'var(--bg-color,#f8fafc)'); bg.setAttribute('rx', '8');
    svg.appendChild(bg);

    for (var i = 0; i < this.points.length; i++) {
      var p = this.points[i];
      var cx = mapX(p.x), cy = mapY(p.y);
      var circle = document.createElementNS(ns, 'circle');
      circle.setAttribute('cx', cx); circle.setAttribute('cy', cy);
      circle.setAttribute('r', 6);
      circle.setAttribute('fill', p.color);
      circle.setAttribute('opacity', '0.7');
      circle.setAttribute('stroke', '#fff');
      circle.setAttribute('stroke-width', '1');
      circle.setAttribute('class', 'vecsearch-dot');
      circle.setAttribute('data-idx', i);
      var title = document.createElementNS(ns, 'title');
      title.textContent = '[' + p.topic + '] ' + p.snippet;
      circle.appendChild(title);
      svg.appendChild(circle);
    }

    var clusterCenters = [
      { topic: 'Revenue', cx: 140, cy: 350 },
      { topic: 'Deliveries', cx: 240, cy: 350 },
      { topic: 'Margins', cx: 340, cy: 350 },
      { topic: 'Costs', cx: 90, cy: 195 },
      { topic: 'Energy', cx: 390, cy: 195 },
      { topic: 'Cash', cx: 90, cy: 75 },
      { topic: 'R&D', cx: 390, cy: 75 },
      { topic: 'Production', cx: 240, cy: 75 },
      { topic: 'Regulatory', cx: 140, cy: 245 },
      { topic: 'Employment', cx: 340, cy: 245 },
    ];
    for (var ci = 0; ci < clusterCenters.length; ci++) {
      var cc = clusterCenters[ci];
      var lx = mapX(cc.cx), ly = mapY(cc.cy);
      var label = document.createElementNS(ns, 'text');
      label.setAttribute('x', lx);
      label.setAttribute('y', ly);
      label.setAttribute('text-anchor', 'middle');
      label.setAttribute('font-size', '9');
      label.setAttribute('font-weight', '600');
      label.setAttribute('fill', this.pointColors[ci % this.pointColors.length]);
      label.setAttribute('opacity', '0.8');
      label.textContent = cc.topic;
      svg.appendChild(label);
    }

    this.dots = svg.querySelectorAll('.vecsearch-dot');
    this.mapX = mapX; this.mapY = mapY;
  };

  VecSearchSim.prototype.search = function (query) {
    query = query.trim();
    if (!query) return;
    this.lastQuery = query;

    var k = parseInt(this.kSlider.value, 10);

    var topicScores = {};
    var allWords = query.toLowerCase().split(/\s+/);
    for (var i = 0; i < this.points.length; i++) {
      var p = this.points[i];
      if (!topicScores[p.topic]) topicScores[p.topic] = 0;
      for (var w = 0; w < allWords.length; w++) {
        var word = allWords[w];
        for (var v = 0; v < p.words.length; v++) {
          if (p.words[v].toLowerCase().indexOf(word) >= 0 || word.indexOf(p.words[v].toLowerCase()) >= 0) {
            topicScores[p.topic] += 1;
            break;
          }
        }
      }
    }

    var maxTS = 0;
    for (var t in topicScores) { if (topicScores[t] > maxTS) maxTS = topicScores[t]; }
    if (maxTS === 0) {
      this.resultsEl.innerHTML = '<div style="padding:10px;font-size:0.82rem;color:#888;text-align:center;">No topic matches found for this query.</div>';
      for (var dd = 0; dd < this.dots.length; dd++) {
        this.dots[dd].setAttribute('opacity', '0.25');
        this.dots[dd].setAttribute('stroke', '#fff');
        this.dots[dd].setAttribute('stroke-width', '1');
        this.dots[dd].setAttribute('r', '5');
      }
      var qm = this.svg.querySelector('.vecsearch-query-marker');
      if (qm) qm.innerHTML = '';
      return;
    }

    var queryVec = { x: 0, y: 0 };
    for (var t2 in topicScores) {
      var score = topicScores[t2] / maxTS;
      for (var i2 = 0; i2 < this.points.length; i2++) {
        if (this.points[i2].topic === t2) {
          queryVec.x += this.points[i2].x * score;
          queryVec.y += this.points[i2].y * score;
        }
      }
    }
    var norm = Math.sqrt(queryVec.x * queryVec.x + queryVec.y * queryVec.y);
    if (norm > 0) { queryVec.x /= norm; queryVec.y /= norm; }

    var scored = [];
    for (var j = 0; j < this.points.length; j++) {
      var pj = this.points[j];
      var pNorm = Math.sqrt(pj.x * pj.x + pj.y * pj.y);
      if (pNorm === 0) { scored.push({ idx: j, sim: 0 }); continue; }
      var sim = (queryVec.x * (pj.x / pNorm) + queryVec.y * (pj.y / pNorm));
      scored.push({ idx: j, sim: sim });
    }

    scored.sort(function (a, b) { return b.sim - a.sim; });
    var topK = scored.slice(0, k);

    for (var d = 0; d < this.dots.length; d++) {
      var idx = parseInt(this.dots[d].getAttribute('data-idx'), 10);
      var isSelected = false;
      for (var s = 0; s < topK.length; s++) {
        if (topK[s].idx === idx) { isSelected = true; break; }
      }
      if (isSelected) {
        this.dots[d].setAttribute('opacity', '1');
        this.dots[d].setAttribute('stroke', '#000');
        this.dots[d].setAttribute('stroke-width', '2.5');
        this.dots[d].setAttribute('r', '8');
      } else {
        this.dots[d].setAttribute('opacity', '0.25');
        this.dots[d].setAttribute('stroke', '#fff');
        this.dots[d].setAttribute('stroke-width', '1');
        this.dots[d].setAttribute('r', '5');
      }
    }

    var qx = 0, qy = 0;
    for (var j2 = 0; j2 < this.points.length; j2++) {
      qx += this.points[j2].x * (topicScores[this.points[j2].topic] || 0);
      qy += this.points[j2].y * (topicScores[this.points[j2].topic] || 0);
    }
    var qNorm = Math.sqrt(qx * qx + qy * qy);
    if (qNorm > 0) { qx /= qNorm; qy /= qNorm; }
    qx *= 200; qy *= 200;
    qx = Math.max(30, Math.min(450, qx + 30));
    qy = Math.max(30, Math.min(370, qy + 30));

    var queryMarker = this.svg.querySelector('.vecsearch-query-marker');
    var ns = 'http://www.w3.org/2000/svg';
    if (!queryMarker) {
      queryMarker = document.createElementNS(ns, 'g');
      queryMarker.setAttribute('class', 'vecsearch-query-marker');
      this.svg.appendChild(queryMarker);
    }
    var mx = this.mapX(qx), my = this.mapY(qy);
    queryMarker.innerHTML =
      '<circle cx="' + mx + '" cy="' + my + '" r="10" fill="none" stroke="#ef4444" stroke-width="2.5"/>' +
      '<circle cx="' + mx + '" cy="' + my + '" r="10" fill="#ef4444" opacity="0.15"/>' +
      '<circle cx="' + mx + '" cy="' + my + '" r="3" fill="#ef4444"/>';

    var resultsHtml = '<div style="font-weight:600;font-size:0.82rem;color:var(--font-color,#555);margin-bottom:8px;">Top ' + k + ' Results</div>';
    for (var r = 0; r < topK.length; r++) {
      var pt = this.points[topK[r].idx];
      var pct = Math.round(topK[r].sim * 100);
      resultsHtml +=
        '<div style="padding:7px 10px;margin-bottom:4px;border-radius:6px;background:var(--bg-color,#f8fafc);border:1px solid #e2e8f0;">' +
        '<div style="font-size:0.72rem;font-weight:600;color:' + pt.color + ';">' + pt.topic + ' &middot; ' + pct + '% match</div>' +
        '<div style="font-size:0.78rem;color:var(--font-color,#555);margin-top:2px;">"' + pt.snippet + '"</div>' +
        '</div>';
    }
    this.resultsEl.innerHTML = resultsHtml;
  };

  VecSearchSim.prototype.randomQuery = function () {
    var queries = [
      'Tesla revenue growth this quarter',
      'vehicle delivery numbers',
      'profit margins and net income',
      'cash flow and balance sheet',
      'battery storage deployments',
      'autonomous driving progress',
      'production capacity expansion',
      'research and development spending',
      'employee headcount',
      'regulatory credits revenue',
      'capital expenditure plans',
      'gross margin improvement',
    ];
    var q = queries[Math.floor(Math.random() * queries.length)];
    this.queryInput.value = q;
    this.search(q);
  };

  function init() {
    var el = document.getElementById('vecsearch-sim');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new VecSearchSim(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
