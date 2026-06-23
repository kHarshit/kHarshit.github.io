(function () {
  'use strict';

  function McpHandshake(root) {
    this.root = root;
    this.step = 0;
    this.maxStep = 3;

    this.nextBtn = root.querySelector('.mcp-hs-next');
    this.backBtn = root.querySelector('.mcp-hs-back');
    this.resetBtn = root.querySelector('.mcp-hs-reset');
    this.stepLabel = root.querySelector('.mcp-hs-step-label');

    this.clientMsg = root.querySelector('.mcp-hs-client-msg');
    this.serverMsg = root.querySelector('.mcp-hs-server-msg');
    this.descEl = root.querySelector('.mcp-hs-desc');

    this.phases = [
      { name: 'Initialize', client: '', server: '', desc: 'Click Next to start the handshake.' },
      { name: 'Protocol Version', client: '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"example-client","version":"1.0.0"}}}', server: '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{"tools":{},"resources":{}},"serverInfo":{"name":"example-server","version":"1.0.0"}}}', desc: 'Client sends init request with protocol version. Server responds with its supported version and capabilities.' },
      { name: 'Capabilities', client: '{"jsonrpc":"2.0","id":2,"method":"tools/list"}', server: '{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"search_jira","description":"Search issues","input_schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}]}}', desc: 'Client lists available tools. Server responds with tool definitions the LLM can use.' },
      { name: 'Ready', client: '&#10003; Connected', server: '&#10003; Connected', desc: 'Handshake complete. Client and server are ready for tool calls and resource access.' }
    ];

    this.buildDots();
    this.render();

    var self = this;
    this.nextBtn.addEventListener('click', function () {
      if (self.step < self.maxStep) { self.step++; self.render(); }
    });
    this.backBtn.addEventListener('click', function () {
      if (self.step > 0) { self.step--; self.render(); }
    });
    this.resetBtn.addEventListener('click', function () { self.step = 0; self.render(); });
  }

  McpHandshake.prototype.buildDots = function () {
    var bar = this.root.querySelector('.mcp-hs-bar');
    if (!bar) return;
    bar.innerHTML = '';
    for (var i = 0; i <= this.maxStep; i++) {
      var dot = document.createElement('span');
      dot.className = 'mcp-hs-dot';
      bar.appendChild(dot);
    }
    this.dots = bar.querySelectorAll('.mcp-hs-dot');
  };

  McpHandshake.prototype.render = function () {
    var p = this.phases[this.step];

    this.backBtn.disabled = (this.step <= 0);
    this.nextBtn.disabled = (this.step >= this.maxStep);
    this.nextBtn.textContent = (this.step >= this.maxStep) ? 'Done' : 'Next';

    if (this.stepLabel) this.stepLabel.textContent = 'Step ' + this.step + ': ' + p.name;
    if (this.descEl) this.descEl.textContent = p.desc;

    if (this.clientMsg) {
      this.clientMsg.innerHTML = p.client;
      this.clientMsg.className = 'mcp-hs-msg mcp-hs-client-msg' + (p.client ? '' : ' mcp-hs-empty');
    }
    if (this.serverMsg) {
      this.serverMsg.innerHTML = p.server;
      this.serverMsg.className = 'mcp-hs-msg mcp-hs-server-msg' + (p.server ? '' : ' mcp-hs-empty');
    }

    // Highlight active phase in diagram
    var phases = this.root.querySelectorAll('.mcp-hs-phase');
    phases.forEach(function (el, i) {
      el.classList.toggle('active', i === this.step);
    }, this);

    if (this.dots) {
      for (var d = 0; d < this.dots.length; d++) {
        this.dots[d].classList.toggle('done', d < this.step);
        this.dots[d].classList.toggle('current', d === this.step);
      }
    }
  };

  function init() {
    var el = document.getElementById('mcp-handshake');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new McpHandshake(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
