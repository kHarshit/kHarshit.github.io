(function () {
  'use strict';

  function McpToolFlow(root) {
    this.root = root;
    this.step = 0;
    this.maxStep = 5;

    this.nextBtn = root.querySelector('.mcp-tf-next');
    this.backBtn = root.querySelector('.mcp-tf-back');
    this.resetBtn = root.querySelector('.mcp-tf-reset');
    this.stepLabel = root.querySelector('.mcp-tf-step-label');
    this.stepDesc = root.querySelector('.mcp-tf-desc');
    this.stageEl = root.querySelector('.mcp-tf-stage');

    this.steps = [
      {
        label: 'User Query',
        desc: 'User asks: "Summarize open high-priority Jira tasks."',
        stage: 'query',
        detail: 'The user sends their request to the AI application. The LLM receives this as its conversational context.'
      },
      {
        label: 'Model Reasons',
        desc: 'LLM reasons: "I need Jira data. I can use search_jira tool."',
        stage: 'reason',
        detail: 'The LLM sees the available tools from all connected MCP servers. It decides which tool to call and what arguments to pass.'
      },
      {
        label: 'Tool Call',
        desc: 'LLM generates structured tool call output.',
        stage: 'call',
        detail: 'The LLM responds with a tool call: { "tool": "search_jira", "arguments": { "query": "priority=high AND status=open" } }'
      },
      {
        label: 'Client Routes',
        desc: 'MCP client intercepts and sends JSON-RPC tools/call to server.',
        stage: 'route',
        detail: 'The AI application intercepts the tool call, finds the right MCP server, and sends a JSON-RPC request: tools/call with the tool name and arguments.'
      },
      {
        label: 'Server Executes',
        desc: 'MCP server runs the tool and returns results.',
        stage: 'execute',
        detail: 'The server executes the tool (e.g., queries Jira API), formats the result as structured content, and sends it back to the client.'
      },
      {
        label: 'Final Response',
        desc: 'LLM uses tool results as context to generate the answer.',
        stage: 'response',
        detail: 'The client inserts the tool results into the LLM\'s context. The LLM now has real Jira data and generates the final response to the user.'
      }
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

  McpToolFlow.prototype.buildDots = function () {
    var bar = this.root.querySelector('.mcp-tf-bar');
    if (!bar) return;
    bar.innerHTML = '';
    for (var i = 0; i <= this.maxStep; i++) {
      var dot = document.createElement('span');
      dot.className = 'mcp-tf-dot';
      bar.appendChild(dot);
    }
    this.dots = bar.querySelectorAll('.mcp-tf-dot');
  };

  McpToolFlow.prototype.render = function () {
    var s = this.steps[this.step];

    this.backBtn.disabled = (this.step <= 0);
    this.nextBtn.disabled = (this.step >= this.maxStep);
    this.nextBtn.textContent = (this.step >= this.maxStep) ? 'Done' : 'Next';

    if (this.stepLabel) this.stepLabel.textContent = 'Step ' + this.step + ': ' + s.label;
    if (this.stepDesc) this.stepDesc.textContent = s.desc;
    this.stageEl.textContent = s.detail;

    this.stageEl.className = 'mcp-tf-stage';
    this.stageEl.classList.add('mcp-tf-stage-' + s.stage);

    // Update flow diagram
    var nodes = this.root.querySelectorAll('.mcp-tf-node');
    nodes.forEach(function (node, i) {
      node.classList.toggle('active', i === this.step);
      node.classList.toggle('done', i < this.step);
    }, this);

    var connectors = this.root.querySelectorAll('.mcp-tf-connector');
    connectors.forEach(function (conn, i) {
      conn.classList.toggle('done', i < this.step);
    }, this);

    this.dots.forEach(function (dot, i) {
      dot.classList.remove('done', 'current');
      if (i < this.step) dot.classList.add('done');
      if (i === this.step) dot.classList.add('current');
    }, this);
  };

  function init() {
    var el = document.getElementById('mcp-toolflow');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new McpToolFlow(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
