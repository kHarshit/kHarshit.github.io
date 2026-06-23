(function () {
  'use strict';

  function McpArch(root) {
    this.root = root;
    this.cards = root.querySelectorAll('.mcp-arch-card');
    this.detailTitle = root.querySelector('.mcp-arch-detail-title');
    this.detailRole = root.querySelector('.mcp-arch-detail-role');
    this.detailDesc = root.querySelector('.mcp-arch-detail-desc');
    this.detailPrims = root.querySelector('.mcp-arch-detail-prims');
    this.detailJson = root.querySelector('.mcp-arch-detail-json');

    var data = {
      host: {
        title: 'MCP Host',
        role: 'AI Application Coordinator',
        desc: 'The AI application (e.g., Claude Desktop, VS Code, custom app) that coordinates one or more MCP clients. It manages the overall workflow and provides the LLM with context from all connected servers.',
        prims: 'Orchestrates clients, aggregates context, manages lifecycle',
        json: JSON.stringify({"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"my-app","version":"1.0.0"}}}, null, 2)
      },
      client: {
        title: 'MCP Client',
        role: 'Connection Manager',
        desc: 'A component instantiated by the host that maintains a dedicated connection to one MCP server. Each server gets its own client instance. The client handles JSON-RPC communication, capability negotiation, and routing tool calls.',
        prims: 'Discovers tools (tools/list), calls tools (tools/call), manages resources, handles prompts',
        json: JSON.stringify({"jsonrpc":"2.0","id":1,"method":"tools/list"}, null, 2)
      },
      server: {
        title: 'MCP Server',
        role: 'Context & Tool Provider',
        desc: 'A program that exposes capabilities via JSON-RPC. Can run locally (filesystem, database) or remotely (SaaS APIs like Jira, Slack). Designed for LLMs to discover and use dynamically.',
        prims: 'Exposes Tools (actions), Resources (data), Prompts (templates), Sampling (LLM access), Logging',
        json: JSON.stringify({"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"search_jira","description":"Search Jira issues","input_schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}]}}, null, 2)
      }
    };

    var self = this;
    this.cards.forEach(function (card) {
      card.addEventListener('click', function () {
        self.cards.forEach(function (c) { c.classList.remove('active'); });
        card.classList.add('active');
        var key = card.getAttribute('data-component');
        var d = data[key];
        if (self.detailTitle) self.detailTitle.textContent = d.title;
        if (self.detailRole) self.detailRole.textContent = d.role;
        if (self.detailDesc) self.detailDesc.textContent = d.desc;
        if (self.detailPrims) self.detailPrims.textContent = d.prims;
        if (self.detailJson) self.detailJson.textContent = d.json;
      });
    });

    // Activate first card by default
    if (this.cards.length > 0) this.cards[0].click();
  }

  function init() {
    var el = document.getElementById('mcp-arch');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new McpArch(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
