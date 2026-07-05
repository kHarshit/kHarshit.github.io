import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

(function () {
  'use strict';

  var HOST_COLOR = 0x6366f1;
  var CLIENT_COLOR = 0x20B2AA;
  var SERVER_COLOR = 0x10b981;

  var NODE_DATA = {
    host: {
      label: 'MCP Host',
      role: 'AI Application',
      desc: 'Coordinates one or more MCP clients. Manages the LLM, aggregates context from all connected servers, and orchestrates tool calls.',
    },
    client_0: { label: 'Client: Files',   role: 'File System',      desc: 'Dedicated connection to the filesystem server. Handles read/write/list operations on local files.' },
    client_1: { label: 'Client: DB',      role: 'Database',         desc: 'Dedicated connection to the database server. Executes SQL queries and returns structured results.' },
    client_2: { label: 'Client: API',     role: 'Jira / SaaS',      desc: 'Dedicated connection to the Jira API server. Searches issues, creates tickets, updates statuses.' },
    client_3: { label: 'Client: Web',     role: 'Web Search',       desc: 'Dedicated connection to the web search server. Performs web searches and scrapes content.' },
    server_0: { label: 'Server: Files',   role: 'fs Tools',         desc: 'Exposes tools: read_file, write_file, list_dir, search_files. Operates on the local filesystem.' },
    server_1: { label: 'Server: DB',      role: 'SQL Tools',        desc: 'Exposes tools: query, execute, get_schema. Connects to a PostgreSQL database via JSON-RPC.' },
    server_2: { label: 'Server: Jira',    role: 'Jira Tools',       desc: 'Exposes tools: search_jira, create_issue, get_issue. Wraps the Jira REST API behind MCP.' },
    server_3: { label: 'Server: Search',  role: 'Web Tools',        desc: 'Exposes tools: web_search, fetch_url, extract_text. Performs web searches and page extraction.' },
  };

  var NODES = [
    { id: 'host',    type: 'host',   angle: null, radius: 0 },
    { id: 'client_0', type: 'client', angle: 0,        radius: 1.8 },
    { id: 'client_1', type: 'client', angle: Math.PI/2, radius: 1.8 },
    { id: 'client_2', type: 'client', angle: Math.PI,   radius: 1.8 },
    { id: 'client_3', type: 'client', angle: 3*Math.PI/2, radius: 1.8 },
    { id: 'server_0', type: 'server', angle: Math.PI/4,        radius: 3.2 },
    { id: 'server_1', type: 'server', angle: 3*Math.PI/4,       radius: 3.2 },
    { id: 'server_2', type: 'server', angle: 5*Math.PI/4,       radius: 3.2 },
    { id: 'server_3', type: 'server', angle: 7*Math.PI/4,       radius: 3.2 },
  ];

  var EDGES = [
    { from: 'host', to: 'client_0' },
    { from: 'host', to: 'client_1' },
    { from: 'host', to: 'client_2' },
    { from: 'host', to: 'client_3' },
    { from: 'client_0', to: 'server_0' },
    { from: 'client_1', to: 'server_1' },
    { from: 'client_2', to: 'server_2' },
    { from: 'client_3', to: 'server_3' },
  ];

  var TYPE_COLORS = { host: HOST_COLOR, client: CLIENT_COLOR, server: SERVER_COLOR };
  var TYPE_RADIUS = { host: 0.35, client: 0.22, server: 0.2 };

  function nodePos(n) {
    if (n.radius === 0) return new THREE.Vector3(0, 0, 0);
    var y = 0.3 * Math.sin(n.angle * 2);
    return new THREE.Vector3(n.radius * Math.cos(n.angle), y, n.radius * Math.sin(n.angle));
  }

  function McpArch3D(container) {
    this.container = container;
    this.detail = document.getElementById('mcp3d-detail');
    this.ddName = document.getElementById('mcp3d-dd-name');
    this.ddRole = document.getElementById('mcp3d-dd-role');
    this.ddDesc = document.getElementById('mcp3d-dd-desc');
    this.meshMap = {};
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.hovered = null;

    this.initScene();
    this.initLights();
    this.initEdges();
    this.initNodes();
    this.initControls();
    this.initHover();
    this.animate();
  }

  McpArch3D.prototype.initScene = function () {
    var w = this.container.clientWidth, h = this.container.clientHeight;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, w / h, 0.1, 20);
    this.camera.position.set(4.5, 3.5, 5.5);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    this.container.appendChild(this.renderer.domElement);
    this.resizeObserver = new ResizeObserver(this.resize.bind(this));
    this.resizeObserver.observe(this.container);
  };

  McpArch3D.prototype.initLights = function () {
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    var d = new THREE.DirectionalLight(0xffffff, 2.0);
    d.position.set(4, 8, 6);
    this.scene.add(d);
    var d2 = new THREE.DirectionalLight(0x8888ff, 0.5);
    d2.position.set(-3, 4, -4);
    this.scene.add(d2);
  };

  var TEAL = '#20B2AA';

  McpArch3D.prototype.makeLabel = function (text, color) {
    var c = document.createElement('canvas');
    var dpr = 2;
    c.width = 512 * dpr;
    c.height = 96 * dpr;
    var ctx = c.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.fillStyle = color || TEAL;
    ctx.font = 'bold 30px -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 256, 48);
    var tex = new THREE.CanvasTexture(c);
    tex.minFilter = THREE.LinearFilter;
    var mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false, sizeAttenuation: true });
    var spr = new THREE.Sprite(mat);
    spr.scale.set(1.6, 0.3, 1);
    return spr;
  };

  McpArch3D.prototype.initNodes = function () {
    var self = this;
    NODES.forEach(function (n) {
      var pos = nodePos(n);
      var color = TYPE_COLORS[n.type];
      var radius = TYPE_RADIUS[n.type];
      var texture = null;

      if (n.type === 'host') {
        var c = document.createElement('canvas');
        c.width = 64; c.height = 64;
        var ctx = c.getContext('2d');
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 42px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('H', 32, 34);
        texture = new THREE.CanvasTexture(c);
      }

      var mat = new THREE.MeshPhysicalMaterial({
        color: color,
        emissive: color,
        emissiveIntensity: 0.2,
        metalness: 0.15,
        roughness: 0.25,
        clearcoat: 0.2,
        map: texture,
      });

      var geo = n.type === 'host'
        ? new THREE.BoxGeometry(radius * 2, radius * 2, radius * 2)
        : new THREE.SphereGeometry(radius, 20, 20);

      var mesh = new THREE.Mesh(geo, mat);
      mesh.position.copy(pos);
      mesh.userData = { node: n };
      self.scene.add(mesh);
      self.meshMap[n.id] = mesh;

      if (n.type !== 'host') {
        var lbl = self.makeLabel(n.id.replace('_', ' '));
        lbl.position.copy(pos);
        lbl.position.y -= radius + 0.18;
        lbl.scale.set(1.4, 0.25, 1);
        self.scene.add(lbl);
      }
    });
  };

  McpArch3D.prototype.initEdges = function () {
    var self = this;
    EDGES.forEach(function (e) {
      var from = nodePos(NODES.find(function (n) { return n.id === e.from; }));
      var to = nodePos(NODES.find(function (n) { return n.id === e.to; }));
      var mid = from.clone().add(to).multiplyScalar(0.5);
      mid.y += 0.3;
      var curve = new THREE.QuadraticBezierCurve3(from, mid, to);
      var pts = curve.getPoints(24);

      var color = e.from === 'host' ? HOST_COLOR : CLIENT_COLOR;
      var geo = new THREE.BufferGeometry().setFromPoints(pts);
      var mat = new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0.25 });
      self.scene.add(new THREE.Line(geo, mat));

      var arrowPos = curve.getPoint(0.45);
      var tangent = curve.getTangent(0.45).normalize();
      var cone = new THREE.Mesh(
        new THREE.ConeGeometry(0.04, 0.1, 6),
        new THREE.MeshBasicMaterial({ color: color, transparent: true, opacity: 0.35 })
      );
      cone.position.copy(arrowPos);
      cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), tangent);
      self.scene.add(cone);
    });
  };

  McpArch3D.prototype.initControls = function () {
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 2;
    this.controls.maxDistance = 12;
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  };

  McpArch3D.prototype.initHover = function () {
    var self = this;
    var meshes = Object.values(this.meshMap);
    this.container.addEventListener('mousemove', function (e) {
      var rect = self.container.getBoundingClientRect();
      self.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      self.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      self.checkHover(meshes);
    });
    this.container.addEventListener('mouseleave', function () { self.hideDetail(); });
  };

  McpArch3D.prototype.checkHover = function (meshes) {
    this.raycaster.setFromCamera(this.mouse, this.camera);
    var intersects = this.raycaster.intersectObjects(meshes);
    if (intersects.length > 0) {
      var hit = intersects[0].object;
      var n = hit.userData.node;
      var d = NODE_DATA[n.id];
      if (d) {
        this.showDetail(d, hit);
        hit.material.emissiveIntensity = 0.5;
        if (this.hovered && this.hovered !== hit) this.hovered.material.emissiveIntensity = 0.2;
        this.hovered = hit;
      }
    } else {
      this.hideDetail();
    }
  };

  McpArch3D.prototype.showDetail = function (d, intersect) {
    this.ddName.textContent = d.label;
    this.ddName.style.color = this.getTypeColor(d.label);
    this.ddRole.textContent = d.role;
    this.ddDesc.textContent = d.desc;
    this.detail.style.display = 'block';
  };

  McpArch3D.prototype.getTypeColor = function (label) {
    if (label.startsWith('MCP Host')) return '#6366f1';
    if (label.startsWith('Client')) return '#20B2AA';
    return '#10b981';
  };

  McpArch3D.prototype.hideDetail = function () {
    this.detail.style.display = 'none';
    if (this.hovered) { this.hovered.material.emissiveIntensity = 0.2; this.hovered = null; }
  };

  McpArch3D.prototype.resize = function () {
    var w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  };

  McpArch3D.prototype.animate = function () {
    var self = this;
    requestAnimationFrame(function () { self.animate(); });
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  var container = document.getElementById('mcp3d-container');
  if (container) { new McpArch3D(container); }
})();
