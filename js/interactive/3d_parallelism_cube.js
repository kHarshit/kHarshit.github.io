import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

(function () {
  'use strict';

  var DIMS = { dp: 4, pp: 3, tp: 2 };
  var SIZE = 0.7;
  var GAP = 0.3;
  var STEP = SIZE + GAP;

  function getPos(dp, pp, tp) {
    return new THREE.Vector3(
      (dp - (DIMS.dp - 1) / 2) * STEP,
      (pp - (DIMS.pp - 1) / 2) * STEP,
      (tp - (DIMS.tp - 1) / 2) * STEP
    );
  }

  function getGPUColor(dp, pp, tp) {
    var hue = 0.58 - (dp / (DIMS.dp - 1)) * 0.50;
    var sat = 0.55 + (tp / (DIMS.tp - 1)) * 0.30;
    var lig = 0.40 + (pp / (DIMS.pp - 1)) * 0.30;
    return new THREE.Color().setHSL(hue, sat, lig);
  }

  function makeLabelSprite(text) {
    var canvas = document.createElement('canvas');
    var dpr = 2;
    canvas.width = 512 * dpr;
    canvas.height = 96 * dpr;
    var ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.fillStyle = 'rgba(0,0,0,0)';
    ctx.fillRect(0, 0, 512, 96);
    ctx.shadowColor = 'rgba(0,0,0,0.8)';
    ctx.shadowBlur = 6;
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 34px -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 256, 48);
    var texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    var material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false, sizeAttenuation: true });
    var sprite = new THREE.Sprite(material);
    sprite.scale.set(3.0, 0.56, 1);
    return sprite;
  }

  function makeDimLabel(text, color) {
    var c = document.createElement('canvas');
    var dpr = 2;
    c.width = 256 * dpr;
    c.height = 256 * dpr;
    var ctx = c.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, 256, 256);
    ctx.shadowColor = 'rgba(0,0,0,0.9)';
    ctx.shadowBlur = 10;
    ctx.fillStyle = color;
    ctx.font = 'bold 60px -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 128, 128);
    var tex = new THREE.CanvasTexture(c);
    tex.minFilter = THREE.LinearFilter;
    var mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false, sizeAttenuation: true });
    var spr = new THREE.Sprite(mat);
    spr.scale.set(1.2, 1.2, 1);
    return spr;
  }

  function Parallelism3D(container) {
    this.container = container;
    this.mode = 'all';
    this.sliceIdx = 0;
    this.gpuData = [];

    this.initScene();
    this.initLights();
    this.initAxes();
    this.initGPUs();
    this.initControls();
    this.setupUI();
    this.animate();
  }

  Parallelism3D.prototype.initScene = function () {
    var w = this.container.clientWidth;
    var h = this.container.clientHeight;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, w / h, 0.1, 100);
    this.camera.position.set(5.5, 4.5, 7.5);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    this.container.appendChild(this.renderer.domElement);

    this.resizeObserver = new ResizeObserver(this.resize.bind(this));
    this.resizeObserver.observe(this.container);
  };

  Parallelism3D.prototype.initLights = function () {
    this.scene.add(new THREE.AmbientLight(0x404060, 0.6));
    var dir = new THREE.DirectionalLight(0xffffff, 1.8);
    dir.position.set(4, 8, 6);
    this.scene.add(dir);
    var dir2 = new THREE.DirectionalLight(0x8888ff, 0.5);
    dir2.position.set(-3, 2, -4);
    this.scene.add(dir2);
    var rim = new THREE.DirectionalLight(0x4488ff, 0.3);
    rim.position.set(-2, -3, 5);
    this.scene.add(rim);
  };

  Parallelism3D.prototype.initAxes = function () {
    var axisLen = 3.8;
    var origin = new THREE.Vector3(-2.8, -2.0, -1.5);

    var xColor = 0xff6b6b;
    var yColor = 0x51cf66;
    var zColor = 0x5c7cfa;

    var xEnd = new THREE.Vector3(origin.x + axisLen, origin.y, origin.z);
    var yEnd = new THREE.Vector3(origin.x, origin.y + axisLen, origin.z);
    var zEnd = new THREE.Vector3(origin.x, origin.y, origin.z + axisLen);

    this.makeArrowLine(origin, xEnd, xColor);
    this.makeArrowCone(xEnd, new THREE.Vector3(1, 0, 0), xColor);

    this.makeArrowLine(origin, yEnd, yColor);
    this.makeArrowCone(yEnd, new THREE.Vector3(0, 1, 0), yColor);

    this.makeArrowLine(origin, zEnd, zColor);
    this.makeArrowCone(zEnd, new THREE.Vector3(0, 0, 1), zColor);

    var dpLabel = makeDimLabel('DP', '#ff6b6b');
    dpLabel.position.set(xEnd.x + 1.0, xEnd.y, xEnd.z);
    this.scene.add(dpLabel);

    var ppLabel = makeDimLabel('PP', '#51cf66');
    ppLabel.position.set(yEnd.x, yEnd.y + 1.0, yEnd.z);
    this.scene.add(ppLabel);

    var tpLabel = makeDimLabel('TP', '#5c7cfa');
    tpLabel.position.set(zEnd.x, zEnd.y, zEnd.z + 1.0);
    this.scene.add(tpLabel);
  };

  Parallelism3D.prototype.makeArrowLine = function (from, to, color) {
    var geo = new THREE.BufferGeometry().setFromPoints([from, to]);
    var mat = new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0.6 });
    this.scene.add(new THREE.Line(geo, mat));
  };

  Parallelism3D.prototype.makeArrowCone = function (pos, dir, color) {
    var cone = new THREE.Mesh(
      new THREE.ConeGeometry(0.12, 0.3, 8),
      new THREE.MeshBasicMaterial({ color: color })
    );
    cone.position.copy(pos);
    cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    this.scene.add(cone);
  };

  Parallelism3D.prototype.initGPUs = function () {
    var geo = new THREE.BoxGeometry(SIZE, SIZE, SIZE);
    var edgeGeo = new THREE.EdgesGeometry(geo);

    for (var dp = 0; dp < DIMS.dp; dp++) {
      for (var pp = 0; pp < DIMS.pp; pp++) {
        for (var tp = 0; tp < DIMS.tp; tp++) {
          var pos = getPos(dp, pp, tp);
          var color = getGPUColor(dp, pp, tp);

          var mat = new THREE.MeshPhysicalMaterial({
            color: color,
            metalness: 0.15,
            roughness: 0.35,
            clearcoat: 0.15,
            clearcoatRoughness: 0.3,
          });

          var mesh = new THREE.Mesh(geo, mat);
          mesh.position.copy(pos);
          this.scene.add(mesh);

          var edgeMat = new THREE.LineBasicMaterial({
            color: 0x444444,
            transparent: true,
            opacity: 0.25,
          });
          var edge = new THREE.LineSegments(edgeGeo.clone(), edgeMat);
          edge.position.copy(pos);
          this.scene.add(edge);

          this.gpuData.push({
            mesh: mesh, edge: edge, mat: mat, edgeMat: edgeMat,
            dp: dp, pp: pp, tp: tp, baseColor: color.clone(), pos: pos,
          });
        }
      }
    }
  };

  Parallelism3D.prototype.initControls = function () {
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 3;
    this.controls.maxDistance = 18;
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  };

  Parallelism3D.prototype.updateHighlight = function (mode, sliceIdx) {
    this.mode = mode;
    this.sliceIdx = sliceIdx;

    for (var i = 0; i < this.gpuData.length; i++) {
      var g = this.gpuData[i];
      var selected = mode === 'all' || this.isSelected(g);

      if (mode === 'all') {
        g.mat.color.copy(g.baseColor);
        g.mat.opacity = 1;
        g.mat.transparent = false;
        g.mat.roughness = 0.35;
        g.edgeMat.opacity = 0.25;
        g.edge.visible = true;
      } else if (selected) {
        g.mat.color.copy(g.baseColor);
        g.mat.opacity = 1;
        g.mat.transparent = false;
        g.mat.roughness = 0.25;
        g.mat.emissive.copy(g.baseColor);
        g.mat.emissiveIntensity = 0.15;
        g.edgeMat.opacity = 0.4;
        g.edge.visible = true;
      } else {
        g.mat.color.setHSL(0, 0, 0.3);
        g.mat.opacity = 0.12;
        g.mat.transparent = true;
        g.mat.roughness = 0.8;
        g.mat.emissive.setHex(0x000000);
        g.mat.emissiveIntensity = 0;
        g.edgeMat.opacity = 0.04;
        g.edge.visible = false;
      }
    }
  };

  Parallelism3D.prototype.isSelected = function (g) {
    switch (this.mode) {
      case 'dp': return g.dp === this.sliceIdx;
      case 'pp': return g.pp === this.sliceIdx;
      case 'tp': return g.tp === this.sliceIdx;
      default: return true;
    }
  };

  Parallelism3D.prototype.setupUI = function () {
    var root = document.getElementById('pc-viz');
    if (!root) return;

    var self = this;
    var buttons = root.querySelectorAll('.pc-btn');
    var sliderRow = document.getElementById('pc-slider-row');
    var slider = document.getElementById('pc-slider');
    var sliceVal = document.getElementById('pc-slice-val');
    var desc = document.getElementById('pc-desc');

    var modeDescs = {
      all: 'All 24 GPUs arranged in a 3D grid. Drag to rotate \u2022 Scroll to zoom',
      dp: 'Data Parallelism: splits the training batch across GPUs. Each DP group (color) processes different data and syncs gradients via AllReduce.',
      pp: 'Pipeline Parallelism: splits model layers across GPUs. Each PP stage (row) holds consecutive layers and passes activations forward.',
      tp: 'Tensor Parallelism: splits weight matrices across GPUs. Each TP rank (column) holds a slice of each layer\u2019s weights.',
    };

    var maxDims = { dp: DIMS.dp - 1, pp: DIMS.pp - 1, tp: DIMS.tp - 1 };

    buttons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        buttons.forEach(function (b) { b.classList.remove('active'); });
        btn.classList.add('active');

        var mode = btn.dataset.mode;

        if (mode === 'all') {
          sliderRow.style.display = 'none';
          self.updateHighlight('all', 0);
          desc.textContent = modeDescs.all;
        } else {
          sliderRow.style.display = 'flex';
          var max = maxDims[mode];
          slider.max = max;
          var val = Math.min(parseInt(slider.value) || 0, max);
          slider.value = val;
          sliceVal.textContent = val + 1;
          self.updateHighlight(mode, val);
          desc.textContent = modeDescs[mode];
        }
      });
    });

    slider.addEventListener('input', function () {
      var val = parseInt(slider.value);
      sliceVal.textContent = val + 1;
      self.updateHighlight(self.mode, val);
    });
  };

  Parallelism3D.prototype.resize = function () {
    var w = this.container.clientWidth;
    var h = this.container.clientHeight;
    if (w === 0 || h === 0) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  };

  Parallelism3D.prototype.animate = function () {
    var self = this;
    requestAnimationFrame(function () { self.animate(); });
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  var container = document.getElementById('pc-container');
  if (container) {
    new Parallelism3D(container);
  }
})();
