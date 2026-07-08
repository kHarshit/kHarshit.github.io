import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

(function () {
  'use strict';

  var BLOCK_COLORS = [
    '#f59e0b', '#10b981', '#6366f1', '#ec4899', '#0891b2', '#dc2626',
    '#8b5cf6', '#14b8a6', '#f97316', '#84cc16', '#06b6d4', '#a855f7',
  ];

  function makeLabelSprite(text, color) {
    color = color || '#20B2AA';
    var canvas = document.createElement('canvas');
    var dpr = 2;
    canvas.width = 512 * dpr;
    canvas.height = 96 * dpr;
    var ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, 512, 96);
    ctx.shadowColor = 'rgba(0,0,0,0.9)';
    ctx.shadowBlur = 8;
    ctx.fillStyle = color;
    ctx.font = 'bold 34px -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 256, 48);
    var texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    var material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false, sizeAttenuation: true });
    var sprite = new THREE.Sprite(material);
    sprite.scale.set(2.4, 0.45, 1);
    return sprite;
  }

  function CudaGrid3D(container) {
    this.container = container;
    this.info = document.getElementById('cg-info');
    this.threadMeshes = [];
    this.blockMeshes = [];
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.hovered = null;

    this.gx = 3; this.gy = 2; this.gz = 2;
    this.bx = 4; this.by = 3; this.bz = 2;

    this.initScene();
    this.initLights();
    this.initAxes();
    this.buildGrid();
    this.initControls();
    this.initHover();
    this.setupUI();
    this.animate();
  }

  CudaGrid3D.prototype.initScene = function () {
    var w = this.container.clientWidth;
    var h = this.container.clientHeight;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 100);
    this.camera.position.set(5, 4, 6);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    this.container.appendChild(this.renderer.domElement);

    this.resizeObserver = new ResizeObserver(this.resize.bind(this));
    this.resizeObserver.observe(this.container);
  };

  CudaGrid3D.prototype.initLights = function () {
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

  CudaGrid3D.prototype.makeArrowLine = function (from, to, color) {
    var geo = new THREE.BufferGeometry().setFromPoints([from, to]);
    var mat = new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0.6 });
    this.scene.add(new THREE.Line(geo, mat));
  };

  CudaGrid3D.prototype.makeArrowCone = function (pos, dir, color) {
    var cone = new THREE.Mesh(
      new THREE.ConeGeometry(0.12, 0.3, 8),
      new THREE.MeshBasicMaterial({ color: color })
    );
    cone.position.copy(pos);
    cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    this.scene.add(cone);
  };

  CudaGrid3D.prototype.initAxes = function () {
    var axisLen = 3.0;
    var origin = new THREE.Vector3(-1.8, -1.5, -1.0);

    var axColor = 0x20B2AA;

    var xEnd = new THREE.Vector3(origin.x + axisLen, origin.y, origin.z);
    var yEnd = new THREE.Vector3(origin.x, origin.y + axisLen, origin.z);
    var zEnd = new THREE.Vector3(origin.x, origin.y, origin.z + axisLen);

    this.makeArrowLine(origin, xEnd, axColor);
    this.makeArrowCone(xEnd, new THREE.Vector3(1, 0, 0), axColor);

    this.makeArrowLine(origin, yEnd, axColor);
    this.makeArrowCone(yEnd, new THREE.Vector3(0, 1, 0), axColor);

    this.makeArrowLine(origin, zEnd, axColor);
    this.makeArrowCone(zEnd, new THREE.Vector3(0, 0, 1), axColor);

    var xLabel = makeLabelSprite('BlockIdx.x');
    xLabel.position.set(xEnd.x + 1.0, xEnd.y, xEnd.z);
    this.scene.add(xLabel);

    var yLabel = makeLabelSprite('BlockIdx.y');
    yLabel.position.set(yEnd.x, yEnd.y + 1.0, yEnd.z);
    this.scene.add(yLabel);

    var zLabel = makeLabelSprite('BlockIdx.z');
    zLabel.position.set(zEnd.x, zEnd.y, zEnd.z + 1.0);
    this.scene.add(zLabel);
  };

  CudaGrid3D.prototype.buildGrid = function () {
    var self = this;
    var blockSpacing = 1.5;
    var threadSpacing = 0.18;
    var totalThreads = this.gx * this.gy * this.gz * this.bx * this.by * this.bz;

    if (totalThreads > 3000) {
      return;
    }

    var gcx = this.gx, gcy = this.gy, gcz = this.gz;
    var bcx = this.bx, bcy = this.by, bcz = this.bz;

    var totalW = gcx * blockSpacing;
    var totalH = gcy * blockSpacing;
    var totalD = gcz * blockSpacing;
    var offsetX = -totalW / 2 + blockSpacing / 2;
    var offsetY = -totalH / 2 + blockSpacing / 2;
    var offsetZ = -totalD / 2 + blockSpacing / 2;

    var globalIdx = 0;

    for (var gx = 0; gx < gcx; gx++) {
      for (var gy = 0; gy < gcy; gy++) {
        for (var gz = 0; gz < gcz; gz++) {
          var blockCenter = new THREE.Vector3(
            offsetX + gx * blockSpacing,
            offsetY + gy * blockSpacing,
            offsetZ + gz * blockSpacing
          );

          var blockIdx = gx * gcy * gcz + gy * gcz + gz;
          var colorHex = BLOCK_COLORS[blockIdx % BLOCK_COLORS.length];
          var color = new THREE.Color(colorHex);

          var tw = (bcx - 1) * threadSpacing;
          var th = (bcy - 1) * threadSpacing;
          var td = (bcz - 1) * threadSpacing;

          var wireGeo = new THREE.BoxGeometry(tw + threadSpacing, th + threadSpacing, td + threadSpacing);
          var edges = new THREE.EdgesGeometry(wireGeo);
          var edgeMat = new THREE.LineBasicMaterial({
            color: colorHex,
            transparent: true,
            opacity: 0.3,
          });
          var wireframe = new THREE.LineSegments(edges, edgeMat);
          wireframe.position.copy(blockCenter);
          this.scene.add(wireframe);
          this.blockMeshes.push(wireframe);

          for (var tx = 0; tx < bcx; tx++) {
            for (var ty = 0; ty < bcy; ty++) {
              for (var tz = 0; tz < bcz; tz++) {
                var tPos = new THREE.Vector3(
                  (tx - (bcx - 1) / 2) * threadSpacing,
                  (ty - (bcy - 1) / 2) * threadSpacing,
                  (tz - (bcz - 1) / 2) * threadSpacing
                );
                tPos.add(blockCenter);

                var sphereGeo = new THREE.SphereGeometry(0.05, 12, 12);
                var sphereMat = new THREE.MeshPhysicalMaterial({
                  color: color,
                  emissive: color,
                  emissiveIntensity: 0.15,
                  metalness: 0.1,
                  roughness: 0.3,
                });
                var sphere = new THREE.Mesh(sphereGeo, sphereMat);
                sphere.position.copy(tPos);

                sphere.userData = {
                  globalIdx: globalIdx,
                  blockX: gx, blockY: gy, blockZ: gz,
                  threadX: tx, threadY: ty, threadZ: tz,
                  gridDimX: gcx, gridDimY: gcy, gridDimZ: gcz,
                  blockDimX: bcx, blockDimY: bcy, blockDimZ: bcz,
                };

                this.scene.add(sphere);
                this.threadMeshes.push(sphere);
                globalIdx++;
              }
            }
          }
        }
      }
    }

    this.buildLegend();
  };

  CudaGrid3D.prototype.initControls = function () {
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 2;
    this.controls.maxDistance = 15;
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  };

  CudaGrid3D.prototype.initHover = function () {
    var self = this;
    var el = this.renderer.domElement;

    el.addEventListener('mousemove', function (event) {
      var rect = el.getBoundingClientRect();
      self.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      self.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      self.raycaster.setFromCamera(self.mouse, self.camera);
      var intersects = self.raycaster.intersectObjects(self.threadMeshes);

      if (intersects.length > 0) {
        var hit = intersects[0].object;
        if (self.hovered !== hit) {
          self.hovered = hit;
          self.showInfo(hit);
        }
      } else {
        if (self.hovered !== null) {
          self.hovered = null;
          self.hideInfo();
        }
      }
    });

    el.addEventListener('mouseleave', function () {
      self.hovered = null;
      self.hideInfo();
    });
  };

  CudaGrid3D.prototype.showInfo = function (mesh) {
    var d = mesh.userData;
    this.info.innerHTML =
      "<div class='cg-tt-label'>Thread #" + d.globalIdx + "</div>" +
      "<div class='cg-tt-detail'>Block: (" + d.blockX + ", " + d.blockY + ", " + d.blockZ + ")  Thread: (" + d.threadX + ", " + d.threadY + ", " + d.threadZ + ")</div>" +
      "<div class='cg-tt-detail'>gridDim=(" + d.gridDimX + "," + d.gridDimY + "," + d.gridDimZ + ")  blockDim=(" + d.blockDimX + "," + d.blockDimY + "," + d.blockDimZ + ")</div>";
    this.info.style.display = 'block';
  };

  CudaGrid3D.prototype.hideInfo = function () {
    this.info.style.display = 'none';
  };

  CudaGrid3D.prototype.clearGrid = function () {
    var i;
    for (i = 0; i < this.threadMeshes.length; i++) {
      var m = this.threadMeshes[i];
      this.scene.remove(m);
      m.geometry.dispose();
      m.material.dispose();
    }
    for (i = 0; i < this.blockMeshes.length; i++) {
      var b = this.blockMeshes[i];
      this.scene.remove(b);
      b.geometry.dispose();
      b.material.dispose();
    }
    this.threadMeshes = [];
    this.blockMeshes = [];
  };

  CudaGrid3D.prototype.buildLegend = function () {
    var legend = document.getElementById('cg-legend');
    if (!legend) return;
    var total = this.gx * this.gy * this.gz;
    var html = '';
    for (var i = 0; i < total; i++) {
      var color = BLOCK_COLORS[i % BLOCK_COLORS.length];
      html += '<span><span class="cg-legend-dot" style="background:' + color + '"></span>Block ' + i + '</span>';
    }
    legend.innerHTML = html;
  };

  CudaGrid3D.prototype.setupUI = function () {
    var self = this;
    var sliders = {
      gx: document.getElementById('cg-slider-gx'),
      gy: document.getElementById('cg-slider-gy'),
      gz: document.getElementById('cg-slider-gz'),
      bx: document.getElementById('cg-slider-bx'),
      by: document.getElementById('cg-slider-by'),
      bz: document.getElementById('cg-slider-bz'),
    };
    var vals = {
      gx: document.getElementById('cg-slider-gx-val'),
      gy: document.getElementById('cg-slider-gy-val'),
      gz: document.getElementById('cg-slider-gz-val'),
      bx: document.getElementById('cg-slider-bx-val'),
      by: document.getElementById('cg-slider-by-val'),
      bz: document.getElementById('cg-slider-bz-val'),
    };

    function update() {
      var ngx = parseInt(sliders.gx.value);
      var ngy = parseInt(sliders.gy.value);
      var ngz = parseInt(sliders.gz.value);
      var nbx = parseInt(sliders.bx.value);
      var nby = parseInt(sliders.by.value);
      var nbz = parseInt(sliders.bz.value);

      var total = ngx * ngy * ngz * nbx * nby * nbz;
      if (total > 3000) {
        return;
      }

      self.gx = ngx; self.gy = ngy; self.gz = ngz;
      self.bx = nbx; self.by = nby; self.bz = nbz;

      vals.gx.textContent = ngx;
      vals.gy.textContent = ngy;
      vals.gz.textContent = ngz;
      vals.bx.textContent = nbx;
      vals.by.textContent = nby;
      vals.bz.textContent = nbz;

      self.clearGrid();
      self.buildGrid();
    }

    for (var key in sliders) {
      if (sliders.hasOwnProperty(key)) {
        sliders[key].addEventListener('input', update);
      }
    }
  };

  CudaGrid3D.prototype.resize = function () {
    var w = this.container.clientWidth;
    var h = this.container.clientHeight;
    if (w === 0 || h === 0) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  };

  CudaGrid3D.prototype.animate = function () {
    var self = this;
    requestAnimationFrame(function () { self.animate(); });
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  var container = document.getElementById('cg-container');
  if (container) { new CudaGrid3D(container); }
})();
