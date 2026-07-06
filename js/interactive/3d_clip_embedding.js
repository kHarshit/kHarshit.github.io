import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

(function () {
  'use strict';

  var MAX_STEPS = 300;
  var PAIRS = [
    { img: '\ud83d\udc15', label: 'dog', text: 'a photo of a dog playing outside' },
    { img: '\ud83c\udfce\ufe0f', label: 'sports car', text: 'a photo of a sports car on the road' },
    { img: '\ud83c\udf05', label: 'sunset', text: 'a photo of a beautiful sunset over the ocean' },
    { img: '\ud83d\udc31', label: 'cat', text: 'a photo of a cat sleeping on a couch' },
    { img: '\ud83c\udf5d', label: 'pasta', text: 'a photo of a plate of pasta with sauce' },
    { img: '\ud83c\udf06', label: 'city', text: 'a photo of a city skyline at night' },
  ];

  var COLORS = ['#20b2aa', '#f59e0b', '#ec4899', '#8b5cf6', '#10b981', '#3b82f6'];
  var TEMPERATURE = 0.5;
  var CLAMP = 4.0;
  var ATTRACT = 0.012;
  var REPEL = 0.004;

  function ClipEmbedding3D(container) {
    this.container = container;
    this.detail = document.getElementById('ce-detail');
    this.ddLabel = document.getElementById('ce-dd-label');
    this.ddCaption = document.getElementById('ce-dd-caption');
    this.stepEl = document.getElementById('ce-step');
    this.lossEl = document.getElementById('ce-loss');
    this.playBtn = document.getElementById('ce-play');
    this.resetBtn = document.getElementById('ce-reset');
    this.speedSlider = document.getElementById('ce-speed');
    this.speedValEl = document.getElementById('ce-speed-val');
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.hovered = null;
    this.hoveredMesh = null;
    this.playing = false;
    this.step = 0;
    this.speed = 1;
    this.accum = 0;

    this.embeddings = [];
    this.allMeshes = [];

    this.initScene();
    this.initLights();
    this.initBackground();
    this.initEmbeddings();
    this.initControls();
    this.initHover();
    this.setupUI();
    this.computeLoss();
    this.animate();
  }

  ClipEmbedding3D.prototype.randomOnSphere = function () {
    var theta = Math.random() * Math.PI * 2;
    var phi = Math.acos(2 * Math.random() - 1);
    var r = 1.5 + Math.random() * 0.4;
    return new THREE.Vector3(
      r * Math.sin(phi) * Math.cos(theta),
      r * Math.sin(phi) * Math.sin(theta) + 1.8,
      r * Math.cos(phi)
    );
  };

  ClipEmbedding3D.prototype.initScene = function () {
    var w = this.container.clientWidth;
    var h = this.container.clientHeight;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 20);
    this.camera.position.set(4.8, 1.0, 5.5);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    this.container.appendChild(this.renderer.domElement);
    this.resizeObserver = new ResizeObserver(this.resize.bind(this));
    this.resizeObserver.observe(this.container);
  };

  ClipEmbedding3D.prototype.initLights = function () {
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    var d = new THREE.DirectionalLight(0xffffff, 2.0);
    d.position.set(4, 8, 6);
    this.scene.add(d);
    var d2 = new THREE.DirectionalLight(0x8888ff, 0.5);
    d2.position.set(-3, 4, -4);
    this.scene.add(d2);
    var d3 = new THREE.DirectionalLight(0xff8844, 0.3);
    d3.position.set(-2, -3, 5);
    this.scene.add(d3);
  };

  ClipEmbedding3D.prototype.initBackground = function () {
    var axes = new THREE.AxesHelper(2);
    axes.position.y = -0.5;
    this.scene.add(axes);
  };

  ClipEmbedding3D.prototype.makeImageTexture = function (symbol, colorHex) {
    var size = 256;
    var c = document.createElement('canvas');
    c.width = size;
    c.height = size;
    var ctx = c.getContext('2d');
    ctx.clearRect(0, 0, size, size);

    ctx.fillStyle = colorHex;
    ctx.fillRect(0, 0, size, size);

    ctx.font = '220px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(symbol, size / 2, size / 2 + 6);

    var tex = new THREE.CanvasTexture(c);
    tex.minFilter = THREE.LinearFilter;
    return tex;
  };

  ClipEmbedding3D.prototype.makeRectTexture = function (text, colorHex) {
    var c = document.createElement('canvas');
    var dpr = 2;
    c.width = 512 * dpr;
    c.height = 160 * dpr;
    var ctx = c.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, 512, 160);

    var r = 20;
    ctx.beginPath();
    ctx.moveTo(r, 0);
    ctx.lineTo(512 - r, 0);
    ctx.quadraticCurveTo(512, 0, 512, r);
    ctx.lineTo(512, 160 - r);
    ctx.quadraticCurveTo(512, 160, 512 - r, 160);
    ctx.lineTo(r, 160);
    ctx.quadraticCurveTo(0, 160, 0, 160 - r);
    ctx.lineTo(0, r);
    ctx.quadraticCurveTo(0, 0, r, 0);
    ctx.closePath();
    ctx.fillStyle = colorHex;
    ctx.fill();

    ctx.fillStyle = '#fff';
    ctx.font = 'bold 72px -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 256, 80);

    var tex = new THREE.CanvasTexture(c);
    tex.minFilter = THREE.LinearFilter;
    return tex;
  };

  ClipEmbedding3D.prototype.initEmbeddings = function () {
    var self = this;

    PAIRS.forEach(function (pair, idx) {
      var imgPos = self.randomOnSphere();
      var txtPos = self.randomOnSphere();
      var color = new THREE.Color(COLORS[idx]);

      var imgTex = self.makeImageTexture(pair.img, COLORS[idx]);
      var imgGeo = new THREE.BoxGeometry(0.65, 0.5, 0.1);
      var imgMat = new THREE.MeshPhysicalMaterial({
        map: imgTex,
        emissive: color,
        emissiveIntensity: 0.1,
        metalness: 0.05,
        roughness: 0.3,
      });
      var imgMesh = new THREE.Mesh(imgGeo, imgMat);
      imgMesh.position.copy(imgPos);
      imgMesh.userData = { type: 'img', pairIdx: idx };
      self.scene.add(imgMesh);
      self.allMeshes.push(imgMesh);

      var txtTex = self.makeRectTexture(pair.label, COLORS[idx]);
      var txtGeo = new THREE.BoxGeometry(0.8, 0.5, 0.1);
      var txtMat = new THREE.MeshPhysicalMaterial({
        map: txtTex,
        emissive: color,
        emissiveIntensity: 0.1,
        metalness: 0.05,
        roughness: 0.3,
      });
      var txtMesh = new THREE.Mesh(txtGeo, txtMat);
      txtMesh.position.copy(txtPos);
      txtMesh.userData = { type: 'txt', pairIdx: idx };
      self.scene.add(txtMesh);
      self.allMeshes.push(txtMesh);

      var lineGeo = new THREE.BufferGeometry().setFromPoints([imgPos, txtPos]);
      var lineMat = new THREE.LineDashedMaterial({
        color: color,
        transparent: true,
        opacity: 0.25,
        dashSize: 0.04,
        gapSize: 0.04,
      });
      var line = new THREE.Line(lineGeo, lineMat);
      line.computeLineDistances();
      self.scene.add(line);

      self.embeddings.push({
        imgMesh: imgMesh,
        txtMesh: txtMesh,
        imgPos: imgPos,
        txtPos: txtPos,
        line: line,
        pairIdx: idx,
      });
    });
  };

  ClipEmbedding3D.prototype.updateLines = function () {
    var self = this;
    this.embeddings.forEach(function (e) {
      var pos = e.line.geometry.attributes.position.array;
      pos[0] = e.imgPos.x;
      pos[1] = e.imgPos.y;
      pos[2] = e.imgPos.z;
      pos[3] = e.txtPos.x;
      pos[4] = e.txtPos.y;
      pos[5] = e.txtPos.z;
      e.line.geometry.attributes.position.needsUpdate = true;
      e.line.computeLineDistances();
    });
  };

  ClipEmbedding3D.prototype.computeLoss = function () {
    var N = this.embeddings.length;
    var tau = TEMPERATURE;
    var imgNorm = [];
    var txtNorm = [];

    for (var i = 0; i < N; i++) {
      imgNorm.push(this.embeddings[i].imgPos.clone().normalize());
      txtNorm.push(this.embeddings[i].txtPos.clone().normalize());
    }

    var sims = [];
    for (var i = 0; i < N; i++) {
      sims[i] = [];
      for (var j = 0; j < N; j++) {
        sims[i][j] = imgNorm[i].dot(txtNorm[j]);
      }
    }

    var loss = 0;
    for (var i = 0; i < N; i++) {
      var rowExp = [];
      var rowSum = 0;
      for (var j = 0; j < N; j++) {
        var v = Math.exp(sims[i][j] / tau);
        rowExp.push(v);
        rowSum += v;
      }
      var prob = rowExp[i] / rowSum;
      loss -= Math.log(Math.max(prob, 1e-10));
    }
    loss /= N;

    this.lossEl.textContent = loss.toFixed(3);
  };

  ClipEmbedding3D.prototype.trainStep = function () {
    var embeddings = this.embeddings;
    var N = embeddings.length;

    for (var i = 0; i < N; i++) {
      var e = embeddings[i];
      var diff = e.txtPos.clone().sub(e.imgPos);
      var dist = diff.length();
      if (dist > 0.001) {
        var dir = diff.normalize();
        e.imgPos.add(dir.clone().multiplyScalar(ATTRACT * dist));
        e.txtPos.add(dir.clone().multiplyScalar(-ATTRACT * dist));
      }
    }

    for (var i = 0; i < N; i++) {
      for (var j = 0; j < N; j++) {
        if (i === j) continue;

        var diff = embeddings[i].imgPos.clone().sub(embeddings[j].txtPos);
        var dist = Math.max(diff.length(), 0.1);
        var force = REPEL / (dist * dist);
        force = Math.min(force, 0.06);
        var dir = diff.normalize();
        embeddings[i].imgPos.add(dir.clone().multiplyScalar(force));
        embeddings[j].txtPos.add(dir.clone().multiplyScalar(-force));

        diff = embeddings[i].txtPos.clone().sub(embeddings[j].imgPos);
        dist = Math.max(diff.length(), 0.1);
        force = REPEL / (dist * dist);
        force = Math.min(force, 0.06);
        dir = diff.normalize();
        embeddings[i].txtPos.add(dir.clone().multiplyScalar(force));
        embeddings[j].imgPos.add(dir.clone().multiplyScalar(-force));
      }
    }

    for (var i = 0; i < N; i++) {
      var e = embeddings[i];
      e.imgPos.x = Math.max(-CLAMP, Math.min(CLAMP, e.imgPos.x));
      e.imgPos.y = Math.max(-CLAMP, Math.min(CLAMP, e.imgPos.y));
      e.imgPos.z = Math.max(-CLAMP, Math.min(CLAMP, e.imgPos.z));
      e.txtPos.x = Math.max(-CLAMP, Math.min(CLAMP, e.txtPos.x));
      e.txtPos.y = Math.max(-CLAMP, Math.min(CLAMP, e.txtPos.y));
      e.txtPos.z = Math.max(-CLAMP, Math.min(CLAMP, e.txtPos.z));

      e.imgMesh.position.copy(e.imgPos);
      e.txtMesh.position.copy(e.txtPos);
    }

    this.updateLines();

    this.step++;
    this.stepEl.textContent = this.step;
    if (this.step >= MAX_STEPS) this.pause();
    this.computeLoss();
  };

  ClipEmbedding3D.prototype.initControls = function () {
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 1.5;
    this.controls.maxDistance = 10;
    this.controls.target.set(0, 1.8, 0);
    this.controls.update();
  };

  ClipEmbedding3D.prototype.initHover = function () {
    var self = this;
    this.container.addEventListener('mousemove', function (e) {
      var rect = self.container.getBoundingClientRect();
      self.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      self.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      self.checkHover();
    });
    this.container.addEventListener('mouseleave', function () { self.hideDetail(); });
  };

  ClipEmbedding3D.prototype.checkHover = function () {
    this.raycaster.setFromCamera(this.mouse, this.camera);
    var intersects = this.raycaster.intersectObjects(this.allMeshes);
    if (intersects.length > 0) {
      var hit = intersects[0].object;
      var ud = hit.userData;
      if (ud.pairIdx !== undefined) {
        var pair = PAIRS[ud.pairIdx];
        this.showDetail(pair, ud);
        if (this.hoveredMesh && this.hoveredMesh !== hit) {
          this.hoveredMesh.material.emissiveIntensity = 0.2;
        }
        hit.material.emissiveIntensity = 0.5;
        this.hoveredMesh = hit;
      }
    } else {
      this.hideDetail();
    }
  };

  ClipEmbedding3D.prototype.showDetail = function (pair, ud) {
    var typeLabel = ud.type === 'img' ? 'Image' : 'Text';
    this.ddLabel.textContent = pair.img + ' ' + pair.label + ' (' + typeLabel + ')';
    this.ddCaption.textContent = '"' + pair.text + '"';
    this.detail.style.display = 'block';
  };

  ClipEmbedding3D.prototype.hideDetail = function () {
    this.detail.style.display = 'none';
    if (this.hoveredMesh) {
      this.hoveredMesh.material.emissiveIntensity = 0.2;
      this.hoveredMesh = null;
    }
  };

  ClipEmbedding3D.prototype.play = function () {
    if (this.step >= MAX_STEPS) this.reset();
    this.playing = true;
    this.playBtn.innerHTML = '\u23f8 Pause';
  };

  ClipEmbedding3D.prototype.pause = function () {
    this.playing = false;
    this.accum = 0;
    this.playBtn.innerHTML = '\u25b6 Play';
  };

  ClipEmbedding3D.prototype.reset = function () {
    var self = this;
    this.pause();
    this.step = 0;
    this.stepEl.textContent = '0';
    this.embeddings.forEach(function (e) {
      var newImgPos = self.randomOnSphere();
      var newTxtPos = self.randomOnSphere();
      e.imgPos.copy(newImgPos);
      e.txtPos.copy(newTxtPos);
      e.imgMesh.position.copy(e.imgPos);
      e.txtMesh.position.copy(e.txtPos);
    });
    this.updateLines();
    this.computeLoss();
  };

  ClipEmbedding3D.prototype.setupUI = function () {
    var self = this;

    this.playBtn.addEventListener('click', function () {
      if (self.playing) {
        self.pause();
      } else {
        self.play();
      }
    });

    this.resetBtn.addEventListener('click', function () {
      self.reset();
    });

    this.speedSlider.addEventListener('input', function () {
      var val = parseInt(self.speedSlider.value, 10);
      self.speed = val / 5;
      self.speedValEl.textContent = self.speed.toFixed(1) + 'x';
    });

    this.speed = parseInt(this.speedSlider.value, 10) / 5;
    this.speedValEl.textContent = this.speed.toFixed(1) + 'x';
  };

  ClipEmbedding3D.prototype.resize = function () {
    var w = this.container.clientWidth;
    var h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  };

  ClipEmbedding3D.prototype.animate = function () {
    var self = this;
    requestAnimationFrame(function () { self.animate(); });

    if (this.playing) {
      this.accum += this.speed;
      while (this.accum >= 1) {
        this.trainStep();
        this.accum -= 1;
      }
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  var container = document.getElementById('ce-container');
  if (container) { new ClipEmbedding3D(container); }
})();
