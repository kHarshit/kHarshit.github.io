import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

(function () {
  'use strict';

  var NOTE_FREQ = {
    'C4':261.63,'C#4':277.18,'D4':293.66,'D#4':311.13,'E4':329.63,
    'F4':349.23,'F#4':369.99,'G4':392.00,'G#4':415.30,'A4':440.00,
    'A#4':466.16,'B4':493.88,'C5':523.25,'C#5':554.37,'D5':587.33,
    'D#5':622.25,'E5':659.25,'F5':698.46,'F#5':739.99,'G5':783.99,
    'G#5':830.61,'A5':880.00,'A#5':932.33,'B5':987.77
  };

  var CHORDS = {
    'C':    { name:'C Major',     notes:['C4','E4','G4'] },
    'D':    { name:'D Major',     notes:['D4','F#4','A4'] },
    'E':    { name:'E Major',     notes:['E4','G#4','B4'] },
    'F':    { name:'F Major',     notes:['F4','A4','C5'] },
    'G':    { name:'G Major',     notes:['G4','B4','D5'] },
    'A':    { name:'A Major',     notes:['A4','C#5','E5'] },
    'B':    { name:'B Major',     notes:['B4','D#5','F#5'] },
    'Am':   { name:'A Minor',     notes:['A4','C5','E5'] },
    'Dm':   { name:'D Minor',     notes:['D4','F4','A4'] },
    'Em':   { name:'E Minor',     notes:['E4','G4','B4'] },
    'Bm':   { name:'B Minor',     notes:['B4','D5','F#5'] },
    'Cmaj7':{ name:'C Major 7',   notes:['C4','E4','G4','B4'] },
    'G7':   { name:'G Dominant 7',notes:['G4','B4','D5','F5'] },
    'D7':   { name:'D Dominant 7',notes:['D4','F#4','A4','C5'] }
  };

  var WHITE_KEYS = [
    { note:'C4', x:0 }, { note:'D4', x:1 }, { note:'E4', x:2 },
    { note:'F4', x:3 }, { note:'G4', x:4 }, { note:'A4', x:5 }, { note:'B4', x:6 },
    { note:'C5', x:7 }, { note:'D5', x:8 }, { note:'E5', x:9 },
    { note:'F5', x:10 }, { note:'G5', x:11 }, { note:'A5', x:12 }, { note:'B5', x:13 }
  ];

  var BLACK_KEYS = [
    { note:'C#4', x:0.7 }, { note:'D#4', x:1.7 },
    { note:'F#4', x:3.7 }, { note:'G#4', x:4.7 }, { note:'A#4', x:5.7 },
    { note:'C#5', x:7.7 }, { note:'D#5', x:8.7 },
    { note:'F#5', x:10.7 }, { note:'G#5', x:11.7 }, { note:'A#5', x:12.7 }
  ];

  var CENTER_OFFSET = 6.5;
  var WHITE_W = 0.92, WHITE_D = 4.0, WHITE_H = 0.22;
  var BLACK_W = 0.54, BLACK_D = 2.4, BLACK_H = 0.40;

  var ACTIVE_COLOR = new THREE.Color('#20B2AA');
  var HOVER_COLOR = new THREE.Color('#5EEAD4');

  function Piano3D(container) {
    this.container = container;
    this.nameEl = document.getElementById('cp3-name');
    this.notesEl = document.getElementById('cp3-notes');
    this.audioCtx = null;
    this.oscillators = [];
    this.gains = [];
    this.activeChord = null;
    this.keyMeshes = {};
    this.keyOrigY = {};
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.mouseDownPos = null;
    this.hoveredNote = null;

    this.initScene();
    this.initLights();
    this.buildKeyboard();
    this.initControls();
    this.initKeyClick();
    this.initHover();
    this.setupUI();
    this.animate();
  }

  Piano3D.prototype.initScene = function () {
    var w = this.container.clientWidth;
    var h = this.container.clientHeight;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf5f0e8);
    this.camera = new THREE.PerspectiveCamera(35, w / h, 0.1, 30);
    this.camera.position.set(4.5, 5.5, 12);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.5;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.container.appendChild(this.renderer.domElement);

    var self = this;
    this.resizeObserver = new ResizeObserver(function () { self.resize(); });
    this.resizeObserver.observe(this.container);
  };

  Piano3D.prototype.initLights = function () {
    this.scene.add(new THREE.AmbientLight(0x3a3530, 0.4));

    var hemi = new THREE.HemisphereLight(0xeeddcc, 0x554433, 0.5);
    this.scene.add(hemi);

    var key = new THREE.DirectionalLight(0xfff0dd, 1.8);
    key.position.set(4, 8, 6);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.near = 0.5;
    key.shadow.camera.far = 20;
    key.shadow.camera.left = -10;
    key.shadow.camera.right = 10;
    key.shadow.camera.top = 10;
    key.shadow.camera.bottom = -10;
    this.scene.add(key);

    var fill = new THREE.DirectionalLight(0xddaa77, 0.25);
    fill.position.set(-4, 2, -5);
    this.scene.add(fill);
  };

  Piano3D.prototype.buildKeyboard = function () {
    var self = this;
    var whiteRest = new THREE.Color('#f0f0f0');
    var blackRest = new THREE.Color('#1a1a1a');

    var whiteMat = new THREE.MeshPhysicalMaterial({
      color: whiteRest, metalness: 0.02, roughness: 0.35,
      clearcoat: 0.05, envMapIntensity: 0.3
    });

    var blackMat = new THREE.MeshPhysicalMaterial({
      color: blackRest, metalness: 0.15, roughness: 0.3,
      clearcoat: 0.1, envMapIntensity: 0.4
    });

    var whiteGeom = new THREE.BoxGeometry(WHITE_W, WHITE_H, WHITE_D);
    var blackGeom = new THREE.BoxGeometry(BLACK_W, BLACK_H, BLACK_D);

    WHITE_KEYS.forEach(function (k) {
      var m = new THREE.Mesh(whiteGeom.clone(), whiteMat.clone());
      var cx = k.x - CENTER_OFFSET;
      m.position.set(cx, WHITE_H / 2, 0);
      m.castShadow = true;
      m.receiveShadow = true;
      m.userData = { note: k.note, isBlack: false, restY: WHITE_H / 2, restColor: whiteRest.clone() };
      self.scene.add(m);
      self.keyMeshes[k.note] = m;
      self.keyOrigY[k.note] = WHITE_H / 2;
    });

    BLACK_KEYS.forEach(function (k) {
      var m = new THREE.Mesh(blackGeom.clone(), blackMat.clone());
      var cx = k.x - CENTER_OFFSET;
      m.position.set(cx, WHITE_H + BLACK_H / 2, -WHITE_D / 2 + BLACK_D / 2 + 0.1);
      m.castShadow = true;
      m.receiveShadow = true;
      m.userData = { note: k.note, isBlack: true, restY: WHITE_H + BLACK_H / 2, restColor: blackRest.clone() };
      self.scene.add(m);
      self.keyMeshes[k.note] = m;
      self.keyOrigY[k.note] = WHITE_H + BLACK_H / 2;
    });
  };

  Piano3D.prototype.initControls = function () {
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 2;
    this.controls.maxDistance = 22;
    this.controls.minPolarAngle = 0.3;
    this.controls.maxPolarAngle = 1.2;
    this.controls.target.set(0, 0.5, 0);
  };

  Piano3D.prototype.lightKey = function (note, on) {
    var mesh = this.keyMeshes[note];
    if (!mesh) return;
    mesh.userData.chordActive = on;
    if (on) {
      mesh.material.color.copy(ACTIVE_COLOR);
      mesh.material.emissive.copy(ACTIVE_COLOR);
      mesh.material.emissiveIntensity = 0.3;
      mesh.position.y = this.keyOrigY[note] - 0.04;
    } else if (this.hoveredNote === note) {
      mesh.material.color.copy(HOVER_COLOR);
      mesh.material.emissive.copy(HOVER_COLOR);
      mesh.material.emissiveIntensity = 0.15;
      mesh.position.y = this.keyOrigY[note] - 0.02;
    } else {
      mesh.material.color.copy(mesh.userData.restColor);
      mesh.material.emissive.setHex(0x000000);
      mesh.material.emissiveIntensity = 0;
      mesh.position.y = this.keyOrigY[note];
    }
  };

  Piano3D.prototype.showChord = function (chordId) {
    var chord = CHORDS[chordId];
    if (!chord) return;
    this.activeChord = chordId;
    this.nameEl.textContent = chord.name;
    this.notesEl.textContent = chord.notes.join(' \u00b7 ');
    this.clearAllKeys();
    this.playSound(chord.notes);
    var self = this;
    chord.notes.forEach(function (n) { self.lightKey(n, true); });
  };

  Piano3D.prototype.clearAllKeys = function () {
    var self = this;
    Object.keys(this.keyMeshes).forEach(function (note) { self.lightKey(note, false); });
  };

  Piano3D.prototype.playSound = function (notes) {
    this.stopSound();
    if (!this.audioCtx) {
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (this.audioCtx.state === 'suspended') this.audioCtx.resume();
    var ctx = this.audioCtx;
    var now = ctx.currentTime;
    var self = this;
    notes.forEach(function (n) {
      var freq = NOTE_FREQ[n];
      if (!freq) return;
      var osc = ctx.createOscillator();
      var gain = ctx.createGain();
      var filter = ctx.createBiquadFilter();
      osc.type = 'triangle';
      osc.frequency.value = freq;
      filter.type = 'lowpass';
      filter.frequency.value = 4000;
      filter.Q.value = 0.5;
      gain.gain.setValueAtTime(0, now);
      gain.gain.linearRampToValueAtTime(0.2, now + 0.01);
      gain.gain.linearRampToValueAtTime(0.15, now + 0.3);
      gain.gain.linearRampToValueAtTime(0, now + 2.0);
      osc.connect(filter);
      filter.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + 2.2);
      self.oscillators.push(osc);
      self.gains.push(gain);
    });
  };

  Piano3D.prototype.stopSound = function () {
    var ctx = this.audioCtx;
    var now = ctx ? ctx.currentTime : 0;
    var self = this;
    this.oscillators.forEach(function (osc, i) {
      try {
        self.gains[i].gain.linearRampToValueAtTime(0, now + 0.05);
        osc.stop(now + 0.06);
      } catch(e) {}
    });
    this.oscillators = [];
    this.gains = [];
  };

  Piano3D.prototype.setupUI = function () {
    var self = this;
    this.container.querySelectorAll('.cp3-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var chordId = btn.getAttribute('data-chord');
        if (self.activeChord === chordId) {
          self.clearAllKeys();
          self.stopSound();
          self.activeChord = null;
          self.nameEl.textContent = '\u2014';
          self.notesEl.textContent = '';
          self.container.querySelectorAll('.cp3-btn').forEach(function (b) { b.classList.remove('active'); });
          return;
        }
        self.container.querySelectorAll('.cp3-btn').forEach(function (b) { b.classList.remove('active'); });
        btn.classList.add('active');
        self.showChord(chordId);
      });
    });
  };

  Piano3D.prototype.initKeyClick = function () {
    var self = this;
    var canvas = this.renderer.domElement;

    canvas.addEventListener('pointerdown', function (e) {
      self.mouseDownPos = { x: e.clientX, y: e.clientY };
    });

    canvas.addEventListener('pointerup', function (e) {
      if (!self.mouseDownPos) return;
      var dx = e.clientX - self.mouseDownPos.x;
      var dy = e.clientY - self.mouseDownPos.y;
      self.mouseDownPos = null;
      if (dx * dx + dy * dy > 16) return;

      var rect = canvas.getBoundingClientRect();
      self.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      self.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      self.raycaster.setFromCamera(self.mouse, self.camera);

      var meshes = [];
      Object.keys(self.keyMeshes).forEach(function (k) { meshes.push(self.keyMeshes[k]); });

      var intersects = self.raycaster.intersectObjects(meshes);
      if (intersects.length > 0) {
        var hit = intersects[0].object;
        var note = hit.userData.note;
        if (note) {
          self.clearAllKeys();
          self.stopSound();
          self.activeChord = null;
          self.nameEl.textContent = note;
          self.notesEl.textContent = '';
          self.container.querySelectorAll('.cp3-btn').forEach(function (b) { b.classList.remove('active'); });
          self.lightKey(note, true);
          self.playSound([note]);
        }
      }
    });
  };

  Piano3D.prototype.initHover = function () {
    var self = this;
    var canvas = this.renderer.domElement;

    canvas.addEventListener('pointermove', function (e) {
      var rect = canvas.getBoundingClientRect();
      self.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      self.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      self.raycaster.setFromCamera(self.mouse, self.camera);

      var meshes = [];
      Object.keys(self.keyMeshes).forEach(function (k) { meshes.push(self.keyMeshes[k]); });

      var intersects = self.raycaster.intersectObjects(meshes);
      var hitNote = null;
      if (intersects.length > 0) {
        hitNote = intersects[0].object.userData.note;
      }

      if (hitNote !== self.hoveredNote) {
        var prevNote = self.hoveredNote;
        self.hoveredNote = null;
        if (prevNote && prevNote !== self.activeChord &&
            (!self.activeChord || self.keyMeshes[prevNote].userData.chordActive !== true)) {
          self.lightKey(prevNote, false);
        }
        self.hoveredNote = hitNote;
        if (hitNote && hitNote !== self.activeChord &&
            (!self.activeChord || self.keyMeshes[hitNote].userData.chordActive !== true)) {
          var mesh = self.keyMeshes[hitNote];
          if (mesh) {
            mesh.material.color.copy(HOVER_COLOR);
            mesh.material.emissive.copy(HOVER_COLOR);
            mesh.material.emissiveIntensity = 0.15;
            mesh.position.y = self.keyOrigY[hitNote] - 0.02;
          }
        }
      }
    });

    canvas.addEventListener('pointerleave', function () {
      var prevNote = self.hoveredNote;
      self.hoveredNote = null;
      if (prevNote && prevNote !== self.activeChord &&
          (!self.activeChord || self.keyMeshes[prevNote].userData.chordActive !== true)) {
        self.lightKey(prevNote, false);
      }
    });
  };

  Piano3D.prototype.resize = function () {
    var w = this.container.clientWidth;
    var h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  };

  Piano3D.prototype.animate = function () {
    var self = this;
    function loop() {
      requestAnimationFrame(loop);
      self.controls.update();
      self.renderer.render(self.scene, self.camera);
    }
    loop();
  };

  var container = document.getElementById('cp3-container');
  if (container) new Piano3D(container);
})();
