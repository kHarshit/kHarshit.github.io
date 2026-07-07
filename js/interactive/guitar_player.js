import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

(function () {
  'use strict';

  var STRING_NAMES = ['E2','A2','D3','G3','B3','E4'];
  var STRING_FREQ = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63];

  var SCALE_LENGTH = 14.0;
  var FRET_POS = [0];
  for (var n_ = 1; n_ <= 22; n_++) {
    FRET_POS.push(-0.02 + SCALE_LENGTH * (1 - Math.pow(2, -n_ / 12)));
  }
  var NUM_FRETS = FRET_POS.length - 1;
  var STRING_SPACING = 0.55;
  var NUM_STRINGS = 6;
  var FB_WIDTH = 3.8;
  var FB_LENGTH = FRET_POS[FRET_POS.length-1] + 0.5;
  var FB_THICK = 0.36;

  var CHORDS = {
    'C':  { name:'C Major', fingering:[-1,3,2,0,1,0], notes:['C4','C3','E3','G3','C4','E4'] },
    'D':  { name:'D Major', fingering:[-1,-1,0,2,3,2], notes:['D3','D4','F#4','A3','D4','F#4'] },
    'E':  { name:'E Major', fingering:[0,2,2,1,0,0], notes:['E2','B2','E3','G#3','B3','E4'] },
    'G':  { name:'G Major', fingering:[3,2,0,0,0,3], notes:['G2','B2','D3','G3','B3','G4'] },
    'A':  { name:'A Major', fingering:[-1,0,2,2,2,0], notes:['A2','A3','C#4','E3','A3','E4'] },
    'Am': { name:'A Minor', fingering:[-1,0,2,2,1,0], notes:['A2','A3','C4','E3','A3','E4'] },
    'Dm': { name:'D Minor', fingering:[-1,-1,0,2,3,1], notes:['D3','F3','A3','D4','F4','A4'] },
    'Em': { name:'E Minor', fingering:[0,2,2,0,0,0], notes:['E2','B2','E3','G3','B3','E4'] },
    'E7': { name:'E Dominant 7', fingering:[0,2,0,1,0,0], notes:['E2','B2','D3','G#3','B3','E4'] },
    'A7': { name:'A Dominant 7', fingering:[-1,0,2,0,2,0], notes:['A2','A3','E4','G3','A3','E4'] },
    'D7': { name:'D Dominant 7', fingering:[-1,-1,0,2,1,2], notes:['D3','F#3','A3','D4','F#4','C5'] },
    'G7': { name:'G Dominant 7', fingering:[3,2,0,0,0,1], notes:['G2','B2','D3','G3','B3','F4'] }
  };

  var NOTE_FREQ_MAP = {
    'C2':65.41,'C#2':69.30,'D2':73.42,'D#2':77.78,'E2':82.41,
    'F2':87.31,'F#2':92.50,'G2':98.00,'G#2':103.83,'A2':110.00,
    'A#2':116.54,'B2':123.47,'C3':130.81,'C#3':138.59,'D3':146.83,
    'D#3':155.56,'E3':164.81,'F3':174.61,'F#3':185.00,'G3':196.00,
    'G#3':207.65,'A3':220.00,'A#3':233.08,'B3':246.94,'C4':261.63,
    'C#4':277.18,'D4':293.66,'D#4':311.13,'E4':329.63,'F4':349.23,
    'F#4':369.99,'G4':392.00,'G#4':415.30,'A4':440.00,'A#4':466.16,
    'B4':493.88,'C5':523.25,'C#5':554.37,'D5':587.33,'D#5':622.25,
    'E5':659.25,'F5':698.46,'F#5':739.99,'G5':783.99,'G#5':830.61,
    'A5':880.00
  };

  var ACTIVE_COLOR = new THREE.Color('#20B2AA');
  var HOVER_COLOR = new THREE.Color('#5EEAD4');
  var STRING_START = -0.8;
  var STRING_END = 14.0;
  var BODY_NECK_Z = 6.5;
  var BODY_BW = 3.8;
  var BODY_BH = 9.0;
  var BODY_BOTTOM_Z = BODY_NECK_Z + BODY_BH;
  var HEADSTOCK_END = -0.25;
  var HEADSTOCK_LENGTH = 1.4;
  var STRING_LENGTH = STRING_END - STRING_START;

  function getStringX(stringIdx) {
    var totalWidth = (NUM_STRINGS - 1) * STRING_SPACING;
    return -totalWidth / 2 + stringIdx * STRING_SPACING;
  }

  function getFretZ(fretIdx) {
    return FRET_POS[fretIdx];
  }

  function noteFromStringFret(stringIdx, fret) {
    var base = STRING_NAMES[stringIdx];
    var noteIdx = [
      'C','C#','D','D#','E','F','F#','G','G#','A','A#','B'
    ];
    var baseNote = base.replace(/\d/, '');
    var baseOct = parseInt(base.charAt(base.length-1));
    var bi = noteIdx.indexOf(baseNote);
    var total = bi + fret;
    var oct = baseOct + Math.floor(total / 12);
    var n = noteIdx[total % 12];
    return n + oct;
  }

  function freqFromNote(noteStr) {
    return NOTE_FREQ_MAP[noteStr] || 440;
  }

  function Guitar3D(container) {
    this.container = container;
    this.nameEl = document.getElementById('gp3-name');
    this.notesEl = document.getElementById('gp3-notes');
    this.audioCtx = null;
    this.oscillators = [];
    this.gains = [];
    this.activeChord = null;
    this.fingerDots = [];
    this.stringMeshes = [];
    this.fretLines = [];
    this.allKeyMeshes = [];

    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.mouseDownPos = null;
    this.hoverMesh = null;
    this.hoveredString = undefined;

    this.initScene();
    this.initLights();
    this.buildBody();
    this.buildFretboard();
    this.buildStrings();
    this.buildFrets();
    this.buildFretMarkers();
    this.buildHeadstock();
    this.initControls();
    this.initClick();
    this.initHover();
    this.setupUI();
    this.animate();
  }

  Guitar3D.prototype.initScene = function () {
    var w = this.container.clientWidth;
    var h = this.container.clientHeight;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf5f0e8);
    this.camera = new THREE.PerspectiveCamera(32, w / h, 0.1, 45);
    this.camera.position.set(-10, 14, 22);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.4;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.container.appendChild(this.renderer.domElement);

    var self = this;
    this.resizeObserver = new ResizeObserver(function () { self.resize(); });
    this.resizeObserver.observe(this.container);
  };

  Guitar3D.prototype.initLights = function () {
    this.scene.add(new THREE.AmbientLight(0x3a3530, 0.5));
    var hemi = new THREE.HemisphereLight(0xeeddcc, 0x554433, 0.4);
    this.scene.add(hemi);
    var key = new THREE.DirectionalLight(0xfff0dd, 2.0);
    key.position.set(5, 7, 6);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    this.scene.add(key);
    var fill = new THREE.DirectionalLight(0xddaa77, 0.3);
    fill.position.set(-3, 2, -4);
    this.scene.add(fill);
  };

  Guitar3D.prototype.buildBody = function () {
    var bw = BODY_BW, bh = BODY_BH;
    var shape = new THREE.Shape();
    shape.moveTo(0, 0);
    shape.bezierCurveTo(0.12*bw, 0.03*bh, 0.32*bw, 0.08*bh, 0.50*bw, 0.16*bh);
    shape.bezierCurveTo(0.62*bw, 0.22*bh, 0.68*bw, 0.28*bh, 0.70*bw, 0.36*bh);
    shape.bezierCurveTo(0.68*bw, 0.44*bh, 0.54*bw, 0.50*bh, 0.54*bw, 0.56*bh);
    shape.bezierCurveTo(0.56*bw, 0.64*bh, 0.85*bw, 0.70*bh, bw, 0.82*bh);
    shape.bezierCurveTo(0.88*bw, 0.94*bh, 0.42*bw, 0.98*bh, 0, bh);
    shape.bezierCurveTo(-0.42*bw, 0.98*bh, -0.88*bw, 0.94*bh, -bw, 0.82*bh);
    shape.bezierCurveTo(-0.85*bw, 0.70*bh, -0.56*bw, 0.64*bh, -0.54*bw, 0.56*bh);
    shape.bezierCurveTo(-0.54*bw, 0.50*bh, -0.68*bw, 0.44*bh, -0.70*bw, 0.36*bh);
    shape.bezierCurveTo(-0.68*bw, 0.28*bh, -0.62*bw, 0.22*bh, -0.50*bw, 0.16*bh);
    shape.bezierCurveTo(-0.32*bw, 0.08*bh, -0.12*bw, 0.03*bh, 0, 0);

    // Cut the sound hole through the body
    var soundHolePath = new THREE.Path();
    soundHolePath.absarc(0, HEADSTOCK_END + FB_LENGTH + 0.95 - BODY_NECK_Z, 0.90, 0, Math.PI * 2, false);
    shape.holes.push(soundHolePath);

    var bodyDepth = 1.5;
    var geom = new THREE.ExtrudeGeometry(shape, {
      depth: bodyDepth, bevelEnabled: true,
      bevelThickness: 0.12, bevelSize: 0.06, bevelSegments: 10
    });
    geom.translate(0, 0, -bodyDepth / 2);

    var mat = new THREE.MeshPhysicalMaterial({
      color: 0xd4a056, roughness: 0.5, metalness: 0.0,
      clearcoat: 0.15, clearcoatRoughness: 0.3,
      side: THREE.DoubleSide
    });
    var bodyTopY = -0.12;
    var bodyY = bodyTopY - bodyDepth / 2;
    var body = new THREE.Mesh(geom, mat);
    body.rotation.x = Math.PI / 2;
    body.position.set(0, bodyY, BODY_NECK_Z);
    body.receiveShadow = true;
    this.scene.add(body);

    var faceY = bodyTopY;
    var ringMat = new THREE.MeshPhysicalMaterial({
      color: 0xc4923e, roughness: 0.4, metalness: 0.15
    });

    var holeZ = HEADSTOCK_END + FB_LENGTH + 1.0;
    var holeGeom = new THREE.RingGeometry(0.78, 0.90, 28);
    var hole = new THREE.Mesh(holeGeom, ringMat);
    hole.rotation.x = -Math.PI / 2;
    hole.position.set(0, faceY + 0.003, holeZ);
    this.scene.add(hole);

    // Dark backplate so the hole reads as a cavity, not a through-hole
    var backPlateMat = new THREE.MeshPhysicalMaterial({
      color: 0x1a1410, roughness: 0.9, metalness: 0.0,
      side: THREE.DoubleSide
    });
    var backGeom = new THREE.CircleGeometry(1.0, 28);
    var backPlate = new THREE.Mesh(backGeom, backPlateMat);
    backPlate.rotation.x = -Math.PI / 2;
    backPlate.position.set(0, bodyTopY - bodyDepth + 0.001, holeZ);
    this.scene.add(backPlate);

    var bridgeMat = new THREE.MeshPhysicalMaterial({
      color: 0x2c1810, roughness: 0.7, metalness: 0.0
    });
    var bridge = new THREE.Mesh(
      new THREE.BoxGeometry(0.55, 0.04, 0.15), bridgeMat
    );
    bridge.position.set(0, faceY + 0.02, BODY_NECK_Z + bh * 0.82);
    this.scene.add(bridge);

    var saddleMat = new THREE.MeshPhysicalMaterial({
      color: 0xf5f0e0, roughness: 0.3, metalness: 0.1
    });
    var saddle = new THREE.Mesh(
      new THREE.BoxGeometry(0.45, 0.025, 0.025), saddleMat
    );
    saddle.position.set(0, faceY + 0.04, BODY_NECK_Z + bh * 0.83);
    this.scene.add(saddle);
  };

  Guitar3D.prototype.buildFretboard = function () {
    var hw = FB_WIDTH / 2;
    var thick = FB_THICK;
    var mat = new THREE.MeshPhysicalMaterial({
      color: 0x5c3a1e, roughness: 0.6, metalness: 0.0
    });

    // Neck portion: thick curved D-shape back
    var neckLen = BODY_NECK_Z - HEADSTOCK_END;
    var neckShape = new THREE.Shape();
    neckShape.moveTo(-hw, 0);
    neckShape.lineTo(hw, 0);
    neckShape.bezierCurveTo(hw, -thick*0.2, hw*0.7, -thick*2.0, 0, -thick*2.5);
    neckShape.bezierCurveTo(-hw*0.7, -thick*2.0, -hw, -thick*0.2, -hw, 0);
    var neckGeom = new THREE.ExtrudeGeometry(neckShape, {
      depth: neckLen, bevelEnabled: true,
      bevelThickness: 0.03, bevelSize: 0.02, bevelSegments: 6
    });
    neckGeom.translate(0, 0, HEADSTOCK_END);
    var neckBoard = new THREE.Mesh(neckGeom, mat);
    neckBoard.receiveShadow = true;
    this.scene.add(neckBoard);

    // Body portion: thin flat back over the body
    var bodyLen = FB_LENGTH - neckLen;
    var bodyShape = new THREE.Shape();
    bodyShape.moveTo(-hw, 0);
    bodyShape.lineTo(hw, 0);
    bodyShape.lineTo(hw, -0.12);
    bodyShape.lineTo(-hw, -0.12);
    var bodyGeom = new THREE.ExtrudeGeometry(bodyShape, {
      depth: bodyLen, bevelEnabled: true,
      bevelThickness: 0.02, bevelSize: 0.01, bevelSegments: 4
    });
    bodyGeom.translate(0, 0, BODY_NECK_Z);
    var bodyBoard = new THREE.Mesh(bodyGeom, mat);
    bodyBoard.receiveShadow = true;
    this.scene.add(bodyBoard);

    var nutGeom = new THREE.BoxGeometry(FB_WIDTH - 0.2, 0.06, 0.08);
    var nutMat = new THREE.MeshPhysicalMaterial({
      color: 0xf0e8d8, roughness: 0.4, metalness: 0.0
    });
    var nut = new THREE.Mesh(nutGeom, nutMat);
    nut.position.set(0, 0.02, -0.02);
    this.scene.add(nut);
  };

  Guitar3D.prototype.buildFrets = function () {
    var fretMat = new THREE.MeshPhysicalMaterial({
      color: 0xe8e0d0, roughness: 0.8, metalness: 0.0, transparent: true, opacity: 0.35
    });
    for (var i = 1; i < FRET_POS.length; i++) {
      var h = 0.025;
      var geom = new THREE.BoxGeometry(FB_WIDTH - 0.2, h, 0.02);
      var mesh = new THREE.Mesh(geom, fretMat);
      mesh.position.set(0, 0.02, FRET_POS[i]);
      this.scene.add(mesh);
      this.fretLines.push(mesh);
    }
  };

  Guitar3D.prototype.buildStrings = function () {
    var self = this;
    var stringMat = new THREE.MeshPhysicalMaterial({
      color: 0xddcbb8, roughness: 0.4, metalness: 0.05
    });

    for (var s = 0; s < NUM_STRINGS; s++) {
      var r = 0.025 + (5 - s) * 0.006;
      var geom = new THREE.CylinderGeometry(r, r, STRING_LENGTH, 6);
      var mat = stringMat.clone();
      var mesh = new THREE.Mesh(geom, mat);
      var sx = getStringX(s);
      mesh.rotation.x = Math.PI / 2;
      mesh.position.set(sx, 0.01, (STRING_START + STRING_END) / 2);
      this.scene.add(mesh);
      this.stringMeshes.push(mesh);
      mesh.userData = { stringIdx: s, isString: true };
      this.allKeyMeshes.push(mesh);
    }
  };

  Guitar3D.prototype.buildFretMarkers = function () {
    var self = this;
    var dotMat = new THREE.MeshPhysicalMaterial({
      color: 0xe8e0d0, roughness: 0.8, metalness: 0.0, transparent: true, opacity: 0.35
    });
    var markerFrets = [3, 5, 7, 9, 12];
    markerFrets.forEach(function (f) {
      if (f >= FRET_POS.length) return;
      if (f === 12) {
        var d1 = new THREE.Mesh(new THREE.CircleGeometry(0.06, 12), dotMat);
        d1.position.set(-0.14, 0.02, FRET_POS[f] - 0.12);
        d1.rotation.x = -Math.PI / 2;
        self.scene.add(d1);
        var d2 = new THREE.Mesh(new THREE.CircleGeometry(0.06, 12), dotMat);
        d2.position.set(0.14, 0.02, FRET_POS[f] - 0.12);
        d2.rotation.x = -Math.PI / 2;
        self.scene.add(d2);
      } else {
        var dot = new THREE.Mesh(new THREE.CircleGeometry(0.07, 12), dotMat);
        dot.position.set(0, 0.02, FRET_POS[f] - 0.12);
        dot.rotation.x = -Math.PI / 2;
        self.scene.add(dot);
      }
    });
  };

  Guitar3D.prototype.buildHeadstock = function () {
    var hsMat = new THREE.MeshPhysicalMaterial({
      color: 0x5c3a1e, roughness: 0.5, metalness: 0.0
    });
    var taper = 0.7;
    var hw = FB_WIDTH * 0.75;
    var shape = new THREE.Shape();
    shape.moveTo(-hw / 2, 0);
    shape.lineTo(hw / 2, 0);
    shape.lineTo(hw * taper / 2, HEADSTOCK_LENGTH);
    shape.lineTo(-hw * taper / 2, HEADSTOCK_LENGTH);
    shape.closePath();

    var geom = new THREE.ExtrudeGeometry(shape, {
      depth: FB_THICK, bevelEnabled: true,
      bevelThickness: 0.02, bevelSize: 0.01, bevelSegments: 4
    });
    geom.translate(0, 0, -FB_THICK / 2);
    var hs = new THREE.Mesh(geom, hsMat);
    hs.rotation.x = -Math.PI / 2;
    hs.position.set(0, -FB_THICK / 2, HEADSTOCK_END);
    this.scene.add(hs);
  };

  Guitar3D.prototype.buildTuners = function () {
    var tunerMat = new THREE.MeshPhysicalMaterial({
      color: 0x888, roughness: 0.3, metalness: 0.7
    });
    var postMat = new THREE.MeshPhysicalMaterial({
      color: 0x666, roughness: 0.4, metalness: 0.5
    });
    var knobMat = new THREE.MeshPhysicalMaterial({
      color: 0xddd, roughness: 0.2, metalness: 0.3
    });

    for (var s = 0; s < NUM_STRINGS; s++) {
      var sx = getStringX(s);
      var side = s < 3 ? 1 : -1;
      var postEnd = HEADSTOCK_END + HEADSTOCK_LENGTH * 0.3 + s * 0.06;

      var post = new THREE.Mesh(
        new THREE.CylinderGeometry(0.02, 0.02, 0.15, 6), postMat
      );
      post.position.set(sx, 0.08, postEnd);
      post.rotation.x = Math.PI / 2;
      this.scene.add(post);

      var peg = new THREE.Mesh(
        new THREE.CylinderGeometry(0.015, 0.015, 0.1, 6), tunerMat
      );
      peg.position.set(sx + side * 0.18, 0.06, postEnd);
      peg.rotation.z = Math.PI / 2 * side;
      this.scene.add(peg);

      var knob = new THREE.Mesh(
        new THREE.SphereGeometry(0.025, 6, 6), knobMat
      );
      knob.position.set(sx + side * 0.26, 0.06, postEnd);
      this.scene.add(knob);
    }
  };

  Guitar3D.prototype.showChord = function (chordId) {
    var chord = CHORDS[chordId];
    if (!chord) return;
    this.activeChord = chordId;
    this.nameEl.textContent = chord.name;
    this.notesEl.textContent = chord.notes ? chord.notes.join(' \u00b7 ') : '';

    this.clearChord();
    this.playSound(chord.notes);

    var self = this;
    var dotMat = new THREE.MeshPhysicalMaterial({
      color: 0x20b2aa, roughness: 0.6, metalness: 0.0,
      transparent: true, opacity: 0.6
    });

    chord.fingering.forEach(function (fret, si) {
      if (fret < 0) {
        self.highlightString(si, true, true);
        return;
      }
      if (fret === 0) {
        self.highlightString(si, true);
        return;
      }
      self.highlightString(si, true);
      var sx = getStringX(si);
      var fz = FRET_POS[fret - 1] + (FRET_POS[fret] - FRET_POS[fret - 1]) / 2;
      var dot = new THREE.Mesh(new THREE.SphereGeometry(0.1, 12, 12), dotMat.clone());
      dot.position.set(sx, 0.08, fz);
      self.scene.add(dot);
      self.fingerDots.push(dot);
    });
  };

  Guitar3D.prototype.highlightString = function (stringIdx, on, muted) {
    if (stringIdx < 0 || stringIdx >= this.stringMeshes.length) return;
    var mesh = this.stringMeshes[stringIdx];
    if (on) {
      if (muted) {
        mesh.material.color.set(0xcc3333);
        mesh.material.emissive.setHex(0xcc3333);
        mesh.material.emissiveIntensity = 0.2;
      } else {
        mesh.material.color.copy(ACTIVE_COLOR);
        mesh.material.emissive.copy(ACTIVE_COLOR);
        mesh.material.emissiveIntensity = 0.3;
      }
    } else {
      mesh.material.color.set(0xddcbb8);
      mesh.material.emissive.setHex(0x000000);
      mesh.material.emissiveIntensity = 0;
    }
  };

  Guitar3D.prototype.clearChord = function () {
    var self = this;
    this.fingerDots.forEach(function (d) {
      self.scene.remove(d);
      d.geometry.dispose();
      d.material.dispose();
    });
    this.fingerDots = [];
    this.stringMeshes.forEach(function (m, i) { self.highlightString(i, false); });
  };

  Guitar3D.prototype.playSound = function (notes) {
    this.stopSound();
    if (!this.audioCtx) {
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (this.audioCtx.state === 'suspended') this.audioCtx.resume();
    var ctx = this.audioCtx;
    var now = ctx.currentTime;
    var self = this;
    notes.forEach(function (n) {
      var freq = freqFromNote(n);
      if (!freq) return;
      setTimeout(function () {
        var osc = ctx.createOscillator();
        var gain = ctx.createGain();
        var filter = ctx.createBiquadFilter();
        osc.type = 'sawtooth';
        osc.frequency.value = freq;
        filter.type = 'lowpass';
        filter.frequency.value = 3000;
        filter.Q.value = 0.5;
        gain.gain.setValueAtTime(0, ctx.currentTime);
        gain.gain.linearRampToValueAtTime(0.06, ctx.currentTime + 0.005);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 1.5);
        osc.connect(filter);
        filter.connect(gain);
        gain.connect(ctx.destination);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 1.6);
        self.oscillators.push(osc);
        self.gains.push(gain);
      }, Math.random() * 30);
    });
  };

  Guitar3D.prototype.stopSound = function () {
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

  Guitar3D.prototype.initControls = function () {
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 5;
    this.controls.maxDistance = 35;
    this.controls.minPolarAngle = 0.2;
    this.controls.maxPolarAngle = 1.3;
    this.controls.target.set(0, 0.3, (HEADSTOCK_END + BODY_BOTTOM_Z) / 2);
  };

  Guitar3D.prototype.initHover = function () {
    var self = this;
    var canvas = this.renderer.domElement;
    var hoverMat = new THREE.MeshPhysicalMaterial({
      color: 0x20b2aa, roughness: 0.6, metalness: 0.0,
      transparent: true, opacity: 0.5
    });

    canvas.addEventListener('pointermove', function (e) {
      var rect = canvas.getBoundingClientRect();
      self.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      self.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      self.raycaster.setFromCamera(self.mouse, self.camera);
      var intersects = self.raycaster.intersectObjects(self.stringMeshes);

      if (intersects.length > 0) {
        var hit = intersects[0].object;
        var si = hit.userData.stringIdx;
        if (si === undefined) { self.removeHover(); return; }
        var pt = intersects[0].point;
        var fret = -1;
        for (var i = 1; i < FRET_POS.length; i++) {
          if (pt.z < FRET_POS[i]) { fret = i - 1; break; }
        }
        if (fret < 0) fret = NUM_FRETS - 1;

        var sx = getStringX(si);
        var fz = FRET_POS[fret] + (FRET_POS[fret + 1] - FRET_POS[fret]) / 2;

        if (!self.hoverMesh) {
          self.hoverMesh = new THREE.Mesh(
            new THREE.SphereGeometry(0.09, 10, 10), hoverMat
          );
          self.scene.add(self.hoverMesh);
        }
        self.hoverMesh.position.set(sx, 0.08, fz);
        self.hoverMesh.visible = true;

        if (self.hoveredString !== si) {
          if (self.hoveredString !== undefined) {
            self.highlightString(self.hoveredString, false);
            if (self.activeChord) {
              var chord = CHORDS[self.activeChord];
              if (chord) {
                var f = chord.fingering[self.hoveredString];
                if (f < 0) self.highlightString(self.hoveredString, true, true);
                else self.highlightString(self.hoveredString, true);
              }
            }
          }
          self.highlightString(si, true);
          self.hoveredString = si;
        }

        var note = noteFromStringFret(si, fret);
        self.nameEl.textContent = note + ' (Str ' + (6 - si) + ' Fr ' + fret + ')';
      } else {
        self.removeHover();
      }
    });

    canvas.addEventListener('pointerleave', function () {
      self.removeHover();
    });
  };

  Guitar3D.prototype.removeHover = function () {
    if (this.hoverMesh) this.hoverMesh.visible = false;
    if (this.hoveredString !== undefined) {
      this.highlightString(this.hoveredString, false);
      this.hoveredString = undefined;
    }
    if (this.activeChord) {
      this.refreshChordStrings();
    } else {
      this.nameEl.textContent = '\u2014';
    }
  };

  Guitar3D.prototype.refreshChordStrings = function () {
    var chord = CHORDS[this.activeChord];
    if (!chord) return;
    chord.fingering.forEach(function (fret, si) {
      if (fret < 0) {
        this.highlightString(si, true, true);
      } else {
        this.highlightString(si, true);
      }
    }, this);
  };

  Guitar3D.prototype.initClick = function () {
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
      var intersects = self.raycaster.intersectObjects(self.stringMeshes);
      if (intersects.length > 0) {
        var hit = intersects[0].object;
        var si = hit.userData.stringIdx;
        if (si === undefined) return;
        var pt = intersects[0].point;
        var localZ = pt.z;
        var fret = -1;
        for (var i = 1; i < FRET_POS.length; i++) {
          if (localZ < FRET_POS[i]) { fret = i - 1; break; }
        }
        if (fret < 0) fret = NUM_FRETS - 1;

        var note = noteFromStringFret(si, fret);
        var freq = freqFromNote(note);
        if (freq) {
          self.clearChord();
          self.stopSound();
          self.activeChord = null;
          self.nameEl.textContent = note + ' (Str ' + (6 - si) + ' Fr ' + fret + ')';
          self.notesEl.textContent = '';
          self.container.querySelectorAll('.gp3-btn').forEach(function (b) { b.classList.remove('active'); });
          self.highlightString(si, true);
          self.hoveredString = si;
          self.playSound([note]);
        }
      }
    });
  };

  Guitar3D.prototype.setupUI = function () {
    var self = this;
    this.container.querySelectorAll('.gp3-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var chordId = btn.getAttribute('data-chord');
        if (self.activeChord === chordId) {
          self.clearChord();
          self.stopSound();
          self.activeChord = null;
          self.nameEl.textContent = '\u2014';
          self.notesEl.textContent = '';
          self.container.querySelectorAll('.gp3-btn').forEach(function (b) { b.classList.remove('active'); });
          return;
        }
        self.container.querySelectorAll('.gp3-btn').forEach(function (b) { b.classList.remove('active'); });
        btn.classList.add('active');
        self.showChord(chordId);
      });
    });
  };

  Guitar3D.prototype.resize = function () {
    var w = this.container.clientWidth;
    var h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  };

  Guitar3D.prototype.animate = function () {
    var self = this;
    function loop() {
      requestAnimationFrame(loop);
      self.controls.update();
      self.renderer.render(self.scene, self.camera);
    }
    loop();
  };

  var container = document.getElementById('gp3-container');
  if (container) new Guitar3D(container);
})();
