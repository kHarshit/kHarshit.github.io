import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// 3D Interactive Guitar — builds a playable acoustic guitar in Three.js
// with fretboard, strings, headstock, tuners, body, and chord playback.
(function () {
  'use strict';

  // --- String data ---
  var STRING_NAMES = ['E2','A2','D3','G3','B3','E4'];
  var STRING_FREQ = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63];

  // --- Fretboard geometry ---
  var SCALE_LENGTH = 14.0;                  // vibrating string length
  var FRET_POS = [0];
  for (var n_ = 1; n_ <= 22; n_++) {
    FRET_POS.push(-0.02 + SCALE_LENGTH * (1 - Math.pow(2, -n_ / 12)));
  }
  var NUM_FRETS = FRET_POS.length - 1;
  var STRING_SPACING = 0.55;                // centre-to-centre
  var NUM_STRINGS = 6;
  var FB_WIDTH = 3.0;                       // fretboard width
  var FB_LENGTH = FRET_POS[FRET_POS.length-1] + 0.5;  // overhang past last fret
  var FB_THICK = 0.36;                      // fretboard thickness

  // --- Chord library ---
  // fingering: -1 = muted, 0 = open, 1-4 = fret number
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

  // --- Note → frequency lookup ---
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

  // --- Colours ---
  var ACTIVE_COLOR = new THREE.Color('#20B2AA');  // teal highlight
  var HOVER_COLOR = new THREE.Color('#5EEAD4');    // lighter teal hover

  // --- Body / headstock layout ---
  var STRING_START = -0.02;       // nut Z
  var STRING_END = 13.05;        // bridge Z
  var BODY_NECK_Z = 6.5;         // neck joint Z
  var BODY_BW = 5.0;             // body half-width
  var BODY_BH = 10.0;             // body length
  var BODY_BOTTOM_Z = BODY_NECK_Z + BODY_BH;
  var HEADSTOCK_END = -0.25;     // nut → headstock transition
  var HEADSTOCK_LENGTH = 3.4;    // headstock extrusion length
  var STRING_LENGTH = STRING_END - STRING_START;

  // --- Helpers ---

  // X-coordinate for the centre of a given string (0 = bass E, 5 = treble E)
  function getStringX(stringIdx) {
    var totalWidth = (NUM_STRINGS - 1) * STRING_SPACING;
    return -totalWidth / 2 + stringIdx * STRING_SPACING;
  }

  // Z-coordinate for a fret centre
  function getFretZ(fretIdx) {
    return FRET_POS[fretIdx];
  }

  // Compute note name from string + fret number
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

  // Look up frequency for a note string (e.g. "A4" → 440)
  function freqFromNote(noteStr) {
    return NOTE_FREQ_MAP[noteStr] || 440;
  }

  // --- Constructor ---
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
    this.buildTuners();
    this.initControls();
    this.initClick();
    this.initHover();
    this.setupUI();
    this.animate();
  }

  // --- Scene setup ---
  Guitar3D.prototype.initScene = function () {
    var w = this.container.clientWidth;
    var h = this.container.clientHeight;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf5f0e8);     // warm cream
    this.camera = new THREE.PerspectiveCamera(32, w / h, 0.1, 45);
    this.camera.position.set(-10, 14, 16);
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

  // --- Lighting ---
  Guitar3D.prototype.initLights = function () {
    this.scene.add(new THREE.AmbientLight(0x3a3530, 0.5));   // dim warm fill
    var hemi = new THREE.HemisphereLight(0xeeddcc, 0x554433, 0.4);
    this.scene.add(hemi);
    var key = new THREE.DirectionalLight(0xfff0dd, 2.0);     // key light
    key.position.set(5, 7, 6);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    this.scene.add(key);
    var fill = new THREE.DirectionalLight(0xddaa77, 0.3);    // rim fill
    fill.position.set(-3, 2, -4);
    this.scene.add(fill);
  };

  // --- Build guitar body with sound hole, ring, bridge, saddle, pins ---
  Guitar3D.prototype.buildBody = function () {
    var bw = BODY_BW, bh = BODY_BH;
    // Dreadnought-style body profile (scaled by bw/bh)
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

    // Sound hole cutout
    var soundHolePath = new THREE.Path();
    soundHolePath.absarc(0, HEADSTOCK_END + FB_LENGTH + 0.95 - BODY_NECK_Z, 0.90, 0, Math.PI * 2, false);
    shape.holes.push(soundHolePath);

    // Extrude body shape to give it depth
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

    // Sound hole rosette ring
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

    // Dark backplate behind the sound hole so it reads as a cavity
    var backPlateMat = new THREE.MeshPhysicalMaterial({
      color: 0x1a1410, roughness: 0.9, metalness: 0.0,
      side: THREE.DoubleSide
    });
    var backGeom = new THREE.CircleGeometry(1.0, 28);
    var backPlate = new THREE.Mesh(backGeom, backPlateMat);
    backPlate.rotation.x = -Math.PI / 2;
    backPlate.position.set(0, bodyTopY - bodyDepth + 0.001, holeZ);
    this.scene.add(backPlate);

    // Bridge (dark wood block on the body top)
    var bridgeZ = HEADSTOCK_END + FB_LENGTH + 2.5;
    var bridgeMat = new THREE.MeshPhysicalMaterial({
      color: 0x5c3a1e, roughness: 0.6, metalness: 0.0
    });
    var bridge = new THREE.Mesh(
      new THREE.BoxGeometry(3.2, 0.20, 0.40), bridgeMat
    );
    bridge.position.set(0, 0.05, bridgeZ);
    this.scene.add(bridge);

    // Saddle (thin ivory strip on the bridge)
    var saddleMat = new THREE.MeshPhysicalMaterial({
      color: 0xf0e8d0, roughness: 0.3, metalness: 0.1
    });
    var saddle = new THREE.Mesh(
      new THREE.BoxGeometry(2.8, 0.08, 0.025), saddleMat
    );
    saddle.position.set(0, 0.19, bridgeZ + 0.12);
    this.scene.add(saddle);

    // Bridge pins (behind saddle, one per string)
    var pinMat = new THREE.MeshPhysicalMaterial({
      color: 0xf0e8d8, roughness: 0.3, metalness: 0.0
    });
    var pinZ = bridgeZ + 0.22;
    for (var p = 0; p < NUM_STRINGS; p++) {
      var pin = new THREE.Mesh(
        new THREE.CylinderGeometry(0.030, 0.040, 0.12, 6), pinMat
      );
      pin.position.set(getStringX(p), 0.05, pinZ);
      this.scene.add(pin);
    }
  };

  // --- Build fretboard (D‑shape extrusion from headstock to end) + nut ---
  Guitar3D.prototype.buildFretboard = function () {
    var hw = FB_WIDTH / 2;
    var thick = FB_THICK;
    var mat = new THREE.MeshPhysicalMaterial({
      color: 0x5c3a1e, roughness: 0.6, metalness: 0.0
    });

    // Single continuous D‑shape from headstock → fretboard end
    var fullLen = FB_LENGTH;
    var fullShape = new THREE.Shape();
    fullShape.moveTo(-hw, 0);
    fullShape.lineTo(hw, 0);
    fullShape.bezierCurveTo(hw, -thick*0.2, hw*0.7, -thick*2.0, 0, -thick*2.5);
    fullShape.bezierCurveTo(-hw*0.7, -thick*2.0, -hw, -thick*0.2, -hw, 0);
    var fullGeom = new THREE.ExtrudeGeometry(fullShape, {
      depth: fullLen, bevelEnabled: true,
      bevelThickness: 0.02, bevelSize: 0.015, bevelSegments: 4
    });
    fullGeom.translate(0, 0, HEADSTOCK_END);
    var fbMesh = new THREE.Mesh(fullGeom, mat);
    fbMesh.receiveShadow = true;
    this.scene.add(fbMesh);

    // Nut (ivory bar at the headjoint)
    var nutGeom = new THREE.BoxGeometry(FB_WIDTH - 0.2, 0.06, 0.08);
    var nutMat = new THREE.MeshPhysicalMaterial({
      color: 0xf0e8d8, roughness: 0.4, metalness: 0.0
    });
    var nut = new THREE.Mesh(nutGeom, nutMat);
    nut.position.set(0, 0.02, -0.02);
    this.scene.add(nut);
  };

  // --- Fret lines across the fretboard ---
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

  // --- Strings (bass → treble, decreasing radius) ---
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

  // --- Inlay fret markers (dots at frets 3,5,7,9, double at 12) ---
  Guitar3D.prototype.buildFretMarkers = function () {
    var self = this;
    var dotMat = new THREE.MeshPhysicalMaterial({
      color: 0xe8e0d0, roughness: 0.8, metalness: 0.0, transparent: true, opacity: 0.35
    });
    var markerFrets = [3, 5, 7, 9, 12];
    markerFrets.forEach(function (f) {
      if (f >= FRET_POS.length) return;
      if (f === 12) {
        // Double dot at 12th fret (centred either side of string plane)
        var d1 = new THREE.Mesh(new THREE.CircleGeometry(0.06, 12), dotMat);
        d1.position.set(-0.14, 0.02, FRET_POS[f] - 0.12);
        d1.rotation.x = -Math.PI / 2;
        self.scene.add(d1);
        var d2 = new THREE.Mesh(new THREE.CircleGeometry(0.06, 12), dotMat);
        d2.position.set(0.14, 0.02, FRET_POS[f] - 0.12);
        d2.rotation.x = -Math.PI / 2;
        self.scene.add(d2);
      } else {
        // Single dot centred on the fretboard
        var dot = new THREE.Mesh(new THREE.CircleGeometry(0.07, 12), dotMat);
        dot.position.set(0, 0.02, FRET_POS[f] - 0.12);
        dot.rotation.x = -Math.PI / 2;
        self.scene.add(dot);
      }
    });
  };

  // --- Headstock (tapered D‑shape extruded backward from the nut) ---
  Guitar3D.prototype.buildHeadstock = function () {
    var hsMat = new THREE.MeshPhysicalMaterial({
      color: 0x5c3a1e, roughness: 0.6, metalness: 0.0,
      side: THREE.DoubleSide
    });
    var taper = 0.7;
    var hw = FB_WIDTH / 2;
    var thick = FB_THICK;
    var d = thick * 2.5;
    var headLen = HEADSTOCK_LENGTH;

    // D‑shape cross-section matching the neck profile
    var shape = new THREE.Shape();
    shape.moveTo(-hw, 0);
    shape.lineTo(hw, 0);
    shape.bezierCurveTo(hw, -d*0.08, hw*0.7, -d*0.8, 0, -d);
    shape.bezierCurveTo(-hw*0.7, -d*0.8, -hw, -d*0.08, -hw, 0);

    var geom = new THREE.ExtrudeGeometry(shape, {
      depth: headLen, bevelEnabled: false
    });

    // Taper width & depth along the extrusion, flip Z to extend in −Z
    var pos = geom.attributes.position;
    for (var i = 0; i < pos.count; i++) {
      var z = pos.getZ(i);
      var t = z / headLen;               // 0 at nut, 1 at tip
      var s = 1 - (1 - taper) * t;
      var zNew = HEADSTOCK_END - z;      // flip to point backward
      pos.setXYZ(i, pos.getX(i) * s, pos.getY(i) * s, zNew);
    }
    geom.computeVertexNormals();

    var hs = new THREE.Mesh(geom, hsMat);
    hs.receiveShadow = true;
    this.scene.add(hs);
  };

  // --- Tuning pegs, posts, knobs, and headstock string segments ---
  Guitar3D.prototype.buildTuners = function () {
    var tunerMat = new THREE.MeshPhysicalMaterial({
      color: 0xd4a050, roughness: 0.3, metalness: 0.6
    });
    var postMat = new THREE.MeshPhysicalMaterial({
      color: 0xbb8833, roughness: 0.4, metalness: 0.4
    });
    var knobMat = new THREE.MeshPhysicalMaterial({
      color: 0xeeddbb, roughness: 0.2, metalness: 0.2
    });
    var segMat = new THREE.MeshPhysicalMaterial({
      color: 0xddcbb8, roughness: 0.4, metalness: 0.05
    });

    var nutZ = -0.02;
    var margin = 0.25;    // distance from headstock tip to first/last post
    var zSpacing = (HEADSTOCK_LENGTH - margin * 2) / (NUM_STRINGS / 2 - 1);

    for (var s_ = 0; s_ < NUM_STRINGS; s_++) {
      var sx = getStringX(s_);
      var side = s_ < 3 ? -1 : 1;          // treble left, bass right
      // Outer strings (0,5) closest to nut for shortest string segments
      var zIdx = s_ < 3 ? s_ : NUM_STRINGS - 1 - s_;
      var postZ = HEADSTOCK_END - margin - zIdx * zSpacing;
      // Compute headstock half-width at this Z (follows taper)
      var extZ = HEADSTOCK_END - postZ;
      var t = extZ / HEADSTOCK_LENGTH;
      var hsScale = 1 - (1 - 0.7) * t;
      var hsHalfW = (FB_WIDTH / 2) * hsScale;
      var postX = side * hsHalfW * 0.88;

      // Tuning post (vertical cylinder)
      var post = new THREE.Mesh(
        new THREE.CylinderGeometry(0.065, 0.065, 0.22, 8), postMat
      );
      post.position.set(postX, 0.02, postZ);
      this.scene.add(post);

      // Peg arm (horizontal, points outward from centre-line)
      var peg = new THREE.Mesh(
        new THREE.CylinderGeometry(0.055, 0.055, 0.36, 8), tunerMat
      );
      peg.position.set(postX + side * 0.18, 0.01, postZ);
      peg.rotation.z = Math.PI / 2 * side;
      this.scene.add(peg);

      // Knob at the outer end of the peg
      var knob = new THREE.Mesh(
        new THREE.SphereGeometry(0.10, 8, 8), knobMat
      );
      knob.position.set(postX + side * 0.36, 0.01, postZ);
      this.scene.add(knob);

      // Angled string segment from nut to tuning post
      var r = 0.025 + (5 - s_) * 0.006;
      var from = new THREE.Vector3(sx, 0.01, nutZ);
      var to   = new THREE.Vector3(postX, 0.04, postZ);
      var dir  = new THREE.Vector3().copy(to).sub(from);
      var len  = dir.length();
      dir.normalize();

      var segGeom = new THREE.CylinderGeometry(r, r, len, 4);
      var seg = new THREE.Mesh(segGeom, segMat.clone());
      var mid = new THREE.Vector3().copy(from).add(to).multiplyScalar(0.5);
      seg.position.copy(mid);
      seg.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
      this.scene.add(seg);
    }
  };

  // --- Display a chord: highlight strings, place finger dots, play audio ---
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
        // Muted string → red
        self.highlightString(si, true, true);
        return;
      }
      if (fret === 0) {
        // Open string → teal
        self.highlightString(si, true);
        return;
      }
      // Fretted string → teal + finger dot
      self.highlightString(si, true);
      var sx = getStringX(si);
      var fz = FRET_POS[fret - 1] + (FRET_POS[fret] - FRET_POS[fret - 1]) / 2;
      var dot = new THREE.Mesh(new THREE.SphereGeometry(0.1, 12, 12), dotMat.clone());
      dot.position.set(sx, 0.08, fz);
      self.scene.add(dot);
      self.fingerDots.push(dot);
    });
  };

  // --- Toggle string highlight: teal (played), red (muted), or unlit ---
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

  // --- Remove finger dots and reset string highlights ---
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

  // --- Play chord notes through Web Audio (sawtooth + lowpass) ---
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
      // Stagger note start slightly for a strum effect
      setTimeout(function () {
        var osc = ctx.createOscillator();
        var gain = ctx.createGain();
        var filter = ctx.createBiquadFilter();
        osc.type = 'sawtooth';
        osc.frequency.value = freq;
        filter.type = 'lowpass';
        filter.frequency.value = 3000;
        filter.Q.value = 0.5;
        // Quick attack, then exponential decay
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

  // --- Stop all active oscillators ---
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

  // --- Orbit controls ---
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

  // --- Hover interaction: highlight string + show hover sphere ---
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
        // Determine which fret the hover point is over
        var pt = intersects[0].point;
        var fret = -1;
        for (var i = 1; i < FRET_POS.length; i++) {
          if (pt.z < FRET_POS[i]) { fret = i - 1; break; }
        }
        if (fret < 0) fret = NUM_FRETS - 1;

        var sx = getStringX(si);
        var fz = FRET_POS[fret] + (FRET_POS[fret + 1] - FRET_POS[fret]) / 2;

        // Show a translucent sphere at the fretted position
        if (!self.hoverMesh) {
          self.hoverMesh = new THREE.Mesh(
            new THREE.SphereGeometry(0.09, 10, 10), hoverMat
          );
          self.scene.add(self.hoverMesh);
        }
        self.hoverMesh.position.set(sx, 0.08, fz);
        self.hoverMesh.visible = true;

        // Highlight the hovered string (restore chord highlights on other strings)
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

  // --- Clear hover state ---
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

  // --- Re-apply chord colours after hover clears ---
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

  // --- Click-to-play a single note ---
  // Detects clicks on strings (ignoring drags) and plays the fret note.
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
      // Ignore drags (threshold = 4px)
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

  // --- Wire chord buttons to showChord ---
  Guitar3D.prototype.setupUI = function () {
    var self = this;
    this.container.querySelectorAll('.gp3-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var chordId = btn.getAttribute('data-chord');
        // Toggle: clicking the active chord clears it
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

  // --- Handle container resize ---
  Guitar3D.prototype.resize = function () {
    var w = this.container.clientWidth;
    var h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  };

  // --- Render loop ---
  Guitar3D.prototype.animate = function () {
    var self = this;
    function loop() {
      requestAnimationFrame(loop);
      self.controls.update();
      self.renderer.render(self.scene, self.camera);
    }
    loop();
  };

  // --- Bootstrap ---
  var container = document.getElementById('gp3-container');
  if (container) new Guitar3D(container);
})();
