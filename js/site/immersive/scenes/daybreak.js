/*
 * Scene for "Still I Rise" (Maya Angelou): from the dirt to a daybreak over
 * a black ocean, rising a little more with every "I rise".
 *
 * I    "You may tread me in the very dirt": low on dark ground at night;
 *      "still, like dust, I'll rise": dust lifts off the earth and climbs.
 * II   "Oil wells pumping in my living room": standing, the black sea ahead
 *      takes on a slick, iridescent sheen.
 * III  "Just like moons and like suns, with the certainty of tides": the moon
 *      rises over the sea, the tide swells up the sand, and spray leaps up
 *      ("hopes springing high").
 * IV   "Bowed head and lowered eyes": looking down at the wet sand, drops
 *      falling like teardrops, rings spreading.
 * V    "Gold mines diggin' in my own back yard": the head lifts; flecks of
 *      gold glint in the sand.
 * VI   "Still, like air, I'll rise": the wind gets up and carries you up and
 *      out over the water.
 * VII  "I dance like I've got diamonds": the moon's path breaks into
 *      dancing diamond glints.
 * VIII "I'm a black ocean, leaping and wide": low over a heaving black sea;
 *      then each "I rise" lifts you higher into "a daybreak that's
 *      wondrously clear", and on the last three the sun comes up.
 *
 * Columns: [unit, z, (unused), tears, wind, height, pitch, yaw, dust, oil, tide,
 *           moon, gold, diamonds, dawn, sun, swell, spray]
 */
import { THREE, isSmall, makeRenderer, fitCamera, softSprite, skyDome, terrain, particleField,
         waveHeight, oceanMaterial, oceanMesh, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// Swell rolling in to the shore (+z), cross-sea and chop.
var WAVES = [[0.2, -1, 0.07, 0.6, 0.9], [-0.6, -0.8, 0.13, 0.3, 1.3], [0.7, -0.6, 0.24, 0.15, 1.8], [0.1, -1, 0.6, 0.05, 2.6]];
var MOON_AZ = -0.42, SUN_AZ = 0.3;                      // moon ahead-left, sun ahead-right
var TIDE = 0.38;                                         // metres the tide rises
var GOLD_AT = 8;                                         // timeline unit where panel V starts (set by keys)

function ground(x, z) {
  var h = z > 0 ? z * 0.03 : z * 0.07;
  h += smooth(30, 110, z) * (5 + 3 * Math.sin(x * 0.031) + 2 * Math.sin(x * 0.083 + z * 0.05));
  var ripple = Math.sin(x * 0.35 + z * 2.1 + Math.sin(x * 0.2) * 1.5) * (1 - smooth(20, 40, z));
  return h + 0.06 * Math.sin(x * 0.9 + z * 0.4) * Math.sin(z * 1.3 - x * 0.2) + 0.035 * ripple;
}

// A warm chord swelling up as the sun breaks the horizon.
function daybreakChord(ac, out) {
  var t = ac.currentTime, lp = ac.createBiquadFilter();
  lp.type = 'lowpass';
  lp.frequency.setValueAtTime(500, t);
  lp.frequency.linearRampToValueAtTime(2200, t + 3);
  lp.connect(out);
  [146.83, 220, 293.66, 369.99, 587.33].forEach(function (f, i) {
    var o = ac.createOscillator(), g = ac.createGain();
    o.type = i % 2 ? 'sine' : 'triangle';
    o.frequency.value = f;
    o.detune.value = (Math.random() - 0.5) * 8;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.09 / (1 + i * 0.4), t + 2.2 + i * 0.25);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 9);
    o.connect(g); g.connect(lp);
    o.start(t); o.stop(t + 9.2);
  });
}

// Twinkling glints (gold on the sand, diamonds on the water): each point
// flashes now and then as a small four-pointed star.
function glintMaterial(color, rate) {
  return new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
    uniforms: { uTime: { value: 0 }, uAmt: { value: 0 }, uScale: { value: 1 }, uColor: { value: new THREE.Color(color) } },
    vertexShader: 'attribute vec2 seed; uniform float uTime; uniform float uAmt; uniform float uScale; varying float vA;\n' +
      'void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;\n' +
      ' float s = sin(uTime * (' + rate.toFixed(2) + ' + seed.x * 2.0) + seed.y * 6.2832);\n' +
      ' vA = pow(max(s, 0.0), 10.0) * uAmt * step(seed.x, uAmt * 1.1);\n' +
      ' gl_PointSize = uScale * (5.0 + 20.0 * vA) * clamp(30.0 / -mv.z, 0.35, 2.0); }',
    fragmentShader: 'uniform vec3 uColor; varying float vA;\n' +
      'void main(){ vec2 c = gl_PointCoord - 0.5; float d = length(c);\n' +
      ' float core = smoothstep(0.18, 0.0, d); float cross = smoothstep(0.035, 0.0, min(abs(c.x), abs(c.y))) * smoothstep(0.5, 0.0, d);\n' +
      ' float a = (core + cross * 0.8) * vA; if (a < 0.003) discard;\n' +
      ' gl_FragColor = vec4(uColor * a, a);\n #include <colorspace_fragment>\n }'
  });
}

function glints(n, place, color, rate) {
  var pos = new Float32Array(n * 3), seed = new Float32Array(n * 2), r = rng(n * 7 + 3);
  for (var i = 0; i < n; i++) { place(r, pos, i * 3); seed[i * 2] = r(); seed[i * 2 + 1] = r(); }
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('seed', new THREE.BufferAttribute(seed, 2));
  var pts = new THREE.Points(geo, glintMaterial(color, rate));
  pts.frustumCulled = false;
  return pts;
}

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(53), azs = 1;          // azs: how far off-centre the moon and sun sit (narrower on portrait)
  var gl = makeRenderer(canvas, { clear: '#02030a' });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#0a0f1e', 0.0018);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.05, 4000);

  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#02030a', mid: '#070b1c', horizon: '#121a33', sun: '#ffb070' }, 1500);
  sky.add(dome.mesh);

  // Stars: a shader point field that twinkles and fades at dawn.
  var SN = small ? 2500 : 5000, sp = [], sa = [];
  for (var i = 0; i < SN; i++) {
    var th = r() * Math.PI * 2, y = 0.03 + r() * 0.97, s = Math.sqrt(1 - y * y);
    sp.push(1300 * s * Math.cos(th), 1300 * y, 1300 * s * Math.sin(th));
    sa.push(0.8 + Math.pow(r(), 3) * 3, r() * 6.28, 1 + r() * 2);
  }
  var starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.Float32BufferAttribute(sp, 3));
  starGeo.setAttribute('star', new THREE.Float32BufferAttribute(sa, 3));
  var starMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
    uniforms: { uTime: { value: 0 }, uAmt: { value: 1 }, uScale: { value: 1 } },
    vertexShader: 'attribute vec3 star; uniform float uTime; uniform float uAmt; uniform float uScale; varying float vA;\n' +
      'void main(){ gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);\n' +
      ' vA = uAmt * (0.7 + 0.3 * sin(uTime * star.z + star.y)); gl_PointSize = star.x * uScale; }',
    fragmentShader: 'varying float vA; void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;\n' +
      ' float a = smoothstep(0.5, 0.0, d) * vA; gl_FragColor = vec4(vec3(0.85, 0.9, 1.0) * a, a);\n #include <colorspace_fragment>\n }'
  });
  var stars = new THREE.Points(starGeo, starMat);
  stars.frustumCulled = false;
  sky.add(stars);

  // The moon: a disc and its halo.
  var moonDisc = new THREE.Mesh(new THREE.CircleGeometry(16, 48), new THREE.MeshBasicMaterial({ fog: false, transparent: true }));
  moonDisc.material.color.setRGB(2.2, 2.2, 2.1);
  var moonGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(200,212,255,0.55)', 'rgba(120,140,220,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
  moonGlow.scale.setScalar(230);
  sky.add(moonDisc, moonGlow);

  var hemi = new THREE.HemisphereLight('#6a7cb4', '#141824', 1.3);
  // Starlight from low behind you before the moon is up: it rakes the
  // ripples, and its glint on the sea falls out of sight.
  var starLight = new THREE.DirectionalLight('#9fb4ec', 1.1);
  starLight.position.set(0.3, 0.35, 1);
  world.add(starLight);
  var moonLight = new THREE.DirectionalLight('#cfdcff', 0);
  var sunLight = new THREE.DirectionalLight('#ffc890', 0);
  world.add(hemi, moonLight, moonLight.target, sunLight, sunLight.target);

  // The shore: dark sand sloping into the sea, dunes behind.
  var dry = new THREE.Color('#8a7e70'), wet = new THREE.Color('#4a453e'), dune = new THREE.Color('#5c5a4a'), cc = new THREE.Color();
  function sandColor(x, z) {
    cc.copy(wet).lerp(dry, smooth(9, 16, z)).lerp(dune, smooth(40, 90, z));
    return cc.multiplyScalar(0.9 + 0.2 * (Math.sin(x * 0.7 + z * 0.5) * Math.sin(x * 0.23 - z * 0.9) * 0.5 + 0.5));
  }
  // A coarse sheet for the whole shore, pushed back in depth so a finer
  // patch round the beach you stand on (where the ripples are) wins.
  var coarseMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 4 });
  world.add(terrain(420, small ? 160 : 240, 0, 70, ground, coarseMat, sandColor));
  var fine = new THREE.PlaneGeometry(70, 52, small ? 160 : 300, small ? 120 : 220).rotateX(-Math.PI / 2).translate(0, 0, 11);
  var fp = fine.attributes.position, fcols = new Float32Array(fp.count * 3);
  for (i = 0; i < fp.count; i++) {
    var fx = fp.getX(i), fz = fp.getZ(i);
    fp.setY(i, ground(fx, fz));
    sandColor(fx, fz);
    fcols[i * 3] = cc.r; fcols[i * 3 + 1] = cc.g; fcols[i * 3 + 2] = cc.b;
  }
  fine.setAttribute('color', new THREE.BufferAttribute(fcols, 3));
  fine.computeVertexNormals();
  var fineMesh = new THREE.Mesh(fine, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9 }));
  world.add(fineMesh);
  // Pebbles for scale close to the ground.
  var pebble = new THREE.InstancedMesh(new THREE.DodecahedronGeometry(0.08, 0), new THREE.MeshStandardMaterial({ color: '#6a6258', roughness: 0.8, flatShading: true }),
                                       small ? 120 : 260);
  var m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), sc = new THREE.Vector3(), pv = new THREE.Vector3(), eu = new THREE.Euler();
  for (i = 0; i < pebble.count; i++) {
    var px = (r() - 0.5) * 24, pz = 8 + r() * 17;
    pv.set(px, ground(px, pz) + 0.02, pz);
    q.setFromEuler(eu.set(r() * 3, r() * 3, r() * 3));
    sc.set(0.4 + r() * 1.0, 0.3 + r() * 0.5, 0.4 + r() * 1.0);
    pebble.setMatrixAt(i, m4.compose(pv, q, sc));
  }
  world.add(pebble);

  var oceanMat = oceanMaterial({ color: '#04070d', specular: '#dfe6ff', shininess: 150, foam: '#c8d2de', waves: WAVES });
  var ou = oceanMat.userData.uniforms, oil = { value: 0 }, shore = { value: 0 };
  // Oil: a thin-film sheen on the black water, strongest at grazing angles.
  var kitCompile = oceanMat.onBeforeCompile;
  oceanMat.onBeforeCompile = function (sh) {
    kitCompile(sh);
    sh.uniforms.uOil = oil;
    sh.uniforms.uShore = shore;
    // The swell dies away up the beach, so the water lies flat at the tide
    // line instead of poking up through the sand.
    sh.vertexShader = 'uniform float uShore;\n' + sh.vertexShader.replace('vec3 wv = waves(wpos.xz);',
      'vec3 wv = waves(wpos.xz) * (1.0 - smoothstep(uShore - 16.0, uShore - 1.0, wpos.z));');
    sh.fragmentShader = 'uniform float uOil;\n' + sh.fragmentShader.replace('totalEmissiveRadiance += uSky * (0.08 + fres * 0.95);',
      'totalEmissiveRadiance += uSky * (0.08 + fres * 0.95);\n' +
      ' vec3 irid = 0.5 + 0.5 * cos(6.2832 * (fres * 1.7 + normal.x * 2.4 + normal.z * 1.6 + vec3(0.0, 0.33, 0.67)));\n' +
      ' totalEmissiveRadiance += mix(vec3(0.5), irid, 0.45) * uOil * (0.004 + fres * 0.035);');
  };
  var ocean = oceanMesh(oceanMat, 1600, small ? 200 : 300);
  world.add(ocean);

  // The surf edge: two sheets of foam sliding up and down the sand at
  // the waterline, following the tide.
  var foamTex = (function () {
    var c = document.createElement('canvas'), fr = rng(77);
    c.width = 512; c.height = 64;
    var x = c.getContext('2d');
    for (var n = 0; n < 900; n++) {
      x.fillStyle = 'rgba(225,232,245,' + (0.12 + fr() * 0.5).toFixed(2) + ')';
      x.beginPath();
      x.ellipse(fr() * 512, 6 + fr() * 54 * fr(), 3 + fr() * 20, 1 + fr() * 2.5, 0, 0, Math.PI * 2);
      x.fill();
    }
    var t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    t.wrapS = THREE.RepeatWrapping;
    t.repeat.set(10, 1);
    return t;
  })();
  var swash = [0, 2.2].map(function (offset) {
    var m = new THREE.Mesh(new THREE.PlaneGeometry(260, 2.6), new THREE.MeshBasicMaterial({ map: foamTex, color: '#8f9cb4', transparent: true, depthWrite: false }));
    m.rotation.x = -Math.PI / 2 + Math.atan(0.03);
    m.userData.offset = offset;
    world.add(m);
    return m;
  });

  // Dust lifting off the ground round you and climbing.
  var DN = small ? 900 : 2000, dPos = new Float32Array(DN * 3), dVel = new Float32Array(DN), dPh = new Float32Array(DN);
  var dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute('position', new THREE.BufferAttribute(dPos, 3));
  var dust = new THREE.Points(dustGeo, new THREE.PointsMaterial({ color: '#ffd29a', size: 0.1, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, map: softSprite('rgba(255,245,225,1)', 'rgba(255,235,200,0)') }));
  dust.frustumCulled = false;
  world.add(dust);
  // Most of it lifts in a loose column ahead and to the right (clear of
  // the text); the rest drifts up all round.
  function gauss() { return (r() + r() + r() - 1.5) * 1.15; }
  function seedDust(k, cz, any) {
    var col = k % 10 < 7, x = col ? 2.6 * azs + gauss() * 1.3 * lerp(0.6, 1, azs) : (r() - 0.5) * 16, z = col ? cz - 7 + gauss() * 2 : cz - 2.5 - r() * 13;
    dPos[k * 3] = x; dPos[k * 3 + 2] = z;
    dPos[k * 3 + 1] = ground(x, z) + (any ? r() * 7 : 0);
    dVel[k] = 0.25 + r() * 0.8; dPh[k] = r() * 6.28;
  }
  for (i = 0; i < DN; i++) seedDust(i, 30, true);

  // Tears: drops falling to the wet sand, rings spreading where they land.
  var TN = 14, tears = [], tearPos = new Float32Array(TN * 3), tearGeo = new THREE.BufferGeometry();
  tearGeo.setAttribute('position', new THREE.BufferAttribute(tearPos, 3));
  var tearPts = new THREE.Points(tearGeo, new THREE.PointsMaterial({ color: '#dfe8ff', size: 0.06, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)') }));
  tearPts.frustumCulled = false;
  world.add(tearPts);
  var ringGeo = new THREE.RingGeometry(0.92, 1, 48).rotateX(-Math.PI / 2);
  for (i = 0; i < TN; i++) {
    var ring = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color: '#c8d6f0', transparent: true, depthWrite: false,
                                                                    blending: THREE.AdditiveBlending, opacity: 0 }));
    world.add(ring);
    tears.push({ ring: ring, x: (r() - 0.5) * 3, dz: 1.8 + r() * 2.4, ph: r(), period: 1.5 + r() * 0.8 });
  }

  // Gold flecks in the sand; diamonds dancing on the moon's path.
  var gold = glints(small ? 500 : 1100, function (rr, p, k) {
    var x = (rr() - 0.5) * 30, z = 2 + rr() * 22;
    p[k] = x; p[k + 1] = ground(x, z) + 0.2; p[k + 2] = z;
  }, '#ffc85a', 1.6);
  world.add(gold);
  var diamondGroup = new THREE.Group();
  var diamonds = glints(small ? 900 : 2000, function (rr, p, k) {
    var d = 18 + Math.pow(rr(), 1.3) * 300, a = (rr() - 0.5) * (0.03 + 20 / d);   // turned towards the moon each frame
    p[k] = Math.sin(a) * d; p[k + 1] = 0.25; p[k + 2] = -Math.cos(a) * d;
  }, '#eef4ff', 2.4);
  diamondGroup.add(diamonds);
  world.add(diamondGroup);

  // Spray leaping off the breakers.
  var SPN = small ? 300 : 700, spPos = new Float32Array(SPN * 3), spVel = new Float32Array(SPN * 3), spLife = new Float32Array(SPN);
  var spGeo = new THREE.BufferGeometry();
  spGeo.setAttribute('position', new THREE.BufferAttribute(spPos, 3));
  var spray = new THREE.Points(spGeo, new THREE.PointsMaterial({ color: '#d6e0f0', size: 0.2, transparent: true, depthWrite: false, opacity: 0.8,
    blending: THREE.AdditiveBlending, map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)') }));
  spray.frustumCulled = false;
  world.add(spray);
  for (i = 0; i < SPN; i++) spPos[i * 3 + 1] = -99;

  // Air: wind-borne motes streaming past.
  var air = particleField({ count: small ? 400 : 900, box: [40, 14, 40], fall: [-0.4, 0.3], size: 0.06, color: '#cfd8ea',
                            map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)'), sway: 0.8, windSpeed: 16 });
  world.add(air.points);

  // ── Per frame ─────────────────────────────────────────────────────────
  var C = {
    night: ['#03050f', '#0c1430', '#2a3a66'], early: ['#0c1434', '#38375e', '#b86d66'], clear: ['#3d6db2', '#8db2dc', '#f7d3a0']
  };
  var tmp = new THREE.Color(), tmp2 = new THREE.Color(), horizon = new THREE.Color();
  var moonDir = new THREE.Vector3(), sunDir = new THREE.Vector3(), H = 800, sprayClock = 0, camBase = 1;

  function skyCol(k, dawn) {
    tmp.set(C.night[k]).lerp(tmp2.set(C.early[k]), smooth(0, 0.55, dawn));
    return tmp.lerp(tmp2.set(C.clear[k]), smooth(0.5, 1, dawn));
  }

  function frame(f) {
    var row = f.row, time = f.time, dt = f.dt, slow = env.reduceMotion;
    var z = row[0], tearsAmt = row[2], height = row[4], dustAmt = row[7], oilAmt = row[8], tide = row[9] * TIDE;
    var moonUp = row[10], goldAmt = row[11], diaAmt = row[12], dawn = row[13], sunUp = row[14], swell = row[15], sprayAmt = row[16];

    // Over land you stand on the sand; over the sea you ride above the swell.
    var target = Math.max(ground(0, z), tide);
    camBase += (target - camBase) * (1 - Math.exp(-dt * 6));
    camera.position.set(Math.sin(time * 0.17) * 0.15, camBase + height, z);
    // Portrait screens are narrow: bring the moon and sun (and the turns
    // towards them) closer to the middle.
    var yaw = row[6] * azs, moonAz = MOON_AZ * azs, sunAz = SUN_AZ * azs;
    // On a phone the centred verse sits on the horizon: until panel V, tilt
    // up a touch so the moonrise and the waterline show below it.
    var lift = azs < 1 ? 0.13 * (1 - smooth(GOLD_AT - 0.8, GOLD_AT, f.u)) : 0;
    camera.rotation.set(row[5] + lift - f.my * 0.06, yaw - f.mx * 0.14, 0, 'YXZ');
    sky.position.copy(camera.position);
    ocean.userData.follow(camera.position);
    ocean.position.y = tide;
    shore.value = tide / 0.03;                      // where the water meets the sand

    // Sky from night through first light to a clear morning.
    dome.uniforms.top.value.copy(skyCol(0, dawn));
    dome.uniforms.mid.value.copy(skyCol(1, dawn));
    horizon.copy(skyCol(2, dawn));
    dome.uniforms.horizon.value.copy(horizon);
    world.fog.color.copy(horizon).lerp(dome.uniforms.mid.value, 0.4).multiplyScalar(0.8);
    world.fog.density = lerp(0.0018, 0.0011, dawn);
    gl.setClearColor(world.fog.color);

    // Moon rising ahead-left; it pales as the day comes.
    var mel = lerp(-0.06, 0.42, moonUp);
    moonDir.set(Math.sin(moonAz) * Math.cos(mel), Math.sin(mel), -Math.cos(moonAz) * Math.cos(mel));
    moonDisc.position.copy(moonDir).multiplyScalar(1100);
    moonGlow.position.copy(moonDir).multiplyScalar(1090);
    sky.updateMatrixWorld();
    moonDisc.lookAt(camera.position);
    var moonVis = smooth(-0.05, 0.02, mel) * (1 - smooth(0.05, 0.35, dawn));
    moonDisc.material.opacity = moonVis * (1 - dawn * 0.5);
    moonGlow.material.opacity = moonVis * (0.8 - dawn * 0.6);
    moonLight.position.copy(camera.position).addScaledVector(moonDir, 300);
    moonLight.target.position.copy(camera.position);
    moonLight.intensity = (0.35 + smooth(-0.03, 0.12, mel) * 1.3) * (1 - smooth(0.1, 0.45, dawn));

    // The sun comes up ahead-right on the last three "I rise".
    var sel = lerp(-0.07, 0.11, sunUp);
    sunDir.set(Math.sin(sunAz) * Math.cos(sel), Math.sin(sel), -Math.cos(sunAz) * Math.cos(sel));
    var toSun = smooth(0.15, 0.5, dawn);
    dome.uniforms.sunDir.value.copy(moonDir).lerp(sunDir, toSun).normalize();
    dome.uniforms.sunColor.value.set('#2a3458').multiplyScalar((1 - toSun) * (1 - smooth(-0.01, 0.12, mel)) * 0.9)
      .add(tmp2.set('#ff9a50').multiplyScalar(smooth(0.3, 1, dawn) * (0.45 + 0.5 * smooth(-0.04, 0.03, sel))));
    sunLight.position.copy(camera.position).addScaledVector(sunDir, 300);
    sunLight.target.position.copy(camera.position);
    sunLight.intensity = smooth(-0.04, 0.05, sel) * 3.2;
    hemi.intensity = 1.3 + dawn * 0.1;
    starLight.intensity = 1.1 * (1 - smooth(0.0, 0.3, dawn));
    hemi.color.set('#5a6a9a').lerp(tmp2.set('#ffe2c4'), dawn);

    starMat.uniforms.uTime.value = time;
    starMat.uniforms.uAmt.value = 1 - smooth(0.15, 0.7, dawn);
    starMat.uniforms.uScale.value = Math.min(window.devicePixelRatio || 1, 2) * (H / 800 + 0.3);

    // The black ocean: slick with oil, then swelling, then lit by the day.
    ou.uTime.value = time;
    ou.uAmp.value = swell;
    ou.uFoamAmt.value = smooth(0.9, 2, swell) * 0.9 * (1 - smooth(0.3, 0.7, dawn));
    ou.uSky.value.copy(horizon).lerp(dome.uniforms.mid.value, 0.5 + dawn * 0.3).multiplyScalar(lerp(0.5, 0.36, dawn));
    oceanMat.color.set('#04070d').lerp(tmp2.set('#123a5e'), dawn);
    oil.value = oilAmt;

    // Swash runs up and drains back at the tide line; it fades out to sea.
    swash.forEach(function (m) {
      var ph = time * 0.7 + m.userData.offset, run = 0.5 + 0.5 * Math.sin(ph), wz = tide / 0.03 + 0.6 + run * 1.6;
      m.position.set(0, ground(0, wz) + 0.03, wz);
      m.material.opacity = (0.2 + 0.3 * (0.5 + 0.5 * Math.cos(ph))) * (1 - smooth(-2, 10, camera.position.z * -1)) * (0.8 + row[16] * 0.4);
    });

    // Dust: drifts up from the ground in front of you while it lasts.
    for (var k = 0; k < DN; k++) {
      var j = k * 3;
      dPos[j + 1] += dVel[k] * dt * (slow ? 0.5 : 1);
      var swirl = 0.25 + (dPos[j + 1] - camBase) * 0.06;            // the column loosens as it climbs
      dPos[j] += Math.sin(time * 0.8 + dPh[k]) * swirl * dt + f.wind * 2 * dt;
      dPos[j + 2] += Math.cos(time * 0.8 + dPh[k]) * swirl * dt;
      if (dPos[j + 1] > ground(dPos[j], dPos[j + 2]) + 9 || Math.abs(dPos[j + 2] - z) > 14) seedDust(k, z, false);
    }
    dustGeo.attributes.position.needsUpdate = true;
    dustGeo.setDrawRange(0, Math.floor(DN * clamp(dustAmt, 0, 1)));
    dust.material.opacity = clamp(dustAmt * 1.4, 0, 1);

    // Tears: each drop falls, then a ring spreads where it lands, ahead
    // of you in the direction you are looking.
    var fwdX = Math.sin(yaw), fwdZ = Math.cos(yaw);
    for (var t = 0; t < TN; t++) {
      var tr = tears[t], on = t < tearsAmt * TN, ph = ((time / tr.period + tr.ph) % 1);
      var tx = camera.position.x - fwdX * tr.dz + fwdZ * tr.x, tz = z - fwdZ * tr.dz - fwdX * tr.x;
      var surf = Math.max(ground(tx, tz), tide + waveHeight(WAVES, tx, tz, time, swell).h * (1 - smooth(shore.value - 16, shore.value - 1, tz))) + 0.03;
      if (!on) { tearPos[t * 3 + 1] = -99; tr.ring.material.opacity = 0; continue; }
      var fall = clamp(ph / 0.3, 0, 1);
      tearPos[t * 3] = tx; tearPos[t * 3 + 1] = fall < 1 ? surf + 2.2 * (1 - fall * fall) : -99; tearPos[t * 3 + 2] = tz;
      var spread = clamp((ph - 0.3) / 0.7, 0, 1);
      tr.ring.position.set(tx, surf, tz);
      tr.ring.scale.setScalar(0.02 + spread * 0.24);
      tr.ring.material.opacity = spread > 0 ? (1 - spread) * 0.7 : 0;
    }
    tearGeo.attributes.position.needsUpdate = true;

    var scale = Math.min(window.devicePixelRatio || 1, 2) * (H / 800 * 0.6 + 0.4);
    [gold, diamonds].forEach(function (g) { g.material.uniforms.uTime.value = time; g.material.uniforms.uScale.value = scale; });
    gold.material.uniforms.uAmt.value = goldAmt;
    diamonds.material.uniforms.uAmt.value = diaAmt;
    diamondGroup.position.set(camera.position.x, tide, camera.position.z);
    diamondGroup.rotation.y = -moonAz;

    // Spray bursts along the break when hopes spring high.
    sprayClock += dt * sprayAmt * (slow ? 12 : 30);
    while (sprayClock > 1) {
      sprayClock -= 1;
      var bx = (Math.random() - 0.5) * 50, bz = tide / 0.03 - 3 - Math.random() * 6;
      for (var n = 0; n < 16; n++) {
        var s = Math.floor(Math.random() * SPN), q3 = s * 3;
        spPos[q3] = bx + (Math.random() - 0.5) * 3; spPos[q3 + 1] = tide + 0.2; spPos[q3 + 2] = bz;
        spVel[q3] = (Math.random() - 0.5) * 1.5; spVel[q3 + 1] = 3 + Math.random() * 5; spVel[q3 + 2] = 0.5 + Math.random() * 1.5;
        spLife[s] = 1.8;
      }
    }
    for (var p = 0; p < SPN; p++) {
      var w = p * 3;
      if (spLife[p] <= 0) { spPos[w + 1] = -99; continue; }
      spLife[p] -= dt;
      spVel[w + 1] -= 6 * dt;
      spPos[w] += spVel[w] * dt; spPos[w + 1] += spVel[w + 1] * dt; spPos[w + 2] += spVel[w + 2] * dt;
    }
    spGeo.attributes.position.needsUpdate = true;

    air.update({ snow: f.wind, wind: f.wind, dt: dt, time: time }, camera.position, slow);

    gl.toneMappingExposure = 1 + dawn * 0.1;
    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) { H = h; azs = w / h < 1 ? 0.45 : 1; fitCamera(gl, camera, w, h, dpr, small); },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('daybreak', {
  renderer: renderer3d,
  maxLines: 6,
  accent: '#ffcf7a',
  emphasis: /^rise\W*$/i,                                 // the refrain
  // Panels: 0-6 stanzas I-VII; the long last stanza splits into 7 "I'm a
  // black ocean", 8 "Into a daybreak" and 9 "I rise / I rise / I rise".
  align: ['left', 'right', 'right', 'center', 'right', 'right', 'right', 'right', 'left', 'center'],
  keys: function (T) {
    function at(i, frac) { i = Math.min(i, T.count - 1); return T.start(i) + frac * (T.end(i) - T.start(i)); }
    GOLD_AT = T.start(Math.min(4, T.count - 1));
    // "I rise" lands roughly where the reading reaches it in each of the
    // last three panels; the camera steps up at each one.
    //   unit          z     -  tears wind height pitch  yaw   dust oil  tide moon  gold dia  dawn sun   swell spray
    return [
      [0,                30, 0,  0.00,  0.05,  0.32,  0.03,  0.00,  0.35,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.45,  0.00],
      [0.7,            29.5, 0,  0.00,  0.05,  0.32,  0.03,  0.00,  0.45,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.45,  0.00],
      [at(0, 0.35),    28.5, 0,  0.00,  0.08,  0.34,  0.06,  0.00,  0.75,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.45,  0.00],  // "tread me in the very dirt"
      [at(0, 0.85),    27.5, 0,  0.00,  0.12,  0.70,  0.42,  0.00,  1.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.45,  0.00],  // "like dust, I'll rise"
      [at(1, 0.2),       24, 0,  0.00,  0.08,  1.65,  0.05,  0.00,  0.45,  0.30,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.50,  0.00],
      [at(1, 0.7),       19, 0,  0.00,  0.06,  1.65,  0.00, -0.06,  0.15,  1.00,  0.00,  0.06,  0.00,  0.00,  0.00,  0.00,  0.60,  0.00],  // "oil wells pumping"
      [at(2, 0.15),      16, 0,  0.00,  0.08,  1.65,  0.03,  0.10,  0.00,  0.40,  0.25,  0.15,  0.00,  0.00,  0.00,  0.00,  0.70,  0.10],  // "like moons"
      [at(2, 0.6),       15, 0,  0.00,  0.12,  1.65,  0.08,  0.24,  0.00,  0.10,  1.00,  0.24,  0.00,  0.00,  0.00,  0.00,  0.90,  0.40],  // "the certainty of tides"
      [at(2, 0.95),    14.5, 0,  0.00,  0.15,  1.65,  0.10,  0.22,  0.00,  0.00,  1.00,  0.28,  0.00,  0.00,  0.00,  0.00,  1.00,  1.00],  // "hopes springing high"
      [at(3, 0.2),       15, 0,  0.30,  0.05,  1.60, -0.42,  0.36,  0.00,  0.00,  0.60,  0.29,  0.00,  0.00,  0.00,  0.00,  0.45,  0.00],  // "bowed head and lowered eyes"
      [at(3, 0.5),       15, 0,  1.00,  0.03,  1.55, -0.48,  0.40,  0.00,  0.00,  0.55,  0.29,  0.00,  0.00,  0.00,  0.00,  0.40,  0.00],  // "falling down like teardrops"
      [at(3, 0.95),    14.5, 0,  0.80,  0.03,  1.55, -0.44,  0.38,  0.00,  0.00,  0.50,  0.29,  0.10,  0.00,  0.00,  0.00,  0.50,  0.00],
      [at(4, 0.25),      13, 0,  0.00,  0.05,  1.60, -0.30,  0.08,  0.00,  0.00,  0.50,  0.29,  0.60,  0.00,  0.00,  0.00,  0.70,  0.00],  // "haughtiness"
      [at(4, 0.7),       14, 0,  0.00,  0.08,  1.65, -0.24,  0.12,  0.00,  0.00,  0.50,  0.30,  1.00,  0.00,  0.00,  0.00,  0.75,  0.00],  // "gold mines"
      [at(5, 0.15),    13.5, 0,  0.00,  0.55,  1.70, -0.05,  0.10,  0.00,  0.00,  0.50,  0.31,  0.40,  0.00,  0.00,  0.00,  0.90,  0.00],
      [at(5, 0.55),      13, 0,  0.00,  1.00,  1.80,  0.00,  0.00,  0.00,  0.00,  0.50,  0.32,  0.00,  0.00,  0.00,  0.00,  1.10,  0.30],  // "kill me with your hatefulness"
      [at(5, 0.95),      -8, 0,  0.00,  0.80,  9.00, -0.06,  0.10,  0.00,  0.00,  0.40,  0.33,  0.00,  0.20,  0.00,  0.00,  1.00,  0.00],  // "like air, I'll rise"
      [at(6, 0.3),      -30, 0,  0.00,  0.35,  6.00, -0.08,  0.30,  0.00,  0.00,  0.30,  0.34,  0.00,  0.80,  0.00,  0.00,  0.80,  0.00],  // "I dance"
      [at(6, 0.8),      -48, 0,  0.00,  0.25,  5.00, -0.07,  0.32,  0.00,  0.00,  0.30,  0.35,  0.00,  1.00,  0.00,  0.00,  0.80,  0.00],  // "like I've got diamonds"
      [at(7, 0.1),      -66, 0,  0.00,  0.45,  3.40,  0.00,  0.10,  0.00,  0.00,  0.30,  0.35,  0.00,  0.30,  0.00,  0.00,  1.40,  0.00],
      [at(7, 0.39),     -76, 0,  0.00,  0.50,  4.60,  0.01,  0.04,  0.00,  0.00,  0.30,  0.35,  0.00,  0.10,  0.02,  0.00,  1.60,  0.00],  // "I rise"
      [at(7, 0.5),      -80, 0,  0.00,  0.50,  4.60,  0.01,  0.02,  0.00,  0.00,  0.30,  0.35,  0.00,  0.00,  0.04,  0.00,  1.70,  0.00],
      [at(7, 0.65),     -86, 0,  0.00,  0.55,  6.00,  0.01,  0.00,  0.00,  0.00,  0.30,  0.35,  0.00,  0.00,  0.06,  0.00,  1.80,  0.00],  // "I rise"
      [at(7, 0.95),     -96, 0,  0.00,  0.60,  6.20,  0.01, -0.04,  0.00,  0.00,  0.30,  0.35,  0.00,  0.00,  0.12,  0.00,  2.00,  0.00],  // "I'm a black ocean ... welling and swelling"
      [at(8, 0.25),    -104, 0,  0.00,  0.45,  6.40,  0.02, -0.06,  0.00,  0.00,  0.30,  0.36,  0.00,  0.00,  0.25,  0.00,  1.70,  0.00],  // "leaving behind nights"
      [at(8, 0.39),    -110, 0,  0.00,  0.40,  9.00,  0.02, -0.07,  0.00,  0.00,  0.30,  0.36,  0.00,  0.00,  0.40,  0.00,  1.50,  0.00],  // "I rise"
      [at(8, 0.52),    -114, 0,  0.00,  0.35,  9.20,  0.02, -0.08,  0.00,  0.00,  0.30,  0.37,  0.00,  0.00,  0.52,  0.05,  1.40,  0.00],  // "into a daybreak"
      [at(8, 0.62),    -118, 0,  0.00,  0.30, 13.00,  0.02, -0.09,  0.00,  0.00,  0.30,  0.37,  0.00,  0.00,  0.62,  0.10,  1.30,  0.00],  // "I rise"
      [at(8, 1.0),     -130, 0,  0.00,  0.25, 13.50,  0.02, -0.10,  0.00,  0.00,  0.30,  0.38,  0.00,  0.00,  0.75,  0.20,  1.10,  0.00],  // "the dream and the hope"
      [at(9, 0.3),     -136, 0,  0.00,  0.20, 14.00,  0.03,  0.03,  0.00,  0.00,  0.30,  0.38,  0.00,  0.00,  0.80,  0.25,  1.00,  0.00],
      [at(9, 0.43),    -140, 0,  0.00,  0.20, 20.00,  0.03,  0.03,  0.00,  0.00,  0.30,  0.38,  0.00,  0.00,  0.86,  0.45,  0.95,  0.00],  // "I rise"
      [at(9, 0.53),    -143, 0,  0.00,  0.20, 20.50,  0.03,  0.03,  0.00,  0.00,  0.30,  0.38,  0.00,  0.00,  0.88,  0.50,  0.95,  0.00],
      [at(9, 0.65),    -147, 0,  0.00,  0.20, 29.00,  0.03,  0.03,  0.00,  0.00,  0.30,  0.38,  0.00,  0.00,  0.93,  0.72,  0.90,  0.00],  // "I rise"
      [at(9, 0.75),    -150, 0,  0.00,  0.20, 29.50,  0.03,  0.03,  0.00,  0.00,  0.30,  0.38,  0.00,  0.00,  0.95,  0.76,  0.90,  0.00],
      [at(9, 0.87),    -154, 0,  0.00,  0.20, 40.00, -0.03,  0.03,  0.00,  0.00,  0.30,  0.38,  0.00,  0.00,  1.00,  1.00,  0.85,  0.00],  // "I rise."
      [T.total,        -170, 0,  0.00,  0.20, 46.00, -0.04,  0.04,  0.00,  0.00,  0.30,  0.38,  0.00,  0.00,  1.00,  1.15,  0.80,  0.00]
    ];
  },
  sound: {
    src: '/audio/ocean.mp3',
    label: 'Play the sea',
    volume: function (row) { return 0.08 + 0.1 * row[9] + 0.15 * row[3] + 0.18 * clamp(row[15] - 0.6, 0, 1.4) - 0.05 * row[13]; },
    cues: [{ stanza: 9, at: 0.43 * 1.6, play: daybreakChord }]
  }
});
