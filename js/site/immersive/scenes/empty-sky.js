/*
 * Scene for "The More Loving One" (W. H. Auden): lying on a hillside,
 * looking up.
 *
 * I   Cold, indifferent stars; a glance down at the lit town on the horizon,
 *     "on earth indifference is the least we have to dread".
 * II  "Were stars to burn with a passion for us": they flare warm, then cool.
 *     "Let the more loving one be me": a lantern beside you glows up.
 * III "I cannot ... say I missed one terribly all day": a day passes, blue
 *     and starless, and the night comes back.
 * IV  "Were all stars to disappear or die": they go out one by one until
 *     the sky is empty, "its total dark sublime"; only your lantern stays.
 *
 * The stars are a shader point field so each can flare and each can die at
 * its own moment. Columns:
 *   [unit, (unused), (unused), (unused), wind, yaw, pitch, passion, warmth, day, empty, sublime]
 */
import { THREE, isSmall, makeRenderer, fitCamera, tinted, merge, softSprite, skyDome, terrain,
         scatter, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

// The hill falls away in front of you into a valley; the town climbs the
// far side, where you can see its lights.
function slope(x, z) {
  var d = Math.hypot(x, z);
  return z * 0.06 + 0.3 * Math.sin(x * 0.2) - smooth(30, 140, d) * 6 + smooth(160, 330, d) * 46;
}

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(91);
  var gl = makeRenderer(canvas, { clear: '#04050c' });
  var world = new THREE.Scene();
  world.fog = new THREE.FogExp2('#0a0d1c', 0.004);
  var camera = new THREE.PerspectiveCamera(60, 1, 0.05, 3000);

  var sky = new THREE.Group();
  world.add(sky);
  var dome = skyDome({ top: '#02030a', mid: '#070b1c', horizon: '#141a33' }, 1500);
  sky.add(dome.mesh);

  // Stars: each with its own size, colour, twinkle and moment of dying.
  var N = small ? 3500 : 7000, pos = [], attr = [];
  for (var i = 0; i < N; i++) {
    var th = r() * Math.PI * 2, y = 0.02 + r() * 0.98, s = Math.sqrt(1 - y * y);
    pos.push(1300 * s * Math.cos(th), 1300 * y, 1300 * s * Math.sin(th));
    var bright = Math.pow(r(), 3);
    attr.push(0.8 + bright * 3.2, r(), r() * 6.28, r());     // size, dies at, phase, warmth bias
  }
  var starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  starGeo.setAttribute('star', new THREE.Float32BufferAttribute(attr, 4));
  var starMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
    uniforms: { uTime: { value: 0 }, uPassion: { value: 0 }, uEmpty: { value: 0 }, uDay: { value: 0 }, uScale: { value: 1 } },
    vertexShader: 'attribute vec4 star; uniform float uTime; uniform float uPassion; uniform float uEmpty; uniform float uDay; uniform float uScale;\n' +
      'varying float vA; varying float vWarm;\n' +
      'void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;\n' +
      ' float alive = smoothstep(uEmpty - 0.04, uEmpty + 0.02, star.y);\n' +
      ' float tw = 0.75 + 0.25 * sin(uTime * (1.5 + star.w * 2.0) + star.z);\n' +
      ' vWarm = uPassion * (0.5 + 0.5 * star.w);\n' +
      ' vA = alive * tw * (1.0 - uDay);\n' +
      ' gl_PointSize = star.x * uScale * (1.0 + vWarm * 1.8); }',
    fragmentShader: 'varying float vA; varying float vWarm;\n' +
      'void main(){ vec2 c = gl_PointCoord - 0.5; float d = length(c); if (d > 0.5) discard;\n' +
      ' float a = smoothstep(0.5, 0.0, d) * vA;\n' +
      ' vec3 col = mix(vec3(0.82, 0.88, 1.0), vec3(1.0, 0.55, 0.25), vWarm);\n' +
      ' gl_FragColor = vec4(col * a, a);\n #include <colorspace_fragment>\n }'
  });
  var starPts = new THREE.Points(starGeo, starMat);
  starPts.frustumCulled = false;
  sky.add(starPts);

  // Day clouds.
  var cloudTex = softSprite('rgba(255,255,255,0.95)', 'rgba(255,255,255,0)'), clouds = [];
  for (i = 0; i < 22; i++) {
    var cl = new THREE.Sprite(new THREE.SpriteMaterial({ map: cloudTex, transparent: true, depthWrite: false, fog: false, opacity: 0 }));
    cl.position.set(-900 + r() * 1800, 350 + r() * 400, -900 + r() * 1400);
    cl.scale.set(280 + r() * 260, 120 + r() * 80, 1);
    sky.add(cl);
    clouds.push(cl);
  }

  var hemi = new THREE.HemisphereLight('#6a78a8', '#0a0a10', 0.4);
  var sunLight = new THREE.DirectionalLight('#fff2dc', 0);
  sunLight.position.set(0.3, 1, -0.2);
  world.add(hemi, sunLight);

  world.add(terrain(800, small ? 100 : 150, 0, -80, slope, new THREE.MeshLambertMaterial({ color: '#16220f' })));
  var blade = merge([tinted(new THREE.ConeGeometry(0.03, 0.6, 3).translate(0, 0.3, 0), '#2c4020')]);
  var grass = new THREE.InstancedMesh(blade, new THREE.MeshLambertMaterial({ vertexColors: true }), small ? 3000 : 7000), up = new THREE.Vector3(0, 1, 0);
  scatter(grass, 20000, function (k, p, q, s, c) {
    var x = (r() - 0.5) * 16, z = -r() * 18 + 2;
    if (Math.hypot(x, z - 1) < 2.2) return false;      // nothing right under your nose
    p.set(x, slope(x, z), z);
    q.setFromAxisAngle(up, r() * 6.28);
    s.setScalar(0.5 + r() * 0.6);
    c.setHSL(0.25, 0.35, 0.18 + r() * 0.1);
  });
  world.add(grass);

  // The town on the horizon, and your lantern on the grass.
  var town = [];
  for (i = 0; i < (small ? 260 : 560); i++) {
    var tx = (r() - 0.5) * 420, tz = -190 - r() * 110;
    town.push(tx, slope(tx, tz) + 0.4 + r() * 2.5, tz);
  }
  var townGeo = new THREE.BufferGeometry();
  townGeo.setAttribute('position', new THREE.Float32BufferAttribute(town, 3));
  var townMat = new THREE.PointsMaterial({ color: '#ffc27a', size: 2.6, transparent: true, depthWrite: false, fog: false,
    blending: THREE.AdditiveBlending, map: softSprite('rgba(255,215,150,1)', 'rgba(255,190,110,0)') });
  world.add(new THREE.Points(townGeo, townMat));

  var lantern = new THREE.Group();
  var lamp = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.11, 0.24, 10), new THREE.MeshBasicMaterial({ color: '#ffcf8a' }));
  var lampGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: softSprite('rgba(255,200,130,1)', 'rgba(255,150,80,0)'),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
  lampGlow.scale.setScalar(1.6);
  var lampLight = new THREE.PointLight('#ffb466', 0, 9, 1.6);
  lantern.add(lamp, lampGlow, lampLight);
  lantern.position.set(0.9, slope(0.9, -1.6) + 0.12, -1.6);
  world.add(lantern);

  var tmp = new THREE.Color(), tmp2 = new THREE.Color(), H = 1;

  function frame(f) {
    var row = f.row, time = f.time, passion = row[6], warmth = row[7], day = row[8], empty = row[9], sublime = row[10];

    // Lying back on the slope, looking up.
    camera.position.set(0, slope(0, 1) + 1.25, 1);
    camera.rotation.set(0, 0, 0);
    camera.lookAt(0, slope(0, 1) + 1.25, -10);
    camera.rotateY(row[4] - f.mx * 0.12);
    camera.rotateX(row[5] - f.my * 0.06);
    sky.position.copy(camera.position);
    sky.rotation.y = time * 0.004;           // the slow turn of the heavens

    starMat.uniforms.uTime.value = time;
    starMat.uniforms.uPassion.value = passion;
    starMat.uniforms.uEmpty.value = empty * 1.06;
    starMat.uniforms.uDay.value = day;
    starMat.uniforms.uScale.value = Math.min(window.devicePixelRatio || 1, 2) * (H / 800 + 0.4);

    // Day, night, and the dark that comes to feel sublime.
    dome.uniforms.top.value.set('#02030a').lerp(tmp.set('#2f62b0'), day);
    dome.uniforms.mid.value.set('#070b1c').lerp(tmp.set('#78a6dc'), day);
    var horizon = tmp2.set('#141a33').lerp(tmp.set('#cfe2f4'), day);
    horizon.lerp(tmp.set('#05060d'), smooth(0.5, 1, empty) * (1 - day));
    horizon.lerp(tmp.set('#1a1238'), sublime * 0.6);
    dome.uniforms.horizon.value.copy(horizon);
    dome.uniforms.top.value.lerp(tmp.set('#000000'), smooth(0.6, 1, empty) * (1 - day));
    world.fog.color.copy(horizon).multiplyScalar(0.6);
    gl.setClearColor(world.fog.color);
    clouds.forEach(function (c) { c.material.opacity = day * 0.8; c.position.x += f.dt * 6; if (c.position.x > 950) c.position.x -= 1900; });
    sunLight.intensity = day * 2.2;
    hemi.intensity = 0.25 + day * 1.1;
    hemi.color.set('#6a78a8').lerp(tmp.set('#cfe2ff'), day);
    townMat.opacity = (1 - day * 0.9) * (1 - empty * 0.75);

    // The more loving one: the lantern, which stays when the stars go.
    var flick = 0.9 + 0.1 * Math.sin(time * 7) * Math.sin(time * 3.1);
    lampGlow.material.opacity = (0.2 + warmth * 0.8) * flick;
    lampLight.intensity = (0.4 + warmth * 3.5) * flick;

    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) { H = h; fitCamera(gl, camera, w, h, dpr, small); camera.fov = w / h < 1 ? 75 : 60; camera.updateProjectionMatrix(); },
    frame: frame,
    destroy: function () { disposeAll(world, gl); }
  };
}

PI.register('empty-sky', {
  renderer: renderer3d,
  align: ['left', 'right', 'left', 'center'],
  keys: [
    //  unit  -  -  -  wind yaw   pitch passion warmth day  empty sublime
    [0.0, 0, 0, 0, 0.1, 0.00, 0.80, 0.0, 0.1, 0.0, 0.00, 0.0],
    [1.3, 0, 0, 0, 0.1, 0.05, 0.88, 0.0, 0.1, 0.0, 0.00, 0.0],   // "Looking up at the stars"
    [2.1, 0, 0, 0, 0.1, 0.00, 0.02, 0.0, 0.1, 0.0, 0.00, 0.0],   // "on earth indifference"
    [2.6, 0, 0, 0, 0.1, 0.00, 0.06, 0.0, 0.1, 0.0, 0.00, 0.0],
    [3.0, 0, 0, 0, 0.1, 0.00, 0.82, 0.2, 0.1, 0.0, 0.00, 0.0],
    [3.5, 0, 0, 0, 0.1, 0.00, 0.86, 1.0, 0.1, 0.0, 0.00, 0.0],   // "burn with a passion for us"
    [3.9, 0, 0, 0, 0.1, -0.12, 0.30, 0.2, 0.4, 0.0, 0.00, 0.0],
    [4.1, 0, 0, 0, 0.1, -0.30, 0.02, 0.0, 1.0, 0.0, 0.00, 0.0],   // "let the more loving one be me"
    [4.6, 0, 0, 0, 0.1, -0.10, 0.62, 0.0, 0.8, 0.3, 0.00, 0.0],
    [5.1, 0, 0, 0, 0.1, 0.00, 0.78, 0.0, 0.6, 1.0, 0.00, 0.0],   // "I missed one terribly all day"
    [5.6, 0, 0, 0, 0.1, 0.00, 0.80, 0.0, 0.6, 0.2, 0.00, 0.0],
    [5.9, 0, 0, 0, 0.1, 0.00, 0.84, 0.0, 0.6, 0.0, 0.00, 0.0],
    [6.4, 0, 0, 0, 0.1, 0.00, 0.86, 0.0, 0.6, 0.0, 0.45, 0.0],   // "Were all stars to disappear or die"
    [6.9, 0, 0, 0, 0.1, 0.00, 0.86, 0.0, 0.5, 0.0, 0.92, 0.0],   // "an empty sky"
    [7.3, 0, 0, 0, 0.1, 0.00, 0.80, 0.0, 0.4, 0.0, 1.00, 0.2],   // "its total dark sublime"
    [8.6, 0, 0, 0, 0.1, -0.18, 0.30, 0.0, 0.35, 0.0, 1.00, 1.0]   // "a little time": back down to the lantern
  ],
  sound: {
    src: '/audio/wind.mp3',
    label: 'Play the night wind',
    volume: function () { return 0.1; }
  }
});
