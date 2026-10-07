/*
 * Scene for "Pale Blue Dot" (Carl Sagan): a journey in and out of scale.
 *
 * It opens on the dot itself, a pale pixel in a band of scattered sunlight,
 * then falls towards Earth while the text lists everyone who ever lived
 * there, until the night side's city lights fill the view. From there it
 * pulls back past the Moon ("a very small stage"), out into the dark
 * ("a lonely speck"), past a barren Mars ("there is nowhere else ... to
 * which our species could migrate"), and settles back on the dot.
 *
 * Distances are in Earth radii; nothing is to scale beyond Earth and Moon.
 * Earth's day, night and cloud maps are generated at load from 3D noise.
 * Keyframes are computed from the prose chunks the engine makes. Columns:
 *   [unit, log10 distance, dark, sunbeam, (unused), azimuth from the sun, elevation, look yaw]
 */
import { THREE, isSmall, makeRenderer, softSprite, starField, disposeAll } from '../kit.js';

var PI = window.PoemImmersive;
var clamp = PI.util.clamp, smooth = PI.util.smooth, lerp = PI.util.lerp, rng = PI.util.rng;

var SUN = new THREE.Vector3(1, 0.06, 0.15).normalize();   // direction to the Sun
var NEAR_AZ = 0.95, FAR_AZ = 2.2;                          // radians from the Sun direction
var MARS_AT = 3500;

function viewDir(az, el, out) {
  // Rotate the Sun direction about the vertical by az, then tilt by el.
  var c = Math.cos(az), s = Math.sin(az);
  out.set(SUN.x * c - SUN.z * s, 0, SUN.x * s + SUN.z * c).normalize();
  out.y = Math.tan(el);
  return out.normalize();
}

// ── Procedural Earth ─────────────────────────────────────────────────────
function noise3(seed) {
  function h(i, j, k) {
    var n = Math.imul(i, 374761393) ^ Math.imul(j, 668265263) ^ Math.imul(k, 1274126177) ^ seed;
    n = Math.imul(n ^ (n >>> 13), 1274126177);
    return ((n ^ (n >>> 16)) >>> 0) / 4294967295;
  }
  function f(t) { return t * t * (3 - 2 * t); }
  return function (x, y, z) {
    var i = Math.floor(x), j = Math.floor(y), k = Math.floor(z), u = f(x - i), v = f(y - j), w = f(z - k);
    var a = lerp(lerp(h(i, j, k), h(i + 1, j, k), u), lerp(h(i, j + 1, k), h(i + 1, j + 1, k), u), v);
    var b = lerp(lerp(h(i, j, k + 1), h(i + 1, j, k + 1), u), lerp(h(i, j + 1, k + 1), h(i + 1, j + 1, k + 1), u), v);
    return lerp(a, b, w);
  };
}

function fbm(n, x, y, z, oct) {
  var sum = 0, amp = 0.5, fr = 1;
  for (var o = 0; o < oct; o++) { sum += amp * n(x * fr, y * fr, z * fr); amp *= 0.5; fr *= 2.03; }
  return sum;
}

// Equirectangular day/night/cloud canvases sampled on the sphere, so there
// is no seam at the date line.
function earthMaps(W) {
  var H = W / 2, land = noise3(7), wob = noise3(11), cloud = noise3(19), city = noise3(23), speck = noise3(29);
  var day = document.createElement('canvas'), night = document.createElement('canvas'), clouds = document.createElement('canvas');
  [day, night, clouds].forEach(function (c) { c.width = W; c.height = H; });
  var dctx = day.getContext('2d'), nctx = night.getContext('2d'), cctx = clouds.getContext('2d');
  var D = dctx.createImageData(W, H), N = nctx.createImageData(W, H), C = cctx.createImageData(W, H);
  for (var y = 0; y < H; y++) {
    var lat = (0.5 - (y + 0.5) / H) * Math.PI, cl = Math.cos(lat), sy = Math.sin(lat);
    for (var x = 0; x < W; x++) {
      var lon = ((x + 0.5) / W) * Math.PI * 2, px = cl * Math.cos(lon) * 1.6, pz = cl * Math.sin(lon) * 1.6, py = sy * 1.6;
      var e = fbm(land, px + 0.6 * wob(px * 2, py * 2, pz * 2), py, pz, 5);
      var ice = Math.abs(lat) > 1.2 + 0.1 * wob(px * 4, py * 4, pz * 4);
      var i4 = (y * W + x) * 4, r, g, b;
      if (ice) { r = 236; g = 241; b = 246; }
      else if (e < 0.53) {                     // ocean, lighter in the shallows
        var sh = smooth(0.42, 0.53, e);
        r = lerp(8, 30, sh); g = lerp(30, 80, sh); b = lerp(78, 140, sh);
      } else {                                 // land: forest, grass, desert by latitude and noise
        var dry = smooth(0.15, 0.5, 1 - Math.abs(lat) * 2.2) * smooth(0.45, 0.7, wob(px * 3, py * 3, pz * 3));
        r = lerp(52, 176, dry); g = lerp(92, 150, dry); b = lerp(42, 96, dry);
        var hi = smooth(0.62, 0.72, e); r = lerp(r, 120, hi); g = lerp(g, 110, hi); b = lerp(b, 96, hi);
      }
      D.data[i4] = r; D.data[i4 + 1] = g; D.data[i4 + 2] = b; D.data[i4 + 3] = 255;
      // City lights on temperate land.
      // City lights: regions from coarse noise, broken into clusters by fine noise.
      var lights = e >= 0.53 && !ice && Math.abs(lat) < 1.0
        ? smooth(0.5, 0.72, city(px * 6, py * 6, pz * 6)) * smooth(0.58, 0.8, speck(px * 60, py * 60, pz * 60)) : 0;
      N.data[i4] = 255; N.data[i4 + 1] = 196; N.data[i4 + 2] = 110; N.data[i4 + 3] = Math.round(lights * 255);
      var c = smooth(0.5, 0.68, fbm(cloud, px * 1.5 + 3, py * 2.2, pz * 1.5, 5));
      C.data[i4] = C.data[i4 + 1] = C.data[i4 + 2] = 255; C.data[i4 + 3] = Math.round(c * 235);
    }
  }
  dctx.putImageData(D, 0, 0); nctx.putImageData(N, 0, 0); cctx.putImageData(C, 0, 0);
  return [day, night, clouds].map(function (c) { var t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t; });
}

function rockTexture(seed, a, b) {
  var c = document.createElement('canvas'), W = 256;
  c.width = W; c.height = W / 2;
  var x = c.getContext('2d'), img = x.createImageData(W, W / 2), n = noise3(seed);
  var ca = new THREE.Color(a), cb = new THREE.Color(b);
  for (var y = 0; y < W / 2; y++) {
    var lat = (0.5 - (y + 0.5) / (W / 2)) * Math.PI;
    for (var i = 0; i < W; i++) {
      var lon = (i + 0.5) / W * Math.PI * 2, px = Math.cos(lat) * Math.cos(lon) * 3, pz = Math.cos(lat) * Math.sin(lon) * 3;
      var v = fbm(n, px, Math.sin(lat) * 3, pz, 4), k = (y * W + i) * 4;
      img.data[k] = lerp(ca.r, cb.r, v) * 255; img.data[k + 1] = lerp(ca.g, cb.g, v) * 255;
      img.data[k + 2] = lerp(ca.b, cb.b, v) * 255; img.data[k + 3] = 255;
    }
  }
  x.putImageData(img, 0, 0);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function renderer3d(canvas, scene, env) {
  var small = isSmall(), r = rng(3);
  var gl = makeRenderer(canvas, { clear: '#000000' });
  gl.autoClear = false;

  // Two passes: a background (stars, Sun) with its own far camera, then the
  // planets with near/far fitted to them each frame, so depth stays precise
  // from a metre above the clouds to tens of thousands of radii out.
  var bg = new THREE.Scene(), bgCam = new THREE.PerspectiveCamera(55, 1, 1, 20000);
  var stars = starField(r, small ? 2500 : 5000, 9000, -1, 1.4);
  bg.add(stars);
  // A band of fainter, denser stars: the Milky Way.
  var mw = [], tilt = new THREE.Euler(0.9, 0.3, 0.2), v = new THREE.Vector3();
  for (var i = 0; i < (small ? 3000 : 7000); i++) {
    var th = r() * Math.PI * 2, y = (r() + r() + r() - 1.5) * 0.12;
    v.set(Math.cos(th), y, Math.sin(th)).normalize().applyEuler(tilt).multiplyScalar(9000);
    mw.push(v.x, v.y, v.z);
  }
  var mwGeo = new THREE.BufferGeometry();
  mwGeo.setAttribute('position', new THREE.Float32BufferAttribute(mw, 3));
  var mwMat = new THREE.PointsMaterial({ color: '#b9c6e8', size: 1, sizeAttenuation: false, transparent: true, opacity: 0.45, depthWrite: false });
  bg.add(new THREE.Points(mwGeo, mwMat));
  var sunGlow = new THREE.Sprite(new THREE.SpriteMaterial({
    map: softSprite('rgba(255,248,230,1)', 'rgba(255,220,170,0)'), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true
  }));
  sunGlow.position.copy(SUN).multiplyScalar(15000);
  sunGlow.scale.setScalar(2600);
  bg.add(sunGlow);

  var world = new THREE.Scene(), camera = new THREE.PerspectiveCamera(55, 1, 0.01, 100);
  world.add(camera);
  var light = new THREE.DirectionalLight('#fff6e8', 3.2);
  light.position.copy(SUN).multiplyScalar(100);
  world.add(light, new THREE.AmbientLight('#203050', 0.08));

  // Earth: day/night blend across the terminator, with a blue rim.
  var maps = earthMaps(small ? 512 : 1024);
  var earthMat = new THREE.ShaderMaterial({
    uniforms: { day: { value: maps[0] }, night: { value: maps[1] }, sun: { value: SUN.clone() } },
    vertexShader: 'varying vec2 vUv; varying vec3 vN; varying vec3 vW;' +
      'void main(){ vUv = uv; vN = normalize(mat3(modelMatrix) * normal); vW = (modelMatrix * vec4(position,1.0)).xyz;' +
      ' gl_Position = projectionMatrix * viewMatrix * vec4(vW,1.0); }',
    fragmentShader: 'uniform sampler2D day; uniform sampler2D night; uniform vec3 sun; varying vec2 vUv; varying vec3 vN; varying vec3 vW;' +
      'void main(){ vec3 n = normalize(vN); float l = dot(n, normalize(sun));' +
      ' vec3 d = texture2D(day, vUv).rgb; vec4 nt = texture2D(night, vUv);' +
      ' vec3 v = normalize(cameraPosition - vW); float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);' +
      ' vec3 c = d * (smoothstep(-0.05, 0.35, l) * 1.15 + 0.02);' +
      ' c += nt.rgb * nt.a * smoothstep(0.08, -0.2, l) * 1.4;' +
      ' vec3 h = normalize(normalize(sun) + v); float ocean = step(d.r + 0.12, d.b);' +
      ' c += vec3(1.0, 0.95, 0.85) * pow(max(dot(n, h), 0.0), 60.0) * 0.6 * ocean * step(0.0, l);' +
      ' c += vec3(0.35, 0.6, 1.0) * rim * smoothstep(-0.3, 0.4, l) * 0.9;' +
      ' gl_FragColor = vec4(c, 1.0);' +
      '\n #include <tonemapping_fragment>\n #include <colorspace_fragment>\n }'
  });
  var earth = new THREE.Mesh(new THREE.SphereGeometry(1, 96, 64), earthMat);
  earth.rotation.z = 0.41;
  world.add(earth);
  var cloudMesh = new THREE.Mesh(new THREE.SphereGeometry(1.012, 96, 64),
    new THREE.MeshLambertMaterial({ map: maps[2], transparent: true, depthWrite: false }));
  cloudMesh.rotation.z = 0.41;
  world.add(cloudMesh);
  var atmo = new THREE.Mesh(new THREE.SphereGeometry(1.08, 64, 32), new THREE.ShaderMaterial({
    side: THREE.BackSide, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { sun: { value: SUN.clone() } },
    vertexShader: 'varying vec3 vN; varying vec3 vW; void main(){ vN = normalize(mat3(modelMatrix) * normal); vW = (modelMatrix * vec4(position,1.0)).xyz; gl_Position = projectionMatrix * viewMatrix * vec4(vW,1.0); }',
    fragmentShader: 'uniform vec3 sun; varying vec3 vN; varying vec3 vW; void main(){ vec3 v = normalize(cameraPosition - vW);' +
      ' float rim = pow(1.0 - abs(dot(normalize(vN), v)), 2.5); float lit = smoothstep(-0.4, 0.5, dot(normalize(vN), normalize(sun)));' +
      ' gl_FragColor = vec4(vec3(0.3, 0.55, 1.0) * rim * lit * 1.2, rim * lit);' +
      '\n #include <colorspace_fragment>\n }'
  }));
  world.add(atmo);

  // The pale blue dot: a fixed-size point that takes over once Earth is
  // smaller than a few pixels.
  var dotGeo = new THREE.BufferGeometry();
  dotGeo.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3));
  var dotMat = new THREE.PointsMaterial({ color: '#b8d2ff', size: 4, sizeAttenuation: false, transparent: true, opacity: 0, depthWrite: false,
                                          map: softSprite('rgba(255,255,255,1)', 'rgba(255,255,255,0)') });
  world.add(new THREE.Points(dotGeo, dotMat));

  var moon = new THREE.Mesh(new THREE.SphereGeometry(0.27, 48, 32), new THREE.MeshLambertMaterial({ map: rockTexture(5, '#5d5a57', '#b4b0aa') }));
  moon.position.set(-42, 10, 42);
  world.add(moon);

  var marsDir = viewDir(FAR_AZ, 0.12, new THREE.Vector3());
  var side = new THREE.Vector3().crossVectors(marsDir, new THREE.Vector3(0, 1, 0)).normalize();
  var mars = new THREE.Mesh(new THREE.SphereGeometry(0.53, 48, 32), new THREE.MeshLambertMaterial({ map: rockTexture(9, '#7a3a22', '#d0784a') }));
  mars.position.copy(marsDir).multiplyScalar(MARS_AT).addScaledVector(side, 3).y += 0.8;
  world.add(mars);

  // Scattered-light bands, as in the Voyager frame, drawn over everything
  // and centred on wherever Earth appears.
  var beams = new THREE.Group();
  camera.add(beams);
  var beamMat = new THREE.MeshBasicMaterial({ color: '#c79b6e', transparent: true, opacity: 0, depthTest: false, depthWrite: false,
                                              blending: THREE.AdditiveBlending });
  [[0, 0.09], [-0.32, 0.05], [0.27, 0.035], [0.55, 0.06]].forEach(function (b) {
    var m = new THREE.Mesh(new THREE.PlaneGeometry(b[1], 6), beamMat);
    m.position.x = b[0];
    m.renderOrder = 10;
    beams.add(m);
  });
  beams.rotation.z = -0.32;

  var dir = new THREE.Vector3(), ndc = new THREE.Vector3(), W = 1, H = 1;

  function frame(f) {
    var row = f.row, D = Math.pow(10, row[0]), dark = f.dark, beam = f.snow;
    viewDir(row[4], row[5], dir);
    camera.position.copy(dir).multiplyScalar(D);
    camera.up.set(0, 1, 0);
    camera.lookAt(0, 0, 0);
    camera.rotateY(row[6] - f.mx * 0.05);
    camera.rotateX(-f.my * 0.04);

    earth.rotation.y += f.dt * 0.012;
    cloudMesh.rotation.y += f.dt * 0.016;

    // Fit near/far to the bodies in front of the camera.
    var near = 1e9, far = 0;
    [[earth.position, 1.08], [moon.position, 0.27], [mars.position, 0.53]].forEach(function (b) {
      var d = camera.position.distanceTo(b[0]);
      near = Math.min(near, d - b[1]);
      far = Math.max(far, d + b[1]);
    });
    camera.near = Math.max(0.002, near * 0.5);
    camera.far = far * 1.5 + 10;
    camera.updateProjectionMatrix();

    // Earth's size on screen decides when the dot takes over.
    var px = (1 / D) / Math.tan(camera.fov * Math.PI / 360) * (H / 2);
    dotMat.opacity = smooth(5, 1.5, px) * (1 - dark * 0.3);
    dotMat.size = 4 * Math.min(window.devicePixelRatio || 1, 2);

    // Bands of sunlight around the dot.
    ndc.set(0, 0, 0).project(camera);
    var halfH = Math.tan(camera.fov * Math.PI / 360), halfW = halfH * camera.aspect;
    // Keep the bands just past the near plane, which moves a long way out.
    var k = camera.near * 1.5;
    beams.position.set(ndc.x * halfW * k, ndc.y * halfH * k, -k);
    beams.scale.setScalar(k);
    beamMat.opacity = beam * 0.09 * (ndc.z < 1 ? 1 : 0);

    stars.material.opacity = 0.85 * (1 - dark * 0.55);
    mwMat.opacity = 0.45 * (1 - dark * 0.7);
    sunGlow.material.opacity = 1 - dark * 0.6;
    gl.toneMappingExposure = 1 - dark * 0.35;

    bgCam.quaternion.copy(camera.quaternion);
    bgCam.fov = camera.fov;
    bgCam.aspect = camera.aspect;
    bgCam.updateProjectionMatrix();
    gl.clear();
    gl.render(bg, bgCam);
    gl.clearDepth();
    gl.render(world, camera);
  }

  return {
    resize: function (w, h, dpr) {
      W = w; H = h;
      gl.setPixelRatio(Math.min(dpr, small ? 1.5 : 1.75));
      gl.setSize(w, h, false);
      camera.aspect = w / h;
      camera.fov = w / h < 1 ? 62 : 45;
      camera.updateProjectionMatrix();
    },
    frame: frame,
    destroy: function () { disposeAll(bg); disposeAll(world, gl); }
  };
}

PI.register('pale-blue-dot', {
  renderer: renderer3d,
  align: ['center', 'left', 'center', 'right', 'center', 'left', 'center', 'right', 'center', 'center', 'center'],
  // Built from the prose chunks: paragraph 1 is chunks 0-1, paragraph 2 is
  // 2-4, 3 is 5-6, 4 is 7-8 and 5 is 9-10. If the text changes, the beats
  // still land in order.
  keys: function (T) {
    var n = T.count;
    function at(i, frac) { i = Math.min(Math.round(i * (n - 1) / 10), n - 1); return lerp(T.start(i), T.end(i), frac); }
    return [
      [0, 4.60, 0.00, 1.0, 0, FAR_AZ, 0.12, 0.12],
      [0.7, 4.58, 0.00, 1.0, 0, FAR_AZ, 0.12, 0.12],       // the dot in a sunbeam
      [at(0, 0.35), 4.30, 0.00, 0.7, 0, FAR_AZ - 0.2, 0.12, 0.06],   // "Look again at that dot."
      [at(0, 1.0), 1.10, 0.00, 0.0, 0, NEAR_AZ + 0.2, 0.25, 0.00],   // "On it everyone you love..."
      [at(1, 0.55), 0.42, 0.00, 0.0, 0, NEAR_AZ + 0.75, 0.30, 0.10], // close: the night side's lights
      [at(1, 1.0), 0.55, 0.00, 0.0, 0, NEAR_AZ + 0.9, 0.28, 0.14],   // "a mote of dust suspended in a sunbeam"
      [at(2, 0.6), 1.20, 0.00, 0.0, 0, NEAR_AZ + 0.6, 0.20, 0.10],   // "a very small stage"
      [at(4, 1.0), 2.40, 0.05, 0.1, 0, NEAR_AZ + 0.9, 0.16, 0.06],   // Earth and Moon: "one corner of this pixel"
      [at(5, 0.6), 3.00, 0.30, 0.2, 0, FAR_AZ - 0.3, 0.12, 0.00],    // "a lonely speck in the great enveloping cosmic dark"
      [at(6, 1.0), 3.40, 0.45, 0.2, 0, FAR_AZ, 0.12, 0.00],
      // Mars sits at MARS_AT; hang just beyond it so it drifts past in frame.
      // Turned a little (last column) to keep Mars in the margin beside the text.
      [at(7, 0.15), Math.log10(MARS_AT + 7), 0.35, 0.2, 0, FAR_AZ, 0.12, -0.10],   // "nowhere else"
      [at(7, 0.95), Math.log10(MARS_AT + 40), 0.30, 0.3, 0, FAR_AZ, 0.12, -0.22],  // "Visit, yes. Settle, not yet."
      [at(8, 1.0), 4.30, 0.25, 0.6, 0, FAR_AZ, 0.12, 0.08],          // "where we make our stand"
      [at(9, 0.6), 4.60, 0.12, 1.0, 0, FAR_AZ, 0.12, 0.12],
      [T.total, 4.70, 0.08, 1.0, 0, FAR_AZ, 0.12, 0.12]              // "the pale blue dot, the only home we've ever known"
    ];
  }
});
