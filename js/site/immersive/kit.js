/*
 * Shared three.js helpers for the immersive poem scenes in ./scenes/.
 *
 * Every scene imports three.js through this file, so the version is pinned
 * in one place, and builds its world from the pieces here: a configured
 * renderer, a gradient sky dome, stars, terrain, instanced scattering, a
 * particle field that travels with the camera (snow, leaves, dust), a
 * path-following camera and teardown.
 */
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js';

export { THREE };

// Phones and tablets get lighter scenes: fewer instances, no shadows.
export function isSmall() {
  return window.innerWidth < 800 || !window.matchMedia('(pointer: fine)').matches;
}

export function makeRenderer(canvas, opts) {
  var gl = new THREE.WebGLRenderer({
    canvas: canvas, antialias: true, powerPreference: 'high-performance',
    logarithmicDepthBuffer: !!(opts && opts.logDepth)
  });
  gl.setClearColor((opts && opts.clear) || '#03060f');
  gl.toneMapping = THREE.ACESFilmicToneMapping;
  gl.outputColorSpace = THREE.SRGBColorSpace;
  gl.shadowMap.enabled = !!(opts && opts.shadows);
  gl.shadowMap.type = THREE.PCFSoftShadowMap;
  return gl;
}

// Wider field of view on portrait screens so both sides of a path stay in shot.
export function fitCamera(gl, camera, w, h, dpr, small) {
  gl.setPixelRatio(Math.min(dpr, small ? 1.5 : 1.75));
  gl.setSize(w, h, false);
  camera.aspect = w / h;
  camera.fov = w / h < 1 ? 70 : 55;
  camera.updateProjectionMatrix();
}

// ── Geometry ─────────────────────────────────────────────────────────────
// Give a geometry one flat vertex colour (non-indexed, so it can be merged).
export function tinted(geo, color) {
  geo = geo.index ? geo.toNonIndexed() : geo;
  var c = new THREE.Color(color), n = geo.attributes.position.count, a = new Float32Array(n * 3);
  for (var i = 0; i < n; i++) { a[i * 3] = c.r; a[i * 3 + 1] = c.g; a[i * 3 + 2] = c.b; }
  geo.setAttribute('color', new THREE.BufferAttribute(a, 3));
  return geo;
}

// Concatenate tinted() geometries into one (position, normal, colour).
export function merge(geos) {
  var total = 0;
  geos.forEach(function (g) { total += g.attributes.position.count; });
  var out = new THREE.BufferGeometry();
  ['position', 'normal', 'color'].forEach(function (name) {
    var arr = new Float32Array(total * 3), o = 0;
    geos.forEach(function (g) { arr.set(g.attributes[name].array, o); o += g.attributes[name].array.length; });
    out.setAttribute(name, new THREE.BufferAttribute(arr, 3));
  });
  geos.forEach(function (g) { g.dispose(); });
  return out;
}

// A broadleaf tree about 6 m tall: a trunk and a few low-poly leaf clumps.
// The crown is white so per-instance colours (autumn golds, spring greens)
// tint it; the trunk takes `bark` multiplied by the same tint.
export function broadleafGeometry(r, bark) {
  var parts = [tinted(new THREE.CylinderGeometry(0.14, 0.24, 3.4, 6).translate(0, 1.7, 0), bark || '#4a3a2e')];
  parts.push(tinted(new THREE.CylinderGeometry(0.05, 0.09, 1.6, 4).rotateZ(0.7).translate(0.5, 3.1, 0), bark || '#4a3a2e'));
  for (var k = 0; k < 5; k++) {
    var rad = 1.1 + r() * 0.7, a = r() * Math.PI * 2, d = k === 0 ? 0 : 0.7 + r() * 0.5;
    parts.push(tinted(new THREE.IcosahedronGeometry(rad, 0)
      .scale(1, 0.85, 1)
      .translate(Math.cos(a) * d, 4.2 + r() * 1.4 + (k === 0 ? 0.6 : 0), Math.sin(a) * d), '#ffffff'));
  }
  return merge(parts);
}

// A soft round sprite on a canvas: `inner` at the centre fading to `outer`.
export function softSprite(inner, outer) {
  var c = document.createElement('canvas');
  c.width = c.height = 64;
  var x = c.getContext('2d'), g = x.createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0, inner);
  g.addColorStop(0.4, inner.replace(/[\d.]+\)$/, '0.55)'));
  g.addColorStop(1, outer);
  x.fillStyle = g;
  x.fillRect(0, 0, 64, 64);
  var t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// ── Sky ──────────────────────────────────────────────────────────────────
// Gradient dome (horizon -> mid -> top) with an optional sun glow. `dark`
// fades it towards near-black. Add it to a group that follows the camera.
export function skyDome(colors, radius) {
  var mat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false,
    uniforms: {
      top: { value: new THREE.Color(colors.top) }, mid: { value: new THREE.Color(colors.mid) },
      horizon: { value: new THREE.Color(colors.horizon) }, dark: { value: 0 },
      sunDir: { value: new THREE.Vector3(0, 1, 0) }, sunColor: { value: new THREE.Color(colors.sun || '#000000') }
    },
    vertexShader: 'varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
    fragmentShader: 'uniform vec3 top; uniform vec3 mid; uniform vec3 horizon; uniform float dark; uniform vec3 sunDir; uniform vec3 sunColor; varying vec3 vP;' +
      'void main(){ float h = clamp(vP.y, 0.0, 1.0);' +
      ' vec3 c = mix(horizon, mid, smoothstep(0.0, 0.22, h)); c = mix(c, top, smoothstep(0.22, 0.75, h));' +
      ' float s = max(dot(normalize(vP), normalize(sunDir)), 0.0);' +
      ' c += sunColor * (pow(s, 600.0) * 4.0 + pow(s, 12.0) * 0.35);' +
      ' gl_FragColor = vec4(mix(c, vec3(0.012,0.016,0.03), dark), 1.0); }'
  });
  return { mesh: new THREE.Mesh(new THREE.SphereGeometry(radius || 1200, 32, 16), mat), uniforms: mat.uniforms };
}

// Stars on a sphere; `minY` keeps them above the horizon (-1 for all round).
export function starField(r, count, radius, minY, size) {
  var pos = [];
  for (var i = 0; i < count; i++) {
    var th = r() * Math.PI * 2, y = minY + r() * (1 - minY), s = Math.sqrt(1 - y * y);
    pos.push(radius * s * Math.cos(th), radius * y, radius * s * Math.sin(th));
  }
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  var mat = new THREE.PointsMaterial({ color: '#dfe7fa', size: size || 1.6, sizeAttenuation: false, fog: false,
                                       transparent: true, opacity: 0.85, depthWrite: false });
  return new THREE.Points(geo, mat);
}

// ── Ground ───────────────────────────────────────────────────────────────
// Distance from (x, z) to a sampled path (an array of Vector3).
export function distanceTo(points) {
  return function (x, z) {
    var best = 1e18;
    for (var i = 0; i < points.length; i++) {
      var dx = points[i].x - x, dz = points[i].z - z, d = dx * dx + dz * dz;
      if (d < best) best = d;
    }
    return Math.sqrt(best);
  };
}

// A square of terrain centred on (cx, cz). `color(x, z, y)` may return a
// THREE.Color for per-vertex colouring (the material needs vertexColors).
export function terrain(size, seg, cx, cz, height, material, color) {
  var geo = new THREE.PlaneGeometry(size, size, seg, seg).rotateX(-Math.PI / 2).translate(cx, 0, cz);
  var p = geo.attributes.position, cols = color ? new Float32Array(p.count * 3) : null;
  for (var i = 0; i < p.count; i++) {
    var x = p.getX(i), z = p.getZ(i), y = height(x, z);
    p.setY(i, y);
    if (cols) { var c = color(x, z, y); cols[i * 3] = c.r; cols[i * 3 + 1] = c.g; cols[i * 3 + 2] = c.b; }
  }
  if (cols) geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  geo.computeVertexNormals();
  var mesh = new THREE.Mesh(geo, material);
  mesh.receiveShadow = true;
  return mesh;
}

// A flat ribbon following a path just above the ground: a road or a track.
export function ribbon(points, offset, width, height, lift) {
  var pos = [];
  for (var j = 0; j < points.length - 1; j++) {
    var a = points[j], b = points[j + 1], dx = b.x - a.x, dz = b.z - a.z, len = Math.hypot(dx, dz) || 1;
    var nx = -dz / len, nz = dx / len, w = width / 2;
    var ya = height(a.x, a.z) + lift, yb = height(b.x, b.z) + lift;
    pos.push(a.x + nx * (offset - w), ya, a.z + nz * (offset - w), b.x + nx * (offset - w), yb, b.z + nz * (offset - w),
             a.x + nx * (offset + w), ya, a.z + nz * (offset + w), a.x + nx * (offset + w), ya, a.z + nz * (offset + w),
             b.x + nx * (offset - w), yb, b.z + nz * (offset - w), b.x + nx * (offset + w), yb, b.z + nz * (offset + w));
  }
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.computeVertexNormals();
  return geo;
}

// Fill an InstancedMesh: place(i, pos, quat, scale, color) sets the
// transform (and optionally a colour) and returns false to reject a try.
export function scatter(mesh, tries, place) {
  var m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3();
  var c = new THREE.Color(), n = 0, cap = mesh.instanceMatrix.count;
  for (var t = 0; t < tries && n < cap; t++) {
    c.setRGB(1, 1, 1);
    if (place(n, p, q, s, c) === false) continue;
    mesh.setMatrixAt(n, m.compose(p, q, s));
    mesh.setColorAt(n, c);
    n++;
  }
  mesh.count = n;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  return n;
}

// ── Particles that travel with the camera ───────────────────────────────
// Points in a box around the camera, wrapped per axis, falling and swaying:
// snow, leaves, pollen. update(f, center) moves them; f.snow sets how many
// are drawn and f.wind pushes them sideways.
export function particleField(o) {
  var N = o.count, B = o.box, pos = new Float32Array(N * 3), speed = new Float32Array(N), phase = new Float32Array(N);
  var cols = o.colors ? new Float32Array(N * 3) : null, tmp = new THREE.Color();
  for (var i = 0; i < N; i++) {
    pos[i * 3] = (Math.random() - 0.5) * B[0];
    pos[i * 3 + 1] = Math.random() * B[1];
    pos[i * 3 + 2] = (Math.random() - 0.5) * B[2];
    speed[i] = o.fall[0] + Math.random() * (o.fall[1] - o.fall[0]);
    phase[i] = Math.random() * 6.28;
    if (cols) { tmp.set(o.colors[i % o.colors.length]); cols[i * 3] = tmp.r; cols[i * 3 + 1] = tmp.g; cols[i * 3 + 2] = tmp.b; }
  }
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  if (cols) geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  var points = new THREE.Points(geo, new THREE.PointsMaterial({
    color: o.colors ? '#ffffff' : (o.color || '#ffffff'), vertexColors: !!cols, size: o.size, map: o.map,
    transparent: true, depthWrite: false, sizeAttenuation: true, alphaTest: o.alphaTest || 0
  }));
  points.frustumCulled = false;

  function wrap(v, c, size) { return c - size / 2 + ((((v - c + size / 2) % size) + size) % size); }

  function update(f, center, slow) {
    var rate = slow ? 0.5 : 1, wx = f.wind * (o.windSpeed || 5.5), sway = o.sway || 0.35;
    for (var i = 0; i < N; i++) {
      var k = i * 3;
      pos[k] += (wx + Math.sin(f.time * 0.9 + phase[i]) * sway) * f.dt * rate;
      pos[k + 1] -= speed[i] * f.dt * rate;
      pos[k + 2] += Math.cos(f.time * 0.7 + phase[i]) * sway * 0.6 * f.dt * rate;
      pos[k] = wrap(pos[k], center.x, B[0]);
      pos[k + 1] = wrap(pos[k + 1], center.y + B[1] * 0.3, B[1]);
      pos[k + 2] = wrap(pos[k + 2], center.z, B[2]);
    }
    geo.attributes.position.needsUpdate = true;
    geo.setDrawRange(0, Math.floor(N * clamp01(f.snow)));
  }
  return { points: points, update: update };
}

function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

// ── Camera ───────────────────────────────────────────────────────────────
// Put the camera at progress t on a path, `eye` metres above the ground,
// looking `ahead` (fraction of the path) further on, then turn by yaw/pitch
// plus a little pointer look-around. Past the end it keeps going straight.
export function followPath(camera, curve, height, t, o) {
  var look = followPath.look || (followPath.look = new THREE.Vector3());
  var tan = followPath.tan || (followPath.tan = new THREE.Vector3());
  curve.getPointAt(t, camera.position);
  camera.position.y = height(camera.position.x, camera.position.z) + o.eye + (o.lift || 0) +
                      Math.sin(o.time * 1.1) * 0.025;
  var ahead = t + (o.ahead || 0.035);
  if (ahead <= 1) curve.getPointAt(ahead, look);
  else { curve.getTangentAt(1, tan); curve.getPointAt(1, look).addScaledVector(tan, (ahead - 1) * curve.getLength()); }
  look.y = height(look.x, look.z) + o.eye - 0.25 + (o.lift || 0);
  camera.lookAt(look);
  camera.rotateY((o.yaw || 0) - o.mx * 0.16);
  camera.rotateX((o.pitch || 0) - o.my * 0.07);
}

// ── Teardown ─────────────────────────────────────────────────────────────
export function disposeAll(world, gl) {
  world.traverse(function (o) {
    if (o.geometry) o.geometry.dispose();
    var mats = o.material ? [].concat(o.material) : [];
    mats.forEach(function (m) {
      ['map', 'bumpMap', 'normalMap', 'emissiveMap'].forEach(function (k) { if (m[k]) m[k].dispose(); });
      if (m.uniforms) Object.keys(m.uniforms).forEach(function (k) { var v = m.uniforms[k].value; if (v && v.isTexture) v.dispose(); });
      m.dispose();
    });
  });
  gl.dispose();
  gl.forceContextLoss();
}
