/*
 * Render the image layers for the `snowy-woods-layers` scene from the
 * three.js scene (js/site/immersive/scenes/snowy-woods-3d.js).
 *
 * One fixed viewpoint is rendered once per depth band, on a transparent
 * background, so each band becomes a cut-out picture of everything at that
 * distance. Trees and houses are clipped to the band; the ground (meshes
 * named 'ground') runs all the way to the camera in every band, so a nearer
 * layer always sits on continuous snow and no seams open between bands as
 * they move apart. Each is cropped to its visible
 * pixels and posted to scripts/immersive/receive-layers.py with a
 * layers.json manifest that places it in the layer engine's 2000x1200 space
 * (horizon at y=720) at a depth z.
 *
 * Usage (with `bundle exec jekyll serve` running):
 *   python3 scripts/immersive/receive-layers.py img/poems/immersive/snowy-woods
 *   open /poems/stopping-by-woods/ and run in the console:
 *     (await import('/scripts/immersive/capture-layers.js')).capture()
 *
 * The same manifest format works for hand-painted layers: see
 * PoemImmersive.imageScene in js/site/immersive/engine.js.
 */
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js';

// Distances in metres from the viewpoint. `z` is the band's depth in layer
// units (25 per metre, roughly its typical distance), which sets how much it
// moves when the camera dollies. `scale` trims resolution for far bands,
// which never get magnified much.
var BANDS = [
  { name: 'mountains', near: 150, far: 1500, z: 7500, scale: 0.6 },
  { name: 'far', near: 70, far: 150, z: 2500, scale: 0.9 },
  { name: 'mid', near: 30, far: 70, z: 1125, scale: 1 },
  { name: 'near', near: 12, far: 30, z: 475, scale: 1 },
  { name: 'foreground', near: 0.1, far: 12, z: 150, scale: 1 }
];

// Layer-space area to render: the 2000x1200 frame plus margins for parallax
// and for portrait screens, which see more above and below.
var AREA = { x: -300, y: -300, w: 2600, h: 2000 };

export async function capture(opts) {
  opts = Object.assign({ t: 0.08, yaw: 0.05, fov: 52, res: 1.5, quality: 0.86,
                         endpoint: 'http://localhost:8765/save' }, opts);
  var PI = window.PoemImmersive;
  if (!PI) throw new Error('Open a poem page with the immersive engine first.');

  // Re-run the scene module and keep its definition instead of starting it.
  var def, register = PI.register;
  PI.register = function (name, scene) { if (name === 'snowy-woods-3d') def = scene; else register(name, scene); };
  try { await import('/js/site/immersive/scenes/snowy-woods-3d.js?capture=' + Date.now()); }
  finally { PI.register = register; }

  var canvas = document.createElement('canvas');
  var view = def.renderer(canvas, def, { reduceMotion: true, capture: true });
  var p = view.parts, cam = p.camera, gl = p.gl, S = opts.res;

  // Let the scene place the camera, lights and fog for this point on the
  // path, then level the camera so the horizon lands on y=720.
  view.resize(64, 64, 1);
  view.frame({ cam: opts.t, dark: 0, snow: 0, wind: 0, row: [opts.t, 0, 0, 0, opts.yaw],
               mx: 0, my: 0, dt: 0, time: 0 });
  var dir = new THREE.Vector3();
  cam.getWorldDirection(dir);
  dir.y = 0;
  cam.lookAt(cam.position.clone().add(dir.normalize()));

  var W = Math.round(AREA.w * S), H = Math.round(AREA.h * S);
  gl.setPixelRatio(1);
  gl.setSize(W, H, false);
  cam.fov = opts.fov;
  cam.aspect = 2000 / 1440;   // full frame centred on the horizon: 1440 = 2 x 720
  cam.setViewOffset(2000 * S, 1440 * S, AREA.x * S, AREA.y * S, W, H);
  p.snow.visible = false;
  p.sky.visible = false;      // the engine draws sky, stars and moon itself

  // Clip by distance along the (level) view axis with two planes per band.
  cam.near = 0.1;
  cam.far = 3000;
  cam.updateProjectionMatrix();
  gl.localClippingEnabled = true;
  var c = cam.position, ground = [], objects = [];
  p.world.traverse(function (o) {
    if (!o.material) return;
    (o.name === 'ground' ? ground : objects).push(o.material);
  });
  function slab(a, b) {
    return [new THREE.Plane(dir.clone(), -dir.dot(c) - a), new THREE.Plane(dir.clone().negate(), dir.dot(c) + b)];
  }

  var manifest = { area: AREA, layers: [] };
  for (var i = 0; i < BANDS.length; i++) {
    var b = BANDS[i];
    var groundSlab = slab(0.1, b.far), objectSlab = slab(b.near, b.far);
    ground.forEach(function (m) { m.clippingPlanes = groundSlab; });
    objects.forEach(function (m) { m.clippingPlanes = objectSlab; });
    gl.setClearColor(0x000000, 0);
    gl.clear();
    gl.render(p.world, cam);

    var crop = cropToAlpha(canvas, W, H, b.scale);
    var blob = await new Promise(function (res) { crop.canvas.toBlob(res, 'image/webp', opts.quality); });
    await post(opts.endpoint, b.name + '.webp', blob);
    manifest.layers.push({
      src: b.name + '.webp', z: b.z,
      x: +(AREA.x + crop.x / S).toFixed(1), y: +(AREA.y + crop.y / S).toFixed(1),
      w: +(crop.w / S).toFixed(1), h: +(crop.h / S).toFixed(1)
    });
    console.log('layer', b.name, crop.canvas.width + 'x' + crop.canvas.height, Math.round(blob.size / 1024) + ' KB');
  }
  await post(opts.endpoint, 'layers.json', new Blob([JSON.stringify(manifest, null, 2)], { type: 'application/json' }));
  view.destroy();
  return manifest;
}

// Copy the rendered frame, find the box of non-transparent pixels and return
// it cropped (and downscaled by `scale`).
function cropToAlpha(src, W, H, scale) {
  var full = document.createElement('canvas');
  full.width = W;
  full.height = H;
  var fx = full.getContext('2d', { willReadFrequently: true });
  fx.drawImage(src, 0, 0);
  var px = fx.getImageData(0, 0, W, H).data;
  var x0 = W, y0 = H, x1 = -1, y1 = -1;
  for (var y = 0; y < H; y++) {
    for (var x = 0; x < W; x++) {
      if (px[(y * W + x) * 4 + 3] > 2) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < 0) { x0 = y0 = 0; x1 = y1 = 1; }
  x0 = Math.max(0, x0 - 4); y0 = Math.max(0, y0 - 4);
  x1 = Math.min(W - 1, x1 + 4); y1 = Math.min(H - 1, y1 + 4);
  var w = x1 - x0 + 1, h = y1 - y0 + 1;
  var out = document.createElement('canvas');
  out.width = Math.round(w * scale);
  out.height = Math.round(h * scale);
  var ox = out.getContext('2d');
  ox.imageSmoothingQuality = 'high';
  ox.drawImage(full, x0, y0, w, h, 0, 0, out.width, out.height);
  return { canvas: out, x: x0, y: y0, w: w, h: h };
}

async function post(endpoint, name, blob) {
  var r = await fetch(endpoint + '?name=' + encodeURIComponent(name), { method: 'POST', body: blob });
  if (!r.ok) throw new Error('Saving ' + name + ' failed: ' + r.status);
}
