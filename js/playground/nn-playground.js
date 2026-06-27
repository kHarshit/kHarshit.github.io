(function () {
  if (typeof MNIST_WEIGHTS === "undefined") return;

  var W = MNIST_WEIGHTS;

  /* ── CNN helpers ──────────────────────────────── */
  function conv2d(input, weight, bias, padding) {
    if (padding === undefined) padding = 1;
    var inCh = input.length, inH = input[0].length, inW = input[0][0].length;
    var outCh = weight.length, kH = weight[0][0].length, kW = weight[0][0][0].length;
    var outH = inH + 2 * padding - kH + 1;
    var outW = inW + 2 * padding - kW + 1;
    if (outH <= 0) outH = 1;
    if (outW <= 0) outW = 1;
    var out = [];
    for (var oc = 0; oc < outCh; oc++) {
      var rows = [];
      for (var h = 0; h < outH; h++) {
        var cols = [];
        for (var w = 0; w < outW; w++) {
          var s = bias[oc];
          for (var ic = 0; ic < inCh; ic++) {
            for (var kh = 0; kh < kH; kh++) {
              var ih = h + kh - padding;
              if (ih < 0 || ih >= inH) continue;
              for (var kw = 0; kw < kW; kw++) {
                var iw = w + kw - padding;
                if (iw < 0 || iw >= inW) continue;
                s += input[ic][ih][iw] * weight[oc][ic][kh][kw];
              }
            }
          }
          cols.push(s);
        }
        rows.push(cols);
      }
      out.push(rows);
    }
    return out;
  }

  function maxPool2d(input) {
    var ch = input.length, inH = input[0].length, inW = input[0][0].length;
    var outH = inH >> 1, outW = inW >> 1;
    var out = [];
    for (var c = 0; c < ch; c++) {
      var rows = [];
      for (var h = 0; h < outH; h++) {
        var cols = [];
        for (var w = 0; w < outW; w++) {
          var max = input[c][h * 2][w * 2];
          max = Math.max(max, input[c][h * 2][w * 2 + 1]);
          max = Math.max(max, input[c][h * 2 + 1][w * 2]);
          max = Math.max(max, input[c][h * 2 + 1][w * 2 + 1]);
          cols.push(max);
        }
        rows.push(cols);
      }
      out.push(rows);
    }
    return out;
  }

  function flatten3d(input) {
    var arr = [];
    for (var c = 0; c < input.length; c++)
      for (var h = 0; h < input[c].length; h++)
        for (var w = 0; w < input[c][h].length; w++)
          arr.push(input[c][h][w]);
    return arr;
  }

  function fc(input, weight, bias, relu) {
    if (relu === undefined) relu = true;
    var out = [];
    for (var j = 0; j < weight.length; j++) {
      var s = bias[j];
      for (var i = 0; i < input.length; i++) s += input[i] * weight[j][i];
      out.push(relu ? (s > 0 ? s : 0) : s);
    }
    return out;
  }

  function relu3d(arr) {
    for (var c = 0; c < arr.length; c++)
      for (var h = 0; h < arr[c].length; h++)
        for (var w = 0; w < arr[c][h].length; w++)
          arr[c][h][w] = Math.max(0, arr[c][h][w]);
    return arr;
  }

  function softmax(logits) {
    var max = logits[0];
    for (var i = 1; i < logits.length; i++) if (logits[i] > max) max = logits[i];
    var exps = [];
    var sum = 0;
    for (var i = 0; i < logits.length; i++) { exps.push(Math.exp(logits[i] - max)); sum += exps[i]; }
    for (var i = 0; i < logits.length; i++) exps[i] /= sum;
    return exps;
  }

  /* ── LeNet-5 forward pass ─────────────────────── */

  function predict(input) {
    var x = [[input.slice(0, 28)]];
    for (var i = 1; i < 28; i++) x[0].push(input.slice(i * 28, i * 28 + 28));

    x = conv2d(x, W.conv1w, W.conv1b, 2);  // Conv1 pad=2 → 28×28
    x = relu3d(x);                           // ReLU
    x = maxPool2d(x);                       // Pool → 14×14
    x = conv2d(x, W.conv2w, W.conv2b, 0);  // Conv2 pad=0 → 10×10
    x = relu3d(x);                           // ReLU
    x = maxPool2d(x);                       // Pool → 5×5
    x = flatten3d(x);                        // 400
    x = fc(x, W.fc1w, W.fc1b, true);        // FC → 120 + ReLU
    x = fc(x, W.fc2w, W.fc2b, true);        // FC → 84 + ReLU
    x = fc(x, W.fc3w, W.fc3b, false);       // FC → 10, no ReLU
    return softmax(x);
  }

  /* ── Drawing canvas ───────────────────────────── */
  var drawCanvas = document.getElementById("drawCanvas");
  var dctx = drawCanvas.getContext("2d");
  var DW = 280, DH = 280;
  var drawing = false;

  dctx.fillStyle = "#000";
  dctx.fillRect(0, 0, DW, DH);
  dctx.strokeStyle = "#fff";
  dctx.lineWidth = 28;
  dctx.lineCap = "round";
  dctx.lineJoin = "round";

  function getPos(e) {
    var rect = drawCanvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (DW / rect.width),
      y: (e.clientY - rect.top) * (DH / rect.height)
    };
  }

  drawCanvas.addEventListener("mousedown", function (e) {
    drawing = true;
    var p = getPos(e);
    dctx.beginPath();
    dctx.moveTo(p.x, p.y);
  });

  drawCanvas.addEventListener("mousemove", function (e) {
    if (!drawing) return;
    var p = getPos(e);
    dctx.lineTo(p.x, p.y);
    dctx.stroke();
    classify();
  });

  drawCanvas.addEventListener("mouseup", function () { drawing = false; });
  drawCanvas.addEventListener("mouseleave", function () { drawing = false; });

  /* ── Touch support ────────────────────────────── */
  drawCanvas.addEventListener("touchstart", function (e) {
    e.preventDefault();
    var t = e.touches[0];
    var rect = drawCanvas.getBoundingClientRect();
    var x = (t.clientX - rect.left) * (DW / rect.width);
    var y = (t.clientY - rect.top) * (DH / rect.height);
    drawing = true;
    dctx.beginPath();
    dctx.moveTo(x, y);
  }, { passive: false });

  drawCanvas.addEventListener("touchmove", function (e) {
    e.preventDefault();
    if (!drawing) return;
    var t = e.touches[0];
    var rect = drawCanvas.getBoundingClientRect();
    var x = (t.clientX - rect.left) * (DW / rect.width);
    var y = (t.clientY - rect.top) * (DH / rect.height);
    dctx.lineTo(x, y);
    dctx.stroke();
    classify();
  }, { passive: false });

  drawCanvas.addEventListener("touchend", function (e) {
    e.preventDefault();
    drawing = false;
  }, { passive: false });

  /* ── Downscale 280×280 → 28×28 ────────────────── */
  function getInput() {
    var imageData = dctx.getImageData(0, 0, DW, DH);
    var data = imageData.data;
    var input = new Array(784);
    var cell = 10;
    for (var row = 0; row < 28; row++) {
      for (var col = 0; col < 28; col++) {
        var sum = 0;
        for (var dy = 0; dy < cell; dy++) {
          for (var dx = 0; dx < cell; dx++) {
            var idx = ((row * cell + dy) * DW + (col * cell + dx)) * 4;
            sum += data[idx];
          }
        }
        input[row * 28 + col] = sum / (cell * cell * 255);
      }
    }
    var maxVal = 0;
    for (var i = 0; i < 784; i++) if (input[i] > maxVal) maxVal = input[i];
    if (maxVal > 0.01) {
      for (var i = 0; i < 784; i++) input[i] /= maxVal;
    }
    return input;
  }

  /* ── Render preview (28×28) ───────────────────── */
  var preCanvas = document.getElementById("previewCanvas");
  var pctx = preCanvas.getContext("2d");
  var PW = 84, PH = 84;

  function renderPreview(input) {
    pctx.fillStyle = "#000";
    pctx.fillRect(0, 0, PW, PH);
    var imgData = pctx.createImageData(PW, PH);
    for (var row = 0; row < 28; row++) {
      for (var col = 0; col < 28; col++) {
        var v = Math.round(Math.max(0, Math.min(255, input[row * 28 + col] * 255)));
        for (var dy = 0; dy < 3; dy++) {
          for (var dx = 0; dx < 3; dx++) {
            var idx = ((row * 3 + dy) * PW + (col * 3 + dx)) * 4;
            imgData.data[idx] = v;
            imgData.data[idx + 1] = v;
            imgData.data[idx + 2] = v;
            imgData.data[idx + 3] = 255;
          }
        }
      }
    }
    pctx.putImageData(imgData, 0, 0);
  }

  /* ── Prediction bars ──────────────────────────── */
  var barsEl = document.getElementById("predictionBars");
  var resultEl = document.getElementById("predictionResult");
  var barColors = ["#e74c3c","#e67e22","#f1c40f","#2ecc71","#1abc9c",
                   "#3498db","#9b59b6","#e91e63","#20B2AA","#34495e"];

  function classify() {
    var input = getInput();
    renderPreview(input);
    var signal = 0;
    for (var i = 0; i < 784; i++) signal += input[i];
    var output = signal < 0.5 ? null : predict(input);

    var pred = 0;
    if (output) {
      for (var i = 1; i < 10; i++) if (output[i] > output[pred]) pred = i;
    }
    var conf = output ? (output[pred] * 100).toFixed(1) : "0.0";

    var html = "";
    for (var i = 0; i < 10; i++) {
      var pct = output ? (output[i] * 100).toFixed(1) : "0.0";
      var barW = output ? Math.max(2, output[i] * 100) : 2;
      var cls = output && i === pred ? "nn-bar-active" : "";
      html += '<div class="nn-bar-row ' + cls + '">';
      html += '<span class="nn-bar-label">' + i + '</span>';
      html += '<span class="nn-bar-track"><span class="nn-bar-fill" style="width:' + barW + '%;background:' + barColors[i] + '"></span></span>';
      html += '<span class="nn-bar-pct">' + pct + '%</span>';
      html += '</div>';
    }
    barsEl.innerHTML = html;
    resultEl.innerHTML = output ? 'Predicted: <strong>' + pred + '</strong>  (' + conf + '% confidence)' : "Draw a digit to see the prediction";
  }

  /* ── Clear ────────────────────────────────────── */
  document.getElementById("clearBtn").addEventListener("click", function () {
    dctx.fillStyle = "#000";
    dctx.fillRect(0, 0, DW, DH);
    barsEl.innerHTML = "";
    resultEl.textContent = "Draw a digit to see the prediction";
    pctx.fillStyle = "#000";
    pctx.fillRect(0, 0, PW, PH);
    classify();
  });

  /* ── Init ─────────────────────────────────────── */
  pctx.fillStyle = "#000";
  pctx.fillRect(0, 0, PW, PH);
  classify();
})();
