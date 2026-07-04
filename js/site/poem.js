document.addEventListener('DOMContentLoaded', function () {
  var dataEl = document.getElementById('poem-explanations');
  var btn = document.getElementById('poem-explain-toggle');
  if (!dataEl || !btn) return;

  var explanations;
  try { explanations = JSON.parse(dataEl.textContent); } catch (e) { return; }
  if (!explanations || explanations.length === 0) { btn.style.display = 'none'; return; }

  var body = document.querySelector('.poem-body');
  if (!body) return;

  // ── Collect lines and group into stanzas ──
  var stanzas = [];
  var isList = false;
  var ul = body.querySelector(':scope > ul');

  if (ul) {
    // List-based (e.g. She Walks in Beauty)
    isList = true;
    var allLines = Array.from(ul.children);
    var current = [];
    allLines.forEach(function (line) {
      current.push(line);
      if (line.querySelector('p')) {
        stanzas.push(current);
        current = [];
      }
    });
    if (current.length > 0) stanzas.push(current);
  } else {
    // Paragraph-based (e.g. V's speech)
    var childEls = Array.from(body.children);
    var hrIdx = childEls.findIndex(function (el) { return el.tagName === 'HR'; });
    if (hrIdx === -1) hrIdx = childEls.length;
    var ps = childEls.slice(0, hrIdx).filter(function (el) { return el.tagName === 'P'; });
    if (ps.length === 0) return;
    var perStanza = Math.ceil(ps.length / explanations.length);
    for (var i = 0; i < ps.length; i += perStanza) {
      stanzas.push(ps.slice(i, i + perStanza));
    }
  }

  if (stanzas.length === 0) return;

  // ── Build layout: each stanza block has content + card side by side ──
  var grid = document.createElement('div');
  grid.className = 'poem-explain-grid';

  stanzas.forEach(function (lines, i) {
    var block = document.createElement('div');

    if (isList) {
      block.className = 'stanza-block';
      var stanzaUl = document.createElement('ul');
      lines.forEach(function (line) { stanzaUl.appendChild(line); });
      block.appendChild(stanzaUl);
    } else {
      block.className = 'stanza-block-column';
      var stanzaDiv = document.createElement('div');
      lines.forEach(function (line) { stanzaDiv.appendChild(line.cloneNode(true)); });
      block.appendChild(stanzaDiv);
    }

    if (i < explanations.length) {
      var card = document.createElement('div');
      card.className = 'stanza-explain-card';
      card.innerHTML = '<div class="stanza-explain-card-inner">' + explanations[i] + '</div>';
      block.appendChild(card);
    }

    grid.appendChild(block);
  });

  // ── Replace original content with grid ──
  if (ul) {
    ul.replaceWith(grid);
  } else {
    var firstP = childEls.find(function (el) { return el.tagName === 'P'; });
    if (firstP) {
      firstP.parentNode.insertBefore(grid, firstP);
      ps.forEach(function (p) { p.remove(); });
    }
  }

  // ── Single toggle button — show/hide all cards ──
  var cards = grid.querySelectorAll('.stanza-explain-card');

  btn.addEventListener('click', function () {
    var showing = grid.classList.toggle('show-explanations');
    cards.forEach(function (c) { c.classList.toggle('open', showing); });
    btn.innerHTML = showing
      ? '<i class="fa fa-info-circle"></i> Hide explanations'
      : '<i class="fa fa-info-circle"></i> Show explanations';
  });
});
