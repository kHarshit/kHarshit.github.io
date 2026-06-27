document.addEventListener('DOMContentLoaded', function () {
  var dataEl = document.getElementById('poem-explanations');
  var btn = document.getElementById('poem-explain-toggle');
  if (!dataEl || !btn) return;

  var explanations;
  try { explanations = JSON.parse(dataEl.textContent); } catch (e) { return; }
  if (!explanations || explanations.length === 0) { btn.style.display = 'none'; return; }

  var ul = document.querySelector('.poem-body > ul');
  if (!ul) return;

  var allLines = Array.from(ul.children);

  // Group <li> into stanzas — kramdown wraps the last line before a blank
  // line in <p>, so li:has(> p) marks the end of a stanza.
  var stanzas = [];
  var current = [];
  allLines.forEach(function (line) {
    current.push(line);
    if (line.querySelector('p')) {
      stanzas.push(current);
      current = [];
    }
  });
  if (current.length > 0) stanzas.push(current);

  // Build layout: each stanza block has <ul> + card side by side
  var grid = document.createElement('div');
  grid.className = 'poem-explain-grid';

  stanzas.forEach(function (lines, i) {
    var block = document.createElement('div');
    block.className = 'stanza-block';

    var stanzaUl = document.createElement('ul');
    lines.forEach(function (line) { stanzaUl.appendChild(line); });
    block.appendChild(stanzaUl);

    if (i < explanations.length) {
      var card = document.createElement('div');
      card.className = 'stanza-explain-card';
      card.innerHTML = '<div class="stanza-explain-card-inner">' + explanations[i] + '</div>';
      block.appendChild(card);
    }

    grid.appendChild(block);
  });

  ul.replaceWith(grid);

  // Single toggle button — show/hide all cards
  var cards = grid.querySelectorAll('.stanza-explain-card');

  btn.addEventListener('click', function () {
    var showing = grid.classList.toggle('show-explanations');
    cards.forEach(function (c) { c.classList.toggle('open', showing); });
    btn.innerHTML = showing
      ? '<i class="fa fa-info-circle"></i> Hide explanations'
      : '<i class="fa fa-info-circle"></i> Show explanations';
  });
});
