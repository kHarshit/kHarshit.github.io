// Contact chat widget — homepage only.
//
// This is the sole consumer of window.GITHUB_TOKEN, which index.html defines
// just above the tag that loads this file. Keeping the two together is
// deliberate: the API credential is only served on the one page that uses it,
// rather than on every page via the shared layout.
(function() {
  var chatContainer = document.getElementById('chat-container');
  var messagesEl = document.getElementById('chat-messages');
  var inputEl = document.getElementById('chat-input');
  var sendBtn = document.getElementById('chat-send');
  if (!chatContainer || !messagesEl || !inputEl || !sendBtn) return;

  // The questions, in order. `answers` is keyed by `key`.
  // Kept as data rather than a switch so a step can be re-asked on its own when
  // the visitor edits it from the review card.
  // `input` retunes the single text field for each question: it drives which
  // on-screen keyboard phones show and which autofill entry the browser offers.
  var STEPS = [
    { key: 'name',  ask: function () { return "Awesome! What's your name?"; },
      label: 'Name',
      input: { type: 'text', autocomplete: 'name' } },
    { key: 'topic', ask: function (a) { return 'Nice to meet you, ' + a.name + "! What's on your mind? (collaboration, job opp, or just saying hi)"; },
      label: 'Message',
      input: { type: 'text', autocomplete: 'off' } },
    { key: 'extra', ask: function () { return "Anything else you'd like to add?"; },
      label: 'Anything else', optional: true,
      input: { type: 'text', autocomplete: 'off' } },
    { key: 'email', ask: function () { return "Got it! What's your email so I can get back to you?"; },
      label: 'Email',
      input: { type: 'email', inputmode: 'email', autocomplete: 'email' },
      // Deliberately loose: just enough to catch "asfd" or a missing @, without
      // rejecting valid-but-unusual addresses.
      validate: function (v) {
        return /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(v) ? null
          : "Hmm, that doesn't look like an email address — mind typing it again?";
      } }
  ];

  var answers = {};
  var stepIndex = 0;
  var editingKey = null;   // set when re-answering a single field from the review
  var phase = 'init';      // init | greeting | asking | review | sending | done

  function addMessage(text, sender) {
    var div = document.createElement('div');
    div.className = 'chat-msg ' + sender;
    var bubble = document.createElement('div');
    bubble.className = 'chat-bubble';
    bubble.appendChild(document.createTextNode(text));
    div.appendChild(bubble);
    messagesEl.appendChild(div);
    scrollToBottom();
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function showTyping() {
    if (document.getElementById('chat-typing')) return;
    var div = document.createElement('div');
    div.className = 'chat-msg bot typing';
    div.id = 'chat-typing';
    var bubble = document.createElement('div');
    bubble.className = 'chat-bubble';
    bubble.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
    div.appendChild(bubble);
    messagesEl.appendChild(div);
    scrollToBottom();
  }

  function hideTyping() {
    var el = document.getElementById('chat-typing');
    if (el) el.remove();
  }

  function botRespond(text, delay) {
    delay = delay || 800;
    showTyping();
    setTimeout(function () {
      hideTyping();
      addMessage(text, 'bot');
    }, delay);
  }

  function addStatusMessage(html) {
    var div = document.createElement('div');
    div.className = 'chat-done';
    div.innerHTML = html;
    messagesEl.appendChild(div);
    scrollToBottom();
  }

  function submitToGitHub() {
    showTyping();

    var token = window.GITHUB_TOKEN || '';

    if (!token) {
      hideTyping();
      phase = 'done';
      addStatusMessage('<div class="chat-status error">⚠️ Chat backend not configured. <a href="mailto:kumar_harshit@outlook.com">Email me directly →</a></div>');
      return;
    }

    var repo = window.GITHUB_REPO || 'kHarshit/kHarshit.github.io';
    var endpoint = 'https://api.github.com/repos/' + repo + '/issues';
    var extras = answers.extra ? '\n\n**Anything else:**\n' + answers.extra : '';
    var body = '**Name:** ' + answers.name + '\n**Email:** ' + answers.email + '\n\n**Message:**\n' + answers.topic + extras;

    fetch(endpoint, {
      method: 'POST',
      headers: {
        'Authorization': 'token ' + token,
        'Content-Type': 'application/json',
        'Accept': 'application/vnd.github.v3+json'
      },
      body: JSON.stringify({
        title: 'Portfolio Contact: ' + answers.name,
        body: body,
        labels: ['contact']
      })
    }).then(function (res) {
      hideTyping();
      if (res.ok) {
        phase = 'done';
        addStatusMessage('<div class="chat-status success">✅ Message sent! I\'ll get back to you soon.</div>');
      } else {
        sendFailed();
      }
    }).catch(sendFailed);
  }

  // A failed send used to be a dead end: input stayed disabled and the typed
  // message was gone. Put the review card back so Send can be retried and the
  // text is still on screen to copy.
  function sendFailed() {
    hideTyping();
    addStatusMessage('<div class="chat-status error">⚠️ Couldn\'t send. Try again, or <a href="mailto:kumar_harshit@outlook.com">email me directly →</a></div>');
    showReview();
  }

  function setInputEnabled(on, placeholder) {
    inputEl.disabled = !on;
    sendBtn.disabled = !on;
    inputEl.placeholder = placeholder || 'Type a message...';
    if (on) inputEl.focus();
  }

  // Reapplied on every step (including when re-answering one field from the
  // review), so switching away from the email question also clears its keyboard
  // and autofill hints rather than leaving them stuck on.
  function applyInputHints(step) {
    var hints = (step && step.input) || { type: 'text', autocomplete: 'off' };
    inputEl.type = hints.type || 'text';
    inputEl.setAttribute('autocomplete', hints.autocomplete || 'off');
    if (hints.inputmode) {
      inputEl.setAttribute('inputmode', hints.inputmode);
    } else {
      inputEl.removeAttribute('inputmode');
    }
  }

  function askStep(i, delay) {
    stepIndex = i;
    phase = 'asking';
    applyInputHints(STEPS[i]);
    setInputEnabled(true);
    botRespond(STEPS[i].ask(answers), delay);
  }

  // \u2500\u2500 Review card \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  // The send only happens from here, so every answer is visible and editable
  // before anything leaves the browser.
  function showReview() {
    phase = 'review';
    setInputEnabled(false, 'Check your details above');

    var card = document.createElement('div');
    card.className = 'chat-review';
    card.id = 'chat-review';

    var heading = document.createElement('p');
    heading.className = 'chat-review-heading';
    heading.textContent = "Here's what I'll send \u2014 all good?";
    card.appendChild(heading);

    STEPS.forEach(function (step) {
      var value = answers[step.key] || '';
      if (step.optional && !value) return;

      var row = document.createElement('div');
      row.className = 'chat-review-row';

      var label = document.createElement('span');
      label.className = 'chat-review-label';
      label.textContent = step.label;

      var val = document.createElement('span');
      val.className = 'chat-review-value';
      val.textContent = value;

      var edit = document.createElement('button');
      edit.type = 'button';
      edit.className = 'chat-review-edit';
      edit.textContent = 'Edit';
      edit.setAttribute('aria-label', 'Edit ' + step.label);
      edit.addEventListener('click', function () { editField(step.key); });

      row.appendChild(label);
      row.appendChild(val);
      row.appendChild(edit);
      card.appendChild(row);
    });

    var actions = document.createElement('div');
    actions.className = 'chat-review-actions';

    var send = document.createElement('button');
    send.type = 'button';
    send.className = 'chat-review-send';
    send.textContent = 'Send';
    send.addEventListener('click', function () {
      removeReview();
      phase = 'sending';
      botRespond('Thanks ' + answers.name + '! Sending your message now...', 500);
      setTimeout(submitToGitHub, 1400);
    });

    var restart = document.createElement('button');
    restart.type = 'button';
    restart.className = 'chat-review-restart';
    restart.textContent = 'Start over';
    restart.addEventListener('click', function () {
      removeReview();
      answers = {};
      addMessage('Let\u2019s start over.', 'bot');
      askStep(0, 400);
    });

    actions.appendChild(send);
    actions.appendChild(restart);
    card.appendChild(actions);

    messagesEl.appendChild(card);
    scrollToBottom();
    send.focus();
  }

  function removeReview() {
    var el = document.getElementById('chat-review');
    if (el) el.remove();
  }

  function editField(key) {
    removeReview();
    editingKey = key;
    for (var i = 0; i < STEPS.length; i++) {
      if (STEPS[i].key === key) { askStep(i, 300); return; }
    }
  }

  function handleUserInput(text) {
    text = text.trim();
    if (phase === 'review' || phase === 'sending' || phase === 'done') return;

    if (phase === 'init' || phase === 'greeting') {
      if (!text) return;
      inputEl.value = '';
      addMessage(text, 'user');
      askStep(0);
      return;
    }

    var step = STEPS[stepIndex];
    if (!text && !step.optional) return;
    inputEl.value = '';
    addMessage(text || '\u2014', 'user');

    if (step.validate) {
      var problem = step.validate(text);
      if (problem) {
        botRespond(problem, 500);   // stay on this step
        return;
      }
    }

    answers[step.key] = text;

    // Editing a single field from the review returns straight to the review.
    if (editingKey) {
      editingKey = null;
      showReview();
      return;
    }

    if (stepIndex + 1 < STEPS.length) {
      askStep(stepIndex + 1);
    } else {
      setInputEnabled(false, 'Check your details above');
      showTyping();
      setTimeout(function () { hideTyping(); showReview(); }, 700);
    }
  }

  // Send greeting after a short delay
  setTimeout(function () {
    addMessage("\uD83D\uDC4B Hey! Want to work together or just chat? Drop a message below!", 'bot');
    phase = 'greeting';
  }, 600);

  sendBtn.addEventListener('click', function () {
    handleUserInput(inputEl.value);
  });

  inputEl.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleUserInput(inputEl.value);
    }
  });
})();
