document.addEventListener('DOMContentLoaded', function () {

  // ── Scroll-to-top button ──────────────────────────────────
  window.addEventListener('scroll', function () {
    var btn = document.getElementById('scroll_top');
    if (!btn) return;
    var scrolled = document.body.scrollTop || document.documentElement.scrollTop;
    btn.style.display = scrolled > 600 ? 'block' : 'none';
  }, { passive: true });

  // ── Typing loop for hero title ────────────────────────────
  var el = document.getElementById('hero-title');
  if (el) {
    var titles = [
      'Machine Learning Engineer',
      'AI Research Engineer',
      'Generative AI Engineer',
      'Computer Vision Engineer',
      'LLM Engineer',
      'Deep Learning Engineer'
    ];
    var titleIndex = 0;
    var charIndex = 0;
    var isDeleting = false;
    var typeSpeed = 45;
    var deleteSpeed = 25;
    var pauseAfterType = 2000;
    var pauseAfterDelete = 500;

    function typeLoop() {
      var currentText = titles[titleIndex];
      if (!isDeleting) {
        el.textContent = currentText.substring(0, charIndex + 1);
        charIndex++;
        if (charIndex === currentText.length) {
          isDeleting = true;
          setTimeout(typeLoop, pauseAfterType);
          return;
        }
        setTimeout(typeLoop, typeSpeed);
      } else {
        el.textContent = currentText.substring(0, charIndex - 1);
        charIndex--;
        if (charIndex === 0) {
          isDeleting = false;
          titleIndex = (titleIndex + 1) % titles.length;
          setTimeout(typeLoop, pauseAfterDelete);
          return;
        }
        setTimeout(typeLoop, deleteSpeed);
      }
    }
    typeLoop();
  }

  // ── Chat widget ────────────────────────────────────────────
  (function() {
    var chatContainer = document.getElementById('chat-container');
    var messagesEl = document.getElementById('chat-messages');
    var inputEl = document.getElementById('chat-input');
    var sendBtn = document.getElementById('chat-send');
    if (!chatContainer || !messagesEl || !inputEl || !sendBtn) return;

    var state = 'init';
    var userName = '';
    var userTopic = '';
    var userExtra = '';
    var userEmail = '';

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
        addStatusMessage('<div class="chat-status error">⚠️ Chat backend not configured. <a href="mailto:kumar_harshit@outlook.com">Email me directly →</a></div>');
        return;
      }

      var repo = window.GITHUB_REPO || 'kHarshit/kHarshit.github.io';
      var endpoint = 'https://api.github.com/repos/' + repo + '/issues';
      var extras = userExtra ? '\n\n**Anything else:**\n' + userExtra : '';
      var body = '**Name:** ' + userName + '\n**Email:** ' + userEmail + '\n\n**Message:**\n' + userTopic + extras;

      fetch(endpoint, {
        method: 'POST',
        headers: {
          'Authorization': 'token ' + token,
          'Content-Type': 'application/json',
          'Accept': 'application/vnd.github.v3+json'
        },
        body: JSON.stringify({
          title: 'Portfolio Contact: ' + userName,
          body: body,
          labels: ['contact']
        })
      }).then(function (res) {
        hideTyping();
        if (res.ok) {
          addStatusMessage('<div class="chat-status success">✅ Message sent! I\'ll get back to you soon.</div>');
        } else {
          addStatusMessage('<div class="chat-status error">⚠️ Couldn\'t send. <a href="mailto:kumar_harshit@outlook.com">Email me directly →</a></div>');
        }
      }).catch(function () {
        hideTyping();
        addStatusMessage('<div class="chat-status error">⚠️ Couldn\'t send. <a href="mailto:kumar_harshit@outlook.com">Email me directly →</a></div>');
      });
    }

    function handleUserInput(text) {
      text = text.trim();
      if (!text || state === 'done') return;
      inputEl.value = '';
      addMessage(text, 'user');

      switch (state) {
        case 'init':
        case 'greeting':
          state = 'ask_name';
          botRespond("Awesome! What's your name?");
          break;
        case 'ask_name':
          userName = text;
          state = 'ask_topic';
          botRespond("Nice to meet you, " + userName + "! What's on your mind? (collaboration, job opp, or just saying hi)");
          break;
        case 'ask_topic':
          userTopic = text;
          state = 'ask_extra';
          botRespond("Anything else you'd like to add?");
          break;
        case 'ask_extra':
          userExtra = text;
          state = 'ask_email';
          botRespond("Got it! What's your email so I can get back to you?");
          break;
        case 'ask_email':
          userEmail = text;
          state = 'done';
          inputEl.disabled = true;
          sendBtn.disabled = true;
          botRespond("Thanks " + userName + "! Sending your message now...", 600);
          setTimeout(submitToGitHub, 1800);
          break;
      }
    }

    // Send greeting after a short delay
    setTimeout(function () {
      addMessage("\uD83D\uDC4B Hey! Want to work together or just chat? Drop a message below!", 'bot');
      state = 'greeting';
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

});

function topFunction() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
