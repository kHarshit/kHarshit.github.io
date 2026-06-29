(function () {
  'use strict';

  var QUESTIONS = [
    // Biology & Medicine
    {
      domain: 'bio',
      question: 'A biochemist observes that an enzyme\'s activity drops to 15% of its maximum when a specific histidine residue is mutated to alanine. The crystal structure shows this histidine is 3.8 Å from the substrate-binding pocket. Which role is this histidine MOST likely playing in catalysis?',
      options: [
        'Direct covalent catalysis via a nucleophilic attack',
        'Electrostatic stabilization of a transition state intermediate',
        'Metal ion coordination in a metalloenzyme active site',
        'Structural maintenance of the protein fold via hydrophobic packing'
      ],
      correct: 1,
      explanation: 'Histidine near the active site at hydrogen-bonding distance (3.8 Å) typically functions in general acid-base or electrostatic stabilization. An 85% activity loss upon mutation to alanine (which removes the imidazole ring) confirms its catalytic role beyond structural integrity.'
    },
    {
      domain: 'bio',
      question: 'A patient presents with progressive muscle weakness, elevated serum creatine kinase, and myotonic discharges on EMG. Genetic testing reveals an expanded CTG repeat in the 3’-UTR of the DMPK gene. The expanded repeat is transcribed but NOT translated. Which molecular mechanism BEST explains the pathogenesis?',
      options: [
        'Loss of function due to haploinsufficiency of DMPK protein',
        'RNA gain-of-function: CUG-repeat RNA sequesters splicing regulators',
        'Polyglutamine toxicity from translated CAG repeats',
        'Chromatin silencing via histone deacetylation at the repeat locus'
      ],
      correct: 1,
      explanation: 'Myotonic dystrophy type 1 is caused by expanded CTG repeats in the 3’-UTR of DMPK. The repeat is transcribed into CUG-repeat RNA, which forms hairpin structures that sequester MBNL1, causing mis-splicing of multiple downstream genes including the chloride channel CLCN1 (explaining myotonia).'
    },
    {
      domain: 'bio',
      question: 'A CAR-T cell therapy targeting CD19 achieves initial tumor regression in a B-cell malignancy. However, the patient relapses two months later and biopsies show tumor cells that have lost CD19 surface expression entirely. Which resistance mechanism is MOST likely?',
      options: [
        'T-cell exhaustion from persistent tonic CAR signaling',
        'Antigen escape via lineage switching to a CD19-negative phenotype',
        'Upregulation of PD-L1 on tumor cells blocking T-cell activation',
        'Frameshift mutation in the CAR transgene due to insertional mutagenesis'
      ],
      correct: 1,
      explanation: 'The most clinically documented mechanism of relapse after CD19-targeting CAR-T is antigen escape — tumor cells downregulate or lose CD19 expression, often through bilineal differentiation or lineage switching to a myeloid phenotype. While T-cell exhaustion and PD-L1 upregulation occur, they cause reduced T-cell activity rather than complete antigen loss.'
    },

    // Mathematics
    {
      domain: 'math',
      question: 'Let f: ℝ² → ℝ be defined by f(x,y) = x³y² / (x⁶ + y⁴) for (x,y) ≠ (0,0) and f(0,0) = 0. Which statement about f at (0,0) is true?',
      options: [
        'f is continuous but not differentiable',
        'f is differentiable and all directional derivatives exist',
        'f is not continuous, but all directional derivatives exist',
        'f has partial derivatives but is not differentiable'
      ],
      correct: 2,
      explanation: 'Along the path y = x³ᵉ², f approaches 1/2 ≠ 0, so f is not continuous at (0,0). Yet for any direction vector (h,k), the directional derivative exists and equals 0. This classic counterexample shows directional derivatives can exist even when a function is discontinuous.'
    },
    {
      domain: 'math',
      question: 'Let G be a finite group of order 2p, where p is an odd prime. Which of the following MUST be true?',
      options: [
        'G is cyclic',
        'G has a normal subgroup of order p',
        'G is abelian',
        'G has exactly p elements of order 2'
      ],
      correct: 1,
      explanation: 'By Cauchy\'s theorem, G has an element of order p, generating a subgroup H of order p. Since [G:H] = 2, H has index 2 in G. Any subgroup of index 2 is automatically normal (the two left cosets equal the two right cosets). G need not be cyclic — the dihedral group D_p has order 2p and is non-abelian for p > 2.'
    },
    {
      domain: 'math',
      question: 'Consider a biased random walk on ℤ starting at 0, moving +1 with probability p and −1 with probability 1−p, where p ≠ 1/2. What is the probability of ever returning to 0?',
      options: [
        '1 − |2p − 1|',
        '(1−p)/p if p > 1/2, else p/(1−p)',
        '2 min(p, 1−p)',
        '1 − p if p > 1/2, else 1 − (1−p) otherwise'
      ],
      correct: 2,
      explanation: 'From any starting point, we either step to +1 (prob p) or −1 (prob 1−p). The probability of reaching 0 from +1 is min(q/p, 1), and from −1 is min(p/q, 1). Combining: P(return) = p·min(q/p,1) + q·min(p/q,1) = 2min(p,q) = 2min(p, 1−p). This is strictly less than 1 for any biased walk, confirming transience.'
    },

    // Physics & Chemistry
    {
      domain: 'phys',
      question: 'A black hole of initial mass M₀ evaporates via Hawking radiation. Given that Hawking temperature T_H ∝ 1/M and treating the black hole as a blackbody radiator, how does its total evaporation lifetime scale with M₀?',
      options: [
        't ∝ M₀',
        't ∝ M₀²',
        't ∝ M₀³',
        't ∝ M₀^(1/2)'
      ],
      correct: 2,
      explanation: 'Power radiated P ∝ T⁴ × R² ∝ (1/M)⁴ × M² = 1/M² (since the Schwarzschild radius R ∝ M). So dM/dt ∝ −1/M², giving M² dM ∝ −dt. Integrating from M₀ to 0: M₀³/3 ∝ t, hence t ∝ M₀³. A solar-mass black hole would take ~10⁶⁷ years to evaporate.'
    },
    {
      domain: 'phys',
      question: 'In a nucleophilic aromatic substitution (SNAr) reaction, which substrate would react FASTEST with sodium methoxide (NaOMe) under identical conditions?',
      options: [
        '1-chloro-4-nitrobenzene',
        'Chlorobenzene',
        '1-chloro-4-methylbenzene',
        '1-chloro-4-methoxybenzene'
      ],
      correct: 0,
      explanation: 'SNAr proceeds via a Meisenheimer complex intermediate. Electron-withdrawing groups (especially nitro) at para or ortho positions strongly stabilize this anionic intermediate through resonance delocalization, dramatically accelerating the reaction. The nitro group is the strongest EWG here. Unsubstituted chlorobenzene is very slow; methyl and methoxy are electron-donating groups that deactivate the ring toward nucleophilic attack.'
    },
    {
      domain: 'phys',
      question: 'For the n=2 hydrogen atom in a weak static electric field (linear Stark effect), first-order degenerate perturbation theory must be applied. Which approach is correct?',
      options: [
        'Use the standard {|2s⟩, |2p_x⟩, |2p_y⟩, |2p_z⟩} basis unchanged',
        'Find linear combinations of the degenerate states that diagonalize the perturbation H′',
        'Use the spherical harmonics Y_l^m since they are always eigenstates of H′',
        'First-order perturbation theory cannot be applied to degenerate energy levels'
      ],
      correct: 1,
      explanation: 'When energy levels are degenerate (all n=2 states are degenerate in hydrogen), first-order perturbation theory requires constructing "good" states by diagonalizing H′ within the degenerate subspace. For the Stark effect H′ = eEz, the matrix elements connect |2s⟩ and |2p_z⟩ (giving a non-zero shift) while |2p_x⟩ and |2p_y⟩ decouple. Applying perturbation theory in the original basis without diagonalizing first gives divergent corrections.'
    },

    // Computer Science
    {
      domain: 'cs',
      question: 'n keys are inserted into a hash table with n buckets using uniform random hashing. What is the expected maximum chain length of any single bucket?',
      options: [
        'Θ(1)',
        'Θ(log n)',
        'Θ(log n / log log n)',
        'Θ(√n)'
      ],
      correct: 2,
      explanation: 'By the Poisson approximation, each bucket\'s load is approximately Poisson(1). The expected maximum of n independent Poisson(1) variables is Θ(log n / log log n), not Θ(log n). This tighter bound follows from a careful tail probability analysis: P(bucket has ≥ k keys) ≈ e^k/k^k, and balancing n × this probability gives the Θ(log n / log log n) bound. This result is a classic in average-case algorithm analysis.'
    },
    {
      domain: 'cs',
      question: 'A multithreaded computation has total work W₁ (sum of all operations across all threads) and span T_∞ (the longest sequential dependency chain). Which of the following are valid lower bounds on execution time on p processors?',
      options: [
        'Only Ω(W₁/p)',
        'Only Ω(T_∞)',
        'Ω(W₁/p) and Ω(T_∞) are both valid, independent lower bounds',
        'Ω(W₁ × T_∞ / p)'
      ],
      correct: 2,
      explanation: 'Both bounds are independent and valid. Ω(W₁/p): even with perfect load balancing, you cannot do more than p units of work per time step. Ω(T_∞): the span is the longest sequential chain; no amount of parallelism can shorten it. The work-stealing algorithm achieves O(W₁/p + T_∞) expected time, matching both lower bounds up to constants.'
    },
    {
      domain: 'cs',
      question: 'What is the exact expected number of element comparisons to sort n distinct elements using randomized quicksort (choosing the pivot uniformly at random)?',
      options: [
        'n(n−1)/2',
        '(3/2) n log₂ n',
        '2n ln n ≈ 1.386 n log₂ n',
        'n log₂ n'
      ],
      correct: 2,
      explanation: 'By linearity of expectation: E[comparisons] = Σ_{i<j} P(elements i and j are compared). Two elements i < j are compared iff one of them is chosen as pivot before any element strictly between them — probability 2/(j−i+1). Summing over all pairs gives 2(n+1)H_n − 4(n−1) ≈ 2n ln n, where H_n is the nth harmonic number. This equals approximately 1.386 n log₂ n.'
    }
  ];

  var DOMAINS = [
    { key: 'all',  label: 'All Domains' },
    { key: 'bio',  label: 'Biology & Medicine' },
    { key: 'math', label: 'Mathematics' },
    { key: 'phys', label: 'Physics & Chemistry' },
    { key: 'cs',   label: 'Computer Science' }
  ];

  var selectedDomain = null;
  var filteredQuestions = [];
  var currentQ = 0;
  var results = [];
  var answered = false;

  function getQuestionsForDomain(domainKey) {
    if (domainKey === 'all') return QUESTIONS.slice();
    return QUESTIONS.filter(function (q) { return q.domain === domainKey; });
  }

  function renderDomainSelector() {
    var container = document.getElementById('bhle-content');
    if (!container) return;
    var html = '<div style="margin-bottom:16px;">';
    html += '<p style="font-size:0.9rem;color:var(--font-color,#555);margin:0 0 14px 0;">Choose a domain to answer HLE-style questions from. Your score will be compared against frontier model performance on HLE.</p>';
    html += '<div style="display:flex;flex-wrap:wrap;gap:10px;">';
    DOMAINS.forEach(function (d) {
      html += '<button class="bhle-domain-btn" data-domain="' + d.key + '" style="';
      html += 'padding:10px 18px;border:1.5px solid #d1d5db;border-radius:8px;background:transparent;';
      html += 'font-size:0.88rem;font-weight:500;cursor:pointer;transition:all 0.15s;';
      html += 'color:var(--font-color,#333);">';
      html += d.label;
      html += ' <span style="font-size:0.75rem;color:#888;">(' + (d.key === 'all' ? QUESTIONS.length : getQuestionsForDomain(d.key).length) + 'Q)</span>';
      html += '</button>';
    });
    html += '</div></div>';
    container.innerHTML = html;

    container.querySelectorAll('.bhle-domain-btn').forEach(function (btn) {
      btn.addEventListener('mouseenter', function () {
        btn.style.borderColor = '#20B2AA';
        btn.style.background = 'rgba(32,178,170,0.07)';
      });
      btn.addEventListener('mouseleave', function () {
        btn.style.borderColor = '#d1d5db';
        btn.style.background = 'transparent';
      });
      btn.addEventListener('click', function () {
        selectedDomain = btn.getAttribute('data-domain');
        filteredQuestions = getQuestionsForDomain(selectedDomain);
        currentQ = 0;
        results = [];
        renderQuestion();
      });
    });
  }

  function renderQuestion() {
    var container = document.getElementById('bhle-content');
    if (!container) return;
    var q = filteredQuestions[currentQ];
    var domainLabel = DOMAINS.filter(function(d){ return d.key === selectedDomain; })[0].label;
    var html = '';
    html += '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">';
    html += '<div class="bhle-question-counter">Question ' + (currentQ + 1) + ' of ' + filteredQuestions.length + ' &middot; <span style="color:#20B2AA;">' + domainLabel + '</span></div>';
    html += '<button class="bhle-change-domain" style="padding:4px 12px;border:1px solid #d1d5db;border-radius:6px;background:transparent;font-size:0.78rem;color:#888;cursor:pointer;">Change domain</button>';
    html += '</div>';
    html += '<div class="bhle-question">';
    html += '<div class="bhle-question-label">HLE-Style Question</div>';
    html += '<div class="bhle-question-text">' + q.question + '</div>';
    html += '<div class="bhle-options">';
    q.options.forEach(function (opt, i) {
      html += '<label class="bhle-option" data-idx="' + i + '">';
      html += '<input type="radio" name="bhle-answer" value="' + i + '">';
      html += opt;
      html += '</label>';
    });
    html += '</div>';
    html += '<button class="bhle-submit" id="bhle-submit" disabled>Submit Answer</button>';
    html += '</div>';
    html += '<div class="bhle-results" id="bhle-results"></div>';
    container.innerHTML = html;

    answered = false;

    container.querySelector('.bhle-change-domain').addEventListener('click', function () {
      selectedDomain = null;
      filteredQuestions = [];
      currentQ = 0;
      results = [];
      renderDomainSelector();
    });

    var options = container.querySelectorAll('.bhle-option');
    options.forEach(function (opt) {
      opt.addEventListener('click', function () {
        if (answered) return;
        options.forEach(function (o) { o.classList.remove('selected'); });
        opt.classList.add('selected');
        opt.querySelector('input[type="radio"]').checked = true;
        document.getElementById('bhle-submit').disabled = false;
      });
      opt.querySelector('input[type="radio"]').addEventListener('change', function () {
        if (answered) return;
        options.forEach(function (o) { o.classList.remove('selected'); });
        opt.classList.add('selected');
        document.getElementById('bhle-submit').disabled = false;
      });
    });

    document.getElementById('bhle-submit').addEventListener('click', submitAnswer);
  }

  function submitAnswer() {
    if (answered) return;
    var selected = document.querySelector('#bhle-content .bhle-option.selected');
    if (!selected) return;
    var idx = parseInt(selected.getAttribute('data-idx'));
    var q = filteredQuestions[currentQ];
    var correct = idx === q.correct;
    answered = true;

    var options = document.querySelectorAll('#bhle-content .bhle-option');
    options.forEach(function (o, i) {
      o.style.pointerEvents = 'none';
      if (i === q.correct) o.classList.add('correct');
      if (i === idx && !correct) o.classList.add('wrong');
    });

    document.getElementById('bhle-submit').disabled = true;
    results.push(correct);

    var resultsDiv = document.getElementById('bhle-results');
    var html = '';
    html += '<div class="bhle-result-card">';
    html += '<div class="bhle-result-header">' + (correct ? '✅ Correct!' : '❌ Incorrect') + '</div>';
    html += '<p style="font-size:0.85rem;color:var(--font-color,#555);margin:0;">' + q.explanation + '</p>';
    html += '</div>';
    html += '<div class="bhle-result-card">';
    html += '<div class="bhle-result-header">Your Score vs. Frontier Models (HLE)</div>';

    var yourPct = (results.filter(function (r) { return r; }).length / filteredQuestions.length) * 100;

    var models = [
      { name: 'You',           score: yourPct, color: '#f59e0b' },
      { name: 'Gemini 3 Pro',  score: 37.5,    color: '#2563eb' },
      { name: 'Claude 4.6',    score: 34.4,    color: '#d97706' },
      { name: 'GPT-5 Pro',     score: 31.6,    color: '#059669' },
      { name: 'DeepSeek-V4',   score: 28,      color: '#7c3aed' },
      { name: 'Human Expert',  score: 90,      color: '#6b7280' }
    ];

    var sorted = models.slice().sort(function (a, b) {
      if (a.name === 'You') return -1;
      if (b.name === 'You') return 1;
      return b.score - a.score;
    });

    sorted.forEach(function (m) {
      var pct = Math.min(m.score / 100 * 100, 100);
      var youClass = m.name === 'You' ? ' bhle-you-bar' : '';
      html += '<div class="bhle-model-bar' + youClass + '">';
      html += '<span class="bhle-model-name">' + m.name + '</span>';
      html += '<div class="bhle-bar-track"><div class="bhle-bar-fill" style="width:' + pct + '%;background:' + m.color + '"></div></div>';
      html += '<span class="bhle-bar-label">' + (m.name === 'You' ? yourPct.toFixed(0) + '%' : m.score + '%') + '</span>';
      html += '</div>';
    });
    html += '</div>';

    if (currentQ < filteredQuestions.length - 1) {
      html += '<button class="bhle-next-btn" id="bhle-next">Next Question →</button>';
    } else {
      html += '<button class="bhle-next-btn" id="bhle-next">↻ Try Again</button>';
      html += ' <button class="bhle-next-btn" id="bhle-change-domain-end" style="margin-left:8px;">Change Domain</button>';
    }

    resultsDiv.innerHTML = html;
    resultsDiv.classList.add('show');

    document.getElementById('bhle-next').addEventListener('click', function () {
      if (currentQ < filteredQuestions.length - 1) {
        currentQ++;
        renderQuestion();
      } else {
        currentQ = 0;
        results = [];
        renderQuestion();
      }
    });

    var endDomainBtn = document.getElementById('bhle-change-domain-end');
    if (endDomainBtn) {
      endDomainBtn.addEventListener('click', function () {
        selectedDomain = null;
        filteredQuestions = [];
        currentQ = 0;
        results = [];
        renderDomainSelector();
      });
    }
  }

  function init() {
    var el = document.getElementById('bhle-widget');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      renderDomainSelector();
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
