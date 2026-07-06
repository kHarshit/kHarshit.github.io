var Question = function(question_string, correct_choice, wrong_choices) {
    this.question_string = question_string;
    this.user_choice_index = null;
    this.submitted = false;
    this.choices = [];

    // Place the correct answer at a random index
    this.correct_choice_index = Math.floor(Math.random() * (wrong_choices.length + 1));

    var wrongs = wrong_choices.slice();
    for (var i = 0; i <= wrong_choices.length; i++) {
        if (i === this.correct_choice_index) {
            this.choices.push(correct_choice);
        } else {
            var wi = Math.floor(Math.random() * wrongs.length);
            this.choices.push(wrongs.splice(wi, 1)[0]);
        }
    }
};

var Quiz = function(questions_data) {
    this.questions_data = questions_data;
    this.questions = [];
    this.current_index = 0;
};

Quiz.prototype.build = function() {
    this.questions = [];
    this.current_index = 0;
    var shuffled = this.questions_data.slice().sort(function() { return Math.random() - 0.5; });
    for (var i = 0; i < shuffled.length; i++) {
        this.questions.push(new Question(
            shuffled[i].question_string,
            shuffled[i].choices.correct,
            shuffled[i].choices.wrong
        ));
    }
};

Quiz.prototype.render = function($container) {
    var self = this;
    this.build();

    var $questionText = $container.find('.quiz-question-text');
    var $options      = $container.find('.quiz-options');
    var $prevBtn      = $container.find('#prev-btn');
    var $nextBtn      = $container.find('#next-btn');
    var $checkBtn     = $container.find('#check-btn');
    var $finishBtn    = $container.find('#finish-btn');
    var $retakeBtn    = $container.find('#retake-btn');
    var $questionArea = $container.find('.quiz-question-area');
    var $footer       = $container.find('.quiz-footer');
    var $results      = $container.find('.quiz-results');
    var $progressText = $container.find('.quiz-progress-text');
    var $progressFill = $container.find('.quiz-progress-fill');

    $results.hide();
    $questionArea.show();
    $footer.show();

    function updateProgress() {
        var total = self.questions.length;
        var current = self.current_index + 1;
        $progressText.text('Question ' + current + ' of ' + total);
        $progressFill.css('width', (current / total * 100) + '%');
    }

    function allSubmitted() {
        return self.questions.every(function(q) { return q.submitted; });
    }

    function updateFooter() {
        var q = self.questions[self.current_index];
        var isLast = self.current_index === self.questions.length - 1;

        $prevBtn.prop('disabled', self.current_index === 0);

        // Check button: show when answer selected and not yet submitted
        if (!q.submitted && q.user_choice_index !== null) {
            $checkBtn.show();
        } else {
            $checkBtn.hide();
        }

        // Next / Finish
        if (isLast) {
            $nextBtn.hide();
            $finishBtn.show().prop('disabled', !allSubmitted());
        } else {
            $nextBtn.show().prop('disabled', !q.submitted);
            $finishBtn.hide();
        }
    }

    function applySubmittedColors($opts, q) {
        $opts.find('.quiz-option').each(function() {
            var idx = parseInt($(this).attr('data-index'));
            $(this).addClass('locked');
            if (idx === q.correct_choice_index) {
                $(this).removeClass('selected incorrect').addClass('correct');
                $(this).find('input').prop('checked', idx === q.user_choice_index);
            } else if (idx === q.user_choice_index) {
                $(this).removeClass('selected correct').addClass('incorrect');
                $(this).find('input').prop('checked', true);
            } else {
                $(this).removeClass('selected');
            }
        });
    }

    function renderQuestion() {
        var q = self.questions[self.current_index];
        $questionText.text(q.question_string);
        $options.empty();

        for (var i = 0; i < q.choices.length; i++) {
            (function(idx) {
                var $label = $('<label class="quiz-option">').attr('data-index', idx);
                var $radio = $('<input type="radio" name="quiz-choice">').val(idx);
                if (q.user_choice_index === idx) {
                    $radio.prop('checked', true);
                    if (!q.submitted) $label.addClass('selected');
                }
                $label.append($radio).append($('<span>').text(q.choices[idx]));
                $options.append($label);
            })(i);
        }

        if (q.submitted) {
            applySubmittedColors($options, q);
        } else {
            $options.find('.quiz-option').on('click', function() {
                var idx = parseInt($(this).attr('data-index'));
                self.questions[self.current_index].user_choice_index = idx;
                $options.find('.quiz-option').removeClass('selected');
                $(this).addClass('selected');
                updateFooter();
            });
        }

        updateFooter();
        updateProgress();
    }

    $prevBtn.off('click').on('click', function() {
        if (self.current_index > 0) { self.current_index--; renderQuestion(); }
    });

    $nextBtn.off('click').on('click', function() {
        if (self.current_index < self.questions.length - 1) { self.current_index++; renderQuestion(); }
    });

    $checkBtn.off('click').on('click', function() {
        var q = self.questions[self.current_index];
        q.submitted = true;
        applySubmittedColors($options, q);
        // Remove click handlers by replacing options (already locked via CSS)
        $options.find('.quiz-option').off('click');
        updateFooter();
    });

    $finishBtn.off('click').on('click', function() {
        var score = 0;
        self.questions.forEach(function(q) {
            if (q.user_choice_index === q.correct_choice_index) score++;
        });

        var pct = score / self.questions.length;
        var emoji, message;
        if (pct === 1)        { emoji = '🎉'; message = 'Perfect score!'; }
        else if (pct >= 0.75) { emoji = '👍'; message = 'Great job!'; }
        else if (pct >= 0.5)  { emoji = '📚'; message = 'Almost there!'; }
        else                  { emoji = '💪'; message = 'Keep practicing!'; }

        $container.find('.quiz-results-emoji').text(emoji);
        $container.find('.quiz-results-message').text(message);
        $container.find('.quiz-results-score').html(
            'You got <b>' + score + ' / ' + self.questions.length + '</b> correct'
        );

        $questionArea.slideUp(200);
        $footer.slideUp(200);
        $results.slideDown(250);
    });

    $retakeBtn.off('click').on('click', function() {
        $results.slideUp(200, function() {
            self.render($container);
        });
    });

    renderQuestion();
};

$(document).ready(function() {
    var quiz = new Quiz(all_questions);
    quiz.render($('#quiz'));
});
