var Question = function(question_string, correct_choice, wrong_choices) {
    this.question_string = question_string;
    this.user_choice_index = null;
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
    // Shuffle question order each build
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
    var $submitBtn    = $container.find('#submit-btn');
    var $retakeBtn    = $container.find('#retake-btn');
    var $questionArea = $container.find('.quiz-question-area');
    var $footer       = $container.find('.quiz-footer');
    var $results      = $container.find('.quiz-results');
    var $progressText = $container.find('.quiz-progress-text');
    var $progressFill = $container.find('.quiz-progress-fill');

    // Reset view state
    $results.hide();
    $questionArea.show();
    $footer.show();

    function updateProgress() {
        var total = self.questions.length;
        var current = self.current_index + 1;
        $progressText.text('Question ' + current + ' of ' + total);
        $progressFill.css('width', (current / total * 100) + '%');
    }

    function allAnswered() {
        return self.questions.every(function(q) { return q.user_choice_index !== null; });
    }

    function updateFooter() {
        var isLast = self.current_index === self.questions.length - 1;
        $prevBtn.prop('disabled', self.current_index === 0);
        if (isLast) {
            $nextBtn.hide();
            $submitBtn.show().prop('disabled', !allAnswered());
        } else {
            $nextBtn.show();
            $submitBtn.hide();
        }
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
                    $label.addClass('selected');
                    $radio.prop('checked', true);
                }
                $label.append($radio).append($('<span>').text(q.choices[idx]));
                $options.append($label);
            })(i);
        }

        $options.find('.quiz-option').on('click', function() {
            var idx = parseInt($(this).attr('data-index'));
            self.questions[self.current_index].user_choice_index = idx;
            $options.find('.quiz-option').removeClass('selected');
            $(this).addClass('selected');
            updateFooter();
        });

        updateFooter();
        updateProgress();
    }

    // Navigation
    $prevBtn.off('click').on('click', function() {
        if (self.current_index > 0) { self.current_index--; renderQuestion(); }
    });
    $nextBtn.off('click').on('click', function() {
        if (self.current_index < self.questions.length - 1) { self.current_index++; renderQuestion(); }
    });

    // Submit
    $submitBtn.off('click').on('click', function() {
        var score = 0;
        self.questions.forEach(function(q) {
            if (q.user_choice_index === q.correct_choice_index) score++;
        });

        var pct = score / self.questions.length;
        var emoji, message;
        if (pct === 1)       { emoji = '🎉'; message = 'Perfect score!'; }
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

    // Retake
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
