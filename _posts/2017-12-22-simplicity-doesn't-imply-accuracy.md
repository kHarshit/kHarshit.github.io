---
layout: post
title: "Simplicity doesn't imply accuracy"
date: 2017-12-22
categories: [Technical Fridays, Data Science]
---

Often, people say things like *beauty lies in simplicity*, *simplicity is the glory of expression, complexity is the enemy of execution*. But to what extent, these statements are true?

> It can scarcely be denied that the supreme goal of all theory is to make the irreducible basic elements as simple and as few as possible without having to surrender the adequate representation of a single datum of experience.
> > &mdash; <cite>Albert Einstein</cite>

**Simplicity** is the quality of being easy to understand or use while **accuracy** is the quality or state of being accurate.

**Occam's razor** (also *Latin: lex parsimoniae* **law of parsimony**) is a problem solving principle which states that among competing hypotheses, the one with the fewest assumptions should be selected i.e. when you have two competing theories that make exactly the same predictions, the simpler one is the better. A common paraphrase is *All things being equal, the simplest solution tends to be the best one.*

Take for example, the probability of correct answer (accuracy) in a multiple choice question decreases with the number of choices given.

In the context of [overfitting]({{ site.url }}{% post_url 2017-10-13-overfitting-and-underfitting %}), excessive complex models may tend to overfit the data (as affected by statistical noise), whereas simpler models may capture the underlying structure better and thus may have better predictive performance.

In *machine learning*, Occam's razor is often taken to mean that given two classifiers with the same training error, the simpler of the two will likely have the lowest test error; but in fact there are many counter-
examples to it.

There exists a trade-off between simplicity and accuracy. Consider the example of *linear regression*. Given a set of *predictors*, one attempts to provide a good fit to the *response*. A common measure of accuracy is the coefficient of determination, $$R^2$$. Increasing the number of predictors will generally incrase the $$R^2$$ (accuracy), but our model will become complex. We can decrease the complexity by decreasing the number of predictors, but now the model may not yield a sufficient level of accuracy.

Also, note that *Simple isn’t always easy*. The statement below demonstrates an assumption that because the desired action is conceptually simple, it must therefore be simple to implement.
<img src="https://imgs.xkcd.com/comics/shouldnt_be_hard.png" style="float: center; display: block; margin: auto; width: auto; max-width: 100%;">
<div style="text-align: center">
    <figcaption>xkcd: <a href="https://xkcd.com/1349/">Shouldn't Be Hard</a></figcaption>
</div>

A computer is a very complicated set of components which effectively can't do anything (simple or complex) until someone has programmed the functionality into it. Even more abstractly, a random silicon crystal can't do anything at all until someone has applied a complex industrial process to it that allows it to read and execute computer code in the first place.

Thus, in the field of *data science*, one must not apply Occam's principle blindly. 
> Until proved otherwise, the more complex theory cometing with a simpler explanation should be put on the back burner, but not thrown onto the trash heap of history until proven false.

**References:**
1. Pedro Domingos, A Few Useful Things to Know about Machine Learning
2. Enriqueta Aragones et al. Accuracy vs. Simplicity: A Complex Trade-Off
3. <a href="https://en.wikipedia.org/wiki/Occam's_razor">Occam's razor - Wikipedia</a>  


