---
layout: post
title: "Correlation is not causation"
date: 2017-10-20
categories: [Technical Fridays]
---

We often calculate correlation during EDA (Exploratory data analysis) to check how strongly two variables are correlated to one another. It's tempting to assume that one variable causes the other. But **correlation proves causation** is a <abbr title="use of invalid or otherwise faulty reasoning  in the construction of an argument">logical fallacy</abbr> known as `Cum Hoc, Ergo Propter Hoc`<sup id="a1">[1](#myfootnote1)</sup>. 

For example,

> As ice cream sales increase, the rate of drowning deaths increases. Therefore, ice cream causes drowning.

In the above example, the month when the sale of ice cream is high plays a significant role. The ice cream is sold at higher rate in the hot summer than during the winter. The people are more likely to go for a swim in the summer thus are more prone to drowning. Hence, the above statement is false.

Have a look at another example<sup id="a2">[2](#myfootnote2)</sup>.

<img src="/img/correlation.png" style="display: block; margin: auto; width: auto; max-width: 100%;">  

The correlation between divorce rate in Maine and per capita consumption of margarine tempts us to believe that consumption of margarine causes divorce, which is incorrect.

On the other hand, Tufte states that saying *Correlation is not causation* is incomplete<sup id="a3">[3](#myfootnote3)</sup>. According to him, the shortest true statement that can be made about causality and correlation is:

> Correlation is not causation but it sure is a hint.
> &mdash; <cite>Tufte</cite>


**Resources:**  
<a name="myfootnote1"></a>1: [Cum Hoc, Ergo Propter Hoc](http://www.fallacyfiles.org/cumhocfa.html): *With this, therefore because of this*, Latin [↩](#a1)  
<a name="myfootnote2"></a>2: [Spurious correlation](http://www.tylervigen.com/spurious-correlations) [↩](#a2)  
<a name="myfootnote3"></a>3: [Correlation does not imply causation - Wikipedia](https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation#cite_ref-Tufte_2006_5_1-1) [↩](#a3)  
