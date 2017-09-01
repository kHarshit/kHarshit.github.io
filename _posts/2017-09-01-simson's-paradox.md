---
layout: post
title: "Simpson's paradox"
date: 2017-09-01
categories: [Technical Fridays, R]
---

In 1973, the University of California, Berkeley was sued for gender bias against women who had applied to graduate schools. The data for fall 1973 showed that men applying were far more likely to get admits than the women.

But, after examining the individual departments, it appeared that no department was significantly biased against women.

It was a case of Simpson's paradox<sup id="a1">[1](#myfootnote1)</sup>, a phenomenon in which a trend appears in different groups of data but disappears or reverses when these groups are combined.

<div style="text-align: center">
<iframe src="https://giphy.com/embed/xT5LMv7ScakR9zwvSw" width="480" height="368" frameBorder="0" class="giphy-embed" allowFullScreen></iframe>
</div>

Let's analyze the UC Berkeley graduate admissions data of 1973.
R comes preloaded with dataset<sup id="a2">[2](#myfootnote2)</sup> UCBAdmissions.

{% highlight r %}
# load in the data
> ucb = UCBAdmissions
# structure of the data
> str(ucb)
 table [1:2, 1:2, 1:6] 512 313 89 19 353 207 17 8 120 205 ...
 - attr(*, "dimnames")=List of 3
  ..$ Admit : chr [1:2] "Admitted" "Rejected"
  ..$ Gender: chr [1:2] "Male" "Female"
  ..$ Dept  : chr [1:6] "A" "B" "C" "D" ...
{% endhighlight %}

`ucb` is a three-dimensional table (like a matrix): `Admit` Status x `Gender` x `Dept`, with counts for each category as the matrix’s values. Here’s the data of six departments:

{% highlight r %}
> ucb
, , Dept = A

          Gender
Admit      Male Female
  Admitted  512     89
  Rejected  313     19

, , Dept = B

          Gender
Admit      Male Female
  Admitted  353     17
  Rejected  207      8

, , Dept = C

          Gender
Admit      Male Female
  Admitted  120    202
  Rejected  205    391

, , Dept = D

          Gender
Admit      Male Female
  Admitted  138    131
  Rejected  279    244

, , Dept = E

          Gender
Admit      Male Female
  Admitted   53     94
  Rejected  138    299

, , Dept = F

          Gender
Admit      Male Female
  Admitted   22     24
  Rejected  351    317

{% endhighlight %}

Let's check the acceptance rate of UC Berkeley in 1973 for graduate applicants.

{% highlight r %}
> apply(ucb, c(1, 2), sum)
          Gender
Admit      Male Female
  Admitted 1198    557
  Rejected 1493   1278
> 1198/(1198+1493)
[1] 0.4451877
> 557/(557+1278)
[1] 0.3035422
{% endhighlight %}

Overall, women have an admission rate of 30.35%, which is much lower than that of men, 44.52%.  
It prompted a lawsuit against UC Berkeley which prompted the study that collected this data.

Let's draw mosaic plot<sup id="a3">[3](#myfootnote3)</sup>, which provides a way to visualize contingency tables.

{% highlight r %}
> mosaicplot(apply(ucb, c(1, 2), sum), color=TRUE
+            main = "Student admissions at UC Berkeley")
{% endhighlight %}

<img src="/img/Rplot_ucb_mosaic_overall.png" style="display: block; margin: auto; width: auto; max-width: 100%;">  

It seems to indicate a gender bias.
However, there is a lurking variable: `Dept`. Here is what happens if we stratify on department: 

{% highlight r %}
> plot(ucb, color=TRUE, main='Student admissions at UC Berkeley')
{% endhighlight %}

<img src="/img/Rplot_ucb_mosaic_analysis.png" style="display: block; margin: auto; width: auto; max-width: 100%;">  

The `Admit` row in our table of contents corresponds to the *width* of columns in the mosaic plot. More people were rejected than admitted because Rejected column is wider. Of the people admitted, a much higher proportion were `Male` because of the height of the rectangles. Of the people rejected, it appears to be pretty even.

A higher proportion of admitted Males were for `Dept` A and B compared to the proportion of admitted Females for the same `Dept`. On the other hand, a higher proportion of admitted Females were for `Dept` C – F. But, higher proportion of the Males were rejected for `Dept` A and B than Females as the widths of the Male rectangles are wider than their Female counterparts. Likewise for `Dept` C – F, a higher proportion of the Females were rejected for `Dept` C – F than Males.

It appears that most departments have no gender bias, and those departments that are biased favor women. First, note that `Dept` A and B have very few female applicants (the columns are narrow). It is also relatively easy to get into those departments---the proportion rejected is lower than other departments, especially F. So one explanation is that more males get in because they are applying to the hungrier, perhaps fastest-growing, departments.

The research paper by Bickel et al.<sup id="a4">[4](#myfootnote4)</sup> concluded that women tended to apply to competitive departments with low rates of admission even among qualified applicants (such as in the English Department), whereas men tended to apply to less-competitive departments with high rates of admission among the qualified applicants (such as in engineering and chemistry).

If you express matters algebraically, the appearance of the Paradox is no surprise.  
For, suppose
<img src="/img/eqn6592.png" style="display: block; margin: auto; width: auto; max-width: 100%;">  
No one would dream of deducing that  
<img src="/img/eqn6594.png" style="display: block; margin: auto; width: auto; max-width: 100%;">  

would they?

**Footnotes:**  
<a name="myfootnote1"></a>1: [Simpson's paradox](https://en.wikipedia.org/wiki/Simpson's_paradox) [↩](#a1)  
<a name="myfootnote2"></a>2: [UCBAdmissions dataset](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/UCBAdmissions.html) [↩](#a2)  
<a name="myfootnote3"></a>3: [Mosaic plot](https://en.wikipedia.org/wiki/Mosaic_plot) [↩](#a3)  
<a name="myfootnote4"></a>4: [Research paper: Berkeley gender bias](http://homepage.stat.uiowa.edu/~mbognar/1030/Bickel-Berkeley.pdf) [↩](#a4)  


