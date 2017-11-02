---
layout: post
title: "Moneyball: Why no prediction can't be made for baseball champion"
date: 2017-08-04
categories: [Technical Fridays, Data Science, R]
---

<img src="/img/moneyball.jpeg" style="float: center; display: block; margin: auto; width: auto; max-width: 100%;">

> How can you not be romantic about baseball?     
> &mdash; <cite>Billy Beans</cite>

Last week, we discussed how Billy Beans and Paul DePosta predicted, using linear regression, about the conditions necessary for Oakland A's to make it to the 2002 playoffs. Although A's made it to the playoffs yet they didn't succeed to win the World Series. Billy Beans justified this by making a claim that sabermatrics can't be used to predict baseball champions. Today, we'll try to analyze why no prediction can't be made for winning the baseball World Series. 

We'll try to make prediction using *logistic regression* in R with the same dataset <em title="click to download">[baseball.csv](/assets/baseball.csv)</em> used last week.

{% highlight r %}
# load in the dataset
> baseball = read.csv("baseball.csv")
> table(baseball$Year)  # 1972, 1981, 1994, and 1995 are missing

1962 1963 1964 1965 1966 1967 1968 1969 1970 1971 1973 1974 1975 1976 1977 1978 1979 
  20   20   20   20   20   20   20   24   24   24   24   24   24   24   26   26   26 
1980 1982 1983 1984 1985 1986 1987 1988 1989 1990 1991 1992 1993 1996 1997 1998 1999 
  26   26   26   26   26   26   26   26   26   26   26   26   28   28   28   30   30 
2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 
  30   30   30   30   30   30   30   30   30   30   30   30   30 
#  we're only analyzing teams that made the playoffs
> baseball = subset(baseball, Playoffs == 1)
# no of teams making to the playoffs in particular year
> table(baseball$Year)

1962 1963 1964 1965 1966 1967 1968 1969 1970 1971 1973 1974 1975 1976 1977 1978 1979 
   2    2    2    2    2    2    2    4    4    4    4    4    4    4    4    4    4 
1980 1982 1983 1984 1985 1986 1987 1988 1989 1990 1991 1992 1993 1996 1997 1998 1999 
   4    4    4    4    4    4    4    4    4    4    4    4    4    8    8    8    8 
2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 
   8    8    8    8    8    8    8    8    8    8    8    8   10 
> table(table(baseball$Year))

 2  4  8 10 
 7 23 16  1 

{% endhighlight %}

Note that it's much harder to win the World Series if there are 10 teams competing for the championship versus just two. Therefore, we will add the predictor variable `NumCompetitors` to the baseball data frame. It will contain the number of total teams making the playoffs in the year of a particular team/year pair.

{% highlight r %}
> playoffTable = table(baseball$Year)
> str(names(playoffTable))
 chr [1:47] "1962" "1963" "1964" "1965" "1966" "1967" "1968" "1969" ...
> playoffTable[c("1990", "2001")]

1990 2001 
   4    8 
> baseball$NumCompetitors = playoffTable[as.character(baseball$Year)]
> table(baseball$NumCompetitors)

  2   4   8  10 
 14  92 128  10 
# Add a variable named WorldSeries to the baseball data frame (takes value 1 if a team won the World Series)
> baseball$WorldSeries = as.numeric(baseball$RankPlayoffs == 1)
> table(baseball$WorldSeries)

  0   1 
197  47 
{% endhighlight %}

When we're not sure which of our variables are useful in predicting a particular outcome, it's often helpful to build bivariate models, which are models that predict the outcome using a single independent variable.
<p id="bigcode">
{% highlight r %}
# logistic regression models with a single independent variable
> summary(glm(WorldSeries ~ Year, data = baseball, family = "binomial"))

Call:
glm(formula = WorldSeries ~ Year, family = "binomial", data = baseball)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.0297  -0.6797  -0.5435  -0.4648   2.1504  

Coefficients:
            Estimate Std. Error z value Pr(>|z|)   
(Intercept) 72.23602   22.64409    3.19  0.00142 **
Year        -0.03700    0.01138   -3.25  0.00115 **
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 239.12  on 243  degrees of freedom
Residual deviance: 228.35  on 242  degrees of freedom
AIC: 232.35

Number of Fisher Scoring iterations: 4

> summary(glm(WorldSeries ~ RS, data = baseball, family = "binomial"))

Call:
glm(formula = WorldSeries ~ RS, family = "binomial", data = baseball)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-0.8254  -0.6819  -0.6363  -0.5561   2.0308  

Coefficients:
             Estimate Std. Error z value Pr(>|z|)
(Intercept)  0.661226   1.636494   0.404    0.686
RS          -0.002681   0.002098  -1.278    0.201

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 239.12  on 243  degrees of freedom
Residual deviance: 237.45  on 242  degrees of freedom
AIC: 241.45

Number of Fisher Scoring iterations: 4

> summary(glm(WorldSeries ~ RA, data = baseball, family = "binomial"))

Call:
glm(formula = WorldSeries ~ RA, family = "binomial", data = baseball)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-0.9749  -0.6883  -0.6118  -0.4746   2.1577  

Coefficients:
             Estimate Std. Error z value Pr(>|z|)  
(Intercept)  1.888174   1.483831   1.272   0.2032  
RA          -0.005053   0.002273  -2.223   0.0262 *
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 239.12  on 243  degrees of freedom
Residual deviance: 233.88  on 242  degrees of freedom
AIC: 237.88

Number of Fisher Scoring iterations: 4

> summary(glm(WorldSeries ~ NumCompetitors, data = baseball, family = "binomial"))

Call:
glm(formula = WorldSeries ~ NumCompetitors, family = "binomial", 
    data = baseball)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-0.9871  -0.8017  -0.5089  -0.5089   2.2643  

Coefficients:
               Estimate Std. Error z value Pr(>|z|)    
(Intercept)     0.03868    0.43750   0.088 0.929559    
NumCompetitors -0.25220    0.07422  -3.398 0.000678 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 239.12  on 243  degrees of freedom
Residual deviance: 226.96  on 242  degrees of freedom
AIC: 230.96

Number of Fisher Scoring iterations: 4

> LogModel = glm(WorldSeries ~ Year + RA + RankSeason + NumCompetitors, data = baseball, family = "binomial")
> summary(LogModel)

Call:
glm(formula = WorldSeries ~ Year + RA + RankSeason + NumCompetitors, 
    family = "binomial", data = baseball)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.0336  -0.7689  -0.5139  -0.4583   2.2195  

Coefficients:
                 Estimate Std. Error z value Pr(>|z|)
(Intercept)    12.5874376 53.6474210   0.235    0.814
Year           -0.0061425  0.0274665  -0.224    0.823
RA             -0.0008238  0.0027391  -0.301    0.764
RankSeason     -0.0685046  0.1203459  -0.569    0.569
NumCompetitors -0.1794264  0.1815933  -0.988    0.323

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 239.12  on 243  degrees of freedom
Residual deviance: 226.37  on 239  degrees of freedom
AIC: 236.37

Number of Fisher Scoring iterations: 4

# and so on ... there will be 12 models!!
{% endhighlight %}
</p>
After analyzing the above models, we found that `Year`, `RA`, `RankSeason` and `NumCompetitors` are <abbr title="a variable is significant if it contains atleast one star (*)">significant</abbr>.  
Let's build a regression model using above four variables.

{% highlight r %}
# a logistic regression model
> LogModel = glm(WorldSeries ~ Year + RA + RankSeason + NumCompetitors, data = baseball, family = "binomial")
> summary(LogModel)

Call:
glm(formula = WorldSeries ~ Year + RA + RankSeason + NumCompetitors, 
    family = "binomial", data = baseball)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.0336  -0.7689  -0.5139  -0.4583   2.2195  

Coefficients:
                 Estimate Std. Error z value Pr(>|z|)
(Intercept)    12.5874376 53.6474210   0.235    0.814
Year           -0.0061425  0.0274665  -0.224    0.823
RA             -0.0008238  0.0027391  -0.301    0.764
RankSeason     -0.0685046  0.1203459  -0.569    0.569
NumCompetitors -0.1794264  0.1815933  -0.988    0.323

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 239.12  on 243  degrees of freedom
Residual deviance: 226.37  on 239  degrees of freedom
AIC: 236.37

Number of Fisher Scoring iterations: 4

{% endhighlight %}

Oops, none of the variables are significant in the multivariate model!  
It maybe due to correlation between the variables.

{% highlight r %}
> cor(baseball[c("Year", "RA", "RankSeason", "NumCompetitors")])
                    Year        RA RankSeason NumCompetitors
Year           1.0000000 0.4762422  0.3852191      0.9139548
RA             0.4762422 1.0000000  0.3991413      0.5136769
RankSeason     0.3852191 0.3991413  1.0000000      0.4247393
NumCompetitors 0.9139548 0.5136769  0.4247393      1.0000000
{% endhighlight %}

The above correlation matrix indicates that `Year/NumCompetitors` has high degree of correlation (0.91).  

Now, let's try to build two-variable model.
<p id="bigcode">
{% highlight r %}
> summary(glm(WorldSeries ~ Year + RA, data=baseball, family=binomial))

Call:
glm(formula = WorldSeries ~ Year + RA, family = binomial, data = baseball)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.0402  -0.6878  -0.5298  -0.4785   2.1370  

Coefficients:
             Estimate Std. Error z value Pr(>|z|)  
(Intercept) 63.610741  25.654830   2.479   0.0132 *
Year        -0.032084   0.013323  -2.408   0.0160 *
RA          -0.001766   0.002585  -0.683   0.4945  
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 239.12  on 243  degrees of freedom
Residual deviance: 227.88  on 241  degrees of freedom
AIC: 233.88

Number of Fisher Scoring iterations: 4

> summary(glm(WorldSeries ~ Year + RankSeason, data=baseball, family=binomial))

Call:
glm(formula = WorldSeries ~ Year + RankSeason, family = binomial, 
    data = baseball)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.0560  -0.6957  -0.5379  -0.4528   2.2673  

Coefficients:
            Estimate Std. Error z value Pr(>|z|)   
(Intercept) 63.64855   24.37063   2.612  0.00901 **
Year        -0.03254    0.01231  -2.643  0.00822 **
RankSeason  -0.10064    0.11352  -0.887  0.37534   
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 239.12  on 243  degrees of freedom
Residual deviance: 227.55  on 241  degrees of freedom
AIC: 233.55

Number of Fisher Scoring iterations: 4

> summary(glm(WorldSeries ~ Year + NumCompetitors, data=baseball, family=binomial))

Call:
glm(formula = WorldSeries ~ Year + NumCompetitors, family = binomial, 
    data = baseball)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.0050  -0.7823  -0.5115  -0.4970   2.2552  

Coefficients:
                Estimate Std. Error z value Pr(>|z|)
(Intercept)    13.350467  53.481896   0.250    0.803
Year           -0.006802   0.027328  -0.249    0.803
NumCompetitors -0.212610   0.175520  -1.211    0.226

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 239.12  on 243  degrees of freedom
Residual deviance: 226.90  on 241  degrees of freedom
AIC: 232.9

Number of Fisher Scoring iterations: 4

> summary(glm(WorldSeries ~ RA + RankSeason, data=baseball, family=binomial))

Call:
glm(formula = WorldSeries ~ RA + RankSeason, family = binomial, 
    data = baseball)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-0.9374  -0.6933  -0.5936  -0.4564   2.1979  

Coefficients:
             Estimate Std. Error z value Pr(>|z|)
(Intercept)  1.487461   1.506143   0.988    0.323
RA          -0.003815   0.002441  -1.563    0.118
RankSeason  -0.140824   0.110908  -1.270    0.204

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 239.12  on 243  degrees of freedom
Residual deviance: 232.22  on 241  degrees of freedom
AIC: 238.22

Number of Fisher Scoring iterations: 4

> summary(glm(WorldSeries ~ RA + NumCompetitors, data=baseball, family=binomial))

Call:
glm(formula = WorldSeries ~ RA + NumCompetitors, family = binomial, 
    data = baseball)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.0433  -0.7826  -0.5133  -0.4701   2.2208  

Coefficients:
                Estimate Std. Error z value Pr(>|z|)   
(Intercept)     0.716895   1.528736   0.469  0.63911   
RA             -0.001233   0.002661  -0.463  0.64313   
NumCompetitors -0.229385   0.088399  -2.595  0.00946 **
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 239.12  on 243  degrees of freedom
Residual deviance: 226.74  on 241  degrees of freedom
AIC: 232.74

Number of Fisher Scoring iterations: 4

> summary(glm(WorldSeries ~ RankSeason + NumCompetitors, data=baseball, family=binomial))

Call:
glm(formula = WorldSeries ~ RankSeason + NumCompetitors, family = binomial, 
    data = baseball)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.0090  -0.7592  -0.5204  -0.4501   2.2562  

Coefficients:
               Estimate Std. Error z value Pr(>|z|)   
(Intercept)     0.12277    0.45737   0.268  0.78837   
RankSeason     -0.07697    0.11711  -0.657  0.51102   
NumCompetitors -0.22784    0.08201  -2.778  0.00546 **
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 239.12  on 243  degrees of freedom
Residual deviance: 226.52  on 241  degrees of freedom
AIC: 232.52

Number of Fisher Scoring iterations: 4
{% endhighlight %}
</p>

None of the models with two independent variables has both variables significant, so none seem promising as compared to a simple bivariate model. Indeed the model with the lowest <abbr title="Akaike information criterion">AIC</abbr> value is the model with just `NumCompetitors` as the independent variable.

This seems to confirm the claim made by Billy Beane in Moneyball that all that matters in the Playoffs is luck, since `NumCompetitors` has nothing to do with the quality of the teams!  
Hence, no prediction can be made for baseball champion.
