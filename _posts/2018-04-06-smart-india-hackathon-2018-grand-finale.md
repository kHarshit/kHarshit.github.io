---
layout: post
title: "Smart India Hackathon 2018 grand finale"
date: 2018-04-06
categories: [Personal]
---

Recently, I attended Smart India Hackathon grand finale 2018<sup id="a1">[1](#myfootnote1)</sup> at Pune, India. We're one of the few teams selected for the finale. Our idea, which was under CSIR<sup id="a2">[2](#myfootnote2)</sup>, was to develop an android application which could give real-time location-based dengue-risk-index using machine learning techniques.

The application followed a client-server architecture consisting of two key components:
* An android app,
* A <a href="https://dengueapp.herokuapp.com">web-server</a>


The hackathon was divided into three phases having evaluation rounds at the end of each phase. During the first phase, we developed the prototype android app which gets user's location using GPS and gives sample dengue risk index to the user. We used TPOT<sup id="a3">[3](#myfootnote3)</sup> to find best machine learing algorithm for the purpose. It suggested gradient boosting. Our machine learing model used weather factors as the predictors.

In the second phase, I built the web-server in django<sup id="a4">[4](#myfootnote4)</sup> to host the machine learing model as an API. Also, we improved the UI of the app. During the evaluation, the evaluators suggested us to show dengue risk on a map and improve visualization part of the app.

In the final phase, we used Google Maps JavaScript API to show dengue risk index on a map to the user. We also improved the overall user experience of the app. We also added a feature so that a user can submit known dengue cases in his/her area.

Although we couldn't win, we learnt a lot from the hackathon.


**Footnotes:**  
<a name="myfootnote1"></a>1: [Smart India Hackathon grand finale 2018](https://innovate.mygov.in/sih2018/) [↩](#a1)  
<a name="myfootnote2"></a>2: [Council of Scientific & Industrial Research (CSIR)](http://www.csir.res.in/) [↩](#a2)  
<a name="myfootnote3"></a>3: [TPOT - Data Science Assistant](http://epistasislab.github.io/tpot/) [↩](#a3)  
<a name="myfootnote4"></a>4: [django - web server](https://www.djangoproject.com/) [↩](#a4)  
