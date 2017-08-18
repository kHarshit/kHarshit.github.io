---
layout: post
title: "Friendship paradox: facebook"
date: 2017-08-18
categories: [Technical Fridays, Python]
---

<div style="text-align: center">
<iframe src="https://giphy.com/embed/3o6Zt6c6km1BV9WeTm" width="480" height="270" frameBorder="0" class="giphy-embed" allowFullScreen></iframe>
</div>

According to a 2012 study by <cite>Pew Research Center’s Internet and American Life Project<sup id="a1">[1](#myfootnote1)</sup></cite>:  
> On Facebook, the average person has 245 friends. However, the average friend of a person has 359 Facebook friends. The finding, that people’s friends have more friends than they do, was nearly universal.
Only those who had among the 10% largest friends lists (over 780 friends) had friends who on average had smaller networks than their own.

It’s just the digital reflection of what’s known as the *friendship paradox*<sup id="a2">[2](#myfootnote2)</sup>, the phenomenon first observed by the sociologist Scott L. Feld in 1991 that most people have fewer friends than their friends have, on average.

The **generalized friendship paradox** states that the friendship paradox applies to other characteristics as well. For example, one's co-authors are on average likely to be more prominent, with more publications, more citations and more collaborators, or one's followers on Twitter have more followers.

Let's check this for our facebook friends.

### Get the data
Navigate to friends list on your facebook profile. Scroll down enough so all your friends are on the page, *Select All* (`Ctrl-A`) and *Copy* (`Ctrl-C`). Then, *paste* (`Ctrl-V`) the content copied into any `regex` editor<sup id="a3">[3](#myfootnote3)</sup>.  
**Note:** The data can also be grabbed by web scrapping using Python's `beautifulsoup` and `requests` library.

### Clean the data
Now, we need to find all instances of *count of friends* from the content pasted. It can be done by regex expression `[,\d]+ friends`. Grab all instances of text like `345 friends` and save it as `txt` file, say, `facebook_friends.txt`.  

**Note:** We won't get count of all of our friends via this method because friends page on facebook lists some friends with *mutual friends* count if they have privacy set to *Only me*.

### Analyse the data
Now, open python console and do some analysis on the data.

{% highlight python %}
Python 3.6.2 |Continuum Analytics, Inc.| (default, Jul 20 2017, 13:51:32) 
[GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux

# Import necessary modules  
In[2]: import pandas as pd
  ...: import matplotlib.pyplot as plt

# read the data
In[3]: df = pd.read_table('facebook_friends.txt', sep=' ', thousands=',', header=None, names=['friend_count', 'text'])
# check the structure of data
In[4]: print(df.head())
   friend_count     text
0           637  friends
1           101  friends
2           350  friends
3          1191  friends
4           300  friends
# description of data
In[5]: print(df.describe())
       friend_count
count    116.000000
mean     594.146552
std      829.748765
min       14.000000
25%      218.500000
50%      380.000000
75%      647.000000
max     4972.000000
{% endhighlight %}

Now, let's plot the data.

{% highlight python %}
# average no of friends
In[6]: avg = df.mean()
# median no of friends
In[7]: median = df.median()
# count of my friends
In[8]: my_friends = 208

# plot a histogram
In[9]: df.hist(bins=20)
In[10]: plt.xlabel('Friend count')
In[11]: plt.ylabel('Number')
In[12]: plt.suptitle("Histogram of Friend Counts")
In[13]: for (x, c) in zip([avg, median, my_friends], ['k', 'b', 'r']):
   ...:    plt.vlines(x, 0, 120, colors=c)
In[14]: plt.show()
{% endhighlight %}

<figure>
<img src="/img/facebook_friends.png" style="display: block; margin: auto; width: auto; max-width: 100%;">
<figcaption>The vertical lines: – black, blue, red – represent the mean, the median, and my own personal friend count, respectively.</figcaption>
</figure>

It appears the paradox holds true for me as well!  
I have 208 facebook friends, but on average, my friends have 594 facebook friends.

**Footnotes:**  
<a name="myfootnote1"></a>1: [2012 Pew Research Center's study](http://www.pewinternet.org/Reports/2012/Facebook-users.aspx) [↩](#a1)  
<a name="myfootnote2"></a>2: [Friendship paradox](https://en.wikipedia.org/wiki/Friendship_paradox) [↩](#a2)  
<a name="myfootnote3"></a>3: [Regex editor](https://regex101.com/) [↩](#a3)  
