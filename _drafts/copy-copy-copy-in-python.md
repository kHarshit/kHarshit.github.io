---
layout: post
title: "Copy Copy Copy in Python"
categories: [Technical Fridays, Python]
---

{% highlight python %}
>>> a = 2
>>> b = 2
{% endhighlight %}

We know that a and b will refer to a integer object 2, but we don't know yet whether they point to the same object or not.
There are two ways the Python interpreter could arrange its memory:


In one case, a and b refer to two different objects that have the same value. In the second case, they refer to the same object.

We can test whether two names refer to the same object using the `is` operator:
{% highlight python %}
>>> a is b  # tests reference
True
>>> a == b  # tests value
True
{% endhighlight %}

This tells us that both a and b refer to the same object, and that it is the second of the two state snapshots that accurately describes the relationship.

Since strings are immutable, Python optimizes resources by making two names that refer to the same string value refer to the same object.  
This is not the case with lists:

{% highlight python %}
>>> a = [1, 2, 3]
>>> b = [1, 2, 3]
>>> a == b
True
>>> a is b
False
{% endhighlight %}

The state snapshot here looks like this:

a and b have the same value but do not refer to the same object.

{% highlight python %}
>>> a = [1, 2, 3]
>>> c = a.copy()
>>> c
[1, 2, 3]
>>> a == c
True
>>> a is c
False
{% endhighlight %}

Since variables refer to objects, if we assign one variable to another, both variables refer to the same object:
{% highlight python %}
>>> a = [1, 2, 3]
>>> b = a
>>> a is b
True
{% endhighlight %}

In this case, the state snapshot looks like this:

Because the same list has two different names, a and b, we say that it is **aliased**. Changes made with one alias affect the other:

{% highlight python %}
>>> b[0] = 5
>>> a
[5, 2, 3]
{% endhighlight %}

Although this behavior can be useful, it is sometimes unexpected or undesirable. In general, it is safer to avoid aliasing when you are working with mutable objects (e.g. lists). Of course, for immutable objects (i.e. strings, tuples), there’s no problem — it is just not possible to change something and get a surprise when you access an alias name. That’s why Python is free to alias strings (and any other immutable kinds of data) when it sees an opportunity to economize.

If we want to modify a list and also keep a copy of the original, we need to be able to make a copy of the list itself, not just the reference. This process is sometimes called **cloning**, to avoid the ambiguity of the word copy.  
The easiest way to clone a list is to use the slice operator:

{% highlight python %}
>>> a = [1, 2, 3]
>>> b = a[:]
>>> b
[1, 2, 3]
{% endhighlight %}

Taking any slice of a creates a new list. In this case the slice happens to consist of the whole list. So now the relationship is like this:

Now, we are free to make changes to b without worrying that we'll inadvertently be changing a:

{% highlight python %}
>>> b[0] = 5
>>> a
[1, 2, 3]
{% endhighlight %}
