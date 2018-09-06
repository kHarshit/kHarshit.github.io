---
layout: post
title: "Dynamic Programming"
date: 2018-07-13
categories: [Algorithms, Python]
---

> Those who cannot remember the past are condemned to repeat it.
> &mdash; <cite>George Santayana</cite>

The Fibonacci sequence is defined as follows:

$$F_1 = F_2 = 1$$

$$F_n = F_{n−1} + F_{n−2}$$

Let's write a program to find the *n*th Fibonacci number.

{% highlight python %}
def fib(n):
    if n ≤ 2:
        return f = 1
    else:
        return f = fib(n − 1) + fib(n − 2)
{% endhighlight %}

Notice, some of the numbers are repeatedly evaluated thus this algorithm takes a large amount of running time (exponential). Let's store the numbers in a dictionary after calculation then we'll look up the number in the dictionary before evaluation to save time.

{% highlight python %}
memo = {}
def fib(n):
    if n in memo:
        return memo[n]
    else:
        if n <= 2 :
            f = 1
        else:
            f = fib(n -1) + fib(n - 2)
            memo[n] = f
        return f
{% endhighlight %}

The above technique of using a look-up table is called **memoization**. This process of using a memoization based algorithm is **dynamic programming**. The running time of dynamic programming is given by  

$$time = \text{number of subproblems} * time / \text{subproblem} = n * \theta(1) = \theta(n)$$

In fact, here we used a *top-down approach* by solving the subproblems we encountered while solving the problem, there also exists a *bottom-up approach* of dynamic programming where we first solve all the subproblems before arriving at the problem itself.

{% highlight python %}
def fib(n):
    fib = {}
    for k in range(1, n):
        if k <= 2:
            f = 1
        else:
            f = fib[k - 1] + fib[k - 2]
            fib[k] = f
    return fib[n]
{% endhighlight %}

A problem must possess the following characteristics in order to be solvable by dynamic programming:

* **Optimal substructure:**  A problem exhibits optimal substructure if an optimal solution to the problem contains within it optimal solutions to subproblems.
* **Overlapping subproblems:**  When the solution of some of the subproblems are same *i.e. subproblems share their solutions*, they are said to be overlapping. That is, a recursive algorithm for the problem solves the same subproblem repeatedly.

> Writes down "1+1+1+1+1+1+1+1 =" on a sheet of paper.  
"What's that equal to?"  
Counting "Eight!"  
Writes down another "1+" on the left.  
"What about that?"  
"Nine!" " How'd you know it was nine so fast?"  
"You just added one more!"  
"So you didn't need to recount because you remembered there were eight! Dynamic Programming is just a fancy way to say remembering stuff to save time later!" 
> &mdash; <cite>Quora answer<sup id="a1">[1](#myfootnote1)</sup></cite>

Dynamic programming differs from [greedy algorithms]({% post_url 2017-11-03-greedy-algorithms %}) in one aspect, that is, it first find the optimal solution to subproblems then make the choice while greedy algorithms first make a greedy choice then solve the subproblems.

## Algorithms solvable using Dynamic programming paradigm:
* 0-1 Knapsack problem
* Parenthesization (Matrix chain multiplication)
* Longest Common Subsequence

<figcaption style="text-align: center;">Travelling Salesman Problem</figcaption>
<img src="https://imgs.xkcd.com/comics/travelling_salesman_problem.png " style="display: block; margin: auto; width: auto; max-width: 100%;">

**References:**  
<a name="#" />1. Introduction to algorithms / Thomas H. Cormen . . . [et al.].—3rd ed.  
<a name="#" />2. [Dynamic Programmin - Wikipedia](https://en.wikipedia.org/wiki/Dynamic_programming)  
<a name="myfootnote1" />3. [How should I explain dynamic programming to a 4-year-old? - Quora](http://qr.ae/TUN9n5) [↩](#a1)
