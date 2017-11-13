---
layout: post
title: "Divide and Conquer"
date: 2017-11-10
categories: [Technical Fridays, Algorithms]
---

Say you want to go to Harvard for your graduate studies in a year. But, the academic fee of $40,000 is intimidating. And, with every passing week, it's getting worse. So, what do you do?  
Think like a computer scientist. Divide the big amount into smaller manageable amount. Say, you take an education loan of $20,000. Hoping ;) that you will be lucky enough to get scholarship of at least $5,000; still you are falling short of $5,000. Doing a little math, you can figure it out that you need to save approx. $416/month i.e. $96/week. Now, you can cut your monthly expenses to save some money. Even though it does not ensure your  acceptance into Harvard, you'll have a handsome amount of money in your bank account after a year.

What we just did was to break a problem into smaller subproblems and the solve them and finally by combining the solutions of the subproblems, we solved the original problem. This approach of problem-solving is known as *Divide and Conquer*.

The three main steps of divide and conquer algorithm paradigm are:
* **Divide** the problem into subproblems
* **Conquer** the subproblems by solving them recursively or directly.
* **Combine** the solutions to the subproblems into the solution to get the solution of the original problem.

## Algorithms using Divide and Conquer paradigm:
* Binary search for searching
* Quicksort and merge sort for sorting
* Karatsuba algorithm for multiplication of large numbers
* Strassen’s algorithm for matrix multiplication

<div style="text-align: center">
<a title="By Swfung8 (Own work) [CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0) or GFDL (http://www.gnu.org/copyleft/fdl.html)], via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File%3AMerge-sort-example-300px.gif"><img width="256" alt="Merge-sort-example-300px" src="https://upload.wikimedia.org/wikipedia/commons/c/cc/Merge-sort-example-300px.gif"/>
<figcaption>Merge-sort</figcaption></a>
</div>

So, next time you come across a problem; think like a computer scientist and don't forget:
> Many a drop make an ocean.


**References:**
1. <a href="https://en.wikipedia.org/wiki/Divide_and_conquer_algorithm">Divide and Conquer algorithm - Wikipedia</a>
2. Introduction to algorithms / Thomas H. Cormen . . . [et al.].—3rd ed.

