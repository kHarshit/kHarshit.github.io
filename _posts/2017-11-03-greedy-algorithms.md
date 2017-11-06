---
layout: post
title: "Greedy Algorithms"
date: 2017-11-03
categories: [Technical Fridays]
---

Let's say you want to count out a certain amount of money, using the fewest possible number of coins.  
Take for example, $6.39, you can choose:
a $5 bill, a $1 bill to make $6, a 25¢ and a 10¢ coin to make $6.35, and four 1¢ to finally make $6.39.  
Note that in each step, we took the highest coin value (taking a greedy step) thus reducing the problem to a smaller problem. This paradigm of problem-solving is known as *Greedy Algorithms*.

A greedy algorithm always makes the choice that looks best at the moment i.e. it makes a locally optimal choice in the hope that this choice will lead to a globally optimal solution.

**NOTE:** Greedy algorithm do not always yield optimal solutions, but for many problems they do.

<figure>
<img src="/img/greedy-search-path-example.gif" style="display: block; margin: auto; width: auto; max-width: 100%;"> 
<figcaption>
With a goal of reaching the largest-sum, at each step, the greedy algorithm will choose what appears to be the optimal immediate choice, so it will choose 12 instead of 3 at the second step, and will not reach the best solution, which contains 99.</figcaption>
</figure> 


But how can we tell whether a greedy algorithm will solve a particular optimization problem? Most problems for which they work will have two properties:

* **Greedy choice property:** We can assemble a globally optimal solution by making locally optimal (greedy) choices. It means that we make the choice that looks best in the current problem, without considering results from subproblems.
* **Optimal substructure:** A problem exhibits optimal substructure if an optimal solution to the problem contains optimal solutions to the sub-problems.


## Greedy algorithms vs Dynamic programming

In dynamic programming, we make choice at each step, but the choice usually depends on the solutions to subproblems. In a greedy algorithm, we make whatever choice seems best at the moment and then solve the subproblem that remains.

Unlike dynamic programming, which solves the subproblems before making the first choice, a greedy algorithm makes its first choice before solving any subproblems.

In other words, a greedy algorithm never reconsiders its choices.

## Algorithms using Greedy algorithm paradigm:
* Dijkstra's Algorithm for finding shortest path
* Prim's algorithm for finding Minimum Spanning Trees (MST)
* Huffman coding for data compression

<div style="text-align: center">
    <a title="By Ken10311120 (Own work) [CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0)], via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File%3AHuffman_algorithm.gif"><img width="256" alt="Huffman algorithm" src="https://upload.wikimedia.org/wikipedia/commons/c/c8/Huffman_algorithm.gif"/>
    <figcaption>Huffman coding</figcaption></a>
</div>


**References:**
1. <a href="https://en.wikipedia.org/wiki/Greedy_algorithm">Greedy algorithms - Wikipedia</a>
2. Introduction to algorithms / Thomas H. Cormen . . . [et al.].—3rd ed.