---
layout: post
title: "Shortest Path: Dijkstra's Algorithm"
date: 2017-11-17
categories: [Technical Fridays, Algorithms]
---

Every day we use Google Maps for commuting. Do you know how it works? 

It uses a variant of one of the fundamental algorithms for finding the shortest path: Dijkstra's algorithm.
Dijkstra's algorithm is a [greedy algorithm]({{ site.url }}{% link _posts/2017-11-03-greedy-algorithms.md %}) i.e. it rely on the **optimal substructure** which means that a shortest path between two vertices contain other shortest paths within it.

> Simplicity is the prerequesite for reliability.
> &mdash; <cite>Edsger Dijkstra</cite>

## Dijkstra's algorithm
1. Dijkstra's algorithm initially marks the distance (from the starting vertex) to every other vertex on the graph with *infinity* and *zero* for our initial vertex.
2. Set the initial vertex as current. Mark all other vertices unvisited. Create a set of all the unvisited vertices called the *unvisited set*.
3. For the current vertex, consider all of its neighbors and calculate their tentative distances. Compare the newly calculated tentative distance to the current assigned value and assign the smaller one.
4. When we are done considering all of the neighbors of the current vertex, mark the current vertex as visited and remove it from the unvisited set. A visited vertex will never be checked again.
5. If the destination vertex has been marked visited (when planning a route between two specific vertices) or if the smallest tentative distance among the vertices in the unvisited set is infinity (when planning a complete traversal; occurs when there is no connection between the initial vertex and remaining unvisited vertices), then stop. The algorithm has finished.
6. Otherwise, select the unvisited vertex that is marked with the smallest tentative distance, set it as the new current vertex, and go back to step 3.

*Note* that all the edges must have non-negative weights otherwise the algorithm doesn't work.

<div style="text-align: center">
<a href="https://commons.wikimedia.org/wiki/File:Dijkstra_Animation.gif#/media/File:Dijkstra_Animation.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/5/57/Dijkstra_Animation.gif" alt="Dijkstra's algorithm runtime" height="222" width="283"></a>
<figcaption>Dijkstra's algorithm to find the shortest path between a and b. It picks the unvisited vertex with the lowest distance, calculates the distance through it to each unvisited neighbor, and updates the neighbor's distance if smaller.</figcaption>
</div>

The process that underlies Dijkstra's algorithm is similar to the greedy process used in Prim's algorithm. Prim's purpose is to find a minimum spanning tree that connects all vertices in the graph; Dijkstra is concerned with only two vertices. Prim's does not evaluate the total weight of the path from the starting vertex, only the individual path.  
Breadth-first search can be viewed as a special-case of Dijkstra's algorithm on unweighted graphs (or when all edge lengths are same).

In 1956, when Edsger W. Dijkstra was first thinking about the problem of finding the shortest path, he had a difficult time trying to find a problem (and its solution) that would be easy to understand for people who didn't come from the computing world! He eventually did come up with a good example problem to showcase the importance of being able to find the shortest path. He chose a map as an example.

> What is the shortest way to travel from Rotterdam to Groningen? It is the algorithm for the shortest path which I designed in about 20 minutes. One morning I was shopping with my young fiancee, and tired, we sat down on the cafe terrace to drink a cup of coffee and I was just thinking about whether I could do this, and then designed the algorithm for the shortest path.


**References:**
1. <a href="https://en.wikipedia.org/wiki/Dijkstra's_algorithm">Dijkstra's algorithm - Wikipedia</a>  
2. Introduction to algorithms / Thomas H. Cormen . . . [et al.].—3rd ed.
