---
layout: post
title: "Structure of the web"
date: 2017-09-08
categories: [Technical Fridays]
---

A study<sup id="a1">[1](#myfootnote1)</sup> of the web's structure reveals that it isn't the fully interconnected network that we've been led to believe. The study suggests that the chance of being able to surf between two randomly chosen pages is less than one in four.

If we consider web pages as `vertices` and hyperlinks as `edges`. Then, the web can be represented as a <abbr title='a graph that is a set of vertices connected by edges, where the edges have a direction associated with them.'>directed graph</abbr>. Now, the question is what does the web graph look like?

In 1999, after the Web had been growing for the better part of a decade, Andrei Broder<sup id="a2">[2](#myfootnote1)</sup> and his colleagues set out to build a global map of the Web, using strongly connected components as the building blocks. They analyzed the data from one of the largest commercial search engines at the time, AltaVista<sup id="a3">[3](#myfootnote3)</sup>.

Their finding included that the web contains a giant strongly connected component. In case you don't know, a strongly connected component is a region from which you can get from any point to any other point along a directed path. So in the context of the web graph, with this giant <abbr title='strongly connected component'>SCC</abbr>, what this means is that from any webpage inside this blob, you can get to any other webpage inside this blob, just by traversing a sequence of hyperlinks. 

The two regions of approximately equal size on the two sides of CORE are named as:
* `IN`: nodes that can reach the giant SCC but cannot be reached from it e.g. new web pages, and
* `OUT`: nodes that can be reached from the giant SCC but cannot reach it e.g. corporate websites.

This structure of web is known as the **Bowtie structure**.

<img src="/img/structure_of_web.gif" style="float: center; display: block; margin: auto; width: auto; max-width: 100%;">

There are pages that belong to none of `IN`, `OUT`, or `the giant SCC` i.e. they can neither reach the giant SCC nor be reached from it. These are clasified as:
* `Tendrils`: the nodes reachable from IN that can't reach the giant SCC, and
            the nodes that can reach OUT but can't be reached from the giant SCC.
            If a tendril node satisfies both conditions then it’s part of a `tube` that travels from IN to OUT without touching the giant SCC, and
* `Disconnected`: nodes that belong to none of the previous catogories.

The study collected the following data:

| Structure | Altavista, May 1999         |
|-----------|-----------------------------|
| Total     | 203.5 million               |
| SCC       | 56.5 million                |
| IN        | 43.3 million                |
| OUT       | 43.1 million                |
| Tendrils  | 43.7 million                |
| Links     | 1466 million (7.2 per page) |
|and others  ||
{:.mbtablestyle}

Taken as a whole, *the bow-tie structure* of the Web provides a high-level view of the Web's structure, based on its reachablility properties and how its strongly connected components fit together.



**Footnotes:**  
<a name="myfootnote1"></a>1, 2: [Graph structure in the Web](/assets/graph_broder.pdf) [↩](#a1)  
<a name="myfootnote3"></a>3: [Altavista](https://en.wikipedia.org/wiki/AltaVista) [↩](#a3)  
