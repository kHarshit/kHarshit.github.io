---
layout: post
title: "Color and color spaces in Computer Vision"
date: 2020-01-17
categories: [Computer Vision]
---

> A picture is worth a millions words.  

<img src="/img/debashis-biswas-dyPFnxxUhYk-unsplash.jpg" style="display: block; margin: auto;  max-width: 100%;">
Photo by [Debashis Biswas](https://unsplash.com/@debashismelts?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/holi-color?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

The color we see is how our brain visually perceive the world. The color of an object is determined by the different wavelengths of light it reflects (and absorbs), which is affected by the object's physical properties.

> Color is a perception, not the physical property of an object ... though it's affected by the object's properties.

## Color space Vs Color model

In order to categorize and represent colors in computers, we use color models such as RGB that mathematically describe colors. On the other hand, a color space is the organization of colors that is used to display or reproduce colors in a medium such as computer screen. It's how you map the real colors to the color model's discrete values e.g. sRGB and Adobe RGB are two different color spaces, both based on the RGB color model i.e. RGB(16,69,201) may be differently displayed in sRGB and AdobeRGB. You can read more about it [here](https://photo.stackexchange.com/questions/48984/what-is-the-difference-or-relation-between-a-color-model-and-a-color-space/48985).

Note that these terms are often used interchangeably.

## Characteristics of color

The color can be characterized by the following properties:

* **hue**: the dominant color, name of the color itself e.g. red, yellow, green.
* **saturation or chroma**: how pure is the color, the dominance of hue in color, purity, strength, intensity, intense vs dull.
* **brightness or value**: how bright or illuminated the color is, black vs white, dark vs light.

<img src="/img/hue_s_v.jpg" style="display: block; margin: auto;  max-width: 100%;">

## Human eye

The human eye responds differently to different wavelengths of light. In fact, it is trichromatic -- it contains three different types of photo-receptors called cones that are sensitive to different wavelengths of light. These are S-cones (short-wavelength), M-cones (middle-wavelength), and L-cones (long-wavelength) historically considered more sensitive to blue, green, and red light respectively.

The below graph shows the cone cells' response to varying wavelengths of light.

<p style="text-align: center"><a href="https://commons.wikimedia.org/wiki/File:Cone-fundamentals-with-srgb-spectrum.svg#/media/File:Cone-fundamentals-with-srgb-spectrum.svg"><img src="https://upload.wikimedia.org/wikipedia/commons/0/04/Cone-fundamentals-with-srgb-spectrum.svg" alt="Cone-fundamentals-with-srgb-spectrum.svg" width="540" height="380"></a><br>By <a href="//commons.wikimedia.org/wiki/User:BenRG" title="User:BenRG">BenRG</a> - <span class="int-own-work" lang="en">Own work</span>, Public Domain, <a href="https://commons.wikimedia.org/w/index.php?curid=7873848">Link</a></p>

As elucidated by the above figure, the peak value of L cone cells lies in greenish-yellow region, not red. Similarly, the S and M cones don't directly correspond to blue and green color. In fact, the responsiveness of the cones to different colors varies from person-to-person.

## RGB

In RGB color model, all the colors are represented by adding the combinations of three primary colors; Red, Green, and Blue. All the primary colors at full intensity form white represented by RGB(255, 255, 255), and at zero intensity gives black (0, 0, 0).

Though RGB model is a convenient model for representing colors, it differs from how human eye perceive colors.

<img src="/img/rgb_cymk.png" style="display: block; margin: auto; width:70%; max-width: 100%;">

## CYMK

Unlike RGB, CYMK is a subtractive color model i.e. the different colors are represented by subtracting some color from white e.g. cyan  is  white  minus  red. Cyan, magenta, and white are the complements of red, green and, blue respectively. The fourth black color is added to yield CYMK for better reproduction of colors.

Conversion from RGB to CMYK: C=1−R, M=1−G, Y=1−B.


## HSV and HSL

HSV (Hue, Saturation, Value) and HSL (Hue, Saturation, Lightness) color models, developed by transforming the RGB color model, were designed to be more intuitive and interpretable. These are cylindrical representation of colors.

Hue, the color itself, ranges from 0 to 360 starting and ending with red. Saturation defines how pure the color is i.e. the dominance of hue in the color. It ranges from 0 (no color saturation) to 1 (full saturation). The Value (in HSV) and Lightness (in HSL), both ranging from 0 (no light, black) at the bottom to 1 (white) at the top, indicates the illumination level. They differ in the fact that full saturation is achieved at V=1 in HSV, while in HSL, it's achieved at L=0.5.

<img src="/img/hsv_hsl.png" style="display: block; margin: auto; max-width: 100%;">

## Delta E

*To be updated soon...*

<section>
	{% include quiz_color.html %}	 
</section>

---

**References & Further Readings:**  

1. [Color space - Wikipedia](https://en.wikipedia.org/wiki/Color_space)  
2. [Color model - Wikipedia](https://en.wikipedia.org/wiki/Color_model)  
2. [Fundamental concepts of processing and image analysis](https://www.dcc.fc.up.pt/~mcoimbra/lectures/MAPI_1415/CV_1415_T1.pdf)  
3. [Introduction to computer vision](http://sun.aei.polsl.pl/~mkawulok/stud/graph/instr.pdf)
