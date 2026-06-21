---
layout: post
title: "Color and color spaces in Computer Vision"
date: 2020-01-17
categories: [Computer Vision]
mathjax: true
excerpt: "Understanding color models (RGB, HSV, LAB) and color spaces in computer vision, how computers represent and work with color."
---

{% include img.html src="/img/debashis-biswas-dyPFnxxUhYk-unsplash.jpg" caption="A picture is worth a millions words. " width="60%" %}

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

## CIE Lab

Before understanding CIE Lab (L\*a\*b\*) color space, let's look at the problems with RGB and HSV. RGB is device-oriented: it describes how much red, green, and blue light a screen emits, not how a human perceives the resulting color. This causes these problems:

<div class="mbgrid mbgrid-2" markdown="1">
<div class="mbcard" style="--mbcard-border: 1.5px solid #d4a0a0; --mbcard-title-color: #e07070" markdown="1">
**Non-uniform perceptual spacing**
equal numerical steps in RGB do not produce equal perceived color differences. For example, changing blue from (0, 0, 200) to (0, 0, 210) looks like a much smaller change than shifting red from (200, 0, 0) to (210, 0, 0), even though the Euclidean distance is identical. Human eyes are simply more sensitive to some regions of color space than others.
</div>
<div class="mbcard" style="--mbcard-border: 1.5px solid #d4a0a0; --mbcard-title-color: #e07070" markdown="1">
**Conflation of luminance and chrominance**
RGB mixes lightness and color information in all three channels. The human visual system separates these: we are far more sensitive to changes in luminance than to changes in hue or saturation at the same luminance level. RGB gives no direct handle on this separation.
</div>
</div>

CIE Lab addresses both issues by mapping colors into a space that is calibrated to human perception, decoupling lightness (L\*) from color (a\*, b\*), and spacing colors so that equal distances feel equally different.

It has three channels:
- **L\*** lightness, ranging from 0 (black) to 100 (white).
- **a\*** green-red axis, ranging from green (-) to red (+).
- **b\*** blue-yellow axis, ranging from blue (-) to yellow (+).

## Delta E

Delta E (`ΔE`) is a single number that quantifies the **perceptual difference between two colors**. It answers the question: *how different do two colors look to a human observer?*

Delta E is computed in the CIE Lab color space. The simplest version, **ΔE\*ab** (CIE76), is just the Euclidean distance between two colors in Lab space:

$$\Delta E^{*}_{ab} = \sqrt{(L^{*}_2 - L^{*}_1)^2 + (a^{*}_2 - a^{*}_1)^2 + (b^{*}_2 - b^{*}_1)^2}$$

Delta E values can be interpreting as follows.

| ΔE value | Perception |
|---|---|---|
| &lt; 1 | Not perceptible by human eyes |
| 1 – 2 | Perceptible only on close observation |
| 2 – 10 | Perceptible at a glance |
| 11 – 49 | Colors are more similar than opposite |
| 100 | Colors are exact opposites |
{:.mbtablestyle}

CIE76 is simple but not perfectly uniform — it has notable inaccuracies in the blue region and for highly saturated colors. Improved formulas were introduced over time:

- **ΔE\*94**: Adds weighting functions for chroma and hue to better match human perception.
- **ΔE\*00**: The current industry standard. Introduces corrections for lightness, chroma, hue, and a rotation term for the blue region. It is significantly more accurate across all colors.

$$\Delta E_{00} = \sqrt{\left(\frac{\Delta L'}{k_L S_L}\right)^2 + \left(\frac{\Delta C'}{k_C S_C}\right)^2 + \left(\frac{\Delta H'}{k_H S_H}\right)^2 + R_T \frac{\Delta C'}{k_C S_C} \frac{\Delta H'}{k_H S_H}}$$

where $$S_L$$, $$S_C$$, $$S_H$$ are weighting functions and $$R_T$$ is the rotation term.

<section>
	{% include quiz_color.html %}	 
</section>

---

**References & Further Readings:**  

1. [Color space - Wikipedia](https://en.wikipedia.org/wiki/Color_space)  
2. [Color model - Wikipedia](https://en.wikipedia.org/wiki/Color_model)  
3. [Fundamental concepts of processing and image analysis](https://www.dcc.fc.up.pt/~mcoimbra/lectures/MAPI_1415/CV_1415_T1.pdf)  
4. [Introduction to computer vision](http://sun.aei.polsl.pl/~mkawulok/stud/graph/instr.pdf)  
5. [Color difference - Wikipedia](https://en.wikipedia.org/wiki/Color_difference)
6. [CIELAB color space - Wikipedia](https://en.wikipedia.org/wiki/CIELAB_color_space)
