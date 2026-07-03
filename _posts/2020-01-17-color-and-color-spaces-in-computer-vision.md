---
layout: post
title: "Color and Color Spaces in Computer Vision"
date: 2020-01-17
permalink: /blog/color-and-color-spaces-in-computer-vision/
categories: [Computer Vision]
mathjax: true
excerpt: "Understanding color models (RGB, HSV, LAB, Luv) and color spaces in computer vision from additive mixing and chromaticity to perceptually uniform CIE spaces and Delta E color difference."
---

## Color

The color we see is how our brain visually perceive the world. The color of an object is determined by the different wavelengths of light it reflects (and absorbs), which is affected by the object's physical properties.

{% include img.html src="/img/blog/color-and-color-spaces-in-computer-vision/debashis-biswas-dyPFnxxUhYk-unsplash.jpg" caption="A picture is worth a millions words. " width="60%" %}

> Color is a perception, not the physical property of an object ... though it's affected by the object's properties.

### Characteristics of color

Any color can be described by three fundamental attributes:

<div class="mbgrid mbgrid-3" markdown="1">
<div class="mbcard" markdown="1">
**Hue**
The dominant color: what we name as <span class="color-red">red</span>, <span class="color-yellow">yellow</span>, <span class="color-green">green</span>, <span class="color-blue">blue</span>, etc. This is what distinguishes a <span class="color-red">red</span> apple from a <span class="color-green">green</span> one.
</div>
<div class="mbcard" markdown="1">
**Saturation (chroma)**
How pure or vivid the color is. A fully saturated color is intense and contains no <span class="color-white">white</span> or <span class="color-gray">gray</span>; a desaturated color appears dull or washed out.
</div>
<div class="mbcard" markdown="1">
**Lightness / Brightness / Value**
How light or dark the color is, ranging from <span class="color-black">black</span> to <span class="color-white">white</span>. Different models use different names (value in HSV, lightness in HSL), but the idea is the same.
</div>
</div>

### Human eye and visible spectrum

Light is a part of the electromagnetic radiation that is visible to the human eye. The **visible spectrum** spans wavelengths from approximately 380 nm (<span class="color-violet">violet</span>) to 780 nm (<span class="color-red">red</span>). Colors corresponding to a single wavelength are called **spectral colors** e.g. light at 570 nm produces a pure spectral <span class="color-yellow">yellow</span>.

{% include img.html src="/img/blog/color-and-color-spaces-in-computer-vision/linear_visible_spectrum.svg" width="80%" caption="The visible spectrum from 380 nm (violet) to 750 nm (red), with wavelength labels and color names (source: Wikipedia)" %}

The human eye does not respond uniformly to all wavelengths. It is **trichromatic**, it contains three types of photo-receptors called **cones** that are sensitive to different but overlapping ranges of wavelengths:

<div class="mbgrid mbgrid-3" markdown="1">
<div class="mbcard" style="--mbcard-border:1.5px solid #93c5fd;--mbcard-title-color:#3b82f6" markdown="1">
**S-cones (short-wavelength)**
Peak sensitivity around 420-440 nm (<span class="color-blue">blue</span>-<span class="color-violet">violet</span>). These cones have the narrowest response curve and are most sensitive to the short-wavelength end of the visible spectrum.
</div>
<div class="mbcard" style="--mbcard-border:1.5px solid #86efac;--mbcard-title-color:#22c55e" markdown="1">
**M-cones (middle-wavelength)**
Peak sensitivity around 534-555 nm (<span class="color-green">green</span>). They respond strongly to the middle of the visible spectrum and overlap significantly with L-cones.
</div>
<div class="mbcard" style="--mbcard-border:1.5px solid #fca5a5;--mbcard-title-color:#ef4444" markdown="1">
**L-cones (long-wavelength)**
Peak sensitivity around 564-580 nm (<span class="color-yellow">yellow</span>-<span class="color-green">green</span>), not <span class="color-red">red</span>. Our perception of <span class="color-red">red</span> comes from the long tail of L-cone sensitivity combined with very low M-cone response at those wavelengths.
</div>
</div>

The graph below shows the normalized response of each cone type across the visible spectrum.

{% include img.html src="/img/blog/color-and-color-spaces-in-computer-vision/cone_fundamentals.svg" width="80%" caption="Normalized response curves of S-cones (blue), M-cones (green), and L-cones (red) across the visible spectrum. (source: Wikipedia)" %}

<span class="color-white">White</span> is **not** a spectral color, no single wavelength produces it. We perceive <span class="color-white">white</span> when S, M, and L cones are stimulated at similar levels by a broad mix of wavelengths (like sunlight). This is also why RGB displays mix <span class="color-red">red</span>, <span class="color-green">green</span>, and <span class="color-blue">blue</span> subpixels to produce <span class="color-white">white</span>.

Because the retina has three different cone types, any color sensation can be matched using just three primary colors with varying intensities. This property is called **trichromacy** and is the foundation of all RGB display technologies.

The responses of the S, M, and L cones form the basis of the **LMS color space**, a color space defined directly by the human visual system's physiological response. Most other color spaces (RGB, XYZ, Lab) are derived from or calibrated against this fundamental space.

In addition to cones, the eye also contains **rods**, highly light-sensitive receptors that handle vision in low-light conditions (scotopic vision). Rods are responsible for our ability to see in dim light but do not contribute to color perception since they can't distinguish different wavelengths of light.

## Color space vs Color model

These two terms are often used interchangeably, but they mean different things:

A **color model** is a mathematical way to organize colors, it defines the axes and the structure. RGB uses a cube with <span class="color-red">red</span>, <span class="color-green">green</span>, and <span class="color-blue">blue</span> axes. HSV uses a cylinder with hue, saturation, and value. The model tells you _how_ colors are arranged, but not what they actually look like.

A **color space** takes a model and maps it to real-world colors. It gives it real-world meaning by defining things like the exact primary colors, white point, and gamma correction. For example, sRGB and Adobe RGB are both based on the RGB model, but they define the primary colors differently. sRGB has a smaller gamut (range of colors) than Adobe RGB. So the same RGB values like (128, 0, 0) will look different on each.

Think of it this way: the color model is the coordinate system, and the color space is the mapping from those coordinates to actual colors you see on a screen.

## RGB

In RGB color model, all the colors are represented by adding the combinations of three primary colors; <span class="color-red">Red</span>, <span class="color-green">Green</span>, and <span class="color-blue">Blue</span>. All the primary colors at full intensity form <span class="color-white">white</span> represented by RGB(255, 255, 255), and at zero intensity gives <span class="color-black">black</span> (0, 0, 0).

Though RGB model is a convenient model for representing colors, it differs from how human eye perceive colors.

{% include interactive/rgb_mixer.html %}

{% include img.html src="/img/blog/color-and-color-spaces-in-computer-vision/rgb_cymk.png" width="70%" caption="RGB (additive) versus CMYK (subtractive) color models." %}

## CYMK

Unlike RGB, CYMK is a subtractive color model i.e. the different colors are represented by subtracting some color from <span class="color-white">white</span> e.g. <span class="color-cyan">cyan</span> is <span class="color-white">white</span> minus <span class="color-red">red</span>. <span class="color-cyan">Cyan</span>, <span class="color-magenta">magenta</span>, and <span class="color-yellow">yellow</span> are the complements of <span class="color-red">red</span>, <span class="color-green">green</span> and, <span class="color-blue">blue</span> respectively. The fourth <span class="color-black">black</span> color is added to yield CYMK for better reproduction of colors.

Conversion from RGB to CMYK: C=1−R, M=1−G, Y=1−B.

## YUV

These color spaces transform RGB into one **brightness** channel and two **color** channels:

- **Y (luma)**: how bright the pixel is. It's a weighted average of R, G, B; notice <span class="color-green">green</span> gets the highest weight (0.587) since our eyes are most sensitive to <span class="color-green">green</span>.
- **U and V (chroma)**: encode the color information as difference signals (<span class="color-blue">blue</span> vs <span class="color-yellow">yellow</span>, <span class="color-red">red</span> vs <span class="color-cyan">cyan</span>).

This design was originally for broadcast TV, black-and-white sets used only the Y channel and ignored U/V, while color TVs decoded all three.

$$
\begin{aligned}
Y &= 0.299R + 0.587G + 0.114B \\
U &= -0.299R - 0.587G + 0.886B \\
V &= 0.701R - 0.587G - 0.114B
\end{aligned}
$$

## HSV and HSL

Both HSV (Hue, Saturation, Value) and HSL (Hue, Saturation, Lightness) are cylindrical color models built from RGB. They were designed to be more intuitive. Instead of mixing amounts of <span class="color-red">red</span>, <span class="color-green">green</span>, and <span class="color-blue">blue</span>, you pick a color (hue), decide how vivid it should be (saturation), and how bright (value/lightness).

- **Hue** (0°-360°): the color itself, starting and ending with <span class="color-red">red</span>.
- **Saturation** (0-1): how pure the color is. 0 is <span class="color-gray">gray</span>, 1 is fully vivid.
- **Value / Lightness** (0-1): how bright the color is. 0 is always <span class="color-black">black</span>. At 1 they both reach <span class="color-white">white</span>, but full saturation is reached at V=1 for HSV and at L=0.5 for HSL.

{% include img.html src="/img/blog/color-and-color-spaces-in-computer-vision/hsv_hsl.png" caption="HSL and HSV: cylindrical representations of color" %}

{% include interactive/hsv_explorer.html %}

## Chromaticity

Imagine you're looking at a <span class="color-red">red</span> apple in sunlight. Now imagine the same apple in shadow, it's darker, but your brain still recognizes it as the same <span class="color-red">red</span>. You instinctively separate the apple's **intrinsic color** from the amount of light hitting it. **Chromaticity** is the mathematical version of this separation: it describes the "color quality" of a pixel independently of its brightness.

The simplest form is **rg chromaticity**, which normalizes RGB values by their total intensity:

$$r = \frac{R}{R + G + B}, \quad g = \frac{G}{R + G + B}, \quad b = \frac{B}{R + G + B}$$

Each channel tells you "what fraction of the total light is red (green / blue)?" Two pixels with the same underlying surface color but different illumination will have similar (r, g, b) ratios, the intensity cancels out.

Since $$r + g + b = 1$$, we only need two of the three values. Plotting every possible (r, g) pair produces a right triangle called the **rg chromaticity diagram**.

{% include interactive/chromaticity_explorer.html %}

Chromaticity is widely used in computer vision: discarding intensity lets algorithms segment objects by material color while staying robust to shadows and illumination gradients.

## CIE Color Space

Before understanding CIE Lab (L\*a\*b\*) color space, let's look at the problems with RGB and HSV. RGB is device-oriented: it describes how much <span class="color-red">red</span>, <span class="color-green">green</span>, and <span class="color-blue">blue</span> light a screen emits, not how a human perceives the resulting color. This causes these problems:

<div class="mbgrid mbgrid-2" markdown="1">
<div class="mbcard" style="--mbcard-border: 1.5px solid #d4a0a0; --mbcard-title-color: #e07070" markdown="1">
**Non-uniform perceptual spacing**
equal numerical steps in RGB do not produce equal perceived color differences. For example, changing <span class="color-blue">blue</span> from (0, 0, 200) to (0, 0, 210) looks like a much smaller change than shifting <span class="color-red">red</span> from (200, 0, 0) to (210, 0, 0), even though the Euclidean distance is identical. Human eyes are simply more sensitive to some regions of color space than others.
</div>
<div class="mbcard" style="--mbcard-border: 1.5px solid #d4a0a0; --mbcard-title-color: #e07070" markdown="1">
**Conflation of luminance and chrominance**
RGB mixes lightness and color information in all three channels. The human visual system separates these: we are far more sensitive to changes in luminance than to changes in hue or saturation at the same luminance level. RGB gives no direct handle on this separation.
</div>
</div>

### XYZ

XYZ is derived from human visual perception experiments, its color matching functions were calibrated to how the average eye responds to light. Each channel is a linear transformation of RGB and carries color information.

It has three channels:
- **X** roughly corresponds to <span class="color-red">red</span> sensation.
- **Y** corresponds to luminance (brightness).
- **Z** roughly corresponds to <span class="color-blue">blue</span> sensation.

A convenient 2D slice of XYZ is the **xy chromaticity** diagram, which normalizes out luminance just like rg chromaticity does for RGB. Plotting x and y for all spectral colors produces the horseshoe-shaped locus.

$$x = \frac{X}{X + Y + Z}, \quad y = \frac{Y}{X + Y + Z}$$

But XYZ is **linear** so equal steps don't feel equal to the eye. Lab and Luv fix this with **non-linear** transforms, creating perceptually uniform spaces where Euclidean distance ≈ perceived difference.

### CIE Lab

CIE Lab was designed for **reflected light** e.g. surfaces, prints, and photographs. It addresses both RGB issues by mapping colors into a space calibrated to human perception, decoupling intensity (L\*) from chromaticity (a\*, b\*), and spacing colors so that equal distances feel equally different.

It has three channels:
- **L\*** lightness, ranging from 0 (<span class="color-black">black</span>) to 100 (<span class="color-white">white</span>).
- **a\*** <span class="color-green">green</span>-<span class="color-red">red</span> axis, ranging from <span class="color-green">green</span> (-) to <span class="color-red">red</span> (+).
- **b\*** <span class="color-blue">blue</span>-<span class="color-yellow">yellow</span> axis, ranging from <span class="color-blue">blue</span> (-) to <span class="color-yellow">yellow</span> (+).

### CIE Luv

CIE Luv (L\*u\*v\*) was developed for **emitted light** e.g. computer screens, TV displays, and other self-luminous devices. It shares the same perceptual uniformity goals as Lab but uses a different chromaticity formulation better suited to additive color mixing.

The three channels are:
- **L\*** lightness, identical to Lab: 0 (<span class="color-black">black</span>) to 100 (<span class="color-white">white</span>).
- **u\*** chromaticity coordinate, roughly <span class="color-red">red</span>-<span class="color-green">green</span> axis.
- **v\*** chromaticity coordinate, roughly <span class="color-blue">blue</span>-<span class="color-yellow">yellow</span> axis.

In practice, Lab is more common in image processing and computer vision, while Luv is often preferred for applications involving display calibration and video processing.

### Delta E

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

The original CIE76 was further improved by following standards. 

- **ΔE\*94**: Adds weighting functions for chroma and hue to better match human perception.
- **ΔE\*00**: The current industry standard. Introduces corrections for lightness, chroma, hue, and a rotation term for the <span class="color-blue">blue</span> region.

$$\Delta E_{00} = \sqrt{\left(\frac{\Delta L'}{k_L S_L}\right)^2 + \left(\frac{\Delta C'}{k_C S_C}\right)^2 + \left(\frac{\Delta H'}{k_H S_H}\right)^2 + R_T \frac{\Delta C'}{k_C S_C} \frac{\Delta H'}{k_H S_H}}$$

where $$S_L$$, $$S_C$$, $$S_H$$ are weighting functions and $$R_T$$ is the rotation term.

{% include interactive/delta_e.html %}

<section>
	{% include quiz/color.html %}	 
</section>


**References & Further Readings:**  

- [Color space - Wikipedia](https://en.wikipedia.org/wiki/Color_space)  
- [Color model - Wikipedia](https://en.wikipedia.org/wiki/Color_model)  
- [Fundamental concepts of processing and image analysis](https://www.dcc.fc.up.pt/~mcoimbra/lectures/MAPI_1415/CV_1415_T1.pdf)  
- [Introduction to computer vision](http://sun.aei.polsl.pl/~mkawulok/stud/graph/instr.pdf)  
- [Color difference - Wikipedia](https://en.wikipedia.org/wiki/Color_difference)
- [CIELAB color space - Wikipedia](https://en.wikipedia.org/wiki/CIELAB_color_space)
- [CIE 1931 color space - Wikipedia](https://en.wikipedia.org/wiki/CIE_1931_color_space)
- [YUV - Wikipedia](https://en.wikipedia.org/wiki/YUV)
- [CIE Luv color space - Wikipedia](https://en.wikipedia.org/wiki/CIELUV)
- [CIE xy chromaticity diagram](https://en.wikipedia.org/wiki/CIE_1931_color_space#Chromaticity_diagram)
- [Visible spectrum - Wikipedia](https://en.wikipedia.org/wiki/Visible_spectrum)
