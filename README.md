# kHarshit.github.io

![logo](/img/favicon_files/favicon-96x96.png)

[https://kharshit.github.io](https://kharshit.github.io)

This is my personal website and blog - Technical Fridays.

It's hosted on [GitHub pages](https://pages.github.com/) and powered by [Jekyll](http://jekyllrb.com).   
Comments are powered by [giscus](https://giscus.app/).  
Built from scratch using HTML5, CSS3, and a little of JavaScript.

## Run

```
gem install bundler jekyll
bundle exec jekyll serve
```

## Custom Markdown Features

### Blockquote (curly quotes)

```
> Color is a perception, not the physical property of an object.
```

### Info callout

```
> Sentence-level vs. corpus-level BLEU requires smoothing.
{: .info-callout}
```

### Inline color names

```html
<span class="color-red">red</span>
<span class="color-green">green</span>
<span class="color-blue">blue</span>
```

Available: `color-red`, `color-green`, `color-blue`, `color-yellow`, `color-cyan`, `color-magenta`, `color-violet`, `color-white`, `color-black`, `color-gray`, `color-orange`, `color-pink`, `color-brown`.

### Styled table

```markdown
| Col1 | Col2 |
|------|------|
{: .mbtablestyle}
```

### Image with caption

```liquid
{% include img.html src="/path/to/image.jpg" caption="My caption" width="80%" %}
```

### Numbered steps

```markdown
<div class="mbsteps" markdown="1">
<div class="mbstep" markdown="1">
**Step one**
Description here.
</div>
<div class="mbstep" markdown="1">
**Step two**
More content.
</div>
</div>
```

### Card grid (2, 3, or 4 columns)

```markdown
<div class="mbgrid mbgrid-3" markdown="1">
<div class="mbcard" markdown="1">
**Title**
Content here.
</div>
<div class="mbcard" markdown="1">
**Another card**
More content.
</div>
</div>
```

Cards support custom CSS variables for styling:

```html
<div class="mbcard" style="--mbcard-border:1.5px solid #93c5fd;--mbcard-title-color:#3b82f6">
```

### Interactive widgets

```liquid
{% include interactive/rgb_mixer.html %}
```

## LICENSE

This work is licensed under a
<a rel="license" href="https://creativecommons.org/licenses/by-nc/4.0/">
    Creative Commons Attribution-NonCommercial 4.0 International License
</a>.<br>
<a rel="license" href="https://creativecommons.org/licenses/by-nc/4.0/">
    <img alt="Creative Commons License" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png">
</a>
