(function () {
  'use strict';

  function ArchCompare(root) {
    this.root = root;
    this.tabs = root.querySelectorAll('.arch-tab');
    this.svg = root.querySelector('.arch-svg');
    this.detailsEl = root.querySelector('.arch-details');
    this.currentArch = 'clip';

    var self = this;
    this.tabs.forEach(function (tab) {
      tab.addEventListener('click', function () {
        self.setArch(tab.getAttribute('data-arch'));
      });
    });

    this.setArch('clip');
  }

  ArchCompare.prototype.setArch = function (arch) {
    this.currentArch = arch;
    this.tabs.forEach(function (tab) {
      var isActive = tab.getAttribute('data-arch') === arch;
      tab.style.background = isActive ? '#fff' : 'transparent';
      tab.style.color = isActive ? '#20B2AA' : '#888';
      tab.style.boxShadow = isActive ? '0 1px 4px rgba(0,0,0,0.1)' : 'none';
    });

    this.renderDiagram(arch);
    this.renderDetails(arch);
  };

  ArchCompare.prototype.renderDiagram = function (arch) {
    var ns = 'http://www.w3.org/2000/svg';
    var svg = this.svg;
    svg.innerHTML = '';

    var w = 600, h = 320;

    var defs = document.createElementNS(ns, 'defs');
    defs.innerHTML =
      '<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">' +
      '<path d="M0,0 L10,5 L0,10 Z" fill="#555"/></marker>';
    svg.appendChild(defs);

    var bg = document.createElementNS(ns, 'rect');
    bg.setAttribute('width', w); bg.setAttribute('height', h);
    bg.setAttribute('fill', 'var(--bg-color,#f8fafc)'); bg.setAttribute('rx', '8');
    svg.appendChild(bg);

    var archs = {
      clip: {
        title: 'CLIP: Contrastive Language-Image Pre-Training',
        blocks: [
          { label: 'Image Encoder\n(ViT / ResNet)', x: 80, y: 120, w: 130, h: 60, trained: true, color: '#10b981' },
          { label: 'Text Encoder\n(Transformer)', x: 390, y: 120, w: 130, h: 60, trained: true, color: '#10b981' },
          { label: 'Projection\nW_i', x: 80, y: 220, w: 130, h: 40, trained: true, color: '#10b981' },
          { label: 'Projection\nW_t', x: 390, y: 220, w: 130, h: 40, trained: true, color: '#10b981' },
          { label: 'Contrastive\nLoss', x: 235, y: 170, w: 130, h: 50, trained: false, color: '#6366f1' },
        ],
        arrows: [
          [145, 180, 145, 220], [455, 180, 455, 220],
          [210, 240, 235, 220], [455, 260, 365, 220],
        ],
        desc: 'CLIP trains dual encoders jointly using a contrastive objective. Image and text embeddings are projected into a shared space and cosine similarity is maximized for correct pairs. Both encoders are trained from scratch on 400M image-text pairs.',
      },
      flava: {
        title: 'FLAVA: Foundational Language And Vision Alignment',
        blocks: [
          { label: 'Image Encoder\n(ViT)', x: 50, y: 100, w: 110, h: 50, trained: true, color: '#10b981' },
          { label: 'Text Encoder\n(Transformer)', x: 190, y: 100, w: 110, h: 50, trained: true, color: '#10b981' },
          { label: 'Multimodal\nEncoder', x: 330, y: 100, w: 110, h: 50, trained: true, color: '#10b981' },
          { label: 'MIM Head', x: 50, y: 190, w: 110, h: 35, trained: true, color: '#f59e0b' },
          { label: 'MLM Head', x: 190, y: 190, w: 110, h: 35, trained: true, color: '#f59e0b' },
          { label: 'MMM Head', x: 330, y: 190, w: 110, h: 35, trained: true, color: '#f59e0b' },
          { label: 'ITM Head', x: 470, y: 140, w: 90, h: 35, trained: true, color: '#f59e0b' },
        ],
        arrows: [
          [105, 150, 105, 190], [245, 150, 245, 190],
          [385, 150, 385, 190], [385, 150, 470, 175],
          [245, 150, 330, 125],
        ],
        desc: 'FLAVA uses three encoders (image, text, multimodal) with four training objectives. MIM and MLM heads reconstruct masked patches/tokens. MMM head handles cross-modal masking. ITM (Image-Text Matching) head classifies if an image-text pair matches.',
      },
      coca: {
        title: 'CoCa: Contrastive Captioner',
        blocks: [
          { label: 'Image Encoder', x: 80, y: 100, w: 120, h: 50, trained: true, color: '#10b981' },
          { label: 'Unimodal Text\nDecoder', x: 250, y: 100, w: 120, h: 50, trained: true, color: '#10b981' },
          { label: 'Multimodal Text\nDecoder', x: 400, y: 100, w: 130, h: 50, trained: true, color: '#10b981' },
          { label: 'Contrastive\nHead', x: 80, y: 200, w: 120, h: 40, trained: true, color: '#6366f1' },
          { label: 'Generative\nHead (Cap.)', x: 400, y: 200, w: 130, h: 40, trained: true, color: '#ec4899' },
        ],
        arrows: [
          [200, 125, 250, 125], [370, 125, 400, 125],
          [140, 150, 140, 200], [465, 150, 465, 200],
          [250, 150, 250, 175, 400, 175, 400, 150],
        ],
        desc: 'CoCa combines contrastive loss (matching image-text pairs) with generative loss (captioning). The unimodal decoder processes text alone, while the multimodal decoder attends to image features. Trained on ALIGN (1.8B pairs) and JFT-3B.',
      },
      llava: {
        title: 'LLaVA: Large Language-and-Vision Assistant',
        blocks: [
          { label: 'Vision Encoder\n(CLIP ViT)\n❄ Frozen', x: 60, y: 100, w: 110, h: 60, trained: false, color: '#94a3b8' },
          { label: 'Linear\nProjection\n🔥 Trained', x: 210, y: 110, w: 90, h: 40, trained: true, color: '#f59e0b' },
          { label: 'Language Model\n(Vicuna / LLaMA)\n🔥 Finetuned', x: 340, y: 100, w: 130, h: 60, trained: true, color: '#10b981' },
          { label: 'Instruction\nTuning Data', x: 520, y: 140, w: 60, h: 30, trained: false, color: '#94a3b8' },
        ],
        arrows: [
          [170, 130, 210, 130], [300, 130, 340, 130],
          [350, 160, 350, 190, 520, 175],
        ],
        desc: 'LLaVA keeps the CLIP vision encoder frozen and connects it to a Vicuna LLM via a trainable linear projection layer. The LLM is fine-tuned end-to-end on multimodal instruction-following data generated by GPT-4 from COCO captions.',
      },
      frozen: {
        title: 'Frozen: Frozen Language Model VLM',
        blocks: [
          { label: 'Vision Encoder\n(NF-ResNet-50)\n🔥 Trained', x: 60, y: 100, w: 110, h: 60, trained: true, color: '#10b981' },
          { label: 'Linear\nMapping\n🔥 Trained', x: 210, y: 110, w: 90, h: 40, trained: true, color: '#f59e0b' },
          { label: 'Language Model\n(Transformer)\n❄ Frozen', x: 340, y: 100, w: 130, h: 60, trained: false, color: '#94a3b8' },
          { label: 'Text\nOutput', x: 510, y: 115, w: 70, h: 30, trained: false, color: '#94a3b8' },
        ],
        arrows: [
          [170, 130, 210, 130], [300, 130, 340, 130], [470, 130, 510, 130],
        ],
        desc: 'Frozen was the first to connect a vision encoder to a frozen LLM via a mapping layer. The 7B-param language model stays frozen to prevent catastrophic forgetting. Only the vision encoder and linear projection are trained, using a visual prefix approach.',
      },
      blip2: {
        title: 'BLIP-2: Bootstrapping Language-Image Pre-training',
        blocks: [
          { label: 'Vision Encoder\n(CLIP ViT)\n❄ Frozen', x: 50, y: 100, w: 110, h: 60, trained: false, color: '#94a3b8' },
          { label: 'Q-Former\n🔥 Trained', x: 200, y: 105, w: 100, h: 50, trained: true, color: '#f59e0b' },
          { label: 'Language Model\n(LLaMA / OPT)\n❄ Frozen', x: 340, y: 100, w: 130, h: 60, trained: false, color: '#94a3b8' },
          { label: 'Learned\nQueries', x: 200, y: 190, w: 100, h: 30, trained: true, color: '#f59e0b' },
        ],
        arrows: [
          [160, 130, 200, 130], [300, 130, 340, 130],
          [250, 155, 250, 190], [200, 190, 200, 155],
        ],
        desc: 'BLIP-2 introduces Q-Former, a lightweight transformer that bridges frozen vision encoders and frozen LLMs. Learnable query tokens interact with image features via cross-attention. Only Q-Former is trained, drastically reducing compute while achieving strong performance.',
      },
    };

    var data = archs[arch];
    if (!data) return;

    var title = document.createElementNS(ns, 'text');
    title.setAttribute('x', w / 2); title.setAttribute('y', 24);
    title.setAttribute('text-anchor', 'middle'); title.setAttribute('font-size', '13');
    title.setAttribute('font-weight', '700'); title.setAttribute('fill', 'var(--font-color,#333)');
    title.textContent = data.title;
    svg.appendChild(title);

    data.blocks.forEach(function (block) {
      var g = document.createElementNS(ns, 'g');

      var rect = document.createElementNS(ns, 'rect');
      rect.setAttribute('x', block.x); rect.setAttribute('y', block.y);
      rect.setAttribute('width', block.w); rect.setAttribute('height', block.h);
      rect.setAttribute('rx', '8');
      rect.setAttribute('fill', block.color);
      rect.setAttribute('opacity', block.trained ? '1' : '0.7');
      g.appendChild(rect);

      var lines = block.label.split('\n');
      var lineH = 14;
      var startY = block.y + block.h / 2 - (lines.length - 1) * lineH / 2;
      lines.forEach(function (line, li) {
        var t = document.createElementNS(ns, 'text');
        t.setAttribute('x', block.x + block.w / 2);
        t.setAttribute('y', startY + li * lineH + 4);
        t.setAttribute('text-anchor', 'middle');
        t.setAttribute('font-size', '10');
        t.setAttribute('font-weight', '600');
        t.setAttribute('fill', block.trained && block.color !== '#94a3b8' ? '#fff' : '#fff');
        t.textContent = line;
        g.appendChild(t);
      });

      svg.appendChild(g);
    });

    data.arrows.forEach(function (arr) {
      var line = document.createElementNS(ns, 'line');
      line.setAttribute('x1', arr[0]); line.setAttribute('y1', arr[1]);
      line.setAttribute('x2', arr[2]); line.setAttribute('y2', arr[3]);
      line.setAttribute('stroke', '#555'); line.setAttribute('stroke-width', '1.5');
      line.setAttribute('marker-end', 'url(#arr)');
      svg.appendChild(line);

      if (arr.length > 4) {
        var line2 = document.createElementNS(ns, 'line');
        line2.setAttribute('x1', arr[2]); line2.setAttribute('y1', arr[3]);
        line2.setAttribute('x2', arr[4]); line2.setAttribute('y2', arr[5]);
        line2.setAttribute('stroke', '#555'); line2.setAttribute('stroke-width', '1.5');
        line2.setAttribute('marker-end', 'url(#arr)');
        svg.appendChild(line2);
      }
    });

    var legend = [
      { color: '#10b981', label: 'Trained' },
      { color: '#94a3b8', label: 'Frozen / Pretrained' },
      { color: '#f59e0b', label: 'Projection / Connector' },
      { color: '#6366f1', label: 'Loss Head' },
      { color: '#ec4899', label: 'Generative Head' },
    ];
    var legendContainer = document.createElementNS(ns, 'g');
    var lx = 30, ly = h - 20;
    legendContainer.setAttribute('transform', 'translate(' + lx + ',' + ly + ')');
    legend.forEach(function (item, li) {
      var lr = document.createElementNS(ns, 'rect');
      lr.setAttribute('x', li * 120); lr.setAttribute('y', 0);
      lr.setAttribute('width', 10); lr.setAttribute('height', 10);
      lr.setAttribute('rx', '2'); lr.setAttribute('fill', item.color);
      legendContainer.appendChild(lr);
      var lt = document.createElementNS(ns, 'text');
      lt.setAttribute('x', li * 120 + 14); lt.setAttribute('y', 9);
      lt.setAttribute('font-size', '9'); lt.setAttribute('fill', '#888');
      lt.textContent = item.label;
      legendContainer.appendChild(lt);
    });
    svg.appendChild(legendContainer);
  };

  ArchCompare.prototype.renderDetails = function (arch) {
    var details = {
      clip: 'CLIP jointly trains image and text encoders with a symmetric contrastive loss. The model processes N image-text pairs in a batch, computing cosine similarities between all pairs. The loss maximizes diagonal (correct pair) scores while minimizing off-diagonal scores. At inference, CLIP can zero-shot classify images by comparing against text prompts for each class.',
      flava: 'FLAVA uses three encoders: a ViT for images, a transformer for text, and a multimodal encoder that fuses both. It trains with four objectives: MIM (mask image patches), MLM (mask text tokens), MMM (mask both for cross-modal prediction), and ITM (binary match/no-match classification). This makes FLAVA a universal model for vision, language, and multimodal tasks.',
      coca: 'CoCa unifies contrastive and generative training in a single model. The image encoder feeds into both a contrastive head (for alignment) and a multimodal text decoder (for caption generation). The unimodal decoder processes text alone, while the multimodal decoder cross-attends to image features — enabling both retrieval and generation from one model.',
      llava: 'LLaVA connects a frozen CLIP vision encoder to Vicuna LLM via a simple linear projection. Stage 1: train only the projection layer on image-caption pairs. Stage 2: fine-tune the projection + LLM end-to-end on GPT-4-generated multimodal instruction data (conversation, detailed description, complex reasoning). The visual encoder stays frozen throughout.',
      frozen: 'Frozen was the pioneering approach for using pretrained LLMs in VLMs. The 7B-parameter transformer LM is kept entirely frozen to avoid catastrophic forgetting. A linear mapping layer projects visual features from a trained vision encoder into the LM\'s embedding space as a visual prefix. Only the vision encoder and mapping are trained.',
      blip2: 'BLIP-2 introduces Q-Former, a lightweight trainable transformer that bridges frozen vision encoders and frozen LLMs. Learnable query tokens attend to image features via cross-attention and produce a fixed-length visual representation readable by the LLM. This approach achieves strong performance while training only ~188M parameters of Q-Former.',
    };
    this.detailsEl.innerHTML = '<strong>' + arch.toUpperCase() + ':</strong> ' + (details[arch] || '');
  };

  function init() {
    var el = document.getElementById('arch-compare');
    if (el && !el.hasAttribute('data-dt-init')) {
      el.setAttribute('data-dt-init', '1');
      new ArchCompare(el);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
