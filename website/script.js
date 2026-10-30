(function () {
  const $ = id => document.getElementById(id);
  const IMG = 32;
  let model = null;

  const drop = $('drop'), fileInput = $('file'), canvas = $('preview'), ctx = canvas.getContext('2d');
  const status = $('status'), bars = $('bars'), top1 = $('top1');

  // Draw a neutral placeholder on the preview canvas
  ctx.fillStyle = '#888'; ctx.globalAlpha = .12; ctx.fillRect(0, 0, 256, 256); ctx.globalAlpha = 1;

  TrafficSignCNN.loadModel('model/').then(m => {
    model = m;
    status.textContent = 'Model loaded. Choose an image to classify.';
  }).catch(err => {
    status.textContent = 'Could not load the model files: ' + err.message;
  });

  function classify(img) {
    // Show the image in the preview canvas (centered, square crop)
    const s = Math.min(img.width, img.height);
    const sx = (img.width - s) / 2, sy = (img.height - s) / 2;
    ctx.imageSmoothingQuality = 'high';
    ctx.clearRect(0, 0, 256, 256);
    ctx.drawImage(img, sx, sy, s, s, 0, 0, 256, 256);

    // Resize to 32x32 (what the model was trained on)
    const tmp = document.createElement('canvas'); tmp.width = tmp.height = IMG;
    const t = tmp.getContext('2d'); t.imageSmoothingQuality = 'high';
    t.drawImage(img, sx, sy, s, s, 0, 0, IMG, IMG);
    const px = t.getImageData(0, 0, IMG, IMG).data;
    const input = new Float32Array(IMG * IMG * 3);
    for (let i = 0, j = 0; i < px.length; i += 4) {
      input[j++] = px[i] / 255; input[j++] = px[i + 1] / 255; input[j++] = px[i + 2] / 255;
    }

    const probs = TrafficSignCNN.predictProbs(model, input);
    const top = TrafficSignCNN.topK(probs, model.names, 3);

    $('top1Name').textContent = top[0].name;
    $('top1Conf').textContent = (top[0].prob * 100).toFixed(1) + '% confidence';
    top1.hidden = false;
    bars.innerHTML = '';
    top.forEach(r => {
      const li = document.createElement('li');
      const pct = (r.prob * 100).toFixed(1);
      const name = document.createElement('span'); name.textContent = r.name;
      const val = document.createElement('span'); val.textContent = pct + '%';
      const track = document.createElement('div'); track.className = 'track';
      const fill = document.createElement('div'); fill.className = 'fill'; fill.style.width = pct + '%';
      track.appendChild(fill);
      li.append(name, val, track);
      bars.appendChild(li);
    });
    status.textContent = 'Top 3 predictions:';
  }

  function handle(file) {
    if (!file || !file.type.startsWith('image/')) return;
    if (!model) { status.textContent = 'Model is still loading, try again in a moment.'; return; }
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { classify(img); URL.revokeObjectURL(url); $('dropText').textContent = 'Click or drag another image'; };
    img.onerror = () => { status.textContent = 'That file could not be read as an image.'; };
    img.src = url;
  }

  fileInput.addEventListener('change', e => handle(e.target.files[0]));
  drop.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } });
  ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', e => handle(e.dataTransfer.files[0]));

  // Hide the "add link" placeholders once real links are filled in (edit href values in index.html)
  document.querySelectorAll('a.pending').forEach(a => {
    if (a.getAttribute('href') && a.getAttribute('href') !== '#') {
      a.classList.remove('pending');
      a.target = '_blank'; a.rel = 'noopener';
      a.textContent = a.getAttribute('href').replace(/^https?:\/\//, '');
    }
  });
})();
