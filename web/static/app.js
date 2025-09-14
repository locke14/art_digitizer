(() => {
  const $ = (sel, ctx=document) => ctx.querySelector(sel);
  const $$ = (sel, ctx=document) => Array.from(ctx.querySelectorAll(sel));

  const form = $('#form');
  const input = $('#file-input');
  const dz = $('.dropzone');
  const preview = $('.preview-grid');
  const submit = $('#submit');
  const themeBtn = $('#theme-toggle');

  // Drag and drop
  ['dragenter','dragover'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('drag'); }));
  ;['dragleave','drop'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('drag'); }));
  dz.addEventListener('drop', e => {
    const files = Array.from(e.dataTransfer.files || []);
    if (files.length) {
      // Append files to input via DataTransfer
      const dt = new DataTransfer();
      Array.from(input.files || []).forEach(f => dt.items.add(f));
      files.forEach(f => dt.items.add(f));
      input.files = dt.files;
      renderPreview();
    }
  });
  dz.addEventListener('click', () => input?.click());
  input?.addEventListener('change', renderPreview);

  function renderPreview() {
    if (!preview) return;
    preview.innerHTML = '';
    const files = Array.from(input.files || []);
    files.slice(0, 24).forEach(f => {
      if (!f.type.startsWith('image/')) return;
      const div = document.createElement('div');
      div.className = 'thumb';
      const img = document.createElement('img');
      const cap = document.createElement('div');
      cap.className = 'cap';
      cap.textContent = f.name;
      div.appendChild(img);
      div.appendChild(cap);
      preview.appendChild(div);
      const reader = new FileReader();
      reader.onload = e => img.src = e.target.result;
      reader.readAsDataURL(f);
    });
  }

  // Submit UX
  form?.addEventListener('submit', () => {
    submit.disabled = true;
    submit.innerHTML = '<span class="spinner"></span> Processing...';
  });

  // Theme persistence + toggle
  const prefersLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
  const savedTheme = localStorage.getItem('theme');
  const initial = savedTheme || (prefersLight ? 'light' : 'dark');
  document.documentElement.setAttribute('data-theme', initial);
  updateThemeButton(initial);

  themeBtn?.addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    updateThemeButton(next);
  });

  function updateThemeButton(theme) {
    if (!themeBtn) return;
    themeBtn.textContent = theme === 'dark' ? 'Switch to Light' : 'Switch to Dark';
  }
})();
