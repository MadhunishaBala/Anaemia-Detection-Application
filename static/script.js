function previewImage(input, zoneId, previewId) {
    const zone    = document.getElementById(zoneId);
    const preview = document.getElementById(previewId);
    if (input.files && input.files[0]) {
      const reader = new FileReader();
      reader.onload = e => { preview.src = e.target.result; };
      reader.readAsDataURL(input.files[0]);
      zone.classList.add('has-file');
    }
  }

  async function runPrediction() {
    const palmFile = document.getElementById('palmInput').files[0];
    const nailFile = document.getElementById('nailInput').files[0];
    const age      = document.getElementById('ageInput').value;
    const gender   = document.getElementById('genderInput').value;
    const btn      = document.getElementById('predictBtn');
    const errMsg   = document.getElementById('errorMsg');
    const resCard  = document.getElementById('resultCard');

    errMsg.style.display  = 'none';
    resCard.style.display = 'none';
    document.getElementById('regressionCard').style.display  = 'none';
    document.getElementById('attentionCard').style.display   = 'none';
    document.getElementById('conflictNote').style.display    = 'none';  // ← reset

    if (!palmFile || !nailFile || !age || gender === '') {
      errMsg.textContent   = '⚠ Please provide both images, age, and gender.';
      errMsg.style.display = 'block'; return;
    }

    btn.classList.add('loading');
    btn.disabled = true;

    const fd = new FormData();
    fd.append('palm',   palmFile);
    fd.append('nail',   nailFile);
    fd.append('age',    age);
    fd.append('gender', gender);

    try {
      const res  = await fetch('/predict', { method: 'POST', body: fd });
      const data = await res.json();

      if (data.error) throw new Error(data.error);

      // Classification card
      const isAnemic = data.label === 'Anemic';
      resCard.className = 'result-card ' + (isAnemic ? 'anemic' : 'non-anemic');
      document.getElementById('resultLabel').textContent = data.label;
      document.getElementById('resultBadge').textContent = isAnemic ? 'Positive' : 'Negative';
      document.getElementById('confPct').textContent     = data.confidence + '%';
      resCard.style.display = 'block';
      setTimeout(() => {
          document.getElementById('barFill').style.width = data.confidence + '%';
      }, 50);

      // Regression card
      const regCard    = document.getElementById('regressionCard');
      const isAnemicReg = data.severity !== 'Non-Anemic';
      regCard.className = 'result-card ' + (isAnemicReg ? 'anemic' : 'non-anemic');
      document.getElementById('hbValue').textContent       = data.hb_level + ' g/dL';
      document.getElementById('severityValue').textContent = data.severity;
      regCard.style.display = 'block';

      // Attention map card             
      if (data.attention_map) {
        document.getElementById('attentionImg').src          = 'data:image/png;base64,' + data.attention_map;
        document.getElementById('attentionCard').style.display = 'block';
      }

      // Conflict note
      if (data.conflict_note) {
          document.getElementById('conflictNote').textContent = data.conflict_note;
          document.getElementById('conflictNote').style.display = 'block';
      } else {
          document.getElementById('conflictNote').style.display = 'none';
      }

    } catch (err) {
      errMsg.textContent   = '✗ ' + err.message;
      errMsg.style.display = 'block';
    } finally {
      btn.classList.remove('loading');
      btn.disabled = false;
    }
}