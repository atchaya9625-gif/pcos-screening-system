document.getElementById('predict-form').addEventListener('submit', async function (e) {
    e.preventDefault();
    const formData = new FormData(this);

    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = '<p>Predicting...</p>';

    const res = await fetch('/predict', {
        method: 'POST',
        body: formData
    });
    const data = await res.json();

    if (data.error) {
        resultDiv.innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
        return;
    }

    let modalityNote = data.image_risk_score !== null
        ? `<p><small>Clinical risk: ${(data.clinical_risk_score * 100).toFixed(1)}% | Image risk: ${(data.image_risk_score * 100).toFixed(1)}% | Combined (fused): ${(data.risk_score * 100).toFixed(1)}%</small></p>`
        : `<p><small>Based on clinical data only (no ultrasound image uploaded). Upload an image for a multimodal prediction.</small></p>`;

    let html = `
        <h3>Result: ${data.prediction}</h3>
        <p><strong>Risk Score:</strong> ${(data.risk_score * 100).toFixed(1)}%</p>
        ${modalityNote}
        <p><strong>Phenotype:</strong> ${data.phenotype}</p>
        <p>${data.phenotype_description}</p>
    `;

    // Explainability section (Stage 2)
    if (data.shap_chart || data.gradcam_chart) {
        html += `<h3>Why this prediction? (Explainability)</h3><div class="explain-grid">`;
        if (data.shap_chart) {
            html += `
                <div class="explain-card">
                    <p class="explain-label">SHAP — Clinical Feature Contributions</p>
                    <img src="${data.shap_chart}" alt="SHAP explanation chart" class="explain-img">
                </div>
            `;
        }
        if (data.gradcam_chart) {
            html += `
                <div class="explain-card">
                    <p class="explain-label">Grad-CAM — Ultrasound Focus Area</p>
                    <img src="${data.gradcam_chart}" alt="Grad-CAM heatmap" class="explain-img">
                </div>
            `;
        }
        html += `</div>`;
    }

    if (data.recommendations) {
        html += `<h3>Your Personalized Plan</h3><div class="nudge-grid">`;
        const slots = [
            ['morning', '🌅 Morning'],
            ['afternoon', '☀️ Afternoon'],
            ['evening', '🌇 Evening'],
            ['night', '🌙 Night']
        ];
        slots.forEach(([key, label]) => {
            const slot = data.recommendations[key];
            html += `
                <div class="nudge-card">
                    <h3>${label}</h3>
                    <p><strong>Food:</strong> ${slot.food}</p>
                    <p><strong>Exercise:</strong> ${slot.exercise}</p>
                </div>
            `;
        });
        html += `</div>`;
    }

    resultDiv.innerHTML = html;
});
