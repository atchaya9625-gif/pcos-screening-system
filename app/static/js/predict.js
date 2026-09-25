// --- Preset Auto-Fill for Demo / Viva ---
const presetPCOS = {
    follicle_r: 16,
    follicle_l: 14,
    cycle: 'I',
    cycle_length: 42,
    amh: 7.5,
    prl: 26.0,
    fsh_lh: 0.65,
    weight_gain: '1',
    hair_growth: '1',
    skin_darkening: '1',
    pimples: '1',
    fast_food: '1'
};

const presetNormal = {
    follicle_r: 6,
    follicle_l: 7,
    cycle: 'R',
    cycle_length: 28,
    amh: 2.8,
    prl: 14.0,
    fsh_lh: 1.45,
    weight_gain: '0',
    hair_growth: '0',
    skin_darkening: '0',
    pimples: '0',
    fast_food: '0'
};

function fillPreset(data) {
    document.getElementById('f_follicle_r').value = data.follicle_r;
    document.getElementById('f_follicle_l').value = data.follicle_l;
    document.getElementById('f_cycle').value = data.cycle;
    document.getElementById('f_cycle_length').value = data.cycle_length;
    document.getElementById('f_amh').value = data.amh;
    document.getElementById('f_prl').value = data.prl;
    document.getElementById('f_fsh_lh').value = data.fsh_lh;
    document.getElementById('f_weight_gain').value = data.weight_gain;
    document.getElementById('f_hair_growth').value = data.hair_growth;
    document.getElementById('f_skin_darkening').value = data.skin_darkening;
    document.getElementById('f_pimples').value = data.pimples;
    document.getElementById('f_fast_food').value = data.fast_food;
}

const btnPcos = document.getElementById('btn-preset-pcos');
const btnNormal = document.getElementById('btn-preset-normal');
if (btnPcos) btnPcos.addEventListener('click', () => fillPreset(presetPCOS));
if (btnNormal) btnNormal.addEventListener('click', () => fillPreset(presetNormal));

// --- Live Ultrasound Image Preview ---
const ultrasoundInput = document.getElementById('ultrasound-input');
const previewContainer = document.getElementById('image-preview-container');
const previewImg = document.getElementById('image-preview');
const previewFilename = document.getElementById('preview-filename');
const removeImageBtn = document.getElementById('remove-image-btn');

if (ultrasoundInput) {
    ultrasoundInput.addEventListener('change', function () {
        const file = this.files && this.files[0];
        if (file) {
            if (!file.type.startsWith('image/')) {
                alert('Please select a valid image file (.jpg, .jpeg, .png).');
                this.value = '';
                previewContainer.style.display = 'none';
                return;
            }
            const reader = new FileReader();
            reader.onload = function (e) {
                previewImg.src = e.target.result;
                previewFilename.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
                previewContainer.style.display = 'flex';
            };
            reader.readAsDataURL(file);
        } else {
            previewContainer.style.display = 'none';
        }
    });
}

if (removeImageBtn) {
    removeImageBtn.addEventListener('click', function () {
        ultrasoundInput.value = '';
        previewContainer.style.display = 'none';
        previewImg.src = '#';
    });
}

// --- Form Submission & Multimodal Analysis ---
document.getElementById('predict-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    // Client-side Validation
    const follicleR = parseFloat(document.getElementById('f_follicle_r').value);
    const follicleL = parseFloat(document.getElementById('f_follicle_l').value);
    const cycleLength = parseFloat(document.getElementById('f_cycle_length').value);
    const amh = parseFloat(document.getElementById('f_amh').value);
    const prl = parseFloat(document.getElementById('f_prl').value);
    const fshLh = parseFloat(document.getElementById('f_fsh_lh').value);

    if (isNaN(follicleR) || follicleR < 0 || isNaN(follicleL) || follicleL < 0) {
        alert('Follicle counts cannot be negative numbers.');
        return;
    }
    if (isNaN(cycleLength) || cycleLength <= 0) {
        alert('Cycle length must be a positive number of days.');
        return;
    }
    if (isNaN(amh) || amh < 0 || isNaN(prl) || prl < 0 || isNaN(fshLh) || fshLh <= 0) {
        alert('Hormone values must be valid positive numbers.');
        return;
    }

    const formData = new FormData(this);
    const submitBtn = document.getElementById('submit-btn');
    const btnText = document.getElementById('btn-text');
    const btnSpinner = document.getElementById('btn-spinner');
    const resultDiv = document.getElementById('result');

    // UI Loading state
    submitBtn.disabled = true;
    btnText.textContent = "Analyzing Multimodal Data...";
    btnSpinner.style.display = "inline-block";
    resultDiv.innerHTML = `
        <div style="text-align: center; padding: 25px;">
            <div class="spinner" style="width: 28px; height: 28px; border-color: rgba(214, 51, 132, 0.2); border-top-color: #d63384; margin: 0 auto 12px auto;"></div>
            <p style="color: #a3336b; font-weight: 600;">Running AI screening & computing SHAP/Grad-CAM explanations...</p>
            <p style="color: #7e6c7c; font-size: 0.85rem;">This may take 4-8 seconds for multimodal feature attribution.</p>
        </div>
    `;

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();

        if (data.error) {
            resultDiv.innerHTML = `<div class="result-header-box" style="background:#fee2e2; border-left: 5px solid #dc2626;"><p style="color:#991b1b; font-weight:600;">⚠️ Error: ${data.error}</p></div>`;
            return;
        }

        const isPositive = data.prediction === 'PCOS Detected';
        const boxClass = isPositive ? 'result-positive' : 'result-negative';
        const icon = isPositive ? '⚠️' : '✅';

        let modalityNote = data.image_risk_score !== null
            ? `<p style="margin-top: 6px; font-size: 0.9rem; color: #554353;"><strong>Multimodal Fusion Breakdown:</strong> Clinical Model: ${(data.clinical_risk_score * 100).toFixed(1)}% | Ultrasound CNN: ${(data.image_risk_score * 100).toFixed(1)}% | <strong>Combined Risk: ${(data.risk_score * 100).toFixed(1)}%</strong></p>`
            : `<p style="margin-top: 6px; font-size: 0.88rem; color: #6b5c68;"><em>Prediction based on 12 clinical biomarkers. (Upload an ultrasound scan for multimodal fusion).</em></p>`;

        let html = `
            <div class="result-header-box ${boxClass}">
                <h2 style="color: ${isPositive ? '#a3336b' : '#2b8a3e'}; font-size: 1.4rem; margin-bottom: 6px;">${icon} ${data.prediction}</h2>
                <p style="font-size: 1.05rem;"><strong>Screening Risk Score:</strong> <span style="font-weight:700;">${(data.risk_score * 100).toFixed(1)}%</span></p>
                ${modalityNote}
                <p style="margin-top: 8px;"><strong>Identified Phenotype:</strong> <span class="badge-tag">${data.phenotype}</span></p>
                <p style="font-size: 0.92rem; color: #4a3848;">${data.phenotype_description}</p>
            </div>
        `;

        // Explainability section (SHAP + Grad-CAM)
        if (data.shap_chart || data.gradcam_chart) {
            html += `<h3>🔍 Why this prediction? (Explainable AI Insights)</h3><div class="explain-grid">`;
            if (data.shap_chart) {
                html += `
                    <div class="explain-card">
                        <p class="explain-label">SHAP — Clinical Feature Impact</p>
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

        // Recommendations
        if (data.recommendations) {
            html += `
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:25px; flex-wrap:wrap; gap:10px;">
                    <h3>🥗 Personalized Continuous Lifestyle Plan</h3>
                    <a href="/dashboard" class="btn btn-secondary" style="padding: 6px 14px; font-size: 13px;">Open in Dashboard ➔</a>
                </div>
                <div class="nudge-grid">
            `;
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
                        <p><strong>Exercise / Yoga:</strong> ${slot.exercise}</p>
                    </div>
                `;
            });
            html += `</div>`;
        }

        resultDiv.innerHTML = html;
        resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch (err) {
        resultDiv.innerHTML = `<div class="result-header-box" style="background:#fee2e2; border-left: 5px solid #dc2626;"><p style="color:#991b1b; font-weight:600;">⚠️ Network error: ${err.message}</p></div>`;
    } finally {
        submitBtn.disabled = false;
        btnText.textContent = "Run Multimodal Screening";
        btnSpinner.style.display = "none";
    }
});
