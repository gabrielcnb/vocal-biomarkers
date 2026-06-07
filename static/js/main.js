// Vocal Biomarkers: Frontend Logic

document.addEventListener("DOMContentLoaded", () => {
    // Tab switching
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
            btn.classList.add("active");
            document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
        });
    });

    // Upload form
    const uploadForm = document.getElementById("upload-form");
    if (uploadForm) {
        uploadForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById("audio-file");
            if (!fileInput.files.length) {
                alert("Selecione um arquivo de \u00e1udio.");
                return;
            }

            const formData = new FormData();
            formData.append("audio", fileInput.files[0]);

            const btn = uploadForm.querySelector("button[type=submit]");
            setLoading(btn, true);

            try {
                const resp = await fetch("/api/predict/upload", {
                    method: "POST",
                    body: formData,
                });
                const data = await resp.json();
                if (resp.ok) {
                    showResult(data);
                } else {
                    showError(data.error || "Erro desconhecido");
                }
            } catch (err) {
                showError("Erro de conex\u00e3o com o servidor.");
            } finally {
                setLoading(btn, false);
            }
        });
    }

    // Manual form
    const manualForm = document.getElementById("manual-form");
    if (manualForm) {
        manualForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const formData = new FormData(manualForm);
            const features = {};
            for (const [key, val] of formData.entries()) {
                features[key] = parseFloat(val) || 0;
            }

            const btn = manualForm.querySelector("button[type=submit]");
            setLoading(btn, true);

            try {
                const resp = await fetch("/api/predict/manual", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(features),
                });
                const data = await resp.json();
                if (resp.ok) {
                    showResult(data);
                } else {
                    showError(data.error || "Erro desconhecido");
                }
            } catch (err) {
                showError("Erro de conex\u00e3o com o servidor.");
            } finally {
                setLoading(btn, false);
            }
        });
    }
});

function setLoading(btn, loading) {
    if (loading) {
        btn.disabled = true;
        btn.dataset.origText = btn.textContent;
        btn.innerHTML = '<span class="loading"></span> Analisando...';
    } else {
        btn.disabled = false;
        btn.textContent = btn.dataset.origText || "Analisar";
    }
}

function showResult(data) {
    const container = document.getElementById("results");
    const content = document.getElementById("result-content");
    container.style.display = "block";

    const isParkinson = data.prediction === 1;
    const confidence = (data.confidence * 100).toFixed(1);
    const probPD = (data.prob_parkinson * 100).toFixed(1);
    const probHealthy = (data.prob_healthy * 100).toFixed(1);
    const barColor = isParkinson ? "#c0392b" : "#2e8b6e";

    let warningsHtml = "";
    if (data.warnings && data.warnings.length > 0) {
        warningsHtml = `
            <div class="alert alert-warning" style="margin-top: 12px;">
                <strong>Avisos:</strong>
                <ul>${data.warnings.map(w => `<li>${w}</li>`).join("")}</ul>
            </div>
        `;
    }

    let missingHtml = "";
    if (data.missing_features && data.missing_features.length > 0) {
        missingHtml = `
            <div class="alert alert-warning" style="margin-top: 12px;">
                <strong>Features ausentes (valor padr\u00e3o utilizado):</strong> ${data.missing_features.join(", ")}
            </div>
        `;
    }

    content.innerHTML = `
        <p class="${isParkinson ? 'result-positive' : 'result-negative'}">
            ${isParkinson ? "Indicadores de Parkinson Detectados" : "Nenhum Indicador de Parkinson Detectado"}
        </p>

        <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${confidence}%; background: ${barColor};">
                Confian\u00e7a: ${confidence}%
            </div>
        </div>

        <div class="prob-grid">
            <div class="prob-item" style="background: ${isParkinson ? '#fdf2f0' : '#fff'};">
                <div class="prob-value" style="color: #c0392b;">${probPD}%</div>
                <div class="prob-label">Probabilidade Parkinson</div>
            </div>
            <div class="prob-item" style="background: ${!isParkinson ? '#e6f5f0' : '#fff'};">
                <div class="prob-value" style="color: #2e8b6e;">${probHealthy}%</div>
                <div class="prob-label">Probabilidade Saud\u00e1vel</div>
            </div>
        </div>

        ${warningsHtml}
        ${missingHtml}
    `;

    container.scrollIntoView({ behavior: "smooth", block: "start" });
}

function showError(message) {
    const container = document.getElementById("results");
    const content = document.getElementById("result-content");
    container.style.display = "block";
    content.innerHTML = `
        <div class="alert alert-warning">
            <strong>Erro:</strong> ${message}
        </div>
    `;
    container.scrollIntoView({ behavior: "smooth", block: "start" });
}
