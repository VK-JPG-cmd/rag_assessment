document.addEventListener('DOMContentLoaded', () => {
  const patientSelect = document.getElementById('patient-select');
  const systemStatus = document.getElementById('system-status');
  const statusText = document.getElementById('status-text');
  
  const pName = document.getElementById('p-name');
  const pAgeGender = document.getElementById('p-age-gender');
  const pBloodType = document.getElementById('p-blood-type');
  const pPcp = document.getElementById('p-pcp');
  const pBadge = document.getElementById('badge-patient-id');
  
  const allergiesList = document.getElementById('p-allergies-list');
  const diagnosesList = document.getElementById('p-diagnoses-list');
  const medsList = document.getElementById('p-meds-list');
  const encountersTimeline = document.getElementById('p-encounters-timeline');
  const quickPromptsContainer = document.getElementById('quick-prompts-container');
  
  const queryForm = document.getElementById('rag-query-form');
  const queryInput = document.getElementById('query-input');
  const categoryFilter = document.getElementById('category-filter');
  const topkFilter = document.getElementById('topk-filter');
  const submitBtn = document.getElementById('submit-btn');
  
  const resultsArea = document.getElementById('results-area');
  const emptyState = document.getElementById('empty-state');
  const ragAnswerContent = document.getElementById('rag-answer-content');
  const resModelBadge = document.getElementById('res-model-badge');
  const citationsBadgesList = document.getElementById('citations-badges-list');
  const evidenceListContainer = document.getElementById('evidence-list-container');
  const evidenceCountBadge = document.getElementById('evidence-count-badge');
  const copyBtn = document.getElementById('copy-btn');

  // Modal elements
  const openUploadModalBtn = document.getElementById('open-upload-modal-btn');
  const uploadModal = document.getElementById('upload-modal');
  const closeModalBtn = document.getElementById('close-modal-btn');
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const selectedFileInfo = document.getElementById('selected-file-info');
  const selectedFileName = document.getElementById('selected-file-name');
  const clearFileBtn = document.getElementById('clear-file-btn');
  const startUploadBtn = document.getElementById('start-upload-btn');
  const downloadTemplateBtn = document.getElementById('download-template-btn');
  const uploadStatus = document.getElementById('upload-status');

  let currentPatientId = null;
  let rawPatientsData = {};
  let selectedFile = null;

  const PATIENT_PROMPT_TEMPLATES = {
    "P101": [
      "Summarize the patient's active diagnoses and prescription history.",
      "Are there any documented drug allergies or adverse reactions?",
      "Review the patient's latest clinical notes and vitals.",
      "Summarize the laboratory findings over time."
    ]
  };

  // --- 1. INITIALIZATION ---
  async function init() {
    try {
      await refreshHealthAndPatients();
    } catch (err) {
      console.error("Initialization error:", err);
      statusText.innerText = "Connection Failed";
      systemStatus.style.background = "#ffe4e6";
      systemStatus.style.color = "#e11d48";
    }
  }

  async function refreshHealthAndPatients(selectTargetId = null) {
    // Check health
    const healthRes = await fetch('/api/health');
    if (healthRes.ok) {
      const health = await healthRes.json();
      statusText.innerText = `RAG Online (${health.chroma_docs_count} Records Indexed)`;
    }

    // Fetch patients list
    const res = await fetch('/api/patients');
    const data = await res.json();
    
    patientSelect.innerHTML = '';
    data.patients.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.patient_id;
      opt.textContent = `${p.patient_id} - ${p.full_name} (${p.gender}, Age ${p.age})`;
      patientSelect.appendChild(opt);
    });

    if (data.patients.length > 0) {
      const target = selectTargetId && data.patients.some(p => p.patient_id === selectTargetId)
        ? selectTargetId 
        : data.patients[0].patient_id;
      currentPatientId = target;
      patientSelect.value = currentPatientId;
      loadPatientDetails(currentPatientId);
    }
  }

  // --- 2. LOAD PATIENT DETAILS ---
  async function loadPatientDetails(patientId) {
    try {
      const res = await fetch(`/api/patient/${patientId}`);
      if (!res.ok) throw new Error("Failed to load patient");
      const p = await res.json();
      rawPatientsData[patientId] = p;

      // Demographics
      const demo = p.demographics || {};
      pName.innerText = demo.full_name || '--';
      pAgeGender.innerText = `${demo.age || 'N/A'} y/o ${demo.gender || 'N/A'}`;
      pBloodType.innerText = demo.blood_type || '--';
      pPcp.innerText = demo.primary_care_physician || '--';
      pBadge.innerText = patientId;

      // Allergies
      allergiesList.innerHTML = '';
      if (p.allergies && p.allergies.length > 0) {
        p.allergies.forEach(a => {
          const item = document.createElement('div');
          item.className = 'allergy-item';
          item.innerHTML = `
            <div class="substance"><i class="fa-solid fa-ban"></i> ${a.substance} (${a.severity || 'Documented'})</div>
            <div class="reaction">Reaction: ${a.reaction || 'Unspecified'} <span style="font-size:0.7rem; color:#94a3b8;">[${a.recorded_date || 'N/A'}]</span></div>
          `;
          allergiesList.appendChild(item);
        });
      } else {
        allergiesList.innerHTML = `<div style="font-size:0.8rem; color:#16a34a; font-weight:600;"><i class="fa-solid fa-circle-check"></i> No Known Drug Allergies (NKDA)</div>`;
      }

      // Diagnoses
      diagnosesList.innerHTML = '';
      if (p.active_diagnoses && p.active_diagnoses.length > 0) {
        p.active_diagnoses.forEach(d => {
          const tag = document.createElement('span');
          tag.className = 'diag-tag';
          tag.innerText = `[${d.code || 'ICD'}] ${d.description || 'Diagnosis'}`;
          diagnosesList.appendChild(tag);
        });
      } else {
        diagnosesList.innerHTML = `<span style="font-size:0.8rem; color:#94a3b8;">No active diagnoses documented.</span>`;
      }

      // Medications
      medsList.innerHTML = '';
      if (p.active_medications && p.active_medications.length > 0) {
        p.active_medications.forEach(m => {
          const card = document.createElement('div');
          card.className = 'med-card';
          card.innerHTML = `
            <div class="med-name">${m.name} <span class="med-dose">(${m.dosage || 'Prescribed'})</span></div>
            <div class="med-ind">Indication: ${m.indication || 'General'} • ${m.route || 'Oral'}</div>
          `;
          medsList.appendChild(card);
        });
      } else {
        medsList.innerHTML = `<span style="font-size:0.8rem; color:#94a3b8;">No active prescriptions recorded.</span>`;
      }

      // Encounters Timeline
      encountersTimeline.innerHTML = '';
      if (p.encounters && p.encounters.length > 0) {
        p.encounters.forEach(enc => {
          const item = document.createElement('div');
          item.className = 'timeline-item';
          item.innerHTML = `
            <div class="date">${enc.encounter_date || 'Date N/A'}</div>
            <div class="type">${enc.encounter_type || 'Encounter'}</div>
            <div class="desc">${enc.chief_complaint || 'Evaluation'}</div>
          `;
          encountersTimeline.appendChild(item);
        });
      } else {
        encountersTimeline.innerHTML = `<span style="font-size:0.8rem; color:#94a3b8;">No historical encounter notes.</span>`;
      }

      // Quick prompt templates
      quickPromptsContainer.innerHTML = '';
      const prompts = PATIENT_PROMPT_TEMPLATES[patientId] || [
        "Summarize all active conditions and medication list.",
        "List all documented allergies and adverse reactions.",
        "Review historical laboratory findings and encounter notes."
      ];

      prompts.forEach(text => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'prompt-pill-btn';
        btn.innerHTML = `<i class="fa-solid fa-arrow-right"></i> ${text}`;
        btn.onclick = () => {
          queryInput.value = text;
          queryForm.dispatchEvent(new Event('submit'));
        };
        quickPromptsContainer.appendChild(btn);
      });

    } catch (err) {
      console.error("Error loading patient file:", err);
    }
  }

  // --- 3. PATIENT SELECTION LISTENER ---
  patientSelect.addEventListener('change', (e) => {
    currentPatientId = e.target.value;
    loadPatientDetails(currentPatientId);
  });

  // --- 4. RAG QUERY SUBMISSION ---
  queryForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query || !currentPatientId) return;

    submitBtn.disabled = true;
    submitBtn.innerHTML = `<span class="loading-spinner"></span> Retrieving...`;

    try {
      const payload = {
        patient_id: currentPatientId,
        query: query,
        top_k: parseInt(topkFilter.value, 10),
        category: categoryFilter.value || null
      };

      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}: ${res.statusText}`);
      }

      const data = await res.json();
      displayRAGResults(data);

    } catch (err) {
      console.error("RAG Query failed:", err);
      alert(`Query failed: ${err.message}`);
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = `<i class="fa-solid fa-paper-plane"></i> Ask RAG`;
    }
  });

  // --- 5. RENDER RESULTS & CITATIONS ---
  function displayRAGResults(data) {
    emptyState.style.display = 'none';
    resultsArea.style.display = 'flex';

    // Render Answer Markdown
    ragAnswerContent.innerHTML = marked.parse(data.answer);
    resModelBadge.innerText = data.model_used || "Clinical Generator";

    // Render Citations
    citationsBadgesList.innerHTML = '';
    if (data.citations && data.citations.length > 0) {
      data.citations.forEach(c => {
        const pill = document.createElement('div');
        pill.className = 'citation-pill';
        const matchPct = Math.round((c.similarity_score || 0) * 100);
        pill.innerHTML = `
          <span class="src-tag"><i class="fa-solid fa-bookmark"></i> ${c.source_id}</span>
          <span class="date-tag">(${c.date})</span>
          <span class="match-tag">${matchPct}% Match</span>
        `;
        citationsBadgesList.appendChild(pill);
      });
    } else {
      citationsBadgesList.innerHTML = `<span style="font-size:0.8rem; color:#94a3b8;">No direct citations referenced.</span>`;
    }

    // Render Evidence Context Chunks
    evidenceListContainer.innerHTML = '';
    const evidence = data.retrieved_evidence || [];
    evidenceCountBadge.innerText = `${evidence.length} Chunks`;

    evidence.forEach((ev, i) => {
      const meta = ev.metadata || {};
      const score = Math.round((ev.similarity_score || 0) * 100);
      const card = document.createElement('div');
      card.className = 'evidence-item';
      card.innerHTML = `
        <div class="evidence-header">
          <div class="title">
            <i class="fa-solid fa-file-lines" style="color:var(--primary);"></i>
            <strong>[#${i+1}] ${meta.record_type || 'Record'}</strong>
            <span style="color:var(--text-muted); font-size:0.75rem;">(${meta.encounter_date || 'N/A'})</span>
          </div>
          <span class="score">${score}% Relevance</span>
        </div>
        <pre class="evidence-text">${escapeHtml(ev.text)}</pre>
      `;
      evidenceListContainer.appendChild(card);
    });

    resultsArea.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // --- 6. UPLOAD DATASET MODAL LOGIC ---
  openUploadModalBtn.addEventListener('click', () => {
    uploadModal.style.display = 'flex';
    resetUploadState();
  });

  closeModalBtn.addEventListener('click', () => {
    uploadModal.style.display = 'none';
  });

  uploadModal.addEventListener('click', (e) => {
    if (e.target === uploadModal) {
      uploadModal.style.display = 'none';
    }
  });

  dropZone.addEventListener('click', () => {
    fileInput.click();
  });

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (ext !== 'json' && ext !== 'csv') {
      showUploadStatus("Please select a valid .json or .csv file", false);
      return;
    }
    selectedFile = file;
    selectedFileName.innerText = `${file.name} (${Math.round(file.size / 1024)} KB)`;
    selectedFileInfo.style.display = 'flex';
    dropZone.style.display = 'none';
    startUploadBtn.disabled = false;
    uploadStatus.style.display = 'none';
  }

  clearFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUploadState();
  });

  function resetUploadState() {
    selectedFile = null;
    fileInput.value = '';
    selectedFileInfo.style.display = 'none';
    dropZone.style.display = 'block';
    startUploadBtn.disabled = true;
    startUploadBtn.innerHTML = `<i class="fa-solid fa-upload"></i> Ingest into Vector DB`;
    uploadStatus.style.display = 'none';
  }

  startUploadBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    startUploadBtn.disabled = true;
    startUploadBtn.innerHTML = `<span class="loading-spinner"></span> Ingesting into Vector DB...`;
    showUploadStatus("Uploading file and generating vector embeddings...", null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch('/api/upload-dataset', {
        method: 'POST',
        body: formData
      });

      const result = await res.json();
      if (!res.ok) {
        throw new Error(result.detail || "Upload failed");
      }

      showUploadStatus(`Success! ${result.message} Total indexed documents: ${result.total_documents_indexed}`, true);
      
      // Auto-select the first newly added patient if available
      const newTargetId = (result.updated_patient_ids && result.updated_patient_ids.length > 0)
        ? result.updated_patient_ids[0]
        : null;

      setTimeout(async () => {
        await refreshHealthAndPatients(newTargetId);
        uploadModal.style.display = 'none';
      }, 1500);

    } catch (err) {
      console.error("Upload error:", err);
      showUploadStatus(`Error: ${err.message}`, false);
      startUploadBtn.disabled = false;
      startUploadBtn.innerHTML = `<i class="fa-solid fa-upload"></i> Try Again`;
    }
  });

  downloadTemplateBtn.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/sample-template');
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'sample_patient_template.json';
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Download template error:", err);
    }
  });

  function showUploadStatus(msg, isSuccess) {
    uploadStatus.style.display = 'block';
    uploadStatus.innerText = msg;
    uploadStatus.className = 'upload-status-box';
    if (isSuccess === true) {
      uploadStatus.classList.add('success');
    } else if (isSuccess === false) {
      uploadStatus.classList.add('error');
    }
  }

  // Copy to clipboard helper
  copyBtn.addEventListener('click', () => {
    const text = ragAnswerContent.innerText;
    navigator.clipboard.writeText(text).then(() => {
      copyBtn.innerHTML = `<i class="fa-solid fa-check"></i>`;
      setTimeout(() => {
        copyBtn.innerHTML = `<i class="fa-regular fa-copy"></i>`;
      }, 2000);
    });
  });

  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // Start app
  init();
});
