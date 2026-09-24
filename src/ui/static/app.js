/**
 * Finance RAG Dashboard — Interactive Client Application
 */

document.addEventListener("DOMContentLoaded", () => {
  // State
  let currentResult = null;
  let activeStep = "decomposer";

  // Elements
  const queryInput = document.getElementById("queryInput");
  const submitBtn = document.getElementById("submitBtn");
  const clearBtn = document.getElementById("clearBtn");
  const sampleChips = document.getElementById("sampleChips");
  const reloadPipelineBtn = document.getElementById("reloadPipelineBtn");

  const emptyState = document.getElementById("emptyState");
  const loadingState = document.getElementById("loadingState");
  const resultsGrid = document.getElementById("resultsGrid");
  const kpiBar = document.getElementById("kpiBar");

  // KPI elements
  const kpiConfidence = document.getElementById("kpiConfidence");
  const kpiGuard = document.getElementById("kpiGuard");
  const kpiChunks = document.getElementById("kpiChunks");
  const kpiLatency = document.getElementById("kpiLatency");
  const kpiCost = document.getElementById("kpiCost");

  // Answer elements
  const answerBody = document.getElementById("answerBody");
  const metricsSection = document.getElementById("metricsSection");
  const metricsGrid = document.getElementById("metricsGrid");
  const citationsSection = document.getElementById("citationsSection");
  const citationsList = document.getElementById("citationsList");
  const copyAnswerBtn = document.getElementById("copyAnswerBtn");
  const exportReportBtn = document.getElementById("exportReportBtn");

  // Inspector elements
  const cycleBadge = document.getElementById("cycleBadge");
  const stepperNav = document.getElementById("stepperNav");
  const inspectorBody = document.getElementById("inspectorBody");

  // Modal elements
  const chunkModal = document.getElementById("chunkModal");
  const modalTitle = document.getElementById("modalTitle");
  const modalBody = document.getElementById("modalBody");
  const modalClose = document.getElementById("modalClose");

  // Status badges
  const statusPill = document.getElementById("statusPill");
  const statusText = document.getElementById("statusText");
  const modelBadge = document.getElementById("modelBadge");

  // Check health on boot
  checkHealth();
  loadSampleQueries();

  // ── Event Handlers ──────────────────────────────────────────────────
  queryInput.addEventListener("input", () => {
    clearBtn.style.display = queryInput.value ? "block" : "none";
  });

  clearBtn.addEventListener("click", () => {
    queryInput.value = "";
    clearBtn.style.display = "none";
    queryInput.focus();
  });

  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      runCurrentQuery();
    }
  });

  submitBtn.addEventListener("click", runCurrentQuery);

  // Stepper tab switching
  stepperNav.querySelectorAll(".step-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      stepperNav.querySelectorAll(".step-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeStep = btn.getAttribute("data-step");
      renderInspectorStep(activeStep);
    });
  });

  // Modal close
  modalClose.addEventListener("click", () => {
    chunkModal.style.display = "none";
  });
  window.addEventListener("click", (e) => {
    if (e.target === chunkModal) {
      chunkModal.style.display = "none";
    }
  });

  // Copy Answer
  copyAnswerBtn.addEventListener("click", () => {
    if (!currentResult || !currentResult.final_answer) return;
    navigator.clipboard.writeText(currentResult.final_answer).then(() => {
      const origText = copyAnswerBtn.querySelector("span").textContent;
      copyAnswerBtn.querySelector("span").textContent = "Copied!";
      setTimeout(() => {
        copyAnswerBtn.querySelector("span").textContent = origText;
      }, 1500);
    });
  });

  // Export Analysis Report
  if (exportReportBtn) {
    exportReportBtn.addEventListener("click", () => {
      if (!currentResult || !currentResult.report_file) return;
      const downloadUrl = `/api/reports/${encodeURIComponent(currentResult.report_file)}`;
      const a = document.createElement("a");
      a.href = downloadUrl;
      a.download = currentResult.report_file;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      const textSpan = exportReportBtn.querySelector("span");
      const orig = textSpan.textContent;
      textSpan.textContent = "Downloaded!";
      setTimeout(() => {
        textSpan.textContent = orig;
      }, 1500);
    });
  }

  // Reload Pipeline Button
  if (reloadPipelineBtn) {
    reloadPipelineBtn.addEventListener("click", async () => {
      reloadPipelineBtn.disabled = true;
      const textSpan = reloadPipelineBtn.querySelector("span");
      const origText = textSpan.textContent;
      textSpan.textContent = "Reloading...";
      try {
        const res = await fetch("/api/reload", { method: "POST" });
        const data = await res.json();
        if (data.status === "reloaded") {
          textSpan.textContent = "Reloaded!";
          setTimeout(() => {
            textSpan.textContent = origText;
            reloadPipelineBtn.disabled = false;
          }, 1500);
        }
      } catch (e) {
        alert("Failed to reload pipeline: " + e.message);
        textSpan.textContent = origText;
        reloadPipelineBtn.disabled = false;
      }
    });
  }

  // ── Functions ────────────────────────────────────────────────────────

  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      if (data.status === "online") {
        statusPill.className = "status-pill status-online";
        statusText.textContent = "CRAG Engine Active";
        modelBadge.textContent = data.model || "GPT-4o";
      }
    } catch (e) {
      statusPill.className = "status-pill badge-rose";
      statusText.textContent = "Offline";
    }
  }

  async function loadSampleQueries() {
    try {
      const res = await fetch("/api/sample-queries");
      const data = await res.json();
      if (data.samples && data.samples.length > 0) {
        sampleChips.innerHTML = "";
        data.samples.slice(0, 6).forEach((s) => {
          const btn = document.createElement("button");
          btn.className = "chip";
          btn.textContent = s.question;
          btn.title = `Difficulty: ${s.difficulty} | Category: ${s.category}`;
          btn.addEventListener("click", () => {
            queryInput.value = s.question;
            clearBtn.style.display = "block";
            runCurrentQuery();
          });
          sampleChips.appendChild(btn);
        });
      }
    } catch (e) {
      sampleChips.querySelectorAll(".chip").forEach((btn) => {
        btn.addEventListener("click", () => {
          queryInput.value = btn.getAttribute("data-query");
          clearBtn.style.display = "block";
          runCurrentQuery();
        });
      });
    }
  }

  async function runCurrentQuery() {
    const query = queryInput.value.trim();
    if (!query) return;

    // Set UI to loading state
    emptyState.style.display = "none";
    resultsGrid.style.display = "none";
    kpiBar.style.display = "none";
    loadingState.style.display = "flex";
    submitBtn.disabled = true;

    // Simulate animated loading progress
    const stepOrder = ["decompose", "retrieve", "rerank", "grade", "generate", "guard"];
    let stepIndex = 0;
    const interval = setInterval(() => {
      document.querySelectorAll(".step-chip").forEach((c) => c.classList.remove("active"));
      if (stepIndex < stepOrder.length) {
        const chip = document.getElementById(`lstep-${stepOrder[stepIndex]}`);
        if (chip) chip.classList.add("active");
        stepIndex++;
      }
    }, 1200);

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      clearInterval(interval);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Query failed");
      }

      currentResult = data;
      renderResults(data);

    } catch (err) {
      clearInterval(interval);
      alert(`Error running query: ${err.message}`);
      emptyState.style.display = "flex";
    } finally {
      loadingState.style.display = "none";
      submitBtn.disabled = false;
    }
  }

  function renderResults(data) {
    kpiBar.style.display = "grid";
    resultsGrid.style.display = "grid";

    // Update Stepper Timers (Issue 2 & 6)
    updateStepperTimers(data.pipeline_trace || []);

    // Confidence
    const conf = (data.confidence || "MEDIUM").toUpperCase();
    kpiConfidence.textContent = conf;
    if (conf === "HIGH") kpiConfidence.className = "kpi-value text-accent";
    else if (conf === "INSUFFICIENT" || conf === "LOW") kpiConfidence.className = "kpi-value badge-amber";
    else kpiConfidence.className = "kpi-value";

    // Guardrail
    const guard = data.hallucination_check || "pass";
    if (guard === "pass") {
      kpiGuard.innerHTML = `<span style="color: var(--accent-emerald);">PASSED</span>`;
    } else {
      kpiGuard.innerHTML = `<span style="color: var(--accent-rose);">UNVERIFIED</span>`;
    }

    // Chunks count (look at reranker or search_results)
    const rerankTrace = findTraceNode(data.pipeline_trace, "reranker");
    const chunkCount = rerankTrace && rerankTrace.chunks ? rerankTrace.chunks.length : (data.search_results ? data.search_results.length : 0);
    kpiChunks.textContent = `${chunkCount} Chunks`;

    // Latency & Cost
    kpiLatency.textContent = `${data.latency_seconds || 0}s`;
    kpiCost.textContent = `$${(data.cost_accumulated || 0).toFixed(4)}`;

    // Export Report Button
    if (exportReportBtn) {
      if (data.report_file) {
        exportReportBtn.style.display = "inline-flex";
        exportReportBtn.title = `Download analysis report (${data.report_file})`;
      } else {
        exportReportBtn.style.display = "none";
      }
    }

    // Cycle Badge
    cycleBadge.textContent = data.cycle_count > 0 ? `Cycle ${data.cycle_count} (Rewritten)` : "Cycle 0 (Direct)";
    cycleBadge.className = data.cycle_count > 0 ? "badge badge-amber" : "badge badge-cyan";

    // 2. Answer Body
    let answerText = data.final_answer || "";

    if (window.marked) {
      answerBody.innerHTML = window.marked.parse(answerText);
    } else {
      answerBody.textContent = answerText;
    }

    // Convert citations [1], [2] to interactive tags
    answerBody.innerHTML = answerBody.innerHTML.replace(/\[(\d+)\]/g, (match, p1) => {
      return `<button class="citation-tag" data-cite="${p1}">[${p1}]</button>`;
    });

    answerBody.querySelectorAll(".citation-tag").forEach((btn) => {
      btn.addEventListener("click", () => {
        const citeNum = btn.getAttribute("data-cite");
        focusCitation(citeNum);
      });
    });

    // 3. Structured Metrics
    if (data.structured_metrics && data.structured_metrics.length > 0) {
      metricsSection.style.display = "block";
      metricsGrid.innerHTML = "";
      data.structured_metrics.forEach((m) => {
        const card = document.createElement("div");
        card.className = "metric-card";
        const valFormatted = typeof m.value === "number" ? m.value.toLocaleString() : m.value;
        card.innerHTML = `
          <span class="metric-name">${escapeHtml(m.name || "Metric")}</span>
          <span class="metric-value">${escapeHtml(valFormatted)} ${escapeHtml(m.unit || "")}</span>
          <div class="metric-meta">
            <span>${escapeHtml(m.period || "")}</span>
            <span>${escapeHtml(m.company || "")}</span>
          </div>
        `;
        metricsGrid.appendChild(card);
      });
    } else {
      metricsSection.style.display = "none";
    }

    // 4. Grounding Citations (Fix Issue 1)
    if (data.citations && data.citations.length > 0) {
      citationsSection.style.display = "block";
      citationsList.innerHTML = "";
      data.citations.forEach((c) => {
        const citeId = c.id || c.index || 1;
        const company = c.company_ticker || c.company || "AAPL";
        const year = c.fiscal_year ? ` (FY${c.fiscal_year})` : "";
        const section = c.section_title || c.section || "10-K Filing";
        const excerpt = c.excerpt || c.evidence_snippet || "";
        const cleanSnippet = stripHtml(excerpt).replace(/^#+\s*/, "").trim();

        const fullContent = c.full_text || c.text_full || c.evidence_snippet || c.excerpt || "";
        const completeContent = c.parent_text
          ? `${fullContent}\n\n### Expanded Parent Context\n${c.parent_text}`
          : fullContent;

        const card = document.createElement("div");
        card.className = "citation-card";
        card.id = `cite-card-${citeId}`;
        card.innerHTML = `
          <div class="citation-header">
            <span class="citation-id">[${citeId}] ${escapeHtml(company)}${escapeHtml(year)}</span>
            <span class="citation-filing">${escapeHtml(section)}</span>
          </div>
          <div class="citation-excerpt">"${escapeHtml(cleanSnippet.slice(0, 180))}..."</div>
        `;
        card.addEventListener("click", () => {
          showDocumentCanvasModal(
            `Citation [${citeId}] — ${company}${year} | ${section}`,
            completeContent
          );
        });
        citationsList.appendChild(card);
      });
    } else {
      citationsSection.style.display = "none";
    }

    // 5. Render Active Inspector Step
    renderInspectorStep(activeStep);
  }

  function updateStepperTimers(trace) {
    const nodeMap = {
      decomposer: "query_decomposer",
      retriever: "retriever",
      reranker: "reranker",
      grader: "grader",
      generator: "generator",
      guard: "hallucination_guard",
    };

    Object.entries(nodeMap).forEach(([stepKey, nodeName]) => {
      const node = findTraceNode(trace, nodeName);
      const btn = stepperNav.querySelector(`[data-step="${stepKey}"]`);
      if (btn) {
        let durBadge = btn.querySelector(".step-duration");
        if (!durBadge) {
          durBadge = document.createElement("span");
          durBadge.className = "step-duration";
          btn.appendChild(durBadge);
        }
        if (node && node.duration_s !== undefined && node.duration_s !== null) {
          durBadge.textContent = `${node.duration_s}s`;
          durBadge.style.display = "inline-block";
        } else {
          durBadge.style.display = "none";
        }
      }
    });
  }

  function focusCitation(citeNum) {
    const card = document.getElementById(`cite-card-${citeNum}`);
    if (card) {
      card.scrollIntoView({ behavior: "smooth", block: "center" });
      card.style.borderColor = "var(--accent-cyan)";
      card.style.boxShadow = "0 0 15px rgba(0, 229, 255, 0.4)";
      setTimeout(() => {
        card.style.borderColor = "";
        card.style.boxShadow = "";
      }, 2000);
    }
  }

  function renderInspectorStep(step) {
    if (!currentResult) {
      inspectorBody.innerHTML = `<div class="empty-state" style="padding: 40px 0;"><p>No query executed yet.</p></div>`;
      return;
    }

    const trace = currentResult.pipeline_trace || [];

    switch (step) {
      case "decomposer":
        renderDecomposerStep(findTraceNode(trace, "query_decomposer"));
        break;
      case "retriever":
        renderRetrieverStep(findTraceNode(trace, "retriever"));
        break;
      case "reranker":
        renderRerankerStep(findTraceNode(trace, "reranker"));
        break;
      case "grader":
        renderGraderStep(findTraceNode(trace, "grader"));
        break;
      case "generator":
        renderGeneratorStep(findTraceNode(trace, "generator"));
        break;
      case "guard":
        renderGuardStep(findTraceNode(trace, "hallucination_guard"));
        break;
      case "raw":
        renderRawState(currentResult);
        break;
      default:
        inspectorBody.innerHTML = `<p>Select a step above to inspect telemetry.</p>`;
    }
  }

  // ── Step 1: Decomposer ───────────────────────────────────────────────
  function renderDecomposerStep(node) {
    if (!node) {
      inspectorBody.innerHTML = `<p class="empty-state">No decomposer trace available.</p>`;
      return;
    }

    const subQueriesHtml = (node.sub_queries || [])
      .map((q) => `<div class="badge" style="padding: 6px 12px; margin-bottom: 6px; display: block; font-family: var(--font-body);">${escapeHtml(q)}</div>`)
      .join("");

    const filtersHtml = Object.entries(node.filters || {})
      .map(([k, v]) => `<span class="badge badge-cyan">${k}: <strong>${escapeHtml(v)}</strong></span>`)
      .join(" ");

    const durationText = node.duration_s !== undefined ? `${node.duration_s}s` : "N/A";

    inspectorBody.innerHTML = `
      <div class="inspector-section">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span class="data-label">Stage 1: Query Decomposition</span>
          <span class="badge badge-cyan">Duration: ${durationText}</span>
        </div>
        <div class="data-group">
          <span class="data-label">Query Classification</span>
          <div><span class="badge badge-emerald">${escapeHtml(node.query_type || "factual_numeric")}</span></div>
        </div>
        <div class="data-group">
          <span class="data-label">Extracted Metadata Filters</span>
          <div class="tag-list">${filtersHtml || "<span class='text-muted'>None (broad search)</span>"}</div>
        </div>
        <div class="data-group">
          <span class="data-label">Sub-Queries Generated (${(node.sub_queries || []).length})</span>
          <div>${subQueriesHtml}</div>
        </div>
        <div class="data-group">
          <span class="data-label">LLM Resource Usage</span>
          <div class="tag-list">
            <span class="badge">Model: ${escapeHtml(node.model || "gpt-4o")}</span>
            <span class="badge">Prompt: ${node.prompt_tokens || 0}</span>
            <span class="badge">Cached: ${node.cached_tokens || 0}</span>
            <span class="badge">Cost: $${(node.cost_usd || 0).toFixed(5)}</span>
          </div>
        </div>
      </div>
    `;
  }

  // ── Step 2: Retriever (Fix Issue 1 & 2) ──────────────────────────────
  function renderRetrieverStep(node) {
    const trace = currentResult.pipeline_trace || [];
    // Ensure we use the latest retriever pass
    const activeNode = node || findTraceNode(trace, "retriever");

    const filtersHtml = Object.entries((activeNode && activeNode.filters) || {})
      .map(([k, v]) => `<span class="badge badge-cyan">${k}: ${escapeHtml(v)}</span>`)
      .join(" ");

    // Chunks list: look inside trace node, with fallback to currentResult.search_results
    let chunks = (activeNode && activeNode.chunks && activeNode.chunks.length > 0)
      ? activeNode.chunks
      : [];

    if (chunks.length === 0 && currentResult.search_results && currentResult.search_results.length > 0) {
      chunks = currentResult.search_results.map((r, i) => ({
        rank: i + 1,
        chunk_id: r.chunk_id,
        score: r.score ? Number(r.score.toFixed(4)) : 0,
        company: r.metadata?.company_ticker || "AAPL",
        fiscal_year: r.metadata?.fiscal_year || "",
        section: r.metadata?.section_title || r.metadata?.section_id || "10-K Section",
        text_preview: (r.text || "").slice(0, 140).replace(/\n/g, " "),
        text_full: r.text || "",
      }));
    }

    let chunksHtml = "";
    chunks.forEach((chunk) => {
      const cleanSnippet = stripHtml(chunk.text_preview || chunk.text_full || "");
      chunksHtml += `
        <div class="chunk-row" data-chunk-rank="${chunk.rank}">
          <div class="chunk-meta-bar">
            <div style="display: flex; gap: 8px; align-items: center;">
              <span class="chunk-rank">#${chunk.rank}</span>
              <span class="badge badge-cyan">${escapeHtml(chunk.company || "AAPL")} ${chunk.fiscal_year ? `FY${escapeHtml(chunk.fiscal_year)}` : ""}</span>
              <span class="badge">${escapeHtml(chunk.section || "10-K Section")}</span>
            </div>
            <div class="score-bar-wrapper">
              <span class="badge badge-emerald">RRF Score: ${chunk.score}</span>
              <span class="badge badge-dim" style="font-size: 10px;">📊 View Canvas</span>
            </div>
          </div>
          <div class="chunk-snippet">${escapeHtml(cleanSnippet.slice(0, 160))}...</div>
        </div>
      `;
    });

    const durationText = activeNode && activeNode.duration_s !== undefined ? `${activeNode.duration_s}s` : "N/A";

    inspectorBody.innerHTML = `
      <div class="inspector-section">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span class="data-label">Stage 2: Hybrid Search (Dense + Sparse RRF)</span>
          <span class="badge badge-cyan">Duration: ${durationText}</span>
        </div>
        <div class="data-group">
          <span class="data-label">Search Statistics</span>
          <div class="tag-list">
            <span class="badge badge-emerald">Raw Candidates: ${(activeNode && activeNode.raw_results) || chunks.length}</span>
            <span class="badge badge-cyan">Unique Deduped: ${(activeNode && activeNode.unique_results) || chunks.length}</span>
          </div>
        </div>
        <div class="data-group">
          <span class="data-label">Qdrant Pre-Filter Conditions</span>
          <div class="tag-list">${filtersHtml || "<span class='text-muted'>No pre-filters applied</span>"}</div>
        </div>
        <div class="data-group">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span class="data-label">All Retrieved Content (${chunks.length} chunks)</span>
            <span class="badge badge-dim">Click row to open Document Canvas</span>
          </div>
          <div class="chunk-table">${chunksHtml || "<p class='text-muted'>No chunks retrieved.</p>"}</div>
        </div>
      </div>
    `;

    // Row click to open Document Canvas modal
    inspectorBody.querySelectorAll(".chunk-row").forEach((row) => {
      row.addEventListener("click", () => {
        const rankStr = row.getAttribute("data-chunk-rank");
        const target = chunks.find((c) => String(c.rank) === rankStr);
        if (target) {
          showDocumentCanvasModal(
            `Retrieved Chunk #${target.rank} — ${target.company} FY${target.fiscal_year} (${target.section})`,
            target.text_full || target.text_preview
          );
        }
      });
    });
  }

  // ── Step 3: Reranker ────────────────────────────────────────────────
  function renderRerankerStep(node) {
    const trace = currentResult.pipeline_trace || [];
    const activeNode = node || findTraceNode(trace, "reranker");

    let chunks = (activeNode && activeNode.chunks && activeNode.chunks.length > 0)
      ? activeNode.chunks
      : [];

    if (chunks.length === 0 && currentResult.reranked_results && currentResult.reranked_results.length > 0) {
      chunks = currentResult.reranked_results.map((r, i) => ({
        rank: i + 1,
        chunk_id: r.chunk_id,
        rerank_score: r.score ? Number(r.score.toFixed(4)) : 0,
        company: r.metadata?.company_ticker || "AAPL",
        fiscal_year: r.metadata?.fiscal_year || "",
        section: r.metadata?.section_title || r.metadata?.section_id || "10-K Section",
        text_preview: (r.text || "").slice(0, 140).replace(/\n/g, " "),
        text_full: r.text || "",
      }));
    }

    let chunksHtml = "";
    chunks.forEach((chunk) => {
      const score = chunk.rerank_score || 0;
      const barPercent = Math.min(100, Math.max(8, Math.round(((score + 10) / 15) * 100)));
      const cleanSnippet = stripHtml(chunk.text_preview || chunk.text_full || "");

      chunksHtml += `
        <div class="chunk-row" data-rank="${chunk.rank}">
          <div class="chunk-meta-bar">
            <div style="display: flex; gap: 8px; align-items: center;">
              <span class="chunk-rank">#${chunk.rank}</span>
              <span class="badge badge-cyan">${escapeHtml(chunk.company || "AAPL")} ${chunk.fiscal_year ? `FY${escapeHtml(chunk.fiscal_year)}` : ""}</span>
              <span class="badge">${escapeHtml(chunk.section || "10-K Section")}</span>
            </div>
            <div class="score-bar-wrapper">
              <div class="score-progress">
                <div class="score-fill" style="width: ${barPercent}%"></div>
              </div>
              <span class="score-number">${score.toFixed(3)}</span>
              <span class="badge badge-dim" style="font-size: 10px;">📊 View Canvas</span>
            </div>
          </div>
          <div class="chunk-snippet">${escapeHtml(cleanSnippet.slice(0, 160))}...</div>
        </div>
      `;
    });

    const durationText = activeNode && activeNode.duration_s !== undefined ? `${activeNode.duration_s}s` : "N/A";

    inspectorBody.innerHTML = `
      <div class="inspector-section">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span class="data-label">Stage 3: Cross-Encoder Reranker</span>
          <span class="badge badge-cyan">Duration: ${durationText}</span>
        </div>
        <div class="data-group">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span class="data-label">Top-K Reranked Evidence (${chunks.length})</span>
            <span class="badge badge-dim">Click row to open Document Canvas</span>
          </div>
          <div class="chunk-table">${chunksHtml || "<p class='text-muted'>No reranked chunks.</p>"}</div>
        </div>
      </div>
    `;

    // Click handler to open Document Canvas modal
    inspectorBody.querySelectorAll(".chunk-row").forEach((row) => {
      row.addEventListener("click", () => {
        const rankStr = row.getAttribute("data-rank");
        const target = chunks.find((c) => String(c.rank) === rankStr);
        if (target) {
          const content = target.parent_text
            ? `${target.text_full}\n\n[Expanded Parent Context]\n${target.parent_text}`
            : target.text_full;
          showDocumentCanvasModal(
            `Rank #${target.rank} — ${target.company} FY${target.fiscal_year} (${target.section})`,
            content
          );
        }
      });
    });
  }

  // ── Step 4: Grader (Fix Issue 3 & 4) ────────────────────────────────
  function renderGraderStep(node) {
    const trace = currentResult.pipeline_trace || [];
    const activeNode = node || findTraceNode(trace, "grader");

    if (!activeNode) {
      inspectorBody.innerHTML = `<p class="empty-state">No grader trace available.</p>`;
      return;
    }

    const relCount = activeNode.relevant ?? activeNode.relevant_count ?? 0;
    const partCount = activeNode.partial ?? activeNode.ambiguous_count ?? 0;
    const irrelCount = activeNode.irrelevant ?? activeNode.irrelevant_count ?? 0;

    const actionBadge =
      activeNode.action === "generate"
        ? "badge badge-emerald"
        : activeNode.action === "rewrite"
        ? "badge badge-amber"
        : "badge badge-rose";

    const grades = activeNode.grades || [];
    const rerankTrace = findTraceNode(trace, "reranker");
    let gradesListHtml = "";

    if (grades.length > 0) {
      gradesListHtml = grades
        .map((g) => {
          const rel = g.relevance || "irrelevant";
          const badgeClass =
            rel === "relevant"
              ? "badge badge-emerald"
              : rel === "partially_relevant"
              ? "badge badge-amber"
              : "badge badge-rose";
          const label = rel.replace("_", " ").toUpperCase();

          // Resilient fallback lookup if metadata was omitted
          const fallbackChunk = (rerankTrace && rerankTrace.chunks ? rerankTrace.chunks[g.chunk_index] : null) || (currentResult.reranked_results ? currentResult.reranked_results[g.chunk_index] : null) || {};
          const fallbackMeta = fallbackChunk.metadata || {};

          const company = g.company || fallbackChunk.company || fallbackMeta.company_ticker || "AAPL";
          const year = g.fiscal_year || fallbackChunk.fiscal_year || fallbackMeta.fiscal_year || "";
          const section = g.section || fallbackChunk.section || fallbackMeta.section_title || fallbackMeta.section_id || "Financial Statements";
          const textSnippet = g.text_preview || fallbackChunk.text_preview || fallbackChunk.text || "";
          const cleanSnippet = stripHtml(textSnippet);

          return `
            <div class="grade-card ${rel}" data-grade-idx="${g.chunk_index}" style="cursor: pointer;">
              <div class="grade-card-header">
                <div style="display: flex; gap: 8px; align-items: center;">
                  <span class="chunk-rank">Chunk #${g.chunk_index}</span>
                  <span class="badge badge-cyan">${escapeHtml(company)} ${year ? `FY${escapeHtml(year)}` : ""}</span>
                  <span class="badge">${escapeHtml(section)}</span>
                </div>
                <div style="display: flex; gap: 6px; align-items: center;">
                  <span class="${badgeClass}">${label}</span>
                  <span class="badge badge-dim" style="font-size: 10px;">📊 View Canvas</span>
                </div>
              </div>
              <div style="font-size: 12px; color: var(--text-muted); font-style: italic;">
                "${escapeHtml(cleanSnippet.slice(0, 160))}..."
              </div>
              <div class="grade-reason">
                <strong>Model Reasoning:</strong> ${escapeHtml(g.reason || "No reasoning provided.")}
              </div>
            </div>
          `;
        })
        .join("");
    } else {
      gradesListHtml = "<p class='text-muted'>No individual chunk grades recorded.</p>";
    }

    const durationText = activeNode.duration_s !== undefined ? `${activeNode.duration_s}s` : "N/A";

    inspectorBody.innerHTML = `
      <div class="inspector-section">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span class="data-label">Stage 4: Document Relevance Grading</span>
          <span class="badge badge-cyan">Duration: ${durationText}</span>
        </div>
        <div class="data-group">
          <span class="data-label">CRAG Decision & Confidence</span>
          <div class="tag-list">
            <span class="${actionBadge}">Action: ${escapeHtml((activeNode.action || "").toUpperCase())}</span>
            <span class="badge badge-cyan">Confidence: ${escapeHtml((activeNode.confidence || "").toUpperCase())}</span>
          </div>
        </div>
        <div class="data-group">
          <span class="data-label">Ternary Relevance Distribution</span>
          <div class="tag-list">
            <span class="badge badge-emerald">Relevant: ${relCount}</span>
            <span class="badge badge-amber">Ambiguous: ${partCount}</span>
            <span class="badge badge-rose">Irrelevant: ${irrelCount}</span>
          </div>
        </div>
        <div class="data-group">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span class="data-label">Evaluated Chunks & Model Explanations (${grades.length})</span>
            <span class="badge badge-dim">Click card to open Document Canvas</span>
          </div>
          <div class="grade-cards-list">${gradesListHtml}</div>
        </div>
        <div class="data-group">
          <span class="data-label">Resource Usage</span>
          <div class="tag-list">
            <span class="badge">Model: ${escapeHtml(activeNode.model || "gpt-4o")}</span>
            <span class="badge">Prompt: ${activeNode.prompt_tokens || 0}</span>
            <span class="badge">Cost: $${(activeNode.cost_usd || 0).toFixed(5)}</span>
          </div>
        </div>
      </div>
    `;

    // Click handler for Grade Card to open Document Canvas (Fix Issue 3)
    inspectorBody.querySelectorAll(".grade-card").forEach((card) => {
      card.addEventListener("click", () => {
        const idx = parseInt(card.getAttribute("data-grade-idx"), 10);
        const g = grades.find((gr) => gr.chunk_index === idx) || {};
        const fallbackChunk = (rerankTrace && rerankTrace.chunks ? rerankTrace.chunks[idx] : null) || (currentResult.reranked_results ? currentResult.reranked_results[idx] : null) || {};
        const text = g.text_full || fallbackChunk.text_full || fallbackChunk.text || g.text_preview || "No content.";
        const parent = g.parent_text || fallbackChunk.parent_text;
        const fullContent = parent ? `${text}\n\n[Expanded Parent Context]\n${parent}` : text;
        const company = g.company || fallbackChunk.company || "AAPL";
        const year = g.fiscal_year || fallbackChunk.fiscal_year || "";
        const section = g.section || fallbackChunk.section || "Section";

        showDocumentCanvasModal(
          `Chunk #${idx} (${(g.relevance || "").toUpperCase()}) — ${company} ${year ? `FY${year}` : ""} (${section})`,
          fullContent
        );
      });
    });
  }

  // ── Step 5: Generator (Fix Issue 5) ─────────────────────────────────
  function renderGeneratorStep(node) {
    const trace = currentResult.pipeline_trace || [];
    const activeNode = node || findTraceNode(trace, "generator");

    if (!activeNode) {
      inspectorBody.innerHTML = `<p class="empty-state">No generator trace available.</p>`;
      return;
    }

    const inputUser = activeNode.llm_input_user || "Context and question passed to model.";
    const outputLlm = activeNode.llm_output || currentResult.final_answer || "No output recorded.";
    const durationText = activeNode.duration_s !== undefined ? `${activeNode.duration_s}s` : "N/A";

    inspectorBody.innerHTML = `
      <div class="inspector-section">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span class="data-label">Stage 5: Grounded Answer Generation</span>
          <span class="badge badge-cyan">Duration: ${durationText}</span>
        </div>
        <div class="data-group">
          <span class="data-label">Token Breakdown & Financial Cost</span>
          <div class="tag-list">
            <span class="badge">Model: ${escapeHtml(activeNode.model || "gpt-4o")}</span>
            <span class="badge">Prompt: ${activeNode.prompt_tokens || 0}</span>
            <span class="badge badge-cyan">Cached (50% off): ${activeNode.cached_tokens || 0}</span>
            <span class="badge">Completion: ${activeNode.completion_tokens || 0}</span>
            <span class="badge badge-emerald">Cost: $${(activeNode.cost_usd || 0).toFixed(5)}</span>
          </div>
        </div>

        <div class="data-group">
          <span class="data-label">Input Prompt to LLM (with Injected Evidence)</span>
          <div class="llm-box">
            <div class="llm-box-header">
              <span>Full Formatted Prompt & Evidence Context</span>
              <button class="btn-ghost" style="padding: 2px 8px; font-size: 11px;" id="copyInputPrompt">Copy</button>
            </div>
            <pre class="llm-box-content">${escapeHtml(inputUser)}</pre>
          </div>
        </div>

        <div class="data-group">
          <span class="data-label">Direct Output from LLM (Pre-Guardrail)</span>
          <div class="llm-box">
            <div class="llm-box-header">
              <span>Raw Completion</span>
              <button class="btn-ghost" style="padding: 2px 8px; font-size: 11px;" id="copyOutputLlm">Copy</button>
            </div>
            <pre class="llm-box-content">${escapeHtml(outputLlm)}</pre>
          </div>
        </div>
      </div>
    `;

    const copyInBtn = document.getElementById("copyInputPrompt");
    if (copyInBtn) {
      copyInBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(inputUser);
        copyInBtn.textContent = "Copied!";
        setTimeout(() => (copyInBtn.textContent = "Copy"), 1500);
      });
    }

    const copyOutBtn = document.getElementById("copyOutputLlm");
    if (copyOutBtn) {
      copyOutBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(outputLlm);
        copyOutBtn.textContent = "Copied!";
        setTimeout(() => (copyOutBtn.textContent = "Copy"), 1500);
      });
    }
  }

  // ── Step 6: Guardrail ────────────────────────────────────────────────
  function renderGuardStep(node) {
    const trace = currentResult.pipeline_trace || [];
    const activeNode = node || findTraceNode(trace, "hallucination_guard");

    if (!activeNode) {
      inspectorBody.innerHTML = `<p class="empty-state">No hallucination guard trace available.</p>`;
      return;
    }

    const passed = (activeNode.result || "pass") === "pass";
    const statusBadge = passed ? "badge badge-emerald" : "badge badge-rose";
    const issues = activeNode.issues || [];
    const durationText = activeNode.duration_s !== undefined ? `${activeNode.duration_s}s` : "N/A";

    let issuesHtml = "";
    if (issues.length > 0) {
      issuesHtml = issues
        .map((issue) => `<div class="badge badge-amber" style="padding: 6px 12px; margin-bottom: 6px; display: block;">⚠️ ${escapeHtml(issue)}</div>`)
        .join("");
    } else {
      issuesHtml = "<span class='text-muted'>No hallucination issues detected. All claims grounded.</span>";
    }

    inspectorBody.innerHTML = `
      <div class="inspector-section">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span class="data-label">Stage 6: Hallucination Guardrail</span>
          <span class="badge badge-cyan">Duration: ${durationText}</span>
        </div>
        <div class="data-group">
          <span class="data-label">Verification Outcome</span>
          <div><span class="${statusBadge}">Result: ${(activeNode.result || "pass").toUpperCase()}</span></div>
        </div>
        <div class="data-group">
          <span class="data-label">Detected Issues (${issues.length})</span>
          <div>${issuesHtml}</div>
        </div>
        <div class="data-group">
          <span class="data-label">Guardrail Resource Usage</span>
          <div class="tag-list">
            <span class="badge">Model: ${escapeHtml(activeNode.model || "gpt-4o")}</span>
            <span class="badge">Cost: $${(activeNode.cost_usd || 0).toFixed(5)}</span>
          </div>
        </div>
      </div>
    `;
  }

  // ── Step 7: Raw State ────────────────────────────────────────────────
  function renderRawState(result) {
    inspectorBody.innerHTML = `
      <div class="inspector-section">
        <div class="data-group">
          <span class="data-label">Full Serialized Pipeline State</span>
          <pre class="code-box">${escapeHtml(JSON.stringify(result, null, 2))}</pre>
        </div>
      </div>
    `;
  }

  // ── Document Canvas Modal ────────────────────────────────────────────
  function showDocumentCanvasModal(title, text) {
    modalTitle.textContent = title;

    const hasHtml = /<\/?(table|tr|td|th|tbody|thead|div|span|p|h\d)[^>]*>/i.test(text);

    modalBody.innerHTML = `
      <div class="canvas-tabs">
        <button class="canvas-tab-btn active" id="tabVisual">📊 Document Canvas (Visual Table)</button>
        <button class="canvas-tab-btn" id="tabRaw">📝 Raw Source</button>
      </div>
      <div class="document-canvas" id="canvasView"></div>
      <pre class="code-box" id="rawView" style="display: none; max-height: 520px;"></pre>
    `;

    const canvasView = document.getElementById("canvasView");
    const rawView = document.getElementById("rawView");
    const tabVisual = document.getElementById("tabVisual");
    const tabRaw = document.getElementById("tabRaw");

    // Marked parses markdown headers/lists AND renders embedded HTML tables faithfully
    if (window.marked) {
      canvasView.innerHTML = window.marked.parse(text);
    } else {
      canvasView.innerHTML = text;
    }

    rawView.textContent = text;

    tabVisual.addEventListener("click", () => {
      tabVisual.classList.add("active");
      tabRaw.classList.remove("active");
      canvasView.style.display = "block";
      rawView.style.display = "none";
    });

    tabRaw.addEventListener("click", () => {
      tabRaw.classList.add("active");
      tabVisual.classList.remove("active");
      rawView.style.display = "block";
      canvasView.style.display = "none";
    });

    chunkModal.style.display = "flex";
  }

  // ── Helpers ─────────────────────────────────────────────────────────
  function findTraceNode(trace, nodeName) {
    if (!trace || trace.length === 0) return null;
    return [...trace].reverse().find((entry) => entry.node === nodeName) || null;
  }

  function stripHtml(html) {
    if (!html) return "";
    const tmp = document.createElement("div");
    tmp.innerHTML = html;
    return tmp.textContent || tmp.innerText || "";
  }

  function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
