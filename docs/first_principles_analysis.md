# Finance RAG — First-Principles Analysis & Critical Design Questions

## Executive Summary

This analysis deconstructs the proposed system into its atomic assumptions and surfaces **34 critical questions** across 8 design dimensions. The goal is to resolve ambiguity *before* writing code — where the cost of change is lowest.

> [!IMPORTANT]
> **Cross-Cutting Design Principle: Cost ↔ Accuracy Balance**
> Every decision below is evaluated against the tension between **token cost** (OpenAI API spend, embedding compute, storage) and **answer accuracy** (faithfulness, precision, recall). Where a tradeoff exists, we flag it explicitly and recommend the Pareto-optimal choice.

---

## 1. Data Ingestion & Parsing — The Foundation Layer

### First Principle: *Garbage in, garbage out. The quality ceiling of any RAG system is set by its parser.*

SEC 10-K/10-Q filings are notoriously hostile to automated parsing:
- **Nested HTML tables** with `colspan`/`rowspan` spanning 3+ levels
- **Inline XBRL tags** wrapping every financial metric
- **Inconsistent formatting** across filers (Apple's 10-K looks nothing like a small-cap's)
- **Exhibits and footnotes** that contain critical context but are structurally separate

> [!CAUTION]
> The spec mentions both `BeautifulSoup` (HTML) and `PyMuPDF` (PDF). SEC EDGAR serves 10-Ks primarily as **HTML** (since ~2020, most are iXBRL). PDF versions are often missing or reformatted. This is a fork-in-the-road decision.

### Critical Questions & Recommendations

---

#### Q1. HTML-first or PDF-first?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | HTML preserves table structure semantically; PDF requires OCR-like heuristics for tables. HTML is strictly superior for financial tables but harder to strip of XBRL noise. |
| **Decision** | ✅ **HTML-first** |

**Recommended Implementation:**
- Use `sec-edgar-downloader` to fetch the primary filing document (10-K HTM/HTML) directly from EDGAR.
- Parse with `BeautifulSoup` (lxml parser for speed).
- Build a dedicated XBRL-stripping pass: remove all `<ix:*>` and `<xbrli:*>` tags while preserving their inner text content.
- Retain `<table>`, `<tr>`, `<td>`, `<th>` structure for downstream table-aware chunking.
- **PDF as fallback only** for pre-2020 filings where only PDF is available — use `PyMuPDF` with `page.get_text("dict")` for layout-aware extraction in these rare cases.

**Cost ↔ Accuracy:** HTML parsing is zero-cost (local CPU) and produces higher-fidelity output than PDF OCR. No tradeoff — strictly dominant.

---

#### Q2. How do we handle multi-page tables?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | If we split naively, we lose the header row → the chunk becomes meaningless numbers without labels. We need a table-aware chunking strategy. |
| **Decision** | ✅ **Layout-aware parsing + header injection + parent-child indexing** |

**Recommended Implementation:**

1. **Layout-Aware Table Detection:** During HTML parsing, identify all `<table>` elements. For each table, detect if it has a `<thead>` or if the first `<tr>` contains `<th>` elements → mark this as the header row.

2. **Header Injection for Fragmented Tables:** SEC filings often repeat headers across pages (detectable by duplicate `<th>` rows). Detect and merge these into a single logical table:
   ```
   Strategy: If two consecutive <table> elements share identical header
   rows (fuzzy match >90%), merge their body rows into a single table.
   Else, treat as separate tables.
   ```

3. **Parent-Child Indexing:**
   - **Parent chunk** = the full reconstructed table (may be 1500+ tokens).
   - **Child chunks** = individual row groups (e.g., 5-row segments) with the header row prepended.
   - The child stores `parent_chunk_id` in metadata → at generation time, the agent can pull the full parent if the child alone is insufficient.

4. **Edge Case — Very Large Tables (>3000 tokens):**
   Split into logical sections (e.g., Current Assets, Non-Current Assets, Total Assets) using row label heuristics. Each section becomes a child chunk with the header injected.

**Cost ↔ Accuracy:** Header injection adds ~50-100 tokens per child chunk (minimal cost), but prevents catastrophic context loss. Parent expansion adds one extra Qdrant fetch (negligible latency).

---

#### Q3. Do we preserve tables as Markdown, HTML, or linearized text?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | Markdown tables are human-readable but break on complex `colspan` layouts. HTML preserves structure but inflates token count. Linearized text is compact but loses visual structure. |
| **Decision** | ✅ **Markdown by default, semantic-stripped HTML for irregular tables as fallback** |

**Recommended Implementation:**

1. **Default Path — Markdown Tables:**
   - Convert `<table>` to Markdown using a custom converter (not `markdownify`, which handles `colspan` poorly).
   - For simple tables (no `colspan`/`rowspan`), emit standard Markdown pipe tables:
     ```markdown
     | Item | FY2024 | FY2023 |
     |------|--------|--------|
     | Revenue | $394.3B | $383.3B |
     ```
   - GPT-4o parses Markdown tables natively and accurately.

2. **Fallback Path — Semantic-Stripped HTML:**
   - If a table has `colspan > 1` or `rowspan > 1` or nested sub-tables, preserve as minimal HTML:
     ```html
     <table><tr><th colspan="3">Consolidated Balance Sheet</th></tr>...</table>
     ```
   - Strip all CSS classes, `style` attributes, `width`/`height` — keep only semantic structure (`<table>`, `<tr>`, `<td>`, `<th>`, `colspan`, `rowspan`).
   - Tag in chunk metadata: `table_format: "html_fallback"` so the generation prompt can adapt.

3. **Detection Heuristic:**
   ```python
   def needs_html_fallback(table_element) -> bool:
       return any(
           cell.get('colspan', '1') != '1' or cell.get('rowspan', '1') != '1'
           for cell in table_element.find_all(['td', 'th'])
       )
   ```

**Cost ↔ Accuracy:** Markdown saves ~40% tokens vs. HTML. HTML fallback only triggers on ~15-20% of financial tables (complex ones), so average token cost stays low while accuracy on irregular tables is preserved.

---

#### Q4. How do we handle footnotes and cross-references?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | Footnotes contain critical context (accounting policy changes, restatements). If we chunk them separately, the LLM loses the reference chain. If we inline them, chunks become enormous. |
| **Decision** | ✅ **Linked metadata with agentic context expansion** |

**Recommended Implementation — Three-Layer Strategy:**

1. **Layer 1 — Explicit Note Extraction During Parsing:**
   - In SEC 10-Ks, all footnotes are centralized in **Item 8: "Notes to Consolidated Financial Statements"**.
   - During HTML parsing, extract each Note into its own labeled parent chunk with a clean identifier (`Note 1`, `Note 2`, etc.).
   - Store with metadata: `{"section": "notes", "note_id": "Note 12", "note_title": "Segment Reporting"}`.

2. **Layer 2 — Regex / LLM Link Extraction:**
   - When chunking tables or MD&A paragraphs, detect reference patterns and store in chunk metadata:
     ```python
     import re
     refs = re.findall(r'(?:See|Refer to)\s+Note\s+(\d+)', chunk_text)
     # Store as: metadata["referenced_notes"] = ["Note 12", "Note 5"]
     ```
   - This is deterministic, zero-cost, and runs at ingestion time.

3. **Layer 3 — Agentic Context Expansion (LangGraph Node):**
   - When the retrieval step pulls a chunk containing `referenced_notes`, the LangGraph agent executes a **secondary deterministic lookup**:
     - Fetch the corresponding Note chunk by `note_id` from Qdrant (exact metadata filter, no vector search).
     - Append `[Appended Context: Note 12 — Segment Reporting]` into the synthesis prompt alongside the primary chunk.
   - This is a **conditional node** in the graph — only fires when references are detected.

**Cost ↔ Accuracy:** Layer 1-2 are zero LLM cost (regex at ingestion). Layer 3 adds one Qdrant metadata lookup (~10ms) and ~200-500 extra generation tokens. This is far cheaper than stuffing all notes into every chunk, and far more accurate than ignoring them.

---

#### Q5. XBRL tag extraction — use or discard?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | XBRL tags contain machine-readable financial concepts (e.g., `us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax`). Could be used as structured metadata for precise filtering, but add parsing complexity. |
| **Decision** | ✅ **Selective extraction: Strip wrappers, extract standard US-GAAP tags as metadata** |

**Recommended Implementation:**

1. **Strip and Discard:**
   - All XML wrapper attributes: `decimals="-6"`, `contextRef="FD2024Q4"`, `unitRef="usd"`.
   - Schema URLs and namespaces (e.g., `xmlns:us-gaap="http://..."`)
   - Obscure custom taxonomy extensions that appear only once in a filing.

2. **Extract and Store (Top ~50 Standard US-GAAP Tags):**
   - Maintain a whitelist of high-value GAAP concepts:
     ```python
     GAAP_WHITELIST = {
         "us-gaap:Revenues", "us-gaap:NetIncomeLoss",
         "us-gaap:EarningsPerShareBasic", "us-gaap:Assets",
         "us-gaap:StockholdersEquity", "us-gaap:OperatingIncomeLoss",
         "us-gaap:CostOfGoodsAndServicesSold", "us-gaap:GrossProfit",
         # ... ~50 total
     }
     ```
   - When a chunk contains an `<ix:nonFraction>` or `<ix:nonNumeric>` tag with a `name` in the whitelist, store:
     ```json
     {"xbrl_concepts": ["us-gaap:Revenues", "us-gaap:NetIncomeLoss"]}
     ```
   - This enables **concept-level filtering** in Qdrant: "find all chunks containing Revenue data" without relying on text matching.

3. **Content Preservation:**
   - The inner text of ALL XBRL tags (whitelisted or not) is always preserved in the chunk text. We only discard the XML wrapper, never the financial data itself.

**Cost ↔ Accuracy:** XBRL extraction adds ~5% parsing time. The metadata payoff is significant: it enables deterministic retrieval of specific financial metrics, reducing reliance on (costly) vector search for precise queries like "What was Apple's EPS in 2024?".

---

## 2. Chunking Strategy — The Make-or-Break Decision

### First Principle: *A chunk must be a self-contained unit of meaning. If a human can't answer a question from a chunk alone, neither can the LLM.*

The spec says "Hierarchical / Semantic Chunking" — but this is underspecified. Financial documents have unique chunking challenges.

### Critical Questions & Recommendations

---

#### Q6. What is the target chunk size?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | BGE-M3 supports up to 8192 tokens but embedding quality degrades beyond ~512. Larger chunks improve context but dilute retrieval precision. Financial tables may need 1500+ tokens to be self-contained. |
| **Decision** | ✅ **Hybrid: ~512 tokens for prose, full-table (uncapped) for tables, with parent-child linking** |

**Recommended Implementation:**

1. **Prose Chunks (MD&A, Risk Factors, Business Description):**
   - Target: **512 tokens** (roughly 350-400 words).
   - Split on paragraph boundaries first; fall back to sentence boundaries if paragraph > 800 tokens.
   - Never split mid-sentence.

2. **Table Chunks:**
   - **Atomic unit** = the full logical table (after multi-page merging from Q2).
   - If a table < 2048 tokens → store as a single chunk (most income statements, balance sheets).
   - If a table > 2048 tokens → split into row-group children (with header injection), each ~512-1024 tokens.
   - Parent-child linkage ensures the full table is always recoverable.

3. **Embedding Strategy for Large Chunks:**
   - Embed only the first 512 tokens of each chunk for the dense vector (where BGE-M3 quality is highest).
   - Store the full text in Qdrant's payload for generation context.
   - Sparse (lexical) vectors use the full text — no truncation needed.

**Cost ↔ Accuracy:** Embedding only 512 tokens saves ~60% compute on large table chunks while maintaining dense retrieval quality. Sparse search over full text ensures no lexical signal is lost.

---

#### Q7. Semantic chunking or section-based chunking?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | Semantic chunking (by embedding similarity) can split mid-table. Section-based chunking (by HTML `<h2>`, `<table>` boundaries) preserves document structure but produces variable-size chunks. |
| **Decision** | ✅ **Section-based chunking** |

**Recommended Implementation:**

1. **Section Hierarchy from HTML Structure:**
   ```
   10-K Filing
   ├── Part I
   │   ├── Item 1: Business          → prose chunking (512 tok)
   │   ├── Item 1A: Risk Factors     → prose chunking (512 tok)
   │   └── Item 1B: Unresolved Staff → prose chunking (512 tok)
   ├── Part II
   │   ├── Item 6: Selected Data     → table chunking
   │   ├── Item 7: MD&A              → prose chunking (512 tok)
   │   ├── Item 8: Financial Stmts   → table chunking
   │   └── Item 8 Notes              → note-per-chunk (Q4 strategy)
   └── Part IV
       └── Exhibits                  → skip / minimal chunking
   ```

2. **Section Detection:**
   - Use regex on HTML headings to detect SEC section boundaries:
     ```python
     SECTION_PATTERNS = [
         r"Item\s+1[Aa]?\s*[.:\-–]", r"Item\s+7[Aa]?\s*[.:\-–]",
         r"Item\s+8\s*[.:\-–]", # ... etc.
     ]
     ```
   - Within each section, apply the appropriate chunking strategy (prose vs. table) based on content type.

3. **Why Not Semantic Chunking:**
   - Semantic chunking uses embedding cosine similarity to find "topic breaks." In financial documents, adjacent paragraphs about revenue and cost-of-revenue have high semantic similarity but are logically distinct accounting items. Semantic chunking would merge them incorrectly.
   - Section-based chunking respects the document's legal structure (which is mandated by SEC regulation and therefore consistent across filers).

**Cost ↔ Accuracy:** Section-based chunking is deterministic (no embedding calls at chunking time), saving ~8,000 embedding API calls per document vs. semantic chunking. Accuracy is higher because SEC filings have reliable structural markers.

---

#### Q8. Do we need parent-child chunk relationships?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | A "small-to-big" strategy retrieves on small chunks but expands to parent sections for generation. Critical for financial tables where the question matches a single row but the answer requires the full table context. |
| **Decision** | ✅ **Yes — implement two-level parent-child hierarchy** |

**Recommended Implementation:**

1. **Parent Level:**
   - A full section or full table (e.g., the entire Consolidated Income Statement, or the full MD&A revenue discussion).
   - Stored in Qdrant with `chunk_type: "parent"`, `chunk_id: "AAPL_10K_2024_item8_income_stmt"`.
   - May or may not have its own embedding (cost consideration — see below).

2. **Child Level:**
   - 512-token prose segments or row-group table segments with injected headers.
   - Stored with `chunk_type: "child"`, `parent_chunk_id: "AAPL_10K_2024_item8_income_stmt"`.
   - Always has dense + sparse embeddings.

3. **Retrieval Flow:**
   ```
   Query → Hybrid search returns child chunks (ranked by relevance)
         → Reranker scores children
         → For top-5 children, fetch parent_chunk_id from metadata
         → Deduplicate parents (multiple children may share a parent)
         → Pass parent text to generation (richer context)
         → Pass child text as "key evidence" highlights
   ```

4. **Whether to Embed Parents:**
   - **Recommended: Don't embed parents initially.** Parents are fetched by ID (deterministic), not by vector similarity. This halves the embedding compute and storage.
   - If retrieval recall proves insufficient later, we can add parent embeddings as a second-pass augmentation.

**Cost ↔ Accuracy:** Storing parent text without embedding adds ~40% more Qdrant payload storage but zero additional embedding cost. The generation accuracy improvement is substantial — the LLM sees the full table context instead of an orphaned row.

---

#### Q9. How do we handle the metadata envelope?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | Over-indexing increases storage; under-indexing cripples filtering. |
| **Decision** | ✅ **Curated schema with indexed filter fields and unindexed payload fields** |

**Recommended Metadata Schema:**

```json
{
  // === Indexed (Qdrant payload index) — used for pre-filtering ===
  "company_ticker": "AAPL",
  "filing_type": "10-K",
  "fiscal_year": 2024,
  "section": "item8_financial_statements",
  "chunk_type": "child",

  // === Unindexed payload — carried for generation context ===
  "company_name": "Apple Inc.",
  "filing_date": "2024-11-01",
  "fiscal_quarter": null,
  "section_title": "Consolidated Statements of Operations",
  "page_number": 42,
  "table_id": "income_stmt_2024",
  "parent_chunk_id": "AAPL_10K_2024_item8_income_stmt",
  "chunk_id": "AAPL_10K_2024_item8_income_stmt_rows_1_5",
  "referenced_notes": ["Note 2", "Note 12"],
  "xbrl_concepts": ["us-gaap:Revenues", "us-gaap:CostOfGoodsAndServicesSold"],
  "table_format": "markdown",
  "token_count": 487
}
```

**Indexing Strategy in Qdrant:**
- Create payload indexes on: `company_ticker`, `filing_type`, `fiscal_year`, `section`, `chunk_type`.
- These five fields cover 95%+ of filter queries (e.g., "Apple 10-K 2024 financial statements").
- All other fields are stored but not indexed — no storage overhead for indexes.

**Cost ↔ Accuracy:** 5 indexed fields add ~5KB per chunk to Qdrant's index. For 8,000 chunks, this is ~40MB total — negligible. The filtering precision gain is massive: it prevents retrieving Microsoft chunks when the user asks about Apple.

---

#### Q10. Overlap strategy between chunks?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | Zero overlap loses boundary context. 20% overlap causes near-duplicate retrieval. Financial tables should have zero overlap (atomic units). Prose sections need some overlap. |
| **Decision** | ✅ **Content-type adaptive: 10% overlap for prose, zero for tables** |

**Recommended Implementation:**

1. **Prose Chunks:** 10% overlap (~50 tokens). This means the last ~50 tokens of chunk N are repeated as the first ~50 tokens of chunk N+1.
   - Prevents losing context at paragraph boundaries.
   - 10% is low enough to avoid significant near-duplicate retrieval issues.
   - At reranking time, if two overlapping chunks both score in top-5, deduplicate by `chunk_id` adjacency.

2. **Table Chunks:** Zero overlap. Tables are atomic — a row belongs to exactly one chunk. The header is injected (not overlapped) as a prefix.

3. **Note Chunks:** Zero overlap. Each Note is self-contained.

**Cost ↔ Accuracy:** 10% overlap adds ~10% more tokens to embed for prose sections (cost: ~800 additional embeddings for a 4-company corpus). The boundary context preservation is worth it for MD&A sections where discussions flow across paragraphs.

---

## 3. Hybrid Search & Embedding — The Retrieval Core

### First Principle: *Retrieval recall at top-25 is the hard ceiling for answer quality. You cannot rerank what you didn't retrieve.*

### Critical Questions & Recommendations

---

#### Q11. What is the dense:sparse fusion ratio?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | Financial queries like *"What was NVIDIA's gross margin in FY2024?"* are highly lexical (sparse wins). Conceptual queries like *"How did Apple's supply chain strategy evolve?"* are semantic (dense wins). |
| **Decision** | ✅ **Query-adaptive weighting via a lightweight router, with RRF as the default** |

**Recommended Implementation:**

1. **Default: Standard RRF (equal weighting).**
   - Reciprocal Rank Fusion with `k=60` (standard constant):
     ```python
     rrf_score = 1 / (k + rank_dense) + 1 / (k + rank_sparse)
     ```
   - This is the starting point — no LLM cost, deterministic, well-understood.

2. **Query-Adaptive Enhancement (Phase 2):**
   - Use the GPT-4o query decomposition node to also classify the query type:
     ```json
     {"query_type": "factual_numeric" | "conceptual" | "comparative"}
     ```
   - Adjust fusion weights:
     - `factual_numeric` → sparse weight 0.7, dense weight 0.3
     - `conceptual` → sparse weight 0.3, dense weight 0.7
     - `comparative` → equal weights (default RRF)
   - This adds zero extra API calls (classification piggybacks on the decomposition prompt).

**Cost ↔ Accuracy:** Default RRF is zero-cost. Query-adaptive weighting reuses an existing GPT-4o call (no incremental cost) and can improve precision by 10-15% on polar query types (purely numeric vs. purely conceptual).

---

#### Q12. BGE-M3 sparse vs. standalone BM25?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | BGE-M3's learned sparse weights may underperform traditional BM25 on exact financial term matching (e.g., ticker symbols, exact dollar amounts). |
| **Decision** | ✅ **BGE-M3 sparse only (start), with BM25 as a measured fallback** |

**Recommended Implementation:**

1. **Phase 1:** Use BGE-M3's native sparse output exclusively.
   - BGE-M3 produces both dense and sparse vectors in a single forward pass — no additional compute.
   - Qdrant supports sparse vectors natively (`SparseVector` type).
   - This gives us hybrid search with a single model, simplifying the stack.

2. **Phase 2 (If Retrieval Recall Is Low on Numeric Queries):**
   - Add a standalone BM25 index (e.g., via Elasticsearch or Qdrant's new keyword search) as a third retrieval signal.
   - Fuse with triple-RRF: `score = α/rank_dense + β/rank_sparse_bge + γ/rank_bm25`.
   - Only pursue this if evaluation shows BGE-M3 sparse underperforms on exact-match queries (e.g., "$394.3 billion" or "AAPL").

**Cost ↔ Accuracy:** Starting with BGE-M3 sparse avoids running a separate BM25 engine (saves infrastructure cost and complexity). If accuracy on numeric queries is fine, we never add BM25. If not, the architecture allows plugging it in without restructuring.

---

#### Q13. Qdrant collection architecture — single collection or per-company?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | Single collection with metadata filtering is simpler but slower at scale. Per-company collections enable isolated searches but complicate cross-company queries. |
| **Decision** | ✅ **Single collection, metadata-filtered** |

**Recommended Implementation:**

```python
from qdrant_client.models import VectorParams, Distance, SparseVectorParams

collection_config = {
    "collection_name": "sec_filings",
    "vectors_config": {
        "dense": VectorParams(size=1024, distance=Distance.COSINE)
    },
    "sparse_vectors_config": {
        "sparse": SparseVectorParams()
    },
}
```

- **Payload Indexes** on `company_ticker`, `fiscal_year`, `filing_type`, `section`, `chunk_type`.
- **Pre-filtered search** (see Q15): queries always include a metadata filter, so Qdrant only searches within the relevant subset.
- **Estimated Scale:** 4 companies × ~2,000 chunks = ~8,000 vectors. Qdrant handles 100K+ vectors trivially on free tier.
- **Cross-company queries** work natively: omit the `company_ticker` filter.

**Cost ↔ Accuracy:** Single collection avoids collection management overhead. At 8K vectors, there is zero performance difference vs. per-company collections. If we scale to 100+ companies (100K+ vectors), we can revisit with sharding.

---

#### Q14. What Qdrant distance metric for dense vectors?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | BGE-M3 embeddings are normalized → cosine and dot product are equivalent. But if we fine-tune later, normalization may change. |
| **Decision** | ✅ **Cosine similarity** |

**Rationale:**
- BGE-M3 documentation recommends cosine similarity.
- Cosine is normalization-invariant — future-proof if we switch embedding models.
- Qdrant applies internal optimizations for cosine (stores normalized vectors, uses dot product internally).
- No measurable performance difference vs. dot product for normalized vectors.

---

#### Q15. Do we need a pre-filter or post-filter strategy in Qdrant?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | Pre-filtering (e.g., `filing_year == 2024`) before vector search is faster and more precise. Post-filtering wastes retrieval budget on irrelevant years. |
| **Decision** | ✅ **Always pre-filter when the query specifies structured constraints** |

**Recommended Implementation:**

1. **Filter Extraction (in Query Decomposition Node):**
   - The GPT-4o decomposition step extracts structured filters from natural language:
     ```json
     {
       "sub_queries": ["What was Apple's total revenue?"],
       "filters": {
         "company_ticker": "AAPL",
         "fiscal_year": 2024,
         "filing_type": "10-K"
       }
     }
     ```
   - This piggybacks on the existing decomposition call — zero incremental cost.

2. **Qdrant Search with Pre-Filter:**
   ```python
   results = client.search(
       collection_name="sec_filings",
       query_vector=("dense", query_embedding),
       query_filter=Filter(
           must=[
               FieldCondition(key="company_ticker", match=MatchValue(value="AAPL")),
               FieldCondition(key="fiscal_year", match=MatchValue(value=2024)),
           ]
       ),
       limit=25
   )
   ```

3. **Fallback — No Filter:**
   - If the query is broad (e.g., "Which tech company had the highest gross margin?"), omit filters and search the full collection.
   - The query decomposition node decides whether to apply filters based on query specificity.

**Cost ↔ Accuracy:** Pre-filtering reduces the search space by ~75% (1 company out of 4), making search faster and more precise. It eliminates false positives from other companies' filings — a common RAG failure mode for financial data.

---

## 4. Reranking — The Precision Layer

### First Principle: *Reranking is a precision tool, not a recall tool. It can only reorder what retrieval already found.*

### Critical Questions & Recommendations

---

#### Q16. Top-25 → Top-5 is aggressive. Is this the right ratio?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | If recall@25 is low, reranking can't fix it. If the query requires 8+ chunks, top-5 may be insufficient. |
| **Decision** | ✅ **Adaptive: Top-25 → Top-5 default, expandable to Top-8 for multi-entity/multi-year queries** |

**Recommended Implementation:**

1. **Default Flow:** Retrieve 25, rerank, take top 5.
   - Covers single-entity, single-year factual queries (80%+ of queries).
   - Keeps generation context tight → fewer tokens → lower GPT-4o cost.

2. **Expansion Trigger:**
   - If the query decomposition produces **3+ sub-queries** or involves **2+ companies/years**, expand to top 8.
   - The decomposition node emits: `{"result_count": 5}` or `{"result_count": 8}`.

3. **Hard Cap:** Never pass more than 10 chunks to generation.
   - 10 chunks × ~512 tokens = ~5,000 tokens of context. GPT-4o handles this easily.
   - Beyond 10 chunks, the "lost in the middle" problem degrades accuracy.

**Cost ↔ Accuracy:** Top-5 saves ~600 generation tokens vs. top-8. Adaptive expansion only triggers on complex queries (~20% of traffic), so average cost stays close to the top-5 baseline.

---

#### Q17. Should reranking use the original query or decomposed sub-queries?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | Query decomposition splits complex queries into sub-queries. Reranking against the wrong query signal degrades precision. |
| **Decision** | ✅ **Rerank against sub-queries independently, then merge** |

**Recommended Implementation:**

1. **Per-Sub-Query Reranking:**
   ```
   Sub-query 1: "Apple revenue FY2024" → retrieve 25 → rerank → top 5
   Sub-query 2: "Microsoft revenue FY2024" → retrieve 25 → rerank → top 5
   ```

2. **Merge & Deduplicate:**
   - Union all reranked results.
   - Deduplicate by `chunk_id` (keep highest score).
   - Sort by reranker score globally.
   - Take top-N (where N is the adaptive limit from Q16).

3. **Single-Query Case:**
   - If no decomposition occurred, rerank against the original query directly.

**Cost ↔ Accuracy:** Per-sub-query reranking costs N × 25 reranker forward passes instead of 25. For 2 sub-queries on CPU, this is ~1 second instead of ~500ms. The precision improvement on comparative queries is substantial — each sub-query gets its own relevance ranking.

---

#### Q18. Reranker latency budget?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | `bge-reranker-large` on CPU scores ~50 pairs/sec. 25 candidates = ~500ms. |
| **Decision** | ✅ **Start with `bge-reranker-large` on CPU; switch to `bge-reranker-v2-m3` if latency is unacceptable** |

**Recommended Implementation:**

1. **Phase 1: `BAAI/bge-reranker-large`**
   - ~500ms for 25 pairs on CPU (acceptable for a 6-8 second total pipeline).
   - Higher accuracy than smaller models (cross-encoder with 560M params).
   - If GPU is available (even a consumer GPU), drops to ~50ms.

2. **Phase 2 (If Needed): `BAAI/bge-reranker-v2-m3`**
   - ~200ms for 25 pairs on CPU.
   - Slightly lower accuracy but may be sufficient after tuning retrieval quality.

3. **Latency Budget Allocation:**
   | Stage | Target | Notes |
   |-------|--------|-------|
   | Query embedding | 100ms | BGE-M3, single query |
   | Qdrant search | 100ms | Pre-filtered, 8K vectors |
   | Reranking | 500ms | 25 pairs, CPU |
   | GPT-4o grading | 1500ms | JSON mode, streaming |
   | GPT-4o generation | 2500ms | Streaming first token |
   | **Total (no CRAG loop)** | **~4.7s** | |
   | **Total (1 CRAG retry)** | **~8.5s** | |

**Cost ↔ Accuracy:** `bge-reranker-large` costs ~$0 (local inference). The latency cost (500ms) is small relative to GPT-4o API latency (4s). Switching to v2-m3 saves 300ms but risks a measurable precision drop — only worth it if users report slowness.

---

## 5. CRAG & LangGraph Orchestration — The Intelligence Layer

### First Principle: *Self-correction loops must converge. An unbounded retry loop is a production incident waiting to happen.*

### Critical Questions & Recommendations

---

#### Q19. What is the maximum number of CRAG correction cycles?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | Unbounded loops burn API credits and create latency spikes. |
| **Decision** | ✅ **Max 2 correction cycles, then graceful degradation** |

**Recommended Implementation:**

```
Cycle 0 (initial): Retrieve → Rerank → Grade
  └── If ALL chunks relevant → Generate answer
  └── If SOME chunks irrelevant → Remove irrelevant, proceed with remaining
  └── If ALL/MOST irrelevant → Trigger Cycle 1

Cycle 1 (rewrite): Rewrite query → Re-retrieve → Re-rerank → Re-grade
  └── If improved → Generate answer
  └── If still insufficient → Trigger Cycle 2

Cycle 2 (fallback): Broader search (relax filters) → Grade
  └── If improved → Generate answer with [LOW CONFIDENCE] flag
  └── If still insufficient → Return "Insufficient evidence" with
      partial context and suggested manual lookup
```

**State Tracking in LangGraph:**
```python
class CRAGState(TypedDict):
    query: str
    rewritten_queries: list[str]
    cycle_count: int  # Hard cap at 2
    retrieved_chunks: list[dict]
    graded_chunks: list[dict]
    confidence: str  # "high" | "medium" | "low" | "insufficient"
```

**Cost ↔ Accuracy:** Each CRAG cycle costs ~1 GPT-4o grading call (~$0.005-0.01). Max 2 retries caps the worst case at 3× the base cost. Empirically, if retrieval fails twice on rewritten queries, a third attempt almost never succeeds — the data likely isn't in the corpus.

---

#### Q20. What does "fallback search" mean concretely?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | The architecture shows a fallback path for ambiguous/incomplete context. Needs concrete definition. |
| **Decision** | ✅ **Three-tier fallback: query rewrite → filter relaxation → graceful refusal** |

**Recommended Implementation:**

1. **Tier 1 — Query Rewrite (Cycle 1):**
   - GPT-4o reformulates the query with alternative terminology:
     - Original: "What was Apple's top line in 2024?"
     - Rewrite: "Apple total net revenue fiscal year 2024 10-K"
   - Re-run hybrid search with the rewritten query, same filters.

2. **Tier 2 — Filter Relaxation (Cycle 2):**
   - Broaden the metadata filter:
     - Remove `fiscal_year` constraint (search across all years).
     - Remove `section` constraint (search all sections, not just financial statements).
   - This catches cases where data is in an unexpected section (e.g., revenue breakdown in MD&A instead of Item 8).

3. **Tier 3 — Graceful Refusal:**
   - If still insufficient, return:
     ```json
     {
       "answer": "I could not find sufficient evidence in the available SEC filings to answer this question.",
       "confidence": "insufficient",
       "partial_evidence": ["Some related context found..."],
       "suggested_action": "Check the original filing at https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=AAPL"
     }
     ```

4. **No Web Search in V1:**
   - Web search (Tavily, Google) introduces unverifiable sources — antithetical to the zero-hallucination constraint.
   - Can be added as an optional V2 feature with explicit "[WEB SOURCE]" labeling.

**Cost ↔ Accuracy:** Tier 1 costs one GPT-4o call. Tier 2 costs one Qdrant search (negligible). Tier 3 costs nothing. Total worst-case fallback cost: ~$0.01. Refusing to answer is infinitely better than hallucinating a financial number.

---

#### Q21. How does query decomposition handle multi-entity, multi-year questions?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | *"Compare the R&D spending of Apple, Microsoft, and NVIDIA from 2021 to 2024 as a percentage of revenue."* decomposes into 12+ sub-queries. |
| **Decision** | ✅ **Smart decomposition with parallel retrieval and aggregated generation** |

**Recommended Implementation:**

1. **Decomposition Strategy — Entity × Metric, NOT Entity × Year:**
   - Instead of 12 sub-queries (4 entities × 3 years), decompose into 3:
     ```json
     [
       "Apple R&D expense and total revenue 2021-2024",
       "Microsoft R&D expense and total revenue 2021-2024",
       "NVIDIA R&D expense and total revenue 2021-2024"
     ]
     ```
   - Each sub-query retrieves chunks across multiple years (the fiscal_year filter is omitted or set to a range).
   - This produces 3 retrieval passes instead of 12 — **4× cheaper**.

2. **Parallel Retrieval:**
   - Sub-queries are independent → run retrieval + reranking in parallel (asyncio / threading).
   - Total retrieval time ≈ single-query time (not 3×).

3. **Aggregated Generation:**
   - All retrieved chunks from all sub-queries are merged, deduplicated, and passed to a single GPT-4o generation call.
   - The generation prompt explicitly requests a comparative table:
     ```
     Format your answer as a comparison table with columns for each company and rows for each year.
     ```

4. **Decomposition Cap:** Max 5 sub-queries per user query. If decomposition produces more, merge related sub-queries.

**Cost ↔ Accuracy:** Decomposing by entity instead of entity × year reduces retrieval passes from O(N×M) to O(N). The generation call is single regardless. Net cost saving: ~60% on complex comparative queries with no accuracy loss (the LLM is good at extracting multi-year data from a single chunk).

---

#### Q22. What is the relevance grading schema?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | Binary vs. ternary grading directly impacts the self-correction trigger and the quality of context passed to generation. |
| **Decision** | ✅ **Ternary: Relevant / Partially Relevant / Irrelevant, with JSON mode** |

**Recommended Implementation:**

```python
GRADING_PROMPT = """You are a financial document relevance grader.
Given a user question and a retrieved document chunk, grade the relevance.

Output JSON:
{
  "relevance": "relevant" | "partially_relevant" | "irrelevant",
  "reason": "one sentence explanation"
}

Criteria:
- "relevant": The chunk directly contains information needed to answer the question.
- "partially_relevant": The chunk contains related context but not the specific data needed.
- "irrelevant": The chunk has no bearing on the question.
"""
```

**Grading-to-Action Mapping:**
| Grade Distribution | Action |
|---|---|
| ≥3 "relevant" chunks | Proceed to generation (high confidence) |
| 1-2 "relevant" + some "partially_relevant" | Proceed with [MEDIUM CONFIDENCE] flag |
| 0 "relevant" but ≥2 "partially_relevant" | Trigger CRAG rewrite (Cycle 1) |
| All "irrelevant" | Trigger CRAG fallback (Cycle 2) |

**Batch Grading for Cost Efficiency:**
- Grade all 5 reranked chunks in a single GPT-4o call (not 5 separate calls):
  ```
  Grade each of the following 5 chunks against the question. Return a JSON array.
  ```
- This reduces from 5 API calls to 1 — **80% cost savings on grading**.

**Cost ↔ Accuracy:** Ternary grading adds ~20 output tokens per chunk vs. binary. But "partially relevant" is critical information — it tells the CRAG loop "you're close but need to refine," preventing unnecessary full retries.

---

#### Q23. How do we handle numerical reasoning?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | Financial queries often require arithmetic: CAGR, percentage changes, ratios, margins. LLMs can make calculation errors. |
| **Decision** | ✅ **GPT-4o for simple arithmetic + Python code execution tool for complex calculations** |

**Recommended Implementation:**

1. **Simple Arithmetic (GPT-4o Direct):**
   - Percentage change: `(new - old) / old × 100`
   - Margins: `gross_profit / revenue × 100`
   - GPT-4o handles these reliably with chain-of-thought prompting.
   - Prompt instruction: `"Show your calculation step by step. Verify by computing the reverse."`

2. **Complex Calculations (Python REPL Tool):**
   - CAGR: `(end_value / start_value)^(1/n) - 1`
   - Multi-year trend analysis
   - Weighted averages across segments
   - Add a `code_executor` tool node in LangGraph:
     ```python
     @tool
     def calculate(expression: str) -> str:
         """Execute a Python expression for financial calculations."""
         # Sandboxed execution with only math/decimal allowed
         allowed_names = {"__builtins__": {}, "math": math, "Decimal": Decimal}
         return str(eval(expression, allowed_names))
     ```

3. **Routing Logic:**
   - The generation node detects if the query requires calculation (keywords: "CAGR", "growth rate", "percentage change", "ratio").
   - If yes, it formulates the calculation as a Python expression and calls the tool.
   - The tool result is incorporated into the final answer.

**Cost ↔ Accuracy:** The Python REPL is zero API cost (local execution). It eliminates GPT-4o's ~5-10% error rate on multi-step arithmetic. The routing logic adds ~50 tokens to the generation prompt (negligible).

---

## 6. Generation & Hallucination Control — The Trust Layer

### First Principle: *In finance, a hallucinated number is worse than no answer. The system must know what it doesn't know.*

### Critical Questions & Recommendations

---

#### Q24. What is the citation format?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | Financial professionals need auditable citations that trace back to source documents. |
| **Decision** | ✅ **Structured citation with filing reference + section + direct EDGAR URL** |

**Recommended Implementation:**

```markdown
Answer: Apple's total net revenue for FY2024 was $394.3 billion. [1]

---
**Sources:**
[1] Apple Inc. 10-K (FY2024), Item 8: Consolidated Statements of Operations, p.42
    SEC Filing URL: https://www.sec.gov/Archives/edgar/data/320193/...
    Evidence: "Net sales: $394,328 (in millions)"
```

**Citation Data Model:**
```python
@dataclass
class Citation:
    index: int                    # [1], [2], ...
    company: str                  # "Apple Inc."
    filing_type: str              # "10-K"
    fiscal_year: int              # 2024
    section: str                  # "Item 8: Financial Statements"
    page_number: int | None       # 42
    evidence_snippet: str         # Direct quote from chunk (max 100 chars)
    sec_filing_url: str | None    # Direct link to EDGAR filing
```

**Generation Prompt Instruction:**
```
When citing information, use numbered references [1], [2], etc.
Every financial number MUST have a citation. If you cannot cite a source, say "I could not verify this figure."
```

**Cost ↔ Accuracy:** Citations add ~100-150 tokens to the output (minor cost increase). The auditability benefit is existential for financial use cases — without citations, the system is untrusted.

---

#### Q25. How do we handle "I don't know"?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | The system must refuse rather than fabricate when context is insufficient. |
| **Decision** | ✅ **Explicit refusal with partial evidence and confidence scoring** |

**Recommended Implementation:**

1. **Confidence Classification (in Generation Prompt):**
   ```
   Before answering, assess your confidence:
   - HIGH: The retrieved context contains the exact data needed.
   - MEDIUM: The context contains related data; some inference is required.
   - LOW: The context is tangential; significant uncertainty exists.
   - INSUFFICIENT: The context does not contain the needed information.

   If confidence is INSUFFICIENT, do NOT attempt to answer.
   Instead, respond with what you DO know and what is missing.
   ```

2. **Response Format for Insufficient Confidence:**
   ```json
   {
     "answer": null,
     "confidence": "insufficient",
     "what_was_found": "Found Apple's total revenue but not the segment-level breakdown for Services.",
     "what_is_missing": "Services segment revenue for FY2024 Q4.",
     "suggested_action": "This data may be in the 10-Q filing for Q4 2024, which is not in the current corpus."
   }
   ```

3. **Hallucination Guardrail (Post-Generation Check):**
   - After generation, run a fast verification prompt:
     ```
     Does the following answer contain any claims not supported by the provided context?
     Answer ONLY "yes" or "no".
     ```
   - If "yes" → flag the answer and either regenerate with stricter instructions or return with [UNVERIFIED] tags.

**Cost ↔ Accuracy:** The guardrail check adds one short GPT-4o call (~$0.001). This is the cheapest insurance against hallucination — the cost of a financial error in production dwarfs the API cost.

---

#### Q26. Do we need structured output for financial metrics?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | Structured output enables downstream dashboards, automated monitoring, and programmatic access. |
| **Decision** | ✅ **Dual output: prose answer + optional structured JSON** |

**Recommended Implementation:**

1. **Default: Prose Answer with Citations** (always returned).

2. **Structured Supplement** (returned when financial metrics are detected):
   ```json
   {
     "metrics": [
       {
         "name": "Total Net Revenue",
         "value": 394328,
         "unit": "millions USD",
         "period": "FY2024",
         "company": "Apple Inc.",
         "source_citation": 1
       }
     ],
     "calculations": [
       {
         "name": "YoY Revenue Growth",
         "formula": "(394328 - 383285) / 383285 * 100",
         "result": 2.88,
         "unit": "percent"
       }
     ]
   }
   ```

3. **When to Include Structured Output:**
   - The generation prompt instructs: `"If your answer includes specific financial metrics, also output them in the structured_metrics JSON field."`
   - This is opt-in at the prompt level — no extra API call.

**Cost ↔ Accuracy:** Structured output adds ~200-300 tokens to the response. The benefit is downstream automation: dashboards, Excel exports, trend analysis. For V1, this can be optional (enabled via query parameter).

---

## 7. Evaluation & Benchmarking — The Accountability Layer

### First Principle: *You can't improve what you can't measure. But you also can't trust a metric you don't understand.*

### Critical Questions & Recommendations

---

#### Q27. Which Ragas metrics are primary vs. secondary?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | Need to define pass/fail criteria and prioritize metric optimization. |
| **Decision** | ✅ **Tiered metric hierarchy with hard gates on Faithfulness** |

**Recommended Metric Framework:**

| Tier | Metric | Target | Purpose | Gate? |
|------|--------|--------|---------|-------|
| 🔴 Primary | **Faithfulness** | ≥ 0.95 | No hallucinated claims | Yes — hard gate |
| 🔴 Primary | **Answer Correctness** | ≥ 0.85 | Factual accuracy vs. ground truth | Yes — hard gate |
| 🟡 Secondary | **Context Precision** | ≥ 0.80 | Retrieval quality (relevant chunks ranked higher) | No — diagnostic |
| 🟡 Secondary | **Context Recall** | ≥ 0.75 | Retrieval completeness (all needed chunks found) | No — diagnostic |
| 🟢 Tertiary | **Answer Relevance** | ≥ 0.80 | Answer addresses the actual question | No — UX metric |

**Hard Gate Logic:**
- If Faithfulness < 0.95 on the eval set → **block deployment** until root cause is identified.
- If Answer Correctness < 0.85 → investigate retrieval quality (Context Precision/Recall) as likely culprit.

**Cost ↔ Accuracy:** Each Ragas metric requires 1-2 LLM judge calls per evaluation sample. For 150 samples × 5 metrics, that's ~750-1500 GPT-4o calls (~$5-10 per eval run). Run full evals only at milestones, not on every code change. Use a fast 20-sample smoke test for iterative development.

---

#### Q28. How do we handle FinanceBench vs. TAT-QA differently?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | FinanceBench tests factual retrieval; TAT-QA tests multi-step arithmetic. They need different rubrics. |
| **Decision** | ✅ **Separate evaluation tracks with shared infrastructure** |

**Recommended Implementation:**

| Dimension | FinanceBench | TAT-QA |
|-----------|-------------|--------|
| **Focus** | Factual retrieval accuracy | Numerical reasoning accuracy |
| **Primary Metric** | Faithfulness + Context Precision | Answer Correctness (exact match on numbers) |
| **Answer Comparison** | Semantic similarity (LLM judge) | Exact numerical match (within 1% tolerance) |
| **Evaluation Frequency** | Every milestone | Every milestone |
| **Sample Size** | Full dataset (~150 questions) | Subset (~100 questions, filtered to SEC-relevant) |

**TAT-QA Special Handling:**
- TAT-QA answers are often exact numbers. Use **numerical tolerance matching** instead of LLM-as-judge:
  ```python
  def numeric_match(predicted: float, expected: float, tolerance: float = 0.01) -> bool:
      return abs(predicted - expected) / max(abs(expected), 1e-9) <= tolerance
  ```
- This is deterministic, zero-cost, and more reliable than LLM judging for numerical accuracy.

**Cost ↔ Accuracy:** Numerical matching for TAT-QA eliminates ~100 LLM judge calls per eval run (saves ~$1-2). More importantly, it's more accurate — LLM judges can't reliably compare "39.4 billion" with "39,400 million".

---

#### Q29. What is the evaluation dataset size and split strategy?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | A biased eval set gives false confidence. Need representative coverage. |
| **Decision** | ✅ **Stratified split by difficulty, query type, and company** |

**Recommended Implementation:**

1. **FinanceBench (~150 questions):**
   - Use the full dataset — it's small enough to evaluate completely.
   - Stratify by difficulty if annotations exist; otherwise, by query type:
     - Factual lookup (e.g., "What was revenue?") — ~60%
     - Comparative (e.g., "Compare X and Y") — ~20%
     - Reasoning (e.g., "Why did margins decline?") — ~20%

2. **TAT-QA (~3,000+ questions, but many non-SEC):**
   - Filter to SEC-relevant questions only (~100-200 questions).
   - Stratify by operation type:
     - Extraction (direct lookup) — ~40%
     - Arithmetic (single-step calculation) — ~30%
     - Multi-step reasoning — ~30%

3. **Development vs. Evaluation Split:**
   - **Dev set (20%):** Used for iterative tuning during development. Run after every change.
   - **Eval set (80%):** Used for milestone evaluations only. Never tune against this set.
   - Random split with stratification, fixed seed for reproducibility.

**Cost ↔ Accuracy:** Dev set of ~30 FinanceBench + ~20 TAT-QA questions costs ~$1-2 per run. Full eval (~150 + ~100 questions) costs ~$10-15 per run. Running full evals only at milestones keeps costs manageable.

---

#### Q30. How do we handle LLM-as-judge unreliability?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | GPT-4o judging its own outputs creates a self-evaluation bias loop. |
| **Decision** | ✅ **GPT-4o as primary judge + numerical verification as complement + periodic human audit** |

**Recommended Implementation:**

1. **Primary Judge: GPT-4o (via Ragas)**
   - Pragmatic choice: Ragas is optimized for GPT-4o, and the bias is known and bounded.
   - For Faithfulness evaluation, the bias is minimal (checking if claims are supported by context is relatively objective).

2. **Complement: Deterministic Verification**
   - For numerical answers, bypass the LLM judge entirely:
     ```python
     if answer_type == "numeric":
         score = numeric_match(predicted, expected)
     else:
         score = ragas_evaluate(predicted, expected, context)
     ```
   - This eliminates the bias loop for ~40% of financial questions.

3. **Periodic Human Audit (Monthly or at Major Milestones):**
   - Randomly sample 20 evaluation results.
   - Human annotator verifies: "Did the judge correctly assess faithfulness?"
   - Track judge agreement rate. If < 90%, investigate prompt issues.

4. **Future Enhancement: Multi-Judge Consensus**
   - If budget permits, run evaluation with both GPT-4o and Claude/Gemini. Take conservative score (minimum of judges).
   - This doubles eval cost but significantly reduces bias risk.

**Cost ↔ Accuracy:** Deterministic verification for numeric answers saves ~40% of judge costs with higher reliability. Human audits cost time, not API dollars. Multi-judge consensus is a V2 enhancement — not needed for initial validation.

---

## 8. Infrastructure & Operability — The Production Layer

### First Principle: *A system that can't be deployed, monitored, and debugged is a demo, not a product.*

### Critical Questions & Recommendations

---

#### Q31. Qdrant Cloud (free tier) or local Docker?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | Free tier has 1GB storage, ~1M vectors. Need to estimate scale. |
| **Decision** | ✅ **Local Docker for development; Qdrant Cloud free tier for staging/demo** |

**Scale Estimation:**
```
4 companies × 200 pages/company × 10 chunks/page = 8,000 chunks
8,000 × (1024 dims × 4 bytes/dim) = ~32 MB dense vectors
8,000 × ~2 KB average sparse vector = ~16 MB sparse vectors
8,000 × ~1 KB average payload = ~8 MB metadata
Total: ~56 MB << 1 GB free tier limit
```

**Docker Setup:**
```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"   # REST API
      - "6334:6334"   # gRPC
    volumes:
      - ./qdrant_data:/qdrant/storage
    environment:
      QDRANT__SERVICE__GRPC_PORT: 6334
```

**Migration Path:**
- Dev: Local Docker (zero cost, full control, fast iteration).
- Demo: Qdrant Cloud free tier (accessible URL, no Docker dependency for reviewers).
- Production: Qdrant Cloud paid tier or self-hosted Kubernetes (future consideration).

**Cost ↔ Accuracy:** Local Docker is free. Qdrant Cloud free tier is free. No cost tradeoff here — it's purely an operational convenience decision.

---

#### Q32. What is the latency target for end-to-end query?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | CRAG loops can push latency to 15-20s. Need to set expectations. |
| **Decision** | ✅ **Target: <8s P50, <15s P95, with streaming for perceived responsiveness** |

**Recommended Implementation:**

1. **Latency Budget (P50 — No CRAG Loop):**
   | Stage | Target | Parallelizable? |
   |-------|--------|----------------|
   | Query decomposition (GPT-4o) | 1.0s | No (first step) |
   | Query embedding (BGE-M3) | 0.1s | Yes (parallel with above if cached) |
   | Qdrant hybrid search | 0.1s | No (needs embedding) |
   | Cross-encoder reranking | 0.5s | No (needs search results) |
   | CRAG grading (GPT-4o) | 1.5s | No (needs reranked chunks) |
   | Generation (GPT-4o) | 2.5s | No (needs graded chunks) |
   | **Total P50** | **~5.7s** | |

2. **Latency Budget (P95 — 1 CRAG Retry):**
   - Add: query rewrite (1s) + re-search (0.2s) + re-rerank (0.5s) + re-grade (1.5s) = +3.2s
   - **Total P95: ~8.9s**

3. **Streaming for Perceived Responsiveness:**
   - Stream the GPT-4o generation response token by token.
   - Show "Searching documents..." → "Analyzing relevance..." → "Generating answer..." progress indicators.
   - First token appears at ~3.2s (after retrieval + grading), even though full response takes ~5.7s.

4. **Cost Optimization: Combine Decomposition + Grading**
   - If the query is simple (no decomposition needed), skip the decomposition call entirely → save ~1s and ~$0.005.
   - Use a fast heuristic: if the query mentions only one company and one metric, bypass decomposition.

**Cost ↔ Accuracy:** Streaming adds zero API cost but dramatically improves perceived latency. Skipping decomposition for simple queries saves ~$0.005/query and ~1s latency, with no accuracy loss.

---

#### Q33. How do we handle OpenAI API costs?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | Each query triggers 3-5 GPT-4o calls. Evaluation runs can cost $50-100+. |
| **Decision** | ✅ **Built-in cost tracking with per-query and per-run budgets** |

**Recommended Implementation:**

1. **Per-Query Cost Estimation:**
   | Call | Input Tokens | Output Tokens | Cost (GPT-4o) |
   |------|-------------|---------------|----------------|
   | Decomposition | ~200 | ~100 | ~$0.002 |
   | Grading (batch) | ~3,000 | ~200 | ~$0.008 |
   | Generation | ~3,500 | ~500 | ~$0.012 |
   | Guardrail check | ~1,000 | ~10 | ~$0.003 |
   | **Total per query** | | | **~$0.025** |
   | With 1 CRAG retry | | | **~$0.040** |

2. **Cost Tracking Module:**
   ```python
   class CostTracker:
       def __init__(self, budget_per_query: float = 0.10, budget_per_run: float = 50.0):
           self.total_cost = 0.0
           self.query_cost = 0.0
           self.budget_per_query = budget_per_query
           self.budget_per_run = budget_per_run

       def log_call(self, model: str, input_tokens: int, output_tokens: int):
           cost = self._calculate_cost(model, input_tokens, output_tokens)
           self.query_cost += cost
           self.total_cost += cost
           if self.query_cost > self.budget_per_query:
               raise BudgetExceededError(f"Query budget exceeded: ${self.query_cost:.3f}")
   ```

3. **Cost-Saving Strategies:**
   - Batch grading (Q22): 1 call instead of 5 → saves ~$0.015/query.
   - Skip decomposition for simple queries: saves ~$0.002/query.
   - Use `gpt-4o-mini` for grading (lower accuracy but 10× cheaper): saves ~$0.007/query.
   - Cache embeddings for repeated queries.

4. **Evaluation Run Budgets:**
   - Dev smoke test (30 questions): ~$1.00
   - Full eval (250 questions): ~$6-10
   - Full eval with CRAG retries: ~$10-15

**Cost ↔ Accuracy:** The cost tracker enforces discipline without sacrificing accuracy. The per-query budget ($0.10) is generous enough to allow 2 CRAG cycles but prevents runaway loops. The GPT-4o-mini substitution for grading is a conscious cost-accuracy tradeoff — test both and compare Faithfulness scores.

---

#### Q34. What is the target Python version and dependency management?

| Aspect | Detail |
|--------|--------|
| **Why It Matters** | Rapid release cycles in the AI/ML ecosystem cause frequent breaking changes. |
| **Decision** | ✅ **Python 3.11+, Poetry with pinned versions, dependency groups** |

**Recommended Implementation:**

```toml
[tool.poetry]
name = "finance-rag"
description = "Enterprise financial intelligence engine using Corrective RAG"
version = "0.1.0"
python = "^3.11"

[tool.poetry.dependencies]
# Core
langchain = "^0.2"
langgraph = "^0.1"
openai = "^1.30"

# Embeddings & Reranking
sentence-transformers = "^3.0"
torch = "^2.3"
FlagEmbedding = "^1.2"

# Vector DB
qdrant-client = "^1.9"

# Data Ingestion
sec-edgar-downloader = "^5.0"
beautifulsoup4 = "^4.12"
lxml = "^5.0"
PyMuPDF = "^1.24"

# Evaluation
ragas = "^0.1"
datasets = "^2.19"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0"
pytest-asyncio = "^0.23"
ruff = "^0.4"
mypy = "^1.10"

[tool.poetry.group.notebook.dependencies]
jupyter = "^1.0"
ipywidgets = "^8.0"
```

**Why Poetry over pip-tools:**
- `poetry.lock` pins every transitive dependency (reproducible builds).
- Dependency groups separate dev/test/notebook deps from production.
- `poetry run` creates isolated virtualenvs automatically.

**Python 3.11+ Rationale:**
- Required by latest `langgraph` and `ragas`.
- 10-60% faster than 3.10 (free performance).
- `tomllib` built-in (no extra dependency for config).

**Cost ↔ Accuracy:** Poetry adds ~30 seconds to initial setup. The reproducibility benefit prevents "works on my machine" failures — a hidden cost that dwarfs any setup overhead.

---

## Decision Summary Matrix

| # | Question | Decision | Tier | Cost Impact | Accuracy Impact |
|---|----------|----------|------|-------------|-----------------|
| Q1 | HTML vs PDF parsing | HTML-first | 🔴 | None | ⬆⬆⬆ |
| Q2 | Multi-page tables | Header injection + parent-child | 🔴 | ⬆ Minimal | ⬆⬆⬆ |
| Q3 | Table format | Markdown + HTML fallback | 🟡 | ⬇ 40% fewer tokens | ⬆⬆ |
| Q4 | Footnotes | Linked metadata + agentic expansion | 🟡 | ⬆ +1 Qdrant lookup | ⬆⬆⬆ |
| Q5 | XBRL tags | Selective extract top GAAP tags | 🟡 | ⬆ +5% parse time | ⬆⬆ |
| Q6 | Chunk size | 512 prose / full-table | 🔴 | Neutral | ⬆⬆⬆ |
| Q7 | Chunking method | Section-based | 🔴 | ⬇ No embed at chunk time | ⬆⬆ |
| Q8 | Parent-child chunks | Yes, two-level | 🔴 | ⬆ +40% payload storage | ⬆⬆⬆ |
| Q9 | Metadata schema | 5 indexed + ~10 unindexed fields | 🟡 | ⬆ +40MB index | ⬆⬆⬆ |
| Q10 | Chunk overlap | 10% prose / 0% tables | 🟢 | ⬆ +10% embeddings | ⬆ |
| Q11 | Dense:sparse ratio | RRF default, adaptive Phase 2 | 🟡 | None (reuses existing call) | ⬆⬆ |
| Q12 | BGE-M3 sparse vs BM25 | BGE-M3 only (start) | 🟢 | ⬇ Single model | Neutral |
| Q13 | Qdrant architecture | Single collection | 🔴 | ⬇ Simpler ops | Neutral |
| Q14 | Distance metric | Cosine | 🟢 | None | Neutral |
| Q15 | Pre/post filter | Pre-filter always | 🟡 | ⬇ Faster search | ⬆⬆ |
| Q16 | Rerank candidate count | Top-5 default, adaptive top-8 | 🟡 | ⬇ Fewer gen tokens | ⬆ |
| Q17 | Rerank query target | Per sub-query, then merge | 🟡 | ⬆ N×25 reranker passes | ⬆⬆ |
| Q18 | Reranker model | bge-reranker-large (start) | 🟢 | None (local) | ⬆⬆ |
| Q19 | Max CRAG cycles | 2 cycles max | 🔴 | ⬆ Max 3× base cost | ⬆⬆ |
| Q20 | Fallback search | Rewrite → relax filters → refuse | 🔴 | ⬆ +$0.01 worst case | ⬆⬆⬆ |
| Q21 | Multi-entity decomposition | By entity, parallel retrieval | 🟡 | ⬇ O(N) not O(N×M) | ⬆⬆ |
| Q22 | Relevance grading | Ternary, batch grading | 🟡 | ⬇ 80% fewer grade calls | ⬆⬆ |
| Q23 | Numerical reasoning | GPT-4o + Python REPL tool | 🟡 | None (local exec) | ⬆⬆⬆ |
| Q24 | Citation format | Filing + section + EDGAR URL | 🟡 | ⬆ +100 output tokens | ⬆⬆⬆ |
| Q25 | "I don't know" | Explicit refusal + confidence score | 🔴 | ⬆ +1 guardrail call | ⬆⬆⬆ |
| Q26 | Structured output | Dual (prose + optional JSON) | 🟡 | ⬆ +200 output tokens | ⬆ |
| Q27 | Ragas metrics priority | Faithfulness ≥0.95 hard gate | 🟢 | ⬆ ~$5-10/eval run | ⬆⬆⬆ |
| Q28 | FinanceBench vs TAT-QA | Separate tracks | 🟢 | Neutral | ⬆ |
| Q29 | Eval dataset strategy | Stratified split, dev/eval | 🟢 | ⬇ Dev set saves $8/run | ⬆ |
| Q30 | Judge unreliability | GPT-4o + numeric verification | 🟢 | ⬇ 40% fewer judge calls | ⬆⬆ |
| Q31 | Qdrant hosting | Docker dev / Cloud demo | 🔴 | $0 | Neutral |
| Q32 | Latency target | <8s P50, streaming | 🟡 | $0 (streaming free) | Neutral |
| Q33 | API cost management | Built-in cost tracker | 🟡 | ⬇ Prevents overruns | Neutral |
| Q34 | Dependency management | Poetry, Python 3.11+ | 🔴 | ⬇ Prevents breakage | Neutral |

---

> [!IMPORTANT]
> All 34 questions now have detailed recommendations. Please review the decisions — particularly any you disagree with or want to modify — and I'll proceed to create the full implementation plan with file structure, module boundaries, and phased build order.
