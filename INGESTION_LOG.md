# 📋 Ingestion & Chunking Technical Execution Log

**Branch:** `feature/table-ingestion-markdown`  
**Focus:** SEC 10-K Financial Table Parsing, Normalization, Header Injection, and Markdown Chunking.  
**Started:** 2026-09-25 01:46 UTC  

---

## 1. Problem Baseline & Diagnosis

### Observed Symptoms
1. **Query Failure:** Query `What was Apple's net income in FY2023?` failed with `INSUFFICIENT` confidence and triggered max CRAG retries (3 cycles), taking 340.43s.
2. **Missing Information in Chunks:** In Qdrant chunk `AAPL_10-K_2023_AAPL_10K_2023_table_22_child_1`, the net income line item (`$96,995`) was present, but:
   * **No statement title:** It had no `"Consolidated Statements of Operations"` heading.
   * **No company or year header:** The date columns (*"September 30, 2023 | 2022 | 2021"*) were lost during table row-group splitting.
   * **70% token noise:** Raw nested HTML tags (`<tr>`, `<td></td>`, `colspan="3"`) degraded dense vector similarity and cross-encoder scores.

---

## 2. Step-by-Step Execution Plan

- [x] **Step 1: Create Table Normalizer Module (`src/ingestion/table_normalizer.py`)**
  - Parse HTML tables into a clean 2D grid.
  - Detect genuine date/column header rows (skip empty spacer rows).
  - Strip empty visual spacer columns (e.g. 18 raw columns collapsed into 4 semantic columns).
  - Merge disconnected currency symbols (`$` + `96,995` $\rightarrow$ `$96,995`).
  - Serialize into clean GitHub-Flavored Markdown.
- [x] **Step 2: Upgrade Table Title & Position Tracking (`src/ingestion/table_extractor.py`)**
  - Fixed `_find_preceding_title`: Uses backward document-order search (`find_previous`) instead of immediate single-sibling check, correctly recovering `"CONSOLIDATED STATEMENTS OF OPERATIONS"`, `"CONSOLIDATED BALANCE SHEETS"`, etc.
  - Integrated `normalize_html_table_to_markdown`: 57 out of 58 tables in AAPL 10-K now convert to pure Markdown (previously 35 fell back to raw HTML).
  - Added exact character offset tracking (`position_in_doc`) during table discovery and multi-page merging.
- [x] **Step 3: Section Splitter Robustness (`src/ingestion/section_splitter.py`)**
  - **Anchored Notes Split:** Fixed `_split_notes_from_item8` which previously matched the introductory Item 8 Table of Contents / Index link at character offset 183,638. Anchored to `Note 1` (`\bnote\s+1[.\s:\-–—]+`) and its preceding header at offset 225,086.
  - **Cover Page Capture:** Preserved document preamble/header prior to Item 1 (offset 0 to 40,336).
  - **Exact Table-Section Association:** Replaced heuristic `:5000` text substring check with O(1) interval check on `position_in_doc`: `section.start_position <= table.position_in_doc < section.end_position`.
  - **Result:** Primary financial statements (Statements of Operations, Comprehensive Income, Balance Sheets, Shareholders' Equity, Cash Flows) are now properly assigned to `item8_financial_statements`, while Note tables are assigned to `item8_notes`.
- [x] **Step 4: Standalone Unit Verification**
  - All 23 ingestion unit tests passing in 0.20s (`test_html_parser.py`, `test_table_extractor.py`, `test_section_splitter.py`, `test_xbrl_extractor.py`).
  - Total test suite (100 tests) passing in 4.94s.
- [x] **Step 5: Ingestion Re-run & Qdrant Verification**
  - Ingestion re-run completed: 3 filings processed, 235 total points indexed (157 vector points, 78 payload parents).
  - Cleaned Markdown tables verified in Qdrant with zero HTML tag noise.
- [x] **Step 6: End-to-End Query Verification**
  - Query: `"What was Apple's net income in FY2023?"`.
  - Result: Direct Cycle 0 PASS. Zero CRAG retries.
  - Final Answer: `"Apple's net income in FY2023 was $96,995 million [1]."`

---

## 3. Technical Changes & Architecture

### A. Table Normalizer (`src/ingestion/table_normalizer.py`)
SEC 10-K tables format financial statements using many visual spacer columns and separate cells for `$` symbols.
- **Before:**
  ```html
  <tr>
    <td colspan="3">Net income</td>
    <td></td>
    <td>$</td>
    <td>96,995</td>
    <td></td>
    <td>$</td>
    <td>99,803</td>
    <td></td>
    <td>$</td>
    <td>94,680</td>
  </tr>
  ```
- **After (`normalize_html_table_to_markdown`):**
  ```markdown
  | Line Item | September 30, 2023 | September 24, 2022 | September 25, 2021 |
  | :--- | :---: | :---: | :---: |
  | Net sales: |  |  |  |
  | Products | $298,085 | $316,199 | $297,392 |
  | Services | 85,200 | 78,129 | 68,425 |
  | Total net sales | 383,285 | 394,328 | 365,817 |
  | Net income | $96,995 | $99,803 | $94,680 |
  ```
- **Token Impact:** Statement of Operations shrunk from ~5,726 characters (~1,500 tokens) to ~1,200 characters (~508 tokens), allowing the complete statement to fit into a **single atomic child chunk** (`child_0`) with zero row-group fragmentation.

### B. Section Splitter & Table Association (`src/ingestion/section_splitter.py`)
- Fixed premature Item 8 split caused by TOC link:
  Primary financial statements are now correctly located in `item8_financial_statements`.
- Tables are mapped via document offsets:
  ```python
  if section.start_position <= table.position_in_doc < section.end_position:
      section.tables.append(table)
  ```
  Resulting in clean distribution: 6 tables in `item8_financial_statements` (including Operations, Balance Sheets, Cash Flows), 32 tables in `item8_notes`, 6 in `item7_mda`, 2 in `item5_market`, 7 in `item15_exhibits`.

---

## 4. Before & After Verification Comparison

| Metric | Before Fix (Commit `43fcaa8`) | After Fix (`feature/table-ingestion-markdown`) |
| :--- | :--- | :--- |
| **Ingested Table Format** | 35 HTML fallback / 23 Markdown | **57 Markdown / 1 HTML fallback** |
| **Operations Statement Chunking** | Split into fragments; header row was blank spacer `<tr><td></td>...</tr>` | **Single atomic child chunk** (`child_0`, 506 tokens) with complete headers and title |
| **Section Tagging** | Tables dumped into `item8_notes` or fallback `item8_financial_statements` | Correctly partitioned: 6 primary statements in `item8_financial_statements`, 32 in `notes` |
| **Search Rank for Net Income** | Low dense score (-3.97 cross-encoder), missing from top candidates | **Rank 1 in Hybrid Search (Score: 0.5476)** |
| **Relevance Grader Action** | `insufficient` $\rightarrow$ rewrite loop | **Direct `generate` (Confidence: medium, Cycle: 0)** |
| **CRAG Retries / Cycles** | 3 cycles (max retries exhausted) | **0 retries (Cycle 0 direct pass)** |
| **Hallucination Guard** | Refusal (`I could not find sufficient evidence...`) | **PASS (0 issues, verified answer `$96,995 million`)** |
| **Query Latency** | 340.43s (5.7 min) | **140s on pure CPU (Single cycle, 59% latency reduction)** |
| **LLM Query Cost** | $0.06385 | **$0.02069 (68% cost reduction)** |
| **Unit Test Coverage** | 100/100 tests passing | **102/102 tests passing (including new normalization, XBRL stripping, and prose tests)** |

---

## 5. Phase 2: Prose Preservation & XBRL Noise Elimination

### A. Problem Diagnosis
1. **Missing Prose in Narrative Sections:**
   - In `data/chunk/chunks_AAPL_2023_item1a_risk_factors.md`, only 1 chunk of 7 tokens existed (`Item 1A. Risk Factors`), dropping ~67k characters of risk factors.
   - Root Cause: In Workiva-generated SEC filings, narrative text resides in `<div><span>...</span></div>` blocks, while only section titles use `<p>`. `ProseChunker._extract_paragraphs()` checked `if p_tags: return paragraphs`, returning just the heading and skipping all body text.
2. **XBRL Taxonomy Metadata Leaking into Prose:**
   - In `AAPL_10-K_2023_cover_page_prose_0`, raw schema tags like `us-gaap:CommonStockMember`, `false 2023 FY 0000320193 P1Y`, and FASB URLs (`http://fasb.org/us-gaap/2023#...`) were indexed as prose.
   - Root Cause: `HTMLParser._strip_xbrl_tags()` unwrapped all `ix:*` tags including non-visual `<ix:header>` and `<ix:hidden>` containers before decomposing standalone tags, dumping thousands of taxonomy schema URLs directly into the DOM.

### B. Implementation
1. **`HTMLParser._strip_xbrl_tags` (`src/ingestion/html_parser.py`):**
   - Decomposes non-visual XBRL containers and taxonomy definitions (`ix:header`, `ix:hidden`, `xbrli:*`, `xbrldi:*`, `link:*`, `xlink:*`) *before* unwrapping inline facts (`ix:nonFraction`, `ix:nonNumeric`).
   - Added decomposition of non-content `<script>` and `<style>` tags.
2. **`ProseChunker._extract_paragraphs` (`src/chunking/prose_chunker.py`):**
   - Decomposes non-content tags (`table`, `script`, `style`, `ix:header`, `ix:hidden`).
   - Extracts all text blocks separated by newlines (`soup.get_text(separator="\n")` + double newline split).
   - Filters out running headers/footers matching regex `^.+?\s*\|\s*\d{4}\s+form\s+10-k\s*\|\s*\d+$` and fragments $\le 10$ characters.
3. **Automated Verification:**
   - Added `test_decomposes_xbrl_header_and_taxonomy` in `tests/test_ingestion/test_html_parser.py`.
   - Added `test_chunks_div_based_paragraphs_with_header` and `test_filters_running_headers` in `tests/test_chunking/test_prose_chunker.py`.
   - Re-ingested all AAPL 10-K filings and re-exported chunk inventory files.

### C. Chunking Metrics Before & After

| Section | Before Tokens (Chunks) | After Tokens (Chunks) | Result |
| :--- | :--- | :--- | :--- |
| **`cover_page`** | 526 tokens (1 chunk) with raw `us-gaap` URLs | **1,404 tokens (3 chunks)** | 100% clean SEC text; zero taxonomy schema URLs |
| **`item1_business`** | 7 tokens (1 chunk) | **2,690 tokens (6 chunks)** | Full narrative business model and segments captured |
| **`item1a_risk_factors`** | 7 tokens (1 chunk) | **11,928 tokens (26 chunks)** | All macroeconomic, supply chain, regulatory risks captured |
| **`item7_mda`** | 7 tokens (1 chunk) | **3,062 tokens (6 chunks)** | Full executive MD&A and segment commentary captured |
| **`item8_notes`** | 7 tokens (1 chunk) | **8,322 tokens (18 chunks)** | Complete accounting policy notes and disclosures captured |
| **Total FY2023 Child Chunks** | 57 chunks (67 KB exported) | **104 chunks (229 KB exported)** | Fully rich hybrid knowledge base |

---

## 6. Phase 3: Exact Document-Order Flow & Checkbox Preservation

### A. Problem Diagnosis
1. **Misplaced Table of Contents (Document Order Desynchronization):**
   - In `data/chunk/chunks_AAPL_2023_cover_page.md`, the Table of Contents table was indexed as Chunk 1. However, in the physical 10-K document, the Table of Contents table is at character offset 11,292, directly following the text header `"TABLE OF CONTENTS"` (offset 11,220).
   - Root Cause:
     1. `ChunkingPipeline._process_section` processed all tables first, and all prose second, separating the two into disconnected lists.
     2. `inspect_chunks.py` sorted by chunk ID string (`_table_11_...` came before `_table_5_...` due to ASCII order, and tables before prose).
2. **Missing Checkboxes & Answers in Cover Page Statements:**
   - Statements like `"Indicate by check mark if the Registrant is a well-known seasoned issuer..."` were indexed without their checked box (`Yes ☒ No ☐`).
   - Root Cause:
     1. Inline `<span>` tags within `<div>` blocks had newlines inserted, breaking `'Yes'`, `'☒'`, `'No'`, `'☐'` into 4 micro-fragments.
     2. Threshold `len(para) <= 10` dropped them as noise (`len("Yes ☒ No ☐") == 10`).

### B. Implementation
1. **`ChunkMetadata` (`src/chunking/metadata.py`):**
   - Added `chunk_index: int = Field(default=0)` to track the exact chronological document sequence across all child and parent chunks.
2. **`TableExtractor` (`src/ingestion/table_extractor.py`):**
   - During extraction, replaces extracted `<table>` elements in the DOM with `<div data-table-placeholder="{table_id}"></div>` and exposes `self.annotated_html`. Empty/merged-away tables are decomposed.
3. **`SectionSplitter` (`src/ingestion/section_splitter.py`):**
   - `_associate_tables` matches tables to sections using `data-table-placeholder="{table_id}"` presence in section HTML.
4. **`ProseChunker` (`src/chunking/prose_chunker.py`):**
   - Extracts leaf block elements (`p`, `div`, headings), preserving inline spans.
   - Detects checkbox characters (`☒`, `☐`, `☑`, `\u2610`, `\u2611`, `\u2612`, `\u25a2`, `\u25a0`).
   - Merges short checkbox answers directly into their preceding statement paragraph (e.g. `"...Rule 405 of the Securities Act.  Yes ☒ No ☐"`).
   - Added `chunk_paragraphs(...)` for programmatic paragraph chunking.
5. **`ChunkingPipeline` (`src/chunking/chunker.py`):**
   - Implemented `_chunk_section_flow`: walks through the section DOM in physical document sequence, seamlessly interleaving prose paragraphs and tables.
   - Flushes accumulated prose before table placeholders, emits table chunks, and resumes prose.
   - In `process_filing`, assigns monotonic `chunk_index` to every chunk.
6. **`inspect_chunks.py` (`scripts/inspect_chunks.py`):**
   - Updated sort key to `p.payload.get("chunk_index", 0)`, guaranteeing exported markdown matches the exact physical document order.

### C. Cover Page Sequence Before & After

| # | Before Phase 3 | After Phase 3 (Exact Document Sequence) |
| :--- | :--- | :--- |
| **Chunk 1** | `AAPL_10K_2023_table_11_child_0` (TOC table) | `cover_page_prose_0` (SEC header, ☒ Annual Report) |
| **Chunk 2** | `AAPL_10K_2023_table_5_child_0` (Address/IRS) | `table_5_child_0` (Address/IRS table) |
| **Chunk 3** | `AAPL_10K_2023_table_7_child_0` (Securities) | `cover_page_prose_1` (Telephone, registered securities) |
| **Chunk 4** | `AAPL_10K_2023_table_9_child_0` (Filer status) | `table_7_child_0` (Securities registered table) |
| **Chunk 5** | `cover_page_prose_0` (Preamble text) | `cover_page_prose_2` (Check mark statements: **Yes ☒ No ☐**) |
| **Chunk 6** | `cover_page_prose_1` (Check mark text, no boxes) | `table_9_child_0` (Filer status table with **☒ / ☐**) |
| **Chunk 7** | `cover_page_prose_2` (TOC header + Forward-looking) | `cover_page_prose_3` (Emerging growth **☐**, Shell company **Yes ☐ No ☒**, Shares, **TABLE OF CONTENTS**) |
| **Chunk 8** | *(none)* | `table_11_child_0` (**Table of Contents table right after header!**) |
| **Chunk 9** | *(none)* | `cover_page_prose_4` (Forward-looking statements) |
