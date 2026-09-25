# 🎯 50 Retrieval Evaluation QA Benchmark (Apple Inc. / AAPL)

**Target Corpus:** SEC Form 10-K Filings (FY2022, FY2023, FY2024)  
**File Location:** [`data/eval/custom_eval.jsonl`](file:///c:/Project/Finance_RAG/data/eval/custom_eval.jsonl)  
**Primary Objective:** Evaluate hybrid search (dense + sparse), cross-encoder reranking, tabular reasoning, narrative extraction, and hallucination guardrails across 50 gold-standard ground-truth questions.

---

## 1. Benchmark Category Distribution

| Category | Count | Focus & Retrieval Capability Tested |
| :--- | :---: | :--- |
| **Category 1: Single-Metric Direct Table Extraction** | **12** | Direct factual lookup from primary financial statements (Operations, Balance Sheet, Cash Flows). Tests date matching, unit scaling, and line-item precision. |
| **Category 2: Multi-Year Comparison & Arithmetic** | **10** | Multi-column and multi-chunk numerical reasoning. Tests year-over-year percentage calculations, margin ratios, and Free Cash Flow computation. |
| **Category 3: Product & Geographic Segment Breakdown** | **10** | Disaggregated footnote & MD&A segment tables. Tests product categories (iPhone, Mac, Services, etc.) and geographic regions (Americas, Europe, Greater China, etc.). |
| **Category 4: Balance Sheet, Liquidity & Cash Flow** | **8** | Non-income-statement balance items: Cash, marketable securities, total assets, share buybacks, and operating cash flow. |
| **Category 5: Narrative, Risks & Qualitative Disclosures** | **5** | Complex prose retrieval over Item 1A Risk Factors, Item 7A Market Risks, and Item 1 Business. Tests semantic search across dense text blocks. |
| **Category 6: Cover Page Form Disclosures & Checkboxes** | **3** | Form disclosures, filer classification checkmarks (`Large accelerated filer ☒`), shell status, par value, and shares outstanding. |
| **Category 7: Negative Guardrail & Abstention Tests** | **2** | Out-of-scope metrics (e.g. competitor financials or nonexistent divisions). Tests CRAG refusal and hallucination rejection. |
| **Total** | **50** | **Comprehensive Full-Spectrum Evaluation** |

---

## 2. Complete Inventory of 50 QA Pairs

### Category 1: Single-Metric Direct Table Extraction (12 QA Pairs)
| # | Question | Ground Truth | Target Section | Difficulty |
| :---: | :--- | :--- | :--- | :---: |
| 1 | What was Apple's total net sales in FY2024? | `$391,035 million` | `item8_financial_statements` | Easy |
| 2 | What was Apple's total net sales in FY2023? | `$383,285 million` | `item8_financial_statements` | Easy |
| 3 | What was Apple's total net sales in FY2022? | `$394,328 million` | `item8_financial_statements` | Easy |
| 4 | What was Apple's net income in FY2024? | `$93,736 million` | `item8_financial_statements` | Easy |
| 5 | What was Apple's net income in FY2023? | `$96,995 million` | `item8_financial_statements` | Easy |
| 6 | What was Apple's gross margin in FY2023? | `$169,148 million` | `item8_financial_statements` | Easy |
| 7 | What was Apple's gross margin in FY2024? | `$180,683 million` | `item8_financial_statements` | Easy |
| 8 | What was Apple's research and development expense in FY2023? | `$29,915 million` | `item8_financial_statements` | Easy |
| 9 | What was Apple's research and development expense in FY2024? | `$31,370 million` | `item8_financial_statements` | Easy |
| 10 | What was Apple's diluted earnings per share in FY2023? | `$6.13` | `item8_financial_statements` | Easy |
| 11 | What was Apple's diluted earnings per share in FY2024? | `$6.08` | `item8_financial_statements` | Easy |
| 12 | What was Apple's total operating expenses in FY2023? | `$54,847 million` | `item8_financial_statements` | Easy |

---

### Category 2: Multi-Year Comparison & Arithmetic (10 QA Pairs)
| # | Question | Ground Truth | Target Section | Difficulty |
| :---: | :--- | :--- | :--- | :---: |
| 13 | What was the percentage change in Apple's total net sales from FY2022 to FY2023? | `-2.80% (decrease from $394,328M to $383,285M)` | `item8_financial_statements` | Medium |
| 14 | Did Apple's net income increase or decrease between FY2022 and FY2023, and by how much? | `Decreased by $2,808M (from $99,803M to $96,995M, or -2.81%)` | `item8_financial_statements` | Medium |
| 15 | What was the growth in Apple's Services revenue from FY2022 to FY2023? | `Grew by $7,071M or 9.05% (from $78,129M to $85,200M)` | `item8_financial_statements` | Medium |
| 16 | How did Apple's R&D expenditure change between FY2022 and FY2023 in dollar terms? | `Increased by $3,664M (from $26,251M to $29,915M)` | `item8_financial_statements` | Medium |
| 17 | What was Apple's operating margin percentage in FY2023? | `29.82% ($114,301M operating income / $383,285M sales)` | `item8_financial_statements` | Hard |
| 18 | What was Apple's gross margin percentage in FY2023 compared to FY2022? | `44.13% in FY2023 vs 43.31% in FY2022` | `item8_financial_statements` | Hard |
| 19 | What was the change in Apple's Products cost of sales from FY2022 to FY2023? | `Decreased by $12,189M (from $201,471M to $189,282M)` | `item8_financial_statements` | Medium |
| 20 | Compare Apple's effective tax rate in FY2023 versus FY2022. | `14.7% in FY2023 compared to 16.2% in FY2022` | `item7_mda` | Medium |
| 21 | What was Apple's Free Cash Flow in FY2023? | `$99,584M ($110,543M operating cash minus $10,959M CapEx)` | `item8_financial_statements` | Hard |
| 22 | What was the change in basic weighted-average shares outstanding between FY2022 and FY2023? | `Decreased by 471,732 thousand shares (due to repurchases)` | `item8_financial_statements` | Medium |

---

### Category 3: Product Category & Geographic Segment Breakdown (10 QA Pairs)
| # | Question | Ground Truth | Target Section | Difficulty |
| :---: | :--- | :--- | :--- | :---: |
| 23 | What was Apple's iPhone net sales in FY2023? | `$200,583 million` | `item7_mda` | Easy |
| 24 | What was Apple's Mac revenue in FY2023? | `$29,357 million` | `item7_mda` | Easy |
| 25 | What was Apple's iPad net sales in FY2023? | `$28,300 million` | `item7_mda` | Easy |
| 26 | What was Apple's Wearables, Home and Accessories revenue in FY2023? | `$39,845 million` | `item7_mda` | Easy |
| 27 | What percentage of Apple's total net sales did iPhone account for in FY2023? | `52.33% ($200,583M / $383,285M)` | `item7_mda` | Medium |
| 28 | What was Apple's net sales in the Americas segment in FY2023? | `$162,560 million` | `item7_mda` | Easy |
| 29 | What was Apple's net sales in Europe in FY2023? | `$94,294 million` | `item7_mda` | Easy |
| 30 | What was Apple's net sales in Greater China in FY2023? | `$72,559 million` | `item7_mda` | Easy |
| 31 | What was Apple's net sales in Japan in FY2023? | `$24,257 million` | `item7_mda` | Easy |
| 32 | Which product category experienced the largest percentage revenue decline for Apple in FY2023? | `Mac (-27%, from $40,177M to $29,357M)` | `item7_mda` | Medium |

---

### Category 4: Balance Sheet, Liquidity & Cash Flow (8 QA Pairs)
| # | Question | Ground Truth | Target Section | Difficulty |
| :---: | :--- | :--- | :--- | :---: |
| 33 | What were Apple's total assets as of September 30, 2023? | `$352,583 million` | `item8_financial_statements` | Easy |
| 34 | What was Apple's cash and cash equivalents balance at the end of FY2023? | `$29,965 million` | `item8_financial_statements` | Easy |
| 35 | What was the total value of Apple's marketable securities (current and non-current combined) as of September 30, 2023? | `$132,134 million ($31,590M current + $100,544M non-current)` | `item8_financial_statements` | Medium |
| 36 | What was Apple's total shareholders' equity as of September 30, 2023? | `$62,146 million` | `item8_financial_statements` | Easy |
| 37 | What were Apple's total liabilities as of September 30, 2023? | `$290,437 million` | `item8_financial_statements` | Easy |
| 38 | What was Apple's cash generated by operating activities in FY2023? | `$110,543 million` | `item8_financial_statements` | Easy |
| 39 | How much did Apple spend on common stock repurchases in FY2023? | `$77,550 million` | `item8_financial_statements` | Easy |
| 40 | How much cash did Apple pay for dividends and dividend equivalents in FY2023? | `$15,025 million` | `item8_financial_statements` | Easy |

---

### Category 5: Narrative, Risks & Qualitative Disclosures (5 QA Pairs)
| # | Question | Ground Truth | Target Section | Difficulty |
| :---: | :--- | :--- | :--- | :---: |
| 41 | Where is the majority of Apple's manufacturing and supply chain outsourced according to Item 1A? | `Outsourced partners located primarily in Asia, with significant operations in mainland China, India, and Vietnam.` | `item1a_risk_factors` | Medium |
| 42 | What risk does Apple disclose regarding single-source component suppliers in Item 1A? | `Relies on single-source or limited-source suppliers for custom silicon and display panels, risking shortages and price fluctuations.` | `item1a_risk_factors` | Medium |
| 43 | What legal proceedings or regulatory scrutiny does Apple disclose regarding the App Store in the 10-K? | `Antitrust investigations, EU Digital Markets Act requirements, and Epic Games litigation.` | `item1a_risk_factors` | Hard |
| 44 | What foreign exchange risk does Apple identify in Item 7A regarding its international sales? | `Strengthening of U.S. dollar adversely impacts translated non-U.S. net sales and gross margins.` | `item7a_market_risk` | Medium |
| 45 | What is Apple's executive office address and telephone number disclosed in the 10-K? | `One Apple Park Way, Cupertino, California 95014; (408) 996-1010.` | `cover_page` | Easy |

---

### Category 6: Cover Page Form Disclosures & Checkboxes (3 QA Pairs)
| # | Question | Ground Truth | Target Section | Difficulty |
| :---: | :--- | :--- | :--- | :---: |
| 46 | Is Apple classified as a large accelerated filer in the FY2023 10-K cover page? | `Yes (Large accelerated filer ☒)` | `cover_page` | Easy |
| 47 | Is Apple designated as a shell company in the FY2023 10-K? | `No (Yes ☐ No ☒)` | `cover_page` | Easy |
| 48 | What is the par value of Apple's common stock and how many shares were outstanding as of October 20, 2023? | `$0.00001 par value; 15,550,061,000 shares outstanding.` | `cover_page` | Medium |

---

### Category 7: Negative Guardrail & Abstention Tests (2 QA Pairs)
| # | Question | Ground Truth | Target Section | Difficulty |
| :---: | :--- | :--- | :--- | :---: |
| 49 | What was Microsoft's Azure cloud revenue in FY2023 according to Apple's 10-K? | `Refusal: Not reported in Apple's Form 10-K.` | `guardrail` | Medium |
| 50 | What was Apple's electric vehicle unit sales in FY2023? | `Refusal: Apple did not report EV sales in its 10-K.` | `guardrail` | Medium |

---

## 3. How to Run the Benchmark

To evaluate your pipeline against this 50-question gold standard set:
```bash
poetry run python scripts/evaluate.py --dataset custom
```
To run a quick dry-run preview:
```bash
poetry run python scripts/evaluate.py --dataset custom --dry-run
```
