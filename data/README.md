# Data Directory — Dataset License Notes

## Evaluation Datasets

| Dataset | File | License | Commercial Use | n | Source |
|---------|------|---------|---------------|---|--------|
| FinanceBench | `eval/financebench_in_corpus.jsonl` | CC BY-NC 4.0 | **Non-commercial only** | 27 questions | https://huggingface.co/datasets/PatronusAI/financebench |
| TAT-QA | `eval/tatqa_eval.jsonl` | CC BY 4.0 | Yes | 50 questions | https://github.com/NExTplusplus/tat-qa |
| AAPL custom QA | `eval/custom_eval.jsonl` | Our own data | No restrictions | — | Internal |
| OOC abstention | `eval/abstention_eval.jsonl` | Our own data | No restrictions | — | Internal |

## Notes

### FinanceBench (`eval/financebench_in_corpus.jsonl`)
- License: **CC BY-NC 4.0** (Attribution-NonCommercial)
- Commercial use is **not permitted**
- This file is **not committed** to the repository due to the non-commercial restriction
- Download separately from: https://huggingface.co/datasets/PatronusAI/financebench
- The 27 in-corpus questions used in this project correspond to questions whose referenced filings
  are present in the ingested Qdrant index (MMM, BA, KO, NFLX, PFE, AAPL 10-K filings)
- After downloading, place the filtered 27-question file at `data/eval/financebench_in_corpus.jsonl`

### TAT-QA (`eval/tatqa_eval.jsonl`)
- License: **CC BY 4.0** (Attribution)
- Commercial use is permitted with attribution
- 50 questions sampled from the TAT-QA dataset, committed as `tatqa_eval.jsonl`
- Source: https://github.com/NExTplusplus/tat-qa

### Custom AAPL QA (`eval/custom_eval.jsonl`)
- Our own AAPL 10-K questions; no restrictions
- Used as the primary clean held-out benchmark (n=50, AAPL50)

### OOC Abstention (`eval/abstention_eval.jsonl`)
- Our own out-of-corpus questions designed to test abstention behavior; no restrictions
- Used for OOC20 abstention gate evaluation (n=20, 100% abstained)
