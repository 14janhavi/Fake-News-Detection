# Indian-News Fake News Detection (multilingual, no pretrained models)

Classifies an Indian news headline / text / URL as FAKE or REAL with a model **trained from scratch** (TF-IDF + Naive Bayes / Logistic Regression / Linear SVM). 2,000 Indian-news items: English, Bengali, Assamese, Odia, Bhojpuri (Hindi-belt, Devanagari).

## Run (Windows: use `set PYTHONPATH=src` instead of the Mac/Linux prefix)
```
pip install scikit-learn pandas numpy scipy joblib requests beautifulsoup4
set PYTHONPATH=src
python src\train_final.py                 # ~1 min, creates models\fakenews_model.joblib
python predict.py --text "your headline"   # or  --url https://...   or no argument for interactive mode
python src\run_experiments.py             # reproduces every number below -> results\results.json
python src\source_shift_check.py          # English unseen-news-outlet test
```
`data/dataset_2000.csv` and `data/splits.csv` are included. To rebuild from the raw files: `python src/build_dataset.py raw` (raw = IFND.csv + 4 *_dataset.csv, see sources below).

## Literature comparison (Phase 1)
| Paper | Data | Labels | Methods | Evidence | Metric / result | Limits |
|---|---|---|---|---|---|---|
| IndicClaimBuster (2025) | ~9K claims; en, hi, bn, Hindi-English code-mix; politics/law/health; Indian news | SUPPORTS / REFUTES (3 gold evidence each) | Smith-Waterman baseline; fine-tuned multilingual SBERT retrieval; mBERT/XLM-R/MuRIL/IndicBERT classifiers | Gold evidence + retrieval pool | Macro-F1 (84.6 gold evidence vs 71.0 retrieved, as reported); Success@1/3/5 | Refuted claims mostly **GPT-generated** (synthetic bias); only 2 labels; code-mix weaker; **data is password-gated** |
| X-FACT (ACL 2021) | 31,189 claims, 25 languages, 85 fact-checkers | 7 labels (true … false, other) | mBERT claim-only, +metadata, attention evidence aggregator | 5 Google snippets/claim | Macro-F1 ~39-42 in-domain, ~15 out-of-domain/zero-shot | Claim-only bias, snippets only ~45% sufficient (Hindi sample), per-source label noise |
| FEVER (2018) | 185K Wikipedia-derived claims | SUPPORTED / REFUTED / NOT ENOUGH INFO | TF-IDF doc retrieval + DrQA + entailment | Wikipedia sentences (annotated) | Label accuracy + FEVER score (needs correct evidence) | English only, synthetic claims (mutations), annotation artefacts |
| AVeriTeC (2023) | 4.5K real claims | Supported/Refuted/NEI/Conflicting | Question-answer evidence pipeline | Web evidence as QA | Veracity acc. + evidence score | **Not among your attached files - summarised from memory; please verify.** |

**Problem statement:** given a headline/claim in any of several languages, estimate whether it is FAKE or REAL using a model with no pretrained weights, optionally supported by retrieved evidence, and evaluate on held-out and unseen-language data.


## Dataset (Phase 2)
- IndicClaimBuster was tried first but its zip is password-protected (request via the authors' form). Replaced by free Indian datasets:
  - **IFND** (Sharma & Garg 2021, github.com/sonalgarg174/Dataset): Indian English, fact-check sites (fake) vs Indian outlets (real).
  - **Bengali / Assamese / Odia / Bhojpuri** (github.com/ujjawalvivek9/multilingual_low_resource_dataset, HATNet): Indian-language articles, labels source-derived and human-verified.
- 400 per language = 200 FAKE + 200 REAL, **balanced**; duplicates, near-duplicates (same first 60 chars) and conflicting labels removed; long articles cut to the first 300 characters (headline + lede) so they resemble headlines.
- Leakage-safe split 70/15/15 stratified by language x label (1400/300/300), zero overlapping items (asserted).
- **Hindi proper is not included**: no openly downloadable Indian Hindi real/fake set was reachable. Bhojpuri (Devanagari script, Hindi-belt) is the closest. Add IndicClaimBuster/Hindi data when available.
- **Gap vs roadmap:** none of these datasets contain evidence passages or Supported/Refuted/NEI labels, so annotated Recall@k and 3-way evidence classification are not possible. Instead "evidence" = the 5 most similar already-labelled articles retrieved from the training collection by TF-IDF cosine.

## Results (held-out test, 300 items, macro-F1)
| Setting | NaiveBayes | LogReg | LinearSVM |
|---|---|---|---|
| Majority | 0.333 | | |
| Claim only | 0.900 | 0.903 | 0.897 |
| Claim + retrieved similar articles | 0.873 | 0.877 | 0.877 |
- Per language (claim-only LogReg): {'as': 0.866, 'bho': 0.95, 'bn': 0.883, 'en': 0.867, 'or': 0.95}.
- Repeated 5-fold CV: {'NaiveBayes': {'mean': 0.895, 'std': 0.016}, 'LogReg': {'mean': 0.908, 'std': 0.018}, 'LinearSVM': {'mean': 0.908, 'std': 0.019}} - models are statistically tied.
- Retrieval: top-5 neighbour vote alone gives 0.81 accuracy; top-1 neighbour is in the same language 99% of the time. Adding retrieved evidence did **not** beat claim-only.
- Error analysis (LogReg): 29/300 errors; FAKE->REAL 19, REAL->FAKE 10.

## IMPORTANT limitations (read before trusting the 90%)
1. **Source/style shortcut.** In English, the news *outlet alone* predicts the label with 99% accuracy (fake items come from fact-check sites such as Augment/IndiaToday-Fact-Check, real items from Tribune, Indian Express, ThePrint...). Top "fake" tokens are `fact`, `check`, `fake`, `viral`, `वायरल`: the model learns *fact-check-article wording*, not falsehood. A fact-check article debunking a rumour will be labelled FAKE.
2. **Unseen outlets:** on English, training on some outlets and testing on outlets never seen gives 0.785 (LogReg) / 0.744 (NB) macro-F1, versus 0.87-0.90 on a random split. Realistic expectation for genuinely new news: ~70-80% English, unknown (likely lower) for other languages.
3. **Unseen languages:** leave-one-language-out macro-F1 is {'as': 0.518, 'bho': 0.489, 'bn': 0.559, 'en': 0.333, 'or': 0.347} i.e. near chance; the model only works for the languages it was trained on.
4. Some non-English texts look machine-translated or scrambled, and labels are source-derived (publisher identifiers not released), so source bias there cannot be measured.
5. The model checks wording, not facts. It cannot verify a claim against the real world. A URL check only reads the page title and first paragraphs.

## Next steps
Get IndicClaimBuster (real Hindi/Bengali claims with gold evidence) to add Hindi, evidence-based Supported/Refuted labels and true Recall@k; add several independent outlets per class so outlet style cannot decide the label; add a Hindi set (e.g. Zenodo Gujarati/Hindi/Marathi/Telugu fake-news data, not reachable from my environment).
