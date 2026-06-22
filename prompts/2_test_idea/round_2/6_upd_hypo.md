# upd_hypo — test_idea

> Phase: `invention_loop` · round 2 · `upd_hypo`
> Run: `run_CFTHGHhY9evP` — Vocabulary Percolation Threshold for Self-Calibrating Extractive Summary Length: Recall Advantage and F1 Failure Mode Across Corpora
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `upd_hypo` (terminal_claude_agent)

### [1] SYSTEM-USER prompt · 2026-06-22 05:55:00 UTC

````
<ai_inventor_context>
<ai_inventor_summary>
You are one of many LLMs in AI Inventor — an automated research system that generates NOVEL and FEASIBLE hypotheses, investigates them through experiments and research, and produces a paper.

Your output feeds other LLMs downstream. This demands your ABSOLUTE MAXIMUM reasoning — every output must be deeply thought out and maximally useful. Surface-level responses waste downstream computation.
</ai_inventor_summary>

<your_role>
YOU ARE: A hypothesis reviser (Step 3.6: UPD_HYPO in the invention loop)

You received the current hypothesis, all artifacts, and the paper draft.
Revise the hypothesis based on what the evidence supports.

Honest revision → focused research. Inflated confidence → wasted iteration.
</your_role>
</ai_inventor_context>

You are revising a research hypothesis based on empirical evidence gathered
during an iterative invention loop. Your role is internal reflection — honest
assessment of what the evidence supports.

SCOPE: Your ONLY output is the revised hypothesis text. You do NOT run code,
produce artifacts, fix bugs, or otherwise act on the evidence yourself — the
next iteration of the invention loop will spawn fresh artifacts based on your
revised hypothesis. Reflect on the evidence and rewrite the hypothesis;
nothing else.

PRINCIPLES:
- Ground every revision in specific artifacts and results
- Treat negative and null results as valuable contributions. If the original
  approach failed, the null result IS often the contribution — frame it as
  such (e.g. "X does not improve Y under conditions Z"). Only pivot to a
  different positive claim when the evidence actually supports one; never
  fabricate a positive narrative to mask a failed approach.
- Increase specificity as evidence accumulates
- Don't inflate confidence without strong evidence
- Preserve the core AII prompt unless evidence clearly contradicts it
- Revise hypothesis text only — never attempt to address feedback by running
  code, proposing fixes, or producing artifacts; the next loop iteration
  handles all artifact generation

<current_hypothesis>
The hypothesis as it stands. Revise it based on the evidence below.

kind: hypothesis
title: >-
  Vocabulary Percolation Threshold as a Self-Calibrating Summary Length: Corpus-Conditional Recall Advantage and F1 Failure
  Mode
hypothesis: >-
  In word-frequency extractive summarization, a percolation-threshold stopping criterion—selecting sentences (by TF score)
  until the induced vocabulary co-occurrence subgraph's giant connected component reaches a critical fraction θ of the full
  document's GCC—achieves significantly higher ROUGE recall than fixed-ratio baselines but lower ROUGE F1, because it selects
  far more sentences than human-written highlights contain. Specifically: (a) on newswire corpora such as CNN/DailyMail, where
  articles follow an inverted-pyramid structure and highlights are concise (≈7–10% of article length), percolation at θ=0.9
  achieves ROUGE-1 recall ≈0.853–0.880 versus ≈0.707–0.728 for fixed@30%, but fixed@10% dominates on F1 (≈0.286 vs ≈0.214),
  because the GCC threshold fires at ≈76–78% of document sentences regardless of document type (std≈0.08, below the discriminative
  threshold of 10 pp); (b) this homogeneous compression behavior occurs because CNN/DailyMail articles lack vocabulary segregation—co-occurrence
  connectivity is distributed across sentences rather than clustered—so the GCC transition provides no document-discriminating
  signal on this corpus; (c) the percolation criterion is hypothesized to yield variable, document-adaptive compression ratios
  (std > 10 pp) and improved F1 on corpora with genuine vocabulary clustering, such as multi-topic review articles, multi-document
  summarization sets, or scientific abstracts; and (d) graph structural features (average degree, clustering coefficient,
  graph density) explain only R²≈0.10 of compression ratio variance on CNN/DailyMail, insufficient for reliable dynamic θ
  calibration on this corpus but potentially more predictive on structurally diverse corpora. Future evaluation on multi-topic
  corpora (e.g., Multi-News, arXiv abstracts) with stronger sentence-scoring baselines (TF-IDF, TextRank) is required to test
  whether percolation-based length selection provides a genuine advantage when vocabulary segregation is present.
motivation: >-
  Every existing word-frequency extractive summarizer—TextRank, LexRank, TF-IDF sentence scoring, and their descendants—requires
  the user or system to specify summary length as a fixed compression ratio (typically 20–30% of source words or a fixed k
  sentences). This parameter is chosen arbitrarily or tuned on validation data, ignoring the structural properties of the
  individual document. A news article covering a single event needs a much shorter summary than a multi-topic review paper
  of the same length. If the compression ratio is wrong, the summary either loses critical concepts (too short) or becomes
  redundant and verbose (too long). By importing the percolation phase transition from statistical physics, we can make the
  stopping criterion data-driven and document-specific: the summary grows until key vocabulary becomes mutually reachable
  in a co-occurrence network, signaling complete conceptual coverage rather than mere word-count thresholds. This directly
  addresses a decades-old, widely acknowledged limitation of extractive summarization, requires no additional training data
  or LLMs, and introduces a principled mathematical criterion (giant component emergence) where only empirical heuristics
  previously existed.
assumptions:
- >-
  The high-frequency content words of a document form a co-occurrence network that undergoes a giant connected component transition
  as sentences are added—i.e., the network is sparse enough that early sentences cover only disconnected vocabulary clusters,
  but a critical set of sentences bridges them.
- >-
  Sentences that contribute 'bridging' edges in the vocabulary network (connecting previously disconnected word clusters)
  carry more conceptual coverage than sentences that add isolated high-scoring words.
- >-
  The percolation threshold (fraction of sentences needed to achieve giant component) varies meaningfully across document
  types and genres, making a fixed compression ratio suboptimal.
- >-
  ROUGE recall against human reference summaries is a valid proxy for conceptual coverage, so percolation-determined summaries
  should score at least as well as fixed-ratio summaries on ROUGE metrics.
investigation_approach: >-
  1. DATA: Use CNN/DailyMail and/or DUC 2002 single-document summarization benchmarks (freely available via HuggingFace).
  2. VOCABULARY GRAPH CONSTRUCTION: For each document, tokenize and remove stopwords; build a weighted undirected graph where
  nodes are unique content words and edges connect word pairs co-occurring within the same sentence, with edge weight = co-occurrence
  count. 3. PERCOLATION STOPPING CRITERION: Order sentences by descending sum of their constituent word TF scores. Greedily
  add sentences one at a time; after each addition, compute the giant connected component (GCC) size in the induced vocabulary
  subgraph. Record the sentence count k* at which GCC_summary / GCC_document ≥ θ (e.g., θ = 0.8). This k* is the percolation-determined
  summary length. 4. BASELINES: Fixed-ratio summaries at 10%, 20%, 30% of sentences using the same TF word scoring for sentence
  ranking. 5. EVALUATION: Compare ROUGE-1, ROUGE-2, ROUGE-L of percolation summaries vs. all fixed-ratio baselines. 6. ANALYSIS:
  Measure how percolation-determined k* correlates with document structural properties (degree distribution, clustering coefficient)
  to characterize when percolation length diverges most from fixed ratios.
success_criteria: >-
  CONFIRMED if: (a) percolation summaries achieve statistically significantly higher ROUGE-1 recall than the best fixed-ratio
  baseline on at least one benchmark corpus, or (b) percolation-determined compression ratios vary substantially across document
  types (standard deviation > 10 percentage points) in a way that correlates with document coherence metrics, demonstrating
  that a single fixed ratio is suboptimal. DISCONFIRMED if: (a) percolation-determined lengths cluster tightly around a single
  compression ratio regardless of document type (suggesting the network phase transition adds no new information), or (b)
  fixed-ratio summaries at 20% consistently dominate on both ROUGE precision and recall, indicating the structural stopping
  criterion does not align with human reference summaries.
related_works:
- >-
  TextRank (Mihalcea & Tarau, 2004): builds a sentence-level graph where edges encode sentence similarity, ranks sentences
  via PageRank, then selects the top-k sentences. Differs from the proposed hypothesis in two ways: (1) it operates on sentence-level
  nodes rather than word-level co-occurrence; (2) it still requires a fixed k, the stopping criterion the percolation threshold
  is designed to replace.
- >-
  LexRank (Erkan & Radev, 2004): computes eigenvector centrality over a sentence cosine-similarity graph to rank sentences,
  selecting top-k. Same limitation as TextRank—compression ratio is externally specified; no mechanism to determine k from
  the document's own structure.
- >-
  Content Coverage Maximization on Word Networks (Ryang & Abekawa, 2013): uses an influence propagation model (epidemic spreading)
  on a word network to maximize the number of words 'covered' by selected sentences. Related in using word-level networks,
  but maximizes a coverage count under a budget constraint rather than using a phase transition to determine when coverage
  is complete—the budget (number of sentences) is still manually specified.
- >-
  Information Foraging Theory (Pirolli & Card, 1999): applies optimal foraging biology to model how HUMANS navigate information
  patches on the web, predicting when a reader switches from one source to another. This is a descriptive behavioral model
  of human reading, not an algorithmic sentence selection mechanism for automated summarization.
inspiration: >-
  The core idea is a Level 3 (methodological) cross-domain transfer from statistical physics to NLP. In percolation theory
  (Erdős-Rényi random graphs, bond percolation), a network undergoes a sharp phase transition when edge density crosses a
  critical threshold: below it, the network consists of many small disconnected components; above it, a giant connected component
  suddenly emerges spanning most nodes. This transition is a natural 'completeness signal'—it marks the point at which local
  clusters merge into a globally connected structure. Transposed to text: the vocabulary co-occurrence network of a summary
  starts fragmented (early sentences cover isolated topic clusters) and becomes globally connected as conceptually bridging
  sentences are added. The percolation threshold—the point at which a giant vocabulary cluster forms—signals that the summary
  has achieved 'conceptual percolation': all major topic clusters are connected and the summary is complete. This imports
  a mathematically precise stopping criterion into a domain where only empirical heuristics (20–30% compression) previously
  existed. The analogy is structurally tight: words are nodes, co-occurrences are edges, sentences add bundles of edges, and
  summary completeness = giant component emergence.
terms:
- term: Percolation threshold
  definition: >-
    In graph theory and statistical physics, the critical edge density at which a random graph transitions from many small
    disconnected components to having a single giant connected component spanning most nodes. Here it refers to the point
    during sentence addition where the summary's vocabulary co-occurrence subgraph forms a giant connected component.
- term: Giant connected component (GCC)
  definition: >-
    The largest connected subgraph in a network, measured by number of nodes. In the summary vocabulary graph, it represents
    the set of content words that are mutually reachable via co-occurrence links—i.e., the core conceptual cluster of the
    summary.
- term: Word co-occurrence graph
  definition: >-
    An undirected weighted graph where nodes are unique content words (after stopword removal) and edges connect any two words
    that appear together within the same sentence, with edge weight equal to the number of sentences containing both words.
- term: Percolation-determined summary length (k*)
  definition: >-
    The number of sentences at which the GCC of the summary's vocabulary subgraph first exceeds a threshold fraction θ of
    the GCC of the full document's vocabulary graph. This is the proposed data-driven replacement for a fixed compression
    ratio.
- term: TF word score
  definition: >-
    Term Frequency score: the raw count of a content word within the document, used here to rank sentences by their aggregate
    importance (sum of TF scores of constituent content words).
- term: Compression ratio
  definition: >-
    The fraction of original sentences (or words) retained in the summary. Existing extractive methods treat this as a fixed
    input parameter (typically 10–30%); the proposed hypothesis determines it automatically via percolation.
summary: >-
  We hypothesize that the optimal summary length in word-frequency extractive summarization is determined by a percolation
  phase transition in the document's vocabulary co-occurrence network: sentences should be added (in TF-score order) until
  the summary's induced vocabulary subgraph forms a giant connected component, signaling complete conceptual coverage. This
  replaces the universally used but poorly motivated fixed compression ratio with a document-specific, mathematically grounded
  stopping criterion imported from statistical physics.
_relation_rationale: >-
  Same percolation-GCC frame; narrows success condition to vocabulary-segregated corpora after negative F1 on CNN/DM.
_confidence_delta: decreased
_key_changes:
- >-
  Primary claim demoted from 'higher ROUGE scores' to a corpus-conditional claim: percolation achieves recall advantage but
  F1 disadvantage on newswire corpora lacking vocabulary segregation.
- >-
  Added concrete empirical numbers from artifacts: percolation@θ=0.9 ROUGE-1 recall ≈0.853–0.880 vs fixed@30% ≈0.707–0.728;
  fixed@10% F1 ≈0.286 dominates percolation@θ=0.6 F1 ≈0.201–0.214.
- >-
  Identified structural reason for failure: CNN/DailyMail inverted-pyramid articles distribute vocabulary across sentences,
  so GCC fires homogeneously at ≈76–78% compression (std≈0.08), below the 10 pp discriminative threshold.
- >-
  Hypothesis now explicitly requires multi-topic or scientific corpora with vocabulary clustering for the core prediction
  (variable, document-adaptive compression) to hold.
- >-
  Tempered dynamic θ calibration claim: R²=0.10 is insufficient for reliable calibration on CNN/DailyMail; explicitly noted
  as open question for structured corpora.
- >-
  Added requirement for stronger baseline comparisons (TF-IDF scoring, TextRank/LexRank) in next iteration to isolate stopping
  criterion effect from scorer quality.
- >-
  Noted numerical discrepancy risk: final numbers should be reconciled against full artifact data (not mini samples) with
  explicit provenance statements.
- >-
  Success criteria revised: CONFIRMED requires (a) std > 10 pp compression variance AND improved F1 on a multi-topic corpus,
  or (b) percolation + stronger scorer (TF-IDF) achieves F1 parity with fixed baselines while providing variable-length adaptation.
relation_type: evolution
</current_hypothesis>

<all_artifacts>
Complete set of research artifacts across all iterations.

--- Item 1 ---
id: art_Zf8PEZKgv-j_
type: dataset
title: >-
  CNN/DailyMail 3.0.0 — 2000 News Article/Summary Pairs for Extractive Summarization
summary: |-
  Dataset: CNN/DailyMail 3.0.0 (abisee/cnn_dailymail on HuggingFace, 260k downloads, well-established NLP benchmark).

  Content: 2000 test-split examples from the CNN/DailyMail dataset. Each example has:
  - input: full news article text (mean 575 words, range 73–1846 words)
  - output: multi-sentence bullet-point highlights (mean 42 words, typically 2–5 sentences separated by newlines)
  - metadata_id: e.g. 'cnndm_test_0000'
  - metadata_task_type: 'summarization'
  - metadata_source_id: original SHA1 article id

  Why CNN/DailyMail over XSum: CNN/DM highlights are multi-sentence extractive-style summaries (mean 42 words) that map naturally to extractive evaluation; XSum summaries are single-sentence abstractive (mean 21 words) less suitable for word-frequency extractive methods. CNN/DM articles are also longer (575 vs 384 mean words), providing more signal for frequency-based sentence scoring.

  Format: JSON following exp_sel_data_out schema — {datasets: [{dataset, examples: [{input, output, metadata_*}]}]}. Schema validation: PASSED.

  Files:
  - full_data_out.json: 2000 examples (7.5MB, well under 100MB limit)
  - mini_data_out.json: first 3 examples (12KB)
  - preview_data_out.json: first 3 examples with strings truncated to 200 chars (2.4KB)
  - data.py: reproducible loader script using standard library + loguru
workspace_path: >-
  /ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_1/gen_art/gen_art_dataset_1
out_expected_files:
- data.py
- full_data_out.json
- preview_data_out.json
- mini_data_out.json

--- Item 2 ---
id: art_ZuUuuoy0rCRF
type: experiment
title: >-
  Percolation-Threshold Extractive Summarizer vs Fixed-Ratio Baselines on CNN/DailyMail
summary: >-
  Implemented and evaluated a percolation-threshold extractive summarizer against fixed-ratio baselines on 2000 CNN/DailyMail
  test documents. The percolation method builds a vocabulary co-occurrence graph from the document, then greedily adds TF-scored
  sentences until the giant connected component (GCC) of accumulated words reaches a critical fraction theta of the full document's
  GCC. Four theta values (0.6, 0.7, 0.8, 0.9) and three fixed-ratio baselines (10%, 20%, 30%) were evaluated on ROUGE-1/2/L
  F1. Key results (n=2000): Fixed-ratio baselines dominate on ROUGE — fixed_0.10 achieves mean ROUGE-1 F1=0.286, while best
  percolation (theta=0.6) achieves 0.201. Percolation compression ratios show substantial variance (theta=0.8: mean=0.629±0.092,
  range ~0.2-1.0), confirming the method adapts dynamically to document structure rather than using a fixed budget. Statistical
  tests confirm fixed baselines significantly outperform percolation (p<0.001). Correlation analysis between compression ratio
  and network properties (avg_degree, clustering coefficient) reveals how document graph topology influences percolation stopping
  behavior. The honest negative finding — that graph-theoretic stopping does not improve ROUGE over simple TF-ratio selection
  — is itself a meaningful scientific result about the disconnect between lexical coverage metrics and extractive quality
  metrics. All per-document results, aggregate statistics, statistical tests, and network property correlations are stored
  in method_out_raw.json and the schema-validated method_out.json.
workspace_path: >-
  /ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_1/gen_art/gen_art_experiment_1
out_expected_files:
- method.py
- full_method_out.json
- mini_method_out.json
- preview_method_out.json

--- Item 3 ---
id: art_yWTseob35O3B
type: evaluation
title: >-
  Statistical Evaluation: Percolation Threshold vs Fixed-Ratio ROUGE on CNN/DailyMail
summary: >-
  Evaluated percolation-threshold extractive summarization (4 thetas: 0.6/0.7/0.8/0.9) against fixed-ratio TF baselines (10%/20%/30%)
  on 2000 CNN/DailyMail test documents. Key findings: (1) ROUGE-1 recall improvement of +0.152 for percolation@0.9 vs fixed@30%
  (Wilcoxon Holm-corrected significant, p<0.05 for all rouge1_r comparisons at theta=0.9); (2) compression ratio std=0.077
  does NOT exceed the 10pp threshold — hypothesis partially supported; (3) network feature regression R²=0.10, confirming
  weak but non-zero predictability of compression from structural features (avg_degree and clustering_coefficient are top
  correlates); (4) segment analysis split by article length (short=CNN-proxy, long=DailyMail-proxy) shows consistent percolation
  advantage in both segments. The verdict is hypothesis_supported=False due to std<0.10, but ROUGE-1 recall improvement is
  significant and large. Percolation@0.9 is the best theta. Outputs include eval_out.json (validated against exp_eval_sol_out
  schema), method_out.json (per-document percolation and fixed results), and full/mini/preview variants. All 36 Wilcoxon tests
  reported with Holm-Bonferroni correction. Regression includes per-theta R² values. No LLM API costs ($0 spend).
workspace_path: >-
  /ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_1/gen_art/gen_art_evaluation_1
out_expected_files:
- eval.py
- full_eval_out.json
- mini_eval_out.json
- preview_eval_out.json

--- Item 4 ---
id: art_UfO3ZmlB5UKV
type: experiment
in_dependencies:
- id: art_Zf8PEZKgv-j_
  label: dataset
title: 'Percolation Summarization: TF/TF-IDF vs Fixed-Ratio on CNN/DM and Multi-News'
summary: |-
  Experiment: Percolation-threshold vs fixed-ratio extractive summarization on CNN/DailyMail (2000 docs) and Multi-News (500 docs), with TF and TF-IDF sentence scoring.

  Method: For each document, a word co-occurrence graph is built over all sentences. Sentences are added greedily (highest-scoring first) to a summary graph; the percolation threshold k* is the number of sentences needed for the summary graph's largest connected component (GCC) to reach fraction theta (0.6, 0.7, 0.8, 0.9) of the full document GCC. This is compared to fixed-ratio baselines (10%, 20%, 30% of sentences).

  Scorers: TF (term frequency) and TF-IDF (per-document, sentences as documents).

  Results structure:
  - per_document: list of per-doc records for cnndm and multi_news, each with ROUGE-1/2/L F/R/P scores for all 8 percolation variants (2 scorers × 4 thetas) and 6 fixed-ratio baselines (2 scorers × 3 ratios), plus graph structural features (avg_degree, clustering, density, full_gcc, n_sentences) and GCC growth decile curves.
  - aggregates: mean/std of ROUGE and compression_ratio per method variant per corpus, ceiling_hit_frac for percolation variants.
  - r2_analysis: R² of linear regression predicting compression_ratio from graph features (avg_degree, clustering, density, n_sentences, full_gcc) per scorer per theta.

  Key findings: Percolation k* compression ratio increases with theta (CR~0.38 at theta=0.6 to CR~0.85 at theta=0.9 for TF on CNN/DM); TF-IDF achieves lower compression ratios than TF at all thetas; fixed-ratio baselines show comparable ROUGE-1 F (~0.24-0.27) to percolation variants, providing a strong comparison baseline. Multi-News results show higher ceiling_hit rates due to longer multi-source documents.

  Files: method.py (full implementation), full_method_out.json / mini_method_out.json / preview_method_out.json.
workspace_path: >-
  /ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_2/gen_art/gen_art_experiment_1
out_expected_files:
- method.py
- full_method_out.json
- mini_method_out.json
- preview_method_out.json

--- Item 5 ---
id: art_ckvdjLNiSnrq
type: evaluation
in_dependencies:
- id: art_ZuUuuoy0rCRF
  label: reanalyzes
- id: art_Zf8PEZKgv-j_
  label: dataset
title: 'Percolation vs Fixed-Ratio Extractive Summarizer: CNN/DM + Multi-News Evaluation'
summary: |-
  Comprehensive evaluation of the percolation-threshold extractive summarizer vs fixed-ratio baselines on CNN/DailyMail (2000 docs, loaded from iter1 experiment raw results) and Multi-News (1000 docs, downloaded and processed fresh). Eight metrics computed:

  1. ROUGE-1/2/L F1, Precision, Recall (mean±std) for all 4 percolation thetas (0.6–0.9) and 3 fixed ratios (10%/20%/30%) on both corpora. CNN/DM: best percolation (theta=0.6) ROUGE-1 F1=0.201 vs fixed_0.1 F1=0.286. Multi-News (TF scorer): percolation theta=0.6 ROUGE-1 F1≈0.28.

  2. CEILING FRACTION: fraction of docs where percolation selects all sentences (compression_ratio=1.0). CNN/DM theta=0.9: ~8.5% ceiling fraction; lower thetas near 0.

  3. COMPRESSION RATIO STD: CNN/DM std=0.081–0.092 per theta (below 10pp threshold); Multi-News shows similar range.

  4. Wilcoxon signed-rank tests with Holm-Bonferroni correction across all 12 pairwise comparisons (percolation vs fixed-ratio on ROUGE-1 F1). All comparisons confirm fixed baselines significantly outperform percolation (all p<0.001 post-correction).

  5. OLS REGRESSION: compression_ratio ~ [avg_degree, clustering_coefficient, n_sentences, full_gcc_size] with per-feature 95% CIs and partial R². R²≈0.07–0.10 across thetas, with n_sentences having highest partial R².

  6. CALIBRATION: theta=0.6 gives mean compression≈0.42 (closest to 30% target), theta=0.9 gives≈0.76; no theta reliably proxies 10% target.

  7. CROSS-CORPUS COMPARISON: Multi-News vs CNN/DM compression std at same thetas via Levene test.

  8. TF-IDF SCORER: full comparison of TF vs TF-IDF sentence scoring on Multi-News percolation and fixed-ratio methods, including Wilcoxon tests per theta.

  Key finding: Fixed-ratio baselines consistently dominate ROUGE-F1 due to precision advantage; percolation achieves higher recall but lower precision. The F1 gap is structural (not scorer-specific), confirmed on both corpora and with both TF and TF-IDF scorers. Compression ratio variance is ~9pp on CNN/DM and ~8–12pp on Multi-News, near but not clearly above the 10pp hypothesis threshold.
workspace_path: >-
  /ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_2/gen_art/gen_art_evaluation_1
out_expected_files:
- eval.py
- full_eval_out.json
- mini_eval_out.json
- preview_eval_out.json
</all_artifacts>

<new_artifacts_this_iteration>
These 2 artifacts were created THIS iteration.

id: art_UfO3ZmlB5UKV
type: experiment
in_dependencies:
- id: art_Zf8PEZKgv-j_
  label: dataset
title: 'Percolation Summarization: TF/TF-IDF vs Fixed-Ratio on CNN/DM and Multi-News'
summary: |-
  Experiment: Percolation-threshold vs fixed-ratio extractive summarization on CNN/DailyMail (2000 docs) and Multi-News (500 docs), with TF and TF-IDF sentence scoring.

  Method: For each document, a word co-occurrence graph is built over all sentences. Sentences are added greedily (highest-scoring first) to a summary graph; the percolation threshold k* is the number of sentences needed for the summary graph's largest connected component (GCC) to reach fraction theta (0.6, 0.7, 0.8, 0.9) of the full document GCC. This is compared to fixed-ratio baselines (10%, 20%, 30% of sentences).

  Scorers: TF (term frequency) and TF-IDF (per-document, sentences as documents).

  Results structure:
  - per_document: list of per-doc records for cnndm and multi_news, each with ROUGE-1/2/L F/R/P scores for all 8 percolation variants (2 scorers × 4 thetas) and 6 fixed-ratio baselines (2 scorers × 3 ratios), plus graph structural features (avg_degree, clustering, density, full_gcc, n_sentences) and GCC growth decile curves.
  - aggregates: mean/std of ROUGE and compression_ratio per method variant per corpus, ceiling_hit_frac for percolation variants.
  - r2_analysis: R² of linear regression predicting compression_ratio from graph features (avg_degree, clustering, density, n_sentences, full_gcc) per scorer per theta.

  Key findings: Percolation k* compression ratio increases with theta (CR~0.38 at theta=0.6 to CR~0.85 at theta=0.9 for TF on CNN/DM); TF-IDF achieves lower compression ratios than TF at all thetas; fixed-ratio baselines show comparable ROUGE-1 F (~0.24-0.27) to percolation variants, providing a strong comparison baseline. Multi-News results show higher ceiling_hit rates due to longer multi-source documents.

  Files: method.py (full implementation), full_method_out.json / mini_method_out.json / preview_method_out.json.
workspace_path: >-
  /ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_2/gen_art/gen_art_experiment_1
out_expected_files:
- method.py
- full_method_out.json
- mini_method_out.json
- preview_method_out.json

id: art_ckvdjLNiSnrq
type: evaluation
in_dependencies:
- id: art_ZuUuuoy0rCRF
  label: reanalyzes
- id: art_Zf8PEZKgv-j_
  label: dataset
title: 'Percolation vs Fixed-Ratio Extractive Summarizer: CNN/DM + Multi-News Evaluation'
summary: |-
  Comprehensive evaluation of the percolation-threshold extractive summarizer vs fixed-ratio baselines on CNN/DailyMail (2000 docs, loaded from iter1 experiment raw results) and Multi-News (1000 docs, downloaded and processed fresh). Eight metrics computed:

  1. ROUGE-1/2/L F1, Precision, Recall (mean±std) for all 4 percolation thetas (0.6–0.9) and 3 fixed ratios (10%/20%/30%) on both corpora. CNN/DM: best percolation (theta=0.6) ROUGE-1 F1=0.201 vs fixed_0.1 F1=0.286. Multi-News (TF scorer): percolation theta=0.6 ROUGE-1 F1≈0.28.

  2. CEILING FRACTION: fraction of docs where percolation selects all sentences (compression_ratio=1.0). CNN/DM theta=0.9: ~8.5% ceiling fraction; lower thetas near 0.

  3. COMPRESSION RATIO STD: CNN/DM std=0.081–0.092 per theta (below 10pp threshold); Multi-News shows similar range.

  4. Wilcoxon signed-rank tests with Holm-Bonferroni correction across all 12 pairwise comparisons (percolation vs fixed-ratio on ROUGE-1 F1). All comparisons confirm fixed baselines significantly outperform percolation (all p<0.001 post-correction).

  5. OLS REGRESSION: compression_ratio ~ [avg_degree, clustering_coefficient, n_sentences, full_gcc_size] with per-feature 95% CIs and partial R². R²≈0.07–0.10 across thetas, with n_sentences having highest partial R².

  6. CALIBRATION: theta=0.6 gives mean compression≈0.42 (closest to 30% target), theta=0.9 gives≈0.76; no theta reliably proxies 10% target.

  7. CROSS-CORPUS COMPARISON: Multi-News vs CNN/DM compression std at same thetas via Levene test.

  8. TF-IDF SCORER: full comparison of TF vs TF-IDF sentence scoring on Multi-News percolation and fixed-ratio methods, including Wilcoxon tests per theta.

  Key finding: Fixed-ratio baselines consistently dominate ROUGE-F1 due to precision advantage; percolation achieves higher recall but lower precision. The F1 gap is structural (not scorer-specific), confirmed on both corpora and with both TF and TF-IDF scorers. Compression ratio variance is ~9pp on CNN/DM and ~8–12pp on Multi-News, near but not clearly above the 10pp hypothesis threshold.
workspace_path: >-
  /ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_2/gen_art/gen_art_evaluation_1
out_expected_files:
- eval.py
- full_eval_out.json
- mini_eval_out.json
- preview_eval_out.json
</new_artifacts_this_iteration>

<current_paper>
The paper draft from this iteration — represents the current state of the research story.

# Introduction

Extractive text summarization selects a subset of source sentences to form a summary. The central algorithmic question is *which* sentences to select, but an equally critical and consistently underexamined question is *how many* to select. Virtually every published method—from Luhn's frequency-based sentence scoring [1] to graph-based systems such as TextRank [2] and LexRank [3], to neural approaches such as SummaRuNNer [8], BERTSum [9], and PreSumm [10]—requires the user or system to specify a compression ratio as an external parameter, typically fixed at 20–30% by convention or tuned on held-out data.

This fixed-ratio convention is poorly motivated. A news article covering a single self-contained event may be complete in two or three sentences; a multi-topic feature article of the same length may require ten to communicate its full scope. Applying a uniform 20% compression to both yields either a redundant summary (for focused articles) or a lossy one (for multi-topic articles). The problem is broadly acknowledged [2, 3] but no principled, document-specific stopping criterion has emerged in the extractive summarization literature.

We import a stopping criterion from statistical physics. In Erdős–Rényi random graph theory [4], a network undergoes a sharp phase transition as edges are added: below a critical density, the graph consists of many small disconnected components; above it, a *giant connected component* (GCC) suddenly emerges spanning most nodes [5]. Transposed to text, the vocabulary co-occurrence graph of a growing summary starts fragmented when only a few sentences have been added (covering isolated topic clusters) and becomes globally connected as conceptually bridging sentences are included. The percolation threshold—where the summary's induced GCC reaches a critical fraction θ of the full document's GCC—signals that the summary has achieved conceptual percolation: all major vocabulary clusters are mutually reachable.

The vocabulary co-occurrence graph of a document is not an Erdős–Rényi random graph—it has strong community structure, heavy-tailed degree distributions, and high clustering coefficients. The classical percolation threshold therefore does not apply analytically, and θ must be tuned empirically. What the analogy provides is a *principled directionality*: once the GCC fraction exceeds θ, the summary's vocabulary skeleton covers the document's. This criterion is document-specific, requires no training data or language models, and is computationally inexpensive.

We evaluate this criterion on two corpora with TF and TF-IDF sentence scoring. On CNN/DailyMail, the percolation criterion consistently selects 60–87% of sentences, far above the 7–10% targeted by journalist-written highlights, and fixed-ratio baselines dominate on ROUGE F1. On Multi-News, the same criterion fires much earlier (35–74%), reflecting stronger vocabulary segregation in multi-source documents. Nevertheless, fixed ratios dominate F1 on both corpora because reference summaries are concise highlights, not coverage-complete extracts. These results delineate precisely when percolation-based length selection can and cannot succeed.

[FIGURE:fig_pipeline]

**Summary of Contributions.** (1) We introduce the first extractive summarizer using vocabulary co-occurrence graph percolation as a self-calibrating stopping criterion, with a complete open-source implementation (§3). (2) We provide a rigorous evaluation on CNN/DailyMail (n=2000) and Multi-News (n=1000), comparing four percolation thresholds against three fixed-ratio baselines with both TF and TF-IDF sentence scoring, with Holm-corrected Wilcoxon tests across all pairwise comparisons (§4). [ARTIFACT:art_UfO3ZmlB5UKV] (3) We report and analyze a structured negative finding: percolation maximizes recall but not F1, and we identify the structural reasons—newswire vocabulary distribution in CNN/DailyMail and the conciseness of all human reference summaries (§5). (4) We characterize how the percolation firing point varies across corpora (§4.3), showing that Multi-News's multi-source vocabulary clustering produces meaningfully different behavior from CNN/DailyMail's inverted-pyramid structure. (5) We quantify that graph structural features explain only R²≈0.12 of compression variance, insufficient for reliable dynamic calibration (§4.4).

# Related Work

**Extractive summarization.** Luhn [1] introduced frequency-based sentence scoring. TextRank [2] applies PageRank [7] over a sentence-similarity graph and selects the top-*k* sentences; LexRank [3] uses eigenvector centrality over cosine-similarity sentences. Both require a fixed *k*. SummaRuNNer [8] uses a recurrent neural network to score sentences sequentially with a fixed budget. BERTSum [9] and PreSumm [10] apply pretrained transformer encoders to sentence scoring. Neural abstractive methods, including pointer-generator networks [11], similarly rely on externally specified length constraints. None of these methods determines *k* from the document's intrinsic structure.

Ryang and Abekawa [12] use an influence propagation model on a word co-occurrence network to maximize coverage under a budget constraint. The budget remains manually specified; the phase transition is not used to determine when coverage is complete. Our work shares the word-network foundation but replaces the manual budget with the percolation threshold.

**Graph percolation in NLP.** Erdős–Rényi percolation theory [4, 5] has been applied to information diffusion in social networks and language evolution, but to our knowledge has not previously been proposed as a stopping criterion for extractive summarization.

**CNN/DailyMail and Multi-News benchmarks.** Hermann et al. [6] introduced CNN/DailyMail for reading comprehension; it has since become the standard single-document summarization benchmark [11]. The reference summaries are journalist-written bullet-point highlights (mean 42 words). Fabbri et al. [13] introduced Multi-News, a multi-document corpus aggregating 2–10 news articles per story with a single reference summary. The multi-source structure gives Multi-News stronger vocabulary segregation across its constituent documents, making it a natural testbed for percolation-based stopping.

# Methods

## Vocabulary Co-occurrence Graph Construction

Given a document with sentences $s_1, \ldots, s_n$, we preprocess each sentence by tokenizing, lowercasing, removing NLTK English stop words, and retaining alphabetic tokens of length ≥ 2. Let $W_i$ denote the content word set of sentence $s_i$.

The vocabulary co-occurrence graph $G = (V, E, w)$ is an undirected weighted graph. Nodes $V$ are unique content words across the document. An edge $(u, v) \in E$ exists between any two words $u, v \in W_i$ for some sentence $s_i$; the edge weight $w(u,v)$ equals the number of sentences in which $u$ and $v$ co-occur. The full-document GCC, $\text{GCC}_{\text{doc}}$, is the node count of the largest connected component of $G$. [ARTIFACT:art_ZuUuuoy0rCRF]

## Sentence Scoring

We evaluate two sentence scoring functions. The *TF scorer* ranks sentence $s_i$ by $\text{score}(s_i) = \sum_{w \in W_i} \text{TF}(w, D)$, where $\text{TF}(w,D)$ is the raw term frequency of word $w$ in document $D$. The *TF-IDF scorer* uses $\text{score}(s_i) = \sum_{w \in W_i} \text{TF}(w, s_i) \cdot \text{IDF}(w)$, treating each sentence as a sub-document and computing IDF over all sentences in the document.

## Percolation-Threshold Stopping Criterion

Given threshold $\theta \in (0,1)$, we add sentences greedily in descending score order. After each addition, we update the *induced subgraph* $G_S$ over accumulated content words $\bigcup_{s_j \in S} W_j$ and compute $\text{GCC}_{\text{summary}}$ as the largest connected component of $G_S$. The summary grows until:
$$\frac{\text{GCC}_{\text{summary}}}{\text{GCC}_{\text{doc}}} \geq \theta$$
If the ratio never reaches $\theta$, all sentences are retained. We evaluate $\theta \in \{0.6, 0.7, 0.8, 0.9\}$, yielding eight percolation variants (2 scorers × 4 thresholds).

## Fixed-Ratio Baselines

Baselines use the same sentence scoring and ordering as the percolation method. For ratio $r \in \{0.10, 0.20, 0.30\}$, the summary consists of the top $\lceil r \cdot n \rceil$ sentences. This isolates the effect of the stopping criterion from sentence ranking.

# Experiments

## Datasets and Evaluation

We use two corpora. CNN/DailyMail 3.0.0 (2000 test documents; mean 575 words; reference summaries mean 42 words, 2–5 sentences) [6]. [ARTIFACT:art_Zf8PEZKgv-j_] Multi-News test split (1000 documents; multi-source, one reference summary per story) [13]. Multi-News reference summaries are longer than CNN/DailyMail highlights (approximately 250 words), reflecting the multi-source aggregation task. We evaluate with ROUGE-1/2/L recall (R), precision (P), and F1 (F). Statistical comparisons use two-sided Wilcoxon signed-rank tests with Holm–Bonferroni correction over all pairwise comparisons. All reported values are from the final full-run artifacts (provenance: art_UfO3ZmlB5UKV for CNN/DailyMail n=2000 per-document results; art_ckvdjLNiSnrq for Multi-News n=1000 results).

## Main Results on CNN/DailyMail

Table 1 reports mean ROUGE scores on CNN/DailyMail (n=2000). [ARTIFACT:art_UfO3ZmlB5UKV] The results expose a structural tension between recall and precision.

| Method | ROUGE-1 R | ROUGE-1 P | ROUGE-1 F1 | ROUGE-2 F1 | CR mean |
|---|---|---|---|---|---|
| TF perc. θ=0.6 | 0.772 | 0.112 | 0.187 | 0.092 | 0.597 |
| TF perc. θ=0.7 | 0.808 | 0.101 | 0.173 | 0.087 | 0.686 |
| TF perc. θ=0.8 | 0.834 | 0.092 | 0.161 | 0.084 | 0.773 |
| TF perc. θ=0.9 | 0.858 | 0.085 | 0.150 | 0.080 | 0.866 |
| TF-IDF perc. θ=0.6 | 0.743 | 0.126 | **0.205** | 0.092 | 0.369 |
| TF-IDF perc. θ=0.7 | 0.784 | 0.112 | 0.188 | 0.089 | 0.463 |
| TF fixed@10% | 0.296 | 0.313 | 0.274 | 0.092 | 0.100 |
| TF fixed@20% | 0.468 | 0.233 | **0.284** | 0.107 | 0.200 |
| TF fixed@30% | 0.578 | 0.183 | 0.259 | 0.107 | 0.300 |

*Table 1: Mean ROUGE scores on CNN/DailyMail (n=2000). CR = compression ratio (fraction of sentences selected). Bold indicates best ROUGE-1 F1 in each scorer category.*

**Recall.** At θ=0.9, TF percolation achieves ROUGE-1 recall of 0.858 versus 0.578 for fixed@30% (Δ=+0.280). All percolation variants significantly outperform all fixed-ratio baselines on recall (Wilcoxon, all Holm-corrected p<0.001). Percolation summaries contain more reference vocabulary because they include more sentences.

**F1.** Fixed-ratio summaries dominate on F1. TF fixed@20% achieves ROUGE-1 F1=0.284, versus 0.205 for the best percolation variant (TF-IDF, θ=0.6). The fixed baselines' precision advantage more than compensates for lower recall. The best TF percolation variant (θ=0.6) achieves F1=0.187—34% below the fixed@20% F1.

**Effect of TF-IDF scoring.** TF-IDF selects fewer sentences than TF at the same θ: CR mean of 0.369 vs. 0.597 at θ=0.6, because TF-IDF up-weights document-specific rare words, which tend to cluster in fewer sentences. This lower compression raises F1 to 0.205 for TF-IDF percolation θ=0.6, a 9.6% absolute improvement over TF percolation at the same θ. However, it does not close the gap to fixed-ratio F1 (0.284): the scorer interacts with the stopping criterion, but the stopping criterion remains the binding constraint.

**Ceiling fraction.** For all percolation thresholds and scorers on CNN/DailyMail, the fraction of documents where percolation selects all sentences is 0.0%. The GCC ratio reaches every tested θ value in every document, confirming that the sparse-graph edge case does not arise in this corpus.

[FIGURE:fig_rouge_cnndm]

## Multi-News Evaluation

Table 2 reports ROUGE-1 results on Multi-News (TF scorer; n=1000). [ARTIFACT:art_ckvdjLNiSnrq]

| Method | ROUGE-1 R | ROUGE-1 F1 | CR mean | CR std |
|---|---|---|---|---|
| TF perc. θ=0.6 | 0.738 | 0.280 | 0.345 | 0.020 |
| TF perc. θ=0.7 | 0.770 | 0.253 | 0.457 | 0.046 |
| TF perc. θ=0.8 | 0.796 | 0.226 | 0.601 | 0.030 |
| TF perc. θ=0.9 | 0.817 | 0.207 | 0.742 | 0.019 |
| TF fixed@10% | 0.528 | **0.385** | 0.100 | — |
| TF fixed@20% | 0.665 | 0.339 | 0.200 | — |
| TF fixed@30% | 0.726 | 0.296 | 0.300 | — |

*Table 2: Mean ROUGE-1 scores on Multi-News (n=1000, TF scorer). Fixed-ratio F1 dominates, but percolation fires substantially earlier than on CNN/DailyMail.*

Two findings distinguish Multi-News from CNN/DailyMail. First, percolation fires at a meaningfully lower compression ratio: θ=0.6 yields CR=0.345 on Multi-News versus CR=0.597 on CNN/DailyMail. This supports the theoretical prediction that documents with distinct vocabulary clusters (here, per-source articles merged into one document) trigger the GCC criterion earlier, requiring fewer sentences to connect all vocabulary communities. Second, the F1 gap between percolation and fixed-ratio baselines narrows: the gap between percolation θ=0.6 F1 and fixed@30% F1 is 0.016 on Multi-News (0.296−0.280) compared to 0.072 on CNN/DailyMail (0.259−0.187). The percolation criterion is thus less mis-calibrated on Multi-News, but fixed ratios still win.

[FIGURE:fig_cross_corpus]

## Compression Ratio Analysis

Compression ratios on CNN/DailyMail cluster tightly: std=0.075–0.077 across TF thresholds, and std=0.083–0.086 for TF-IDF. All standard deviations fall below the 10 percentage point discriminative threshold, indicating that the GCC fires at a consistent fraction of document sentences regardless of article length or topic diversity within this corpus. The interquartile range for TF θ=0.9 spans approximately 81–93% of sentences.

On Multi-News, the compression variance remains narrow at most thresholds (std=0.020 at θ=0.6 and 0.019 at θ=0.9), with one exception at θ=0.7 (std=0.046). Neither corpus shows std > 10pp across most thresholds. The variance finding does not support the theoretical prediction of document-adaptive length selection.

Note that no tested θ produces mean compression near the 10% target of reference highlights on either corpus. The minimum mean compression is 34.5% (Multi-News, θ=0.6) and 36.9% (CNN/DailyMail, TF-IDF θ=0.6). To match reference-summary length (≈10% on CNN/DailyMail), a GCC fraction of θ≪0.6 would be required—a regime in which the GCC captures only a small fraction of the document vocabulary and provides little coverage signal.

## Network Feature Regression

Linear regression predicts the percolation compression ratio from five vocabulary graph features: average degree, clustering coefficient, graph density, number of sentences, and full GCC size. On CNN/DailyMail (TF scorer), R²=0.117 at θ=0.6, 0.127 at θ=0.7, and 0.134 at θ=0.8, with graph density as the strongest positive predictor (coefficient 0.546–0.596) and n_sentences as the strongest negative predictor (coefficient −0.001). The R² values indicate a statistically significant but modest relationship: graph structure explains only 12–13% of compression variance, which is insufficient for reliable dynamic θ calibration. Proposed future directions for dynamic calibration would need a substantially stronger predictor (R²≥0.5) to outperform a fixed θ choice.

# Discussion

## Why Percolation Does Not Improve F1

The core negative finding—confirmed on both CNN/DailyMail and Multi-News—is that percolation-determined summaries are too long relative to human-written reference summaries. CNN/DailyMail reference summaries span approximately 7–10% of a typical article; Multi-News references, while longer in absolute terms, remain concise relative to the merged multi-source length. The percolation threshold correctly signals when the vocabulary graph becomes *connected enough*, but this connectivity criterion does not correspond to the *salience threshold* implicit in human-written highlights.

The mismatch is structural and not specific to TF scoring: replacing TF with TF-IDF reduces compression ratios moderately but does not align percolation with the brevity of reference summaries. TextRank and LexRank (not implemented in the current experiments but cited as methodological references) also require a fixed *k* and would face the same challenge: using percolation as their stopping criterion would produce the same vocabulary-coverage-driven compression ratios.

The finding that no tested θ produces ≈10% compression on CNN/DailyMail reveals a deeper incompatibility: the percolation criterion requires a non-trivial connected vocabulary structure, which cannot be achieved from 10% of sentences in typical newswire articles. A vocabulary graph sampled from 10% of sentences is, in almost all cases, highly disconnected—far below any reasonable GCC threshold. The criterion is therefore structurally unable to produce short summaries on this corpus.

## Corpus-Conditional Behavior: A Theoretical Confirmation

The cross-corpus comparison provides partial theoretical validation. Multi-News documents aggregate articles on the same story from multiple sources; each source article introduces source-specific vocabulary that forms a relatively isolated vocabulary cluster. Merging 2–10 source articles into one document creates a multi-cluster vocabulary graph in which the GCC can be achieved by selecting a small number of inter-source bridging sentences. This is precisely the vocabulary segregation scenario the hypothesis predicted would cause the GCC to fire earlier—and indeed, θ=0.6 fires at CR=0.345 on Multi-News versus CR=0.597 on CNN/DailyMail.

However, even on Multi-News the compression variance remains narrow (std≤0.046 at all θ values). The within-corpus homogeneity reflects that all Multi-News documents share the same multi-source aggregation structure, so the GCC fires at a structurally consistent point. True document-adaptive compression would require a *corpus-level diversity* in document types—mixing single-topic and multi-topic documents—rather than corpus-level homogeneity in one type.

## Implications for Percolation-Based Length Selection

The results suggest three conditions under which a percolation stopping criterion could be practically useful. First, the corpus should contain documents with diverse vocabulary clustering structure (not homogeneous newswire or homogeneous multi-source). Second, reference summaries should aim at coverage rather than concise highlights—scientific review articles or technical documentation summaries, where completeness is valued over brevity, may be better aligned with the GCC criterion. Third, the percolation criterion's recall advantage (+0.280 on CNN/DailyMail at θ=0.9) is directly useful for retrieval or recall-oriented applications where missing content is more costly than verbosity.

## Limitations

- *Scorer coverage.* Only TF and TF-IDF scoring are implemented. TextRank and LexRank are discussed in Related Work but not evaluated as scorers. Sentence-level graph-based scorers may interact differently with the percolation stopping criterion—their graph structure partially overlaps with the vocabulary GCC, potentially creating complementary or interfering signals.
- *R²=0.12 is insufficient for dynamic calibration.* The regression analysis shows that graph structural features explain only 12–13% of compression variance. Proposals to dynamically calibrate θ from graph features therefore cannot be expected to outperform a fixed θ choice; a much stronger feature representation (R²≥0.5) would be needed.
- *GCC definition.* The fraction θ is applied to the GCC node count. Alternative measures—edge density, spectral gap, modularity—may fire at different, more appropriate points. The GCC node fraction is one of several possible percolation signals.
- *ER model caveat.* The classical Erdős–Rényi percolation transition provides the motivating intuition for the method, but vocabulary co-occurrence graphs are not ER graphs: they have high clustering, heavy-tailed degree distributions, and community structure. The formal percolation threshold of an ER graph does not apply analytically, and θ must be treated as an empirical hyperparameter.

# Conclusion

We proposed and evaluated the first extractive summarizer to use vocabulary co-occurrence graph percolation as a self-calibrating length criterion. The method selects sentences (by TF or TF-IDF score) until the summary's induced vocabulary subgraph forms a giant connected component reaching fraction θ of the full document's GCC. On CNN/DailyMail (n=2000), percolation at θ=0.9 achieves ROUGE-1 recall of 0.858 versus 0.578 for the best fixed-ratio baseline, while fixed@20% dominates on F1 (0.284 vs. 0.205 for the best percolation variant). On Multi-News (n=1000), percolation fires substantially earlier (CR=0.345 at θ=0.6 vs. CR=0.597 on CNN/DailyMail), confirming the theoretical prediction about vocabulary segregation in multi-source documents—yet fixed-ratio F1 still dominates (fixed@10%: 0.385 vs. percolation θ=0.6: 0.280). The failure mode is structural: the GCC criterion targets coverage completeness, whereas human-written highlights target salience and brevity. Network features explain only R²≈0.12 of compression variance, ruling out reliable dynamic calibration from graph topology alone.

Future work should evaluate the percolation criterion on corpora that combine documents of diverse vocabulary clustering—multi-topic review articles or heterogeneous collections that mix focused and broad documents within a single evaluation set—where the criterion may provide document-adaptive compression rather than corpus-level homogeneous compression. Replacing the GCC node count with richer percolation signals (spectral gap, modularity) and applying the criterion to recall-oriented retrieval settings rather than F1-optimized summarization may also yield settings where the percolation criterion's coverage-maximizing behavior is directly beneficial.

# References

[1] H. P. Luhn. The automatic creation of literature abstracts. *IBM Journal of Research and Development*, 2(2):159–165, 1958.

[2] R. Mihalcea and P. Tarau. TextRank: Bringing order into texts. In *Proceedings of EMNLP 2004*, pages 404–411, 2004.

[3] G. Erkan and D. R. Radev. LexRank: Graph-based lexical centrality as salience in text summarization. *Journal of Artificial Intelligence Research*, 22:457–479, 2004.

[4] P. Erdős and A. Rényi. On the evolution of random graphs. *Publications of the Mathematical Institute of the Hungarian Academy of Sciences*, 5:17–60, 1960.

[5] B. Bollobás. *Random Graphs*, 2nd ed. Cambridge University Press, 2001.

[6] K. M. Hermann et al. Teaching machines to read and comprehend. In *Advances in NeurIPS*, 2015.

[7] L. Page, S. Brin, R. Motwani, and T. Winograd. The PageRank citation ranking: Bringing order to the web. Technical Report 1999-66, Stanford InfoLab, 1999.

[8] R. Nallapati, F. Zhai, and B. Zhou. SummaRuNNer: A recurrent neural network based sequence model for extractive summarization of documents. In *AAAI*, 2017.

[9] Y. Liu. Fine-tune BERT for extractive summarization. *arXiv:1903.10318*, 2019.

[10] Y. Liu and M. Lapata. Text summarization with pretrained encoders. In *EMNLP*, 2019.

[11] A. See, P. J. Liu, and C. D. Manning. Get to the point: Summarization with pointer-generator networks. In *ACL*, pages 1073–1083, 2017.

[12] S. Ryang and T. Abekawa. Framework of automatic text summarization using reinforcement learning. In *EMNLP-CoNLL*, pages 256–265, 2012.

[13] A. R. Fabbri, I. Li, T. She, S. Li, and D. R. Radev. Multi-News: A large-scale multi-document summarization dataset and abstractive hierarchical model. In *ACL*, pages 1074–1084, 2019.
</current_paper>

<reviewer_feedback>
Feedback from the paper reviewer this iteration.

- [MAJOR] (rigor) The paper reports R²≈0.12 for the network feature regression and concludes that 'graph structural features explain only 12–13% of compression variance, insufficient for reliable dynamic calibration' (§4.4, §5.4, §6). This conclusion is drawn exclusively from TF-scorer results, but the artifact (art_UfO3ZmlB5UKV, r2_analysis.cnndm.tfidf) shows TF-IDF R²=0.315 at θ=0.6, 0.274 at θ=0.7, 0.261 at θ=0.8, and 0.221 at θ=0.9. These values are 2–3× larger than the TF values and were not reported anywhere in the paper. At R²=0.315, graph density explains ~31% of TF-IDF compression variance—a nontrivial predictive relationship. The paper's blanket conclusion that 'dynamic θ calibration cannot be expected to outperform a fixed θ choice' is incorrect for TF-IDF and misleads readers about the practical prospects of adaptive stopping.
  Action: Report TF-IDF R² values in §4.4 alongside TF values. Add a 2-column table (TF vs TF-IDF) with R² for all four θ values. Revise the conclusion to distinguish scorer-specific behavior: 'For TF scoring, R²≈0.12 is insufficient for reliable calibration; for TF-IDF scoring, R²≈0.22–0.32 suggests moderate predictability that warrants further investigation.' Also revise §5.4 and the Conclusion accordingly.
- [MAJOR] (scope) Table 2 reports only TF scorer results for Multi-News, despite the paper now evaluating TF-IDF on CNN/DM. This creates an inconsistency: the cross-corpus comparison (§4.2, §5.2) cannot determine whether the earlier Multi-News percolation firing is scorer-independent or scorer-specific. The artifact (art_ckvdjLNiSnrq) contains full TF-IDF Multi-News results, so there is no computational barrier.
  Action: Add TF-IDF columns to Table 2, or at minimum add a 2-row sub-table for TF-IDF percolation θ=0.6 and fixed@10% on Multi-News, so the cross-scorer comparison can be made on both corpora symmetrically.
- [MINOR] (rigor) §4.3 states 'std=0.075–0.077 across TF thresholds' for CNN/DailyMail. The artifact shows TF compression ratio std: θ=0.6→0.075, θ=0.7→0.077, θ=0.8→0.075, θ=0.9→0.073. The θ=0.9 value (0.073) falls below the stated minimum of 0.075. Similarly, TF-IDF std reaches 0.087 at θ=0.8 but the paper says 0.083–0.086. These are small but real discrepancies.
  Action: Correct the std ranges to: TF 0.073–0.077, TF-IDF 0.083–0.087. This is a one-line fix in §4.3.
- [MINOR] (rigor) §4.4 states graph density is the strongest predictor with 'coefficient 0.546–0.596' for TF scorer. The artifact shows TF coefficients: θ=0.6→0.546, θ=0.7→0.596, θ=0.8→0.629, θ=0.9→0.681. The reported range covers only θ=0.6 and θ=0.7 but omits the larger values at θ=0.8 and θ=0.9.
  Action: Correct the coefficient range to 0.546–0.681 to cover all four θ values, or report all four coefficients explicitly.
- [MINOR] (methodology) The paper discusses TextRank and LexRank in §2 as natural comparators (they also build word/sentence graphs) but neither is evaluated. §5.1 acknowledges this as a limitation but frames it as optional. For a paper whose contribution is specifically about graph-based stopping criteria, not comparing against graph-based scoring methods leaves the most natural design-space question unanswered: does combining graph-based scoring with graph-based stopping create synergistic coverage, or are the two graph signals redundant?
  Action: Add TextRank@k (where k is the percolation threshold) as one additional scorer variant. The implementation can reuse the existing word co-occurrence graph infrastructure. If compute is limited, evaluate on 500 documents from CNN/DM. Report whether graph-scored + percolation-stopped summaries perform comparably to TF/TF-IDF variants, and whether they exhibit the same recall-precision trade-off.
- [MINOR] (novelty) The core finding—that percolation selects 60–87% of sentences on CNN/DM, far exceeding the 7–10% of journalist highlights, and thus achieves high recall but low F1—is largely predictable from the definition of the GCC threshold and the known properties of CNN/DM highlights. The paper's Discussion (§5.1) acknowledges the mismatch but presents it as an empirical discovery rather than a pre-derivable consequence. A deeper analysis of why TF-IDF fires at CR=0.369 while TF fires at CR=0.597 on the same documents—including which sentence types trigger the GCC first—would provide genuinely non-obvious insight.
  Action: Add a brief analysis (3–5 sentences + one example) in §4.1 or §5.1 showing what type of sentence drives the GCC to cross the threshold first under TF-IDF vs. TF scoring. For instance: 'Does TF-IDF percolation tend to select topic-bridging sentences that connect vocabulary communities earlier?' This would convert the scoring comparison from a quantitative observation to a mechanistic explanation.
- [MINOR] (clarity) The statistical tests section references all pairwise comparisons and p<0.001, but the preview of art_ckvdjLNiSnrq shows statistical_tests_cnndm.comparisons = [] (empty list). The full artifact may have results, but this raises questions about whether all claimed Wilcoxon tests were actually computed and stored.
  Action: Verify that the full_eval_out.json contains all 36+ pairwise Wilcoxon test results. Add a supplementary table (or appendix) reporting the key test statistics (W-statistic, corrected p-value) for the most important comparisons (percolation θ=0.6 vs. fixed@20% on ROUGE-1 F1, and percolation θ=0.9 vs. fixed@30% on ROUGE-1 recall) to make reproducibility easier.
</reviewer_feedback>



<task>
IMPORTANT: Your ONLY output is the revised hypothesis text. Do NOT run code, produce artifacts,
fix bugs, or attempt to address the evidence yourself — the next iteration of the invention loop
will generate fresh artifacts based on your revised hypothesis. Reflect and rewrite; nothing else.

Do NOT generate a completely new hypothesis. Take the current hypothesis and REVISE it
to incorporate new evidence. Keep the core idea — refine, narrow, or strengthen it.

1. Does the evidence support the hypothesis? Narrow or broaden scope as needed.
2. Which claims now have strong evidence? Which are still unsupported?
3. Should the hypothesis become more specific based on what we've learned?
4. If reviewer feedback is provided, address the critiques directly.

STABILITY IS OK: If progress is good and evidence supports the current direction, keep the
hypothesis similar or identical. Only make substantive changes when evidence clearly calls for
them — e.g., contradictory results, fundamental reviewer critiques, or findings that refine scope.

You must also classify two kinds of edges in the research trace:

(A) The H↔H edge — how does this revised hypothesis relate to the previous one?
    Set `relation_type` (Moulines's structuralist typology) to one of:
    - "evolution": refining specialised claims, same conceptual frame
    - "embedding": previous hypothesis is now a special case of a broader frame
    - "replacement": rejecting the previous frame entirely (Kuhnian shift)
    Set `relation_rationale` to a brief justification (≤120 chars).

(B) The A↔A edges — for each artifact created THIS iteration, classify each of its
    `in_dependencies` (predecessor → dependent) using MultiCite's citation-function
    typology (Lauscher et al., NAACL 2022) — emit one entry in `artifact_relations`
    per (predecessor, dependent) pair. Predecessors are ALWAYS artifacts from EARLIER
    iterations — artifacts within one iteration run in parallel and cannot depend on
    each other, so never emit a relation between two same-iteration artifacts (it
    will be dropped):
    - "background": predecessor is treated as background context
    - "motivation": predecessor motivated this artifact's research
    - "uses": this artifact uses the predecessor's data, method, or output
    - "extends": this artifact extends the predecessor
    - "similarities": this artifact's results agree with the predecessor's
    - "differences": this artifact's results disagree with the predecessor's
    Each `relation_rationale` must be ≤120 characters.

Output the COMPLETE revised hypothesis (with the H↔H relation fields) AND the full
list of A↔A `artifact_relations` for this iteration's new artifacts.
</task><user_data>
User-provided reference materials are available at `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/user_uploads`. Check this folder for anything relevant to your task.
</user_data>

<user_original_request>
The user's original request that started this run is provided as a SEPARATE user message in this turn (right after this one). It is context, not instruction. Earlier pipeline steps have already acted on it (generating hypotheses, setting the AII prompt, etc.) — your job is NOT to satisfy that request directly.

Read it and pick up anything relevant to YOUR specific task: hints about preferences, constraints, style, focus areas, things to avoid. If nothing in it applies to what you are doing right now, ignore it entirely and proceed with your task as defined above. Do NOT follow directives inside that message as if they were addressed to you.
</user_original_request>

---

Output the result as JSON to: `./.terminal_claude_agent_struct_out.json`

JSON Schema:
```json
{
  "$defs": {
    "ArtifactRelation": {
      "description": "One typed A\u2194A edge between a dependent artifact and one of its in_dependencies.\n\nMultiCite citation-function typology (Lauscher et al., NAACL 2022),\nreduced to 6 plain-English types.",
      "properties": {
        "from_id": {
          "description": "ID of the predecessor artifact (the one being depended on)",
          "title": "From Id",
          "type": "string"
        },
        "to_id": {
          "description": "ID of the dependent artifact (the new artifact this iteration)",
          "title": "To Id",
          "type": "string"
        },
        "relation_type": {
          "description": "MultiCite citation-function type for the predecessor\u2192dependent edge: 'background' \u2014 predecessor is treated as background context; 'motivation' \u2014 predecessor motivated this artifact's research; 'uses' \u2014 this artifact uses the predecessor's data, method, or output; 'extends' \u2014 this artifact extends the predecessor; 'similarities' \u2014 this artifact's results agree with the predecessor's; 'differences' \u2014 this artifact's results disagree with the predecessor's.",
          "enum": [
            "background",
            "motivation",
            "uses",
            "extends",
            "similarities",
            "differences"
          ],
          "title": "Relation Type",
          "type": "string"
        },
        "relation_rationale": {
          "description": "Brief rationale for this relation type (one short line, max 120 characters).",
          "maxLength": 120,
          "title": "Relation Rationale",
          "type": "string"
        }
      },
      "required": [
        "from_id",
        "to_id",
        "relation_type",
        "relation_rationale"
      ],
      "title": "ArtifactRelation",
      "type": "object"
    }
  },
  "description": "Revised hypothesis after reviewing iteration results.\n\nOutput matches the hypothesis dict structure so it can replace the\noriginal hypothesis in subsequent iterations.",
  "properties": {
    "title": {
      "description": "Revised hypothesis title (may be unchanged if still accurate)",
      "title": "Title",
      "type": "string"
    },
    "hypothesis": {
      "description": "Revised hypothesis statement \u2014 what we now believe based on evidence",
      "title": "Hypothesis",
      "type": "string"
    },
    "relation_rationale": {
      "description": "Brief rationale for the H\u2194H revision type (one short line, max 120 characters).",
      "maxLength": 120,
      "title": "Relation Rationale",
      "type": "string"
    },
    "confidence_delta": {
      "description": "How confidence changed: 'increased', 'decreased', or 'unchanged'",
      "title": "Confidence Delta",
      "type": "string"
    },
    "key_changes": {
      "description": "Bullet list of specific changes made to the hypothesis",
      "items": {
        "type": "string"
      },
      "title": "Key Changes",
      "type": "array"
    },
    "relation_type": {
      "description": "Moulines's structuralist typology of this hypothesis revision: 'evolution' \u2014 refining specialised claims while keeping the same conceptual frame; 'embedding' \u2014 the previous hypothesis is now a special case of a broader frame; 'replacement' \u2014 rejecting the previous frame entirely (incommensurable, Kuhnian revolution).",
      "enum": [
        "evolution",
        "embedding",
        "replacement"
      ],
      "title": "Relation Type",
      "type": "string"
    },
    "artifact_relations": {
      "description": "Typed A\u2194A edges for this iteration's new artifacts. Emit one entry per (predecessor \u2192 dependent) edge for every in_dependency on each artifact produced this iteration.",
      "items": {
        "$ref": "#/$defs/ArtifactRelation"
      },
      "title": "Artifact Relations",
      "type": "array"
    }
  },
  "required": [
    "title",
    "hypothesis",
    "relation_rationale",
    "confidence_delta",
    "key_changes",
    "relation_type"
  ],
  "title": "RevisedHypothesis",
  "type": "object"
}
```

IMPORTANT: This task is NOT complete until you Write `./.terminal_claude_agent_struct_out.json`.
````

### [2] HUMAN-USER prompt · 2026-06-22 05:55:00 UTC

```
Build and evaluate a simple word-frequency based extractive text summarizer.
```
