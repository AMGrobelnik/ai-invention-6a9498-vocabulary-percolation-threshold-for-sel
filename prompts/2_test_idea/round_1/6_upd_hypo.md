# upd_hypo — test_idea

> Phase: `invention_loop` · round 1 · `upd_hypo`
> Run: `run_CFTHGHhY9evP` — Vocabulary Percolation Threshold for Self-Calibrating Extractive Summary Length: Recall Advantage and F1 Failure Mode Across Corpora
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `upd_hypo` (terminal_claude_agent)

### [1] SYSTEM-USER prompt · 2026-06-22 05:08:31 UTC

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
  Vocabulary Percolation Threshold as a Self-Calibrating Summary Length in Word-Frequency Extractive Summarization
hypothesis: >-
  In word-frequency extractive summarization, the optimal number of sentences to include in a summary is not a fixed compression
  ratio (e.g., 20–30%) but emerges from a phase transition in the document's vocabulary co-occurrence network. Specifically:
  building a weighted word co-occurrence graph over the document's content words and greedily adding sentences (ranked by
  aggregate TF word score) until the induced subgraph's giant connected component reaches a critical fraction of the full
  document's giant component size yields summaries that are more coherent and cover key concepts more completely than summaries
  produced at an arbitrary fixed compression ratio. We hypothesize that this percolation-threshold stopping criterion will
  achieve higher ROUGE-1/2/L scores than fixed-length baselines (10%, 20%, 30% compression) on standard benchmark corpora,
  and that the threshold-determined length will vary significantly across document types in a way that is predictable from
  structural properties of the vocabulary network (average degree, clustering coefficient).
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
</all_artifacts>

<new_artifacts_this_iteration>
These 3 artifacts were created THIS iteration.

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
</new_artifacts_this_iteration>

<current_paper>
The paper draft from this iteration — represents the current state of the research story.

# Introduction

Extractive text summarization—selecting a subset of source sentences to form a summary—has been studied for over six decades, beginning with Luhn's frequency-based sentence scoring [1]. The central algorithmic question is which sentences to select; but an equally critical and consistently underexamined question is *how many* sentences to select. Virtually every published method, from TF-IDF scoring to graph-based systems such as TextRank [2] and LexRank [3], requires the user or system to specify a compression ratio (the fraction of sentences retained) as an external parameter, typically fixed at 20–30% by convention or tuned on held-out data.

This fixed-ratio convention is poorly motivated. A news article covering a single, self-contained event is conceptually complete in two or three sentences; a multi-topic feature article of the same word count may require ten to communicate its scope. A uniform 20% compression applied to both will either be redundant (event article) or lossy (feature article). Despite broad acknowledgment of this limitation [2, 3], no principled, document-specific stopping criterion has emerged in the extractive summarization literature.

We import a stopping criterion from statistical physics. In Erdős–Rényi random graph theory [4], a network undergoes a sharp phase transition as edges are added: below a critical density, the graph consists of many small disconnected components; above it, a *giant connected component* (GCC) suddenly emerges spanning most nodes [5]. This transition marks the point at which local clusters merge into a globally connected structure—a natural completeness signal. Transposed to text, the vocabulary co-occurrence graph of a growing summary starts fragmented (early sentences cover isolated topic clusters) and becomes globally connected as conceptually bridging sentences are added. The percolation threshold—where the summary's induced GCC reaches a critical fraction of the full document's GCC—signals that the summary has achieved conceptual percolation: all major topic clusters are mutually reachable.

Our investigation evaluates whether this percolation-threshold stopping criterion improves ROUGE scores over fixed-ratio baselines on the CNN/DailyMail benchmark [6]. The empirical results reveal an important and instructive negative finding: the graph-theoretic criterion does not improve ROUGE F1, because it systematically selects too many sentences (mean 78% of the document), sacrificing precision. However, percolation summaries achieve significantly higher ROUGE-1 recall (+0.152 over the best fixed-ratio baseline), and the variance analysis exposes a structural mismatch between the percolation objective (conceptual coverage) and the reference summary objective (concise highlights). These results constitute a clear empirical baseline and a principled analysis of why naive graph-percolation does not align with human summarization preferences in news corpora.

[FIGURE:fig_pipeline]

**Summary of Contributions.** (1) We introduce the first extractive summarizer that uses vocabulary graph percolation as a stopping criterion, with a complete open-source implementation (§3). (2) We provide a rigorous evaluation on 2000 CNN/DailyMail documents comparing four percolation thresholds against three fixed-ratio baselines across ROUGE-1/2/L recall and F1, with Holm-corrected Wilcoxon tests (§4). (3) We report and analyze an honest negative finding: percolation maximizes recall but not F1, and we identify the structural reason—CNN/DailyMail reference summaries are concise highlights, not coverage-complete extracts (§5). (4) We quantify the predictability of percolation length from graph structural features (R²=0.10) and characterize how compression ratio varies with vocabulary network topology (§5).

# Related Work

**Extractive summarization.** Luhn [1] introduced frequency-based sentence scoring, the foundation of word-frequency extractive methods. TextRank [2] applies PageRank [7] over a sentence-similarity graph and selects the top-*k* sentences; LexRank [3] uses eigenvector centrality over cosine-similarity sentences. Both operate at the sentence level and require a fixed *k*. SummaRuNNer [8] uses a recurrent neural network to score sentences sequentially, retaining a fixed number. BERTSum [9] and PreSumm [10] apply pretrained transformer encoders to sentence classification. Neural abstractive methods, including pointer-generator networks [11], similarly rely on length constraints specified externally. None of these methods determines *k* from the document's intrinsic structure.

Ryang and Abekawa [12] use an influence propagation model on a word co-occurrence network to maximize coverage under a budget (sentence count) constraint. The budget remains manually specified; they do not use a phase transition to determine when coverage is complete. Our work shares the word-network foundation but replaces the manual budget with the percolation threshold.

**Graph percolation in NLP.** Random graph percolation theory [4, 5] has been applied to model information diffusion in social networks and language evolution, but to our knowledge has not previously been proposed as a stopping criterion for extractive summarization. Our approach adapts the giant-component emergence criterion as a data-driven completeness signal.

**CNN/DailyMail benchmark.** Hermann et al. [6] introduced the CNN/DailyMail corpus for reading comprehension; it has since become the standard benchmark for single-document summarization [11]. The reference summaries are journalist-written bullet-point highlights (mean 42 words), a property that significantly shapes our results (§5).

# Methods

## Vocabulary Co-occurrence Graph Construction

Given a document with sentences $s_1, \ldots, s_n$, we first preprocess each sentence: tokenize, lowercase, remove stop words (NLTK English stop list), and retain alphabetic tokens of length ≥ 2. Let $W_i$ denote the content word set of sentence $s_i$.

The vocabulary co-occurrence graph $G = (V, E, w)$ is an undirected weighted graph. Nodes $V$ are unique content words across the document. An edge $(u, v) \in E$ exists between any two words $u, v \in W_i$ for some sentence $s_i$; the edge weight $w(u, v)$ equals the number of sentences in which both $u$ and $v$ co-occur. The full-document GCC, $\text{GCC}_{\text{doc}}$, is the node count of the largest connected component of $G$. [ARTIFACT:art_ZuUuuoy0rCRF]

## Sentence Scoring and Ordering

Each sentence $s_i$ receives a TF score:
$$\text{score}(s_i) = \sum_{w \in W_i} \text{TF}(w, D)$$
where $\text{TF}(w, D)$ is the raw term frequency of word $w$ in document $D$. Sentences are ranked in descending score order and added greedily to the summary.

## Percolation-Threshold Stopping Criterion

Given a threshold $\theta \in (0, 1)$, we build the summary incrementally. At each step, we add the next highest-scoring sentence $s_i$ and update the *induced subgraph* $G_S$ over the accumulated content words $\bigcup_{s_j \in S} W_j$. We compute $\text{GCC}_{\text{summary}} = $ node count of the largest connected component of $G_S$. The summary grows until:
$$\frac{\text{GCC}_{\text{summary}}}{\text{GCC}_{\text{doc}}} \geq \theta$$
The stopping index $k^*$ is the percolation-determined summary length. We evaluate $\theta \in \{0.6, 0.7, 0.8, 0.9\}$. If the ratio never reaches $\theta$ (which occurs when the document graph is sparse), all sentences are retained.

## Fixed-Ratio Baselines

The baselines use the same TF scoring and sentence ordering as the percolation method. For a fixed ratio $r \in \{0.10, 0.20, 0.30\}$, the summary consists of the top $\lceil r \cdot n \rceil$ sentences. Fixing the method (TF scoring) while varying only the stopping criterion isolates the effect of percolation vs. fixed-ratio length selection.

# Experiments

## Dataset and Evaluation

We use the CNN/DailyMail 3.0.0 test split (2000 documents) loaded via HuggingFace [6]. [ARTIFACT:art_Zf8PEZKgv-j_] Articles have a mean of 575 words (range 73–1846). Reference summaries are multi-sentence bullet-point highlights with a mean of 42 words (2–5 sentences). We evaluate with ROUGE-1, ROUGE-2, and ROUGE-L recall (R) and F1 (F). Statistical comparisons use two-sided Wilcoxon signed-rank tests with Holm–Bonferroni correction over 36 pairwise comparisons.

## Main Results

Table 1 reports mean ROUGE scores across all 2000 documents. [ARTIFACT:art_yWTseob35O3B] The results expose a clear tension between recall and precision.

| Method | ROUGE-1 R | ROUGE-2 R | ROUGE-L R | ROUGE-1 F1 | ROUGE-2 F1 |
|---|---|---|---|---|---|
| percolation θ=0.6 | 0.788 | 0.390 | 0.544 | 0.214 | 0.107 |
| percolation θ=0.7 | 0.825 | 0.428 | 0.582 | 0.194 | 0.102 |
| percolation θ=0.8 | 0.856 | 0.462 | 0.613 | 0.178 | 0.097 |
| percolation θ=0.9 | **0.880** | **0.488** | **0.638** | 0.154 | 0.081 |
| fixed 10% | 0.516 | 0.193 | 0.330 | **0.286** | **0.108** |
| fixed 20% | 0.650 | 0.276 | 0.428 | 0.251 | 0.105 |
| fixed 30% | 0.728 | 0.337 | 0.491 | 0.222 | 0.100 |

*Table 1: Mean ROUGE scores across 2000 CNN/DailyMail test documents. Bold indicates best per column.*

**Recall.** Percolation at θ=0.9 achieves the highest ROUGE-1 recall of 0.880, versus 0.728 for the best fixed-ratio baseline (fixed@30%), a difference of Δ=+0.152. All percolation θ values significantly outperform all fixed-ratio baselines on ROUGE-1 recall (Wilcoxon, all Holm-corrected p<0.001). This confirms that percolation-determined summaries contain more of the reference vocabulary—they include more sentences.

**F1.** Fixed-ratio summaries dominate on F1. The 10% fixed baseline achieves ROUGE-1 F1=0.286, versus 0.214 for the best percolation method (θ=0.6). The fixed baselines' precision advantage more than compensates for their lower recall. Percolation at θ=0.9 achieves F1=0.154—less than half the fixed@10% value.

[FIGURE:fig_rouge_comparison]

## Compression Ratio Analysis

The percolation threshold at θ=0.9 selects a mean of 77.8% of sentences (std=0.083, range 11.1%–100%). The interquartile range spans 72.9%–83.3%. The hypothesis required std>10 percentage points (0.10) for the compression ratio to be judged substantially variable; the observed std=0.083 falls below this threshold, indicating that in the CNN/DailyMail corpus the percolation threshold consistently fires at a high fraction of sentences regardless of document length or structure.

The CNN and DailyMail segments yield slightly different mean compression ratios (CNN: 79.7%, DailyMail: 75.8%), reflecting differences in article length and vocabulary density, but both are far above the 20–30% range targeted by the hypothesis.

[FIGURE:fig_compression_dist]

## Network Structure and Length Predictability

Linear regression predicts the percolation compression ratio (θ=0.9) from three vocabulary graph features: average degree, clustering coefficient, and graph density. The regression achieves R²=0.10, explaining a modest but statistically significant fraction of variance. Pearson correlation analysis identifies the strongest predictors: graph density (r=0.230, p<10⁻²⁵), clustering coefficient (r=0.202, p<10⁻¹⁹), and number of sentences (r=−0.250, p<10⁻²⁹). Documents with denser, more clustered vocabulary graphs trigger percolation at a slightly higher compression ratio, while longer documents trigger it at a lower ratio. Per-theta R² values range from 0.086 to 0.105 across the four thresholds, indicating the relationship is stable across θ choices.

[FIGURE:fig_network_regression]

## Segment Analysis

We split the 2000 documents by source: 1020 CNN articles (shorter, single-focus) and 980 DailyMail articles (longer, multi-topic). Across both segments, percolation achieves higher ROUGE-1 recall than all fixed-ratio baselines, and fixed-ratio achieves higher F1. The advantage of percolation recall over fixed@30% is consistent: CNN Δ=+0.176 (0.867 vs. 0.691), DailyMail Δ=+0.126 (0.893 vs. 0.767). This confirms the recall advantage is not an artifact of one sub-corpus.

# Discussion

## Why Percolation Does Not Improve F1

The core negative finding is that percolation-determined summaries are too long for CNN/DailyMail reference summaries. CNN/DailyMail highlights are concise journalist-authored bullet points (mean 42 words, 2–5 sentences), representing approximately 7–10% of a typical article. An extractive method optimized to maximize vocabulary co-occurrence coverage will naturally select far more sentences than those highlights contain. The percolation threshold does correctly signal when the vocabulary graph becomes "connected enough," but that connectivity criterion aligns poorly with the selectivity criterion implicit in journalist-written highlights.

This mismatch exposes a conceptually important distinction: *graph connectivity completeness* is not the same as *human summary completeness*. A human summarizer selects the most salient sentences; the percolation criterion selects sentences until vocabulary coverage is achieved. In a focused news article, vocabulary connectivity may require 70–80% of sentences because mid-document sentences use vocabulary that bridges early and late topic clusters—this bridging vocabulary may be conceptually unimportant to the human reader.

## The Variable-Length Failure Mode

The hypothesis assumed that the percolation threshold would vary substantially across documents, producing short summaries for focused articles and longer ones for multi-topic documents. The data shows this does not occur in CNN/DailyMail: the compression ratio clusters around 78% for θ=0.9 (std=0.083). This homogeneity arises because CNN/DailyMail articles, though diverse in topic, are structurally similar: they are newswire articles written in inverted pyramid style, with vocabulary distributed throughout rather than clustered in discrete topic segments. The graph phase transition therefore fires at a structurally consistent point across documents.

## Implications and Paths Forward

The percolation criterion is most meaningful when documents have truly distinct vocabulary clusters—for example, multi-topic review articles, multi-document summarization sets, or long-form documents where sections cover different subjects. In such settings, the criterion would fire earlier for focused documents and later for multi-topic ones, producing the variable-length behavior the hypothesis anticipated. Future work should evaluate percolation stopping on multi-document or scientific summarization corpora where vocabulary segregation is stronger.

The R²=0.10 network-to-length regression, while modest, shows that graph structure does predict percolation behavior. Graph density and clustering coefficient are the dominant predictors; future percolation designs could use these features to *calibrate* θ dynamically—choosing a smaller θ for dense graphs (where GCC grows quickly) to prevent over-inclusion.

## Limitations

- *Single corpus.* All experiments use CNN/DailyMail, a homogeneous newswire corpus. The findings may not generalize to multi-topic or scientific documents.
- *TF scoring only.* The sentence scoring function is term frequency. Richer sentence rankers (TF-IDF, BERTScore) may interact differently with the percolation stopping criterion.
- *GCC definition.* The fraction θ is applied to the GCC node count. Alternative percolation measures (edge density, modularity, spectral gap) may fire at different, more appropriate points.
- *No training-free calibration.* The threshold θ is swept empirically over {0.6, 0.7, 0.8, 0.9}; a principled method for setting θ without held-out data remains an open problem.

# Conclusion

We proposed and evaluated the first extractive summarizer to use vocabulary co-occurrence graph percolation as a self-calibrating length criterion. On 2000 CNN/DailyMail documents, the percolation method at θ=0.9 achieves ROUGE-1 recall of 0.880—significantly and substantially above the fixed-ratio maximum of 0.728—while fixed-ratio baselines dominate on F1 (0.286 vs. 0.214). The percolation threshold consistently selects approximately 78% of document sentences in this corpus, rather than the varied compression ratios anticipated by the original hypothesis, indicating that CNN/DailyMail's inverted-pyramid structure lacks the vocabulary segregation needed for percolation to discriminate document types. Network regression (R²=0.10) confirms a modest but real relationship between graph topology and percolation behavior. These results establish a clear empirical benchmark and identify the conditions—structural vocabulary clustering—under which percolation-based length determination is most likely to succeed.

Future directions include: (1) evaluation on multi-topic or scientific corpora with stronger vocabulary segregation; (2) dynamic θ calibration using graph features without validation data; (3) hybrid methods that use the percolation-based GCC growth curve as a feature for a learned length predictor rather than a hard threshold.

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
</current_paper>

<reviewer_feedback>
Feedback from the paper reviewer this iteration.

- [MAJOR] (rigor) Systematic numerical discrepancies between paper and artifacts. Checking mini_method_out.json against Table 1: (a) percolation θ=0.6 ROUGE-1 recall=0.759 in artifact vs. 0.788 in paper; θ=0.9 recall=0.853 vs. 0.880; fixed@10% recall=0.504 vs. 0.516; fixed@30% recall=0.707 vs. 0.728. (b) All percolation F1 values in Table 1 are inflated: θ=0.6 F1=0.2011 in artifact vs. 0.214 in paper. (c) Compression ratio for θ=0.9: artifact shows mean=76.17%, std=0.0893; paper reports 77.8%, std=0.083. (d) The recall advantage headline in abstract/intro claims +0.152 but correct difference is 0.853−0.707=0.146. These suggest the paper was finalized from a different or intermediate experimental run without reconciling the numbers.
  Action: Re-run all aggregate statistics from the artifact data and update every number in the paper. Specifically update Table 1, the compression ratio statistics in §4.3 (including mean, std, IQR, CNN/DailyMail segment values), and the Δ=+0.152 headline in abstract and §4.2. Add a data provenance statement pointing to the exact artifact hash used for final numbers.
- [MAJOR] (scope) Single-corpus evaluation with a weak baseline scorer makes the conclusions fragile. All experiments use CNN/DailyMail with TF sentence scoring. The paper's own Discussion (§5.3) identifies multi-topic corpora as the condition under which percolation should succeed—but this condition is never tested. Without at least one additional corpus the claim that 'percolation is most meaningful when documents have truly distinct vocabulary clusters' is speculation unsupported by evidence.
  Action: Add evaluation on at least one multi-topic corpus such as Multi-News (Fabbri et al., 2019), arXiv abstracts, or WikiSum. This directly tests the paper's core theoretical prediction and would either validate the hypothesis or reveal a deeper limitation. If computational budget is a constraint, a random sample of 500 documents is sufficient to establish the directional result.
- [MAJOR] (methodology) Baselines are too weak: only TF scoring with fixed ratios. No TF-IDF, no BERTScore-based sentence ranking, no TextRank or LexRank which are directly cited in the paper. Since TF scoring is itself suboptimal, any advantage or disadvantage of the percolation stopping criterion may be confounded with sentence scoring quality. Specifically, TextRank and LexRank also build word/sentence graphs and would be natural comparisons since the paper claims a graph-theoretic advance.
  Action: Add at minimum TF-IDF sentence scoring with both fixed-ratio and percolation stopping criteria, so the interaction between scorer quality and stopping criterion can be assessed. Adding TextRank@k-variable (where k is percolation-determined) would directly answer whether graph-based scoring + graph-based stopping is complementary.
- [MINOR] (novelty) The negative finding (percolation selects ~78% of sentences → poor F1 against short highlights) is predictable from first principles given that CNN/DailyMail highlights are 7–10% of article length. The paper acknowledges this but presents it as a discovery rather than a pre-derivable consequence. A reader familiar with the corpus would not be surprised. The paper would be stronger if it showed this reasoning was non-obvious a priori or if it found a surprising aspect of the failure mode.
  Action: Strengthen the Discussion by showing what a hypothetical 'well-calibrated' percolation threshold would look like—i.e., what θ value would produce 10% compression and whether GCC reaches it at a consistent point. This would reframe the failure as a calibration problem with a potential solution rather than a fundamental incompatibility.
- [MINOR] (evidence) R²=0.10 for network feature regression is reported as meaningful but explains only 10% of variance. The paper states this confirms 'a modest but real relationship' but the practical implication—that graph features cannot reliably predict compression ratio—argues against using them for the proposed 'dynamic θ calibration' future direction.
  Action: Either temper the claim about dynamic θ calibration (R²=0.10 is insufficient for reliable calibration) or show the partial R² contribution of each predictor and the confidence intervals on the regression coefficients to clarify whether any individual feature is actionably predictive.
- [MINOR] (methodology) The percolation criterion uses the GCC node count, but the vocabulary co-occurrence graph constructed from a document is not a random Erdős–Rényi graph—it has strong community structure, heavy-tailed degree distributions, and high clustering coefficients. The theoretical motivation from classical random graph percolation theory therefore does not directly apply. The paper does not acknowledge this gap.
  Action: Add a brief caveat noting that the Erdős–Rényi model provides intuition but does not formally apply to vocabulary co-occurrence graphs, and that the percolation threshold θ is therefore empirically tuned rather than theoretically derived. Alternatively, cite literature on percolation in heterogeneous networks that is closer to the actual graph structure.
- [MINOR] (clarity) The paper conflates two distinct stopping behaviors: (a) the percolation ratio never reaches θ (all sentences retained) and (b) percolation fires at a high but sub-100% fraction. The frequency of case (a) is not reported—how often do documents hit the 100% ceiling? This is important for interpreting the 'range 11.1%–100%' reported for θ=0.9.
  Action: Report the fraction of documents that hit the 100% ceiling for each θ value. This directly informs whether the 'sparse graph' edge case is common or rare and affects the practical applicability of the method.
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

### [2] HUMAN-USER prompt · 2026-06-22 05:08:31 UTC

```
Build and evaluate a simple word-frequency based extractive text summarizer.
```

### [3] SYSTEM-USER prompt · 2026-06-22 05:09:15 UTC

```
<validation-feedback>
Attempt 1 failed validation.

Schema validation found 1 problem — fix ALL of them at once:
  - at `relation_rationale`: 'Same percolation-GCC frame; narrows success condition to vocabulary-segregated corpora after negative F1 result on CNN/DM.' is too long (at most 120 characters, got 122)
Every required field must be present and every field type must match the schema.

Please use the Write tool to overwrite `.terminal_claude_agent_struct_out.json` with corrected JSON. Do not invent new fields; match the schema you were given.
</validation-feedback>
```
