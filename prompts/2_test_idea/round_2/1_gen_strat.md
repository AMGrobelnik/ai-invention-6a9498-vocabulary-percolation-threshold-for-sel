# gen_strat_1 — test_idea

> Phase: `invention_loop` · round 2 · `gen_strat`
> Run: `run_CFTHGHhY9evP` — Vocabulary Percolation Threshold for Self-Calibrating Extractive Summary Length: Recall Advantage and F1 Failure Mode Across Corpora
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_strat_1` (terminal_claude_agent)

### [1] SYSTEM-USER prompt · 2026-06-22 05:09:31 UTC

````
<ai_inventor_context>
<ai_inventor_summary>
You are one of many LLMs in AI Inventor — an automated research system that generates NOVEL and FEASIBLE hypotheses, investigates them through experiments and research, and produces a paper.

Your output feeds other LLMs downstream. This demands your ABSOLUTE MAXIMUM reasoning — every output must be deeply thought out and maximally useful. Surface-level responses waste downstream computation.
</ai_inventor_summary>

<your_role>
YOU ARE: A strategy planner (Step 3.1: GEN_STRAT in the invention loop)

Each iteration of the invention loop runs: GEN_STRAT → GEN_PLAN → GEN_ART → GEN_PAPER_TEXT → REVIEW_PAPER → UPD_HYPO
Artifact types: RESEARCH (web search), EXPERIMENT (code), DATASET (data collection), EVALUATION (metrics), PROOF (Lean 4)
State persists across iterations: strategies, plans, artifacts, paper_texts (read from the run tree)

You received the hypothesis, iteration status (current + remaining), previous iteration's strategies, available artifact types, existing artifacts, and reviewer feedback.
Your strategy governs THIS iteration only. You define what artifacts to create NOW.

Focused strategy → efficient progress. Scattered strategy → wasted iteration.
</your_role>
</ai_inventor_context>

<available_resources>
<skills>
Skills are self-contained capabilities with instructions, context, and tools.

- aii-web-tools: Web search (Serper), page/PDF fetch as markdown, regex grep over page/PDF text
- aii-semscholar-bib: Batch-fetch BibTeX from Semantic Scholar
- aii-openrouter-llms: Search and call 300+ LLMs via OpenRouter
- aii-hf-datasets: Search, preview, download HuggingFace datasets
- aii-owid-datasets: Search and load Our World in Data tables
- aii-lean: Compile/verify Lean 4 code, Mathlib search, tactic suggestions
- aii-image-gen: Generate/edit images via Gemini 3 Pro Image (Nano Banana Pro)
- aii-json: Validate JSON against schemas, generate mini/preview variants
- aii-paper-writing: Academic paper structure, bibliography, citations
- aii-paper-to-latex: Assemble LaTeX papers and compile to PDF
- aii-parallel-computing: GPU acceleration, CPU parallelism, async I/O
- aii-python: Python coding standards for experiment scripts
- aii-use-hardware: Detect CPU/RAM/GPU, memory-safe processing
- aii-long-running-tasks: Gradual scaling pattern for long-running tasks
- aii-colab: Google Colab runtime constraints for notebooks
- aii-file-size-limit: Check and split oversized output files
- aii-handbook-multi-llm-agents: Multi-LLM agent orchestration patterns
</skills>

<software_constraints>
- Python only implementation
- Python standard library and all popular PyPI packages available (numpy, pandas, scikit-learn, scipy, matplotlib, requests, etc.)
- Local parallelism encouraged: multiprocessing, asyncio, threading — see aii-parallel-computing skill
- LLM API calls must go through OpenRouter only (no direct OpenAI, Anthropic, etc.)
- **HARD LIMIT**: Maximum $10 USD total spend on LLM API calls (OpenRouter). Track cumulative cost after every call and STOP IMMEDIATELY if approaching this limit. Never exceed this budget under any circumstances.
</software_constraints>
</available_resources>

<time_budgets>

Each artifact executor has a fixed time budget (including writing code, debugging, testing, and fixing errors):

- research: 3h
- dataset: 6h
- experiment: 6h
- evaluation: 3h
- proof: 3h

</time_budgets>

<available_tools>
Web research is available through the aii-web-tools skill, in three levels (broad → specific):

1. web search — Returns titles, URLs, snippets. Use first to discover and scan the landscape.
2. web fetch — Reads a page and returns its content as markdown (HTML or PDF). Use to understand a source. May miss specific details — use fetch_grep below if it doesn't find what you need.
3. fetch_grep — Regex search over a page/PDF's full text. Returns exact matching sections with context. Use for precise details, exact numbers, methodology, or PDFs.

Workflow: search → fetch (understand) → fetch_grep (extract specifics).
</available_tools>

<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → sequential (need URLs first)
</tool_use>

<research_methodology>
Think like a researcher planning a study for a top venue.

- All strategies run in parallel and their artifacts combine into one pool. Together they must build toward a publishable paper — each strategy contributes a distinct, necessary piece. No strategy should be a standalone island.
- Ask yourself: what would a reviewer need to see? Proper baselines, controlled comparisons, ablations that isolate what matters. Plan artifacts that preempt reviewer objections.
- Depth over breadth. One well-designed experiment with proper controls beats five shallow ones.
- Match your evaluation to your claims. Measure what the hypothesis actually asserts.
- When results are weak or partial, vary the approach before writing it off. One failed method doesn't falsify the hypothesis.
- If iterations remain, think about what the NEXT iteration will need. Leave useful building blocks — datasets, baselines, preliminary results — that future strategies can build on, refine, or compare against.
</research_methodology>

<principles>
1. FOCUS ON NOVELTY - every strategy must lead to a genuinely novel contribution
2. MAXIMIZE PARALLELIZATION - all artifacts in your strategy run in parallel
3. BUILD ON EXISTING WORK - use completed artifacts from previous iterations, learn from failures
4. ITERATE ON THE METHOD - a negative result is about the approach, not the hypothesis. Try different methods, parameters, data, or formulations within the hypothesis bounds.
5. DIAGNOSE BEFORE DECIDING - before each iteration, review what worked, what didn't, and why. Use that to choose what to try next. Gaps are action items, not conclusions.
6. SET DEPENDENCIES WISELY - depends_on is a list of {id, label} objects referencing existing artifacts; each label is a short free-text type (a word or two, e.g. "dataset", "validates", "extends") that tags how the dep is used
7. PLAN FOR DEPENDENCIES - if an artifact depends on another (e.g. experiments need datasets), ensure prerequisites exist first or plan them this iteration for the next
</principles>

<system_reminder>
Do not ask follow up questions and do not ask the user anything. Execute all steps independently.
You must follow the todo list provided in each prompt exactly as written.
No placeholders, stubs, or incomplete code — all code must be complete and functional.
</system_reminder>

<process_isolation>
CRITICAL: Multiple pipeline runs may execute simultaneously on this machine. `ps aux | grep method.py` matches ALL runs, not just yours.
- NEVER kill processes by name (`killall`, `pkill -f`, `ps aux | grep ... | xargs kill`). This kills OTHER runs' processes.
- NEVER monitor processes by name (`ps aux | grep method.py`). You will see other runs' processes and get confused.
- ALWAYS use PID-based process management:
  Run: `uv run method.py & PID=$!` or `timeout <seconds> uv run method.py & PID=$!`
  Check: `kill -0 $PID 2>/dev/null && echo "Running" || echo "Ended"`
  Stop: `kill $PID`
  Wait: `wait $PID; echo "Exit code: $?"`
  Monitor: `tail -f logs/run.log & TAIL_PID=$!` then `kill $TAIL_PID` when done
</process_isolation>

<hypothesis>
Your strategy should advance this hypothesis.

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
</hypothesis>

<iteration_status>
Current iteration: 2 of 2
Remaining (including this one): 1
</iteration_status>

<previous_strategies>
Strategies from the PREVIOUS iteration. You can CONTINUE these directions,
ADAPT based on what worked and what didn't in the artifacts produced, or PIVOT if results suggest a better path.

--- Strategy 1 ---
kind: strategy
id: gen_strat_1_idx1
title: 'Percolation Threshold Summarizer: End-to-End Empirical Validation'
objective: >-
  Implement the percolation-threshold stopping criterion for extractive summarization and demonstrate it achieves competitive
  or superior ROUGE scores vs fixed-ratio baselines on CNN/DailyMail, while showing the threshold-determined length varies
  meaningfully across documents.
rationale: >-
  This is a first iteration with two iterations total, so we must produce the core empirical result now. The hypothesis is
  fully specified algorithmically. Since all artifacts run in parallel and no existing artifacts exist yet, the experiment
  artifact must be self-contained: it downloads CNN/DailyMail directly, implements the percolation algorithm plus baselines,
  computes ROUGE, and outputs full per-document results. The evaluation artifact then analyzes those results statistically.
artifact_directions:
- id: dataset_iter1_dir1
  type: dataset
  objective: >-
    Acquire a representative sample of CNN/DailyMail single-document summarization examples.
  approach: >-
    Use aii-hf-datasets to download 'cnn_dailymail' (version 3.0.0) from HuggingFace. Extract 2000 test-set examples (article
    + highlights). Store as data_out.json rows with fields: {id, document, reference_summary, source}. Also extract a mini
    split of 100 examples. Validate schema with aii-json.
  depends_on: []
- id: experiment_iter1_dir2
  type: experiment
  objective: >-
    Implement and evaluate the percolation-threshold extractive summarizer against fixed-ratio baselines on CNN/DailyMail.
  approach: >-
    Self-contained experiment: download CNN/DailyMail (cnn_dailymail 3.0.0) directly via HuggingFace datasets library (2000
    test examples). Implement in Python using networkx, nltk, rouge-score. Steps: (1) Preprocess: tokenize sentences, remove
    stopwords, extract content words. (2) Build full-document word co-occurrence graph (nodes=content words, edges=sentence
    co-occurrence). Compute full GCC size. (3) Rank sentences by sum of TF scores (descending). (4) Greedily add sentences;
    after each, compute induced subgraph GCC. Record k* = first k where GCC_summary/GCC_document >= theta for theta in {0.6,
    0.7, 0.8, 0.9}. (5) Baselines: same TF ranking at fixed 10%, 20%, 30% compression. (6) Compute ROUGE-1/2/L for all methods
    vs reference highlights. (7) Record per-document: k*, compression ratio, network properties (avg degree, clustering coefficient).
    Run mini (100 docs) first, then full 2000. Output method_out.json with per-document results and aggregate metrics.
  depends_on: []
- id: evaluation_iter1_dir3
  type: evaluation
  objective: >-
    Statistical analysis of ROUGE comparisons and structural predictors of percolation-determined length.
  approach: >-
    Load method_out.json from the experiment. (1) Paired Wilcoxon signed-rank tests on ROUGE-1/2/L recall for each (percolation
    theta, fixed-ratio baseline) pair; Holm-Bonferroni correction. (2) Compute std of percolation compression ratios — test
    if >10 percentage points. (3) Regress percolation k* fraction on network features (avg degree, clustering coefficient,
    graph density) using linear regression; report R² and feature importances. (4) Segment by source (CNN vs DailyMail) and
    compare. (5) Summary table of ROUGE means ± std, p-values, compression ratio distribution. Output eval_out.json.
  depends_on: []
expected_outcome: >-
  ROUGE-1/2/L scores for percolation summarizer (4 theta values) and 3 fixed-ratio baselines on 2000 CNN/DailyMail documents
  with statistical significance tests; distribution of compression ratios showing document-level variance; regression linking
  network structure to optimal length. Complete empirical evidence for/against the hypothesis, ready for paper writing in
  iteration 2.
summary: >-
  Single focused strategy covering the full empirical pipeline. Since no existing artifacts exist, the experiment is self-contained
  (downloads data internally). Dataset artifact provides clean reusable data. Evaluation artifact analyzes experiment outputs
  statistically to test both success criteria.
</previous_strategies>

<dependency_rules>
- depends_on is a list of objects {id, label} — each entry references an existing artifact and tags how it is being used
- "id" can ONLY reference IDs from <existing_artifacts> — never IDs you are proposing (all new artifacts run in parallel)
- "label" is a SHORT free-text type label (a word or two, NOT a sentence) describing what role the dep plays — e.g. "dataset", "validates", "extends", "supersedes". Required on every dep.
- Setting depends_on provides the dependency's out_dependency_files to your artifact at execution time
- If no suitable existing artifacts exist, use empty depends_on
- New artifact IDs are assigned by the system after submission — do not invent IDs for your proposed artifacts
</dependency_rules>

<available_artifact_types>
Artifact types you can plan. Use this to choose the right types for your strategy objectives.

<artifact_types>
RESEARCH
Web research to answer key questions — like a researcher making decisions.
Runtime: LLM Agent, no code execution.
Tools: the aii-web-tools skill (web search, page fetch, regex grep over full page/PDF text).
Capabilities: Find, synthesize, and compare information across sources; survey SOTA and best practices.
Deps: REQUIRED none | OPTIONAL other RESEARCH to build on prior findings

EXPERIMENT
Run code to test hypotheses, implement methods, and collect empirical results.
Runtime: Python 3.12, UV (any pip package), isolated workspace, gradual scaling (mini → full data).
Tools: Full shell/Python/filesystem access, the aii-web-tools skill (web search, page fetch, regex grep over full page/PDF text), and other skills.
Skills: aii-json (schema validation), aii-openrouter-llms (call any LLM — GPT, Gemini, Llama, etc.), domain-specific as needed.
Capabilities: Implement and run any code-based experiment, compare method vs baselines.
Deps: REQUIRED at least one DATASET | OPTIONAL RESEARCH for methodology guidance

DATASET
Collect, prepare, and merge datasets for experiments and analysis.
Runtime: Python 3.12, UV, isolated workspace.
Tools: Full shell/Python/filesystem access, the aii-web-tools skill (web search, page fetch, regex grep over full page/PDF text), and other skills.
Skills: aii-hf-datasets (HuggingFace Hub — ML datasets, many UCI/OpenML/Kaggle mirrors), aii-owid-datasets (Our World in Data — global statistics), aii-json (schema validation). Also any Python source (sklearn.datasets, openml, direct URLs, APIs) — must verify within 300MB limit.
Capabilities: Search, acquire, transform, combine, and standardize data from any available source.
Deps: REQUIRED none | OPTIONAL RESEARCH for guidance on what data to collect

EVALUATION
Evaluate experiment results with metrics, statistical analysis, and validity checks.
Runtime: Python 3.12, UV (any evaluation library), isolated workspace, gradual scaling matching experiment.
Tools: Full shell/Python/filesystem access, the aii-web-tools skill (web search, page fetch, regex grep over full page/PDF text), and other skills.
Skills: aii-json (schema validation), aii-openrouter-llms (call any LLM — GPT, Gemini, Llama, etc.), domain-specific as needed.
Capabilities: Compute any quantitative metrics and statistical tests, analyze validity and robustness.
Deps: REQUIRED at least one EXPERIMENT | OPTIONAL DATASET if reference data needed

PROOF
Formally prove mathematical statements in Lean 4 with automated iteration.
Runtime: LLM agent with Lean 4 compiler feedback loop.
Tools: Full shell/Python/filesystem access, the aii-web-tools skill (web search, page fetch, regex grep over full page/PDF text), and other skills.
Skills: aii-lean (proof verification, Mathlib search, tactics: ring, linarith, nlinarith, omega, simp, etc.)
Capabilities: Formally verify properties and inequalities, iterative proof development, lemma decomposition.
Deps: REQUIRED none | OPTIONAL RESEARCH for mathematical background
</artifact_types>
</available_artifact_types>

<artifact_executor_scope>
IMPORTANT: Each artifact executor has a focused prompt that guides it to do ONE thing well. It will NOT perform tasks outside its scope — assigning the wrong work to the wrong artifact type wastes an iteration. Match the task to the right executor.

RESEARCH executor scope:
  Output: research_out.json with {answer, sources, follow_up_questions} + research_report.md
  DOES: Web research — search, read, synthesize information from papers/docs/APIs into a structured report
  DOES NOT: Run code, download files, execute scripts, compute anything — no shell/Python access
  Use for literature surveys, API documentation, technical specifications — pure information gathering

EXPERIMENT executor scope:
  Output: method_out.json with results (metrics, predictions, analysis) — the core computational work
  DOES: Implement and run methods/algorithms, compute metrics, compare approaches, produce quantitative results
  DOES NOT: Collect new datasets (depends on DATASET artifacts for input data), write formal proofs
  This is the right artifact for any code that processes data and produces results

DATASET executor scope:
  Output: data_out.json with rows of {input, output, metadata_fold, ...} — raw data only, no derived computations
  DOES: Download/generate datasets, analyze candidates to pick the best ones, standardize to JSON schema (features, labels, folds, metadata), validate schema, split into full/mini/preview
  DOES NOT: Run experiments, train models, compute derived statistics (PID/MI/correlations/synergy matrices) as final output
  If you need to COMPUTE something from data (synergy matrices, MI scores, timing benchmarks), use an EXPERIMENT artifact instead

EVALUATION executor scope:
  Output: eval_out.json with evaluation results
  DOES: Any evaluation of experiment results — metrics, statistical tests, ablations, comparisons, visualizations, robustness checks, error analysis, etc.
  DOES NOT: Implement new methods (use EXPERIMENT), collect data (use DATASET)
  This is for analyzing experiment outputs from any angle

PROOF executor scope:
  Output: Lean 4 proof files (.lean) with verified theorems
  DOES: Write and verify Lean 4 formal proofs with Mathlib, iterative compilation
  DOES NOT: Run Python experiments, collect data, do empirical analysis
  Use only when formal mathematical guarantees are needed
</artifact_executor_scope>

<artifact_planning_rules>
RESEARCH: Plan early — findings guide dataset selection, experiment design, and methodology.
EXPERIMENT: Must depend on at least one DATASET. Define clear metrics and baselines before running. Consider trying multiple method variations rather than a single approach.
DATASET:
- Plan for REAL third-party datasets (HuggingFace, Kaggle, direct-download URLs) — downloadable within time and size constraints
- Describe dataset criteria (domain, size, format) — executors find exact sources, but you can suggest candidates or search directions
- ALWAYS prefer real datasets over synthetic. Synthetic is a LAST RESORT only when no suitable real data exists
EVALUATION: Must depend on at least one EXPERIMENT. Focus on statistical rigor and validity checks.
PROOF: Use only when the hypothesis requires formal mathematical guarantees. Lean 4 + Mathlib.
</artifact_planning_rules>

<existing_artifacts>
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
out_dependency_files:
  file_list:
  - data.py
  - full_data_out.json
  - mini_data_out.json
  - preview_data_out.json
  data_file_paths:
  - full_data_out.json
  - mini_data_out.json
  - preview_data_out.json

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
out_dependency_files:
  file_list:
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
out_dependency_files:
  file_list:
  - eval.py
  - full_eval_out.json
  - mini_eval_out.json
  - preview_eval_out.json
</existing_artifacts>

<current_paper>
The current paper draft — represents the research story so far.

Use this to understand what's working, what's not, and what gaps remain.
Gaps and weak results signal what to try differently — not what to conclude.

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
Paper reviewer feedback from the previous iteration. Your strategy MUST address these critiques.
Prioritize major issues — these are the most impactful improvements to make.

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
Generate 1 research strategy for THIS iteration.

**ARTIFACT LIMIT: Each strategy may contain AT MOST 3 artifact directions.** Focus on the highest-impact artifacts. Quality over quantity.

Each strategy should:
1. Define a clear OBJECTIVE - what novel contribution we're building toward
2. Plan artifacts to execute NOW - specify type, objective, approach, and depends_on for each
3. Account for parallel execution - all strategies and all planned artifacts run simultaneously, their artifacts are combined into one shared pool


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
    "ArtifactDep": {
      "description": "A single dependency on an existing artifact, with a short type label.\n\n``id`` and ``label`` are LLM-generated at strategy time. ``label`` is free-text but\nshort \u2014 a word or two naming the type of dependency, not a sentence.\n\n``relation_type`` and ``relation_rationale`` are populated later, in upd_hypo,\nusing the MultiCite citation-function typology (Lauscher et al., NAACL 2022).\nThey are absent at strategy time and may stay absent for legacy runs.",
      "properties": {
        "id": {
          "description": "ID of an existing artifact this artifact depends on",
          "title": "Id",
          "type": "string"
        },
        "label": {
          "description": "Short free-text label naming the type of this dependency (a word or two, not a sentence)",
          "title": "Label",
          "type": "string"
        }
      },
      "required": [
        "id",
        "label"
      ],
      "title": "ArtifactDep",
      "type": "object"
    },
    "ArtifactDirection": {
      "description": "High-level direction for an artifact to execute this iteration.\n\nID is code-assigned (LLMPrompt only \u2014 visible in prompts, not LLM-generated).",
      "properties": {
        "type": {
          "description": "Type of artifact to create",
          "enum": [
            "experiment",
            "research",
            "proof",
            "evaluation",
            "dataset"
          ],
          "title": "Type",
          "type": "string"
        },
        "objective": {
          "description": "What we want to achieve with this artifact",
          "title": "Objective",
          "type": "string"
        },
        "approach": {
          "description": "High-level direction/method",
          "title": "Approach",
          "type": "string"
        },
        "depends_on": {
          "description": "Existing artifacts this depends on, each with a short type label",
          "items": {
            "$ref": "#/$defs/ArtifactDep"
          },
          "title": "Depends On",
          "type": "array"
        }
      },
      "required": [
        "type",
        "objective",
        "approach"
      ],
      "title": "ArtifactDirection",
      "type": "object"
    },
    "Strategy": {
      "description": "A research strategy.\n\nContent fields have LLMPrompt + LLMStructOut markers.\n``id`` is code-assigned (LLMPrompt only \u2014 visible in prompts, not LLM-generated).\n\nID format: gen_strat_idx{N}",
      "properties": {
        "title": {
          "description": "Short name for this strategy",
          "title": "Title",
          "type": "string"
        },
        "objective": {
          "description": "The novel contribution we're building toward",
          "title": "Objective",
          "type": "string"
        },
        "rationale": {
          "description": "Why this strategy is promising",
          "title": "Rationale",
          "type": "string"
        },
        "artifact_directions": {
          "description": "Artifacts to execute THIS iteration",
          "items": {
            "$ref": "#/$defs/ArtifactDirection"
          },
          "title": "Artifact Directions",
          "type": "array"
        },
        "expected_outcome": {
          "description": "What we'll have after this iteration's artifacts complete",
          "title": "Expected Outcome",
          "type": "string"
        },
        "summary": {
          "default": "",
          "description": "Brief summary of the strategy and its expected contribution",
          "title": "Summary",
          "type": "string"
        }
      },
      "required": [
        "title",
        "objective",
        "rationale",
        "artifact_directions",
        "expected_outcome"
      ],
      "title": "Strategy",
      "type": "object"
    }
  },
  "description": "Top-level wrapper for LLM strategy generation output.",
  "properties": {
    "strategies": {
      "description": "List of generated strategies",
      "items": {
        "$ref": "#/$defs/Strategy"
      },
      "title": "Strategies",
      "type": "array"
    }
  },
  "required": [
    "strategies"
  ],
  "title": "Strategies",
  "type": "object"
}
```

IMPORTANT: This task is NOT complete until you Write `./.terminal_claude_agent_struct_out.json`.
````

### [2] HUMAN-USER prompt · 2026-06-22 05:09:31 UTC

```
Build and evaluate a simple word-frequency based extractive text summarizer.
```

### [3] SYSTEM-USER prompt · 2026-06-22 05:10:15 UTC

```
<verification_results>
Your previous response had issues that need fixing:

DEPENDENCY ERRORS (depends_on can ONLY reference IDs from <existing_artifacts>):
  - Strategy 1: Artifact 'experiment_iter2_dir1' (experiment): dependency 'art_ZuUuuoy0rCRF' has type 'experiment' which is not allowed (allowed: {'dataset', 'research'})
  - Strategy 1: Artifact 'evaluation_iter2_dir2' (evaluation): dependency 'art_yWTseob35O3B' has type 'evaluation' which is not allowed (allowed: {'dataset', 'experiment'})

INSUFFICIENT VALID ARTIFACTS:
  Required: at least 1 valid artifacts
  Found: 0 valid out of 2 total
  Artifacts with invalid types, duplicate IDs, or invalid dependencies don't count as valid.

</verification_results>

<task>
Fix ALL issues above and regenerate your strategies:

1. Fix dependency errors:
   - depends_on is a list of {id, label} objects — every entry MUST have a non-empty short label
   - id can ONLY reference IDs from <existing_artifacts>
   - You CANNOT reference artifacts you are proposing in this strategy as dependencies (they all run in parallel)
   - Follow the dependency type rules (e.g., experiments require datasets)
   - If no suitable existing artifacts exist, use depends_on: []
2. Ensure at least 1 artifacts are fully valid (correct types, no ID conflicts, valid dependencies)

Output the corrected JSON with the fixed strategies.
</task>
```
