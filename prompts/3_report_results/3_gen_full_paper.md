# gen_full_paper — report_results

> Phase: `gen_paper_repo` · `gen_full_paper`
> Run: `run_CFTHGHhY9evP` — Vocabulary Percolation Threshold for Self-Calibrating Extractive Summary Length: Recall Advantage and F1 Failure Mode Across Corpora
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_full_paper` (terminal_claude_agent)

### [1] SYSTEM-USER prompt · 2026-06-22 06:21:16 UTC

````
<research_methodology>
Write like an experienced academic. Reviewers judge both the science and the writing.

- Claims must be proportional to evidence. Choose verbs carefully — "demonstrate," "observe," and "hypothesize" mean different things.
- Every result needs: what was measured, on what data, the numbers, and what they mean.
- Methodology must be specific enough to reproduce. Related work must be organized by theme, not a literature dump.
- State limitations honestly. Avoid both overclaiming and excessive hedging.
</research_methodology>

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

<workspace>
Your workspace: `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/4_gen_paper_repo/_4_assemble_paper/paper/workspace`

CRITICAL: Every file you create, write, or save MUST be inside this workspace directory (subdirectories OK). You MUST NOT write files anywhere outside this path — external paths are READ-ONLY. Use absolute paths for all file operations.

EVERY file write MUST start with `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/4_gen_paper_repo/_4_assemble_paper/paper/workspace/`:
GOOD: `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/4_gen_paper_repo/_4_assemble_paper/paper/workspace/file.py`, `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/4_gen_paper_repo/_4_assemble_paper/paper/workspace/results/out.json`
BAD: `/tmp/file.py`, `~/output.json`, `./file.py`, any path outside the workspace
</workspace>

<task>
Create a publication-ready top-conference LaTeX paper with BibTeX from <paper_text> and <available_figures>, compile to PDF.
</task>

<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → sequential (need URLs first)
</tool_use>

<paper_text>
title: >-
  Vocabulary Percolation Threshold for Self-Calibrating Extractive Summary Length: Recall Advantage and F1 Failure Mode Across
  Corpora
abstract: >-
  Every extractive summarization method—from TF-IDF scoring to graph-based systems such as TextRank and LexRank—requires the
  user to specify summary length as a fixed compression ratio, typically 20–30% of source sentences, chosen without regard
  to the individual document's structure. We propose and evaluate a self-calibrating stopping criterion based on vocabulary
  co-occurrence graph percolation: sentences (ranked by term frequency or TF-IDF score) are added to the summary until the
  giant connected component (GCC) of the induced vocabulary subgraph reaches a critical fraction θ of the full document's
  GCC. We evaluate this criterion on CNN/DailyMail (n=2000) and Multi-News (n=1000) with both TF and TF-IDF sentence scoring,
  comparing against fixed-ratio baselines at 10%, 20%, and 30%. On CNN/DailyMail, percolation at θ=0.9 achieves ROUGE-1 recall
  of 0.858—significantly above the best fixed-ratio baseline (fixed@30%: 0.578)—but fixed@20% dominates on ROUGE-1 F1 (0.284
  vs. 0.150), because the GCC fires homogeneously at 60–87% of sentences in inverted-pyramid newswire articles. On Multi-News,
  percolation fires substantially earlier (34.5% mean compression at θ=0.6), demonstrating that the criterion is sensitive
  to vocabulary clustering: multi-source documents aggregate distinct per-source vocabularies that the GCC can bridge at lower
  sentence counts. Nevertheless, fixed-ratio F1 dominates on Multi-News as well (fixed@10%: 0.385 vs. percolation θ=0.6: 0.280),
  because even multi-document reference summaries remain concise relative to merged source length. TF-IDF scoring reduces
  compression ratios moderately (37% vs. 60% at θ=0.6 on CNN/DailyMail) without closing the F1 gap. Graph structural features
  explain R²=0.12–0.13 of compression ratio variance—a relationship too weak for reliable dynamic threshold calibration. These
  results constitute a rigorous, cross-corpus empirical baseline for percolation-based length selection and identify the structural
  conditions under which it can and cannot succeed.
paper_text: |-
  # Introduction

  Extractive text summarization selects a subset of source sentences to form a summary. The central algorithmic question is *which* sentences to select, but an equally critical and consistently underexamined question is *how many* to select. Virtually every published method—from Luhn's frequency-based sentence scoring [1] to graph-based systems such as TextRank [2] and LexRank [3], to neural approaches such as SummaRuNNer [8], BERTSum [9], and PreSumm [10]—requires the user or system to specify a compression ratio as an external parameter, typically fixed at 20–30% by convention or tuned on held-out data.

  This fixed-ratio convention is poorly motivated. A news article covering a single self-contained event may be complete in two or three sentences; a multi-topic feature article of the same length may require ten to communicate its full scope. Applying a uniform 20% compression to both yields either a redundant summary (for focused articles) or a lossy one (for multi-topic articles). The problem is broadly acknowledged [2, 3] but no principled, document-specific stopping criterion has emerged in the extractive summarization literature.

  We import a stopping criterion from statistical physics. In Erdős–Rényi random graph theory [4], a network undergoes a sharp phase transition as edges are added: below a critical density, the graph consists of many small disconnected components; above it, a *giant connected component* (GCC) suddenly emerges spanning most nodes [5]. Transposed to text, the vocabulary co-occurrence graph of a growing summary starts fragmented when only a few sentences have been added (covering isolated topic clusters) and becomes globally connected as conceptually bridging sentences are included. The percolation threshold—where the summary's induced GCC reaches a critical fraction θ of the full document's GCC—signals that the summary has achieved conceptual percolation: all major vocabulary clusters are mutually reachable.

  The vocabulary co-occurrence graph of a document is not an Erdős–Rényi random graph—it has strong community structure, heavy-tailed degree distributions, and high clustering coefficients. The classical percolation threshold therefore does not apply analytically, and θ must be tuned empirically. What the analogy provides is a *principled directionality*: once the GCC fraction exceeds θ, the summary's vocabulary skeleton covers the document's. This criterion is document-specific, requires no training data or language models, and is computationally inexpensive.

  We evaluate this criterion on two corpora with TF and TF-IDF sentence scoring. On CNN/DailyMail, the percolation criterion consistently selects 60–87% of sentences, far above the 7–10% targeted by journalist-written highlights, and fixed-ratio baselines dominate on ROUGE F1. On Multi-News, the same criterion fires much earlier (35–74%), reflecting stronger vocabulary segregation in multi-source documents. Nevertheless, fixed ratios dominate F1 on both corpora because reference summaries are concise highlights, not coverage-complete extracts. These results delineate precisely when percolation-based length selection can and cannot succeed.

  [FIGURE:fig_pipeline]

  **Summary of Contributions.** (1) We introduce the first extractive summarizer using vocabulary co-occurrence graph percolation as a self-calibrating stopping criterion, with a complete open-source implementation (§3). (2) We provide a rigorous evaluation on CNN/DailyMail (n=2000) and Multi-News (n=1000), comparing four percolation thresholds against three fixed-ratio baselines with both TF and TF-IDF sentence scoring, with Holm-corrected Wilcoxon tests across all pairwise comparisons (§4). \footnote{Code: \url{https://github.com/AMGrobelnik/ai-invention-6a9498-vocabulary-percolation-threshold-for-sel/tree/main/round-2/experiment-1}} (3) We report and analyze a structured negative finding: percolation maximizes recall but not F1, and we identify the structural reasons—newswire vocabulary distribution in CNN/DailyMail and the conciseness of all human reference summaries (§5). (4) We characterize how the percolation firing point varies across corpora (§4.3), showing that Multi-News's multi-source vocabulary clustering produces meaningfully different behavior from CNN/DailyMail's inverted-pyramid structure. (5) We quantify that graph structural features explain only R²≈0.12 of compression variance, insufficient for reliable dynamic calibration (§4.4).

  # Related Work

  **Extractive summarization.** Luhn [1] introduced frequency-based sentence scoring. TextRank [2] applies PageRank [7] over a sentence-similarity graph and selects the top-*k* sentences; LexRank [3] uses eigenvector centrality over cosine-similarity sentences. Both require a fixed *k*. SummaRuNNer [8] uses a recurrent neural network to score sentences sequentially with a fixed budget. BERTSum [9] and PreSumm [10] apply pretrained transformer encoders to sentence scoring. Neural abstractive methods, including pointer-generator networks [11], similarly rely on externally specified length constraints. None of these methods determines *k* from the document's intrinsic structure.

  Ryang and Abekawa [12] use an influence propagation model on a word co-occurrence network to maximize coverage under a budget constraint. The budget remains manually specified; the phase transition is not used to determine when coverage is complete. Our work shares the word-network foundation but replaces the manual budget with the percolation threshold.

  **Graph percolation in NLP.** Erdős–Rényi percolation theory [4, 5] has been applied to information diffusion in social networks and language evolution, but to our knowledge has not previously been proposed as a stopping criterion for extractive summarization.

  **CNN/DailyMail and Multi-News benchmarks.** Hermann et al. [6] introduced CNN/DailyMail for reading comprehension; it has since become the standard single-document summarization benchmark [11]. The reference summaries are journalist-written bullet-point highlights (mean 42 words). Fabbri et al. [13] introduced Multi-News, a multi-document corpus aggregating 2–10 news articles per story with a single reference summary. The multi-source structure gives Multi-News stronger vocabulary segregation across its constituent documents, making it a natural testbed for percolation-based stopping.

  # Methods

  ## Vocabulary Co-occurrence Graph Construction

  Given a document with sentences $s_1, \ldots, s_n$, we preprocess each sentence by tokenizing, lowercasing, removing NLTK English stop words, and retaining alphabetic tokens of length ≥ 2. Let $W_i$ denote the content word set of sentence $s_i$.

  The vocabulary co-occurrence graph $G = (V, E, w)$ is an undirected weighted graph. Nodes $V$ are unique content words across the document. An edge $(u, v) \in E$ exists between any two words $u, v \in W_i$ for some sentence $s_i$; the edge weight $w(u,v)$ equals the number of sentences in which $u$ and $v$ co-occur. The full-document GCC, $\text{GCC}_{\text{doc}}$, is the node count of the largest connected component of $G$. \footnote{Code: \url{https://github.com/AMGrobelnik/ai-invention-6a9498-vocabulary-percolation-threshold-for-sel/tree/main/round-1/experiment-1}}

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

  Table 1 reports mean ROUGE scores on CNN/DailyMail (n=2000).  The results expose a structural tension between recall and precision.

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

  Table 2 reports ROUGE-1 results on Multi-News (TF scorer; n=1000). \footnote{Code: \url{https://github.com/AMGrobelnik/ai-invention-6a9498-vocabulary-percolation-threshold-for-sel/tree/main/round-2/evaluation-1}}

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
summary: >-
  We propose and rigorously evaluate a percolation-threshold stopping criterion for extractive summarization: sentences are
  added until the induced vocabulary co-occurrence subgraph's giant connected component reaches fraction θ of the full document's
  GCC. On CNN/DailyMail (n=2000), the criterion achieves ROUGE-1 recall of 0.858 at θ=0.9 (vs. 0.578 for fixed@30%) but fails
  on F1 (best perc. 0.205 vs. fixed@20% 0.284) because newswire vocabulary is distributed throughout articles and requires
  60–87% of sentences to percolate. On Multi-News (n=1000), percolation fires earlier (34.5% at θ=0.6), confirming the theoretical
  prediction about vocabulary clustering in multi-source documents, but fixed ratios still dominate F1. TF-IDF scoring reduces
  compression moderately without closing the F1 gap. Graph features explain only R²≈0.12 of compression variance, ruling out
  dynamic calibration from topology alone. The results establish a clear empirical baseline and identify the structural conditions—coverage-oriented
  references and diverse vocabulary clustering across document types—under which percolation-based length selection is most
  likely to succeed.
</paper_text>

<available_figures>
--- Item 1 ---
id: fig_pipeline
title: Percolation-Threshold Extractive Summarization Pipeline
caption: >-
  Overview of the percolation-threshold extractive summarization pipeline. A document's sentences are scored (TF or TF-IDF)
  and sorted. Sentences are added greedily; after each addition, the induced vocabulary co-occurrence subgraph is updated
  and its GCC size tracked. When the GCC fraction $\text{GCC}_{\text{summary}}/\text{GCC}_{\text{doc}}$ reaches threshold
  $\theta$, sentence selection stops. The result is a self-calibrating summary whose length is determined by the document's
  own vocabulary graph structure.
image_gen_detailed_description: >-
  Horizontal left-to-right flow diagram with five main stages connected by thick arrows, on a clean white background with
  sans-serif font. Stage 1 (gray box, leftmost): 'Document' — shows a small document icon with 10 horizontal lines representing
  sentences. Stage 2 (blue box): 'Sentence Scoring (TF / TF-IDF)' — shows a bar chart inside the box with bars of varying
  heights labeled s1 through s5 on x-axis. Stage 3 (green box): 'Greedy Addition Loop' — shows a circular arrow icon inside
  indicating iteration. Below this box, a small annotation reads 'add highest-scoring sentence'. Stage 4 (orange box): 'GCC
  Fraction Tracking' — shows a small line graph rising from left to right with a dashed horizontal line labeled 'θ' that the
  curve crosses, with x-axis labeled 'Sentences added' and y-axis labeled 'GCC fraction'. Stage 5 (purple box, rightmost):
  'Summary Output' — shows 3 lines representing selected sentences. Between Stage 3 and Stage 4, a downward-curved arrow labeled
  'update induced subgraph' connects Stage 3 back to itself, and a rightward arrow from Stage 4 labeled 'GCC ≥ θ → STOP' goes
  to Stage 5. At the bottom, a note box: 'θ ∈ {0.6, 0.7, 0.8, 0.9}'. All boxes have rounded corners, color-coded backgrounds,
  white text labels. Arrow from Stage 4 back to Stage 3 labeled 'GCC < θ → continue'. Clean academic diagram style.
aspect_ratio: '21:9'
summary: >-
  Hero pipeline diagram showing how the percolation-threshold stopping criterion works: score sentences, add greedily, track
  GCC fraction, stop at threshold.
figure_path: figures/fig_pipeline_v0.jpg

--- Item 2 ---
id: fig_rouge_cnndm
title: 'ROUGE-1 F1 and Recall: Percolation vs. Fixed-Ratio on CNN/DailyMail'
caption: >-
  ROUGE-1 F1 (left panel) and ROUGE-1 Recall (right panel) for all evaluated methods on CNN/DailyMail (n=2000). Percolation
  variants (TF and TF-IDF, θ=0.6–0.9) are shown in blue/teal; fixed-ratio baselines (TF, 10%/20%/30%) are shown in orange.
  Fixed-ratio baselines dominate on F1; percolation achieves substantially higher recall at the cost of lower precision.
image_gen_detailed_description: >-
  Side-by-side two-panel grouped bar chart on white background with sans-serif font. Left panel title: 'ROUGE-1 F1'. Right
  panel title: 'ROUGE-1 Recall'. X-axis labels (same for both panels, rotated 30 degrees): 'TF θ=0.6', 'TF θ=0.7', 'TF θ=0.8',
  'TF θ=0.9', 'TF-IDF θ=0.6', 'TF-IDF θ=0.7', 'Fixed 10%', 'Fixed 20%', 'Fixed 30%'. Y-axis: 0.0 to 1.0 for Recall panel,
  0.0 to 0.35 for F1 panel. Left panel (F1) bar values: TF θ=0.6: 0.187, TF θ=0.7: 0.173, TF θ=0.8: 0.161, TF θ=0.9: 0.150,
  TF-IDF θ=0.6: 0.205, TF-IDF θ=0.7: 0.188, Fixed 10%: 0.274, Fixed 20%: 0.284, Fixed 30%: 0.259. Right panel (Recall) bar
  values: TF θ=0.6: 0.772, TF θ=0.7: 0.808, TF θ=0.8: 0.834, TF θ=0.9: 0.858, TF-IDF θ=0.6: 0.743, TF-IDF θ=0.7: 0.784, Fixed
  10%: 0.296, Fixed 20%: 0.468, Fixed 30%: 0.578. Colors: percolation TF bars in steel blue (#4472C4), percolation TF-IDF
  bars in teal (#5DA0A0), fixed-ratio bars in orange (#ED7D31). A horizontal dashed red line in the F1 panel at 0.284 labeled
  'best fixed F1'. A horizontal dashed red line in the Recall panel at 0.858 labeled 'best perc. recall'. Legend in upper
  right.
aspect_ratio: '21:9'
summary: >-
  Dual bar chart showing the recall-precision tradeoff: percolation excels at recall, fixed ratios win on F1, across all tested
  variants on CNN/DailyMail.
figure_path: figures/fig_rouge_cnndm_v0.jpg

--- Item 3 ---
id: fig_cross_corpus
title: Compression Ratio and ROUGE-1 F1 Across Corpora and Thresholds
caption: >-
  Left panel: Mean compression ratio (fraction of sentences selected) vs. percolation threshold θ for CNN/DailyMail (blue)
  and Multi-News (green), TF scorer. Multi-News percolation fires at substantially lower compression ratios than CNN/DailyMail,
  reflecting stronger vocabulary clustering in multi-source documents. Right panel: ROUGE-1 F1 vs. compression ratio for all
  method-corpus combinations, showing that fixed-ratio baselines (orange squares for CNN/DM, red squares for Multi-News) dominate
  F1 at any given compression level.
image_gen_detailed_description: >-
  Two-panel figure on white background with sans-serif font. Left panel: Line chart titled 'Compression Ratio vs. Threshold
  θ'. X-axis: theta values 0.6, 0.7, 0.8, 0.9. Y-axis: Compression ratio (fraction), 0.0 to 1.0. Two lines with markers: CNN/DailyMail
  (blue solid line with circle markers) with values θ=0.6: 0.597, θ=0.7: 0.686, θ=0.8: 0.773, θ=0.9: 0.866. Multi-News (green
  dashed line with triangle markers) with values θ=0.6: 0.345, θ=0.7: 0.457, θ=0.8: 0.601, θ=0.9: 0.742. A horizontal dotted
  gray line at 0.10 labeled '10% reference target'. Legend in upper left: 'CNN/DailyMail (TF)' and 'Multi-News (TF)'. Right
  panel: Scatter plot titled 'ROUGE-1 F1 vs. Compression Ratio'. X-axis: Compression ratio 0.0 to 1.0. Y-axis: ROUGE-1 F1,
  0.0 to 0.45. Points: CNN/DM percolation TF (blue circles): (0.597, 0.187), (0.686, 0.173), (0.773, 0.161), (0.866, 0.150).
  CNN/DM percolation TF-IDF (teal circles): (0.369, 0.205), (0.463, 0.188). CNN/DM fixed-ratio (orange squares): (0.10, 0.274),
  (0.20, 0.284), (0.30, 0.259). Multi-News percolation TF (green triangles): (0.345, 0.280), (0.457, 0.253), (0.601, 0.226),
  (0.742, 0.207). Multi-News fixed-ratio (red squares): (0.10, 0.385), (0.20, 0.339), (0.30, 0.296). Labels on fixed-ratio
  squares showing their ratio value. Legend in lower right.
aspect_ratio: '21:9'
summary: >-
  Shows that Multi-News percolation fires earlier than CNN/DM due to vocabulary clustering, and that fixed ratios Pareto-dominate
  percolation on the compression vs F1 tradeoff.
figure_path: figures/fig_cross_corpus_v0.jpg
</available_figures>

<figure_requirements>
CRITICAL: Include ALL figures from <available_figures>. No exceptions.

- Every figure MUST use \includegraphics{figures/filename.jpg}
- Do NOT skip, convert to tables, or describe without inserting
- Each needs: \begin{figure*|figure}[placement], \includegraphics, \caption, \label, \end{...} — pick env + placement by the figure's `aspect_ratio` field (see PLACEMENT below). Constrain every \includegraphics with `width=\linewidth,height=0.4\textheight,keepaspectratio` (single-column) or `width=\textwidth,height=0.45\textheight,keepaspectratio` (figure*). Use exactly these option keys — `max height=` is NOT valid LaTeX
- Use the `caption` field from each figure for \caption{...} — do NOT invent new captions
- Place figures where their [FIGURE:fig_id] markers appear in paper_text
- VERIFICATION: paper.tex MUST have exact same number of \includegraphics as <available_figures>
- Do NOT generate new figure images (no matplotlib, no PIL, no image generation). Use ONLY the pre-generated figures from <available_figures>. They were already created by a previous pipeline step.

PLACEMENT BY ASPECT RATIO (use the `aspect_ratio` field on each figure):
- `21:9` (architecture diagrams / hero figures): \begin{figure*}[!t] (full two-column width, top of page). The hero architecture diagram should appear EARLY in the paper — typically at the top of page 2. Marker placement in paper_text already determines this; preserve it.
- `16:9` (comparisons, multi-panel results): \begin{figure*}[!t] for full-width or \begin{figure}[!htbp] for single-column.
- `4:3` / `1:1` / `3:2` / `3:4` / `9:16`: \begin{figure}[!htbp] (single-column).
</figure_requirements>

<artifact_links>
The paper_text contains \footnote{Code: \url{...}} references linking to artifact source code
on GitHub. Include \usepackage{hyperref} and \usepackage{url}.
Preserve these exactly as-is — do not remove, rewrite, or convert them to plain text.
The URLs will not resolve yet (the repo is deployed after compilation) — do NOT try to verify or fix them.
</artifact_links>

<headings>
NEVER use inline math (``$...$``) inside ``\section{...}`` / ``\subsection{...}`` / ``\subsubsection{...}`` arguments — hyperref's bookmark builder errors out (``Token not allowed in a PDF string``) and the PDF outline breaks. If a section heading needs a math-looking term, use the text equivalent (``d star`` not ``$d^*$``, ``alpha-equivalent`` not ``$\alpha$-equivalent``) or wrap it in ``\texorpdfstring{$math$}{plain}``. Inline math inside body paragraphs is fine.
</headings>

FIRST, add ALL of these to your todo list using your task/todo-tracking tool:

CRITICAL: Todo content must be copied exactly as is written here, with NO CHANGES. These todos are intentionally detailed so that another LLM could read each one without any external context and understand exactly what it has to do.

<todos>
TODO 1. Read and STRICTLY follow these skills: aii-paper-to-latex, aii-semscholar-bib.
TODO 2. Review <paper_text> and <available_figures>. Copy all figure images into ./figures/ in your workspace. Count figures — MUST include every one. Plan placements per section. Build `./references.bib` via aii_semscholar_bib__fetch — collect DOIs/ArXiv IDs from <paper_text> and batch-fetch all BibTeX in one call. Do NOT fabricate entries.
TODO 3. Create `./paper.tex` per aii-paper-to-latex skill's setup, write ALL sections, insert ALL figures from <available_figures>, include `./references.bib` via \bibliography. Compile to PDF per skill's process. Fix errors.
TODO 4. CRITICAL VERIFICATION: Run `grep -c 'includegraphics' paper.tex`, confirm count equals figures in <available_figures>. If not, add missing figures. Verify `./paper.pdf` was created.
TODO 5. VISUAL REVIEW: Write Python script to convert EVERY page of paper.pdf to PNG at 150 DPI (use pdf2image or pymupdf). Then read ALL page screenshots — each page image costs ~1,600 tokens so a 15-page paper is only ~24K tokens. You MUST read every page. The ONLY exception is if all page images would not fit in your remaining context — in that case, read as many as fit and state which pages you are skipping and why. Check every page for layout issues, overlapping figures, cut-off text, bad spacing, formatting problems. Fix issues and recompile.
TODO 6. FINAL READ: Check page count (`pdfinfo paper.pdf` or pymupdf). Read entire paper.pdf — check for missing sections, unclear explanations, inconsistencies, typos. Fix and recompile. The ONLY exception is if all pages would not fit in your remaining context — in that case, read as many pages as fit and state which pages you are skipping and why.
</todos>

---

Output the result as JSON to: `./.terminal_claude_agent_struct_out.json`

JSON Schema:
```json
{
  "$defs": {
    "FullPaperExpectedFiles": {
      "description": "All expected output files from full paper generation.",
      "properties": {
        "paper_tex_path": {
          "description": "Path to LaTeX source file. Example: 'paper.tex'",
          "title": "Paper Tex Path",
          "type": "string"
        },
        "paper_pdf_path": {
          "description": "Path to compiled PDF. Example: 'paper.pdf'",
          "title": "Paper Pdf Path",
          "type": "string"
        },
        "references_bib_path": {
          "description": "Path to BibTeX bibliography file. Example: 'references.bib'",
          "title": "References Bib Path",
          "type": "string"
        },
        "figure_paths": {
          "description": "Paths to all figure image files. Example: ['figures/fig1_v0.jpg', 'figures/fig2_v0.jpg']",
          "items": {
            "type": "string"
          },
          "title": "Figure Paths",
          "type": "array"
        }
      },
      "required": [
        "paper_tex_path",
        "paper_pdf_path",
        "references_bib_path",
        "figure_paths"
      ],
      "title": "FullPaperExpectedFiles",
      "type": "object"
    }
  },
  "description": "Full paper \u2014 structured output from paper generation.",
  "properties": {
    "title": {
      "description": "Short descriptive title for this paper generation task (roughly 30-90 characters)",
      "maxLength": 90,
      "minLength": 30,
      "title": "Title",
      "type": "string"
    },
    "summary": {
      "description": "Brief summary of the generated paper: sections written, figures included, compilation status",
      "maxLength": 5000,
      "minLength": 500,
      "title": "Summary",
      "type": "string"
    },
    "out_expected_files": {
      "$ref": "#/$defs/FullPaperExpectedFiles",
      "description": "All output files you created. Must include paper.tex, paper.pdf, references.bib, and paths to all figure files."
    }
  },
  "required": [
    "title",
    "summary",
    "out_expected_files"
  ],
  "title": "FullPaper",
  "type": "object"
}
```

IMPORTANT: This task is NOT complete until you Write `./.terminal_claude_agent_struct_out.json`.
````

### [2] HUMAN-USER prompt · 2026-06-22 06:21:16 UTC

```
Build and evaluate a simple word-frequency based extractive text summarizer.
```

### [3] SKILL-INPUT — aii-paper-to-latex · 2026-06-22 06:21:22 UTC

The agent loaded the **aii-paper-to-latex** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-paper-to-latex
description: LaTeX paper assembly and compilation. Covers document setup, figure inclusion from pre-generated JPEGs, compilation process, and output files. Use when assembling a paper from pre-written text and pre-generated figures into a compiled PDF.
---

## LaTeX Paper Assembly

Assembles a research paper from paper text, pre-generated figure JPEGs, and bibliography into a compiled PDF.

### Document Setup

```latex
\documentclass[11pt,letterpaper]{article}
\usepackage{graphicx, geometry, amsmath, hyperref, natbib, booktabs, xcolor, listings}
\geometry{margin=1in}
\hypersetup{colorlinks=true, linkcolor=black, citecolor=black, urlcolor=black}
```

### Figure Inclusion

CRITICAL: Include ALL figures. Every figure MUST appear in the paper.

```latex
\begin{figure}[!htbp]
  \centering
  \includegraphics[width=0.92\textwidth,keepaspectratio]{figures/filename.jpg}
  \caption{Descriptive caption.}
  \label{fig:label}
\end{figure}
```

Rules:
- ALWAYS use `[!htbp]` float placement (NOT `[t]` or `[h]` alone)
- ALWAYS constrain with `width` and `keepaspectratio` to prevent page takeover
- Every figure needs `\caption`, `\label`, and a `\ref` in the text
- Do NOT convert figures to tables or describe them without inserting the image
- Do NOT skip any figures

### Compilation Process

Run each command separately (do NOT chain with `&&` — pdflatex often exits non-zero on warnings, which would skip bibtex and leave citations as `??`):

```bash
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

All four commands are required. Skipping bibtex causes `??` in all citations.
Fix any errors between runs. Verify `./paper.pdf` was created.

### Output Files

- `./paper.tex` — LaTeX source
- `./references.bib` — bibliography file
- `./paper.pdf` — compiled PDF
- `./figures/*.jpg` — all figure images (pre-generated, copied into workspace)
````

### [4] SKILL-INPUT — aii-semscholar-bib · 2026-06-22 06:21:29 UTC

The agent loaded the **aii-semscholar-bib** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-semscholar-bib
description: Build bibliographies using Semantic Scholar. Batch-fetch BibTeX for papers by DOI, ArXiv ID, or title. Use when writing papers, generating reference lists, or building .bib files.
---

## Tool: `aii_semscholar_bib__fetch`

Batch-fetch BibTeX entries from Semantic Scholar. Pass all references in a single call — the tool handles batching internally.

### How it works

1. **DOI/ArXiv refs** → batched into POST /paper/batch calls (up to 500 per API call, auto-chunked)
2. **Title-only refs** → individual GET /paper/search/match (1s delay between)
3. **Post-process** → fix entry type, fix citation key (AuthorYYYY), inject DOI

The ability server runs a single worker (`max_threads: 1`). Multiple concurrent tool calls are queued — each runs independently (no cross-request aggregation). Batching happens within each request.

### Input format

```json
{
  "references": [
    {"doi": "10.48550/arXiv.1706.03762", "author": "Vaswani", "year": 2017},
    {"arxiv": "2201.11903", "author": "Wei", "year": 2022},
    {"title": "Tree of Thoughts", "author": "Yao", "year": 2023}
  ]
}
```

Each reference object can have:
- `doi` — DOI string (ArXiv DOIs like `10.48550/arXiv.XXXX.XXXXX` auto-convert to ArXiv IDs)
- `arxiv` — ArXiv ID (e.g. `"2305.14325"`)
- `title` — Paper title (used for search/match when no DOI/ArXiv)
- `author` — First author last name (for cleaner citation key)
- `year` — Publication year (int, for citation key)

At least one of `doi`, `arxiv`, or `title` is required per reference.

### Output format

```json
{
  "success": true,
  "bib_text": "@inproceedings{Vaswani2017, ...}\n\n@article{Wei2022, ...}",
  "total": 3,
  "found": 3,
  "failed_count": 0,
  "entries": [{"citation_key": "Vaswani2017", "bibtex": "...", "title": "...", "doi": "...", "arxiv": ""}],
  "failed": []
}
```

### Workflow

1. Collect DOIs, ArXiv IDs, or titles for all papers you need to cite
2. Call `aii_semscholar_bib__fetch` with the full list in **one call**
3. Save `bib_text` from the response to your `references.bib` file
4. Check `failed` — for any missed papers, follow the **fallback procedure** below

### Fallback for failed references (MANDATORY)

NEVER fabricate BibTeX. For each failed reference:
1. **WebSearch** for `"Title" author year` (try `site:arxiv.org` too)
2. **WebFetch** the paper page → extract title, authors, year, venue, DOI/ArXiv ID
3. If DOI/ArXiv found → retry `aii_semscholar_bib__fetch` with it
4. Last resort: write BibTeX by hand using **only verified info from the actual paper page**

---

### CLI (for manual use / debugging)

```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-semscholar-bib" && \
$SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_semscholar_bib__fetch.py --refs '[
  {"doi": "10.48550/arXiv.1706.03762", "author": "Vaswani", "year": 2017},
  {"arxiv": "2201.11903", "author": "Wei", "year": 2022},
  {"title": "Tree of Thoughts", "author": "Yao", "year": 2023}
]'
```

`--json, -j` — output raw JSON instead of .bib text

**If the script fails** with a connection error (ability server not running): create a local `.venv`, install server deps from `server_requirements.txt` into it, then import the `@aii_ability` function from the script and call it directly — bypassing the server:
```bash
uv venv .venv --python=3.12 && uv pip install --python=.venv/bin/python -r "$SKILL_DIR/scripts/server_requirements.txt"
```
````
