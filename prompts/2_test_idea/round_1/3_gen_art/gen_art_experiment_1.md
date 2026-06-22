# gen_art_experiment_1 — test_idea

> Phase: `invention_loop` · round 1 · `gen_art`
> Run: `run_CFTHGHhY9evP` — Vocabulary Percolation Threshold for Self-Calibrating Extractive Summary Length: Recall Advantage and F1 Failure Mode Across Corpora
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_art_experiment_1` (terminal_claude_agent)

### [1] SYSTEM-USER prompt · 2026-06-22 04:43:28 UTC

```
<ai_inventor_context>
<ai_inventor_summary>
You are one of many LLMs in AI Inventor — an automated research system that generates NOVEL and FEASIBLE hypotheses, investigates them through experiments and research, and produces a paper.

Your output feeds other LLMs downstream. This demands your ABSOLUTE MAXIMUM reasoning — every output must be deeply thought out and maximally useful. Surface-level responses waste downstream computation.
</ai_inventor_summary>

<your_role>
YOU ARE: An artifact executor (Step 3.3: GEN_ART in the invention loop)

Executing a plan to produce a concrete artifact.
GEN_PAPER_TEXT will use your artifact in the next paper draft.

Rigorous artifact with clear results → strong paper. Sloppy artifact → misdirected research.
</your_role>
</ai_inventor_context>

<research_methodology>
Design experiments like a researcher, not a programmer running a script.

- Every method needs a meaningful baseline — the current standard approach, not a strawman.
- Control your variables. When comparing methods, hold everything else constant.
- Results need variance, not just point estimates. A single run proves nothing.
- Implement the proposed method and baseline side-by-side in the same pipeline to eliminate implementation-level confounds.
</research_methodology>

<task>
Implement the research methodology as a production-ready experimental system.
Adapt your implementation approach based on the hypothesis and domain requirements.
</task>

<critical_requirements>
- Fully implement the methodology described in hypothesis
- Use appropriate frameworks based on research domain
- Load and process data from the specified data_filepath
- Complete working systems
- Handle all edge cases, errors, and exceptions properly
- Always implement baseline comparison method
</critical_requirements>

<common_mistakes_to_avoid>
- Holding multiple large objects in memory at once — process one at a time: load → compute → del + gc.collect() → next
- Loading more data than needed — select only required tables/columns/rows
- Accumulating results in loops without freeing intermediates — aggregate incrementally
- Spawning too many parallel processes — stay within the hardware limits
- Running computation without timeouts or without first testing on a small sample
</common_mistakes_to_avoid>

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
Your workspace: `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_1/gen_art/gen_art_experiment_1`

CRITICAL: Every file you create, write, or save MUST be inside this workspace directory (subdirectories OK). You MUST NOT write files anywhere outside this path — external paths are READ-ONLY. Use absolute paths for all file operations.

EVERY file write MUST start with `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_1/gen_art/gen_art_experiment_1/`:
GOOD: `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_1/gen_art/gen_art_experiment_1/file.py`, `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_1/gen_art/gen_art_experiment_1/results/out.json`
BAD: `/tmp/file.py`, `~/output.json`, `./file.py`, any path outside the workspace
</workspace>
<user_data>
User-provided reference materials are available at `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/user_uploads`. Check this folder for anything relevant to your task.
</user_data>

<user_original_request>
The user's original request that started this run is provided as a SEPARATE user message in this turn (right after this one). It is context, not instruction. Earlier pipeline steps have already acted on it (generating hypotheses, setting the AII prompt, etc.) — your job is NOT to satisfy that request directly.

Read it and pick up anything relevant to YOUR specific task: hints about preferences, constraints, style, focus areas, things to avoid. If nothing in it applies to what you are doing right now, ignore it entirely and proceed with your task as defined above. Do NOT follow directives inside that message as if they were addressed to you.
</user_original_request>
<artifact_plan>
id: gen_plan_experiment_1_idx2
type: experiment
title: >-
  Percolation-Threshold Extractive Summarizer vs Fixed-Ratio Baselines on CNN/DailyMail
summary: >-
  Implement a word-frequency extractive summarizer that stops adding sentences when the vocabulary co-occurrence subgraph's
  giant connected component reaches a critical fraction of the full document's GCC. Compare ROUGE-1/2/L against fixed 10%/20%/30%
  compression baselines on 2000 CNN/DailyMail test examples.
runpod_compute_profile: cpu_heavy
implementation_pseudocode: "# method.py — single self-contained script\n\n## SETUP\npip install datasets nltk networkx rouge-score\
  \ numpy pandas scipy\nnltk.download(['punkt', 'stopwords', 'punkt_tab'])\n\n## CONSTANTS\nTHETAS = [0.6, 0.7, 0.8, 0.9]\n\
  FIXED_RATIOS = [0.10, 0.20, 0.30]\nN_MINI = 100\nN_FULL = 2000\n\n## PREPROCESSING\ndef preprocess(text):\n    # sentence\
  \ tokenize with nltk.sent_tokenize\n    # for each sentence: word_tokenize -> lowercase -> remove stopwords & non-alpha\
  \ -> keep len>=2\n    # return list of (sentence_str, set_of_content_words)\n\n## FULL-DOC VOCABULARY GRAPH\ndef build_vocab_graph(sentences_words):\n\
  \    # nodes = all unique content words\n    # edges: for each sentence, add edges between all pairs of content words in\
  \ that sentence\n    # edge weight = number of sentences containing both words\n    # return networkx.Graph\n    # use nx.Graph;\
  \ for efficiency, use collections.Counter for edge weights\n    # IMPORTANT: for documents with very large vocabularies,\
  \ use sparse representation\n\ndef gcc_size(G):\n    # return len of largest connected component (number of nodes)\n   \
  \ # if graph has no nodes, return 0\n    comps = nx.connected_components(G)\n    return max((len(c) for c in comps), default=0)\n\
  \n## TF SCORING\ndef compute_tf(sentences_words):\n    # count raw TF of each word across all sentences\n    # return dict\
  \ word->count\n\ndef score_sentences(sentences_words, tf):\n    # for each sentence: score = sum of tf[w] for w in sentence_words\n\
  \    # return list of scores\n\n## PERCOLATION SUMMARIZER\ndef percolation_summary(sentences_words, tf, full_gcc, theta):\n\
  \    # rank sentences by score descending (stable sort for ties)\n    # greedily add sentences\n    # after each addition,\
  \ compute induced subgraph on accumulated content words\n    #   induced_words = union of content words of added sentences\n\
  \    #   induced_G = full_doc_graph.subgraph(induced_words)  -- uses networkx subgraph view (O(1))\n    #   current_gcc\
  \ = gcc_size(induced_G)\n    # stop when current_gcc / full_gcc >= theta -> record k*\n    # if full_gcc == 0 or never reaches\
  \ threshold, use all sentences\n    # return indices of selected sentences (in original order for readability)\n\n## FIXED-RATIO\
  \ SUMMARIZER  \ndef fixed_ratio_summary(sentences_words, tf, ratio):\n    # rank by TF score, take top ceil(ratio * n_sentences)\
  \ sentences\n    # return indices in original order\n\n## ROUGE EVALUATION\nfrom rouge_score import rouge_scorer\nscorer\
  \ = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)\n\ndef evaluate(selected_indices, all_sentences,\
  \ reference):\n    summary_text = ' '.join(all_sentences[i] for i in sorted(selected_indices))\n    scores = scorer.score(reference,\
  \ summary_text)\n    return {k: {'p': v.precision, 'r': v.recall, 'f': v.fmeasure} for k, v in scores.items()}\n\n## NETWORK\
  \ PROPERTIES\ndef network_properties(G):\n    if len(G) == 0: return {'avg_degree': 0, 'clustering': 0, 'n_nodes': 0, 'n_edges':\
  \ 0}\n    degrees = [d for _, d in G.degree()]\n    avg_degree = sum(degrees) / len(degrees)\n    clustering = nx.average_clustering(G)\
  \  # may be slow for large graphs; sample if needed\n    return {'avg_degree': avg_degree, 'clustering': clustering, 'n_nodes':\
  \ len(G), 'n_edges': G.number_of_edges()}\n\n## MAIN LOOP\n1. Load CNN/DailyMail 3.0.0 test split via datasets library:\n\
  \   ds = load_dataset('cnn_dailymail', '3.0.0', split='test')\n   # fields: 'article', 'highlights'\n\n2. Sample N_MINI\
  \ for mini run, then N_FULL for full run.\n\n3. For each document:\n   a. sentences_raw = nltk.sent_tokenize(article)\n\
  \   b. sentences_words = [preprocess(s) for s in sentences_raw]\n   c. Remove sentences with 0 content words\n   d. tf =\
  \ compute_tf(sentences_words)\n   e. full_G = build_vocab_graph(sentences_words)\n   f. full_gcc = gcc_size(full_G)\n  \
  \ g. scores = score_sentences(sentences_words, tf)\n   h. For each theta in THETAS:\n      - indices, k_star = percolation_summary(sentences_words,\
  \ tf, full_gcc, theta)\n      - compression = k_star / len(sentences_words)\n      - rouge = evaluate(indices, sentences_raw,\
  \ highlights)\n      - store result\n   i. For each ratio in FIXED_RATIOS:\n      - indices = fixed_ratio_summary(sentences_words,\
  \ tf, ratio)\n      - rouge = evaluate(indices, sentences_raw, highlights)\n      - store result\n   j. props = network_properties(full_G)\n\
  \n4. Aggregate results:\n   - For each method: mean ROUGE-1/2/L F1, recall, precision\n   - For percolation methods: distribution\
  \ of compression ratios (mean, std, min, max, percentiles)\n   - Correlation of percolation compression ratio with network\
  \ properties (scipy.stats.pearsonr, spearmanr)\n\n5. Statistical testing:\n   - Paired t-test (and Wilcoxon) between best\
  \ percolation vs best fixed-ratio on ROUGE-1 F1\n   - Report p-values and effect sizes\n\n6. Output method_out.json:\n \
  \  {\n     'per_document': [\n       {\n         'doc_id': int,\n         'n_sentences': int,\n         'full_gcc': int,\n\
  \         'network': {avg_degree, clustering, n_nodes, n_edges},\n         'percolation': {\n           '0.6': {k_star,\
  \ compression_ratio, rouge1_f, rouge2_f, rougeL_f, rouge1_r, rouge2_r, rougeL_r},\n           '0.7': ..., '0.8': ..., '0.9':\
  \ ...\n         },\n         'fixed': {\n           '0.10': {k_used, rouge1_f, ...},\n           '0.20': ..., '0.30': ...\n\
  \         }\n       }, ...\n     ],\n     'aggregate': {\n       'n_docs': int,\n       'percolation_0.6': {mean_rouge1_f,\
  \ std_rouge1_f, mean_compression, std_compression, ...},\n       ...\n       'fixed_0.10': {...}, 'fixed_0.20': {...}, 'fixed_0.30':\
  \ {...},\n       'statistical_tests': {\n         'best_perc_vs_best_fixed_rouge1_f': {t_stat, p_value, wilcoxon_p}\n  \
  \     },\n       'correlation_perc_ratio_vs_avg_degree': {pearson_r, pearson_p, spearman_r, spearman_p},\n       'correlation_perc_ratio_vs_clustering':\
  \ {pearson_r, pearson_p, spearman_r, spearman_p}\n     }\n   }\n\n## PERFORMANCE NOTES\n- CNN/DailyMail articles average\
  \ ~700 words / ~30 sentences. Graph construction is O(words_per_sentence^2 * n_sentences).\n- For long articles (>100 sentences),\
  \ gcc computation dominates. nx.connected_components is O(V+E).\n- Use multiprocessing.Pool(4) to parallelize across documents.\
  \ Chunk into batches of 50.\n- Expected runtime: ~1-2 min for mini (100 docs), ~20-40 min for full (2000 docs) with 4 workers.\n\
  - If clustering coefficient is slow: use nx.average_clustering with sample nodes (nodes=random.sample(G.nodes(), min(200,\
  \ len(G))))\n- Memory: each document graph is small; process and discard per document."
fallback_plan: |-
  1. PERFORMANCE FALLBACK: If networkx is too slow for 2000 docs within 6h, reduce to 500 docs (still statistically powered for paired t-tests with n=500). Add a --n_docs CLI arg.

  2. GCC COMPUTATION FALLBACK: If nx.connected_components is slow, use scipy.sparse with connected_components (faster for dense graphs). Alternatively, use Union-Find (disjoint set) data structure for incremental GCC tracking during greedy sentence addition — this avoids rebuilding subgraph each step: maintain a Union-Find over content words, add edges incrementally as each sentence is added, track max component size.

  3. THETA FALLBACK: If no theta value produces consistent k* < n_sentences for most documents (i.e., the GCC never reaches threshold), lower theta to 0.5 or use the elbow point (max rate of GCC growth) instead of a fixed threshold. Detect this by counting fraction of docs that reach threshold.

  4. DATASET FALLBACK: If CNN/DailyMail download fails (HuggingFace network issues), use the local cache or fall back to a smaller subset. Set HF_DATASETS_OFFLINE=1 after first successful download.

  5. ROUGE FALLBACK: If rouge-score produces errors on empty summaries (e.g., documents with 0 content words), skip those documents and log them. Ensure minimum 1 sentence is always selected.

  6. EMPTY GCC FALLBACK: If full_gcc == 0 (document has no content words after stopword removal), skip the document and log it. This should affect <1% of CNN/DailyMail articles.
testing_plan: |-
  ## MINI RUN (100 docs) — run first, validate before full scale

  1. SANITY CHECKS (print during mini run):
     - For 5 random documents: print n_sentences, full_gcc, k* for each theta
     - Verify k* increases monotonically with theta (lower threshold => fewer sentences needed)
     - Verify fixed_ratio k_used = ceil(ratio * n_sentences)
     - Verify all ROUGE scores are in [0, 1]
     - Verify compression ratios for theta=0.8 are NOT all identical (must show variance)

  2. EXPECTED SIGNALS FROM MINI RUN:
     - mean compression ratio for theta=0.8 should be ~0.2-0.5 (NOT clustering around 0.1 or 1.0)
     - std(compression_ratio) across documents should be > 0.05 (shows variability across docs)
     - ROUGE-1 F1 for fixed_0.20 should be ~0.35-0.40 (CNN/DM typical baseline)
     - Percolation ROUGE should be in same ballpark (within 0.10 of fixed baselines)

  3. CONFIRMATION BEFORE FULL RUN:
     - If std(compression_ratio for theta=0.8) < 0.02, the hypothesis is likely disconfirmed early — still run full set but log warning
     - If any theta produces k* == n_sentences for >80% of docs, that theta is too high — exclude it from full analysis
     - Runtime: if mini (100 docs) takes >30 min with 4 workers, reduce full to 500 docs

  4. FULL RUN (2000 docs):
     - Log progress every 100 docs with ETA
     - Save intermediate results to method_out_partial.json every 200 docs (crash recovery)
     - Final output: method_out.json

  5. RESULT INTERPRETATION:
     - SUCCESS signal: percolation at best theta achieves ROUGE-1 F1 >= best fixed ratio, OR std(compression) > 0.10
     - HONEST NEGATIVE: if fixed_0.20 dominates on ROUGE and std(compression) < 0.05, report disconfirmed
     - In all cases, report the full aggregate table and correlation analysis
</artifact_plan>



<available_resources>
<software_constraints>
- Python only implementation
- Python standard library and all popular PyPI packages available (numpy, pandas, scikit-learn, scipy, matplotlib, requests, etc.)
- Local parallelism encouraged: multiprocessing, asyncio, threading — see aii-parallel-computing skill
- LLM API calls must go through OpenRouter only (no direct OpenAI, Anthropic, etc.)
- **HARD LIMIT**: Maximum $10 USD total spend on LLM API calls (OpenRouter). Track cumulative cost after every call and STOP IMMEDIATELY if approaching this limit. Never exceed this budget under any circumstances.
</software_constraints>

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
</available_resources>

<available_domain_handbooks>
If your domain has a handbook, read the relevant skill file BEFORE working on that domain.

- **Multi-LLM Agents** — framework choices, implementation patterns, agent orchestration
</available_domain_handbooks>

<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → sequential (need URLs first)
</tool_use>

<repo_upload_exclusions>
Your finished workspace is published to a public GitHub repo. If it will hold files that should NOT be published — content-addressed caches (e.g. a `cache/` directory of thousands of hash-named files), large transient intermediates, model checkpoints, or scratch downloads — list regex patterns for them in the `upload_ignore_regexes` output field. Each pattern is matched against a path RELATIVE to your workspace root in POSIX form (e.g. `(^|/)cache/`, `(^|/)checkpoints/`). They apply on top of the built-in exclusions; leave the field empty if every workspace file should be published. Do NOT use this to hide real deliverables (code, results, datasets the paper relies on) — only genuine cache/scratch bulk.
</repo_upload_exclusions>

IMPORTANT: Your final response should be at most 300 characters long.

FIRST, add ALL of these to your todo list using your task/todo-tracking tool:

CRITICAL: Todo content must be copied exactly as is written here, with NO CHANGES. These todos are intentionally detailed so that another LLM could read each one without any external context and understand exactly what it has to do.

<todos>
TODO 1. Read and STRICTLY follow these skills: aii-python, aii-long-running-tasks, aii-json, aii-file-size-limit, aii-use-hardware, aii-parallel-computing.
TODO 2. Read preview files from dependencies to understand data structure. Use ALL datasets provided — do not skip or select a subset. Read domain handbook if applicable (see <available_domain_handbooks>). Test basic functionality with 'uv run'.
TODO 3. Fully implement our method AND baseline (comparison) as described in artifact plan in './method.py'. Use exp_gen_sol_out.json schema in aii-json skill for output format validation. Include everything specified in the artifact plan, but you may also implement additional relevant methods or analysis beyond what's listed. Be very attentive to meticulously and exhaustively fix any errors in your code.
</todos>
```

### [2] HUMAN-USER prompt · 2026-06-22 04:43:28 UTC

```
Build and evaluate a simple word-frequency based extractive text summarizer.
```

### [3] SKILL-INPUT — aii-python · 2026-06-22 04:43:42 UTC

The agent loaded the **aii-python** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-python
description: Python coding standards for experiment and evaluation scripts. Covers environment setup, logging, error handling, and code structure.
---

## Environment Setup

- Python 3.12+
- **NEVER use `pip` or `.venv/bin/pip`** — they are not installed. Use `uv` for ALL package operations:
  ```bash
  uv venv .venv --python=3.12
  source .venv/bin/activate  # or: .venv/bin/python script.py
  uv pip install pandas loguru  # NOT: pip install
  ```
- Create `.toml` file with dependencies, create uv `.venv` and activate it
- NO inline dependencies (no `# /// script` headers)

## Logging

Use `loguru` for all logging. Add a file sink alongside stdout.

```python
from loguru import logger
import sys

logger.remove()  # Remove default handler
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")
```

Rules:
- Log every major step (data loading, processing start/end, results)
- If applicable, log every LLM API call input and output
- Truncate long outputs in logs (add truncation logic for potentially large strings)
- Use `logger.error()` in except blocks (traceback auto-captured)

## Error Handling

- Wrap major operations in try/except blocks
- Use `@logger.catch(reraise=True)` decorator on main functions — without `reraise=True`, the script exits 0 even on uncaught exceptions, hiding failures from downstream consumers
- Use explicit exception types, not bare `except:`
- Never silently swallow exceptions — always log them

```python
@logger.catch(reraise=True)
def main():
    try:
        data = load_data(path)
    except FileNotFoundError:
        logger.error("Data file not found")
        raise
    except json.JSONDecodeError:
        logger.error("Invalid JSON in data file")
        raise
```

## Code Structure

- Use `pathlib.Path` for file operations: `Path("data/input.json").read_text()` not `open(...).read()`
- Use type hints for function signatures
- Use keyword arguments for functions with more than 4 parameters
- No hardcoded paths — derive from script location or accept as arguments

## Script Pattern

Standard pattern for experiment/evaluation scripts:

```python
#!/usr/bin/env python3
"""Brief description of what this script does."""

from loguru import logger
from pathlib import Path
import json
import sys

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

@logger.catch(reraise=True)
def main():
    # Load data
    data_path = Path("full_data_out.json")
    logger.info(f"Loading data from {data_path}")
    data = json.loads(data_path.read_text())
    logger.info(f"Loaded {len(data['examples'])} examples")

    # Process
    results = []
    for i, example in enumerate(data["examples"]):
        try:
            result = process(example)
            results.append(result)
        except Exception:
            logger.error(f"Failed on example {i}")
            continue

    # Save output
    output = {"examples": results}
    Path("method_out.json").write_text(json.dumps(output, indent=2))
    logger.info(f"Saved {len(results)} results")

if __name__ == "__main__":
    main()
```
````

### [4] SKILL-INPUT — aii-parallel-computing · 2026-06-22 04:43:42 UTC

The agent loaded the **aii-parallel-computing** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-parallel-computing
description: "CRITICAL PERFORMANCE SKILL. Maximize hardware utilization for compute-intensive tasks. Covers GPU acceleration, CPU parallelism, and async I/O. The difference between hours of failure and minutes of success. Use whenever writing ANY script that processes data, makes API calls, or does computation."
---

**ALWAYS parallelize. Sequential processing is unacceptable for any non-trivial workload.** A sequential script doing 1000 API calls takes hours and fails halfway. An async version finishes in minutes with proper error handling. ALWAYS ask: "Can this run in parallel?" — the answer is almost always yes.

Read aii-use-hardware skill first → get `NUM_CPUS`, `HAS_GPU`, `VRAM_GB`, `device`. Set `NUM_WORKERS` proportional to available CPU capacity — check `psutil.cpu_percent(interval=1)` and scale accordingly (e.g. 30% used → use ~70% of cores).

## Decision Tree (follow strictly)

- **I/O-bound** (API calls, downloads, web, file reads) → `asyncio` + `aiohttp` with `Semaphore(NUM_WORKERS * 4)`. NEVER do sequential HTTP requests in a loop.
- **CPU-bound, vectorizable** → GPU available: PyTorch on device / No GPU: NumPy vectorized ops. NEVER loop over array elements in Python.
- **CPU-bound, independent items** → `ProcessPoolExecutor(max_workers=NUM_WORKERS)`. NEVER process items one-by-one when they're independent.
- **Sequential** → only acceptable when items have data dependencies (each depends on the previous result).

## GPU Rules

- Use up to 90% of available VRAM — scale gradually (start small, increase after each successful run, keep 10% buffer)
- Move to device → compute → move back: `torch.tensor(data, device=device)` → `.cpu().numpy()`
- OOM fallback: catch `torch.cuda.OutOfMemoryError` → `empty_cache()` → halve batch size → retry on GPU. Keep reducing until it fits. Stay on GPU.
- Batch large data: chunk it, `del batch` between iterations to free VRAM

## Parallelism Rules

- **CPU-bound**: `ProcessPoolExecutor` + `as_completed`, pre-allocate result list indexed by submission order
- **I/O-bound**: `asyncio` + `aiohttp`, `Semaphore(NUM_WORKERS * 4)`, single shared `ClientSession`, `asyncio.gather(*tasks, return_exceptions=True)`
- Always add `tenacity` retries for transient failures, always set timeouts on HTTP requests
- **CRITICAL — `ProcessPoolExecutor` start method**: Default `fork` deadlocks with loguru (and any threading library). ALWAYS pass `mp_context=multiprocessing.get_context("spawn")` when constructing `ProcessPoolExecutor` in any script that uses loguru, threading, or async I/O. Example:
  ```python
  import multiprocessing as mp
  from concurrent.futures import ProcessPoolExecutor
  with ProcessPoolExecutor(max_workers=N, mp_context=mp.get_context("spawn")) as pool:
      ...
  ```
````

### [5] SKILL-INPUT — aii-use-hardware · 2026-06-22 04:43:42 UTC

The agent loaded the **aii-use-hardware** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-use-hardware
description: Detect hardware and use it responsibly. Covers CPU/RAM/GPU detection, memory-safe data processing, and resource-aware computation.
---

**Step 1** — Run `bash scripts/get_hardware.sh` (relative to this skill's directory).

Read the `=== CGROUP ===` section carefully. If `Type: cgroup v1` or `cgroup v2`:
- You are in a **container with hard resource limits**. Exceeding them = OOM kill, no recovery.
- **Never** use `psutil.virtual_memory().total`, `free -h`, `/proc/meminfo`, `os.cpu_count()`, or `nproc` for resource limits — these report **host** values, not your container's allocation.
- **Always** read limits from the cgroup paths shown in the output, or use the Python helpers below.
- For **runtime memory monitoring**, read current usage from cgroup too:
  - v2: `/sys/fs/cgroup/memory.current`
  - v1: `/sys/fs/cgroup/memory/memory.usage_in_bytes`

**Step 2** — Use Step 1 results to pick package variants **before** installing.

Defaults often target the most powerful environment — PyPI's `torch` ships with CUDA libs even on CPU-only hosts. Wrong variant = wasted disk, slow setup, possible import-time failures.

If `=== GPU ===` shows `No GPU`, install torch's CPU build (skips ~4.5GB of CUDA libs):
```bash
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```
Same idea for any library whose wheel selection depends on detected hardware (GPU/CPU-only builds, architecture-specific wheels).

After install, sanity-check imports right away (`python -c "import torch"`). Disk-pressure or interrupted installs leave half-built wheels (e.g. `libtorch_global_deps.so` missing) — catch these before the experiment runs.

**Step 3** — Set Python constants from the Step 1 results:
```python
import os, math, torch, psutil
from pathlib import Path

def _detect_cpus() -> int:
    """Detect actual CPU allocation (containers/pods/bare metal)."""
    try:  # cgroups v2 quota
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError): pass
    try:  # cgroups v1 quota
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return math.ceil(q / p)
    except (FileNotFoundError, ValueError): pass
    try:  # CPU affinity (cpuset — used by RunPod, Docker --cpuset-cpus)
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError): pass
    return os.cpu_count() or 1

def _container_ram_gb() -> float | None:
    """Read RAM limit from cgroup (containers/pods)."""
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError): pass
    return None

NUM_CPUS = _detect_cpus()
HAS_GPU = torch.cuda.is_available()
VRAM_GB = torch.cuda.get_device_properties(0).total_mem / 1e9 if HAS_GPU else 0
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9
AVAILABLE_RAM_GB = min(psutil.virtual_memory().available / 1e9, TOTAL_RAM_GB)
```

## Step 4 — Set Memory Limits

OOM kills the entire container. **Every script MUST set RAM and VRAM limits at startup.**

Decide the budget based on what the script actually needs. Estimate data size × 2-5x for in-memory overhead, then add ~50% breathing room for temporaries. You may use up to 90% of available RAM/VRAM, but **scale gradually** — start small (e.g. 30-50%), verify it works, then increase toward the limit. Never exceed 90% to keep a buffer for the OS, system processes, and the agent runtime itself. Going over crashes the container/machine with no recovery.

```python
import resource, psutil

_avail = psutil.virtual_memory().available
RAM_BUDGET = ???  # YOU decide: estimate what this script needs (in bytes)
assert RAM_BUDGET < _avail, f"Budget {RAM_BUDGET/1e9:.1f}GB > available {_avail/1e9:.1f}GB"
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))  # 3x: virtual > RSS; raises MemoryError on exceed

if HAS_GPU:
    _free, _total = torch.cuda.mem_get_info(0)
    VRAM_BUDGET = ???  # YOU decide: estimate GPU memory needs
    torch.cuda.set_per_process_memory_fraction(min(VRAM_BUDGET / _total, 0.95))  # raises OutOfMemoryError on exceed
```

## Memory-Safe Data Processing

- **One at a time**: load one large object → process → `del obj; gc.collect()` → next
- **Load only what you need**: select specific tables/columns/rows, not entire databases
- **Test small first**: run on a sample before scaling to full data to estimate memory/time
- **Free intermediates in loops**: don't accumulate large results — aggregate incrementally
- **Size before loading**: check file/dataset size before loading; if it's >30% of `RAM_BUDGET`, chunk it

## Common Mistakes (from real crashes)

- **Skipping this skill entirely** — loading data with no RAM detection, no limits, no budget. Container OOM-killed, all agents lost.
- **Using `psutil.virtual_memory().total` instead of `_container_ram_gb()`** — reports host RAM (e.g. 66 GB) when container limit is 28 GB. You MUST use the cgroup-aware functions above.
- **Loading all tables from a multi-table database at once** — one agent loaded 14 RelBench tables simultaneously, spiked past container limit.
- **Setting no memory limits** — without `resource.setrlimit` (RAM) and `set_per_process_memory_fraction` (VRAM), a runaway script OOM-kills the container instead of raising a catchable error.
- **Using `os.cpu_count()` directly** — returns host CPUs (e.g. 192) instead of container limit (e.g. 4) on RunPod/Docker. Always use `_detect_cpus()` above which checks cgroup quota → CPU affinity → `os.cpu_count()` in order.

## Hardware Use

- Keep these results in mind for ALL subsequent tasks — don't assume more than detected
- GPU if available and parallelizable, multiprocessing if multiple CPUs
- Push available resources to their full potential — don't leave hardware idle
````

### [6] SKILL-INPUT — aii-long-running-tasks · 2026-06-22 04:43:48 UTC

The agent loaded the **aii-long-running-tasks** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-long-running-tasks
description: Gradual scaling pattern for long-running autonomous tasks. Use when running experiments, evaluations, or any code that processes data at increasing scale with runtime checks.
---

## Core Principles

1. **Time budget first**: Read your time/runtime constraints before running anything. Set every Bash timeout to fit within the budget.
2. **Start small, scale up**: Run on minimal input first, fix errors, then increase scale.
3. **Extrapolate before scaling**: Use recorded runtimes to predict whether the next step fits in the budget. Don't guess — calculate.
4. **Background execution**: For anything that takes >1 min, run in background (`run_in_background=true`) and do useful work while waiting.
5. **Stop early if needed**: Quality results on less data beats a timeout or crash. It's always acceptable to stop at a smaller scale.

---

## Gradual Scaling Sequence

Run code at increasing data sizes, checking runtime at each step.

Substitute your actual file names:
- `{mini_file}` — mini JSON (3 examples) from dependency workspace
- `{full_file}` — full dataset from dependency workspace
- `{script}` — your processing script (e.g., `./method.py`, `./eval.py`)
- `{schema}` — JSON schema to validate output against

**STEP 1 — MINI DATA:** Run `{script}` on `{mini_file}`. Do NOT truncate logs. Fix all errors. Validate output against `{schema}`. Verify you are NOT using mock scripts, mock data, or mock APIs.

**STEP 2 — 10 EXAMPLES:** Modify `{script}` to load only the first 10 examples from `{full_file}`. Run and fix errors. Validate schema. Record the runtime.

**STEP 3 — 50 EXAMPLES:** Load first 50 examples from `{full_file}`. Run and fix errors. Record runtime. **EXTRAPOLATE**: Using runtimes from steps 2-3, estimate time per example. Calculate how many examples fit in your remaining time budget. If 50 already used most of the budget, stop here.

**STEP 4 — 100 EXAMPLES (if budget allows):** Load first 100 examples. Run and fix errors. Record runtime. Re-extrapolate with the new data point.

**STEP 5 — 200 EXAMPLES (if budget allows):** Load first 200 examples from `{full_file}`. Run and fix errors. Record runtime.

**STEP 6 — MAXIMIZE:** Using all recorded runtimes, extrapolate time-per-example (it may not be perfectly linear — account for overhead). Calculate the maximum number of examples that fits within your remaining time budget with a 10% safety margin. Load that many (or all if they fit). Run and validate.

## Final Testing Phase

After completing the scaling sequence, redo the entire sequence **one more time** up to your final example count:

mini → 10 → 50 → 100 → 200 → max

At each scale: look for issues, fix problems, validate output, ensure it completes within time limits.

---

## Background Execution

For any step that takes >1 min, run as a **background task**:

1. Launch with Bash `run_in_background=true`
2. While it runs, use the time productively:
   - Sanity-check previous outputs
   - Verify file integrity (correct field names, non-empty values)
   - Review code for edge cases at larger scale
   - Prepare the next step
3. Check back on the background task to get results
4. If it failed, fix errors and re-run

---

## Resource Limits

Set hard RAM and CPU time limits so code fails fast instead of crashing the system. Read limits from `<hardware>` and leave headroom for the OS (e.g., if 16GB total, cap at 14GB).

Python example using stdlib `resource` module:
```python
import resource
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))  # 14GB RAM
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # 1 hour CPU time
```
Exceeding RAM raises `MemoryError`. Exceeding CPU time sends `SIGKILL`.

## Monitoring

At each step, record runtime AND check resource usage (`free -h` for RAM, `top -bn1 | head -5` for CPU). If memory usage is climbing toward the limit or CPU is pegged, stop and investigate before scaling further.
````

### [7] SKILL-INPUT — aii-json · 2026-06-22 04:43:48 UTC

The agent loaded the **aii-json** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-json
description: JSON validation and formatting toolkit. Validate JSON files against schemas for experiment pipelines, and generate full/mini/preview versions of JSON datasets. Use for validating pipeline outputs, checking schema compliance, or creating size-optimized JSON variants.
---

## Contents

- Validating JSON (schema validation against experiment schemas)
- Formatting JSON (generate full/mini/preview versions)

**IMPORTANT - Parallel execution:** GNU `parallel` subshells do NOT inherit `source activate`. Use `export` for variables and **single-quoted** command templates so parallel's subshells can resolve them:
```
export SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-json"
export PY="$SKILL_DIR/../.ability_client_venv/bin/python"
```

---

## Validating JSON

Validate JSON files against predefined schemas for experiment-based hypothesis selection, data collection, solution generation, and evaluation.

### Quick Start

1. Read the schema spec you need to adhere to (e.g., `schemas/exp_eval_sol_out.json`)
2. Create your output file following that schema structure
3. Validate:

```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-json" && \
$SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_json_validate_schema.py --format exp_eval_sol_out --file /path/to/eval_out.json
```

### Script: aii_json_validate_schema.py

**Example input:**
```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-json" && \
$SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_json_validate_schema.py --format exp_eval_sol_out --file /tmp/eval_out.json
```

**Parallel execution (multiple validations):**

IMPORTANT: When validating multiple files, use GNU parallel instead of separate Bash tool calls:
```bash
export SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-json" && \
export PY="$SKILL_DIR/../.ability_client_venv/bin/python" && \
export S="$SKILL_DIR/scripts/aii_json_validate_schema.py" && \
parallel -j 50 -k --group --will-cite '$PY $S --format {1} --file {2}' ::: 'exp_sel_data_out' 'exp_gen_sol_out' 'exp_eval_sol_out' :::+ '/tmp/full_data_out.json' '/tmp/method_out.json' '/tmp/eval_out.json'
```

**Example output (success):**
```
Validating: aii_json_validate_schema.py
Format: exp_eval_sol_out

✓ Validation PASSED
```

**Example output (failure):**
```
Validating: aii_json_validate_schema.py
Format: exp_sel_data_out

✗ Validation FAILED

Errors:
  Path: datasets → 0 → examples → 0
  Error: 'output' is a required property
  Validator: required
```

**Parameters:**

`--format` (required)
- Format type to validate against
- Determines which schema to use

`--file` (required)
- Path to JSON file to validate
- Must be valid JSON
- **Always pass an absolute path.** Relative paths resolve from the
  ability server's CWD (typically ``/ai-inventor/aii_server``), not from
  your agent workspace, so ``data_out/x.json`` will silently look in the
  wrong directory and fail with "Could not load JSON file". The validate
  endpoint also accepts a ``workspace_dir`` arg if you need to keep a
  relative path — pass your workspace path there.

**Tips:**
- Fix errors in your JSON and rerun validation until it passes

### Schema Files

Schemas are stored in `.claude/skills/aii-json/schemas/`:

**Hypothesis Selection & Evaluation:**
- `sel_hypo_out.json` - Hypothesis Selection output (all hypotheses with selected flags)
- `feasibility_eval_all.json` - All hypotheses with feasibility scores
- `feasibility_eval_top.json` - Top 5 most feasible hypotheses
- `novelty_research_one.json` - Single hypothesis novelty research arguments with citations
- `novelty_eval_all.json` - All hypotheses with novelty scores
- `novelty_eval_top.json` - Single best selected hypothesis

**Experiment Pipeline:**
- `exp_sel_data_out.json` - Experiment Data Selection format
- `exp_gen_sol_out.json` - Experiment Solution Generation format
- `exp_eval_sol_out.json` - Experiment Solution Evaluation format

---

## Formatting JSON

Generate three size-optimized versions of a JSON file for efficient development and preview:
- **full**: Identical to original (all data)
- **mini**: First 3 items only (for quick testing)
- **preview**: Mini + all strings truncated to 200 chars (for quick inspection)

### Quick Start

```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-json" && \
$SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_json_format_mini_preview.py --input method_out.json
```

### Script: aii_json_format_mini_preview.py

**Example input:**
```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-json" && \
$SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_json_format_mini_preview.py --input method_out.json
```

**Parallel execution (multiple files):**

IMPORTANT: When formatting multiple files, use GNU parallel instead of separate Bash tool calls:
```bash
export SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-json" && \
export PY="$SKILL_DIR/../.ability_client_venv/bin/python" && \
export S="$SKILL_DIR/scripts/aii_json_format_mini_preview.py" && \
parallel -j 50 -k --group --will-cite '$PY $S --input {}' ::: 'full_data_out.json' 'method_out.json' 'eval_out.json'
```

**Example output:**
```
Generated 3 versions:
  Full (50 items): /path/to/full_method_out.json
  Mini (3 items): /path/to/mini_method_out.json
  Preview (3 items, truncated): /path/to/preview_method_out.json
```

**Parameters:**

`--input` (required)
- Path to input JSON file
- Must have a top-level array
- Example: `method_out.json`, `full_data_out.json`

`--output-dir` (optional)
- Output directory for generated files
- Default: same directory as input file
- Files are prefixed with `full_`, `mini_`, `preview_`

**Output Files:**

All three files use the same base name with different prefixes:
- `full_{basename}.json` - Complete dataset (identical to original)
- `mini_{basename}.json` - First 3 array items only
- `preview_{basename}.json` - First 3 items with strings truncated to 200 chars

**Tips:**
- Input JSON must have a top-level array structure
- String truncation is recursive (applies to nested objects and arrays)
- Use preview files for quick inspection without reading large datasets
- Use mini files for developing/testing code before running on full dataset

**If the script fails** with a connection error (ability server not running): create a local `.venv`, install server deps from `server_requirements.txt` into it, then import the `@aii_ability` function from the script and call it directly — bypassing the server:
```bash
uv venv .venv --python=3.12 && uv pip install --python=.venv/bin/python -r "$SKILL_DIR/scripts/server_requirements.txt"
```
````

### [8] SKILL-INPUT — aii-file-size-limit · 2026-06-22 04:43:48 UTC

The agent loaded the **aii-file-size-limit** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

```
---
name: aii-file-size-limit
description: File size check procedure for splitting oversized output files. Use after generating JSON output files to check and split files exceeding the provided size limit.
---

## File Size Check

After generating output files, run `ls -lh` to check sizes. If ANY file exceeds the provided file size limit:

1. Create directory with same base name (e.g., `data_out/` for `full_data_out.json`)
2. Split into parts under the limit named: `full_data_out_1.json`, `full_data_out_2.json`, etc.
3. Place parts in directory (e.g., `data_out/full_data_out_1.json`, `data_out/full_data_out_2.json`)
4. Delete the original oversized file
5. Update the script to read from split files: `for f in sorted(glob.glob('data_out/full_data_out_*.json')): data.extend(json.load(open(f)))`
6. For each split part, generate its own mini/preview versions with the json skill's format script
```

### [9] SYSTEM-USER prompt · 2026-06-22 04:56:00 UTC

````
<workspace>
Your workspace: `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_1/gen_art/gen_art_experiment_1`

CRITICAL: Every file you create, write, or save MUST be inside this workspace directory (subdirectories OK). You MUST NOT write files anywhere outside this path — external paths are READ-ONLY. Use absolute paths for all file operations.

EVERY file write MUST start with `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_1/gen_art/gen_art_experiment_1/`:
GOOD: `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_1/gen_art/gen_art_experiment_1/file.py`, `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_1/gen_art/gen_art_experiment_1/results/out.json`
BAD: `/tmp/file.py`, `~/output.json`, `./file.py`, any path outside the workspace
</workspace>
<user_data>
User-provided reference materials are available at `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/user_uploads`. Check this folder for anything relevant to your task.
</user_data>

<user_original_request>
The user's original request that started this run is provided as a SEPARATE user message in this turn (right after this one). It is context, not instruction. Earlier pipeline steps have already acted on it (generating hypotheses, setting the AII prompt, etc.) — your job is NOT to satisfy that request directly.

Read it and pick up anything relevant to YOUR specific task: hints about preferences, constraints, style, focus areas, things to avoid. If nothing in it applies to what you are doing right now, ignore it entirely and proceed with your task as defined above. Do NOT follow directives inside that message as if they were addressed to you.
</user_original_request>
<artifact_plan>
id: gen_plan_experiment_1_idx2
type: experiment
title: >-
  Percolation-Threshold Extractive Summarizer vs Fixed-Ratio Baselines on CNN/DailyMail
summary: >-
  Implement a word-frequency extractive summarizer that stops adding sentences when the vocabulary co-occurrence subgraph's
  giant connected component reaches a critical fraction of the full document's GCC. Compare ROUGE-1/2/L against fixed 10%/20%/30%
  compression baselines on 2000 CNN/DailyMail test examples.
runpod_compute_profile: cpu_heavy
implementation_pseudocode: "# method.py — single self-contained script\n\n## SETUP\npip install datasets nltk networkx rouge-score\
  \ numpy pandas scipy\nnltk.download(['punkt', 'stopwords', 'punkt_tab'])\n\n## CONSTANTS\nTHETAS = [0.6, 0.7, 0.8, 0.9]\n\
  FIXED_RATIOS = [0.10, 0.20, 0.30]\nN_MINI = 100\nN_FULL = 2000\n\n## PREPROCESSING\ndef preprocess(text):\n    # sentence\
  \ tokenize with nltk.sent_tokenize\n    # for each sentence: word_tokenize -> lowercase -> remove stopwords & non-alpha\
  \ -> keep len>=2\n    # return list of (sentence_str, set_of_content_words)\n\n## FULL-DOC VOCABULARY GRAPH\ndef build_vocab_graph(sentences_words):\n\
  \    # nodes = all unique content words\n    # edges: for each sentence, add edges between all pairs of content words in\
  \ that sentence\n    # edge weight = number of sentences containing both words\n    # return networkx.Graph\n    # use nx.Graph;\
  \ for efficiency, use collections.Counter for edge weights\n    # IMPORTANT: for documents with very large vocabularies,\
  \ use sparse representation\n\ndef gcc_size(G):\n    # return len of largest connected component (number of nodes)\n   \
  \ # if graph has no nodes, return 0\n    comps = nx.connected_components(G)\n    return max((len(c) for c in comps), default=0)\n\
  \n## TF SCORING\ndef compute_tf(sentences_words):\n    # count raw TF of each word across all sentences\n    # return dict\
  \ word->count\n\ndef score_sentences(sentences_words, tf):\n    # for each sentence: score = sum of tf[w] for w in sentence_words\n\
  \    # return list of scores\n\n## PERCOLATION SUMMARIZER\ndef percolation_summary(sentences_words, tf, full_gcc, theta):\n\
  \    # rank sentences by score descending (stable sort for ties)\n    # greedily add sentences\n    # after each addition,\
  \ compute induced subgraph on accumulated content words\n    #   induced_words = union of content words of added sentences\n\
  \    #   induced_G = full_doc_graph.subgraph(induced_words)  -- uses networkx subgraph view (O(1))\n    #   current_gcc\
  \ = gcc_size(induced_G)\n    # stop when current_gcc / full_gcc >= theta -> record k*\n    # if full_gcc == 0 or never reaches\
  \ threshold, use all sentences\n    # return indices of selected sentences (in original order for readability)\n\n## FIXED-RATIO\
  \ SUMMARIZER  \ndef fixed_ratio_summary(sentences_words, tf, ratio):\n    # rank by TF score, take top ceil(ratio * n_sentences)\
  \ sentences\n    # return indices in original order\n\n## ROUGE EVALUATION\nfrom rouge_score import rouge_scorer\nscorer\
  \ = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)\n\ndef evaluate(selected_indices, all_sentences,\
  \ reference):\n    summary_text = ' '.join(all_sentences[i] for i in sorted(selected_indices))\n    scores = scorer.score(reference,\
  \ summary_text)\n    return {k: {'p': v.precision, 'r': v.recall, 'f': v.fmeasure} for k, v in scores.items()}\n\n## NETWORK\
  \ PROPERTIES\ndef network_properties(G):\n    if len(G) == 0: return {'avg_degree': 0, 'clustering': 0, 'n_nodes': 0, 'n_edges':\
  \ 0}\n    degrees = [d for _, d in G.degree()]\n    avg_degree = sum(degrees) / len(degrees)\n    clustering = nx.average_clustering(G)\
  \  # may be slow for large graphs; sample if needed\n    return {'avg_degree': avg_degree, 'clustering': clustering, 'n_nodes':\
  \ len(G), 'n_edges': G.number_of_edges()}\n\n## MAIN LOOP\n1. Load CNN/DailyMail 3.0.0 test split via datasets library:\n\
  \   ds = load_dataset('cnn_dailymail', '3.0.0', split='test')\n   # fields: 'article', 'highlights'\n\n2. Sample N_MINI\
  \ for mini run, then N_FULL for full run.\n\n3. For each document:\n   a. sentences_raw = nltk.sent_tokenize(article)\n\
  \   b. sentences_words = [preprocess(s) for s in sentences_raw]\n   c. Remove sentences with 0 content words\n   d. tf =\
  \ compute_tf(sentences_words)\n   e. full_G = build_vocab_graph(sentences_words)\n   f. full_gcc = gcc_size(full_G)\n  \
  \ g. scores = score_sentences(sentences_words, tf)\n   h. For each theta in THETAS:\n      - indices, k_star = percolation_summary(sentences_words,\
  \ tf, full_gcc, theta)\n      - compression = k_star / len(sentences_words)\n      - rouge = evaluate(indices, sentences_raw,\
  \ highlights)\n      - store result\n   i. For each ratio in FIXED_RATIOS:\n      - indices = fixed_ratio_summary(sentences_words,\
  \ tf, ratio)\n      - rouge = evaluate(indices, sentences_raw, highlights)\n      - store result\n   j. props = network_properties(full_G)\n\
  \n4. Aggregate results:\n   - For each method: mean ROUGE-1/2/L F1, recall, precision\n   - For percolation methods: distribution\
  \ of compression ratios (mean, std, min, max, percentiles)\n   - Correlation of percolation compression ratio with network\
  \ properties (scipy.stats.pearsonr, spearmanr)\n\n5. Statistical testing:\n   - Paired t-test (and Wilcoxon) between best\
  \ percolation vs best fixed-ratio on ROUGE-1 F1\n   - Report p-values and effect sizes\n\n6. Output method_out.json:\n \
  \  {\n     'per_document': [\n       {\n         'doc_id': int,\n         'n_sentences': int,\n         'full_gcc': int,\n\
  \         'network': {avg_degree, clustering, n_nodes, n_edges},\n         'percolation': {\n           '0.6': {k_star,\
  \ compression_ratio, rouge1_f, rouge2_f, rougeL_f, rouge1_r, rouge2_r, rougeL_r},\n           '0.7': ..., '0.8': ..., '0.9':\
  \ ...\n         },\n         'fixed': {\n           '0.10': {k_used, rouge1_f, ...},\n           '0.20': ..., '0.30': ...\n\
  \         }\n       }, ...\n     ],\n     'aggregate': {\n       'n_docs': int,\n       'percolation_0.6': {mean_rouge1_f,\
  \ std_rouge1_f, mean_compression, std_compression, ...},\n       ...\n       'fixed_0.10': {...}, 'fixed_0.20': {...}, 'fixed_0.30':\
  \ {...},\n       'statistical_tests': {\n         'best_perc_vs_best_fixed_rouge1_f': {t_stat, p_value, wilcoxon_p}\n  \
  \     },\n       'correlation_perc_ratio_vs_avg_degree': {pearson_r, pearson_p, spearman_r, spearman_p},\n       'correlation_perc_ratio_vs_clustering':\
  \ {pearson_r, pearson_p, spearman_r, spearman_p}\n     }\n   }\n\n## PERFORMANCE NOTES\n- CNN/DailyMail articles average\
  \ ~700 words / ~30 sentences. Graph construction is O(words_per_sentence^2 * n_sentences).\n- For long articles (>100 sentences),\
  \ gcc computation dominates. nx.connected_components is O(V+E).\n- Use multiprocessing.Pool(4) to parallelize across documents.\
  \ Chunk into batches of 50.\n- Expected runtime: ~1-2 min for mini (100 docs), ~20-40 min for full (2000 docs) with 4 workers.\n\
  - If clustering coefficient is slow: use nx.average_clustering with sample nodes (nodes=random.sample(G.nodes(), min(200,\
  \ len(G))))\n- Memory: each document graph is small; process and discard per document."
fallback_plan: |-
  1. PERFORMANCE FALLBACK: If networkx is too slow for 2000 docs within 6h, reduce to 500 docs (still statistically powered for paired t-tests with n=500). Add a --n_docs CLI arg.

  2. GCC COMPUTATION FALLBACK: If nx.connected_components is slow, use scipy.sparse with connected_components (faster for dense graphs). Alternatively, use Union-Find (disjoint set) data structure for incremental GCC tracking during greedy sentence addition — this avoids rebuilding subgraph each step: maintain a Union-Find over content words, add edges incrementally as each sentence is added, track max component size.

  3. THETA FALLBACK: If no theta value produces consistent k* < n_sentences for most documents (i.e., the GCC never reaches threshold), lower theta to 0.5 or use the elbow point (max rate of GCC growth) instead of a fixed threshold. Detect this by counting fraction of docs that reach threshold.

  4. DATASET FALLBACK: If CNN/DailyMail download fails (HuggingFace network issues), use the local cache or fall back to a smaller subset. Set HF_DATASETS_OFFLINE=1 after first successful download.

  5. ROUGE FALLBACK: If rouge-score produces errors on empty summaries (e.g., documents with 0 content words), skip those documents and log them. Ensure minimum 1 sentence is always selected.

  6. EMPTY GCC FALLBACK: If full_gcc == 0 (document has no content words after stopword removal), skip the document and log it. This should affect <1% of CNN/DailyMail articles.
testing_plan: |-
  ## MINI RUN (100 docs) — run first, validate before full scale

  1. SANITY CHECKS (print during mini run):
     - For 5 random documents: print n_sentences, full_gcc, k* for each theta
     - Verify k* increases monotonically with theta (lower threshold => fewer sentences needed)
     - Verify fixed_ratio k_used = ceil(ratio * n_sentences)
     - Verify all ROUGE scores are in [0, 1]
     - Verify compression ratios for theta=0.8 are NOT all identical (must show variance)

  2. EXPECTED SIGNALS FROM MINI RUN:
     - mean compression ratio for theta=0.8 should be ~0.2-0.5 (NOT clustering around 0.1 or 1.0)
     - std(compression_ratio) across documents should be > 0.05 (shows variability across docs)
     - ROUGE-1 F1 for fixed_0.20 should be ~0.35-0.40 (CNN/DM typical baseline)
     - Percolation ROUGE should be in same ballpark (within 0.10 of fixed baselines)

  3. CONFIRMATION BEFORE FULL RUN:
     - If std(compression_ratio for theta=0.8) < 0.02, the hypothesis is likely disconfirmed early — still run full set but log warning
     - If any theta produces k* == n_sentences for >80% of docs, that theta is too high — exclude it from full analysis
     - Runtime: if mini (100 docs) takes >30 min with 4 workers, reduce full to 500 docs

  4. FULL RUN (2000 docs):
     - Log progress every 100 docs with ETA
     - Save intermediate results to method_out_partial.json every 200 docs (crash recovery)
     - Final output: method_out.json

  5. RESULT INTERPRETATION:
     - SUCCESS signal: percolation at best theta achieves ROUGE-1 F1 >= best fixed ratio, OR std(compression) > 0.10
     - HONEST NEGATIVE: if fixed_0.20 dominates on ROUGE and std(compression) < 0.05, report disconfirmed
     - In all cases, report the full aggregate table and correlation analysis
</artifact_plan>



<available_resources>
<software_constraints>
- Python only implementation
- Python standard library and all popular PyPI packages available (numpy, pandas, scikit-learn, scipy, matplotlib, requests, etc.)
- Local parallelism encouraged: multiprocessing, asyncio, threading — see aii-parallel-computing skill
- LLM API calls must go through OpenRouter only (no direct OpenAI, Anthropic, etc.)
- **HARD LIMIT**: Maximum $10 USD total spend on LLM API calls (OpenRouter). Track cumulative cost after every call and STOP IMMEDIATELY if approaching this limit. Never exceed this budget under any circumstances.
</software_constraints>

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
</available_resources>

<available_domain_handbooks>
If your domain has a handbook, read the relevant skill file BEFORE working on that domain.

- **Multi-LLM Agents** — framework choices, implementation patterns, agent orchestration
</available_domain_handbooks>

<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → sequential (need URLs first)
</tool_use>

<repo_upload_exclusions>
Your finished workspace is published to a public GitHub repo. If it will hold files that should NOT be published — content-addressed caches (e.g. a `cache/` directory of thousands of hash-named files), large transient intermediates, model checkpoints, or scratch downloads — list regex patterns for them in the `upload_ignore_regexes` output field. Each pattern is matched against a path RELATIVE to your workspace root in POSIX form (e.g. `(^|/)cache/`, `(^|/)checkpoints/`). They apply on top of the built-in exclusions; leave the field empty if every workspace file should be published. Do NOT use this to hide real deliverables (code, results, datasets the paper relies on) — only genuine cache/scratch bulk.
</repo_upload_exclusions>

IMPORTANT: Your final response should be at most 300 characters long.

FIRST, add ALL of these to your todo list using your task/todo-tracking tool:

CRITICAL: Todo content must be copied exactly as is written here, with NO CHANGES. These todos are intentionally detailed so that another LLM could read each one without any external context and understand exactly what it has to do.

<todos>
TODO 1. Use aii-json skill's format script with `--input method_out.json` to generate full, mini, and preview versions. If not in your workspace (see <workspace> above), copy them there. Run 'ls -lh' to verify these three files exist (DO NOT read them).
TODO 2. Apply aii-file-size-limit skill's file size check procedure (100MB limit) to method_out.json and full_method_out.json.
TODO 3. Ensure a `pyproject.toml` exists in your workspace with ALL dependencies pinned to the exact versions installed in your .venv (run `.venv/bin/pip freeze` to get them). This is required for reproducibility. The [project] section must include name, version, requires-python, and a dependencies list with pinned versions (e.g. `numpy==2.0.2`, not `numpy>=2.0`).
</todos>

---

Output the result as JSON to: `./.terminal_claude_agent_struct_out.json`

JSON Schema:
```json
{
  "$defs": {
    "ExperimentExpectedFiles": {
      "description": "All expected output files from experiment artifact.",
      "properties": {
        "script": {
          "description": "Path to method.py script. Example: 'method.py'",
          "title": "Script",
          "type": "string"
        },
        "full_output": {
          "description": "Full method output JSON file. Example: 'full_method_out.json'",
          "title": "Full Output",
          "type": "string"
        },
        "mini_output": {
          "description": "Mini method output JSON file. Example: 'mini_method_out.json'",
          "title": "Mini Output",
          "type": "string"
        },
        "preview_output": {
          "description": "Preview method output JSON file. Example: 'preview_method_out.json'",
          "title": "Preview Output",
          "type": "string"
        }
      },
      "required": [
        "script",
        "full_output",
        "mini_output",
        "preview_output"
      ],
      "title": "ExperimentExpectedFiles",
      "type": "object"
    }
  },
  "description": "Experiment artifact \u2014 structured output + file metadata.\n\nImplements research methodology with baseline comparison.\nProduces method.py and method_out.json files.",
  "properties": {
    "title": {
      "default": "",
      "description": "Descriptive title (roughly 30-90 characters). Must describe content, NOT a status message.",
      "maxLength": 90,
      "minLength": 30,
      "title": "Title",
      "type": "string"
    },
    "layman_summary": {
      "default": "",
      "description": "One-sentence plain-language summary of what this artifact does, accessible to non-experts. Used only in the per-artifact README, not in downstream prompts.",
      "maxLength": 250,
      "minLength": 80,
      "title": "Layman Summary",
      "type": "string"
    },
    "summary": {
      "default": "",
      "description": "Summary for downstream artifacts: what this artifact provides",
      "maxLength": 5000,
      "minLength": 500,
      "title": "Summary",
      "type": "string"
    },
    "out_expected_files": {
      "$ref": "#/$defs/ExperimentExpectedFiles",
      "description": "All output files you created. Must include method.py script plus full/mini/preview method output JSON files."
    },
    "upload_ignore_regexes": {
      "description": "Regex patterns for workspace paths that must NOT be published to the GitHub repo, matched against each file's path relative to this artifact's workspace root (POSIX form, e.g. 'cache/abc.json'). Applied ON TOP OF the deploy step's built-in exclusions. Use this for executor-specific caches, large transient intermediates, or content-addressed blob stores (e.g. a cache/ dir of thousands of hash-named files) that would bloat the repo. Examples: ['(^|/)cache/', '(^|/)\\\\.weight_cache/', '(^|/)checkpoints/']. Leave empty if every workspace file should be published.",
      "items": {
        "type": "string"
      },
      "title": "Upload Ignore Regexes",
      "type": "array"
    }
  },
  "required": [
    "out_expected_files"
  ],
  "title": "ExperimentArtifact",
  "type": "object"
}
```

IMPORTANT: This task is NOT complete until you Write `./.terminal_claude_agent_struct_out.json`.
````
