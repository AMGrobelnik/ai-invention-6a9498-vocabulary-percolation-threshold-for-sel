# gen_art_evaluation_1 — test_idea

> Phase: `invention_loop` · round 1 · `gen_art`
> Run: `run_CFTHGHhY9evP` — Vocabulary Percolation Threshold for Self-Calibrating Extractive Summary Length: Recall Advantage and F1 Failure Mode Across Corpora
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_art_evaluation_1` (terminal_claude_agent)

### [1] SYSTEM-USER prompt · 2026-06-22 04:43:23 UTC

````
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

<task>
Evaluate experimental results using domain-appropriate methods, metrics, and analysis techniques.
When in doubt, prefer more metrics over fewer — but only ones that make sense for the domain.
</task>

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
Your workspace: `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_1/gen_art/gen_art_evaluation_1`

CRITICAL: Every file you create, write, or save MUST be inside this workspace directory (subdirectories OK). You MUST NOT write files anywhere outside this path — external paths are READ-ONLY. Use absolute paths for all file operations.

EVERY file write MUST start with `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_1/gen_art/gen_art_evaluation_1/`:
GOOD: `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_1/gen_art/gen_art_evaluation_1/file.py`, `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_1/gen_art/gen_art_evaluation_1/results/out.json`
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
id: gen_plan_evaluation_1_idx3
type: evaluation
title: 'Statistical Evaluation: Percolation Threshold vs Fixed-Ratio ROUGE Comparison'
summary: >-
  Load method_out.json from the experiment, run paired Wilcoxon signed-rank tests with Holm-Bonferroni correction on ROUGE-1/2/L
  recall for each (percolation theta, fixed-ratio baseline) pair, test whether percolation compression ratio std > 10pp, regress
  k* fraction on network features, segment by CNN vs DailyMail, and output eval_out.json.
runpod_compute_profile: cpu_heavy
metrics_descriptions: |-
  ## Metrics

  ### 1. ROUGE Scores (Primary)
  - **ROUGE-1 recall, precision, F1** — unigram overlap between generated and reference summary
  - **ROUGE-2 recall, precision, F1** — bigram overlap
  - **ROUGE-L recall, precision, F1** — longest common subsequence
  - Computed per document using the `rouge-score` Python package (pip install rouge-score)
  - Primary comparison: recall scores (hypothesis targets conceptual coverage)
  - Computed for: percolation summaries at each theta (e.g., 0.6, 0.7, 0.8, 0.9) vs fixed-ratio baselines (10%, 20%, 30% of sentences)

  ### 2. Paired Wilcoxon Signed-Rank Tests
  - Non-parametric paired test: for each (percolation_theta, fixed_ratio) pair, test whether ROUGE-1/2/L recall differs significantly
  - Use `scipy.stats.wilcoxon` with `alternative='greater'` (percolation > fixed) and `alternative='two-sided'` for honest reporting
  - Apply Holm-Bonferroni correction across all pairwise comparisons using `statsmodels.stats.multitest.multipletests(method='holm')`
  - Report: test statistic, raw p-value, corrected p-value, reject/not-reject at alpha=0.05
  - Total comparisons: 4 thetas × 3 fixed ratios × 3 ROUGE metrics = 36 tests (correct all together)

  ### 3. Compression Ratio Distribution
  - For each document: compute percolation k*/N (fraction of sentences selected)
  - Report: mean, std, min, max, 25th/50th/75th percentiles across all documents
  - **Key test**: is std > 0.10 (10 percentage points)? Use one-sample t-test or direct std comparison
  - Also report by corpus segment (CNN vs DailyMail separately)

  ### 4. Network Feature Regression
  - Features from method_out.json per document: avg_degree, clustering_coefficient, graph_density, num_nodes (vocabulary size), num_sentences
  - Target: percolation_compression_ratio (k*/N) at the best-performing theta
  - Fit OLS linear regression via `sklearn.linear_model.LinearRegression` with standardized features
  - Report: R², coefficients, feature importances (standardized betas)
  - Also fit per-theta to see which threshold produces most predictable compression
  - Secondary: Pearson/Spearman correlation of each feature vs k*/N

  ### 5. Segment Analysis (CNN vs DailyMail)
  - Split documents by source field in method_out.json
  - Compute all ROUGE stats and compression ratio stats separately
  - Report mean ROUGE-1 recall ± std for each method × segment
  - Test whether percolation advantage is consistent across both corpora

  ### 6. Summary Table
  - Table: rows = methods (percolation@0.6/0.7/0.8/0.9, fixed@10%/20%/30%), columns = ROUGE-1/2/L recall mean ± std
  - Best cell highlighted (annotated in JSON)
  - p-values from Wilcoxon tests vs best fixed-ratio baseline per ROUGE metric

  ## Output Format (eval_out.json)
  ```json
  {
    "n_documents": int,
    "rouge_summary_table": [
      {"method": str, "rouge1_recall_mean": float, "rouge1_recall_std": float, ...}
    ],
    "wilcoxon_tests": [
      {"method_a": str, "method_b": str, "metric": str, "statistic": float, "p_raw": float, "p_corrected": float, "significant": bool}
    ],
    "compression_ratio_stats": {
      "overall": {"mean": float, "std": float, "min": float, "max": float, "p25": float, "p50": float, "p75": float},
      "cnn": {...},
      "dailymail": {...},
      "std_exceeds_10pp": bool
    },
    "regression": {
      "r2": float,
      "features": ["avg_degree", "clustering_coefficient", "graph_density"],
      "coefficients": [float, ...],
      "correlations": {"pearson": {...}, "spearman": {...}}
    },
    "verdict": {
      "hypothesis_supported": bool,
      "best_percolation_theta": float,
      "best_fixed_ratio": float,
      "rouge1_recall_improvement": float,
      "rouge1_significant": bool,
      "compression_ratio_variable": bool,
      "notes": str
    }
  }
  ```
metrics_justification: |-
  ## Justification

  **Why Wilcoxon signed-rank?** ROUGE scores across documents are non-normal and paired (same documents evaluated by all methods). Wilcoxon is the correct non-parametric paired test. Holm-Bonferroni is preferred over Bonferroni (less conservative, controls FWER) and over BH FDR (FWER is more appropriate when we want to avoid claiming false positives for specific pairs).

  **Why ROUGE recall as primary?** The hypothesis claims percolation achieves better *conceptual coverage* — recall directly measures how much of the reference summary's content is captured. Precision would penalize the percolation method for producing longer summaries when theta is low. F1 is reported for completeness.

  **Why std > 10pp as threshold?** The hypothesis explicitly states this as the success criterion for demonstrating that fixed compression ratios are suboptimal. This is a simple, interpretable threshold that can be verified directly from the data.

  **Why network feature regression?** The hypothesis claims percolation length is *predictable* from structural properties (avg degree, clustering coefficient). R² from OLS tells us how much variance in compression ratio is explained by document structure. High R² would validate the mechanistic claim; low R² would suggest the threshold is noisy rather than structurally determined.

  **Why segment by CNN vs DailyMail?** These sub-corpora have systematically different document lengths and styles (CNN shorter/harder news vs DailyMail longer/softer). If percolation improves one but not the other, this gives insight into when the method is most useful and tests assumption #3 (variation across document types).

  **Implementation notes for executor:**
  - Install: `pip install rouge-score scipy statsmodels scikit-learn pandas numpy`
  - Load method_out.json (expected: list of dicts with fields: document_id, source, percolation_summaries={theta: {sentences, rouge1, rouge2, rougeL}}, fixed_summaries={ratio: {...}}, network_features={avg_degree, clustering_coefficient, graph_density, ...})
  - If method_out.json has a different schema, adapt field access accordingly — print first record to inspect
  - Handle edge cases: documents where percolation never reaches threshold (k*=N, all sentences), very short docs with < 3 sentences
  - Use `rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)` for consistency
  - For Wilcoxon: skip documents where method produced identical summaries (zero difference), use `zero_method='wilcox'` default
  - Cap sample at 5000 documents if method_out.json is very large (random seed 42)
  - The verdict dict should directly address the hypothesis success/disconfirmation criteria
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

- **Multi-LLM Agents** — evaluation metrics, agent orchestration patterns, benchmark design
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
TODO 2. Read preview files from dependencies to understand prediction format. Evaluate ALL experiments provided — do not skip or select a subset. Avoid re-training or re-executing the method unless absolutely necessary; prefer loading predictions from each dependency's method_out.json / predict_* fields. Read domain handbook if applicable (see <available_domain_handbooks>). Decide evaluation metrics based on artifact plan. Test basic functionality with 'uv run'.
TODO 3. Fully implement evaluation as described in artifact plan in './eval.py'. Use exp_eval_sol_out.json schema in aii-json skill for output format validation. Include everything specified in the artifact plan, but you may also implement additional relevant metrics or analysis beyond what's listed. Be very attentive to meticulously and exhaustively fix any errors in your code.
</todos>
````

### [2] HUMAN-USER prompt · 2026-06-22 04:43:23 UTC

```
Build and evaluate a simple word-frequency based extractive text summarizer.
```

### [3] SKILL-INPUT — aii-python · 2026-06-22 04:43:45 UTC

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

### [4] SKILL-INPUT — aii-json · 2026-06-22 04:43:49 UTC

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

### [5] SYSTEM-USER prompt · 2026-06-22 04:57:46 UTC

````
<workspace>
Your workspace: `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_1/gen_art/gen_art_evaluation_1`

CRITICAL: Every file you create, write, or save MUST be inside this workspace directory (subdirectories OK). You MUST NOT write files anywhere outside this path — external paths are READ-ONLY. Use absolute paths for all file operations.

EVERY file write MUST start with `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_1/gen_art/gen_art_evaluation_1/`:
GOOD: `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_1/gen_art/gen_art_evaluation_1/file.py`, `/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop/iter_1/gen_art/gen_art_evaluation_1/results/out.json`
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
id: gen_plan_evaluation_1_idx3
type: evaluation
title: 'Statistical Evaluation: Percolation Threshold vs Fixed-Ratio ROUGE Comparison'
summary: >-
  Load method_out.json from the experiment, run paired Wilcoxon signed-rank tests with Holm-Bonferroni correction on ROUGE-1/2/L
  recall for each (percolation theta, fixed-ratio baseline) pair, test whether percolation compression ratio std > 10pp, regress
  k* fraction on network features, segment by CNN vs DailyMail, and output eval_out.json.
runpod_compute_profile: cpu_heavy
metrics_descriptions: |-
  ## Metrics

  ### 1. ROUGE Scores (Primary)
  - **ROUGE-1 recall, precision, F1** — unigram overlap between generated and reference summary
  - **ROUGE-2 recall, precision, F1** — bigram overlap
  - **ROUGE-L recall, precision, F1** — longest common subsequence
  - Computed per document using the `rouge-score` Python package (pip install rouge-score)
  - Primary comparison: recall scores (hypothesis targets conceptual coverage)
  - Computed for: percolation summaries at each theta (e.g., 0.6, 0.7, 0.8, 0.9) vs fixed-ratio baselines (10%, 20%, 30% of sentences)

  ### 2. Paired Wilcoxon Signed-Rank Tests
  - Non-parametric paired test: for each (percolation_theta, fixed_ratio) pair, test whether ROUGE-1/2/L recall differs significantly
  - Use `scipy.stats.wilcoxon` with `alternative='greater'` (percolation > fixed) and `alternative='two-sided'` for honest reporting
  - Apply Holm-Bonferroni correction across all pairwise comparisons using `statsmodels.stats.multitest.multipletests(method='holm')`
  - Report: test statistic, raw p-value, corrected p-value, reject/not-reject at alpha=0.05
  - Total comparisons: 4 thetas × 3 fixed ratios × 3 ROUGE metrics = 36 tests (correct all together)

  ### 3. Compression Ratio Distribution
  - For each document: compute percolation k*/N (fraction of sentences selected)
  - Report: mean, std, min, max, 25th/50th/75th percentiles across all documents
  - **Key test**: is std > 0.10 (10 percentage points)? Use one-sample t-test or direct std comparison
  - Also report by corpus segment (CNN vs DailyMail separately)

  ### 4. Network Feature Regression
  - Features from method_out.json per document: avg_degree, clustering_coefficient, graph_density, num_nodes (vocabulary size), num_sentences
  - Target: percolation_compression_ratio (k*/N) at the best-performing theta
  - Fit OLS linear regression via `sklearn.linear_model.LinearRegression` with standardized features
  - Report: R², coefficients, feature importances (standardized betas)
  - Also fit per-theta to see which threshold produces most predictable compression
  - Secondary: Pearson/Spearman correlation of each feature vs k*/N

  ### 5. Segment Analysis (CNN vs DailyMail)
  - Split documents by source field in method_out.json
  - Compute all ROUGE stats and compression ratio stats separately
  - Report mean ROUGE-1 recall ± std for each method × segment
  - Test whether percolation advantage is consistent across both corpora

  ### 6. Summary Table
  - Table: rows = methods (percolation@0.6/0.7/0.8/0.9, fixed@10%/20%/30%), columns = ROUGE-1/2/L recall mean ± std
  - Best cell highlighted (annotated in JSON)
  - p-values from Wilcoxon tests vs best fixed-ratio baseline per ROUGE metric

  ## Output Format (eval_out.json)
  ```json
  {
    "n_documents": int,
    "rouge_summary_table": [
      {"method": str, "rouge1_recall_mean": float, "rouge1_recall_std": float, ...}
    ],
    "wilcoxon_tests": [
      {"method_a": str, "method_b": str, "metric": str, "statistic": float, "p_raw": float, "p_corrected": float, "significant": bool}
    ],
    "compression_ratio_stats": {
      "overall": {"mean": float, "std": float, "min": float, "max": float, "p25": float, "p50": float, "p75": float},
      "cnn": {...},
      "dailymail": {...},
      "std_exceeds_10pp": bool
    },
    "regression": {
      "r2": float,
      "features": ["avg_degree", "clustering_coefficient", "graph_density"],
      "coefficients": [float, ...],
      "correlations": {"pearson": {...}, "spearman": {...}}
    },
    "verdict": {
      "hypothesis_supported": bool,
      "best_percolation_theta": float,
      "best_fixed_ratio": float,
      "rouge1_recall_improvement": float,
      "rouge1_significant": bool,
      "compression_ratio_variable": bool,
      "notes": str
    }
  }
  ```
metrics_justification: |-
  ## Justification

  **Why Wilcoxon signed-rank?** ROUGE scores across documents are non-normal and paired (same documents evaluated by all methods). Wilcoxon is the correct non-parametric paired test. Holm-Bonferroni is preferred over Bonferroni (less conservative, controls FWER) and over BH FDR (FWER is more appropriate when we want to avoid claiming false positives for specific pairs).

  **Why ROUGE recall as primary?** The hypothesis claims percolation achieves better *conceptual coverage* — recall directly measures how much of the reference summary's content is captured. Precision would penalize the percolation method for producing longer summaries when theta is low. F1 is reported for completeness.

  **Why std > 10pp as threshold?** The hypothesis explicitly states this as the success criterion for demonstrating that fixed compression ratios are suboptimal. This is a simple, interpretable threshold that can be verified directly from the data.

  **Why network feature regression?** The hypothesis claims percolation length is *predictable* from structural properties (avg degree, clustering coefficient). R² from OLS tells us how much variance in compression ratio is explained by document structure. High R² would validate the mechanistic claim; low R² would suggest the threshold is noisy rather than structurally determined.

  **Why segment by CNN vs DailyMail?** These sub-corpora have systematically different document lengths and styles (CNN shorter/harder news vs DailyMail longer/softer). If percolation improves one but not the other, this gives insight into when the method is most useful and tests assumption #3 (variation across document types).

  **Implementation notes for executor:**
  - Install: `pip install rouge-score scipy statsmodels scikit-learn pandas numpy`
  - Load method_out.json (expected: list of dicts with fields: document_id, source, percolation_summaries={theta: {sentences, rouge1, rouge2, rougeL}}, fixed_summaries={ratio: {...}}, network_features={avg_degree, clustering_coefficient, graph_density, ...})
  - If method_out.json has a different schema, adapt field access accordingly — print first record to inspect
  - Handle edge cases: documents where percolation never reaches threshold (k*=N, all sentences), very short docs with < 3 sentences
  - Use `rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)` for consistency
  - For Wilcoxon: skip documents where method produced identical summaries (zero difference), use `zero_method='wilcox'` default
  - Cap sample at 5000 documents if method_out.json is very large (random seed 42)
  - The verdict dict should directly address the hypothesis success/disconfirmation criteria
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

- **Multi-LLM Agents** — evaluation metrics, agent orchestration patterns, benchmark design
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
TODO 1. Use aii-json skill's format script with `--input eval_out.json` to generate full, mini, and preview versions. If not in your workspace (see <workspace> above), copy them there. Run 'ls -lh' to verify these three files exist (DO NOT read them).
TODO 2. Apply aii-file-size-limit skill's file size check procedure (100MB limit) to eval_out.json and full_eval_out.json.
TODO 3. Ensure a `pyproject.toml` exists in your workspace with ALL dependencies pinned to the exact versions installed in your .venv (run `.venv/bin/pip freeze` to get them). This is required for reproducibility. The [project] section must include name, version, requires-python, and a dependencies list with pinned versions (e.g. `numpy==2.0.2`, not `numpy>=2.0`).
</todos>

---

Output the result as JSON to: `./.terminal_claude_agent_struct_out.json`

JSON Schema:
```json
{
  "$defs": {
    "EvaluationExpectedFiles": {
      "description": "All expected output files from evaluation artifact.",
      "properties": {
        "script": {
          "description": "Path to eval.py script. Example: 'eval.py'",
          "title": "Script",
          "type": "string"
        },
        "full_output": {
          "description": "Full evaluation JSON file. Example: 'full_eval_out.json'",
          "title": "Full Output",
          "type": "string"
        },
        "mini_output": {
          "description": "Mini evaluation JSON file. Example: 'mini_eval_out.json'",
          "title": "Mini Output",
          "type": "string"
        },
        "preview_output": {
          "description": "Preview evaluation JSON file. Example: 'preview_eval_out.json'",
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
      "title": "EvaluationExpectedFiles",
      "type": "object"
    }
  },
  "description": "Evaluation artifact \u2014 structured output + file metadata.\n\nEvaluates both proposed and baseline methods with appropriate metrics.\nProduces eval.py and eval_out.json files.",
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
      "$ref": "#/$defs/EvaluationExpectedFiles",
      "description": "All output files you created. Must include eval.py script plus full/mini/preview evaluation JSON files."
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
  "title": "EvaluationArtifact",
  "type": "object"
}
```

IMPORTANT: This task is NOT complete until you Write `./.terminal_claude_agent_struct_out.json`.
````

### [6] SYSTEM-USER prompt · 2026-06-22 04:58:46 UTC

```
<validation-feedback>
Attempt 1 failed validation.

Schema validation found 1 problem — fix ALL of them at once:
  - at `layman_summary`: 'This evaluation tests whether a graph-based extractive summarizer that stops adding sentences once the word-network reaches a connectivity threshold outperforms simple fixed-length baselines on 2000 CNN/DailyMail news articles, using ROUGE recall scores and statistical significance tests.' is too long (at most 250 characters, got 289)
Every required field must be present and every field type must match the schema.

Please use the Write tool to overwrite `.terminal_claude_agent_struct_out.json` with corrected JSON. Do not invent new fields; match the schema you were given.
</validation-feedback>
```
