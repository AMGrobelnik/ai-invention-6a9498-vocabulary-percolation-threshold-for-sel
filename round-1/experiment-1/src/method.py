#!/usr/bin/env python3
"""Percolation-Threshold Extractive Summarizer vs Fixed-Ratio Baselines on CNN/DailyMail."""

import json
import math
import multiprocessing as mp
import os
import resource
import sys
import gc
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import stats

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
Path("logs").mkdir(exist_ok=True)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ─── Hardware & Memory ────────────────────────────────────────────────────────
def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except Exception:
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return math.ceil(q / p)
    except Exception:
        pass
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1

NUM_CPUS = _detect_cpus()
RAM_GB = 29.0
RAM_BUDGET = int(RAM_GB * 0.75 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
NUM_WORKERS = max(1, NUM_CPUS)

# ─── Constants ────────────────────────────────────────────────────────────────
THETAS = [0.6, 0.7, 0.8, 0.9]
FIXED_RATIOS = [0.10, 0.20, 0.30]
N_MINI = 10
N_FULL = 2000

# ─── NLTK setup (done inside worker after spawn) ─────────────────────────────
def ensure_nltk():
    import nltk
    nltk_data_dir = "/tmp/nltk_data"
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.insert(0, nltk_data_dir)
    for pkg in ["punkt", "punkt_tab", "stopwords"]:
        try:
            nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
        except Exception:
            pass

# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess(text: str) -> tuple[list[str], list[set[str]]]:
    """Return (raw_sentences, content_word_sets)."""
    import nltk
    from nltk.corpus import stopwords
    stop = set(stopwords.words("english"))
    raw_sents = nltk.sent_tokenize(text)
    word_sets = []
    for s in raw_sents:
        words = set(
            w.lower() for w in nltk.word_tokenize(s)
            if w.isalpha() and len(w) >= 2 and w.lower() not in stop
        )
        word_sets.append(words)
    return raw_sents, word_sets


# ─── Vocabulary Graph ─────────────────────────────────────────────────────────
def build_vocab_graph(sentences_words: list[set[str]]):
    import networkx as nx
    G = nx.Graph()
    edge_counter: Counter = Counter()
    for words in sentences_words:
        wlist = sorted(words)
        for i in range(len(wlist)):
            for j in range(i + 1, len(wlist)):
                edge_counter[(wlist[i], wlist[j])] += 1
    G.add_nodes_from({w for ws in sentences_words for w in ws})
    for (u, v), wt in edge_counter.items():
        G.add_edge(u, v, weight=wt)
    return G


def gcc_size(G) -> int:
    import networkx as nx
    if len(G) == 0:
        return 0
    return max((len(c) for c in nx.connected_components(G)), default=0)


# ─── TF Scoring ──────────────────────────────────────────────────────────────
def compute_tf(sentences_words: list[set[str]]) -> dict[str, int]:
    tf: Counter = Counter()
    for words in sentences_words:
        tf.update(words)
    return dict(tf)


def score_sentences(sentences_words: list[set[str]], tf: dict[str, int]) -> list[float]:
    return [sum(tf.get(w, 0) for w in ws) for ws in sentences_words]


# ─── Percolation Summarizer ───────────────────────────────────────────────────
def percolation_summary(
    sentences_words: list[set[str]],
    tf: dict[str, int],
    full_G,
    full_gcc: int,
    theta: float,
) -> tuple[list[int], int]:
    import networkx as nx
    n = len(sentences_words)
    if n == 0:
        return [], 0
    scores = score_sentences(sentences_words, tf)
    ranked = sorted(range(n), key=lambda i: -scores[i])
    if full_gcc == 0:
        return list(range(n)), n
    accumulated_words: set[str] = set()
    selected = []
    k_star = n
    for idx in ranked:
        selected.append(idx)
        accumulated_words |= sentences_words[idx]
        sub = full_G.subgraph(accumulated_words)
        cur_gcc = max((len(c) for c in nx.connected_components(sub)), default=0)
        if cur_gcc / full_gcc >= theta:
            k_star = len(selected)
            break
    return sorted(selected), k_star


# ─── Fixed-Ratio Summarizer ───────────────────────────────────────────────────
def fixed_ratio_summary(
    sentences_words: list[set[str]], tf: dict[str, int], ratio: float
) -> tuple[list[int], int]:
    n = len(sentences_words)
    k = max(1, math.ceil(ratio * n))
    scores = score_sentences(sentences_words, tf)
    ranked = sorted(range(n), key=lambda i: -scores[i])
    selected = sorted(ranked[:k])
    return selected, k


# ─── ROUGE Evaluation ────────────────────────────────────────────────────────
def evaluate(selected_indices: list[int], all_sentences: list[str], reference: str) -> dict:
    from rouge_score import rouge_scorer as rs_module
    scorer = rs_module.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    if not selected_indices:
        return {
            "rouge1_f": 0.0, "rouge1_r": 0.0, "rouge1_p": 0.0,
            "rouge2_f": 0.0, "rouge2_r": 0.0, "rouge2_p": 0.0,
            "rougeL_f": 0.0, "rougeL_r": 0.0, "rougeL_p": 0.0,
        }
    summary_text = " ".join(all_sentences[i] for i in sorted(selected_indices))
    scores = scorer.score(reference, summary_text)
    out = {}
    for k, v in scores.items():
        out[f"{k}_f"] = v.fmeasure
        out[f"{k}_r"] = v.recall
        out[f"{k}_p"] = v.precision
    return out


# ─── Network Properties ───────────────────────────────────────────────────────
def network_properties(G) -> dict:
    import networkx as nx
    import random
    if len(G) == 0:
        return {"avg_degree": 0.0, "clustering": 0.0, "n_nodes": 0, "n_edges": 0}
    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees)
    sample_nodes = random.sample(list(G.nodes()), min(200, len(G)))
    clustering = nx.average_clustering(G, nodes=sample_nodes)
    return {
        "avg_degree": round(avg_degree, 4),
        "clustering": round(clustering, 4),
        "n_nodes": len(G),
        "n_edges": G.number_of_edges(),
    }


# ─── Per-Document Worker ─────────────────────────────────────────────────────
def process_doc(args) -> dict | None:
    doc_id, article, highlights = args
    try:
        ensure_nltk()
        import networkx as nx

        raw_sents, word_sets = preprocess(article)
        # Filter empty sentences
        paired = [(s, ws) for s, ws in zip(raw_sents, word_sets) if ws]
        if not paired:
            return None
        raw_sents_filt = [p[0] for p in paired]
        word_sets_filt = [p[1] for p in paired]
        n = len(word_sets_filt)

        tf = compute_tf(word_sets_filt)
        full_G = build_vocab_graph(word_sets_filt)
        full_gcc = gcc_size(full_G)
        props = network_properties(full_G)

        result = {
            "doc_id": doc_id,
            "n_sentences": n,
            "full_gcc": full_gcc,
            "network": props,
            "percolation": {},
            "fixed": {},
        }

        for theta in THETAS:
            indices, k_star = percolation_summary(word_sets_filt, tf, full_G, full_gcc, theta)
            rouge = evaluate(indices, raw_sents_filt, highlights)
            compression = k_star / n if n > 0 else 1.0
            result["percolation"][str(theta)] = {
                "k_star": k_star,
                "compression_ratio": round(compression, 4),
                **rouge,
            }

        for ratio in FIXED_RATIOS:
            indices, k_used = fixed_ratio_summary(word_sets_filt, tf, ratio)
            rouge = evaluate(indices, raw_sents_filt, highlights)
            result["fixed"][str(ratio)] = {"k_used": k_used, **rouge}

        return result
    except Exception as e:
        logger.error(f"Doc {doc_id} failed: {e}")
        return None


# ─── Aggregation ─────────────────────────────────────────────────────────────
def aggregate(per_doc: list[dict]) -> dict:
    def col(method_key, field):
        vals = []
        for d in per_doc:
            entry = d.get(method_key, {})
            if isinstance(entry, dict) and field in entry:
                vals.append(entry[field])
        return np.array(vals, dtype=float)

    agg = {"n_docs": len(per_doc)}

    for theta in THETAS:
        k = f"percolation_{theta}"
        bucket = {}
        for metric in ["rouge1_f", "rouge2_f", "rougeL_f", "rouge1_r", "rouge2_r", "rougeL_r",
                        "rouge1_p", "rouge2_p", "rougeL_p"]:
            v = col(f"percolation.{theta}", metric)
            if len(v) == 0:
                v = np.array([d["percolation"].get(str(theta), {}).get(metric, np.nan) for d in per_doc])
            if len(v) == 0:
                continue
            # direct extraction
        # Use direct extraction
        for theta in THETAS:
            bucket = {}
            for metric in ["rouge1_f", "rouge2_f", "rougeL_f", "rouge1_r", "rouge2_r", "rougeL_r",
                            "compression_ratio"]:
                v = np.array([d["percolation"][str(theta)][metric] for d in per_doc
                               if str(theta) in d.get("percolation", {})])
                bucket[f"mean_{metric}"] = round(float(np.nanmean(v)), 4)
                bucket[f"std_{metric}"] = round(float(np.nanstd(v)), 4)
                if metric == "compression_ratio":
                    bucket["min_compression"] = round(float(np.nanmin(v)), 4)
                    bucket["max_compression"] = round(float(np.nanmax(v)), 4)
                    bucket["p25_compression"] = round(float(np.nanpercentile(v, 25)), 4)
                    bucket["p75_compression"] = round(float(np.nanpercentile(v, 75)), 4)
            agg[f"percolation_{theta}"] = bucket

    for ratio in FIXED_RATIOS:
        bucket = {}
        for metric in ["rouge1_f", "rouge2_f", "rougeL_f", "rouge1_r", "rouge2_r", "rougeL_r"]:
            v = np.array([d["fixed"][str(ratio)][metric] for d in per_doc
                           if str(ratio) in d.get("fixed", {})])
            bucket[f"mean_{metric}"] = round(float(np.nanmean(v)), 4)
            bucket[f"std_{metric}"] = round(float(np.nanstd(v)), 4)
        agg[f"fixed_{ratio}"] = bucket

    # Best percolation vs best fixed on rouge1_f
    best_theta = max(THETAS, key=lambda t: agg.get(f"percolation_{t}", {}).get("mean_rouge1_f", 0))
    best_ratio = max(FIXED_RATIOS, key=lambda r: agg.get(f"fixed_{r}", {}).get("mean_rouge1_f", 0))
    perc_r1 = np.array([d["percolation"][str(best_theta)]["rouge1_f"] for d in per_doc
                         if str(best_theta) in d.get("percolation", {})])
    fixed_r1 = np.array([d["fixed"][str(best_ratio)]["rouge1_f"] for d in per_doc
                          if str(best_ratio) in d.get("fixed", {})])
    n_paired = min(len(perc_r1), len(fixed_r1))
    if n_paired > 1:
        t_stat, p_val = stats.ttest_rel(perc_r1[:n_paired], fixed_r1[:n_paired])
        wilc_stat, wilc_p = stats.wilcoxon(perc_r1[:n_paired] - fixed_r1[:n_paired])
        agg["statistical_tests"] = {
            "best_perc_theta": best_theta,
            "best_fixed_ratio": best_ratio,
            "best_perc_vs_best_fixed_rouge1_f": {
                "t_stat": round(float(t_stat), 4),
                "p_value": round(float(p_val), 6),
                "wilcoxon_p": round(float(wilc_p), 6),
                "mean_diff": round(float(np.mean(perc_r1[:n_paired] - fixed_r1[:n_paired])), 4),
            }
        }

    # Correlations: percolation compression_ratio vs network props
    comp_08 = np.array([d["percolation"]["0.8"]["compression_ratio"] for d in per_doc
                         if "0.8" in d.get("percolation", {})])
    avg_deg = np.array([d["network"]["avg_degree"] for d in per_doc
                         if "0.8" in d.get("percolation", {})])
    clust = np.array([d["network"]["clustering"] for d in per_doc
                       if "0.8" in d.get("percolation", {})])

    def corr_dict(x, y):
        if len(x) < 3:
            return {}
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        return {"pearson_r": round(float(pr), 4), "pearson_p": round(float(pp), 6),
                "spearman_r": round(float(sr), 4), "spearman_p": round(float(sp), 6)}

    agg["correlation_perc_ratio_vs_avg_degree"] = corr_dict(comp_08, avg_deg)
    agg["correlation_perc_ratio_vs_clustering"] = corr_dict(comp_08, clust)

    return agg


# ─── Main ─────────────────────────────────────────────────────────────────────
@logger.catch(reraise=True)
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_docs", type=int, default=N_FULL)
    parser.add_argument("--mini", action="store_true")
    args_cli = parser.parse_args()

    n_docs = N_MINI if args_cli.mini else args_cli.n_docs

    logger.info(f"NUM_CPUS={NUM_CPUS}, NUM_WORKERS={NUM_WORKERS}, n_docs={n_docs}")

    ensure_nltk()

    logger.info("Loading CNN/DailyMail dataset...")
    from datasets import load_dataset
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
    total = len(ds)
    n_docs = min(n_docs, total)
    logger.info(f"Dataset loaded: {total} test examples. Using {n_docs}.")

    tasks = [(i, ds[i]["article"], ds[i]["highlights"]) for i in range(n_docs)]

    per_doc_results = []
    partial_path = Path("method_out_partial.json")
    batch_size = 50
    batch_num = 0

    logger.info(f"Processing {n_docs} docs with {NUM_WORKERS} workers, batch_size={batch_size}")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, mp_context=mp.get_context("spawn")) as pool:
        for batch_start in range(0, len(tasks), batch_size):
            batch = tasks[batch_start: batch_start + batch_size]
            futures = {pool.submit(process_doc, t): t[0] for t in batch}
            batch_results = []
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    batch_results.append(res)
            batch_results.sort(key=lambda x: x["doc_id"])
            per_doc_results.extend(batch_results)
            batch_num += 1
            done = batch_start + len(batch)
            logger.info(f"Progress: {done}/{n_docs} docs processed ({len(per_doc_results)} valid)")
            # Save partial every 200 docs
            if done % 200 < batch_size:
                partial_path.write_text(json.dumps({"per_document": per_doc_results}, indent=2))
                logger.info(f"Partial save: {len(per_doc_results)} docs")

    logger.info(f"Aggregating {len(per_doc_results)} results...")
    agg = aggregate(per_doc_results)

    # Sanity checks
    for theta in THETAS:
        k = f"percolation_{theta}"
        if k in agg:
            logger.info(f"theta={theta}: mean_rouge1_f={agg[k]['mean_rouge1_f']:.4f}, "
                        f"mean_compression={agg[k]['mean_compression_ratio']:.4f}±{agg[k]['std_compression_ratio']:.4f}")
    for ratio in FIXED_RATIOS:
        k = f"fixed_{ratio}"
        if k in agg:
            logger.info(f"fixed ratio={ratio}: mean_rouge1_f={agg[k]['mean_rouge1_f']:.4f}")

    full_out = {"per_document": per_doc_results, "aggregate": agg}
    out_path = Path("method_out_raw.json")
    out_path.write_text(json.dumps(full_out, indent=2))
    logger.info(f"Saved raw output to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # ─── Convert to exp_gen_sol_out schema ────────────────────────────────────
    examples = []
    for doc in per_doc_results:
        doc_id = doc["doc_id"]
        article = ds[doc_id]["article"]
        highlights = ds[doc_id]["highlights"]
        # Build predict fields: one per method
        predict_fields = {}
        for theta in THETAS:
            key = str(theta)
            field_key = f"predict_percolation_{str(theta).replace('.', '_')}"
            if key in doc["percolation"]:
                r1 = doc["percolation"][key]["rouge1_f"]
                comp = doc["percolation"][key]["compression_ratio"]
                k_star = doc["percolation"][key]["k_star"]
                predict_fields[field_key] = (
                    f"rouge1_f={r1:.4f}|rouge2_f={doc['percolation'][key]['rouge2_f']:.4f}|"
                    f"rougeL_f={doc['percolation'][key]['rougeL_f']:.4f}|"
                    f"compression={comp:.4f}|k_star={k_star}"
                )
        for ratio in FIXED_RATIOS:
            key = str(ratio)
            field_key = f"predict_fixed_{str(ratio).replace('.', '_')}"
            if key in doc["fixed"]:
                r1 = doc["fixed"][key]["rouge1_f"]
                predict_fields[field_key] = (
                    f"rouge1_f={r1:.4f}|rouge2_f={doc['fixed'][key]['rouge2_f']:.4f}|"
                    f"rougeL_f={doc['fixed'][key]['rougeL_f']:.4f}|"
                    f"k_used={doc['fixed'][key]['k_used']}"
                )
        example = {
            "input": article[:2000],  # truncate for schema storage
            "output": highlights,
            **predict_fields,
            "metadata_doc_id": doc_id,
            "metadata_n_sentences": doc["n_sentences"],
            "metadata_full_gcc": doc["full_gcc"],
            "metadata_avg_degree": doc["network"]["avg_degree"],
            "metadata_clustering": doc["network"]["clustering"],
        }
        examples.append(example)

    schema_out = {
        "metadata": {
            "method_name": "PercolationThresholdExtractSummarizer",
            "description": "Word-frequency extractive summarizer using percolation threshold on vocabulary co-occurrence graph",
            "thetas": THETAS,
            "fixed_ratios": FIXED_RATIOS,
            "n_docs": len(per_doc_results),
            "aggregate": agg,
        },
        "datasets": [
            {
                "dataset": "cnn_dailymail_3.0.0_test",
                "examples": examples,
            }
        ],
    }

    method_out_path = Path("method_out.json")
    method_out_path.write_text(json.dumps(schema_out, indent=2))
    logger.info(f"Saved method_out.json ({method_out_path.stat().st_size / 1e6:.1f} MB)")

    # Clean up partial
    if partial_path.exists():
        partial_path.unlink()

    logger.info("Done!")


if __name__ == "__main__":
    main()
