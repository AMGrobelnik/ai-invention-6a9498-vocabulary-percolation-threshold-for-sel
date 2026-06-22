#!/usr/bin/env python3
"""
Comprehensive evaluation: CNN/DM + Multi-News percolation vs fixed-ratio baselines.
Metrics: ROUGE P/R/F, ceiling fraction, compression std, Wilcoxon+Holm, OLS regression,
calibration, cross-corpus comparison, TF-IDF scorer.
"""

import gc
import json
import math
import multiprocessing as mp
import os
import resource
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import stats

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
Path("logs").mkdir(exist_ok=True)
logger.add("logs/eval.log", rotation="30 MB", level="DEBUG")

# ── Hardware ──────────────────────────────────────────────────────────────────
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
RAM_BUDGET = int(RAM_GB * 0.70 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
NUM_WORKERS = max(1, NUM_CPUS)

THETAS = [0.6, 0.7, 0.8, 0.9]
FIXED_RATIOS = [0.10, 0.20, 0.30]
MULTIDOC_N = 1000  # Multi-News docs to process
CNNDM_RAW = Path(
    "/ai-inventor/aii_data/runs/run_CFTHGHhY9evP/3_invention_loop"
    "/iter_1/gen_art/gen_art_experiment_1/method_out_raw.json"
)

# ── NLTK setup ────────────────────────────────────────────────────────────────
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

# ── Text processing ───────────────────────────────────────────────────────────
def preprocess(text: str) -> tuple[list[str], list[set[str]]]:
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

def compute_tf(sentences_words: list[set[str]]) -> dict[str, float]:
    tf: Counter = Counter()
    for words in sentences_words:
        tf.update(words)
    return dict(tf)

def compute_tfidf(sentences_words: list[set[str]]) -> dict[str, float]:
    n = len(sentences_words)
    tf: Counter = Counter()
    df: Counter = Counter()
    for words in sentences_words:
        tf.update(words)
        for w in words:
            df[w] += 1
    tfidf = {}
    for w, f in tf.items():
        idf = math.log((1 + n) / (1 + df[w])) + 1.0
        tfidf[w] = f * idf
    return tfidf

def score_sentences(sentences_words: list[set[str]], tf: dict[str, float]) -> list[float]:
    return [sum(tf.get(w, 0.0) for w in ws) for ws in sentences_words]

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

def percolation_summary(sentences_words, scorer, full_G, full_gcc, theta):
    import networkx as nx
    n = len(sentences_words)
    if n == 0:
        return [], n
    scores = score_sentences(sentences_words, scorer)
    ranked = sorted(range(n), key=lambda i: -scores[i])
    if full_gcc == 0:
        return sorted(range(n)), n
    accumulated: set[str] = set()
    selected = []
    k_star = n
    for idx in ranked:
        selected.append(idx)
        accumulated |= sentences_words[idx]
        sub = full_G.subgraph(accumulated)
        cur = max((len(c) for c in nx.connected_components(sub)), default=0)
        if cur / full_gcc >= theta:
            k_star = len(selected)
            break
    return sorted(selected), k_star

def fixed_ratio_summary(sentences_words, scorer, ratio):
    n = len(sentences_words)
    k = max(1, math.ceil(ratio * n))
    scores = score_sentences(sentences_words, scorer)
    ranked = sorted(range(n), key=lambda i: -scores[i])
    return sorted(ranked[:k]), k

def rouge_score_doc(selected_indices, all_sentences, reference):
    from rouge_score import rouge_scorer as rs_module
    scorer = rs_module.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    if not selected_indices:
        return {k: 0.0 for k in [
            "rouge1_f","rouge1_r","rouge1_p","rouge2_f","rouge2_r","rouge2_p",
            "rougeL_f","rougeL_r","rougeL_p"]}
    summary = " ".join(all_sentences[i] for i in sorted(selected_indices))
    scores = scorer.score(reference, summary)
    out = {}
    for k, v in scores.items():
        out[f"{k}_f"] = v.fmeasure
        out[f"{k}_r"] = v.recall
        out[f"{k}_p"] = v.precision
    return out

def network_props(G) -> dict:
    import networkx as nx, random
    if len(G) == 0:
        return {"avg_degree": 0.0, "clustering": 0.0, "n_nodes": 0, "n_edges": 0}
    degs = [d for _, d in G.degree()]
    avg_deg = sum(degs) / len(degs)
    sample = random.sample(list(G.nodes()), min(200, len(G)))
    clust = nx.average_clustering(G, nodes=sample)
    return {"avg_degree": round(avg_deg, 4), "clustering": round(clust, 4),
            "n_nodes": len(G), "n_edges": G.number_of_edges()}

# ── Worker for Multi-News ─────────────────────────────────────────────────────
def process_multidoc(args) -> dict | None:
    doc_id, article_text, summary = args
    try:
        ensure_nltk()
        raw_sents, word_sets = preprocess(article_text)
        paired = [(s, ws) for s, ws in zip(raw_sents, word_sets) if ws]
        if not paired:
            return None
        raw_sents_f = [p[0] for p in paired]
        word_sets_f = [p[1] for p in paired]
        n = len(word_sets_f)

        tf = compute_tf(word_sets_f)
        tfidf = compute_tfidf(word_sets_f)
        full_G = build_vocab_graph(word_sets_f)
        full_gcc = gcc_size(full_G)
        props = network_props(full_G)

        result = {
            "doc_id": doc_id,
            "n_sentences": n,
            "full_gcc": full_gcc,
            "network": props,
            "percolation_tf": {},
            "percolation_tfidf": {},
            "fixed_tf": {},
            "fixed_tfidf": {},
        }

        for theta in THETAS:
            for scorer_name, scorer in [("tf", tf), ("tfidf", tfidf)]:
                indices, k_star = percolation_summary(word_sets_f, scorer, full_G, full_gcc, theta)
                rouge = rouge_score_doc(indices, raw_sents_f, summary)
                compression = k_star / n if n > 0 else 1.0
                result[f"percolation_{scorer_name}"][str(theta)] = {
                    "k_star": k_star,
                    "compression_ratio": round(compression, 4),
                    **rouge,
                }

        for ratio in FIXED_RATIOS:
            for scorer_name, scorer in [("tf", tf), ("tfidf", tfidf)]:
                indices, k_used = fixed_ratio_summary(word_sets_f, scorer, ratio)
                rouge = rouge_score_doc(indices, raw_sents_f, summary)
                result[f"fixed_{scorer_name}"][str(ratio)] = {"k_used": k_used, **rouge}

        return result
    except Exception as e:
        logger.error(f"Doc {doc_id} failed: {e}")
        return None

# ── Statistical utilities ─────────────────────────────────────────────────────
def holm_bonferroni(p_values: list[float]) -> list[float]:
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted[orig_idx] = min(1.0, p * (n - rank))
    # Make monotone
    sorted_adj = [adjusted[i] for i, _ in sorted(enumerate(p_values), key=lambda x: x[1])]
    running_max = 0.0
    for rank, (orig_idx, _) in enumerate(sorted(enumerate(p_values), key=lambda x: x[1])):
        running_max = max(running_max, sorted_adj[rank])
        adjusted[orig_idx] = running_max
    return adjusted

def wilcoxon_safe(x, y):
    diff = np.array(x) - np.array(y)
    nonzero = diff[diff != 0]
    if len(nonzero) < 10:
        return float("nan"), float("nan")
    stat, p = stats.wilcoxon(diff)
    return float(stat), float(p)

def collect_arrays(per_doc: list[dict], method_key: str, sub_key: str, field: str) -> np.ndarray:
    vals = []
    for d in per_doc:
        entry = d.get(method_key, {}).get(sub_key, {})
        if field in entry:
            vals.append(entry[field])
    return np.array(vals, dtype=float)

# ── Aggregate CNN/DM (from raw) ────────────────────────────────────────────────
def aggregate_cnndm(per_doc: list[dict]) -> dict:
    agg = {"n_docs": len(per_doc)}
    metrics = ["rouge1_f","rouge1_r","rouge1_p","rouge2_f","rouge2_r","rouge2_p",
               "rougeL_f","rougeL_r","rougeL_p","compression_ratio"]

    for theta in THETAS:
        bucket = {}
        key = str(theta)
        for m in metrics:
            v = collect_arrays(per_doc, "percolation", key, m)
            if len(v) == 0:
                continue
            bucket[f"mean_{m}"] = round(float(np.mean(v)), 4)
            bucket[f"std_{m}"] = round(float(np.std(v)), 4)
            if m == "compression_ratio":
                bucket["min_compression"] = round(float(np.min(v)), 4)
                bucket["max_compression"] = round(float(np.max(v)), 4)
                bucket["p25_compression"] = round(float(np.percentile(v, 25)), 4)
                bucket["p75_compression"] = round(float(np.percentile(v, 75)), 4)
                # CEILING FRACTION: fraction where compression == 1.0
                bucket["ceiling_fraction"] = round(float(np.mean(v >= 0.9999)), 4)
                # STD > 0.10 test
                bucket["std_exceeds_10pp"] = bool(np.std(v) > 0.10)
        agg[f"percolation_{theta}"] = bucket

    for ratio in FIXED_RATIOS:
        bucket = {}
        key = str(ratio)
        for m in ["rouge1_f","rouge1_r","rouge1_p","rouge2_f","rouge2_r","rouge2_p",
                  "rougeL_f","rougeL_r","rougeL_p"]:
            v = collect_arrays(per_doc, "fixed", key, m)
            if len(v) == 0:
                continue
            bucket[f"mean_{m}"] = round(float(np.mean(v)), 4)
            bucket[f"std_{m}"] = round(float(np.std(v)), 4)
        agg[f"fixed_{ratio}"] = bucket

    return agg

# ── OLS regression analysis ───────────────────────────────────────────────────
def regression_analysis(per_doc: list[dict]) -> dict:
    import statsmodels.api as sm

    results = {}
    for theta in THETAS:
        key = str(theta)
        comp = []
        avg_deg = []
        clust = []
        n_sents = []
        full_gcc_vals = []
        for d in per_doc:
            if key not in d.get("percolation", {}):
                continue
            comp.append(d["percolation"][key]["compression_ratio"])
            avg_deg.append(d["network"]["avg_degree"])
            clust.append(d["network"]["clustering"])
            n_sents.append(d["n_sentences"])
            full_gcc_vals.append(d["full_gcc"])

        if len(comp) < 10:
            continue

        y = np.array(comp)
        X = np.column_stack([avg_deg, clust, n_sents, full_gcc_vals])
        # Normalize features
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std == 0] = 1
        X_norm = (X - X_mean) / X_std
        X_with_const = sm.add_constant(X_norm)

        try:
            model = sm.OLS(y, X_with_const).fit()
            params = model.params
            conf = model.conf_int()
            pvals = model.pvalues
            r2 = model.rsquared
            feature_names = ["intercept", "avg_degree", "clustering_coefficient",
                             "n_sentences", "full_gcc_size"]
            coefs = {}
            for i, name in enumerate(feature_names):
                coefs[name] = {
                    "coef": round(float(params[i]), 6),
                    "ci_lower": round(float(conf[0][i]), 6),
                    "ci_upper": round(float(conf[1][i]), 6),
                    "p_value": round(float(pvals[i]), 6),
                }
            # Partial R² for each predictor (via exclusion)
            partial_r2 = {}
            for feat_idx, fname in enumerate(["avg_degree", "clustering_coefficient",
                                               "n_sentences", "full_gcc_size"]):
                cols_excl = [j for j in range(4) if j != feat_idx]
                X_excl = np.column_stack([X_norm[:, j] for j in cols_excl])
                X_excl_c = sm.add_constant(X_excl)
                model_excl = sm.OLS(y, X_excl_c).fit()
                part_r2 = max(0.0, r2 - model_excl.rsquared)
                partial_r2[fname] = round(float(part_r2), 6)

            results[f"theta_{key}"] = {
                "r2": round(float(r2), 4),
                "adj_r2": round(float(model.rsquared_adj), 4),
                "n": len(comp),
                "coefficients": coefs,
                "partial_r2": partial_r2,
            }
        except Exception as e:
            logger.warning(f"Regression failed for theta={theta}: {e}")

    return results

# ── Wilcoxon + Holm tests ─────────────────────────────────────────────────────
def statistical_tests(per_doc: list[dict], corpus: str = "cnndm") -> dict:
    # All pairwise: percolation_theta vs fixed_ratio for ROUGE-1 F1
    comparisons = []
    labels = []
    for theta in THETAS:
        for ratio in FIXED_RATIOS:
            perc_vals = collect_arrays(per_doc, "percolation", str(theta), "rouge1_f")
            fixed_vals = collect_arrays(per_doc, "fixed", str(ratio), "rouge1_f")
            n = min(len(perc_vals), len(fixed_vals))
            if n < 10:
                continue
            stat, p = wilcoxon_safe(perc_vals[:n], fixed_vals[:n])
            comparisons.append({
                "percolation_theta": theta,
                "fixed_ratio": ratio,
                "wilcoxon_stat": round(stat, 4) if not math.isnan(stat) else None,
                "p_raw": round(p, 8) if not math.isnan(p) else None,
                "mean_diff": round(float(np.mean(perc_vals[:n] - fixed_vals[:n])), 4),
                "n": n,
            })
            labels.append((theta, ratio))

    # Holm-Bonferroni correction
    p_vals = [c["p_raw"] if c["p_raw"] is not None else 1.0 for c in comparisons]
    if p_vals:
        adj_ps = holm_bonferroni(p_vals)
        for c, adj_p in zip(comparisons, adj_ps):
            c["p_holm"] = round(adj_p, 8)
            c["significant_holm_05"] = bool(adj_p < 0.05)

    return {"corpus": corpus, "comparisons": comparisons}

# ── Calibration analysis ──────────────────────────────────────────────────────
def calibration_analysis(per_doc: list[dict]) -> dict:
    targets = [0.10, 0.20, 0.30]
    result = {}
    for theta in THETAS:
        key = str(theta)
        v = collect_arrays(per_doc, "percolation", key, "compression_ratio")
        if len(v) == 0:
            continue
        mean_cr = float(np.mean(v))
        std_cr = float(np.std(v))
        # Which target is closest?
        closest = min(targets, key=lambda t: abs(mean_cr - t))
        result[f"theta_{key}"] = {
            "mean_compression_ratio": round(mean_cr, 4),
            "std_compression_ratio": round(std_cr, 4),
            "within_doc_std": round(std_cr, 4),
            "closest_target": closest,
            "abs_deviation_from_closest": round(abs(mean_cr - closest), 4),
        }
    # Also evaluate fixed ratios
    for ratio in FIXED_RATIOS:
        key = str(ratio)
        # Fixed ratio always gives ratio * n / n ≈ ratio (small noise from ceil)
        result[f"fixed_{key}"] = {
            "nominal_compression": ratio,
            "comment": "fixed ratio provides near-constant compression by design",
        }
    return result

# ── Aggregate Multi-News ──────────────────────────────────────────────────────
def aggregate_multidoc(per_doc: list[dict]) -> dict:
    agg = {"n_docs": len(per_doc)}
    metrics = ["rouge1_f","rouge1_r","rouge1_p","rouge2_f","rouge2_r","rouge2_p",
               "rougeL_f","rougeL_r","rougeL_p","compression_ratio"]

    for scorer_name in ["tf", "tfidf"]:
        for theta in THETAS:
            bucket = {}
            method_key = f"percolation_{scorer_name}"
            key = str(theta)
            for m in metrics:
                v = collect_arrays(per_doc, method_key, key, m)
                if len(v) == 0:
                    continue
                bucket[f"mean_{m}"] = round(float(np.mean(v)), 4)
                bucket[f"std_{m}"] = round(float(np.std(v)), 4)
                if m == "compression_ratio":
                    bucket["ceiling_fraction"] = round(float(np.mean(v >= 0.9999)), 4)
                    bucket["std_exceeds_10pp"] = bool(np.std(v) > 0.10)
                    bucket["min_compression"] = round(float(np.min(v)), 4)
                    bucket["max_compression"] = round(float(np.max(v)), 4)
            agg[f"percolation_{scorer_name}_{theta}"] = bucket

        for ratio in FIXED_RATIOS:
            bucket = {}
            method_key = f"fixed_{scorer_name}"
            key = str(ratio)
            for m in ["rouge1_f","rouge1_r","rouge1_p","rouge2_f","rouge2_r","rouge2_p",
                      "rougeL_f","rougeL_r","rougeL_p"]:
                v = collect_arrays(per_doc, method_key, key, m)
                if len(v) == 0:
                    continue
                bucket[f"mean_{m}"] = round(float(np.mean(v)), 4)
                bucket[f"std_{m}"] = round(float(np.std(v)), 4)
            agg[f"fixed_{scorer_name}_{ratio}"] = bucket

    return agg

# ── Cross-corpus comparison ────────────────────────────────────────────────────
def cross_corpus_comparison(cnndm_per_doc: list[dict], multidoc_per_doc: list[dict]) -> dict:
    result = {}
    for theta in THETAS:
        key = str(theta)
        cnndm_cr = collect_arrays(cnndm_per_doc, "percolation", key, "compression_ratio")
        mn_cr = collect_arrays(multidoc_per_doc, "percolation_tf", key, "compression_ratio")
        if len(cnndm_cr) == 0 or len(mn_cr) == 0:
            continue
        cnndm_std = float(np.std(cnndm_cr))
        mn_std = float(np.std(mn_cr))
        result[f"theta_{key}"] = {
            "cnndm_compression_std": round(cnndm_std, 4),
            "multinews_compression_std": round(mn_std, 4),
            "cnndm_compression_mean": round(float(np.mean(cnndm_cr)), 4),
            "multinews_compression_mean": round(float(np.mean(mn_cr)), 4),
            "multinews_std_exceeds_10pp": bool(mn_std > 0.10),
            "cnndm_std_exceeds_10pp": bool(cnndm_std > 0.10),
            "multinews_higher_variance": bool(mn_std > cnndm_std),
        }
        # Test whether std difference is significant via Levene
        lev_stat, lev_p = stats.levene(cnndm_cr, mn_cr)
        result[f"theta_{key}"]["levene_stat"] = round(float(lev_stat), 4)
        result[f"theta_{key}"]["levene_p"] = round(float(lev_p), 6)
    return result

# ── TF vs TF-IDF comparison on Multi-News ─────────────────────────────────────
def tfidf_vs_tf_comparison(multidoc_per_doc: list[dict]) -> dict:
    result = {}
    for theta in THETAS:
        key = str(theta)
        tf_r1 = collect_arrays(multidoc_per_doc, "percolation_tf", key, "rouge1_f")
        tfidf_r1 = collect_arrays(multidoc_per_doc, "percolation_tfidf", key, "rouge1_f")
        n = min(len(tf_r1), len(tfidf_r1))
        if n < 10:
            continue
        stat, p = wilcoxon_safe(tf_r1[:n], tfidf_r1[:n])
        result[f"theta_{key}"] = {
            "tf_mean_rouge1_f": round(float(np.mean(tf_r1[:n])), 4),
            "tfidf_mean_rouge1_f": round(float(np.mean(tfidf_r1[:n])), 4),
            "mean_diff_tf_minus_tfidf": round(float(np.mean(tf_r1[:n] - tfidf_r1[:n])), 4),
            "wilcoxon_stat": round(stat, 4) if not math.isnan(stat) else None,
            "wilcoxon_p": round(p, 6) if not math.isnan(p) else None,
        }
        # Also compression ratio comparison
        tf_cr = collect_arrays(multidoc_per_doc, "percolation_tf", key, "compression_ratio")
        tfidf_cr = collect_arrays(multidoc_per_doc, "percolation_tfidf", key, "compression_ratio")
        n2 = min(len(tf_cr), len(tfidf_cr))
        if n2 > 0:
            result[f"theta_{key}"]["tf_compression_std"] = round(float(np.std(tf_cr[:n2])), 4)
            result[f"theta_{key}"]["tfidf_compression_std"] = round(float(np.std(tfidf_cr[:n2])), 4)

    for ratio in FIXED_RATIOS:
        key = str(ratio)
        tf_r1 = collect_arrays(multidoc_per_doc, "fixed_tf", key, "rouge1_f")
        tfidf_r1 = collect_arrays(multidoc_per_doc, "fixed_tfidf", key, "rouge1_f")
        n = min(len(tf_r1), len(tfidf_r1))
        if n < 10:
            continue
        stat, p = wilcoxon_safe(tf_r1[:n], tfidf_r1[:n])
        result[f"fixed_{key}"] = {
            "tf_mean_rouge1_f": round(float(np.mean(tf_r1[:n])), 4),
            "tfidf_mean_rouge1_f": round(float(np.mean(tfidf_r1[:n])), 4),
            "mean_diff_tf_minus_tfidf": round(float(np.mean(tf_r1[:n] - tfidf_r1[:n])), 4),
            "wilcoxon_stat": round(stat, 4) if not math.isnan(stat) else None,
            "wilcoxon_p": round(p, 6) if not math.isnan(p) else None,
        }
    return result

# ── Build eval_out schema ─────────────────────────────────────────────────────
def build_eval_out(cnndm_per_doc, cnndm_agg, multidoc_per_doc, multidoc_agg,
                   reg_result, stat_tests_cnndm, stat_tests_mn,
                   calib_cnndm, cross_corpus, tfidf_vs_tf) -> dict:
    # Build flat metrics_agg
    metrics_agg = {}

    # CNN/DM key metrics
    for theta in THETAS:
        k = f"percolation_{theta}"
        b = cnndm_agg.get(k, {})
        safe = str(theta).replace(".", "_")
        metrics_agg[f"cnndm_perc_{safe}_rouge1_f"] = b.get("mean_rouge1_f", 0.0)
        metrics_agg[f"cnndm_perc_{safe}_rouge1_r"] = b.get("mean_rouge1_r", 0.0)
        metrics_agg[f"cnndm_perc_{safe}_rouge1_p"] = b.get("mean_rouge1_p", 0.0)
        metrics_agg[f"cnndm_perc_{safe}_compression_mean"] = b.get("mean_compression_ratio", 0.0)
        metrics_agg[f"cnndm_perc_{safe}_compression_std"] = b.get("std_compression_ratio", 0.0)
        metrics_agg[f"cnndm_perc_{safe}_ceiling_fraction"] = b.get("ceiling_fraction", 0.0)

    for ratio in FIXED_RATIOS:
        b = cnndm_agg.get(f"fixed_{ratio}", {})
        safe = str(ratio).replace(".", "_")
        metrics_agg[f"cnndm_fixed_{safe}_rouge1_f"] = b.get("mean_rouge1_f", 0.0)
        metrics_agg[f"cnndm_fixed_{safe}_rouge1_r"] = b.get("mean_rouge1_r", 0.0)
        metrics_agg[f"cnndm_fixed_{safe}_rouge1_p"] = b.get("mean_rouge1_p", 0.0)

    # Multi-News key metrics
    for theta in THETAS:
        safe = str(theta).replace(".", "_")
        b_tf = multidoc_agg.get(f"percolation_tf_{theta}", {})
        b_tfidf = multidoc_agg.get(f"percolation_tfidf_{theta}", {})
        metrics_agg[f"mn_perc_tf_{safe}_rouge1_f"] = b_tf.get("mean_rouge1_f", 0.0)
        metrics_agg[f"mn_perc_tf_{safe}_compression_std"] = b_tf.get("std_compression_ratio", 0.0)
        metrics_agg[f"mn_perc_tfidf_{safe}_rouge1_f"] = b_tfidf.get("mean_rouge1_f", 0.0)
        metrics_agg[f"mn_perc_tfidf_{safe}_compression_std"] = b_tfidf.get("std_compression_ratio", 0.0)

    # N docs
    metrics_agg["cnndm_n_docs"] = float(cnndm_agg.get("n_docs", 0))
    metrics_agg["multinews_n_docs"] = float(multidoc_agg.get("n_docs", 0))

    # Best comparison: best percolation vs best fixed on CNN/DM ROUGE-1 F1
    best_perc_f1 = max(
        cnndm_agg.get(f"percolation_{t}", {}).get("mean_rouge1_f", 0.0) for t in THETAS
    )
    best_fixed_f1 = max(
        cnndm_agg.get(f"fixed_{r}", {}).get("mean_rouge1_f", 0.0) for r in FIXED_RATIOS
    )
    metrics_agg["cnndm_best_perc_rouge1_f"] = best_perc_f1
    metrics_agg["cnndm_best_fixed_rouge1_f"] = best_fixed_f1
    metrics_agg["cnndm_fixed_minus_perc_rouge1_f"] = round(best_fixed_f1 - best_perc_f1, 4)

    # Regression R² for theta=0.8 (most studied)
    reg_08 = reg_result.get("theta_0.8", {})
    metrics_agg["cnndm_regression_r2_theta08"] = reg_08.get("r2", 0.0)

    # Build per-example entries for CNN/DM
    cnndm_examples = []
    for d in cnndm_per_doc:
        doc_id = d["doc_id"]
        predict_fields = {}
        eval_fields = {}
        for theta in THETAS:
            k = str(theta)
            safe = k.replace(".", "_")
            if k in d.get("percolation", {}):
                p = d["percolation"][k]
                predict_fields[f"predict_percolation_{safe}"] = (
                    f"rouge1_f={p['rouge1_f']:.4f}|rouge1_r={p['rouge1_r']:.4f}|"
                    f"rouge1_p={p['rouge1_p']:.4f}|compression={p['compression_ratio']:.4f}"
                )
                eval_fields[f"eval_perc_{safe}_rouge1_f"] = p["rouge1_f"]
                eval_fields[f"eval_perc_{safe}_compression"] = p["compression_ratio"]
        for ratio in FIXED_RATIOS:
            k = str(ratio)
            safe = k.replace(".", "_")
            if k in d.get("fixed", {}):
                f_ = d["fixed"][k]
                predict_fields[f"predict_fixed_{safe}"] = (
                    f"rouge1_f={f_['rouge1_f']:.4f}|rouge1_r={f_['rouge1_r']:.4f}|"
                    f"rouge1_p={f_['rouge1_p']:.4f}"
                )
                eval_fields[f"eval_fixed_{safe}_rouge1_f"] = f_["rouge1_f"]

        cnndm_examples.append({
            "input": f"doc_id={doc_id}|n_sentences={d['n_sentences']}",
            "output": f"full_gcc={d['full_gcc']}|avg_degree={d['network']['avg_degree']}|"
                      f"clustering={d['network']['clustering']}",
            **predict_fields,
            **eval_fields,
            "metadata_doc_id": doc_id,
            "metadata_n_sentences": d["n_sentences"],
            "metadata_full_gcc": d["full_gcc"],
            "metadata_avg_degree": d["network"]["avg_degree"],
            "metadata_clustering": d["network"]["clustering"],
        })

    # Build per-example entries for Multi-News
    mn_examples = []
    for d in multidoc_per_doc:
        doc_id = d["doc_id"]
        predict_fields = {}
        eval_fields = {}
        for theta in THETAS:
            k = str(theta)
            safe = k.replace(".", "_")
            for sn in ["tf", "tfidf"]:
                mkey = f"percolation_{sn}"
                if k in d.get(mkey, {}):
                    p = d[mkey][k]
                    predict_fields[f"predict_perc_{sn}_{safe}"] = (
                        f"rouge1_f={p['rouge1_f']:.4f}|compression={p['compression_ratio']:.4f}"
                    )
                    eval_fields[f"eval_perc_{sn}_{safe}_rouge1_f"] = p["rouge1_f"]
                    eval_fields[f"eval_perc_{sn}_{safe}_compression"] = p["compression_ratio"]
        for ratio in FIXED_RATIOS:
            k = str(ratio)
            safe = k.replace(".", "_")
            for sn in ["tf", "tfidf"]:
                mkey = f"fixed_{sn}"
                if k in d.get(mkey, {}):
                    f_ = d[mkey][k]
                    predict_fields[f"predict_fixed_{sn}_{safe}"] = (
                        f"rouge1_f={f_['rouge1_f']:.4f}"
                    )
                    eval_fields[f"eval_fixed_{sn}_{safe}_rouge1_f"] = f_["rouge1_f"]

        mn_examples.append({
            "input": f"doc_id={doc_id}|n_sentences={d['n_sentences']}",
            "output": f"full_gcc={d['full_gcc']}|avg_degree={d['network']['avg_degree']}|"
                      f"clustering={d['network']['clustering']}",
            **predict_fields,
            **eval_fields,
            "metadata_doc_id": doc_id,
            "metadata_n_sentences": d["n_sentences"],
            "metadata_full_gcc": d["full_gcc"],
            "metadata_avg_degree": d["network"]["avg_degree"],
            "metadata_clustering": d["network"]["clustering"],
        })

    return {
        "metadata": {
            "description": "Percolation-threshold extractive summarizer evaluation on CNN/DM and Multi-News",
            "cnndm_aggregate": cnndm_agg,
            "multinews_aggregate": multidoc_agg,
            "regression_analysis": reg_result,
            "statistical_tests_cnndm": stat_tests_cnndm,
            "statistical_tests_multinews": stat_tests_mn,
            "calibration_cnndm": calib_cnndm,
            "cross_corpus_compression_std": cross_corpus,
            "tfidf_vs_tf_multinews": tfidf_vs_tf,
        },
        "metrics_agg": metrics_agg,
        "datasets": [
            {"dataset": "cnn_dailymail_3_0_0_test", "examples": cnndm_examples},
            {"dataset": "multi_news_test", "examples": mn_examples},
        ],
    }

# ── Main ──────────────────────────────────────────────────────────────────────
@logger.catch(reraise=True)
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_mini", type=int, default=None, help="Limit docs for mini run")
    parser.add_argument("--n_multidoc", type=int, default=MULTIDOC_N)
    args_cli = parser.parse_args()

    logger.info(f"NUM_CPUS={NUM_CPUS}, NUM_WORKERS={NUM_WORKERS}")

    # ── Load CNN/DM raw results ────────────────────────────────────────────────
    logger.info(f"Loading CNN/DM raw results from {CNNDM_RAW}")
    raw = json.loads(CNNDM_RAW.read_text())
    cnndm_per_doc = raw["per_document"]
    if args_cli.n_mini:
        cnndm_per_doc = cnndm_per_doc[:args_cli.n_mini]
    logger.info(f"Loaded {len(cnndm_per_doc)} CNN/DM per-doc results")

    cnndm_agg = aggregate_cnndm(cnndm_per_doc)
    logger.info(f"CNN/DM aggregated: theta=0.6 mean_rouge1_f={cnndm_agg.get('percolation_0.6',{}).get('mean_rouge1_f','?')}")

    # ── Run Multi-News ─────────────────────────────────────────────────────────
    ensure_nltk()
    logger.info("Loading Multi-News dataset from raw files...")
    from huggingface_hub import hf_hub_download
    src_path = hf_hub_download(repo_id="multi_news", repo_type="dataset",
                               filename="data/test.src.cleaned", local_dir="/tmp/multi_news")
    tgt_path = hf_hub_download(repo_id="multi_news", repo_type="dataset",
                               filename="data/test.tgt", local_dir="/tmp/multi_news")

    src_lines = Path(src_path).read_text(encoding="utf-8").splitlines()
    tgt_lines = Path(tgt_path).read_text(encoding="utf-8").splitlines()
    n_mn = min(args_cli.n_multidoc, len(src_lines))
    logger.info(f"Multi-News test split: {len(src_lines)} examples. Using {n_mn}.")

    def clean_mn_text(line: str) -> str:
        return line.replace("NEWLINE_CHAR", "\n").replace("|||||", "\n\n")

    tasks_mn = [
        (i, clean_mn_text(src_lines[i]), clean_mn_text(tgt_lines[i]))
        for i in range(n_mn)
    ]
    del src_lines, tgt_lines
    gc.collect()

    mn_per_doc = []
    batch_size = 50
    logger.info(f"Processing {n_mn} Multi-News docs with {NUM_WORKERS} workers")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, mp_context=mp.get_context("spawn")) as pool:
        for batch_start in range(0, len(tasks_mn), batch_size):
            batch = tasks_mn[batch_start: batch_start + batch_size]
            futures = {pool.submit(process_multidoc, t): t[0] for t in batch}
            batch_results = []
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    batch_results.append(res)
            batch_results.sort(key=lambda x: x["doc_id"])
            mn_per_doc.extend(batch_results)
            done = batch_start + len(batch)
            logger.info(f"Multi-News progress: {done}/{n_mn} ({len(mn_per_doc)} valid)")

    logger.info(f"Multi-News done: {len(mn_per_doc)} valid docs")

    # ── Analysis ───────────────────────────────────────────────────────────────
    logger.info("Aggregating Multi-News...")
    multidoc_agg = aggregate_multidoc(mn_per_doc)

    logger.info("Running OLS regression analysis (CNN/DM)...")
    reg_result = regression_analysis(cnndm_per_doc)
    for k, v in reg_result.items():
        logger.info(f"Regression {k}: R²={v['r2']}, adj_R²={v['adj_r2']}")

    logger.info("Running statistical tests (CNN/DM)...")
    stat_tests_cnndm = statistical_tests(cnndm_per_doc, corpus="cnndm")

    logger.info("Running statistical tests (Multi-News)...")
    # Build wrapper for multidoc tests (use TF scorer)
    class MNWrapper:
        pass
    mn_wrapped = []
    for d in mn_per_doc:
        wrapped = {
            "percolation": d.get("percolation_tf", {}),
            "fixed": d.get("fixed_tf", {}),
        }
        mn_wrapped.append(wrapped)
    stat_tests_mn = statistical_tests(mn_wrapped, corpus="multinews")

    logger.info("Running calibration analysis...")
    calib_cnndm = calibration_analysis(cnndm_per_doc)

    logger.info("Running cross-corpus comparison...")
    cross_corpus = cross_corpus_comparison(cnndm_per_doc, mn_per_doc)
    for k, v in cross_corpus.items():
        logger.info(f"Cross-corpus {k}: CNN/DM std={v['cnndm_compression_std']}, MN std={v['multinews_compression_std']}")

    logger.info("Running TF vs TF-IDF comparison on Multi-News...")
    tfidf_vs_tf = tfidf_vs_tf_comparison(mn_per_doc)

    # ── Assemble output ────────────────────────────────────────────────────────
    logger.info("Building eval_out...")
    eval_out = build_eval_out(
        cnndm_per_doc, cnndm_agg,
        mn_per_doc, multidoc_agg,
        reg_result,
        stat_tests_cnndm, stat_tests_mn,
        calib_cnndm, cross_corpus, tfidf_vs_tf,
    )

    out_path = Path("eval_out.json")
    out_path.write_text(json.dumps(eval_out, indent=2))
    logger.info(f"Saved eval_out.json ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Summary
    logger.info("=== SUMMARY ===")
    logger.info(f"CNN/DM n_docs={cnndm_agg['n_docs']}")
    logger.info(f"Multi-News n_docs={multidoc_agg['n_docs']}")
    for theta in THETAS:
        k = f"percolation_{theta}"
        b = cnndm_agg.get(k, {})
        logger.info(f"CNN/DM theta={theta}: rouge1_f={b.get('mean_rouge1_f','?')}, "
                    f"compression_std={b.get('std_compression_ratio','?')}, "
                    f"ceiling_frac={b.get('ceiling_fraction','?')}")
    for theta in THETAS:
        safe = str(theta)
        b = multidoc_agg.get(f"percolation_tf_{theta}", {})
        logger.info(f"Multi-News (TF) theta={theta}: rouge1_f={b.get('mean_rouge1_f','?')}, "
                    f"compression_std={b.get('std_compression_ratio','?')}")

    logger.info("Done!")

if __name__ == "__main__":
    main()
