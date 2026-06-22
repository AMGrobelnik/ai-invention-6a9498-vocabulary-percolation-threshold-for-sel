#!/usr/bin/env python3
"""Percolation-Threshold Extractive Summarizer: run experiment + statistical evaluation."""

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import nltk
import numpy as np
from loguru import logger
from rouge_score import rouge_scorer

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

WORKSPACE = Path(__file__).parent
THETAS = [0.6, 0.7, 0.8, 0.9]
FIXED_RATIOS = [0.10, 0.20, 0.30]
N_FULL = 2000
RANDOM_SEED = 42


# ── NLTK setup ────────────────────────────────────────────────────────────────
def ensure_nltk():
    # Ensure /root/nltk_data is on the path (directories already exist)
    import os
    nltk_root = "/root/nltk_data"
    if nltk_root not in nltk.data.path:
        nltk.data.path.insert(0, nltk_root)
    # Only download if directories are missing
    for pkg, subdir in [("punkt_tab", "tokenizers/punkt_tab"), ("stopwords", "corpora/stopwords")]:
        full_path = os.path.join(nltk_root, subdir)
        if not os.path.isdir(full_path):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


# ── Text processing ───────────────────────────────────────────────────────────
def preprocess(text: str, stop_words: set) -> list[str]:
    try:
        tokens = nltk.word_tokenize(text.lower())
    except Exception:
        tokens = text.lower().split()
    return [w for w in tokens if w.isalpha() and len(w) >= 2 and w not in stop_words]


def build_vocab_graph(sentences_words: list[list[str]]) -> nx.Graph:
    G = nx.Graph()
    for words in sentences_words:
        unique = list(set(words))
        G.add_nodes_from(unique)
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                if G.has_edge(unique[i], unique[j]):
                    G[unique[i]][unique[j]]["weight"] += 1
                else:
                    G.add_edge(unique[i], unique[j], weight=1)
    return G


def gcc_size(G: nx.Graph) -> int:
    if len(G) == 0:
        return 0
    return max((len(c) for c in nx.connected_components(G)), default=0)


def compute_tf(sentences_words: list[list[str]]) -> dict:
    c: Counter = Counter()
    for words in sentences_words:
        c.update(words)
    return dict(c)


def score_sentences(sentences_words: list[list[str]], tf: dict) -> list[float]:
    return [sum(tf.get(w, 0) for w in words) for words in sentences_words]


# ── Percolation summarizer ────────────────────────────────────────────────────
def percolation_summary(
    sentences_words: list[list[str]], tf: dict, full_G: nx.Graph, full_gcc: int, theta: float
) -> tuple[list[int], int]:
    n = len(sentences_words)
    if n == 0 or full_gcc == 0:
        return list(range(n)), n

    scores = score_sentences(sentences_words, tf)
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)

    accumulated_words: set = set()
    selected: list[int] = []
    for idx in ranked:
        selected.append(idx)
        accumulated_words.update(sentences_words[idx])
        sub = full_G.subgraph(accumulated_words)
        cur_gcc = gcc_size(sub)
        if cur_gcc / full_gcc >= theta:
            break

    k_star = len(selected)
    return selected, k_star


# ── Fixed-ratio summarizer ────────────────────────────────────────────────────
def fixed_ratio_summary(sentences_words: list[list[str]], tf: dict, ratio: float) -> list[int]:
    n = len(sentences_words)
    k = max(1, math.ceil(ratio * n))
    scores = score_sentences(sentences_words, tf)
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    return sorted(ranked[:k])


# ── ROUGE ─────────────────────────────────────────────────────────────────────
def evaluate_rouge(
    selected_indices: list[int], all_sentences: list[str], reference: str, scorer_obj
) -> dict:
    summary_text = " ".join(all_sentences[i] for i in sorted(selected_indices))
    if not summary_text.strip():
        summary_text = all_sentences[0] if all_sentences else ""
    scores = scorer_obj.score(reference, summary_text)
    return {k: {"p": v.precision, "r": v.recall, "f": v.fmeasure} for k, v in scores.items()}


# ── Network properties ────────────────────────────────────────────────────────
def network_properties(G: nx.Graph) -> dict:
    if len(G) == 0:
        return {"avg_degree": 0.0, "clustering_coefficient": 0.0, "graph_density": 0.0,
                "num_nodes": 0, "num_edges": 0}
    degrees = [d for _, d in G.degree()]
    avg_degree = float(np.mean(degrees))
    try:
        clustering = nx.average_clustering(G)
    except Exception:
        clustering = 0.0
    density = nx.density(G)
    return {
        "avg_degree": avg_degree,
        "clustering_coefficient": clustering,
        "graph_density": density,
        "num_nodes": len(G),
        "num_edges": G.number_of_edges(),
    }


# ── Per-document processing ───────────────────────────────────────────────────
def process_document(doc_id: int, article: str, highlights: str, source: str,
                     stop_words: set, scorer_obj) -> dict | None:
    try:
        sentences_raw = nltk.sent_tokenize(article)
        sentences_words = [preprocess(s, stop_words) for s in sentences_raw]
        # Filter out empty sentences
        valid_mask = [len(w) > 0 for w in sentences_words]
        sentences_raw_f = [s for s, v in zip(sentences_raw, valid_mask) if v]
        sentences_words_f = [w for w, v in zip(sentences_words, valid_mask) if v]

        if len(sentences_raw_f) < 2:
            return None

        tf = compute_tf(sentences_words_f)
        full_G = build_vocab_graph(sentences_words_f)
        full_gcc = gcc_size(full_G)
        net_props = network_properties(full_G)
        n = len(sentences_words_f)

        percolation_results = {}
        for theta in THETAS:
            indices, k_star = percolation_summary(sentences_words_f, tf, full_G, full_gcc, theta)
            rouge = evaluate_rouge(indices, sentences_raw_f, highlights, scorer_obj)
            compression = k_star / n
            percolation_results[str(theta)] = {
                "k_star": k_star,
                "compression_ratio": compression,
                "rouge1_r": rouge["rouge1"]["r"],
                "rouge2_r": rouge["rouge2"]["r"],
                "rougeL_r": rouge["rougeL"]["r"],
                "rouge1_p": rouge["rouge1"]["p"],
                "rouge2_p": rouge["rouge2"]["p"],
                "rougeL_p": rouge["rougeL"]["p"],
                "rouge1_f": rouge["rouge1"]["f"],
                "rouge2_f": rouge["rouge2"]["f"],
                "rougeL_f": rouge["rougeL"]["f"],
            }

        fixed_results = {}
        for ratio in FIXED_RATIOS:
            indices = fixed_ratio_summary(sentences_words_f, tf, ratio)
            rouge = evaluate_rouge(indices, sentences_raw_f, highlights, scorer_obj)
            fixed_results[str(ratio)] = {
                "k_used": len(indices),
                "compression_ratio": len(indices) / n,
                "rouge1_r": rouge["rouge1"]["r"],
                "rouge2_r": rouge["rouge2"]["r"],
                "rougeL_r": rouge["rougeL"]["r"],
                "rouge1_p": rouge["rouge1"]["p"],
                "rouge2_p": rouge["rouge2"]["p"],
                "rougeL_p": rouge["rougeL"]["p"],
                "rouge1_f": rouge["rouge1"]["f"],
                "rouge2_f": rouge["rouge2"]["f"],
                "rougeL_f": rouge["rougeL"]["f"],
            }

        return {
            "doc_id": doc_id,
            "source": source,
            "n_sentences": n,
            "full_gcc": full_gcc,
            "network": net_props,
            "percolation": percolation_results,
            "fixed": fixed_results,
        }
    except Exception as e:
        logger.error(f"Doc {doc_id} failed: {e}")
        return None


# ── Statistical evaluation ────────────────────────────────────────────────────
def statistical_evaluation(per_doc: list[dict]) -> dict:
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    n = len(per_doc)
    logger.info(f"Running stats on {n} documents")

    # Build arrays per method
    methods_perc = [str(t) for t in THETAS]
    methods_fixed = [str(r) for r in FIXED_RATIOS]
    rouge_metrics = ["rouge1_r", "rouge2_r", "rougeL_r"]

    def get_scores(method_type: str, key: str, metric: str) -> np.ndarray:
        arr = []
        for d in per_doc:
            v = d[method_type][key].get(metric, np.nan)
            arr.append(v)
        return np.array(arr)

    # ── Rouge summary table ──────────────────────────────────────────────────
    summary_rows = []
    for t in THETAS:
        key = str(t)
        row = {"method": f"percolation@{t}"}
        for m in ["rouge1_r", "rouge2_r", "rougeL_r", "rouge1_f", "rouge2_f", "rougeL_f"]:
            arr = get_scores("percolation", key, m)
            row[f"{m}_mean"] = float(np.mean(arr))
            row[f"{m}_std"] = float(np.std(arr))
        summary_rows.append(row)

    for r in FIXED_RATIOS:
        key = str(r)
        row = {"method": f"fixed@{int(r*100)}pct"}
        for m in ["rouge1_r", "rouge2_r", "rougeL_r", "rouge1_f", "rouge2_f", "rougeL_f"]:
            arr = get_scores("fixed", key, m)
            row[f"{m}_mean"] = float(np.mean(arr))
            row[f"{m}_std"] = float(np.std(arr))
        summary_rows.append(row)

    # ── Wilcoxon tests (36 comparisons) ─────────────────────────────────────
    wilcoxon_raw = []
    for t in THETAS:
        for r in FIXED_RATIOS:
            for metric in rouge_metrics:
                a = get_scores("percolation", str(t), metric)
                b = get_scores("fixed", str(r), metric)
                diff = a - b
                # skip zero-diff docs using wilcox default
                try:
                    stat, p = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
                except ValueError:
                    stat, p = 0.0, 1.0
                wilcoxon_raw.append({
                    "method_a": f"percolation@{t}",
                    "method_b": f"fixed@{int(r*100)}pct",
                    "metric": metric,
                    "statistic": float(stat),
                    "p_raw": float(p),
                })

    p_raws = [x["p_raw"] for x in wilcoxon_raw]
    reject, p_corrected, _, _ = multipletests(p_raws, alpha=0.05, method="holm")
    wilcoxon_tests = []
    for i, x in enumerate(wilcoxon_raw):
        wilcoxon_tests.append({**x, "p_corrected": float(p_corrected[i]), "significant": bool(reject[i])})

    # ── Compression ratio stats ──────────────────────────────────────────────
    # Use best-performing theta (pick theta with highest mean rouge1_r)
    best_theta = max(THETAS, key=lambda t: np.mean(get_scores("percolation", str(t), "rouge1_r")))

    def compression_stats(docs: list[dict]) -> dict:
        ratios = np.array([d["percolation"][str(best_theta)]["compression_ratio"] for d in docs])
        return {
            "mean": float(np.mean(ratios)),
            "std": float(np.std(ratios)),
            "min": float(np.min(ratios)),
            "max": float(np.max(ratios)),
            "p25": float(np.percentile(ratios, 25)),
            "p50": float(np.percentile(ratios, 50)),
            "p75": float(np.percentile(ratios, 75)),
        }

    # Split by article length as CNN/DM proxy (CNN shorter, DailyMail longer)
    n_sentences_list = [d["n_sentences"] for d in per_doc]
    median_sents = float(np.median(n_sentences_list))
    cnn_docs = [d for d in per_doc if d["n_sentences"] <= median_sents]
    dm_docs = [d for d in per_doc if d["n_sentences"] > median_sents]

    overall_stats = compression_stats(per_doc)
    overall_stats["std_exceeds_10pp"] = overall_stats["std"] > 0.10

    comp_ratio_stats = {
        "overall": overall_stats,
        "cnn": compression_stats(cnn_docs) if cnn_docs else {},
        "dailymail": compression_stats(dm_docs) if dm_docs else {},
        "std_exceeds_10pp": overall_stats["std"] > 0.10,
        "best_theta_used": float(best_theta),
    }

    # ── Regression: network features → compression ratio ────────────────────
    feature_names = ["avg_degree", "clustering_coefficient", "graph_density", "num_nodes", "n_sentences"]
    X_rows = []
    y_vals = []
    for d in per_doc:
        net = d["network"]
        X_rows.append([
            net["avg_degree"],
            net["clustering_coefficient"],
            net["graph_density"],
            net["num_nodes"],
            float(d["n_sentences"]),
        ])
        y_vals.append(d["percolation"][str(best_theta)]["compression_ratio"])

    X = np.array(X_rows)
    y = np.array(y_vals)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    reg = LinearRegression()
    reg.fit(X_scaled, y)
    y_pred = reg.predict(X_scaled)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    pearson_corr = {}
    spearman_corr = {}
    for i, feat in enumerate(feature_names):
        p_r, p_p = stats.pearsonr(X[:, i], y)
        s_r, s_p = stats.spearmanr(X[:, i], y)
        pearson_corr[feat] = {"r": float(p_r), "p": float(p_p)}
        spearman_corr[feat] = {"r": float(s_r), "p": float(s_p)}

    regression = {
        "r2": float(r2),
        "best_theta": float(best_theta),
        "features": feature_names,
        "coefficients": [float(c) for c in reg.coef_],
        "intercept": float(reg.intercept_),
        "correlations": {"pearson": pearson_corr, "spearman": spearman_corr},
    }

    # ── Per-theta regression ─────────────────────────────────────────────────
    per_theta_r2 = {}
    for t in THETAS:
        yt = np.array([d["percolation"][str(t)]["compression_ratio"] for d in per_doc])
        reg_t = LinearRegression()
        reg_t.fit(X_scaled, yt)
        yp = reg_t.predict(X_scaled)
        ss_r = np.sum((yt - yp) ** 2)
        ss_t = np.sum((yt - np.mean(yt)) ** 2)
        per_theta_r2[str(t)] = float(1 - ss_r / ss_t) if ss_t > 0 else 0.0
    regression["per_theta_r2"] = per_theta_r2

    # ── Segment analysis ─────────────────────────────────────────────────────
    def seg_rouge_stats(docs: list[dict], method_type: str, key: str, metric: str) -> dict:
        arr = np.array([d[method_type][key][metric] for d in docs]) if docs else np.array([])
        if len(arr) == 0:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "n": len(arr)}

    segment_analysis = {"cnn": {}, "dailymail": {}}
    for seg_name, seg_docs in [("cnn", cnn_docs), ("dailymail", dm_docs)]:
        for t in THETAS:
            segment_analysis[seg_name][f"percolation@{t}"] = seg_rouge_stats(
                seg_docs, "percolation", str(t), "rouge1_r")
        for r in FIXED_RATIOS:
            segment_analysis[seg_name][f"fixed@{int(r*100)}pct"] = seg_rouge_stats(
                seg_docs, "fixed", str(r), "rouge1_r")

    # ── Verdict ──────────────────────────────────────────────────────────────
    # Find best percolation vs best fixed on rouge1_r
    best_perc_mean = max(
        float(np.mean(get_scores("percolation", str(t), "rouge1_r"))) for t in THETAS
    )
    best_fixed_mean = max(
        float(np.mean(get_scores("fixed", str(r), "rouge1_r"))) for r in FIXED_RATIOS
    )
    best_fixed_ratio = max(
        FIXED_RATIOS, key=lambda r: float(np.mean(get_scores("fixed", str(r), "rouge1_r")))
    )

    # Is the best percolation theta significantly better than best fixed on rouge1_r?
    a = get_scores("percolation", str(best_theta), "rouge1_r")
    b = get_scores("fixed", str(best_fixed_ratio), "rouge1_r")
    try:
        _, p_best = stats.wilcoxon(a - b, alternative="greater", zero_method="wilcox")
    except ValueError:
        p_best = 1.0

    # find if any wilcoxon test for rouge1_r is significant
    rouge1_sig = any(
        wt["significant"] for wt in wilcoxon_tests if wt["metric"] == "rouge1_r"
    )

    verdict = {
        "hypothesis_supported": rouge1_sig and comp_ratio_stats["std_exceeds_10pp"],
        "best_percolation_theta": float(best_theta),
        "best_fixed_ratio": float(best_fixed_ratio),
        "best_percolation_rouge1r_mean": float(best_perc_mean),
        "best_fixed_rouge1r_mean": float(best_fixed_mean),
        "rouge1_recall_improvement": float(best_perc_mean - best_fixed_mean),
        "rouge1_significant": rouge1_sig,
        "compression_ratio_variable": comp_ratio_stats["std_exceeds_10pp"],
        "p_best_pair_wilcoxon": float(p_best),
        "notes": (
            "Percolation threshold method compared to fixed-ratio TF baselines. "
            f"Best theta={best_theta}, best fixed={best_fixed_ratio}. "
            f"ROUGE-1 recall improvement={best_perc_mean - best_fixed_mean:.4f}. "
            f"Compression std={overall_stats['std']:.4f} ({'>' if comp_ratio_stats['std_exceeds_10pp'] else '<='} 0.10). "
            f"Wilcoxon rouge1 significant: {rouge1_sig}."
        ),
    }

    return {
        "n_documents": n,
        "rouge_summary_table": summary_rows,
        "wilcoxon_tests": wilcoxon_tests,
        "compression_ratio_stats": comp_ratio_stats,
        "regression": regression,
        "segment_analysis": segment_analysis,
        "verdict": verdict,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
@logger.catch(reraise=True)
def main():
    ensure_nltk()
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))
    scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    method_out_path = WORKSPACE / "method_out.json"

    # Check if method_out.json already exists (cache)
    if method_out_path.exists():
        logger.info("Loading cached method_out.json")
        per_doc = json.loads(method_out_path.read_text())
    else:
        logger.info("Loading CNN/DailyMail dataset...")
        from datasets import load_dataset
        ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
        logger.info(f"Dataset size: {len(ds)}")

        rng = random.Random(RANDOM_SEED)
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        sample_indices = indices[:N_FULL]

        per_doc = []
        for i, idx in enumerate(sample_indices):
            row = ds[idx]
            article = row["article"]
            highlights = row["highlights"]
            # CNN/DM source: approximate by article length (CNN shorter, DailyMail longer)
            # In 3.0.0, ids are hashes with no prefix; split by median as proxy
            source = "cnn_dailymail"

            result = process_document(i, article, highlights, source, stop_words, scorer_obj)
            if result is not None:
                per_doc.append(result)

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{len(sample_indices)}, valid={len(per_doc)}")

        logger.info(f"Total valid documents: {len(per_doc)}")
        method_out_path.write_text(json.dumps(per_doc, indent=2))
        logger.info("Saved method_out.json")

    logger.info("Running statistical evaluation...")
    eval_result = statistical_evaluation(per_doc)

    # ── Build eval_out.json in exp_eval_sol_out schema ───────────────────────
    # Schema requires: metrics_agg (flat dict of numbers) + datasets (list with dataset+examples)
    metrics_agg = {
        "n_documents": eval_result["n_documents"],
        "best_percolation_rouge1r_mean": eval_result["verdict"]["best_percolation_rouge1r_mean"],
        "best_fixed_rouge1r_mean": eval_result["verdict"]["best_fixed_rouge1r_mean"],
        "rouge1_recall_improvement": eval_result["verdict"]["rouge1_recall_improvement"],
        "compression_ratio_mean": eval_result["compression_ratio_stats"]["overall"]["mean"],
        "compression_ratio_std": eval_result["compression_ratio_stats"]["overall"]["std"],
        "std_exceeds_10pp": float(eval_result["compression_ratio_stats"]["std_exceeds_10pp"]),
        "regression_r2": eval_result["regression"]["r2"],
        "best_percolation_theta": eval_result["verdict"]["best_percolation_theta"],
        "best_fixed_ratio": eval_result["verdict"]["best_fixed_ratio"],
        "hypothesis_supported": float(eval_result["verdict"]["hypothesis_supported"]),
        "rouge1_significant": float(eval_result["verdict"]["rouge1_significant"]),
        "p_best_pair_wilcoxon": eval_result["verdict"]["p_best_pair_wilcoxon"],
    }
    # Add per-method ROUGE means
    for row in eval_result["rouge_summary_table"]:
        method_key = row["method"].replace("@", "_at_").replace("%", "pct").replace(".", "_")
        for metric in ["rouge1_r", "rouge2_r", "rougeL_r"]:
            k = f"{method_key}_{metric}_mean"
            metrics_agg[k] = row[f"{metric}_mean"]

    # Examples: one per document with all predictions
    examples = []
    for d in per_doc:
        input_str = f"doc_id={d['doc_id']} source={d['source']} n_sentences={d['n_sentences']}"
        output_str = json.dumps(d["percolation"])
        example = {
            "input": input_str,
            "output": output_str,
        }
        # Per-example eval metrics (schema allows eval_*)
        for t in THETAS:
            key = str(t)
            example[f"eval_rouge1r_perc_{str(t).replace('.','_')}"] = d["percolation"][key]["rouge1_r"]
            example[f"eval_compression_perc_{str(t).replace('.','_')}"] = d["percolation"][key]["compression_ratio"]
        for r in FIXED_RATIOS:
            key = str(r)
            example[f"eval_rouge1r_fixed_{str(int(r*100))}pct"] = d["fixed"][key]["rouge1_r"]
        # predict fields (schema allows predict_*)
        best_theta = eval_result["verdict"]["best_percolation_theta"]
        example["predict_percolation_summary"] = str(d["percolation"][str(best_theta)]["compression_ratio"])
        examples.append(example)

    eval_out = {
        "metadata": {
            "evaluation_name": "Percolation vs Fixed-Ratio Extractive Summarization",
            "n_documents": eval_result["n_documents"],
            "thetas": THETAS,
            "fixed_ratios": FIXED_RATIOS,
            "verdict": eval_result["verdict"],
            "rouge_summary_table": eval_result["rouge_summary_table"],
            "wilcoxon_tests": eval_result["wilcoxon_tests"],
            "compression_ratio_stats": eval_result["compression_ratio_stats"],
            "regression": eval_result["regression"],
            "segment_analysis": eval_result["segment_analysis"],
        },
        "metrics_agg": metrics_agg,
        "datasets": [
            {
                "dataset": "cnn_dailymail",
                "examples": examples,
            }
        ],
    }

    eval_out_path = WORKSPACE / "eval_out.json"
    eval_out_path.write_text(json.dumps(eval_out, indent=2))
    logger.info(f"Saved eval_out.json ({eval_out_path.stat().st_size // 1024} KB)")

    # Print summary
    v = eval_result["verdict"]
    logger.info("=" * 60)
    logger.info(f"VERDICT: hypothesis_supported={v['hypothesis_supported']}")
    logger.info(f"  Best theta={v['best_percolation_theta']}, best fixed={v['best_fixed_ratio']}")
    logger.info(f"  ROUGE-1 recall improvement: {v['rouge1_recall_improvement']:.4f}")
    logger.info(f"  Compression std > 10pp: {v['compression_ratio_variable']}")
    logger.info(f"  ROUGE-1 Wilcoxon significant: {v['rouge1_significant']}")
    logger.info(f"  Regression R²: {eval_result['regression']['r2']:.4f}")
    logger.info("=" * 60)

    return eval_out


if __name__ == "__main__":
    main()
