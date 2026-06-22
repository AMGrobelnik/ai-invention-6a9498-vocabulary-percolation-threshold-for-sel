#!/usr/bin/env python3
"""Percolation-threshold vs fixed-ratio extractive summarization on CNN/DM and Multi-News."""

import gc
import json
import math
import os
import resource
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import nltk
import numpy as np
from loguru import logger
from rouge_score import rouge_scorer as rs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

# ── Logging ───────────────────────────────────────────────────────────────────
WS = Path(__file__).parent
(WS / "logs").mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(WS / "logs/run.log"), rotation="30 MB", level="DEBUG")

# ── Hardware / memory limits ──────────────────────────────────────────────────
def _container_ram_gb() -> float:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return 8.0

TOTAL_RAM_GB = _container_ram_gb()
RAM_BUDGET = int(TOTAL_RAM_GB * 0.70 * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

# ── Constants ─────────────────────────────────────────────────────────────────
THETAS = [0.6, 0.7, 0.8, 0.9]
FIXED_RATIOS = [0.10, 0.20, 0.30]
MAX_SENTENCES = 100  # cap for very long Multi-News docs

# ── NLTK setup (done once at import) ─────────────────────────────────────────
logger.info("Loading NLTK resources...")
for res_id, res_type in [("punkt_tab", "tokenizers"), ("punkt", "tokenizers"), ("stopwords", "corpora")]:
    try:
        nltk.data.find(f"{res_type}/{res_id}")
    except LookupError:
        nltk.download(res_id, quiet=True)

from nltk.corpus import stopwords as _sw
from nltk.tokenize import sent_tokenize

STOPWORDS = set(_sw.words("english"))
PUNCT_TABLE = str.maketrans("", "", string.punctuation)
ROUGE_SCORER = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

logger.info("NLTK ready.")

# ── Text helpers ──────────────────────────────────────────────────────────────
def tokenize_sentences(text: str) -> list[str]:
    sents = sent_tokenize(text.strip())
    return [s.strip() for s in sents if s.strip() and len(s.split()) >= 3]


def get_content_words(sentence: str) -> list[str]:
    tokens = sentence.lower().translate(PUNCT_TABLE).split()
    return [w for w in tokens if w not in STOPWORDS and len(w) > 2]


# ── Graph helpers ─────────────────────────────────────────────────────────────
def build_vocab_graph(cw_per_sent: list[list[str]]) -> nx.Graph:
    G: nx.Graph = nx.Graph()
    for cws in cw_per_sent:
        unique = list(set(cws))
        G.add_nodes_from(unique)
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                u, v = unique[i], unique[j]
                if G.has_edge(u, v):
                    G[u][v]["weight"] += 1
                else:
                    G.add_edge(u, v, weight=1)
    return G


def gcc_size(G: nx.Graph) -> int:
    if len(G.nodes) == 0:
        return 0
    return len(max(nx.connected_components(G), key=len))


# ── Scoring ───────────────────────────────────────────────────────────────────
def tf_scores(cw_per_sent: list[list[str]]) -> list[float]:
    freq: Counter = Counter()
    for cws in cw_per_sent:
        freq.update(cws)
    return [
        sum(freq[w] for w in cws) / len(cws) if cws else 0.0
        for cws in cw_per_sent
    ]


def tfidf_scores(sentences: list[str]) -> list[float]:
    if len(sentences) < 2:
        return [1.0] * len(sentences)
    try:
        vec = TfidfVectorizer(stop_words="english", min_df=1)
        X = vec.fit_transform(sentences)
        return list(np.asarray(X.sum(axis=1)).flatten())
    except Exception:
        return [0.0] * len(sentences)


# ── Percolation pipeline ──────────────────────────────────────────────────────
def percolation_summary(
    cw_per_sent: list[list[str]],
    sent_scores: list[float],
    full_gcc: int,
) -> dict[str, Any]:
    n = len(sent_scores)
    order = sorted(range(n), key=lambda i: sent_scores[i], reverse=True)

    results: dict[float, int | None] = {t: None for t in THETAS}
    G_summary: nx.Graph = nx.Graph()
    gcc_curve: list[float] = []

    for step, idx in enumerate(order):
        cws = list(set(cw_per_sent[idx]))
        G_summary.add_nodes_from(cws)
        for i in range(len(cws)):
            for j in range(i + 1, len(cws)):
                u, v = cws[i], cws[j]
                if G_summary.has_edge(u, v):
                    G_summary[u][v]["weight"] += 1
                else:
                    G_summary.add_edge(u, v, weight=1)
        cur_gcc = gcc_size(G_summary)
        ratio = cur_gcc / full_gcc if full_gcc > 0 else 0.0
        gcc_curve.append(ratio)
        for t in THETAS:
            if results[t] is None and ratio >= t:
                results[t] = step + 1

    k_star = {}
    ceiling_hit = {}
    for t in THETAS:
        if results[t] is None:
            ceiling_hit[t] = True
            k_star[t] = n
        else:
            ceiling_hit[t] = False
            k_star[t] = results[t]

    compression = {t: k_star[t] / n for t in THETAS}
    return {
        "k_star": k_star,
        "compression": compression,
        "ceiling_hit": ceiling_hit,
        "gcc_curve": gcc_curve,
    }


# ── ROUGE evaluation ──────────────────────────────────────────────────────────
def evaluate_summary(selected: list[int], sentences: list[str], reference: str) -> dict[str, float]:
    if not selected:
        return {k: 0.0 for k in ["rouge1_f","rouge1_r","rouge1_p","rouge2_f","rouge2_r","rougeL_f","rougeL_r"]}
    text = " ".join(sentences[i] for i in selected)
    s = ROUGE_SCORER.score(reference, text)
    return {
        "rouge1_f": s["rouge1"].fmeasure,
        "rouge1_r": s["rouge1"].recall,
        "rouge1_p": s["rouge1"].precision,
        "rouge2_f": s["rouge2"].fmeasure,
        "rouge2_r": s["rouge2"].recall,
        "rougeL_f": s["rougeL"].fmeasure,
        "rougeL_r": s["rougeL"].recall,
    }


def fixed_summary(n: int, scores: list[float], ratio: float) -> list[int]:
    k = max(1, round(n * ratio))
    top_k = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
    return sorted(top_k)


# ── Per-document processing ───────────────────────────────────────────────────
def process_document(article_text: str, reference_text: str, doc_id: str, corpus_name: str) -> dict | None:
    sentences = tokenize_sentences(article_text)
    if len(sentences) < 3:
        return None
    if len(sentences) > MAX_SENTENCES:
        sentences = sentences[:MAX_SENTENCES]

    cw_per_sent = [get_content_words(s) for s in sentences]
    full_G = build_vocab_graph(cw_per_sent)
    full_gcc = gcc_size(full_G)

    deg_vals = [d for _, d in full_G.degree()]
    avg_degree = float(np.mean(deg_vals)) if deg_vals else 0.0
    clustering = float(nx.average_clustering(full_G)) if full_G.nodes else 0.0
    density = float(nx.density(full_G))

    tf_s = tf_scores(cw_per_sent)
    tfidf_s = tfidf_scores(sentences)

    # Build record in exp_gen_sol_out schema format
    # input = article, output = reference, predict_* = summary texts, metadata_* = metrics
    record: dict[str, Any] = {
        "input": article_text,
        "output": reference_text,
        "metadata_doc_id": doc_id,
        "metadata_corpus": corpus_name,
        "metadata_n_sentences": len(sentences),
        "metadata_full_gcc": full_gcc,
        "metadata_avg_degree": round(avg_degree, 4),
        "metadata_clustering": round(clustering, 4),
        "metadata_density": round(density, 6),
    }

    for scorer_name, scores in [("tf", tf_s), ("tfidf", tfidf_s)]:
        perc = percolation_summary(cw_per_sent, scores, full_gcc)

        for theta in THETAS:
            k_star = perc["k_star"][theta]
            selected = sorted(
                sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k_star]
            )
            variant = f"{scorer_name}_theta{int(theta*10)}"
            summary_text = " ".join(sentences[i] for i in selected)
            rouge = evaluate_summary(selected, sentences, reference_text)
            record[f"predict_{variant}"] = summary_text
            record[f"metadata_{variant}_k_star"] = k_star
            record[f"metadata_{variant}_compression_ratio"] = round(perc["compression"][theta], 4)
            record[f"metadata_{variant}_ceiling_hit"] = perc["ceiling_hit"][theta]
            record[f"metadata_{variant}_rouge1_f"] = round(rouge["rouge1_f"], 5)
            record[f"metadata_{variant}_rouge1_r"] = round(rouge["rouge1_r"], 5)
            record[f"metadata_{variant}_rouge1_p"] = round(rouge["rouge1_p"], 5)
            record[f"metadata_{variant}_rouge2_f"] = round(rouge["rouge2_f"], 5)
            record[f"metadata_{variant}_rouge2_r"] = round(rouge["rouge2_r"], 5)
            record[f"metadata_{variant}_rougeL_f"] = round(rouge["rougeL_f"], 5)
            record[f"metadata_{variant}_rougeL_r"] = round(rouge["rougeL_r"], 5)

        # GCC decile curve (compact form)
        curve = perc["gcc_curve"]
        nc = len(curve)
        deciles = []
        for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            idx = max(0, min(int(frac * nc) - 1, nc - 1))
            deciles.append(round(curve[idx], 4))
        record[f"metadata_{scorer_name}_gcc_deciles"] = deciles

        for ratio in FIXED_RATIOS:
            selected = fixed_summary(len(sentences), scores, ratio)
            variant = f"{scorer_name}_fixed{int(ratio*100)}"
            summary_text = " ".join(sentences[i] for i in selected)
            rouge = evaluate_summary(selected, sentences, reference_text)
            record[f"predict_{variant}"] = summary_text
            record[f"metadata_{variant}_rouge1_f"] = round(rouge["rouge1_f"], 5)
            record[f"metadata_{variant}_rouge1_r"] = round(rouge["rouge1_r"], 5)
            record[f"metadata_{variant}_rouge1_p"] = round(rouge["rouge1_p"], 5)
            record[f"metadata_{variant}_rouge2_f"] = round(rouge["rouge2_f"], 5)
            record[f"metadata_{variant}_rouge2_r"] = round(rouge["rouge2_r"], 5)
            record[f"metadata_{variant}_rougeL_f"] = round(rouge["rougeL_f"], 5)
            record[f"metadata_{variant}_rougeL_r"] = round(rouge["rougeL_r"], 5)

    return record


# ── Aggregate statistics ──────────────────────────────────────────────────────
def aggregate(records: list[dict]) -> dict:
    valid = [r for r in records if "metadata_doc_id" in r]
    agg: dict[str, Any] = {}

    for scorer_name in ["tf", "tfidf"]:
        agg[scorer_name] = {}

        for theta in THETAS:
            variant = f"{scorer_name}_theta{int(theta*10)}"
            sub = [r for r in valid if f"metadata_{variant}_rouge1_f" in r]
            if not sub:
                continue
            m: dict[str, Any] = {"n": len(sub)}
            for metric in ["rouge1_f","rouge1_r","rouge1_p","rouge2_f","rouge2_r","rougeL_f","rougeL_r"]:
                arr = [r[f"metadata_{variant}_{metric}"] for r in sub]
                m[f"{metric}_mean"] = round(float(np.mean(arr)), 5)
                m[f"{metric}_std"] = round(float(np.std(arr)), 5)
            cr_arr = [r[f"metadata_{variant}_compression_ratio"] for r in sub]
            m["compression_ratio_mean"] = round(float(np.mean(cr_arr)), 5)
            m["compression_ratio_std"] = round(float(np.std(cr_arr)), 5)
            m["ceiling_hit_frac"] = round(float(np.mean([r[f"metadata_{variant}_ceiling_hit"] for r in sub])), 4)
            agg[scorer_name][f"theta{int(theta*10)}"] = m

        for ratio in FIXED_RATIOS:
            variant = f"{scorer_name}_fixed{int(ratio*100)}"
            sub = [r for r in valid if f"metadata_{variant}_rouge1_f" in r]
            if not sub:
                continue
            m = {"n": len(sub)}
            for metric in ["rouge1_f","rouge1_r","rouge1_p","rouge2_f","rouge2_r","rougeL_f","rougeL_r"]:
                arr = [r[f"metadata_{variant}_{metric}"] for r in sub]
                m[f"{metric}_mean"] = round(float(np.mean(arr)), 5)
                m[f"{metric}_std"] = round(float(np.std(arr)), 5)
            agg[scorer_name][f"fixed{int(ratio*100)}"] = m

    return agg


def r2_analysis(records: list[dict]) -> dict:
    valid = [r for r in records if "metadata_doc_id" in r]
    result: dict[str, Any] = {}
    for scorer_name in ["tf", "tfidf"]:
        result[scorer_name] = {}
        for theta in THETAS:
            variant = f"{scorer_name}_theta{int(theta*10)}"
            sub = [r for r in valid if f"metadata_{variant}_compression_ratio" in r]
            if len(sub) < 5:
                continue
            cr_arr = np.array([r[f"metadata_{variant}_compression_ratio"] for r in sub])
            feat = np.column_stack([
                [r.get("metadata_avg_degree", 0.0) for r in sub],
                [r.get("metadata_clustering", 0.0) for r in sub],
                [r.get("metadata_density", 0.0) for r in sub],
                [r.get("metadata_n_sentences", 0) for r in sub],
                [r.get("metadata_full_gcc", 0) for r in sub],
            ])
            reg = LinearRegression().fit(feat, cr_arr)
            result[scorer_name][f"theta{int(theta*10)}"] = {
                "r2": round(float(reg.score(feat, cr_arr)), 5),
                "features": ["avg_degree", "clustering", "density", "n_sentences", "full_gcc"],
                "coefficients": [round(float(c), 6) for c in reg.coef_],
                "n": len(sub),
            }
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
@logger.catch(reraise=True)
def main():
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cnndm", type=int, default=2000)
    parser.add_argument("--max-mn", type=int, default=500)
    parser.add_argument("--skip-mn", action="store_true")
    args_cli = parser.parse_args()

    logger.info(f"RAM: {TOTAL_RAM_GB:.1f} GB container limit")

    # ── Load CNN/DM ───────────────────────────────────────────────────────────
    cnndm_path = WS.parent.parent.parent / "iter_1/gen_art/gen_art_dataset_1/full_data_out.json"
    logger.info(f"Loading CNN/DM from {cnndm_path}")
    cnndm_data = json.loads(cnndm_path.read_text())
    cnndm_examples = cnndm_data["datasets"][0]["examples"][: args_cli.max_cnndm]
    del cnndm_data
    gc.collect()
    logger.info(f"CNN/DM: {len(cnndm_examples)} examples")

    # ── Process CNN/DM ────────────────────────────────────────────────────────
    cnndm_records: list[dict] = []
    t0 = time.time()
    for i, ex in enumerate(cnndm_examples):
        try:
            rec = process_document(ex["input"], ex["output"], ex["metadata_id"], "cnndm")
            if rec is not None:
                cnndm_records.append(rec)
        except Exception as e:
            logger.error(f"CNN/DM doc {i} failed: {e}")
            cnndm_records.append({"doc_id": ex.get("metadata_id", str(i)), "corpus": "cnndm", "_error": str(e)})
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(cnndm_examples) - i - 1) / rate
            logger.info(f"CNN/DM: {i+1}/{len(cnndm_examples)} | rate={rate:.1f} doc/s | ETA={eta:.0f}s")

    errors_cnndm = sum(1 for r in cnndm_records if "_error" in r)
    logger.info(f"CNN/DM done: {len(cnndm_records)} records, {errors_cnndm} errors in {time.time()-t0:.1f}s")
    del cnndm_examples
    gc.collect()

    # ── Load and process Multi-News ───────────────────────────────────────────
    mn_records: list[dict] = []
    if not args_cli.skip_mn:
        try:
            from datasets import load_dataset
            logger.info("Loading Multi-News from HuggingFace...")
            mn_ds = load_dataset("multi_news", split="validation", trust_remote_code=True)
            total_mn = len(mn_ds)
            step = max(1, total_mn // args_cli.max_mn)
            mn_indices = list(range(0, total_mn, step))[: args_cli.max_mn]
            logger.info(f"Multi-News: {len(mn_indices)} examples sampled from {total_mn}")

            t0 = time.time()
            for j, idx in enumerate(mn_indices):
                try:
                    ex = mn_ds[int(idx)]
                    article = ex["document"].replace("|||||", "\n")
                    ref = ex["summary"]
                    rec = process_document(article, ref, f"mn_{idx}", "multi_news")
                    if rec is not None:
                        mn_records.append(rec)
                except Exception as e:
                    logger.error(f"Multi-News doc {idx} failed: {e}")
                    mn_records.append({"doc_id": f"mn_{idx}", "corpus": "multi_news", "_error": str(e)})
                if (j + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    rate = (j + 1) / elapsed
                    eta = (len(mn_indices) - j - 1) / rate
                    logger.info(f"Multi-News: {j+1}/{len(mn_indices)} | rate={rate:.1f} doc/s | ETA={eta:.0f}s")

            errors_mn = sum(1 for r in mn_records if "_error" in r)
            logger.info(f"Multi-News done: {len(mn_records)} records, {errors_mn} errors in {time.time()-t0:.1f}s")
            del mn_ds
            gc.collect()
        except Exception as e:
            logger.warning(f"Multi-News failed: {e}. Skipping.")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    logger.info("Computing aggregate statistics...")
    agg_cnndm = aggregate(cnndm_records)
    agg_mn = aggregate(mn_records) if mn_records else {}
    r2_cnndm = r2_analysis(cnndm_records)
    r2_mn = r2_analysis(mn_records) if mn_records else {}

    # ── Build output (exp_gen_sol_out schema) ─────────────────────────────────
    datasets_list = []
    if cnndm_records:
        datasets_list.append({"dataset": "cnndm", "examples": cnndm_records})
    if mn_records:
        datasets_list.append({"dataset": "multi_news", "examples": mn_records})

    output = {
        "metadata": {
            "description": "Percolation-threshold vs fixed-ratio extractive summarization",
            "thetas": THETAS,
            "fixed_ratios": FIXED_RATIOS,
            "scorers": ["tf", "tfidf"],
            "max_sentences_per_doc": MAX_SENTENCES,
            "n_cnndm": len(cnndm_records),
            "n_multi_news": len(mn_records),
            "aggregates": {"cnndm": agg_cnndm, "multi_news": agg_mn},
            "r2_analysis": {"cnndm": r2_cnndm, "multi_news": r2_mn},
        },
        "datasets": datasets_list,
    }

    out_path = WS / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Saved method_out.json ({size_mb:.1f} MB)")

    # ── Summary table ─────────────────────────────────────────────────────────
    for corpus_name, agg in [("cnndm", agg_cnndm), ("multi_news", agg_mn)]:
        if not agg:
            continue
        logger.info(f"\n=== {corpus_name.upper()} ===")
        for scorer_name in ["tf", "tfidf"]:
            if scorer_name not in agg:
                continue
            logger.info(f"  [{scorer_name}]")
            for variant, stats in sorted(agg[scorer_name].items()):
                r1 = stats.get("rouge1_f_mean", float("nan"))
                r2 = stats.get("rouge2_f_mean", float("nan"))
                cr = stats.get("compression_ratio_mean", float("nan"))
                ch = stats.get("ceiling_hit_frac", float("nan"))
                n = stats.get("n", 0)
                cr_str = f"{cr:.3f}" if cr == cr else "n/a"
                ch_str = f"{ch:.3f}" if ch == ch else "n/a"
                logger.info(f"    {variant:12s}: R1={r1:.3f} R2={r2:.3f} CR={cr_str} CH={ch_str} n={n}")


if __name__ == "__main__":
    main()
