#!/usr/bin/env python3
"""Load CNN/DailyMail 3.0.0 test split from temp/datasets/, standardize to exp_sel_data_out schema, save full_data_out.json."""

from loguru import logger
from pathlib import Path
import json
import sys

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WS = Path(__file__).parent


@logger.catch(reraise=True)
def load_cnndm() -> list[dict]:
    path = WS / "temp/datasets/full_abisee_cnn_dailymail_3.0.0_test.json"
    logger.info(f"Loading CNN/DailyMail from {path}")
    raw = json.loads(path.read_text())
    logger.info(f"CNN/DailyMail: {len(raw)} rows total, taking first 2000")
    examples = []
    for i, ex in enumerate(raw[:2000]):
        article = ex.get("article", "").strip()
        highlights = ex.get("highlights", "").strip()
        if not article or not highlights:
            logger.warning(f"Skipping empty row {i}")
            continue
        examples.append({
            "input": article,
            "output": highlights,
            "metadata_id": f"cnndm_test_{i:04d}",
            "metadata_row_index": i,
            "metadata_task_type": "summarization",
            "metadata_source_id": ex.get("id", ""),
        })
    logger.info(f"CNN/DailyMail: {len(examples)} examples built")
    return examples



@logger.catch(reraise=True)
def main():
    cnndm_examples = load_cnndm()

    out = {
        "metadata": {
            "description": "CNN/DailyMail 3.0.0 test split — 2000 news article/summary pairs for extractive summarization",
            "task": "single-document news summarization",
            "source": "abisee/cnn_dailymail",
            "config": "3.0.0",
            "split": "test",
            "n_examples": len(cnndm_examples),
        },
        "datasets": [
            {"dataset": "cnn_dailymail", "examples": cnndm_examples},
        ],
    }

    out_path = WS / "full_data_out.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info(f"Saved {len(cnndm_examples)} examples to {out_path}")

    ins = [len(e["input"].split()) for e in cnndm_examples]
    outs = [len(e["output"].split()) for e in cnndm_examples]
    logger.info(
        f"input_words mean={sum(ins)/len(ins):.0f} min={min(ins)} max={max(ins)}"
    )
    logger.info(f"output_words mean={sum(outs)/len(outs):.0f}")


if __name__ == "__main__":
    main()
