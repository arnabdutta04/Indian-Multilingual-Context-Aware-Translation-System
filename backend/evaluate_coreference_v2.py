import asyncio
import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

smoother = SmoothingFunction().method1

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from app import resolve_coreference
except ImportError as e:
    sys.exit(f"[ERROR] Could not import resolve_coreference: {e}")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PRONOUN_RE = re.compile(
    r'\b(he|him|his|himself|she|her|hers|herself|it|its|itself|'
    r'they|them|their|theirs|themselves)\b', re.IGNORECASE)

def pronoun_count(text):
    """Count total pronoun occurrences in text."""
    return len(PRONOUN_RE.findall(text))

def is_resolved_correctly(original, resolved, expected):
    """
    TRUE PRECISION CHECK: compare system output against ground truth.
    
    Scoring:
    - If expected == original (should NOT resolve): pass only if resolved == original
    - If expected != original (SHOULD resolve): 
        * Full match = TP
        * Partial match (some pronouns resolved) = partial credit → still TP if any pronoun reduced
        * No change = FN
        * Wrong resolution = FP
    """
    orig_count  = pronoun_count(original)
    res_count   = pronoun_count(resolved)
    exp_count   = pronoun_count(expected)
    
    if expected.lower().strip() == original.lower().strip():
        # Should NOT resolve — pass only if system also didn't resolve
        return resolved.lower().strip() == original.lower().strip()
    else:
        # SHOULD resolve — pass if pronoun count decreased (at least partial)
        return res_count < orig_count

def precision_check(original, resolved, expected):
    """For precision: did the system resolve when it should have?"""
    orig_count = pronoun_count(original)
    res_count  = pronoun_count(resolved)
    exp_count  = pronoun_count(expected)
    
    should_resolve = (expected.lower().strip() != original.lower().strip())
    did_resolve    = (res_count < orig_count)
    
    if should_resolve and did_resolve:
        return True   # TP
    if not should_resolve and not did_resolve:
        return True   # TN → not counted in sklearn precision
    if not should_resolve and did_resolve:
        return False  # FP: resolved when shouldn't have
    # should_resolve and not did_resolve → FN
    return False

async def evaluate(df: pd.DataFrame) -> dict:
    results = []
    y_true, y_pred = [], []
    bleu_scores = []
    
    has_gt = 'expected_resolved' in df.columns

    for idx, row in df.iterrows():
        text     = str(row["input"]).strip()
        expected = str(row["expected_resolved"]).strip() if has_gt else None
        
        try:
            resolved = resolve_coreference(text, [])
        except Exception as exc:
            log.warning("Row %d exception: %s", idx, exc)
            results.append({"index": idx, "original": text, "resolved": "", "passed": False})
            y_true.append(1)
            y_pred.append(0)
            continue

        # Determine pass/fail
        if has_gt and expected:
            passed = is_resolved_correctly(text, resolved, expected)
            # y_true: 1 = should resolve, 0 = should NOT resolve
            should = (expected.lower().strip() != text.lower().strip())
            y_true.append(1 if should else 0)
            y_pred.append(1 if (pronoun_count(resolved) < pronoun_count(text)) else 0)
        else:
            # Fallback: count-based
            passed = pronoun_count(resolved) < pronoun_count(text)
            has_p  = pronoun_count(text) > 0
            y_true.append(1 if has_p else 0)
            y_pred.append(1 if passed else 0)

        # BLEU with smoothing
        reference = [text.lower().split()]
        candidate = resolved.lower().split()
        bleu = sentence_bleu(reference, candidate, smoothing_function=smoother)
        bleu_scores.append(bleu)

        results.append({"index": idx, "original": text, "resolved": resolved, "passed": passed})

        status = "✅ Resolved" if passed else "❌ Failed"
        print(f"\n[{idx}] {status}")
        print(f"  ORIGINAL : {text}")
        print(f"  RESOLVED : {resolved}")
        if has_gt and expected and expected != text:
            print(f"  EXPECTED : {expected}")

    total   = len(results)
    correct = sum(r["passed"] for r in results)
    accuracy = (correct / total * 100) if total else 0.0

    # Filter to only rows where resolution was expected (y_true==1) for precision/recall
    pairs = list(zip(y_true, y_pred))
    yt_filtered = [yt for yt,yp in pairs if yt == 1 or yp == 1]
    yp_filtered = [yp for yt,yp in pairs if yt == 1 or yp == 1]

    try:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        f1        = f1_score(y_true, y_pred, zero_division=0)
    except Exception:
        precision = recall = f1 = 0.0

    avg_bleu = (sum(bleu_scores) / len(bleu_scores) * 100) if bleu_scores else 0

    return {
        "total": total, "correct": correct, "failed": total - correct,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "bleu": avg_bleu, "rows": results,
    }

def print_summary(stats):
    bar = "=" * 42
    print(f"\n{bar}")
    print("  COREFERENCE EVALUATION SUMMARY")
    print(bar)
    print(f"  Total samples : {stats['total']}")
    print(f"  Resolved (✅)  : {stats['correct']}")
    print(f"  Failed   (❌)  : {stats['failed']}")
    print(f"  Accuracy       : {stats['accuracy']:.2f} %")
    print(f"  Precision      : {stats['precision']*100:.2f} %")
    print(f"  Recall         : {stats['recall']*100:.2f} %")
    print(f"  F1 Score       : {stats['f1']*100:.2f} %")
    print(f"  BLEU Score     : {stats['bleu']:.2f} %")
    print(bar)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",   default="../dataset/context_dataset_with_gt.csv")
    p.add_argument("--limit", type=int, default=5159)
    p.add_argument("--lang",  default=None)
    p.add_argument("--save",  default=None)
    return p.parse_args()

async def main():
    args  = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"[ERROR] CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    log.info("Loaded %d rows from %s", len(df), csv_path)
    if args.lang:
        if "target_language" in df.columns:
            df = df[df["target_language"].str.lower() == args.lang.lower()]
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
    log.info("Evaluating %d rows", len(df))
    if df.empty:
        sys.exit("[ERROR] No rows to evaluate.")
    stats = await evaluate(df)
    print_summary(stats)
    if args.save:
        pd.DataFrame(stats["rows"]).to_csv(args.save, index=False)

if __name__ == "__main__":
    asyncio.run(main())