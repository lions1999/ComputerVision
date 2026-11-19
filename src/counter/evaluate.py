"""Compare predicted line crossings with a manual ground truth: match events one-to-one, report the metrics."""
import argparse
import csv
from collections import Counter


def load(path):
    with open(path, newline="") as f:
        return [{"t": float(r["time_s"]), "cls": r["class"], "dir": r["direction"]} for r in csv.DictReader(f)]


def match(gt, pred, tol):
    """One-to-one pairs (i_gt, j_pred): same direction, |dt| <= tol seconds; same-class pairs first, then closest.
    Class goes first because the ground truth has no position: two vehicles crossing in different lanes a frame
    apart are ambiguous by time alone, and a time-only match swapped a car and a truck.
    shortcut: greedy, not globally optimal; fine while same-direction crossings are further apart than tol."""
    cand = sorted((g["cls"] != p["cls"], abs(g["t"] - p["t"]), i, j) for i, g in enumerate(gt)
                  for j, p in enumerate(pred) if g["dir"] == p["dir"] and abs(g["t"] - p["t"]) <= tol)
    used_g, used_p, pairs = set(), set(), []
    for _, _, i, j in cand:
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            pairs.append((i, j))
    return sorted(pairs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("gt", help="csv from annotate.py")
    p.add_argument("pred", help="csv from run.py")
    p.add_argument("--tol", type=float, default=1.0, help="max time difference for a match, seconds")
    a = p.parse_args()

    gt, pred = load(a.gt), load(a.pred)
    pairs = match(gt, pred, a.tol)
    tp, fn, fp = len(pairs), len(gt) - len(pairs), len(pred) - len(pairs)
    prec = tp / (tp + fp) if pred else 0.0
    rec = tp / (tp + fn) if gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    for d in sorted({e["dir"] for e in gt + pred}):
        ng, np_ = sum(e["dir"] == d for e in gt), sum(e["dir"] == d for e in pred)
        print(f"{d:<18} gt {ng:>3}  pred {np_:>3}  count error {np_ - ng:+d} ({(np_ - ng) / ng:+.0%})" if ng
              else f"{d:<18} gt   0  pred {np_:>3}")
    print(f"matched (tol {a.tol}s): TP {tp}  FN {fn}  FP {fp}")
    print(f"precision {prec:.3f}  recall {rec:.3f}  F1 {f1:.3f}")

    conf = Counter((gt[i]["cls"], pred[j]["cls"]) for i, j in pairs)
    print(f"class accuracy on matched: {sum(n for (g, q), n in conf.items() if g == q)}/{tp}")
    for (g, q), n in sorted(conf.items()):
        print(f"  gt {g:<10} -> pred {q:<10} {n}")

    mg, mp = {i for i, _ in pairs}, {j for _, j in pairs}
    for i, e in enumerate(gt):
        if i not in mg:
            print(f"  missed (FN): t {e['t']:.2f}s {e['cls']} {e['dir']}")
    for j, e in enumerate(pred):
        if j not in mp:
            print(f"  extra  (FP): t {e['t']:.2f}s {e['cls']} {e['dir']}")
    for i, j in pairs:
        if gt[i]["cls"] != pred[j]["cls"]:
            print(f"  class: t {gt[i]['t']:.2f}s gt {gt[i]['cls']} -> pred {pred[j]['cls']}")


if __name__ == "__main__":
    main()
