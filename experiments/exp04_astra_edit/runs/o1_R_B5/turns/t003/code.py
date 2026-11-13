# Hypothesis: stdout was lost/truncated last turn; re-print a compact inventory (variables from last turn should persist).
import sys
summary = {
  "WORK": str(WORK), "notes_exists": NOTES.exists(), "n_tensors": len(NAMES),
  "layer5_pooler": [n for n in NAMES if "layer.5" in n or "pooler" in n],
  "n_q": len(practice), "types": dict(collections.Counter(q["type"] for q in practice)),
  "par_counts": dict(collections.Counter(len(q["paragraphs"]) for q in practice)),
  "total_pars": sum(len(q["paragraphs"]) for q in practice),
  "tok_len_mean": float(np.mean(lens)), "tok_len_max": int(max(lens)),
  "edited": str(edited()),
}
print(json.dumps(summary, indent=1)); sys.stdout.flush()
q = practice[0]; print("Q:", q["question"], "| A:", q["answer"], "|", q["type"])
for p, g in zip(q["paragraphs"], q["gold"]): print(" ", g, p[:100].replace("\n", " "))
sys.stdout.flush()
