```python
# Hypothesis: the lexical-bias diagnosis can be evaluated from cached baseline activations without another model pass.
print("Diagnostic state:", {k: k in globals() for k in [
    "drows", "dscores", "dgold", "dlength", "doverlap", "groups_d",
    "base_d", "weak_gold", "neuron_stats"]})
print("Baseline score range:", float(np.min(dscores)), float(np.max(dscores)))
if "base_d" in globals():
    print("Discovery baseline:", {k: v for k, v in base_d.items() if not isinstance(v, np.ndarray)})
if "neuron_stats" in globals():
    print("Completed layer summaries:", len(neuron_stats))
```
