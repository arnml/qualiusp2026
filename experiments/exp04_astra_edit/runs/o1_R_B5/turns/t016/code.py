# Hypothesis: (prep, no pass) per-head edit = scale the 32 input columns of layer-5 attention.output.dense belonging to a head; check sizes/norms.
Wao = load_original("bert.encoder.layer.5.attention.output.dense.weight"); Wv5 = load_original("bert.encoder.layer.5.attention.self.value.weight")
print("attn.output.dense", tuple(Wao.shape), "| value", tuple(Wv5.shape))
hn_ = [float(Wao[:, h*32:(h+1)*32].norm()) for h in range(12)]
print("L5 per-head output-column norms:", np.round(hn_, 2))
# weight-only alignment: how strongly can each head's output subspace write onto the readout axis r (via LN gamma)?
gA5 = load_original("bert.encoder.layer.5.attention.output.LayerNorm.weight").numpy()
align = [np.linalg.norm((Wao[:, h*32:(h+1)*32].numpy() * gA5[:, None]).T @ r_) / np.linalg.norm(Wao[:, h*32:(h+1)*32].numpy() * gA5[:, None]) for h in range(12)]
print("L5 per-head |proj of output subspace on r| / norm:", np.round(align, 3))
def head_scale_edit(layer, scales):
    """scales: {head: factor}; multiplies the head's 32 columns of attention.output.dense.weight (bias untouched)."""
    W = load_original(f"bert.encoder.layer.{layer}.attention.output.dense.weight").clone()
    for h, f in scales.items(): W[:, h*32:(h+1)*32] *= f
    return W
Wt = head_scale_edit(5, {0: 2.0, 6: 2.0}); print("changed entries for 2 heads:", int((Wt != Wao).sum()))
