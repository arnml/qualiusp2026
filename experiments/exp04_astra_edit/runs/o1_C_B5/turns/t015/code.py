# Hypothesis: for items whose layer-5 input is ~unchanged (cos>0.99 at L4 output), the layer-5 thinning alone changes final CLS/score only slightly.
c4=cos(Ce[:,4],C[:,5]); c5=cos(Ce[:,5],C[:,6]); ds=se-so
for th in [0.995,0.99,0.98]:
    m=c4>th
    print(f"items with L4-out cos>{th}: {m.sum():4d}  final cos mean {c5[m].mean():.3f} min {c5[m].min():.3f}  |score diff| mean {np.abs(ds[m]).mean():.2f} (score std {so.std():.2f})  mean diff {ds[m].mean():+.2f}")
print("all items: |score diff| mean %.2f, mean diff %+.2f"%(np.abs(ds).mean(), ds.mean()))
# within-question rank fidelity restricted to questions where all paragraphs have L4-out cos>0.99
qs_ok=[qi for qi in range(200) if all(c4[idx[(qi,p)]]>0.99 for p in range(len(practice[qi]['paragraphs'])))]
print("questions fully intact before L5:", len(qs_ok))
