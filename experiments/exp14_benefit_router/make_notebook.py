"""Writes exp14_colab.ipynb (edit this file, not the notebook).  python make_notebook.py <pinned commit>
Deviation 1 (PROTOCOL.md): the two frozen models are re-trained on the second Google account with exp12's code at 975371a,
checked against the registered hashes and exp12's test recall, and router_train / dev_fresh are re-ranked."""
import json
import sys

PIN = sys.argv[1]
E12_PIN = "975371a6873153deb5a0a4c33c3644579b39dd96"  # exp12 registration commit (training code)

cells = [
("markdown", """# exp14: benefit router cascade (Colab GPU)

Runtime → Change runtime type → **T4 GPU**, then *Run all*. A1 is read from Drive (either Google account, sha-checked); MuSiQue is downloaded from the official release and hash-checked.
**Deviation 1:** M1-ctrl and M1-iter (seed 0) are re-trained with exp12's code at its registration commit, compared with the registered hashes and with exp12's test recall, and `router_train` / `dev_fresh` are re-ranked.
Everything is written to Drive (`MyDrive/qualiusp2026/exp14_<commit>`); after a disconnect, *Run all* again: finished models and rankings are skipped."""),
("code", """from google.colab import drive
drive.mount('/content/drive')
import os, pathlib, subprocess, hashlib, json, shutil
PIN = '__PIN__'  # registration commit (pinned code, fresh output folder)
E12_PIN = '__E12__'  # exp12's training code
OUT = f'/content/drive/MyDrive/qualiusp2026/exp14_{PIN[:7]}'
for d in ('models', 'results', 'fidelity'): pathlib.Path(OUT, d).mkdir(parents=True, exist_ok=True)
def sha(path): return hashlib.sha256(open(path, 'rb').read()).hexdigest()
A1_SHA = '605c986eb9a9a82b10ac43c43c77b2a0f976c04c7ed40f21cc66f2a5b19d950d'
CANDS = ['/content/drive/MyDrive/qualiusp2026/exp12_inputs/a1', '/content/drive/MyDrive/qualiusp2026/exp02_results/full/model']
A1 = next(c for c in CANDS if os.path.exists(f'{c}/model.safetensors'))
assert sha(f'{A1}/model.safetensors') == A1_SHA; print('A1:', A1)
for path, commit in (('/content/qualiusp2026', PIN), ('/content/e12', E12_PIN)):
    if not os.path.exists(path):
        subprocess.run(['git', 'clone', '-q', 'https://github.com/arnml/qualiusp2026.git', path], check=True)
    subprocess.run(['git', '-C', path, 'checkout', '-q', commit], check=True)
    assert subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD']).decode().strip() == commit
if not os.path.exists('/content/musique_raw'):
    !pip -q install gdown && cd /content && gdown -q 1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h -O musique.zip && unzip -q -o musique.zip -d musique_raw
src = subprocess.check_output('find /content/musique_raw -name musique_ans_v1.0_dev.jsonl', shell=True).decode().split()[0].rsplit('/', 1)[0]
def subset(root, split_file, keys, want):
    D = f'{root}/experiments/exp08_musique/data'
    if os.path.islink(D): os.unlink(D)
    os.makedirs(D, exist_ok=True)
    if not os.path.exists(f'{D}/musique_ans_v1.0_train.jsonl'):
        shutil.copy(f'{src}/musique_ans_v1.0_dev.jsonl', D)
        sp = json.load(open(split_file)); ids = set().union(*(sp[k] for k in keys))
        with open(f'{src}/musique_ans_v1.0_train.jsonl') as f, open(f'{D}/musique_ans_v1.0_train.jsonl', 'w') as g:
            for line in f:
                if json.loads(line)['id'] in ids: g.write(line)
    assert sha(f'{D}/musique_ans_v1.0_dev.jsonl') == '15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b', 'dev file differs'
    assert sha(f'{D}/musique_ans_v1.0_train.jsonl') == want, f'train subset differs: {root}'
subset('/content/qualiusp2026', '/content/qualiusp2026/experiments/exp14_benefit_router/split_ids.json',
       ('b_train', 'b_tune', 'b_eval', 'router_train'), '4093c17f2266fcf3fb0e3647aff18d892fc28edecea62f6f0609cd76d7f2c49b')
subset('/content/e12', '/content/e12/experiments/exp12_matched_control/split_ids.json',
       ('train',), '6259adbcb10bc4665e9f963e09b2fe549b6c0e3082c9d511af48544826ae75f0')
print('inputs verified')
os.environ.update(A1_DIR=A1, MODELS_DIR=f'{OUT}/models', TOKENIZERS_PARALLELISM='false')
import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, torch.cuda.get_device_name(0))
assert 'T4' in torch.cuda.get_device_name(0)
!pip -q install tiktoken
!git -C /content/qualiusp2026 rev-parse HEAD > {OUT}/results/commit.txt"""),
("markdown", "## Deviation 1: re-train M1-ctrl and M1-iter (seed 0) with exp12's code, then check identity and fidelity"),
("code", """os.chdir('/content/e12/experiments/exp12_matched_control')
for mode, tag in (('ctrl', 'm1_ctrl'), ('iter', 'm1_iter_s0')):
    if os.path.exists(f'{OUT}/models/{tag}/model/model.safetensors'):
        print('done:', tag); continue
    !set -o pipefail; python train.py --mode {mode} --tag {tag} --seed 0 --n-questions 1000 2>&1 | grep -v Warning | tail -n 4
    assert os.path.exists(f'{OUT}/models/{tag}/model/model.safetensors'), f'training failed: {tag}'
H = {t: sha(f'{OUT}/models/{t}/model/model.safetensors') for t in ('m1_iter_s0', 'm1_ctrl')}
REG = {'m1_iter_s0': '2c91c40f1310ca02', 'm1_ctrl': 'ed87244f462412f9'}
same = all(H[t][:16] == REG[t] for t in REG)
print('hashes', {t: h[:16] for t, h in H.items()}, '| registered models reproduced:', same)
json.dump(H, open(f'{OUT}/results/model_sha.json', 'w'))  # always the models actually used; analyze.py asserts them on every ranking
# Fidelity (descriptive): exp12's test set with exp12's evaluate.py; exp12 reported iter_iter 0.7475, ctrl_oneshot 0.7075
os.environ['RESULTS_DIR'] = f'{OUT}/fidelity'
!python evaluate.py --split test --only iter_iter ctrl_oneshot 2>&1 | grep -v -e Warning -e Loading"""),
("markdown", "## Rank with the (re-trained) models, then the pre-registered analysis"),
("code", """os.environ['RESULTS_DIR'] = f'{OUT}/results'
os.chdir('/content/qualiusp2026/experiments/exp14_benefit_router')
for sp in ('b_train', 'b_tune', 'b_eval', 'router_train', 'dev_fresh'):
    !python rank.py --split {sp} --only ctrl_oneshot iter_iter 2>&1 | grep -v -e Warning -e Loading -e 'Token indices'
!python rank.py --split b_eval --only iter_oneshot ctrl_iter 2>&1 | grep -v -e Warning -e Loading -e 'Token indices'
!python analyze.py"""),
]
nb = {"cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(True), **({"outputs": [], "execution_count": None} if t == "code" else {})}
                for t, s in cells],
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}
txt = json.dumps(nb, indent=1).replace("__PIN__", PIN).replace("__E12__", E12_PIN)
open("exp14_colab.ipynb", "w").write(txt)
