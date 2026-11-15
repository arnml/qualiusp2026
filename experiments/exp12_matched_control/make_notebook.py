"""Writes exp12_colab.ipynb (edit this file, not the notebook).  python make_notebook.py <pinned commit>"""
import json
import sys

PIN = sys.argv[1]

cells = [
("markdown", """# exp12: pair-matched control (Colab GPU)

Runtime → Change runtime type → **T4 GPU**, then *Run all*. A1 is read from Drive (`MyDrive/qualiusp2026/exp02_results/full/model`); MuSiQue is downloaded from the official release; both are hash-checked.
Models and results are written to Drive (`MyDrive/qualiusp2026/exp12_<commit>`). Training checkpoints every 500 steps, so after a
disconnect just *Run all* again: finished models and evaluation configs are skipped. Code comes from the public repo (PROTOCOL.md)."""),
("code", """from google.colab import drive
drive.mount('/content/drive')
import os, pathlib, subprocess
PIN = '__PIN__'  # registration commit (PROTOCOL: pinned code, fresh output folder)
OUT = f'/content/drive/MyDrive/qualiusp2026/exp12_{PIN[:7]}'
A1 = '/content/drive/MyDrive/qualiusp2026/exp02_results/full/model'  # exp02's A1 (sha256 checked below)
pathlib.Path(OUT, 'models').mkdir(parents=True, exist_ok=True); pathlib.Path(OUT, 'results').mkdir(exist_ok=True)
if not os.path.exists('/content/qualiusp2026'):
    !git clone -q https://github.com/arnml/qualiusp2026.git /content/qualiusp2026
!cd /content/qualiusp2026 && git fetch -q && git checkout -q {PIN} && git log --oneline -1
assert subprocess.check_output(['git', '-C', '/content/qualiusp2026', 'rev-parse', 'HEAD']).decode().strip() == PIN
# Inputs (deviation 1): A1 from exp02's Drive folder; MuSiQue from the official release (StonyBrookNLP/musique).
# Both are checked against sha256 hashes of the local files used in exp08-exp11.
import hashlib, json
def sha(path): return hashlib.sha256(open(path, 'rb').read()).hexdigest()
assert sha(f'{A1}/model.safetensors') == '605c986eb9a9a82b10ac43c43c77b2a0f976c04c7ed40f21cc66f2a5b19d950d'
D = '/content/qualiusp2026/experiments/exp08_musique/data'
if os.path.islink(D): os.unlink(D)  # left by the first attempt (Drive inputs from another account)
if not os.path.exists(f'{D}/musique_ans_v1.0_dev.jsonl'):
    !pip -q install gdown && cd /content && gdown -q 1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h -O musique.zip && unzip -q -o musique.zip -d musique_raw
    os.makedirs(D, exist_ok=True)
    src = subprocess.check_output('find /content/musique_raw -name musique_ans_v1.0_dev.jsonl', shell=True).decode().split()[0].rsplit('/', 1)[0]
    !cp {src}/musique_ans_v1.0_dev.jsonl {D}/
    ids = set(json.load(open('/content/qualiusp2026/experiments/exp12_matched_control/split_ids.json'))['train'])
    with open(f'{src}/musique_ans_v1.0_train.jsonl') as f, open(f'{D}/musique_ans_v1.0_train.jsonl', 'w') as g:
        for line in f:
            if json.loads(line)['id'] in ids: g.write(line)
assert sha(f'{D}/musique_ans_v1.0_dev.jsonl') == '15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b', 'dev file differs'
assert sha(f'{D}/musique_ans_v1.0_train.jsonl') == '6259adbcb10bc4665e9f963e09b2fe549b6c0e3082c9d511af48544826ae75f0', 'train subset differs'
print('inputs verified')
os.environ.update(A1_DIR=A1, MODELS_DIR=f'{OUT}/models', RESULTS_DIR=f'{OUT}/results', TOKENIZERS_PARALLELISM='false')
os.chdir('/content/qualiusp2026/experiments/exp12_matched_control')
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install tiktoken
!git rev-parse HEAD > {OUT}/results/commit.txt"""),
("markdown", "## Train (PROTOCOL: M1-ctrl seed 0 = primary; seeds 1–2 and GPU M1-iter = replication)"),
("code", """RUNS = [('ctrl', 'm1_ctrl', 0), ('iter', 'm1_iter_s0', 0), ('ctrl', 'm1_ctrl_s1', 1), ('iter', 'm1_iter_s1', 1),
        ('ctrl', 'm1_ctrl_s2', 2), ('iter', 'm1_iter_s2', 2)]
for mode, tag, seed in RUNS:
    if os.path.exists(f'{OUT}/models/{tag}/model/model.safetensors'):
        print('done:', tag); continue
    !python train.py --mode {mode} --tag {tag} --seed {seed} --n-questions 1000 2>&1 | grep -v Warning | tail -n 4"""),
("markdown", "## Evaluate"),
("code", """!python evaluate.py --split test --only iter_iter ctrl_oneshot iter_oneshot ctrl_iter iter_iter_s1 ctrl_oneshot_s1 iter_iter_s2 ctrl_oneshot_s2 2>&1 | grep -v -e Warning -e Loading
!python evaluate.py --split dev300 --only ctrl_iter iter_iter g_r1 g_r2 g_r4 g_r8 g_rand_r2 2>&1 | grep -v -e Warning -e Loading"""),
("markdown", "## Secondary: r* from dev300 (analyze.py), then test for r* only; then all pre-registered tests"),
("code", """rstar = !python analyze.py --rstar
rstar = rstar[-1].strip(); print('r* =', rstar)
if rstar != 'none':
    !python evaluate.py --split test --only g_r{rstar} 2>&1 | grep -v -e Warning -e Loading
!python analyze.py"""),
]
nb = {"cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(True), **({"outputs": [], "execution_count": None} if t == "code" else {})}
                for t, s in cells],
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}
txt = json.dumps(nb, indent=1).replace("__PIN__", PIN)
open("exp12_colab.ipynb", "w").write(txt)
