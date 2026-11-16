"""Writes exp13_colab.ipynb (edit this file, not the notebook).  python make_notebook.py <pinned commit>"""

import json
import sys

PIN = sys.argv[1]

cells = [
    (
        "markdown",
        """# exp13: 4-hop confirmation and cascade (Colab GPU)

Runtime → Change runtime type → **T4 GPU**, then *Run all*. A1 is read from Drive (`MyDrive/qualiusp2026/exp02_results/full/model`); MuSiQue is downloaded from the official release; both are hash-checked.
Models and results are written to Drive (`MyDrive/qualiusp2026/exp12_<commit>`). Training checkpoints every 500 steps, so after a
disconnect just *Run all* again: finished models and evaluation configs are skipped. Code comes from the public repo (PROTOCOL.md). Models: exp12's frozen GPU models on Drive.""",
    ),
    (
        "code",
        """from google.colab import drive
drive.mount('/content/drive')
import os, pathlib, subprocess
PIN = '__PIN__'  # registration commit (PROTOCOL: pinned code, fresh output folder)
OUT = f'/content/drive/MyDrive/qualiusp2026/exp13_{PIN[:7]}'
M12 = '/content/drive/MyDrive/qualiusp2026/exp12_975371a/models'  # frozen exp12 models
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
    ids = json.load(open('/content/qualiusp2026/experiments/exp13_cascade/split_ids.json'))
    ids = set(ids['router_train']) | set(ids['f4_train'])
    with open(f'{src}/musique_ans_v1.0_train.jsonl') as f, open(f'{D}/musique_ans_v1.0_train.jsonl', 'w') as g:
        for line in f:
            if json.loads(line)['id'] in ids: g.write(line)
assert sha(f'{D}/musique_ans_v1.0_dev.jsonl') == '15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b', 'dev file differs'
assert sha(f'{D}/musique_ans_v1.0_train.jsonl') == 'c47ce51185d2447ebede240dc31acdd6acc8822219cbaab4d540f6f2dd1e65fe', 'train subset differs'
print('inputs verified')
os.environ.update(A1_DIR=A1, MODELS_DIR=M12, RESULTS_DIR=f'{OUT}/results', TOKENIZERS_PARALLELISM='false')
os.chdir('/content/qualiusp2026/experiments/exp13_cascade')
for t in ('m1_iter_s0', 'm1_ctrl', 'm1_iter_s1', 'm1_ctrl_s1', 'm1_iter_s2', 'm1_ctrl_s2'):
    print(t, sha(f'{M12}/{t}/model/model.safetensors')[:16])
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install tiktoken
!git rev-parse HEAD > {OUT}/results/commit.txt""",
    ),
    ("markdown", "## Rank (frozen exp12 models), then the pre-registered analysis"),
    (
        "code",
        """!python rank.py --split f4_dev 2>&1 | grep -v -e Warning -e Loading
!python rank.py --split f4_train 2>&1 | grep -v -e Warning -e Loading
!python rank.py --split router_train --only ctrl_oneshot 2>&1 | grep -v -e Warning -e Loading
!python rank.py --split dev_fresh --only ctrl_oneshot iter_iter 2>&1 | grep -v -e Warning -e Loading
!python analyze.py""",
    ),
]
nb = {
    "cells": [
        {
            "cell_type": t,
            "metadata": {},
            "source": s.splitlines(True),
            **({"outputs": [], "execution_count": None} if t == "code" else {}),
        }
        for t, s in cells
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
txt = json.dumps(nb, indent=1).replace("__PIN__", PIN)
open("exp13_colab.ipynb", "w").write(txt)
