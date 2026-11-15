"""Writes exp16_colab.ipynb (edit this file, not the notebook).  python make_notebook.py <pinned commit>"""
import json
import sys

PIN = sys.argv[1]
HP = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/1908d6afbbead072334abe2965f91bd2709910ab/distractor"
TW = "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/612bc5039a457880d9e7d84c3b0a4cf154b70e4f"
cells = [
("markdown", """# exp16: rank-2 replication on 2WikiMultiHopQA and HotpotQA (Colab GPU)

Runtime: T4 GPU, then *Run all*. A1 from Drive (`exp02_results/full/model`, sha-checked); datasets from pinned Hugging Face
revisions (evaluation files sha-checked). Models and results go to `MyDrive/qualiusp2026/exp16_<commit>`. Code: public repo, pinned."""),
("code", f"""from google.colab import drive
drive.mount('/content/drive')
import os, subprocess, hashlib, pathlib
PIN = '{PIN}'
OUT = f'/content/drive/MyDrive/qualiusp2026/exp16_{{PIN[:7]}}'
A1 = '/content/drive/MyDrive/qualiusp2026/exp02_results/full/model'
pathlib.Path(OUT, 'models').mkdir(parents=True, exist_ok=True); pathlib.Path(OUT, 'results').mkdir(exist_ok=True)
if not os.path.exists('/content/qualiusp2026'):
    !git clone -q https://github.com/arnml/qualiusp2026.git /content/qualiusp2026
!cd /content/qualiusp2026 && git fetch -q && git checkout -q {{PIN}} && git log --oneline -1
assert subprocess.check_output(['git', '-C', '/content/qualiusp2026', 'rev-parse', 'HEAD']).decode().strip() == PIN
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
assert sha(f'{{A1}}/model.safetensors') == '605c986eb9a9a82b10ac43c43c77b2a0f976c04c7ed40f21cc66f2a5b19d950d'
D = '/content/data'; os.makedirs(D, exist_ok=True)
FILES = {{'hotpot_validation.parquet': '{HP}/validation-00000-of-00001.parquet',
         'hotpot_train0.parquet': '{HP}/train-00000-of-00002.parquet', 'hotpot_train1.parquet': '{HP}/train-00001-of-00002.parquet',
         'twowiki_dev.parquet': '{TW}/dev.parquet', 'twowiki_train.parquet': '{TW}/train.parquet'}}
for f, u in FILES.items():
    if not os.path.exists(f'{{D}}/{{f}}'):
        !wget -q -O {{D}}/{{f}} "{{u}}"
assert sha(f'{{D}}/hotpot_validation.parquet') == 'c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6'
assert sha(f'{{D}}/twowiki_dev.parquet') == 'c0d8b60b9026b728fb07ad74c5252a0f188f6942e8ba5c02df4dfa369502ea8d'
print('train file sha:', {{f: sha(f'{{D}}/{{f}}')[:16] for f in FILES if 'train' in f}})
print('inputs verified')
os.environ.update(A1_DIR=A1, DATA_DIR=D, MODELS_DIR=f'{{OUT}}/models', RESULTS_DIR=f'{{OUT}}/results', TOKENIZERS_PARALLELISM='false')
os.chdir('/content/qualiusp2026/experiments/exp16_rank2_replication')
!pip -q install pyarrow
import torch, transformers
assert 'T4' in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
print('torch', torch.__version__, 'transformers', transformers.__version__, torch.cuda.get_device_name(0))"""),
("markdown", "## Train the matched pairs (seed 0)"),
("code", """for ds in ('tw', 'hp'):
    for mode in ('iter', 'ctrl'):
        !python train.py --ds {ds} --mode {mode} --tag {ds}_{mode} 2>&1 | grep -v -e Warning -e 'Token indices' | tail -n 3
    !python train.py --ds {ds} --mode ctrl --tag {ds}_ctrl_s1 --seed 1 2>&1 | grep -v -e Warning -e 'Token indices' | tail -n 3"""),
("markdown", "## Evaluate (tuning and test sets, all configurations), then the pre-registered analysis"),
("code", """C = 'iter_iter ctrl_oneshot iter_oneshot ctrl_iter g_r1 g_r2 g_r4 g_r8 g_rand_r2 g_noise_r2'
for ds in ('tw', 'hp'):
    for sp in ('tune', 'test'):
        !python evaluate.py --ds {ds} --split {ds}_{sp} --only {C} 2>&1 | grep -v -e Warning -e Loading -e 'Token indices'
!python analyze.py"""),
]
nb = {"cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(True), **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells],
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}
open("exp16_colab.ipynb", "w").write(json.dumps(nb, indent=1))
