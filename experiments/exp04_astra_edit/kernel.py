"""Persistent Python kernel for the editor. Protocol: one JSON line in, one JSON line out.

in : {"code": "..."}      out: {"output": "..."}
State (variables) persists between messages. Started and restarted by harness.py.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback

_real_stdout = sys.stdout
ns: dict = {"__name__": "__kernel__"}
exec("from kernel_api import *\nimport numpy as np, torch, scipy, sklearn, pandas as pd\n", ns)
_real_stdout.write(json.dumps({"output": "kernel ready"}) + "\n")
_real_stdout.flush()

for line in sys.stdin:
    msg = json.loads(line)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            exec(compile(msg["code"], "<cell>", "exec"), ns)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    _real_stdout.write(json.dumps({"output": buf.getvalue()}) + "\n")
    _real_stdout.flush()
