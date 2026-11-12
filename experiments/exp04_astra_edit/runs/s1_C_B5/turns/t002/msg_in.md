OUTPUT:
```
Traceback (most recent call last):
  File "C:\code\qualiusp2026\experiments\exp04_astra_edit\kernel.py", line 26, in <module>
    exec(compile(msg["code"], "<cell>", "exec"), ns)
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<cell>", line 8, in <module>
  File "C:\Users\devar\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\Lib\pathlib\_local.py", line 555, in write_text
    return PathBase.write_text(self, data, encoding, errors, newline)
           ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\devar\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\Lib\pathlib\_abc.py", line 651, in write_text
    with self.open(mode='w', encoding=encoding, errors=errors, newline=newline) as f:
         ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\devar\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\Lib\pathlib\_local.py", line 537, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '\\work\\notes.md'

```
Passes left: 5/5. Turn 1/40.