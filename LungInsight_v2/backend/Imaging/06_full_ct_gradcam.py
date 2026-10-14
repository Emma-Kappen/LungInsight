from pathlib import Path as _Path

_source = _Path(__file__).resolve().parent / "visualization" / "06_full_ct_gradcam.py"
exec(compile(_source.read_bytes(), __file__, "exec"), globals())
