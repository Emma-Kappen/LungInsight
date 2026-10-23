from pathlib import Path as _Path

_source = _Path(__file__).resolve().parent / "inference" / "05_gradcam.py"
exec(compile(_source.read_bytes(), __file__, "exec"), globals())
