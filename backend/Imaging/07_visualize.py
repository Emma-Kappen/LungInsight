from pathlib import Path as _Path

_source = _Path(__file__).resolve().parent / "visualization" / "07_visualize.py"
exec(compile(_source.read_bytes(), __file__, "exec"), globals())
