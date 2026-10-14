from pathlib import Path as _Path

_source = _Path(__file__).resolve().parent / "validation" / "attest_heart_mask_review.py"
exec(compile(_source.read_bytes(), __file__, "exec"), globals())
