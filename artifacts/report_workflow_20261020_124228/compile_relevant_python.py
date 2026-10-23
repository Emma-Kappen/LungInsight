import py_compile, sys
from pathlib import Path
errors=[]
for item in sys.argv[1:]:
    try:
        py_compile.compile(item, doraise=True)
    except Exception as exc:
        errors.append(f"{item}: {exc}")
if errors:
    print("\n".join(errors))
    raise SystemExit(1)
print(f"Compiled {len(sys.argv)-1} Python files successfully.")
