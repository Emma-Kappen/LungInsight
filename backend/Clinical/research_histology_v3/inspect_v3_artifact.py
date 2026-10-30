import sys
import joblib
import json

path = sys.argv[1]
obj = joblib.load(path)

print("Top-level type:", type(obj).__name__)

if isinstance(obj, dict):
    print("Top-level keys:", list(obj.keys()))

    for key, value in obj.items():
        if key == "models" and isinstance(value, dict):
            print("\nMODEL ENTRIES:")
            for model_name, entry in value.items():
                print("\nModel:", model_name)
                print("Entry type:", type(entry).__name__)

                if isinstance(entry, dict):
                    print("Entry keys:", list(entry.keys()))
                    print("Features:", entry.get("features"))
                    print("Target:", entry.get("target"))
                    print("Model class:", type(entry.get("pipeline")).__name__)

                    pipeline = entry.get("pipeline")
                    if pipeline is not None:
                        print("Pipeline steps:", getattr(pipeline, "named_steps", {}).keys())

        elif key != "models":
            try:
                print(f"\n{key}:", json.dumps(value, default=str)[:5000])
            except Exception:
                print(f"\n{key}:", repr(value)[:5000])
else:
    print(repr(obj)[:10000])
