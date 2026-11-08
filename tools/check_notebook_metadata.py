"""Audit Python notebook metadata and syntax without executing any notebook cells."""
import argparse
import ast
import json
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

LANGUAGE = dict(name="python", codemirror_mode=dict(name="ipython", version=3),
                pygments_lexer="ipython3", mimetype="text/x-python", file_extension=".py",
                nbconvert_exporter="python")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = sorted([*root.glob("notebooks/**/*.ipynb"), *root.parent.glob("epkd-vision/kaggle/*.ipynb")])
    transform = TransformerManager()
    changed, code_cells, failures = [], 0, []
    for path in paths:
        nb = json.loads(path.read_text(encoding="utf-8"))
        original_cells = json.dumps([{k:v for k,v in c.items() if k != "id"} for c in nb["cells"]], sort_keys=True)
        meta = nb.setdefault("metadata", {})
        correct = (all(meta.get("language_info", {}).get(k) == v for k,v in LANGUAGE.items()) and
                   meta.get("kernelspec", {}).get("language") == "python")
        format_fix = nb.get("nbformat_minor", 0) < 5 or any("id" not in c for c in nb["cells"])
        if not correct or format_fix:
            changed.append(path)
            if args.fix:
                meta.setdefault("language_info", {}).update(LANGUAGE)
                meta["kernelspec"] = dict(name="python3", display_name="Python 3", language="python")
                nb["nbformat_minor"] = 5
                for i, c in enumerate(nb["cells"]):
                    c.setdefault("id", f"notebook-cell-{i}")
                # Only editor metadata/cell IDs change; all source/outputs/cell types stay intact.
                assert original_cells == json.dumps([{k:v for k,v in c.items() if k != "id"} for c in nb["cells"]], sort_keys=True)
                path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        try:
            nbformat.validate(nbformat.from_dict(nb))
            for i, cell in enumerate(nb["cells"]):
                if cell["cell_type"] == "code":
                    code = "".join(cell["source"])
                    compile(transform.transform_cell(code), f"{path.name}:cell{i}", "exec",
                            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                    code_cells += 1
        except Exception as error:
            failures.append(dict(notebook=str(path.relative_to(root.parent)), error=str(error)))
    report = dict(notebooks=len(paths), code_cells=code_cells,
                  metadata_fixed=len(changed) if args.fix else 0,
                  remaining_metadata_issues=0 if args.fix else len(changed),
                  failures=failures, executed_cells=0)
    (root/"NOTEBOOK_METADATA_CHECKS.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if failures or (changed and not args.fix):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
