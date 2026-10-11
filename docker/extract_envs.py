#!/usr/bin/env python3
"""Pull the embedded Project.toml / Manifest.toml out of each Pluto notebook into envs/<name>/."""
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out = root / "envs"
found = 0


def grab(text, name):
    m = re.search(name + r' = """\n(.*?)"""', text, re.S)
    if not m:
        return None
    # Pluto escapes these inside the triple-quoted string
    return m.group(1).replace('\\"', '"').replace("\\$", "$").replace("\\\\", "\\")


for nb in sorted(root.glob("*.jl")):
    text = nb.read_text(encoding="utf-8")
    project = grab(text, "PLUTO_PROJECT_TOML_CONTENTS")
    manifest = grab(text, "PLUTO_MANIFEST_TOML_CONTENTS")
    if project is None or manifest is None:
        print(f"skip {nb.name}: no embedded packages")
        continue
    d = out / nb.stem
    d.mkdir(parents=True, exist_ok=True)
    (d / "Project.toml").write_text(project, encoding="utf-8")
    (d / "Manifest.toml").write_text(manifest, encoding="utf-8")
    print(f"extracted {nb.name} -> {d}")
    found += 1

# The Dockerfile does COPY envs/, so the folder must exist even if empty
out.mkdir(exist_ok=True)
print(f"{found} notebook environment(s) extracted")
