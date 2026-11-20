#!/usr/bin/env python3
"""Bundle JL entry points and local Python dependencies, never datasets/results."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent


def dependencies():
    found = set()
    pending = list(ROOT.glob('jl_*.py'))
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        tree = ast.parse(path.read_text(encoding='utf-8'))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parent = path.parent
                    for _ in range(node.level - 1):
                        parent = parent.parent
                    prefix = '.'.join(parent.relative_to(ROOT).parts)
                    module = '.'.join(x for x in (prefix, node.module) if x)
                else:
                    module = node.module or ''
                modules.append(module)
                modules.extend(module + '.' + alias.name for alias in node.names)
        for module in modules:
            parts = module.split('.')
            for n in range(1, len(parts) + 1):
                stem = ROOT.joinpath(*parts[:n])
                for candidate in (stem.with_suffix('.py'), stem / '__init__.py'):
                    if candidate.is_file() and candidate not in found:
                        pending.append(candidate)
    return sorted(found)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive')
    args = p.parse_args()
    files = dependencies() + list(ROOT.glob('jl_*.txt'))
    manifest = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    with tarfile.open(args.archive, 'w:gz') as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
        import io
        payload = json.dumps(manifest, indent=2).encode()
        entry = tarfile.TarInfo('jl_release_manifest.json')
        entry.size = len(payload)
        archive.addfile(entry, io.BytesIO(payload))
    print(json.dumps({'archive': args.archive, 'files': list(manifest)}, indent=2))


if __name__ == '__main__':
    main()
