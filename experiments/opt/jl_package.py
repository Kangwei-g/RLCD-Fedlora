"""Collect source dependencies for experiment provenance."""
import ast
from pathlib import Path
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
                modules.extend((alias.name for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parent = path.parent
                    for _ in range(node.level - 1):
                        parent = parent.parent
                    prefix = '.'.join(parent.relative_to(ROOT).parts)
                    module = '.'.join((x for x in (prefix, node.module) if x))
                else:
                    module = node.module or ''
                modules.append(module)
                modules.extend((module + '.' + alias.name for alias in node.names))
        for module in modules:
            parts = module.split('.')
            for n in range(1, len(parts) + 1):
                stem = ROOT.joinpath(*parts[:n])
                for candidate in (stem.with_suffix('.py'), stem / '__init__.py'):
                    if candidate.is_file() and candidate not in found:
                        pending.append(candidate)
    return sorted(found)
