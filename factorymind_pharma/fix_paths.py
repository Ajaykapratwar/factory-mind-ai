import glob
from pathlib import Path

for path in glob.glob('dashboard/pages/*.py'):
    p = Path(path)
    content = p.read_text(encoding='utf-8')
    target = '_BASE / "data"'
    replacement = '_BASE.parent / "data"'
    if target in content:
        content = content.replace(target, replacement)
        p.write_text(content, encoding='utf-8')
        print(f'Fixed {path}')
