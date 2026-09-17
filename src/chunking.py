import ast
from pathlib import Path

def get_target_files(raw_dir: Path):
    target_files = []
    base = Path(raw_dir)
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".py":
            target_files.append((path,"python"))
        elif path.suffix in [".md", ".txt", ".rst"]:
            target_files.append((path,"markdown"))
        else:
            continue
    return target_files

def get_trees(target_files):
    python_tree = []
    markdown_tree = []
    for i in target_files:
        if i[1] == "python":
            python_tree.append(ast.parse(i[0]))
        elif i[1] == "markdown":
            markdown_tree.append(ast.parse(i[0]))
        else:
            continue
    return (python_tree, markdown_tree)



