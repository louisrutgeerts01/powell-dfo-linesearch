import os

# Define the repo structure (relative to current directory)
structure = {
    "requirements.txt": "numpy\nmatplotlib\n",
    "src": {
        "powell.py": "",
        "line_search.py": "",
        "functions.py": "",
        "utils.py": "",
    },
    "scripts": {
        "run_experiment.py": "",
    },
    "experiments": {
        "compare_rosenbrock.ipynb": "",
    },
    "plots": {},
    "tests": {
        "test_line_search.py": "",
        "test_powell.py": "",
    },
}

def make_structure(base, struct):
    for name, content in struct.items():
        path = os.path.join(base, name)
        if isinstance(content, dict):  # folder
            os.makedirs(path, exist_ok=True)
            make_structure(path, content)
        else:  # file
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

if __name__ == "__main__":
    make_structure(".", structure)
    print("✅ Repo structure created in current directory.")