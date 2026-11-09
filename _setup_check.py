"""Check installed packages and perform backup."""
import sys, os, shutil

root = os.path.dirname(os.path.abspath(__file__))
print("Root:", root)
print("Python:", sys.executable)

# Check packages
for pkg in ['lime', 'shap', 'captum']:
    try:
        __import__(pkg)
        print(f"{pkg}: INSTALLED")
    except ImportError:
        print(f"{pkg}: NOT FOUND")

# Create backup directory
backup_dir = os.path.join(root, '.backup')
os.makedirs(backup_dir, exist_ok=True)

files_to_backup = [
    (os.path.join(root, 'app', 'main.py'), os.path.join(backup_dir, 'main.py')),
    (os.path.join(root, 'app', 'explain.py'), os.path.join(backup_dir, 'explain.py')),
    (os.path.join(root, 'app', 'templates', 'index.html'), os.path.join(backup_dir, 'index.html')),
    (os.path.join(root, 'app', 'static', 'style.css'), os.path.join(backup_dir, 'style.css')),
    (os.path.join(root, 'app', 'static', 'script.js'), os.path.join(backup_dir, 'script.js')),
]

for src, dst in files_to_backup:
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Backed up: {src} -> {dst}")
    else:
        print(f"WARNING: Source not found: {src}")

print("Done.")
