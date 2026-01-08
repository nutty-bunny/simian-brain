#!/usr/bin/env python3
import subprocess
import os

# Base directory for scripts
base_dir = "/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/scripts/sustained-attention"

# List of scripts to run in order
scripts = [
    "1-seed-2-FEF.py",
    "2-seed-2-LIP.py",
    "3-seed-2-seed.py",
    "4-aggregate-me.py",
]

for script in scripts:
    script_path = os.path.join(base_dir, script)
    print(f"Running {script_path}...")
    result = subprocess.run(["python3", script_path])
    
    if result.returncode != 0:
        print(f"Error running {script_path}, stopping.")
        break
else:
    print("All scripts completed successfully.")
