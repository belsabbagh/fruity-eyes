import subprocess
from os import listdir, path

if __name__ == "__main__":
    python_files = [f for f in listdir() if f.endswith(".py") and f != "main.py"]
    for python_file in python_files:
        print(f"Running {path.basename(python_file)}...")
        subprocess.run(["python", python_file])
        print(f"Done running {path.basename(python_file)}!")
