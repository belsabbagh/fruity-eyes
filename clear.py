from os.path import exists
from shutil import rmtree

if __name__ == "__main__":
    valid = False
    print("WARNING: This will delete all files in the `out` and `data` folders. ")
    while not valid:
        answer = input("Are you sure you want to continue? (y/n)\n")
        if answer == "y":
            valid = True
            dirs = ["out", "data"]
            for d in dirs:
                if exists(d):
                    rmtree(d)
            print(
                "Done. Please run `python 0-initialize-files.py` to reconstruct data files."
            )
        elif answer != "n":
            print("Invalid input. Please enter 'y' or 'n'.")
            continue
        exit()
