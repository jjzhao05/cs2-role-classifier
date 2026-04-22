from pathlib import Path
import subprocess
import time

# ===== CONFIG =====
PYTHON = r"C:\Users\Jonathan Zhao\AppData\Local\Microsoft\WindowsApps\python3.11.exe"

UNZIP_SCRIPT = Path("file_unzipper.py")
PARSER_SCRIPT = Path("demo_parser.py")

DEMOS_DIR = Path(r"C:\Users\Jonathan Zhao\Documents\GitHub\cs2-role-classifier\demos")
OUTPUT_CSV = Path("output.csv")
# ==================


def run_script(script_path, args=None):
    if args is None:
        args = []

    cmd = [PYTHON, str(script_path)] + args

    print(f"\n[run] {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[error] script failed: {script_path}")
        exit(1)


def main():
    print("=== CS2 Pipeline Start ===")
    start_time = time.time()

    # 1. Extract + dedupe demos
    print("\n[step 1] extracting demos...")
    run_script(UNZIP_SCRIPT)

    # 2. Parse demos → CSV
    print("\n[step 2] parsing demos...")
    run_script(PARSER_SCRIPT, [str(DEMOS_DIR), str(OUTPUT_CSV)])

    print("\n=== Pipeline Complete ===")
    
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"\n[time] Total runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")


if __name__ == "__main__":
    main()