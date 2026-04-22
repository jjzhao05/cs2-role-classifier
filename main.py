from pathlib import Path
import subprocess
import time

# ===== CONFIG =====
PYTHON = r"C:\Users\Jonathan Zhao\AppData\Local\Microsoft\WindowsApps\python3.11.exe"

UNZIP_SCRIPT = Path("file_unzipper.py")
PARSER_SCRIPT = Path("demo_parser.py")
CLUSTER_SCRIPT = Path("compare_clustering_methods.py")
PLOT_SCRIPT = Path("plotter.py")

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


def time_step(name, func):
    print(f"\n[{name}] starting...")
    start = time.time()

    func()

    end = time.time()
    elapsed = end - start

    print(f"[{name}] done in {elapsed:.2f}s ({elapsed/60:.2f} min)")
    return elapsed


def main():
    print("=== CS2 Pipeline Start ===")
    total_start = time.time()

    timings = {}

    timings["extract"] = time_step(
        "Extracting and deduplicating demos",
        lambda: run_script(UNZIP_SCRIPT),
    )

    timings["parse"] = time_step(
        "Parsing demos into feature dataset",
        lambda: run_script(PARSER_SCRIPT, [str(DEMOS_DIR), str(OUTPUT_CSV)]),
    )

    timings["cluster"] = time_step(
        "Running clustering algorithms",
        lambda: run_script(CLUSTER_SCRIPT),
    )

    timings["plot"] = time_step(
        "Generating visualizations",
        lambda: run_script(PLOT_SCRIPT),
    )

    total_end = time.time()
    total_elapsed = total_end - total_start

    print("\n=== Pipeline Complete ===")

    print("\n[time] Stage breakdown:")
    for name, t in timings.items():
        print(f"  {name:<10}: {t:8.2f}s ({t/60:.2f} min)")

    print(f"\n[time] Total runtime: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")


if __name__ == "__main__":
    main()