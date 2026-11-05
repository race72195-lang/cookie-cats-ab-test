import shutil
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
TARGET = DATA_DIR / "cookie_cats.csv"

try:
    import kagglehub

    path = Path(kagglehub.dataset_download("yufengsui/mobile-games-ab-testing"))
    source = next(path.glob("*.csv"))
    shutil.copy(source, TARGET)
    print("Done:", TARGET)
except Exception as error:  # network problem, changed dataset name, no access...
    print("Automatic download failed:", error)
    print(
        "\nDownload the file manually:\n"
        "  1. Open kaggle.com and search for 'Mobile Games A/B Testing - Cookie Cats'.\n"
        "  2. Click Download and unzip the archive.\n"
        f"  3. Put the file cookie_cats.csv into this folder: {DATA_DIR}\n"
        "Then run:  python ab_test_analysis.py"
    )
