from pathlib import Path
import hashlib
import pandas as pd
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path("data/raw/room_type")

SPLITS  = {
    "train":("train_df.csv","train_images"),
    "val":("val_df.csv","val_images"),
    "train":("test_df.csv","test_images")
}

OUT_PATH = Path("data/processed/data_manifest.parquet")
REPORT_PATH = Path("reports/data_integrity.md")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def check_image(path:Path, image_id:str, split:str ):
    result = {
        "image_id_ext": image_id,
        "split": split,
        "local_path": str(path),
        "hash_sha256": None,
        "width": None,
        "height": None,
        "status": "ok",
        "error": None,
    }

    if not path.exists():
        result["status"] = "missing"
        result["error"] = "file not found"
        return result
    
    try:
        result["hash_sha256"] = sha256_file(path)

        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            result["width"], result["height"] = img.size

    except (UnidentifiedImageError, OSError) as e:
        result["status"] = "corrupted"
        result["error"] = str(e)

    return result

def process_split(split, csv_name, folder_name):
    df = pd.read_csv(BASE_DIR / csv_name)

    # ⚠️ ВАЖНО: проверь имя колонки!
    image_col = "image_id_ext" if "image_id_ext" in df.columns else df.columns[0]

    folder = BASE_DIR / folder_name

    rows = []
    for img_name in df[image_col].astype(str):
        path = folder / img_name
        rows.append(check_image(path, img_name, split))

    return rows

def main():
    all_rows = []

    for split, (csv_name, folder_name) in SPLITS.items():
        print(f"Processing {split}...")
        all_rows.extend(process_split(split, csv_name, folder_name))

    manifest = pd.DataFrame(all_rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest.to_parquet(OUT_PATH, index=False)

    total = len(manifest)
    missing = (manifest["status"] == "missing").sum()
    corrupted = (manifest["status"] == "corrupted").sum()

    REPORT_PATH.write_text(f"""
# Data integrity report

Total: {total}
Missing: {missing} ({missing/total:.2%})
Corrupted: {corrupted} ({corrupted/total:.2%})
""")

    print("Done!")
    print(f"Manifest → {OUT_PATH}")
    print(f"Report → {REPORT_PATH}")


if __name__ == "__main__":
    main()