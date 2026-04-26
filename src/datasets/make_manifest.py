from pathlib import Path
import hashlib
import pandas as pd
from PIL import Image, UnidentifiedImageError



BASE_DIR = Path("data/raw/room_type")

SPLITS  = {
    "train":("train_df.csv","train_images"),
    "val":("val_df.csv","val_images"),
    "test":("test_df.csv","test_images")
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

def process_split(split, df, folder_name):
    image_col = "image_id_ext" 
    folder = BASE_DIR / folder_name

    rows = []
    for img_name in df[image_col].astype(str):
        #path = folder / img_name
        img_name = f"{img_name}.jpg"
        path = folder / folder.name / img_name
        rows.append(check_image(path, img_name, split))

    return rows

def build_report(manifest, max_rows=50):
    missing_df = manifest[manifest["status"] == "missing"]
    corrupted_df = manifest[manifest["status"] == "corrupted"]
    ok_df = manifest[manifest["status"] == "ok"]

    def format_block(df, title):
        if len(df) == 0:
            return f"\n## {title}\nНет\n"

        sample = df.head(max_rows)

        lines = "\n".join(
            sample[["image_id_ext", "local_path", "error"]]
            .fillna("")
            .astype(str)
            .apply(
                lambda row: f"- {row['image_id_ext']} | {row['local_path']} | {row['error']}",
                axis=1,
            )
        )

        return f"\n## {title} ({len(df)})\n\n{lines}\n"

    total = len(manifest)

    report = f"""# Data integrity report

Total: {total}
OK: {len(ok_df)}
Missing: {len(missing_df)} ({len(missing_df)/total:.2%})
Corrupted: {len(corrupted_df)} ({len(corrupted_df)/total:.2%})
"""

    report += format_block(missing_df, "Missing")
    report += format_block(corrupted_df, "Corrupted")

    return report

def main():
    all_rows = []
    expected_total = 0
    processed_splits = set()

    for split, (csv_name, folder_name) in SPLITS.items():
        print(f"Processing {split}...")

        csv_path = BASE_DIR / csv_name
        df = pd.read_csv(csv_path)

        # считаем ожидаемое количество строк
        expected_total += len(df)
        processed_splits.add(split)

        # обрабатываем
        rows = process_split(split, df, folder_name)
        all_rows.extend(rows)

    # 🔍 Проверка split'ов
    expected_splits = set(SPLITS.keys())
    if processed_splits != expected_splits:
        raise ValueError(
            f"Split mismatch! Expected {expected_splits}, got {processed_splits}"
        )

    # 🔍 Проверка количества строк
    if len(all_rows) != expected_total:
        raise ValueError(
            f"Row count mismatch! Expected {expected_total}, got {len(all_rows)}"
        )

    # создаём manifest
    manifest = pd.DataFrame(all_rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest.to_parquet(OUT_PATH, index=False)

    # разделяем статусы
    missing_df = manifest[manifest["status"] == "missing"]
    corrupted_df = manifest[manifest["status"] == "corrupted"]

    # сохраняем списки
    missing_df.to_csv("reports/missing_files.csv", index=False)
    corrupted_df.to_csv("reports/corrupted_files.csv", index=False)

    # генерируем отчёт
    report = build_report(manifest)
    REPORT_PATH.write_text(report, encoding="utf-8")

    # финальный лог
    print("\n=== DONE ===")
    print(f"Expected rows: {expected_total}")
    print(f"Manifest rows: {len(manifest)}")
    print(f"Splits: {sorted(processed_splits)}")
    print(f"Missing: {len(missing_df)}")
    print(f"Corrupted: {len(corrupted_df)}")
    print(f"Manifest → {OUT_PATH}")
    print(f"Report → {REPORT_PATH}")


if __name__ == "__main__":
    main()

