from pathlib import Path
import argparse
import json

import pandas as pd


ID_COLS = [
    "image_id_ext",
    "item_id",
    "target",
    "label",
    "fold",
    "source_dataset",
    "pred",
]


def sorted_class_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]

    return sorted(cols, key=lambda c: int(c.split("_")[1]))


def export_file(
    input_path: Path,
    output_path: Path,
    run_name: str,
) -> dict:
    df = pd.read_parquet(input_path)

    if "image_id_ext" not in df.columns:
        raise ValueError(f"{input_path}: нет колонки image_id_ext")

    prob_cols = sorted_class_cols(df, "prob_")
    logit_cols = sorted_class_cols(df, "logit_")

    if not prob_cols:
        raise ValueError(f"{input_path}: нет колонок prob_*")

    keep_cols = []

    for col in ID_COLS:
        if col in df.columns:
            keep_cols.append(col)

    keep_cols += logit_cols
    keep_cols += prob_cols

    out = df[keep_cols].copy()
    out["run_name"] = run_name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(out),
        "cols": len(out.columns),
        "has_logits": bool(logit_cols),
        "num_probs": len(prob_cols),
        "columns": out.columns.tolist(),
    }


def export_run(run_dir: Path, logits_root: Path) -> dict:
    run_name = run_dir.name

    oof_path = run_dir / "oof_predictions.parquet"
    test_path = run_dir / "shadow_holdout_predictions.parquet"

    if not oof_path.exists():
        raise FileNotFoundError(f"{run_name}: нет {oof_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"{run_name}: нет {test_path}")

    out_dir = logits_root / run_name

    oof_info = export_file(
        input_path=oof_path,
        output_path=out_dir / "oof_logits.parquet",
        run_name=run_name,
    )

    test_info = export_file(
        input_path=test_path,
        output_path=out_dir / "test_logits.parquet",
        run_name=run_name,
    )

    metadata = {
        "run_name": run_name,
        "oof": oof_info,
        "test": test_info,
    }

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--oof-root",
        default="artifacts/oof",
        help="Папка с OOF-предсказаниями моделей",
    )
    parser.add_argument(
        "--logits-root",
        default="artifacts/logits",
        help="Куда сохранять единый экспорт logits/probabilities",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Имя конкретного run. Если не указано, экспортируются все run'ы",
    )

    args = parser.parse_args()

    oof_root = Path(args.oof_root)
    logits_root = Path(args.logits_root)

    if args.run_name:
        run_dirs = [oof_root / args.run_name]
    else:
        run_dirs = sorted([p for p in oof_root.iterdir() if p.is_dir()])

    all_metadata = []

    for run_dir in run_dirs:
        try:
            metadata = export_run(run_dir, logits_root)
            all_metadata.append(metadata)
            print(f"[OK] {run_dir.name}")
        except Exception as e:
            print(f"[SKIP] {run_dir.name}: {e}")

    print("\nExport finished")
    print(f"Exported runs: {len(all_metadata)}")


if __name__ == "__main__":
    main()