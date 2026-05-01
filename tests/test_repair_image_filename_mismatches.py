from pathlib import Path

import pandas as pd
from PIL import Image

from src.datasets.repair_image_filename_mismatches import repair_dataset


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(path)


def test_repair_dataset_copies_url_named_image_to_expected_image_id_ext(tmp_path):
    base_dir = tmp_path / "data" / "raw"
    image_dir = base_dir / "val_images" / "val_images"
    write_image(image_dir / "source.jpg")

    pd.DataFrame(
        [
            {
                "item_id": "1",
                "image": "http://labelimages.avito.ru/source.jpg",
                "image_id_ext": "expected",
                "result": 0,
                "label": "label",
                "ratio": 1.0,
            }
        ]
    ).to_csv(base_dir / "val_df.csv", index=False)

    stats, rows = repair_dataset(base_dir, ["val"], download=False)

    assert (image_dir / "expected.jpg").exists()
    assert stats["val.mismatched"] == 1
    assert stats["val.created_from_local"] == 1
    assert rows[0]["status"] == "created_from_local"
