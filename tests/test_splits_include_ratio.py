from __future__ import annotations

import pandas as pd
import pytest

from src.datasets.data02_build_splits import rows_to_records
from src.utils.labeled_data import load_labeled_csv


def test_rows_to_records_preserves_ratio():
    df = pd.DataFrame(
        {
            "image_id_ext": ["1.jpg"],
            "item_id": [10],
            "result": [5],
            "label": ["кабинет"],
            "ratio": [0.666667],
            "image": ["http://example/1.jpg"],
            "source_dataset": ["train_df"],
            "local_path": ["data/raw/train_images/train_images/1.jpg"],
            "width": [640],
            "height": [480],
            "status": ["ok"],
            "content_hash": ["abc"],
        }
    )

    records = rows_to_records(df)

    assert records[0]["ratio"] == pytest.approx(0.666667)


def test_load_labeled_csv_validates_ratio_range(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "item_id,image,image_id_ext,result,label,ratio\n1,http://example/1.jpg,1,0,кухня,1.2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside"):
        load_labeled_csv(
            path,
            required_columns={"item_id", "image", "image_id_ext", "result", "label", "ratio"},
            source_dataset="train_df",
            ratio_column="ratio",
        )
