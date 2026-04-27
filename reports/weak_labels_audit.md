# Weak labels audit

## Output

- Parquet: `/home/alex/Yandex.Disk/mystuff/MIFI/Hakaton_2_semestr/hakathon_avito_CV2/hakathon_room_cv/data/processed/weak_labels_v1.parquet`
- Report: `/home/alex/Yandex.Disk/mystuff/MIFI/Hakaton_2_semestr/hakathon_avito_CV2/hakathon_room_cv/reports/weak_labels_audit.md`

## Sources

- `heuristics_cabinet.csv` -> class_id `5`
- `heuristics_detskaya.csv` -> class_id `6`
- `heuristics_dressing_room.csv` -> class_id `11`


## Counts

- Rows before deduplication: `50340`
- Missing image files: `141`
- Duplicates with train: `0`
- Duplicates inside weak by image_id_ext: `357`
- Duplicates inside weak by hash: `22266`
- Rows after cleaning: `27717`

## Class distribution

class_id
5     14667
6      2933
11    10117
Name: count, dtype: int64

## Source distribution

source
heuristics_cabinet          14667
heuristics_dressing_room    10117
heuristics_detskaya          2933
Name: count, dtype: int64

## DOD checklist

- [x] Manifest contains `image_id_ext`
- [x] Manifest contains `class_id`
- [x] Manifest contains `weak_weight`
- [x] Manifest contains `source`
- [x] Manifest contains `source_flag`
- [x] Manifest contains `is_in_train`
- [x] Duplicates with train removed
- [x] Duplicates by `image_id_ext` removed
