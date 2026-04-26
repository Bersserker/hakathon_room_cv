# DATA-02 Leakage Report

## Inputs and policy
- version: `splits_v1`
- timestamp_utc: `2026-04-23T18:29:17Z`
- train_csv: `data/raw/train_df.csv`
- val_csv: `data/raw/val_df.csv`
- n_folds: `5`
- group_key: `item_id`
- splitter: `StratifiedGroupKFold`
- duplicate_policy_image_id_ext: `drop_duplicates_keep_first_before_split`
- duplicate_policy_hash: `report_only_when_manifest_available`
- val_df_status: `separate_shadow_holdout`
- val_df_reason: safe default: keep original val_df outside train k-folds to avoid tuning leakage

## Train/val overlap checks
| key | train_unique | val_unique | intersection_count | sample |
| --- | --- | --- | --- | --- |
| item_id | 4230 | 380 | 0 | - |
| image_id_ext | 4562 | 500 | 0 | - |
| image | 4562 | 500 | 0 | - |

## Duplicate checks
| dataset | removed_image_id_ext_rows | duplicate_image_url_rows | rows_after_filters |
| --- | --- | --- | --- |
| train_df | 0 | 0 | 4562 |
| val_df | 0 | 0 | 500 |

## Manifest integration
- used_manifest: `False`
- manifest_path: `data/processed/data_manifest.parquet`
- manifest_exists: `False`
- manifest_hash_source_column: `None`
- hash/status checks: `pending until data/processed/data_manifest.parquet exists`

## Shadow holdout
- `val_df` fixed as `separate_shadow_holdout`.
- rows_in_shadow_holdout_after_filters: `500`
- item_groups_in_shadow_holdout_after_filters: `380`

## Fold summary
| fold | validation_rows | validation_item_groups | training_rows | training_item_groups |
| --- | --- | --- | --- | --- |
| 0 | 912 | 845 | 3650 | 3385 |
| 1 | 913 | 846 | 3649 | 3384 |
| 2 | 913 | 848 | 3649 | 3382 |
| 3 | 912 | 846 | 3650 | 3384 |
| 4 | 912 | 845 | 3650 | 3385 |

## Class distribution by fold
| result | label | fold_0 | fold_1 | fold_2 | fold_3 | fold_4 | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | кухня / столовая | 50 | 49 | 50 | 49 | 50 | 248 |
| 1 | кухня-гостиная | 40 | 40 | 40 | 39 | 40 | 199 |
| 2 | универсальная комната | 49 | 49 | 50 | 50 | 49 | 247 |
| 3 | гостиная | 50 | 50 | 50 | 50 | 49 | 249 |
| 4 | спальня | 51 | 50 | 50 | 50 | 50 | 251 |
| 5 | кабинет | 14 | 15 | 15 | 15 | 15 | 74 |
| 6 | детская | 43 | 43 | 43 | 43 | 43 | 215 |
| 7 | ванная комната | 51 | 51 | 51 | 51 | 51 | 255 |
| 8 | туалет | 51 | 51 | 51 | 51 | 51 | 255 |
| 9 | совмещенный санузел | 51 | 51 | 51 | 50 | 51 | 254 |
| 10 | коридор / прихожая | 50 | 50 | 50 | 50 | 50 | 250 |
| 11 | гардеробная / кладовая / постирочная | 12 | 12 | 12 | 12 | 11 | 59 |
| 12 | балкон / лоджия | 50 | 50 | 49 | 50 | 50 | 249 |
| 13 | вид из окна / с балкона | 46 | 46 | 46 | 46 | 47 | 231 |
| 14 | дом снаружи / двор | 49 | 50 | 50 | 50 | 50 | 249 |
| 15 | подъезд / лестничная площадка | 51 | 50 | 50 | 51 | 51 | 253 |
| 16 | другое | 54 | 54 | 54 | 54 | 54 | 270 |
| 17 | предметы интерьера / быт.техника | 50 | 50 | 50 | 50 | 50 | 250 |
| 18 | не могу дать ответ / не ясно | 50 | 51 | 50 | 50 | 50 | 251 |
| 19 | комната без мебели | 50 | 51 | 51 | 51 | 50 | 253 |

## Pending checks
- merge `data/processed/data_manifest.parquet` to add `local_path`, `width`, `height`, `status`, `content_hash`
- exclude non-`ok` manifest statuses from train pool and shadow holdout
- run content-hash leakage checks across train folds and between train pool vs shadow holdout

## Re-run
```bash
python3 scripts/data02_build_splits.py --train-csv data/raw/train_df.csv --val-csv data/raw/val_df.csv --manifest data/processed/data_manifest.parquet --output-json data/splits/splits_v1.json --report-md reports/leakage_report.md --n-folds 5
```
