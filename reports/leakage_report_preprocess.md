# DATA-02 Leakage Report

## Inputs and policy
- version: `splits_v1`
- timestamp_utc: `2026-05-09T09:33:27Z`
- train_csv: `data/preprocess/train_df.csv`
- val_csv: `data/preprocess/val_df.csv`
- n_folds: `5`
- group_key: `item_id_content_hash_component`
- splitter: `StratifiedGroupKFold`
- duplicate_policy_image_id_ext: `drop_duplicates_keep_first_before_split`
- duplicate_policy_hash: `group_connected_components_before_split`
- val_df_status: `separate_shadow_holdout`
- val_df_reason: safe default: keep original val_df outside train k-folds to avoid tuning leakage

## Train/val overlap checks
| key | train_unique | val_unique | intersection_count | sample |
| --- | --- | --- | --- | --- |
| item_id | 4228 | 372 | 0 | - |
| image_id_ext | 4560 | 463 | 0 | - |
| image | 4560 | 463 | 0 | - |

## Duplicate checks
| dataset | removed_image_id_ext_rows | duplicate_image_url_rows | rows_after_filters |
| --- | --- | --- | --- |
| train_df | 0 | 0 | 4560 |
| val_df | 0 | 0 | 463 |

## Manifest integration
- used_manifest: `True`
- manifest_path: `data/preprocess/data_manifest.parquet`
- manifest_exists: `True`
- manifest_hash_source_column: `hash_sha256`
- manifest_duplicate_image_id_ext_rows: `0`
- train_status_counts: `{'ok': 4560}`
- val_status_counts: `{'ok': 463}`
- hash_overlap_train_vs_shadow_holdout: `0`
- hash_overlap_across_folds: `0`

## Shadow holdout
- `val_df` fixed as `separate_shadow_holdout`.
- rows_in_shadow_holdout_after_filters: `463`
- item_groups_in_shadow_holdout_after_filters: `372`

## Fold summary
| fold | validation_rows | validation_item_groups | training_rows | training_item_groups |
| --- | --- | --- | --- | --- |
| 0 | 912 | 847 | 3648 | 3381 |
| 1 | 913 | 850 | 3647 | 3378 |
| 2 | 911 | 841 | 3649 | 3387 |
| 3 | 913 | 846 | 3647 | 3382 |
| 4 | 911 | 844 | 3649 | 3384 |

## Class distribution by fold
| result | label | fold_0 | fold_1 | fold_2 | fold_3 | fold_4 | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | кухня / столовая | 50 | 49 | 49 | 50 | 50 | 248 |
| 1 | кухня-гостиная | 40 | 40 | 40 | 39 | 40 | 199 |
| 2 | универсальная комната | 50 | 49 | 49 | 50 | 49 | 247 |
| 3 | гостиная | 50 | 50 | 50 | 49 | 50 | 249 |
| 4 | спальня | 50 | 50 | 50 | 51 | 50 | 251 |
| 5 | кабинет | 15 | 15 | 14 | 15 | 14 | 73 |
| 6 | детская | 43 | 43 | 43 | 43 | 43 | 215 |
| 7 | ванная комната | 51 | 51 | 51 | 51 | 51 | 255 |
| 8 | туалет | 51 | 51 | 51 | 51 | 51 | 255 |
| 9 | совмещенный санузел | 51 | 51 | 51 | 50 | 51 | 254 |
| 10 | коридор / прихожая | 50 | 50 | 50 | 50 | 50 | 250 |
| 11 | гардеробная / кладовая / постирочная | 12 | 12 | 11 | 12 | 12 | 59 |
| 12 | балкон / лоджия | 49 | 50 | 50 | 50 | 50 | 249 |
| 13 | вид из окна / с балкона | 46 | 46 | 46 | 46 | 47 | 231 |
| 14 | дом снаружи / двор | 50 | 50 | 50 | 50 | 49 | 249 |
| 15 | подъезд / лестничная площадка | 51 | 51 | 50 | 51 | 50 | 253 |
| 16 | другое | 53 | 54 | 54 | 54 | 54 | 269 |
| 17 | предметы интерьера / быт.техника | 50 | 50 | 50 | 50 | 50 | 250 |
| 18 | не могу дать ответ / не ясно | 50 | 50 | 51 | 50 | 50 | 251 |
| 19 | комната без мебели | 50 | 51 | 51 | 51 | 50 | 253 |

## Pending checks
- none

## Re-run
```bash
python3 scripts/data02_build_splits.py --train-csv data/preprocess/train_df.csv --val-csv data/preprocess/val_df.csv --manifest data/preprocess/data_manifest.parquet --output-json data/preprocess/splits_v1.json --report-md reports/leakage_report_preprocess.md --n-folds 5
```
