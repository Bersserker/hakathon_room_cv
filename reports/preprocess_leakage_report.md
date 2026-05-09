# DATA-01 Preprocess and leakage report

- generated_at_utc: `2026-05-09T09:33:27Z`
- raw_dir: `data/raw`
- out_dir: `data/preprocess`
- decision: `ok_no_exact_leakage_after_filtering`

## Outputs
- train_csv: `data/preprocess/train_df.csv`
- val_csv: `data/preprocess/val_df.csv`
- test_csv: `data/preprocess/test_df.csv`
- manifest: `data/preprocess/data_manifest.parquet`
- summary_json: `data/preprocess/preprocess_summary.json`

## Rows
| split | raw_rows | saved_rows |
| --- | --- | --- |
| train | 4562 | 4560 |
| val | 500 | 463 |
| test | 48003 | 48003 |

## Manifest status
| split | status | rows |
| --- | --- | --- |
| train | ok | 4562 |
| val | ok | 500 |
| test | ok | 48003 |

## Dropped exact overlaps with test
| split | dropped_rows | image_id_ext_matches | image_url_matches | content_hash_matches | sample_image_id_ext |
| --- | --- | --- | --- | --- | --- |
| train | 2 | 0 | 0 | 2 | 15587295234, 15816354051 |
| val | 37 | 0 | 0 | 37 | 14310005725, 14439071106, 14463135002, 14323904422, 14381003479, 14420132142, 14343795533, 14358449028, 14358458742, 14419842367 |

## Overlaps after filtering
| pair | key | intersection_count | sample |
| --- | --- | --- | --- |
| train_vs_val | item_id | 0 | - |
| train_vs_val | image_id_ext_file | 0 | - |
| train_vs_val | image | 0 | - |
| train_vs_val | content_hash | 0 | - |
| train_vs_test | item_id | 0 | - |
| train_vs_test | image_id_ext_file | 0 | - |
| train_vs_test | image | 0 | - |
| train_vs_test | content_hash | 0 | - |
| val_vs_test | item_id | 258 | 1257180750520, 1259970000823, 1289862000897, 1292244250496, 1293968251777, 1297957750433, 1301198000137, 1301252750847, 1301279000258, 1301348750131 |
| val_vs_test | image_id_ext_file | 0 | - |
| val_vs_test | image | 0 | - |
| val_vs_test | content_hash | 0 | - |

## Leakage policy
- Fatal leakage keys: `image_id_ext`, `image` URL, `content_hash`.
- Exact train/test and val/test overlaps are removed from labeled CSVs before split building.
- `item_id` overlaps with test are reported only: the image model does not consume item_id as a feature.
- Fold-level item/content-hash leakage is checked by DATA-02 split builder.
