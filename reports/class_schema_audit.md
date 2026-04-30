# Class Schema Audit

## Decision
- Confirmed target schema: **20 classes `0..19`**.
- `Predicted` must be an integer from `0` to `19` inclusive.
- Class `19` is valid because it is present in `train_df.result` and in `room_type_sample_submission.csv`.
- `val_df` / shadow holdout has no class `18`; OOF remains the primary all-class evaluation gate.

## Sources checked
- `data/raw/train_df.csv`
- `data/raw/val_df.csv`
- `data/raw/room_type_sample_submission.csv`
- `data/splits/splits_v1.json`
- `configs/data/class_mapping.yaml`

## Unique classes
- train_df.result.unique(): `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`
- val_df.result.unique(): `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19]`
- sample_submission.Predicted.unique(): `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`

## Class support
| class_id | label | train_support | val_support | sample_predicted_support |
| --- | --- | --- | --- | --- |
| 0 | кухня / столовая | 248 | 34 | 2402 |
| 1 | кухня-гостиная | 199 | 16 | 2370 |
| 2 | универсальная комната | 247 | 23 | 2404 |
| 3 | гостиная | 249 | 30 | 2325 |
| 4 | спальня | 251 | 24 | 2320 |
| 5 | кабинет | 74 | 22 | 2395 |
| 6 | детская | 215 | 34 | 2411 |
| 7 | ванная комната | 255 | 23 | 2321 |
| 8 | туалет | 255 | 33 | 2433 |
| 9 | совмещенный санузел | 254 | 31 | 2394 |
| 10 | коридор / прихожая | 250 | 29 | 2327 |
| 11 | гардеробная / кладовая / постирочная | 59 | 24 | 2498 |
| 12 | балкон / лоджия | 249 | 23 | 2375 |
| 13 | вид из окна / с балкона | 231 | 21 | 2462 |
| 14 | дом снаружи / двор | 249 | 22 | 2434 |
| 15 | подъезд / лестничная площадка | 253 | 33 | 2380 |
| 16 | другое | 270 | 28 | 2479 |
| 17 | предметы интерьера / быт.техника | 250 | 27 | 2388 |
| 18 | не могу дать ответ / не ясно | 251 | 0 | 2381 |
| 19 | комната без мебели | 253 | 23 | 2504 |

## Validator rule
`src/inference/validate_submission.py` loads `configs/data/class_mapping.yaml` and rejects:
- missing/extra/duplicate `image_id_ext`;
- missing or non-integer `Predicted`;
- any class outside `0..19`;
- row count mismatch against `test_df.csv`.

## Re-run
```bash
python -m src.inference.validate_submission --submission data/raw/room_type_sample_submission.csv --test-csv data/raw/test_df.csv --class-mapping configs/data/class_mapping.yaml
```
