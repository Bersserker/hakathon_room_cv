# Weak Images Download Report: weak_images_v1

## Policy
- version: `weak_images_v1`
- weak_weight: `0.35`
- max_texts: `0`
- drop_catalog: `True`
- drop_person: `True`
- min_size: `64x64`
- max_added_per_class: `{'5': 180, '6': 80, '11': 200}`
- candidate_score: `perform_top_microcat_prob - perform_top_other_classes_prob + 0.1 * crop_area - 0.01 * n_texts`
- leakage_checked: `image_id_ext` and `sha256` against train/val/test manifests/splits

## Input rows by source
| source | rows |
| --- | --- |
| heuristics_cabinet | 20451 |
| heuristics_detskaya | 14739 |
| heuristics_dressing_room | 15150 |

## Drop counts by reason
| reason | rows |
| --- | --- |
| corrupted_image | 0 |
| duplicate_hash_sha256 | 21 |
| duplicate_image_id_ext | 1 |
| empty_image_id_ext | 0 |
| is_catalog | 0 |
| leakage_hash_sha256 | 0 |
| leakage_image_id_ext | 0 |
| max_texts | 428 |
| missing_image | 49753 |
| person_found | 2 |
| quota | 0 |
| small_image | 0 |

## Selection
- rows_after_gates: `157`
- rows_after_dedup: `135`
- selected_rows: `135`
- output_image_dir: `data/raw/weak_images/weak_images`

## Selected rows by class
| class_id | rows |
| --- | --- |
| 11 | 47 |
| 5 | 65 |
| 6 | 23 |
