# Weak Labels Audit: weak_labels_v1

## Policy and assumptions
- version: `weak_labels_v1`
- source_identity: `heuristic CSV filename stem`
- weak_weight: `0.5` for every v1 weak label
- train_overlap_policy: `drop rows overlapping train by image_id_ext or hash_sha256`
- duplicate_policy: `drop weak duplicates by image_id_ext, then by hash_sha256 when hash is available; keep deterministic first sorted by class/source/image_id_ext`
- no images are downloaded; hashes are reused from `data/processed/data_manifest.parquet`.

## Source to class mapping
| source | class_id |
| --- | --- |
| heuristics_cabinet | 5 |
| heuristics_detskaya | 6 |
| heuristics_dressing_room | 11 |

## Input rows by source
| source | rows |
| --- | --- |
| heuristics_cabinet | 20451 |
| heuristics_detskaya | 14739 |
| heuristics_dressing_room | 15150 |

## Missing hashes
- missing_hash_rows: `50340`

## Train overlaps removed
- by image_id_ext: `0`
- by hash_sha256: `0`
- either key: `0`

## Weak internal duplicates removed
- by image_id_ext: `357`
- by hash_sha256: `0`

## Final rows by class
| class_id | rows |
| --- | --- |
| 5 | 20451 |
| 6 | 14739 |
| 11 | 14793 |

## Final rows by source/class
| source | class_id | rows |
| --- | --- | --- |
| heuristics_cabinet | 5 | 20451 |
| heuristics_detskaya | 6 | 14739 |
| heuristics_dressing_room | 11 | 14793 |

## Final row count
- rows: `49983`

## Re-run
```bash
python3 scripts/build_weak_labels_v1.py
```
