# Adversarial validation: train/dev pool vs shadow holdout

## Executive summary
- generated_at_utc: `2026-05-06T15:25:24Z`
- main covariate-shift gate: **class-balanced visual classifier**.
- main result: Class-balanced visual ROC-AUC = 0.5523 ± 0.0389 across folds (negligible/no clear separability).
- raw mode is a warning signal because it includes known label distribution differences.
- low adversarial AUC does **not** prove absence of all shift; it only means these features and this classifier did not find strong separability.
- shadow holdout remains diagnostic only and must not be used for room-classifier tuning/model selection.

## Interpretation thresholds
- ROC-AUC near 0.50–0.60: negligible / no clear separability.
- 0.60–0.70: weak shift.
- 0.70–0.75: moderate shift.
- 0.75–0.85: notable shift.
- >0.85: severe shift.

## Excluded features
The adversarial classifiers exclude target labels, label names, item ids, image ids, URLs, hashes, local paths, source names, and model outputs.

`checksum`, `class_id`, `content_hash`, `domain`, `domain_label`, `hash`, `hash_sha256`, `image`, `image_id_ext`, `item_id`, `label`, `label_name`, `local_path`, `path`, `pred`, `predicted`, `result`, `source_dataset`, `target`, `url`

## Label shift report
| class_id | label | train_count | holdout_count | train_pct | holdout_pct | pct_diff_holdout_minus_train | present_in_train | present_in_holdout |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 18 | не могу дать ответ / не ясно | 251 | 0 | 0.0550 | 0.0000 | -0.0550 | True | False |
| 11 | гардеробная / кладовая / постирочная | 59 | 24 | 0.0129 | 0.0480 | 0.0351 | True | True |
| 5 | кабинет | 74 | 22 | 0.0162 | 0.0440 | 0.0278 | True | True |
| 6 | детская | 215 | 34 | 0.0471 | 0.0680 | 0.0209 | True | True |
| 0 | кухня / столовая | 248 | 34 | 0.0544 | 0.0680 | 0.0136 | True | True |
| 1 | кухня-гостиная | 199 | 16 | 0.0436 | 0.0320 | -0.0116 | True | True |
| 14 | дом снаружи / двор | 249 | 22 | 0.0546 | 0.0440 | -0.0106 | True | True |
| 15 | подъезд / лестничная площадка | 253 | 33 | 0.0555 | 0.0660 | 0.0105 | True | True |
| 8 | туалет | 255 | 33 | 0.0559 | 0.0660 | 0.0101 | True | True |
| 7 | ванная комната | 255 | 23 | 0.0559 | 0.0460 | -0.0099 | True | True |
| 19 | комната без мебели | 253 | 23 | 0.0555 | 0.0460 | -0.0095 | True | True |
| 13 | вид из окна / с балкона | 231 | 21 | 0.0506 | 0.0420 | -0.0086 | True | True |
| 12 | балкон / лоджия | 249 | 23 | 0.0546 | 0.0460 | -0.0086 | True | True |
| 2 | универсальная комната | 247 | 23 | 0.0541 | 0.0460 | -0.0081 | True | True |
| 4 | спальня | 251 | 24 | 0.0550 | 0.0480 | -0.0070 | True | True |
| 9 | совмещенный санузел | 254 | 31 | 0.0557 | 0.0620 | 0.0063 | True | True |
| 3 | гостиная | 249 | 30 | 0.0546 | 0.0600 | 0.0054 | True | True |
| 10 | коридор / прихожая | 250 | 29 | 0.0548 | 0.0580 | 0.0032 | True | True |
| 16 | другое | 270 | 28 | 0.0592 | 0.0560 | -0.0032 | True | True |
| 17 | предметы интерьера / быт.техника | 250 | 27 | 0.0548 | 0.0540 | -0.0008 | True | True |

## Class-balanced mode audit
- shared_classes: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19]`
- excluded_classes: `[18]`
- missing_from_holdout_classes: `[18]`
- missing_from_train_classes: `[]`
- rows_before: `5062`
- rows_after: `1000`
- class_18_note: Class 18 is absent from shadow holdout and is treated as label shift, not covariate shift.

## Raw mode and class-balanced mode metrics
| mode | feature_set | rows | features | roc_auc_mean | roc_auc_std | roc_auc_ci95_half_width | pr_auc_mean | balanced_accuracy_mean | brier_score_mean | log_loss_mean | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | metadata | 5062 | 10 | 0.5520 | 0.0323 | 0.0283 | 0.1201 | 0.5302 | 0.2452 | 0.6845 | negligible/no clear separability |
| class_balanced | metadata | 1000 | 10 | 0.5663 | 0.0434 | 0.0381 | 0.5665 | 0.5310 | 0.2466 | 0.6862 | negligible/no clear separability |
| raw | visual | 5062 | 768 | 0.5740 | 0.0101 | 0.0089 | 0.1331 | 0.5460 | 0.2196 | 0.8242 | negligible/no clear separability |
| class_balanced | visual | 1000 | 768 | 0.5523 | 0.0389 | 0.0341 | 0.5693 | 0.5340 | 0.4109 | 2.2833 | negligible/no clear separability |

## Metadata shift
Metadata-only classifiers use annotation consensus ratio, dimensions, aspect ratio, megapixels, and orientation/shape indicators only.
| mode | feature_set | rank | feature | mean_abs_coefficient | max_abs_coefficient |
| --- | --- | --- | --- | --- | --- |
| raw | metadata | 1 | width | 2.4480 | 2.7538 |
| raw | metadata | 2 | megapixels | 2.3176 | 2.6478 |
| raw | metadata | 3 | aspect_ratio | 0.6450 | 0.7700 |
| raw | metadata | 4 | height | 0.5886 | 0.7024 |
| raw | metadata | 5 | is_wide | 0.1703 | 0.2250 |
| raw | metadata | 6 | is_portrait | 0.1034 | 0.1734 |
| raw | metadata | 7 | is_landscape | 0.0975 | 0.1830 |
| raw | metadata | 8 | ratio | 0.0879 | 0.1432 |
| raw | metadata | 9 | is_tall | 0.0651 | 0.1489 |
| raw | metadata | 10 | is_squareish | 0.0367 | 0.0858 |
| class_balanced | metadata | 1 | megapixels | 1.6174 | 1.7815 |
| class_balanced | metadata | 2 | width | 1.1973 | 1.4087 |
| class_balanced | metadata | 3 | height | 0.7445 | 0.8561 |
| class_balanced | metadata | 4 | is_portrait | 0.1448 | 0.2366 |
| class_balanced | metadata | 5 | is_landscape | 0.1388 | 0.2373 |
| class_balanced | metadata | 6 | ratio | 0.1269 | 0.1782 |
| class_balanced | metadata | 7 | aspect_ratio | 0.1156 | 0.2483 |
| class_balanced | metadata | 8 | is_wide | 0.0956 | 0.1385 |
| class_balanced | metadata | 9 | is_tall | 0.0433 | 0.1368 |
| class_balanced | metadata | 10 | is_squareish | 0.0220 | 0.0410 |

## Visual shift
Visual classifiers use frozen image embeddings; no room-classifier logits/probabilities are included.
| mode | feature_set | roc_auc_mean | roc_auc_std | pr_auc_mean | balanced_accuracy_mean | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| raw | visual | 0.5740 | 0.0101 | 0.1331 | 0.5460 | negligible/no clear separability |
| class_balanced | visual | 0.5523 | 0.0389 | 0.5693 | 0.5340 | negligible/no clear separability |

## Per-class adversarial summaries
| mode | feature_set | class_id | label | train_count | holdout_count | roc_auc | mean_p_for_train_rows | mean_p_for_holdout_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | metadata | 0 | кухня / столовая | 248 | 34 | 0.4869 | 0.4922 | 0.4889 |
| raw | metadata | 1 | кухня-гостиная | 199 | 16 | 0.5518 | 0.4771 | 0.4869 |
| raw | metadata | 2 | универсальная комната | 247 | 23 | 0.5471 | 0.4719 | 0.4846 |
| raw | metadata | 3 | гостиная | 249 | 30 | 0.4814 | 0.4783 | 0.4718 |
| raw | metadata | 4 | спальня | 251 | 24 | 0.4896 | 0.4945 | 0.4892 |
| raw | metadata | 5 | кабинет | 74 | 22 | 0.8151 | 0.4756 | 0.5881 |
| raw | metadata | 6 | детская | 215 | 34 | 0.5342 | 0.4872 | 0.4952 |
| raw | metadata | 7 | ванная комната | 255 | 23 | 0.6271 | 0.4870 | 0.5048 |
| raw | metadata | 8 | туалет | 255 | 33 | 0.5218 | 0.4832 | 0.4893 |
| raw | metadata | 9 | совмещенный санузел | 254 | 31 | 0.5587 | 0.4917 | 0.4987 |
| raw | metadata | 10 | коридор / прихожая | 250 | 29 | 0.4885 | 0.4840 | 0.4828 |
| raw | metadata | 11 | гардеробная / кладовая / постирочная | 59 | 24 | 0.6289 | 0.4771 | 0.5117 |
| raw | metadata | 12 | балкон / лоджия | 249 | 23 | 0.4880 | 0.4855 | 0.4822 |
| raw | metadata | 13 | вид из окна / с балкона | 231 | 21 | 0.6592 | 0.4911 | 0.5212 |
| raw | metadata | 14 | дом снаружи / двор | 249 | 22 | 0.5368 | 0.5023 | 0.5066 |
| raw | metadata | 15 | подъезд / лестничная площадка | 253 | 33 | 0.5747 | 0.4835 | 0.4948 |
| raw | metadata | 16 | другое | 270 | 28 | 0.5402 | 0.5588 | 0.5796 |
| raw | metadata | 17 | предметы интерьера / быт.техника | 250 | 27 | 0.4973 | 0.4939 | 0.4889 |
| raw | metadata | 18 | не могу дать ответ / не ясно | 251 | 0 |  | 0.4682 |  |
| raw | metadata | 19 | комната без мебели | 253 | 23 | 0.4841 | 0.4804 | 0.4789 |
| class_balanced | metadata | 0 | кухня / столовая | 34 | 34 | 0.5130 | 0.5062 | 0.5123 |
| class_balanced | metadata | 1 | кухня-гостиная | 16 | 16 | 0.3906 | 0.5099 | 0.4889 |
| class_balanced | metadata | 2 | универсальная комната | 23 | 23 | 0.6569 | 0.4628 | 0.4861 |
| class_balanced | metadata | 3 | гостиная | 30 | 30 | 0.5411 | 0.4824 | 0.4834 |
| class_balanced | metadata | 4 | спальня | 24 | 24 | 0.5920 | 0.4925 | 0.5052 |
| class_balanced | metadata | 5 | кабинет | 22 | 22 | 0.8244 | 0.4794 | 0.6115 |
| class_balanced | metadata | 6 | детская | 34 | 34 | 0.5333 | 0.4991 | 0.5135 |
| class_balanced | metadata | 7 | ванная комната | 23 | 23 | 0.6295 | 0.4775 | 0.5111 |
| class_balanced | metadata | 8 | туалет | 33 | 33 | 0.6125 | 0.4791 | 0.4953 |
| class_balanced | metadata | 9 | совмещенный санузел | 31 | 31 | 0.4079 | 0.5034 | 0.4826 |
| class_balanced | metadata | 10 | коридор / прихожая | 29 | 29 | 0.5600 | 0.4816 | 0.4895 |
| class_balanced | metadata | 11 | гардеробная / кладовая / постирочная | 24 | 24 | 0.7231 | 0.4579 | 0.5282 |
| class_balanced | metadata | 12 | балкон / лоджия | 23 | 23 | 0.4631 | 0.4922 | 0.4867 |
| class_balanced | metadata | 13 | вид из окна / с балкона | 21 | 21 | 0.6429 | 0.4891 | 0.5202 |
| class_balanced | metadata | 14 | дом снаружи / двор | 22 | 22 | 0.6126 | 0.5014 | 0.5214 |
| class_balanced | metadata | 15 | подъезд / лестничная площадка | 33 | 33 | 0.5312 | 0.4937 | 0.5001 |
| class_balanced | metadata | 16 | другое | 28 | 28 | 0.5281 | 0.5489 | 0.5581 |
| class_balanced | metadata | 17 | предметы интерьера / быт.техника | 27 | 27 | 0.4156 | 0.5101 | 0.4910 |
| class_balanced | metadata | 19 | комната без мебели | 23 | 23 | 0.5331 | 0.4825 | 0.4855 |
| raw | visual | 0 | кухня / столовая | 248 | 34 | 0.4872 | 0.3326 | 0.2920 |

_Showing 40 of 78 rows._

## Top examples for inspection
| mode | feature_set | example_type | rank | image_id_ext | item_id | result | label | domain | p_shadow_holdout | fold | local_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | metadata | holdout_like_train_examples | 1 | 15888269104.jpg | 1370230003074 | 9 | совмещенный санузел | train_dev_pool | 0.9286 | 3 | data/raw/train_images/train_images/15888269104.jpg |
| raw | metadata | holdout_like_train_examples | 2 | 14171940094.jpg | 1296061750253 | 14 | дом снаружи / двор | train_dev_pool | 0.9071 | 0 | data/raw/train_images/train_images/14171940094.jpg |
| raw | metadata | holdout_like_train_examples | 3 | 15680828680.jpg | 1358940251360 | 16 | другое | train_dev_pool | 0.8801 | 1 | data/raw/train_images/train_images/15680828680.jpg |
| raw | metadata | holdout_like_train_examples | 4 | 15491460339.jpg | 1346374000971 | 9 | совмещенный санузел | train_dev_pool | 0.8676 | 4 | data/raw/train_images/train_images/15491460339.jpg |
| raw | metadata | holdout_like_train_examples | 5 | 15826033032.jpg | 1367821500446 | 17 | предметы интерьера / быт.техника | train_dev_pool | 0.8515 | 1 | data/raw/train_images/train_images/15826033032.jpg |
| raw | metadata | holdout_like_train_examples | 6 | 15661252076.jpg | 1361828001870 | 16 | другое | train_dev_pool | 0.8500 | 2 | data/raw/train_images/train_images/15661252076.jpg |
| raw | metadata | holdout_like_train_examples | 7 | 15889690014.jpg | 1354419004855 | 7 | ванная комната | train_dev_pool | 0.8444 | 2 | data/raw/train_images/train_images/15889690014.jpg |
| raw | metadata | holdout_like_train_examples | 8 | 15737445232.jpg | 1364553500288 | 9 | совмещенный санузел | train_dev_pool | 0.8395 | 0 | data/raw/train_images/train_images/15737445232.jpg |
| raw | metadata | holdout_like_train_examples | 9 | 15905730475.jpg | 1370835505490 | 16 | другое | train_dev_pool | 0.8302 | 4 | data/raw/train_images/train_images/15905730475.jpg |
| raw | metadata | holdout_like_train_examples | 10 | 15703412347.jpg | 1282865000026 | 14 | дом снаружи / двор | train_dev_pool | 0.8298 | 1 | data/raw/train_images/train_images/15703412347.jpg |
| raw | metadata | holdout_like_train_examples | 11 | 15890054297.jpg | 1370290505205 | 8 | туалет | train_dev_pool | 0.8249 | 3 | data/raw/train_images/train_images/15890054297.jpg |
| raw | metadata | holdout_like_train_examples | 12 | 15704216432.jpg | 1363395752791 | 16 | другое | train_dev_pool | 0.8235 | 1 | data/raw/train_images/train_images/15704216432.jpg |
| raw | metadata | holdout_like_train_examples | 13 | 15986666622.jpg | 1373725750385 | 13 | вид из окна / с балкона | train_dev_pool | 0.8216 | 1 | data/raw/train_images/train_images/15986666622.jpg |
| raw | metadata | holdout_like_train_examples | 14 | 15678732862.jpg | 1308311250094 | 7 | ванная комната | train_dev_pool | 0.8127 | 0 | data/raw/train_images/train_images/15678732862.jpg |
| raw | metadata | holdout_like_train_examples | 15 | 15502273355.jpg | 1354348751719 | 19 | комната без мебели | train_dev_pool | 0.7962 | 2 | data/raw/train_images/train_images/15502273355.jpg |
| raw | metadata | holdout_like_train_examples | 16 | 15748859417.jpg | 1340933250062 | 16 | другое | train_dev_pool | 0.7956 | 1 | data/raw/train_images/train_images/15748859417.jpg |
| raw | metadata | holdout_like_train_examples | 17 | 15835132515.jpg | 1368163755347 | 12 | балкон / лоджия | train_dev_pool | 0.7874 | 4 | data/raw/train_images/train_images/15835132515.jpg |
| raw | metadata | holdout_like_train_examples | 18 | 14472819385.jpg | 1309881750202 | 16 | другое | train_dev_pool | 0.7858 | 3 | data/raw/train_images/train_images/14472819385.jpg |
| raw | metadata | holdout_like_train_examples | 19 | 15834869010.jpg | 1368169502714 | 16 | другое | train_dev_pool | 0.7624 | 1 | data/raw/train_images/train_images/15834869010.jpg |
| raw | metadata | holdout_like_train_examples | 20 | 15432850052.jpg | 1352189252047 | 0 | кухня / столовая | train_dev_pool | 0.7620 | 1 | data/raw/train_images/train_images/15432850052.jpg |
| raw | metadata | train_like_holdout_examples | 1 | 14383835948.jpg | 1307666250646 | 17 | предметы интерьера / быт.техника | shadow_holdout | 0.3024 | 0 | data/raw/val_images/val_images/14383835948.jpg |
| raw | metadata | train_like_holdout_examples | 2 | 15803926481.jpg | 1367085503633 | 1 | кухня-гостиная | shadow_holdout | 0.3080 | 3 | data/raw/val_images/val_images/15803926481.jpg |
| raw | metadata | train_like_holdout_examples | 3 | 14474443479.jpg | 1309866250969 | 0 | кухня / столовая | shadow_holdout | 0.3108 | 0 | data/raw/val_images/val_images/14474443479.jpg |
| raw | metadata | train_like_holdout_examples | 4 | 14395798694.jpg | 1309772252076 | 0 | кухня / столовая | shadow_holdout | 0.3119 | 0 | data/raw/val_images/val_images/14395798694.jpg |
| raw | metadata | train_like_holdout_examples | 5 | 14465710523.jpg | 1309667501167 | 3 | гостиная | shadow_holdout | 0.3134 | 0 | data/raw/val_images/val_images/14465710523.jpg |
| raw | metadata | train_like_holdout_examples | 6 | 14465723841.jpg | 1309667501167 | 3 | гостиная | shadow_holdout | 0.3134 | 0 | data/raw/val_images/val_images/14465723841.jpg |
| raw | metadata | train_like_holdout_examples | 7 | 14489708423.jpg | 1310264500339 | 6 | детская | shadow_holdout | 0.3511 | 2 | data/raw/val_images/val_images/14489708423.jpg |
| raw | metadata | train_like_holdout_examples | 8 | 14489708472.jpg | 1310264500339 | 4 | спальня | shadow_holdout | 0.3511 | 2 | data/raw/val_images/val_images/14489708472.jpg |
| raw | metadata | train_like_holdout_examples | 9 | 14489708526.jpg | 1310264500339 | 6 | детская | shadow_holdout | 0.3511 | 2 | data/raw/val_images/val_images/14489708526.jpg |
| raw | metadata | train_like_holdout_examples | 10 | 14486292306.jpg | 1310218001296 | 12 | балкон / лоджия | shadow_holdout | 0.3864 | 0 | data/raw/val_images/val_images/14486292306.jpg |
| raw | metadata | train_like_holdout_examples | 11 | 14486292716.jpg | 1310218001296 | 19 | комната без мебели | shadow_holdout | 0.3864 | 0 | data/raw/val_images/val_images/14486292716.jpg |
| raw | metadata | train_like_holdout_examples | 12 | 15620932559.jpg | 1287376251246 | 6 | детская | shadow_holdout | 0.3871 | 1 | data/raw/val_images/val_images/15620932559.jpg |
| raw | metadata | train_like_holdout_examples | 13 | 15838959315.jpg | 1368283501631 | 4 | спальня | shadow_holdout | 0.3872 | 0 | data/raw/val_images/val_images/15838959315.jpg |
| raw | metadata | train_like_holdout_examples | 14 | 15593681262.jpg | 1358892001131 | 6 | детская | shadow_holdout | 0.3876 | 3 | data/raw/val_images/val_images/15593681262.jpg |
| raw | metadata | train_like_holdout_examples | 15 | 14478121480.jpg | 1309913500532 | 19 | комната без мебели | shadow_holdout | 0.3882 | 1 | data/raw/val_images/val_images/14478121480.jpg |
| raw | metadata | train_like_holdout_examples | 16 | 14493462625.jpg | 1310380251781 | 2 | универсальная комната | shadow_holdout | 0.3889 | 1 | data/raw/val_images/val_images/14493462625.jpg |
| raw | metadata | train_like_holdout_examples | 17 | 14449760914.jpg | 1309354250671 | 16 | другое | shadow_holdout | 0.3998 | 3 | data/raw/val_images/val_images/14449760914.jpg |
| raw | metadata | train_like_holdout_examples | 18 | 14361874520.jpg | 1306878751747 | 16 | другое | shadow_holdout | 0.3998 | 3 | data/raw/val_images/val_images/14361874520.jpg |
| raw | metadata | train_like_holdout_examples | 19 | 14488254283.jpg | 1310239750445 | 16 | другое | shadow_holdout | 0.4007 | 3 | data/raw/val_images/val_images/14488254283.jpg |
| raw | metadata | train_like_holdout_examples | 20 | 14443810553.jpg | 1309223000194 | 12 | балкон / лоджия | shadow_holdout | 0.4055 | 0 | data/raw/val_images/val_images/14443810553.jpg |
| raw | metadata | confidently_separated_holdout_examples | 1 | 15808564023.jpg | 1367201502375 | 5 | кабинет | shadow_holdout | 0.8412 | 1 | data/raw/val_images/val_images/15808564023.jpg |
| raw | metadata | confidently_separated_holdout_examples | 2 | 15626086434.jpg | 1359028000503 | 16 | другое | shadow_holdout | 0.8393 | 4 | data/raw/val_images/val_images/15626086434.jpg |
| raw | metadata | confidently_separated_holdout_examples | 3 | 14449760909.jpg | 1309354250671 | 16 | другое | shadow_holdout | 0.8217 | 3 | data/raw/val_images/val_images/14449760909.jpg |
| raw | metadata | confidently_separated_holdout_examples | 4 | 15808331145.jpg | 1367197502177 | 6 | детская | shadow_holdout | 0.7566 | 3 | data/raw/val_images/val_images/15808331145.jpg |
| raw | metadata | confidently_separated_holdout_examples | 5 | 14476994681.jpg | 1309916000003 | 16 | другое | shadow_holdout | 0.7289 | 2 | data/raw/val_images/val_images/14476994681.jpg |
| raw | metadata | confidently_separated_holdout_examples | 6 | 15816500049.jpg | 1367607505990 | 17 | предметы интерьера / быт.техника | shadow_holdout | 0.7141 | 0 | data/raw/val_images/val_images/15816500049.jpg |
| raw | metadata | confidently_separated_holdout_examples | 7 | 15807164544.jpg | 1367201752942 | 5 | кабинет | shadow_holdout | 0.7135 | 2 | data/raw/val_images/val_images/15807164544.jpg |
| raw | metadata | confidently_separated_holdout_examples | 8 | 14490595634.jpg | 1310312500913 | 16 | другое | shadow_holdout | 0.7108 | 1 | data/raw/val_images/val_images/14490595634.jpg |
| raw | metadata | confidently_separated_holdout_examples | 9 | 14298463344.jpg | 1303148501754 | 16 | другое | shadow_holdout | 0.7055 | 2 | data/raw/val_images/val_images/14298463344.jpg |
| raw | metadata | confidently_separated_holdout_examples | 10 | 14484146939.jpg | 1310164751491 | 16 | другое | shadow_holdout | 0.6725 | 4 | data/raw/val_images/val_images/14484146939.jpg |
| raw | metadata | confidently_separated_holdout_examples | 11 | 15824373816.jpg | 1367740500136 | 16 | другое | shadow_holdout | 0.6711 | 4 | data/raw/val_images/val_images/15824373816.jpg |
| raw | metadata | confidently_separated_holdout_examples | 12 | 15815546103.jpg | 1367610508389 | 17 | предметы интерьера / быт.техника | shadow_holdout | 0.6599 | 4 | data/raw/val_images/val_images/15815546103.jpg |
| raw | metadata | confidently_separated_holdout_examples | 13 | 15641457555.jpg | 1360977500552 | 11 | гардеробная / кладовая / постирочная | shadow_holdout | 0.6532 | 3 | data/raw/val_images/val_images/15641457555.jpg |
| raw | metadata | confidently_separated_holdout_examples | 14 | 15560971349.jpg | 1362721754229 | 11 | гардеробная / кладовая / постирочная | shadow_holdout | 0.6532 | 3 | data/raw/val_images/val_images/15560971349.jpg |
| raw | metadata | confidently_separated_holdout_examples | 15 | 15794366460.jpg | 1366713751032 | 5 | кабинет | shadow_holdout | 0.6531 | 3 | data/raw/val_images/val_images/15794366460.jpg |
| raw | metadata | confidently_separated_holdout_examples | 16 | 15802684133.jpg | 1367037501391 | 5 | кабинет | shadow_holdout | 0.6531 | 3 | data/raw/val_images/val_images/15802684133.jpg |
| raw | metadata | confidently_separated_holdout_examples | 17 | 15808499817.jpg | 1367201754182 | 5 | кабинет | shadow_holdout | 0.6531 | 3 | data/raw/val_images/val_images/15808499817.jpg |
| raw | metadata | confidently_separated_holdout_examples | 18 | 15825123785.jpg | 1367783750156 | 17 | предметы интерьера / быт.техника | shadow_holdout | 0.6531 | 3 | data/raw/val_images/val_images/15825123785.jpg |
| raw | metadata | confidently_separated_holdout_examples | 19 | 15191237004.jpg | 1352190501609 | 16 | другое | shadow_holdout | 0.6524 | 2 | data/raw/val_images/val_images/15191237004.jpg |
| raw | metadata | confidently_separated_holdout_examples | 20 | 15788528387.jpg | 1366551500414 | 16 | другое | shadow_holdout | 0.6524 | 2 | data/raw/val_images/val_images/15788528387.jpg |

_Showing 60 of 240 rows._

## Re-run
```bash
python scripts/run_adversarial_validation.py --splits-json data/splits/splits_v1.json --output-dir artifacts/diagnostics/adversarial_validation --report-md reports/adversarial_validation.md --embedding-cache artifacts/diagnostics/adversarial_validation/clip_embeddings.parquet --clip-model-name openai/clip-vit-base-patch32 --n-folds 5 --seed 26042026
```
