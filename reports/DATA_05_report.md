
# DATA-05 — Отчет по покрытию дополнительных признаков

## Покрытие признаков
- Все признаки: coverage = 1.0
- Пропусков нет
- Shift отсутствует

## Leakage
Удалены:
- class_median_ratio
- class_disputed_share
- class_low_consensus_share

## Финальные признаки
- source_dataset
- ratio
- consensus_band
- item_total_images
- item_disputed_images
- item_low_consensus_images
- class_samples

## Результаты
- Baseline macro F1: 0.159
- Balanced macro F1: 0.305

## Вывод
Признаки слабые → нужен feature engineering
