# Error Taxonomy V1

- source_oof: `artifacts/oof/cv03_balanced_sampler/oof_predictions.parquet`
- top_pairs: `10`

| true | true_label | pred | pred_label | count | taxonomy |
| --- | --- | --- | --- | --- | --- |
| 3 | гостиная | 2 | универсальная комната | 45 | visual similarity: living/common room |
| 19 | комната без мебели | 18 | не могу дать ответ / не ясно | 38 | ambiguous empty-room vs unclear image |
| 8 | туалет | 9 | совмещенный санузел | 36 | close sanitary classes |
| 9 | совмещенный санузел | 8 | туалет | 35 | close sanitary classes |
| 7 | ванная комната | 9 | совмещенный санузел | 34 | bathroom vs combined bathroom context |
| 2 | универсальная комната | 3 | гостиная | 34 | visual similarity: common rooms |
| 2 | универсальная комната | 4 | спальня | 34 | generic room vs bedroom furniture cue |
| 1 | кухня-гостиная | 0 | кухня / столовая | 31 | open-space kitchen-living vs kitchen-only |
| 15 | подъезд / лестничная площадка | 10 | коридор / прихожая | 30 | building hallway vs apartment hallway |
| 10 | коридор / прихожая | 18 | не могу дать ответ / не ясно | 30 | poor quality / ambiguous hallway |

## Samples for manual review

### true `3` → pred `2`

- reason: visual similarity: living/common room
- sample_image_id_ext: `['14474048398.jpg', '15357483221.jpg', '15572826456.jpg', '15522277518.jpg', '15840422574.jpg', '14092618448.jpg', '14030668815.jpg', '14364831917.jpg', '14475394206.jpg', '15439220567.jpg']`

### true `19` → pred `18`

- reason: ambiguous empty-room vs unclear image
- sample_image_id_ext: `['14975824465.jpg', '16018844749.jpg', '14419517182.jpg', '14493043195.jpg', '15154686245.jpg', '15431684560.jpg', '15615861980.jpg', '15757030586.jpg', '15516712451.jpg', '15784803413.jpg']`

### true `8` → pred `9`

- reason: close sanitary classes
- sample_image_id_ext: `['14486309628.jpg', '15914605885.jpg', '15522656412.jpg', '12465667933.jpg', '13695734416.jpg', '14365844152.jpg', '14450080447.jpg', '14462144474.jpg', '15220163203.jpg', '15280773774.jpg']`

### true `9` → pred `8`

- reason: close sanitary classes
- sample_image_id_ext: `['13299769056.jpg', '15763325435.jpg', '15642837447.jpg', '15714435128.jpg', '15598578068.jpg', '15845469716.jpg', '15885375927.jpg', '15922408828.jpg', '15990230344.jpg', '16015024126.jpg']`

### true `7` → pred `9`

- reason: bathroom vs combined bathroom context
- sample_image_id_ext: `['14338914996.jpg', '15676043427.jpg', '15561318631.jpg', '16008450008.jpg', '14152543315.jpg', '14348118307.jpg', '14484495375.jpg', '14488147176.jpg', '15913936053.jpg', '15174742709.jpg']`

### true `2` → pred `3`

- reason: visual similarity: common rooms
- sample_image_id_ext: `['12825912425.jpg', '15635306014.jpg', '15453042439.jpg', '15919803172.jpg', '15223481709.jpg', '15246354785.jpg', '15249179255.jpg', '15254209965.jpg', '15267524635.jpg', '15272389612.jpg']`

### true `2` → pred `4`

- reason: generic room vs bedroom furniture cue
- sample_image_id_ext: `['14440474030.jpg', '15714256014.jpg', '16008658505.jpg', '15328397972.jpg', '15490594073.jpg', '15679558974.jpg', '14209051524.jpg', '15317085893.jpg', '15460714008.jpg', '15454166985.jpg']`

### true `1` → pred `0`

- reason: open-space kitchen-living vs kitchen-only
- sample_image_id_ext: `['15573351496.jpg', '15698627495.jpg', '15968720078.jpg', '14440719710.jpg', '15153912375.jpg', '15513738513.jpg', '15494418544.jpg', '15775840811.jpg', '15822101851.jpg', '15632930049.jpg']`

### true `15` → pred `10`

- reason: building hallway vs apartment hallway
- sample_image_id_ext: `['15828493854.jpg', '15113712891.jpg', '14470140860.jpg', '14470199001.jpg', '14572449400.jpg', '15279842786.jpg', '15191392244.jpg', '15361257433.jpg', '15510542283.jpg', '15924279145.jpg']`

### true `10` → pred `18`

- reason: poor quality / ambiguous hallway
- sample_image_id_ext: `['13656888254.jpg', '14330216041.jpg', '14258925001.jpg', '15608640002.jpg', '14199369491.jpg', '15830217542.jpg', '14429861681.jpg', '14475392704.jpg', '15359617621.jpg', '15503850383.jpg']`
