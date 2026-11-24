# Build log

## 21 Nov 2026: leaf photos split

| split | healthy | sick | total |
|---|---|---|---|
| train | 1113 | 1092 | 2205 |
| val | 233 | 238 | 471 |
| test | 245 | 261 | 506 |

Source: mohanty/PlantVillage on Hugging Face (data.zip + leaf-map.json), colour photos.
Split by leaf (no leaf in two splits). Seed 42.
Photos with no leaf match: 6749.
Print leaves H1-H6 and S1-S6 come from test. Never train on them.

6,749 photos had no leaf match, same as the official loader, so lab accuracy may be slightly high.

- Worked: kept 1,591 healthy tomato photos and the same number of sick ones, an equal share from each of the 9 diseases, then split 70/15/15 by leaf.
- Broke: the dataset's Hugging Face loader failed ("BuilderConfig 'color' not found"), because newer versions of the datasets library no longer run its setup script.
- Fixed: downloaded data.zip and leaf-map.json directly and copied the dataset's own leaf-matching logic into model/split_leaves_v4.py.






## 25 Nov 2026: leaf model trained

- Model: MobileNetV2 (transfer learning), 224 x 224 photos, Google Colab T4 GPU.
- Stage 1 (top layer only): best val accuracy 97.9%.
- Stage 2 (fine-tune last 30 layers): best val accuracy XX.X%.
- Worked: training stopped by itself once val accuracy stopped improving.
- Odd result: train accuracy (~96%) was lower than val accuracy (97.9%).
- Why: training photos are blurred, brightened and flipped on purpose, and dropout only runs during training, so training is harder than val. Val photos are left untouched.




## 25 Nov 2026: lab test

- Test accuracy: 97.8% (495 of 506 held-out photos), close to val (97.9%), so the model works on new photos.
- Mistakes: 9 healthy leaves called sick, 2 sick leaves called healthy. Mostly false alarms, which is the safer mistake for a plant checker.
- Lab photos have plain backgrounds and good light, so this is the best case.
- The real test is the car camera later in the project.