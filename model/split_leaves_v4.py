# split_leaves_v4.py
# Gets the PlantVillage leaf photos, keeps only tomato leaves (colour photos),
# sorts them into healthy and sick, makes the two groups about the same size,
# and splits them into train (70%), val (15%) and test (15%).
# All photos of the same real leaf stay in the same split, so the test is fair.
# It also picks 12 test photos to print: 6 healthy and 6 sick.
#
# Run it from the repo folder:   python model/split_leaves_v4.py
# Needs:                         pip install huggingface_hub
#
# The dataset's own loader no longer works, so this script downloads its two
# files directly: data.zip (all the photos) and leaf-map.json (which photos
# belong to the same leaf). It works out the leaf of each photo the same way
# the dataset's own loader (plant_village.py) does.

import os
import csv
import json
import random
import zipfile
from huggingface_hub import hf_hub_download

REPO = "mohanty/PlantVillage"
OUT_DIR = "data"          # put "data/" in .gitignore so photos never go to GitHub
RESULTS_DIR = "results"
SEED = 42                 # same seed = same split every time you run it
TRAIN_SHARE = 0.70
VAL_SHARE = 0.15          # whatever is left (15%) goes to test

random.seed(SEED)


def find_leaf_id(class_name, file_name, leaf_map):
    # Copied from the dataset's own loader, so leaf IDs match theirs.
    image_identifier = file_name.replace("_final_masked", "")
    if "___" in image_identifier:
        image_identifier = image_identifier.split("___")[-1]
    image_identifier = image_identifier.split("copy")[0]
    image_identifier = image_identifier.replace(".jpg", "").replace(".JPG", "")
    image_identifier = image_identifier.replace(".png", "").replace(".PNG", "")
    image_identifier = image_identifier.strip()

    lookup_key = image_identifier.lower().strip()
    if lookup_key in leaf_map:
        suggestions = leaf_map[lookup_key]
        if len(suggestions) == 1:
            return suggestions[0]
        for suggestion in suggestions:
            if class_name in suggestion:
                return suggestion
    # No match in the map: treat the photo as its own leaf.
    return "fallback_" + image_identifier


def split_leaf_ids(leaf_id_list):
    # Shuffle the leaves, then cut the list into train, val and test.
    ids = list(leaf_id_list)
    ids.sort()
    random.shuffle(ids)
    train_end = int(len(ids) * TRAIN_SHARE)
    val_end = int(len(ids) * (TRAIN_SHARE + VAL_SHARE))
    return {
        "train": ids[:train_end],
        "val": ids[train_end:val_end],
        "test": ids[val_end:],
    }


# ---------- 1. Download the two files ----------
print("Downloading data.zip (about 2 GB, only the first time)...")
zip_path = hf_hub_download(REPO, "data.zip", repo_type="dataset")
print("Downloading leaf-map.json...")
map_path = hf_hub_download(REPO, "leaf_grouping/leaf-map.json", repo_type="dataset")

with open(map_path, "r") as f:
    leaf_map = json.load(f)

photo_zip = zipfile.ZipFile(zip_path, "r")

# ---------- 2. Keep only colour tomato photos and group them by leaf ----------
# Paths inside the zip look like color/Tomato___Late_blight/abc.JPG
# (sometimes with raw/ in front).
healthy = {}   # leaf_id -> list of paths inside the zip
sick = {}      # disease name -> {leaf_id -> list of paths}
fallback_count = 0

for path in photo_zip.namelist():
    if path.endswith("/"):
        continue   # a folder, not a photo
    parts = path.split("/")
    if "color" not in parts:
        continue
    i = parts.index("color")
    if len(parts) < i + 3:
        continue
    class_name = parts[i + 1]      # e.g. Tomato___Late_blight
    file_name = parts[i + 2]
    if not class_name.startswith("Tomato"):
        continue

    disease = class_name.split("___")[-1]
    leaf = find_leaf_id(class_name, file_name, leaf_map)
    if leaf.startswith("fallback_"):
        fallback_count = fallback_count + 1

    if disease == "healthy":
        if leaf not in healthy:
            healthy[leaf] = []
        healthy[leaf].append(path)
    else:
        if disease not in sick:
            sick[disease] = {}
        if leaf not in sick[disease]:
            sick[disease][leaf] = []
        sick[disease][leaf].append(path)

healthy_photo_count = 0
for leaf in healthy:
    healthy_photo_count = healthy_photo_count + len(healthy[leaf])

total_sick_before = 0
for disease in sick:
    for leaf in sick[disease]:
        total_sick_before = total_sick_before + len(sick[disease][leaf])

print("Tomato photos found: healthy", healthy_photo_count, "| sick", total_sick_before)
print("Diseases found:", sorted(sick.keys()))
print("Photos with no leaf match (treated as their own leaf):", fallback_count)

if healthy_photo_count == 0 or total_sick_before == 0:
    print("No tomato photos found. First 10 paths in the zip:")
    names = photo_zip.namelist()
    for n in range(min(10, len(names))):
        print("  ", names[n])
    print("Send this output to Claude.")
    raise SystemExit

# ---------- 3. Shrink the sick group to about the healthy size ----------
# Take the same number of photos from each disease, so every disease is included.
per_disease_target = healthy_photo_count // len(sick)
kept_sick = {}   # disease -> {leaf_id -> paths}

for disease in sorted(sick.keys()):
    leaf_list = sorted(sick[disease].keys())
    random.shuffle(leaf_list)
    kept_sick[disease] = {}
    count = 0
    for leaf in leaf_list:
        if count >= per_disease_target:
            break
        kept_sick[disease][leaf] = sick[disease][leaf]
        count = count + len(sick[disease][leaf])

# ---------- 4. Split by leaf into train / val / test ----------
# photos_in[split][group] = list of (path in zip, disease)
photos_in = {}
for split in ["train", "val", "test"]:
    photos_in[split] = {"healthy": [], "sick": []}

healthy_split = split_leaf_ids(healthy.keys())
for split in healthy_split:
    for leaf in healthy_split[split]:
        for path in healthy[leaf]:
            photos_in[split]["healthy"].append((path, "healthy"))

# Split each disease on its own, so every split gets every disease.
for disease in sorted(kept_sick.keys()):
    disease_split = split_leaf_ids(kept_sick[disease].keys())
    for split in disease_split:
        for leaf in disease_split[split]:
            for path in kept_sick[disease][leaf]:
                photos_in[split]["sick"].append((path, disease))

# ---------- 5. Copy the photos out of the zip into folders ----------
saved_files = {}   # (split, group) -> list of (file path, disease)
for split in ["train", "val", "test"]:
    for group in ["healthy", "sick"]:
        folder = os.path.join(OUT_DIR, "split", split, group)
        os.makedirs(folder, exist_ok=True)
        saved_files[(split, group)] = []
        print("Saving", split, group, "...")
        number = 0
        for item in photos_in[split][group]:
            zip_name = item[0]
            disease = item[1]
            extension = os.path.splitext(zip_name)[1].lower()
            number = number + 1
            out_path = os.path.join(folder, disease + "_" + str(number) + extension)
            data = photo_zip.read(zip_name)
            with open(out_path, "wb") as out:
                out.write(data)
            saved_files[(split, group)].append((out_path, disease))

# ---------- 6. Pick the 12 photos to print (from test only) ----------
print_dir = os.path.join(OUT_DIR, "print_leaves")
os.makedirs(print_dir, exist_ok=True)
picks = []   # (code, disease, saved file)

healthy_test = list(saved_files[("test", "healthy")])
random.shuffle(healthy_test)
for i in range(6):
    picks.append(("H" + str(i + 1), "healthy", healthy_test[i][0]))

# One sick photo from each of 6 different diseases, so the test is varied.
sick_by_disease = {}
for item in saved_files[("test", "sick")]:
    disease = item[1]
    if disease not in sick_by_disease:
        sick_by_disease[disease] = []
    sick_by_disease[disease].append(item[0])

disease_names = sorted(sick_by_disease.keys())
random.shuffle(disease_names)
for i in range(6):
    disease = disease_names[i % len(disease_names)]
    choice = random.choice(sick_by_disease[disease])
    picks.append(("S" + str(i + 1), disease, choice))

with open(os.path.join(print_dir, "print_leaves.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["code", "disease", "original_file"])
    for pick in picks:
        code = pick[0]
        source = pick[2]
        extension = os.path.splitext(source)[1]
        with open(source, "rb") as src:
            data = src.read()
        with open(os.path.join(print_dir, code + extension), "wb") as dst:
            dst.write(data)
        writer.writerow([code, pick[1], source])

# ---------- 7. Write the counts for the build log ----------
os.makedirs(RESULTS_DIR, exist_ok=True)
lines = []
lines.append("| split | healthy | sick | total |")
lines.append("|---|---|---|---|")
for split in ["train", "val", "test"]:
    h = len(photos_in[split]["healthy"])
    s = len(photos_in[split]["sick"])
    lines.append("| " + split + " | " + str(h) + " | " + str(s) + " | " + str(h + s) + " |")
lines.append("")
lines.append("Source: mohanty/PlantVillage on Hugging Face (data.zip + leaf-map.json), colour photos.")
lines.append("Split by leaf (no leaf in two splits). Seed " + str(SEED) + ".")
lines.append("Photos with no leaf match: " + str(fallback_count) + ".")
lines.append("Print leaves H1-H6 and S1-S6 come from test. Never train on them.")

with open(os.path.join(RESULTS_DIR, "split_counts.md"), "w") as f:
    f.write("\n".join(lines) + "\n")

photo_zip.close()

print()
for line in lines:
    print(line)
print()
print("Done. Copy the table above into docs/build-log.md")
