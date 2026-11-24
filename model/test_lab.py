# test_lab.py
# Runs the trained leaf model ONCE on the held-out test photos (lab photos).
# Saves:
#   results/lab_test.txt       accuracy and counts
#   results/confusion_lab.png  confusion matrix picture
#   results/lab_mistakes.csv   every photo the model got wrong
#
# Run it from the repo folder:   python model/test_lab.py

import os
import csv
import json
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")   # save pictures without opening a window
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join("model", "leaf_model.keras")
NAMES_PATH = os.path.join("model", "class_names.json")
TEST_DIR = os.path.join("data", "split", "test")
RESULTS_DIR = "results"
IMAGE_SIZE = (224, 224)

os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------- 1. Load the model and the class names ----------
print("Loading the model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(NAMES_PATH, "r") as f:
    class_names = json.load(f)   # ["healthy", "sick"]

# ---------- 2. Load the test photos (no shuffling, no changes) ----------
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=32,
    label_mode="binary",
    shuffle=False,
)
file_paths = test_ds.file_paths

# ---------- 3. Ask the model about every test photo ----------
true_labels = []
predicted_labels = []
confidences = []   # model's "sick" score, 0 to 1

for images, labels in test_ds:
    scores = model.predict(images, verbose=0)
    for i in range(len(scores)):
        sick_score = float(scores[i][0])
        true_labels.append(int(labels[i][0]))
        confidences.append(sick_score)
        if sick_score >= 0.5:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)

# ---------- 4. Count right and wrong answers ----------
# matrix[actual][predicted]: 0 = healthy, 1 = sick
matrix = [[0, 0], [0, 0]]
correct = 0
mistakes = []
for i in range(len(true_labels)):
    actual = true_labels[i]
    guess = predicted_labels[i]
    matrix[actual][guess] = matrix[actual][guess] + 1
    if actual == guess:
        correct = correct + 1
    else:
        mistakes.append([file_paths[i], class_names[actual], class_names[guess], round(confidences[i], 3)])

total = len(true_labels)
accuracy = correct / total

lines = []
lines.append("Lab test (held-out PlantVillage photos)")
lines.append("Photos tested: " + str(total))
lines.append("Correct: " + str(correct))
lines.append("Accuracy: " + str(round(accuracy * 100, 1)) + "%")
lines.append("Healthy photos called sick: " + str(matrix[0][1]))
lines.append("Sick photos called healthy: " + str(matrix[1][0]))

with open(os.path.join(RESULTS_DIR, "lab_test.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

with open(os.path.join(RESULTS_DIR, "lab_mistakes.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "actual", "predicted", "sick_score"])
    for row in mistakes:
        writer.writerow(row)

# ---------- 5. Draw the confusion matrix ----------
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(matrix, cmap="Greens")
biggest = max(matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1])
for actual in range(2):
    for guess in range(2):
        value = matrix[actual][guess]
        # white text on dark squares, black text on light squares
        if value > biggest / 2:
            colour = "white"
        else:
            colour = "black"
        ax.text(guess, actual, str(value),
                ha="center", va="center", fontsize=16, color=colour)
ax.set_xticks([0, 1])
ax.set_xticklabels(class_names)
ax.set_yticks([0, 1])
ax.set_yticklabels(class_names)
ax.set_xlabel("Model said")
ax.set_ylabel("Actually was")
ax.set_title("Lab test: " + str(round(accuracy * 100, 1)) + "% correct")
fig.savefig(os.path.join(RESULTS_DIR, "confusion_lab.png"), dpi=120, bbox_inches="tight")

print()
for line in lines:
    print(line)
print()
print("Saved results/lab_test.txt, results/confusion_lab.png and results/lab_mistakes.csv")
