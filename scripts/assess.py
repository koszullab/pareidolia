import itertools as it
import pathlib
import numpy as np
import pandas as pd
import pareidolia.hic_utils as pah
import cooler
import matplotlib.pyplot as plt


DATA = pathlib.Path("data_test")
# Synthetic matrices and their known loop coordinates
COOLS = [str(c) for c in DATA.glob("B_[1-6]*.cool")]
LOOPS = np.loadtxt(DATA / "B_loops.txt")
# Matrices with a diagonal gradient
COOLS_COMP = [str(c) for c in DATA.glob("smooth_[1-6]*.cool")]


# Run loop change detection between matrices with and without loops
cools = COOLS + COOLS_COMP
conds = ["B"] * len(COOLS) + ["S"] * len(COOLS_COMP)
obs_pos = pah.change_detection_pipeline(
    cools,
    conds,
    min_dist=50000,
    subsample=False,
)
# Build a set of fuzzy (+/3 pixels around) positions found
fuzzy_obs = set()
for pos in obs_pos.loc[:, ["bin1", "bin2"]].values:
    for shift in it.product(range(-3, 4), range(-3, 4)):
        fuzzy_obs.add((pos[0] + shift[0], pos[1] + shift[1]))
# Same for targets
valid_pos = set()
for pos in LOOPS:
    for shift in it.product(range(-3, 4), range(-3, 4)):
        valid_pos.add((pos[0] + shift[0], pos[1] + shift[1]))
# Count the number of real loop positions that were found
found = 0
for target in LOOPS:
    if tuple(target.astype(int)) in fuzzy_obs:
        found += 1
# Compute number of detections that are correct
correct = 0
for obs in obs_pos.loc[:, ["bin1", "bin2"]].values:
    if tuple(obs.astype(int)) in valid_pos:
        correct += 1

print(f"Found: {obs_pos.shape[0]}, targets: {LOOPS.shape[0]}")
print(f"Recall: {found / LOOPS.shape[0]}")
print(f"Precision: {correct / obs_pos.shape[0]}")


fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
ax[0].imshow(
    np.log(cooler.Cooler(COOLS_COMP[0]).matrix(balance=True)[:]), cmap="afmhot_r"
)
ax[1].imshow(np.log(cooler.Cooler(COOLS[0]).matrix(balance=True)[:]), cmap="afmhot_r")
ax[1].scatter(obs_pos.bin1, obs_pos.bin2, c="r")
ax[1].scatter([c[0] for c in LOOPS], [c[1] for c in LOOPS], c="g", marker="x")
plt.show()
