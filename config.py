"""All settings in one place.

Change a value here rather than hunting for it across files. The two class
maps (12-class and 4-class) are the main thing you switch between.
"""

# data
DATA_DIR = "data"                  # scraper writes here; classifier reads here
DATASET_ROOT = "data/m9-bayonet"   # folder of <finish>/<phase>/*.png

# The M9 Bayonet blade sits in a fixed region of each render, so a fixed crop
# isolates the finish and drops the background. (left, top, bottom-to-remove)
CROP_LEFT, CROP_TOP, CROP_BOTTOM_REMOVE = 1321, 329, 234
IMG_H, IMG_W = 64, 338

MAX_PER_CLASS = 30   # cap images per class; the whole point is "little data"
TEST_FRAC = 0.20
SEED = 42

# classes
# Every Doppler and Gamma Doppler finish.
LABELS_12 = {
    ("doppler", "phase-1"): 0,  ("doppler", "phase-2"): 1,
    ("doppler", "phase-3"): 2,  ("doppler", "phase-4"): 3,
    ("doppler", "ruby"):    4,  ("doppler", "sapphire"): 5,
    ("doppler", "black-pearl"): 6,
    ("gamma-doppler", "phase-1"): 7,  ("gamma-doppler", "phase-2"): 8,
    ("gamma-doppler", "phase-3"): 9,  ("gamma-doppler", "phase-4"): 10,
    ("gamma-doppler", "emerald"): 11,
}


# Only gamma doppler phase 3 and 4 and doppler phase 3 and 4 are hard to tell apart so that is what we are focusing on.
LABELS_4 = {
    ("doppler",       "phase-3"): 0,
    ("doppler",       "phase-4"): 1,
    ("gamma-doppler", "phase-3"): 2,
    ("gamma-doppler", "phase-4"): 3,
}

# Pick which task to run.
LABELS = LABELS_12          # or LABELS_4
IDX_TO_LABEL = {v: f"{k[0]} / {k[1]}" for k, v in LABELS.items()}
NUM_CLASSES = len(LABELS)

# model
EMBEDDING_DIM = 16   # small on purpose, 64 worked too but was "too easy"

# training
EPOCHS = 10          # more makes no real difference
LR = 1e-3
BATCH_SIZE = 32
MARGIN = 1.0         # triplet-loss margin

# classifier
KNN_NEIGHBORS = 3

# scraper
# csgoskins.gg serves pattern renders straight from a CDN, no scraping of the
# page needed: <CDN>/<knife>-<finish>-<variant>/<seed>.png
CDN_BASE = "https://cdn.csgoskins.gg/public/images/patterns/v1"
SCRAPE_N_SEEDS = 15        # pattern seeds per variant, you can add to that but the images are quite big.
SCRAPE_DELAY = 0.3          # seconds between requests — be polite to the CDN
SCRAPE_SEED_MAX = 1000      # seeds are drawn from 1 to 1000

# Which knives to pull. The classifier only uses m9-bayonet, but the CDN has
# the same finishes for every knife, so the scraper can fetch the lot.
SCRAPE_KNIVES = [
    "m9-bayonet"
    #, "karambit", "butterfly-knife", "gut-knife", "flip-knife",
    # "ursus-knife", "survival-knife", "skeleton-knife", "nomad-knife",
    # "stiletto-knife", "navaja-knife", "talon-knife", "huntsman-knife",
    # "shadow-daggers", "bowie-knife", "falchion-knife", "paracord-knife", # if you like you can dowload all of the knives
]

# Variants of doppler and gamma doppler finishes.
SCRAPE_VARIANTS = {
    "doppler": ["phase-1", "phase-2", "phase-3", "phase-4",
                "ruby", "sapphire", "black-pearl"],
    "gamma-doppler": ["phase-1", "phase-2", "phase-3", "phase-4", "emerald"],
}

# output
OUTPUT_DIR = "outputs"
