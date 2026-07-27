"""All settings in one place.

Change a value here rather than hunting for it across files. The two class
maps (12-class and 4-class) are the main thing you switch between.
"""

# data
DATA_DIR = "data"                  # scraper writes here; classifier reads here
DATASET_ROOT = "data/m9-bayonet"   

# Because the original image is a full knife we need to crop the image to have only the blade.
# The parameters below may differ and most likely will differ for other knives if you chose to do them as well.  
CROP_LEFT, CROP_TOP, CROP_BOTTOM_REMOVE = 1321, 329, 234
IMG_H, IMG_W = 64, 338

MAX_PER_CLASS = 10   # images per class
TEST_FRAC = 0.80
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


# Here is another problem defined where we only compare the 4 most similar to each other classes
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
EMBEDDING_DIM = 16   # tested on 8, 16, 32 and 64 but everything above 16 worked too easy

# training
EPOCHS = 10          # doesnt really matter you can train on 1 epoch and it still will have great results
LR = 1e-3
BATCH_SIZE = 32
MARGIN = 1.0         # triplet-loss margin

# classifier
KNN_NEIGHBORS = 3

# scraper
CDN_BASE = "https://cdn.csgoskins.gg/public/images/patterns/v1"
SCRAPE_N_SEEDS = 15        # pattern seeds per variant, you can add to that but the images are quite big.
SCRAPE_DELAY = 0.3          # seconds between requests
SCRAPE_SEED_MAX = 1000      # seeds are drawn from 1 to 1000

# you can scrape images for all knives if you wish
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
