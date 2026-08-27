
#cfg mifliore= 13.5 (nzomma), nstart è stato testato con tutte le config aventi cgf=13.5 (cfg era stato testato con nfe=33 ed nstart=0),
# nstart migliore penso sia lo  tra 0, 4 e 8, nfe penso sia il 200, se la gioca col nostro 33 originale (nfe testato con cfg=13.5 e n_staart=0)


from pathlib import Path
import copy
import random
import time
import os
import json
import sys
import torch
from pathlib import Path
from torchvision.utils import save_image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, ConcatDataset, Dataset
#from torchvision.models import mobilenet_v2
from torchvision.models import resnet18

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)


# 1. PATH MANAGEMENT
# Get the absolute path of the current directory
PROJECT_DIR = Path.cwd().resolve()

# If Jupyter started from the root folder (/workspace), force entry into seai_project
if PROJECT_DIR.name != "seai_project":
    PROJECT_DIR = PROJECT_DIR / "seai_project"

BASE_DIR = PROJECT_DIR / "dataset"
JSON_PATH = BASE_DIR / "captions/blip_qwen_captions_flowAlign.json"
FLOWALIGN_IN_DIR = BASE_DIR / "flowalign_inputs"

# Folder where we will save the generated variants
FLOWALIGN_OUT_DIR = BASE_DIR / "flowalign_outputs" / "edited"
FLOWALIGN_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Absolute and verified path for FlowAlign
FLOWALIGN_PATH = PROJECT_DIR / "FlowAlign"

# Insert the path at the TOP of the list (index 0) to give it absolute priority
if str(FLOWALIGN_PATH) not in sys.path:
    sys.path.insert(0, str(FLOWALIGN_PATH))


# --- SETUP PATHS ---
#BASE_DIR = Path.cwd() / "dataset"
DATA_ROOT = BASE_DIR / "oxford_pets"
OUTPUT_DIR = BASE_DIR / "classifier_outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("DATA_ROOT:", DATA_ROOT.resolve())
print("OUTPUT_DIR:", OUTPUT_DIR.resolve())









IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
SEED = 42

FLOWALIGN_IMG_SIZE = 1024 

# Classifier params
NUM_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# --- DATASET SPLIT PARAMS ---
# Total available in the dataset: ~4970 dogs, ~2370 cats
N_MAJORITY_TRAIN = 1000  # Dogs
N_MINORITY_TRAIN = 100   # Cats (10:1 Imbalance)
N_VAL_PER_CLASS = 300    # Balanced (300 dogs, 300 cats)
N_TEST_PER_CLASS = 600   # Balanced (600 dogs, 600 cats)

# 0=Dog (Majority), 1=Cat (Minority)
class_names = ["dog", "cat"]
num_classes = 2

# --- TRANSFORMS ---
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

print("Classes:", class_names)
print(f"Train split: {N_MAJORITY_TRAIN} Majority, {N_MINORITY_TRAIN} Minority")




from pathlib import Path
import shutil

folder = Path("/workspace/seai_project/dataset/flowAl_testImages2")

for item in folder.iterdir():
    if item.is_dir():
        shutil.rmtree(item)
    else:
        item.unlink()

print("Cartella svuotata:", folder)





import random
import json
import torch

from diffusion.editing.sd3_edit import get_editor
from utils import util
from torchvision.utils import save_image

# ============================================================
# OUTPUT TEST
# ============================================================

TEST_OUT_DIR = BASE_DIR / "flowAl_testImages2"
TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD EXISTING JSON
# ============================================================

with open(JSON_PATH, "r", encoding="utf-8") as f:
    caption_data = json.load(f)

# Consideriamo solo immagini con 9 varianti
eligible = [
    item for item in caption_data
    if len(item["variants"]) == 9
    and (FLOWALIGN_IN_DIR / item["image_id"]).exists()
]

item = random.choice(eligible)
# per scegliere l'immgine invece di prenderla a caso
#item = next(x for x in eligible if x["image_id"] == "pet_train_328.jpg")

image_id = item["image_id"]
src_prompt = item["original_caption"]
variants = item["variants"]

print("Selected image:", image_id)
print("Source prompt:", src_prompt)
print("Number of variants:", len(variants))

# ============================================================
# LOAD FLOWALIGN
# ============================================================

print("\nLoading FlowAlign...")

sampler = get_editor("flowalign")
sampler = sampler.to(device="cuda")

print("FlowAlign loaded.\n")

# ============================================================
# SOURCE IMAGE
# ============================================================

img_path = str(FLOWALIGN_IN_DIR / image_id)

src_img = util.load_img(
    img_path,
    img_size=(FLOWALIGN_IMG_SIZE, FLOWALIGN_IMG_SIZE)
)

src_img = src_img * 2.0 - 1.0

# ============================================================
# THREE PARAMETER SETS
# ============================================================

parameter_sets = [
    {"name": "nfe33_cfg8_nstart0", "NFE": 33, "n_start": 0, "cfg_scale": 8.0},
    {"name": "nfe33_cfg10_nstart0", "NFE": 33, "n_start": 0, "cfg_scale": 10.0},
    {"name": "original_33_8_8", "NFE": 33, "n_start": 8, "cfg_scale": 8.0},
    {"name": "nfe33_cfg10_nstart8", "NFE": 33, "n_start": 8, "cfg_scale": 10.0}
]

# ============================================================
# GENERATION
# ============================================================

total_generated = 0

for params in parameter_sets:

    # sottocartella diversa per ogni configurazione
    current_out_dir = TEST_OUT_DIR / params["name"]
    current_out_dir.mkdir(parents=True, exist_ok=True)

    print("\n========================================")
    print("PARAMETER SET:", params["name"])
    print("NFE:", params["NFE"])
    print("n_start:", params["n_start"])
    print("cfg_scale:", params["cfg_scale"])
    print("========================================")

    for var_idx, tgt_prompt in enumerate(variants):

        print(
            f" -> Variant {var_idx+1}/{len(variants)}: "
            f"{tgt_prompt}"
        )

        try:

            output = sampler.sample(
                src_img=src_img,
                src_prompt=src_prompt,
                tgt_prompt=tgt_prompt,
                null_prompt="",

                NFE=params["NFE"],
                img_shape=(FLOWALIGN_IMG_SIZE, FLOWALIGN_IMG_SIZE),
                n_start=params["n_start"],
                cfg_scale=params["cfg_scale"],

                src_prompt_emb=None,
                tgt_prompt_emb=None,
                null_prompt_emb=None
            )

            output_file = (
                current_out_dir /
                f"{image_id.split('.')[0]}_var{var_idx+1}.jpg"
            )

            save_image(
                output,
                output_file,
                normalize=True
            )

            total_generated += 1

            print("    ✅ saved")

        except Exception as e:
            print(f"    ❌ ERROR: {e}")

print("\nFinished.")
print("Total generated:", total_generated)

torch.cuda.empty_cache()