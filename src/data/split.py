import os
import shutil
import random

SOURCE_DIR = "data/processed/brain_tumor"
SPLIT_DIR = "data/splits/brain_tumor"
VAL_SPLIT = 0.2
SEED = 42


def split():
    source_train = os.path.join(SOURCE_DIR, "train")
    source_test = os.path.join(SOURCE_DIR, "test")

    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)

    split_train = os.path.join(SPLIT_DIR, "train")
    split_val = os.path.join(SPLIT_DIR, "val")
    split_test = os.path.join(SPLIT_DIR, "test")

    os.makedirs(split_train, exist_ok=True)
    os.makedirs(split_val, exist_ok=True)
    os.makedirs(split_test, exist_ok=True)

    rng = random.Random(SEED)

    for cls in os.listdir(source_train):
        cls_source = os.path.join(source_train, cls)
        if not os.path.isdir(cls_source):
            continue

        images = [img for img in os.listdir(cls_source) if os.path.isfile(os.path.join(cls_source, img))]
        rng.shuffle(images)

        split_idx = int(len(images) * VAL_SPLIT)
        val_images = set(images[:split_idx])

        os.makedirs(os.path.join(split_train, cls), exist_ok=True)
        os.makedirs(os.path.join(split_val, cls), exist_ok=True)

        for img in images:
            src = os.path.join(cls_source, img)
            if img in val_images:
                dst = os.path.join(split_val, cls, img)
            else:
                dst = os.path.join(split_train, cls, img)
            shutil.copy2(src, dst)

    if os.path.exists(source_test):
        for cls in os.listdir(source_test):
            cls_source = os.path.join(source_test, cls)
            if not os.path.isdir(cls_source):
                continue
            os.makedirs(os.path.join(split_test, cls), exist_ok=True)
            for img in os.listdir(cls_source):
                src = os.path.join(cls_source, img)
                if os.path.isfile(src):
                    dst = os.path.join(split_test, cls, img)
                    shutil.copy2(src, dst)

    print("Validation split created")


if __name__ == "__main__":
    split()