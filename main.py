import argparse
import os
import random
import numpy as np
import tensorflow as tf
from data_generator import DataGenerator
from maml import MAML, reliability_image, fuzzy_lr_scaling, task_weight_fuzzy
from rule_extractor import extract_fuzzy_rules
from fuzzy_utils import fuzzy_lr_scaling  # برای scaling تیون شده

# ---------------------------------------------------------------------------
# Utility: robust boolean parser
# ---------------------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 't', 'y', '1'):
        return True
    if v in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

# ---------------------------------------------------------------------------
# Image‑loading helper
# ---------------------------------------------------------------------------
def load_and_preprocess(img_path, img_size, channels):
    """
    Read image from disk, resize to img_size×img_size, scale to [0,1],
    and flatten to 1‑D tensor.
    """
    img_bytes = tf.io.read_file(img_path)
    if channels == 3:
        img = tf.image.decode_jpeg(img_bytes, channels=3)
    else:
        img = tf.image.decode_png(img_bytes, channels=1)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 255.0
    return tf.reshape(img, [-1])   # [dim_input]

# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--datasource',          default='miniimagenet')
parser.add_argument('--num_classes',     type=int,   default=5)
parser.add_argument('--meta_batch_size', type=int,   default=4)
parser.add_argument('--update_batch_size', type=int, default=5)
parser.add_argument('--query_batch_size',  type=int, default=15)
parser.add_argument('--num_updates',      type=int,   default=5)
parser.add_argument('--meta_lr',          type=float, default=1e-3)
parser.add_argument('--update_lr',        type=float, default=1e-2)
parser.add_argument('--iters',            type=int,   default=10000)  # افزایش به 20000 برای آموزش بهتر
parser.add_argument('--seed',             type=int,   default=42)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# ---------------------------------------------------------------------------
# Build data generator and model
# ---------------------------------------------------------------------------
dg   = DataGenerator(args.update_batch_size + args.query_batch_size,
                     args.meta_batch_size, args)
maml = MAML(dg.dim_input, dg.dim_output, args)

# ---------------------------------------------------------------------------
# Helper: sample one meta‑batch with fuzzy reliability tensors
# ---------------------------------------------------------------------------
def sample_meta_batch():
    tasks = []
    folders   = dg.metatrain_character_folders
    N, K, Q   = args.num_classes, args.update_batch_size, args.query_batch_size
    img_size  = dg.img_size[0] if isinstance(dg.img_size, (list, tuple)) else dg.img_size
    channels  = 3 if args.datasource == 'miniimagenet' else 1

    for _ in range(args.meta_batch_size):
        class_folders = random.sample(folders, N)
        xa, ya, ra, xb, yb = [], [], [], [], []

        for cls_idx, cfolder in enumerate(class_folders):
            img_files = [os.path.join(cfolder, f)
                         for f in os.listdir(cfolder)
                         if os.path.isfile(os.path.join(cfolder, f))]
            if len(img_files) < K + Q:
                img_files = list(np.random.choice(img_files, K + Q, replace=True))
            else:
                img_files = random.sample(img_files, K + Q)

            support_paths = img_files[:K]
            query_paths   = img_files[K:K + Q]

            # -------- support --------
            for p in support_paths:
                img = load_and_preprocess(p, img_size, channels)
                xa.append(img)
                ya.append(tf.one_hot(cls_idx, N))
                # reliability per sample (scalar tensor)
                ra.append(reliability_image(img[None])[0])

            # -------- query ---------
            for p in query_paths:
                img = load_and_preprocess(p, img_size, channels)
                xb.append(img)
                yb.append(tf.one_hot(cls_idx, N))

        tasks.append(( 
            tf.stack(xa), tf.stack(ya), tf.stack(ra),   # support + reliability
            tf.stack(xb), tf.stack(yb)                  # query
        ))
    return tasks

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for it in range(args.iters):
    meta_batch = sample_meta_batch()
    meta_loss, meta_acc, _ = maml.meta_train_step(meta_batch)  # حالا meta_acc هم برمی‌گرده

    # --- logging ---
    if it % 100 == 0:
        print(f'Iter {it:6d} │ meta‑loss = {meta_loss.numpy():.4f} │ meta‑acc = {meta_acc.numpy():.4f}')
        # fuzzy adjustment of meta learning-rate
        scale = fuzzy_lr_scaling(meta_loss)
        maml.optimizer.learning_rate.assign(args.meta_lr * scale)

        # استخراج رول‌ها و اعمال آنها بر تسک‌ها
        for (xa, ya, ra, xb, yb) in meta_batch:
            avg_rel = tf.reduce_mean(ra)
            task_weight = task_weight_fuzzy(avg_rel)
            print(f"Avg rel: {avg_rel.numpy():.4f} │ Task Weight: {task_weight.numpy()}")  # اضافه کردن پرینت avg_rel برای دیباگ

        # استخراج قوانین فازی برای تسک‌ها (حالا با FCM)
        fuzzy_rules = extract_fuzzy_rules(xa.numpy(), n_clusters=3)
        print(f"Extracted Fuzzy Rules (FCM-based): {fuzzy_rules}")

print('Training finished.')