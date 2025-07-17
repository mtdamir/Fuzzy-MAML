import argparse
import os
import random
import numpy as np
import tensorflow as tf

from data_generator import DataGenerator
from maml import MAML

# --------------------------------------------------------------------------------------
# Utility: robust boolean parser (argparse and bool don't mix well)
# --------------------------------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 't', 'y', '1'):  # English comment: accept truthy forms
        return True
    if v in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


# --------------------------------------------------------------------------------------
# Image loading utilities
# --------------------------------------------------------------------------------------
def load_and_preprocess(img_path, img_size, channels):
    """Load an image from disk, resize to [img_size, img_size], normalize to [0,1], flatten."""
    img_bytes = tf.io.read_file(img_path)
    if channels == 3:
        img = tf.image.decode_jpeg(img_bytes, channels=3)
    else:
        img = tf.image.decode_png(img_bytes, channels=1)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.reshape(img, [-1])  # flatten
    return img


def one_hot(label_idx, num_classes):
    return tf.one_hot(label_idx, num_classes, dtype=tf.float32)


# --------------------------------------------------------------------------------------
# Meta-batch sampling for image datasources (omniglot / miniimagenet)
# --------------------------------------------------------------------------------------
def sample_meta_batch_images(dg, args, train=True):
    """Sample a meta-batch of image classification tasks.

    Returns list of length meta_batch_size; each element is (xa, ya, xb, yb):
        xa: [N*K, dim_input]   support inputs
        ya: [N*K, N]           support labels (one-hot)
        xb: [N*Q, dim_input]   query inputs
        yb: [N*Q, N]           query labels
    """
    folders = dg.metatrain_character_folders if train else dg.metaval_character_folders
    N = args.num_classes
    K = args.update_batch_size
    Q = args.query_batch_size

    # dg.img_size comes from DataGenerator: tuple like (84,84)
    if isinstance(dg.img_size, (tuple, list)):
        img_size = dg.img_size[0]
    else:
        img_size = dg.img_size
    channels = 3 if args.datasource == 'miniimagenet' else 1

    tasks = []
    for _ in range(args.meta_batch_size):
        # sample N distinct classes
        class_folders = random.sample(folders, N)

        support_imgs = []
        support_lbls = []
        query_imgs = []
        query_lbls = []

        for class_idx, class_folder in enumerate(class_folders):
            # collect all image files in this class folder
            img_files = [os.path.join(class_folder, f)
                         for f in os.listdir(class_folder)
                         if os.path.isfile(os.path.join(class_folder, f))]
            # sample (K+Q) images (with replacement if not enough)
            if len(img_files) < K + Q:
                img_files = list(np.random.choice(img_files, size=K+Q, replace=True))
            else:
                img_files = random.sample(img_files, K + Q)
            support_paths = img_files[:K]
            query_paths   = img_files[K:K+Q]

            # load & preprocess support
            for p in support_paths:
                support_imgs.append(load_and_preprocess(p, img_size, channels))
                support_lbls.append(one_hot(class_idx, N))
            # load & preprocess query
            for p in query_paths:
                query_imgs.append(load_and_preprocess(p, img_size, channels))
                query_lbls.append(one_hot(class_idx, N))

        # stack into tensors
        xa = tf.stack(support_imgs, axis=0)
        ya = tf.stack(support_lbls, axis=0)
        xb = tf.stack(query_imgs, axis=0)
        yb = tf.stack(query_lbls, axis=0)
        tasks.append((xa, ya, xb, yb))
    return tasks


# --------------------------------------------------------------------------------------
# Meta-batch sampling for sinusoid regression
# --------------------------------------------------------------------------------------
def sample_meta_batch_sinusoid(dg, args, train=True):
    """Sample sinusoid tasks using DataGenerator.generate_sinusoid_batch.

    We split the generated sequence into support/query the same way: first K
    points per task = support; rest = query.
    """
    # generate() signature: (inputs, outputs, amp, phase)
    batch_x, batch_y, amp, phase = dg.generate_sinusoid_batch(train=train)
    # batch_x shape: [meta_batch_size, num_samples_per_class, 1]
    # batch_y shape: [meta_batch_size, num_samples_per_class, 1]
    # For sinusoid we ignore classes: treat whole sequence as 1 class output.
    K = args.update_batch_size
    N = 1  # single function output

    tasks = []
    for task_idx in range(args.meta_batch_size):
        x_all = batch_x[task_idx]  # [T, 1]
        y_all = batch_y[task_idx]  # [T, 1]
        xa = tf.convert_to_tensor(x_all[:K], dtype=tf.float32)
        ya = tf.convert_to_tensor(y_all[:K], dtype=tf.float32)
        xb = tf.convert_to_tensor(x_all[K:], dtype=tf.float32)
        yb = tf.convert_to_tensor(y_all[K:], dtype=tf.float32)
        tasks.append((xa, ya, xb, yb))
    return tasks


# --------------------------------------------------------------------------------------
# Argparse / main
# --------------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    # data / experiment
    p.add_argument('--datasource', type=str, default='miniimagenet', help='sinusoid | omniglot | miniimagenet')
    p.add_argument('--num_classes', type=int, default=5, help='N-way classification')
    p.add_argument('--meta_batch_size', type=int, default=4, help='tasks per meta-update')
    p.add_argument('--update_batch_size', type=int, default=1, help='K-shot (support) per class')
    p.add_argument('--query_batch_size', type=int, default=15, help='query per class')
    p.add_argument('--num_updates', type=int, default=5, help='inner gradient steps')
    # optimization
    p.add_argument('--meta_lr', type=float, default=0.001, help='outer (meta) learning rate')
    p.add_argument('--update_lr', type=float, default=0.01, help='inner loop step size')
    # model arch
    p.add_argument('--num_filters', type=int, default=64, help='conv filters per layer')
    p.add_argument('--max_pool', type=str2bool, default=True, help='use VALID/max-pool pattern (true/false)')
    p.add_argument('--norm', type=str, default='batch_norm', help='batch_norm | layer_norm | None')
    # misc
    p.add_argument('--test_set', action='store_true', help='sample from held-out test set (else val)')
    p.add_argument('--iters', type=int, default=60000, help='meta-training iterations')
    p.add_argument('--val_interval', type=int, default=1000, help='print/validate every N iters')
    p.add_argument('--seed', type=int, default=42, help='rng seed')
    return p.parse_args()


# --------------------------------------------------------------------------------------
# Validation helper: run one meta-batch in eval mode, report loss/acc
# --------------------------------------------------------------------------------------
def meta_eval_once(maml, dg, args):
    if args.datasource == 'sinusoid':
        batch = sample_meta_batch_sinusoid(dg, args, train=False)
    else:
        batch = sample_meta_batch_images(dg, args, train=False)
    # No weight update here (just forward meta loss) -> use gradient tape but don't apply
    with tf.GradientTape() as tape:
        losses = []
        for (xa, ya, xb, yb) in batch:
            # clone initial weights
            fast = {k: v for k, v in maml.weights.items()}
            for _ in range(maml.num_updates):
                with tf.GradientTape() as inner_tape:
                    inner_tape.watch(list(fast.values()))
                    pred_a = maml.forward(xa, fast)
                    loss_a = maml.loss_fn(pred_a, ya)
                grads = inner_tape.gradient(loss_a, list(fast.values()))
                new_fast = {}
                for (name, w_prev, g) in zip(fast.keys(), fast.values(), grads):
                    if g is None:
                        new_fast[name] = w_prev
                    else:
                        new_fast[name] = w_prev - maml.update_lr * g
                fast = {k: tf.stop_gradient(v) for k, v in new_fast.items()}
            pred_b = maml.forward(xb, fast)
            loss_b = maml.loss_fn(pred_b, yb)
            losses.append(loss_b)
        meta_loss = tf.add_n(losses) / tf.cast(len(batch), tf.float32)
    # optional accuracy (classification only)
    acc = None
    if maml.classification:
        # evaluate last task predictions from loop above? we recompute quickly
        # Simpler: just compute accuracy on the last fast weights of last task
        # Better: aggregate across tasks; we do that here by re‑looping w/out grads.
        correct = 0
        total = 0
        for (xa, ya, xb, yb) in batch:
            fast = {k: v for k, v in maml.weights.items()}
            for _ in range(maml.num_updates):
                pred_a = maml.forward(xa, fast)
                loss_a = maml.loss_fn(pred_a, ya)
                grads = tf.gradients(loss_a, list(fast.values()))  # best effort; some None
                new_fast = {}
                for (name, w_prev, g) in zip(fast.keys(), fast.values(), grads):
                    if g is None:
                        new_fast[name] = w_prev
                    else:
                        new_fast[name] = w_prev - maml.update_lr * g
                fast = new_fast
            pred_b = maml.forward(xb, fast)
            pred_cls = tf.argmax(pred_b, axis=-1)
            true_cls = tf.argmax(yb, axis=-1)
            correct += tf.reduce_sum(tf.cast(tf.equal(pred_cls, true_cls), tf.float32))
            total += tf.cast(tf.size(true_cls), tf.float32)
        acc = correct / total
    return meta_loss, acc


# --------------------------------------------------------------------------------------
# Main training entry point
# --------------------------------------------------------------------------------------

def main():
    args = parse_args()

    # set seeds for reproducibility ----------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # DataGenerator: we pass num_samples_per_class = K+Q just to satisfy constructor ---------\n    # (we do *our own* sampling in sample_meta_batch_* helpers.)
    dg = DataGenerator(num_samples_per_class=args.update_batch_size + args.query_batch_size,
                       batch_size=args.meta_batch_size,
                       args=args)

    # Build MAML model ----------------------------------------------------------------------
    maml = MAML(dim_input=dg.dim_input, dim_output=dg.dim_output, args=args)

    # Training loop -------------------------------------------------------------------------
    for itr in range(args.iters):
        if args.datasource == 'sinusoid':
            batch = sample_meta_batch_sinusoid(dg, args, train=True)
        else:
            batch = sample_meta_batch_images(dg, args, train=True)
        loss = maml.meta_train_step(batch)
        if itr % 100 == 0:
            print(f"Iteration {itr}, meta-loss: {loss.numpy():.6f}")
        if (itr % args.val_interval == 0) and (itr > 0):
            val_loss, val_acc = meta_eval_once(maml, dg, args)
            if val_acc is not None:
                print(f"  [val] loss={val_loss.numpy():.6f}, acc={val_acc.numpy():.4f}")
            else:
                print(f"  [val] loss={val_loss.numpy():.6f}")

    print('Training complete.')


if __name__ == '__main__':
    main()