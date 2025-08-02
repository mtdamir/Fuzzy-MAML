import sys
import numpy as np
import tensorflow as tf
from utils import mse, xent, conv_block
from fuzzy_utils import fuzzy_lr_scaling, task_weight_fuzzy  # import fuzzy functions
from rule_extractor import extract_fuzzy_rules  # import rule extractor

try:
    import special_grads
except KeyError as e:
    print("WARN: Cannot define MaxPoolGrad: %s" % e, file=sys.stderr)

# ------------------------- Fuzzy helper functions -------------------------
# sample‑level reliability for images: simple brightness rule (0=dark → low r)
@tf.function
def reliability_image(batch_pixels):
    """Input [B, dim_input] in [0,1]; return [B] in [0,1]."""
    mean_pix = tf.reduce_mean(batch_pixels, axis=-1)  # brightness
    return tf.clip_by_value(mean_pix, 0.0, 1.0)

# task‑level importance fuzzy rule (low/med/high based on avg reliability)
@tf.function
def task_weight_fuzzy(avg_rel):
    # triangular membership
    if avg_rel < 0.3:
        return 0.8
    elif avg_rel < 0.6:
        return 0.5
    else:
        return 0.2

# meta‑lr scaling (same as before)
@tf.function
def fuzzy_lr_scaling(avg_loss):
    if avg_loss < 0.5:
        return 1.2
    elif avg_loss < 1.0:
        return 1.0
    else:
        return 0.8

class MAML:
    def __init__(self, dim_input, dim_output, args):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = float(args.update_lr)
        self.meta_lr = float(args.meta_lr)
        self.num_updates = int(args.num_updates)
        self.meta_batch_size = int(args.meta_batch_size)
        self.num_classes = int(args.num_classes)
        self.datasource = args.datasource
        self.num_filters = int(getattr(args, 'num_filters', 64))
        self.max_pool = bool(getattr(args, 'max_pool', False))
        self.norm = getattr(args, 'norm', 'batch_norm')

        if self.datasource == 'sinusoid':
            self.classification = False
            self.loss_fn = mse
            self.dim_hidden = [40, 40]
            self.channels = None
        else:
            self.classification = True
            self.loss_fn = xent
            self.channels = 3 if self.datasource == 'miniimagenet' else 1
            self.img_size = int(np.sqrt(self.dim_input / self.channels))
            self.dim_hidden = self.num_filters

        self.weights = self._init_weights()
        self.optimizer = tf.keras.optimizers.Adam(self.meta_lr)

    # ---------------- weight init (same as previous) -------------------------
    def _init_weights(self):
        if self.datasource == 'sinusoid':
            return self._build_fc(self.dim_hidden)
        return self._build_conv()

    def _build_fc(self, hidden):
        w = {}
        dims = [self.dim_input] + hidden + [self.dim_output]
        for i in range(len(dims)-1):
            w[f'w{i+1}'] = tf.Variable(tf.random.truncated_normal([dims[i], dims[i+1]], stddev=0.01))
            w[f'b{i+1}'] = tf.Variable(tf.zeros([dims[i+1]]))
        return w

    def _build_conv(self):
        w, k = {}, 3
        w['conv1'] = tf.Variable(tf.keras.initializers.GlorotNormal()([k,k,self.channels,self.num_filters]))
        w['b1'] = tf.Variable(tf.zeros([self.num_filters]))
        for i in range(2,5):
            w[f'conv{i}'] = tf.Variable(tf.keras.initializers.GlorotNormal()([k,k,self.num_filters,self.num_filters]))
            w[f'b{i}'] = tf.Variable(tf.zeros([self.num_filters]))
        w['w5'] = tf.Variable(tf.keras.initializers.GlorotNormal()([self.num_filters, self.dim_output]))
        w['b5'] = tf.Variable(tf.zeros([self.dim_output]))
        return w

    # ---------------- forward (same as previous) -------------------------------
    def forward_fc(self, x, w):
        h = tf.nn.relu(tf.matmul(x, w['w1']) + w['b1'])
        h = tf.nn.relu(tf.matmul(h, w['w2']) + w['b2'])
        return tf.matmul(h, w['w3']) + w['b3']

    def forward_conv(self, x, w):
        x = tf.reshape(x, [-1, self.img_size, self.img_size, self.channels])
        h = x
        for i in range(1,5):
            h = conv_block(h, w[f'conv{i}'], w[f'b{i}'], None, tf.nn.relu,
                            'VALID' if self.max_pool else 'SAME')
            h = tf.nn.relu(h)
        h = tf.keras.layers.GlobalAveragePooling2D()(h)
        return tf.matmul(h, w['w5']) + w['b5']

    def forward(self, x, w):
        return self.forward_fc(x, w) if not self.classification else self.forward_conv(x, w)

    # ---------------- meta‑train (with fuzzy logic) -------------------------------
    def meta_train_step(self, batch):
        with tf.GradientTape() as outer_tape:
            meta_losses = []
            task_weights = []
            for (xa, ya, ra, xb, yb) in batch:
                # inner loop
                fast = {k:v for k,v in self.weights.items()}
                for _ in range(self.num_updates):
                    with tf.GradientTape() as inner:
                        inner.watch(list(fast.values()))
                        pred_a = self.forward(xa, fast)
                        loss_a_elem = tf.nn.softmax_cross_entropy_with_logits(labels=ya, logits=pred_a)
                        loss_a = tf.reduce_mean(ra * loss_a_elem)  # reliability weighting
                    grads = inner.gradient(loss_a, list(fast.values()))
                    fast = {n: w - self.update_lr*tf.stop_gradient(g) if g is not None else w
                            for (n,w,g) in zip(fast.keys(), fast.values(), grads)}

                # query loss
                pred_b = self.forward(xb, fast)
                loss_b = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yb, logits=pred_b))
                
                # fuzzy task importance based on avg reliability
                task_weight = task_weight_fuzzy(tf.reduce_mean(ra))
                meta_losses.append(task_weight * loss_b)
                task_weights.append(task_weight)

            meta_loss = tf.add_n(meta_losses) / tf.cast(len(meta_losses), tf.float32)

        # outer gradients
        grads = outer_tape.gradient(meta_loss, list(self.weights.values()))
        grads = [g if g is not None else tf.zeros_like(v) for g,v in zip(grads, self.weights.values())]
        self.optimizer.apply_gradients(zip(grads, self.weights.values()))

        # return meta_loss for logging
        return meta_loss, tf.reduce_mean(task_weights)
