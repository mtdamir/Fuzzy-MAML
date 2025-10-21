import sys
import numpy as np
import tensorflow as tf
from utils import mse, xent, conv_block
from fuzzy_utils import fuzzy_lr_scaling  # task_weight_fuzzy حذف
from rule_extractor import extract_fuzzy_rules  # import rule extractor

# ------------------------- Fuzzy helper functions -------------------------
@tf.function
def reliability_image(batch_pixels):
    """Input [B, dim_input] in [0,1]; return [B] in [0,1]."""
    mean_pix = tf.reduce_mean(batch_pixels, axis=-1)  # brightness
    var_pix = tf.math.reduce_variance(batch_pixels, axis=-1)  # variance برای نویز/عدم قطعیت
    return tf.clip_by_value(mean_pix * (1 - 0.5 * var_pix), 0.0, 1.0)

class MAML:
    def __init__(self, dim_input, dim_output, args):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = float(args.update_lr)
        self.meta_lr = float(args.meta_lr)
        self.num_updates = int(args.num_updates)  # حالا 3 در args
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
        
        # Cosine decay scheduler
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.CosineDecay(
                self.meta_lr, decay_steps=args.iters, alpha=0.01
            )
        )

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

    # اضافه: extract intermediate features (برای FCM روی features نه raw)
    def extract_features(self, x, w):
        x = tf.reshape(x, [-1, self.img_size, self.img_size, self.channels])
        h = x
        for i in range(1,5):
            h = conv_block(h, w[f'conv{i}'], w[f'b{i}'], None, tf.nn.relu,
                            'VALID' if self.max_pool else 'SAME')
            h = tf.nn.relu(h)
        h = tf.keras.layers.GlobalAveragePooling2D()(h)  # features قبل FC
        return h

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
        if not self.classification:
            return self.forward_fc(x, w)
        else:
            h = self.extract_features(x, w)  # استفاده از extract_features
            return tf.matmul(h, w['w5']) + w['b5']  # ادامه به logits

    # ---------------- meta‑train (with fuzzy logic) -------------------------------
    def meta_train_step(self, batch):
        with tf.GradientTape() as outer_tape:
            meta_losses = []
            meta_accs = []  # لیست برای acc هر تسک
            task_weights = []
            avg_rels = []  # برای dynamic thresholds
            for (xa, ya, ra, xb, yb) in batch:
                # استخراج features به جای raw xa
                features_a = self.extract_features(xa, self.weights).numpy()  # به numpy برای FCM

                # ادغام FCM: استخراج عضویت‌ها برای وزن‌دهی فازی بیشتر
                _, u = extract_fuzzy_rules(features_a, n_clusters=5, return_u=True)  # n=5
                u_mean = tf.reduce_mean(tf.convert_to_tensor(u, dtype=tf.float32), axis=1)  # میانگین عضویت per sample
                
                # inner loop
                fast = {k:v for k,v in self.weights.items()}
                for _ in range(self.num_updates):
                    with tf.GradientTape() as inner:
                        inner.watch(list(fast.values()))
                        pred_a = self.forward(xa, fast)
                        loss_a_elem = tf.nn.softmax_cross_entropy_with_logits(labels=ya, logits=pred_a)
                        # ترکیب FCM با reliability: نقش فازی داده‌های قدیمی
                        loss_a = tf.reduce_mean(ra * u_mean * loss_a_elem)  # حالا ra * u_mean * elem
                    grads = inner.gradient(loss_a, list(fast.values()))
                    fast = {n: w - self.update_lr*tf.stop_gradient(g) if g is not None else w
                            for (n,w,g) in zip(fast.keys(), fast.values(), grads)}

                # query loss
                pred_b = self.forward(xb, fast)
                loss_b = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yb, logits=pred_b))
                
                # محاسبه acc روی query set
                pred_classes = tf.argmax(pred_b, axis=1)
                true_classes = tf.argmax(yb, axis=1)
                acc_b = tf.reduce_mean(tf.cast(tf.equal(pred_classes, true_classes), tf.float32))
                
                # جمع avg_rel برای dynamic
                avg_rel = tf.reduce_mean(ra)
                avg_rels.append(avg_rel)
                
                # dynamic task_weight_fuzzy
                if avg_rels:  # بعد اولین
                    rel_mean = tf.reduce_mean(avg_rels)
                    rel_std = tf.math.reduce_std(avg_rels)
                    low_thresh = rel_mean - rel_std
                    high_thresh = rel_mean
                    task_weight = tf.cond(avg_rel < low_thresh, lambda: 0.8,
                                          lambda: tf.cond(avg_rel < high_thresh, lambda: 0.5, lambda: 0.2))
                else:
                    task_weight = 0.5  # default برای اولین
                
                meta_losses.append(task_weight * loss_b)
                meta_accs.append(acc_b)  # acc بدون وزن‌دهی (برای لاگ میانگین ساده)
                task_weights.append(task_weight)

            meta_loss = tf.add_n(meta_losses) / tf.cast(len(meta_losses), tf.float32)
            meta_acc = tf.reduce_mean(meta_accs)  # میانگین acc تسک‌ها

        # outer gradients
        grads = outer_tape.gradient(meta_loss, list(self.weights.values()))
        grads = [g if g is not None else tf.zeros_like(v) for g,v in zip(grads, self.weights.values())]
        
        # fuzzy scaling on grads
        scale = fuzzy_lr_scaling(meta_loss)
        grads = [g * scale if g is not None else None for g in grads]
        
        self.optimizer.apply_gradients(zip(grads, self.weights.values()))

        # return meta_loss, meta_acc, mean_task_weights برای logging
        return meta_loss, meta_acc, tf.reduce_mean(task_weights)