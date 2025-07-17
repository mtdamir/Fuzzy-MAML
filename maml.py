from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf

try:
    import special_grads  # optional / legacy
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined: %s' % e, file=sys.stderr)

from utils import mse, xent, conv_block


class MAML:
    def __init__(self, dim_input=1, dim_output=1, args=None):
        """Initialize MAML hyperparameters and create learnable weights."""
        # ----- core dims -----
        self.dim_input = dim_input    # flattened input size
        self.dim_output = dim_output  # output size (num classes or 1 for regression)

        # ----- hyperparams from args -----
        self.update_lr = float(args.update_lr)
        self.meta_lr = float(args.meta_lr)
        self.num_updates = int(args.num_updates)
        self.meta_batch_size = int(args.meta_batch_size)
        self.num_classes = int(args.num_classes)
        self.datasource = args.datasource
        self.num_filters = int(getattr(args, 'num_filters', 64))
        self.max_pool = bool(getattr(args, 'max_pool', False))
        self.norm = getattr(args, 'norm', 'batch_norm')

        # ----- data‑source specific settings -----
        if self.datasource == 'sinusoid':
            self.classification = False
            self.loss_fn = mse
            self.dim_hidden = [40, 40]
            self.channels = None
            self.img_size = None
        elif self.datasource in ('omniglot', 'miniimagenet'):
            self.classification = True
            self.loss_fn = xent
            self.channels = 3 if self.datasource == 'miniimagenet' else 1
            self.img_size = int(np.sqrt(self.dim_input / self.channels))  # correct!
            self.dim_hidden = self.num_filters  # conv channels per layer
        else:
            raise ValueError('Unrecognized datasource: %s' % self.datasource)

        # ----- create learnable initial weights -----
        self.weights = self._construct_weights()
        # ----- outer optimizer -----
        self.optimizer = tf.keras.optimizers.Adam(self.meta_lr)

    # ==================================================================
    # Weight construction helpers
    # ==================================================================
    def _construct_weights(self):
        if self.datasource == 'sinusoid':
            return self._build_fc(self.dim_hidden)
        else:
            return self._build_conv()

    def _build_fc(self, hidden_dims):
        """Build fully‑connected weight dict."""
        w = {}
        dims = [self.dim_input] + list(hidden_dims) + [self.dim_output]
        for i in range(len(dims) - 1):
            w[f'w{i+1}'] = tf.Variable(
                tf.random.truncated_normal([dims[i], dims[i+1]], stddev=0.01),
                trainable=True)
            w[f'b{i+1}'] = tf.Variable(tf.zeros([dims[i+1]]), trainable=True)
            if i < len(hidden_dims) and self.norm != 'None':
                w[f'norm{i+1}_gamma'] = tf.Variable(tf.ones([dims[i+1]]), trainable=True)
                w[f'norm{i+1}_beta']  = tf.Variable(tf.zeros([dims[i+1]]), trainable=True)
        return w

    def _build_conv(self):
        """Build four convolutional blocks + linear classifier."""
        w = {}
        k = 3
        # conv1 uses input channels; conv2‑4 use num_filters
        w['conv1'] = tf.Variable(
            tf.keras.initializers.GlorotNormal()([k, k, self.channels, self.num_filters]),
            trainable=True)
        w['b1'] = tf.Variable(tf.zeros([self.num_filters]), trainable=True)
        for i in range(2, 5):
            w[f'conv{i}'] = tf.Variable(
                tf.keras.initializers.GlorotNormal()([k, k, self.num_filters, self.num_filters]),
                trainable=True)
            w[f'b{i}'] = tf.Variable(tf.zeros([self.num_filters]), trainable=True)
        if self.norm != 'None':
            for i in range(1, 5):
                w[f'norm{i}_gamma'] = tf.Variable(tf.ones([self.num_filters]), trainable=True)
                w[f'norm{i}_beta']  = tf.Variable(tf.zeros([self.num_filters]), trainable=True)
        # final linear classifier
        w['w5'] = tf.Variable(
            tf.keras.initializers.GlorotNormal()([self.num_filters, self.dim_output]),
            trainable=True)
        w['b5'] = tf.Variable(tf.zeros([self.dim_output]), trainable=True)
        return w

    # ==================================================================
    # Forward networks
    # ==================================================================
    def _maybe_batchnorm(self, x, gamma, beta):
        # simple batch stat BN over all but last dim
        axes = list(range(len(x.shape) - 1))
        mean, var = tf.nn.moments(x, axes=axes)
        return tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)

    def forward_fc(self, x, weights):
        h = x
        num_hidden = len(self.dim_hidden)
        for i in range(1, num_hidden + 1):
            h = tf.matmul(h, weights[f'w{i}']) + weights[f'b{i}']
            if self.norm != 'None':
                h = self._maybe_batchnorm(h, weights[f'norm{i+1}_gamma'], weights[f'norm{i+1}_beta'])
            h = tf.nn.relu(h)
        return tf.matmul(h, weights[f'w{num_hidden+1}']) + weights[f'b{num_hidden+1}']

    def forward_conv(self, x, weights):
        # reshape flat -> image
        x = tf.reshape(x, [-1, self.img_size, self.img_size, self.channels])
        h = x
        for i in range(1, 5):
            h = conv_block(
                h,
                weights[f'conv{i}'],
                weights[f'b{i}'],
                norm_layer=None,
                activation=tf.nn.relu,
                max_pool_pad='VALID' if self.max_pool else 'SAME')
            if self.norm != 'None':
                h = self._maybe_batchnorm(h, weights[f'norm{i}_gamma'], weights[f'norm{i}_beta'])
            h = tf.nn.relu(h)
        h = tf.keras.layers.GlobalAveragePooling2D()(h)
        return tf.matmul(h, weights['w5']) + weights['b5']

    def forward(self, x, weights):
        return self.forward_fc(x, weights) if self.datasource == 'sinusoid' else self.forward_conv(x, weights)

    # ==================================================================
    # Meta‑training step (First‑Order MAML, gradient path preserved!)
    # ==================================================================
    def meta_train_step(self, batch_data):
        """Perform one meta‑training update.

        batch_data: list of (xa, ya, xb, yb) tensors.
        Shapes:
            xa: [K_total, dim_input]
            ya: [K_total, dim_output]
            xb: [Q_total, dim_input]
            yb: [Q_total, dim_output]
        where K_total = num_classes * update_batch_size.
        """
        with tf.GradientTape() as outer_tape:
            meta_losses = []
            for (xa, ya, xb, yb) in batch_data:
                # start inner loop from *initial* weights (tensors); we will update tensor copies
                fast = {k: v for k, v in self.weights.items()}
                for _ in range(self.num_updates):
                    # compute support loss wrt current fast tensors
                    with tf.GradientTape() as inner_tape:
                        # need to watch because `fast` entries become tensors (not Variables) after first update
                        inner_tape.watch(list(fast.values()))
                        pred_a = self.forward(xa, fast)
                        loss_a = self.loss_fn(pred_a, ya)
                    grads = inner_tape.gradient(loss_a, list(fast.values()))
                    # FOMAML: drop 2nd derivatives by detaching *grads*, not weights
                    new_fast = {}
                    for (name, w_prev, g) in zip(fast.keys(), fast.values(), grads):
                        if g is None:
                            new_fast[name] = w_prev  # no update
                        else:
                            new_fast[name] = w_prev - self.update_lr * tf.stop_gradient(g)
                    fast = new_fast  # keep computational link to w_prev!
                # query loss using adapted fast weights
                pred_b = self.forward(xb, fast)
                loss_b = self.loss_fn(pred_b, yb)
                meta_losses.append(loss_b)
            meta_loss = tf.add_n(meta_losses) / tf.cast(len(batch_data), tf.float32)
        # gradient of meta-loss w.r.t. *initial* learnable weights
        grads = outer_tape.gradient(meta_loss, list(self.weights.values()))
        # defensive fallback: replace any None gradient with zeros
        safe_grads = []
        for g, v in zip(grads, self.weights.values()):
            if g is None:
                safe_grads.append(tf.zeros_like(v))
            else:
                safe_grads.append(g)
        self.optimizer.apply_gradients(zip(safe_grads, list(self.weights.values())))
        return meta_loss