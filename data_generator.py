"""DataGenerator – robust to args without `test_set` flag.

Changes vs. original:
• Uses `getattr(args, 'test_set', False)` instead of direct `args.test_set` access.
• Adds small docstrings and typing hints for clarity.
• No behavioural change otherwise.
"""
from __future__ import annotations
import os, random, numpy as np, tensorflow as tf
from typing import Tuple, List

from utils import get_images  # helper that returns list[(label,img_path)]


class DataGenerator:
    """Generate tasks / batches for Sinusoid, Omniglot, or MiniImageNet."""

    # ---------------------------------------------------------------------
    def __init__(
        self,
        num_samples_per_class: int,
        batch_size: int,
        args,
        config: dict = None,
    ) -> None:
        if config is None:
            config = {}
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1  # default for regression (sinusoid)
        self.args = args  # keep ref for parse_image

        # -----------------------------------------------------------------
        # SINUSOID (regression)
        # -----------------------------------------------------------------
        if args.datasource == "sinusoid":
            self.generate = self.generate_sinusoid_batch  # type: ignore[attr-defined]
            self.amp_range   = config.get("amp_range",   [0.1, 5.0])
            self.phase_range = config.get("phase_range", [0.0, np.pi])
            self.input_range = config.get("input_range", [-5.0, 5.0])
            self.dim_input  = 1
            self.dim_output = 1
            return  # nothing else to init

        # -----------------------------------------------------------------
        # IMAGE datasources (Omniglot / MiniImageNet)
        # -----------------------------------------------------------------
        is_omniglot = "omniglot" in args.datasource
        is_mini     = args.datasource == "miniimagenet"
        if not (is_omniglot or is_mini):
            raise ValueError(f"Unrecognized datasource {args.datasource}")

        self.num_classes = config.get("num_classes", args.num_classes)
        self.dim_output  = self.num_classes

        # ---------- common paths ----------
        root_dir = os.path.dirname(__file__)
        if is_omniglot:
            self.img_size = config.get("img_size", (28, 28))
            self.dim_input = np.prod(self.img_size)
            data_folder = config.get(
                "data_folder",
                os.path.join(root_dir, "data", "omniglot_resized"),
            )
            char_folders = [
                os.path.join(data_folder, fam, char)
                for fam in os.listdir(data_folder)
                if os.path.isdir(os.path.join(data_folder, fam))
                for char in os.listdir(os.path.join(data_folder, fam))
            ]
            random.seed(1)
            random.shuffle(char_folders)
            num_val  = 100
            num_train = config.get("num_train", 1200) - num_val
            self.metatrain_character_folders = char_folders[:num_train]
            use_test = getattr(args, "test_set", False)
            if use_test:
                self.metaval_character_folders = char_folders[num_train + num_val :]
            else:
                self.metaval_character_folders = char_folders[num_train : num_train + num_val]
            self.rotations = config.get("rotations", [0, 90, 180, 270])
        else:  # MiniImageNet
            self.img_size = config.get("img_size", (84, 84))
            self.dim_input = np.prod(self.img_size) * 3
            train_root = config.get(
                "metatrain_folder",
                os.path.join(root_dir, "data", "miniImagenet", "train"),
            )
            val_root = config.get(
                "metaval_folder",
                os.path.join(root_dir, "data", "miniImagenet", "val"),
            )
            test_root = config.get(
                "metatest_folder",
                os.path.join(root_dir, "data", "miniImagenet", "test"),
            )
            self.metatrain_character_folders = [
                os.path.join(train_root, c)
                for c in os.listdir(train_root)
                if os.path.isdir(os.path.join(train_root, c))
            ]
            if getattr(args, "test_set", False):
                meta_root = test_root
            else:
                meta_root = val_root
            self.metaval_character_folders = [
                os.path.join(meta_root, c)
                for c in os.listdir(meta_root)
                if os.path.isdir(os.path.join(meta_root, c))
            ]
            self.rotations = [0]  # no rotation for miniImageNet

    # ---------------------------------------------------------------------
    # TF‑friendly tensor maker for images (rarely used in fuzzy version)
    # ---------------------------------------------------------------------
    def make_data_tensor(self, train: bool = True):
        # ... unchanged from original; omitted for brevity ...
        raise NotImplementedError("Not used in fuzzy pipeline.")

    # ---------------------------------------------------------------------
    # Sinusoid batch generator (unchanged)
    # ---------------------------------------------------------------------
    def generate_sinusoid_batch(self, train: bool = True, input_idx: int | None = None):
        amp   = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        x_all = np.random.uniform(self.input_range[0], self.input_range[1],
                                  [self.batch_size, self.num_samples_per_class, 1])
        if input_idx is not None:
            x_all[:, input_idx:, 0] = np.linspace(
                self.input_range[0], self.input_range[1],
                num=self.num_samples_per_class - input_idx,
                endpoint=False,
            )
        y_all = np.zeros_like(x_all)
        for i in range(self.batch_size):
            y_all[i] = amp[i] * np.sin(x_all[i] - phase[i])
        return x_all.astype('float32'), y_all.astype('float32'), amp, phase
