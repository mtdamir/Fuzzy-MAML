""" Code for loading data. """
import numpy as np
import os
import random
import tensorflow as tf

from utils import get_images

class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, num_samples_per_class, batch_size, args, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
            args: parsed command-line arguments
        """
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)
        self.args = args  # ذخیره args به عنوان ویژگی کلاس

        if args.datasource == 'sinusoid':
            self.generate = self.generate_sinusoid_batch
            self.amp_range = config.get('amp_range', [0.1, 5.0])
            self.phase_range = config.get('phase_range', [0, np.pi])
            self.input_range = config.get('input_range', [-5.0, 5.0])
            self.dim_input = 1
            self.dim_output = 1
        elif 'omniglot' in args.datasource:
            self.num_classes = config.get('num_classes', args.num_classes)
            self.img_size = config.get('img_size', (28, 28))
            self.dim_input = np.prod(self.img_size)
            self.dim_output = self.num_classes
            data_folder = config.get('data_folder', os.path.join(os.path.dirname(__file__), 'data', 'omniglot_resized'))

            character_folders = [os.path.join(data_folder, family, character)
                for family in os.listdir(data_folder)
                if os.path.isdir(os.path.join(data_folder, family))
                for character in os.listdir(os.path.join(data_folder, family))]
            random.seed(1)
            random.shuffle(character_folders)
            num_val = 100
            num_train = config.get('num_train', 1200) - num_val
            self.metatrain_character_folders = character_folders[:num_train]
            if args.test_set:
                self.metaval_character_folders = character_folders[num_train + num_val:]
            else:
                self.metaval_character_folders = character_folders[num_train:num_train + num_val]
            self.rotations = config.get('rotations', [0, 90, 180, 270])
        elif args.datasource == 'miniimagenet':
            self.num_classes = config.get('num_classes', args.num_classes)
            self.img_size = config.get('img_size', (84, 84))
            self.dim_input = np.prod(self.img_size) * 3
            self.dim_output = self.num_classes
            metatrain_folder = config.get('metatrain_folder', os.path.join(os.path.dirname(__file__), 'data', 'miniImagenet', 'train'))
            if args.test_set:
                metaval_folder = config.get('metaval_folder', os.path.join(os.path.dirname(__file__), 'data', 'miniImagenet', 'test'))
            else:
                metaval_folder = config.get('metaval_folder', os.path.join(os.path.dirname(__file__), 'data', 'miniImagenet', 'val'))

            metatrain_folders = [os.path.join(metatrain_folder, label)
                for label in os.listdir(metatrain_folder)
                if os.path.isdir(os.path.join(metatrain_folder, label))]
            metaval_folders = [os.path.join(metaval_folder, label)
                for label in os.listdir(metaval_folder)
                if os.path.isdir(os.path.join(metaval_folder, label))]
            self.metatrain_character_folders = metatrain_folders
            self.metaval_character_folders = metaval_folders
            self.rotations = config.get('rotations', [0])
        else:
            raise ValueError('Unrecognized data source')

    def make_data_tensor(self, train=True):
        if train:
            folders = self.metatrain_character_folders
            num_total_batches = 1000
        else:
            folders = self.metaval_character_folders
            num_total_batches = 600

        print('Generating filenames')
        all_filenames = []
        for _ in range(num_total_batches):
            sampled_character_folders = random.sample(folders, self.num_classes)
            random.shuffle(sampled_character_folders)
            labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False)
            filenames = [li[1] for li in labels_and_images]
            all_filenames.extend(filenames)

        def parse_image(filename):
            image = tf.io.read_file(filename)
            if self.args.datasource == 'miniimagenet':
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.image.resize(image, self.img_size)
                image = tf.reshape(image, [self.dim_input])
                image = tf.cast(image, tf.float32) / 255.0
            else:
                image = tf.image.decode_png(image)
                image = tf.image.resize(image, self.img_size)
                image = tf.reshape(image, [self.dim_input])
                image = tf.cast(image, tf.float32) / 255.0
                image = 1.0 - image
            return image

        dataset = tf.data.Dataset.from_tensor_slices(all_filenames)
        dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size * self.num_classes * self.num_samples_per_class)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        all_image_batches, all_label_batches = [], []
        for batch_images in dataset.take(self.batch_size):
            if self.args.datasource == 'omniglot':
                rotations = tf.random.uniform([self.num_classes], maxval=4, dtype=tf.int32)
            label_batch = tf.repeat(tf.range(self.num_classes, dtype=tf.int32), self.num_samples_per_class)
            new_list, new_label_list = [], []
            for k in range(self.num_samples_per_class):
                class_idxs = tf.range(self.num_classes)
                class_idxs = tf.random.shuffle(class_idxs)
                true_idxs = class_idxs * self.num_samples_per_class + k
                new_list.append(tf.gather(batch_images, true_idxs))
                if self.args.datasource == 'omniglot':
                    new_list[-1] = tf.stack([tf.reshape(tf.image.rot90(
                        tf.reshape(new_list[-1][ind], self.img_size + [1]),
                        k=rotations[class_idxs[ind]]), [self.dim_input])
                        for ind in range(self.num_classes)])
                new_label_list.append(tf.gather(label_batch, true_idxs))
            new_list = tf.concat(new_list, 0)
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches

    def generate_sinusoid_batch(self, train=True, input_idx=None):
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class, 1])
            if input_idx is not None:
                init_inputs[:, input_idx:, 0] = np.linspace(self.input_range[0], self.input_range[1], num=self.num_samples_per_class - input_idx, retstep=False)
            outputs[func] = amp[func] * np.sin(init_inputs[func] - phase[func])
        return init_inputs, outputs, amp, phase
