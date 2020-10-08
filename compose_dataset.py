import tensorflow as tf
import tensorflow_datasets as tfds
import random
import librosa
import numpy as np
import csv

class Datagen():
    def __init__(self, data_fp, batch_size=32, sr=16000, n_mfcc=13, hop_length=160, cmn=True, max_chars=100, shuffle=True):
        self.data_fp = data_fp
        self.batch_size = batch_size
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.cmn = cmn
        self.max_chars = max_chars
        self.shuffle = shuffle
        self.max_ts = (10*sr//self.hop_length)+1
        self.augments = [] # If augments are provided via tensor_batch

        self.generate_encoder_decoder()

    def generate_encoder_decoder(self):
        vocab = " 'abcdefghijklmnopqrstuvwxzy"
        self.encoder = dict((v, k) for k, v in enumerate(vocab))
        self.decoder = dict(enumerate(vocab))

    def load_audio(self, fn):
        """
        Load audio from filename

        :param fn: array containing str filename of converted wav (ending in .mp3)
        :return: np.array of audio signal
        """
        fn = fn
        fn = fn[:-3] + 'wav'  # Take off mp3 ending and append wav
        fp = f"./en/converted/" + fn
        sig, _ = librosa.load(fp, sr=self.sr)
        return sig


    def speed_aug(self, sig, speed_delta=.25, max_dur=9.9):
        """
        Apply speed augmentation onto audio signal

        :param sig: tensorflow audio signal in tf.float32
        :param speed_delta: max speed augmentation
        :param max_dur: maximum duration in seconds
        :return: np.array
        """
        dur = librosa.get_duration(sig, sr=self.sr)
        max_slowdown = max(dur*1/max_dur, 1/(1 + speed_delta)) # 25% slowdown max; if we don't exceed max_dur
        min_speedup = 1/(1 - speed_delta) # 25% speed up max
        rate = np.random.uniform(max_slowdown, min_speedup)
        sig = librosa.effects.time_stretch(sig, rate)
        return sig


    def shift_aug(self, sig, max_dur=9.9, noise_strength=.004):
        """
        Shift audio by random amount

        :param sig: tf.float32 audio signal
        :param max_dur: maximum duration of audio
        :param noise_strength: controls coefficient to increase or decrease noise impact
        :return: np.array
        """
        sig_shape = sig.shape[0]
        max_shift = int(self.sr * max_dur) - sig_shape
        random_shift = np.random.randint(0, max_shift)
        noise = noise_strength * np.random.randn(random_shift)
        shifted_sig = np.concatenate((noise, sig), axis=0)
        return shifted_sig


    def mfcc_feats(self, sig):
        """
        Get mfcc, delta, and delta_delta features
         - 25ms window = 16kHz * (25/1000) = 400 frames
         - 10ms hop = 16kHz * (10/1000) = 160 frames
         - 23ms n_fft ~ 2^9 = 512 frames

        :param sig: numpy array of audio signal
        :return: np.array
        """

        mfcc = librosa.feature.mfcc(sig, n_mfcc=self.n_mfcc, n_fft=512, sr=self.sr, win_length=400, hop_length=self.hop_length)
        delta_mfcc = librosa.feature.delta(mfcc, order=1)
        delta_delta_mfcc = librosa.feature.delta(mfcc, order=2)
        all_features = np.vstack((mfcc, delta_mfcc, delta_delta_mfcc))

        # Tranpose time axis first
        all_features = all_features.T
        # MFCC, with delta, and delta_delta in shape of (timesteps, features)
        return all_features


    def cepstral_mean_norm(self, sig):
        """
        Normalize across time across (over the mean/std of each feature)
        :param sig: numpy array of audio signal
        :return: np.array
        """
        mean = np.mean(sig, axis=0)
        std = np.std(sig, axis=0)
        return (sig - mean) / std


    def zero_pad(self, sig):
        """
        Pad time dimension (axis=0) to max_ts
        :param sig:
        :return: tf.float32 tensor
        """
        pad_end = sig.shape[0]  # Get time axis length
        padded = np.pad(sig, ((0, self.max_ts - pad_end), (0, 0)))
        return padded


    def build_encoding(self, label):
        """
        Build a vocab from a string of unique vocab
        :param vocab: string of unique chars
        :return: tfds encoder object
        """
        encoded = np.array([self.encoder[c] for c in label])
        return encoded

    def get_lengths(self, train_data, label):
        """

        :param train_data: batch of training data
        :param label: batch of corresponding labels
        :return: input_len, label_len
                input_len will contain values of number of timesteps; after batch aggregation size is (samples, 1)
                label_len will contain length for each label in batch; after batch aggregation size is (samples, 1)
        """

        ts_len = train_data.shape[1]

        input_length = ts_len//4 * np.ones((self.batch_size,1), dtype='int32')
        label_length = np.array([[len(sample)] for sample in label], dtype='int32')
        return input_length, label_length


    def pad_labels(self, label, pad_value=0):
        """
        Pad labels to the max_chars in our dataset. Doesn't really matter what we pad with
        since it'll be masked when we give the actual label length calculated previous.
        For clarity's sake, we pad with -1.
        :param label: tf.int32 tensor with individual encoded labels
        :return: np.int32 array of size (max_chars, 1)
        """
        padded = pad_value * np.ones((len(label), self.max_chars))

        for i, vals in enumerate(label):
            padded[i, :len(vals)] = vals

        return padded


    def batch_load(self, shuffle=True):
        """
        Provide csv data file; get back a tf batch generator
        :param augments: list of functions to augment audio signal
        :return: tf batch generator
        """
        train_fps = []
        labels = []

        with open(self.data_fp) as csvfile:
            reader = csv.reader(csvfile)
            for fp, lab in reader:
                train_fps.append(fp)
                labels.append(lab)

        if shuffle:
            shuffler = np.random.permutation(len(train_fps))
            train_fps = [train_fps[i] for i in shuffler]
            labels = [labels[i] for i in shuffler]

        # Perform transformations in batches
        for end_idx in range(self.batch_size, len(train_fps) + 1, self.batch_size):
            batch_fps = train_fps[end_idx - self.batch_size:end_idx]
            batch_labels = labels[end_idx - self.batch_size:end_idx]

            # Map filenames to file data
            signals = [self.load_audio(f) for f in batch_fps]

            # Perform any augmentations
            for aug_func in self.augments:
                signals = [aug_func(sig) for sig in signals]

            # Perform MFCC feature transform
            signals = [self.mfcc_feats(sig) for sig in signals]
            if self.cmn:
                signals = [self.cepstral_mean_norm(sig) for sig in signals]

            # Zero pad signals to max_ts
            signals = [self.zero_pad(sig) for sig in signals]
            signals = np.array(signals) # Finalize as np.array
            signals = np.expand_dims(signals, axis=-1) # Add in channel

            # Encode labels
            batch_labels = [self.build_encoding(lab) for lab in batch_labels]
            batch_labels = self.pad_labels(batch_labels)

            # Build additional length inputs for CTC cost func
            input_length, label_length = self.get_lengths(signals, batch_labels)

            data_inputs = signals, batch_labels, input_length, label_length
            data_outputs = batch_labels

            yield data_inputs, data_outputs


    def tensor_batch(self, augments=None):
        if augments:
            self.augments = augments

        types = ((tf.float32, tf.int32, tf.int32, tf.int32),
                 (tf.int32))
        shape = (([self.batch_size, self.max_ts, 3 * self.n_mfcc, 1],
                  [self.batch_size, self.max_chars],
                  [self.batch_size, 1],
                  [self.batch_size, 1]),
                 ([self.batch_size, self.max_chars]))

        dataset = tf.data.Dataset.from_generator(
            self.batch_load,
            args=[],
            output_types=types,
            output_shapes=shape
        ).prefetch(1)

        return dataset



# These lines are for a quick debug run
# test_fp = './en/train_clean.csv'
# datagen = Datagen(test_fp)
# augments = [datagen.speed_aug, datagen.shift_aug]
# data = datagen.tensor_batch(augments=augments)
#
# for ins, outs in data.take(1):
#     for i in ins:
#         print(i.shape)
#     print(outs.shape)