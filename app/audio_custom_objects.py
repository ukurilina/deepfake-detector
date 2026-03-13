import tensorflow as tf
from tensorflow.keras import layers


def flatten_segments(x):
    shape = tf.shape(x)
    batch_size = shape[0]
    segments = shape[1]
    flattened = tf.reshape(x, (batch_size * segments, 48000))
    return flattened, batch_size, segments


def restore_segments(x):
    preds, batch_size, segments = x
    return tf.reshape(preds, (batch_size, segments, 1))


class MRSTFT(layers.Layer):
    def __init__(self, sr=16000, n_mels=80, **kwargs):
        super().__init__(**kwargs)
        self.sr = sr
        self.n_mels = n_mels
        self.fft_sizes = [512, 1024, 2048]
        self.hop = 256
        self.mel_filters = []

        for fft in self.fft_sizes:
            bins = fft // 2 + 1
            mel = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins=n_mels,
                num_spectrogram_bins=bins,
                sample_rate=sr,
                lower_edge_hertz=80.0,
                upper_edge_hertz=7600.0,
            )
            self.mel_filters.append(mel)

    def call(self, audio):
        outputs = []

        for fft, mel in zip(self.fft_sizes, self.mel_filters):
            spec = tf.signal.stft(
                audio,
                frame_length=fft,
                frame_step=self.hop,
                fft_length=fft,
            )
            spec = tf.abs(spec)
            spec = tf.matmul(spec, mel)
            outputs.append(spec)

        min_frames = tf.reduce_min(tf.stack([tf.shape(o)[1] for o in outputs]))
        cropped = [o[:, :min_frames, :] for o in outputs]
        return tf.concat(cropped, axis=-1)


class MaskedMean(layers.Layer):
    def call(self, x):
        mask = tf.reduce_sum(tf.abs(x), axis=-1) > 0
        mask = tf.cast(mask, tf.float32)

        squeezed = tf.squeeze(x, axis=-1)
        num = tf.reduce_sum(squeezed * mask, axis=1)
        den = tf.reduce_sum(mask, axis=1) + 1e-6
        return num / den


def get_audio_custom_objects():
    return {
        "flatten_segments": flatten_segments,
        "restore_segments": restore_segments,
        "MRSTFT": MRSTFT,
        "MaskedMean": MaskedMean,
    }

