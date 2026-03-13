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


class ExpandDims(layers.Layer):
    """Compat layer for models serialized with keras.ops.expand_dims nodes."""

    def __init__(self, axis=-1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = axis

    def call(self, x):
        return tf.expand_dims(x, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


class Mean(layers.Layer):
    def __init__(self, axis=None, keepdims=False, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return tf.reduce_mean(x, axis=self.axis, keepdims=self.keepdims)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "keepdims": self.keepdims})
        return config


class Sum(layers.Layer):
    def __init__(self, axis=None, keepdims=False, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return tf.reduce_sum(x, axis=self.axis, keepdims=self.keepdims)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "keepdims": self.keepdims})
        return config


class Log(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, x):
        return tf.math.log(x)


class Add(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            return tf.add(inputs[0], inputs[1])
        return tf.add_n(inputs)


class Multiply(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            return tf.multiply(inputs[0], inputs[1])
        result = inputs[0]
        for item in inputs[1:]:
            result = tf.multiply(result, item)
        return result


class Softmax(layers.Layer):
    def __init__(self, axis=-1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = axis

    def call(self, x):
        return tf.nn.softmax(x, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


class Dense(layers.Dense):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._compat_quantization_config = quantization_config

    def get_config(self):
        config = super().get_config()
        config["quantization_config"] = self._compat_quantization_config
        return config


def get_audio_custom_objects(include_ops=True):
    objects = {
        "flatten_segments": flatten_segments,
        "restore_segments": restore_segments,
        "MRSTFT": MRSTFT,
        "MaskedMean": MaskedMean,
        "Dense": Dense,
    }

    if include_ops:
        objects.update(
            {
                "ExpandDims": ExpandDims,
                "Mean": Mean,
                "Sum": Sum,
                "Log": Log,
                "Add": Add,
                "Multiply": Multiply,
                "Softmax": Softmax,
            }
        )

    return objects

