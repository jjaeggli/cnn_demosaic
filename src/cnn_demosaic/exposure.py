# Module for correcting exposure / luminance signal in the raw image.

import tensorflow as tf

from cnn_demosaic import transform


class Exposure:
    def __init__(self, model, hist_size=32):
        self.model = model
        self.hist_size = hist_size

    def process(self, img_arr):
        orig_shape = img_arr.shape
        # Compute the histogram.
        if len(img_arr.shape) == 3 and img_arr.shape[-1] == 3:
            # The input shape is assumed to be a 2D RGB array.
            new_shape = (img_arr.shape[0] * img_arr.shape[1], 3)
            img_arr = tf.reshape(img_arr, new_shape)
        luma_arr = transform.tf_rgb_luma_fn(img_arr)
        img_hist = tf.histogram_fixed_width(luma_arr, [0.0, 1.0], self.hist_size)
        img_hist = tf.reshape(img_hist, (1, self.hist_size))

        # Use the histogram to determine processing params.
        levels, gamma, curve = self.model.predict(img_hist)

        # Apply the processing params.
        output_arr = tf.pow(img_arr, *gamma[0])
        output_arr = transform.tf_levels_fn(output_arr, levels[0][0], levels[0][1])
        output_arr = transform.tf_s_curve_fn(output_arr, *curve[0])
        output_arr = tf.reshape(output_arr, orig_shape)

        return output_arr
