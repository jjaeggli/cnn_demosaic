# Module for correcting exposure / luminance signal in the raw image.

import tensorflow as tf

from cnn_demosaic import transform
from cnn_demosaic.profile import profile


class Exposure:
    def __init__(self, model, hist_size=32):
        self.model = model
        self.hist_size = hist_size

    @profile()
    def process(self, img_arr):
        orig_shape = img_arr.shape
        # Compute the histogram.
        if len(img_arr.shape) == 3 and img_arr.shape[-1] == 3:
            # The input shape is assumed to be a 2D RGB array.
            new_shape = (img_arr.shape[0] * img_arr.shape[1], 3)
            img_arr = tf.reshape(img_arr, new_shape)

        # Use the histogram to determine processing params.
        levels, gamma, curve = self.get_processing_params(img_arr)

        # Apply the processing params.
        output_arr = self.apply_parameters(img_arr, levels[0], gamma[0], curve[0])
        output_arr = tf.reshape(output_arr, orig_shape)
        return output_arr

    @profile()
    def get_processing_params(self, img_arr):
        luma_arr = transform.tf_rgb_luma_fn(img_arr)
        img_hist = tf.histogram_fixed_width(luma_arr, [0.0, 1.0], self.hist_size)
        # Use the histogram to determine processing params.
        img_hist = tf.reshape(img_hist, (1, self.hist_size))
        return self.model.predict(img_hist, verbose=0)

    @profile()
    def apply_parameters(self, img_arr, levels, gamma, curve):
        """
        Applies the parameters returned by the levels model.

        Args:
            gamma: Gamma exponent
            levels: Tuple (min level, max level)
            curve: Tuple (offset, contrast, slope)
        """
        output_arr = tf.pow(img_arr, gamma)
        output_arr = transform.tf_levels_fn(output_arr, *levels)
        output_arr = transform.tf_s_curve_fn(output_arr, *curve)
        return output_arr
