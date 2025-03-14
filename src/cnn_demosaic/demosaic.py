# Defines the demosaic process using a trained model.

import numpy as np
from tqdm import tqdm

# TODO(jjaeggli): add progress callback for UI.
# TODO(jjaeggli): add debug logging using a logging handler.


class Demosaic:
    def __init__(self, model, per_tile_fn=None, xtrans=True):
        self.model = model
        self.per_tile_fn = per_tile_fn
        self.xtrans = xtrans

        # tile_size should be determined by the model, and should be a multiple
        # of the minimum sensor pattern shape.
        # margin should be a multiple of the minimum sensor pattern shape.

        if xtrans:
            self.tile_size = 36
            self.margin = 6
        else:  # Use Bayer defaults
            self.tile_size = 32
            self.margin = 2

    def demosaic(self, img_array):
        # Computed tile width
        m = self.margin
        # Size of the tile
        S_t = self.tile_size
        # Size of the subtile
        S_s = S_t - 2 * m

        # This is working with the assumption that the mosaiced image
        # contains an array of monochrome values and not RGB pixels.
        img_row, img_col = img_array.shape

        # Trim the array to a multiple of the margin.
        img_row_out = img_row - img_row % m
        img_col_out = img_col - img_col % m
        img_array = img_array[:img_row_out, :img_col_out]

        # working rows and columns are the number of regular subtiles that are
        # contained within the image.
        w_row = img_row_out // S_s
        w_col = img_col_out // S_s

        # Pad the array by the margin
        img_array = pad_array(img_array, margin=m, xtrans=self.xtrans)

        # If the working subtile + margin extends outside of the last row
        # or column. This should not happen.
        img_row_padded, img_col_padded = img_array.shape
        assert (w_row * S_s + m) <= img_row_padded
        assert (w_col * S_s + m) <= img_col_padded

        # The output image has 3 axis for RGB values.
        img_out = np.zeros((img_row_out, img_col_out, 3), dtype=np.float32)

        # Generate row coordinates. This allows handling image dimensions
        # which do not align with the tile size.
        row_coords = []

        for r in range(w_row):
            row_start = r * S_s
            row_end = row_start + S_t
            row_coords.append((row_start, row_end))

        # If the last working row does not end with the last padded row,
        # create a final row for this stripe.
        if row_coords[-1][1] != img_row_padded:
            row_start = img_row_padded - S_t
            row_coords.append((row_start, img_row_padded))

        for row_start, row_end in tqdm(row_coords):
            stripe = img_array[row_start:row_end]
            assert stripe.shape[0] == S_t, "Stripe rows must equal the tile size."

            col_coords = []

            for c in range(w_col):
                col_start = c * S_s
                col_end = col_start + S_t
                col_coords.append((col_start, col_end))

            # If the last working row does not end with the last padded row,
            # create a final row for this stripe.
            if col_coords[-1][1] != img_col_padded:
                col_start = img_col_padded - S_t
                col_coords.append((col_start, img_col_padded))

            tiles = []
            post_process_functions = []

            for col_start, col_end in col_coords:
                tile = stripe[:, col_start:col_end]

                assert col_end <= img_col_padded, "Tile column end is out of bounds."
                assert tile.shape == (S_t, S_t), "Tile shape is invalid."

                # Process the tile with the per-tile function.
                if self.per_tile_fn is not None:
                    tile, post_process_fn = self.per_tile_fn(tile)
                    post_process_functions.append(post_process_fn)

                tiles.append(tile)

            tiles = np.array(tiles)

            # Do our thing.
            output = self.model.predict(tiles)

            # Out row start coordinate is relative to the non-padded image.
            # row_start is relative to the padded image. So compensating for
            # image and tile padding:
            # out_row_start = row_start + m - m
            out_row_start = row_start
            out_row_end = out_row_start + S_s

            for c in range(output.shape[0]):
                sub_tile = output[c]

                col_start, _ = col_coords[c]

                out_col_start = col_start
                out_col_end = out_col_start + S_s

                # If we use the pre_process_tile_fn, we need to apply the corresponding reversal
                # function to the tile output. Ie. if the pre_process_tile_fn is to perform
                # normalization, the output function should denormalize the tile.
                if post_process_functions:
                    sub_tile = post_process_functions[c](sub_tile)

                # Trim the margin of the tile into the subtile shape by removing the margin, and
                # apply the subtile to the output array.
                sub_tile = sub_tile[m:-m, m:-m]
                img_out[out_row_start:out_row_end, out_col_start:out_col_end] = sub_tile

        return img_out


def pad_array(input_array, margin, xtrans=True):
    """Pad the array with the given margin dimensions."""
    input_rows, input_cols = input_array.shape
    is_valid_xtrans = input_rows % 6 == 0 and input_rows % 6 == 0
    if xtrans and not is_valid_xtrans:
        raise ValueError("The image dimensions do not conform to a valid XTrans pattern shape.")
    output_rows = input_rows + margin * 2
    output_cols = input_cols + margin * 2

    # This approach simply extends the image by the margin width on all dimensions, and duplicates
    # and shifts the margins into these areas. This is done instead of mirroring to maintain the
    # existing sensor pattern so that the edge pixels are properly demosaiced.

    output_array = np.zeros((output_rows, output_cols), dtype=np.float32)
    # Broadcast the input array within the output array.
    output_array[margin:-margin, margin:-margin] = input_array

    # Copy left and right margin data
    output_array[margin:-margin, :margin] = input_array[:, :margin]
    output_array[margin:-margin, -margin:] = input_array[:, -margin:]

    # Copy top and bottom margin data
    output_array[:margin, :] = output_array[margin : 2 * margin, :]
    output_array[-margin:, :] = output_array[-2 * margin : -margin, :]
    return output_array
