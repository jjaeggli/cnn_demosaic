# cnn_demosaic

Demosaicing for X-Trans and Bayer sensors using convolutional neural networks (CNN).

## Introduction

The goal of this project is to improve image processing for Fuji cameras within open-source raw image processing software. Secondarily, the technique should be usable within bulk image processing pipelines, such as time-lapse, film digitization, and archival workflows.

Open source raw image processing software provide solutions which can be used by amateur and professional photographers, artists, and uses in other fields, they are extensible, configurable, and through open file formats and CLIs, can be used for bulk image processing and in processing pipelines. Dark Table and Raw Therapee are the two most users will be familiar with. Both of these tools provide non-destructive image editing. While these too tools provide trade-offs between different features and paradigms, they both fundamentally rely on the Markesteijn algorithm from the underlying dcraw tool. This algorithmic approach is fast (less than one second typically) and has been the best available for some time.

Commercial products have promised improved quality through the use of *AI*, and numerous papers show the promise of using ML techniques for demosaicing raw images at higher quality (I don't understand most of this work). This research and these products have not yet made their way to the open source community, at least not in a usable form. This project emerged from a personal research project to see if it was remotely possible to perform image demosaicing with entry-level ML knowledge and brute force. The answer is clear: *Yes, it absolutely is!* If you string enough CNNs together, and spend the requisite days training the model, they can produce output which exceeds, at least according to human perception, the best open source algorithmic approaches.

This approach will be slower than Markesteijn, by at least one order of magnitude, however, keep in mind that speed and quality can provide different trade-offs. In the current state, the model has been optimized for quality rather than speed, and for X-Trans sensors and not Bayer sensors. The initial research used a Bayer array pattern for training, and so training a Bayer-compatible model is entirely possible.

## Current Status

- [x] Output as 16-bit floating point OpenEXR images
- [ ] Output as 16-bit integer PNG images
- [ ] Proper color-space conversion from *camera RGB* to *sRGB*
- [ ] Include image metadata in output
- [ ] Train X-Trans models with noise-augmented images for combined ISO-optimized de-noise and demosaicing
- [ ] Train Bayer models

## Samples

I've provided two sample images processed in both RawTherapee using the Markesteijn with capture sharpening and false color suppression disabled. The CNN-Demosaic image is processed using the CLI and then adjusted and white balanced so that contrast and saturation are comparable to the RawTherapee image. Please note that the CLI still has issues producing accurate color output, which is visible to some degree in the output of these images.

### Example 1

Fuji X-E2 Sample Image

Markesteijn

![DSCF5657_crop_markesteijn](./assets/DSCF5657_crop_markesteijn.png)

CNN-Demosaic

![DSCF5657_crop_cnn](./assets/DSCF5657_crop_cnn.png)

[download](https://github.com/jjaeggli/cnn_demosaic/raw/refs/heads/main/assets/DSCF5657.exr) processed image
[download](https://github.com/jjaeggli/cnn_demosaic/raw/refs/heads/main/assets/DSCF5657.raf) raw image

Credit: Jacob Jaeggli

From the examples above, it is clear there is a significant decrease in false color and linear artifacts in the CNN-Demosaic output versus Markesteijn. There is a more subtle increase in detail, and subjectively a more pleasing and natural appearance.

## Installation

This installation uses Tensorflow and recommends a CUDA compatible GPU.

### Installation from source

Clone the repository from GitHub. Create a separate virtualenv for the package to avoid conflicts
with specific versions of tensorflow or other packages. Additionally, if you use a non-CUDA version
of tensorflow, you may want to customize package dependencies prior to install.

```
python -m venv path-for-virtualenv
source path-for-virtualenv/bin/activate
```

From the root of the GitHub repository:

```
pip install .
```

## Usage

Note: Color conversion is not currently implemented. Color output will be incorrect and not easily
converted to a color-corrected image.

Images are currently output in the 16-bit floating point OpenEXR format. This provides a great deal
of flexibility for adjusting the output levels and preserving the dynamic range, as demosaicing is
a floating point operation, and the image from the sensor typically contains a great deal of
information beyond what is displayed in a processed image.

To process a single Fuji RAF image, first activate the virtualenv, then call:

```
cnn_demosaic path/to/image/DSCF0001.RAF
```

The processed image will be saved to the source image path with the EXR extension.
