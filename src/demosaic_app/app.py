# Application which allows adjusting the processing parameters of an image.

from cnn_demosaic.process_raw import EXPOSURE_WEIGHTS
import streamlit as st
import numpy as np
import pyexr

from cnn_demosaic import color_model
from cnn_demosaic import color
from cnn_demosaic import config
from cnn_demosaic import exposure_model
from cnn_demosaic import exposure
from cnn_demosaic import transform
from demosaic_app import operations

# TODO(jjaeggli): Add file chooser control and remove hardcoded value.
INPUT_EXR_PATH = "/home/jake/git/cnn_demosaic/assets/DSCF5657.exr"

# TODO(jjaeggli): Catalog available weights in the weights module.
COLOR_MODELS = {
    "Color": "color_transform.weights.h5",
    "Landscape": "color_transform_landscape.weights.h5",
}


def get_exposure_processor():
    if "exp" not in st.session_state:
        model_path = config.get_weights_resource_path(None, config.EXPOSURE_WEIGHTS)
        exp_model = exposure_model.create_exposure_model()
        exp = exposure.Exposure(exp_model)
        st.session_state["exp"] = exp
        return exp
    else:
        return st.session_state["exp"]


def predict_exposure(exr_img):
    """Use the exposure model to predict processing parameters for the image."""
    exp = get_exposure_processor()

    img_arr = exr_img.reshape((exr_img.size // 3, 3))
    levels, gamma, curve = exp.get_processing_params(img_arr)
    st.session_state["exp_levels"] = levels
    st.session_state["exp_gamma"] = gamma[0][0]
    st.session_state["exp_curve"] = curve[0]


def apply_exposure(img_arr, levels, gamma, curve):
    exp = get_exposure_processor()
    return exp.apply_parameters(img_arr, levels, gamma, curve)


def auto_button_click():
    pass
    # TODO(jjaeggli): Reset levels to auto-level parameters.


def main_app():
    st.title("Process Image")

    # Set defaults only once when the session starts
    if "color1_name" not in st.session_state:
        img_width, img_height = 256, 256
        st.session_state["default_percentage"] = 50
        st.session_state["img_width"] = img_width
        st.session_state["img_height"] = img_height

    if "thumbnail_base" not in st.session_state:
        exr_img = pyexr.read(INPUT_EXR_PATH)[:, :, :3]
        thumb_base = operations.create_thumbnail(exr_img)
        st.session_state["thumbnail_base"] = thumb_base

    img_width = st.session_state["img_width"]
    img_height = st.session_state["img_height"]

    curr_image1_np = None
    curr_image2_np = None

    st.subheader("Levels")

    st.button(
        "Auto",
        key=None,
        help=None,
        on_click=None,
        type="secondary",
        icon=None,
        disabled=False,
        use_container_width=False,
    )

    predict_exposure(st.session_state["thumbnail_base"])

    gamma_default = 1.0

    if "exp_gamma" in st.session_state:
        gamma_default = st.session_state["exp_gamma"]
        if gamma_default < 0.0:
            gamma_default = 0.0
        curve = st.session_state["exp_curve"]

    gamma = st.slider("Gamma", 0.0, 3.0, gamma_default, step=0.1)

    contrast = st.slider("Contrast", 0.0, 10.0, 1.2, step=0.1)

    slope = st.slider("Slope", 0.0, 10.0, 1.3, step=0.1)

    shift = st.slider("Shift", 0.0, 10.0, 1.4, step=0.1)

    # TODO(jjaeggli): Apply the levels to the base thumbnail, and display the luma channel.

    st.subheader("Color Models")
    col1, col2 = st.columns(2)
    with col1:
        color1_name = st.selectbox(
            "Select model",
            list(COLOR_MODELS.keys()),  # Options are the color names
            index=0,
            key="selectbox_color1",  # Unique key for the widget
        )
        color1_rgb = COLOR_MODELS[color1_name]
        curr_image1_np = st.session_state.thumbnail_base
        st.image(curr_image1_np, caption="Image 1", use_container_width=True)

    with col2:
        color2_name = st.selectbox(
            "Select model",
            list(COLOR_MODELS.keys()),  # Options are the color names
            index=0,
            key="selectbox_color2",  # Unique key for the widget
        )

        color2_rgb = COLOR_MODELS[color2_name]
        curr_image2_np = st.session_state.thumbnail_base
        st.image(curr_image2_np, caption="Image 2", use_container_width=True)

    st.subheader("Blending Control")
    percentage = st.slider(
        "Color balance",
        0,
        100,
        st.session_state["default_percentage"],
    )
    st.write(f"Current Blend: **{percentage}%**")

    st.subheader("Blended Image")
    if curr_image1_np is not None and curr_image2_np is not None:
        blended_image_np = operations.blend_images(curr_image1_np, curr_image2_np, percentage)
        st.image(
            blended_image_np, caption=f"Blended Image ({percentage}%)", use_container_width=True
        )


if __name__ == "__main__":
    main_app()
