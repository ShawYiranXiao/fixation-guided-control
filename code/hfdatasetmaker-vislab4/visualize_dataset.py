# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import streamlit as st
from datasets import load_from_disk, Image
import pandas as pd
import base64
from io import BytesIO
from PIL import Image as PILImage

#  Load the dataset
dataset = load_from_disk("processed_dataset_3")
df = dataset.to_pandas()


#  Decode datasets.Image records
def decode_image(image_data):
    if image_data is None:
        return None
    try:
        return Image().decode_example(image_data)
    except Exception as e:
        st.warning(f"Unable to decode datasets.Image data: {e}")
        return None  # Return None when image decoding fails


df["image"] = df["image"].apply(decode_image)  #  Replace the original image column
# df["image"] = df["deepgaze_feature"].apply(decode_image)  #  Replace the original image column


#  Render image previews with HTML img elements
def image_to_html(img, index):
    if img is None:
        return "No Image"
    try:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return (
            f'<a href="data:image/png;base64,{img_base64}" target="_blank">'
            f'<img src="data:image/png;base64,{img_base64}" width="400"></a>'
        )
    except Exception as e:
        st.warning(f"Unable to process image: {e}")
        return "Error Image"


df["image_html"] = [
    image_to_html(img, i) for i, img in enumerate(df["image"])
]  #  Create a separate HTML preview column

#  Render image previews as HTML rather than passing PIL images to st.dataframe
st.write("### Hugging Face dataset preview")
st.markdown(df.to_html(escape=False), unsafe_allow_html=True)
