import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageColor
from skimage.color import rgb2hsv, hsv2rgb
from skimage.filters import gaussian


# Mocking evaluate function (replace with actual implementation)
def evaluate(image_path, model_path):
    """Mock parsing function to simulate semantic segmentation."""
    h, w, _ = cv2.imread(image_path).shape
    # Create a dummy parsing map with all zeros for demonstration.
    return np.zeros((h, w), dtype=np.uint8)


# Sharpening function
def sharpen(img):
    """Apply a sharpening filter to the image."""
    img = img.astype(np.float32) / 255.0
    blurred = gaussian(img, sigma=2, multichannel=True)
    sharpened = np.clip(img + 1.5 * (img - blurred), 0, 1) * 255
    return sharpened.astype(np.uint8)


# Function to apply virtual makeup
def apply_makeup(image, parsing, part, color):
    """Apply the specified color to a given facial part."""
    hsv_image = rgb2hsv(image)
    target_color = np.array(color) / 255.0
    target_hsv = rgb2hsv(target_color.reshape(1, 1, 3)).squeeze()

    if part in [12, 13]:  # Lips
        hsv_image[..., 0:2] = target_hsv[0:2]
    elif part == 17:  # Hair
        hsv_image[..., 0] = target_hsv[0]

    modified_image = hsv2rgb(hsv_image) * 255
    modified_image = modified_image.astype(np.uint8)
    if part == 17:
        modified_image = sharpen(modified_image)
    modified_image[parsing != part] = image[parsing != part]
    return modified_image


# Streamlit App
DEMO_IMAGE = 'imgs/116.png'
MODEL_PATH = 'cp/79999_iter.pth'

st.title("Virtual Makeup Application")
st.sidebar.title("Customize Your Look")
st.sidebar.subheader("Choose Your Options")

# Upload image or use demo
img_file_buffer = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if img_file_buffer:

    image = np.array(Image.open(img_file_buffer))
    demo_image_path = 'uploaded_image.jpg'
    Image.fromarray(image).save(demo_image_path)
else:
    demo_image_path = DEMO_IMAGE
    image = np.array(Image.open(DEMO_IMAGE))

# Display the original image
st.subheader("Original Image")
st.image(image, use_column_width=True)

# Process the image
try:
    h, w, _ = image.shape
    resized_image = cv2.resize(image, (1024, 1024))

    # Obtain the segmentation mask
    parsing = evaluate(demo_image_path, MODEL_PATH)
    parsing = cv2.resize(parsing, (1024, 1024), interpolation=cv2.INTER_NEAREST)

    # Sidebar color pickers
    hair_color_hex = st.sidebar.color_picker("Pick Hair Color", "#FF5733")
    hair_color = ImageColor.getcolor(hair_color_hex, "RGB")
    lip_color_hex = st.sidebar.color_picker("Pick Lip Color", "#FFC0CB")
    lip_color = ImageColor.getcolor(lip_color_hex, "RGB")

    # Apply makeup
    parts = [17, 12, 13]
    colors = [hair_color, lip_color, lip_color]
    output_image = resized_image.copy()

    for part, color in zip(parts, colors):
        output_image = apply_makeup(output_image, parsing, part, color)

    # Resize back to original dimensions
    output_image = cv2.resize(output_image, (w, h))

    # Display the result
    st.subheader("Output Image")
    st.image(output_image, use_column_width=True)

except Exception as e:
    st.error(f"An error occurred: {e}")
