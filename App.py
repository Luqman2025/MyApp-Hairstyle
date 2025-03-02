import streamlit as st
import pandas as pd
import numpy as np
import yaml
import cv2
import os
import logging
import traceback
from App1.image_segmentation.segment_inference import face_segment
from App1.runners.image_editing import Diffusion

from App1.image_landmark_transform.face_landmark import face_landmark_transform
from App1.image_artifact_fill.artifact_fill import face_artifact_fill
from App1.inference import resize_image, dict2namespace
from skimage.filters import gaussian
from App2.test import evaluate
from PIL import Image, ImageColor

# Set up Streamlit page configuration
st.set_page_config(page_title="Virtual Hair TryOn", layout="wide")
st.markdown("""<meta name="google-adsense-account" content="ca-pub-1461221419808763">""", unsafe_allow_html=True)


# Sidebar menu using st.radio
st.sidebar.title("Menu")
App = st.sidebar.radio(
    "Select an option:",
    ["Hair Color Change", "Hairstyle Change"]
)

# Hair Color Change functionality
def App2():
    st.title('Virtual Makeup')
    st.sidebar.title('Virtual Makeup')
    st.sidebar.subheader('Parameters')

    table = {
        'hair': 17,
        'upper_lip': 12,
        'lower_lip': 13,
    }

    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", 'png'])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
        demo_image = img_file_buffer
    else:
        demo_image = 'App2/imgs/116.png'
        image = np.array(Image.open(demo_image))

    new_image = image.copy()
    st.subheader('Original Image')
    st.image(image, use_container_width=True)

    cp = 'App2/cp/79999_iter.pth'
    ori = image.copy()
    h, w, _ = ori.shape

    image = cv2.resize(image, (1024, 1024))
    parsing = evaluate(demo_image, cp)
    parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

    parts = [table['hair'], table['upper_lip'], table['lower_lip']]

    hair_color = st.sidebar.color_picker('Pick the Hair Color', '#000')
    hair_color = ImageColor.getcolor(hair_color, "RGB")

    lip_color = st.sidebar.color_picker('Pick the Lip Color', '#edbad1')
    lip_color = ImageColor.getcolor(lip_color, "RGB")

    colors = [hair_color, lip_color, lip_color]

    def hair(image, parsing, part=17, color=[230, 50, 20]):
        b, g, r = color
        tar_color = np.zeros_like(image)
        tar_color[:, :, 0] = b
        tar_color[:, :, 1] = g
        tar_color[:, :, 2] = r

        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

        if part == 12 or part == 13:
            image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
        else:
            image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

        changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

        if part == 17:
            changed = sharpen(changed)

        changed[parsing != part] = image[parsing != part]
        return changed

    def sharpen(img):
        img = img * 1.0
        gauss_out = gaussian(img, sigma=5)

        alpha = 1.5
        img_out = (img - gauss_out) * alpha + img

        img_out = img_out / 255.0

        mask_1 = img_out < 0
        mask_2 = img_out > 1

        img_out = img_out * (1 - mask_1)
        img_out = img_out * (1 - mask_2) + mask_2
        img_out = np.clip(img_out, 0, 1)
        img_out = img_out * 255
        return np.array(img_out, dtype=np.uint8)

    for part, color in zip(parts, colors):
        image = hair(image, parsing, part, color)

    image = cv2.resize(image, (w, h))
    st.subheader('Output Image')
    st.image(image, use_container_width=True)

# Hairstyle Change functionality
class App1:
    def __init__(self):
        self.args = self.create_args()
        self.init_App()
        self.config = self.load_config()  # No caching here
        run = st.button('RUN')
        if self.args['App1/target_image'] and self.args['App1/source_image'] and run:
            self.pipeline()

    def init_App(self):
        st.title('Realistic Hairstyle Try-On')
        st.subheader('Input Images')
        self.args['App1/target_image'] = st.file_uploader('Target image (The person whose FACE you desire)',
                                                          type=['png', 'jpg', 'jpeg'])
        self.args['App1/source_image'] = st.file_uploader('Source image (The person whose HAIR you desire)',
                                                          type=['png', 'jpg', 'jpeg'])
        if self.args['App1/target_image'] and self.args['App1/source_image']:
            self.target_image = self.read_image_from_streamlit(self.args['App1/target_image'])
            self.source_image = self.read_image_from_streamlit(self.args['App1/source_image'])

            images = [self.target_image, self.source_image]
            captions = ['Target image', 'Source image']
            st.image(images, width=300, caption=captions, use_container_width=True)

    def create_args(self):
        args = {
            'App1/seg_model_path': os.path.join("App1/image_segmentation", "face_segment_checkpoints_256.pth.tar"),
            'App1/image_size': (256, 256),
            'App1/input_image_size': (256, 256),
            'App1/label_config': os.path.join("App1/image_segmentation", "label.yml"),
            'App1/exp': 'App1/exp',
            'App1/verbose': 'info',
            'App1/sample': True,
            'App1/image_folder': 'App1/images',
            'App1/ni': True,
            'App1/is_erode_mask': True,
            'App1/erode_kernel_size': 5  # Default kernel size for erosion
        }
        return args

    def load_config(self):
        config_file_path = os.path.join("App1/configs", "celeba.yml")
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
        return dict2namespace(config)  # No caching here

    def read_image_from_streamlit(self, uploaded_file):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        return opencv_image

    def pipeline(self):
        segment = face_segment(seg_model_path=self.args['App1/seg_model_path'],
                               label_config=self.args['App1/label_config'],
                               input_image_size=self.args['App1/input_image_size'])

        target_mask = segment.segmenting(image=self.target_image)
        source_mask = segment.segmenting(image=self.source_image)

        target_image = resize_image(self.target_image, self.args['App1/image_size'])
        source_image = resize_image(self.source_image, self.args['App1/image_size'])
        target_mask = resize_image(target_mask, self.args['App1/image_size'])
        source_mask = resize_image(source_mask, self.args['App1/image_size'])

        transform_outputs = face_landmark_transform(target_image, target_mask, source_image, source_mask)
        transformed_image, transformed_mask = transform_outputs["result_image"], transform_outputs["result_mask"]
        transformed_segment = segment.segmenting(image=transformed_image)

        filled_image = face_artifact_fill(target_image, target_mask, transformed_image, transformed_mask,
                                          transformed_segment)

        before_images = [ filled_image]
        captions = ['Transformed image']
        st.image(before_images, width=300, caption=captions, clamp=True, use_container_width=True)

        sde_mask = transform_outputs['only_fixed_face']
        if self.args['App1/is_erode_mask']:
            kernel_size = self.args['App1/erode_kernel_size']
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            sde_mask = cv2.erode(sde_mask, kernel, iterations=1)

        try:
            runner = Diffusion(image_folder=self.args['App1/image_folder'],
                               sample_step=self.args['App1/sample_step'],
                               total_noise_levels=self.args['App1/t'],
                               config=self.config)
            self.show_images = runner.image_editing_sample_for_streamlit(filled_image, sde_mask)
            images = list(self.show_images.values())
            captions = list(self.show_images.keys())
            st.image(images, width=100, caption=captions, clamp=True, use_container_width=True)
            for it in range(self.args['App1/sample_step']):
                st.image(self.show_images[f'samples_{it}'], width=300, caption=f'Final image {it + 1}', clamp=True,
                         use_container_width=True)

        except Exception:
            logging.error(traceback.format_exc())

        return 0

# Main App logic
if __name__ == "__main__":
    if App == "Hair Color Change":
        App2()
    elif App == "Hairstyle Change":
        App1()
