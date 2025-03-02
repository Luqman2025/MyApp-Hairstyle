# import cv2
# import os
# import numpy as np
# from skimage.filters import gaussian
# from test import evaluate
# import streamlit as st
# from PIL import Image, ImageColor
#
# def sharpen(img):
#     img = img * 1.0
#     gauss_out = gaussian(img, sigma=5, multichannel=True)
#
#     alpha = 1.5
#     img_out = (img - gauss_out) * alpha + img
#
#     img_out = img_out / 255.0
#
#     mask_1 = img_out < 0
#     mask_2 = img_out > 1
#
#     img_out = img_out * (1 - mask_1)
#     img_out = img_out * (1 - mask_2) + mask_2
#     img_out = np.clip(img_out, 0, 1)
#     img_out = img_out * 255
#     return np.array(img_out, dtype=np.uint8)
#
#
# def hair(image, parsing, part=17, color=[230, 50, 20]):
#     b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
#     tar_color = np.zeros_like(image)
#     tar_color[:, :, 0] = b
#     tar_color[:, :, 1] = g
#     tar_color[:, :, 2] = r
#     np.repeat(parsing[:, :, np.newaxis], 3, axis=2)
#
#     image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)
#
#     if part == 12 or part == 13:
#         image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
#     else:
#         image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]
#
#     changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
#
#     if part == 17:
#         changed = sharpen(changed)
#
#
#     changed[parsing != part] = image[parsing != part]
#     return changed
#
# DEMO_IMAGE = 'imgs/116.jpg'
#
# st.title('Virtual Makeup')
#
# st.sidebar.title('Virtual Makeup')
# st.sidebar.subheader('Parameters')
#
# table = {
#         'hair': 17,
#         'upper_lip': 12,
#         'lower_lip': 13,
#
#     }
#
# img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])
#
# if img_file_buffer is not None:
#     image = np.array(Image.open(img_file_buffer))
#     demo_image = img_file_buffer
#
# else:
#     demo_image = DEMO_IMAGE
#     image = np.array(Image.open(demo_image))
#
# #st.set_option('deprecation.showfileUploaderEncoding', False)
#
# new_image = image.copy()
#
#
#
#
#
# st.subheader('Original Image')
#
# st.image(image,use_column_width = True)
#
#
# cp = 'cp/79999_iter.pth'
# ori = image.copy()
# h,w,_ = ori.shape
#
# #print(h)
# #print(w)
# image = cv2.resize(image,(1024,1024))
#
# parsing = evaluate(demo_image, cp)
# parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)
#
# parts = [table['hair'], table['upper_lip'], table['lower_lip']]
#
# hair_color = st.sidebar.color_picker('Pick the Hair Color', '#000')
# hair_color = ImageColor.getcolor(hair_color, "RGB")
#
# lip_color = st.sidebar.color_picker('Pick the Lip Color', '#edbad1')
#
# lip_color = ImageColor.getcolor(lip_color, "RGB")
#
#
#
# colors = [hair_color, lip_color, lip_color]
#
# for part, color in zip(parts, colors):
#     image = hair(image, parsing, part, color)
#
# image = cv2.resize(image,(w,h))
#
#
# st.subheader('Output Image')
#
# st.image(image,use_column_width = True)



import cv2
import os
import numpy as np
from skimage.filters import gaussian
import streamlit as st
from PIL import Image, ImageColor

# Mocking the evaluate function for demonstration
def evaluate(image_path, model_path):
    # Dummy parsing mask for demonstration
    h, w, _ = cv2.imread(image_path).shape
    return np.zeros((h, w), dtype=np.uint8)

def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)
    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img
    img_out = np.clip(img_out, 0, 1) * 255
    return img_out.astype(np.uint8)

def hair(image, parsing, part=17, color=[230, 50, 20]):
    b, g, r = color
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if part in [12, 13]:
        image_hsv[:, :, :2] = tar_hsv[:, :, :2]
    else:
        image_hsv[:, :, :1] = tar_hsv[:, :, :1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    if part == 17:
        changed = sharpen(changed)

    changed[parsing != part] = image[parsing != part]
    return changed

# Streamlit app setup
DEMO_IMAGE = 'imgs/116.jpg'
MODEL_PATH = 'cp/79999_iter.pth'

st.title('Virtual Makeup')
st.sidebar.title('Virtual Makeup')
st.sidebar.subheader('Parameters')

# Sidebar file uploader
img_file_buffer = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if img_file_buffer:
    image = np.array(Image.open(img_file_buffer))
    demo_image_path = 'uploaded_image.jpg'
    Image.fromarray(image).save(demo_image_path)
else:
    demo_image_path = DEMO_IMAGE
    image = np.array(Image.open(DEMO_IMAGE))

# Display the original image
st.subheader('Original Image')
st.image(image, use_column_width=True)

# Process the uploaded/demo image
try:
    ori = image.copy()
    h, w, _ = ori.shape
    image = cv2.resize(image, (1024, 1024))

    # Evaluate parsing mask
    parsing = evaluate(demo_image_path, MODEL_PATH)
    parsing = cv2.resize(parsing, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Get color inputs from user
    hair_color_hex = st.sidebar.color_picker('Pick the Hair Color', '#000000')
    hair_color = ImageColor.getcolor(hair_color_hex, "RGB")
    lip_color_hex = st.sidebar.color_picker('Pick the Lip Color', '#FF0000')
    lip_color = ImageColor.getcolor(lip_color_hex, "RGB")

    # Apply makeup
    parts = [17, 12, 13]
    colors = [hair_color, lip_color, lip_color]
    for part, color in zip(parts, colors):
        image = hair(image, parsing, part, color)

    image = cv2.resize(image, (w, h))

    # Display the output image
    st.subheader('Output Image')
    st.image(image, use_column_width=True)
except Exception as e:
    st.error(f"An error occurred: {e}")


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

# Dummy user credentials
USER_CREDENTIALS = {
    "username": "Luqman",
    "password": "Luqman06"
}


# Function for login
def login():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type='password')

    if st.sidebar.button("Login"):
        if username == USER_CREDENTIALS["username"] and password == USER_CREDENTIALS["password"]:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.sidebar.success("You are now logged in.")

        else:
            st.session_state.logged_in = False
            st.sidebar.error("Incorrect username or password.")


# Define the app class for hairstyle change
class App1:
    def __init__(self):
        self.args = self.create_args()
        self.init_app()
        self.config = self.load_config()  # No caching here
        run = st.button('RUN')
        if self.args['app1/target_image'] and self.args['app1/source_image'] and run:
            self.pipeline()

    def init_app(self):
        st.title('Realistic Hairstyle Try-On')
        st.subheader('Input Images')
        self.args['app1/target_image'] = st.file_uploader('Target image (The person whose FACE you desire)',
                                                          type=['png', 'jpg', 'jpeg'])
        self.args['app1/source_image'] = st.file_uploader('Source image (The person whose HAIR you desire)',
                                                          type=['png', 'jpg', 'jpeg'])
        if self.args['app1/target_image'] and self.args['app1/source_image']:
            self.target_image = self.read_image_from_streamlit(self.args['app1/target_image'])
            self.source_image = self.read_image_from_streamlit(self.args['app1/source_image'])

            images = [self.target_image, self.source_image]
            captions = ['Target image', 'Source image']
            st.image(images, width=300, caption=captions, use_container_width=True)

    def create_args(self):
        args = {
            'app1/seg_model_path': os.path.join("app1/image_segmentation", "face_segment_checkpoints_256.pth.tar"),
            'app1/image_size': (256, 256),
            'app1/input_image_size': (256, 256),
            'app1/label_config': os.path.join("app1/image_segmentation", "label.yml"),
            'app1/exp': 'app1/exp',
            'app1/verbose': 'info',
            'app1/sample': True,
            'app1/image_folder': 'app1/images',
            'app1/ni': True,
            'app1/is_erode_mask': True
        }
        return args

    def load_config(self):
        config_file_path = os.path.join("app1/configs", "celeba.yml")
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
        return dict2namespace(config)  # No caching here

    def read_image_from_streamlit(self, uploaded_file):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        return opencv_image

    def pipeline(self):
        segment = face_segment(seg_model_path=self.args['app1/seg_model_path'],
                               label_config=self.args['app1/label_config'],
                               input_image_size=self.args['app1/input_image_size'])

        target_mask = segment.segmenting(image=self.target_image)
        source_mask = segment.segmenting(image=self.source_image)

        target_image = resize_image(self.target_image, self.args['app1/image_size'])
        source_image = resize_image(self.source_image, self.args['app1/image_size'])
        target_mask = resize_image(target_mask, self.args['app1/image_size'])
        source_mask = resize_image(source_mask, self.args['app1/image_size'])

        transform_outputs = face_landmark_transform(target_image, target_mask, source_image, source_mask)
        transformed_image, transformed_mask = transform_outputs["result_image"], transform_outputs["result_mask"]
        transformed_segment = segment.segmenting(image=transformed_image)

        filled_image = face_artifact_fill(target_image, target_mask, transformed_image, transformed_mask,
                                          transformed_segment)

        before_images = [target_mask, source_mask, transformed_image, filled_image]
        captions = ['Target mask', 'Source mask', 'Transformed image', 'Filled image']
        st.image(before_images, width=300, caption=captions, clamp=True, use_container_width=True)

        sde_mask = transform_outputs['only_fixed_face']
        if self.args['app1/is_erode_mask']:
            kernel_size = self.args['app1/erode_kernel_size']
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            sde_mask = cv2.erode(sde_mask, kernel, iterations=1)

        try:
            runner = Diffusion(image_folder=self.args['app1/image_folder'],
                               sample_step=self.args['app1/sample_step'],
                               total_noise_levels=self.args['app1/t'],
                               config=self.config)
            self.show_images = runner.image_editing_sample_for_streamlit(filled_image, sde_mask)
            images = list(self.show_images.values())
            captions = list(self.show_images.keys())
            st.image(images, width=100, caption=captions, clamp=True, use_container_width=True)
            for it in range(self.args['app1/sample_step']):
                st.image(self.show_images[f'samples_{it}'], width=300, caption=f'Final image {it + 1}', clamp=True,
                         use_container_width=True)

        except Exception:
            logging.error(traceback.format_exc())

        return 0


# Hair color change implementation (app2)
def app2():
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
        demo_image = 'app2/imgs/116.jpg'
        image = np.array(Image.open(demo_image))

    new_image = image.copy()
    st.subheader('Original Image')
    st.image(image, use_container_width=True)

    cp = 'app2/cp/79999_iter.pth'
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


# Main app with tabs
if __name__ == '__main__':
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        login()
    else:
        st.sidebar.title("Choose Application")
        app_choice = st.sidebar.radio("Select", ("Hairstyle Change", "Hair Color Change"))

        if app_choice == "Hairstyle Change":
            App1()
        else:
            app2()