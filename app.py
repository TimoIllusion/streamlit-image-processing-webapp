import tempfile
import os
import logging
from zipfile import ZipFile

import streamlit as st
import cv2
import numpy as np
from moviepy.editor import VideoFileClip

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure imageio uses system ffmpeg
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

from models import ModelA, ModelB, ModelC, CustomModel
from frame_processor import FrameProcessor


def process_images(uploaded_files, model_instance):
    logger.info("Processing images")
    processed_images = []
    total_images = len(uploaded_files)
    progress_bar = st.progress(0)
    for idx, uploaded_file in enumerate(uploaded_files):
        logger.info(f"Processing image {idx + 1}/{total_images}")
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        processed_image = model_instance.execute(image)
        processed_images.append((uploaded_file.name, processed_image))
        progress_bar.progress((idx + 1) / total_images)
    progress_bar.empty()
    return processed_images


def process_video(uploaded_file, model_instance, bitrate):
    logger.info("Processing video")
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.flush()
    tfile.close()

    # Define the output video path
    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_path = temp_video_file.name

    # Use VideoFileClip for on-the-fly processing
    with st.spinner("Processing video..."):
        clip = VideoFileClip(tfile.name)

        # Get total frames using clip.reader.nframes
        total_frames = clip.reader.nframes
        if total_frames is None or total_frames == 0:
            total_frames = int(clip.fps * clip.duration)

        progress_bar = st.progress(0)

        frame_processor = FrameProcessor(model_instance, total_frames, progress_bar)

        processed_clip = clip.fl_image(frame_processor.process)
        processed_clip.write_videofile(
            video_path,
            codec="libx264",
            audio=False,
            bitrate=f"{bitrate}k" if bitrate else None,
            verbose=False,
            logger=None,
        )
        progress_bar.empty()
        clip.close()
        processed_clip.close()

    # Read the video file and store in session state
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()

    # Clean up temporary files
    os.remove(video_path)
    os.remove(tfile.name)

    return video_bytes


def save_images_to_zip(processed_images):
    with tempfile.TemporaryDirectory() as tmpdirname:
        zip_path = os.path.join(tmpdirname, "processed_images.zip")
        with ZipFile(zip_path, "w") as zipObj:
            for name, img in processed_images:
                img_path = os.path.join(tmpdirname, name)
                cv2.imwrite(img_path, img)
                zipObj.write(img_path, arcname=name)
        with open(zip_path, "rb") as fp:
            zip_bytes = fp.read()
    return zip_bytes


def main():
    st.title("Image Processing WebApp")
    col1, col2 = st.columns(2)

    with col1:
        st.header("Upload Media")
        media_type = st.selectbox("Select Media Type", ["Images", "Video"])
        if media_type == "Images":
            uploaded_files = st.file_uploader(
                "Choose images", type=["jpg", "png", "jpeg"], accept_multiple_files=True
            )
        else:
            uploaded_file = st.file_uploader(
                "Choose a video",
                type=["mp4", "avi", "mov"],
                accept_multiple_files=False,
            )
            if uploaded_file:
                uploaded_files = [uploaded_file]  # Make it a list for consistency
            else:
                uploaded_files = []

        # Model selection
        model_config = st.selectbox(
            "Select Image Processing Model Configuration",
            ["Model A", "Model B", "Model C", "Custom Model"],
            key="model_config",
        )

        # Custom model selection, if Cutom model is selected
        if model_config == "Custom Model":
            custom_model_cfg_file = st.file_uploader(
                "Choose a custom model config file",
                type=["py"],
                accept_multiple_files=False,
            )
            custom_model_chckpoint_file = st.file_uploader(
                "Choose a custom model checkpoint file",
                type=["pth"],
                accept_multiple_files=False,
            )

        # Always display processing parameters (using 3 number inputs for color)
        st.subheader("Processing Parameters")
        st.write("Select the color for the visualization:")
        rgb_col1, rgb_col2, rgb_col3 = st.columns(3)

        with rgb_col1:
            r = st.number_input("Red", min_value=0, max_value=255, value=0, step=1)

        with rgb_col2:
            g = st.number_input("Green", min_value=0, max_value=255, value=255, step=1)

        with rgb_col3:
            b = st.number_input("Blue", min_value=0, max_value=255, value=0, step=1)

        color = (r, g, b)

        # Bitrate input (only if video is selected)
        if media_type == "Video":
            bitrate = st.number_input(
                "Enter video bitrate (Kbps)",
                min_value=100,
                max_value=100000,
                value=5000,
                step=500,
            )
        else:
            bitrate = None  # Not applicable for images

        process_button = st.button("Process")

    with col2:
        st.header("Preview / Download")
        if "processed" not in st.session_state:
            st.session_state["processed"] = False

        if process_button and uploaded_files:
            # Clear previous processed data
            st.session_state["processed"] = True
            st.session_state["media_type"] = media_type
            st.session_state.pop("processed_images", None)
            st.session_state.pop("video_bytes", None)
            st.session_state.pop("zip_bytes", None)

            params = {"color": color}

            # Instantiate the model class based on selection
            with st.spinner("Loading model..."):
                if model_config == "Model A":
                    model_instance = ModelA(params)
                elif model_config == "Model B":
                    model_instance = ModelB(params)
                elif model_config == "Model C":
                    model_instance = ModelC(params)
                elif model_config == "Custom Model":
                    params["checkpoint_file"] = custom_model_chckpoint_file
                    params["config_file"] = custom_model_cfg_file
                    model_instance = CustomModel(params)
                else:
                    st.error("Invalid model selected.")
                    return

            if media_type == "Images":
                processed_images = process_images(uploaded_files, model_instance)
                st.session_state["processed_images"] = processed_images

                # Save processed images to a zip file and store in session state
                zip_bytes = save_images_to_zip(processed_images)
                st.session_state["zip_bytes"] = zip_bytes

                # Display images
                st.image(
                    [img for _, img in processed_images],
                    channels="BGR",
                    caption=[name for name, _ in processed_images],
                )

                # Download button for images
                st.download_button(
                    label="Download Processed Images",
                    data=st.session_state["zip_bytes"],
                    file_name="processed_images.zip",
                    mime="application/zip",
                )
            else:
                video_bytes = process_video(uploaded_files[0], model_instance, bitrate)
                st.session_state["video_bytes"] = video_bytes

                # Display video
                st.video(st.session_state["video_bytes"])

                # Download button for video
                st.download_button(
                    label="Download Processed Video",
                    data=st.session_state["video_bytes"],
                    file_name="processed_video.mp4",
                    mime="video/mp4",
                )
        elif not uploaded_files and process_button:
            st.warning("Please upload files to process.")
        elif st.session_state.get("processed"):
            # Display stored processed data
            if st.session_state["media_type"] == "Images":
                processed_images = st.session_state["processed_images"]
                st.image(
                    [img for _, img in processed_images],
                    channels="BGR",
                    caption=[name for name, _ in processed_images],
                )
                # Download button for images
                st.download_button(
                    label="Download Processed Images",
                    data=st.session_state["zip_bytes"],
                    file_name="processed_images.zip",
                    mime="application/zip",
                )
            else:
                st.video(st.session_state["video_bytes"])
                # Download button for video
                st.download_button(
                    label="Download Processed Video",
                    data=st.session_state["video_bytes"],
                    file_name="processed_video.mp4",
                    mime="video/mp4",
                )


if __name__ == "__main__":
    main()
