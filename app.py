import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import logging
from zipfile import ZipFile
from abc import ABC, abstractmethod
from moviepy.editor import VideoFileClip

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define a class to process frames and keep track of progress
class FrameProcessor:
    def __init__(self, model_instance, total_frames, progress_bar):
        self.model_instance = model_instance
        self.total_frames = total_frames
        self.progress_bar = progress_bar
        self.frame_count = 0

    def process(self, frame):
        # Convert frame from RGB to BGR for OpenCV processing
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        processed_frame_bgr = self.model_instance.execute(frame_bgr)
        # Convert back to RGB for moviepy
        processed_frame_rgb = cv2.cvtColor(processed_frame_bgr, cv2.COLOR_BGR2RGB)
        # Update progress bar
        self.frame_count += 1
        progress = min(self.frame_count / self.total_frames, 1.0)
        self.progress_bar.progress(progress)
        return processed_frame_rgb


class ImageProcessor(ABC):
    def __init__(self, params):
        self.params = params
        self.setup()

    @abstractmethod
    def setup(self): ...

    @abstractmethod
    def execute(self, image): ...


# Model A
class ModelA(ImageProcessor):
    def setup(self):
        # Any setup specific to Model A
        logger.info("Setting up Model A")

    def execute(self, image):
        color = self.params.get("color", (0, 255, 0))
        processed_image = image.copy()
        height, width = processed_image.shape[:2]
        cv2.rectangle(
            processed_image,
            (int(width * 0.25), int(height * 0.25)),
            (int(width * 0.75), int(height * 0.75)),
            color,
            5,
        )
        return processed_image


# Model B
class ModelB(ImageProcessor):
    def setup(self):
        # Any setup specific to Model B
        logger.info("Setting up Model B")

    def execute(self, image):
        color = self.params.get("color", (255, 0, 0))
        processed_image = image.copy()
        height, width = processed_image.shape[:2]
        cv2.rectangle(
            processed_image,
            (int(width * 0.10), int(height * 0.10)),
            (int(width * 0.90), int(height * 0.90)),
            color,
            5,
        )
        return processed_image


# Model C
class ModelC(ImageProcessor):
    def setup(self):
        # Any setup specific to Model C
        logger.info("Setting up Model C")

    def execute(self, image):
        color = self.params.get("color", (0, 0, 255))
        processed_image = image.copy()
        height, width = processed_image.shape[:2]
        cv2.rectangle(
            processed_image,
            (int(width * 0.40), int(height * 0.40)),
            (int(width * 0.60), int(height * 0.60)),
            color,
            5,
        )
        return processed_image


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
            ["Model A", "Model B", "Model C"],
            key="model_config",
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
            if model_config == "Model A":
                model_instance = ModelA(params)
            elif model_config == "Model B":
                model_instance = ModelB(params)
            elif model_config == "Model C":
                model_instance = ModelC(params)
            else:
                st.error("Invalid model selected.")
                return

            if media_type == "Images":
                logger.info("Processing images")
                processed_images = []
                total_images = len(uploaded_files)
                progress_bar = st.progress(0)
                for idx, uploaded_file in enumerate(uploaded_files):
                    logger.info(f"Processing image {idx + 1}/{total_images}")
                    file_bytes = np.asarray(
                        bytearray(uploaded_file.read()), dtype=np.uint8
                    )
                    image = cv2.imdecode(file_bytes, 1)
                    processed_image = model_instance.execute(image)
                    processed_images.append((uploaded_file.name, processed_image))
                    progress_bar.progress((idx + 1) / total_images)
                progress_bar.empty()
                st.session_state["processed_images"] = processed_images

                # Save processed images to a zip file and store in session state
                with tempfile.TemporaryDirectory() as tmpdirname:
                    zip_path = os.path.join(tmpdirname, "processed_images.zip")
                    with ZipFile(zip_path, "w") as zipObj:
                        for name, img in processed_images:
                            img_path = os.path.join(tmpdirname, name)
                            cv2.imwrite(img_path, img)
                            zipObj.write(img_path, arcname=name)
                    with open(zip_path, "rb") as fp:
                        zip_bytes = fp.read()
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
                logger.info("Processing video")
                uploaded_file = uploaded_files[0]
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                tfile.flush()
                tfile.close()

                # Define the output video path
                temp_video_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                )
                video_path = temp_video_file.name

                # Use VideoFileClip for on-the-fly processing
                with st.spinner("Processing video..."):
                    clip = VideoFileClip(tfile.name)

                    # Get total frames using clip.reader.nframes
                    total_frames = clip.reader.nframes
                    if total_frames is None or total_frames == 0:
                        total_frames = int(clip.fps * clip.duration)

                    progress_bar = st.progress(0)

                    frame_processor = FrameProcessor(
                        model_instance, total_frames, progress_bar
                    )

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
                st.session_state["video_bytes"] = video_bytes

                # Clean up temporary files
                os.remove(video_path)
                os.remove(tfile.name)

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
