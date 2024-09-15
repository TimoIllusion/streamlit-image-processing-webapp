# frame_processor.py

import cv2


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
