import time
import logging
from abc import ABC, abstractmethod

import cv2

logger = logging.getLogger(__name__)


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
        time.sleep(2)

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
        time.sleep(2)

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
        time.sleep(2)

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


class CustomModel(ImageProcessor):

    def setup(self):
        # Any setup specific to Custom Model
        logger.info("Setting up Custom Model")
        time.sleep(2)
        logger.info(self.params.get("checkpoint_file").name)
        logger.info(self.params.get("config_file").name)

    def execute(self, image):
        color = self.params.get("color", (255, 255, 255))
        processed_image = image.copy()
        height, width = processed_image.shape[:2]
        cv2.rectangle(
            processed_image,
            (int(width * 0.30), int(height * 0.30)),
            (int(width * 0.70), int(height * 0.70)),
            color,
            5,
        )
        return processed_image
