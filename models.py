# models.py

import cv2
import logging
from abc import ABC, abstractmethod

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
