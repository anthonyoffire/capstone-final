import random
import warnings
import cv2 as cv
import numpy as np
from PIL import Image
import argparse

# Random seed for reproducibility
SEED = 1337

#----------------------------------------------------------------------
# Shuffler: Modified from image-shuffler package by Gabriel Stork
#           Changes: Input is PIL Image instead of reading from file
#                    Uses random seed for research reproducibility
#                    shuffle returns a PIL Image
#                    Added callable wrapper class for transforms
#                    Added argument validator for this specific case 
#----------------------------------------------------------------------
class Shuffler:
    def __init__(self, image: Image.Image) -> None:
        if not isinstance(image, Image.Image):
            raise TypeError("Expected PIL Image")

        self.original = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

        if self.original is None:
            raise ValueError("Image is None")

        self.shuffled = self.original.copy()
        self.x = self.original.shape[1]
        self.y = self.original.shape[0]

        self._pieces = []

    def _check_argument(self, matrix: tuple) -> None:
        if len(matrix) != 2 or not all(isinstance(x, int) for x in matrix):
            raise ValueError(
                'matrix must be 2-dimensional containing only integer numbers'
            )
        elif min(matrix) <= 0:
            raise ValueError('matrix values must be greater than 0')
        elif matrix[0] > self.x or matrix[1] > self.y:
            raise ValueError('number of splits greater than pixels')

    def _check_pixel_loss(self, x_missing: int, y_missing: int) -> None:
        new_x = self.x - x_missing
        new_y = self.y - y_missing

        if (x_missing + y_missing) > 0:
            warnings.warn(
                'Splitting images into non-integer intervals causes pixel '
                f'loss. Original Shape: ({self.x}, {self.y}) New Shape: '
                f'({new_x}, {new_y})',
                stacklevel=2,
            )

    def _split(self, x: int, y: int, x_list: list, y_list: list) -> None:
        if len(self._pieces) > 0:
            self._pieces.clear()

        for y_n, y_piece in enumerate(y_list):
            for x_n, x_piece in enumerate(x_list):
                self._pieces.append(
                    self.original[y_n * y:y_piece, x_n * x:x_piece]
                )

    def _generate_image(self, cols: int) -> None:
        chunks = [
            np.hstack(self._pieces[i:i + cols])
            for i in range(0, len(self._pieces), cols)
        ]
        self.shuffled = np.vstack(chunks)

    def shuffle(self, grid_size: int, seed: int = None) -> Image.Image:
        matrix = (grid_size, grid_size)
        self._check_argument(matrix)

        x = int(self.x / matrix[0])
        x_missing = self.x - (x * matrix[0])
        x_list = list(range(x, self.x + 1, x))

        y = int(self.y / matrix[1])
        y_missing = self.y - (y * matrix[1])
        y_list = list(range(y, self.y + 1, y))

        self._check_pixel_loss(x_missing, y_missing)
        self._split(x, y, x_list, y_list)

        if seed is not None:
            random.seed(seed)
        random.shuffle(self._pieces)

        self._generate_image(matrix[0])
        
        return Image.fromarray(cv.cvtColor(self.shuffled, cv.COLOR_BGR2RGB))

#----------------------------------------------------------------------
# PIL_Shuffler: Callable wrapper for Shuffler
#----------------------------------------------------------------------
class PIL_Shuffler:
    def __init__(self, grid_size: int):
        self.grid_size = grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.grid_size == 1: # No shuffle
            return img
        shuffler = Shuffler(img)
        return shuffler.shuffle(self.grid_size, seed=SEED)

def validate_grid_size(size_str: str) -> int:
    try:
        size = int(size_str)
    except:
        raise argparse.ArgumentTypeError(f"Invalid integer for grid size: '{size_str}'")
    acceptable_sizes = {1, 2, 4, 8, 16, 32, 64, 128}
    if size not in acceptable_sizes:
        raise argparse.ArgumentTypeError(f"Grid size must be a power of 2, <= 128, > 0")
    return size