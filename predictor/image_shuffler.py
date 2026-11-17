import random
from typing import NoReturn, overload

import cv2 as cv
import numpy as np
from PIL import Image
import argparse
import torch
from torch.utils.data import get_worker_info

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

    def _split(self, grid_size: int, tile_width: int, tile_height: int) -> None:
        self._pieces = [
            self.original[
                row * tile_height:(row + 1) * tile_height,
                column * tile_width:(column + 1) * tile_width,
            ]
            for row in range(grid_size)
            for column in range(grid_size)
        ]

    def _generate_image(self, cols: int) -> None:
        chunks = [
            np.hstack(self._pieces[i:i + cols])
            for i in range(0, len(self._pieces), cols)
        ]
        self.shuffled = np.vstack(chunks)

    def shuffle(
        self,
        grid_size: int,
        seed: int | None = None,
    ) -> Image.Image | None:
        if not isinstance(grid_size, int) or grid_size <= 0:
            raise ValueError("grid_size must be a positive integer")
        if self.y % grid_size or self.x % grid_size:
            return None

        tile_width = self.x // grid_size
        tile_height = self.y // grid_size
        self._split(grid_size, tile_width, tile_height)

        rng = random if seed is None else random.Random(seed)
        rng.shuffle(self._pieces)

        self._generate_image(grid_size)

        return Image.fromarray(cv.cvtColor(self.shuffled, cv.COLOR_BGR2RGB))

#----------------------------------------------------------------------
# PIL_Shuffler: Callable wrapper for Shuffler
#----------------------------------------------------------------------
class PIL_Shuffler:
    def __init__(
        self,
        grid_size: int,
        seed: int = SEED,
        increment_seed: bool = False,
    ):
        if not isinstance(grid_size, int) or grid_size <= 0:
            raise ValueError("grid_size must be a positive integer")
        self.grid_size = grid_size
        self.seed = seed
        self.increment_seed = increment_seed
        self._image_count = 0
        self._tensor_permutations: dict[torch.device, torch.Tensor] = {}
        self._tensor_pixel_indices: dict[
            torch.device, tuple[int, int, torch.Tensor]
        ] = {}

    @overload
    def __call__(self, img: Image.Image) -> Image.Image: ...

    @overload
    def __call__(self, img: torch.Tensor) -> torch.Tensor | None: ...

    def __call__(
        self,
        img: Image.Image | torch.Tensor,
    ) -> Image.Image | torch.Tensor | None:
        if isinstance(img, Image.Image):
            if img.height % self.grid_size or img.width % self.grid_size:
                self._report_invalid_size(img.height, img.width)
            seed = (
                self._next_seeds(1)[0]
                if self.increment_seed
                else self.seed
            )
            if self.grid_size == 1:
                return img
            shuffled = Shuffler(img).shuffle(self.grid_size, seed=seed)
            if shuffled is None:
                self._report_invalid_size(img.height, img.width)
            return shuffled

        if isinstance(img, torch.Tensor):
            return self._shuffle_tensor(img)

        raise TypeError("Expected a PIL image or torch tensor")

    def _shuffle_tensor(self, images: torch.Tensor) -> torch.Tensor | None:
        single_image = images.ndim == 3
        if single_image:
            images = images.unsqueeze(0)
        elif images.ndim != 4:
            raise ValueError("Tensor images must have CHW or BCHW dimensions")

        batch, channels, height, width = images.shape
        if height % self.grid_size or width % self.grid_size:
            return None
        if self.increment_seed:
            seeds = self._next_seeds(batch)
            if self.grid_size == 1 or batch == 0:
                return images.squeeze(0) if single_image else images
            pixel_indices = self._get_tensor_pixel_indices_for_seeds(
                height, width, seeds, images.device
            )
            shuffled = torch.gather(
                images.flatten(2),
                2,
                pixel_indices[:, None, :].expand(-1, channels, -1),
            )
            shuffled = shuffled.reshape(batch, channels, height, width)
            return shuffled.squeeze(0) if single_image else shuffled

        if self.grid_size == 1:
            return images.squeeze(0) if single_image else images

        if height == self.grid_size and width == self.grid_size:
            shuffled = images.flatten(2).index_select(
                2, self._get_tensor_permutation(images.device)
            ).reshape_as(images)
            return shuffled.squeeze(0) if single_image else shuffled

        pixel_indices = self._get_tensor_pixel_indices(
            images.device, height, width
        )
        shuffled = images.flatten(2).index_select(2, pixel_indices)
        shuffled = shuffled.reshape(batch, channels, height, width)
        return shuffled.squeeze(0) if single_image else shuffled

    def _get_tensor_pixel_indices(
        self,
        device: torch.device,
        height: int,
        width: int,
    ) -> torch.Tensor:
        cached = self._tensor_pixel_indices.get(device)
        if cached is not None and cached[:2] == (height, width):
            return cached[2]

        indices = self._make_tensor_pixel_indices(height, width, self.seed)
        pixel_indices = torch.tensor(indices, dtype=torch.long, device=device)
        self._tensor_pixel_indices[device] = (height, width, pixel_indices)
        return pixel_indices

    def _get_tensor_pixel_indices_for_seeds(
        self,
        height: int,
        width: int,
        seeds: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        grid_size = self.grid_size
        tile_height = height // grid_size
        tile_width = width // grid_size
        tile_count = grid_size * grid_size
        permutations = np.empty((len(seeds), tile_count), dtype=np.int64)
        for index, seed in enumerate(seeds):
            permutations[index] = self._make_tile_permutation(seed)

        y, x = np.indices((height, width))
        destination_tiles = (
            (y // tile_height) * grid_size + (x // tile_width)
        ).reshape(-1)
        source_tiles = permutations[:, destination_tiles]
        source_y = (
            (source_tiles // grid_size) * tile_height
            + (y % tile_height).reshape(1, -1)
        )
        source_x = (
            (source_tiles % grid_size) * tile_width
            + (x % tile_width).reshape(1, -1)
        )
        indices = source_y * width + source_x
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _make_tensor_pixel_indices(
        self,
        height: int,
        width: int,
        seed: int,
    ) -> np.ndarray:
        grid_size = self.grid_size
        tile_height = height // grid_size
        tile_width = width // grid_size
        tile_permutation = self._make_tile_permutation(seed)
        y, x = np.indices((height, width))
        source_tiles = np.asarray(tile_permutation)[
            (y // tile_height) * grid_size + (x // tile_width)
        ]
        source_y = (
            (source_tiles // grid_size) * tile_height + y % tile_height
        )
        source_x = (
            (source_tiles % grid_size) * tile_width + x % tile_width
        )
        return (source_y * width + source_x).reshape(-1)

    def _make_tile_permutation(self, seed: int) -> list[int]:
        permutation = list(range(self.grid_size * self.grid_size))
        random.Random(seed).shuffle(permutation)
        return permutation

    def _next_seeds(self, count: int) -> list[int]:
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            worker_count = 1
        else:
            worker_id = worker_info.id
            worker_count = worker_info.num_workers

        first_image = self._image_count
        self._image_count += count
        return [
            self.seed + worker_id + (first_image + index) * worker_count
            for index in range(count)
        ]

    def _get_tensor_permutation(self, device: torch.device) -> torch.Tensor:
        permutation = self._tensor_permutations.get(device)
        if permutation is None:
            indices = list(range(self.grid_size * self.grid_size))
            random.Random(self.seed).shuffle(indices)
            permutation = torch.tensor(indices, dtype=torch.long, device=device)
            self._tensor_permutations[device] = permutation
        return permutation

    def _report_invalid_size(self, height: int, width: int) -> NoReturn:
        message = (
            f"Image dimensions h={height} and w={width} must both be "
            f"divisible by grid size g={self.grid_size}."
        )
        print(f"Error: {message}")
        raise ValueError(message)

def validate_grid_size(size_str: str) -> int:
    try:
        size = int(size_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid integer for grid size: '{size_str}'"
        ) from exc
    if size <= 0:
        raise argparse.ArgumentTypeError("Grid size must be a positive integer.")
    return size