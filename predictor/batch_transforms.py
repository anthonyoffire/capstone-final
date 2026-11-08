"""Apply the image shuffle and tensor augmentations to a whole batch."""

import random

import torch
import torch.nn.functional as F

from image_shuffler import SEED


class BatchImageTransforms:
    def __init__(
        self,
        grid_size: int,
        image_size: int,
        device: str | torch.device,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        seed: int = SEED,
    ):
        if image_size % grid_size != 0:
            raise ValueError("grid_size must evenly divide image_size")

        self.grid_size = grid_size
        self.device = torch.device(device)
        indices = list(range(grid_size * grid_size))
        random.Random(seed).shuffle(indices)
        self.permutation = torch.tensor(
            indices, dtype=torch.long, device=self.device
        )
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.mean = torch.tensor(mean, device=self.device)[None, :, None, None]
        self.std = torch.tensor(std, device=self.device)[None, :, None, None]

    def _shuffle_tiles(self, images: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = images.shape
        tile_height = height // self.grid_size
        tile_width = width // self.grid_size
        tiles = (
            images.reshape(
                batch,
                channels,
                self.grid_size,
                tile_height,
                self.grid_size,
                tile_width,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(
                batch,
                channels,
                self.grid_size * self.grid_size,
                tile_height,
                tile_width,
            )
        )
        tiles = tiles.index_select(2, self.permutation)
        return (
            tiles.reshape(
                batch,
                channels,
                self.grid_size,
                self.grid_size,
                tile_height,
                tile_width,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(batch, channels, height, width)
        )

    def _augment(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        flip = torch.rand(
            batch, device=self.device, generator=self.generator
        ) < 0.5
        images = torch.where(flip[:, None, None, None], images.flip(-1), images)

        angles = (
            torch.rand(batch, device=self.device, generator=self.generator) * 20.0
            - 10.0
        )
        radians = torch.deg2rad(angles)
        cosines = torch.cos(radians)
        sines = torch.sin(radians)
        theta = torch.zeros((batch, 2, 3), device=self.device)
        theta[:, 0, 0] = cosines
        theta[:, 0, 1] = sines
        theta[:, 1, 0] = -sines
        theta[:, 1, 1] = cosines
        grid = F.affine_grid(theta, images.shape, align_corners=False)
        images = F.grid_sample(
            images,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )

        brightness = 0.8 + 0.4 * torch.rand(
            batch, device=self.device, generator=self.generator
        )
        contrast = 0.8 + 0.4 * torch.rand(
            batch, device=self.device, generator=self.generator
        )
        brightness = brightness[:, None, None, None]
        contrast = contrast[:, None, None, None]

        def adjust_brightness(tensor: torch.Tensor) -> torch.Tensor:
            return (tensor * brightness).clamp(0.0, 1.0)

        def adjust_contrast(tensor: torch.Tensor) -> torch.Tensor:
            grayscale = (
                tensor[:, 0:1] * 0.2989
                + tensor[:, 1:2] * 0.5870
                + tensor[:, 2:3] * 0.1140
            )
            grayscale_mean = grayscale.mean(dim=(-2, -1), keepdim=True)
            return (
                (tensor - grayscale_mean) * contrast + grayscale_mean
            ).clamp(0.0, 1.0)

        brightness_first = (
            torch.rand(batch, device=self.device, generator=self.generator) < 0.5
        )[:, None, None, None]
        bright_then_contrast = adjust_contrast(adjust_brightness(images))
        contrast_then_bright = adjust_brightness(adjust_contrast(images))
        return torch.where(
            brightness_first, bright_then_contrast, contrast_then_bright
        )

    def __call__(self, images: torch.Tensor, augment: bool = False) -> torch.Tensor:
        images = images.to(dtype=torch.float32).div_(255.0)
        images = self._shuffle_tiles(images)
        if augment:
            images = self._augment(images)
        return (images - self.mean) / self.std
