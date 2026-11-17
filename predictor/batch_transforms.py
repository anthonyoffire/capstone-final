"""Apply the image shuffle and tensor augmentations to a whole batch."""

import torch
import torch.nn.functional as F

from image_shuffler import PIL_Shuffler, SEED


class BatchImageTransforms:
    def __init__(
        self,
        grid_size: int,
        device: str | torch.device,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        seed: int = SEED,
        increment_seed: bool = False,
    ):
        self.grid_size = grid_size
        self.device = torch.device(device)
        self.shuffler = PIL_Shuffler(
            grid_size,
            seed=seed,
            increment_seed=increment_seed,
        )
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.mean = torch.tensor(mean, device=self.device)[None, :, None, None]
        self.std = torch.tensor(std, device=self.device)[None, :, None, None]

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
        if augment:
            images = self._augment(images)
        shuffled = self.shuffler(images)
        if shuffled is None:
            height, width = images.shape[-2:]
            message = (
                f"Image dimensions h={height} and w={width} must both be "
                f"divisible by grid size g={self.grid_size}."
            )
            print(f"Error: {message}")
            raise ValueError(message)
        images = shuffled
        return (images - self.mean) / self.std
