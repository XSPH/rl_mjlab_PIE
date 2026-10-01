"""Reset-safe chronological sensor history without cross-episode samples."""

import torch


class HistoryBuffer:
    def __init__(self, num_envs: int, length: int, sample_shape: tuple[int, ...], device,
                 dtype=torch.float32):
        self.data = torch.zeros((num_envs, length, *sample_shape), device=device, dtype=dtype)

    def append(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        if mask is None:
            self.data[:, :-1] = self.data[:, 1:].clone()
            self.data[:, -1] = values
        else:
            ids = mask.nonzero(as_tuple=False).flatten()
            self.data[ids, :-1] = self.data[ids, 1:].clone()
            self.data[ids, -1] = values[ids]

    def reset(self, values: torch.Tensor, ids: torch.Tensor) -> None:
        self.data[ids] = values[ids].unsqueeze(1)
