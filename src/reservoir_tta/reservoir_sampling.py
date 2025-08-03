import random
import torch

class StyleReservoir:
    """
    Fixed-size buffer of style vectors using reservoir sampling.
    Approximates uniform sampling from all seen vectors.
    """

    def __init__(self, max_size: int, seed: int = 42):
        self.max_size = max_size
        self.buffer = []  # List of torch.Tensor vectors
        self.total_seen = 0
        random.seed(seed)

    def update(self, new_vectors: torch.Tensor):
        """
        Add new style vectors to the reservoir using reservoir sampling.

        Args:
            new_vectors (Tensor): shape (B, D)
        """
        for vec in new_vectors:
            self.total_seen += 1
            if len(self.buffer) < self.max_size:
                self.buffer.append(vec.clone())
            else:
                i = random.randint(0, self.total_seen - 1)
                if i < self.max_size:
                    self.buffer[i] = vec.clone()

    def get(self) -> torch.Tensor:
        """
        Returns: Tensor of all style vectors in the buffer (N, D)
        """
        if not self.buffer:
            return torch.empty(0)
        return torch.stack(self.buffer)

    def __len__(self):
        return len(self.buffer)

    def reset(self):
        self.buffer = []
        self.total_seen = 0
