import torch
import torch.nn.functional as F
from typing import List, Tuple
from src.reservoir_tta.reservoir_sampling import StyleReservoir

class OnlineStyleClustering:
    """
    Online clustering of style vectors using a DP-Means–like method.
    Each style vector is assigned to an existing cluster or forms a new one
    if it exceeds a threshold distance τ.
    """

    def __init__(self, tau: float, max_reservoir_size: int = 500, device='cpu'):
        self.tau = tau  # Distance threshold for creating new clusters
        self.device = device

        self.reservoir = StyleReservoir(max_size=max_reservoir_size)
        self.centroids: List[torch.Tensor] = []  # List of (D,) tensors

    def _distance(self, x: torch.Tensor, y: torch.Tensor) -> float:
        return F.pairwise_distance(x.unsqueeze(0), y.unsqueeze(0), p=2).item()

    def _assign_cluster(self, vec: torch.Tensor) -> Tuple[int, float]:
        """
        Assign a single style vector to the nearest cluster.
        If too far from all, return -1 to indicate a new cluster.
        """
        if not self.centroids:
            return -1, float('inf')

        dists = [self._distance(vec, c) for c in self.centroids]
        min_dist = min(dists)
        min_idx = dists.index(min_dist)

        if min_dist > self.tau:
            return -1, min_dist
        return min_idx, min_dist

    def update(self, style_vectors: torch.Tensor) -> List[int]:
        """
        Process a batch of style vectors.
        Returns a list of cluster assignments for each vector.
        """
        assignments = []
        for vec in style_vectors:
            cluster_idx, dist = self._assign_cluster(vec)
            if cluster_idx == -1:
                # New cluster
                self.centroids.append(vec.clone().detach().to(self.device))
                assignments.append(len(self.centroids) - 1)
            else:
                # Existing cluster
                assignments.append(cluster_idx)

            # Always update reservoir for later re-clustering or analysis
            self.reservoir.update(vec.unsqueeze(0))

        return assignments

    def get_centroids(self) -> torch.Tensor:
        if not self.centroids:
            return torch.empty(0)
        return torch.stack(self.centroids)

    def reset(self):
        self.centroids.clear()
        self.reservoir.reset()

    def __len__(self):
        return len(self.centroids)
