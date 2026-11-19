"""Phi-alpha PCA implemented with ordinary PCA in the transformed coordinates."""

import numpy as np
from sklearn.decomposition import PCA


def signed_power(values, exponent):
    """Apply sign(x) * abs(x)**exponent elementwise, including at zero."""
    if exponent <= 0:
        raise ValueError("The power exponent must be positive.")
    if exponent == 1:
        return values
    return np.copysign(np.power(np.abs(values), exponent), values)


class AlphaPCA:
    """Learn principal directions on powered inputs and invert the score map."""

    def __init__(self, alpha, n_components, random_state):
        self.alpha = alpha
        self.pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=random_state,
        )

    def fit(self, images):
        self.pca.fit(signed_power(images, self.alpha))
        return self

    def transform(self, images):
        scores = self.pca.transform(signed_power(images, self.alpha))
        return signed_power(scores, 1 / self.alpha)
