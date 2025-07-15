# Copyright 2025 Martin Becker
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Optional, Sequence

import torch
from torch.utils.data import TensorDataset

from renda.random import check_seed, temp_seed


class GalaxyDataset(TensorDataset):
    def __init__(
        self,
        num_classes: int = 3,
        num_samples_per_class: int | Sequence[int] = 480,
        num_rotations: float | int = 1.0,
        sigma: float | int = 0.3,
        seed: Optional[int] = 0,
    ) -> None:
        if not (isinstance(num_classes, int) and num_classes > 0):
            raise ValueError("`num_classes` must be a positive `int`")

        if isinstance(num_samples_per_class, int) and num_samples_per_class > 0:
            _num_samples_per_class = (num_samples_per_class,) * num_classes
        elif (
            isinstance(num_samples_per_class, Sequence)
            and len(num_samples_per_class) == num_classes
            and all(isinstance(n, int) for n in num_samples_per_class)
            and all(n > 0 for n in num_samples_per_class)
        ):
            _num_samples_per_class = num_samples_per_class
        else:
            raise ValueError("`num_samples_per_class` must be a positive `int`")

        if not (isinstance(num_rotations, (float, int)) and num_rotations > 0):
            raise ValueError("`num_rotations` must be a positive `float` or `int`")

        if not (isinstance(sigma, (float, int)) and sigma > 0):
            raise ValueError("`sigma` must be a positive `float` or `int`")

        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class
        self._num_samples_per_class = _num_samples_per_class
        self.num_rotations = num_rotations
        self.sigma = sigma
        self.seed = check_seed(seed, allow_none=True)

        with temp_seed(self.seed):
            t_min = 0.0
            t_max = 2.0 * math.pi * self.num_rotations
            phi_abs_max = math.pi / self.num_classes

            X = []
            Y = []

            for y, n in enumerate(self._num_samples_per_class, start=1):
                # Rejection sampling of curve parameter values
                # These represent the current spiral arm with no expansion
                t = []
                t_count = 0
                while t_count < n:
                    t_ = self.sigma * t_max * (0.5 * torch.randn(2 * n) + 1.0)
                    t_ = t_[(t_min <= t_).logical_and(t_ <= t_max)]
                    t_count += t_.numel()
                    t.append(t_)
                t = torch.cat(t)
                t = t[0:n]

                # Rejection sampling of noisy phase values
                # These represent the extent of the current spiral arm
                phi = []
                phi_count = 0
                while phi_count < n:
                    phi_ = self.sigma * phi_abs_max * torch.randn(2 * n)
                    phi_ = phi_[phi_.abs() <= phi_abs_max]
                    phi_count += phi_.numel()
                    phi.append(phi_)
                phi = torch.cat(phi)
                phi = phi[0:n]

                # Rotation of the current spiral arm by a class-based phase offset
                # This makes the spiral arms distinguishable
                phi_offset = (y - 1) * 2.0 * math.pi / self.num_classes
                phi += phi_offset

                # Features
                X_ = torch.Tensor(2, n)
                X_[0, :] = t * torch.cos(t + phi)
                X_[1, :] = t * torch.sin(t + phi)
                X_ = X_.t() / t_max
                X.append(X_)

                # Class labels
                Y_ = y * torch.ones(n, 1)
                Y.append(Y_)

            self.X = torch.cat(X)
            self.Y = torch.cat(Y)

        super().__init__(self.X, self.Y)
