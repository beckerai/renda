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
from typing import Sequence

import torch
from torch.utils.data import TensorDataset

from renda._checks import _check_scalar, _check_scalar_or_sequence, _check_seed
from renda.random import temp_seed


class GalaxyDataset(TensorDataset):
    def __init__(
        self,
        num_classes: int = 3,
        num_samples_per_class: int | Sequence[int] = 480,
        num_rotations: float | int = 1.0,
        entwinement: float | int = 0.3,
        seed: int | None = None,
    ) -> None:
        self.num_classes = _check_scalar(
            scalar=num_classes,
            type_=int,
            name="num_classes",
            gt=0,
        )
        self.num_samples_per_class = _check_scalar_or_sequence(
            scalar_or_sequence=num_samples_per_class,
            type_=int,
            name="num_samples_per_class",
            length=self.num_classes,
            gt=0,
        )
        self.num_rotations = _check_scalar(
            scalar=num_rotations,
            type_=float | int,
            name="num_rotations",
            gt=0,
        )
        self.entwinement = _check_scalar(
            scalar=entwinement,
            type_=float | int,
            name="entwinement",
            gt=0,
        )
        self.seed = _check_seed(seed)

        with temp_seed(seed):
            self._X, self._Y = self._generate_dataset()

        super().__init__(self._X, self._Y)

    def _generate_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.num_samples_per_class, int):
            num_samples_per_class = (self.num_samples_per_class,) * self.num_classes
        else:
            num_samples_per_class = self.num_samples_per_class

        t_min = 0.0
        t_max = 2.0 * math.pi * self.num_rotations
        phi_abs_max = math.pi / self.num_classes

        X_ = []
        Y_ = []

        for y, n in enumerate(num_samples_per_class):
            # Sampling of curve parameter values
            # They represent the current spiral arm with no expansion
            t_ = []
            t_count = 0
            while t_count < n:
                t__ = self.entwinement * t_max * (0.5 * torch.randn(2 * n) + 1.0)
                t__ = t__[(t_min <= t__).logical_and(t__ <= t_max)]
                t_count += t__.numel()
                t_.append(t__)
            t = torch.cat(t_)
            t = t[0:n]

            # Rejection sampling of noisy phase values
            # They represent the expansion of the current spiral arm
            phi_ = []
            phi_count = 0
            while phi_count < n:
                phi__ = self.entwinement * phi_abs_max * torch.randn(2 * n)
                phi__ = phi__[phi__.abs() <= phi_abs_max]
                phi_count += phi__.numel()
                phi_.append(phi__)
            phi = torch.cat(phi_)
            phi = phi[0:n]

            # Rotation of the current spiral arm by a class-based phase offset
            # This makes the spiral arms distinguishable
            phi_offset = y * 2.0 * math.pi / self.num_classes
            phi += phi_offset

            # Features
            X__ = torch.Tensor(2, n)
            X__[0, :] = t * torch.cos(t + phi)
            X__[1, :] = t * torch.sin(t + phi)
            X__ = X__.t() / t_max
            X_.append(X__)

            # Class labels
            Y__ = y * torch.ones(n, 1)
            Y_.append(Y__)

        X = torch.cat(X_)
        Y = torch.cat(Y_)

        return X, Y
