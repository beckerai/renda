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
import random
from typing import Any, Final, Iterable

import numpy as np
import torch

from renda._checks import _check_seed


MIN_SEED: Final[int] = 0
MAX_SEED: Final[int] = 4294967295  # 2^32 - 1 (uint32)


class temp_seed:
    _seed: int | None
    _random_state: tuple[Any, ...]
    _np_random_state: dict[str, Any]
    _torch_rng_state: torch.Tensor
    _torch_cuda_rng_state_all: Iterable[torch.Tensor]

    def __init__(self, seed: int | None) -> None:
        self._seed = _check_seed(seed)

    def __enter__(self) -> None:  # pragma: no cover
        if self._seed is not None:
            # Store random states
            self._random_state = random.getstate()
            self._np_random_state = np.random.get_state()
            self._torch_rng_state = torch.get_rng_state()
            self._torch_cuda_rng_state_all = torch.cuda.get_rng_state_all()

            # Seed everything
            random.seed(self._seed)
            np.random.seed(self._seed)
            torch.manual_seed(self._seed)
            torch.cuda.manual_seed_all(self._seed)

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # pragma: no cover
        if self._seed is not None:
            # Restore random states
            random.setstate(self._random_state)
            np.random.set_state(self._np_random_state)
            torch.set_rng_state(self._torch_rng_state)
            torch.cuda.set_rng_state_all(self._torch_cuda_rng_state_all)
