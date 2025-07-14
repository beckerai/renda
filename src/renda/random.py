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
from typing import Any, Optional

import numpy as np
import torch


MIN_SEED = 0
MAX_SEED = 4294967295  # 2^32 - 1 (uint32)


def is_seed(value: Any) -> bool:
    return isinstance(value, int) and MIN_SEED <= value <= MAX_SEED


def ensure_seed(value: int) -> int:
    if isinstance(value, int):
        # This only works because MIN_SEED = 0 and MIN_SEED > 0
        # A more general solutions would be nice
        return value % (MAX_SEED + 1)
    else:
        raise TypeError("`value` must be of type `int`")


def check_seed(value: Any, allow_none: bool = False) -> Any:
    if not isinstance(allow_none, bool):
        raise TypeError("`allow_none` must be of type `bool`")

    if is_seed(value) or (value is None and allow_none):
        return value
    else:
        raise ValueError(
            f"`seed` must be an `int` between `MIN_SEED = {MIN_SEED}` and "
            f"`MAX_SEED = {MAX_SEED}`" + allow_none * " or None"
        )


class temp_seed:
    def __init__(self, seed: Optional[int]) -> None:
        self._seed = check_seed(seed, allow_none=True)

        self._random_state = None
        self._np_random_state = None
        self._torch_rng_state = None
        self._torch_cuda_rng_state_all = None

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
