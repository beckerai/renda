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
from __future__ import annotations

import operator
import random
from typing import Any, Iterable

import numpy as np
import torch

from renda._checks import _check_scalar, _check_seed, _CheckError


MIN_SEED = 0
MAX_SEED = 4294967295  # 2^32 - 1 (uint32)


class Seed:
    def __init__(self, value: int | None | Seed) -> None:
        if isinstance(value, Seed):
            value = value.value

        self._value = _check_scalar(
            scalar=value,
            type_=int | Seed,
            allow_none=True,
            name="value",
            ge=MIN_SEED,
            le=MAX_SEED,
        )

    @property
    def value(self) -> int | None:
        return self._value

    def __str__(self):
        return str(self._value)

    def __add__(self, other: int | Seed | None) -> Seed:
        return self._operator(other, operator.add)

    def __sub__(self, other: int | Seed | None) -> Seed:
        return self._operator(other, operator.sub)

    def __mul__(self, other: int | Seed | None) -> Seed:
        return self._operator(other, operator.mul)

    def _operator(self, other, operator_):
        _check_scalar(
            scalar=other,
            type_=int | Seed,
            allow_none=True,
            name="other",
        )

        value = self._value
        if isinstance(other, Seed):
            other = other.value

        if value is None or other is None:
            return Seed(None)
        else:
            try:
                result = operator_(value, other)
                result %= MAX_SEED + 1
                result = Seed(result)
            except (TypeError, _CheckError):
                raise TypeError()
            return result

    def __radd__(self, other: int | Seed | None) -> Seed:
        return self + other

    def __rsub__(self, other: int | Seed | None) -> Seed:
        return self - other

    def __rmul__(self, other: int | Seed | None) -> Seed:
        return self * other

    def __iadd__(self, other: int | Seed | None) -> Seed:
        return self + other

    def __isub__(self, other: int | Seed | None) -> Seed:
        return self - other

    def __imul__(self, other: int | Seed | None) -> Seed:
        return self * other


def _int_to_seed(int_: int) -> int:
    if not isinstance(int_, int):
        raise TypeError(f"`int_` must be of type `int`, got `{int_}`")

    # This only works because MIN_SEED = 0 and MAX_SEED > 0
    # A more general solutions would be nice
    return int_ % (MAX_SEED + 1)


class temp_seed:
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
