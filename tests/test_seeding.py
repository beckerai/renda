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

import numpy as np
import pytest
import torch

from renda._exceptions import _CheckError
from renda.seeding import MAX_SEED, MIN_SEED, temp_seed


@pytest.mark.parametrize(
    (
        "set_global_seed",
        "get_10_random_numbers",
        "all_equal",
    ),
    (
        pytest.param(
            random.seed,
            lambda: [random.random() for _ in range(10)],
            lambda a, b: a == b,
            id="random",
        ),
        pytest.param(
            np.random.seed,
            lambda: np.random.rand(10),
            lambda a, b: np.all(a == b),
            id="numpy",
        ),
        pytest.param(
            torch.manual_seed,
            lambda: torch.rand(10),
            lambda a, b: torch.all(torch.eq(a, b)),
            id="torch",
        ),
        pytest.param(
            torch.cuda.manual_seed_all,
            lambda: torch.rand(10, device=torch.device("cuda")),
            lambda a, b: torch.all(torch.eq(a, b)),
            id="torch_cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="requires torch with CUDA support",
            ),
        ),
    ),
)
class TestTempSeed:
    def test_same_seeds(
        self,
        set_global_seed,
        get_10_random_numbers,
        all_equal,
    ):
        set_global_seed(42)
        with temp_seed(0):
            a = get_10_random_numbers()
        with temp_seed(0):
            b = get_10_random_numbers()

        assert all_equal(a, b)

    def test_different_seeds(
        self,
        set_global_seed,
        get_10_random_numbers,
        all_equal,
    ):
        set_global_seed(42)
        with temp_seed(0):
            a = get_10_random_numbers()
        with temp_seed(1):
            b = get_10_random_numbers()

        assert not all_equal(a, b)

    def test_none_as_seed(
        self,
        set_global_seed,
        get_10_random_numbers,
        all_equal,
    ):
        set_global_seed(42)
        a = get_10_random_numbers()

        set_global_seed(42)
        with temp_seed(None):
            b = get_10_random_numbers()

        assert all_equal(a, b)

    def test_independence_of_nested_calls(
        self,
        set_global_seed,
        get_10_random_numbers,
        all_equal,
    ):
        set_global_seed(42)
        with temp_seed(0):
            a1 = get_10_random_numbers()
            with temp_seed(0):
                b1 = get_10_random_numbers()
                b2 = get_10_random_numbers()
            a2 = get_10_random_numbers()

        assert all_equal(a1, b1)
        assert all_equal(a2, b2)

    def test_independence_of_global_rng(
        self,
        set_global_seed,
        get_10_random_numbers,
        all_equal,
    ):
        set_global_seed(42)
        get_10_random_numbers()
        a = get_10_random_numbers()

        set_global_seed(42)
        get_10_random_numbers()
        with temp_seed(0):
            get_10_random_numbers()
        b = get_10_random_numbers()

        assert all_equal(a, b)


@pytest.mark.parametrize(
    "seed",
    (
        pytest.param(0.0, id="0.0"),
        pytest.param(MIN_SEED - 1, id="MIN_SEED - 1"),
        pytest.param(MAX_SEED + 1, id="MAX_SEED + 1"),
        pytest.param("zero", id="zero"),
        pytest.param(lambda: 0, id="lambda: 0"),
    ),
)
def test_temp_seed_seed_arg_invalid(seed):
    with pytest.raises(_CheckError):
        temp_seed(seed)
