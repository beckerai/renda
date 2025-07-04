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

from renda.random import MAX_SEED, MIN_SEED, ensure_seed, is_seed, temp_seed


@pytest.mark.parametrize(
    ("value", "result"),
    (
        pytest.param(42, True, id="42"),
        pytest.param(MIN_SEED, True, id="MIN_SEED"),
        pytest.param(MAX_SEED, True, id="MAX_SEED"),
        pytest.param(4.2, False, id="4.2"),
        pytest.param("forty-two", False, id="forty-two"),
        pytest.param(True, False, id="True"),
        pytest.param(MIN_SEED - 1, False, id="MIN_SEED_minus_1"),
        pytest.param(MAX_SEED + 1, False, id="MAX_SEED_plus_1"),
        pytest.param(None, False, id="None"),
    ),
)
def test_is_seed(value, result):
    assert is_seed(value) is result


@pytest.mark.parametrize(
    ("value", "value_expected"),
    (
        pytest.param(42, 42, id="42"),
        pytest.param(MIN_SEED, MIN_SEED, id="MIN_SEED"),
        pytest.param(MAX_SEED, MAX_SEED, id="MAX_SEED"),
        pytest.param(MIN_SEED - 1, MAX_SEED, id="MIN_SEED_minus_1"),
        pytest.param(MAX_SEED + 1, MIN_SEED, id="MAX_SEED_plus_1"),
        pytest.param(MIN_SEED - 42, MAX_SEED - 41, id="MIN_SEED_minus_42"),
        pytest.param(MAX_SEED + 42, MIN_SEED + 41, id="MAX_SEED_plus_42"),
    ),
)
def test_ensure_seed(value, value_expected):
    value_suggested = ensure_seed(value)
    assert is_seed(value_suggested)
    assert value_suggested == value_expected


@pytest.mark.parametrize(
    "value",
    (
        pytest.param(4.2, id="float"),
        pytest.param("forty-two", id="str"),
        pytest.param(True, id="bool"),
        pytest.param(None, id="NoneType"),
        pytest.param(lambda: 0, id="function"),
    ),
)
def test_ensure_seed_type_error(value):
    with pytest.raises(TypeError, match="`value` must be of type `int`"):
        ensure_seed(value)


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
        pytest.param(4.2, id="4.2"),
        pytest.param("forty-two", id="forty-two"),
        pytest.param(True, id="True"),
        pytest.param(MIN_SEED - 1, id="MIN_SEED_minus_1"),
        pytest.param(MAX_SEED + 1, id="MAX_SEED_plus_1"),
    ),
)
def test_temp_seed_value_error(seed):
    error_message = (
        f"`seed` must be an `int` between `MIN_SEED = {MIN_SEED}` and "
        f"`MAX_SEED = {MAX_SEED}` or None."
    )
    with pytest.raises(ValueError, match=error_message):
        with temp_seed(seed):
            pass
