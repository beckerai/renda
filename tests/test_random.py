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

from renda._checks import _check_seed
from renda._exceptions import _CheckError
from renda.random import MAX_SEED, MIN_SEED, Seed, _int_to_seed, temp_seed


@pytest.mark.parametrize(
    "value",
    (
        pytest.param(1, id="1"),
        pytest.param(MIN_SEED, id="MIN_SEED"),
        pytest.param(MAX_SEED, id="MAX_SEED"),
        pytest.param(None, id="None"),
    ),
)
def test_seed_value_arg_valid(value):
    seed = Seed(value)
    assert seed.value == value

    value = Seed(value)
    seed = Seed(value)
    assert seed.value == value.value


@pytest.mark.parametrize(
    "value",
    (
        pytest.param(0.0, id="0.0"),
        pytest.param("zero", id="zero"),
        pytest.param(MIN_SEED - 1, id="MIN_SEED - 1"),
        pytest.param(MAX_SEED + 1, id="MAX_SEED + 1"),
        pytest.param(lambda: 0, id="lambda: 0"),
    ),
)
def test_seed_value_arg_invalid(value):
    with pytest.raises(_CheckError):
        Seed(value)


def test_seed_value_immutable():
    match = "^property 'value' of 'Seed' object has no setter$"
    seed = Seed(0)
    with pytest.raises(AttributeError, match=match):
        seed.value = 1


@pytest.mark.parametrize(
    ("value", "other", "result"),
    (
        pytest.param(1, 1, 2, id="1 + 1 = 2"),
        pytest.param(MAX_SEED, 1, MIN_SEED, id="MAX_SEED + 1 = MIN_SEED"),
        pytest.param(MAX_SEED, 2, MIN_SEED + 1, id="MAX_SEED + 2 = MIN_SEED + 1"),
        pytest.param(None, 1, None, id="None + 1 = None"),
        pytest.param(None, None, None, id="None + None = None"),
    ),
)
class TestSeedAdd:
    def test_seed_add_int(self, value, other, result):
        seed = Seed(value)
        result_ = seed + other
        assert result_.value == result
        result_ = other + seed
        assert result_.value == result
        seed += other
        assert seed.value == result

    def test_seed_add_seed(self, value, other, result):
        seed = Seed(value)
        other = Seed(other)
        result_ = seed + other
        assert result_.value == result
        result_ = other + seed
        assert result_.value == result
        seed += other
        assert seed.value == result


@pytest.mark.parametrize(
    ("value", "other", "result"),
    (
        pytest.param(1, 1, 0, id="1 - 1 = 0"),
        pytest.param(MIN_SEED, 1, MAX_SEED, id="MIN_SEED - 1 = MAX_SEED"),
        pytest.param(MIN_SEED, 2, MAX_SEED - 1, id="MIN_SEED - 2 = MAX_SEED - 1"),
        pytest.param(None, 1, None, id="None - 1 = None"),
        pytest.param(None, None, None, id="None - None = None"),
    ),
)
class TestSeedSub:
    def test_seed_sub_int(self, value, other, result):
        seed = Seed(value)
        result_ = seed - other
        assert result_.value == result
        seed -= other
        assert seed.value == result

    def test_seed_sub_seed(self, value, other, result):
        seed = Seed(value)
        other = Seed(other)
        result_ = seed - other
        assert result_.value == result
        seed -= other
        assert seed.value == result


@pytest.mark.parametrize(
    ("value", "other", "result"),
    (
        pytest.param(2, 2, 4, id="2 * 2 = 4"),
        pytest.param(MAX_SEED, 2, MAX_SEED - 1, id="MAX_SEED * 2 = MAX_SEED - 1"),
        pytest.param(MAX_SEED, 3, MAX_SEED - 2, id="MAX_SEED * 3 = MAX_SEED - 2"),
        pytest.param(None, 1, None, id="None * 1 = None"),
        pytest.param(None, None, None, id="None * None = None"),
    ),
)
class TestSeedMul:
    def test_seed_mul_int(self, value, other, result):
        seed = Seed(value)
        result_ = seed * other
        assert result_.value == result
        result_ = other * seed
        assert result_.value == result
        seed *= other
        assert seed.value == result

    def test_seed_mul_seed(self, value, other, result):
        seed = Seed(value)
        other = Seed(other)
        result_ = seed * other
        assert result_.value == result
        result_ = other * seed
        assert result_.value == result
        seed *= other
        assert seed.value == result


@pytest.mark.parametrize(
    ("int_", "seed_expected"),
    (
        # Seeds
        pytest.param(1, 1, id="1"),
        pytest.param(MIN_SEED, MIN_SEED, id="MIN_SEED"),
        pytest.param(MAX_SEED, MAX_SEED, id="MAX_SEED"),
        # Boolean seeds change in type (bool -> int)
        pytest.param(False, 0, id="False"),
        pytest.param(True, 1, id="True"),
        # Non-seeds of type int are mapped onto seeds
        pytest.param(MIN_SEED - 1, MAX_SEED, id="MIN_SEED - 1"),
        pytest.param(MAX_SEED + 1, MIN_SEED, id="MAX_SEED + 1"),
        pytest.param(MIN_SEED - 2, MAX_SEED - 1, id="MIN_SEED - 2"),
        pytest.param(MAX_SEED + 2, MIN_SEED + 1, id="MAX_SEED + 2"),
        pytest.param(MIN_SEED - 10, MAX_SEED - 9, id="MIN_SEED_- 10"),
        pytest.param(MAX_SEED + 10, MIN_SEED + 9, id="MAX_SEED + 10"),
    ),
)
def test_int_to_seed_int_arg_valid(int_, seed_expected):
    seed_suggested = _int_to_seed(int_)
    assert _check_seed(seed_suggested) == seed_suggested
    assert seed_suggested == seed_expected


@pytest.mark.parametrize(
    "int_",
    (
        pytest.param(0.0, id="0.0"),
        pytest.param("zero", id="zero"),
        pytest.param(None, id="None"),
        pytest.param(lambda: 0, id="lambda: 0"),
    ),
)
def test_int_to_seed_int_arg_invalid(int_):
    match = "`int_` must be of type `int`, got `.*`"
    with pytest.raises(TypeError, match=match):
        _int_to_seed(int_)


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
