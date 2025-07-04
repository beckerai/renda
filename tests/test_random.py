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
import pytest

from renda.random import MAX_SEED, MIN_SEED, ensure_seed, is_seed


@pytest.mark.parametrize(
    "value",
    (
        pytest.param(42, id="int_42"),
        pytest.param(MIN_SEED, id="MIN_SEED"),
        pytest.param(MAX_SEED, id="MAX_SEED"),
    ),
)
def test_is_seed(value):
    assert is_seed(value)


@pytest.mark.parametrize(
    "value",
    (
        pytest.param(4.2, id="float"),
        pytest.param("forty-two", id="string"),
        pytest.param(True, id="boolean"),
        pytest.param(MIN_SEED - 1, id="int_lt_MIN_SEED"),
        pytest.param(MAX_SEED + 1, id="int_gt_MAX_SEED"),
        pytest.param(None, id="None"),
    ),
)
def test_is_seed_invalid_value(value):
    assert is_seed(value) is False


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
