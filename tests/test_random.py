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

from renda.random import MAX_SEED, MIN_SEED, is_seed


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
