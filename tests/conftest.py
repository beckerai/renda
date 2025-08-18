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

from renda.seeding import MAX_SEED, MIN_SEED


@pytest.fixture(
    scope="session",
    params=(
        0.0,
        MIN_SEED - 1,
        MAX_SEED + 1,
        "zero",
        lambda: 0,
    ),
    ids=(
        "0.0",
        "MIN_SEED - 1",
        "MAX_SEED + 1",
        "zero",
        "lambda: 0",
    ),
)
def non_seed(request):
    yield request.param
