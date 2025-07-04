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
from typing import Any


MIN_SEED = 0
MAX_SEED = 4294967295  # 2^32 - 1 (uint32)


def is_seed(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and MIN_SEED <= value <= MAX_SEED
    )


def ensure_seed(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("`value` must be of type `int`")
    # This only works because MIN_SEED = 0 and MIN_SEED > 0
    # A more general solutions would be nice
    return value % (MAX_SEED + 1)
