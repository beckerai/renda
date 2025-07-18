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
import operator
from types import UnionType
from typing import Any


def check_scalar(
    value: Any,
    type_: type | UnionType | tuple[Any, ...],
    **kwargs: Any,
) -> Any:
    # -----------
    # Type check
    # -----------
    try:
        value_is_invalid = not isinstance(value, type_)
    except TypeError:
        raise TypeError(
            f"`type_` must be a `type` or a `tuple` of types, checking if "
            f"`value={value}` is an instance of `type_={type_}` failed"
        )

    name = kwargs.pop("name", "value")
    if not isinstance(name, str):
        raise TypeError(f"`name` must be of type `str`, got `{name}`")

    if value_is_invalid:
        if isinstance(type_, tuple):
            type_str = ", ".join(t.__qualname__ for t in type_)
            type_str = f"({type_str})"
        else:
            type_str = type_.__qualname__

        raise TypeError(
            f"`{name}` must be of `type_={type_str}`, got "
            f"`{name}={value}` of type `{type(value).__qualname__}`"
        )

    # -----------------
    # Operators checks
    # -----------------
    supported_operators = {
        "ge": ">=",
        "gt": ">",
        "le": "<=",
        "lt": "<",
        "eq": "==",
        "ne": "!=",
        "in_": "in",
        "not_in": "not in",
    }
    if not kwargs.keys() <= supported_operators.keys():
        raise TypeError(
            f"unknown operator keyword(s) `{', `'.join(kwargs.keys())}`, "
            f"supported operator keywords are "
            f"`{', `'.join(supported_operators.keys())}`"
        )

    for k, v in kwargs.items():
        operator_symbol = supported_operators[k]
        if k == "in_":
            operator_ = lambda a, b: operator.contains(b, a)  # noqa: E731
        elif k == "not_in":
            operator_ = lambda a, b: not operator.contains(b, a)  # noqa: E731
        else:
            operator_ = getattr(operator, k)

        try:
            value_is_invalid = not operator_(value, v)
        except TypeError as original_type_error:
            if k in ("in_", "not_in"):
                raise TypeError(f"`{k}` must be iterable, got `{v}`")
            elif k in ("ge", "gt", "le", "lt"):
                raise TypeError(
                    f"`{k}` (`{operator_symbol}`) not supported between "
                    f"instances of `{type(value).__qualname__}` and "
                    f"`{type(v).__qualname__}`"
                )
            else:
                raise original_type_error

        if value_is_invalid:
            raise ValueError(
                f"`{name}={value}` does not satisfy `{operator_symbol} {v}`"
            )

    return value
