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


class CheckError(Exception):
    pass


def check_scalar(
    value: Any,
    type_: type | UnionType | tuple[Any, ...],
    name: str = "value",
    **operators: Any,
) -> Any:
    # -----------
    # Type check
    # -----------
    try:
        check_failed = not isinstance(value, type_)
    except TypeError:
        raise TypeError(
            f"`type_` must be a type, a tuple of types, or a union, got `{type_}`"
        )

    if not isinstance(name, str):
        raise TypeError(f"`name` must be a `str`, got `{name}`")

    if check_failed:
        if isinstance(type_, tuple):
            type_str = "`, `".join(t.__qualname__ for t in type_[:-1])
            type_str = "`" + type_str + "` or `" + type_[-1].__qualname__ + "`"
        else:
            type_str = "`" + type_.__qualname__ + "`"

        raise CheckError(
            f"`{name}` must be of type {type_str}, got "
            f"`{value}` of type `{type(value).__qualname__}`"
        )

    # ----------------
    # Operator checks
    # ----------------
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

    unsupported_operators = []
    for op_key in operators.keys():
        if op_key not in supported_operators:
            unsupported_operators.append(op_key)

    if len(unsupported_operators) > 0:
        raise TypeError(
            f"unsupported operator keyword(s) "
            f"`{'`, `'.join(unsupported_operators)}`, "
            f"supported operator keywords are "
            f"`{'`, `'.join(supported_operators.keys())}`"
        )

    for op_key, op_arg in operators.items():
        op_symbol = supported_operators[op_key]
        if op_key == "in_":
            op = lambda a, b: operator.contains(b, a)  # noqa: E731
        elif op_key == "not_in":
            op = lambda a, b: not operator.contains(b, a)  # noqa: E731
        else:
            op = getattr(operator, op_key)

        try:
            check_failed = not op(value, op_arg)
        except TypeError as e:
            if op_key in ("in_", "not_in"):
                raise TypeError(f"`{op_key}` must be iterable, got `{op_arg}`")
            elif op_key in ("ge", "gt", "le", "lt"):
                raise TypeError(
                    f"`{op_symbol}` (`{op_key}`) not supported between "
                    f"instances of `{type(value).__qualname__}` and "
                    f"`{type(op_arg).__qualname__}`"
                )
            else:
                raise e  # pragma: no cover

        if check_failed:
            raise CheckError(
                f"`{name} {op_symbol} {op_arg}` not satisfied, got `{value}`"
            )

    return value
