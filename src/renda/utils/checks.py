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
from typing import Any, Optional, Sequence


def _check_type(
    type_: type | UnionType | tuple[Any, ...],
) -> type | UnionType | tuple[Any, ...]:
    try:
        isinstance(object(), type_)
    except TypeError:
        raise TypeError(
            f"`type_` must be a type, a tuple of types, or a union, got `{type_}`"
        )

    return type_


def _check_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"`name` must be a `str`, got `{name}`")

    return name


_SUPPORTED_OPERATORS = {
    "ge": ">=",
    "gt": ">",
    "le": "<=",
    "lt": "<",
    "eq": "==",
    "ne": "!=",
    "in_": "in",
    "not_in": "not in",
}


def _check_operators(operators: Any) -> dict[str:Any]:
    unsupported_operators = []
    for op_key in operators.keys():
        if op_key not in _SUPPORTED_OPERATORS:
            unsupported_operators.append(op_key)

    if len(unsupported_operators) > 0:
        raise TypeError(
            f"unsupported operator keyword(s) "
            f"`{'`, `'.join(unsupported_operators)}`, "
            f"supported operator keywords are "
            f"`{'`, `'.join(_SUPPORTED_OPERATORS.keys())}`"
        )

    return operators


class CheckError(Exception):
    pass


def check_scalar(
    value: Any,
    type_: type | UnionType | tuple[Any, ...],
    name: str = "value",
    **operators: Any,
) -> Any:
    type_ = _check_type(type_)
    name = _check_name(name)
    operators = _check_operators(operators)

    # -----------
    # Type check
    # -----------
    if not isinstance(value, type_):
        if isinstance(type_, tuple):
            type_str = ", ".join(f"`{t.__qualname__}`" for t in type_[:-1])
            type_str = f"{type_str} or `{type_[-1].__qualname__}`"
        else:
            type_str = f"`{type_.__qualname__}`"

        raise CheckError(
            f"`{name}` must be of type {type_str}, got "
            f"`{value}` of type `{type(value).__qualname__}`"
        )

    # ----------------
    # Operator checks
    # ----------------
    for op_key, op_arg in operators.items():
        op_symbol = _SUPPORTED_OPERATORS[op_key]
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


def check_sequence(
    sequence: Sequence[Any],
    type_: type | UnionType | tuple[Any, ...],
    name: str = "sequence",
    **operators: Any,
) -> Sequence[Any]:
    if not isinstance(sequence, Sequence):
        raise CheckError(f"`{name}` must be a sequence, got `{sequence}`")

    type_ = _check_type(type_)
    name = _check_name(name)
    operators = _check_operators(operators)

    error_message = ""
    for index, value in enumerate(sequence):
        try:
            check_scalar(value, type_, f"{name}[{index}]", **operators)
        except CheckError as e:
            error_message = f"{error_message}\n  - {e}"

    if len(error_message) > 0:
        raise CheckError(error_message)

    return sequence


def check_scalar_or_sequence(
    value_or_sequence: Any | Sequence[Any],
    type_: type | UnionType | tuple[Any, ...],
    name: Optional[str] = None,
    **operators: Any,
) -> Any | Sequence[Any]:
    if isinstance(value_or_sequence, Sequence):
        name = name or "sequence"
        check_function = check_sequence
    else:
        name = name or "value"
        check_function = check_scalar

    type_ = _check_type(type_)
    name = _check_name(name)
    operators = _check_operators(operators)

    check_function(value_or_sequence, type_, name, **operators)

    return value_or_sequence
