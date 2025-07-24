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
from typing import Any, Sequence


# The second argument of `isinstance()` must be of this type
_TYPE_TYPE = type | UnionType | tuple[Any, ...]


class CheckError(Exception):
    pass


def check_scalar(
    scalar: Any,
    type_: _TYPE_TYPE,
    name: str = "scalar",
    **operators: Any,
) -> Any:
    type_ = _check_type(type_)
    name = _check_name(name)
    operators = _check_operators(operators)

    check_error_message = ""

    # -----------
    # Type check
    # -----------
    if not isinstance(scalar, type_):
        if isinstance(type_, tuple):
            type_str = ", ".join(f"`{t.__qualname__}`" for t in type_[:-1])
            type_str = f"{type_str} or `{type_[-1].__qualname__}`"
        elif isinstance(type_, UnionType):
            type_str = f"`{type_}`"
        else:
            type_str = f"`{type_.__qualname__}`"

        check_error_message = (
            f"{check_error_message}\n"
            f"  - `{name}` must be of type {type_str}, got "
            f"`{scalar}` of type `{type(scalar).__qualname__}`"
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
            condition_not_satisfied = not op(scalar, op_arg)
        except TypeError as e:
            if op_key in ("in_", "not_in"):
                raise TypeError(f"`{op_key}` must be iterable, got `{op_arg}`")
            elif op_key in ("ge", "gt", "le", "lt"):
                raise TypeError(
                    f"`{op_symbol}` (`{op_key}`) not supported between "
                    f"instances of `{type(scalar).__qualname__}` and "
                    f"`{type(op_arg).__qualname__}`"
                )
            else:
                raise e  # pragma: no cover

        if condition_not_satisfied:
            check_error_message = (
                f"{check_error_message}\n"
                f"  - `{name} {op_symbol} {op_arg}` not satisfied, got `{scalar}`"
            )

    if len(check_error_message) > 0:
        raise CheckError(check_error_message)

    return scalar


def check_sequence(
    sequence: Sequence[Any],
    type_: _TYPE_TYPE,
    name: str = "sequence",
    length: int | None = None,
    **operators: Any,
) -> Sequence[Any]:
    type_ = _check_type(type_)
    name = _check_name(name)
    length = _check_length(length)
    operators = _check_operators(operators)

    if not isinstance(sequence, Sequence):
        raise CheckError(f"`{name}` must be a sequence, got `{sequence}`")

    check_error_message = ""

    if length is not None and len(sequence) != length:
        check_error_message = (
            f"{check_error_message}\n  - `{name}` must have length `{length}`, "
            f"but `len({name}) = {len(sequence)}`)"
        )

    for index, scalar in enumerate(sequence):
        try:
            check_scalar(scalar, type_, f"{name}[{index}]", **operators)
        except CheckError as e:
            check_error_message = f"{check_error_message}{e}"

    if len(check_error_message) > 0:
        raise CheckError(check_error_message)

    return sequence


def check_scalar_or_sequence(
    scalar_or_sequence: Any | Sequence[Any],
    type_: _TYPE_TYPE,
    name: str = "scalar_or_sequence",
    length: int | None = None,
    **operators: Any,
) -> Any | Sequence[Any]:
    type_ = _check_type(type_)
    name = _check_name(name)
    length = _check_length(length)
    operators = _check_operators(operators)

    if isinstance(scalar_or_sequence, Sequence):
        check_sequence(scalar_or_sequence, type_, name, length, **operators)
    else:
        check_scalar(scalar_or_sequence, type_, name, **operators)

    return scalar_or_sequence


def _check_type(type_: _TYPE_TYPE) -> _TYPE_TYPE:
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


def _check_length(length: int | None) -> int | None:
    if not ((isinstance(length, int) and length > 0) or length is None):
        raise TypeError(f"`length` must be a positive `int` or `None`, got `{length}`")

    return length


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


def _check_operators(operators: dict[str, Any]) -> dict[str, Any]:
    unsupported_operators = []
    for op_key in operators.keys():
        if op_key not in _SUPPORTED_OPERATORS.keys():
            unsupported_operators.append(op_key)

    if len(unsupported_operators) > 0:
        raise TypeError(
            f"unsupported operator keyword(s) "
            f"`{'`, `'.join(unsupported_operators)}`, "
            f"supported operator keywords are "
            f"`{'`, `'.join(_SUPPORTED_OPERATORS.keys())}`"
        )

    return operators
