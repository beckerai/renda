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
import re
from types import UnionType
from typing import Any, Sequence

from renda.utils._messages import _BUG_MESSAGE


# The second argument of `isinstance()` must be of this type
__TYPE_TYPE = type | UnionType | tuple[Any, ...]


class _CheckError(Exception):
    pass


def _check_scalar(
    scalar: Any,
    type_: __TYPE_TYPE,
    name: str = "scalar",
    **operators: Any,
) -> Any:
    type_ = __check_type(type_)
    name = __check_name(name)
    operators = __check_operators(operators)

    check_error_message = ""

    # -----------
    # Type check
    # -----------
    type_condition_not_satisfied = not isinstance(scalar, type_)

    if type_condition_not_satisfied:
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
        op_symbol = __SUPPORTED_OPERATORS[op_key]

        if op_key == "in_":
            op = lambda a, b: operator.contains(b, a)  # noqa: E731
        elif op_key == "not_in":
            op = lambda a, b: not operator.contains(b, a)  # noqa: E731
        else:
            op = getattr(operator, op_key)

        try:
            op_condition_not_satisfied = not op(scalar, op_arg)
        except TypeError as e:
            if type_condition_not_satisfied:
                # We assume that this is a consequential error and that passing
                # a scalar of the correct type will fix this
                continue
            else:
                # We are using _check_scalar incorrectly

                # Known incorrect uses
                if op_key in ("ge", "gt", "le", "lt"):
                    p = "^'.*' not supported between instances of '.*' and '.*'$"
                    if re.match(p, str(e)):
                        raise TypeError(
                            f"`{op_symbol}` (`{op_key}`) not supported between "
                            f"instances of `{type(scalar).__qualname__}` and "
                            f"`{type(op_arg).__qualname__}`{_BUG_MESSAGE}"
                        )
                elif op_key in ("in_", "not_in"):
                    p = "^argument of type .* is not iterable$"
                    if re.match(p, str(e)):
                        raise TypeError(
                            f"`{op_key}` must be iterable, got `{op_arg}`"
                            f"{_BUG_MESSAGE}`"
                        )

                # Unknown incorrect uses
                raise TypeError(f"{e}{_BUG_MESSAGE}")  # pragma: no cover

        if op_condition_not_satisfied:
            check_error_message = (
                f"{check_error_message}\n"
                f"  - `{name} {op_symbol} {op_arg}` not satisfied, got `{scalar}`"
            )

    if len(check_error_message) > 0:
        raise _CheckError(check_error_message)

    return scalar


def _check_sequence(
    sequence: Sequence[Any],
    type_: __TYPE_TYPE,
    name: str = "sequence",
    length: int | None = None,
    **operators: Any,
) -> Sequence[Any]:
    type_ = __check_type(type_)
    name = __check_name(name)
    length = __check_length(length)
    operators = __check_operators(operators)

    if not isinstance(sequence, Sequence):
        raise _CheckError(f"`{name}` must be a sequence, got `{sequence}`")

    check_error_message = ""

    if length is not None and len(sequence) != length:
        check_error_message = (
            f"{check_error_message}\n  - `{name}` must have length `{length}`, "
            f"but `len({name}) = {len(sequence)}`"
        )

    for index, scalar in enumerate(sequence):
        try:
            _check_scalar(scalar, type_, f"{name}[{index}]", **operators)
        except _CheckError as e:
            check_error_message = f"{check_error_message}{e}"

    if len(check_error_message) > 0:
        raise _CheckError(check_error_message)

    return sequence


def _check_scalar_or_sequence(
    scalar_or_sequence: Any | Sequence[Any],
    type_: __TYPE_TYPE,
    name: str = "scalar_or_sequence",
    length: int | None = None,
    **operators: Any,
) -> Any | Sequence[Any]:
    type_ = __check_type(type_)
    name = __check_name(name)
    length = __check_length(length)
    operators = __check_operators(operators)

    if isinstance(scalar_or_sequence, Sequence):
        _check_sequence(scalar_or_sequence, type_, name, length, **operators)
    else:
        _check_scalar(scalar_or_sequence, type_, name, **operators)

    return scalar_or_sequence


def __check_type(type_: __TYPE_TYPE) -> __TYPE_TYPE:
    try:
        isinstance(object(), type_)
    except TypeError:
        raise TypeError(
            f"`type_` must be a type, a tuple of types, or a union, got `{type_}`"
        )

    return type_


def __check_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"`name` must be a `str`, got `{name}`")

    return name


def __check_length(length: int | None) -> int | None:
    if not ((isinstance(length, int) and length > 0) or length is None):
        raise TypeError(f"`length` must be a positive `int` or `None`, got `{length}`")

    return length


__SUPPORTED_OPERATORS = {
    "ge": ">=",
    "gt": ">",
    "le": "<=",
    "lt": "<",
    "eq": "==",
    "ne": "!=",
    "in_": "in",
    "not_in": "not in",
}


def __check_operators(operators: dict[str, Any]) -> dict[str, Any]:
    unsupported_operators = []
    for op_key in operators.keys():
        if op_key not in __SUPPORTED_OPERATORS.keys():
            unsupported_operators.append(op_key)

    if len(unsupported_operators) > 0:
        raise TypeError(
            f"unsupported operator keyword(s) "
            f"`{'`, `'.join(unsupported_operators)}`, "
            f"supported operator keywords are "
            f"`{'`, `'.join(__SUPPORTED_OPERATORS.keys())}`"
        )

    return operators
