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
from types import NoneType, UnionType
from typing import Any, Final, Sequence, get_args

from renda._exceptions import _CheckError
from renda._messages import _BUG_MESSAGE


# The second argument of `isinstance()` must be of this type
__TYPE_TYPE = type | UnionType | tuple[Any, ...]


def _check_scalar(
    scalar: Any,
    type_: __TYPE_TYPE,
    name: str = "scalar",
    **operators: Any,
) -> Any:
    type_ = __check_type_arg(type_)
    name = __check_name_arg(name)
    operators = __check_operators_arg(operators)

    if scalar is None and __type_includes_none(type_):
        return scalar

    check_error_message = ""

    # ---------------------
    # Check type condition
    # ---------------------
    type_condition_not_satisfied = not isinstance(scalar, type_)
    if type_condition_not_satisfied:
        type_str = __get_type_str(type_)
        check_error_message = (
            f"{check_error_message}\n"
            f"  - `{name}` must be of type {type_str}, got `{scalar}` of "
            f"type `{type(scalar).__qualname__}`"
        )

    # --------------------------
    # Check operator conditions
    # --------------------------
    for op_key, op_arg in operators.items():
        op_symbol = __SUPPORTED_OPERATORS[op_key]

        # Get operator
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
                # We assume that this is a consequential error
                # Passing a scalar of the correct type should fix this
                continue
            else:
                # We are using _check_scalar incorrectly
                # It is a bug if using our public API leads to this error

                # Known incorrect use :/
                if op_key in ("ge", "gt", "le", "lt"):
                    pattern = "^'.*' not supported between instances of '.*' and '.*'$"
                    if re.match(pattern, str(e)):
                        raise TypeError(
                            f"`{op_symbol}` (`{op_key}`) not supported between "
                            f"instances of `{type(scalar).__qualname__}` and "
                            f"`{type(op_arg).__qualname__}`{_BUG_MESSAGE}"
                        )
                elif op_key in ("in_", "not_in"):
                    pattern = "^argument of type '.*' is not iterable$"
                    if re.match(pattern, str(e)):
                        raise TypeError(
                            f"`{op_key}` must be iterable, got `{op_arg}`"
                            f"{_BUG_MESSAGE}"
                        )

                # Unknown incorrect use :(
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
    type_ = __check_type_arg(type_)
    name = __check_name_arg(name)
    length = __check_length_arg(length)
    operators = __check_operators_arg(operators)

    check_error_message = ""

    # ---------------------
    # Check type condition
    # ---------------------
    type_condition_not_satisfied = not (
        isinstance(sequence, Sequence)
        and all(isinstance(element, type_) for element in sequence)
    )
    if type_condition_not_satisfied:
        type_str = __get_type_str(type_)
        check_error_message = (
            f"{check_error_message}\n"
            f"  - `{name}` must be a sequence with elements of type "
            f"{type_str}, got `{sequence}`"
        )

    if isinstance(sequence, Sequence):
        # ---------------------------------------------
        # Check length condition / operator conditions
        # ---------------------------------------------
        try:
            __check_sequence_length_and_elements(
                sequence=sequence,
                type_=type_,
                name=name,
                length=length,
                **operators,
            )
        except _CheckError as e:
            check_error_message = f"{check_error_message}{e}"

    if len(check_error_message) > 0:
        raise _CheckError(check_error_message)

    return sequence


def __check_sequence_length_and_elements(
    sequence: Sequence[Any],
    type_: __TYPE_TYPE,
    name: str = "sequence",
    length: int | None = None,
    **operators: Any,
) -> Sequence[Any]:
    check_error_message = ""

    # -----------------------
    # Check length condition
    # -----------------------
    if length is not None and len(sequence) != length:
        check_error_message = (
            f"{check_error_message}\n  - `{name}` must have length "
            f"`{length}`, but `len({name}) = {len(sequence)}`"
        )

    # --------------------------
    # Check operator conditions
    # --------------------------
    for index, scalar in enumerate(sequence):
        try:
            _check_scalar(
                scalar=scalar,
                type_=type_,
                name=f"{name}[{index}]",  # Clearly indicate element-level check
                **operators,
            )
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
    type_ = __check_type_arg(type_)
    name = __check_name_arg(name)
    length = __check_length_arg(length)
    operators = __check_operators_arg(operators)

    check_error_message = ""

    # ---------------------
    # Check type condition
    # ---------------------
    type_condition_not_satisfied = not (
        isinstance(scalar_or_sequence, type_)
        or isinstance(scalar_or_sequence, Sequence)
        and all(isinstance(element, type_) for element in scalar_or_sequence)
    )
    if type_condition_not_satisfied:
        type_str = __get_type_str(type_)
        check_error_message = (
            f"{check_error_message}\n"
            f"  - `{name}` must be of type {type_str}, or a sequence with "
            f"elements of type {type_str}, got `{scalar_or_sequence}`"
        )

    if isinstance(scalar_or_sequence, Sequence):
        # ---------------------------------------------
        # Check length condition / operator conditions
        # ---------------------------------------------
        try:
            # We don't want to check the type again
            # So instead of _check_sequence, we call its helper function
            __check_sequence_length_and_elements(
                sequence=scalar_or_sequence,
                type_=type_,
                name=name,
                length=length,
                **operators,
            )
        except _CheckError as e:
            check_error_message = f"{check_error_message}{e}"
    else:
        # -------------
        # Check scalar
        # -------------
        try:
            # We don't want to check the type again
            # So instead of the desired type, we pass the actual type
            _check_scalar(
                scalar=scalar_or_sequence,
                type_=type(scalar_or_sequence),
                name=name,
                **operators,
            )
        except _CheckError as e:
            check_error_message = f"{check_error_message}{e}"

    if len(check_error_message) > 0:
        raise _CheckError(check_error_message)

    return scalar_or_sequence


def __check_type_arg(type_: __TYPE_TYPE) -> __TYPE_TYPE:
    try:
        isinstance(object(), type_)
    except TypeError:
        raise TypeError(
            f"`type_` must be a type, a tuple of types, or a union, got `{type_}`"
        )

    return type_


def __check_name_arg(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"`name` must be a `str`, got `{name}`")

    return name


def __check_length_arg(length: int | None) -> int | None:
    if not ((isinstance(length, int) and length > 0) or length is None):
        raise TypeError(f"`length` must be a positive `int` or `None`, got `{length}`")

    return length


__SUPPORTED_OPERATORS: Final = {
    "ge": ">=",
    "gt": ">",
    "le": "<=",
    "lt": "<",
    "eq": "==",
    "ne": "!=",
    "in_": "in",
    "not_in": "not in",
}


def __check_operators_arg(operators: dict[str, Any]) -> dict[str, Any]:
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


def __type_includes_none(type_: __TYPE_TYPE) -> bool:
    return (
        type_ is NoneType
        or (isinstance(type_, tuple) and NoneType in type_)
        or (isinstance(type_, UnionType) and NoneType in get_args(type_))
    )


def __get_type_str(type_: __TYPE_TYPE) -> str:
    if isinstance(type_, type):
        return f"`{type_.__qualname__}`"
    else:
        if not isinstance(type_, tuple):
            type_ = get_args(type_)

        return (
            f"{', '.join(f'`{t.__qualname__}`' for t in type_[:-1])} or "
            f"`{type_[-1].__qualname__}`"
        )
