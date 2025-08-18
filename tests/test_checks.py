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
from collections import OrderedDict
from types import NoneType

import pytest

from renda._checks import (
    _check_scalar,
    _check_scalar_or_sequence,
    _check_seed,
    _check_sequence,
)
from renda._exceptions import _CheckError
from renda.seeding import MAX_SEED, MIN_SEED


# ==============================================================================
# Test special checks
# ==============================================================================
@pytest.mark.parametrize(
    "seed",
    (
        pytest.param(1, id="1"),
        pytest.param(MIN_SEED, id="MIN_SEED"),
        pytest.param(MAX_SEED, id="MAX_SEED"),
        pytest.param(None, id="None"),
        pytest.param(False, id="False"),
        pytest.param(True, id="True"),
    ),
)
def test_check_seed_satisfied(seed):
    assert _check_seed(seed) == seed


@pytest.mark.parametrize(
    "value",
    (
        pytest.param(0.0, id="0.0"),
        pytest.param(MIN_SEED - 1, id="MIN_SEED - 1"),
        pytest.param(MAX_SEED + 1, id="MAX_SEED + 1"),
        pytest.param("zero", id="zero"),
        pytest.param(lambda: 0, id="lambda: 0"),
    ),
)
def test_check_seed_not_satisfied(value):
    with pytest.raises(_CheckError):
        _check_seed(value)


# ==============================================================================
# Test general checks
# ==============================================================================
@pytest.mark.parametrize(
    ("scalar", "type_"),
    (
        pytest.param(True, bool, id="bool"),
        pytest.param(1, int, id="int"),
        pytest.param(1.0, float, id="float"),
        pytest.param(None, NoneType, id="None (NoneType)"),
        pytest.param(None, (int, NoneType), id="None ((int, NoneType))"),
        pytest.param(None, int | None, id="None (int | None)"),
    ),
)
def test_check_scalar_type_condition_satisfied(scalar, type_):
    assert _check_scalar(scalar, type_) == scalar


@pytest.mark.parametrize(
    ("scalar", "type_"),
    (
        pytest.param(True, float, id="bool vs. float"),
        pytest.param(1, bool | float, id="int vs. bool | float"),
        pytest.param(1.0, (bool, int), id="float vs. (bool, int)"),
        pytest.param(1.0, NoneType, id="float vs. NoneType"),
        pytest.param(1.0, (int, NoneType), id="float vs. (int, NoneType)"),
        pytest.param(1.0, int | None, id="float vs. int | None"),
    ),
)
def test_check_scalar_type_condition_not_satisfied(scalar, type_):
    match = "`scalar` must be of type `.*`, got `.*` of type `.*`"
    with pytest.raises(_CheckError, match=match):
        _check_scalar(scalar, type_)


@pytest.mark.parametrize(
    "type_",
    (
        pytest.param(0, id="0"),
        pytest.param("zero", id="zero"),
        pytest.param(None, id="None"),
        pytest.param(lambda: 0, id="lambda: 0"),
    ),
)
def test_check_scalar_type_arg_invalid(type_):
    match = "^`type_` must be a type, a tuple of types, or a union, got `.*`$"
    with pytest.raises(TypeError, match=match):
        _check_scalar(0, type_)


def test_check_scalar_name_arg_valid():
    match = "`a_string` must be of type `.*`, got `.*` of type `.*`"
    with pytest.raises(_CheckError, match=match):
        _check_scalar(0, float, name="a_string")


@pytest.mark.parametrize(
    "name",
    (
        pytest.param(0, id="0"),
        pytest.param(0.0, id="0.0"),
        pytest.param(None, id="None"),
        pytest.param(lambda: 0, id="lambda: 0"),
    ),
)
def test_check_scalar_name_arg_invalid(name):
    match = "^`name` must be a `str`, got `.*`$"
    with pytest.raises(TypeError, match=match):
        _check_scalar(0, int, name=name)


def __get_pytest_params(*args):
    pytest_params = []
    for group_params, subgroup in args:
        for subgroup_params in subgroup:
            op_key, op_symbol = group_params
            scalar, type_, op_arg = subgroup_params

            id_ = f"{op_key}: {scalar} {op_symbol} {op_arg}"
            if not isinstance(scalar, type_):
                id_ = f"{id_} (type not satisfied)"

            pytest_params.append(
                pytest.param(
                    op_key,
                    op_symbol,
                    scalar,
                    type_,
                    op_arg,
                    id=id_,
                ),
            )

    return pytest_params


@pytest.mark.parametrize(
    ("op_key", "op_symbol", "scalar", "type_", "op_arg"),
    __get_pytest_params(
        (
            ("ge", ">="),  # (op_key, op_symbol)
            (
                (False, bool, False),  # (scalar, type_, op_arg)
                (0, int, 0),
                (0.0, float, 0.0),
                # `op_arg` doesn't have to match type_
                (0, int, 0.0),
                (0.0, float, 0),
            ),
        ),
        (
            ("gt", ">"),
            (
                (True, bool, False),
                (1, int, 0),
                (1.0, float, 0.0),
                # `op_arg` doesn't have to match type_
                (1, int, 0.0),
                (1.0, float, 0),
            ),
        ),
        (
            ("le", "<="),
            (
                (True, bool, True),
                (1, int, 1),
                (1.0, float, 1.0),
                # `op_arg` doesn't have to match `type_`
                (1, int, 1.0),
                (1.0, float, 1),
            ),
        ),
        (
            ("lt", "<"),
            (
                (False, bool, True),
                (0, int, 1),
                (0.0, float, 1.0),
                # `op_arg` doesn't have to match `type_`
                (0, int, 1.0),
                (0.0, float, 1),
            ),
        ),
        (
            ("eq", "=="),
            (
                (False, bool, False),
                (0, int, 0),
                (0.0, float, 0.0),
                # `op_arg` doesn't have to match `type_`
                (0, int, 0.0),
                (0.0, float, 0),
            ),
        ),
        (
            ("ne", "!="),
            (
                (False, bool, True),
                (0, int, 1),
                (0.0, float, 1.0),
                # `op_arg` doesn't have to match `type_`
                (0, int, 1.0),
                (0.0, float, 1),
            ),
        ),
        (
            ("in_", "in"),
            (
                (False, bool, (False, True)),
                (0, int, (0, 1)),
                (0.0, float, (0.0, 1.0)),
                # `op_arg` doesn't have to match `type_`
                (0, int, (0.0, 1.0)),
                (0.0, float, (0, 1)),
            ),
        ),
        (
            ("not_in", "not in"),
            (
                (False, bool, (True,)),
                (-1, int, (0, 1)),
                (-1.0, float, (0.0, 1.0)),
                # `op_arg` doesn't have to match `type_`
                (-1, int, (0.0, 1.0)),
                (-1.0, float, (0, 1)),
            ),
        ),
    ),
)
def test_check_scalar_operator_condition_satisfied(
    op_key, op_symbol, scalar, type_, op_arg
):
    assert _check_scalar(scalar, type_, **{op_key: op_arg}) == scalar


@pytest.mark.parametrize(
    ("op_key", "op_symbol", "scalar", "type_", "op_arg"),
    __get_pytest_params(
        (
            ("ge", ">="),  # (op_key, op_symbol)
            (
                (False, bool, True),  # (scalar, type_, op_arg)
                (0, int, 1),
                (0.0, float, 1.0),
                # `scalar` doesn't have to match `type_`, but the error will be listed
                (0.0, int, 1),
                (0, float, 1.0),
            ),
        ),
        (
            ("gt", ">"),
            (
                (False, bool, False),
                (0, int, 0),
                (0.0, float, 0.0),
                # `scalar` doesn't have to match `type_`, but the error will be listed
                (0.0, int, 0),
                (0, float, 0.0),
            ),
        ),
        (
            ("le", "<="),
            (
                (True, bool, False),
                (1, int, 0),
                (1.0, float, 0.0),
                # `scalar` doesn't have to match `type_`, but the error will be listed
                (1.0, int, 0),
                (1, float, 0.0),
            ),
        ),
        (
            ("lt", "<"),
            (
                (True, bool, True),
                (1, int, 1),
                (1.0, float, 1.0),
                # `scalar` doesn't have to match `type_`, but the error will be listed
                (1.0, int, 1),
                (1, float, 1.0),
            ),
        ),
        (
            ("eq", "=="),
            (
                (False, bool, True),
                (0, int, 1),
                (0.0, float, 1.0),
                # `scalar` doesn't have to match `type_`, but the error will be listed
                (0.0, int, 1),
                (0, float, 1.0),
            ),
        ),
        (
            ("ne", "!="),
            (
                (False, bool, False),
                (0, int, 0),
                (0.0, float, 0.0),
                # `scalar` doesn't have to match `type_`, but the error will be listed
                (0.0, int, 0),
                (0, float, 0.0),
            ),
        ),
        (
            ("in_", "in"),
            (
                (False, bool, (True,)),
                (-1, int, (0, 1)),
                (-1.0, float, (0.0, 1.0)),
                # `scalar` doesn't have to match `type_`, but the error will be listed
                (-1.0, int, (0, 1)),
                (-1, float, (0.0, 1.0)),
            ),
        ),
        (
            ("not_in", "not in"),
            (
                (False, bool, (False, True)),
                (0, int, (0, 1)),
                (0.0, float, (0.0, 1.0)),
                # `scalar` doesn't have to match `type_`, but the error will be listed
                (0.0, int, (0, 1)),
                (0, float, (0.0, 1.0)),
            ),
        ),
    ),
)
def test_check_scalar_operator_condition_not_satisfied(
    op_key, op_symbol, scalar, type_, op_arg
):
    if isinstance(scalar, type_):
        match = "^\n  - `scalar .{1,6} .*` not satisfied, got `.*`$"
    else:
        match = (
            "^\n  - `scalar` must be of type `.*`, got `.*` of type `.*`\n"
            "  - `scalar .{1,6} .*` not satisfied, got `.*`$"
        )
    match = match.replace("(", "\\(")
    match = match.replace(")", "\\)")

    with pytest.raises(_CheckError, match=match):
        _check_scalar(scalar, type_, **{op_key: op_arg})


@pytest.mark.parametrize(
    ("op_key", "op_symbol", "scalar", "type_", "op_arg"),
    (
        pytest.param("ge", ">=", 0, int, 1j, id="ge"),
        pytest.param("gt", ">", 0, int, 1j, id="gt"),
        pytest.param("le", "<=", 0, int, 1j, id="le"),
        pytest.param("lt", "<", 0, int, 1j, id="lt"),
        pytest.param("in_", "in", 0, int, 0, id="in_"),
        pytest.param("not_in", "not in", 0, int, 0, id="not_in"),
    ),
)
def test_check_scalar_operator_arg_invalid(op_key, op_symbol, scalar, type_, op_arg):
    if op_key in ("ge", "gt", "le", "lt"):
        match = (
            "^`.{1,6}` \\(`.{2,6}`\\) not supported between instances of `.*` and `.*`"
        )
    elif op_key in ("in_", "not_in"):
        match = "^`.{2,6}` must be iterable, got `.*`"
    else:
        match = ""

    with pytest.raises(TypeError, match=match):
        _check_scalar(scalar, type_, **{op_key: op_arg})


def test_check_scalar_operator_unsupported():
    match = "^unsupported operator keyword\\(s\\) `foo`, `bar`"
    operators = OrderedDict([("foo", 0), ("ge", 1), ("bar", 2), ("lt", 3)])

    with pytest.raises(TypeError, match=match):
        _check_scalar(0, int, **operators)


@pytest.mark.parametrize(
    ("op_key", "op_symbol", "scalar", "type_", "op_arg"),
    __get_pytest_params(
        (
            ("ge", ">="),  # (op_key, op_symbol)
            (
                (1j, complex, 0),  # (scalar, type_, op_arg)
                (1j, int, 0),
            ),
        ),
        (
            ("gt", ">"),
            (
                (1j, complex, 0),
                (1j, int, 0),
            ),
        ),
        (
            ("le", "<="),
            (
                (1j, complex, 0),
                (1j, int, 0),
            ),
        ),
        (
            ("lt", "<"),
            (
                (1j, complex, 0),
                (1j, int, 0),
            ),
        ),
    ),
)
def test_check_scalar_operator_not_applicable_to_scalar(
    op_key,
    op_symbol,
    scalar,
    type_,
    op_arg,
):
    if isinstance(scalar, type_):
        error = TypeError
        if op_key in ("ge", "gt", "le", "lt"):
            match = (
                "^`.{1,6}` \\(`.{2,6}`\\) not supported between instances "
                "of `.*` and `.*`"
            )
        elif op_key in ("in_", "not_in"):
            match = "^`.{2,6}` must be iterable, got `.*`"
        else:
            match = ""
    else:
        error = _CheckError
        match = "`scalar` must be of type `.*`, got `.*` of type `.*`"

    with pytest.raises(error, match=match):
        _check_scalar(scalar, type_, **{op_key: op_arg})


@pytest.mark.parametrize(
    "sequence",
    (
        pytest.param(False, id="bool"),
        pytest.param(0, id="int"),
        pytest.param(0.0, id="float"),
        pytest.param(None, id="NoneType"),
        pytest.param(lambda: 0, id="function"),
    ),
)
def test_check_sequence_sequence_arg_invalid(sequence):
    match = "`sequence` must be a sequence with elements of type `.*`, got `.*`"
    with pytest.raises(_CheckError, match=match):
        _check_sequence(sequence, int)


@pytest.mark.parametrize(
    ("sequence", "type_"),
    (
        pytest.param((False, True), bool, id="bool"),
        pytest.param([1, 2, 3], int, id="int"),
        pytest.param([1.0, 2.0, 3.0], float, id="float"),
        pytest.param([None, None], NoneType, id="None (NoneType)"),
        pytest.param([None, 1], (int, NoneType), id="None (int, NoneType)"),
        pytest.param([None, 1], int | None, id="None (int | None)"),
    ),
)
def test_check_sequence_type_condition_satisfied(sequence, type_):
    assert _check_sequence(sequence, type_) == sequence


@pytest.mark.parametrize(
    ("sequence", "type_"),
    (
        pytest.param((0, True), bool, id="int vs. bool"),
        pytest.param([1.0, 2, 3], int, id="float vs. int"),
        pytest.param([1, 2.0, 3.0], float, id="int vs. float"),
        pytest.param([1.0, 2.0], NoneType, id="float vs. NoneType"),
        pytest.param([1.0, 2.0], (int, NoneType), id="float vs. (int, NoneType)"),
        pytest.param([1.0, 2.0], int | None, id="float vs. int | None"),
    ),
)
def test_check_sequence_type_condition_not_satisfied(sequence, type_):
    match = "`sequence` must be a sequence with elements of type `.*`, got `.*`"
    with pytest.raises(_CheckError, match=match):
        _check_sequence(sequence, type_)


@pytest.mark.parametrize(
    "type_",
    (
        pytest.param(0, id="0"),
        pytest.param("zero", id="zero"),
        pytest.param(None, id="None"),
        pytest.param(lambda: 0, id="lambda: 0"),
    ),
)
def test_check_sequence_type_arg_invalid(type_):
    match = "^`type_` must be a type, a tuple of types, or a union, got `.*`$"
    with pytest.raises(TypeError, match=match):
        _check_sequence((1, 2, 3), type_)


def test_check_sequence_name_arg_valid():
    match = "`a_string` must be a sequence with elements of type `.*`, got `.*`"
    with pytest.raises(_CheckError, match=match):
        _check_sequence((1, 2, 3), float, name="a_string")


@pytest.mark.parametrize(
    "name",
    (
        pytest.param(0, id="0"),
        pytest.param(0.0, id="0.0"),
        pytest.param(None, id="None"),
        pytest.param(lambda: 0, id="lambda: 0"),
    ),
)
def test_check_sequence_name_arg_invalid(name):
    match = "^`name` must be a `str`, got `.*`$"
    with pytest.raises(TypeError, match=match):
        _check_sequence((1, 2, 3), int, name=name)


@pytest.mark.parametrize(
    ("sequence", "length"),
    (
        pytest.param((1, 2), 2, id="(1, 2) with length=3"),
        pytest.param((1, 2), None, id="(1, 2) with length=None"),
        pytest.param((1, 2, 3), 3, id="(1, 2, 3) with length=4"),
        pytest.param((1, 2, 3), None, id="(1, 2, 3) with length=None"),
    ),
)
def test_check_sequence_length_condition_satisfied(sequence, length):
    assert _check_sequence(sequence, int, length=length) == sequence


def test_check_sequence_length_condition_not_satisfied():
    match = "`sequence` must have length `4`, but `len\\(sequence\\) = 3`"
    with pytest.raises(_CheckError, match=match):
        _check_sequence((1, 2, 3), int, length=4)


@pytest.mark.parametrize(
    "length",
    (
        pytest.param(0, id="0"),
        pytest.param(-1, id="-1"),
        pytest.param("zero", id="zero"),
        pytest.param(lambda: 0, id="lambda: 0"),
    ),
)
def test_check_sequence_length_arg_invalid(length):
    match = "^`length` must be a positive `int` or `None`, got `.*`$"
    with pytest.raises(TypeError, match=match):
        _check_sequence((1, 2, 3), int, length=length)


def test_check_sequence_operator_unsupported():
    match = "^unsupported operator keyword\\(s\\) `foo`, `bar`"
    operators = OrderedDict([("foo", 0), ("ge", 1), ("bar", 2), ("lt", 3)])

    with pytest.raises(TypeError, match=match):
        _check_sequence((1, 2, 3), int, **operators)


@pytest.mark.parametrize(
    ("sequence", "type_", "length", "operators", "match"),
    (
        pytest.param(
            (False, True, 2, 3.0),
            bool,
            4,
            {"le": True},
            (
                "^\n  - `s` must be a sequence with elements of type `.*`, got .*\n"
                "  - `s\\[2\\]` must be of type `bool`, got .*\n"
                "  - `s\\[2\\] <= True` not satisfied, got .*\n"
                "  - `s\\[3\\]` must be of type `bool`, got .*\n"
                "  - `s\\[3] <= True` not satisfied, got"
            ),
            id="scenario0",
        ),
        pytest.param(
            (False, 1, 2, 3.0),
            int,
            4,
            {"ge": 1},
            (
                "^\n  - `s` must be a sequence with elements of type `.*`, got .*\n"
                "  - `s\\[0\\] >= 1` not satisfied, got .*\n"
                "  - `s\\[3\\]` must be of type `int`, got"
            ),
            id="scenario1",
        ),
        pytest.param(
            (0.0, True, 2.0, 3.0),
            float,
            5,
            {"gt": 0.0, "lt": 3.0},
            (
                "^\n  - `s` must be a sequence with elements of type `.*`, got .*\n"
                "  - `s` must have length `5`, but .*\n"
                "  - `s\\[0\\] > 0.0` not satisfied, got .*\n"
                "  - `s\\[1\\]` must be of type `float`, got .*\n"
                "  - `s\\[3\\] < 3.0` not satisfied"
            ),
            id="scenario2",
        ),
        pytest.param(
            (0, 1j, 2j, 3j, 4j, 5),
            complex,
            5,
            {"not_in": (2j, 4j)},
            (
                "^\n  - `s` must be a sequence with elements of type `.*`, got .*\n"
                "  - `s` must have length `5`, but .*\n"
                "  - `s\\[0\\]` must be of type `complex`, got .*\n"
                "  - `s\\[2\\] not in \\(2j, 4j\\)` not satisfied, got .*\n"
                "  - `s\\[4\\] not in \\(2j, 4j\\)` not satisfied, got .*\n"
                "  - `s\\[5\\]` must be of type `complex`, got"
            ),
            id="scenario3",
        ),
    ),
)
def test_check_sequence_with_multiple_conditions(
    sequence,
    type_,
    length,
    operators,
    match,
):
    with pytest.raises(_CheckError, match=match):
        _check_sequence(sequence, type_, name="s", length=length, **operators)


@pytest.mark.parametrize(
    "scalar_or_sequence",
    (
        pytest.param(None, id="NoneType"),
        pytest.param(lambda: 0, id="function"),
    ),
)
def test_check_scalar_or_sequence_arg_invalid(scalar_or_sequence):
    match = "`scalar_or_sequence` must be of type `.*`, or a sequence"
    with pytest.raises(_CheckError, match=match):
        _check_scalar_or_sequence(scalar_or_sequence, int)


@pytest.mark.parametrize(
    ("scalar_or_sequence", "type_"),
    (
        pytest.param(False, bool, id="bool"),
        pytest.param(0, int, id="int"),
        pytest.param(0.0, float, id="float"),
        pytest.param((False, True), bool, id="sequence of bool"),
        pytest.param([1, 2, 3], int, id="sequence of int"),
        pytest.param([1.0, 2.0, 3.0], float, id="sequence of float"),
        pytest.param(None, NoneType, id="None (NoneType)"),
        pytest.param(None, (int, NoneType), id="None ((int, NoneType))"),
        pytest.param(None, int | None, id="None (int | None)"),
        pytest.param([None, None], NoneType, id="None (NoneType)"),
        pytest.param([None, 1], (int, NoneType), id="None (int, NoneType)"),
        pytest.param([None, 1], int | None, id="None (int | None)"),
    ),
)
def test_check_scalar_or_sequence_type_condition_satisfied(scalar_or_sequence, type_):
    assert _check_scalar_or_sequence(scalar_or_sequence, type_) == scalar_or_sequence


@pytest.mark.parametrize(
    ("scalar_or_sequence", "type_"),
    (
        pytest.param(0, bool, id="bool"),
        pytest.param(0.0, int, id="int"),
        pytest.param(0, float, id="float"),
        pytest.param((0, True), bool, id="sequence of bool"),
        pytest.param([1.0, 2, 3], int, id="sequence of int"),
        pytest.param([1, 2.0, 3.0], float, id="sequence of float"),
        pytest.param(1.0, NoneType, id="float vs. NoneType"),
        pytest.param(1.0, (int, NoneType), id="float vs. (int, NoneType)"),
        pytest.param(1.0, int | None, id="float vs. int | None"),
        pytest.param([1.0, 2.0], NoneType, id="float vs. NoneType"),
        pytest.param([1.0, 2.0], (int, NoneType), id="float vs. (int, NoneType)"),
        pytest.param([1.0, 2.0], int | None, id="float vs. int | None"),
    ),
)
def test_check_scalar_or_sequence_type_condition_not_satisfied(
    scalar_or_sequence, type_
):
    match = "`scalar_or_sequence` must be of type `.*`, or a sequence"
    with pytest.raises(_CheckError, match=match):
        _check_scalar_or_sequence(scalar_or_sequence, type_)


@pytest.mark.parametrize(
    "type_",
    (
        pytest.param(0, id="0"),
        pytest.param("zero", id="zero"),
        pytest.param(None, id="None"),
        pytest.param(lambda: 0, id="lambda: 0"),
    ),
)
def test_check_scalar_or_sequence_type_arg_invalid(type_):
    match = "^`type_` must be a type, a tuple of types, or a union, got `.*`$"
    with pytest.raises(TypeError, match=match):
        _check_scalar_or_sequence(0, type_)
    with pytest.raises(TypeError, match=match):
        _check_scalar_or_sequence((1, 2, 3), type_)


def test_check_scalar_or_sequence_name_arg_valid():
    match = "`a_string` must be of type `.*`, or a sequence"
    with pytest.raises(_CheckError, match=match):
        _check_scalar_or_sequence((1, 2, 3), float, name="a_string")


@pytest.mark.parametrize(
    "name",
    (
        pytest.param(0, id="0"),
        pytest.param(0.0, id="0.0"),
        pytest.param(None, id="None"),
        pytest.param(lambda: 0, id="lambda: 0"),
    ),
)
def test_check_scalar_or_sequence_name_arg_invalid(name):
    match = "^`name` must be a `str`, got `.*`$"
    with pytest.raises(TypeError, match=match):
        _check_scalar_or_sequence(0, int, name=name)
    with pytest.raises(TypeError, match=match):
        _check_scalar_or_sequence((1, 2, 3), int, name=name)


@pytest.mark.parametrize(
    ("scalar_or_sequence", "length"),
    (
        pytest.param((1, 2), 2, id="(1, 2) with length=3"),
        pytest.param((1, 2), None, id="(1, 2) with length=None"),
        pytest.param((1, 2, 3), 3, id="(1, 2, 3) with length=4"),
        pytest.param((1, 2, 3), None, id="(1, 2, 3) with length=None"),
    ),
)
def test_scalar_or_sequence_length_condition_satisfied(scalar_or_sequence, length):
    scalar_or_sequence_ = _check_scalar_or_sequence(
        scalar_or_sequence=scalar_or_sequence,
        type_=int,
        length=length,
    )
    assert scalar_or_sequence_ == scalar_or_sequence


def test_scalar_or_sequence_length_condition_not_satisfied():
    match = "`sos` must have length `4`, but `len\\(sos\\) = 3`"
    with pytest.raises(_CheckError, match=match):
        _check_scalar_or_sequence((1, 2, 3), int, name="sos", length=4)


@pytest.mark.parametrize(
    "length",
    (
        pytest.param(0, id="0"),
        pytest.param(-1, id="-1"),
        pytest.param("forty-two", id="forty-two"),
        pytest.param(lambda: 0, id="lambda: 0"),
    ),
)
def test_check_scalar_or_sequence_length_arg_invalid(length):
    match = "^`length` must be a positive `int` or `None`, got `.*`$"
    with pytest.raises(TypeError, match=match):
        _check_scalar_or_sequence((1, 2, 3), int, length=length)


def test_check_scalar_or_sequence_operator_unsupported():
    match = "^unsupported operator keyword\\(s\\) `foo`, `bar`"
    operators = OrderedDict([("foo", 0), ("ge", 1), ("bar", 2), ("lt", 3)])

    with pytest.raises(TypeError, match=match):
        _check_scalar_or_sequence(0, int, **operators)

    with pytest.raises(TypeError, match=match):
        _check_scalar_or_sequence((1, 2, 3), int, **operators)


@pytest.mark.parametrize(
    ("scalar_or_sequence", "type_", "length", "operators", "match"),
    (
        pytest.param(
            (False, True, 2, 3.0),
            bool,
            4,
            {"le": True},
            (
                "^\n  - `sos` must be of type `.*`, or a sequence with "
                "elements of type `.*`, got .*\n"
                "  - `sos\\[2\\]` must be of type `bool`, got .*\n"
                "  - `sos\\[2\\] <= True` not satisfied, got .*\n"
                "  - `sos\\[3\\]` must be of type `bool`, got .*\n"
                "  - `sos\\[3] <= True` not satisfied, got"
            ),
            id="scenario0",
        ),
        pytest.param(
            (False, 1, 2, 3.0),
            int,
            4,
            {"ge": 1},
            (
                "^\n  - `sos` must be of type `.*`, or a sequence with "
                "elements of type `.*`, got .*\n"
                "  - `sos\\[0\\] >= 1` not satisfied, got .*\n"
                "  - `sos\\[3\\]` must be of type `int`, got"
            ),
            id="scenario1",
        ),
        pytest.param(
            (0.0, True, 2.0, 3.0),
            float,
            5,
            {"gt": 0.0, "lt": 3.0},
            (
                "^\n  - `sos` must be of type `.*`, or a sequence with "
                "elements of type `.*`, got .*\n"
                "  - `sos` must have length `5`, but .*\n"
                "  - `sos\\[0\\] > 0.0` not satisfied, got .*\n"
                "  - `sos\\[1\\]` must be of type `float`, got .*\n"
                "  - `sos\\[3\\] < 3.0` not satisfied"
            ),
            id="scenario2",
        ),
        pytest.param(
            0,
            float,
            None,
            {"gt": 1.0, "in_": (1, 2, 3)},
            (
                "^\n  - `sos` must be of type `.*`, or a sequence with "
                "elements of type `.*`, got .*\n"
                "  - `sos > 1.0` not satisfied, got .*\n"
                "  - `sos in \\(1, 2, 3\\)` not satisfied, got"
            ),
            id="scenario3",
        ),
    ),
)
def test_check_scalar_or_sequence_with_multiple_conditions(
    scalar_or_sequence,
    type_,
    length,
    operators,
    match,
):
    with pytest.raises(_CheckError, match=match):
        _check_scalar_or_sequence(
            scalar_or_sequence,
            type_,
            name="sos",
            length=length,
            **operators,
        )
