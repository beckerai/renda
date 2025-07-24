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

import pytest

from renda.utils.checks import (
    CheckError,
    check_scalar,
    check_scalar_or_sequence,
    check_sequence,
)


@pytest.mark.parametrize(
    ("scalar", "type_"),
    (
        pytest.param(True, bool, id="bool"),
        pytest.param(1, int, id="int"),
        pytest.param(1.0, float, id="float"),
    ),
)
def test_check_scalar_type(scalar, type_):
    assert check_scalar(scalar, type_) == scalar


@pytest.mark.parametrize(
    ("scalar", "type_"),
    (
        pytest.param(True, float, id="bool vs. float"),
        pytest.param(1, bool | float, id="int vs. bool | float"),
        pytest.param(1.0, (bool, int), id="float vs. (bool, int)"),
    ),
)
def test_check_scalar_type_not_satisfied(scalar, type_):
    match = "`scalar` must be of type"
    with pytest.raises(CheckError, match=match):
        check_scalar(scalar, type_)


@pytest.mark.parametrize(
    "type_",
    (
        pytest.param(42, id="0"),
        pytest.param("forty-two", id="forty-two"),
        pytest.param(None, id="None"),
        pytest.param(lambda: 0, id="lambda: 0"),
    ),
)
def test_check_scalar_type_arg_invalid(type_):
    match = f"`type_` must be a type, a tuple of types, or a union, got `{type_}`"
    with pytest.raises(TypeError, match=match):
        check_scalar(0, type_)


def test_check_scalar_name():
    try:
        check_scalar(0, int, name="a_string", ge=1)
    except CheckError as e:
        assert "`a_string >= 1` not satisfied, got `0`" in str(e)


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
    match = f"`name` must be a `str`, got `{name}`"
    with pytest.raises(TypeError, match=match):
        check_scalar(0, int, name=name)


def test_check_scalar_unsupported_operators_keywords():
    match = "unsupported operator keyword\\(s\\) `foo`, `bar`"
    operators = OrderedDict([("foo", 0), ("ge", 1), ("bar", 2), ("lt", 3)])
    with pytest.raises(TypeError, match=match):
        check_scalar(0, int, **operators)


@pytest.mark.parametrize(
    ("scalar", "type_", "ge"),
    (
        pytest.param(False, bool, False, id="False >= False"),
        pytest.param(0, int, 0, id="0 >= 0"),
        pytest.param(0.0, float, 0.0, id="0.0 >= 0.0"),
    ),
)
def test_check_scalar_ge(scalar, type_, ge):
    assert check_scalar(scalar, type_, ge=ge) == scalar


@pytest.mark.parametrize(
    ("scalar", "type_", "ge"),
    (
        pytest.param(False, bool, True, id="False >= True"),
        pytest.param(0, int, 1, id="0 >= 1"),
        pytest.param(0.0, float, 1.0, id="0.0 >= 1.0"),
    ),
)
def test_check_scalar_ge_not_satisfied(scalar, type_, ge):
    match = f"`scalar >= {ge}` not satisfied, got `{scalar}`"
    with pytest.raises(CheckError, match=match):
        check_scalar(scalar, type_, ge=ge)


@pytest.mark.parametrize(
    ("scalar", "type_", "gt"),
    (
        pytest.param(True, bool, False, id="True > False"),
        pytest.param(1, int, 0, id="1 > 0"),
        pytest.param(1.0, float, 0.0, id="1.0 > 0.0"),
    ),
)
def test_check_scalar_gt(scalar, type_, gt):
    assert check_scalar(scalar, type_, gt=gt) == scalar


@pytest.mark.parametrize(
    ("scalar", "type_", "gt"),
    (
        pytest.param(False, bool, False, id="False > False"),
        pytest.param(0, int, 0, id="0 > 0"),
        pytest.param(0.0, float, 0.0, id="0.0 > 0.0"),
    ),
)
def test_check_scalar_gt_not_satisfied(scalar, type_, gt):
    match = f"`scalar > {gt}` not satisfied, got `{scalar}`"
    with pytest.raises(CheckError, match=match):
        check_scalar(scalar, type_, gt=gt)


@pytest.mark.parametrize(
    ("scalar", "type_", "le"),
    (
        pytest.param(True, bool, True, id="True <= True"),
        pytest.param(1, int, 1, id="1 <= 1"),
        pytest.param(1.0, float, 1.0, id="1.0 <= 1.0"),
    ),
)
def test_check_scaler_le(scalar, type_, le):
    assert check_scalar(scalar, type_, le=le) == scalar


@pytest.mark.parametrize(
    ("scalar", "type_", "le"),
    (
        pytest.param(True, bool, False, id="True <= False"),
        pytest.param(1, int, 0, id="1 <= 0"),
        pytest.param(1.0, float, 0.0, id="1.0 <= 0.0"),
    ),
)
def test_check_scalar_le_not_satisfied(scalar, type_, le):
    match = f"`scalar <= {le}` not satisfied, got `{scalar}`"
    with pytest.raises(CheckError, match=match):
        check_scalar(scalar, type_, le=le)


@pytest.mark.parametrize(
    ("scalar", "type_", "lt"),
    (
        pytest.param(False, bool, True, id="False < True"),
        pytest.param(0, int, 1, id="0 < 1"),
        pytest.param(0.0, float, 1.0, id="0.0 < 1.0"),
    ),
)
def test_check_scaler_lt(scalar, type_, lt):
    assert check_scalar(scalar, type_, lt=lt) == scalar


@pytest.mark.parametrize(
    ("scalar", "type_", "lt"),
    (
        pytest.param(True, bool, True, id="True < True"),
        pytest.param(1, int, 1, id="1 < 1"),
        pytest.param(1.0, float, 1.0, id="1.0 < 1.0"),
    ),
)
def test_check_scalar_lt_not_satisfied(scalar, type_, lt):
    match = f"`scalar < {lt}` not satisfied, got `{scalar}`"
    with pytest.raises(CheckError, match=match):
        check_scalar(scalar, type_, lt=lt)


@pytest.mark.parametrize(
    ("operator_keyword", "operator_symbol"),
    (
        pytest.param("ge", ">=", id="ge"),
        pytest.param("gt", ">", id="gt"),
        pytest.param("le", "<=", id="le"),
        pytest.param("lt", "<", id="lt"),
    ),
)
def test_check_scalar_ge_gt_le_lt_arg_invalid(operator_keyword, operator_symbol):
    match = (
        f"`{operator_symbol}` \\(`{operator_keyword}`\\) not supported "
        f"between instances of `int` and `str`"
    )
    operators = {operator_keyword: "a_string"}
    with pytest.raises(TypeError, match=match):
        check_scalar(0, int, **operators)


@pytest.mark.parametrize(
    ("scalar", "type_", "eq"),
    (
        pytest.param(False, bool, False, id="False == False"),
        pytest.param(0, int, 0, id="0 == 0"),
        pytest.param(0.0, float, 0.0, id="0.0 == 0.0"),
    ),
)
def test_check_scalar_eq(scalar, type_, eq):
    assert check_scalar(scalar, type_, eq=eq) == scalar


@pytest.mark.parametrize(
    ("scalar", "type_", "eq"),
    (
        pytest.param(False, bool, True, id="False == True"),
        pytest.param(0, int, 1, id="0 == 1"),
        pytest.param(0.0, float, 1.0, id="0.0 == 1.0"),
    ),
)
def test_check_scalar_eq_not_satisfied(scalar, type_, eq):
    match = f"`scalar == {eq}` not satisfied, got `{scalar}`"
    with pytest.raises(CheckError, match=match):
        check_scalar(scalar, type_, eq=eq)


@pytest.mark.parametrize(
    ("scalar", "type_", "ne"),
    (
        pytest.param(False, bool, True, id="False != True"),
        pytest.param(0, int, 1, id="0 != 1"),
        pytest.param(0.0, float, 1.0, id="0.0 != 1.0"),
    ),
)
def test_check_scalar_ne(scalar, type_, ne):
    assert check_scalar(scalar, type_, ne=ne) == scalar


@pytest.mark.parametrize(
    ("scalar", "type_", "ne"),
    (
        pytest.param(False, bool, False, id="False != False"),
        pytest.param(0, int, 0, id="0 != 0"),
        pytest.param(0.0, float, 0.0, id="0.0 != 0.0"),
    ),
)
def test_check_scalar_ne_not_satisfied(scalar, type_, ne):
    match = f"`scalar != {ne}` not satisfied, got `{scalar}`"
    with pytest.raises(CheckError, match=match):
        check_scalar(scalar, type_, ne=ne)


@pytest.mark.parametrize(
    ("scalar", "type_", "in_"),
    (
        pytest.param(False, bool, (False, True), id="False in (False, True)"),
        pytest.param(0, int, (0, 1), id="0 in (0, 1)"),
        pytest.param(0.0, float, (0.0, 1.0), id="0.0 in (0.0, 1.0)"),
    ),
)
def test_check_scalar_in(scalar, type_, in_):
    assert check_scalar(scalar, type_, in_=in_) == scalar


@pytest.mark.parametrize(
    ("scalar", "type_", "in_"),
    (
        pytest.param(False, bool, (True,), id="False in (True,)"),
        pytest.param(-1, int, (0, 1), id="-1 in (0, 1)"),
        pytest.param(-1.0, float, (0.0, 1.0), id="-1.0 in (0.0, 1.0)"),
    ),
)
def test_check_scalar_in_not_satisfied(scalar, type_, in_):
    match = f"`scalar in {in_}` not satisfied, got `{scalar}`"
    match = match.replace("(", "\\(")
    match = match.replace(")", "\\)")
    with pytest.raises(CheckError, match=match):
        check_scalar(scalar, type_, in_=in_)


@pytest.mark.parametrize(
    ("scalar", "type_", "not_in"),
    (
        pytest.param(False, bool, (True,), id="False not in (True,)"),
        pytest.param(-1, int, (0, 1), id="-1 not in (0, 1)"),
        pytest.param(-1.0, float, (0.0, 1.0), id="-1.0 not in (0.0, 1.0)"),
    ),
)
def test_check_scalar_not_in(scalar, type_, not_in):
    assert check_scalar(scalar, type_, not_in=not_in) == scalar


@pytest.mark.parametrize(
    ("scalar", "type_", "not_in"),
    (
        pytest.param(False, bool, (False, True), id="False not in (True,)"),
        pytest.param(0, int, (0, 1), id="0 not in (0, 1)"),
        pytest.param(0.0, float, (0.0, 1.0), id="0.0 not in (0.0, 1.0)"),
    ),
)
def test_check_scalar_not_in_not_satisfied(scalar, type_, not_in):
    match = f"`scalar not in {not_in}` not satisfied, got `{scalar}`"
    match = match.replace("(", "\\(")
    match = match.replace(")", "\\)")
    with pytest.raises(CheckError, match=match):
        check_scalar(scalar, type_, not_in=not_in)


@pytest.mark.parametrize(
    "operator",
    (
        pytest.param("in_", id="in_"),
        pytest.param("not_in", id="not_in"),
    ),
)
def test_check_scalar_in_not_in_arg_invalid(operator):
    match = f"`{operator}` must be iterable, got `0`"
    operators = {operator: 0}
    with pytest.raises(TypeError, match=match):
        check_scalar(0, int, **operators)


@pytest.mark.parametrize(
    ("sequence", "type_"),
    (
        pytest.param((False, True), bool, id="bool"),
        pytest.param([1, 2, 3], int, id="int"),
        pytest.param([1.0, 2.0, 3.0], float, id="float"),
    ),
)
def test_check_sequence(sequence, type_):
    assert check_sequence(sequence, type_) == sequence


@pytest.mark.parametrize(
    ("sequence", "type_", "operators", "match"),
    (
        pytest.param(
            (False, True, 2, 3.0),
            bool,
            {"le": True},
            (
                "`sequence[2]` must be of type `bool`",
                "`sequence[2] <= True` not satisfied",
                "`sequence[3]` must be of type `bool`",
            ),
            id="bool",
        ),
        pytest.param(
            (False, 1, 2, 3.0),
            int,
            {"ge": 1},
            (
                "`sequence[0] >= 1` not satisfied",
                "`sequence[3]` must be of type `int`",
            ),
            id="int",
        ),
        pytest.param(
            (0.0, True, 2.0, 3.0),
            float,
            {"lt": 3.0},
            (
                "`sequence[1]` must be of type `float`",
                "`sequence[3] < 3.0` not satisfied",
            ),
            id="float",
        ),
    ),
)
def test_check_sequence_conditions_not_satisfied(sequence, type_, operators, match):
    match = ".*\\n.*".join(match)
    match = match.replace("[", "\\[")
    match = match.replace("]", "\\]")
    with pytest.raises(CheckError, match=match):
        check_sequence(sequence, type_, **operators)


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
def test_check_sequence_arg_invalid(sequence):
    match = f"`sequence` must be a sequence, got `{sequence}`"
    with pytest.raises(CheckError, match=match):
        check_sequence(sequence, int)


@pytest.mark.parametrize(
    "type_",
    (
        pytest.param(0, id="0"),
        pytest.param("forty-two", id="forty-two"),
        pytest.param(None, id="None"),
        pytest.param(lambda: 0, id="lambda: 0"),
    ),
)
def test_check_sequence_type_arg_invalid(type_):
    match = f"`type_` must be a type, a tuple of types, or a union, got `{type_}`"
    with pytest.raises(TypeError, match=match):
        check_sequence((1, 2, 3), type_)


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
    match = f"`name` must be a `str`, got `{name}`"
    with pytest.raises(TypeError, match=match):
        check_sequence((1, 2, 3), int, name=name)


@pytest.mark.parametrize(
    ("sequence", "length"),
    (
        pytest.param((1, 2, 3), 3, id="(1, 2, 3) with length=3"),
        pytest.param((1, 2, 3), None, id="(1, 2, 3) with length=None"),
        pytest.param((1, 2, 3, 4), 4, id="(1, 2, 3, 4) with length=4"),
        pytest.param((1, 2, 3, 4), None, id="(1, 2, 3, 4) with length=None"),
    ),
)
def test_check_sequence_length(sequence, length):
    assert check_sequence(sequence, int, length=length) == sequence


def test_check_sequence_length_not_satisfied():
    match = "`sequence` must have length `4`, but `len\\(sequence\\) = 3`"
    with pytest.raises(CheckError, match=match):
        check_sequence((1, 2, 3), int, length=4)


@pytest.mark.parametrize(
    "length",
    (
        pytest.param(0, id="0"),
        pytest.param(-1, id="-1"),
        pytest.param("forty-two", id="forty-two"),
        pytest.param(lambda: 0, id="lambda: 0"),
    ),
)
def test_check_sequence_length_arg_invalid(length):
    match = "`length` must be a positive `int` or `None`"
    with pytest.raises(TypeError, match=match):
        check_sequence((1, 2, 3), int, length=length)


def test_check_sequence_unsupported_operators_keywords():
    match = "unsupported operator keyword\\(s\\) `foo`, `bar`"
    operators = OrderedDict([("foo", 0), ("ge", 1), ("bar", 2), ("lt", 3)])
    with pytest.raises(TypeError, match=match):
        check_sequence((1, 2, 3), int, **operators)


@pytest.mark.parametrize(
    "scalar_or_sequence",
    (
        pytest.param(0, id="0"),
        pytest.param((1, 2, 3), id="(1, 2, 3)"),
    ),
)
def test_check_scalar_or_sequence(scalar_or_sequence):
    assert check_scalar_or_sequence(scalar_or_sequence, int) == scalar_or_sequence


@pytest.mark.parametrize(
    "type_",
    (
        pytest.param(42, id="0"),
        pytest.param("forty-two", id="forty-two"),
        pytest.param(None, id="None"),
        pytest.param(lambda: 0, id="lambda: 0"),
    ),
)
def test_check_scalar_or_sequence_type_arg_invalid(type_):
    match = f"`type_` must be a type, a tuple of types, or a union, got `{type_}`"
    with pytest.raises(TypeError, match=match):
        check_scalar_or_sequence(0, type_)
    with pytest.raises(TypeError, match=match):
        check_scalar_or_sequence((1, 2, 3), type_)


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
    match = f"`name` must be a `str`, got `{name}`"
    with pytest.raises(TypeError, match=match):
        check_scalar_or_sequence(0, int, name=name)
    with pytest.raises(TypeError, match=match):
        check_scalar_or_sequence((1, 2, 3), int, name=name)


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
    match = "`length` must be a positive `int` or `None`"
    with pytest.raises(TypeError, match=match):
        check_scalar_or_sequence((1, 2, 3), int, length=length)


def test_check_scalar_or_sequence_unsupported_operators_keywords():
    match = "unsupported operator keyword\\(s\\) `foo`, `bar`"
    operators = OrderedDict([("foo", 0), ("ge", 1), ("bar", 2), ("lt", 3)])
    with pytest.raises(TypeError, match=match):
        check_scalar_or_sequence(0, int, **operators)
    with pytest.raises(TypeError, match=match):
        check_scalar_or_sequence((1, 2, 3), int, **operators)
