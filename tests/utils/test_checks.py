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

from renda.utils.checks import check_scalar


@pytest.mark.parametrize(
    ("value", "type_"),
    (
        pytest.param(True, bool, id="bool"),
        pytest.param(1, int, id="int"),
        pytest.param(1.0, float, id="float"),
    ),
)
def test_check_scalar_type(value, type_):
    assert check_scalar(value, type_) == value


@pytest.mark.parametrize(
    ("value", "type_"),
    (
        pytest.param(True, float, id="bool vs. float"),
        pytest.param(1, (bool, float), id="int vs. (bool, float)"),
        pytest.param(1.0, (bool, int), id="float vs. (bool, int)"),
    ),
)
def test_check_scalar_type_fails(value, type_):
    with pytest.raises(TypeError):
        check_scalar(value, type_)


@pytest.mark.parametrize(
    "type_",
    (
        pytest.param(42, id="0"),
        pytest.param("forty-two", id="forty-two"),
        pytest.param(None, id="None"),
        pytest.param(lambda: 0, id="lambda: 0"),
    ),
)
def test_check_scalar_type_invalid(type_):
    match = (
        f"`type_` must be a `type` or a `tuple` of types, unable to check if "
        f"`value=0` is an instance of `type_={type_}`"
    )
    with pytest.raises(TypeError, match=match):
        check_scalar(0, type_)


def test_check_scalar_name():
    try:
        check_scalar(0, int, ge=1, name="a_string")
    except ValueError as e:
        assert str(e) == "`a_string=0` does not satisfy `>= 1`"


@pytest.mark.parametrize(
    "name",
    (
        pytest.param(0, id="0"),
        pytest.param(0.0, id="0.0"),
        pytest.param(None, id="None"),
        pytest.param(lambda: 0, id="lambda: 0"),
    ),
)
def test_check_scalar_name_invalid(name):
    match = (
        f"`name` must be of type `str`, got `name={name}` of type "
        f"`{type(name).__qualname__}`"
    )
    with pytest.raises(TypeError, match=match):
        check_scalar(0, int, name=name)


def test_check_scalar_operator_invalid():
    match = "unsupported operator keyword\\(s\\) `foo`, `bar`, .*"
    kwargs = OrderedDict([("foo", 0), ("ge", 1), ("bar", 2)])
    with pytest.raises(TypeError, match=match):
        check_scalar(0, int, **kwargs)


@pytest.mark.parametrize(
    ("value", "type_", "ge"),
    (
        pytest.param(False, bool, False, id="False >= False"),
        pytest.param(0, int, 0, id="0 >= 0"),
        pytest.param(0.0, float, 0.0, id="0.0 >= 0.0"),
    ),
)
def test_check_scalar_ge(value, type_, ge):
    assert check_scalar(value, type_, ge=ge) == value


@pytest.mark.parametrize(
    ("value", "type_", "ge"),
    (
        pytest.param(False, bool, True, id="False >= True"),
        pytest.param(0, int, 1, id="0 >= 1"),
        pytest.param(0.0, float, 1.0, id="0.0 >= 1.0"),
    ),
)
def test_check_scalar_ge_fails(value, type_, ge):
    match = f"`value={value}` does not satisfy `>= {ge}`"
    with pytest.raises(ValueError, match=match):
        check_scalar(value, type_, ge=ge)


@pytest.mark.parametrize(
    ("value", "type_", "gt"),
    (
        pytest.param(True, bool, False, id="True > False"),
        pytest.param(1, int, 0, id="1 > 0"),
        pytest.param(1.0, float, 0.0, id="1.0 > 0.0"),
    ),
)
def test_check_scalar_gt(value, type_, gt):
    assert check_scalar(value, type_, gt=gt) == value


@pytest.mark.parametrize(
    ("value", "type_", "gt"),
    (
        pytest.param(False, bool, False, id="False > False"),
        pytest.param(0, int, 0, id="0 > 0"),
        pytest.param(0.0, float, 0.0, id="0.0 > 0.0"),
    ),
)
def test_check_scalar_gt_fails(value, type_, gt):
    match = f"`value={value}` does not satisfy `> {gt}`"
    with pytest.raises(ValueError, match=match):
        check_scalar(value, type_, gt=gt)


@pytest.mark.parametrize(
    ("value", "type_", "le"),
    (
        pytest.param(True, bool, True, id="True <= True"),
        pytest.param(1, int, 1, id="1 <= 1"),
        pytest.param(1.0, float, 1.0, id="1.0 <= 1.0"),
    ),
)
def test_check_scaler_le(value, type_, le):
    assert check_scalar(value, type_, le=le) == value


@pytest.mark.parametrize(
    ("value", "type_", "le"),
    (
        pytest.param(True, bool, False, id="True <= False"),
        pytest.param(1, int, 0, id="1 <= 0"),
        pytest.param(1.0, float, 0.0, id="1.0 <= 0.0"),
    ),
)
def test_check_scalar_le_fails(value, type_, le):
    match = f"`value={value}` does not satisfy `<= {le}`"
    with pytest.raises(ValueError, match=match):
        check_scalar(value, type_, le=le)


@pytest.mark.parametrize(
    ("value", "type_", "lt"),
    (
        pytest.param(False, bool, True, id="False < True"),
        pytest.param(0, int, 1, id="0 < 1"),
        pytest.param(0.0, float, 1.0, id="0.0 < 1.0"),
    ),
)
def test_check_scaler_lt(value, type_, lt):
    assert check_scalar(value, type_, lt=lt) == value


@pytest.mark.parametrize(
    ("value", "type_", "lt"),
    (
        pytest.param(True, bool, True, id="True < True"),
        pytest.param(1, int, 1, id="1 < 1"),
        pytest.param(1.0, float, 1.0, id="1.0 < 1.0"),
    ),
)
def test_check_scalar_lt_fails(value, type_, lt):
    match = f"`value={value}` does not satisfy `< {lt}`"
    with pytest.raises(ValueError, match=match):
        check_scalar(value, type_, lt=lt)


@pytest.mark.parametrize(
    ("operator", "operator_symbol"),
    (
        pytest.param("ge", ">=", id="ge"),
        pytest.param("gt", ">", id="gt"),
        pytest.param("le", "<=", id="le"),
        pytest.param("lt", "<", id="lt"),
    ),
)
def test_check_scalar_ge_gt_le_lt_invalid(operator, operator_symbol):
    match = (
        f"`{operator}` \\(`{operator_symbol}`\\) not supported between "
        f"instances of `int` and `str`"
    )
    kwargs = {operator: "a_string"}
    with pytest.raises(TypeError, match=match):
        check_scalar(0, int, **kwargs)


@pytest.mark.parametrize(
    ("value", "type_", "eq"),
    (
        pytest.param(False, bool, False, id="False == False"),
        pytest.param(0, int, 0, id="0 == 0"),
        pytest.param(0.0, float, 0.0, id="0.0 == 0.0"),
    ),
)
def test_check_scalar_eq(value, type_, eq):
    assert check_scalar(value, type_, eq=eq) == value


@pytest.mark.parametrize(
    ("value", "type_", "eq"),
    (
        pytest.param(False, bool, True, id="False == True"),
        pytest.param(0, int, 1, id="0 == 1"),
        pytest.param(0.0, float, 1.0, id="0.0 == 1.0"),
    ),
)
def test_check_scalar_eq_fails(value, type_, eq):
    match = f"`value={value}` does not satisfy `== {eq}`"
    with pytest.raises(ValueError, match=match):
        check_scalar(value, type_, eq=eq)


@pytest.mark.parametrize(
    ("value", "type_", "ne"),
    (
        pytest.param(False, bool, True, id="False != True"),
        pytest.param(0, int, 1, id="0 != 1"),
        pytest.param(0.0, float, 1.0, id="0.0 != 1.0"),
    ),
)
def test_check_scalar_ne(value, type_, ne):
    assert check_scalar(value, type_, ne=ne) == value


@pytest.mark.parametrize(
    ("value", "type_", "ne"),
    (
        pytest.param(False, bool, False, id="False != False"),
        pytest.param(0, int, 0, id="0 != 0"),
        pytest.param(0.0, float, 0.0, id="0.0 != 0.0"),
    ),
)
def test_check_scalar_ne_fails(value, type_, ne):
    match = f"`value={value}` does not satisfy `!= {ne}`"
    with pytest.raises(ValueError, match=match):
        check_scalar(value, type_, ne=ne)


@pytest.mark.parametrize(
    ("value", "type_", "in_"),
    (
        pytest.param(False, bool, (False, True), id="False in (False, True)"),
        pytest.param(0, int, (0, 1), id="0 in (0, 1)"),
        pytest.param(0.0, float, (0.0, 1.0), id="0.0 in (0.0, 1.0)"),
    ),
)
def test_check_scalar_in(value, type_, in_):
    assert check_scalar(value, type_, in_=in_) == value


@pytest.mark.parametrize(
    ("value", "type_", "in_"),
    (
        pytest.param(False, bool, (True,), id="False in (True,)"),
        pytest.param(-1, int, (0, 1), id="-1 in (0, 1)"),
        pytest.param(-1.0, float, (0.0, 1.0), id="-1.0 in (0.0, 1.0)"),
    ),
)
def test_check_scalar_in_fails(value, type_, in_):
    match = f"`value={value}` does not satisfy `in {in_}`"
    match = match.replace("(", "\\(")
    match = match.replace(")", "\\)")
    with pytest.raises(ValueError, match=match):
        check_scalar(value, type_, in_=in_)


@pytest.mark.parametrize(
    ("value", "type_", "not_in"),
    (
        pytest.param(False, bool, (True,), id="False not in (True,)"),
        pytest.param(-1, int, (0, 1), id="-1 not in (0, 1)"),
        pytest.param(-1.0, float, (0.0, 1.0), id="-1.0 not in (0.0, 1.0)"),
    ),
)
def test_check_scalar_not_in(value, type_, not_in):
    assert check_scalar(value, type_, not_in=not_in) == value


@pytest.mark.parametrize(
    ("value", "type_", "not_in"),
    (
        pytest.param(False, bool, (False, True), id="False not in (True,)"),
        pytest.param(0, int, (0, 1), id="0 not in (0, 1)"),
        pytest.param(0.0, float, (0.0, 1.0), id="0.0 not in (0.0, 1.0)"),
    ),
)
def test_check_scalar_not_in_fails(value, type_, not_in):
    match = f"`value={value}` does not satisfy `not in {not_in}`"
    match = match.replace("(", "\\(")
    match = match.replace(")", "\\)")
    with pytest.raises(ValueError, match=match):
        check_scalar(value, type_, not_in=not_in)


@pytest.mark.parametrize(
    "operator",
    (
        pytest.param("in_", id="in_"),
        pytest.param("not_in", id="not_in"),
    ),
)
def test_check_scalar_in_not_in_invalid(operator):
    match = f"`{operator}` must be iterable, got `0`"
    kwargs = {operator: 0}
    with pytest.raises(TypeError, match=match):
        check_scalar(0, int, **kwargs)
