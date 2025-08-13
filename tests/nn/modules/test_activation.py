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
import inspect

import pytest
import torch

from renda._exceptions import _CheckError
from renda.nn.modules.activation import Activation


@pytest.mark.parametrize(
    ("activation", "check_activation_attribute"),
    (
        pytest.param(
            "Sigmoid",
            lambda a: isinstance(a, torch.nn.Sigmoid),
            id="'Sigmoid'",
        ),
        pytest.param(
            "sigmoid",
            inspect.isfunction,
            id="'sigmoid'",
        ),
        pytest.param(
            torch.nn.ReLU(),
            lambda a: isinstance(a, torch.nn.ReLU),
            id="torch.nn.ReLU()",
        ),
        pytest.param(
            torch.nn.functional.relu,
            inspect.isfunction,
            id="torch.nn.functional.relu",
        ),
        pytest.param(
            torch.nn.LeakyReLU,
            lambda a: isinstance(a, torch.nn.LeakyReLU),
            id="torch.nn.LeakyReLU",
        ),
        pytest.param(
            torch.nn.functional.leaky_relu,
            inspect.isfunction,
            id="torch.nn.functional.leaky_relu",
        ),
    ),
)
def test_activation(activation, check_activation_attribute):
    activation_ = Activation(activation=activation)

    assert isinstance(activation_, torch.nn.Module)
    assert check_activation_attribute(activation_.activation)

    # Test if forward implementation works
    # We know this is true for all our examples
    input_ = torch.ones(5, 2)
    output = activation_(input_)
    assert output.shape == torch.Size([5, 2])


@pytest.mark.parametrize(
    ("activation", "match"),
    (
        pytest.param(
            "AString",
            "^`activation` \\(when passed as a `str`\\)",
            id="'AString'",
        ),
        pytest.param(
            "a_string",
            "^`activation` \\(when passed as a `str`\\)",
            id="'a_string'",
        ),
        pytest.param(
            torch.Tensor,
            "^`activation` \\(when passed as a `class`\\)",
            id="torch.Tensor",
        ),
        pytest.param(
            0,
            "^`activation` must be one of the following",
            id="0",
        ),
    ),
)
def test_activation_activation_arg_invalid(activation, match):
    with pytest.raises(_CheckError, match=match):
        Activation(activation=activation)
