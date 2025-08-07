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
import pytest
import torch
from torch.nn import Module

from renda._exceptions import _CheckError
from renda.nn.modules.linear import DecoderLinear, EncoderLinear
from renda.random import MAX_SEED, MIN_SEED


@pytest.mark.parametrize(
    "seed",
    (
        pytest.param(1, id="1"),
        pytest.param(MIN_SEED, id="MIN_SEED"),
        pytest.param(MAX_SEED, id="MAX_SEED"),
    ),
)
@pytest.mark.parametrize("bias", (False, True))
@pytest.mark.parametrize("out_features", (1, 4))
@pytest.mark.parametrize("in_features", (1, 25))
@pytest.mark.parametrize("linear_class", (EncoderLinear, DecoderLinear))
def test_linear(linear_class, in_features, out_features, bias, seed):
    linear = linear_class(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        seed=seed,
    )

    assert isinstance(linear, Module)

    assert linear.in_features == in_features
    assert linear.out_features == out_features
    if bias:
        assert linear.bias.shape == torch.Size([out_features])
    else:
        assert linear.bias is None
    assert linear.seed == seed

    if linear_class is EncoderLinear:
        assert linear.weight.shape == torch.Size([out_features, in_features])

    elif linear_class is DecoderLinear:
        assert linear.weight.shape == torch.Size([in_features, out_features])

    input_ = torch.randn(1000, in_features)
    output = linear(input_)
    assert output.shape == torch.Size([1000, out_features])


@pytest.mark.parametrize("in_features", (0, pytest.param(-25, id="(-25)"), 25.0))
@pytest.mark.parametrize("linear_class", (EncoderLinear, DecoderLinear))
def test_linear_in_features_arg_invalid(linear_class, in_features):
    with pytest.raises(_CheckError):
        linear_class(in_features=in_features, out_features=4)


@pytest.mark.parametrize("out_features", (0, pytest.param(-4, id="(-4)"), 4.0))
@pytest.mark.parametrize("linear_class", (EncoderLinear, DecoderLinear))
def test_linear_out_features_arg_invalid(linear_class, out_features):
    with pytest.raises(_CheckError):
        linear_class(in_features=25, out_features=out_features)


@pytest.mark.parametrize(
    "bias",
    (
        pytest.param(0, id="0"),
        pytest.param(0.0, id="0.0"),
        pytest.param("zero", id="zero"),
        pytest.param(lambda: 0, id="(lambda: 0)"),
    ),
)
@pytest.mark.parametrize("linear_class", (EncoderLinear, DecoderLinear))
def test_linear_bias_arg_invalid(linear_class, bias):
    with pytest.raises(_CheckError):
        linear_class(in_features=25, out_features=4, bias=bias)


@pytest.mark.parametrize("linear_class", (EncoderLinear, DecoderLinear))
def test_linear_seed_arg_same_seeds(linear_class):
    La = linear_class(in_features=25, out_features=4, seed=0)
    Lb = linear_class(in_features=25, out_features=4, seed=0)
    assert torch.all(torch.eq(La.weight, Lb.weight))
    assert torch.all(torch.eq(La.bias, Lb.bias))


@pytest.mark.parametrize("linear_class", (EncoderLinear, DecoderLinear))
def test_linear_seed_arg_different_seeds(linear_class):
    La = linear_class(in_features=25, out_features=4, seed=0)
    Lb = linear_class(in_features=25, out_features=4, seed=1)
    assert not torch.all(torch.eq(La.weight, Lb.weight))
    assert not torch.all(torch.eq(La.bias, Lb.bias))


@pytest.mark.parametrize("linear_class", (EncoderLinear, DecoderLinear))
def test_linear_seed_arg_none_as_seed(linear_class):
    La = linear_class(in_features=25, out_features=4, seed=None)
    Lb = linear_class(in_features=25, out_features=4, seed=None)
    assert not torch.all(torch.eq(La.weight, Lb.weight))
    assert not torch.all(torch.eq(La.bias, Lb.bias))


@pytest.mark.parametrize(
    "seed",
    (
        pytest.param(0.0, id="0.0"),
        pytest.param("zero", id="zero"),
        pytest.param(MIN_SEED - 1, id="(MIN_SEED - 1)"),
        pytest.param(MAX_SEED + 1, id="(MAX_SEED + 1)"),
        pytest.param(lambda: 0, id="(lambda: 0)"),
    ),
)
@pytest.mark.parametrize("linear_class", (EncoderLinear, DecoderLinear))
def test_linear_seed_arg_invalid(linear_class, seed):
    with pytest.raises(_CheckError):
        linear_class(in_features=25, out_features=4, seed=seed)
