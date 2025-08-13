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
import math

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn import functional as F
from torch.nn import init

from renda._checks import _check_scalar, _check_seed
from renda._messages import _BUG_MESSAGE
from renda.random import temp_seed


class _Linear(Module):
    def __init__(
        self,
        # -------
        # Linear
        # -------
        in_features: int,
        out_features: int,
        bias: bool,
        # -----
        # Init
        # -----
        seed: int | None,
    ) -> None:
        super().__init__()

        self.in_features = _check_scalar(
            scalar=in_features,
            type_=int,
            name="in_features",
            gt=0,
        )
        self.out_features = _check_scalar(
            scalar=out_features,
            type_=int,
            name="out_features",
            gt=0,
        )
        bias = _check_scalar(
            scalar=bias,
            type_=bool,
            name="bias",
        )
        self.seed = _check_seed(seed)

        if type(self) is EncoderLinear:
            self.weight = Parameter(torch.empty(out_features, in_features))
        elif type(self) is DecoderLinear:
            self.weight = Parameter(torch.empty(in_features, out_features))
        else:  # pragma: no cover
            raise TypeError(
                f"unsupported subclass `{type(self).__name__}` of `_Linear`, "
                f"supported subclasses are `EncoderLinear` and `DecoderLinear`"
                f"{_BUG_MESSAGE}"
            )

        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        with temp_seed(self.seed):
            # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
            # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
            # https://github.com/pytorch/pytorch/issues/57109
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.bias, -bound, bound)


class EncoderLinear(_Linear):
    def __init__(
        self,
        # -------
        # Linear
        # -------
        in_features: int,
        out_features: int,
        bias: bool = True,
        # -----
        # Init
        # -----
        seed: int | None = None,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            seed=seed,
        )

    def forward(self, input_: Tensor) -> Tensor:
        return F.linear(input_, self.weight, self.bias)


class DecoderLinear(_Linear):
    def __init__(
        self,
        # -------
        # Linear
        # -------
        in_features: int,
        out_features: int,
        bias: bool = True,
        # -----
        # Init
        # -----
        seed: int | None = None,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            seed=seed,
        )

    def forward(self, input_: Tensor) -> Tensor:
        return F.linear(input_, self.weight.t(), self.bias)
