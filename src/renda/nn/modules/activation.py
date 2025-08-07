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
from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module

from renda._exceptions import _CheckError


class Activation(Module):
    def __init__(
        self,
        activation: str | Module,
    ) -> None:
        super().__init__()


def _foo(activation) -> Callable[[Tensor], Tensor]:
    if isinstance(activation, str):
        activation_ = getattr(torch.nn, activation, None)
        if activation_ is not None:
            return activation_()

        activation_ = getattr(torch.nn.functional, activation, None)
        if activation_ is not None:
            return activation_

    if inspect.isclass(activation):
        activation_ = activation()
    else:
        activation_ = activation

    if isinstance(activation_, Callable):
        return activation_

    raise _CheckError()
