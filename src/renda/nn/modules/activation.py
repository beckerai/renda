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
from typing import Any, Callable

import torch
from torch import Tensor
from torch.nn import Module

from renda._exceptions import _CheckError


class Activation(Module):
    activation: Module | Callable[..., Any]

    def __init__(
        self,
        activation: str | type[Module] | Module | Callable[..., Any],
    ) -> None:
        super().__init__()

        # ----
        # str
        # ----
        if isinstance(activation, str):
            modules = (torch.nn, torch.nn.functional)
            for module in modules:
                activation_ = getattr(module, activation, None)
                if activation_ is not None:
                    break
            else:
                raise _CheckError(
                    f"`activation` (when passed as a `str`) must be the name "
                    f"of a class from `torch.nn` or a function from "
                    f"`torch.nn.functional`, got `{activation}`"
                )
        else:
            activation_ = activation

        # ----------------
        # Module subclass
        # ----------------
        if inspect.isclass(activation_):
            if issubclass(activation_, Module):
                activation_ = activation_()
            else:
                raise _CheckError(
                    f"`activation` (when passed as a `class`) must be a "
                    f"`torch.nn.Module` subclass, got `{activation}`"
                )

        # -------
        # Module
        # -------
        if isinstance(activation_, Module):
            self.activation = activation_

        # ---------
        # function
        # ---------
        elif callable(activation_):
            self.activation = lambda input_: activation_(input_)

        # ------------------
        # raise _CheckError
        # ------------------
        else:
            raise _CheckError(
                f"`activation` must be one of the following (got {activation}):\n"
                f"  - the name (`str`) of a class/function from one of these modules:\n"
                f"    `torch.nn`, `torch.nn.functional`\n"
                f"  - a `torch.nn.Module` subclass\n"
                f"  - an instance of a `torch.nn.Module` subclass\n"
                f"  - a function"
            )

    def forward(self, input_: Tensor) -> Tensor:
        return self.activation(input_)
