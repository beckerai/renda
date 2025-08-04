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
from torch.utils.data import Dataset

from renda._exceptions import _CheckError
from renda.data.galaxy import GalaxyDataset
from renda.random import MAX_SEED, MIN_SEED


def test_galaxy_dataset():
    galaxy_dataset = GalaxyDataset()

    assert galaxy_dataset.num_classes == 3
    assert galaxy_dataset.num_samples_per_class == 480
    assert galaxy_dataset.num_rotations == 1.0
    assert galaxy_dataset.entwinement == 0.3

    assert isinstance(galaxy_dataset, Dataset)
    assert len(galaxy_dataset) == 1440

    C = galaxy_dataset.num_classes
    N_per_class = galaxy_dataset.num_samples_per_class
    N = len(galaxy_dataset)
    assert C * N_per_class == N

    X, Y = galaxy_dataset[:]
    assert X.shape == torch.Size([N, 2])
    assert Y.shape == torch.Size([N, 1])
    assert len(Y.unique()) == C


@pytest.mark.parametrize(
    "num_classes",
    (
        pytest.param(1, id="1 (min)"),
        pytest.param(3, id="3 (default)"),
        pytest.param(10, id="10 (other)"),
    ),
)
def test_galaxy_dataset_num_classes_arg_valid(num_classes):
    galaxy_dataset = GalaxyDataset(
        num_classes=num_classes,
    )
    assert galaxy_dataset.num_classes == num_classes

    C = galaxy_dataset.num_classes
    N_per_class = galaxy_dataset.num_samples_per_class
    N = len(galaxy_dataset)
    assert N_per_class * C == N


@pytest.mark.parametrize(
    "num_classes",
    (
        pytest.param(0, id="0"),
        pytest.param(-1, id="-1"),
        pytest.param(1.0, id="1.0"),
    ),
)
def test_galaxy_dataset_num_classes_arg_invalid(num_classes):
    with pytest.raises(_CheckError):
        GalaxyDataset(num_classes=num_classes)


@pytest.mark.parametrize(
    "num_samples_per_class",
    (
        pytest.param(1, id="1 (min)"),
        pytest.param(480, id="480 (default)"),
        pytest.param(100, id="100 (other)"),
        pytest.param((10, 20, 30), id="(10, 20, 30) (tuple)"),
        pytest.param([10, 20, 30], id="[10, 20, 30] (list)"),
    ),
)
def test_galaxy_dataset_num_samples_per_class_arg_valid(num_samples_per_class):
    galaxy_dataset = GalaxyDataset(num_samples_per_class=num_samples_per_class)

    C = galaxy_dataset.num_classes
    N_per_class = galaxy_dataset.num_samples_per_class
    N = len(galaxy_dataset)
    assert N_per_class * C == N or sum(N_per_class) == N


@pytest.mark.parametrize(
    "num_samples_per_class",
    (
        pytest.param(0, id="0"),
        pytest.param(-1, id="-1"),
        pytest.param(1.0, id="1.0"),
        pytest.param((10, 20, 30, 40), id="tuple, too many elements"),
        pytest.param([100, 20], id="list, too few elements"),
        pytest.param((10.5, 20, 30), id="tuple, element oy type float"),
    ),
)
def test_galaxy_dataset_num_samples_per_class_arg_invalid(num_samples_per_class):
    with pytest.raises(_CheckError):
        GalaxyDataset(num_samples_per_class=num_samples_per_class)


@pytest.mark.parametrize(
    "num_rotations",
    (
        pytest.param(0.5, id="0.5"),
        pytest.param(1.0, id="1.0 (default)"),
        pytest.param(1.5, id="1.5"),
        pytest.param(2.0, id="2 (int)"),
    ),
)
def test_galaxy_dataset_num_rotations_arg_valid(num_rotations):
    galaxy_dataset = GalaxyDataset(num_rotations=num_rotations)
    assert galaxy_dataset.num_rotations == num_rotations


@pytest.mark.parametrize(
    "num_rotations",
    (
        pytest.param(0.0, id="0.0"),
        pytest.param(-0.5, id="-0.5"),
        pytest.param(-1, id="-1 (int)"),
    ),
)
def test_galaxy_dataset_num_rotations_arg_invalid(num_rotations):
    with pytest.raises(_CheckError):
        GalaxyDataset(num_rotations=num_rotations)


@pytest.mark.parametrize(
    "entwinement",
    (
        pytest.param(0.1, id="0.1"),
        pytest.param(0.3, id="0.3 (== default)"),
        pytest.param(0.5, id="0.5"),
        pytest.param(1, id="1 (int)"),
    ),
)
def test_galaxy_dataset_entwinement_arg_valid(entwinement):
    galaxy_dataset = GalaxyDataset(entwinement=entwinement)
    assert galaxy_dataset.entwinement == entwinement


@pytest.mark.parametrize(
    "entwinement",
    (
        pytest.param(0.0, id="0.0"),
        pytest.param(-0.5, id="-0.5"),
        pytest.param(-1, id="-1 (int)"),
    ),
)
def test_galaxy_dataset_entwinement_arg_invalid(entwinement):
    with pytest.raises(_CheckError):
        GalaxyDataset(entwinement=entwinement)


def test_galaxy_dataset_seed_arg_same_seeds():
    Xa, Ya = GalaxyDataset(seed=0)[:]
    Xb, Yb = GalaxyDataset(seed=0)[:]
    assert torch.all(torch.eq(Xa, Xb))
    assert torch.all(torch.eq(Ya, Yb))


def test_galaxy_dataset_seed_arg_different_seeds():
    Xa, Ya = GalaxyDataset(seed=0)[:]
    Xb, Yb = GalaxyDataset(seed=1)[:]
    assert not torch.all(torch.eq(Xa, Xb))
    assert torch.all(torch.eq(Ya, Yb))


def test_galaxy_dataset_seed_arg_none_as_seed():
    Xa, Ya = GalaxyDataset(seed=None)[:]
    Xb, Yb = GalaxyDataset(seed=None)[:]
    assert not torch.all(torch.eq(Xa, Xb))
    assert torch.all(torch.eq(Ya, Yb))


@pytest.mark.parametrize(
    "seed",
    (
        pytest.param(0.0, id="0.0"),
        pytest.param("zero", id="zero"),
        pytest.param(MIN_SEED - 1, id="MIN_SEED - 1"),
        pytest.param(MAX_SEED + 1, id="MAX_SEED + 1"),
        pytest.param(lambda: 0, id="lambda: 0"),
    ),
)
def test_galaxy_dataset_seed_arg_invalid(seed):
    with pytest.raises(_CheckError):
        GalaxyDataset(seed=seed)
