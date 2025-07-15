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

from renda.data.galaxy import GalaxyDataset
from renda.random import MAX_SEED, MIN_SEED


@pytest.mark.parametrize(
    ("num_classes", "num_samples_per_class", "num_rotations", "sigma", "seed"),
    (
        pytest.param(3, 480, 1.0, 0.3, 0, id="defaults"),
        pytest.param(3, (960, 240, 240), 1, 3, None, id="test1"),
        pytest.param(5, 288, True, True, False, id="test2"),
    ),
)
def test_galaxy_dataset(
    num_classes,
    num_samples_per_class,
    num_rotations,
    sigma,
    seed,
):
    galaxy_dataset = GalaxyDataset(
        num_classes=num_classes,
        num_samples_per_class=num_samples_per_class,
        num_rotations=num_rotations,
        sigma=sigma,
        seed=seed,
    )

    assert galaxy_dataset.num_classes == num_classes
    assert galaxy_dataset.num_samples_per_class == num_samples_per_class
    assert galaxy_dataset.num_rotations == num_rotations
    assert galaxy_dataset.sigma == sigma
    assert galaxy_dataset.seed == seed

    assert isinstance(galaxy_dataset, Dataset)
    assert len(galaxy_dataset) == 1440

    X, Y = galaxy_dataset[:]
    assert X.shape[1] == 2
    assert len(Y.unique()) == num_classes


def test_galaxy_dataset_for_same_seeds():
    Xa, Ya = GalaxyDataset(seed=0)[:]
    Xb, Yb = GalaxyDataset(seed=0)[:]
    assert torch.all(torch.eq(Xa, Xb))
    assert torch.all(torch.eq(Ya, Yb))


def test_galaxy_dataset_for_different_seeds():
    Xa, Ya = GalaxyDataset(seed=0)[:]
    Xb, Yb = GalaxyDataset(seed=1)[:]
    assert not torch.all(torch.eq(Xa, Xb))
    assert torch.all(torch.eq(Ya, Yb))


def test_galaxy_dataset_for_none_as_seed():
    Xa, Ya = GalaxyDataset(seed=None)[:]
    Xb, Yb = GalaxyDataset(seed=None)[:]
    assert not torch.all(torch.eq(Xa, Xb))
    assert torch.all(torch.eq(Ya, Yb))


@pytest.mark.parametrize(
    "num_classes",
    (
        pytest.param(0, id="0"),
        pytest.param(-1, id="-1"),
        pytest.param(4.2, id="4.2"),
        pytest.param("forty-two", id="forty-two"),
        pytest.param(None, id="None"),
    ),
)
def test_galaxy_dataset_for_invalid_num_classes(num_classes):
    with pytest.raises(ValueError):
        GalaxyDataset(num_classes=num_classes)


@pytest.mark.parametrize(
    "num_samples_per_class",
    (
        pytest.param(0, id="0"),
        pytest.param(-1, id="-1"),
        pytest.param((100, 100), id="(100, 100)"),
        pytest.param((100, 100, 100, 100), id="(100, 100, 100, 100)"),
        pytest.param(4.2, id="4.2"),
        pytest.param((4.2, 4.2, 4.2), id="(4.2, 4.2, 4.2)"),
        pytest.param("forty-two", id="forty-two"),
        pytest.param(None, id="None"),
    ),
)
def test_galaxy_dataset_for_invalid_num_samples_per_class(num_samples_per_class):
    with pytest.raises(ValueError):
        GalaxyDataset(num_samples_per_class=num_samples_per_class)


@pytest.mark.parametrize(
    "num_rotations",
    (
        pytest.param(0.0, id="0.0"),
        pytest.param(0, id="0"),
        pytest.param(-1.0, id="-1.0"),
        pytest.param(-1, id="-1"),
        pytest.param((1, 1, 1), id="(1, 1, 1)"),
        pytest.param("forty-two", id="forty-two"),
        pytest.param(None, id="None"),
    ),
)
def test_galaxy_dataset_for_invalid_num_rotations(num_rotations):
    with pytest.raises(ValueError):
        GalaxyDataset(num_rotations=num_rotations)


@pytest.mark.parametrize(
    "sigma",
    (
        pytest.param(0.0, id="0.0"),
        pytest.param(0, id="0"),
        pytest.param(-1.0, id="-1.0"),
        pytest.param(-1, id="-1"),
        pytest.param((1, 1, 1), id="(1, 1, 1)"),
        pytest.param("forty-two", id="forty-two"),
        pytest.param(None, id="None"),
    ),
)
def test_galaxy_dataset_for_invalid_sigma(sigma):
    with pytest.raises(ValueError):
        GalaxyDataset(sigma=sigma)


@pytest.mark.parametrize(
    "seed",
    (
        pytest.param(4.2, id="4.2"),
        pytest.param("forty-two", id="forty-two"),
        pytest.param(MIN_SEED - 1, id="MIN_SEED_minus_1"),
        pytest.param(MAX_SEED + 1, id="MAX_SEED_plus_1"),
    ),
)
def test_galaxy_dataset_for_invalid_seed(seed):
    with pytest.raises(ValueError):
        GalaxyDataset(seed=seed)
