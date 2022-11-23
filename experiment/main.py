import argparse

import jax
import jax.numpy as jnp
from jax.experimental.compilation_cache.compilation_cache import initialize_cache
from flax.jax_utils import replicate, unreplicate
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T

from .train import create_train_state, train_step, cross_replica_mean, scale_step


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--scaling_epochs', type=int)
    parser.add_argument('--scaling_rate', type=float)
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    scaling_epochs = args.scaling_epochs
    scaling_rate = args.scaling_rate

    device_count = jax.local_device_count()
    assert batch_size % device_count == 0, f'batch_size should be divisible by {device_count}'

    root = 'torchvision/datasets'
    specimen = jnp.empty((28, 28, 1))

    transform = T.Compose([
        T.ToTensor(),
        lambda X: torch.permute(X, (1, 2, 0)),  # (C, H, W) -> (H, W, C)
    ])

    train_dataset = MNIST(root, train=True, download=True, transform=transform)

    key = jax.random.PRNGKey(42)
    num_classes = 10


    print('===> Training Alice')
    key_init, key = jax.random.split(key)
    alice = create_train_state(key_init, num_classes, learning_rate, specimen)
    alice = replicate(alice)
    train_loader = DataLoader(train_dataset, batch_size)
    for epoch in range(epochs):
        alice, loss = train_epoch(alice, device_count, train_loader)
        with jnp.printoptions(precision=3):
            print(f'Epoch {epoch + 1}, train loss: {loss}')

    # Sync the batch statistics across replicas so that evaluation is deterministic.
    alice = alice.replace(batch_stats=cross_replica_mean(alice.batch_stats))


    print('===> Training Bob')
    key_init, key = jax.random.split(key)
    bob = create_train_state(key_init, num_classes, learning_rate, specimen)
    bob = replicate(bob)
    train_loader = DataLoader(train_dataset, batch_size)
    for epoch in range(epochs):
        bob, loss = train_epoch(bob, device_count, train_loader)
        with jnp.printoptions(precision=3):
            print(f'Epoch {epoch + 1}, train loss: {loss}')

    # Sync the batch statistics across replicas so that evaluation is deterministic.
    bob = bob.replace(batch_stats=cross_replica_mean(bob.batch_stats))


    print('===> Scaling')
    factor = {
        'params': jax.tree_util.tree_map(lambda _: 0., alice.params),
        'batch_stats': jax.tree_util.tree_map(lambda _: 0., alice.batch_stats),
    }
    factor = replicate(factor)
    for epoch in range(scaling_epochs):
        key_scale, key = jax.random.split(key)
        factor, loss = scale_epoch(factor, key_scale, alice, bob, scaling_rate, device_count, train_loader)
        with jnp.printoptions(precision=3):
            print(f'Epoch {epoch + 1}, scale loss: {loss}')
            print(f'{unreplicate(factor)}')


def train_epoch(state, device_count, loader):
    epoch_loss = 0
    for X, y in loader:
        remainder = X.shape[0] % device_count
        if remainder != 0:
            X = X[:-remainder]
            y = y[:-remainder]

        image = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
        label = jnp.array(y).reshape(device_count, -1, *y.shape[1:])

        state, loss = train_step(state, image, label)
        epoch_loss += loss.sum()

    return state, epoch_loss


def scale_epoch(factor, key, alice, bob, scaling_rate, device_count, loader):
    epoch_loss = 0
    for X, y in loader:
        remainder = X.shape[0] % device_count
        if remainder != 0:
            X = X[:-remainder]
            y = y[:-remainder]

        image = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
        label = jnp.array(y).reshape(device_count, -1, *y.shape[1:])

        key_scale, key = jax.random.split(key)
        factor, loss = scale_step(factor, replicate(key_scale), alice, bob, scaling_rate, image, label)
        epoch_loss += loss.sum()

    return factor, epoch_loss


if __name__ == '__main__':
    initialize_cache('jit_cache')
    cli()
