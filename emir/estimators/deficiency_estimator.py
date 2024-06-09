import torch
import torch.nn as nn
import logging

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from .ftensordataloader import FastTensorDataLoader

import torch.nn.utils.parametrize as parametrize

from torch.func import grad, vmap


@dataclass(frozen=True)
class GANDeficiencyArgs:

    gan_batch_size: int = 64
    gan_n_epochs: int = 100
    critic_repeats: int = 5

    # Generator hyperparameters
    gen_hidden_dim: int = 16
    gen_n_layers: int = 5
    gen_lr: float = 0.0001

    # Critic hyperparameters
    critic_hidden_dim: int = 16
    critic_n_layers: int = 3
    critic_lr: float = 0.001

    # noise hyperparameters
    noise_dim: int = 16
    noise_std = 0.1

    # Wasserstein hyperparameters
    disc_clip: float = 0.01

    device: str = "cpu"


class GANDeficiencyEstimator:
    def __init__(self, args: GANDeficiencyArgs, x_dim: int, y_dim: int):
        """
        \\delta (X ⟶ Y)
        """
        self.args = args
        self.logger = logging.getLogger(__name__)

        self.x_dim = x_dim
        self.y_dim = y_dim

        self.gen = Generator(
            x_dim=self.x_dim,
            hidden_dim=self.args.gen_hidden_dim,
            y_dim=self.y_dim,
        )
        self.crit = Critic(y=self.y_dim, hidden_dim=self.args.critic_hidden_dim)

        self.adversarial_loss = nn.BCEWithLogitsLoss(reduction="none")

        self.recorded_gen_loss = []
        self.recorded_critic_loss = []

    def eval(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.fit_estimator(x, y)

        with torch.no_grad():
            output = self.batch_eval(x, y)

        return output

    def batch_eval(
        self, x: torch.Tensor, y: torch.Tensor, per_samples=False
    ) -> torch.Tensor:

        true_labels = torch.ones(y.shape[0], 1)
        fake_labels = torch.zeros(x.shape[0], 1)

        data_loader = FastTensorDataLoader(
            x,
            fake_labels,
            y,
            true_labels,
            batch_size=self.args.gan_batch_size,
        )

        deficiencies = []
        for x_batch, fake_labels_batch, y_batch, true_labels_batch in data_loader:
            x_batch = x_batch.to(self.args.device)
            fake_labels_batch = fake_labels_batch.to(self.args.device)
            y_batch = y_batch.to(self.args.device)
            true_labels_batch = true_labels_batch.to(self.args.device)

            deficiency = self.deficiency_per_sample(
                x_batch, fake_labels_batch, y_batch, true_labels_batch
            )
            deficiencies.append(deficiency)

        deficiencies = torch.cat(deficiencies)

        if not per_samples:
            return deficiencies.mean()
        else:
            return deficiencies

    def deficiency_per_sample(
        self,
        x: torch.Tensor,
        fake_labels: torch.Tensor,
        y: torch.Tensor,
        true_labels: torch.Tensor,
    ) -> torch.Tensor:
        fake_y = self.gen(x)

        # Get the critic predictions
        critic_fake_pred = self.crit(y, fake_y)
        critic_real_pred = self.crit(y, y)

        # Compute loss

        fake_loss = self.adversarial_loss(critic_fake_pred, fake_labels)
        real_loss = self.adversarial_loss(critic_real_pred, true_labels)

        return torch.concat([fake_loss, real_loss], dim=0)

    def fit_estimator(self, x: torch.Tensor, y: torch.Tensor):

        # Make true and fake labels
        true_labels = torch.ones(y.shape[0], 1) * 1
        fake_labels = torch.zeros(x.shape[0], 1)

        # Create the data loader

        # Create the optimizers
        gen_optim = torch.optim.RMSprop(self.gen.parameters(), lr=self.args.gen_lr)
        critic_optim = torch.optim.RMSprop(
            self.crit.parameters(), lr=self.args.critic_lr
        )

        # Move the models to the device
        self.gen.to(self.args.device)
        self.crit.to(self.args.device)

        # Training loop
        for epoch in range(self.args.gan_n_epochs):
            data_loader = FastTensorDataLoader(
                x,
                fake_labels,
                y,
                true_labels,
                batch_size=self.args.gan_batch_size,
                shuffle=True,
            )

            for x_batch, fake_labels_batch, y_batch, true_labels_batch in data_loader:
                # Move the data to the device
                x_batch = x_batch.to(self.args.device)
                fake_labels_batch = fake_labels_batch.to(self.args.device)
                y_batch = y_batch.to(self.args.device)
                true_labels_batch = true_labels_batch.to(self.args.device)

                # Train the critic
                _critic_loss = []
                for _ in range(self.args.critic_repeats):
                    critic_loss = self.critic_training_step(
                        x_batch,
                        fake_labels_batch,
                        y_batch,
                        true_labels_batch,
                        critic_optim,
                    )

                    _critic_loss.append(critic_loss)

                self.recorded_critic_loss.append(
                    torch.stack(_critic_loss).mean().item()
                )

                # Train the generator
                gen_loss = self.generator_training_step(
                    x_batch, fake_labels_batch, y_batch, true_labels_batch, gen_optim
                )

                self.recorded_gen_loss.append(gen_loss.item())

            self.logger.info(
                f"Epoch {epoch+1}/{self.args.gan_n_epochs}, Critic loss: {self.recorded_critic_loss[-1]}, Generator loss: {self.recorded_gen_loss[-1]}"
            )

    def critic_training_step(self, x, fake_labels, y, true_labels, critic_optim):
        # Zero the gradients
        critic_optim.zero_grad()

        # create fake y from x
        fake_y = self.gen(x)

        # Get the critic predictions
        critic_fake_pred = self.crit(y, fake_y)
        critic_real_pred = self.crit(y, y)

        fake_loss = self.adversarial_loss(critic_fake_pred, fake_labels).mean()
        real_loss = self.adversarial_loss(critic_real_pred, true_labels).mean()

        loss = (fake_loss + real_loss) / 2
        loss.backward()

        # Step the optimizer
        critic_optim.step()

        return loss

    def generator_training_step(self, x, fake_labels, y, true_labels, gen_optim):
        # Zero the gradients
        gen_optim.zero_grad()

        # create fake y from x
        fake_y = self.gen(x)

        # Get the critic predictions
        critic_fake_pred = self.crit(y, fake_y)

        # Compute loss
        loss = self.adversarial_loss(critic_fake_pred, true_labels).mean()

        loss.backward()

        # Step the optimizer
        gen_optim.step()

        return loss


class GANTrickedDeficiencyEstimator(GANDeficiencyEstimator):

    def __init__(self, args: GANDeficiencyArgs, x_dim: int, y_dim: int):
        super().__init__(args, x_dim, y_dim)

    def fit_estimator(self, x: torch.Tensor, y: torch.Tensor):
        # Make true and fake labels
        true_labels = torch.ones(y.shape[0], 1) * 0.9
        fake_labels = torch.ones(x.shape[0], 1) * 0.1

        # Create the data loader

        # Create the optimizers
        gen_optim = torch.optim.AdamW(self.gen.parameters(), lr=self.args.gen_lr)
        critic_optim = torch.optim.AdamW(self.crit.parameters(), lr=self.args.critic_lr)

        # Move the models to the device
        self.gen.to(self.args.device)
        self.crit.to(self.args.device)

        # Training loop
        for epoch in range(self.args.gan_n_epochs):
            data_loader = FastTensorDataLoader(
                x,
                fake_labels,
                y,
                true_labels,
                batch_size=self.args.gan_batch_size,
                shuffle=True,
            )
            _critic_loss = []

            for x_batch, fake_labels_batch, y_batch, true_labels_batch in data_loader:
                # Move the data to the device
                x_batch = x_batch.to(self.args.device)
                fake_labels_batch = fake_labels_batch.to(self.args.device)
                y_batch = y_batch.to(self.args.device)
                true_labels_batch = true_labels_batch.to(self.args.device)

                critic_loss = self.critic_training_step(
                    x_batch,
                    fake_labels_batch,
                    y_batch,
                    true_labels_batch,
                    critic_optim,
                )

                _critic_loss.append(critic_loss)

            self.recorded_critic_loss.append(torch.stack(_critic_loss).mean().item())

            _gen_loss = []
            for x_batch, fake_labels_batch, y_batch, true_labels_batch in data_loader:
                x_batch = x_batch.to(self.args.device)
                fake_labels_batch = fake_labels_batch.to(self.args.device)
                y_batch = y_batch.to(self.args.device)
                true_labels_batch = true_labels_batch.to(self.args.device)

                # Train the generator
                gen_loss = self.generator_training_step(
                    x_batch, fake_labels_batch, y_batch, true_labels_batch, gen_optim
                )

                _gen_loss.append(gen_loss)

            self.recorded_gen_loss.append(torch.stack(_gen_loss).mean().item())

            self.logger.info(
                f"Epoch {epoch+1}/{self.args.gan_n_epochs}, Critic loss: {self.recorded_critic_loss[-1]}, Generator loss: {self.recorded_gen_loss[-1]}"
            )


class WassersteinDeficiencyEstimator(GANDeficiencyEstimator):

    def __init__(self, args: GANDeficiencyArgs, x_dim: int, y_dim: int):
        super().__init__(args, x_dim, y_dim)

    def critic_training_step(self, x, fake_labels, y, true_labels, critic_optim):
        # Zero the gradients
        critic_optim.zero_grad()

        # create fake y from x
        fake_y = self.gen(x)

        # Get the critic predictions
        critic_fake_pred = self.crit(y, fake_y)
        critic_real_pred = self.crit(y, y)

        # Compute loss
        loss = critic_fake_pred.mean() - critic_real_pred.mean()

        loss.backward()
        # Step the optimizer
        critic_optim.step()

        # clip the weights
        for p in self.crit.parameters():
            p.data.clamp_(min=-self.args.disc_clip, max=self.args.disc_clip)

        return loss

    def generator_training_step(self, x, fake_labels, y, true_labels, gen_optim):
        # Zero the gradients
        gen_optim.zero_grad()

        # create fake y from x
        fake_y = self.gen(x)

        # add noise
        noise = torch.randn_like(fake_y) * self.args.noise_std
        fake_y = fake_y + noise

        # Get the critic predictions
        critic_fake_pred = self.crit(y, fake_y)

        # Compute loss
        loss = -critic_fake_pred.mean()

        loss.backward()

        # Step the optimizer
        gen_optim.step()

        return loss


class WassersteinGPDeficiencyEstimator(WassersteinDeficiencyEstimator):
    def __init__(self, args: GANDeficiencyArgs, x_dim: int, y_dim: int):
        super().__init__(args, x_dim, y_dim)

        self.critic_gradient = vmap(
            grad(lambda x, y: self.crit(x, y).squeeze(), argnums=1), in_dims=0
        )

    def critic_training_step(self, x, fake_labels, y, true_labels, critic_optim):
        # Zero the gradients
        critic_optim.zero_grad()

        # create fake y from x
        fake_y = self.gen(x)

        # Get the critic predictions
        critic_fake_pred = self.crit(y, fake_y)
        critic_real_pred = self.crit(y, y)

        # Compute loss
        loss = critic_fake_pred.mean() - critic_real_pred.mean()

        # Compute the gradient penalty
        eps = torch.rand(y.shape[0], 1, device=self.args.device)
        y_hat = eps * y + (1 - eps) * fake_y

        gradients = self.critic_gradient(y, y_hat)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        loss = loss + 10 * gradient_penalty

        loss.backward()
        # Step the optimizer
        critic_optim.step()

        return loss


class Generator(nn.Module):
    def __init__(self, x_dim: int, hidden_dim: int, y_dim: int, n_layers: int = 3):
        super(Generator, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Define the model
        self.model = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
                for _ in range(n_layers)
            ],
            nn.Linear(hidden_dim, y_dim),
        )

    def forward(self, x):
        return self.model(x)


class Critic(nn.Module):

    def __init__(self, y: int, hidden_dim: int, n_layers: int = 3):
        super(Critic, self).__init__()
        self.y = y
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Define the model
        self.model = nn.Sequential(
            nn.Linear(y, hidden_dim),
            nn.ReLU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU())
                for _ in range(n_layers)
            ],
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, _, x):
        return self.model(x)


class SmartCritic(nn.Module):
    def __init__(self, y: int, hidden_dim: int, n_layers: int = 3):
        super(SmartCritic, self).__init__()
        self.y = y
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Define the model
        self.model = nn.Sequential(
            nn.Linear(2 * y, hidden_dim),
            nn.ReLU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU())
                for _ in range(n_layers)
            ],
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))
