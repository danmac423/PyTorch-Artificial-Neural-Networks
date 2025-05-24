import lightning as L
import torch
import torch.nn as nn
import torchvision

import wandb


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        embedding_dim: int,
        img_channels: int,
        features_g: int = 64,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.img_channels = img_channels
        self.features_g = features_g

        self.embedding = nn.Embedding(
            num_embeddings=num_classes, embedding_dim=embedding_dim
        )
        self.main = nn.Sequential(
            # Block 1: (N, latent_dim + embedding_dim, 1, 1) -> (N, features_g * 4, 4, 4)
            nn.ConvTranspose2d(
                latent_dim + embedding_dim, features_g * 4, 4, 1, 0, bias=False
            ),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            # Block 2: (N, features_g * 4, 4, 4) -> (N, features_g * 2, 8, 8)
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            # Block 3: (N, features_g * 2, 8, 8) -> (N, features_g, 16, 16)
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            # Block 4: (N, features_g, 16, 16) -> (N, img_channels, 32, 32)
            nn.ConvTranspose2d(features_g, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_embedded = self.embedding(y)

        combined_input = torch.cat([z, y_embedded], dim=1)

        x = combined_input.view(-1, self.latent_dim + self.embedding_dim, 1, 1)

        return self.main(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        img_channels: int = 3,
        features_d: int = 64,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.img_channels = img_channels
        self.features_d = features_d

        self.embedding = nn.Embedding(
            num_embeddings=num_classes, embedding_dim=embedding_dim
        )

        self.main = nn.Sequential(
            # Block 1: (N, img_channels + embedding_dim, 32, 32) -> (N, features_d, 16, 16)
            nn.Conv2d(img_channels + embedding_dim, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 2: (N, features_d, 16, 16) -> (N, features_d * 2, 8, 8)
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 3: (N, features_d * 2, 8, 8) -> (N, features_d * 4, 4, 4)
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 4: (N, features_d * 4, 4, 4) -> (N, 1, 1, 1)
            nn.Conv2d(features_d * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, img: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_embedded: torch.Tensor = self.embedding(y)

        y_embedded_spatial = y_embedded.unsqueeze(-1).unsqueeze(-1)
        y_embedded_spatial = y_embedded_spatial.expand(-1, -1, img.size(2), img.size(3))

        combined_input = torch.cat([img, y_embedded_spatial], dim=1)

        return self.main(combined_input).view(-1, 1)


class ConditionalDCGAN(L.LightningModule):
    def __init__(
        self,
        generator_params: dict,
        discriminator_params: dict,
        optimizer_params: dict,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.generator_params = generator_params
        self.discriminator_params = discriminator_params
        self.optimizer_params = optimizer_params

        self.generator = Generator(**generator_params)
        self.discriminator = Discriminator(**discriminator_params)

        self.criterion = nn.BCELoss()

        self.register_buffer("val_z", torch.randn(64, generator_params["latent_dim"]))
        self.register_buffer(
            "val_y",
            torch.randint(low=0, high=discriminator_params["num_classes"], size=(64,)),
        )

        self.automatic_optimization = False

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.generator(z, y)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        real_images, labels = batch

        real_images = real_images.to(self.device)
        labels = labels.to(self.device)

        noise = torch.randn(
            real_images.size(0),
            self.generator_params["latent_dim"],
            device=self.device,
        )

        valid = torch.ones(real_images.size(0), 1, device=self.device)
        fake = torch.zeros(real_images.size(0), 1, device=self.device)

        # Train Generator
        opt_g.zero_grad()

        fake_images = self.generator(noise, labels)

        g_loss = self.criterion(self.discriminator(fake_images, labels), valid)

        self.log("train/generator_loss", g_loss, prog_bar=True)

        self.manual_backward(g_loss)
        opt_g.step()

        # Train Discriminator
        opt_d.zero_grad()

        real_loss = self.criterion(self.discriminator(real_images, labels), valid)
        fake_loss = self.criterion(
            self.discriminator(fake_images.detach(), labels), fake
        )

        d_loss = (real_loss + fake_loss) / 2

        self.log("train/discriminator_loss", d_loss, prog_bar=True)

        self.manual_backward(d_loss)
        opt_d.step()

    def on_train_epoch_end(self):
        self.generator.eval()

        with torch.inference_mode():
            sample_images = self.generator(self.val_z, self.val_y)
            grid = torchvision.utils.make_grid(sample_images, nrow=8, normalize=True)
            grid_image = wandb.Image(grid, caption="Generated Images")
            wandb.log({"generated_images": grid_image})

        self.generator.train()

    def configure_optimizers(self):
        generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), **self.optimizer_params
        )
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), **self.optimizer_params
        )
        return generator_optimizer, discriminator_optimizer
