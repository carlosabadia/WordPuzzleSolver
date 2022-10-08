import torch
import torchvision
from torch import nn


def create_model(num_classes: int = 32,
                          seed: int = 42):
    """Creates a feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of classes in the classifier head.
            Defaults to 32.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): vit feature extractor model.
        transforms (torchvision.transforms): vit image transforms.
    """
    IMG_SIZE = 28
    transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()])

        # Create a convolutional neural network
    class Model(nn.Module):
            def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
                super().__init__()
                self.block_1 = nn.Sequential(
                        nn.Conv2d(in_channels=input_shape,
                                        out_channels=hidden_units,
                                  kernel_size=3,  # how big is the square that's going over the image?
                                  stride=1,  # default
                                        padding=1),  # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
                    nn.ReLU(),
                        nn.Conv2d(in_channels=hidden_units,
                                  out_channels=hidden_units,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2,
                                     stride=2)  # default stride value is same as kernel_size
                )
                self.block_2 = nn.Sequential(
                    nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                        nn.Linear(in_features=hidden_units*7*7,
                                  out_features=output_shape)
                )

            def forward(self, x: torch.Tensor):
                # x = self.block_1(x)
                # print(x.shape)
                # x = self.block_2(x)
                # print(x.shape)
                # x = self.classifier(x)
                # print(x.shape)
                x = self.classifier(self.block_2(self.block_1(x)))
                return x
    return Model, transforms
