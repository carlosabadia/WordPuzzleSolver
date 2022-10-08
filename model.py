import torch
import torchvision

from torch import nn


def create_vit16_model(num_classes:int=101, 
                          seed:int=42):
    """Creates an vit16 feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of classes in the classifier head. 
            Defaults to 3.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): vit feature extractor model. 
        transforms (torchvision.transforms): vit image transforms.
    """
    # Create vit pretrained weights, transforms and model
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT;
    transforms = weights.transforms()
    model = torchvision.models.vit_b_16(weights=weights)

    # Freeze all layers in base model
    for param in model.parameters():
        param.requires_grad = False

    # Change classifier head with random seed for reproducibility
    torch.manual_seed(seed)
    model.heads = nn.Sequential(nn.Linear(in_features=768, # keep this the same as original model
                                          out_features=num_classes)) # update to reflect target number of classes
    
    return model, transforms
