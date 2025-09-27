
import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import numpy as np

model = smp.Unet(
    encoder_name="resnet34",  # you can choose other encoders like 'resnet50'
    encoder_weights=None,      # set to 'imagenet' for pretrained
    in_channels=3,
    classes=1                  
)
model.eval()  
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def segment_retina(img_path):
    """
    Simulates retinal segmentation using a U-Net model from SMP.
    
    Args:
        img_path (str): Path to the retinal image.
        
    Returns:
        np.ndarray: Binary segmentation mask (0 or 1) of shape (256, 256)
    """
    image = Image.open(img_path).convert('RGB')
    x = preprocess(image).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        mask = model(x).squeeze().numpy()  # remove batch dimension

    # Convert to binary mask (simulation)
    mask = (mask > 0.5).astype(np.uint8)
    return mask
