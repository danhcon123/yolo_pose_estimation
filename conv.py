import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Reproducible random values
torch.manual_seed(7)

# ------------------------------------------------------------
# 1. YOLO-style convolution block: Conv2d -> BatchNorm2d -> SiLU
# ------------------------------------------------------------
class YoloConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,  # common when BatchNorm follows the convolution
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ------------------------------------------------------------
# 2. Random RGB input image
#    Tensor layout in PyTorch: [batch, channels, height, width]
# ------------------------------------------------------------
img = Image.open("cideon/image.png").convert("RGB")
input_image = T.ToTensor()(img).unsqueeze(0)

# Use 8 output channels to keep the visualization simple.
# YOLO models normally use more channels in real layers.
block = YoloConvBlock(
    in_channels=3,
    out_channels=8,
    kernel_size=3,
    stride=2,
    padding=1,
)

# Inference mode avoids updating BatchNorm statistics during the demo.
block.eval()

with torch.no_grad():
    output = block(input_image)

# ------------------------------------------------------------
# 3. Print the important dimensions
# ------------------------------------------------------------
print("YOLO-style block: Conv2d(kernel_size=3, stride=2, padding=1) -> BatchNorm2d -> SiLU")
print(f"Input tensor shape : {tuple(input_image.shape)}  = [batch, RGB channels, height, width]")
print(f"Output tensor shape: {tuple(output.shape)} = [batch, feature channels, height, width]")
print()
print("Because stride = 2, the spatial resolution is reduced approximately by half:")
print("64 x 64  ->  32 x 32")
print()
print("The block generates 8 output feature maps because out_channels = 8.")
print("Below, only output feature map 0 is displayed.")

# ------------------------------------------------------------
# 4. Prepare arrays for visualization
# ------------------------------------------------------------
input_np = input_image[0].permute(1, 2, 0).numpy()
kernel_np = block.conv.weight[0, 0].detach().numpy()  # output channel 0, input channel R
feature_map_np = output[0, 0].detach().numpy()

# ------------------------------------------------------------
# 5. Plot each visual separately (no subplots)
# ------------------------------------------------------------
plt.figure(figsize=(5, 5))
plt.imshow(input_np)
plt.title("Random RGB input image: 64 × 64 × 3")
plt.axis("off")
plt.show()

plt.figure(figsize=(4, 4))
plt.imshow(kernel_np)
plt.title("Example kernel slice: output channel 0, input channel R")
plt.axis("off")
plt.colorbar()
plt.show()

plt.figure(figsize=(5, 5))
plt.imshow(feature_map_np)
plt.title("Output feature map 0 after Conv2d + BatchNorm + SiLU: 32 × 32")
plt.axis("off")
plt.colorbar()
plt.show()

# ------------------------------------------------------------
# 6. Show numerical kernel values for one slice
# ------------------------------------------------------------
print("\nExample 3 x 3 kernel slice (output channel 0, input channel R):")
print(np.round(kernel_np, 4))
