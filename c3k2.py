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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.act  = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ------------------------------------------------------------
# 2. C3k2 block
# ------------------------------------------------------------
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, k=(3, 3), e=0.5):
        super().__init__()
        mid = int(out_channels * e)
        self.cv1 = YoloConvBlock(in_channels,  mid,          kernel_size=k[0], stride=1, padding=k[0]//2)
        self.cv2 = YoloConvBlock(mid,           out_channels, kernel_size=k[1], stride=1, padding=k[1]//2)
        self.add = shortcut and (in_channels == out_channels)

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k2(nn.Module):
    def __init__(self, in_channels, out_channels, n=2, c3k=False, e=0.5, shortcut=True):
        super().__init__()
        mid = int(out_channels * e)
        k   = (3, 3)
        self.cv1 = YoloConvBlock(in_channels, mid,       kernel_size=1, stride=1, padding=0)
        self.cv2 = YoloConvBlock(in_channels, mid,       kernel_size=1, stride=1, padding=0)
        self.cv3 = YoloConvBlock(2 * mid,     out_channels, kernel_size=1, stride=1, padding=0)
        self.bottlenecks = nn.Sequential(
            *[Bottleneck(mid, mid, shortcut=shortcut, k=k, e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        return self.cv3(torch.cat([self.bottlenecks(self.cv1(x)), self.cv2(x)], dim=1))


# ------------------------------------------------------------
# 3. Load image
# ------------------------------------------------------------
img         = Image.open("input/dog.png").convert("RGB")
input_image = T.ToTensor()(img).unsqueeze(0)   # [1, 3, H, W]

# ------------------------------------------------------------
# 4. Build blocks and run pipeline
# ------------------------------------------------------------
conv_block = YoloConvBlock(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1)
c3k2_block = C3k2(in_channels=8, out_channels=16, n=2, c3k=False, e=0.5, shortcut=True)

conv_block.eval()
c3k2_block.eval()

with torch.no_grad():
    conv_out  = conv_block(input_image)   # [1,  8, H/2, W/2]
    c3k2_out  = c3k2_block(conv_out)     # [1, 16, H/2, W/2]

# ------------------------------------------------------------
# 5. Print shapes
# ------------------------------------------------------------
print("YOLO-style block: Conv2d(kernel_size=3, stride=2, padding=1) -> BatchNorm2d -> SiLU")
print(f"Input tensor shape  : {tuple(input_image.shape)}")
print(f"After ConvBlock     : {tuple(conv_out.shape)}  (stride=2 halves H,W)")
print(f"After C3k2          : {tuple(c3k2_out.shape)} (same H,W, more channels)")

# ------------------------------------------------------------
# 6. Prepare arrays
# ------------------------------------------------------------
input_np       = input_image[0].permute(1, 2, 0).numpy()
kernel_np      = conv_block.conv.weight[0, 0].detach().numpy()
feature_map_np = conv_out[0, 0].detach().numpy()

# ------------------------------------------------------------
# 7. Figure 1 — Input vs ConvBlock output (side by side)
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(input_np)
axes[0].set_title(f"Input image\n{tuple(input_image.shape[-2:])} × 3 (RGB)")
axes[0].axis("off")

im = axes[1].imshow(feature_map_np, cmap="viridis")
axes[1].set_title(f"Output feature map 0\n{tuple(conv_out.shape[-2:])} (after Conv + BN + SiLU)")
axes[1].axis("off")
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

plt.suptitle("YOLO Conv Block: stride=2 halves spatial resolution", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 8. Figure 2 — Kernel slice
# ------------------------------------------------------------
plt.figure(figsize=(3, 3))
plt.imshow(kernel_np)
plt.title("Kernel slice\n(out ch 0, in ch R)")
plt.axis("off")
plt.colorbar()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 9. Figure 3 — All 16 C3k2 feature maps in 4×4 grid
# ------------------------------------------------------------
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    fm = c3k2_out[0, i].detach().numpy()
    im = ax.imshow(fm, cmap="viridis")
    ax.set_title(f"ch {i}", fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("All 16 C3k2 output feature maps (4×4 grid)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("c3k2_all_feature_maps.png", dpi=150, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 10. Print kernel values
# ------------------------------------------------------------
print("\nExample 3×3 kernel slice (output channel 0, input channel R):")
print(np.round(kernel_np, 4))