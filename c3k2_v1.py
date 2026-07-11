"""
Visualize how an input image is transformed by a YOLO Conv block and a C3k2 block
(structure following YOLO26, Jocher et al. 2026, Fig. S2 / Ultralytics C2f-based C3k2).

Produces three poster-ready figures:
  fig1_pipeline.png   -- Input -> Conv -> C3k2 at a glance (with tensor shapes)
  fig2_inside_c3k2.png -- what happens INSIDE C3k2: chunk -> bottlenecks -> dense concat
  fig3_channel_grid.png -- all 16 output channels, sorted by activation strength

Note: with random (untrained) weights the maps are random projections -- label them
as such, or load pretrained weights for more meaningful responses (see bottom).
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(7)

# ------------------------------------------------------------
# Building blocks (corrected C3k2, C2f-based as in YOLO26)
# ------------------------------------------------------------
class YoloConvBlock(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(c2)
        self.act  = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, k=(3, 3), e=0.5):
        super().__init__()
        mid = int(c2 * e)
        self.cv1 = YoloConvBlock(c1, mid, k=k[0], s=1, p=k[0] // 2)
        self.cv2 = YoloConvBlock(mid, c2, k=k[1], s=1, p=k[1] // 2)
        self.add = shortcut and (c1 == c2)

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k2(nn.Module):
    """C3k2 (c3k=False variant): 1x1 conv -> chunk(2) -> n bottlenecks,
    ALL intermediate outputs kept -> concat (2+n)*c -> 1x1 conv."""
    def __init__(self, c1, c2, n=2, e=0.5, shortcut=True):
        super().__init__()
        self.c   = int(c2 * e)
        self.cv1 = YoloConvBlock(c1, 2 * self.c, k=1, s=1, p=0)
        self.cv2 = YoloConvBlock((2 + n) * self.c, c2, k=1, s=1, p=0)
        self.m   = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut=shortcut, k=(3, 3), e=0.5)
            for _ in range(n)
        )

    def forward(self, x, return_intermediates=False):
        y = list(self.cv1(x).chunk(2, dim=1))          # [y0 bypass, y1 working path]
        inter = {"y0": y[0], "y1": y[1]}
        for i, m in enumerate(self.m):
            y.append(m(y[-1]))                          # keep every intermediate output
            inter[f"b{i+1}"] = y[-1]
        cat = torch.cat(y, dim=1)                       # (2+n)*c channels
        out = self.cv2(cat)
        if return_intermediates:
            inter["cat"], inter["out"] = cat, out
            return out, inter
        return out


# ------------------------------------------------------------
# Helpers for display
# ------------------------------------------------------------
def norm01(a):
    """Normalize a 2D map to [0,1] for consistent display."""
    a = a - a.min()
    return a / (a.max() + 1e-8)

def summary_map(t):
    """Summarize a [1,C,H,W] tensor as one 2D map (mean absolute activation)."""
    return norm01(t[0].abs().mean(dim=0).numpy())

def best_channel(t):
    """Pick the channel with the highest spatial variance (most 'interesting')."""
    var = t[0].flatten(1).var(dim=1)
    idx = int(var.argmax())
    return norm01(t[0, idx].numpy()), idx


# ------------------------------------------------------------
# Run the pipeline
# ------------------------------------------------------------
img = Image.open("input/dog.png").convert("RGB")
x   = T.ToTensor()(img).unsqueeze(0)                    # [1, 3, H, W]

conv = YoloConvBlock(3, 8)                              # stride 2 -> H/2, W/2
c3k2 = C3k2(8, 16, n=2)
conv.eval(); c3k2.eval()

with torch.no_grad():
    f_conv        = conv(x)                             # [1,  8, H/2, W/2]
    f_out, inter  = c3k2(f_conv, return_intermediates=True)  # [1, 16, H/2, W/2]

CMAP = "inferno"   # reads well on posters, also in grayscale print

# ------------------------------------------------------------
# FIGURE 1 -- Pipeline at a glance: Input -> Conv -> C3k2
# ------------------------------------------------------------
panels = [
    (np.asarray(img),            f"Input image\n{tuple(x.shape)}",         None),
    (summary_map(f_conv),        f"After Conv (s=2)\n{tuple(f_conv.shape)}", CMAP),
    (summary_map(f_out),         f"After C3k2\n{tuple(f_out.shape)}",       CMAP),
]
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, (im, title, cmap) in zip(axes, panels):
    ax.imshow(im, cmap=cmap)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
# arrows between panels
for xpos, label in [(0.355, "Conv\n3x3, s=2"), (0.665, "C3k2\nn=2")]:
    fig.text(xpos, 0.55, r"$\longrightarrow$", fontsize=22, ha="center")
    fig.text(xpos, 0.70, label, fontsize=9, ha="center")
fig.suptitle("From image to features: Conv halves resolution, C3k2 refines features",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("fig1_pipeline.png", dpi=200, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# FIGURE 2 -- Inside C3k2: chunk -> bottlenecks -> dense concat
# ------------------------------------------------------------
c = c3k2.c
steps = [
    (summary_map(f_conv),        f"Input\n8 ch"),
    (summary_map(inter["y0"]),   f"y0: bypass\n{c} ch"),
    (summary_map(inter["y1"]),   f"y1: working path\n{c} ch"),
    (summary_map(inter["b1"]),   f"Bottleneck 1(y1)\n{c} ch"),
    (summary_map(inter["b2"]),   f"Bottleneck 2(b1)\n{c} ch"),
    (summary_map(inter["cat"]),  f"Concat all 4\n{4*c} ch"),
    (summary_map(inter["out"]),  f"1x1 Conv fuse\n16 ch"),
]
fig, axes = plt.subplots(1, len(steps), figsize=(3 * len(steps), 3.4))
for ax, (im, title) in zip(axes, steps):
    ax.imshow(im, cmap=CMAP)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
fig.suptitle("Inside C3k2: 1x1 conv -> chunk(2) -> sequential bottlenecks -- "
             "ALL intermediate outputs are concatenated (dense connectivity)",
             fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("fig2_inside_c3k2.png", dpi=200, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# FIGURE 3 -- All 16 output channels, sorted by activation variance
# ------------------------------------------------------------
var_order = f_out[0].flatten(1).var(dim=1).argsort(descending=True)
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, ch in zip(axes.flat, var_order):
    ax.imshow(norm01(f_out[0, ch].numpy()), cmap=CMAP)
    ax.set_title(f"ch {int(ch)}", fontsize=9)
    ax.axis("off")
fig.suptitle("All 16 C3k2 output channels (sorted by activation variance)",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("fig3_channel_grid.png", dpi=200, bbox_inches="tight")
plt.show()

print("Saved: fig1_pipeline.png, fig2_inside_c3k2.png, fig3_channel_grid.png")

# ------------------------------------------------------------
# OPTIONAL: use trained weights instead of random ones
# ------------------------------------------------------------
# With random weights the maps are arbitrary projections. For feature maps that
# visibly respond to edges/body parts, copy the first layers of a trained model:
#
#   from ultralytics import YOLO
#   yolo = YOLO("yolo26n-pose.pt")
#   layers = yolo.model.model          # nn.Sequential of the real network
#   conv_trained = layers[0]           # first Conv (3 -> C)
#   c3k2_trained = layers[2]           # first C3k2 in the backbone
#   # then run: f = conv_trained(x); out = c3k2_trained(layers[1](f)) etc.
#
# and add a caption "untrained random weights" / "pretrained weights" accordingly.