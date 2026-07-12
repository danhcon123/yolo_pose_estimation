"""
Feature pyramid visualization from the REAL trained YOLO26 backbone.

Taps the yolo26n-pose model at every pyramid level (P1/2 ... P5/32) via forward
hooks and renders:

  figA_feature_pyramid.png    -- input + one summary map per level, panels drawn at
                                 PROPORTIONAL size (true relative feature-map size)
  figC_pyramid_equal_size.png -- SAME levels at EQUAL display size with nearest-
                                 neighbor rendering: resolution loss shows as
                                 visibly coarser pixel blocks (poster-friendly)
  figB_p3_channels.png        -- the 6 strongest channels at P3/8 (trained filters
                                 visibly responding to the object's structure)

Requires:  pip install ultralytics   (downloads yolo26n-pose.pt on first run)
Fallback:  set USE_TRAINED = False to run the random-weight demo blocks instead.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

IMG_PATH    = "input/dog.png"
USE_TRAINED = True
CMAP        = "viridis"   # same colormap as the C3k2 script / seminar paper figures

def norm01(a):
    a = a - a.min()
    return a / (a.max() + 1e-8)

def summary_map(t):
    """[1,C,H,W] -> normalized 2D map (mean absolute activation over channels)."""
    return norm01(t[0].abs().mean(dim=0).cpu().numpy())

# ------------------------------------------------------------
# 1. Collect feature maps per backbone layer via forward hooks
# ------------------------------------------------------------
features = {}   # layer_index -> output tensor

if USE_TRAINED:
    from ultralytics import YOLO
    yolo   = YOLO("yolo26n-pose.pt")
    layers = yolo.model.model            # top-level nn.Sequential of the network

    hooks = []
    def make_hook(idx):
        def hook(module, inp, out):
            if torch.is_tensor(out) and out.dim() == 4:
                features[idx] = out.detach()
        return hook

    # hook only the backbone (layers before the neck's first Upsample)
    for i, layer in enumerate(layers):
        if layer.__class__.__name__ == "Upsample":
            break
        hooks.append(layer.register_forward_hook(make_hook(i)))

    yolo.predict(IMG_PATH, verbose=False)   # runs letterbox preprocessing + forward
    for h in hooks:
        h.remove()

    # keep the LAST output at each distinct spatial resolution = pyramid levels
    # (e.g. Conv downsamples to P3, the following C3k2 refines at P3 -> keep C3k2)
    by_res = {}
    for idx in sorted(features):
        h, w = features[idx].shape[-2:]
        by_res[(h, w)] = (idx, features[idx])
    pyramid = sorted(by_res.values(), key=lambda t: -t[1].shape[-1])  # fine -> coarse

    layer_names = [f"{layers[i].__class__.__name__} (layer {i})" for i, _ in pyramid]
    level_names = [f"P{k+1}/{2**(k+1)}" for k in range(len(pyramid))]
    maps   = [summary_map(t) for _, t in pyramid]
    shapes = [tuple(t.shape[1:]) for _, t in pyramid]     # (C, H, W)

    from PIL import Image
    input_img = np.asarray(Image.open(IMG_PATH).convert("RGB"))

else:
    # Fallback: random-weight cascade (label it as untrained in the caption!)
    from PIL import Image
    import torchvision.transforms as T
    from visualize_c3k2 import YoloConvBlock, C3k2   # reuse the demo blocks

    img = Image.open(IMG_PATH).convert("RGB")
    input_img = np.asarray(img)
    x = T.ToTensor()(img).unsqueeze(0)

    stages, chans = [], [16, 32, 64, 128, 256]
    c_in = 3
    for c_out in chans:
        stages.append(torch.nn.Sequential(YoloConvBlock(c_in, c_out), C3k2(c_out, c_out)))
        c_in = c_out
    maps, shapes, level_names, layer_names = [], [], [], []
    with torch.no_grad():
        f = x
        for k, s in enumerate(stages):
            s.eval(); f = s(f)
            maps.append(summary_map(f)); shapes.append(tuple(f.shape[1:]))
            level_names.append(f"P{k+1}/{2**(k+1)}"); layer_names.append("Conv+C3k2 (random)")

# ------------------------------------------------------------
# 2. FIGURE A -- feature pyramid with proportionally sized panels
# ------------------------------------------------------------
widths = [shapes[0][2] * 2] + [s[2] for s in shapes]      # input twice P1 width
fig = plt.figure(figsize=(16, 5))
gs  = gridspec.GridSpec(1, len(widths), width_ratios=widths, wspace=0.35)

ax = fig.add_subplot(gs[0])
ax.imshow(input_img)
ax.set_title(f"Input\n{input_img.shape[0]}x{input_img.shape[1]}x3", fontsize=10)
ax.axis("off")

for k, (m, (C, H, W)) in enumerate(zip(maps, shapes)):
    ax = fig.add_subplot(gs[k + 1])
    ax.imshow(m, cmap=CMAP)
    ax.set_title(f"{level_names[k]}\n{H}x{W}x{C}", fontsize=10)
    ax.set_xlabel(layer_names[k], fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

fig.suptitle("YOLO26 backbone: resolution shrinks (Conv, s=2), features grow more "
             "abstract (C3k2) -- panel sizes proportional to true feature-map size",
             fontsize=13, fontweight="bold")
plt.savefig("figA_feature_pyramid.png", dpi=200, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 2b. FIGURE C -- SAME display size, nearest-neighbor upscaling:
#     every level shown at input scale, so the resolution loss
#     appears as visibly coarser pixel blocks instead of tiny tiles
# ------------------------------------------------------------
n = len(maps) + 1
fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.6))

axes[0].imshow(input_img)
axes[0].set_title(f"Input\n{input_img.shape[0]}x{input_img.shape[1]}x3", fontsize=10)
axes[0].axis("off")

for k, (m, (C, H, W)) in enumerate(zip(maps, shapes)):
    ax = axes[k + 1]
    # interpolation="nearest": each feature-map pixel becomes a visible block
    ax.imshow(m, cmap=CMAP, interpolation="nearest")
    ax.set_title(f"{level_names[k]}\n{H}x{W}x{C}", fontsize=10)
    ax.set_xlabel(layer_names[k], fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

fig.suptitle("Same levels at equal display size: coarser pixel blocks = lower "
             "resolution (nearest-neighbor rendering, no smoothing)",
             fontsize=13, fontweight="bold")
plt.savefig("figC_pyramid_equal_size.png", dpi=200, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 3. FIGURE B -- decomposition of ONE level: summary map (as in
#    fig C) on the left + 6 individual channels of the SAME output
# ------------------------------------------------------------
if USE_TRAINED:
    LEVEL_LAYER = 2                       # layer 2 = first C3k2 (P2/4 output)
    t = features[LEVEL_LAYER]             # [1, C, H, W]
    C, H, W = t.shape[1:]

    # 6 channels with the highest spatial variance (most structured maps);
    # replace with e.g. [0, 5, 12, 20, 33, 47] to hand-pick channels instead
    sel = t[0].flatten(1).var(dim=1).argsort(descending=True)[:6]

    fig, axes = plt.subplots(1, 7, figsize=(18, 3.4))

    # leftmost panel: EXACTLY the map shown in fig C for this level
    axes[0].imshow(summary_map(t), cmap=CMAP, interpolation="nearest")
    axes[0].set_title(f"Kanalmittel (wie Abb. C)\n{H}x{W}x{C}", fontsize=10)
    axes[0].set_xlabel(f"C3k2 (layer {LEVEL_LAYER})", fontsize=7)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for spine in axes[0].spines.values():
        spine.set_visible(False)

    # panels 2-7: individual channels of the same tensor
    for ax, ch in zip(axes[1:], sel):
        ax.imshow(norm01(t[0, ch].cpu().numpy()), cmap=CMAP, interpolation="nearest")
        ax.set_title(f"ch {int(ch)} von {C}", fontsize=9)
        ax.axis("off")

    fig.suptitle(f"Ein Level im Detail: das Kanalmittel (links) fasst {C} Kanaele "
                 "zusammen -- rechts 6 einzelne Kanaele derselben C3k2-Ausgabe",
                 fontsize=12, fontweight="bold")
    plt.savefig("figB_level_channels.png", dpi=200, bbox_inches="tight")
    plt.show()

print("Saved: figA_feature_pyramid.png, figC_pyramid_equal_size.png"
      + (", figB_level_channels.png" if USE_TRAINED else ""))