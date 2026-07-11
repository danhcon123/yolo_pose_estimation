"""
Feature pyramid visualization from the REAL trained YOLO26 backbone.

Taps the yolo26n-pose model at every pyramid level (P1/2 ... P5/32) via forward
hooks and renders:

  figA_feature_pyramid.png -- input + one summary map per level, panels drawn at
                              PROPORTIONAL size so the reader literally sees the
                              resolution shrinking while abstraction grows
  figB_p3_channels.png     -- the 6 strongest channels at P3/8 (trained filters
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
CMAP        = "inferno"

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
# 3. FIGURE B -- strongest individual channels at P3/8 (trained only)
# ------------------------------------------------------------
if USE_TRAINED:
    # pick the pyramid level closest to P3 (index 2 if 5 levels were captured)
    idx = min(2, len(pyramid) - 1)
    t   = pyramid[idx][1]
    var_order = t[0].flatten(1).var(dim=1).argsort(descending=True)[:6]

    fig, axes = plt.subplots(1, 6, figsize=(16, 3.2))
    for ax, ch in zip(axes, var_order):
        ax.imshow(norm01(t[0, ch].cpu().numpy()), cmap=CMAP)
        ax.set_title(f"ch {int(ch)}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Trained filters at {level_names[idx]}: individual channels respond "
                 "to distinct structures of the object", fontsize=12, fontweight="bold")
    plt.savefig("figB_p3_channels.png", dpi=200, bbox_inches="tight")
    plt.show()

print("Saved: figA_feature_pyramid.png" + (", figB_p3_channels.png" if USE_TRAINED else ""))