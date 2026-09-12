from pathlib import Path
import h5py
import numpy as np
import torch.nn.functional as F
import torch
from src.current.models.unet import UNet
from src.current.inference.predictor import Predictor

DATA_DIR = Path(r"D:\Portfolio\NeuroSense\Data\Raw")

mat_path = DATA_DIR / "3022.mat"

with h5py.File(mat_path, "r") as f:
    cjdata = f["cjdata"]

    label = int(np.array(cjdata["label"]).flatten()[0])

    pid_raw = cjdata["PID"][()]

    if isinstance(pid_raw, np.ndarray):
        pid = "".join(chr(int(x)) for x in pid_raw.flatten())
    elif isinstance(pid_raw, bytes):
        pid = pid_raw.decode("utf-8")
    else:
        pid = str(pid_raw)

    image = np.array(cjdata["image"])
    mask = np.array(cjdata["tumorMask"])

    print("========== FAILURE RAW SAMPLE ==========")

    print("file:", mat_path.name)
    print("label:", label)
    print("PID:", pid)

    print("\nIMAGE")
    print("shape:", image.shape)
    print("min:", image.min())
    print("max:", image.max())
    print("mean:", image.mean())
    print("std:", image.std())

    print("\nMASK")
    print("tumor pixels:", np.sum(mask > 0))
    print("mask unique:", np.unique(mask))

    ys, xs = np.where(mask > 0)

    print("\nMASK BBOX")
    print("x:", xs.min(), xs.max())
    print("y:", ys.min(), ys.max())

    print("========================================")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(
        in_channels=1,
        out_channels=1
    )

    checkpoint = torch.load(
        r"D:\Portfolio\NeuroSense\src\current\training\models\best_model.pth",
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    image_tensor = torch.tensor(image).float()

    # normalize مثل pipeline خودت
    image_tensor = (image_tensor - image_tensor.mean()) / image_tensor.std()

    image_tensor = F.interpolate(
        image_tensor.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    with torch.no_grad():
        output = model(image_tensor.to(DEVICE))

    prob = torch.sigmoid(output)[0, 0].cpu()

    print("\n===== MODEL PROBABILITY =====")
    print("min:", prob.min().item())
    print("max:", prob.max().item())
    print("mean:", prob.mean().item())

    ys, xs = torch.where(prob > 0.5)

    print("\nPROB BBOX")

    if len(xs) > 0:
        print(
            "x:",
            xs.min().item(),
            xs.max().item()
        )
        print(
            "y:",
            ys.min().item(),
            ys.max().item()
        )
    else:
        print("No prediction")

    # ys, xs = torch.where(prob_mask > 0.5)
    #
    # print("PROB BBOX")
    # print(
    #     xs.min().item(),
    #     xs.max().item(),
    #     ys.min().item(),
    #     ys.max().item()
    # )
    mask = torch.tensor(mask).float()

    mask_resized = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode="nearest"
    )[0, 0]

    ys, xs = torch.where(mask_resized > 0)

    print("RESIZED MASK BBOX")
    print(
        xs.min().item(),
        xs.max().item(),
        ys.min().item(),
        ys.max().item()
    )

    image_pt = torch.load(
        r"D:\Portfolio\NeuroSense\Data\Processed\test\images\3022.pt"
    )

    mask_pt = torch.load(
        r"D:\Portfolio\NeuroSense\Data\Processed\test\masks\3022.pt"
    )
    print("\nCOMPARE")

    pred = (prob > 0.5).float()

    print("prediction pixels:", pred.sum())

    print("mask pixels:", mask_resized.sum())

    intersection = (pred * mask_resized).sum()

    print("intersection:", intersection)

    dice = (2 * intersection) / (
            pred.sum() + mask_resized.sum() + 1e-8
    )

    print("manual dice:", dice.item())
    print("IMAGE PT")
    print(image_pt.shape)
    print(image_pt.min())
    print(image_pt.max())
    print(image_pt.mean())

    print("\nMASK PT")
    print(mask_pt.shape)
    print(mask_pt.sum())
    print(torch.unique(mask_pt))

    ys, xs = torch.where(mask_pt[0] > 0)

    print("PT MASK BBOX")
    print(
        xs.min().item(),
        xs.max().item(),
        ys.min().item(),
        ys.max().item()
    )
