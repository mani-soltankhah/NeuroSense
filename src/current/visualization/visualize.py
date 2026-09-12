import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from src.current.utils.metrics import dice_score, iou_score

matplotlib.use('TkAgg')


def tensor_to_numpy(tensor):
    tensor = tensor.squeeze()
    tensor = tensor.detach().cpu().numpy()
    return tensor


def normalize_image(image):
    """Normalize image to [0, 1] for better visualization contrast."""
    image = image - image.min()
    max_val = image.max()
    if max_val > 0:
        image = image / max_val
    return image


def categorize_dice(dice):
    if dice >= 0.88:
        return "Excellent"
    elif dice >= 0.80:
        return "Good"
    elif dice >= 0.70:
        return "Fair"
    elif dice >= 0.50:
        return "Poor"
    else:
        return "Very Poor"


def create_segmentation_figure(image, mask, prediction, dice, iou):
    image = tensor_to_numpy(image)
    mask = tensor_to_numpy(mask)
    prediction = tensor_to_numpy(prediction)

    image_norm = normalize_image(image)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_norm, cmap='gray')
    axes[0].set_title('Original MRI', fontsize=14, fontweight='bold')

    axes[1].imshow(image_norm, cmap='gray')
    masked_gt = np.ma.masked_where(mask == 0, mask)
    axes[1].imshow(masked_gt, cmap='Reds', alpha=0.6, interpolation='none')
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold', color='darkred')

    axes[2].imshow(image_norm, cmap='gray')
    masked_pred = np.ma.masked_where(prediction == 0, prediction)
    axes[2].imshow(masked_pred, cmap='Blues', alpha=0.6, interpolation='none')
    axes[2].set_title(f'''
    Prediction
    Dice: {dice:.3f}
    IoU: {iou:.3f}
    ''', fontsize=14, fontweight='bold', color='darkblue')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    return fig


def create_probability_figure(image, mask, probability, prediction, dice, iou):
    image = tensor_to_numpy(image)
    mask = tensor_to_numpy(mask)
    probability = tensor_to_numpy(probability)
    prediction = tensor_to_numpy(prediction)

    image_norm = normalize_image(image)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. MRI
    axes[0].imshow(image_norm, cmap='gray')
    axes[0].set_title('Original MRI')
    axes[0].axis('off')

    # 2. Ground Truth
    axes[1].imshow(image_norm, cmap='gray')

    masked_gt = np.ma.masked_where(mask == 0, mask)

    axes[1].imshow(
        masked_gt,
        cmap='Reds',
        alpha=0.6,
        interpolation='none'
    )

    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # 3. Probability Map
    im = axes[2].imshow(
        probability,
        cmap='jet',
        vmin=0,
        vmax=1
    )

    axes[2].set_title('Probability Map')
    axes[2].axis('off')

    fig.colorbar(
        im,
        ax=axes[2],
        fraction=0.046,
        pad=0.04
    )

    # 4. Binary Prediction
    axes[3].imshow(image_norm, cmap='gray')

    masked_pred = np.ma.masked_where(
        prediction == 0,
        prediction
    )

    axes[3].imshow(
        masked_pred,
        cmap='Blues',
        alpha=0.6,
        interpolation='none'
    )

    axes[3].set_title(
        f'Prediction\nDice: {dice:.3f} | IoU: {iou:.3f}'
    )

    axes[3].axis('off')

    plt.tight_layout()

    return fig


def show_segmentation(image, mask, prediction):
    dice = dice_score(prediction, mask)
    iou = iou_score(prediction, mask)
    fig = create_segmentation_figure(image, mask, prediction, dice, iou)
    plt.show()
    plt.close(fig)


def save_segmentation(fig, save_path, dpi=300, bbox_inches='tight'):
    fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)


def visualize_dataset(loader, predictor, output_dir, num_last_images=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_images = len(loader.dataset)
    start_idx = 0
    if num_last_images is not None and num_last_images > 0:
        start_idx = max(0, total_images - num_last_images)

    sample_index = 0
    current_idx = 0
    all_dice = 0
    all_iou = 0
    results = []
    counts = {"Excellent": 0, "Good": 0, "Fair": 0, "Poor": 0, "Very Poor": 0}
    for images, masks in loader:
        probabilities, predictions = predictor.predict(images)
        probabilities = probabilities.cpu()
        predictions = predictions.cpu()
        for i in range(images.size(0)):
            if current_idx == 411:
                probability = probabilities[0]

                print("\n===== PROBABILITY DEBUG =====")
                print("shape:", probability.shape)
                print("min:", probability.min().item())
                print("max:", probability.max().item())
                print("mean:", probability.mean().item())

                print("unique values:", torch.unique(probability)[:20])

                print("percentiles:")
                print("p01:", torch.quantile(probability, 0.01).item())
                print("p10:", torch.quantile(probability, 0.10).item())
                print("p25:", torch.quantile(probability, 0.25).item())
                print("p50:", torch.quantile(probability, 0.50).item())
                print("p75:", torch.quantile(probability, 0.75).item())
                print("p90:", torch.quantile(probability, 0.90).item())
                print("p99:", torch.quantile(probability, 0.99).item())
                print("=============================\n")
            if current_idx >= start_idx:
                dice = dice_score(predictions[i], masks[i])
                iou = iou_score(predictions[i], masks[i])
                all_dice += dice
                all_iou += iou
                fig = create_probability_figure(
                    images[i],
                    masks[i],
                    probabilities[i],
                    predictions[i],
                    dice,
                    iou
                )
                category = categorize_dice(dice)
                counts[category] += 1

                filename = f"sample_{sample_index:04d}.png"
                category_dir = output_dir / category.lower().replace(" ", "_")
                category_dir.mkdir(parents=True, exist_ok=True)
                filepath = category_dir / filename

                save_segmentation(fig, filepath)
                sample_index += 1
                print(f'''
                sample {sample_index:04d} done
                dice: {dice:.3f}
                iou: {iou:.3f}
                ''')
                results.append({
                    "index": current_idx,
                    "dice": dice.item(),
                    "iou": iou.item(),
                    "gt_pixels": masks[i].sum().item(),
                    "pred_pixels": predictions[i].sum().item()
                })

            current_idx += 1
    print(f"Average dice: {all_dice / sample_index:.3f}")
    print(f"Average iou: {all_iou / sample_index:.3f}")
    print(f"{sample_index} images have been saved to {output_dir}")
    print(counts)
    print(results)
