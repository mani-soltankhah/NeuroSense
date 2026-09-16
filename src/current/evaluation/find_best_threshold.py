import torch


def test_multiple_thresholds(model, test_loader, device):
    """تست چندین threshold و پیدا کردن بهترین"""

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print("=" * 70)
    print("تست Threshold های مختلف")
    print("=" * 70)

    results = {}

    for thresh in thresholds:
        total_dice = 0
        total_iou = 0
        count = 0

        model.eval()
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            with torch.no_grad():
                output = model(images)
                probability = torch.sigmoid(output)
                prediction = (probability > thresh).float()

            # محاسبه Dice
            intersection = (prediction * masks).sum()
            union = prediction.sum() + masks.sum()
            dice = (2. * intersection + 1e-8) / (union + 1e-8)

            # محاسبه IoU
            iou = intersection / (union - intersection + 1e-8)

            total_dice += dice.item()
            total_iou += iou.item()
            count += 1

        avg_dice = total_dice / count
        avg_iou = total_iou / count
        results[thresh] = {'dice': avg_dice, 'iou': avg_iou}

        print(f"Threshold {thresh:.1f} → Dice: {avg_dice:.3f} | IoU: {avg_iou:.3f}")

    # پیدا کردن بهترین threshold
    best_thresh = max(results, key=lambda x: results[x]['dice'])
    best_dice = results[best_thresh]['dice']
    best_iou = results[best_thresh]['iou']

    print("\n" + "=" * 70)
    print(f"✅ بهترین Threshold: {best_thresh}")
    print(f"   Dice: {best_dice:.3f}")
    print(f"   IoU: {best_iou:.3f}")
    print("=" * 70)

    return best_thresh, results
