import os
import torch
import matplotlib.pyplot as plt


class SegmentationVisualizer:

    def __init__(
            self,
            model,
            loader,
            device,
            threshold,
            save_dir
    ):
        self.model = model
        self.loader = loader
        self.device = device
        self.threshold = threshold
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    def dice(self, pred, mask):
        intersection = (pred * mask).sum()

        return (
                (2 * intersection + 1e-8)
                /
                (pred.sum() + mask.sum() + 1e-8)
        ).item()

    def iou(self, pred, mask):
        intersection = (pred * mask).sum()

        union = (
                pred.sum()
                +
                mask.sum()
                -
                intersection
        )

        return (
                (intersection + 1e-8)
                /
                (union + 1e-8)
        ).item()

    def overlay(self, image, mask, prediction, title):
        image = image.squeeze().cpu().numpy()
        mask = mask.squeeze().cpu().numpy()
        prediction = prediction.squeeze().cpu().numpy()

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 4, 1)
        plt.imshow(image, cmap="gray")
        plt.title("MRI")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(prediction, cmap="gray")
        plt.title("Prediction")
        plt.axis("off")

        plt.subplot(1, 4, 4)

        plt.imshow(image, cmap="gray")

        plt.imshow(
            mask,
            alpha=0.4,
            cmap="Greens"
        )

        plt.imshow(
            prediction,
            alpha=0.4,
            cmap="Reds"
        )

        plt.title(title)
        plt.axis("off")

    def run(self):
        self.model.eval()

        with torch.no_grad():
            for idx, (images, masks) in enumerate(self.loader):
                images = images.to(self.device)
                masks = masks.to(self.device)

                output = self.model(images)

                probability = torch.sigmoid(output)

                prediction = (
                        probability > self.threshold
                ).float()

                dice = self.dice(
                    prediction,
                    masks
                )

                iou = self.iou(
                    prediction,
                    masks
                )

                title = (
                    f"Dice:{dice:.3f} "
                    f"IoU:{iou:.3f}"
                )

                self.overlay(
                    images[0],
                    masks[0],
                    prediction[0],
                    title
                )

                path = os.path.join(
                    self.save_dir,
                    f"sample_{idx + 1:04d}.png"
                )

                plt.savefig(
                    path,
                    bbox_inches="tight"
                )

                plt.close()

                print(
                    f"saved {idx + 1}: "
                    f"Dice={dice:.3f}"
                )
