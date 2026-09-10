import torch
import torch.nn as nn


class BrainTumorMultiTaskCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # 224 -> 112
            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 112 -> 56
            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 56 -> 28
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 28 -> 28
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(
                128,
                3
            )
        )


        # ====================================================
        # SEGMENTATION HEAD
        # ====================================================

        self.segmentation = nn.Sequential(

            # 28 -> 56
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),


            # 56 -> 112
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),


            # 112 -> 224
            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),


            # 224 -> 224
            nn.Conv2d(
                16,
                1,
                kernel_size=1
            )
        )


    def forward(self, x):

        # ====================================================
        # SHARED FEATURES
        # ====================================================

        features = self.encoder(x)


        # ====================================================
        # CLASSIFICATION
        # ====================================================

        class_logits = self.classifier(
            features
        )


        # ====================================================
        # SEGMENTATION
        # ====================================================

        mask_logits = self.segmentation(
            features
        )
        return class_logits, mask_logits


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):

        super().__init__()

        self.smooth = smooth


    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)


        probs = probs.view(-1)

        targets = targets.view(-1)


        intersection = (
            probs * targets
        ).sum()


        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            probs.sum()
            + targets.sum()
            + self.smooth
        )


        return 1.0 - dice