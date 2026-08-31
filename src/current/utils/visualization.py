import matplotlib.pyplot as plt


def show_sample(image, mask):
    image = image.squeeze().numpy()
    mask = mask.squeeze().numpy()

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("MRI")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(image, cmap="gray")
    plt.imshow(mask, alpha=0.4, cmap="jet")
    plt.title("Overlay")
    plt.axis("off")

    plt.show()
