import matplotlib.pyplot as plt
import numpy as np


def visualise_samples(dataset, n=4):
    """
    Show n examples.
    The most important sanity check — do the images and labels look right?
    """
    fig, axes = plt.subplots(n, 2, figsize=(n * 6, n * 6))

    for i in range(n):
        indx = np.random.randint(0, len(dataset))
        ax_img = axes[i][0]
        ax_mask = axes[i][1]
        ax_img.imshow(
            dataset[indx][0].permute(1, 2, 0),
        )
        ax_mask.imshow(dataset[indx][1].permute(1, 2, 0))

    plt.suptitle("images and corresponding ground truth masks", fontsize=12)
    plt.tight_layout()
    plt.show()
