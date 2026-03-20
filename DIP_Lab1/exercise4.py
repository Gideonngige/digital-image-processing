import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data


def unsharp_mask(image, sigma=2.0, k=1.5):
    # Step 1: Create blurred version
    blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigmaX=sigma)
    # Note: kernel size (0,0) tells OpenCV to compute size from sigma automatically
    # Step 2: Compute the unsharp mask (high-frequency content)
    mask = image.astype(np.float32) - blurred
    # Step 3: Add k times the mask back to the original
    sharpened = image.astype(np.float32) + k * mask
    # Step 4: Clip and return
    return np.clip(sharpened, 0, 255).astype(np.uint8)

original = data.camera()

k_values = [0.5, 1.0, 3.5, 4.0, 3.0]

results = [unsharp_mask(original, sigma=2.0, k=k) for k in k_values]

fig, axes = plt.subplots(1, 6, figsize=(20, 4))
fig.suptitle('Unsharp Masking: Effect of Sharpening Strength k', fontsize=13, fontweight='bold')

# Show original in first panel
axes[0].imshow(original, cmap='gray')
axes[0].set_title('Original', fontsize=10)
axes[0].axis('off')

# Show each sharpened version
for i, (img, k) in enumerate(zip(results, k_values)):
    axes[i+1].imshow(img, cmap='gray')
    axes[i+1].set_title(f'k = {k}', fontsize=10)
    axes[i+1].axis('off')

plt.tight_layout()
plt.savefig('exercise4_output.png', dpi=150, bbox_inches='tight')
plt.show()