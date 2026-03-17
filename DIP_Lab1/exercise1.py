import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# Load the image
original = data.camera()

print('Image shape:', original.shape)
print('Data type:', original.dtype)
print('Min value:', original.min())
print('Max value:', original.max())

# apply Gaussian blur to simulate a blurry image
blurred = cv2.GaussianBlur(original, (5, 5), sigmaX=2)

print('Blurred image shape:', blurred.shape)

# define the Laplacian high-pass sharpening kernel
kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]], dtype=np.float32)
# apply the kernel to the blurred image
laplacian = cv2.filter2D(blurred.astype(np.float32), ddepth=-1, kernel=kernel)

# Add the Laplacian to the blurred image to get the sharpened image
sharpened_float = blurred.astype(np.float32) + laplacian

# Clip the values to the valid range [0, 255] and convert back to uint8
sharpened = np.clip(sharpened_float, 0, 255).astype(np.uint8)

print('Sharpened image range:', sharpened.min(), 'to', sharpened.max())

# compute a difference image to visualize the sharpening effect
diff = sharpened.astype(np.float32) - original.astype(np.float32)

diff_display = np.clip(diff + 128, 0, 255).astype(np.uint8)

print('Difference image range:', diff.min(), 'to', diff.max())

# Create a 2x2 figure, 10x10 inches
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('Image Sharpening: Laplacian Filter', fontsize=16, fontweight='bold')

# Panel 1 1L Original
axes[0, 0].imshow(original, cmap='gray')
axes[0, 0].set_title('1. Original Image', fontsize=12)
axes[0, 0].axis('off')

# Panel 2: Blurred
axes[0, 1].imshow(blurred, cmap='gray')
axes[0, 1].set_title('2. Blurred (Gaussian 5x5, sigma=2)', fontsize=12)
axes[0, 1].axis('off')

# Panel 3: Sharpened
axes[1, 0].imshow(sharpened, cmap='gray')
axes[1, 0].set_title('3. Sharpened (Laplacian Filter)', fontsize=12)
axes[1, 0].axis('off')

# Panel 4: Difference
axes[1, 1].imshow(diff_display, cmap='gray')
axes[1, 1].set_title('4. Difference (Sharpened - Original + 128)', fontsize=12)
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('execise1_output.png', dpi=150, bbox_inches='tight')
plt.show()
print('Figure saved as exercise1_output.png')