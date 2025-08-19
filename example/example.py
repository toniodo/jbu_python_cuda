import cv2
import torch
import matplotlib.pyplot as plt

import jbu_cuda


sigma_s = 4
radius = 10
sigma_r = 0.2


# Load the original image
original_image = cv2.imread('example_aerial.png')

# Downsample the image
downsampled_image = cv2.resize(original_image, (original_image.shape[1] // 14, original_image.shape[0] // 14), interpolation=cv2.INTER_LANCZOS4)
downsampled_image_up = cv2.resize(downsampled_image, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_LINEAR)

guidance = torch.tensor(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)).unsqueeze(0).unsqueeze(0) / 255.0

result:torch.Tensor = jbu_cuda.upsample(guidance.float().to('cuda') , torch.tensor(downsampled_image_up).permute(2,0,1).unsqueeze(0).to('cuda') / 255.0, radius, sigma_s, sigma_r)
torch.cuda.empty_cache()

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

# Display each image in a subplot
axs[0].imshow(guidance[0, 0])
axs[0].set_title('Guidance Image')
axs[0].axis('off')  # Hide axes

axs[1].imshow(downsampled_image_up)
axs[1].set_title('Downsampled Image')
axs[1].axis('off')  # Hide axes

axs[2].imshow(result.cpu()[0].permute(1,2,0))
axs[2].set_title('Result Image (JBU)')
axs[2].axis('off')  # Hide axes


# Show the plot
plt.tight_layout()
plt.show()
