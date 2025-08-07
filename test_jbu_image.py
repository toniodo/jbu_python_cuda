import cv2
import torch
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load


sigma_s = 5
radius = 10
sigma_r = 0.2


# Load the original image
original_image = cv2.imread('/d/adomingu/Documents/Codes/ovdsat/data/simd/train/0003.jpg')

# Downsample the image
downsampled_image = cv2.resize(original_image, (original_image.shape[1] // 14, original_image.shape[0] // 14), interpolation=cv2.INTER_LANCZOS4)
downsampled_image_up = cv2.resize(downsampled_image, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_LINEAR)
# Upsample the image using JBU
load(
    name="jbu_filter",
    sources=["jbu_filter.cu"],
    verbose=True,
    with_cuda=True,
    is_python_module=False
)

def gkern(radius, std_dev):
    """
    Generate a Gaussian kernel.

    Args:
    - radius (int): The radius of the kernel.
    - std_dev (float): The standard deviation of the Gaussian distribution.

    Returns:
    - kernel (torch.Tensor): The Gaussian kernel.
    """

    # Calculate the kernel size
    kernel_size = 2 * radius + 1

    # Create a grid of coordinates
    x = torch.arange(kernel_size) - radius
    y = torch.arange(kernel_size) - radius
    x, y = torch.meshgrid(x, y)

    # Calculate the Gaussian distribution
    gaussian = torch.exp(-((x**2 + y**2) / (2 * std_dev**2)))

    # Normalize the kernel
    kernel = gaussian / torch.sum(gaussian)

    return kernel

gaussian_kernel = gkern(radius, sigma_s).to('cuda')
guidance = torch.tensor(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)).unsqueeze(0).unsqueeze(0) / 255.0

#sigma_r = torch.std(guidance, dim=(2,3)).item()

result:torch.Tensor = torch.ops.jbu_cuda.jbu_filter_global(guidance.float().to('cuda') , torch.tensor(downsampled_image_up).permute(2,0,1).unsqueeze(0).to('cuda') / 255.0, radius, gaussian_kernel, sigma_r)
torch.cuda.empty_cache()

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

# Display each image in a subplot
axs[0].imshow(guidance[0, 0])
axs[0].set_title('Guidance Image')
axs[0].axis('off')  # Hide axes

axs[1].imshow(downsampled_image_up)
axs[1].set_title('Input Image')
axs[1].axis('off')  # Hide axes

axs[2].imshow(result.cpu()[0].permute(1,2,0))
axs[2].set_title('Result Image')
axs[2].axis('off')  # Hide axes


# Show the plot
plt.tight_layout()
plt.show()
