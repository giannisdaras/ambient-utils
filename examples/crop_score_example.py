"""
Example script for using CropScore with regular grid-based patches.
"""

import torch
import os
import ambient_utils
from ambient_utils.search import CropScore
from configs import (
    DATASETS, SEARCH_MODES, TEST_IMAGES, 
    get_base_path, calculate_num_clusters, get_index_path
)


def run_crop_score_example(dataset_key: str = "afhqv2", 
                          test_image_key: str = "ood_dog",
                          output_prefix: str = "crop_score"):
    """
    Run CropScore example with specified configuration.
    
    Args:
        dataset_key: Key for dataset configuration
        test_image_key: Key for test image path
        output_prefix: Prefix for output files
    """
    # Get configurations
    dataset_config = DATASETS[dataset_key]
    search_config = SEARCH_MODES["crop_score"]
    base_path = get_base_path()
    
    # Calculate number of clusters
    num_clusters = calculate_num_clusters(dataset_config, search_config)
    
    # Get index path
    index_path = get_index_path(base_path, dataset_config, search_config, "crop_score")
    
    # Create output directory
    os.makedirs(os.path.join(base_path, f"datasets/faiss_resolution_{search_config.resolution}"), exist_ok=True)
    
    print(f"Running CropScore example:")
    print(f"  Dataset: {dataset_config.description}")
    print(f"  Patch size: {search_config.patch_size}")
    print(f"  Resolution: {search_config.resolution}")
    print(f"  Index path: {index_path}")
    print(f"  Test image: {TEST_IMAGES[test_image_key]}")
    
    # Initialize CropScore
    crop_score = CropScore(
        dataset_path=os.path.join(base_path, dataset_config.path),
        use_gpu=True,
        index_type='ivf',
        index_path=index_path,
        num_clusters=num_clusters,
        patch_size=search_config.patch_size,
        device=torch.device("cpu"),
        batch_size=search_config.batch_size,
        resolution=search_config.resolution
    )
    
    # Load and process test image
    test_image = ambient_utils.load_image(TEST_IMAGES[test_image_key], device=torch.device("cpu"))[:, :3] * 2 - 1
    sigma_t = torch.tensor(search_config.sigma_value).unsqueeze(0)
    
    # Add noise to test image
    noisy_test_image = test_image + sigma_t * torch.randn_like(test_image)
    ambient_utils.save_images(noisy_test_image, f"{output_prefix}_noisy_test_image.png")
    
    # Denoise using CropScore
    denoised = crop_score(noisy_test_image, sigma_t=sigma_t)
    ambient_utils.save_images(denoised, f"{output_prefix}_denoised_patch_size_{search_config.patch_size}_sigma_{search_config.sigma_value}.png")
    
    print(f"Results saved:")
    print(f"  Noisy image: {output_prefix}_noisy_test_image.png")
    print(f"  Denoised image: {output_prefix}_denoised_patch_size_{search_config.patch_size}_sigma_{search_config.sigma_value}.png")


if __name__ == "__main__":
    # Example usage
    run_crop_score_example(dataset_key="afhqv2", test_image_key="ood_dog", output_prefix="crop_score_afhq")
    
    # You can also run with different datasets:
    # run_crop_score_example(dataset_key="cifar10", test_image_key="ood_cat", output_prefix="crop_score_cifar") 