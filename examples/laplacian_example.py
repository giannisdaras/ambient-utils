"""
Example script for using LaplacianCropScore with Laplacian pyramid levels.
"""

import torch
import os
import ambient_utils
from ambient_utils.search import LaplacianCropScore
from configs import (
    DATASETS, SEARCH_MODES, LAPLACIAN_CONFIGS, TEST_IMAGES, 
    get_base_path, calculate_num_clusters, get_index_path
)


def run_laplacian_crop_score_example(dataset_key: str = "afhqv2", 
                                   test_image_key: str = "ood_dog",
                                   laplacian_config_key: str = "default",
                                   output_prefix: str = "laplacian_crop_score"):
    """
    Run LaplacianCropScore example with specified configuration.
    
    Args:
        dataset_key: Key for dataset configuration
        test_image_key: Key for test image path
        laplacian_config_key: Key for Laplacian configuration
        output_prefix: Prefix for output files
    """
    # Get configurations
    dataset_config = DATASETS[dataset_key]
    search_config = SEARCH_MODES["laplacian_crop_score"]
    laplacian_config = LAPLACIAN_CONFIGS[laplacian_config_key]
    base_path = get_base_path()
    
    # Calculate number of clusters
    num_clusters = calculate_num_clusters(dataset_config, search_config)
    
    # Get index path
    index_path = get_index_path(base_path, dataset_config, search_config, "laplacian_crop_score", laplacian_config)
    
    # Create output directory
    os.makedirs(os.path.join(base_path, f"datasets/faiss_resolution_{search_config.resolution}"), exist_ok=True)
    
    print(f"Running LaplacianCropScore example:")
    print(f"  Dataset: {dataset_config.description}")
    print(f"  Patch size: {search_config.patch_size}")
    print(f"  Resolution: {search_config.resolution}")
    print(f"  Laplacian levels: {laplacian_config.laplacian_levels}")
    print(f"  Laplacian level of interest: {laplacian_config.laplacian_level_of_interest}")
    print(f"  Index path: {index_path}")
    print(f"  Test image: {TEST_IMAGES[test_image_key]}")
    
    # Initialize LaplacianCropScore
    laplacian_crop_score = LaplacianCropScore(
        dataset_path=os.path.join(base_path, dataset_config.path),
        use_gpu=True,
        index_type='ivf',
        index_path=index_path,
        num_clusters=num_clusters,
        patch_size=search_config.patch_size,
        device=torch.device("cpu"),
        batch_size=search_config.batch_size,
        resolution=search_config.resolution,
        keep_ratio=search_config.keep_ratio,
        non_overlapping=search_config.non_overlapping,
        pca_dim=search_config.pca_dim,
        laplacian_levels=laplacian_config.laplacian_levels,
        laplacian_level_of_interest=laplacian_config.laplacian_level_of_interest
    )
    
    # Load and process test image
    test_image = ambient_utils.load_image(TEST_IMAGES[test_image_key], device=torch.device("cpu"))[:, :3] * 2 - 1
    sigma_t = torch.tensor(search_config.sigma_value).unsqueeze(0)
    
    # Add noise to test image
    noisy_test_image = test_image + sigma_t * torch.randn_like(test_image)
    ambient_utils.save_images(noisy_test_image, f"{output_prefix}_noisy_test_image.png")
    
    # Apply LaplacianCropScore to denoise the image
    denoised_image = laplacian_crop_score(
        x_t=noisy_test_image,
        sigma_t=sigma_t,
        temperature=search_config.temperature,
        n_neighbors=search_config.n_neighbors,
        nprobe=search_config.n_probe
    )
    
    # Save results
    ambient_utils.save_images(denoised_image, f"{output_prefix}_denoised.png")
    ambient_utils.save_images(test_image, f"{output_prefix}_original.png")
    
    # Calculate and print metrics
    mse = torch.mean((denoised_image - test_image) ** 2).item()
    print(f"Denoising completed:")
    print(f"  MSE between original and denoised: {mse:.6f}")
    print(f"  Results saved:")
    print(f"    Original image: {output_prefix}_original.png")
    print(f"    Noisy image: {output_prefix}_noisy_test_image.png")
    print(f"    Denoised image: {output_prefix}_denoised.png")


def compare_laplacian_levels_crop_score(dataset_key: str = "afhqv2", 
                                      test_image_key: str = "ood_dog",
                                      output_prefix: str = "laplacian_crop_score_comparison"):
    """
    Compare different Laplacian levels using LaplacianCropScore.
    
    Args:
        dataset_key: Key for dataset configuration
        test_image_key: Key for test image path
        output_prefix: Prefix for output files
    """
    print(f"Comparing different Laplacian levels using LaplacianCropScore...")
    
    # Get base configurations
    dataset_config = DATASETS[dataset_key]
    search_config = SEARCH_MODES["laplacian_crop_score"]
    base_path = get_base_path()
    num_clusters = calculate_num_clusters(dataset_config, search_config)
    
    # Load and prepare test image
    test_image = ambient_utils.load_image(TEST_IMAGES[test_image_key], device=torch.device("cpu"))[:, :3] * 2 - 1
    sigma_t = torch.tensor(search_config.sigma_value).unsqueeze(0)
    noisy_test_image = test_image + sigma_t * torch.randn_like(test_image)
    
    # Save noisy image once
    ambient_utils.save_images(noisy_test_image, f"{output_prefix}_noisy_test_image.png")
    
    # Process through different Laplacian levels
    for level_key, laplacian_config in LAPLACIAN_CONFIGS.items():
        print(f"Processing Laplacian level: {level_key} (level {laplacian_config.laplacian_level_of_interest})")
        
        # Get index path for this level
        index_path = get_index_path(base_path, dataset_config, search_config, "laplacian_crop_score", laplacian_config)
        
        # Initialize LaplacianCropScore for this level
        laplacian_crop_score = LaplacianCropScore(
            dataset_path=os.path.join(base_path, dataset_config.path),
            use_gpu=True,
            index_type='ivf',
            index_path=index_path,
            num_clusters=num_clusters,
            patch_size=search_config.patch_size,
            device=torch.device("cpu"),
            batch_size=search_config.batch_size,
            resolution=search_config.resolution,
            keep_ratio=search_config.keep_ratio,
            non_overlapping=search_config.non_overlapping,
            pca_dim=search_config.pca_dim,
            laplacian_levels=laplacian_config.laplacian_levels,
            laplacian_level_of_interest=laplacian_config.laplacian_level_of_interest
        )
        
        # Apply denoising
        denoised_image = laplacian_crop_score(
            x_t=noisy_test_image,
            sigma_t=sigma_t,
            temperature=search_config.temperature,
            n_neighbors=search_config.n_neighbors,
            nprobe=search_config.n_probe
        )
        
        # Save denoised result
        ambient_utils.save_images(
            denoised_image, 
            f"{output_prefix}_{level_key}_level_{laplacian_config.laplacian_level_of_interest}_denoised.png"
        )
        
        # Calculate MSE
        mse = torch.mean((denoised_image - test_image) ** 2).item()
        print(f"  MSE for {level_key}: {mse:.6f}")
    
    print(f"Laplacian level comparison completed:")
    print(f"  Noisy image: {output_prefix}_noisy_test_image.png")
    print(f"  Denoised images: {output_prefix}_*_denoised.png")


if __name__ == "__main__":
    # Example usage with default Laplacian configuration
    run_laplacian_crop_score_example(
        dataset_key="afhqv2", 
        test_image_key="ood_dog", 
        laplacian_config_key="default",
        output_prefix="laplacian_crop_score_default"
    )
    
    # Compare different Laplacian levels
    # compare_laplacian_levels_crop_score(
    #     dataset_key="afhqv2", 
    #     test_image_key="ood_dog",
    #     output_prefix="laplacian_crop_score_comparison"
    # )
    
    # Example usage with fine detail Laplacian configuration
    # run_laplacian_crop_score_example(
    #     dataset_key="afhqv2", 
    #     test_image_key="ood_dog", 
    #     laplacian_config_key="fine_detail",
    #     output_prefix="laplacian_crop_score_fine"
    # ) 