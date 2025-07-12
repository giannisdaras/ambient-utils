"""
Configuration file for FAISS search examples.
Contains different setups for various datasets and search modes.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    path: str
    resolution: int
    size: int
    description: str


@dataclass
class SearchConfig:
    """Configuration for search parameters."""
    patch_size: int
    resolution: int
    keep_ratio: float
    n_probe: int
    n_neighbors: int
    sigma_value: float
    batch_size: int
    dataset_noise_level: float
    non_overlapping: bool
    temperature: float
    pca_dim: Optional[int] = None
    num_clusters: Optional[int] = None


@dataclass
class LaplacianConfig:
    """Configuration for Laplacian pyramid parameters."""
    laplacian_levels: int = 4
    laplacian_level_of_interest: int = 2


# Dataset configurations
DATASETS = {
    "cifar10": DatasetConfig(
        name="cifar10-32x32",
        path="datasets/cifar10-32x32",
        resolution=32,
        size=50_000,
        description="CIFAR-10 dataset with 32x32 resolution"
    ),
    "afhqv2": DatasetConfig(
        name="afhqv2",
        path="datasets/afhqv2-64x64",
        resolution=64,
        size=16_000,
        description="AFHQv2 dataset with 64x64 resolution"
    ),
    "imagenet_sd": DatasetConfig(
        name="imagenet-sd",
        path="datasets/img512-sd.zip",
        resolution=64,
        size=1_200_000,
        description="ImageNet SD dataset with 64x64 resolution"
    )
}

SEARCH_MODES = {
    "crop_score": SearchConfig(
        patch_size=64,
        resolution=64,
        keep_ratio=1.0,
        n_probe=512,
        n_neighbors=2048,
        sigma_value=4.0,
        batch_size=16_000,
        dataset_noise_level=0.0,
        non_overlapping=False,
        temperature=1.0,
        num_clusters=128
    ),
    "crop_score_pixel_level": SearchConfig(
        patch_size=8,
        resolution=64,
        keep_ratio=0.1,
        n_probe=32,
        n_neighbors=128,
        sigma_value=4.0,
        batch_size=16_000,
        dataset_noise_level=0.0,
        non_overlapping=False,
        temperature=1.0
    ),
    "crop_score_pixel_level_non_overlapping": SearchConfig(
        patch_size=64,
        resolution=64,
        keep_ratio=0.1,
        n_probe=512,
        n_neighbors=2048,
        sigma_value=4.0,
        batch_size=16_000,
        dataset_noise_level=0.0,
        non_overlapping=True,
        temperature=1.0
    ),
    "laplacian_crop_score": SearchConfig(
        patch_size=8,
        resolution=64,
        keep_ratio=0.1,
        n_probe=32,
        n_neighbors=128,
        sigma_value=4.0,
        batch_size=16_000,
        dataset_noise_level=0.0,
        non_overlapping=False,
        temperature=1.0
    )
}

# Laplacian configurations
LAPLACIAN_CONFIGS = {
    "default": LaplacianConfig(),
    "fine_detail": LaplacianConfig(laplacian_levels=4, laplacian_level_of_interest=0),
    "medium_detail": LaplacianConfig(laplacian_levels=4, laplacian_level_of_interest=1),
    "coarse_detail": LaplacianConfig(laplacian_levels=4, laplacian_level_of_interest=3)
}

# Test image paths
TEST_IMAGES = {
    "ood_dog": "/home1/07362/gdaras/freefusion/images/ood_dog.jpg",
    "ood_cat": "/home1/07362/gdaras/freefusion/images/ood_cat.png",
    "cifar_sample": "/scratch/07362/gdaras/datasets/cifar10-32x32/00000/img00000000.png",
    "afhq_sample": "/scratch/07362/gdaras/datasets/afhqv2-64x64/00000/img00000000.png"
}


def get_base_path():
    """Get the base path for storing indices and results."""
    return os.environ.get("SCRATCH", "/scratch/07362/gdaras/")


def calculate_num_clusters(dataset_config: DatasetConfig, search_config: SearchConfig) -> int:
    """Calculate the number of clusters based on dataset and search configuration."""
    if search_config.num_clusters is not None:
        return search_config.num_clusters
    
    if search_config.non_overlapping:
        return int((dataset_config.size * dataset_config.resolution ** 2) ** 0.5)
    else:
        return int((dataset_config.size * search_config.patch_size ** 2) ** 0.5)


def get_index_path(base_path: str, dataset_config: DatasetConfig, search_config: SearchConfig, 
                   mode: str, laplacian_config: Optional[LaplacianConfig] = None) -> str:
    """Generate the index path based on configuration."""
    dataset_name = dataset_config.name
    resolution = search_config.resolution
    patch_size = search_config.patch_size
    
    # Add laplacian prefix if using laplacian
    laplacian_str = ""
    if laplacian_config is not None:
        laplacian_str = f"laplacian_l{laplacian_config.laplacian_level_of_interest}_"
    
    # Add mode-specific prefixes
    if mode == "crop_score_pixel_level":
        extra_str = "_non_overlapping" if search_config.non_overlapping else ""
        return os.path.join(base_path, f"datasets/faiss_resolution_{resolution}/faiss_pixel_level_{extra_str}{laplacian_str}index_{dataset_name}-{dataset_config.resolution}x{dataset_config.resolution}_ivf_patch_size_{patch_size}.index")
    elif mode == "laplacian_crop_score":
        extra_str = "_non_overlapping" if search_config.non_overlapping else ""
        return os.path.join(base_path, f"datasets/faiss_resolution_{resolution}/faiss_laplacian_pixel_level_{extra_str}{laplacian_str}index_{dataset_name}-{dataset_config.resolution}x{dataset_config.resolution}_ivf_patch_size_{patch_size}.index")
    else:
        return os.path.join(base_path, f"datasets/faiss_resolution_{resolution}/{laplacian_str}faiss_index_{dataset_name}_ivf_{patch_size}.index") 