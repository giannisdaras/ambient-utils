from abc import ABC, abstractmethod
import torch
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional
from ambient_utils.dataset import ImageFolderDataset
import os
from torch.utils.data import DataLoader
import ambient_utils
import torch.nn.functional as F
from ambient_utils.loss import from_noise_pred_to_x0_pred_ve


def process_laplacian_level(image: torch.Tensor, levels: int, level_of_interest: int) -> torch.Tensor:
    """
    Process an image through Laplacian pyramid and return the level of interest.
    Args:
        image: Input tensor of shape [B, C, H, W]
        levels: Number of pyramid levels
        level_of_interest: Which level to use (0 = finest, levels-1 = coarsest)
    Returns:
        processed_image: Tensor of shape [B, C, H//2^level_of_interest, W//2^level_of_interest]
    """
    if level_of_interest == 0:
        return image
    pyramid = create_laplacian_pyramid(image, levels=levels)
    return pyramid[level_of_interest]


def create_laplacian_pyramid(image, levels=4, lowest_level_size=4):
    """
    Create a Laplacian pyramid from an input image.
    
    Args:
        image: Input tensor of shape (B, C, H, W)
        levels: Number of pyramid levels
    
    Returns:
        pyramid: List of Laplacian images at each level
    """
    pyramid = []
    B, C, H, W = image.shape
    
    # Calculate target sizes for each level
    # Level 0: original resolution (H, W)
    # Level levels-1: 2x2 resolution
    target_heights = []
    target_widths = []
    
    for level in range(levels):
        if level == 0:
            # Highest level: original resolution
            target_heights.append(H)
            target_widths.append(W)
        elif level == levels - 1:
            # Lowest level: 2x2
            target_heights.append(lowest_level_size)
            target_widths.append(lowest_level_size)
        else:
            # Intermediate levels: linearly interpolated
            progress = level / (levels - 1)
            target_heights.append(int(2 + (H - 2) * (1 - progress)))
            target_widths.append(int(2 + (W - 2) * (1 - progress)))
    
    current = image
    
    for level in range(levels):
        if level == 0:
            # For the highest level, we need the difference between original and downsampled
            target_h, target_w = target_heights[level + 1], target_widths[level + 1]
            downsampled = torch.nn.functional.interpolate(current, size=(target_h, target_w), 
                                                        mode='bilinear', align_corners=False)
            upsampled = torch.nn.functional.interpolate(downsampled, size=(H, W), 
                                                      mode='bilinear', align_corners=False)
            laplacian = current - upsampled
            pyramid.append(laplacian)
            current = downsampled
        elif level < levels - 1:
            # For intermediate levels
            target_h, target_w = target_heights[level + 1], target_widths[level + 1]
            downsampled = torch.nn.functional.interpolate(current, size=(target_h, target_w), 
                                                        mode='bilinear', align_corners=False)
            upsampled = torch.nn.functional.interpolate(downsampled, size=current.shape[-2:], 
                                                      mode='bilinear', align_corners=False)
            laplacian = current - upsampled
            pyramid.append(laplacian)
            current = downsampled
        else:
            # For the lowest level, just add the residual
            pyramid.append(current)
    
    return pyramid[::-1]

def reconstruct_from_laplacian_pyramid(pyramid):
    """
    Reconstruct image from Laplacian pyramid.
    
    Args:
        pyramid: List of Laplacian images
    
    Returns:
        reconstructed: Reconstructed image
    """
    pyramid = pyramid[::-1]
    reconstructed = pyramid[-1]  # Start with the lowest frequency component
    
    # Reconstruct by adding Laplacian components back
    for i in range(len(pyramid) - 2, -1, -1):
        # Upsample current reconstruction to match the next level's size
        target_size = pyramid[i].shape[-2:]
        upsampled = torch.nn.functional.interpolate(reconstructed, size=target_size, 
                                                   mode='bilinear', align_corners=False)
        # Add Laplacian component
        reconstructed = upsampled + pyramid[i]
    
    return reconstructed


def split_image_into_patches(image: torch.Tensor, patch_size: int, mode: str = 'regular', keep_ratio: float = 1.0, non_overlapping: bool = False) -> torch.Tensor:
    """
    Split an image into patches.
    
    Args:
        image: torch.Tensor of shape [B, C, H, W]
        patch_size: Size of each patch
        mode: 'regular' for grid-based patches, 'pixel_level' for pixel-centered patches
        keep_ratio: Ratio of patches to keep (for pixel-level mode)
        non_overlapping: Whether to use non-overlapping patches (for pixel-level mode)
    Returns:
        patches: torch.Tensor of shape [B, C, H // patch_size, W // patch_size, patch_size, patch_size] for regular mode
                or [B, C, H, W, patch_size, patch_size] for pixel-level mode
    """
    if mode == 'regular':
        patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        return patches
    
    elif mode == 'pixel_level':
        # Make patch_size odd if it's even
        if patch_size % 2 == 0:
            patch_size += 1
        
        half_patch = patch_size // 2
        
        # Pad the image to handle edge cases
        padded_image = F.pad(image, (half_patch, half_patch, half_patch, half_patch), mode='reflect')
        
        # Use unfold to extract all patches at once
        patches_h = padded_image.unfold(2, patch_size, 1)  # [B, C, H, W, patch_size]

        if non_overlapping:
            patches_h = patches_h[:, :, ::patch_size]

        dropped_h = torch.rand(patches_h.shape[2]) < keep_ratio 
        patches_h = patches_h[:, :, dropped_h, :]

        # Then unfold in width dimension
        patches = patches_h.unfold(3, patch_size, 1)  # [B, C, H, W, patch_size, patch_size]
        if non_overlapping:
            patches = patches[:, :, :, ::patch_size]

        dropped_w = torch.rand(patches.shape[3]) < keep_ratio 
        patches = patches[:, :, :, dropped_w]
        
        # Rearrange dimensions to match expected output format
        patches = patches.permute(0, 1, 2, 3, 4, 5)  # [B, C, H, W, patch_size, patch_size]
        return patches
    
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def assemble_patches(patches: torch.Tensor) -> torch.Tensor:
    """
    Assemble patches into an image.
    
    Args:
        patches: torch.Tensor of shape [B, num_patches_per_row, num_patches_per_col, 3, patch_size, patch_size]
    Returns: 
        image: torch.Tensor of shape [B, C, H, W]
    """
    B, num_patches_per_row, num_patches_per_col, C, patch_size, _ = patches.shape
    image = patches.permute(0, 3, 1, 4, 2, 5)
    image = image.reshape(B, C, num_patches_per_row * patch_size, num_patches_per_col * patch_size)
    return image


class BaseFAISS(ABC):
    """
    Base class for FAISS-based approximate nearest neighbors search.
    
    This abstract base class provides common functionality for both
    disk-based and in-memory FAISS implementations.
    """
    
    def __init__(self, use_gpu: bool = True, 
                 index_type: str = 'ivf', device: torch.device = None,
                 dtype: torch.dtype = torch.float32,
                 num_clusters: int = 4096):
        """
        Initialize base FAISS neighbors search.
        
        Args:
            use_gpu: Whether to use GPU acceleration for FAISS
            device: torch device for tensor operations
            dtype: data type for tensors
            index_type: Type of FAISS index ('ivf', 'flat', 'hnsw')
        """
        self.use_gpu = use_gpu
        self.index_type = index_type
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.index = None
        self.num_clusters = num_clusters


    def _create_index(self, dataset_dim: int, pca_dim: int = None):
        """Create FAISS index based on the specified type."""
        if self.index_type == 'flat':
            self.index = faiss.IndexFlatL2(dataset_dim)
        elif self.index_type == 'ivf':
            nlist = self.num_clusters
            quantizer = faiss.IndexFlatL2(dataset_dim)
            self.index = faiss.IndexIVFFlat(quantizer, dataset_dim, nlist)
        elif self.index_type == 'hnsw':
            self.index = faiss.IndexHNSWFlat(dataset_dim, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        if pca_dim is not None:
            pca_matrix = faiss.PCAMatrix(dataset_dim, pca_dim, 0, True)
            pca_index = faiss.IndexPreTransform(pca_matrix, self.index)
            # make sure the distances are computed wrt the true images
            self.index = faiss.IndexRefineFlat(pca_index)
        
 
    def _check_gpu_availability(self):
        """Check FAISS GPU availability and provide diagnostics."""
        try:
            num_gpus = faiss.get_num_gpus()
            print(f"FAISS reports {num_gpus} GPU (s) available")
            
            if num_gpus == 0:
                print("No FAISS GPU support detected. This could be due to:")
                print("  - FAISS compiled without GPU support")
                print("  - No CUDA installation")
                print("  - No compatible GPU drivers")
                return False
                
            # Try to create GPU resources to test availability
            try:
                res = faiss.StandardGpuResources()
                print("FAISS GPU resources created successfully")
                return True
            except Exception as e:
                print(f"Failed to create FAISS GPU resources: {e}")
                return False
                
        except Exception as e:
            print(f"Error checking FAISS GPU availability: {e}")
            return False

    def _move_to_gpu(self):
        """Move FAISS index to GPU if requested and available."""
        if not self.use_gpu:
            return
            
        if not self._check_gpu_availability():
            print("Falling back to CPU")
            self.use_gpu = False
            return
            
        try:
            # Check if index is already on GPU
            if hasattr(self.index, 'getDevice') and self.index.getDevice() >= 0:
                return
                
            # Create GPU resources and move to GPU
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
            
            # Verify the move was successful
            if hasattr(gpu_index, 'getDevice') and gpu_index.getDevice() >= 0:
                self.index = gpu_index
            else:
                print("GPU move failed, falling back to CPU")
                self.use_gpu = False
                
        except Exception as e:
            print(f"Error moving FAISS index to GPU: {e}")
            print("Falling back to CPU")
            self.use_gpu = False

    def _search_neighbors(self, x_t_np: np.ndarray, n_neighbors: int = 100, nprobe: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Search for nearest neighbors using FAISS.
        
        Args:
            x_t_np: numpy array of shape [B, D] to search for
            n_neighbors: number of nearest neighbors to retrieve
        Returns:
            distances: tensor of shape [B, n_neighbors]
            indices: tensor of shape [B, n_neighbors]
        """
        self.index.nprobe = nprobe
        # faiss.omp_set_num_threads(os.cpu_count())

        distances, indices = self.index.search(x_t_np, n_neighbors)
        distances = torch.from_numpy(distances).to(self.device, dtype=self.dtype)
        indices = torch.from_numpy(indices).to(self.device)
        return distances, indices

    
    def find_nearest_matches(self, generated_samples: torch.Tensor, n_neighbors: int = 100, nprobe: int = 10) -> Tuple[list, list, list]:
        """
        Find the nearest match in the dataset for each generated sample using FAISS.
        
        Args:
            generated_samples: tensor of shape [B, C, H, W] - the generated images
            
        Returns:
            nearest_indices: list of indices of nearest matches in the dataset
            distances: list of L2 distances to the nearest matches
            nearest_samples: list of the actual nearest sample images
        """
        B, C, H, W = generated_samples.shape
        
        # Flatten generated samples
        generated_flat = generated_samples.reshape(B, -1)  # [B, D]
        
        # Convert to numpy for FAISS search
        generated_np = generated_flat.cpu().numpy().astype(np.float32)
        
        # Search for nearest neighbors
        self.index.nprobe = nprobe
        try:
            param = faiss.GpuParameterSpace()
            param.set_index_parameter(self.index, "nprobe", nprobe)
        except Exception as e:
            print(f"Error setting nprobe: {e}. Probably working on CPU.")


        distances, indices = self.index.search(generated_np, n_neighbors)  # [B, n_neighbors]
        # Convert to tensors
        distances = torch.from_numpy(distances)
        indices = torch.from_numpy(indices)
        
        distances = distances.to(self.device, dtype=self.dtype)  # [B, n_neighbors]
        
        # Get the actual nearest samples (implemented by subclasses)
        nearest_samples = self._get_nearest_samples(indices, C, H, W)

        return indices.cpu().tolist(), distances.cpu().tolist(), nearest_samples
    
    @abstractmethod
    def _get_neighbor_samples(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Get neighbor samples for the given indices.
        
        Args:
            indices: tensor of shape [B, n_neighbors]
            
        Returns:
            batch_neighbors: tensor of shape [B, n_neighbors, D]
        """
        pass
    
    @abstractmethod
    def _get_nearest_samples(self, indices: torch.Tensor, C: int, H: int, W: int) -> list:
        """
        Get nearest samples for the given indices.
        
        Args:
            indices: tensor of shape [B, n_neighbors]
            C, H, W: image dimensions
            
        Returns:
            nearest_samples: list of lists of sample tensors, shape [B][n_neighbors]
        """
        pass



class FAISSIndex(BaseFAISS):
    """
    FAISS-based approximate nearest neighbors search.
    
    This class builds FAISS indices from dataset files without loading
    the entire dataset into memory, making it suitable for very large datasets.
    Supports both regular grid-based patches and pixel-level patches.
    """
    
    def __init__(self, dataset_path: str, 
                 use_gpu: bool = True, index_type: str = 'ivf',
                 index_path: Optional[str] = None, device: torch.device = None,
                 dtype: torch.dtype = torch.float32, batch_size: int = 16,
                 num_clusters: int = 4096, patch_size: int = None, pca_dim: int = None, 
                 noise_level: float = 0.0, resolution: int = 64, 
                 patch_mode: str = 'regular', keep_ratio: float = 1.0, non_overlapping: bool = False):
        """
        Initialize FAISS approximate nearest neighbors search.
        
        Args:
            dataset_path: Path to the dataset (directory or zip file)
            use_gpu: Whether to use GPU acceleration for FAISS
            index_type: Type of FAISS index ('ivf', 'flat', 'hnsw')
            index_path: Path to save/load the FAISS index (optional)
            device: torch device for tensor operations
            dtype: data type for tensors
            batch_size: Batch size for processing dataset during index building
            patch_mode: 'regular' for grid-based patches, 'pixel_level' for pixel-centered patches
            keep_ratio: Ratio of patches to keep (for pixel-level mode)
            non_overlapping: Whether to use non-overlapping patches (for pixel-level mode)
        """
        super().__init__(use_gpu=use_gpu, index_type=index_type, device=device, dtype=dtype, num_clusters=num_clusters)
        self.resolution = resolution
        self.noise_level = noise_level
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.pca_dim = pca_dim
        self.patch_mode = patch_mode
        self.keep_ratio = keep_ratio
        self.non_overlapping = non_overlapping
        
        if patch_size is None:
            self.patch_size = self.get_dataset().resolution
        else:
            self.patch_size = patch_size
        
        # Default index path if not provided
        if index_path is None:
            dataset_name = Path(dataset_path).stem
            base_path = os.environ.get("SCRATCH", "/scratch/07362/gdaras/")
            mode_str = f"_{patch_mode}" if patch_mode == 'pixel_level' else ""
            index_path = f"{base_path}/faiss_index_{dataset_name}_{index_type}_{self.patch_size}{mode_str}.index"
            print(f"Will be saving index to {index_path}")
        else:
            print(f"Will be saving index to {index_path}")
        self.index_path = index_path
        
        # Dataset info
        self.dataset_dim = None
        self.dataset_size = None
        self.dataset = None  # Will be initialized lazily when first needed

        # Build or load the index
        self._build_or_load_index()
    
    def _get_dataset(self):
        """Lazy initialization of dataset object."""
        if self.dataset is None:
            self.dataset = ImageFolderDataset(self.dataset_path, use_labels=False, cache=False, only_positive=False, resolution=self.resolution)
        return self.dataset
    
    def _get_dataset_info(self):
        """Get dataset dimension and size without loading all data."""
        # Create a temporary dataset object to get info
        temp_dataset = ImageFolderDataset(self.dataset_path, use_labels=False, cache=False, resolution=self.resolution)
        num_patches_per_image = (temp_dataset.resolution // self.patch_size)**2
        self.dataset_size = len(temp_dataset) * num_patches_per_image
        self.dataset_dim = (temp_dataset.resolution**2) * temp_dataset.num_channels
    
    def _build_index_from_disk(self):
        """Build FAISS index by processing dataset in batches."""
        
        # Create dataset object
        dataset = ImageFolderDataset(self.dataset_path, use_labels=False, cache=False, only_positive=False, resolution=self.resolution)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        # Initialize index based on patch mode
        if self.patch_mode == 'regular':
            num_patches_per_image = (dataset.resolution // self.patch_size)**2
            patch_dim = self.dataset_dim // num_patches_per_image
        else:  # pixel_level
            real_patch_size = self.patch_size + 1 if (self.patch_size % 2 == 0) else self.patch_size
            patch_dim = real_patch_size ** 2 * 3
            
        self._create_index(patch_dim, self.pca_dim)
        
        # Process dataset in batches
        first_batch = True
        for batch in tqdm(dataloader, desc="Indexing dataset"):   
            batch['image'] = batch['image'] + self.noise_level * torch.randn_like(batch['image'])

            if self.patch_mode == 'regular':
                batch_patches = split_image_into_patches(batch['image'], self.patch_size, mode='regular')
                batch_vectors = batch_patches.reshape(batch_patches.shape[0] * batch_patches.shape[1] * batch_patches.shape[2], -1)
            else:  # pixel_level
                batch['image'] = F.interpolate(batch['image'].to(self.dtype), size=(self.resolution, self.resolution), mode='bilinear', align_corners=False)
                batch_patches = split_image_into_patches(batch['image'], self.patch_size, mode='pixel_level', keep_ratio=self.keep_ratio, non_overlapping=self.non_overlapping)
                batch_patches = batch_patches.permute(0, 2, 3, 1, 4, 5)  # B, H, W, C, patch_size, patch_size
                batch_vectors = batch_patches.reshape(batch_patches.shape[0] * batch_patches.shape[1] * batch_patches.shape[2], -1)

            if self.index_type == 'ivf' and first_batch:
                # Train the index with first batch
                print("Training FAISS index...")
                if self.patch_mode == 'pixel_level':
                    self._move_to_gpu()
                self.index.train(batch_vectors)
                first_batch = False
            
            # Add vectors to index
            self.index.add(batch_vectors)
            
        # Move to GPU if requested
        self._move_to_gpu()
        
        # Save index to disk
        if self.index_path:
            print(f"Saving FAISS index to: {self.index_path}")
            # Check if index is on GPU and move to CPU before saving
            if hasattr(self.index, 'getDevice') and self.index.getDevice() >= 0:
                index_cpu = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(index_cpu, self.index_path)
            else:
                faiss.write_index(self.index, self.index_path)
    
    def _load_index_from_disk(self):
        """Load existing FAISS index."""
        print(f"Loading FAISS index from: {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        self._move_to_gpu()
    
    def _build_or_load_index(self):
        """Build new index or load existing one."""
        # Get dataset info first
        self._get_dataset_info()
        
        # Check if index already exists
        if os.path.exists(self.index_path):
            try:
                self._load_index_from_disk()
                print("Successfully loaded existing FAISS index")
                return
            except Exception as e:
                print(f"Failed to load existing index: {e}")
                print("Will build new index...")
        
        # Build new index
        self._build_index_from_disk()

    
    def _get_neighbor_samples(self, indices: torch.Tensor) -> torch.Tensor:
        """Get neighbor samples."""
        B = indices.shape[0]
        batch_neighbors = []
        
        for b in range(B):
            # Get the nearest neighbors for this batch item
            neighbor_indices = indices[b]  # [n_neighbors]
            neighbor_samples = []
            
            for idx in neighbor_indices:
                # Load sample from disk
                dataset = self._get_dataset()
                sample = dataset[idx]['image'][:3]
                sample_flat = sample.reshape(-1)  # [D]
                neighbor_samples.append(sample_flat)
            
            neighbor_samples = torch.stack(neighbor_samples)  # [n_neighbors, D]
            batch_neighbors.append(neighbor_samples)
        
        return torch.stack(batch_neighbors)  # [B, n_neighbors, D]
    
    def _get_nearest_samples(self, indices: torch.Tensor, C: int, H: int, W: int) -> list:
        """Get nearest samples."""
        B, n_neighbors = indices.shape
        
        if self.patch_mode == 'pixel_level':
            # For pixel-level mode, use direct reconstruction
            nearest_samples = np.zeros((B, n_neighbors, 3, H, W))
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            cpu_index.make_direct_map()
            nearest_samples = cpu_index.reconstruct_batch(indices.reshape(-1)).reshape(B, n_neighbors, 3, H, W)
            return nearest_samples
        else:
            # For regular mode, use direct reconstruction from FAISS index
            # The index contains the actual patch vectors, so we can reconstruct them directly
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            cpu_index.make_direct_map()
            
            # Reconstruct all patches at once
            patch_vectors = cpu_index.reconstruct_batch(indices.reshape(-1))  # [B * n_neighbors, patch_dim]
            
            # Reshape to patch format
            patch_dim = patch_vectors.shape[1]
            patch_size = int(np.sqrt(patch_dim // 3))  # Assuming RGB channels
            nearest_samples = patch_vectors.reshape(B, n_neighbors, 3, patch_size, patch_size)
            
            return nearest_samples

class FAISSLaplacianIndex(FAISSIndex):
    """
    FAISS-based approximate nearest neighbors search for Laplacian pyramid levels.
    
    This class builds FAISS indices from dataset files by storing a particular level
    of the Laplacian pyramid instead of the original images.
    """
    
    def __init__(self, dataset_path: str, 
                 use_gpu: bool = True, index_type: str = 'ivf',
                 index_path: Optional[str] = None, device: torch.device = None,
                 dtype: torch.dtype = torch.float32, batch_size: int = 16,
                 num_clusters: int = 4096, patch_size: int = None, pca_dim: int = None, 
                 noise_level: float = 0.0, resolution: int = 64, 
                 patch_mode: str = 'regular', keep_ratio: float = 1.0, non_overlapping: bool = False,
                 laplacian_levels: int = 4, laplacian_level_of_interest: int = 2):
        """
        Initialize FAISS Laplacian approximate nearest neighbors search.
        
        Args:
            dataset_path: Path to the dataset (directory or zip file)
            use_gpu: Whether to use GPU acceleration for FAISS
            index_type: Type of FAISS index ('ivf', 'flat', 'hnsw')
            index_path: Path to save/load the FAISS index (optional)
            device: torch device for tensor operations
            dtype: data type for tensors
            batch_size: Batch size for processing dataset during index building
            patch_mode: 'regular' for grid-based patches, 'pixel_level' for pixel-centered patches
            keep_ratio: Ratio of patches to keep (for pixel-level mode)
            non_overlapping: Whether to use non-overlapping patches (for pixel-level mode)
            laplacian_levels: Number of pyramid levels
            laplacian_level_of_interest: Which level to use (0 = finest, levels-1 = coarsest)
        """
        self.laplacian_levels = laplacian_levels
        self.laplacian_level_of_interest = laplacian_level_of_interest
        
            
        super().__init__(dataset_path=dataset_path, 
                        use_gpu=use_gpu, 
                        index_type=index_type,
                        index_path=index_path, 
                        device=device, 
                        dtype=dtype, 
                        batch_size=batch_size,
                        num_clusters=num_clusters, 
                        patch_size=patch_size, 
                        pca_dim=pca_dim, 
                        noise_level=noise_level, 
                        resolution=resolution, 
                        patch_mode=patch_mode, 
                        keep_ratio=keep_ratio, 
                        non_overlapping=non_overlapping)
    
    def _build_index_from_disk(self):
        """Build FAISS index by processing dataset in batches with Laplacian pyramid."""
        
        # Create dataset object
        dataset = ImageFolderDataset(self.dataset_path, use_labels=False, cache=False, only_positive=False, resolution=self.resolution)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        real_patch_size = self.patch_size + 1 if (self.patch_size % 2 == 0) else self.patch_size
        patch_dim = real_patch_size ** 2 * 3

            
        self._create_index(patch_dim, self.pca_dim)
        
        # Process dataset in batches
        first_batch = True
        for batch in tqdm(dataloader, desc="Indexing dataset with Laplacian pyramid"):   
            batch['image'] = batch['image'] + self.noise_level * torch.randn_like(batch['image'])
            
            # Process images through Laplacian pyramid
            laplacian_images = process_laplacian_level(batch['image'], 
                                                      self.laplacian_levels, 
                                                      self.laplacian_level_of_interest)

            batch_patches = split_image_into_patches(laplacian_images, self.patch_size, mode='pixel_level', keep_ratio=self.keep_ratio, non_overlapping=self.non_overlapping)
            batch_patches = batch_patches.permute(0, 2, 3, 1, 4, 5)  # B, H, W, C, patch_size, patch_size
            batch_vectors = batch_patches.reshape(batch_patches.shape[0] * batch_patches.shape[1] * batch_patches.shape[2], -1)

            if self.index_type == 'ivf' and first_batch:
                print("Training FAISS index...")
                self._move_to_gpu()
                self.index.train(batch_vectors)
                first_batch = False
            
            # Add vectors to index
            self.index.add(batch_vectors)
            
        # Move to GPU if requested
        self._move_to_gpu()
        
        # Save index to disk
        if self.index_path:
            print(f"Saving FAISS Laplacian index to: {self.index_path}")
            # Check if index is on GPU and move to CPU before saving
            if hasattr(self.index, 'getDevice') and self.index.getDevice() >= 0:
                index_cpu = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(index_cpu, self.index_path)
            else:
                faiss.write_index(self.index, self.index_path)


class CropScore():
    """
    Compute a per-crop score.
    """
    
    def __init__(self, dataset_path: str, 
                 use_gpu: bool = True, index_type: str = 'ivf',
                 index_path: Optional[str] = None, device: torch.device = None,
                 dtype: torch.dtype = torch.float32, batch_size: int = 16,
                 num_clusters: int = 4096, patch_size: int = None, pca_dim: int = None, resolution: int = 64):
        """
        Initialize CropScore.
        
        Args:
            dataset_path: Path to the dataset (directory or zip file)
            use_gpu: Whether to use GPU acceleration for FAISS
            index_type: Type of FAISS index ('ivf', 'flat', 'hnsw')
            index_path: Path to save/load the FAISS index (optional)
            device: torch device for tensor operations
            dtype: data type for tensors
            batch_size: Batch size for processing dataset during index building
        """
        self.patch_size = patch_size
        self.faiss_disk = FAISSIndex(dataset_path=dataset_path, 
                                         index_path=index_path, 
                                         patch_size=patch_size, 
                                         use_gpu=use_gpu, 
                                         index_type=index_type, 
                                         device=device, 
                                         dtype=dtype, 
                                         batch_size=batch_size, 
                                         num_clusters=num_clusters, 
                                         pca_dim=pca_dim,
                                         resolution=resolution)
    

    def find_neighbors(self, image: torch.Tensor, n_neighbors: int = 10, nprobe: int = 20000) -> torch.Tensor:
        """
        Find patch-level neighbors.
        
        Args:
            image: torch.Tensor of shape [B, C, H, W]
        Returns:
            batched_nearest_samples: torch.Tensor of shape [B, num_patches_per_row, num_patches_per_col, n_neighbors, 3, patch_size, patch_size]
            batched_distances: torch.Tensor of shape [B, num_patches_per_row, num_patches_per_col, n_neighbors]
        """
        patches = split_image_into_patches(image, self.patch_size, mode='regular')
        num_patches_per_row = image.shape[2] // self.patch_size
        num_patches_per_col = image.shape[3] // self.patch_size
        batched_patches = patches.reshape(patches.shape[0] * patches.shape[1] * patches.shape[2], patches.shape[3], patches.shape[4], patches.shape[5])
        _, batched_distances, batched_nearest_samples = self.faiss_disk.find_nearest_matches(batched_patches, n_neighbors=n_neighbors, nprobe=nprobe)        
        batched_nearest_samples = torch.from_numpy(np.stack(batched_nearest_samples)) 
        batched_nearest_samples = batched_nearest_samples.to(self.faiss_disk.device)
        batched_nearest_samples = batched_nearest_samples.reshape(image.shape[0], num_patches_per_row, num_patches_per_col, *batched_nearest_samples.shape[1:])
        batched_distances = torch.from_numpy(np.stack(batched_distances))

        batched_distances = batched_distances.reshape(image.shape[0], num_patches_per_row, num_patches_per_col, *batched_distances.shape[1:])
        return batched_nearest_samples, batched_distances


    def __call__(self, x_t: torch.Tensor, 
                sigma_t: torch.Tensor, temperature: float = 1.0, 
                n_neighbors: int = 10, nprobe: int = 10) -> torch.Tensor:
        """
        Compute score from nearest neighbors using softmax weighting.
        
        Args:
            x_t: input tensor [B, C, H, W]
            sigma_t: noise level [B]
            
        Returns:
            score: computed score tensor [B, C, H, W]
        """
        x_t_index_interpolated = F.interpolate(x_t, size=(self.faiss_disk.resolution, self.faiss_disk.resolution), mode='bilinear', align_corners=False)
        samples, distances = self.find_neighbors(image=x_t_index_interpolated, n_neighbors=n_neighbors, nprobe=nprobe) # samples: [B, num_patches_per_row, num_patches_per_col, n_neighbors, 3, patch_size, patch_size]
        batch_size, num_patches_per_row, num_patches_per_col, n_neighbors, _, patch_size, _ = samples.shape
        sigma2 = (sigma_t ** 2).to(x_t.device)
        D = x_t[0].numel()
        
        # Compute log-likelihoods for the subset

        log_norm = -0.5 * D * torch.log(2 * torch.pi * sigma2)
        log_likelihoods = log_norm - distances.to(x_t.device) / (2 * sigma2)
        
        # Compute softmax weights over the subset
        weights = F.softmax(log_likelihoods / temperature, dim=-1)  # (B, num_patches_per_row, num_patches_per_col, n_neighbors)
        weights = weights.to(x_t.device)
        
        # Compute weighted differences
        samples_perm = samples.permute(0, 3, 1, 2, 4, 5, 6).reshape(batch_size * n_neighbors, num_patches_per_row, num_patches_per_col, 3, patch_size, patch_size) # [B * n_neighbors, num_patches_per_row, num_patches_per_col, 3, patch_size, patch_size]
        assembled = assemble_patches(samples_perm).reshape(batch_size, n_neighbors, 3, x_t_index_interpolated.shape[2], x_t_index_interpolated.shape[3]) # [B, n_neighbors, 3, H, W]
        assembled = F.interpolate(assembled.reshape(batch_size * n_neighbors, 3, x_t_index_interpolated.shape[2], x_t_index_interpolated.shape[3]), size=(x_t.shape[2], x_t.shape[3]), mode='bilinear', align_corners=False).reshape(batch_size, n_neighbors, 3, x_t.shape[2], x_t.shape[3])

        diffs = assembled - x_t[:, None] # [B, n_neighbors, 3, H, W]


        # if you want non-overlapping patches, you need to adjust this harcoded thingy.
        # weighted_diffs = (weights.permute(0, 3, 1, 2).reshape(batch_size, n_neighbors, 1, 1, 1) * diffs)
        # weighted_diffs = weighted_diffs.reshape(-1, *weighted_diffs.shape[2:]).unsqueeze(-1).unsqueeze(-1)

        patched_diffs = split_image_into_patches(diffs.reshape(batch_size * n_neighbors, 3, x_t.shape[2], x_t.shape[3]), self.patch_size)
        weighted_diffs = weights.permute(0, 3, 1, 2).reshape(batch_size * n_neighbors, num_patches_per_row, num_patches_per_col)[:, :, :, None, None, None] * patched_diffs


        weighted_diffs = assemble_patches(weighted_diffs).reshape(batch_size, n_neighbors, 3, x_t.shape[2], x_t.shape[3])
        score = weighted_diffs.sum(dim=1) / sigma2  # [B, 3, patch_size, patch_size]   
        denoised = from_noise_pred_to_x0_pred_ve(x_t, sigma_t.to(x_t.device), score.to(x_t.device))
        return denoised

class CropScorePixelLevel():
    """
    Compute a per-crop score using pixel-level patches.
    """
    
    def __init__(self, dataset_path: str, 
                 use_gpu: bool = True, index_type: str = 'ivf',
                 index_path: Optional[str] = None, device: torch.device = None,
                 dtype: torch.dtype = torch.float32, batch_size: int = 16,
                 num_clusters: int = 4096, patch_size: int = None, keep_ratio: float = 0.1, pca_dim: int = None, non_overlapping: bool = False, 
                 resolution: int = 64):
        """
        Initialize CropScorePixelLevel.
        
        Args:
            dataset_path: Path to the dataset (directory or zip file)
            use_gpu: Whether to use GPU acceleration for FAISS
            index_type: Type of FAISS index ('ivf', 'flat', 'hnsw')
            index_path: Path to save/load the FAISS index (optional)
            device: torch device for tensor operations
            dtype: data type for tensors
            batch_size: Batch size for processing dataset during index building
        """
        self.patch_size = patch_size
        self.resolution = resolution
        self.faiss_disk = FAISSIndex(dataset_path=dataset_path, 
                                         index_path=index_path, 
                                         patch_size=patch_size, 
                                         use_gpu=use_gpu, 
                                         index_type=index_type, 
                                         device=device, 
                                         dtype=dtype, 
                                         batch_size=batch_size, 
                                         num_clusters=num_clusters, 
                                         patch_mode='pixel_level',
                                         keep_ratio=keep_ratio,
                                         pca_dim=pca_dim,
                                         non_overlapping=non_overlapping,
                                         resolution=resolution)
    
    

    def find_neighbors(self, image: torch.Tensor, n_neighbors: int = 10, nprobe: int = 20000) -> torch.Tensor:
        """
        Find pixel-level neighbors.
        
        Args:
            image: torch.Tensor of shape [B, C, H, W]
        Returns:
            batched_nearest_samples: torch.Tensor of shape [B, H, W, n_neighbors, 3, patch_size, patch_size]
            batched_distances: torch.Tensor of shape [B, H, W, n_neighbors]
        """
        patches = split_image_into_patches(image, self.patch_size, mode='pixel_level') # B, C, H, W, patch_size, patch_size
        patches_perm = patches.permute(0, 2, 3, 1, 4, 5) # B, H, W, C, patch_size, patch_size
        batched_patches = patches_perm.reshape(patches_perm.shape[0] * patches_perm.shape[1] * patches_perm.shape[2], 3, patches_perm.shape[-2], patches_perm.shape[-1]) # B * H * W, 3, patch_size, patch_size
        _, batched_distances, batched_nearest_samples = self.faiss_disk.find_nearest_matches(batched_patches, n_neighbors=n_neighbors, nprobe=nprobe)

        batched_nearest_samples = torch.from_numpy(np.stack(batched_nearest_samples))  # B * H * W, n_neighbors, 3, patch_size, patch_size
        batched_nearest_samples = batched_nearest_samples.reshape(image.shape[0], image.shape[-2], image.shape[-1], *batched_nearest_samples.shape[1:]) # B, H, W, n_neighbors, 3, patch_size, patch_size
        batched_distances = torch.from_numpy(np.stack(batched_distances))
        batched_distances = batched_distances.reshape(image.shape[0], image.shape[-2], image.shape[-1], *batched_distances.shape[1:])  # B, H, W, n_neighbors
        return batched_nearest_samples, batched_distances


    def __call__(self, x_t: torch.Tensor, 
                sigma_t: torch.Tensor, temperature: float = 1.0, 
                n_neighbors: int = 10, nprobe: int = 10) -> torch.Tensor:
        """
        Compute score from nearest neighbors using softmax weighting.
        
        Args:
            x_t: input tensor [B, C, H, W]
            sigma_t: noise level [B]
            
        Returns:
            score: computed score tensor [B, C, H, W]
        """
        samples, distances = self.find_neighbors(image=x_t, n_neighbors=n_neighbors, nprobe=nprobe) # samples: [B, num_patches_per_row, num_patches_per_col, n_neighbors, 3, patch_size, patch_size]
        batch_size, num_patches_per_row, num_patches_per_col, n_neighbors, _, patch_size, _ = samples.shape
        sigma2 = sigma_t ** 2
        D = x_t[0].numel()
        
        # Compute log-likelihoods for the subset

        log_norm = -0.5 * D * torch.log(2 * torch.pi * sigma2)
        log_likelihoods = log_norm - distances / (2 * sigma2)
        
        # Compute softmax weights over the subset
        weights = F.softmax(log_likelihoods / temperature, dim=-1)  # (B, num_patches_per_row, num_patches_per_col, n_neighbors)
        
        # Compute weighted differences
        real_patch_size = self.patch_size + 1 if self.patch_size % 2 == 0 else self.patch_size
        samples_perm = samples.permute(0, 3, 4, 1, 2, 5, 6).reshape(batch_size * n_neighbors, 3, x_t.shape[2], x_t.shape[3], real_patch_size, real_patch_size)
        diffs = samples_perm.to(x_t.device) - x_t[:, :, :, :, None, None]

        weights = weights.permute(0, 3, 1, 2).reshape(batch_size * n_neighbors, num_patches_per_row, num_patches_per_col)

        diffs_on_central_pixel = diffs[:, :, :, :, real_patch_size // 2, real_patch_size // 2] # (B * n_neighbors, 3, H, W)
        weighted_diffs = weights[:, None].to(x_t.device) * diffs_on_central_pixel  # (B * n_neighbors, 3, H, W)
        weighted_diffs = weighted_diffs.reshape(batch_size, n_neighbors, 3, x_t.shape[2], x_t.shape[3])
        score = weighted_diffs.sum(dim=1) / sigma2.to(x_t.device)  # [B, 3, H, W]   
        denoised = from_noise_pred_to_x0_pred_ve(x_t, sigma_t.to(x_t.device), score.to(x_t.device))
        return denoised
    


class LaplacianCropScore():
    """
    Compute a per-crop score using pixel-level patches with Laplacian pyramid processing.
    """
    
    def __init__(self, dataset_path: str, 
                 use_gpu: bool = True, index_type: str = 'ivf',
                 index_path: Optional[str] = None, device: torch.device = None,
                 dtype: torch.dtype = torch.float32, batch_size: int = 16,
                 num_clusters: int = 4096, patch_size: int = None, keep_ratio: float = 0.1, pca_dim: int = None, non_overlapping: bool = False, 
                 resolution: int = 64, laplacian_levels: int = 4, laplacian_level_of_interest: int = 2):
        """
        Initialize LaplacianCropScore.
        
        Args:
            dataset_path: Path to the dataset (directory or zip file)
            use_gpu: Whether to use GPU acceleration for FAISS
            index_type: Type of FAISS index ('ivf', 'flat', 'hnsw')
            index_path: Path to save/load the FAISS index (optional)
            device: torch device for tensor operations
            dtype: data type for tensors
            batch_size: Batch size for processing dataset during index building
            laplacian_levels: Number of pyramid levels
            laplacian_level_of_interest: Which level to use (0 = finest, levels-1 = coarsest)
        """
        self.patch_size = patch_size
        self.resolution = resolution
        self.faiss_disk = FAISSLaplacianIndex(dataset_path=dataset_path, 
                                         index_path=index_path, 
                                         patch_size=patch_size, 
                                         use_gpu=use_gpu, 
                                         index_type=index_type, 
                                         device=device, 
                                         dtype=dtype, 
                                         batch_size=batch_size, 
                                         num_clusters=num_clusters, 
                                         patch_mode='pixel_level',
                                         keep_ratio=keep_ratio,
                                         pca_dim=pca_dim,
                                         non_overlapping=non_overlapping,
                                         resolution=resolution,
                                         laplacian_levels=laplacian_levels,
                                         laplacian_level_of_interest=laplacian_level_of_interest)
    
    

    def find_neighbors(self, image: torch.Tensor, n_neighbors: int = 10, nprobe: int = 20000) -> torch.Tensor:
        """
        Find pixel-level neighbors.
        
        Args:
            image: torch.Tensor of shape [B, C, H, W]
        Returns:
            batched_nearest_samples: torch.Tensor of shape [B, H, W, n_neighbors, 3, patch_size, patch_size]
            batched_distances: torch.Tensor of shape [B, H, W, n_neighbors]
        """
        patches = split_image_into_patches(image, self.patch_size, mode='pixel_level') # B, C, H, W, patch_size, patch_size
        patches_perm = patches.permute(0, 2, 3, 1, 4, 5) # B, H, W, C, patch_size, patch_size
        batched_patches = patches_perm.reshape(patches_perm.shape[0] * patches_perm.shape[1] * patches_perm.shape[2], 3, patches_perm.shape[-2], patches_perm.shape[-1]) # B * H * W, 3, patch_size, patch_size
        _, batched_distances, batched_nearest_samples = self.faiss_disk.find_nearest_matches(batched_patches, n_neighbors=n_neighbors, nprobe=nprobe)

        batched_nearest_samples = torch.from_numpy(np.stack(batched_nearest_samples))  # B * H * W, n_neighbors, 3, patch_size, patch_size
        batched_nearest_samples = batched_nearest_samples.reshape(image.shape[0], image.shape[-2], image.shape[-1], *batched_nearest_samples.shape[1:]) # B, H, W, n_neighbors, 3, patch_size, patch_size
        batched_distances = torch.from_numpy(np.stack(batched_distances))
        batched_distances = batched_distances.reshape(image.shape[0], image.shape[-2], image.shape[-1], *batched_distances.shape[1:])  # B, H, W, n_neighbors
        return batched_nearest_samples, batched_distances


    def __call__(self, pyramid_level: torch.Tensor, 
                sigma_t: torch.Tensor, temperature: float = 1.0, 
                n_neighbors: int = 10, nprobe: int = 10) -> torch.Tensor:
        """
        Compute score from nearest neighbors using softmax weighting.
        
        Args:
            pyramid_level: input tensor [B, C, H, W]
            sigma_t: noise level [B]
            
        Returns:
            score: computed score tensor [B, C, H, W]
        """
        samples, distances = self.find_neighbors(image=pyramid_level, n_neighbors=n_neighbors, nprobe=nprobe) # samples: [B, num_patches_per_row, num_patches_per_col, n_neighbors, 3, patch_size, patch_size]
        batch_size, num_patches_per_row, num_patches_per_col, n_neighbors, _, patch_size, _ = samples.shape
        sigma2 = sigma_t ** 2
        D = pyramid_level[0].numel()
        
        # Compute log-likelihoods for the subset

        log_norm = -0.5 * D * torch.log(2 * torch.pi * sigma2)
        log_likelihoods = log_norm - distances / (2 * sigma2)
        
        # Compute softmax weights over the subset
        weights = F.softmax(log_likelihoods / temperature, dim=-1)  # (B, num_patches_per_row, num_patches_per_col, n_neighbors)
        
        # Compute weighted differences
        real_patch_size = self.patch_size + 1 if self.patch_size % 2 == 0 else self.patch_size
        samples_perm = samples.permute(0, 3, 4, 1, 2, 5, 6).reshape(batch_size * n_neighbors, 3, pyramid_level.shape[2], pyramid_level.shape[3], real_patch_size, real_patch_size)
        diffs = samples_perm - pyramid_level[:, :, :, :, None, None]

        weights = weights.permute(0, 3, 1, 2).reshape(batch_size * n_neighbors, num_patches_per_row, num_patches_per_col)

        diffs_on_central_pixel = diffs[:, :, :, :, real_patch_size // 2, real_patch_size // 2] # (B * n_neighbors, 3, H, W)
        weighted_diffs = weights[:, None] * diffs_on_central_pixel  # (B * n_neighbors, 3, H, W)
        weighted_diffs = weighted_diffs.reshape(batch_size, n_neighbors, 3, pyramid_level.shape[2], pyramid_level.shape[3])
        score = weighted_diffs.sum(dim=1) / sigma2  # [B, 3, H, W]
        denoised_pyramid_level = from_noise_pred_to_x0_pred_ve(pyramid_level, sigma_t, score)
        return denoised_pyramid_level
    


if __name__ == "__main__":
    print("FAISS Search Examples")
    print("====================")
    print("This module provides FAISS-based approximate nearest neighbors search for image patches.")
    print("For usage examples, see the 'examples/' directory:")



