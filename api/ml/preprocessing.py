from typing import Dict, List, Optional, Any, Union, Callable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import albumentations as A
from pathlib import Path
import logging
import json
import dask.dataframe as dd
from concurrent.futures import ThreadPoolExecutor
import cv2
import h5py

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data preprocessing and augmentation pipeline."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[Union[str, Path]] = None
    ):
        self.config = config or {}
        self.cache_dir = Path(cache_dir) if cache_dir else Path("preprocessed_data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Set up data transformations."""
        self.image_transforms = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        self.text_transforms = {
            "lowercase": lambda x: x.lower(),
            "remove_special_chars": lambda x: x.replace("[^a-zA-Z0-9\s]", ""),
            "remove_extra_spaces": lambda x: " ".join(x.split())
        }
        
        self.numerical_transforms = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler()
        }
    
    def preprocess_tabular(
        self,
        data: Union[pd.DataFrame, str, Path],
        numerical_cols: List[str],
        categorical_cols: List[str],
        target_col: Optional[str] = None,
        scaler: str = "standard"
    ) -> Dict[str, Any]:
        """Preprocess tabular data."""
        try:
            # Load data if path provided
            if isinstance(data, (str, Path)):
                data = dd.read_csv(data).compute()
            
            # Handle numerical features
            numerical_processor = self.numerical_transforms[scaler]
            numerical_features = numerical_processor.fit_transform(data[numerical_cols])
            
            # Handle categorical features
            categorical_features = pd.get_dummies(data[categorical_cols])
            
            # Combine features
            processed_data = np.hstack([
                numerical_features,
                categorical_features.values
            ])
            
            # Split data if target provided
            if target_col:
                X = processed_data
                y = data[target_col].values
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=self.config.get("test_size", 0.2),
                    random_state=self.config.get("random_state", 42)
                )
                
                return {
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                    "numerical_processor": numerical_processor,
                    "feature_names": numerical_cols + list(categorical_features.columns)
                }
            
            return {
                "processed_data": processed_data,
                "numerical_processor": numerical_processor,
                "feature_names": numerical_cols + list(categorical_features.columns)
            }
            
        except Exception as e:
            logger.error(f"Failed to preprocess tabular data: {str(e)}")
            raise
    
    def preprocess_image(
        self,
        image_paths: Union[List[str], List[Path]],
        target_size: tuple = (224, 224),
        augment: bool = True,
        batch_size: int = 32
    ) -> torch.utils.data.Dataset:
        """Preprocess image data."""
        try:
            class ImageDataset(Dataset):
                def __init__(self, image_paths, transform, target_size):
                    self.image_paths = image_paths
                    self.transform = transform
                    self.target_size = target_size
                
                def __len__(self):
                    return len(self.image_paths)
                
                def __getitem__(self, idx):
                    image = cv2.imread(str(self.image_paths[idx]))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, self.target_size)
                    
                    if self.transform:
                        image = self.transform(image=image)["image"]
                    
                    return torch.tensor(image).permute(2, 0, 1)
            
            transform = self.image_transforms if augment else None
            dataset = ImageDataset(image_paths, transform, target_size)
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to preprocess image data: {str(e)}")
            raise
    
    def preprocess_text(
        self,
        texts: Union[List[str], pd.Series],
        max_length: Optional[int] = None,
        transforms: Optional[List[str]] = None
    ) -> List[str]:
        """Preprocess text data."""
        try:
            transforms = transforms or ["lowercase", "remove_special_chars", "remove_extra_spaces"]
            
            processed_texts = texts
            for transform in transforms:
                if transform in self.text_transforms:
                    processed_texts = [
                        self.text_transforms[transform](text)
                        for text in processed_texts
                    ]
            
            if max_length:
                processed_texts = [
                    text[:max_length] for text in processed_texts
                ]
            
            return processed_texts
            
        except Exception as e:
            logger.error(f"Failed to preprocess text data: {str(e)}")
            raise
    
    def save_preprocessed(
        self,
        data: Any,
        name: str,
        format: str = "numpy"
    ):
        """Save preprocessed data to cache."""
        try:
            save_path = self.cache_dir / f"{name}.{format}"
            
            if format == "numpy":
                np.save(save_path, data)
            elif format == "torch":
                torch.save(data, save_path)
            elif format == "hdf5":
                with h5py.File(save_path, "w") as f:
                    f.create_dataset("data", data=data)
            elif format == "json":
                with save_path.open("w") as f:
                    json.dump(data, f)
            
            logger.info(f"Saved preprocessed data to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save preprocessed data: {str(e)}")
            raise
    
    def load_preprocessed(
        self,
        name: str,
        format: str = "numpy"
    ) -> Any:
        """Load preprocessed data from cache."""
        try:
            load_path = self.cache_dir / f"{name}.{format}"
            
            if format == "numpy":
                return np.load(load_path)
            elif format == "torch":
                return torch.load(load_path)
            elif format == "hdf5":
                with h5py.File(load_path, "r") as f:
                    return f["data"][:]
            elif format == "json":
                with load_path.open("r") as f:
                    return json.load(f)
            
        except Exception as e:
            logger.error(f"Failed to load preprocessed data: {str(e)}")
            raise
    
    def create_data_loader(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4
    ) -> DataLoader:
        """Create PyTorch DataLoader."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
