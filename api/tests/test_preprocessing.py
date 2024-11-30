import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import cv2
from unittest.mock import Mock, patch
from ..ml.preprocessing import DataPreprocessor

@pytest.fixture
def data_preprocessor():
    return DataPreprocessor(
        cache_dir="test_cache"
    )

@pytest.fixture
def sample_tabular_data():
    return pd.DataFrame({
        'num1': np.random.randn(100),
        'num2': np.random.randn(100),
        'cat1': np.random.choice(['A', 'B', 'C'], 100),
        'cat2': np.random.choice(['X', 'Y'], 100),
        'target': np.random.randint(0, 2, 100)
    })

def test_preprocess_tabular(data_preprocessor, sample_tabular_data):
    result = data_preprocessor.preprocess_tabular(
        data=sample_tabular_data,
        numerical_cols=['num1', 'num2'],
        categorical_cols=['cat1', 'cat2'],
        target_col='target'
    )
    
    assert 'X_train' in result
    assert 'X_test' in result
    assert 'y_train' in result
    assert 'y_test' in result
    assert 'numerical_processor' in result
    assert 'feature_names' in result
    
    assert result['X_train'].shape[1] == len(result['feature_names'])
    assert len(result['y_train']) + len(result['y_test']) == len(sample_tabular_data)

def test_preprocess_image(data_preprocessor):
    with patch('cv2.imread') as mock_imread:
        with patch('cv2.cvtColor') as mock_cvtColor:
            with patch('cv2.resize') as mock_resize:
                # Mock image data
                mock_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                mock_imread.return_value = mock_image
                mock_cvtColor.return_value = mock_image
                mock_resize.return_value = mock_image
                
                image_paths = [
                    Path("test1.jpg"),
                    Path("test2.jpg")
                ]
                
                dataset = data_preprocessor.preprocess_image(
                    image_paths=image_paths,
                    target_size=(224, 224),
                    augment=True
                )
                
                assert len(dataset) == 2
                sample = dataset[0]
                assert isinstance(sample, torch.Tensor)
                assert sample.shape == (3, 224, 224)

def test_preprocess_text(data_preprocessor):
    texts = [
        "Hello World!",
        "Testing 123",
        "Special @#$ Characters"
    ]
    
    processed = data_preprocessor.preprocess_text(
        texts=texts,
        max_length=10,
        transforms=['lowercase', 'remove_special_chars']
    )
    
    assert len(processed) == 3
    assert all(len(text) <= 10 for text in processed)
    assert all(text.islower() for text in processed)
    assert not any('@' in text for text in processed)

def test_save_and_load_preprocessed(data_preprocessor):
    test_data = np.random.randn(100, 10)
    
    # Test numpy format
    data_preprocessor.save_preprocessed(
        data=test_data,
        name="test_data",
        format="numpy"
    )
    
    loaded_data = data_preprocessor.load_preprocessed(
        name="test_data",
        format="numpy"
    )
    
    assert np.array_equal(test_data, loaded_data)
    
    # Test torch format
    torch_data = torch.tensor(test_data)
    data_preprocessor.save_preprocessed(
        data=torch_data,
        name="test_torch",
        format="torch"
    )
    
    loaded_torch = data_preprocessor.load_preprocessed(
        name="test_torch",
        format="torch"
    )
    
    assert torch.equal(torch_data, loaded_torch)

def test_create_data_loader(data_preprocessor):
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.data = torch.randn(100, 10)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = DummyDataset()
    
    loader = data_preprocessor.create_data_loader(
        dataset=dataset,
        batch_size=32,
        shuffle=True
    )
    
    assert isinstance(loader, torch.utils.data.DataLoader)
    assert loader.batch_size == 32
    assert loader.shuffle == True
    
    # Test batch iteration
    batch = next(iter(loader))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape[0] <= 32  # Last batch might be smaller
    assert batch.shape[1] == 10
