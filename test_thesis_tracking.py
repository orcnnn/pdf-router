#!/usr/bin/env python3
"""
Test script to verify thesis_id change detection and timestamp-based pushing functionality
"""

import tempfile
import os
from unittest.mock import Mock, patch
from router import PDFRouter
from utils import ProcessorLabel
import datasets
from loguru import logger

def test_thesis_id_change_detection():
    """Test that the system correctly detects thesis_id changes and pushes batches"""
    
    # Create mock data simulating different thesis_ids
    mock_data = [
        {"thesis_id": "thesis_001", "images": Mock(), "predictions": '{"labels": [{"class": "Metin", "confidence": 0.9}]}'},
        {"thesis_id": "thesis_001", "images": Mock(), "predictions": '{"labels": [{"class": "Metin", "confidence": 0.9}]}'},
        {"thesis_id": "thesis_002", "images": Mock(), "predictions": '{"labels": [{"class": "Metin", "confidence": 0.9}]}'},
        {"thesis_id": "thesis_002", "images": Mock(), "predictions": '{"labels": [{"class": "Metin", "confidence": 0.9}]}'},
        {"thesis_id": "thesis_003", "images": Mock(), "predictions": '{"labels": [{"class": "Metin", "confidence": 0.9}]}'},
    ]
    
    # Mock the dataset
    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    
    # Create a temporary router instance
    router = PDFRouter(
        model_name="test-model",
        debug=True,
        use_vllm=False,
        use_marker=True,
        vlm_batch_size=2,
        buffer_size=10
    )
    
    # Variables to track push calls
    push_calls = []
    
    def mock_push_to_hub(repo_id, private=False):
        # Extract the split name from the DatasetDict
        split_names = list(self.keys())
        if split_names:
            push_calls.append({
                'repo_id': repo_id,
                'split_name': split_names[0],
                'data_count': len(self[split_names[0]])
            })
        logger.info(f"Mock push called: repo_id={repo_id}, splits={split_names}")
    
    # Mock external dependencies
    with patch('datasets.load_dataset') as mock_load_dataset, \
         patch('datasets.DatasetDict.push_to_hub', mock_push_to_hub), \
         patch('router.send_to_marker_map') as mock_marker, \
         patch('router.get_dataset_config_names') as mock_get_configs, \
         patch('utils.get_splits') as mock_get_splits:
        
        # Setup mocks
        mock_load_dataset.return_value = mock_dataset
        mock_get_configs.return_value = ['test_split']
        mock_get_splits.return_value = []
        
        # Mock marker processing to just add text field
        def mock_marker_process(sample):
            sample['text'] = f"Processed text for {sample.get('thesis_id', 'unknown')}"
            sample['processor_used'] = 'marker'
            return sample
        mock_marker.side_effect = mock_marker_process
        
        # Run the processing
        try:
            router.process_splits(
                ds_name="test/dataset",
                output_ds_name="test/output",
                streaming=True,
                push_while_streaming=False,  # We'll handle pushing manually
                limit=5
            )
            
            logger.info(f"Test completed. Number of push calls: {len(push_calls)}")
            for call in push_calls:
                logger.info(f"Push call: {call}")
                
            # Expected: 3 pushes (one for each thesis_id)
            expected_pushes = 3
            if len(push_calls) == expected_pushes:
                logger.success(f"✅ Test PASSED: Expected {expected_pushes} pushes, got {len(push_calls)}")
                return True
            else:
                logger.error(f"❌ Test FAILED: Expected {expected_pushes} pushes, got {len(push_calls)}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Test FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    logger.info("Starting thesis_id change detection test...")
    success = test_thesis_id_change_detection()
    
    if success:
        logger.success("🎉 All tests passed!")
    else:
        logger.error("💥 Some tests failed!")
        exit(1)
