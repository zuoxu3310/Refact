#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name:         feature_processor.py
 Description:  Main feature processing class.
 Author:       MASA
---------------------------------
'''
import numpy as np
import pandas as pd
import copy

from utils.features.technical_indicators import generate_technical_features
from utils.features.scaling import scale_and_split_features
from utils.features.fine_data import process_fine_data

class FeatureProcesser:
    """
    Process training data by generating technical indicators,
    scaling features, and preparing fine-grained data.
    """
    def __init__(self, config):
        """
        Initialize the feature processor.
        
        Args:
            config: Configuration object with processing parameters
        """
        self.config = config
        self.rawColLst = None
        self.techIndicatorLst = None
    
    def preprocess_feat(self, data):
        """
        Preprocess features through multiple processing steps.
        
        Args:
            data: DataFrame with raw stock data
            
        Returns:
            dict: Processed data with train/valid/test splits and fine-grained data
                - train: pd.DataFrame
                - valid: pd.DataFrame
                - test: pd.DataFrame
                - bftrain: pd.DataFrame
                - extra_train: dict {daily_market, fine_market, fine_stock}: pd.DataFrame
                - extra_valid: dict {daily_market, fine_market, fine_stock}: pd.DataFrame
                - extra_test: dict {daily_market, fine_market, fine_stock}: pd.DataFrame
        """
        # Generate technical features
        data = self.gen_feat(data=data)
        
        # Scale features and split into datasets
        data = self.scale_feat(data=data)
        
        # Process fine-grained data
        data = self.process_finedata(data=data)

        return data
    
    def gen_feat(self, data):
        """
        Generate technical indicators.
        
        Args:
            data: DataFrame with raw stock data
            
        Returns:
            DataFrame with generated technical features
        """
        # Generate technical indicators for the data
        datax, self.rawColLst, self.techIndicatorLst = generate_technical_features(data, self.config)
        return datax

    def scale_feat(self, data):
        """
        Scale features and split into train/valid/test sets.
        
        Args:
            data: DataFrame with generated features
            
        Returns:
            Dictionary with dataset splits
        """
        # Scale features and split data into train/valid/test sets
        dataset_dict = scale_and_split_features(
            data, 
            self.rawColLst, 
            self.techIndicatorLst, 
            self.config
        )
        return dataset_dict

    def process_finedata(self, data):
        """
        Process fine-grained data.
        
        Args:
            data: Dictionary with dataset splits
            
        Returns:
            Dictionary with dataset splits and additional fine data
        """
        # Process fine-grained market and stock data
        processed_data = process_fine_data(data, self.config)
        return processed_data