#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name:         featGen.py
 Description:  Technical feature generation.
 Author:       MASA
---------------------------------
'''
import numpy as np
import pandas as pd
import copy
import os
from talib import abstract
import sys
sys.path.append(".")

# Import from the new package structure
from utils.features import FeatureProcesser, dc_feature_generation

# Re-export the main components for backward compatibility
__all__ = ['FeatureProcesser', 'dc_feature_generation']