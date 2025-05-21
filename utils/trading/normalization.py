#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name:         normalization.py
 Description:  Action normalization methods for trading environments.
 Author:       MASA
---------------------------------
'''
import numpy as np

def softmax_normalization(actions, bound_flag=1):
    """
    Normalize actions using softmax function.
    
    Args:
        actions: Array of action values
        bound_flag: Flag to indicate direction (1 for long, -1 for short)
        
    Returns:
        Array of normalized weights
    """
    if np.sum(np.abs(actions)) == 0:  
        norm_weights = np.array([1/len(actions)]*len(actions)) * bound_flag
    else:
        norm_weights = np.exp(actions)/np.sum(np.abs(np.exp(actions)))
    return norm_weights

def sum_normalization(actions, bound_flag=1):
    """
    Normalize actions using sum normalization.
    
    Args:
        actions: Array of action values
        bound_flag: Flag to indicate direction (1 for long, -1 for short)
        
    Returns:
        Array of normalized weights
    """
    if np.sum(np.abs(actions)) == 0:
        norm_weights = np.array([1/len(actions)]*len(actions)) * bound_flag
    else:
        norm_weights = actions / np.sum(np.abs(actions))
    return norm_weights

def get_weights_normalization_fn(norm_method, bound_flag=1):
    """
    Get the appropriate normalization function based on method.
    
    Args:
        norm_method: Normalization method ('softmax' or 'sum')
        bound_flag: Flag to indicate direction (1 for long, -1 for short)
        
    Returns:
        Function for normalizing weights
    """
    if norm_method == 'softmax':
        return lambda actions: softmax_normalization(actions, bound_flag)
    elif norm_method == 'sum':
        return lambda actions: sum_normalization(actions, bound_flag)
    else:
        raise ValueError(f"Unexpected normalization method: {norm_method}")