#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name:         directional_change.py
 Description:  Directional Change (DC) feature generation functions
 Author:       MASA
---------------------------------
'''
import numpy as np

def dc_feature_generation(data, dc_threshold):
    """
    Directional Change (DC) implementation.
    
    Args:
        data: Array of price data
        dc_threshold: Threshold for directional change
        
    Returns:
        List of DC events (True for upturn, False for downturn)
    """
    # Directional Change (DC) implementation.
    dc_event_lst = [True] 

    ph = data[0]
    pl = data[0]
    # Training dataset DC patterns
    for idx in range(1, len(data)):
        if dc_event_lst[-1]:
            if data[idx] <= (ph * (1 - dc_threshold)):
                dc_event_lst.append(False) # Downturn Event
                pl= data[idx]
            else:
                dc_event_lst.append(dc_event_lst[-1]) # No DC pattern
                if ph < data[idx]:
                    ph = data[idx]
        else:
            if data[idx] >= (pl * (1 + dc_threshold)):
                dc_event_lst.append(True)  # Upturn Event
                ph = data[idx]
            else:
                dc_event_lst.append(dc_event_lst[-1])  # No DC pattern
                if pl > data[idx]:
                    pl = data[idx]
    # Uptrend event: True
    # Downtrend evvent: False
    return dc_event_lst