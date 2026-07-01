"""
core/book.py
------------
Order book feature extractors shared across Optiver Volatility and Optiver Close.
"""

import numpy as np
import pandas as pd


def weighted_average_price(bid_price, ask_price, bid_size, ask_size):
    return (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)

def bid_ask_spread(bid_price, ask_price):
    return ask_price - bid_price

def log_return(prices):
    return np.log(prices).diff()

def realized_volatility(log_returns):
    return np.sqrt(np.sum(log_returns ** 2))

def order_imbalance(bid_size, ask_size):
    return (bid_size - ask_size) / (bid_size + ask_size)

def depth_ratio(size_near, size_far):
    return size_far / (size_near + 1e-8)

def total_depth(bid_size1, bid_size2, ask_size1, ask_size2):
    return bid_size1 + bid_size2 + ask_size1 + ask_size2

def wap_combined(wap1, wap2, weight1=0.7, weight2=0.3):
    return weight1 * wap1 + weight2 * wap2
