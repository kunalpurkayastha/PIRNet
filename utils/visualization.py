import matplotlib.pyplot as plt
import numpy as np

def plot_slice(image, slice_num=None, title=None):
    if slice_num is None:
        slice_num = image.shape[2] // 2
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image[:, :, slice_num], cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def plot_comparison(original, prediction, slice_num=None):
    if slice_num is None:
        slice_num = original.shape[2] // 2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    ax1.imshow(original[:, :, slice_num], cmap='gray')
    ax1.set_title('Original')
    ax1.axis('off')
    
    ax2.imshow(prediction[:, :, slice_num], cmap='gray')
    ax2.set_title('Prediction')
    ax2.axis('off')
    
    plt.show()