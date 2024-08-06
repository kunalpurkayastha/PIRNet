import numpy as np
import nibabel as nib
from skimage.transform import resize

def load_and_preprocess(file_path, target_size=(256, 256, 256)):
    # Load NIfTI file
    img = nib.load(file_path)
    data = img.get_fdata()
    
    # Normalize
    data = (data - data.min()) / (data.max() - data.min())
    
    # Resize
    data = resize(data, target_size, mode='constant', anti_aliasing=True)
    
    return data

def augment(image):
    # Simple augmentation example
    if np.random.rand() > 0.5:
        image = np.flip(image, axis=0)
    if np.random.rand() > 0.5:
        image = np.flip(image, axis=1)
    return image