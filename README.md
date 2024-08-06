
# PIRNet

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Configuration](#configuration)
5. [Training](#training)
6. [Testing](#testing)
7. [Visualization](#visualization)


## Project Structure

```
project/
│
├── data/
│   ├── preprocess.py
│   └── dataset.py
│
├── model/
│   ├── inception_block.py
│   ├── residual_block.py
│   ├── pyramid_pooling.py
│   └── unet.py
│
├── utils/
│   └── visualization.py
│
├── train.py
├── test.py
├── config.py
└── README.md
```

## Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/yourusername/mri-scan-analysis.git
    cd mri-scan-analysis
    ```

2. Create a virtual environment and activate it:
    ```sh
    conda create -n PIRNET python=3.8
    conda activate PIRNET
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Data Preparation

1. Organize your MRI scan data in the following structure:
    ```
    data/
    ├── train/
    │   ├── image1.nii.gz
    │   ├── image2.nii.gz
    │   └── ...
    └── test/
        ├── image1.nii.gz
        ├── image2.nii.gz
        └── ...
    ```

2. Adjust the `MRIDataset` class in `data/dataset.py` if your data format differs.

## Configuration

Edit `config.py` to set the hyperparameters and paths:

- `batch_size`: Number of samples per batch
- `epochs`: Number of training epochs
- `lr`: Learning rate
- `data_dir`: Directory containing the dataset
- `output_dir`: Directory to save model checkpoints and outputs
- `image_size`: Size of input images

## Training

To train the model, run:
```sh
python train.py
```

This script will:
1. Load and preprocess the data
2. Initialize the model
3. Train the model for the specified number of epochs
4. Save the best model based on validation loss
5. Visualize sample results every 10 epochs

Training progress will be printed to the console, showing train and validation loss for each epoch.

## Testing

To test the trained model on the test set, run:
```sh
python test.py
```

This script will:
1. Load the best trained model
2. Run inference on the test set
3. Visualize the results for each test sample

## Visualization

The `utils/visualization.py` module provides functions to visualize the MRI scans and model predictions:

- `plot_slice(image, slice_num, title)`: Plot a single slice of a 3D image
- `plot_comparison(original, prediction, slice_num)`: Plot original and predicted images side by side

These functions are used in both training and testing scripts to visualize results.
