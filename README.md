# Unet_Segmentation
# Project Structure

Project structured as follows:
.
└── src/
    ├── carvana_dataset.py
    ├── unet.py
    ├── unet_parts.py
    ├── main.py
    ├── inference.py
    ├── data/
    │   ├── manual_test
    │   ├── manual_test_mask
    │   ├── train
    │   └── train_mask
    └── models/

`carvana_dataset.py` creates the PyTorch dataset. `unet.py` is the file that contains the U-Net architecture. `unet_parts.py` contains the building blocks for the U-Net. `main.py` file contains the training loop. `inference.py` contains necessary functions to easily run inference for single and multiple images.

The `models/` directory is to save and store the trained models.

The `data/` directory contains the data you're going to train on. `train/` contains images and `train_mask/` contains masks for the images. `manual_test/` and `manual_test_mask/` are optional directories for showcasing the inference.

# Training 

In order to train the model you must run the Unet_Segmentation.ipynb on GG Colab. File has hyperparameters of LEARNING_RATE, BATCH_SIZE and EPOCHS. You can change them as you like.

You must give your data directory and the directory you want to save your model to DATA_PATH and MODEL_SAVE_PATH variables in Unet_Segmentation.ipynb file.

By the end of the training your model will be saved into the MODEL_SAVE_PATH.

# Inference

If you want to run prediction on multiple images, you must use pred_show_image_grid() function by giving your data path, model path and device as arguments.

If you want to run the prediction on single image, you must use single-image-inference() function by giving image path, model path and your device as arguments.

