# EXAMPLE

## Environment Setup

1. Create a Conda environment using the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```
2. In Jupyter Notebook, select the "sam" environment to work in.

## Training the Network

1. Open and run Notebook 1 to begin the training process and to familiarize yourself with the codebase.
2. The training process generates tensorboard logs in the "logs" folder, which includes RMSE, R2, and R metrics for each epoch, along with the saved model.

## Using TensorBoard

1. To visualize the training process, run the following command:
    ```bash
    tensorboard --logdir=logs/
    ```
2. Open your web browser and navigate to `localhost:6006` to view the TensorBoard dashboard.
![Tensorboard preview metrices](logs/image.png)

## Understanding the File Structure

- **codes/**: Contains metrics, data classes, and utility functions.
- **architecture/**: Includes the UNet architecture, UNet blocks, and transformer components.
- **models/**: Stores pretrained models for SAM and YOLO, which are used for segmentation augmentation.
- **Training_Generated/**: Stores masks and temporary RGB files for augmentation purposes.

## Understanding the Code

- **Notebook 1**: The main architecture is detailed here, as illustrated in `Network.jpg`. The notebook is thoroughly commented, and you are encouraged to experiment with the hyperparameters.
- **Notebooks 2 and 3**: Similar to Notebook 1, these notebooks explore different architectures. It's recommended to understand Notebook 1 before proceeding with these.
- **Notebook 4**: Utilized for manually generating Regions of Interest (ROI) from foundation models for augmentation purposes. Ensure to download the pretrained models and place them in the `models/` folder:
    - SAM: [Segment-Anything Model (SAM_B)](https://github.com/facebookresearch/segment-anything)
    - EfficientVIT: [EfficientVIT (L1)](https://github.com/mit-han-lab/efficientvit)
- **Notebook 5**: Explores image depth and its inversion. Note that areas of lower mass are typically brighter in depth images (found in `Training_Generated/Depth/`).

## Additional Resources

- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [PyTorch Documentation](https://pytorch.org/docs/)

For any questions or contributions, please feel free to open an issue or pull request.
