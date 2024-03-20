# Cloth Garments Segmentation 

## Environment Setup

1. Create a Conda environment using the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```
2. In Jupyter Notebook, select the "sam" environment to work in.

3. Download the fashion dataset and put the upper.zip to folder "dataset", then extract there, there will be folder "upper" which we will be using this.

## Training the Network

1. Open and run Notebook 1 to begin the training process and to familiarize yourself with the codebase.
2. The training process generates result output in the "logs/result" folder, which includes generated output at specific batch.

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
- **Training_Generated/**: Stores masks and temporary RGB files for augmentation purposes.

## Understanding the Code

- **Notebook 1**: The main architecture is detailed here, as illustrated in `Network.jpg`. The notebook is thoroughly commented, and you are encouraged to experiment with the hyperparameters.

## Additional Resources

- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [PyTorch Documentation](https://pytorch.org/docs/)

For any questions or contributions, please feel free to open an issue or pull request.
