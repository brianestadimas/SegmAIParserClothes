# Garment Segmentation Project

This project focuses on segmenting garments in images using advanced neural network architectures. The goal is to accurately identify and separate different clothing items in a given image, which can be crucial for applications in fashion technology, retail, and augmented reality.

## Architecture Overview

We utilize two main architectures for our segmentation tasks: SegNet and a Diffusion model. These models have been chosen for their efficiency and effectiveness in handling complex image segmentation tasks.

![Network Architecture](network.png)

## Environment Setup

### Installing Anaconda

1. Download Anaconda for your operating system from the [Anaconda website](https://www.anaconda.com/products/distribution).
2. Follow the installation instructions provided on the download page for your OS.
3. Once installed, you can verify the installation by opening a terminal (or Anaconda Prompt) and typing:
    ```
    conda list
    ```

### Creating the Project Environment

1. Clone the project repository to your local machine.
2. Navigate to the project directory.
3. Create a new Conda environment using the `environment.yml` file provided in the project:
    ```
    conda env create -f environment.yml
    ```

### Setting Up VSCode and Jupyter Notebook

1. Install Visual Studio Code (VSCode) from the [official website](https://code.visualstudio.com/).
2. Open VSCode and install the Jupyter Notebook extension:
    - Open the Extensions view by clicking on the square icon on the sidebar or pressing `Ctrl+Shift+X`.
    - Search for "Jupyter" and install the extension provided by Microsoft.
3. Open the project folder in VSCode.

### Running the Notebooks

1. Open the Command Palette (`Ctrl+Shift+P`) and search for "Python: Select Interpreter". Choose the Conda environment you created earlier.
2. Open Notebook #1 from the project directory.
3. Select "sam" as the kernel.
4. Run all cells in the notebook by opening the Command Palette and searching for "Jupyter: Run All Cells".

## Dataset Structure

Before starting with the training or testing, organize your dataset as follows:
```
/dataset/upper
|-- cloth
|-- cloth_align
|-- cloth_align_parse-bytedance
|-- ...
```
## FOR TRAINING YOU MAY USE NOTEBOOK #1

Select the kernel (top-right) as "SAM" and select 'Run All'
Make sure Python and Jupyter extensions installed in Vscode

## Testing the Model

To evaluate the model's performance on unseen data, follow these steps:

1. **Prepare the Input Data:**
   - Place all images that you wish to test inside the `logs/input` folder within the project directory. Ensure that these images are in a format compatible with your model (typically `.jpg` or `.png`).

2. **Execute the Testing Notebook:**
   - Open **Notebook #3** from the project's Jupyter notebook interface in VSCode. This notebook is specifically designed for the testing phase.
   - Before running the notebook, ensure that the correct Conda environment is activated and the "sam" kernel is selected. This can be done from the Jupyter notebook interface in VSCode.
   - Execute all cells in the notebook by using the "Run All" command. This process will read the images from `logs/input`, perform garment segmentation using the trained model, and save the results.

3. **Review the Output:**
   - Once Notebook #3 has finished running, navigate to the `logs/output` folder. Here, you'll find the segmented outputs of your input images.
   - These output images will show the garment segmentation results, which can then be used for further analysis or processing as required by your project objectives.



