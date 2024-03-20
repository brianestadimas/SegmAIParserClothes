

import pandas as pd
import pydicom
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import Dataset
import torch.optim as optim
from torchvision import transforms
import numpy as np
import torch
from torch import nn
import os
from codes.metrics import relative_root_mean_squared_error, r_score
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error
from transformers import DPTImageProcessor, DPTForDepthEstimation
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import cv2
import torch.multiprocessing as mp

class DC_Data(Dataset):
    def __init__(self, root_csv, processor, model, device, root='', augmentation=None, png=False):
        self.df = pd.read_excel(root_csv)
        self.root = root
        self.png = png
        self.processor = processor
        self.model = model
        self.device = device
        self.transform = transforms.Compose([transforms.Resize((256, 256))]) if augmentation is None \
                            else transforms.Compose([augmentation, transforms.Resize((256, 256))]) 
        self.data = []
        if self.png:
            self.df['image_path'] = self.df.apply(lambda x: os.path.join(self.root, x[0].replace('.dcm', '.png')), axis=1)
            self.df = self.df[self.df['image_path'].apply(os.path.exists)]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        if self.png:
            # Load the image directly as a PNG
            image_path = self.root + self.df.iloc[index, 0].replace('.dcm', '.png')
            image = Image.open(image_path).convert('L')
            rgb_img = image.convert('RGB')
            inverse_image = ImageOps.invert(image)
        else:
            dicom_path = self.root + self.df.iloc[index, 0]
            dicom_data = pydicom.dcmread(dicom_path)
            pixel_array = dicom_data.pixel_array
            image = pixel_array.astype(np.float32)
            
            rgb_img = np.stack((pixel_array,) * 3, axis=-1).astype(np.uint8)
            rgb_img = Image.fromarray(rgb_img).convert('RGB')
            image = torch.from_numpy(image.transpose(0,1)/255)
            rgb_img = rgb_img

            # Display inversed image
            inverse_image = 1.0 - image
            image = image.unsqueeze(0)
            inverse_image = inverse_image.unsqueeze(0)
        
        # Depth Estimation
        inputs = self.processor(images=rgb_img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=rgb_img.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        # visualize the prediction
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth_image = Image.fromarray(formatted).convert('L')
        image = self.transform(image)
        inverse_image = self.transform(inverse_image)
        depth_image = self.transform(depth_image)
        
        if not isinstance(image, torch.Tensor):
            image = transforms.functional.to_tensor(image)
            inverse_image = transforms.functional.to_tensor(inverse_image)
        
        if not isinstance(depth_image, torch.Tensor):
            depth_image = transforms.functional.to_tensor(depth_image)
        
        mass = self.df.iloc[index, 1]
        return image, inverse_image, depth_image, mass