import torch
import torch.nn.functional as F
import torch, numpy, random, csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torchvision import transforms
import torch, torchvision
from torch import nn
from PIL import Image

class FocalLoss(nn.Module):
	def __init__(self, gamma=2, alpha=0.25):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha
	def __call__(self, input, target):
		if len(target.shape) == 1:
			target = torch.nn.functional.one_hot(target, num_classes=64)
		loss = torchvision.ops.sigmoid_focal_loss(input, target.float(), alpha=self.alpha, gamma=self.gamma,
												  reduction='mean')
		return loss


def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    out = F.pad(x, pads, "constant", 0)

    return out, pads

def unpad(x, pad):
    if pad[2]+pad[3] > 0:
        x = x[:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,pad[0]:-pad[1]]
    return x

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax, class_name="", color="green"):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.text(x0, y0 - 10, f"{class_name}")
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2)
    )


def expand_box(box, image, expansion_rate=0.1):
    # Calculate expansion amounts
    width = box[2] - box[0]
    height = box[3] - box[1]
    expand_width = width * expansion_rate
    expand_height = height * expansion_rate
    
    # Create a new expanded box
    new_box = [
        max(box[0] - expand_width, 0),  # left
        max(box[1] - expand_height, 0),  # up
        min(box[2] + expand_width, image.shape[1]),  # right
        min(box[3] + expand_height, image.shape[0])  # down
    ]
    return new_box

def tensor_to_numpy(tensor):
    # Convert from Torch Tensor to numpy
    # and handle the case of 1-channel and 3-channel images
    np_image = tensor.numpy()
    if np_image.shape[0] == 1:
        # For grayscale, repeat the channel to get RGB
        np_image = np.repeat(np_image, 3, axis=0)
    # Transpose from (C, H, W) to (H, W, C) for plotting
    np_image = np.transpose(np_image, (1, 2, 0))
    return np_image

def save_image_helper(tensor, filepath):
    """
    Save a torch tensor as an image.
    """
    img = tensor.detach().cpu().numpy()
    img = img.transpose(1, 2, 0)  # Convert from CHW to HWC format for PIL
    img = ((img + 1) * 0.5 * 255).astype('uint8')  # Scale from [-1, 1] to [0, 255]
    Image.fromarray(img).save(filepath)

class ScaleToCenterTransform:
    """Scales the image towards the center and crops it back to the output size."""
    def __init__(self, output_size, scale_factor=1.2):
        self.output_size = output_size
        self.scale_factor = scale_factor
        self.pre_crop_size = int(output_size * scale_factor)
    
    def __call__(self, img):
        # Resize to scale up
        resize_transform = transforms.Resize(self.pre_crop_size)
        img = resize_transform(img)
        # Center crop to the output size
        crop_transform = transforms.CenterCrop(self.output_size)
        img = crop_transform(img)
        return img