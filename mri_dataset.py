from __future__ import print_function, division
import os
import torch
# import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
from torchvision.io import read_video
from torch import from_numpy
import skimage
import cv2
from skimage import metrics as sm
import torchvision.transforms.functional as F

from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def get_dataloader(dataset, batch_size):
    dataloader = DataLoader(dataset= dataset, batch_size = batch_size, shuffle= False)
    return dataloader

class MRIDataset(Dataset):
    """MRI dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the videos.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.paths = self.load_paths(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        vid_name = self.paths[idx]
        sample = self.load_image_batches(vid_name, 1, 24)
        
        if self.transform:
            if isinstance(sample[0], list):
                print("NOT VALID")
            else:
                sample = self.transform(sample)

        return sample
    
    def make_dataset(self, root_dir):
        paths = self.load_paths(root_dir)
        samples = []
        for idx in len(paths):
            vid_name = self.paths[idx]
            sample = self.load_image_batches(vid_name, 1, 24)
            samples.append(samples)
        
        return samples
    



    # function to extract the paths for files froma path
    def load_paths(self, data_path):
        files = []
        files.append(glob.glob(data_path, 
                    recursive = True))
        return files[0]
    




    def load_image_batches(self, path, step = 10, smooth = 5):
        # path = "C:/Users/Misha/OneDrive - University of Sussex/FYP/Participants/Participant_12/Processed_data/Video/Subject_12_03.mp4"

        frames, _, _ = read_video(str(path), output_format="TCHW") # returns video frames, audio frames, metadata for the video and audio
        # print(frames.shape)
        images1 = []
        images2 = []
        img = np.array([np.moveaxis(frames[a].numpy()[:,:,:], 0, -1) for a in range(len(frames))])
        # print(img.shape)
        # print(sm.structural_similarity(img[0, 200:500, 600:900, 0], img[1, 200:500, 600:900, 0]))
        # len(frames)-2
        for i in range(0, len(frames)-2, step):
            img1 = img[i, 100:600, 500:1000, :]
            img2 = img[i+1, 100:600, 500:1000, :]
            if (sm.structural_similarity(img1[:,:,0], img2[:,:,0]) < 0.90):

                new_shape = (img1.shape[0] , img1.shape[1] , img1.shape[2])
                blurred1 = skimage.transform.resize(image=img1, output_shape=new_shape).astype(np.float32)
                blurred2 = skimage.transform.resize(image=img2, output_shape=new_shape).astype(np.float32)

                blurred_1 = np.moveaxis(cv2.bilateralFilter(blurred1,smooth,160,160), -1, 0)
                blurred_2 = np.moveaxis(cv2.bilateralFilter(blurred2,smooth,160,160), -1, 0)

                images1.append(torch.from_numpy(blurred_1))
                images2.append(torch.from_numpy(blurred_2))

        if len(images2) < 1: #when the video is too short/ redundant video
            img1_batch = images1
            img2_batch = images2
        else:
            img1_batch = torch.stack(images1) # making predictions between 2 pairs of frames 53 and 83, and 84 and 130
            img2_batch = torch.stack(images2)

        return img1_batch, img2_batch
    


    def plot(self, idx, tens_num, **imshow_kwargs):
        print(type(self))
        imgs = self[idx][tens_num]    
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0])
        _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                img = F.to_pil_image(img.to("cpu"))
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.tight_layout()


    # def run_raft(self):
    #     flows = []
    #     for idx in range(len(self)):
    #         print(type(self[idx])) #supposed to be a tuple
    #         print(self[idx][0].shape)
    #         flows.append(RaftMRI(self[idx]))

    #     return flows

    def run_raft(self):
        device = 'cpu'
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights, progress=True).to(device)
        model = model.eval()


        sample1 = self
        flows = []
        print(len(sample1))
        for idx in range(len(sample1)):
            # print(sample1[idx][0])
            # print(type(sample1[idx][0]))

            if isinstance(sample1[idx][0], list): continue
            else:
                print(idx)
                img1__batch = sample1[idx][0]
                img2__batch = sample1[idx][1]
                # get list of predictions
                list_of_flows = model(img1__batch.to(device), img2__batch.to(device))
                # get the last one - most accurate
                predicted_flow = list_of_flows[-1]
                flows.append(predicted_flow)

        return flows

class Rescale(object):
    def __init__(self, output_size) -> None:
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        img1_batch = F.resize(sample[0], size=[self.output_size, self.output_size], antialias=False) # resize frames to ensure they are divisable by 8 
        img2_batch = F.resize(sample[1], size=[self.output_size, self.output_size], antialias=False)

        return (img1_batch, img2_batch)
    

class RaftTransforms(object):
    def __call__(self, sample):
        weights = Raft_Large_Weights.DEFAULT
        raft_transforms = weights.transforms()
        return raft_transforms(sample[0], sample[1])
    


class RaftMRI(object):

    def __init__(self, device = 'cpu'):
        self.device = device

    def __call__(self, sample1):
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights, progress=True).to(self.device)
        model = model.eval()


        img1__batch = sample1[0]
        img2__batch = sample1[1]
        # get list of predictions
        list_of_flows = model(img1__batch.to(self.device), img2__batch.to(self.device))
        # get the last one - most accurate
        predicted_flow = list_of_flows[-1]

        return predicted_flow

