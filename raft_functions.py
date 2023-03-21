import cv2
import numpy as np

import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from skimage import metrics as sm

def convert_images(img_batch):
    # print(img_batch.shape)
    img2_batch = [(img2 + 1) / 2 for img2 in img_batch] # upsampling the image back to original
    # print(len(img2_batch))
    img2 = np.array([np.moveaxis(img2_batch[a].numpy()[:,:,:], 0, -1) for a in range(len(img2_batch))])
    return img2

def convert_flow(predicted_flows):
    flows_tchw = [predicted_flows[a].detach().numpy()[:,:,:] for a in range(predicted_flows.shape[0])] # make it back to numpy arrays for all N predicted flows
    flows = np.moveaxis(flows_tchw, 0, -1) # change the dimmensions from TCHW to THWC
    return flows


def draw_flow(img, flow, step=16):
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (_x2, _y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis


def remap_forward(image, flow):
    flow[np.isnan(flow)] = 0
    x = np.linspace(0, image.shape[0], image.shape[0])
    y = np.linspace(0, image.shape[1], image.shape[1])
    xv, yv = np.meshgrid(x,y)
    map_x = -flow[:, :, 0] + xv
    map_y = -flow[:, :, 1] + yv

    mapped_img = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR) #pull back mapping to avoid holes in the image
    return mapped_img

def remap_backward(image, flow):
    flow[np.isnan(flow)] = 0
    x = np.linspace(0, image.shape[0], image.shape[0])
    y = np.linspace(0, image.shape[1], image.shape[1])
    xv, yv = np.meshgrid(x,y)
    map_x = flow[:, :, 0] + xv
    map_y = flow[:, :, 1] + yv

    mapped_img = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR) #pull back mapping to avoid holes in the image
    return mapped_img


def draw_process_evaluate(image1, image2, mapped):
    plt.subplot(1,3,1)
    plt.imshow(image1)
    plt.subplot(1,3,2)
    plt.imshow(image2)
    plt.subplot(1,3,3)
    plt.imshow(mapped, cmap = "gray")
    plt.show()
    print('Structural similarity between the 2 images:', sm.structural_similarity(image2[:, :, 0], mapped[:,:,0]))


def draw_remap_with_flow(image, image2, flow):
    flow[np.isnan(flow)] = 0
    x = np.linspace(0, image.shape[0], image.shape[0])
    y = np.linspace(0, image.shape[1], image.shape[1])
    xv, yv = np.meshgrid(x,y)
    map_x = flow[:, :, 0] + xv
    map_y = flow[:, :, 1] + yv

    mapped_img = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR) #pull back mapping to avoid holes in the image
    
    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.subplot(1,3,2)
    plt.imshow(image2)
    plt.subplot(1,3,3)
    plt.imshow(mapped_img, cmap = "gray")
    plt.show()

    print('Second image warped into the first:', sm.structural_similarity(image2[:, :, 0], mapped_img[:,:,0]))