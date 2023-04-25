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
    img2 = np.array([np.moveaxis(img2_batch[a].numpy()[:,:,:], 0, -1) for a in range(len(img_batch))])
    return img2

def convert_flow(predicted_flows):
    flows_tchw = [predicted_flows[a].detach().numpy()[:,:,:] for a in range(predicted_flows.shape[0])] # make it back to numpy arrays for all N predicted flows
    flows = np.moveaxis(flows_tchw, 1, -1) # change the dimmensions from TCHW to THWC
    return flows


def draw_flow(img, flow, step=16):
        h, w = img.shape[:2] # compute high and width 
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int) # create grid of points x, y separated by a distance of step 
        fx, fy = flow[y,x].T # extract flow vectors fx, fy from the flow at grid points y,x
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2) #create a set of line segments lines joining the grid points to their corresponding displaced positions according to the flow vectors
        lines = np.int32(lines + 0.5) 
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # convert grayscale img to color img (BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0)) # draw line segments line on the color img
        for (x1, y1), (_x2, _y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1) # draw circles at the start points of the line segments 
        return vis # return result


def remap_forward(image, flow):
    flow[np.isnan(flow)] = 0
    x = np.linspace(0, image.shape[1], image.shape[1])
    y = np.linspace(0, image.shape[0], image.shape[0])
    xv, yv = np.meshgrid(x,y)
    map_x = -flow[:, :, 0] + xv
    map_y = -flow[:, :, 1] + yv

    mapped_img = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR) #pull back mapping to avoid holes in the image
    return mapped_img

def remap_backward(image, flow):
    flow[np.isnan(flow)] = 0
    x = np.linspace(0, image.shape[1], image.shape[1])
    y = np.linspace(0, image.shape[0], image.shape[0])
    xv, yv = np.meshgrid(x,y)
    map_x = flow[:, :, 0] + xv
    map_y = flow[:, :, 1] + yv

    mapped_img = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR) #pull back mapping to avoid holes in the image
    return mapped_img


def draw_process_evaluate(image_from, image_to, mapped):
    plt.subplot(1,3,1)
    plt.imshow(image_from)
    plt.subplot(1,3,2)
    plt.imshow(image_to)
    plt.subplot(1,3,3)
    plt.imshow(mapped, cmap = "gray")
    plt.show()
    print('Structural similarity between the 2 images:', sm.structural_similarity(image_to[:, :, 0], mapped[:,:,0], data_range = 1))


def draw_remap_with_flow(image_from, image_to, flow):
    flow[np.isnan(flow)] = 0
    x = np.linspace(0, image_from.shape[0], image_from.shape[0])
    y = np.linspace(0, image_from.shape[1], image_from.shape[1])
    xv, yv = np.meshgrid(x,y)
    map_x = flow[:, :, 0] + xv
    map_y = flow[:, :, 1] + yv

    mapped_img = cv2.remap(image_from, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR) #pull back mapping to avoid holes in the image
    
    plt.subplot(1,3,1)
    plt.imshow(image_from)
    plt.subplot(1,3,2)
    plt.imshow(image_to)
    plt.subplot(1,3,3)
    plt.imshow(mapped_img, cmap = "gray")
    plt.show()

    print('Second image warped into the first:', sm.structural_similarity(image_to[:, :, 0], mapped_img[:,:,0]))


from torchvision.io import read_video

def load_image_batches(path, step = 10):
    # path = "C:/Users/Misha/OneDrive - University of Sussex/FYP/Participants/Participant_12/Processed_data/Video/Subject_12_01.mp4"

    frames, _, _ = read_video(str(path), output_format="TCHW") # returns video frames, audio frames, metadata for the video and audio

    images1 = []
    images2 = []
    for i in range(0, len(frames)-2, step):
        images1.append(frames[i, :, 200:500, 600:900])
        images2.append(frames[i+1, :, 200:500, 600:900])
    img1_batch = torch.stack(images1) # making predictions between 2 pairs of frames 53 and 83, and 84 and 130
    img2_batch = torch.stack(images2)

    return img1_batch, img2_batch


from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import Raft_Small_Weights
from torchvision.models.optical_flow import raft_large


def preprocess(img1_batch, img2_batch, transf = 'large', s = 352):
    if transf == 'large':
        weights = Raft_Large_Weights.DEFAULT
    else: 
        weights = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    # img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
    # img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
    img1_batch = F.resize(img1_batch, size=[s, s], antialias=False) # resize frames to ensure they are divisable by 8 
    img2_batch = F.resize(img2_batch, size=[s, s], antialias=False)
    return transforms(img1_batch, img2_batch)

def raft(img1_batch, img2_batch, raft_ = 'large', s = 352, preprocess = False):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    # set the weights
    if raft_ == 'large':
        weights = Raft_Large_Weights.DEFAULT
    else: 
        weights = Raft_Small_Weights.DEFAULT
    # prepare the images for the model
    if preprocess:
        img1_batch, img2_batch = preprocess(img1_batch, img2_batch, transf = raft_, s = s)
    # create model
    model = raft_large(weights, progress=True).to(device)
    model = model.eval()

    # get list of predictions
    list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
    # get the last one - most accurate
    predicted_flow = list_of_flows[-1]

    return predicted_flow


# def evaluate_raft(img1_batch, img2_batch, predicted_flows, remapping = 'forward'):
#     images1 = convert_images(img1_batch) # all frames 1
#     images2 = convert_images(img2_batch) # all frames 2

#     flows = convert_flow(predicted_flows)
#     similarity = []
#     print(images1.shape)
#     print(flows.shape)
#     if remapping == 'forward':
#         mapped = remap_forward(images1[0], flows[0])
#         similarity.append(sm.structural_similarity(images2[0 , :, :, 0], mapped[:,:,0]))
        
#     else:
#         mapped = remap_backward(images2, flows)
#         return [sm.structural_similarity(images1[:,:,0], mapped[:,:,0])]

# def evaluate_raft_(img1_batch, img2_batch, predicted_flows, remapping = 'forward'):
#     images1 = convert_images(img1_batch) # all frames 1
#     images2 = convert_images(img2_batch) # all frames 2

#     flows = convert_flow(predicted_flows)
#     similarity = []
#     mapped = []
#     if remapping == 'forward':
#         for i in range(len(flows)):
#             mapped.append(remap_forward(images1[i], flows[i]))
#             print(mapped[i].shape)
#             print(images2[i].shape)
#             similarity.append([sm.structural_similarity(images2[i][: , :, 0], mapped[i][:, :, 0]  )])
        
#     else:
#         for i in range(len(flows)):
#             mapped.append(remap_backward(images2[i], flows[i]))
#             print(mapped[i].shape)
#             print(images2[i].shape)
#             similarity.append([sm.structural_similarity(images1[i][: , :, 0], mapped[i][:, :, 0]  )])

#     return similarity


# def evaluate_raft_2(images1, images2, flows, remapping = 'forward'):

#     similarity = []
#     mapped = []
#     if remapping == 'forward':
#         for i in range(len(flows)):
#             mapped.append(remap_forward(images1[i], flows[i][:,:,:]))
#             # draw_process_evaluate(images1[i], images2[i], remap_forward(images1[i], flows[i][:,:,:]))
#             # plt.imshow(draw_flow(images1[i][:,:,0], -flows[i][:,:,:], step=10)), plt.show()
#             similarity.append([sm.structural_similarity(images2[i][: , :, 0], mapped[i][:, :, 0], data_range= 1)])
        
#     else:
#         for i in range(len(flows)):
#             mapped.append(remap_backward(images2[i], flows[i][:,:,:]))
#             draw_process_evaluate(images2[i], images1[i], remap_backward(images2[i], -flows[i][:,:,:]))
#             plt.imshow(draw_flow(images2[i][:,:,0], -flows[i][:,:,:], step=20)), plt.show()
#             similarity.append([sm.structural_similarity(images1[i][: , :, 0], mapped[i][:, :, 0], data_range=1)])
    
#     return np.mean(similarity)

import flowpy        

def evaluate_raft_2(img1_batch, img2_batch, predicted_flows, remapping = 'forward'):
    print(img1_batch.shape)

    images1 = convert_images(img1_batch) # all frames 1
    images2 = convert_images(img2_batch) # all frames 2
    print(images1.shape)
    flows = convert_flow(predicted_flows)

    similarity = []
    mapped = []
    if remapping == 'forward':
        for i in range(len(flows)):
            flows[np.isnan(flows)] = 0
            mapped.append(flowpy.forward_warp(images1[i], flows[i]))
            draw_process_evaluate(images1[i], images2[i],flowpy.forward_warp(images1[i], flows[i]))
            plt.imshow(draw_flow(images1[i][:,:,0], -flows[i][:,:,:], step=13)), plt.show()
            similarity.append([sm.structural_similarity(images2[i][: , :, 0], mapped[i][:, :, 0], data_range= 1)])
            print()
        
    else:
        for i in range(len(flows)):
            # mapped.append(remap_backward(images2[i], flows[i][:,:,:]))
            # draw_process_evaluate(images2[i], images1[i], remap_backward(images2[i], flows[i][:,:,:]))
            mapped.append(flowpy.backward_warp(images2[i], flows[i]))
            draw_process_evaluate(images2[i], images1[i], flowpy.backward_warp(images2[i], flows[i]))
            plt.imshow(draw_flow(images2[i][:,:,0], flows[i][:,:,:], step=12)), plt.show()
            similarity.append([sm.structural_similarity(images1[i][: , :, 0], mapped[i][:, :, 0], data_range=1)])
    
    print(np.mean(similarity))
    return np.mean(similarity)





from torchvision.io import read_video
from torch import from_numpy
import skimage
def load_image_batches(path, step = 10, smooth = 5):
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
            # print(sm.structural_similarity(img[i, 200:500, 600:900, 0], img[i+1, 200:500, 600:900, 0]))
            # plt.imshow(img1, cmap = 'gray'), plt.show()
            # plt.imshow(img[i+1, 200:500, 600:900, 0], cmap = 'gray'), plt.show()
            # blur = cv2.bilateralFilter(img1, smooth, 160, 160)
            # blurred = np.moveaxis(blur, -1, 0)
            # blurred2 = np.moveaxis(cv2.bilateralFilter(img2,smooth,160,160), -1, 0)
            # print(img1.dtype)
            new_shape = (img1.shape[0] , img1.shape[1] , img1.shape[2])
            blurred1 = skimage.transform.resize(image=img1, output_shape=new_shape).astype(np.float32)
            blurred2 = skimage.transform.resize(image=img2, output_shape=new_shape).astype(np.float32)
            # print(blurred1.shape)

            blurred_1 = np.moveaxis(cv2.bilateralFilter(blurred1,smooth,160,160), -1, 0)
            blurred_2 = np.moveaxis(cv2.bilateralFilter(blurred2,smooth,160,160), -1, 0)

            # print(blurred.shape)
            # plt.imshow(blurred, cmap= 'gray'), plt.show()
            images1.append(torch.from_numpy(blurred_1))
            images2.append(torch.from_numpy(blurred_2))
            # images1.append(frames[i, :, 200:500, 600:900])
            # images2.append(frames[i+1, :, 200:500, 600:900])

    if len(images2) < 1: #when the video is too short/ redundant video
        img1_batch = images1
        img2_batch = images2
    else:
        img1_batch = torch.stack(images1) # making predictions between 2 pairs of frames 53 and 83, and 84 and 130
        img2_batch = torch.stack(images2)

    return img1_batch, img2_batch


from torchvision.io import read_video
from torch import from_numpy
import skimage
from skimage import metrics as sm
import numpy as np
import cv2

def load_image_batches_all(paths, step = 10, smooth = 5):
    # path = "C:/Users/Misha/OneDrive - University of Sussex/FYP/Participants/Participant_12/Processed_data/Video/Subject_12_03.mp4"
    img1_batch_all = []
    img2_batch_all = []

    for num, path in enumerate(paths):
        img1_batch, img2_batch = load_image_batches(path, step, smooth)
        if len(img1_batch) < 1: continue
        img1_batch_all.append(img1_batch)
        img2_batch_all.append(img2_batch)

    return img1_batch_all, img2_batch_all




import glob

# function to extract the paths for files froma path
def load_paths(data_path):
    files = []
    files.append(glob.glob(data_path, 
                recursive = True))
    return files[0]


def flows_all(img1_batch, img2_batch, s = 336):
    
    img1__batch_all = []
    img2__batch_all = []
    flows = []

    for i in range(len(img1_batch)):
        img1__batch, img2__batch = preprocess(img1_batch[i], img2_batch[i], s = 336)
        img1__batch_all.append(img1__batch)
        img2__batch_all.append(img2__batch)
        predicted_flow = raft(img1__batch, img2__batch)
        flows.append(predicted_flow)

    return img1__batch_all, img2__batch_all, flows



print(1)
