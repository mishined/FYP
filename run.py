from raft_functions import *

print("get all the paths")
paths = load_paths("/Users/men22/OneDrive - University of Sussex/FYP/Participants/VIDEOS_ALL/*.mp4")
print(len(paths))
print("get all the images")
img1_batch, img2_batch = load_image_batches_all(paths, 1, 24)

print("get the flow")
img1__batch, img2__batch, flows = flows_all(img1_batch, img2_batch)
print("time for pickle")
import pickle
with open('flows1.pickle','wb') as temp:
  pickle.dump([img1__batch, img2__batch, flows], temp)