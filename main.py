
from mri_dataset import *
from raft_functions import *

if __name__ == '__main__':

    video_dataset = MRIDataset("/Users/men22/OneDrive - University of Sussex/FYP/Participants/VIDEOS_ALL/*.mp4", 
                               transform=None)

    video_dataset_transformed = MRIDataset("/Users/men22/OneDrive - University of Sussex/FYP/Participants/VIDEOS_ALL/*.mp4", 
                               transform=transforms.Compose([Rescale(256),
                               RaftTransforms()]))
    
    print(len(video_dataset_transformed[0][1]))
    # video_dataset.plot(2, 1)
    # plt.show()

    flows = video_dataset_transformed.run_raft()

    print(len(flows))
    print(flows[0].shape)

    evaluate_raft_2(video_dataset_transformed[0][0], video_dataset_transformed[0][1], flows[0])



    # print(video_dataset_transformed.paths)


    