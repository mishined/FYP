
from mri_dataset import *
from raft_functions import *
import pickle


if __name__ == '__main__':

    # video_dataset = MRIDataset("/Users/men22/OneDrive - University of Sussex/FYP/Participants/VIDEOS_ALL/*.mp4", 
                            #    transform=None)

    video_dataset_transformed = MRIDataset("/Users/men22/OneDrive - University of Sussex/FYP/Participants/VIDEOS_ALL/*.mp4", 
                               transform=transforms.Compose([Rescale(256),
                               RaftTransforms()]))
    
    # print((video_dataset_transformed[2][0]))
    # print(isinstance(video_dataset_transformed[2][0], list))

    # video_dataset_transformed.plot(0, 1)
    # plt.show()

    # flows = video_dataset_transformed.run_raft()

    # print(len(flows))
    # print(flows[0].shape)

    # evaluate_raft_2(video_dataset_transformed[0][0], video_dataset_transformed[0][1], flows[0])


    from torch.utils.data import Subset
    data_subset = Subset(video_dataset_transformed, range(1280, 1370))
    del video_dataset_transformed

    # print(video_dataset_transformed.paths)

    video_dataloader = get_dataloader(data_subset, batch_size=1)

    # print(len(data_subset))

    # for i, videos in enumerate(video_dataloader):
    #     sample1, sample2 = videos
    #     print(type(sample1))
    flows = []
    for i, videos in enumerate(video_dataloader):

        # print((videos[0].shape))
        sample1, sample2 = videos
        del videos
        if isinstance(sample1, list): continue
        print(i)
        # print(type(sample1))
        # if isinstance(sample1[0], list): continue
        flow = raft(sample1[0], sample2[0])
        del sample1, sample2
        flows.append(flow)

    print(len(flows))
    
    # # with open('flows2.pickle','wb') as temp:
    # #     pickle.dump(flows, temp)

    import pickle
    with open('flows16.pickle','wb') as temp:
        pickle.dump([data_subset, flows], temp)

#flows 3 onwards are all