# FYP
Final Year Project - Analysing speech motion from MRI Data

Meeting 0:
- confirmed a project idea and chose a project to continue with
- discussed two perspective the project can go to 
  - super-resolution to improve MRI video quality
  - analysing the motion in the MRI video to find how different people pronounce phonems



Diary
- tried different optical flow techniques - RAFT didnt work
- using Dense Oprical Flow - find the flow of 1 image - improve hyperparameters - which ones and why and what they mean
- change the hyperparameters to current best - 

     cv2.calcOpticalFlowFarneback(prev_gray,
                                    gray,
                                    None,
                                    pyr_scale = 0.5,
                                    levels = 3,
                                    winsize = 10,           
                                    iterations = 6,
                                    poly_n = 5,
                                    poly_sigma = 1.2,
                                    flags = 0)

- Looked at the optical flow and saw some differences
- managed to draw the arrows
- tried remapping the optical flow to the old image to see if it the task is successful or not
- managed to make RAFT work - no
- remap successful 
  - using warping - different results - what do they mean?
- evaluation / loss techniques 
