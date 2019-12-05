# CSE 5524 Project

## Team member:
  
Mengxi Zhou (.2656)
  
Jincheng Li (.7004)
  
Yun Jiang (.1982)

## Problem definition:
    
Given a dataset consisting of cat images and an unseen input cat image, try to find the top 5 images in the dataset that are most similar to the input image in terms of the posture.

## Proposed approach:
    
Get a relatively large dataset (aims for 500+ cat images).
    
Utilize techniques, such as edge detections, thresholding-based segmentation, or some clustering methods (if we’re allowed to use, say k-means clustering), to do image segmentation. This gives the approximated region containing the cat.
Find the interest points within the “cat region”. Do this for every image in the dataset and the target image.
Do Difference-of-Gaussians (DoG) to get scale-invariant regions.
Build SIFT descriptors.
    
For each image pair (one image from the dataset and the new image), do Hungarian algorithm to find the best matching between these two images and assign a similarity score to the old image (the one from the dataset). Or maybe use other greedy algorithms to only find the top, say 30, best matching interest points, in case the overall interest points matching is not desired.
Find the top 5 images with the highest similarity scores from the dataset.

## Dataset:
    
A lot of datasets are publicly available for cat images. Such as https://www.kaggle.com/crawford/cat-dataset
Also, our team member Yun has a personal collection of cat images.

## Work Arrangement:
    
Following our proposed approach, each member in the team will take care of:
    
Mengxi: 2. Image Segmentation and 6. Matching algorithm
Jincheng: 5. SIFT
Yun: 4. DoG
    
Once the whole pipeline is built, other general tasks, such as parameter tuning and result reasoning, will first be done by each team member individually. And finally, we’ll combine the results together.

## Evaluation:
    
The evaluation will be straightforward, we can directly see from the output whether the extracted images have similar postures to the input image or not, and how it comes (we can probably interpret this from the SIFT descriptors and matching points).
