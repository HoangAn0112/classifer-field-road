## Objective
Create a two class classifier : Field & Road using a small dataset 

## Data: 
This dataset has 153 images for training and 10 images for testing, which is relatively small. Images within the same group are quite diverse, while some from different groups could look quite the same (example: fields images contain dirt paths between crop rows) (1)

The classes are imbalanced - road samples are almost double field samples, and 2 images have wrong labels (fields instead of roads) (2)

Also, samples don’t have the same resolution and format (3)

## Method:
With a computer vision task like this my intuitive idea was using CNNs, which are easy to deal with images data, efficient and have been the cornerstone of many breakthroughs in this field. I chose to work with PyTorch framework and to solve the listed problems, I propose these methods:
(3) Image preprocessing: using `torchvision` to input different formats


