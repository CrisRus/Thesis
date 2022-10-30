Image segmentation:
Proceess 
* load Image
* Increase contrast, because images are monochomatic (black and white or something similar)
* find the outline using threshold
* draw the outline as a new images
* using floodfill algorithm fill in the outline image from all 4 corners with white pixels
* invert the pixel values of the outline image. This results in the mask for the original image
* substract the outline image from the original image using bitwise operation. The resulting image is the areas of the lungs from the original image
* aggregate the images
* flatten each image into onedimensional array and add them to the CSV file

Classification is done using sklearn python library

Describe the dataset:
What it originally includes, that it was published in kaggle, provide a link, mention the privacy issues regarding medical data
Describe how many images were originally in the dataset, and how many were left after segmentation and cleaning.

Results:
List classification algorithms that were used and describe results for each. 
Show confusion matrices and cross validation results.

Analysis and discussion:
describe why the results are the way they are. 
NB -> too many classes and too many datapoints
LR -> failed to converge, thus larger split in cross validation (+- 6% instead of 4%)
SVC gives the most consistent results although the accuracy is less than anticipated