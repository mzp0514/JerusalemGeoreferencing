# Jerusalem 1840-1949 Historical Maps Georeferencing

## Motivation

The creation of large digital databases on urban development is a strategic challenge, which could lead to new discoveries in urban planning, environmental sciences, sociology, economics, and in a considerable number of scientific and social fields. Digital geohistorical data can also be used and valued by cultural institutions. These historical data could also be studied to better understand and optimize the construction of new infrastructures in cities nowadays, and provide humanities scientists with accurate variables that are essential to simulate and analyze urban ecosystems. Now there are many geographic information system platforms that can be directly applied, such as QGIS, ARCGIS, etc. How to digitize and standardize geo-historical data has become the focus of research. We hope to propose a model that can associate geographic historical data with today's digital maps, analyze and study them under the same geographic information platform, same coordinate projection, and the same scale. Eventually it can eliminate errors caused by scaling, rotation, and the deformation of the map carrier that may exist in historical data and the entire process is automated and efficient.

The scale is restricted to Jerusalem in our project. Jerusalem is one of the oldest cities in the world and is considered holy to the three major Abrahamic religions—Judaism, Christianity, and Islam. We did georeferencing among Jerusalem’s historical maps from 1840 to 1949 and the modern map from OpenStreetMap so that the overlaid maps will reveal changes over time and enable map analysis and discovery. We focused on the wall of the Old City as the morphological feature to do georeferencing because the region outside the Old City has seen many new constructions while the Old City has not great changes and the closed polygon of the wall is relatively more consistent than other features like road networks. More specifically, we used dhSegment, a historical document segmentation tool, to extract the wall of the Old City of Jerusalem and proposed an alignment algorithm exploiting the geometrical features of the wall to align the maps.

## Usage

```
pip install -r requirements.txt
```

```
python align.py
```




## Methodology

### Dataset

- 126 historical maps of Jerusalem from 1837 to 1938. (https://drive.google.com/drive/folders/0AJMw_BIszSkYUk9PVA)

- Modern geographical data of Jerusalem from OpenStreetMap.

 

### Wall Extraction

We first used dhSegment to do image segmentation in order to extract the wall polygon from the maps. dhSegment [1] is a generic approach for Historical Document Processing, which relies on a Convolutional Neural Network to predict pixelwise characteristics. The whole process include preprocessing, training, predicting and postprocessing.

 

#### Preprocessing

To make the image data fit into the neural networks and to make the old city region more dominant in the image, we cropped and scaled the original image by hand for subsequent uses.

 

#### Map annotating

We used Procreate to pixel-wisely label 46 maps according to whether the pixel belong to the old city with different colors (RGB(0, 0, 0) for pixels inside the wall, RGB(255, 0, 255) for pixels outside the wall). 



#### Training

We divided the 46 annotated patches into 36 and 6 for training and validating respectively, and trained the model with learning rate 5e-5, batch size: 2, N_epochs: 40, and data augmentation like rotation, scaling and color.

 

#### Predicting

By doing prediction on the testing set, we got the predicted old city region and naturally got the contour of it, i.e. the wall.

 

#### Postprocessing

We removed the minor noises by retain the largest connected region and drew the contour of it. We put the contour on the original image and find that it generally fits well, although there are some minor deviations.

 

### Wall Alignment

After obtaining the wall polygons, we try to fit them to the the reference wall polygon from the OpenStreetMap and obtain the coordinates of the source polygon as well as the four vertices of the source image in the coordinate system of the reference image.

#### Overview

The overall algorithm proposed is shown as follows. We do scaling, transformation and rotation iteratively for a given iteration numbers in order to get a better alignment result since the initial scaling isn't the accurate enough with the angle difference of the source polygon and reference polygon.



For $i = 0$ to $iters$:

​		Estimate and do scaling

​		For each point pair $P$ in selected point pair set:

​				Calculate and do transformation

​				For $k = -6$ to $6$:

​						Do rotation of angle = $k(iters - i)$ around $P$

​						Calculate overlapping area

​						Update maximum overlapping area

 

#### Scaling

The method to do scaling is shown in the image below. 

 

#### Translation & Rotation

The goal of the translation and rotation is to obtain optimal translation and rotation to maximize the overlapping area of the two polygons. Since it’s computationally expensive to search for the translation pixel by pixel, and the angle degree by degree, we here narrow down the search space to translations that can fit the key point pairs respectively together (up-left points, up-right points, bottom-left points, bottom-right points and the centroids) and rotation angle set, which starts with [-30, 30] of step 5 and whose range and step will get smaller after every iteration (it's intuitive since after one iteration, we will need more accurate rotation angles).

For every translation, we fix the key point together, rotate the wall and calculate the overlapping area. To calculate the overlapping area, we use the FloodFill algorithm to fill the area bounded by the source polygon and reference polygon respectively and use the logical-and operation to obtain the overlapping region. We then count the number of pixels that are not zero to be the overlapping area. 

After the iterations, we get the maximized overlapping area and the coordinates of the source polygon as well as the four vertices of the source image in the coordinate system of the reference image.



### Coordinate System Transformation

Since the final goal is to get the latitudes and longitudes in the geographic coordinate system of the four vertices of the raw maps, we finally transformed the coordinates of the image patches in the reference image coordinate system to the latitudes and longitudes of the raw maps through mathematical computations. 





### Coordinate System Transformation

The results are stored in a csv file shown as follows.

(P16)



## Limitations

-  Although the alignment algorithm is not sensitive to the minor flaws of the segmentation results, the alignment performance is still subject to the segmentation results. Specifically, when the segmentation gives very bad predictions, the alignment algorithm just can't work.
- The project can only do linear transformation, i.e. scaling, translation and rotation, but it can't do deformation.
- Only the closed polygon of the wall are utilized and segmented.  The morphological feature of road networks and buildings can also be exploited in future works.



## References

[1] Oliveira, Sofia Ares, Benoit Seguin, and Frederic Kaplan. "dhSegment: A generic deep-learning approach for document segmentation." 2018 16th International Conference on Frontiers in Handwriting Recognition (ICFHR). IEEE, 2018.