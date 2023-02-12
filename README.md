# MPI-image-blure
In this project we will be partitioning images over rows and sending pieces of images (collections of rows) to other processes for image processing. The OpenCV library was used to visualized the results and worked with images.

Our approach will be to blur each pixel by setting it equal to the average of the surrounding pixels. The level of the blur will dictate how many pixels we use in the average. We will use a square box of dimension (2 level+1)×(2 level+1) centered on a given pixel to compute the blurred pixel. This way, a level=0 blur corresponds to the original image.

## Part 1: Serial blur function 

Here are some results of the program with different values of Level. As you see in the pictures below, by increasing Level, the picture has become more blurred.
![image](https://user-images.githubusercontent.com/57262710/218325870-4b5d2224-ab50-4530-96f7-b37870f7bf23.png)

## Part 2: Parallel image blure

we are going to blur an image in parallel and compare the timing and results with serial processing. It should be noted that the target image is loaded only in processor zero, and other processors just receive a part of the image that they will calculate.
The program was run with 4 processors and level =5, and pictures below show the results. Proper window names were selected to differentiate between the images produced by the code. Furthermore, the mean square error (MSE) between channel values of the processed image in serial and parallel is calculated to verify the parallel code.

Image section received by processors
![image](https://user-images.githubusercontent.com/57262710/218326126-d5625c99-6d04-485a-90ff-f15ea0eba0ad.png)

Image sections proccessed by each processors
![image](https://user-images.githubusercontent.com/57262710/218326214-d3624ffe-6cd2-48a6-a221-a45aeb0b4d84.png)

Final processed image
![image](https://user-images.githubusercontent.com/57262710/218326297-f4e5b28e-9a47-4d90-ba5e-2a4cd43c21ab.png)
