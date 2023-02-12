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

## Part 3: Partition the image over columns

The code written for part 2 is modified so that the work is divided based on columns of image, not rows. To do this, same as previous part, the number of working columns for each processor has been computed using the parallelRange function. Then for each processor, a unique vector data type is defined to send a specified number of columns since the number of columns that should be sent to each processor is different.

Image section received by processors

![image](https://user-images.githubusercontent.com/57262710/218326553-ac254241-342a-4eb7-bd2c-31ce1e6072e2.png)

Image sections proccessed by each processors

![image](https://user-images.githubusercontent.com/57262710/218326584-43e2647f-66b3-46b5-8e0f-0b3f5ab41fac.png)

Compare the serial and parallel images

![image](https://user-images.githubusercontent.com/57262710/218326620-a0fe3c09-387d-4451-bb1e-1ca43c1bf600.png)
