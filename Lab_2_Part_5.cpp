#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <cassert>
#include <mpi.h>
#include <cmath>
//#include <windows.h>


using namespace std;

void parallelRange(int globalstart, int globalstop, int irank, int nproc, int& localstart, int& localstop, int& localcount)
{
	int nrows = globalstop - globalstart + 1;
	int divisor = nrows / nproc;
	int remainder = nrows % nproc;
	int offset;
	if (irank < remainder) offset = irank;
	else offset = remainder;

	localstart = irank * divisor + globalstart + offset;
	localstop = localstart + divisor - 1;
	if (remainder > irank) localstop += 1;
	localcount = localstop - localstart + 1;
}


void imageBlur(const cv::Mat& in, cv::Mat& out, int level, int colstart, int colstop)
{
	int num_cols = in.cols;
	int num_rows = in.rows;
	int dummy = 0;
	//out = in.clone();
	std::vector<double> channel;
	channel.resize(3);
	double avg = 0;
	double n_pixel = 0;
	for (int icol = colstart; icol < colstop; icol++)
	{
		for (int irow = 0; irow < num_rows; irow++)
		{
			for (int blur_row = irow - level; blur_row < irow + level; blur_row++)
			{
				for (int blur_col = icol - level; blur_col < icol + level; blur_col++)
				{
					if (blur_row >= 0 && blur_row < num_rows && blur_col >= 0 && blur_col < num_cols)
					{
						channel[0] += (double)in.at<cv::Vec3b>(blur_row, blur_col).val[0];
						channel[1] += (double)in.at<cv::Vec3b>(blur_row, blur_col).val[1];
						channel[2] += (double)in.at<cv::Vec3b>(blur_row, blur_col).val[2];
						n_pixel++; // count the number of pixel values added
					}

				}
			}

			if (n_pixel != 0)
			{
				for (int i = 0; i < channel.size(); i++)
				{
					avg = (double)(channel[i] / n_pixel);
					assert(avg <= 255);
					assert(n_pixel < ((2 * level + 1)* (2 * level + 1)));
					out.at<cv::Vec3b>(irow, icol).val[i] = (uchar)avg;
					channel[i] = 0;
				}
				n_pixel = 0;
			}
		}
	}
}


int main(int argc, char** argv)
{

	int rank, nproc;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// create a contiguous data type for one row of the target image by knowing the number of columns
	MPI_Datatype vec3b;
	MPI_Type_contiguous(3, MPI_UNSIGNED_CHAR, &vec3b);
	MPI_Type_commit(&vec3b);

	std::stringstream ss;
	ss << rank;
	std::string srank = ss.str();

	if (rank == 0)
	{
		// define level in processor zero
		int level = 16;

		// load the original image and disply it
		cv::Mat image;
		image = cv::imread("rabbit.jpg", 1);  // Read the file
		if (!image.data) // Check for invalid input
		{
			cout << "Could not open or find the image" << std::endl;
		}
		std::cout << "Image size: " << image.size() << std::endl;

		cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
		cv::imshow("Original Image", image);
		cv::waitKey(1000);

		// extract number of rows and columns of the image
		int m = image.rows; // number of rows
		int n = image.cols; // number of columns

		// create data type handler for a column of image
		MPI_Datatype image_col;


		// blure the image using only processor zero (serial) and calculate timing
		double serial_time_start = 0.0;
		serial_time_start = MPI_Wtime();
		cv::Mat SerialProcessedImage = cv::Mat::zeros(cv::Size(n, m), CV_8UC3);

		imageBlur(image, SerialProcessedImage, level, 0, n);

		double serial_time_stop = MPI_Wtime();
		//------------------------------------------------------------------------------
		// ------------------------------Start parallel bluring------------------------- 
		//------------------------------------------------------------------------------

		MPI_Barrier(MPI_COMM_WORLD);
		double parallel_time_start = MPI_Wtime();
		// create buffer for parallelRange function and information about sending data to other processors
		std::vector<int> localstarts(nproc), localstops(nproc), localcounts(nproc), sendcounts(nproc), sendstarts(nproc), sendstops(nproc);

		// information which will be sent to other processors
		int dataInfo[5] = { 0 };
		dataInfo[2] = m;
		dataInfo[1] = level;
		dataInfo[4] = n;
		// Using parallelRange to specify each processor with specific number of rows to process and all number of receiving rows
		for (int irank = 0; irank < nproc; irank++)
		{
			parallelRange(0, n - 1, irank, nproc, localstarts[irank], localstops[irank], localcounts[irank]);

			if (irank != 0 && irank != nproc - 1)
			{
				sendcounts[irank] = localcounts[irank] + 2 * level;
				sendstarts[irank] = localstarts[irank] - level;
			}
			else if (irank == 0 && nproc != 1)
			{
				sendcounts[irank] = localcounts[irank] + level;
			}
			else if (irank == nproc - 1 && nproc != 1)
			{
				sendcounts[irank] = localcounts[irank] + level;
				sendstarts[irank] = localstarts[irank] - level;
			}
			else if (irank == 0 && nproc == 1)
			{
				sendcounts[irank] = localcounts[irank];
				sendstarts[irank] = localstarts[irank];
			}


			if (sendstarts[irank] < 0)
			{
				sendstarts[irank] = 0;
			}

			if (localcounts[irank] > n)
			{
				localcounts[irank] = n;
				sendcounts[irank] = n;
			}

			dataInfo[0] = sendcounts[irank];
			dataInfo[3] = localcounts[irank];

			if (irank != 0)
			{
				// send data information to other processors
				MPI_Send(&dataInfo[0], 5, MPI_INT, irank, 2, MPI_COMM_WORLD);
			}

		}

		int allColscount = sendcounts[0]; //dataInfo[0]
		int myColscount = localcounts[0]; //dataInfo[3]

		// create buffers for subimage and sub processing image
		cv::Mat myProcessedImage;
		cv::Mat myImage;
		myImage = image(cv::Rect(0, 0, allColscount, m));

		// distribute the image to all processors
		for (int irank = 1; irank < nproc; irank++)
		{
			// create a vector data type for n columns of the target image and send it to other processors at once
			MPI_Type_vector(m, sendcounts[irank], n, vec3b, &image_col);
			MPI_Type_commit(&image_col);
			MPI_Send(&image.data[sendstarts[irank] * 3], 1, image_col, irank, 99, MPI_COMM_WORLD);
		}

		// blur the subimage
		cv::Mat fullProcessedImage = cv::Mat::zeros(cv::Size(n, m), CV_8UC3);
		imageBlur(myImage, fullProcessedImage, level, 0, myColscount);
		myProcessedImage = fullProcessedImage(cv::Rect(0, 0, allColscount, m));

		// gather the all pieces of blured image from all processors 
		for (int irank = 1; irank < nproc; irank++)
		{
			MPI_Type_vector(m, localcounts[irank], n, vec3b, &image_col);
			MPI_Type_commit(&image_col);
			MPI_Recv(&fullProcessedImage.data[localstarts[irank] * 3], 1, image_col, irank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		// finish parallel processing
		MPI_Barrier(MPI_COMM_WORLD);
		double parallel_time_stop = MPI_Wtime();

		// display the results
		std::cout << "I am " << rank << ", myColscount = " << myColscount << ", allColscount = " << allColscount << ", myImage size = " << myImage.size() << std::endl;

		double serial_time = serial_time_stop - serial_time_start;
		double parallel_time = parallel_time_stop - parallel_time_start;

		std::cout << "Serial Time = " << serial_time << " seconds." << std::endl;
		std::cout << "Parallel Time = " << parallel_time << " seconds." << std::endl;
		std::cout << "Speedup = " << serial_time / parallel_time << std::endl;

		// calculate the MSE between processed image in Serial and Parallel
		double sum = 0;
		for (int irow = 0; irow < m; irow++)
		{
			for (int icol = 0; icol < n; icol++)
			{
				sum += pow((double)SerialProcessedImage.at<cv::Vec3b>(irow, icol).val[0] - (double)fullProcessedImage.at<cv::Vec3b>(irow, icol).val[0], 2);
				sum += pow((double)SerialProcessedImage.at<cv::Vec3b>(irow, icol).val[1] - (double)fullProcessedImage.at<cv::Vec3b>(irow, icol).val[1], 2);
				sum += pow((double)SerialProcessedImage.at<cv::Vec3b>(irow, icol).val[2] - (double)fullProcessedImage.at<cv::Vec3b>(irow, icol).val[2], 2);
			}
		}
		double MSE = sum / ((double)(n) * (double)(m) * 3);
		std::cout << "MSE between processed image in Serial and Parallel = " << MSE << std::endl;

		// save the subimage and sub processed image into different file
		std::string imageName = "Sub-image in processor " + srank;
		//cv::namedWindow(imageName, cv::WINDOW_AUTOSIZE);
		//cv::imshow(imageName, myImage);
		cv::imwrite(imageName + " .jpg", myImage);

		imageName = "Processed image in processor " + srank;
		//cv::namedWindow(imageName, cv::WINDOW_AUTOSIZE);
		//cv::imshow(imageName, myProcessedImage);
		cv::imwrite(imageName + " .jpg", myProcessedImage);

		// display the processed image in serial and parallel
		cv::namedWindow("Full processed image in parallel", cv::WINDOW_AUTOSIZE);
		cv::imshow("Full processed image in parallel", fullProcessedImage);

		cv::namedWindow("Full processed image in serial", cv::WINDOW_AUTOSIZE);
		cv::imshow("Full processed image in serial", SerialProcessedImage);
		cv::waitKey(10);

	}
	else
	{
		MPI_Barrier(MPI_COMM_WORLD);
		int dataInfo[5];
		// receive required information
		MPI_Recv(&dataInfo[0], 5, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		// translate the received information
		int allColscount = dataInfo[0];
		int level = dataInfo[1];
		int m = dataInfo[2];
		int myColscount = dataInfo[3];
		int n = dataInfo[4];


		// create buffer for received subimage and processed subimage
		cv::Mat myImage = cv::Mat::zeros(cv::Size(allColscount, m), CV_8UC3);
		cv::Mat myProcessedImage = cv::Mat::zeros(cv::Size(allColscount, m), CV_8UC3);

		// receive the subimage 
		MPI_Datatype image_col;
		MPI_Type_vector(m, allColscount, allColscount, vec3b, &image_col);
		MPI_Type_commit(&image_col);
		MPI_Recv(&myImage.data[0], 1, image_col, 0, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		// blur the subimage
		imageBlur(myImage, myProcessedImage, level, level, myColscount + level);

		// send the processed subimage to the processor zero
		MPI_Type_vector(m, myColscount, allColscount, vec3b, &image_col);
		MPI_Type_commit(&image_col);
		MPI_Send(&myProcessedImage.data[level * 3], 1, image_col, 0, 3, MPI_COMM_WORLD);

		MPI_Barrier(MPI_COMM_WORLD);

		// display the results
		std::cout << "I am " << rank << ", myColscount = " << myColscount << ", allColscount = " << allColscount << ", myImage size = " << myImage.size() << std::endl;

		string imageName = "Sub-image in processor " + srank;

		// save the subimage and sub processed image into different file
		//cv::namedWindow(imageName, cv::WINDOW_AUTOSIZE);	
		//cv::imshow(imageName, myImage);
		cv::imwrite(imageName + " .jpg", myImage);

		imageName = "Processed image in processor " + srank;
		//cv::namedWindow(imageName, cv::WINDOW_AUTOSIZE);	
		//cv::imshow(imageName, myProcessedImage);
		//cv::waitKey(10);
		cv::imwrite(imageName + " .jpg", myProcessedImage);

	}

	MPI_Barrier(MPI_COMM_WORLD);		//synchronize
	MPI_Finalize();

}
