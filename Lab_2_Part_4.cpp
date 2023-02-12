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


void imageBlur(const cv::Mat& in, cv::Mat& out, int level, int rowstart, int rowstop)
{
	int num_cols = in.cols;
	int num_rows = in.rows;
	int dummy = 0;
	//out = in.clone();
	std::vector<double> channel;
	channel.resize(3);
	double avg = 0;
	double n_pixel = 0;
	for (int irow = rowstart; irow < rowstop; irow++)
	{
		for (int icol = 0; icol < num_cols; icol++)
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
		int level = 2;

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

		// create a contiguous data type for one row of the target image by knowing the number of columns
		MPI_Datatype image_row;
		MPI_Type_contiguous(n, vec3b, &image_row);
		MPI_Type_commit(&image_row);

		// blure the image using only processor zero (serial) and calculate timing
		double serial_time_start = 0.0;
		serial_time_start = MPI_Wtime();
		cv::Mat SerialProcessedImage = cv::Mat::zeros(cv::Size(n, m), CV_8UC3);

		imageBlur(image, SerialProcessedImage, level, 0, m);

		double serial_time_stop = MPI_Wtime();
		//------------------------------------------------------------------------------
		// ------------------------------Start parallel bluring------------------------- 
		//------------------------------------------------------------------------------

		MPI_Barrier(MPI_COMM_WORLD);
		double parallel_time_start = MPI_Wtime();
		// create buffer for parallelRange function and information about sending data to other processors
		std::vector<int> localstarts(nproc), localstops(nproc), localcounts(nproc), sendcounts(nproc), sendstarts(nproc), sendstops(nproc);

		// information which will be sent to other processors
		int dataInfo[4] = { 0 };
		dataInfo[2] = n;
		dataInfo[1] = level;

		// Using parallelRange to calculate specific number of rows to be send to processors and all number of receiving rows
		for (int irank = 0; irank < nproc; irank++)
		{
			parallelRange(0, m - 1, irank, nproc, localstarts[irank], localstops[irank], localcounts[irank]);

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

			if (localcounts[irank] > m)
			{
				localcounts[irank] = m;
				sendcounts[irank] = m;
			}

			dataInfo[0] = sendcounts[irank];
			dataInfo[3] = localcounts[irank];

			if (irank != 0)
			{
				// send data information to other processors
				MPI_Send(&dataInfo[0], 4, MPI_INT, irank, 99, MPI_COMM_WORLD);
			}

		}

		int allRowscount = sendcounts[0]; //dataInfo[0]
		int myRowscount = localcounts[0]; //dataInfo[3]

		// create buffers for subimage and sub processing image
		cv::Mat myImage = cv::Mat::zeros(cv::Size(n, allRowscount), CV_8UC3);
		cv::Mat myProcessedImage = cv::Mat::zeros(cv::Size(n, allRowscount), CV_8UC3);

		// distribute the image to all processors
		MPI_Scatterv(&image.data[0], &sendcounts[0], &sendstarts[0], image_row, &myImage.data[0], allRowscount, image_row, 0, MPI_COMM_WORLD);

		// blur the subimage
		imageBlur(myImage, myProcessedImage, level, 0, myRowscount);

		// gather the all pieces of blured image from all processors including processer zero
		cv::Mat fullProcessedImage = cv::Mat::zeros(cv::Size(n, m), CV_8UC3);
		MPI_Gatherv(&myProcessedImage.data[0], myRowscount, image_row, &fullProcessedImage.data[0], &localcounts[0], &localstarts[0], image_row, 0, MPI_COMM_WORLD);

		// finish parallel processing
		MPI_Barrier(MPI_COMM_WORLD);
		double parallel_time_stop = MPI_Wtime();

		// display the results
		std::cout << "I am " << rank << ", myRowscount = " << myRowscount << ", allRowscount = " << allRowscount << ", myImage size = " << myImage.size() << std::endl;
		
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
		double MSE = sum / ((double)(n) * (double)(m)*3);
		std::cout << "MSE between processed image in Serial and Parallel = " << MSE << std::endl;

		std::string imageName = "Sub-image in processor " + srank;
		cv::namedWindow(imageName, cv::WINDOW_AUTOSIZE);	
		cv::imshow(imageName, myImage);                   				

		imageName = "Processed image in processor " + srank;
		cv::namedWindow(imageName, cv::WINDOW_AUTOSIZE);	
		cv::imshow(imageName, myProcessedImage);

		cv::namedWindow("Full processed image in parallel", cv::WINDOW_AUTOSIZE);	
		cv::imshow("Full processed image in parallel", fullProcessedImage);

		cv::namedWindow("Full processed image in serial", cv::WINDOW_AUTOSIZE);
		cv::imshow("Full processed image in serial", SerialProcessedImage);

		cv::waitKey(0);


	}
	else
	{
		MPI_Barrier(MPI_COMM_WORLD);
		int dataInfo[4];
		// receive required information
		MPI_Recv(&dataInfo[0], 4, MPI_INT, 0, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		// translate the received information
		int allRowscount = dataInfo[0];
		int level = dataInfo[1];
		int n = dataInfo[2];
		int myRowscount = dataInfo[3];

		// create a contiguous data type for one row of the target image by knowing the number of columns
		MPI_Datatype image_row;
		MPI_Type_contiguous(n, vec3b, &image_row);
		MPI_Type_commit(&image_row);

		// create buffer for received subimage and processed subimage
		cv::Mat myImage = cv::Mat::zeros(cv::Size(n, allRowscount), CV_8UC3);
		cv::Mat myProcessedImage = cv::Mat::zeros(cv::Size(n, allRowscount), CV_8UC3);

		// receive the subimage 
		MPI_Scatterv(NULL, NULL, NULL, NULL, &myImage.data[0], allRowscount, image_row, 0, MPI_COMM_WORLD);


		// blur the subimage
		imageBlur(myImage, myProcessedImage, level, level, myRowscount + level);

		// send the processed subimage to the processor zero
		MPI_Gatherv(&myProcessedImage.data[level * n * 3], myRowscount, image_row, NULL, NULL, NULL, NULL, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);

		// display the results
		std::cout << "I am " << rank << ", myRowscount = " << myRowscount << ", allRowscount = " << allRowscount << ", myImage size = " << myImage.size() << std::endl;

		string imageName = "Sub-image in processor " + srank;

		cv::namedWindow(imageName, cv::WINDOW_AUTOSIZE);	// Create a window for display.
		cv::imshow(imageName, myImage);

		imageName = "Processed image in processor " + srank;
		cv::namedWindow(imageName, cv::WINDOW_AUTOSIZE);	// Create a window for display.
		cv::imshow(imageName, myProcessedImage);
		cv::waitKey(100);
	}


	MPI_Barrier(MPI_COMM_WORLD);		//synchronize
	MPI_Finalize();

}
