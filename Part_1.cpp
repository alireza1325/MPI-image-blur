#include <iostream>
#include <opencv2/core/core.hpp>            //changes may be required to opencv includes
#include <opencv2/highgui/highgui.hpp>
#include <vector>
//#include <mpi.h>

using namespace std;



void imageBlur(const cv::Mat& in, cv::Mat& out, int level, int rowstart, int rowstop)
{
    int num_cols = in.cols;
    int dummy = 0;
    out = in.clone();
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
                    if (blur_row >= rowstart && blur_row <= rowstop && blur_col >= 0 && blur_col < in.cols )
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
            //out.at<cv::Vec3b>(irow, icol) = in.at<cv::Vec3b>(irow, icol);
        }
    }
}


int main(int argc, char** argv)
{
    cv::Mat image;
    image = cv::imread("rabbit.jpg", 1);  // Read the file
    if (!image.data) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
    }
    else {

        cv::Mat processed_image;
        imageBlur(image, processed_image, 5, 0, image.rows-1);
        

        cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);	// Create a window for display.
        cv::imshow("Original Image", image);                   				// Show our image inside it.

        cv::namedWindow("Blurred Image", cv::WINDOW_AUTOSIZE);	// Create display window.
        cv::imshow("Blurred Image", processed_image);      // Show our image inside it.


        cv::waitKey(0);	//wait 10 seconds before closing image (or a keypress to close)


    }
}
