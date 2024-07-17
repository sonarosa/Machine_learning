#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
using namespace cv;
using namespace std;
// Function to load an image
Mat loadImage(const string &filename) {
    Mat image = imread(filename, IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error: Unable to load image " << filename << endl;
        exit(1);
    }
    return image;
}
// Function to display an image
void displayImage(const Mat &image, const string &windowName) {
    namedWindow(windowName, WINDOW_AUTOSIZE);
    imshow(windowName, image);
    waitKey(0);
}
// Function to save an image
void saveImage(const Mat &image, const string &filename) {
    if (!imwrite(filename, image)) {
        cerr << "Error: Unable to save image " << filename << endl;
    }
}
// Function to apply convolution using OpenMP
void applyConvolution(const Mat &src, Mat &dst, const Mat &kernel) {
    int kernelRows = kernel.rows;
    int kernelCols = kernel.cols;
    int halfKernelRows = kernelRows / 2;
    int halfKernelCols = kernelCols / 2;
    #pragma omp parallel for collapse(2)
    for (int i = halfKernelRows; i < src.rows - halfKernelRows; i++) {
        for (int j = halfKernelCols; j < src.cols - halfKernelCols; j++) {
            Vec3f sum = Vec3f(0, 0, 0);
            for (int m = -halfKernelRows; m <= halfKernelRows; m++) {
                for (int n = -halfKernelCols; n <= halfKernelCols; n++) {
                    Vec3b pixel = src.at<Vec3b>(i + m, j + n);
                    float value = kernel.at<float>
                    (m + halfKernelRows, n + halfKernelCols);
                    sum[0] += pixel[0] * value;
                    sum[1] += pixel[1] * value;
                    sum[2] += pixel[2] * value;
                }
            }
            dst.at<Vec3b>(i, j) = Vec3b(sum[0], sum[1], sum[2]);
        }
    }
}
// Function to measure execution time
double measureExecutionTime(void (*func)(const Mat&, Mat&, const Mat&),
const Mat &src, Mat &dst, const Mat &kernel) {
    double start = omp_get_wtime();
    func(src, dst, kernel);
    double end = omp_get_wtime();
    return end - start;
}
int main() {
    // Load image
    string filename = "input.jpg";
    Mat image = loadImage(filename);
    displayImage(image, "Original Image");
    // Define kernels
    Mat blurKernel = (Mat_<float>(3, 3) << 1/9.0, 1/9.0, 1/9.0, 
    1/9.0, 1/9.0, 1/9.0, 1/9.0, 1/9.0, 1/9.0);
    Mat sharpenKernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5,
    -1, 0, -1, 0);
    Mat edgeDetectionKernel = (Mat_<float>(3, 3) << -1, -1, -1,
    -1, 8, -1, -1, -1, -1);
    // Apply and measure convolution operations
    Mat blurImage = image.clone();
    Mat sharpenImage = image.clone();
    Mat edgeImage = image.clone();
    double blurTime = measureExecutionTime(applyConvolution, 
    image, blurImage, blurKernel);
    double sharpenTime = measureExecutionTime(applyConvolution, 
    image, sharpenImage, sharpenKernel);
    double edgeTime = measureExecutionTime(applyConvolution, 
    image, edgeImage, edgeDetectionKernel);
    // Display results
    displayImage(blurImage, "Blurred Image");
    displayImage(sharpenImage, "Sharpened Image");
    displayImage(edgeImage, "Edge Detection Image");
    // Save results
    saveImage(blurImage, "blurred.jpg");
    saveImage(sharpenImage, "sharpened.jpg");
    saveImage(edgeImage, "edgedetection.jpg");
    // Print performance analysis
    cout << "Execution Time (with parallelization):\n";
    cout << "Blurring: " << blurTime << " seconds\n";
    cout << "Sharpening: " << sharpenTime << " seconds\n";
    cout << "Edge Detection: " << edgeTime << " seconds\n";
    return 0;
}
!pip install nvcc4jupyter
%load_ext nvcc4jupyter
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
