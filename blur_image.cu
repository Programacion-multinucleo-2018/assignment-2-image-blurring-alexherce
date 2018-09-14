#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <omp.h>

using namespace std;

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

// GPU Blur
__global__ void blur_kernel(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep) {
	
	//2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	if ((xIndex < width) && (yIndex < height)) {
		
		// Location of pixel in output
		const int outputIndex = yIndex * colorWidthStep + (xIndex * 3);
		int blue = 0;
		int green = 0;
		int red = 0;
		int det = 0;

		// Iterate filter matrix
		for (int x = -2; x < 3; x++) {
			for (int y = -2; y < 3; y++) {

				int inputIndex = (y + yIndex) * colorWidthStep + ((x + xIndex) * 3);

				// Check if it is inside borders
				if ((xIndex + x < width) && (yIndex + y < height) && (xIndex + x > 0) && (yIndex + y > 0)) {
					blue += input[inputIndex];
					green += input[inputIndex + 1];
					red += input[inputIndex + 2];
					det++;
				}
			}
		}

		// Store in output. Divide between number of iterations (ideally divide by 25 since its a 5x5 filter)
		output[outputIndex] = static_cast<unsigned char>(blue / det);
		output[outputIndex + 1] = static_cast<unsigned char>(green / det);
		output[outputIndex + 2] = static_cast<unsigned char>(red / det);
	}
}

// CPU OMP Blur
void blur_omp(const cv::Mat& input, cv::Mat& output) {

	int i, j, x, y;

	// Get number of processors
	int nProcessors = omp_get_max_threads();
	std::cout << "CPU processors available: " << nProcessors << std::endl;

	// Set number of processors to use with OpenMP
	omp_set_num_threads(8);

	// Iterate over image
	#pragma omp parallel for private(i, j, x, y) shared(input, output)
	for (i = 0; i < input.rows; i++) {
		for (j = 0; j < input.cols; j++) {

			int blue = 0;
			int green = 0;
			int red = 0;
			int det = 0;

			int inputIndexX = 0;
			int inputIndexY = 0;

			// Iterate filter matrix
			for (x = -2; x < 3; x++) {
				for (y = -2; y < 3; y++) {

					inputIndexX = i + x;
					inputIndexY = j + y;
					
					// Check if it is inside borders
					if ((i + x < input.rows) && (j + y < input.cols) && (i + x > 0) && (j + y > 0)) {
						blue += input.at<cv::Vec3b>(inputIndexX, inputIndexY)[0];
						green += input.at<cv::Vec3b>(inputIndexX, inputIndexY)[1];
						red += input.at<cv::Vec3b>(inputIndexX, inputIndexY)[2];
						det++;
					}
				}
			}
			// Store in output. Divide between number of iterations (ideally divide by 25 since its a 5x5 filter)
			output.at<cv::Vec3b>(i, j)[0] = blue / det;
            output.at<cv::Vec3b>(i, j)[1] = green / det;
            output.at<cv::Vec3b>(i, j)[2] = red / det;
		}
	}
}

// CPU Blur
void blur_cpu(const cv::Mat& input, cv::Mat& output) {

	int i, j, x, y;

	// Iterate over image
	for (i = 0; i < input.rows; i++) {
		for (j = 0; j < input.cols; j++) {

			int blue = 0;
			int green = 0;
			int red = 0;
			int det = 0;

			int inputIndexX = 0;
			int inputIndexY = 0;

			// Iterate filter matrix
			for (x = -2; x < 3; x++) {
				for (y = -2; y < 3; y++) {

					inputIndexX = i + x;
					inputIndexY = j + y;

					// Check if it is inside borders
					if ((i + x < input.rows) && (j + y < input.cols) && (i + x > 0) && (j + y > 0)) {
						blue += input.at<cv::Vec3b>(inputIndexX, inputIndexY)[0];
						green += input.at<cv::Vec3b>(inputIndexX, inputIndexY)[1];
						red += input.at<cv::Vec3b>(inputIndexX, inputIndexY)[2];
						det++;
					}
				}
			}
			// Store in output. Divide between number of iterations (ideally divide by 25 since its a 5x5 filter)
			output.at<cv::Vec3b>(i, j)[0] = blue / det;
			output.at<cv::Vec3b>(i, j)[1] = green / det;
			output.at<cv::Vec3b>(i, j)[2] = red / det;
		}
	}
}

/*
MEMORY ERRORS WITH 4K IMAGE

void blur_cpu(const cv::Mat& input, cv::Mat& output) {

	int i, j, x, y;

	// Iterate over image
	for (i = 0; i < input.cols; i++) {
		for (j = 0; j < input.rows; j++) {
			int blue = 0;
			int green = 0;
			int red = 0;
			int count = 0;

			int output_index = j * input.step + (3 * i);

			// Iterate filter matrix
			for (x = -2; x < 3; x++) {
				for (y = -2; y < 3; y++) {

					int input_index = (y + yIndex) * colorWidthStep + ((x + xIndex) * 3);

					// Check if it is inside borders
					if ((i + x < input.rows) && (j + y < input.cols) && (i + x > 0) && (j + y > 0)) {
						blue += input.data[input_index];
						green += input.data[input_index + 1];
						red += input.data[input_index + 2];
						count++;
					}
				}
			}
			// Store in output. Divide between number of iterations (ideally divide by 25 since its a 5x5 filter)
			output.data[output_index] = static_cast<unsigned char>(blue / count);
			output.data[output_index + 1] = static_cast<unsigned char>(green / count);
			output.data[output_index + 2] = static_cast<unsigned char>(red / count);
		}
	}
}
*/

void blur_image(const cv::Mat& input, cv::Mat& output, cv::Mat& outputOMP, cv::Mat& outputCPU)
{

	// Set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	SAFE_CALL(cudaSetDevice(dev), "Error setting device");

	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;

	// Calculate total number of bytes of input and output image
	// Step = cols * number of colors	
	size_t inputBytes = input.step * input.rows;
	size_t outputBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, inputBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, outputBytes), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), inputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
	const dim3 block(32, 32);

	// Calculate grid size to cover the whole image
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows / block.y));
	
	//const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
	//const dim3 grid(16, 4);

	printf("blur_kernel<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

	// Start timer
	auto start_gpu = chrono::high_resolution_clock::now();
	
	// Launch the color conversion kernel
	blur_kernel << <grid, block >> > (d_input, d_output, input.cols, input.rows, static_cast<int>(input.step));

	// End timer and print result
	auto end_gpu = chrono::high_resolution_clock::now();
	chrono::duration<float, std::milli> duration_ms = end_gpu - start_gpu;
	printf("GPU elapsed %f ms\n", duration_ms.count());

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, outputBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");

	// Blur with OpenMP
	start_gpu = chrono::high_resolution_clock::now();

	blur_omp(input, outputOMP);

	end_gpu = chrono::high_resolution_clock::now();
	duration_ms = end_gpu - start_gpu;
	printf("CPU OMP elapsed %f ms\n", duration_ms.count());

	// Blur with CPU
	start_gpu = chrono::high_resolution_clock::now();

	blur_cpu(input, outputCPU);

	end_gpu = chrono::high_resolution_clock::now();
	duration_ms = end_gpu - start_gpu;
	printf("CPU elapsed %f ms\n", duration_ms.count());
}

int main(int argc, char *argv[])
{
	string imagePath;

	if (argc < 2)
		imagePath = "wallpaper4k.jpg";
	else
		imagePath = argv[1];

	// Read input image from the disk
	cv::Mat input = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

	if (input.empty())
	{
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

	cv::Mat output(input.rows, input.cols, input.type());
	cv::Mat outputCPU(input.rows, input.cols, input.type());
	cv::Mat outputOMP(input.rows, input.cols, input.type());

	//Call the wrapper function
	blur_image(input, output, outputOMP, outputCPU);

	/* ********* DISPLAY IMAGES **********/
	//Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("GPU", cv::WINDOW_NORMAL);
	namedWindow("OpenMP", cv::WINDOW_NORMAL);
	namedWindow("CPU", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("Input", input);
	imshow("GPU", output);
	imshow("OpenMP", outputOMP);
	imshow("CPU", outputCPU);

	//Wait for key press
	cv::waitKey();

	return 0;
}