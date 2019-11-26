
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys\timeb.h>
#include "main.h"
#include "lodepng.h"
// For an M (rows) by N (columns) maze
# define M 199
# define N 199

void postProcessing(char* inputImage);

__device__ bool insertPoint(Point array[2 * (N < M ? N : M)], Point point);

__global__ void checkPoint(Point* d_points, Point* d_new_points, unsigned char* image, unsigned nb_threads, int* insert_index, int run_num) //bool* maze
{
	// shared array to hold resulting new points to visit for next iteration
	__shared__ Point new_points[2 * (N < M ? N : M)];

	for (int index = (blockIdx.x * nb_threads + threadIdx.x); index < N; index += nb_threads) {

		new_points[index] = Point(-1, -1);

		__syncthreads();

		// If Point is empty (ie, still at initial Point(-1,-1) value), skip process but still copy back results
		if (d_points[index].getR() >= 0) {
			// Get row and col for current point
			int row = d_points[index].getR();
			int col = d_points[index].getC();

			// If point above is in bounds and not a wall
			if (row - 1 >= 0 && image[(row - 1) * 4 * N + col * 4 + 2] > 250) {
				// Insert in shared array and get insertion index
				if (insertPoint(new_points, Point((row - 1), col))) {
					new_points[atomicAdd(&insert_index[0], 1)] = Point((row - 1), col);
					image[(row - 1) * 4 * N + col * 4] = 255 - run_num;
					image[(row - 1) * 4 * N + col * 4 + 1] = 0;
					image[(row - 1) * 4 * N + col * 4 + 2] = 0;
				}
			}

			
			// If point to the left is in bounds and not a wall
			if (col - 1 >= 0 && image[row * 4 * N + (col - 1) * 4 + 2] > 250) {
				// Insert in shared array and get insertion index
				if (insertPoint(new_points, Point(row, (col - 1)))) {
					new_points[atomicAdd(&insert_index[0], 1)] = Point(row, (col - 1));
					image[row * 4 * N + (col - 1) * 4] = 255 - run_num;
					image[row * 4 * N + (col - 1) * 4 + 1] = 0;
					image[row * 4 * N + (col - 1) * 4 + 2] = 0;
				}
			}

			// If point below is in bounds and not a wall
			if (row + 1 < M  && image[(row + 1) * 4 * N + col * 4 + 2] > 250) {
				// Insert in shared array and get insertion index
				if (insertPoint(new_points, Point((row + 1), col))) {
					new_points[atomicAdd(&insert_index[0], 1)] = Point((row + 1), col);
					image[(row + 1) * 4 * N + col * 4] = 255 - run_num;
					image[(row + 1) * 4 * N + col * 4 + 1] = 0;
					image[(row + 1) * 4 * N + col * 4 + 2] = 0;
				}
			}

			// If point to the left is in bounds and not a wall
			if (col + 1 < N && image[row * 4 * N + (col + 1) * 4 + 2] > 250) {
				// Insert in shared array and get insertion index
				if (insertPoint(new_points, Point(row, (col + 1)))) {
					new_points[atomicAdd(&insert_index[0], 1)] = Point(row, (col + 1));
					image[row * 4 * N + (col + 1) * 4] = 255 - run_num;
					image[row * 4 * N + (col + 1) * 4 + 1] = 0;
					image[row * 4 * N + (col + 1) * 4 + 2] = 0;
				}
			}
		}

		// Synchronize threads and copy shared array into result array
		__syncthreads();
		d_new_points[index] = new_points[index];
		d_new_points[index + (N < M ? N : M)] = new_points[index + (N < M ? N : M)];
	}
}

// Insert given point at the first available position in the given array (avoiding duplicate points)
__device__ bool insertPoint(Point array[2 * (N < M ? N : M)], Point point) {
	int i;
	// Cycle through array points until the end or an empty point (ie, still at initial Point(-1,-1) value) is reached
	for (i = 0; i < 2 * (N < M ? N : M) && array[i].getR() >= 0; i++) {
		// If duplicate point found (ie, point we want to insert is already in the array) do nothing and return
		if (point.getR() == array[i].getR() && point.getC() == array[i].getC()) {
			return false;
		}
	}
	if (i < 2 * (N < M ? N : M)) return true;
	return false;
}

int main(int argc, char* argv[])
{
	struct timeb start_time, end_time, cuda_start, cuda_end;
	ftime(&start_time);
	const int diagonalSize = 2 * (N < M ? N : M);

	if (argc < 2) {
		printf("Invalid arguments! Usage: ./ParallelMazeSolver <name of input png> (optional)<number of threads>\n");
		return -1;
	}

	char* input_filename = argv[1];
	double cuda_total = 0, total_time = 0;
	unsigned total_threads = diagonalSize;
	if (argc == 3) total_threads = atoi(argv[2]);
	unsigned nb_threads = total_threads;
	unsigned nb_blocks = 1;
	bool pathFound = false;

	// Max threads per block is 1024
	while (nb_threads > 1024) {
		nb_blocks++;
		nb_threads = total_threads / nb_blocks;
	}

	unsigned error;
	unsigned char* image, * image_copy;
	unsigned image_width, image_height;

	// Decode image
	error = lodepng_decode32_file(&image, &image_width, &image_height, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error)); 

	printf("Input file: %s, maze width: %d, maze height: %d\n", input_filename, image_width, image_height);
	printf("Number of blocks: %d, number of threads: %d\n", nb_blocks, nb_threads);

	// Check that maze is non-empty
	if (image_width * image_height != 0) {

		// Array of points to be visited at each iteration, initialized with all Point(-1,-1) entries
		Point points[diagonalSize];
		// Set first point to be visited - the arrival point (because we are backtracking), the last point of the maze (assuming square maze)
		points[0] = Point(M - 1, N - 1);

		cudaMallocManaged((void**)& image_copy, image_width * image_height * 4 * sizeof(unsigned char));
		cudaMemcpy(image_copy, image, image_width * image_height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);

		// Temporary points matrix, initialized with  all Point(-1,-1) entries
		Point temp_points[N];

		// Cuda copies of the points to be visited and resulting new points to visit for next iteration
		Point* d_points, * d_new_points;

		cudaMallocManaged((void**)& d_points, diagonalSize * sizeof(Point));
		cudaMallocManaged((void**)& d_new_points, diagonalSize * sizeof(Point));

		int* d_data;
		cudaMalloc((void**)& d_data, 1 * sizeof(int));
		cudaMemset(d_data, 0, 1 * sizeof(int));

		int run_num = 0;

		// While there are still points to visit
		while (points[0].getR() != -1) {

			if (points[0].getR() == 0 && points[0].getC() == 0) {
				pathFound = true;
				break;
			}

			cudaMemcpy(d_points, &points, diagonalSize * sizeof(Point), cudaMemcpyHostToDevice);
			cudaMemcpy(d_new_points, &temp_points, diagonalSize * sizeof(Point), cudaMemcpyHostToDevice);

			ftime(&cuda_start);

			cudaMemset(d_data, 0, 1 * sizeof(int));

			// Call to device function with N threads (at most N points), points to be visited, and array to hold resulting new points to visit for next iteration
			checkPoint << <nb_blocks, nb_threads >> > (d_points, d_new_points, image_copy, nb_threads, d_data, run_num); //d_maze

			cudaDeviceSynchronize();

			ftime(&cuda_end);

			cuda_total += 1000 * (cuda_end.time - cuda_start.time) + (cuda_end.millitm - cuda_start.millitm);

			// Copy the resulting new points to visit for next iteration into the points to be visited array
			cudaMemcpy(&points, d_new_points, diagonalSize * sizeof(Point), cudaMemcpyDeviceToHost);

			run_num++;
			if (run_num % 10 == 0) run_num = 0;
		}

		lodepng_encode32_file("solvedMaze.png", image_copy, image_width, image_height);

		// Free cuda copies memory
		cudaFree(image_copy);
		cudaFree(d_points);
		cudaFree(d_new_points);
	}

	// Check that path has been found
	if (pathFound) printf("Path found!\n");
	else printf("Path not found!\n");

	ftime(&end_time);
	total_time = 1000 * (end_time.time - start_time.time) + (end_time.millitm - start_time.millitm);

	printf("Total execution time: %d, Parallel execution time: %d\n", (int)total_time, (int)cuda_total);


	struct timeb s_time, e_time;
	ftime(&s_time);
	postProcessing("solvedMaze.png");
	ftime(&e_time);
	float post_time = 1000 * (e_time.time - s_time.time) + (e_time.millitm - s_time.millitm);

	printf("Post processing time: %d", (int)post_time);

	return 0;
}

int findNextIndex(unsigned char* image, int index, int indexValue, int h, int w) {
	int nextIndex = -1;

	if (indexValue == 255) indexValue = 245;

	int r = index / (w * 4);
	int c = index % (w * 4);

	unsigned current = r * 4 * w + c;
	unsigned right = current + 4;
	unsigned down = current + 4 * w;
	unsigned left = current - 4;
	unsigned up = current - 4 * w;

	if (r == 0 && c == 0) {
		if (image[right] == indexValue + 1) {
			nextIndex = right;
		}
		if (image[down] == indexValue + 1) {
			nextIndex = down;
		}
	}

	else if (r == 0) {
		if (image[right] == indexValue + 1) {
			nextIndex = right;
		}
		if (image[down] == indexValue + 1) {
			nextIndex = down;
		}
		if (image[left] == indexValue + 1) {
			nextIndex = left;
		}
	}

	else if (r == h - 1) {
		if (image[right] == indexValue + 1) {
			nextIndex = right;
		}
		if (image[up] == indexValue + 1) {
			nextIndex = up;
		}
		if (image[left] == indexValue + 1) {
			nextIndex = left;
		}
	}

	else if (c == 0) {
		if (image[up] == indexValue + 1) {
			nextIndex = up;
		}
		if (image[down] == indexValue + 1) {
			nextIndex = down;
		}
		if (image[right] == indexValue + 1) {
			nextIndex = right;
		}
	}

	else if (c == w - 1) {
		if (image[up] == indexValue + 1) {
			nextIndex = up;
		}
		if (image[down] == indexValue + 1) {
			nextIndex = down;
		}
		if (image[left] == indexValue + 1) {
			nextIndex = left;
		}
	}

	else {
		if (image[up] == indexValue + 1) {
			nextIndex = up;
		}
		if (image[down] == indexValue + 1) {
			nextIndex = down;
		}
		if (image[left] == indexValue + 1) {
			nextIndex = left;
		}
		if (image[right] == indexValue + 1) {
			nextIndex = right;
		}
	}

	return nextIndex;
}

void postProcessing(char* inputImage) {
	unsigned error;
	unsigned char* image;
	unsigned image_width, image_height;

	error = lodepng_decode32_file(&image, &image_width, &image_height, inputImage);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	int index = 0;

	while (index != image_width * image_height - 1) {
		int indexValue = image[index];
		image[index] = (char)0;
		image[index + 1] = (char)255;

		index = findNextIndex(image, index, indexValue, image_width, image_height);

		if (index == -1) {
			lodepng_encode32_file("solvedMaze.png", image, image_width, image_height);
			return;
		}
	}
}

