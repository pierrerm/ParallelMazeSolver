
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys\timeb.h>
#include "main.h"
#include "lodepng.h"
// For an M (rows) by N (columns) maze
# define M 20
# define N 20

__device__ bool insertPoint(Point array[2 * (N < M ? N : M)], Point point);

__global__ void checkPoint(Point* d_points, Point* d_new_points, unsigned char* image, unsigned nb_threads, unsigned int* index) //bool* maze
{
	// shared array to hold resulting new points to visit for next iteration
	__shared__ Point new_points[2 * (N < M ? N : M)];

	//for (int index = (blockIdx.x * nb_threads + threadIdx.x); index < N; index += nb_threads) {
	for (int index = (blockIdx.x * nb_threads + threadIdx.x); index < N; index += nb_threads) {

		new_points[index] = Point(-1, -1);

		__syncthreads();

		// If Point is empty (ie, still at initial Point(-1,-1) value), skip process but still copy back results
		if (d_points[index].getR() >= 0) {
			// Get row and col for current point
			int row = d_points[index].getR();
			int col = d_points[index].getC();

			// If point above is in bounds and not a wall
			//if (row - 1 >= 0 && maze[(row - 1) * N + col]) {
			if (row - 1 >= 0 && image[(row - 1) * 4 * N + col * 4 + 2] > 250) {
				//printf("Point above: x: %d, y: %d, value: %d\n", (row-1), col, maze[(row - 1) * N + col]);
				// Insert in shared array and get insertion index
				if (insertPoint(new_points, Point((row - 1), col))) {
					//new_points[atomicInc((unsigned int *)insertIndex[0], N * M)] = Point((row - 1), col);
					printf("ye %d\n", atomicAdd((unsigned int*)index, 1));
					image[(row - 1) * 4 * N + col * 4] = 255;
					image[(row - 1) * 4 * N + col * 4 + 1] = 0;
					image[(row - 1) * 4 * N + col * 4 + 2] = 0;
				}
				// If successfully inserted, print details
				//printf("row: %d, col: %d, i: %d\n", (row - 1), col, i);
				/*if (i >= 0) {
					printf("Point %d, r: %d, c: %d\n", i, new_points[i].getR(), new_points[i].getC());
				} */
			}

			// Synchronize threads before checkpoint points to the left
			//__syncthreads();

			//printf("Entering Left Point if statement\n");

			// If point to the left is in bounds and not a wall
			//if (col - 1 >= 0 && maze[row * N + (col - 1)]) {
			if (col - 1 >= 0 && image[row * 4 * N + (col - 1) * 4 + 2] > 250) {
				//printf("Point to the left: x: %d, y: %d, value: %d\n", row, (col - 1), maze[row * N + (col - 1)]);
				// Insert in shared array and get insertion index
				if (insertPoint(new_points, Point(row, (col - 1)))) {
					//new_points[atomicInc((unsigned int*)insertIndex[0], N * M)] = Point(row, (col - 1));
					printf("ye %d\n", atomicAdd((unsigned int*)index, 1));
					image[row * 4 * N + (col - 1) * 4] = 255;
					image[row * 4 * N + (col - 1) * 4 + 1] = 0;
					image[row * 4 * N + (col - 1) * 4 + 2] = 0;
				}
				/*int i = insertPoint(new_points, Point(row, (col - 1)));
				printf("row: %d, col: %d, i: %d\n", row, (col - 1), i);
				if (i >= 0) {
					printf("Point %d, r: %d, c: %d\n", i, new_points[i].getR(), new_points[i].getC());
					image[row * 4 * N + (col - 1) * 4] = 255;
					image[row * 4 * N + (col - 1) * 4 + 1] = 0;
					image[row * 4 * N + (col - 1) * 4 + 2] = 0;
					// If successfully inserted, print details
					//printf("Point %d, r: %d, c: %d\n", i, row, (col-1));
				}*/
			}

			//__syncthreads();

			// If point below is in bounds and not a wall
			//if (row + 1 >= 0 && maze[(row - 1) * N + col]) {
			if (row + 1 < M  && image[(row + 1) * 4 * N + col * 4 + 2] > 250) {
				//printf("Point above: x: %d, y: %d, value: %d\n", (row-1), col, maze[(row - 1) * N + col]);
				// Insert in shared array and get insertion index
				if (insertPoint(new_points, Point((row + 1), col))) {
					//new_points[atomicInc((unsigned int*)insertIndex[0], N * M)] = Point((row + 1), col);
					printf("ye %d\n", atomicAdd((unsigned int*)index, 1));
					image[(row + 1) * 4 * N + col * 4] = 255;
					image[(row + 1) * 4 * N + col * 4 + 1] = 0;
					image[(row + 1) * 4 * N + col * 4 + 2] = 0;
				}
				/*int i = insertPoint(new_points, Point((row + 1), col));
				// If successfully inserted, print details
				printf("row: %d, col: %d, i: %d\n", (row + 1), col, i);
				if (i >= 0) {
					printf("Point %d, r: %d, c: %d\n", i, new_points[i].getR(), new_points[i].getC());
					image[(row + 1) * 4 * N + col * 4] = 255;
					image[(row + 1) * 4 * N + col * 4 + 1] = 0;
					image[(row + 1) * 4 * N + col * 4 + 2] = 0;
					//printf("Point %d, r: %d, c: %d\n", i, (row - 1), col);
				} */
			}

			//printf("Entering Left Point if statement\n");
			//__syncthreads();

			// If point to the left is in bounds and not a wall
			//if (col - 1 >= 0 && maze[row * N + (col - 1)]) {
			if (col + 1 < N && image[row * 4 * N + (col + 1) * 4 + 2] > 250) {
				//printf("Point to the left: x: %d, y: %d, value: %d\n", row, (col - 1), maze[row * N + (col - 1)]);
				// Insert in shared array and get insertion index
				if (insertPoint(new_points, Point(row, (col + 1)))) {
					//new_points[atomicInc((unsigned int*)insertIndex[0], N * M)] = Point(row, (col + 1));
					printf("ye %d\n", atomicAdd((unsigned int*)index, 1));
					image[row * 4 * N + (col + 1) * 4] = 255;
					image[row * 4 * N + (col + 1) * 4 + 1] = 0;
					image[row * 4 * N + (col + 1) * 4 + 2] = 0;
				}
				/* int i = insertPoint(new_points, Point(row, (col + 1)));
				printf("row: %d, col: %d, i: %d\n", row, (col + 1), i);
				if (i >= 0) {
					printf("Point %d, r: %d, c: %d\n", i, new_points[i].getR(), new_points[i].getC());
					image[row * 4 * N + (col + 1) * 4] = 255;
					image[row * 4 * N + (col + 1) * 4 + 1] = 0;
					image[row * 4 * N + (col + 1) * 4 + 2] = 0;
					// If successfully inserted, print details
					//printf("Point %d, r: %d, c: %d\n", i, row, (col-1));
				} */
			}
		}

		// Synchronize threads and copy shared array into result array
		__syncthreads();
		d_new_points[index] = new_points[index];
		d_new_points[index + (N < M ? N : M)] = new_points[index + (N < M ? N : M)];
		printf("index: %d, row: %d\n", index, d_new_points[index].getR());
		//printf("index: %d, point: x:%d, y:%d\n", index, new_points[index].getR(), new_points[index].getC());
	}
}

// insert given point at the first available position in the given array (avoiding duplicate points)
__device__ bool insertPoint(Point array[2 * (N < M ? N : M)], Point point) {
	int i;
	// Cycle through array points until the end or an empty point (ie, still at initial Point(-1,-1) value) is reached
	for (i = 0; i < 2 * (N < M ? N : M) && array[i].getR() >= 0; i++) {
		// if duplicate point found (ie, point we want to insert is already in the array) do nothing and return
		if (point.getR() == array[i].getR() && point.getC() == array[i].getC()) {
			//printf("point already exists\n");
			return false;
		}
	}
	if (i < 2 * (N < M ? N : M)) return true;
	/*// If empty point has been found (and we are not at the end of the array), insert point and return insertion index
	if (i < 2 * (N < M ? N : M)) {
		array[i] = point;
		return i;
	}
	// If array end is reached, do nothing and return
	//printf("end of array reached\n");*/
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

	/* // The maze, false is wall, true is path
	bool maze[M * N] = {true, true, true, true,
						true, true, false, true,
						true, true, false, true,
						true, false, true, true};

	printf("Maze size: %d rows by %d columns\n", M, N); */

	printf("Input file: %s, maze width: %d, maze height: %d\n", input_filename, image_width, image_height);
	printf("Number of blocks: %d, number of threads: %d\n", nb_blocks, nb_threads);

	// Check that maze is non-empty
	if (image_width * image_height != 0) {

		// Array of points to be visited at each iteration, initialized with all Point(-1,-1) entries
		Point points[diagonalSize];
		// Set first point to be visited - the arrival point (because we are backtracking), the last point of the maze (assuming square maze)
		points[0] = Point(M - 1, N - 1);
		
		/* Point* failedPoints;
		cudaMallocManaged((void**)& failedPoints, N * N * sizeof(Point)); */

		/*// Cuda copy of the maze
		bool* d_maze;
		cudaMallocManaged((void**)& d_maze, M * N * sizeof(bool));
		cudaMemcpy(d_maze, &maze, M * N * sizeof(bool), cudaMemcpyHostToDevice);*/

		cudaMallocManaged((void**)& image_copy, image_width * image_height * 4 * sizeof(unsigned char));
		cudaMemcpy(image_copy, image, image_width * image_height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);

		// Temporary points matrix, initialized with  all Point(-1,-1) entries
		Point temp_points[N];

		// Cuda copies of the points to be visited and resulting new points to visit for next iteration
		Point* d_points, * d_new_points;

		cudaMallocManaged((void**)& d_points, diagonalSize * sizeof(Point));
		cudaMallocManaged((void**)& d_new_points, diagonalSize * sizeof(Point));

		// While there are still points to visit
		while (points[0].getR() != -1) {

			if (points[0].getR() == 0 && points[0].getC() == 0) {
				pathFound = true;
				break;
			}

			cudaMemcpy(d_points, &points, diagonalSize * sizeof(Point), cudaMemcpyHostToDevice);
			cudaMemcpy(d_new_points, &temp_points, diagonalSize * sizeof(Point), cudaMemcpyHostToDevice);

			ftime(&cuda_start);

			unsigned int* d_index = 0;
			cudaMalloc((void**)& d_index, sizeof(unsigned int));

			// Call to device function with N threads (at most N points), points to be visited, and array to hold resulting new points to visit for next iteration
			checkPoint << <nb_blocks, nb_threads >> > (d_points, d_new_points, image_copy, nb_threads, d_index); //d_maze

			cudaDeviceSynchronize();

			ftime(&cuda_end);

			cuda_total += 1000 * (cuda_end.time - cuda_start.time) + (cuda_end.millitm - cuda_start.millitm);

			// Copy the resulting new points to visit for next iteration into the points to be visited array
			cudaMemcpy(&points, d_new_points, diagonalSize * sizeof(Point), cudaMemcpyDeviceToHost);
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

	printf("Total execution time: %d, Parallel execution time: %d", (int)total_time, (int)cuda_total);

	return 0;
}

