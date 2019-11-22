
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "main.h"
// For an N by N maze
# define N 4

__device__ int insertPoint(Point array[N], Point point);

__global__ void checkPoint(Point* d_points, Point* d_new_points, bool* maze, unsigned nb_threads)
{
	// shared array to hold resulting new points to visit for next iteration
	__shared__ Point new_points[N];

	for (int index = (blockIdx.x * nb_threads + threadIdx.x); index < N; index += nb_threads) {

		// If Point is empty (ie, still at initial Point(-1,-1) value), do nothing
		if (d_points[index].getR() < 0) return;

		// Get row and col for current point
		int row = d_points[index].getR();
		int col = d_points[index].getC();

		// If point above is in bounds and not a wall
		if (row - 1 > 0 && maze[(row - 1) * N + col]) {
			// Insert in shared array and get insertion index
			int i = insertPoint(new_points, Point((row - 1), col));
			// If successfully inserted, print details
			if (i != -1) {
				printf("Point %d, r: %d, c: %d\n", i, (row - 1), col);
			}
		}

		// Synchronize threads before checkoint points to the left
		__syncthreads();

		// If point to the left is in bounds and not a wall
		if (col - 1 > 0 && maze[row * N + (col - 1)]) {
			// Insert in shared array and get insertion index
			int i = insertPoint(new_points, Point(row, (col - 1)));
			if (i != -1) {
				// If successfully inserted, print details
				printf("Point %d, r: %d, c: %d\n", i, row, (col-1));
			}
		}

		// Synchronize threads and copy shared array into result array
		__syncthreads();
		d_new_points[index] = new_points[index];
	}
}

// insert given point at the first available position in the given array (avoiding duplicate points)
__device__ int insertPoint(Point array[N], Point point) {
	int i;
	// cycle through array points until the end or an empty point (ie, still at initial Point(-1,-1) value) is reached
	for (i = 0; i < (sizeof(array) / sizeof(array[0])) && array[i].getR() >= 0; i++) {
		// if duplicate point found (ie, point we want to insert is already in the array) do nothing and return
		if (point.getR() == array[i].getR() && point.getC() == array[i].getC()) return -1;
	}
	// if empty point has been found (and we are not at the end of the array), insert point and return insertion index
	if (i < (sizeof(array) / sizeof(array[0]))) {
		array[i] = point;
		return i;
	}
	// if array end is reached, do nothing and return
	return -1;
}

int main()
{
	unsigned total_threads = N;
	unsigned nb_threads = total_threads;
	unsigned nb_blocks = 1;

	// Max threads per block is 1024
	while (nb_threads > 1024) {
		nb_blocks++;
		nb_threads = total_threads / nb_blocks;
	}

	// The maze, false is wall, true is path
	bool maze[N * N] = {true, true, true, true,
						true, true, false, true,
						true, true, false, true,
						true, false, true, true};
	
	// Size of the maze (ie number of points)
	int size = sizeof(maze) / sizeof(maze[0]);

	printf("size: %d\n", size);	

	// Iteration counter
	int counter = 0;

	// Check that maze is non-empty
	if (size != 0) {

		// Array of points to be visited at each iteration, initialized with all Point(-1,-1) entries
		Point points[N];
		// Set first point to be visited - the arrival point (because we are backtracking), the last point of the maze (assuming square maze)
		points[0] = Point(sqrt(size) - 1, sqrt(size) - 1);
		
		/* Point* failedPoints;
		cudaMallocManaged((void**)& failedPoints, N * N * sizeof(Point)); */

		// Cuda copy of the maze
		bool* d_maze;
		cudaMallocManaged((void**)& d_maze, N * N * sizeof(bool));
		cudaMemcpy(d_maze, &maze, N * N * sizeof(bool), cudaMemcpyHostToDevice);

		// While there are still points to visit
		while (points[0].getR() != -1) {

			// Temporary points matrix, initialized with  all Point(-1,-1) entries
			Point temp_points[N];

			// Cuda copies of the points to be visited and resulting new points to visit for next iteration
			Point* d_points,* d_new_points;

			cudaMallocManaged((void**)& d_points, N * sizeof(Point));
			cudaMallocManaged((void**)& d_new_points, N * sizeof(Point));

			cudaMemcpy(d_points, &points, N * sizeof(Point), cudaMemcpyHostToDevice);
			cudaMemcpy(d_new_points, &temp_points, N * sizeof(Point), cudaMemcpyHostToDevice);

			// Call to device function with N threads (at most N points), points to be visited, and array to hold resulting new points to visit for next iteration
			checkPoint << <nb_blocks, nb_threads >> > (d_points, d_new_points, d_maze, N);

			// Copy the resulting new points to visit for next iteration into the points to be visited array
			cudaMemcpy(&points, d_new_points, N * sizeof(Point), cudaMemcpyDeviceToHost);

			// Free cuda copies memory
			cudaFree(d_points);
			cudaFree(d_new_points);

			counter++;
		}
	}

	printf("counter: %d\n", counter);

	// With up-left moves only, the path to the end will be exactly 2N - 2 iterations/points long
	if (counter == (2*N - 2)) printf("Path found!\n");
	else printf("Path not found!\n");

	return 0;
}

