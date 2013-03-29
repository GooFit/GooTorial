#include <iostream> 
#include <cmath> 
#include <cassert> 

void checkError (cudaError_t err, int line) {
  if (err == cudaSuccess) return;
  std::cout << "Error code " << err << " : " << cudaGetErrorString(err) << " " << " on line " << line << ", aborting.\n";
  assert(false); 
}

#define CUDACALL(x) checkError(x, __LINE__)

__global__ void dev_calculate_Gaussians (double* data, double mean, double sigma) {
  // EXERCISE: Write this function so that each thread updates one
  // index of the array data. The output value should be the Gaussian
  // probability of the input value, given mean and sigma. 
  // (Optionally add a separate array for the output so that the input data
  // are not overwritten.) 
}

__global__ void dev_reduce_vector (double* data, double* result) {
  // EXERCISE: Write this function so it takes the sum of
  // the values in data and puts them into result. 
  // NB: You should assume that the size of data is smaller
  // than one block - you need not worry about synchronising
  // across blocks. 

}

int main (int argc, char** argv) {
  int sizeOfVector = atoi(argv[1]); 

  // EXERCISE: Check that the sizeOfVector variable
  // is small enough that the GPU is able to launch
  // that many threads in a single block. 




  double mean = 5;
  double sigma = 3; 

  // Generate a host-side vector and fill it with random numbers. 
  double* host_data = new double[sizeOfVector];
  for (int i = 0; i < sizeOfVector; ++i) {
    host_data[i] = (rand() % 11) - 5; 
  }

  // Host-side numbers to check against device-side ones. 
  double* host_probs = new double[sizeOfVector]; 
  double host_sum = 0; 
  for (int i = 0; i < sizeOfVector; ++i) {
    host_probs[i] = exp(-0.5 * pow((host_data[i] - mean) / sigma, 2));
    host_probs[i] /= (sigma * sqrt(2*M_PI)); 
    host_sum += host_probs[i]; 
  }

  double* dev_data = 0; 
  // EXERCISE: Create a device-side array with sizeOfVector elements and copy the host data into it.
  // EXERCISE: Launch a one-block kernel which will run the method dev_calculate_Gaussians
  //           on each element of dev_data. 
  // EXERCISE: Copy back the results of the calculation into host_data. 

  // Check for reasonableness
  double tolerance = 1e-6;  
  for (int i = 0; i < sizeOfVector; ++i) {
    if (fabs(host_data[i] - host_probs[i]) <= tolerance) continue;
    std::cout << "Problem with entry " << i << ": " 
	      << host_probs[i] << " " << host_data[i] << " "
	      << (host_probs[i] - host_data[i])  
	      << std::endl; 
  }

  std::cout << "Sum from CPU: " << host_sum << std::endl; 
  double* device_sum = 0; 
  // EXERCISE: Allocate a single double on the device, putting its address into device_sum. 
  // EXERCISE: Launch a kernel to sum the elements of dev_data and put the result in device_sum.
  // EXERCISE: Copy the result back into host_sum.

  std::cout << "Sum from GPU: " << host_sum << std::endl; 
  
  
  return 0; 
}
