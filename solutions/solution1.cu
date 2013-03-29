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
  data[threadIdx.x] = exp(-0.5 * pow((data[threadIdx.x] - mean) / sigma, 2));
  data[threadIdx.x] /= (sigma * sqrt(2*M_PI));
}

__global__ void dev_reduce_vector (double* data, double* result) {
  int currentArraySize = blockDim.x;
  while (currentArraySize > 1) {
    int secondHalfBegin = (1 + currentArraySize) / 2;
    if (threadIdx.x + secondHalfBegin < currentArraySize) {
      data[threadIdx.x] += data[secondHalfBegin + threadIdx.x];
    }
    __syncthreads();
    currentArraySize = secondHalfBegin;
  }
  if (0 == threadIdx.x) (*result) = data[0];
}

int main (int argc, char** argv) {
  int sizeOfVector = atoi(argv[1]); 
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);
  assert(sizeOfVector < devProp.maxThreadsPerBlock);

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

  // Create a device-side array and copy the data into it.
  double* dev_data = 0; 
  CUDACALL(cudaMalloc((void**) &dev_data, sizeOfVector*sizeof(double))); 
  CUDACALL(cudaMemcpy(dev_data, host_data, sizeOfVector*sizeof(double), cudaMemcpyHostToDevice)); 
  dev_calculate_Gaussians<<<1, sizeOfVector>>>(dev_data, mean, sigma); 

  // Copy back results
  CUDACALL(cudaMemcpy(host_data, dev_data, sizeOfVector*sizeof(double), cudaMemcpyDeviceToHost));

  // Check for reasonableness
  double tolerance = 1e-6;  
  for (int i = 0; i < sizeOfVector; ++i) {
    if (fabs(host_data[i] - host_probs[i]) <= tolerance) continue;
    std::cout << "Problem with entry " << i << ": " 
	      << host_probs[i] << " " << host_data[i] << " "
	      << (host_probs[i] - host_data[i])  
	      << std::endl; 
  }

  double* device_sum_address;
  std::cout << "Sum from CPU: " << host_sum << std::endl; 
  CUDACALL(cudaMalloc((void**) &device_sum_address, sizeof(double)));
  dev_reduce_vector<<<1, sizeOfVector>>>(dev_data, device_sum_address);
  CUDACALL(cudaMemcpy(&host_sum, device_sum_address, sizeof(double), cudaMemcpyDeviceToHost));
  std::cout << "Sum from GPU: " << host_sum << std::endl; 
  
  
  return 0; 
}
