#include <iostream> 
#include <cmath> 
#include <cassert> 
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/transform_reduce.h"
#include "rootstuff/TRandom.hh"
#include "rootstuff/TMinuit.hh" 

using namespace thrust; 

struct GaussFunctor : public thrust::unary_function<double, double> {
  // EXERCISE: Write this struct so that it has an operator method
  // which takes a double and returns the Gaussian probability of 
  // that double. The mean and sigma of the distribution can be
  // passed as arguments to the operator function, or can be 
  // member variables of the struct; take your choice. 
  // NB: You may want to do the exercises in the main method before
  // these ones in the structs - follow the execution path of the program. 

  GaussFunctor (double m, double s) 
    : mean(m)
    , sigma(s)
  {} 

  __device__ double operator () (double xval) {
    double ret = (xval - mean);
    ret /= sigma;
    ret *= ret;
    ret = exp(-0.5*ret);
    ret /= sigma;
    ret /= sqrt(2*M_PI);
    return ret; 
  }

  double mean;
  double sigma; 
}; 

struct NllFunctor : public thrust::unary_function<double, double> {
  // EXERCISE: Write this struct similarly to the one above, but instead
  // of the Gaussian probability, it should return the negative log likelihood,
  // ie -2*ln(P(x)) where P(x) is the Gaussian function. 
  // (The factor 2 is to get correct errors. You can put it in fcn_nll if you prefer.) 


  NllFunctor (double m, double s) 
    : mean(m)
    , sigma(s)
  {}

  __device__ double operator () (double xval) {
    // Taking the log of an exp - simplify: 
    double ret = -0.5*pow((xval - mean) / sigma, 2);
    ret -= log(sigma); 
    // Ignore constant term sqrt(2pi)
    return -2*ret; 
  }

  double mean;
  double sigma; 
}; 


device_vector<double>* device_data = 0; 

void fcn_nll (int& npar, double* deriv, double& fVal, double param[], int flag) {
  double mean = param[0];
  double sigma = param[1];

  // EXERCISE: Create an NllFunctor and use it to calculate the negative-log-likelihood
  // of the data stored in device_data, using a Thrust transform_reduce call. 
  // Store the result in fVal. 

  double initVal = 0; 
  fVal = transform_reduce(device_data->begin(), device_data->end(), 
			  NllFunctor(mean, sigma),
			  initVal, 
			  thrust::plus<double>()); 
}

int main (int argc, char** argv) {
  int sizeOfVector = atoi(argv[1]); 
  // NOTE: This time we will use Thrust, 
  // so we need not limit ourselves to 
  // small vectors - the library will take
  // care of reductions for us.

  double mean = 5;
  double sigma = 3; 

  // Generate a host-side vector and fill it with random numbers. 
  TRandom donram(42); 
  host_vector<double> host_data;
  // EXERCISE: Create a thrust::host_vector and fill it with random numbers. 
  for (int i = 0; i < sizeOfVector; ++i) {
    double currVal = donram.Gaus(mean, sigma); 
    host_data.push_back(currVal); 
  }

  // EXERCISE: Create the device_vector called 'device_data' (use the existing
  //           pointer) and copy the host data into it. 
  device_data = new device_vector<double>(host_data);
  // EXERCISE: Use Thrust to calculate the Gaussian probability of 
  //           each generated event, storing the results in another device_vector.
  device_vector<double> device_results(device_data->size());
  thrust::transform(device_data->begin(), device_data->end(), 
		    device_results.begin(), 
		    GaussFunctor(mean, sigma)); 
  // EXERCISE: Copy the result back to the host side. Compare the numbers with
  //           ones calculated by the CPU. 
  host_vector<double> host_results = device_results; 
  double tolerance = 1e-6; 
  double cpusum = 0; 
  for (unsigned int i = 0; i < host_data.size(); ++i) {
    double currVal = exp(-0.5*(pow((host_data[i] - mean) / sigma, 2))) / (sigma * sqrt(2*M_PI));
    cpusum += currVal; 
    if (fabs(currVal - host_results[i]) < tolerance) continue;
    std::cout << "Problem with event " << i << ": "
	      << currVal << " != " << host_results[i] << std::endl; 
  }

  // EXERCISE: Use Thrust to calculate the sum of the device-side results vector;
  //           compare with the same sum calculated on the host. 
  double gpusum = reduce(device_results.begin(), device_results.end());
  if (fabs(gpusum - cpusum) > tolerance)
    std::cout << "Problem with sums: " << gpusum << " != " << cpusum << std::endl; 

  TMinuit minuit(2);
  minuit.DefineParameter(0, "mean", mean,   0.01, -10.0, 10.0); 
  minuit.DefineParameter(1, "sigma", sigma, 0.01,   0.0, 10.0); 
  // EXERCISE: Implement fcn_nll so that this fit will work! 
  minuit.SetFCN(fcn_nll);
  minuit.Migrad(); 

  // Cleanup. 
  if (device_data) delete device_data; 
  
  return 0; 
}
