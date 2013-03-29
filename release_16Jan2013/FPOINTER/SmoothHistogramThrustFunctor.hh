#ifndef SMOOTHHISTOGRAM_THRUST_FUNCTOR_HH
#define SMOOTHHISTOGRAM_THRUST_FUNCTOR_HH

#include "ThrustPdfFunctor.hh" 
#include "BinnedDataSet.hh" 

class SmoothHistogramThrustFunctor : public ThrustPdfFunctor {
public:
  SmoothHistogramThrustFunctor (std::string n, BinnedDataSet* x, Variable* smoothing); 
  __host__ virtual fptype normalise () const;

private:
  thrust::device_vector<fptype>* dev_base_histogram; 
  thrust::device_vector<fptype>* dev_smoothed_histogram; 
  fptype totalEvents; 
  fptype* host_constants;
};

#endif
