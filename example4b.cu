#include "Variable.hh" 
#include "GaussianThrustFunctor.hh" 
#include "PdfBuilder.hh" 
#include "UnbinnedDataSet.hh" 

#include "TRandom.hh" 
#include "TH1F.h"
#include "TCanvas.h" 

#include <sys/time.h>
#include <sys/times.h>
#include <iostream>

using namespace std; 

int main (int argc, char** argv) {
  Variable* xvar = new Variable("xvar", -5, 5); 

  TRandom donram(42); 
  UnbinnedDataSet data(xvar);
  for (int i = 0; i < 10000; ++i) {
    fptype val = donram.Gaus(0.2, 1.1);
    if (fabs(val) > 5) {--i; continue;} 
    data.addEvent(val); 
  }

  Variable* mean = new Variable("mean", 0, 1, -10, 10);
  Variable* sigm = new Variable("sigm", 1, 0.5, 1.5); 
  GaussianThrustFunctor gauss("gauss", xvar, mean, sigm); 

  timeval startTime, stopTime, totalTime;

  gauss.setData(&data);
  PdfFunctor fitter(&gauss); 
  gettimeofday(&startTime, NULL);
  fitter.fit(); 
  gettimeofday(&stopTime, NULL);
 
  timersub(&stopTime, &startTime, &totalTime);
  std::cout << "Wallclock time  : " << totalTime.tv_sec + totalTime.tv_usec/1000000.0 << " seconds." << std::endl;

  return 0;
}
