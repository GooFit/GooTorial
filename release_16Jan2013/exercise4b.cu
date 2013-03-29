#include "Variable.hh" 
#include "GaussianThrustFunctor.hh" 
#include "ExpThrustFunctor.hh" 
#include "StepThrustFunctor.hh" 
#include "ProdThrustFunctor.hh" 
#include "ConvolutionThrustFunctor.hh" 
#include "MappedThrustFunctor.hh" 
#include "PdfBuilder.hh" 
#include "UnbinnedDataSet.hh" 

#include "TRandom.hh" 
#include "TH1F.h"
#include "TH2F.h"
#include "TStyle.h" 
#include "TCanvas.h" 

#include <sys/time.h>
#include <sys/times.h>
#include <iostream>

using namespace std; 

int main (int argc, char** argv) {
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(10);
  gStyle->SetFrameFillColor(10);
  gStyle->SetFrameBorderMode(0);
  gStyle->SetPadColor(0);
  gStyle->SetTitleColor(1);
  gStyle->SetStatColor(0);
  gStyle->SetFillColor(0);
  gStyle->SetFuncWidth(1);
  gStyle->SetLineWidth(1);
  gStyle->SetLineColor(1);
  gStyle->SetPalette(1, 0);

  vector<Variable*> vars; 
  Variable* xvar = new Variable("xvar", -3, 6); vars.push_back(xvar);
  Variable* yvar = new Variable("yvar", 0, 1); vars.push_back(yvar);
  UnbinnedDataSet data(vars);

  TH1F xvarHist("xvarHist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  TH1F xvarHistLo("xvarHistLo", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  TH1F xvarHistHi("xvarHistHi", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  xvarHist.SetStats(false); 
  xvarHistLo.SetStats(false); 
  xvarHistHi.SetStats(false); 

  TRandom donram(42); 
  double totalData = 0; 
  // Generating exponential convolved with
  // Gaussian resolution which varies with
  // position in a second variable. 
  for (int i = 0; i < 100000; ++i) {
    xvar->value = -log(donram.Uniform(0, 1)); 
    yvar->value = donram.Uniform(); 
    if (yvar->value < 0.5) xvar->value += donram.Gaus(0.05, 0.33);
    else xvar->value += donram.Gaus(0.08, 0.50);
    if ((xvar->value < xvar->lowerlimit) || (xvar->value > xvar->upperlimit)) {
      i--;
      continue; 
    }
    data.addEvent(); 
    xvarHist.Fill(xvar->value);
    if (yvar->value < 0.5) xvarHistLo.Fill(xvar->value);
    else xvarHistHi.Fill(xvar->value);
    totalData++; 
  }

  // EXERCISE: Create a PDF modelling the distribution created above. 



  //PdfFunctor fitter(total);
  //fitter.fit(); 
  //fitter.getMinuitValues(); 

  // EXERCISE: Draw the data and PDF on a histogram;
  // in addition, create separate histograms for the low and high
  // y areas - that is, one for each resolution function. 
 
  return 0;
}
