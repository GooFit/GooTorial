#include "Variable.hh" 
#include "GaussianThrustFunctor.hh" 
#include "ExpThrustFunctor.hh" 
#include "StepThrustFunctor.hh" 
#include "ProdThrustFunctor.hh" 
#include "ConvolutionThrustFunctor.hh" 
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
  UnbinnedDataSet data(vars);

  TH1F xvarHist("xvarHist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  xvarHist.SetStats(false); 

  TRandom donram(42); 
  double totalData = 0; 
  // Generating exponential convolved with Gaussian
  for (int i = 0; i < 100000; ++i) {
    xvar->value = -log(donram.Uniform(0, 1)); 
    xvar->value += donram.Gaus(0.05, 0.33);
    if ((xvar->value < xvar->lowerlimit) || (xvar->value > xvar->upperlimit)) {
      i--;
      continue; 
    }
    data.addEvent(); 
    xvarHist.Fill(xvar->value);
    totalData++; 
  }

  // EXERCISE: Create a PDF modelling an exponential
  // decay convolved with a Gaussian resolution. 
  // NB! ExpThrustFunctor does not automatically go
  // to zero at a negative argument, unlike decay
  // functions! 

  //PdfFunctor fitter(total);
  //fitter.fit(); 

  // EXERCISE: Draw the data and PDF on a histogram. 
  //fitter.getMinuitValues(); 
 
  TH1F pdfHist("pdfHist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  pdfHist.SetStats(false);
  TCanvas foo;


  // Create a grid of points and extract the PDF probabilities. 



  // Fill, then normalise, the pdfHist.



  foo.SetLogy(true); 
  xvarHist.SetMarkerStyle(8);
  xvarHist.SetMarkerSize(0.5);
  xvarHist.Draw("p"); 
  pdfHist.SetLineColor(kBlue);
  pdfHist.SetLineWidth(3); 
  pdfHist.Draw("lsame"); 
  foo.SaveAs("conv.png"); 
  
  return 0;
}
