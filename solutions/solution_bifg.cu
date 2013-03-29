#include "Variable.hh" 
#include "PdfBuilder.hh" 
#include "UnbinnedDataSet.hh" 

#include "TRandom.h" 
#include "TH1F.h"
#include "TH2F.h"
#include "TStyle.h" 
#include "TCanvas.h" 

#include "BifurGaussThrustFunctor.hh"

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

  Variable* xvar = new Variable("xvar", -100, 100); 
  xvar->numbins = 1000; // For such a large range, want more bins for better accuracy in normalisation. 
  UnbinnedDataSet bifgdata(xvar);

  TH1F bifgHist("bifgHist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  bifgHist.SetStats(false); 

  TRandom donram(42); 
  double totalData = 0; 

  double leftSigma = 13;
  double rightSigma = 29;
  double leftIntegral = 0.5 / (leftSigma * sqrt(2*M_PI));
  double rightIntegral = 0.5 / (rightSigma * sqrt(2*M_PI));
  double totalIntegral = leftIntegral + rightIntegral; 
  double bifpoint = -10; 

  for (int i = 0; i < 100000; ++i) {
    if (donram.Uniform() < (leftIntegral / totalIntegral)) {
      xvar->value = bifpoint - 1;
      while ((xvar->value < bifpoint) || (xvar->value > xvar->upperlimit)) xvar->value = donram.Gaus(bifpoint, rightSigma);
    }
    else {
      xvar->value = bifpoint + 1;
      while ((xvar->value > bifpoint) || (xvar->value < xvar->lowerlimit)) xvar->value = donram.Gaus(bifpoint, leftSigma);
    }
    bifgdata.addEvent(); 
    bifgHist.Fill(xvar->value); 
    totalData++; 
  }

  // EXERCISE: Write one of LandauThrustFunctor, BifurGaussThrustFunctor, 
  // or NovoSibirskThrustFunctor. Then use your new class to fit one
  // of the distributions created above. If you feel ambitious, do two
  // or all three. 

  // There is no solution for this exercise! However, if you get it to work
  // well, please give me the code and I will put it in the next release of
  // GooFit. 

  UnbinnedDataSet* data = &bifgdata;

  Variable *mean = new Variable("mean",0,-15,15);
  Variable *sigmaLeft = new Variable("sigmaLeft",leftSigma,leftSigma-5,leftSigma+5);
  Variable *sigmaRight = new Variable("sigmaRight",rightSigma,rightSigma-5,rightSigma+5);
  BifurGaussThrustFunctor* total = new BifurGaussThrustFunctor("total", xvar, mean, sigmaLeft, sigmaRight);
  
  total->setData(data);
  PdfFunctor fitter(total);
  fitter.fit(); 
  fitter.getMinuitValues(); 

  TH1F pdfHist("pdfHist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  pdfHist.SetStats(false);

  UnbinnedDataSet grid(xvar);
  double step = (xvar->upperlimit - xvar->lowerlimit)/xvar->numbins;
  for (int i = 0; i < xvar->numbins; ++i) {
    xvar->value = xvar->lowerlimit + (i + 0.5) * step;
    grid.addEvent(); 
  }

  TCanvas foo;

  total->setData(&grid);
  vector<vector<double> > pdfVals;
  total->getCompProbsAtDataPoints(pdfVals); 

  double totalPdf = 0; 
  for (int i = 0; i < grid.getNumEvents(); ++i) {
    grid.loadEvent(i); 
    pdfHist.Fill(xvar->value, pdfVals[0][i]);
    totalPdf += pdfVals[0][i]; 
  }

  for (int i = 0; i < xvar->numbins; ++i) {
    double val = pdfHist.GetBinContent(i+1); 
    val /= totalPdf; 
    val *= totalData;
    pdfHist.SetBinContent(i+1, val); 
  }

  foo.SetLogy(true); 
  bifgHist.SetMarkerStyle(8);
  bifgHist.SetMarkerSize(0.5);
  bifgHist.Draw("p"); 
  pdfHist.SetLineColor(kBlue);
  pdfHist.SetLineWidth(3); 
  pdfHist.Draw("lsame"); 
  foo.SaveAs("bifurgauss.png"); 
 
  return 0;
}
