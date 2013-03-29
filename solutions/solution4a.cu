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

  Variable* tau = new Variable("tau", -1.0, -2.0, 0.0); 
  ExpThrustFunctor* expfunc = new ExpThrustFunctor("expfunc", xvar, tau);

  Variable* zero = new Variable("zero", 0); 
  StepThrustFunctor* step = new StepThrustFunctor("step", xvar, zero); 

  vector<FunctorBase*> comps;
  comps.push_back(expfunc); 
  comps.push_back(step); 
  ProdThrustFunctor* decay = new ProdThrustFunctor("decay", comps); 

  Variable* resbias = new Variable("resbias", 0, -1, 1); 
  Variable* ressigm = new Variable("ressigm", 0.2, 0.01, 0.66); 
  GaussianThrustFunctor* resfunc = new GaussianThrustFunctor("resfunc", xvar, resbias, ressigm);

  ConvolutionThrustFunctor* total = new ConvolutionThrustFunctor("total", xvar, decay, resfunc);
  total->setIntegrationConstants(-5, 9, 0.01); 
  total->setData(&data);
  PdfFunctor fitter(total);
  fitter.fit(); 

  // EXERCISE: Draw the data and PDF on a histogram. 
  fitter.getMinuitValues(); 
 
  TH1F pdfHist("pdfHist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  pdfHist.SetStats(false);

  UnbinnedDataSet grid(xvar);
  for (int i = 0; i < xvar->numbins; ++i) {
    double step = (xvar->upperlimit - xvar->lowerlimit)/xvar->numbins;
    xvar->value = xvar->lowerlimit + (i + 0.5) * step;
    grid.addEvent(); 
  }

  total->setData(&grid);
  vector<vector<double> > pdfVals;
  total->getCompProbsAtDataPoints(pdfVals); 

  TCanvas foo;

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
  xvarHist.SetMarkerStyle(8);
  xvarHist.SetMarkerSize(0.5);
  xvarHist.Draw("p"); 
  pdfHist.SetLineColor(kBlue);
  pdfHist.SetLineWidth(3); 
  pdfHist.Draw("lsame"); 
  foo.SaveAs("conv.png"); 


  
  return 0;
}
