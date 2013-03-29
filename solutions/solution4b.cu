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
  double totalLo = 0;
  double totalHi = 0; 
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
    if (yvar->value < 0.5) {xvarHistLo.Fill(xvar->value); totalLo++;}
    else {xvarHistHi.Fill(xvar->value); totalHi++;}
    totalData++; 
  }

  // EXERCISE: Create a PDF modelling the distribution created above. 

  Variable* tau = new Variable("tau", -1.0, -2.0, 0.0); 
  ExpThrustFunctor* expfunc = new ExpThrustFunctor("expfunc", xvar, tau);

  Variable* zero = new Variable("zero", 0); 
  StepThrustFunctor* step = new StepThrustFunctor("step", xvar, zero); 

  vector<FunctorBase*> comps;
  comps.push_back(expfunc); 
  comps.push_back(step); 
  ProdThrustFunctor* decay = new ProdThrustFunctor("decay", comps); 

  Variable* resbiaslo = new Variable("resbiaslo", 0, -1, 1); 
  Variable* ressigmlo = new Variable("ressigmlo", 0.2, 0.01, 0.66); 
  GaussianThrustFunctor* resfunclo = new GaussianThrustFunctor("resfunclo", xvar, resbiaslo, ressigmlo);

  Variable* resbiashi = new Variable("resbiashi", 0, -1, 1); 
  Variable* ressigmhi = new Variable("ressigmhi", 0.2, 0.01, 0.66); 
  GaussianThrustFunctor* resfunchi = new GaussianThrustFunctor("resfunchi", xvar, resbiashi, ressigmhi);

  ConvolutionThrustFunctor* convlo = new ConvolutionThrustFunctor("convlo", xvar, decay, resfunclo);
  convlo->setIntegrationConstants(-5, 9, 0.01); 
  ConvolutionThrustFunctor* convhi = new ConvolutionThrustFunctor("convhi", xvar, decay, resfunchi);
  convhi->setIntegrationConstants(-5, 9, 0.01); 


  // Function to return 0 below 0.5, and 1 above it.
  Variable* onehalf = new Variable("onehalf", 0.5); 
  StepThrustFunctor* mapper = new StepThrustFunctor("mapper", yvar, onehalf); 

  vector<ThrustPdfFunctor*> targets;
  targets.push_back(convlo);
  targets.push_back(convhi); 
  MappedThrustFunctor* total = new MappedThrustFunctor("total", mapper, targets); 

  total->setData(&data);
  PdfFunctor fitter(total);
  fitter.fit(); 
  fitter.getMinuitValues(); 

  // EXERCISE: Draw the data and PDF on a histogram;
  // in addition, create separate histograms for the low and high
  // y areas - that is, one for each resolution function. 

  TH1F pdfHist("pdfHist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  TH1F pdfHistLo("pdfHistLo", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  TH1F pdfHistHi("pdfHistHi", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  pdfHist.SetStats(false);
  pdfHistLo.SetStats(false);
  pdfHistHi.SetStats(false);

  UnbinnedDataSet grid(vars);

  for (int i = 0; i < xvar->numbins; ++i) {
    double step = (xvar->upperlimit - xvar->lowerlimit)/xvar->numbins;
    xvar->value = xvar->lowerlimit + (i + 0.5) * step;
    for (int j = 0; j < yvar->numbins; ++j) {
      step = (yvar->upperlimit - yvar->lowerlimit)/yvar->numbins;
      yvar->value = yvar->lowerlimit + (j + 0.5) * step;
      grid.addEvent(); 
    }
  }

  total->setData(&grid);
  vector<vector<double> > pdfVals;
  total->getCompProbsAtDataPoints(pdfVals); 

  TCanvas foo;

  double totalPdf = 0; 
  double totalPdfLo = 0; 
  double totalPdfHi = 0; 
  for (int i = 0; i < grid.getNumEvents(); ++i) {
    grid.loadEvent(i); 
    pdfHist.Fill(xvar->value, pdfVals[0][i]);
    totalPdf += pdfVals[0][i]; 

    if (yvar->value < 0.5) {
      pdfHistLo.Fill(xvar->value, pdfVals[0][i]);
      totalPdfLo += pdfVals[0][i]; 
    }
    else {
      pdfHistHi.Fill(xvar->value, pdfVals[0][i]);
      totalPdfHi += pdfVals[0][i]; 
    }
  }

  for (int i = 0; i < xvar->numbins; ++i) {
    double val = pdfHist.GetBinContent(i+1); 
    val /= totalPdf; 
    val *= totalData;
    pdfHist.SetBinContent(i+1, val); 

    val = pdfHistLo.GetBinContent(i+1); 
    val /= totalPdfLo; 
    val *= totalLo;
    pdfHistLo.SetBinContent(i+1, val); 

    val = pdfHistHi.GetBinContent(i+1); 
    val /= totalPdfHi; 
    val *= totalHi;
    pdfHistHi.SetBinContent(i+1, val); 
  }

  foo.SetLogy(true); 
  xvarHist.SetMarkerStyle(8);
  xvarHist.SetMarkerSize(0.5);
  xvarHist.Draw("p"); 
  pdfHist.SetLineColor(kBlue);
  pdfHist.SetLineWidth(3); 
  pdfHist.Draw("lsame"); 
  foo.SaveAs("conv.png"); 

  xvarHistLo.SetMarkerStyle(8);
  xvarHistLo.SetMarkerSize(0.5);
  xvarHistLo.Draw("p"); 
  pdfHistLo.SetLineColor(kBlue);
  pdfHistLo.SetLineWidth(3); 
  pdfHistLo.Draw("lsame"); 
  foo.SaveAs("convLo.png"); 

  xvarHistHi.SetMarkerStyle(8);
  xvarHistHi.SetMarkerSize(0.5);
  xvarHistHi.Draw("p"); 
  pdfHistHi.SetLineColor(kBlue);
  pdfHistHi.SetLineWidth(3); 
  pdfHistHi.Draw("lsame"); 
  foo.SaveAs("convHi.png"); 


  
  return 0;
}
