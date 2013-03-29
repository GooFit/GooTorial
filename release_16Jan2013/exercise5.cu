#include "Variable.hh" 
#include "PdfBuilder.hh" 
#include "UnbinnedDataSet.hh" 

#include "TRandom.h" 
#include "TH1F.h"
#include "TH2F.h"
#include "TStyle.h" 
#include "TCanvas.h" 
#include "RooNovosibirsk.h" 

#include <sys/time.h>
#include <sys/times.h>
#include <iostream>

using namespace std; 

double novosib (double x, double peak, double width, double tail) {
  double qa=0,qb=0,qc=0,qx=0,qy=0;

  if(fabs(tail) < 1.e-7) 
    qc = 0.5*pow(((x-peak)/width),2);
  else {
    qa = tail*sqrt(log(4.));
    qb = sinh(qa)/qa;
    qx = (x-peak)/width*qb;
    qy = 1.+tail*qx;
  
    //---- Cutting curve from right side

    if( qy > 1.E-7) 
      qc = 0.5*(pow((log(qy)/tail),2) + tail*tail);
    else
      qc = 15.0;
  }

  //---- Normalize the result

  return exp(-qc);
}


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
  UnbinnedDataSet landdata(xvar);
  UnbinnedDataSet bifgdata(xvar);
  UnbinnedDataSet novodata(xvar);

  TH1F landHist("landHist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  TH1F bifgHist("bifgHist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  TH1F novoHist("novoHist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  landHist.SetStats(false); 
  bifgHist.SetStats(false); 
  novoHist.SetStats(false); 

  TRandom donram(42); 
  double totalData = 0; 

  double maxNovo = 0; 
  for (double x = xvar->lowerlimit; x < xvar->upperlimit; x += 0.01) {
    double curr = novosib(x, 0.3, 0.5, 1.0);
    if (curr < maxNovo) continue;
    maxNovo = curr; 
  }

  double leftSigma = 13;
  double rightSigma = 29;
  double leftIntegral = 0.5 / (leftSigma * sqrt(2*M_PI));
  double rightIntegral = 0.5 / (rightSigma * sqrt(2*M_PI));
  double totalIntegral = leftIntegral + rightIntegral; 
  double bifpoint = -10; 

  for (int i = 0; i < 100000; ++i) {
    xvar->value = xvar->upperlimit + 1; 
    while (fabs(xvar->value) > xvar->upperlimit) {
      xvar->value = donram.Landau(-50, 1); 
    }
    landdata.addEvent(); 
    landHist.Fill(xvar->value); 

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

    while (true) {
      xvar->value = donram.Uniform(xvar->lowerlimit, xvar->upperlimit);
      double y = donram.Uniform(0, maxNovo); 
      if (y < novosib(xvar->value, 0.3, 0.5, 1.0)) break;
    }
    novodata.addEvent(); 
    novoHist.Fill(xvar->value); 

    totalData++; 
  }

  // EXERCISE: Write one of LandauThrustFunctor, BifurGaussThrustFunctor, 
  // or NovoSibirskThrustFunctor. Then use your new class to fit one
  // of the distributions created above. If you feel ambitious, do two
  // or all three. 

  // There is no solution for this exercise! However, if you get it to work
  // well, please give me the code and I will put it in the next release of
  // GooFit. 

  UnbinnedDataSet* data = &landdata;
  //data = &bifgdata;
  //data = &novodata;

  ThrustPdfFunctor* total = 0; // Replace with your PDF constructor. 
  
  if (total) {
    total->setData(data);
    PdfFunctor fitter(total);
    fitter.fit(); 
    fitter.getMinuitValues(); 
  }

  TH1F pdfHist("pdfHist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  pdfHist.SetStats(false);

  UnbinnedDataSet grid(xvar);
  double step = (xvar->upperlimit - xvar->lowerlimit)/xvar->numbins;
  for (int i = 0; i < xvar->numbins; ++i) {
    xvar->value = xvar->lowerlimit + (i + 0.5) * step;
    grid.addEvent(); 
  }

  TCanvas foo;

  if (total) {
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
  }

  foo.SetLogy(true); 
  landHist.SetMarkerStyle(8);
  landHist.SetMarkerSize(0.5);
  landHist.Draw("p"); 
  pdfHist.SetLineColor(kBlue);
  pdfHist.SetLineWidth(3); 
  pdfHist.Draw("lsame"); 
  foo.SaveAs("landau.png"); 

  bifgHist.SetMarkerStyle(8);
  bifgHist.SetMarkerSize(0.5);
  bifgHist.Draw("p"); 
  foo.SaveAs("bifurgauss.png"); 

  novoHist.SetMarkerStyle(8);
  novoHist.SetMarkerSize(0.5);
  novoHist.Draw("p"); 
  foo.SaveAs("novosibirsk.png"); 
 
  return 0;
}
