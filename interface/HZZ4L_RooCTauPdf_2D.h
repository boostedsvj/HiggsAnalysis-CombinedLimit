/*****************************************************************************
 * Project: RooFit                                                           *
 *                                                                           *
  * This code was autogenerated by RooClassFactory                            * 
 *****************************************************************************/

#ifndef HZZ4L_ROOCTAUPDF_2D
#define HZZ4L_ROOCTAUPDF_2D

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooRealVar.h"
#include "RooCategoryProxy.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include "TH3F.h"
#include "TH1.h"
#include "RooDataHist.h"
#include "RooHistFunc.h"
#include "RooListProxy.h"


class HZZ4L_RooCTauPdf_2D : public RooAbsPdf {
protected:

	RooRealProxy kd;
	RooRealProxy ksmd;
	RooRealProxy ctau;

	RooListProxy _coefList;  //  List of histogram pdfs
	Double_t evaluate() const;
public:
	HZZ4L_RooCTauPdf_2D() {};
	HZZ4L_RooCTauPdf_2D(
		const char *name,
		const char *title,
		RooAbsReal& _kd,
		RooAbsReal& _ksmd,
		RooAbsReal& _ctau,
		const RooArgList& inCoefList,

		double _ctau_min,
		double _ctau_max
		);

	HZZ4L_RooCTauPdf_2D(const HZZ4L_RooCTauPdf_2D& other, const char* name = 0);
	virtual TObject* clone(const char* newname) const { return new HZZ4L_RooCTauPdf_2D(*this, newname); }
	inline virtual ~HZZ4L_RooCTauPdf_2D() { delete[] Integral_T; }

	Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName = 0) const;
	Double_t analyticalIntegral(Int_t code, const char* rangeName = 0) const;
	const RooArgList& coefList() const { return _coefList; }

	int nbins_ctau;
	double* Integral_T;
	double ctau_min;
	double ctau_max;

private:
	int findNeighborBins() const; // Returns index_low for ctau pdfs
	Double_t interpolateBin() const;
	Double_t interpolateIntegral() const;

	ClassDef(HZZ4L_RooCTauPdf_2D, 1) // Your description goes here...
};
 
#endif
