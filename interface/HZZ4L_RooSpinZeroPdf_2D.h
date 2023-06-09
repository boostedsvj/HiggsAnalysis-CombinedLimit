/*****************************************************************************
 * Project: RooFit                                                           *
 *                                                                           *
  * This code was autogenerated by RooClassFactory                            * 
 *****************************************************************************/

#ifndef HZZ4L_ROOSPINZEROPDF_2D
#define HZZ4L_ROOSPINZEROPDF_2D

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

using namespace RooFit;


class HZZ4L_RooSpinZeroPdf_2D : public RooAbsPdf {
protected:

  RooRealProxy kd;
  RooRealProxy kdint;
  RooRealProxy ksmd;
  RooRealProxy fai1;
  RooRealProxy fai2;
  RooRealProxy phi1;
  RooRealProxy phi2;
  RooListProxy _coefList;  //  List of funcficients
  //  TIterator* _coefIter ;    //! Iterator over funcficient lis
  Double_t evaluate() const;
public:
  HZZ4L_RooSpinZeroPdf_2D() {};
  HZZ4L_RooSpinZeroPdf_2D(const char *name, const char *title,
    RooAbsReal& _kd,
    RooAbsReal& _kdint,
    RooAbsReal& _ksmd,
    RooAbsReal& _fai1,
    RooAbsReal& _fai2,
    RooAbsReal& _phi1,
    RooAbsReal& _phi2,
    const RooArgList& inCoefList);

  HZZ4L_RooSpinZeroPdf_2D(const HZZ4L_RooSpinZeroPdf_2D& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new HZZ4L_RooSpinZeroPdf_2D(*this, newname); }
  inline virtual ~HZZ4L_RooSpinZeroPdf_2D() {}

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const;
  const RooArgList& coefList() const { return _coefList; }

  double Integral_T1;
  double Integral_T2;
  double Integral_T3;
  double Integral_T4;
  double Integral_T5;
  double Integral_T6;
  double Integral_T7;
  double Integral_T8;
  double Integral_T9;

private:

  ClassDef(HZZ4L_RooSpinZeroPdf_2D, 1) // Your description goes here...
};
 
#endif
