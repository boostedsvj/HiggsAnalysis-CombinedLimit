/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooPower.h,v 1.10 2007/07/12 20:30:49 wouter Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef HiggsAnalysis_CombinedLimit_HGGRooPdfs_h
#define HiggsAnalysis_CombinedLimit_HGGRooPdfs_h

#include "RooAbsPdf.h"
#include "RooRealProxy.h"

class RooRealVar;
class RooAbsReal;

class RooPowerCMS : public RooAbsPdf {
public:
  RooPowerCMS() {} ;
  RooPowerCMS(const char *name, const char *title,
		 RooAbsReal& _x, RooAbsReal& _c);
  RooPowerCMS(const RooPowerCMS& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooPowerCMS(*this,newname); }
  inline virtual ~RooPowerCMS() { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const ;
  const RooAbsReal & base() const { return x.arg(); }
  const RooAbsReal & exponent() const { return c.arg(); }

protected:
  RooRealProxy x;
  RooRealProxy c;

  Double_t evaluate() const;

private:
  ClassDef(RooPowerCMS,1) // Exponential PDF
};

#endif
