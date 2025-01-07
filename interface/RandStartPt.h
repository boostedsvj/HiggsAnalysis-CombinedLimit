#ifndef HiggsAnalysis_CombinedLimit_RandStartPt_h
#define HiggsAnalaysis_CombinedLimit_RandStartPt_h

#include "HiggsAnalysis/CombinedLimit/interface/CascadeMinimizer.h"
#include "HiggsAnalysis/CombinedLimit/interface/MultiDimFit.h"

#include <map>
#include <vector>
#include <string>
#include "RooRealVar.h"
#include "RooAbsReal.h"
#include "RooArgSet.h"
#include "RooCategory.h"

class RandStartPt {
  private:
      RooAbsReal& nll_;
      std::vector<RooRealVar* > &specifiedvars_;
      std::vector<float> &specifiedvals_;
      bool skipdefaultstart_;
      std::string parameterRandInitialValranges_;
      int numrandpts_;
      int verbosity_;
      bool fastscan_;
      bool hasmaxdeltaNLLforprof_;
      float maxdeltaNLLforprof_;
      std::vector<std::string> &specifiednuis_;
      std::vector<std::string> &specifiedfuncnames_;
      std::vector<RooAbsReal*> &specifiedfunc_;
      std::vector<float> &specifiedfuncvals_; 
      std::vector<std::string> &specifiedcatnames_; 
      std::vector<RooCategory*> &specifiedcat_; 
      std::vector<int> &specifiedcatvals_;
      unsigned int nOtherFloatingPOI_;
      std::map<std::string, std::vector<float>> rand_ranges_dict_;
      std::map<std::string, std::vector<float>> prev_dict_;
      bool debug_;
      int numdebugpts_;
      std::map<std::string, std::vector<float>> debug_ranges_dict_;
      std::map<std::string, std::vector<float>> debug_prev_dict_;
  public:
      RandStartPt(RooAbsReal& nll, std::vector<RooRealVar* > &specifiedvars, std::vector<float> &specifiedvals, bool skipdefaultstart, const std::string& parameterRandInitialValranges, int numrandpts, int verbose, bool fastscan, bool hasmaxdeltaNLLforprof, float maxdeltaNLLforprof, std::vector<std::string> &specifiednuis, std::vector<std::string> &specifiedfuncnames, std::vector<RooAbsReal*> &specifiedfunc, std::vector<float> &specifiedfuncvals, std::vector<std::string> &specifiedcatnames, std::vector<RooCategory*> &specifiedcat, std::vector<int> &specifiedcatvals, unsigned int nOtherFloatingPOI);
      void getRangesDictFromInString(const std::string& params_ranges_string_in, std::map<std::string, std::vector<float>>& ranges_dict, std::map<std::string, std::vector<float>>& prev_dict);
      std::vector<std::vector<float>> vectorOfPointsToTry (int num, std::map<std::string, std::vector<float>>& ranges_dict, std::map<std::string, std::vector<float>>& prev_dict);
      void commitBestNLLVal(unsigned int idx, float &nllVal, double &probVal);
      void setProfPOIvalues(unsigned int startptIdx, std::vector<std::vector<float>> &nested_vector_of_wc_vals);
      void setValSpecifiedObjs();
      void setDebug(int num, const std::string& ranges);
      void doRandomStartPt1DGridScan(double &xval, unsigned int poiSize, std::vector<float> &poival, std::vector<RooRealVar* > &poivars, std::unique_ptr <RooArgSet> &param, RooArgSet &snap, float &deltaNLL, double &nll_init, CascadeMinimizer &minimObj, int &status);
      void doRandomStartPt2DGridScan(double &xval, double &yval, unsigned int poiSize, std::vector<float> &poival, std::vector<RooRealVar* > &poivars, std::unique_ptr <RooArgSet> &param, RooArgSet &snap, float &deltaNLL, double &nll_init, MultiDimFit::GridType gridType, double deltaX, double deltaY, CascadeMinimizer &minimObj, int &status);

};
#endif
