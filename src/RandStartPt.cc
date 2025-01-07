#include "HiggsAnalysis/CombinedLimit/interface/RandStartPt.h"
#include "HiggsAnalysis/CombinedLimit/interface/Combine.h"
#include "HiggsAnalysis/CombinedLimit/interface/utils.h"
#include "HiggsAnalysis/CombinedLimit/interface/CascadeMinimizer.h"

#include <vector>
#include <iostream>
#include <cmath>

#include "TMath.h"
#include "TFile.h"
#include "RooArgSet.h"
#include "RooArgList.h"
#include "RooRealVar.h"
#include "RooMinimizer.h"
#include <boost/algorithm/string.hpp>

#include <Math/Minimizer.h>
#include <Math/MinimizerOptions.h>
#include <Math/QuantFuncMathCore.h>
#include <Math/ProbFunc.h>

RandStartPt::RandStartPt(RooAbsReal& nll, std::vector<RooRealVar* > &specifiedvars, std::vector<float> &specifiedvals, bool skipdefaultstart, const std::string& parameterRandInitialValranges, int numrandpts, int verbose, bool fastscan, bool hasmaxdeltaNLLforprof, float maxdeltaNLLforprof, std::vector<std::string> &specifiednuis, std::vector<std::string> &specifiedfuncnames, std::vector<RooAbsReal*> &specifiedfunc, std::vector<float> &specifiedfuncvals, std::vector<std::string> &specifiedcatnames, std::vector<RooCategory*> &specifiedcat, std::vector<int> &specifiedcatvals, unsigned int nOtherFloatingPOI) :
    nll_(nll),
    specifiedvars_(specifiedvars),
    specifiedvals_(specifiedvals),
    skipdefaultstart_(skipdefaultstart), 
    parameterRandInitialValranges_(parameterRandInitialValranges), 
    numrandpts_(numrandpts), 
    verbosity_(verbose),
    fastscan_(fastscan),
    hasmaxdeltaNLLforprof_(hasmaxdeltaNLLforprof),
    maxdeltaNLLforprof_(maxdeltaNLLforprof),
    specifiednuis_(specifiednuis),
    specifiedfuncnames_(specifiedfuncnames),
    specifiedfunc_(specifiedfunc),
    specifiedfuncvals_(specifiedfuncvals),
    specifiedcatnames_(specifiedcatnames),
    specifiedcat_(specifiedcat),
    specifiedcatvals_(specifiedcatvals),
    nOtherFloatingPOI_(nOtherFloatingPOI),
    debug_(false),
    numdebugpts_(0)
    {
        if (!parameterRandInitialValranges_.empty()) {
            RandStartPt::getRangesDictFromInString(parameterRandInitialValranges_, rand_ranges_dict_, prev_dict_);
        }
    }

std::vector<std::vector<float>> RandStartPt::vectorOfPointsToTry (int num, std::map<std::string, std::vector<float>>& ranges_dict, std::map<std::string, std::vector<float>>& prev_dict){
    std::vector<std::vector<float>> wc_vals_vec_of_vec = {};
    int n_prof_params = specifiedvars_.size();

    int n_default_starts = 0;
    if(!skipdefaultstart_) {
        std::vector<float> default_start_pt_vec;
        for (int prof_param_idx = 0; prof_param_idx<n_prof_params; prof_param_idx++){
            default_start_pt_vec.push_back(specifiedvars_[prof_param_idx]->getVal());
        }
        wc_vals_vec_of_vec.push_back(default_start_pt_vec);
        ++n_default_starts;
        // another default starting point from prev values (only if prev value usage requested for any param)
        if (!parameterRandInitialValranges_.empty()) {
            bool modified = false;
            for (int prof_param_idx = 0; prof_param_idx<n_prof_params; prof_param_idx++){
                const auto& poi_name = specifiedvars_[prof_param_idx]->GetName();
                if (prev_dict.find(poi_name) != prev_dict.end()){
                    default_start_pt_vec[prof_param_idx] = prev_dict[poi_name][0];
                    modified = true;
                }
            }
            if (modified) {
                wc_vals_vec_of_vec.push_back(default_start_pt_vec);
                ++n_default_starts;
            }
        }
    }

    // Append the random points to the vector of points to try
    float prof_start_pt_range_max = 20.0; // Default to 20 if we're not asking for custom ranges

    for (int pt_idx = 0; pt_idx<num; pt_idx++){
        std::vector<float> wc_vals_vec;
        for (int prof_param_idx=0; prof_param_idx<n_prof_params; prof_param_idx++) {
            bool no_rand = false;
            if (!parameterRandInitialValranges_.empty()) {
                const auto& poi_name = specifiedvars_[prof_param_idx]->GetName();
                if (prev_dict.find(poi_name) != prev_dict.end()){ //if the value from the previous grid step should be used as the initial value
                    float prev_val = prev_dict[poi_name][0];
                    float prev_err = prev_dict[poi_name][1];
                    float prev_fac = prev_dict[poi_name][2];
                    float prev_lo = prev_val - prev_fac*prev_err;
                    float prev_hi = prev_val + prev_fac*prev_err;
                    prof_start_pt_range_max = std::max(prev_lo, prev_hi);
                    if (prev_fac==0) no_rand = true;
                }
                else if (ranges_dict.find(poi_name) != ranges_dict.end()){   //if the random starting point range for this floating POI was supplied during runtime
                    float rand_range_lo = ranges_dict[poi_name][0];
                    float rand_range_hi = ranges_dict[poi_name][1];
                    prof_start_pt_range_max = std::max(abs(rand_range_lo),abs(rand_range_hi));
                }
                else {   //if the random starting point range for this floating POI was not supplied during runtime, set the default low to -20 and high to +20
                    ranges_dict.insert({poi_name,{-1*prof_start_pt_range_max,prof_start_pt_range_max}});
                }
            }
            //Get a random number in the range [-prof_start_pt_range_max,prof_start_pt_range_max]
            //unless just using prev value directly w/ no variation
            float rand_num = no_rand ? prof_start_pt_range_max : (rand()*2.0*prof_start_pt_range_max)/RAND_MAX - prof_start_pt_range_max;
            wc_vals_vec.push_back(rand_num);
        }
        // avoid redundancy w/ default starting points above
        bool novel = true;
        for (int npt = 0; npt < n_default_starts; ++npt) {
            if (wc_vals_vec==wc_vals_vec_of_vec[npt]) {
                novel = false;
                break;
            }
        }
        if (!novel) continue;
        wc_vals_vec_of_vec.push_back(wc_vals_vec);
    }

    //Print vector of points to try
    if (verbosity_ > 1) {
        std::cout<<"List of points to try for : "<<std::endl;
        for (const auto& vals_vec: wc_vals_vec_of_vec){
            int index = 0;
            std::cout<<"\tThe vals at this point: "<<std::endl;
            for (auto val:vals_vec){
                std::cout << "\t\tPoint val for: "<<specifiedvars_[index]->GetName() <<" = "<< val << std::endl;
                index++;
            }
        }
    }
    return wc_vals_vec_of_vec;
}

// Extract the ranges map from the input string
// Assumes the string is formatted with colons like "poi_name1=lo_lim,hi_lim:poi_name2=lo_lim,hi_lim"
// Alternative form: poi_name1=prev,none:po1_name2=prev,2*err
// where prev indicates the poi value from the previous step in the grid should be used (only makes sense in 1d)
// and x*err indicates that the range should be some factor multiplied by the poi error from the previous step
// if none, no random variation of this poi
void RandStartPt::getRangesDictFromInString(const std::string& params_ranges_string_in, std::map<std::string, std::vector<float>>& ranges_dict, std::map<std::string, std::vector<float>>& prev_dict) {
    std::vector<std::string> params_ranges_string_lst;
    boost::split(params_ranges_string_lst, params_ranges_string_in, boost::is_any_of(":"));
    for (UInt_t p = 0; p < params_ranges_string_lst.size(); ++p) {
        std::vector<std::string> params_ranges_string;
        boost::split(params_ranges_string, params_ranges_string_lst[p], boost::is_any_of("=,"));
        if (params_ranges_string.size() != 3) {
            std::cout << "Error parsing expression : " << params_ranges_string_lst[p] << std::endl;
        }
        std::string wc_name =params_ranges_string[0];
        if (params_ranges_string[1]=="prev"){
            float factor = 0;
            if (params_ranges_string[2]!="none"){
                std::vector<std::string> err_range_string;
                boost::split(err_range_string, params_ranges_string[2], boost::is_any_of("*"));
                if (err_range_string.size()==1) factor = 1;
                else if (err_range_string.size()==2) factor = atof(err_range_string[0].c_str());
                else std::cout << "Error parsing expression : " << params_ranges_string[2] << std::endl;
            }
            //dict form: prev value (updated in place), prev error (updated in place), range factor (constant)
            prev_dict.insert({wc_name,{0,0,factor}});
        }
        else {
            float lim_lo = atof(params_ranges_string[1].c_str());
            float lim_hi = atof(params_ranges_string[2].c_str());
            ranges_dict.insert({wc_name,{lim_lo,lim_hi}});
        }
    }
}

void RandStartPt::commitBestNLLVal(unsigned int idx, float &nllVal, double &probVal){
    bool best = idx==0 or nll_.getVal() < nllVal;
    if (best or debug_){
        if (verbosity_ > 1) std::cout << "Committing point " << idx << " w/ nll " << nll_.getVal() << ", ref nll " << nllVal << ", diff " << nllVal - nll_.getVal() << std::endl;
        Combine::commitPoint(true, /*quantile=*/probVal);
        nllVal = nll_.getVal();

        //update prev values
        if (!prev_dict_.empty() and best){
            for (size_t prof_param_idx=0; prof_param_idx<specifiedvars_.size(); prof_param_idx++) {
                const auto& poi_name = specifiedvars_[prof_param_idx]->GetName();
                if (prev_dict_.find(poi_name) != prev_dict_.end()){
                    prev_dict_[poi_name][0] = specifiedvars_[prof_param_idx]->getVal();
                    prev_dict_[poi_name][1] = specifiedvars_[prof_param_idx]->getError();
					if (verbosity_ > 1) std::cout << "Updating prev value for " << poi_name << ": " << prev_dict_[poi_name][0] << " +- " << prev_dict_[poi_name][1] << std::endl;
                }
            }
        }
    }
}

void RandStartPt::setProfPOIvalues(unsigned int startptIdx, std::vector<std::vector<float>> &nested_vector_of_wc_vals){
    if (verbosity_ > 1) std::cout << "\n\tStart pt idx: " << startptIdx << std::endl;
    for (unsigned int var_idx = 0; var_idx<specifiedvars_.size(); var_idx++){
        if (verbosity_ > 1) std::cout << "\t\tThe var name: " << specifiedvars_[var_idx]->GetName() << std::endl;
        if (verbosity_ > 1) std::cout << "\t\t\tRange before: " << specifiedvars_[var_idx]->getMin() << " " << specifiedvars_[var_idx]->getMax() << std::endl;
        if (verbosity_ > 1) std::cout << "\t\t\t" << specifiedvars_[var_idx]->GetName() << " before setting: " << specifiedvars_[var_idx]->getVal() << " +- " << specifiedvars_[var_idx]->getError() << std::endl;
        specifiedvars_[var_idx]->setVal(nested_vector_of_wc_vals.at(startptIdx).at(var_idx));
        if (verbosity_ > 1) std::cout << "\t\t\tRange after: " << specifiedvars_[var_idx]->getMin() << " " << specifiedvars_[var_idx]->getMax() << std::endl;   
        if (verbosity_ > 1) std::cout << "\t\t\t" << specifiedvars_[var_idx]->GetName() << " after  setting: " << specifiedvars_[var_idx]->getVal() << " +- " << specifiedvars_[var_idx]->getError() << std::endl;   
    }
} 

void RandStartPt::setValSpecifiedObjs(){
    for(unsigned int j=0; j<specifiednuis_.size(); j++){
        specifiedvals_[j]=specifiedvars_[j]->getVal();
    }
    for(unsigned int j=0; j<specifiedfuncnames_.size(); j++){
        specifiedfuncvals_[j]=specifiedfunc_[j]->getVal();
    }
    for(unsigned int j=0; j<specifiedcatnames_.size(); j++){
        specifiedcatvals_[j]=specifiedcat_[j]->getIndex();
    }
}

void RandStartPt::setDebug(int num, const std::string& ranges) {
    debug_ = true;
    numdebugpts_ = num;
    if (!ranges.empty()) {
        RandStartPt::getRangesDictFromInString(ranges, debug_ranges_dict_, debug_prev_dict_);
    }
    //populate debug prev dict from default prev dict
    if (!prev_dict_.empty()){
        for (size_t prof_param_idx=0; prof_param_idx<specifiedvars_.size(); prof_param_idx++) {
            const auto& poi_name = specifiedvars_[prof_param_idx]->GetName();
            if (prev_dict_.find(poi_name) != prev_dict_.end() and debug_prev_dict_.find(poi_name) != debug_prev_dict_.end()){
                debug_prev_dict_[poi_name][0] = prev_dict_[poi_name][0];
                debug_prev_dict_[poi_name][1] = prev_dict_[poi_name][1];
            }
        }
    }
}


void RandStartPt::doRandomStartPt1DGridScan(double &xval, unsigned int poiSize, std::vector<float> &poival, std::vector<RooRealVar* > &poivars, std::unique_ptr <RooArgSet> &param, RooArgSet &snap, float &deltaNLL, double &nll_init, CascadeMinimizer &minimObj, int &status){
    float current_best_nll = 0;
    //the nested vector to hold random starting points to try
    std::vector<std::vector<float>> nested_vector_of_wc_vals = debug_ ? vectorOfPointsToTry (numdebugpts_, debug_ranges_dict_, debug_prev_dict_) : vectorOfPointsToTry (numrandpts_, rand_ranges_dict_, prev_dict_);
    for (unsigned int start_pt_idx = 0; start_pt_idx<nested_vector_of_wc_vals.size(); start_pt_idx++){
        *param = snap;
        poival[0] = xval;
        poivars[0]->setVal(xval);

        //Loop over prof POIs and set their values
        setProfPOIvalues(start_pt_idx, nested_vector_of_wc_vals);

        //now we minimize
        nll_.clearEvalErrorLog();
	deltaNLL = nll_.getVal() - nll_init;
	if (nll_.numEvalErrors() > 0){
            deltaNLL = 9990;
            setValSpecifiedObjs();
            Combine::commitPoint(true, /*quantile=*/0);
            continue;
         }
         bool ok = fastscan_ || (hasmaxdeltaNLLforprof_ && (nll_.getVal() - nll_init) > maxdeltaNLLforprof_) || utils::countFloating(*param)==0 ?
                            true :
                            minimObj.minimize(verbosity_-1);
         if (ok) {
             deltaNLL = nll_.getVal() - nll_init;
             double qN = 2*(deltaNLL);
             double prob = ROOT::Math::chisquared_cdf_c(qN, poiSize + nOtherFloatingPOI_);
             setValSpecifiedObjs();
             //finally, commit best NLL value
             status = minimObj.status();
             commitBestNLLVal(start_pt_idx, current_best_nll, prob);
         }
    }

    if (debug_) debug_ = false;
}

void RandStartPt::doRandomStartPt2DGridScan(double &xval, double &yval, unsigned int poiSize, std::vector<float> &poival, std::vector<RooRealVar* > &poivars, std::unique_ptr <RooArgSet> &param, RooArgSet &snap, float &deltaNLL, double &nll_init, MultiDimFit::GridType gridType, double deltaX, double deltaY, CascadeMinimizer &minimObj, int &status){
    float current_best_nll = 0;
    //the nested vector to hold random starting points to try
    std::vector<std::vector<float>> nested_vector_of_wc_vals = debug_ ? vectorOfPointsToTry (numdebugpts_, debug_ranges_dict_, debug_prev_dict_) : vectorOfPointsToTry (numrandpts_, rand_ranges_dict_, prev_dict_);
    for (unsigned int start_pt_idx = 0; start_pt_idx<nested_vector_of_wc_vals.size(); start_pt_idx++){
        *param = snap;
        poival[0] = xval;
        poival[1] = yval;
        poivars[0]->setVal(xval);
        poivars[1]->setVal(yval);

        //Loop over prof POIs and set their values
        setProfPOIvalues(start_pt_idx, nested_vector_of_wc_vals);
       
        //now we minimize
        nll_.clearEvalErrorLog();
        nll_.getVal();
        deltaNLL = nll_.getVal() - nll_init;
        if (nll_.numEvalErrors() > 0) {
            setValSpecifiedObjs();
            deltaNLL = 9999;
            Combine::commitPoint(true, /*quantile=*/0);
            if (gridType == MultiDimFit::G3x3){
                for (int i2 = -1; i2 <= +1; ++i2){
                    for (int j2 = -1; j2 <= +1; ++j2) {
                        if (i2 == 0 && j2 == 0) continue;
                        poival[0] = xval + 0.33333333*i2*deltaX;
                        poival[1] = yval + 0.33333333*j2*deltaY;
                        setValSpecifiedObjs();
                        deltaNLL = 9999; Combine::commitPoint(true, /*quantile=*/0);
                    }
                }
            }
            continue;
        }
        bool ok = fastscan_ || (hasmaxdeltaNLLforprof_ && (nll_.getVal() - nll_init) > maxdeltaNLLforprof_) ?
                            true :
                            minimObj.minimize(verbosity_-1);
        if (ok) {
            deltaNLL = nll_.getVal() - nll_init;
            double qN = 2*(deltaNLL);
            double prob = ROOT::Math::chisquared_cdf_c(qN, poiSize + nOtherFloatingPOI_);
            setValSpecifiedObjs();
            status = minimObj.status();
            commitBestNLLVal(start_pt_idx, current_best_nll, prob);
        }
        if (gridType == MultiDimFit::G3x3){
            bool forceProfile = !fastscan_  && std::min(fabs(deltaNLL - 1.15), fabs(deltaNLL - 2.995)) < 0.5;
            utils::CheapValueSnapshot center(*param);
            double x0 = xval, y0 = yval;
            for (int i2 = -1; i2 <= +1; ++i2){
                for (int j2 = -1; j2 <= +1; ++j2){
                    if (i2 == 0 && j2 == 0) continue;
                    center.writeTo(*param);
                    xval = x0 + 0.33333333*i2*deltaX;
                    yval = y0 + 0.33333333*j2*deltaY;
                    poival[0] = xval; poivars[0]->setVal(xval);
                    poival[1] = yval; poivars[1]->setVal(yval);
                    nll_.clearEvalErrorLog(); nll_.getVal();
                    if (nll_.numEvalErrors() > 0){
                        setValSpecifiedObjs();
                        deltaNLL = 9999; Combine::commitPoint(true, /*quantile*/0);
                        continue;
                    }
                    deltaNLL = nll_.getVal() - nll_init;
                    if (forceProfile || (fastscan_ && std::min(fabs(deltaNLL - 1.15), fabs(deltaNLL - 2.995)) < 0.5)) {
                        minimObj.minimize(verbosity_-1);
                        deltaNLL = nll_.getVal() - nll_init;
                    }
                    double qN = 2*(deltaNLL);
                    double prob = ROOT::Math::chisquared_cdf_c(qN, poiSize + nOtherFloatingPOI_);
                    setValSpecifiedObjs();
                    status = minimObj.status();
                    commitBestNLLVal(start_pt_idx, current_best_nll, prob);
                }
            }
        }
    }

    if (debug_) debug_ = false;
}
