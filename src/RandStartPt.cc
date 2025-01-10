#include "HiggsAnalysis/CombinedLimit/interface/RandStartPt.h"
#include "HiggsAnalysis/CombinedLimit/interface/Combine.h"
#include "HiggsAnalysis/CombinedLimit/interface/utils.h"
#include "HiggsAnalysis/CombinedLimit/interface/CascadeMinimizer.h"

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

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

RandStartPt::RandStartPt(RooAbsReal& nll, std::vector<RooRealVar* > &specifiedvars, std::vector<float> &specifiedvals, std::vector<float> &specifiederrs, bool skipdefaultstart, const std::string& parameterRandInitialValranges, int numrandpts, int verbose, bool fastscan, bool hasmaxdeltaNLLforprof, float maxdeltaNLLforprof, std::vector<std::string> &specifiednuis, std::vector<std::string> &specifiedfuncnames, std::vector<RooAbsReal*> &specifiedfunc, std::vector<float> &specifiedfuncvals, std::vector<std::string> &specifiedcatnames, std::vector<RooCategory*> &specifiedcat, std::vector<int> &specifiedcatvals, unsigned int nOtherFloatingPOI, const std::string& extfilename) :
    nll_(nll),
    specifiedvars_(specifiedvars),
    specifiedvals_(specifiedvals),
    specifiederrs_(specifiederrs),
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
    numdebugpts_(0),
    extfilename_(extfilename),
    extfile_(nullptr),
    exttree_(nullptr)
    {
        //set up reading initial values from external file
        //external file should also have been generated using MultiDimFit with args --saveSpecifiedNuis and --saveSpecifiedNuisErrors
        if (!extfilename_.empty()) {
            extfile_ = TFile::Open(extfilename_.c_str());
            exttree_ = (TTree*)extfile_->Get("limit");
            //copy nuisance vectors to use as branches
            extvals_ = specifiedvals_;
            exterrs_ = specifiederrs_;
            for (unsigned i = 0; i < specifiedvars_.size(); ++i){
                exttree_->SetBranchAddress(specifiedvars_[i]->GetName(), &extvals_[i]);
                exttree_->SetBranchAddress((std::string("error_")+specifiedvars_[i]->GetName()).c_str(), &exterrs_[i]);
            }
        }

        if (!parameterRandInitialValranges_.empty()) {
            RandStartPt::getRangesDictFromInString(parameterRandInitialValranges_, rand_ranges_dict_, prev_dict_, ext_dict_);
        }
    }

//helper function to compute err-dependent ranges
namespace {
bool compute_range(std::map<std::string, std::vector<float>>& val_dict, const std::string& poi_name, float& prof_start_pt_range_max){
    bool no_rand = false;
    float val = val_dict[poi_name][0];
    float err = val_dict[poi_name][1];
    float fac = val_dict[poi_name][2];
    float lo = err!=-1 ? val - fac*err : val - fac*val;
    float hi = err!=-1 ? val + fac*err : val + fac*val;
    prof_start_pt_range_max = std::max(lo, hi);
    if (fac==0) no_rand = true;
    return no_rand;
}
}

std::vector<std::vector<float>> RandStartPt::vectorOfPointsToTry (std::vector<RooRealVar* > &pois, int num, std::map<std::string, std::vector<float>>& ranges_dict, std::map<std::string, std::vector<float>>& prev_dict, std::map<std::string, std::vector<float>>& ext_dict){
    std::vector<std::vector<float>> wc_vals_vec_of_vec = {};
    int n_prof_params = specifiedvars_.size();

    // get ext values from tree
    if (!ext_dict.empty()) {
        std::stringstream scut;
        scut.precision(16);
        for (unsigned p = 0; p < pois.size(); ++p){
            scut << pois[p]->GetName() << "==" << pois[p]->getVal() << "&&";
        }
        std::string cut = scut.str();
        cut.pop_back(); cut.pop_back();
        // if ext file was also generated using pointsRandProf, it may have multiple likelihoods per iteration
        // take the lowest likelihood
        int nentries = exttree_->Draw("Entry$:deltaNLL", cut.c_str(), "goff");
        double* a_entry = exttree_->GetV1();
        double* a_dnll = exttree_->GetV2();
        int argmin = std::distance(a_dnll, std::min_element(a_dnll, a_dnll+nentries));
        int min_entry = a_entry[argmin];
        // for debugging
        if (verbosity_ > 1) {
            std::cout << "Selecting ext value " << cut << " : entry " << min_entry << " w/ dnll " << a_dnll[argmin] << std::endl;
        }
        // load branch values for minimum likelihood for current poi values
        exttree_->GetEntry(min_entry);
        // populate dict
        for (int prof_param_idx=0; prof_param_idx<n_prof_params; prof_param_idx++) {
            const auto& poi_name = specifiedvars_[prof_param_idx]->GetName();
            if (ext_dict.find(poi_name) != ext_dict.end()){
                ext_dict[poi_name][0] = extvals_[prof_param_idx];
                if (ext_dict[poi_name][1]!=-1) ext_dict[poi_name][1] = exterrs_[prof_param_idx];
            }
        }
    }

    int n_default_starts = 0;
    if(!skipdefaultstart_) {
        std::vector<float> default_start_pt_vec;
        for (int prof_param_idx = 0; prof_param_idx<n_prof_params; prof_param_idx++){
            default_start_pt_vec.push_back(specifiedvars_[prof_param_idx]->getVal());
        }
        wc_vals_vec_of_vec.push_back(default_start_pt_vec);
        ++n_default_starts;
        // another default starting point from prev or ext values (only if prev or ext value usage requested for any param)
        if (!parameterRandInitialValranges_.empty()) {
            bool modified = false;
            for (int prof_param_idx = 0; prof_param_idx<n_prof_params; prof_param_idx++){
                const auto& poi_name = specifiedvars_[prof_param_idx]->GetName();
                if (prev_dict.find(poi_name) != prev_dict.end()){
                    default_start_pt_vec[prof_param_idx] = prev_dict[poi_name][0];
                    modified = true;
                }
                else if (ext_dict.find(poi_name) != ext_dict.end()){
                    default_start_pt_vec[prof_param_idx] = ext_dict[poi_name][0];
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
                    no_rand = compute_range(prev_dict, poi_name, prof_start_pt_range_max);
                }
                else if (ext_dict.find(poi_name) != ext_dict.end()){ //if value from external file should be used as initial value
                    no_rand = compute_range(ext_dict, poi_name, prof_start_pt_range_max);
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

//helper function to parse error-dependent ranges
namespace {
bool parse_range_factor(const std::string& params_ranges_string_2, float& factor){
    factor = 0;
    bool use_err = false;
    if (params_ranges_string_2!="none") {
        std::vector<std::string> err_range_string;
        boost::split(err_range_string, params_ranges_string_2, boost::is_any_of("*"));
        if (err_range_string.size()==1) {
            if (err_range_string[0]=="err") { factor = 1; use_err = true; }
            else factor = atof(err_range_string[0].c_str()); //range is relative to poi rather than err in this case
        }
        else if (err_range_string.size()==2) { factor = atof(err_range_string[0].c_str()); use_err = true; }
        else std::cout << "Error parsing expression : " << params_ranges_string_2 << std::endl;
    }
    return use_err;
}
}

// Extract the ranges map from the input string
// Assumes the string is formatted with colons like "poi_name1=lo_lim,hi_lim:poi_name2=lo_lim,hi_lim"
// Alternative form: poi_name1=prev,none:poi_name2=prev,2*err
// where prev indicates the poi value from the previous step in the grid should be used (only makes sense in 1d)
// and x*err indicates that the range should be some factor multiplied by the poi error from the previous step
// if none, no random variation of this poi
// Alternative form 2: poi_name1=ext,none:poi_name2=ext,2*err
// where ext indicates that the poi value from specified external file should be used
void RandStartPt::getRangesDictFromInString(const std::string& params_ranges_string_in, std::map<std::string, std::vector<float>>& ranges_dict, std::map<std::string, std::vector<float>>& prev_dict, std::map<std::string, std::vector<float>>& ext_dict) {
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
            bool use_err = parse_range_factor(params_ranges_string[2], factor);
            //dict form: prev value (updated in place), prev error (updated in place), range factor (constant)
            //-1 means that prev error will not be used
            prev_dict.insert({wc_name,{0, use_err ? 0.f : -1.f, factor}});
        }
        else if (params_ranges_string[1]=="ext"){
            float factor = 0;
            bool use_err = parse_range_factor(params_ranges_string[2], factor);
            ext_dict.insert({wc_name,{0, use_err ? 0.f : -1.f, factor}});
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
                    if (prev_dict_[poi_name][1]!=-1) prev_dict_[poi_name][1] = specifiedvars_[prof_param_idx]->getError();
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
        if (!specifiederrs_.empty()) specifiederrs_[j]=specifiedvars_[j]->getError();
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
        RandStartPt::getRangesDictFromInString(ranges, debug_ranges_dict_, debug_prev_dict_, debug_ext_dict_);
    }
    //populate debug prev dict from default prev dict
    if (!prev_dict_.empty()){
        for (size_t prof_param_idx=0; prof_param_idx<specifiedvars_.size(); prof_param_idx++) {
            const auto& poi_name = specifiedvars_[prof_param_idx]->GetName();
            if (prev_dict_.find(poi_name) != prev_dict_.end() and debug_prev_dict_.find(poi_name) != debug_prev_dict_.end()){
                debug_prev_dict_[poi_name][0] = prev_dict_[poi_name][0];
                if (debug_prev_dict_[poi_name][1]!=-1) debug_prev_dict_[poi_name][1] = prev_dict_[poi_name][1];
            }
        }
    }
}


void RandStartPt::doRandomStartPt1DGridScan(double &xval, unsigned int poiSize, std::vector<float> &poival, std::vector<RooRealVar* > &poivars, std::unique_ptr <RooArgSet> &param, RooArgSet &snap, float &deltaNLL, double &nll_init, CascadeMinimizer &minimObj, int &status){
    float current_best_nll = 0;
    poival[0] = xval;
    poivars[0]->setVal(xval);
    //the nested vector to hold random starting points to try
    std::vector<std::vector<float>> nested_vector_of_wc_vals = debug_ ? vectorOfPointsToTry (poivars, numdebugpts_, debug_ranges_dict_, debug_prev_dict_, debug_ext_dict_) : vectorOfPointsToTry (poivars, numrandpts_, rand_ranges_dict_, prev_dict_, ext_dict_);
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
    poival[0] = xval;
    poival[1] = yval;
    poivars[0]->setVal(xval);
    poivars[1]->setVal(yval);
    //the nested vector to hold random starting points to try
    std::vector<std::vector<float>> nested_vector_of_wc_vals = debug_ ? vectorOfPointsToTry (poivars, numdebugpts_, debug_ranges_dict_, debug_prev_dict_, debug_ext_dict_) : vectorOfPointsToTry (poivars, numrandpts_, rand_ranges_dict_, prev_dict_, debug_ext_dict_);
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
