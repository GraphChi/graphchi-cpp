
/**
 * @file
 * @author  Aapo Kyrola <akyrola@cs.cmu.edu>
 * @version 1.0
 *
 * @section LICENSE
 *
 * Copyright [2012] [Aapo Kyrola, Guy Blelloch, Carlos Guestrin / Carnegie Mellon University]
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 
 *
 * @section DESCRIPTION
 *
 * Analyses output of label propagation algorithms such as connected components
 * and community detection. Memory efficient implementation.
 *
 * @author Aapo Kyrola
 */


#include <vector>
#include <algorithm>
#include <errno.h>
#include <assert.h>
#include <fstream>

#include "io/stripedio.hpp"
#include "logger/logger.hpp"
#include "util/merge.hpp"
#include "util/ioutil.hpp"
#include "util/qsort.hpp"
#include "api/chifilenames.hpp"
#include "engine/auxdata/vertex_data.hpp"


#ifndef DEF_GRAPHCHI_LABELANALYSIS
#define DEF_GRAPHCHI_LABELANALYSIS

using namespace graphchi;

template <typename LabelType>
struct labelcount_tt {
    LabelType label;
    unsigned int count;  // Count excludes the vertex which has its own id as the label. (Important optimization)
    labelcount_tt(LabelType l, int c) : label(l), count(c) {}
    labelcount_tt() {}
};

template <typename LabelType>
bool label_count_greater(const labelcount_tt<LabelType> &a, const labelcount_tt<LabelType> &b) {
    return a.count > b.count;
}

template <typename LabelType>
void analyze_labels(std::string basefilename, int printtop = 20) {    
    typedef labelcount_tt<LabelType> labelcount_t;
    /**
     * NOTE: this implementation is quite a mouthful. Cleaner implementation
     * could be done by using a map implementation. But STL map takes too much
     * memory, and I want to avoid Boost dependency - which would have boost::unordered_map.
     */
    metrics m("labelanalysis");
    stripedio * iomgr = new stripedio(m);
    
    /* Initialize the vertex-data reader */
    vid_t readwindow = 1024 * 1024;
    vid_t numvertices = (vid_t) get_num_vertices(basefilename);
    vertex_data_store<LabelType> * vertexdata =
    new vertex_data_store<LabelType>(basefilename, numvertices, iomgr);
    
    std::vector<labelcount_t> curlabels;
    bool first = true;
    vid_t curvid = 0;
    LabelType * buffer = (LabelType*) calloc(readwindow, sizeof(LabelType));
    
    /* Iterate the vertex values and maintain the top-list */
    vid_t st = 0;
    vid_t en = numvertices - 1;
    
    while(st <= numvertices - 1) {
        en = st + readwindow - 1;
        if (en >= numvertices - 1) en = numvertices - 1;
        
        /* Load the vertex values */
        vertexdata->load(st, en);
        
        int nt = en - st + 1;
        
        /* Mark vertices with its own label with 0xffffffff so they will be ignored */
        for(int i=0; i < nt; i++) { 
            LabelType l = *vertexdata->vertex_data_ptr(i + st);
            if (l == curvid) buffer[i] = 0xffffffff;
            else buffer[i] = l;
            curvid++;
        }
        
        /* First sort the buffer */
        quickSort(buffer, nt, std::less<LabelType>());
        
        /* Then collect */
        std::vector<labelcount_t> newlabels;
        newlabels.reserve(nt);
        LabelType lastlabel = LabelType(0xffffffff);
        for(int i=0; i < nt; i++) {
            if (buffer[i] != LabelType(0xffffffff)) {
                if (buffer[i] != lastlabel) {
                    newlabels.push_back(labelcount_t(buffer[i], 1));
                } else {
                    newlabels[newlabels.size() - 1].count ++;
                }
                lastlabel = buffer[i];
            }
        }
        
        if (first) {
            for(int i=0; i < (int)newlabels.size(); i++) {
                curlabels.push_back(newlabels[i]);
            }
            
        } else {
            /* Merge current and new label counts */
            int cl = 0;
            int nl = 0;
            std::vector< labelcount_t > merged;
            merged.reserve(curlabels.size() + newlabels.size());
            while(cl < (int)curlabels.size() && nl < (int)newlabels.size()) {
                if (newlabels[nl].label == curlabels[cl].label) {
                    merged.push_back(labelcount_t(newlabels[nl].label, newlabels[nl].count + curlabels[cl].count));
                    nl++; cl++;
                } else {
                    if (newlabels[nl].label < curlabels[cl].label) {
                        merged.push_back(newlabels[nl]);
                        nl++;
                    } else {
                        merged.push_back(curlabels[cl]);
                        cl++;
                    }
                }
            }
            while(cl < (int)curlabels.size()) merged.push_back(curlabels[cl++]);
            while(nl < (int)newlabels.size()) merged.push_back(newlabels[nl++]);
            
            curlabels = merged;
        }
        
        first = false;
        st += readwindow;
    }
    
    /* Sort */
    std::sort(curlabels.begin(), curlabels.end(), label_count_greater<LabelType>);
    
    /* Write output file */
    std::string outname = basefilename + ".components";
    std::ofstream resf;
    resf.open(outname.c_str());
    if (resf == NULL) {
        logstream(LOG_ERROR) << "Could not write label outputfile : " << outname << std::endl;
        return;
    }
    for(int i=0; i < (int) curlabels.size(); i++) {
        resf << curlabels[i].label << "," << curlabels[i].count + 1 << std::endl;
    }
    resf.close();
    
    std::cout << "Total number of different labels (components/communities): " << curlabels.size() << std::endl;
    std::cout << "List of labels was written to file: " << outname << std::endl;
    
    for(int i=0; i < (int)std::min((size_t)printtop, curlabels.size()); i++) {
        std::cout << (i+1) << ". label: " << curlabels[i].label << ", size: " << curlabels[i].count + 1 << std::endl;
    }
    
    free(buffer);
    
    delete vertexdata;
    delete iomgr;
}

#endif


