
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
 * Dynamic data version: manages a block.
 */

#ifndef graphchi_xcode_dynamicblock_hpp
#define graphchi_xcode_dynamicblock_hpp

namespace graphchi {
    
    int get_block_uncompressed_size(std::string blockfilename, int defaultsize) {
        std::string szfilename = blockfilename + ".bsize";
        FILE * f = fopen(szfilename.c_str(), "r");
        if (f != NULL) {
            int sz;
            fread(&sz, 1, sizeof(int), f);
            fclose(f);
            return sz;
        } else {
            return defaultsize;
        }
    }
    
    void write_block_uncompressed_size(std::string blockfilename, int size) {
        std::string szfilename = blockfilename + ".bsize";
        FILE * f = fopen(szfilename.c_str(), "w");
        fwrite(&size, 1, sizeof(int), &size);
        fclose(f);
    }
    
    
    template <typename ET>
    struct dynamicdata_block {
        int nedges;
        uint8_t * data;
        ET * chivecs;
        
        dynamicdata_block() : data(NULL), chivecs(NULL) {}
        
        dynamicdata_block(int nedges, uint8_t * data) {
            chivecs = new ET[nedges];
            uint8_t * ptr = data;
            for(int i=0; i < nedges; i++) {
                uint16_t * sz = ((uint16_t *) ptr);
                ptr += sizeof(uint16_t);
                chivecs[i] = ET(sz, ptr);
                ptr += (*sz) * sizeof(typename ET::element_type_t)
            }
        }
        
        ET * edgevec(i) {
            return chivecs[i];
        }
        
        void write(uint8_t ** outdata, int & size) {
            // First compute size
            size = 0;
            for(int i=0; i < chivecs.size(); i++) {
                size += chivecs[i].size() * sizeof(typename ET::element_type_t) + sizeof(uint16_t);
            }
            
            *outdata = (uint8_t *) malloc(size);
            uint8_t * ptr = *outdata;
            for(int i=0; i < chivecs.size(); i++) {
                ET & vec = chivecs[i];
                ((uint16_t *) ptr)[0] = vec.size();
                ptr += sizeof(uint16_t);
                vec.write(ptr);
                ptr += vec.size() * sizeof(typename ET::element_type_t);
            }
        }
        
        ~dynamicdata_block() {
            if (chivecs != NULL)
                delete [] chivecs;
        }
        
    };

    
};


#endif
