
// NOTE, copied from GraphLab v 0.5

#ifndef DENSE_BITSET_HPP
#define DENSE_BITSET_HPP
#include <cstdio>
#include <cstdlib>
#include <stdint.h>

namespace graphchi {
    class dense_bitset {
    public:
        dense_bitset() : array(NULL), len(0) {
            generate_bit_masks();
        }
        
        dense_bitset(size_t size) : array(NULL), len(size) {
            resize(size);
            clear();
            generate_bit_masks();
        }
        
        
        virtual ~dense_bitset() {free(array);}
        
        void resize(size_t n) {
            len = n;
            //need len bits
            arrlen =  n / (8*sizeof(size_t)) + 1;
            array = (size_t*)realloc(array, sizeof(size_t) * arrlen);
        }
        
        void clear() {
            for (size_t i = 0;i < arrlen; ++i) array[i] = 0;
        }
        
        void setall() {
            memset(array, 0xff,  arrlen * sizeof(size_t));
        }
        
        inline bool get(uint32_t b) const{
            uint32_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            return array[arrpos] & (size_t(1) << size_t(bitpos));
        }
        
        //! Set the bit returning the old value
        inline bool set_bit(uint32_t b) {
            // use CAS to set the bit
            uint32_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            const size_t mask(size_t(1) << size_t(bitpos)); 
            return __sync_fetch_and_or(array + arrpos, mask) & mask;
        }
        
        //! Set the state of the bit returning the old value
        inline bool set(uint32_t b, bool value) {
            if (value) return set_bit(b);
            else return clear_bit(b);
        }
        
        //! Clear the bit returning the old value
        inline bool clear_bit(uint32_t b) {
            // use CAS to set the bit
            uint32_t arrpos, bitpos;
            bit_to_pos(b, arrpos, bitpos);
            const size_t test_mask(size_t(1) << size_t(bitpos)); 
            const size_t clear_mask(~test_mask); 
            return __sync_fetch_and_and(array + arrpos, clear_mask) & test_mask;
        }
        
        inline void clear_bits(uint32_t fromb, uint32_t tob) { // tob is inclusive
            // Careful with alignment
            const size_t bitsperword = sizeof(size_t)*8;
            while((fromb%bitsperword != 0)) {
                clear_bit(fromb);
                if (fromb>=tob) return;
                fromb++;
            }
            
            while((tob%bitsperword != 0)) {
                clear_bit(tob);
                if(tob<=fromb) return;
                tob--;
            }
            clear_bit(tob);

            uint32_t from_arrpos = fromb / (8 * (int) sizeof(size_t));
            uint32_t to_arrpos = tob / (8 * (int)  sizeof(size_t)); 
            memset(&array[from_arrpos], 0, (to_arrpos-from_arrpos) * (int)  sizeof(size_t));
        }
        
                
        inline size_t size() const {
            return len;
        }
        
    private:
                
        
        inline static void bit_to_pos(uint32_t b, uint32_t &arrpos, uint32_t &bitpos) {
            // the compiler better optimize this...
            arrpos = b / (8 * (int)sizeof(size_t));
            bitpos = b & (8 * (int)sizeof(size_t) - 1);
        }
        
        void generate_bit_masks() {
            below_selectedbit[0] = size_t(-2);
            for (size_t i = 0;i < 8 * sizeof(size_t) ; ++i) {
                selectbit[i] = size_t(1) << i;
                notselectbit[i] = ~selectbit[i];
                if (i > 0)  below_selectedbit[i] = below_selectedbit[i-1] - selectbit[i];
            }
        }
        
        // returns 0 on failure
        inline size_t next_bit_in_block(const uint32_t &b, const size_t &block) {
            // use CAS to set the bit
            size_t x = block & below_selectedbit[b] ;
            if (x == 0) return 0;
            else return __builtin_ctzl(x);
        }
        
        // returns 0 on failure
        inline size_t first_bit_in_block(const size_t &block) {
            // use CAS to set the bit
            if (block == 0) return 0;
            else return __builtin_ctzl(block);
        }
        
        size_t* array;
        size_t len;
        size_t arrlen;
        // selectbit[i] has a bit in the i'th position
        size_t selectbit[8 * sizeof(size_t)];
        size_t notselectbit[8 * sizeof(size_t)];
        size_t below_selectedbit[8 * sizeof(size_t)];
    };
    
}
#endif
