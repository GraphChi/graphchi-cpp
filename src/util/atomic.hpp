#ifndef ATOMIC_HPP
#define ATOMIC_HPP

// Note, stolen from GraphLab.

namespace graphchi {
    /**
     * \brief atomic object toolkit
     * 
     * A templated class for creating atomic numbers.
     */
    
    template<typename T>
    class atomic{
    public:
        volatile T value;
        atomic(const T& value = 0) : value(value) { }
        T inc() { return __sync_add_and_fetch(&value, 1);  }
        T dec() { return __sync_sub_and_fetch(&value, 1);  }
        
        
        //! Lvalue implicit cast
        operator T() const { return value; }
        
        //! Performs an atomic increment by 1, returning the new value
        T operator++() { return inc(); }
        
        //! Performs an atomic decrement by 1, returning the new value
        T operator--() { return dec(); }
        
        //! Performs an atomic increment by 'val', returning the new value
        T inc(const T val) { return __sync_add_and_fetch(&value, val);  }
        
        //! Performs an atomic decrement by 'val', returning the new value
        T dec(const T val) { return __sync_sub_and_fetch(&value, val);  }
        
        //! Performs an atomic increment by 'val', returning the new value
        T operator+=(const T val) { return inc(val); }
        
        //! Performs an atomic decrement by 'val', returning the new value
        T operator-=(const T val) { return dec(val); }
        
        //! Performs an atomic increment by 1, returning the old value
        T inc_ret_last() { return __sync_fetch_and_add(&value, 1);  }
        
        //! Performs an atomic decrement by 1, returning the old value
        T dec_ret_last() { return __sync_fetch_and_sub(&value, 1);  }
        
        //! Performs an atomic increment by 1, returning the old value
        T operator++(int) { return inc_ret_last(); }
        
        //! Performs an atomic decrement by 1, returning the old value
        T operator--(int) { return dec_ret_last(); }
        
        //! Performs an atomic increment by 'val', returning the old value
        T inc_ret_last(const T val) { return __sync_fetch_and_add(&value, val);  }
        
        //! Performs an atomic decrement by 'val', returning the new value
        T dec_ret_last(const T val) { return __sync_fetch_and_sub(&value, val);  }
        
        //! Performs an atomic exchange with 'val', returning the previous value
        T exchange(const T val) { return __sync_lock_test_and_set(&value, val);  }

    };
    
    
    /**
     atomic instruction that is equivalent to the following::
     
     if a==oldval, then {    \
     a = newval;           \
     return true;          \
     }
     return false;
     */
    template<typename T>
    bool atomic_compare_and_swap(T& a, const T &oldval, const T &newval) {
        return __sync_bool_compare_and_swap(&a, oldval, newval);
    };
    
    
    template <>
    inline bool atomic_compare_and_swap(double& a, const double &oldval, const double &newval) {
        return __sync_bool_compare_and_swap(reinterpret_cast<uint64_t*>(&a), 
                                            *reinterpret_cast<const uint64_t*>(&oldval), 
                                            *reinterpret_cast<const uint64_t*>(&newval));
    };
    
    template <>
    inline bool atomic_compare_and_swap(float& a, const float &oldval, const float &newval) {
        return __sync_bool_compare_and_swap(reinterpret_cast<uint32_t*>(&a), 
                                            *reinterpret_cast<const uint32_t*>(&oldval), 
                                            *reinterpret_cast<const uint32_t*>(&newval));
    };
    
    template<typename T>
    void atomic_exchange(T& a, T& b) {
        b =__sync_lock_test_and_set(&a, b);
    };
    
}
#endif
