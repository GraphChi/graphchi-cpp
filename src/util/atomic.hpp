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
        T inc(T val) { return __sync_add_and_fetch(&value, val);  }
        T dec(T val) { return __sync_sub_and_fetch(&value, val);  }
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
