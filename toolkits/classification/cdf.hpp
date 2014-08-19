/**  
 * Copyright (c) 2013 GraphLab Inc.
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 * For more about this software visit:
              http://graphlab.org
*/
#ifndef CDF_HPP
#define CDF_HPP

#include <cmath>

// IMPLEMENTATION OF GAUSSIAN CFD
// TAKEN FROM : http://www.johndcook.com/cpp_phi.html

// constants
const double phi_a1 =  0.254829592;
const double phi_a2 = -0.284496736;
const double phi_a3 =  1.421413741;
const double phi_a4 = -1.453152027;
const double phi_a5 =  1.061405429;
const double phi_p  =  0.3275911;


double phi(double x)
{
    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x)/sqrt(2.0);

    // A&S formula 7.1.26
    double t = 1.0/(1.0 + phi_p*x);
    double y = 1.0 - (((((phi_a5*t + phi_a4)*t) + phi_a3)*t + phi_a2)*t + phi_a1)*t*exp(-x*x);

    return 0.5*(1.0 + sign*y);
}


#endif
