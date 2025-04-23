/*
Nate Klein
24-703 HW #3
Estimates roots of nonlinear equations.
*/

#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <time.h>
#include <vector>
using namespace std;

ifstream data_in("input.txt");
ofstream data_out("output.txt");

float f1(float x);
float f2(float x);

int main() {
    
    // iterator number and other temp variables
    int i;

    // bracketing values
    float bounds[4];
    const float dx = 0.01;

    float x;
    float val;

    // read in the global bounds for both functions
    for(i=0;i<4;i++) data_in>>bounds[i];

    // Function 1
    data_out<<"Roots for f1(x)"<<endl;

    // loop through the bracketed region
    for (i=0;i<((bounds[1]-bounds[0])/dx);i++) {

        x = bounds[0] + i*dx;

        // check for sign changes
        if (f1(x) / f1(x+dx) < 0) {

            // set the initial guess and get the value at the guess
            val = f1(x);

            // while the error > 10^-6, do the N-R method
            while (abs(val) > pow(10, -6)) {
                // forward difference approximtaion
                float dfdx = (f1(x+dx) - val) / dx;
                // N-R method
                x = x - val/dfdx;
                // update value
                val = f1(x);
            }
            data_out<<x<<endl;
        }
    }

}

// Equations to find the roots of
float f1(float x) {
    float fx = exp(x)+2.*x-3.;
    return fx;
}

float f2(float x) {
    float fx = exp(x) + x*x - 3*x - 2;
    return fx;
}