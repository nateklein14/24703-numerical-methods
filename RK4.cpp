// RK4.cpp
//
// This program implements a fourth-order Runge-Kutta algorithm to solve a single first-order ordinary differential equation 
//
//*****************
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <time.h>
#include <stdio.h>
using namespace std;
//*****************

// declaration of functions/subroutines

double f1(double t, double x1, double x2, double x3, double x4);	// the right hand side of the ODE dx/dt = f1(t,x)
double f2(double t, double x1, double x2, double x3, double x4);
double f3(double t, double x1, double x2, double x3, double x4);
double f4(double t, double x1, double x2, double x3, double x4);

// declaration of output stream

ofstream data_out("output.txt"); // output data

int main() {
	
	// variable declarations
    int i,j,k; // loop counters 
    
    double ttotal = 50.; // total time
    double delt = 0.1; // time step
    double t = 0.; // running time
    const int n = int(ttotal/delt); // number of points in discretization after initial value
    
    double k11, k21, k31, k41;
    double k12, k22, k32, k42; // RK parameters
    double k13, k23, k33, k43;
    double k14, k24, k34, k44;
    
    double x10 = 0.; // initial condition
    double x20 = 1.;
    double x30 = 0.;
    double x40 = 0.;
    double x1 = x10;
    double x2 = x20;
    double x3 = x30;
    double x4 = x40;

    
    data_out<<"t\tx1\tx2"<<endl;
    data_out<<t<<'\t'<<x1<<'\t'<<x2<<endl; // initial condition
    
    for(i=1;i<=n;i++) {
	    
	    k11 = delt*f1(t,x1,x2,x3,x4);
      k12 = delt*f2(t,x1,x2,x3,x4);
      k13 = delt*f3(t,x1,x2,x3,x4);
      k14 = delt*f4(t,x1,x2,x3,x4);
	    k21 = delt*f1(t + 0.5*delt, x1 + 0.5*k11, x2 + 0.5*k12, x3 + 0.5*k13, x4 + 0.5*k14);
      k22 = delt*f2(t + 0.5*delt, x1 + 0.5*k11, x2 + 0.5*k12, x3 + 0.5*k13, x4 + 0.5*k14);
      k23 = delt*f3(t + 0.5*delt, x1 + 0.5*k11, x2 + 0.5*k12, x3 + 0.5*k13, x4 + 0.5*k14);
      k24 = delt*f4(t + 0.5*delt, x1 + 0.5*k11, x2 + 0.5*k12, x3 + 0.5*k13, x4 + 0.5*k14);
	    k31 = delt*f1(t + 0.5*delt, x1 + 0.5*k21, x2 + 0.5*k22, x3 + 0.5*k23, x4 + 0.5*k24);
      k32 = delt*f2(t + 0.5*delt, x1 + 0.5*k21, x2 + 0.5*k22, x3 + 0.5*k23, x4 + 0.5*k24);
      k33 = delt*f3(t + 0.5*delt, x1 + 0.5*k21, x2 + 0.5*k22, x3 + 0.5*k23, x4 + 0.5*k24);
      k34 = delt*f4(t + 0.5*delt, x1 + 0.5*k21, x2 + 0.5*k22, x3 + 0.5*k23, x4 + 0.5*k24);
	    k41 = delt*f1(t + delt, x1 + k31, x2 + k32, x3 + k33, x4 + k34);
      k42 = delt*f2(t + delt, x1 + k31, x2 + k32, x3 + k33, x4 + k34);
      k43 = delt*f3(t + delt, x1 + k31, x2 + k32, x3 + k33, x4 + k34);
      k44 = delt*f4(t + delt, x1 + k31, x2 + k32, x3 + k33, x4 + k34);
	    
	    x1 = x1 + (k11 + 2.*k21 + 2.*k31 + k41)/6.;
      x2 = x2 + (k12 + 2.*k22 + 2.*k32 + k42)/6.;
      x3 = x3 + (k13 + 2.*k23 + 2.*k33 + k43)/6.;
      x4 = x4 + (k14 + 2.*k24 + 2.*k34 + k44)/6.;
	    t = t + delt;
	    
	    data_out<<t<<'\t'<<x1<<'\t'<<x2<<endl;
    }
}	
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double f1(double t, double x1, double x2, double x3, double x4) {
  return x3;
}

double f2(double t, double x1, double x2, double x3, double x4) {
  return x4;
}

double f3(double t, double x1, double x2, double x3, double x4) {
  
  double f; 
  f = 2.*exp(-t*0.5) - 3.*x1 + x2;
  return f;
}

double f4(double t, double x1, double x2, double x3, double x4) {

  double f;
  f = x1/3. - 2.*x2;
  return f;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
