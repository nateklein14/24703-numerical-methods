// LU.cpp
//
// This program decomposes a matrix into LU format and solves a system of equations given a b vector
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

double q(double x);	// heat generation function, x is the position

// declaration of output stream

ofstream data_out("output.txt"); // output data


// the main part of the program - should always be type "int"

int main() {
	
	// variable declarations
    int i,j,k; // loop counters 
    
    const int n = 21; // number of points in discretization
    
    double Tleft = 5.; // left boundary condition
    double Tright = 30;// right boundary condition
    double length = 4.; // length of system
    double tcond = 5.; // thermal conductivity
    double Ri = 1.;
    
    double dx = length/(double(n)-1.); // discretization length
    
    double A[n][n]; // coefficient matrix
    double L[n][n]; // lower triangular matrix
    double U[n][n]; // upper triangular matrix
    
    double x[n]; // solution vector
    double y[n]; // intermediate solution vector
    double b[n]; // right side vector
    
    double sum; // dummy variable to keep track of sums in decomposition

	// coefficient matrix for conduction problem (you could enter or read in any square matrix of size n)
	
  
	for(i=0;i<n;i++) {
		for(j=0;j<n;j++) A[i][j] = 0.;
	}
	
  
  // T only BCs
  
  A[0][0] = 1.;
  A[n-1][n-1] = 1.;
  

  // Q(0) BC
  /*
  A[0][0] = tcond/(2.*dx);
  A[0][2] = -tcond/(2.*dx);
  A[n-1][n-1] = 1.;
  */

  // Q(L) BC 
  /*
	A[0][0] = 1.; 
	A[n-1][n-3] = tcond/(2.*dx);
  A[n-1][n-1] = -tcond/(2.*dx);
  */
	
  for(i=1;i<n-1;i++) { //internal nodes
     double ri = Ri + double(i)*dx;
	   A[i][i+1] = 1./(ri*dx) + 1./(dx*dx);
	   A[i][i] = -2./(dx*dx);
	   A[i][i-1] = 1./(dx*dx) - 1./(ri*dx);
   }
  
  /*
  A[0][0] = 3.00;
  A[0][1] = 6.03;
  A[0][2] = 1.99;
  A[1][0] = 1.27;
  A[1][1] = 4.16;
  A[1][2] = -1.23;
  A[2][0] = 0.99;
  A[2][1] = -4.81;
  A[2][2] = 9.34;
  */
 
   // right side vector
   
   // known temperature boundary conditions
   
   b[0] = Tleft;
   b[n-1] = Tright;
   
   // heat generation at internal nodes
   
   for(i=1;i<n-1;i++) b[i] = -1.*dx*dx/tcond*q(double(i)*dx);
  
   // initialize L and U
   
   for(i=0;i<n;i++) {
	   for(j=0;j<n;j++){
		   L[i][j] = 0.;
		   U[i][j] = 0.;
	   }
   }
    
   for(i=0;i<n;i++) U[i][i] = 1.; // for Crout reduction
  
   // Do the LU decomposition using the Crout reduction
   
   for(i=0;i<n;i++) { // loop over pairs of L columns and U rows. The three levels of loop indicate n^3 behavior
	  
     // first, column i of L
     
   		for(j=i;j<n;j++) { // i is the row
			sum=0.;
			for(k=0;k<=i-1;k++) sum = sum + L[j][k]*U[k][i];
			L[j][i] = A[j][i] - sum;
		}
		
	// second, row i of U
		   
		  for(j=i+1;j<n;j++){ // j is the column
			   sum = 0.;
			   for(k=0;k<=i-1;k++) sum = sum + L[i][k]*U[k][j];
			   U[i][j] = (A[i][j] - sum)/L[i][i];
		   }
   }
   
   // output intermediate data to screen
   
   for(i=0;i<n;i++) {
	   for(j=0;j<n;j++) cout<<A[i][j]<<'\t';
	   cout<<endl;
   }
   cout<<endl;		   
    
   for(i=0;i<n;i++) {
	   for(j=0;j<n;j++) cout<<L[i][j]<<'\t';
	   cout<<endl;
   }
   cout<<endl;	
   
   for(i=0;i<n;i++) {
	   for(j=0;j<n;j++) cout<<U[i][j]<<'\t';
	   cout<<endl;
   }
   cout<<endl;	
   
   // solve the system of equations
   
   // could loop over a series of b vectors here once the decomposition has been done
   
   // first, find the y vector
   
   y[0] = b[0]/L[0][0];
   for(i=1;i<n;i++) {
	   sum = 0.;
	   for(j=0;j<i;j++) sum = sum + L[i][j]*y[j];
	   y[i] = (b[i]-sum)/L[i][i];
   }
   
   for(i=0;i<n;i++) cout<<y[i]<<'\t';
   cout<<endl<<endl;;
   
   // second, find the x vector
   
   x[n-1] = y[n-1];
   for(i=1;i<n;i++) {
	   j = n-i-1;
	   sum = 0;
	   for(k=j+1;k<n;k++) sum = sum + U[j][k]*x[k];
	   x[j] = y[j] - sum;
   }
   
   // output data
   
   for(i=0;i<n;i++) cout<<x[i]<<'\t';
   for(i=0;i<n;i++) data_out<<(i)*dx<<'\t'<<x[i]<<endl;
   cout<<endl;

}	
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double q(double x) {
    double f;
    
    f = 0;
     
  return f;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
