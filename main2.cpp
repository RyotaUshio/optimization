#include <Eigen/Core>
#include "continuous.hpp"
#include <iostream>
#include <cmath>

using namespace Eigen;
using namespace Continuous;


int main()
{
  // dimension of x \in R^n
  int n = 20;
  
  // initial guess
  VectorXd x0 = VectorXd::Constant(n, 12);
  
  // objective function definition
  auto g = [](int n) -> objFunc
    {
      return objFunc(
		     // objective function itself
		     [n](VectorXd x) -> double
    {
      double sum;
      int i;
      for(sum=0, i=0; i<n-1; i++)
	sum += 100 * std::pow(x(i+1)-x(i)*x(i), 2) + std::pow(1 - x(i), 2);
      return sum;
    },
	    
		     // gradient
		     [n](VectorXd x) -> VectorXd 
    {
      VectorXd g = VectorXd::Zero(n);
      for(int i=0; i<n-1; i++)
	{
	  g(i) += -400 * x(i) * (x(i+1) - x(i)*x(i)) + 2 * (x(i) - 1);
	  g(i+1) += 200 * (x(i+1) - x(i)*x(i));
	}
      return g;
    },
	    
		     // Hessian matrix
		     [n](VectorXd x) -> MatrixXd
    {
      MatrixXd h = MatrixXd::Zero(n, n);
      for(int i=0; i<n-1; i++)
	{
	  h(i, i)     +=  400 * (3*x(i)*x(i) - x(i+1)) + 2;
	  h(i, i+1)   += -400*x(i);
	  h(i+1, i)   += -400*x(i);
	  h(i+1, i+1) +=  200;
	}
      return h;
    }
		     );      
    };
  
  objFunc g_n = g(n);
  problem prob(g_n);
  //gradientDescent solver(true);
  Newton solver;
  VectorXd x_star = solver(prob, x0);
  std::cout << x_star << std::endl;
}
