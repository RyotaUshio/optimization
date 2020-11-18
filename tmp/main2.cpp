#include "continuous.hpp"
#include <Eigen/Dense>
#include <iostream>

using namespace Continuous;
using namespace Eigen;


int main()
{
  // dimension of x \in R^n
  int n = 2;
  
  // initial guess
  VectorXd x0 = VectorXd::Constant(n, 5);
  
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

  gradientDescent grad_solver("gradient_descent_log2.out");
  NewtonsMethod newton_solver("newtons_method_log2.out");
  MatrixXd H0 = MatrixXd::Identity(n, n);
  quasiNewtonMethod quasi_newton_solver("quasi_newton_method_log2.out", H0);
  
  VectorXd
    x_star_grad = grad_solver(prob, x0),
    x_star_newton = newton_solver(prob, x0),
    x_star_quasi_newton = quasi_newton_solver(prob, x0);
}
