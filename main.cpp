#include <Eigen/Core>
#include "continuous.hpp"
#include <iostream>
#include <cmath>

using namespace Eigen;
using namespace Continuous;


int main()
{
  // initial guess
  VectorXd x0(2);
  x0 << 1.2, 1.2;
  
  // objective function definition
  objFunc f(
	    // objective function itself
	    [](VectorXd x) -> double
	    {
	      return std::pow(100 * (x(1)-x(0)*x(0)), 2) + std::pow(1 - x(0), 2);
	    },
	    // gradient
	    [](VectorXd x) -> VectorXd 
	    {
	      VectorXd g(2);
	      g << -400 * x(0) * (x(1) - x(0)*x(0)) + 2 * (x(0) - 1),
		200 * (x(1) - x(0)*x(0));
	      return g;
	    },
	    // Hessian matrix
	    [](VectorXd x) -> MatrixXd
	    {
	      MatrixXd h(2, 2);
	      h << 400 * (x(0)*x(0) + 2*x(0) - x(1)) + 2,
		-400*x(0),
		-400*x(0),
		200;
	      return h;
	    }
	    );

  problem prob(f);
  gradientDescent solver;
  solver.eps = 0.0000001;
  VectorXd x_star = solver(prob, x0);
  std::cout << x_star << std::endl;
}
