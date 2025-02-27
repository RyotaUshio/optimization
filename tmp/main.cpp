#include "continuous.hpp"
#include <Eigen/Core>
#include <iostream>

using namespace Continuous;
using namespace Eigen;


int main(int argc, char* argv[])
{
  // initial guess
  VectorXd x0(2);
  x0 << 1.2, 1.2;
  set_x0(x0, argc, argv);

  
  // objective function definition
  objFunc f(
	    // objective function itself
	    [](VectorXd x) -> double
	    {
	      return 100 * std::pow(x(1)-x(0)*x(0), 2) + std::pow(1 - x(0), 2);
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
	      h << 400 * (3*x(0)*x(0) - x(1)) + 2,
		-400*x(0),
		-400*x(0),
		200;
	      return h;
	    }
	    );


  problem prob(f);
  gradientDescent grad_solver("gradient_descent_log.out");
  NewtonsMethod newton_solver("newtons_method_log.out");
  MatrixXd H0 = MatrixXd::Identity(2, 2);
  //MatrixXd H0 = f.hesse(x0);
  quasiNewtonMethod quasi_newton_solver("quasi_newton_method_log.out", H0);

  VectorXd
    x_star_grad = grad_solver(prob, x0),
    x_star_newton = newton_solver(prob, x0),
    x_star_quasi_newton = quasi_newton_solver(prob, x0);

  std::cout << "gradientDescent: (" << x_star_grad.transpose() << "), k = "
  	    << grad_solver.k << std::endl
  	    << "newtonsMethod: (" << x_star_newton.transpose() << "), k = "
  	    << newton_solver.k << std::endl
  	    << "quasiNewtonMethod: (" << x_star_quasi_newton.transpose() << "), k = "
  	    << quasi_newton_solver.k << std::endl;
}
