#include <Eigen/Core>
#include "continuous.hpp"
#include <iostream>
#include <cmath>

using namespace Eigen;
using namespace Continuous;


// definition of objective function
double func(VectorXd x)
{
  return std::pow(100 * (x(1)-x(0)*x(0)), 2) + std::pow(1 - x(0), 2);
}

// gradient
VectorXd grad(VectorXd x)
{
  VectorXd g(2);
  g << -400 * x(0) * (x(1) - x(0)*x(0)) + 2 * (x(0) - 1),
    200 * (x(1) - x(0)*x(0));

  return g;
}

// Hessian matrix
MatrixXd hesse(VectorXd x)
{
  MatrixXd h(2, 2);
  h << 400 * (x(0)*x(0) + 2*x(0) - x(1)) + 2,
    -400*x(0),
    -400*x(0),
    200;

  return h;
}


int main()
{
  VectorXd x0(2);
  x0 << 1.2, 1.2;
  
  objFunc f(func, grad, hesse);
  problem prob(f);
  gradientDescent solver;
  solver.eps = 0.1;
  VectorXd x_star = solver(prob, x0);
  std::cout << x_star << std::endl;
}
