#include "continuous.hpp"
#include <iostream>
#include <functional>
#include <vector>
#include <exception>
#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace Continuous; //// Continuous Optimization Problem ////


void Continuous::set_x0(VectorXd& x0, int argc, char* argv[]) // set the inital guess x0 from command line
{
  int c;
  double base = 10;
  double exp = 0;
  while ((c = getopt(argc, argv, "k:e:b:")) != -1 ) {
    switch (c)
      {
      case 'k':
	iterativeSolver::k_max = std::atoi(optarg);
	break;
      case 'e':
	exp = std::atof(optarg);
	break;
      case 'b':
	base = std::atof(optarg);
	break;
      }
  }
  if (exp)
    iterativeSolver::eps = std::pow(base, exp);

  int x0_argc = argc - optind;
  int i;
  if (x0_argc)
    {
      if (x0_argc == 1)
      	{
      	  for(i=0; i<x0.size(); i++)
  	    {
  	      x0(i) = std::atof(argv[optind]);
  	    }
      	}
      else
      	{
      	  if (x0_argc != x0.size())
      	    {
      	      std::cerr << "Continuous::set_x0(): "
      			<< "Number of command line arguments do not match x0.size()"
      			<< std::endl;
      	      std::exit(1);
      	    }

      	  for(i=0; i<x0.size(); i++)
  	    {
  	      x0(i) = std::atof(argv[i+optind]);
  	    }
  	}
    }

  // std::cout << "x0 = " << x0.transpose() << std::endl;
  // std::cout << "k_max = " << iterativeSolver::k_max << std::endl;
  // std::cout << "eps = " << iterativeSolver::eps << std::endl;
}



//// objFunc: objective function class
objFunc::objFunc(funcType f)
  : func(f) {}

objFunc::objFunc(funcType f, gradType g)
  : func(f), grad(g) {}

objFunc::objFunc(funcType f, gradType g, hesseType h)
  : func(f), grad(g), hesse(h) {}

double objFunc::operator()(VectorXd x) const
{
  return func(x);
}

  

//// problem: Optimization problem class
problem::problem(objFunc& func)
  : f(func) {}

problem::problem(objFunc& func, constraint& cons)
  : f(func), c(&cons) {}



//// iterativeSolver: optimization problem solver with iterative method    
iterativeSolver::iterativeSolver() {}
//  : eps(std::pow(10, -8)) {}

double iterativeSolver::eps = std::pow(10, -8);
int iterativeSolver::k_max = std::numeric_limits<int>::max();


bool iterativeSolver::converge(problem& prob, VectorXd& x) const // convergence test
{
  double grad_norm = (prob.f.grad(x)).norm();
  return (grad_norm < eps) or (k > k_max);
}

VectorXd iterativeSolver::operator()(problem& prob, VectorXd& x0) // body
{
  VectorXd x = x0;
  k = 0;
  while (not converge(prob, x))
    {
      x = update(prob, x);
      k++;
    }
  return x;
}   



//// base class for solver with line search algorithm
lineSearchSolver::lineSearchSolver(bool wolfe)
  : c1(0.0001), c2(0.9), rho(0.5), alpha0(1.0), use_wolfe(wolfe), log(false) {}

lineSearchSolver::lineSearchSolver(const char* filename, bool wolfe)
  : lineSearchSolver(wolfe)
{
  log = true;
  logname = filename;
  logout.open(filename, std::ios::out);
}

// step size alpha
double lineSearchSolver::alpha(problem& prob, VectorXd& x, VectorXd& d)
{
  return use_wolfe ? alpha_wolfe(prob, x, d) : alpha_armijo(prob, x, d);
}   

// Armijo's condition
bool lineSearchSolver::Armijo(problem& prob, VectorXd& x, double a, VectorXd& d) const
{
  double
    lhs = prob.f(x + a*d),
    rhs = prob.f(x) + c1*a*(prob.f.grad(x)).dot(d);
  return lhs <= rhs;
}
    
// curvature condition of Wolfe's condition
bool lineSearchSolver::curvature_condition(problem& prob, VectorXd& x, double a, VectorXd& d) const
{
  double
    lhs = (prob.f.grad(x + a*d)).dot(d),
    rhs = (prob.f.grad(x)).dot(d) * c2;
  return lhs >= rhs;
}

// backtracking
double lineSearchSolver::alpha_armijo(problem& prob, VectorXd& x, VectorXd& d)
{
  double a = alpha0;
  while(not Armijo(prob, x, a, d))
    a *= rho;
  return a;
}

// return alpha that satisfying Wolfe's condition
double lineSearchSolver::alpha_wolfe(problem& prob, VectorXd& x, VectorXd& d)
{
  double amin = 0, amax = alpha0, a;
  while (Armijo(prob, x, amax, d))
    amax *= 2.0;
  while (true)
    {
      a = (amin + amax) / 2.0;
      if (not Armijo(prob, x, a, d))
	amax = a;
      else if (curvature_condition(prob, x, a, d))	    
	return a;
      else
	amin = a;
    }
}

// improve approximate solution x for the next step of iteration
VectorXd lineSearchSolver::update(problem& prob, VectorXd& x)
{
  VectorXd d = dir(prob, x);
  double a = alpha(prob, x, d);
  VectorXd x_new = x + a*d;
  if (log)
    logout << x.transpose() << " " << a << std::endl;
  return x_new;
}


  
//// Gradient Descent solver class
gradientDescent::gradientDescent(bool wolfe)
  : lineSearchSolver(wolfe) {}

gradientDescent::gradientDescent(const char* filename, bool wolfe)
  : lineSearchSolver(filename, wolfe) {}

// search direction d: steepest descent direction
VectorXd gradientDescent::dir(problem& prob, VectorXd& x)
{
  return - prob.f.grad(x);
}



//// Newton's Method solver class
NewtonsMethod::NewtonsMethod(bool line_search, bool wolfe)
  : lineSearchSolver(wolfe), use_line_search(line_search) {}

NewtonsMethod::NewtonsMethod(const char* filename, bool line_search, bool wolfe)
  : lineSearchSolver(filename, wolfe), use_line_search(line_search) {}

// search direction d: the Newton direction
VectorXd NewtonsMethod::dir(problem& prob, VectorXd& x)
{
  return (prob.f.hesse(x)).colPivHouseholderQr().solve(-(prob.f.grad(x)));
}

// step size alpha
double NewtonsMethod::alpha(problem& prob, VectorXd& x, VectorXd& d)
{
  return (not use_line_search) ? 1.0 : lineSearchSolver::alpha(prob, x, d);
}



//// Quasi-Newton Method solver class
quasiNewtonMethod::quasiNewtonMethod(Eigen::MatrixXd H0, std::string method, bool wolfe)
  : lineSearchSolver(wolfe)
{
  set_H(H0);
  set_hesse_method(method);
}

quasiNewtonMethod::quasiNewtonMethod(const char* filename, Eigen::MatrixXd H0, std::string method, bool wolfe)
  : lineSearchSolver(filename, wolfe)
{
  set_H(H0);
  set_hesse_method(method);
}

void quasiNewtonMethod::set_H(Eigen::MatrixXd H0)
{
  if (H0.rows() != H0.cols())
    {
      std::cerr << "Continuous::quasiNewtonMethod::quasiNewtonMethod(): "
		<< "H0 must be a square matrix"
		<< std::endl;
      std::exit(1);
    }
  H = H0;
}

void quasiNewtonMethod::set_hesse_method(std::string method)
{
  if (method != "BFGS" and method != "DFP")
	{
	  std::cerr << "Continuous::quasiNewtonMethod::set_hesse_method(): "
		    << "Hessian approximation method must be \"BFGS\" or \"DFP\""
		    << std::endl;
	  std::exit(1);
	}
  hesse_method = method;
}

// search direction d
Eigen::VectorXd quasiNewtonMethod::dir(problem& prob, Eigen::VectorXd& x)
{
  return -H*prob.f.grad(x);
}

// update H with the BFGS formula
void quasiNewtonMethod::BFGS (problem& prob, Eigen::VectorXd& x_old, Eigen::VectorXd& x_new)
{
  Eigen::VectorXd
	s = x_new - x_old,
	y = prob.f.grad(x_new) - prob.f.grad(x_old);
  int dim = H.rows(); // dimension of H
  double denom = s.dot(y);
  Eigen::MatrixXd tmp = Eigen::MatrixXd::Identity(dim, dim) - y*(s.transpose()) / denom;
  H = (tmp.transpose())*H*tmp + s*(s.transpose())/denom;
}

// update H with the DFP formula
void quasiNewtonMethod::DFP(problem& prob, Eigen::VectorXd& x_old, Eigen::VectorXd& x_new)
{
  Eigen::VectorXd
	s = x_new - x_old,
	y = prob.f.grad(x_new) - prob.f.grad(x_old);
  H = H - H*y*(y.transpose())*H/(y.dot(H*y)) + s*(s.transpose())/(y.dot(s));
}

Eigen::VectorXd quasiNewtonMethod::update(problem& prob, Eigen::VectorXd& x)
{
  // update x
  Eigen::VectorXd x_new = lineSearchSolver::update(prob, x);
  // update H
  (hesse_method == "BFGS") ? BFGS(prob, x, x_new) : DFP(prob, x, x_new);
  return x_new;
}
