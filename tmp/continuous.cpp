#include "continuous.hpp"
#include <iostream>
#include <functional>
#include <vector>
#include <cmath>
#include <cstdio>
#include <Eigen/Dense>

using namespace Eigen;
using namespace Continuous; //// Continuous Optimization Problem ////


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
iterativeSolver::iterativeSolver()
  : eps(std::pow(10, -8)) {}


bool iterativeSolver::converge(problem& prob, VectorXd& x) // convergence test
{
  double grad_norm = (prob.f.grad(x)).norm();
  return (grad_norm < eps);
}

VectorXd iterativeSolver::operator()(problem& prob, VectorXd& x0) // body
{
  VectorXd x = x0;
  while (not converge(prob, x))
    x = update(prob, x);
  return x;
}   



//// base class for solver with line search algorithm
lineSearchSolver::lineSearchSolver(bool wolfe)
  : c1(0.0001), c2(0.9), rho(0.5), alpha0(1.0), use_wolfe(wolfe) {}


// step size alpha
double lineSearchSolver::alpha(problem& prob, VectorXd& x, VectorXd& d)
{
  return use_wolfe ? alpha_wolfe(prob, x, d) : alpha_armijo(prob, x, d);
}   

// Armijo's condition
bool lineSearchSolver::Armijo(problem& prob, VectorXd& x, double a, VectorXd& d)
{
  double lhs, rhs;
  lhs = prob.f(x + a*d);
  rhs = prob.f(x) + c1*a*(prob.f.grad(x)).dot(d);
  return lhs <= rhs;
}
    
// curvature condition of Wolfe's condition
bool lineSearchSolver::curvature_condition(problem& prob, VectorXd& x, double a, VectorXd& d)
{
  double lhs, rhs;
  lhs = (prob.f.grad(x + a*d)).dot(d);
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
  double amin = 0;
  double amax = alpha0;
  while (Armijo(prob, x, amax, d))
    amax *= 2.0;
  double a;
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
  return x + a*d;
}


  
//// Gradient Descent solver class
gradientDescent::gradientDescent(bool wolfe)
  : lineSearchSolver(wolfe) {}

// search direction d: steepest descent direction
VectorXd gradientDescent::dir(problem& prob, VectorXd& x)
{
  return - prob.f.grad(x);
}



//// Newton's Method solver class
NewtonsMethod::NewtonsMethod(bool line_search, bool wolfe)
  : lineSearchSolver(wolfe), use_line_search(line_search) {}
    
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
