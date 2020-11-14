#ifndef __CONTINUOUS_HPP_INCLUDED__
#define __CONTINUOUS_HPP_INCLUDED__


#include <iostream>
#include <functional>
#include <vector>
#include <cmath>
#include <cstdio>
#include <Eigen/Dense>

using std::cout;
using std::cerr;
using std::endl;
using namespace Eigen;
using funcType = std::function<double(VectorXd)>;
using gradType = std::function<VectorXd(VectorXd)>;
using hesseType = std::function<MatrixXd(VectorXd)>;


namespace Continuous {
  //// Continuous Optimization Problem ////
  
  struct objFunc {
    //// objective function class
    const funcType func; // objective function itself
    const gradType grad; // gradient
    const hesseType hesse; // Hessian matrix


    objFunc(funcType f)
      : func(f) {}
    objFunc(funcType f, gradType g)
      : func(f), grad(g) {}
    objFunc(funcType f, gradType g, hesseType h)
      : func(f), grad(g), hesse(h) {}

    double operator()(VectorXd x) const
    {
      return func(x);
    }
  };

  

  // base class for constraints
  struct constraint {};

  
  // equality constraints
  struct eqConstraint: public constraint {
    std::vector<funcType> g;
  };


  // inequality constraints
  struct ineqConstraint: public eqConstraint {
    std::vector<funcType> h;
  };

  

  // Optimization problem class
  struct problem {
    const objFunc f;
    const constraint* c;

    problem(objFunc& func)
      : f(func) {}
    problem(objFunc& func, constraint& cons)
      : f(func), c(&cons) {}
  };



  //// optimization problem solver with iterative method
  struct iterativeSolver {
    
    double eps; // tolerant norm of gradient vector

    
    iterativeSolver()
      : eps(std::pow(10, -8)) {}


    bool converge(problem& prob, VectorXd& x) // convergence test
    {
      double grad_norm = (prob.f.grad(x)).norm();
      return (grad_norm < eps);
    }

    
    VectorXd operator()(problem& prob, VectorXd& x0) // body
    {
      VectorXd x = x0;
      while (not converge(prob, x))
	x = update(prob, x);

      return x;
    }
    
    virtual VectorXd update(problem& prob, VectorXd& x)=0; // updates approximate solution x
  };

  

  //// base class for solver with line search algorithm
  struct lineSearchSolver: public iterativeSolver {

      // whether use Wolfe's condition or not
    bool use_wolfe;
    // constants in Wolfe's condition
    double c1; // 減少条件
    double c2; // 曲率条件
    // constants in backtracking
    double rho; // 縮小率
    double alpha0; // initial step size


    lineSearchSolver(bool wolfe=false)
      : c1(0.0001), c2(0.9), rho(0.5), alpha0(1.0), use_wolfe(wolfe) {}

    
    virtual VectorXd dir(problem& prob, VectorXd& x)=0; // computes searching direction

    // step size alpha
    virtual double alpha(problem& prob, VectorXd& x, VectorXd& d)
    {
      return use_wolfe ? alpha_wolfe(prob, x, d) : alpha_armijo(prob, x, d);
    }
    

    // Armijo's condition
    bool Armijo(problem& prob, VectorXd& x, double a, VectorXd& d)
    {
      double lhs, rhs;
      lhs = prob.f(x + a*d);
      rhs = prob.f(x) + c1*a*(prob.f.grad(x)).dot(d);
      return lhs <= rhs;
    }

    
    // curvature condition of Wolfe's condition
    bool curvature_condition(problem& prob, VectorXd& x, double a, VectorXd& d)
    {
      double lhs, rhs;
      lhs = (prob.f.grad(x + a*d)).dot(d);
      rhs = (prob.f.grad(x)).dot(d) * c2;
      return lhs >= rhs;
    }


    double alpha_armijo(problem& prob, VectorXd& x, VectorXd& d)
    {
      double a = alpha0;
      while(not Armijo(prob, x, a, d))
	a *= rho;
      return a;
    }

    double alpha_wolfe(problem& prob, VectorXd& x, VectorXd& d)
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

    
    VectorXd update(problem& prob, VectorXd& x) override
    {
      VectorXd d = dir(prob, x);
      double a = alpha(prob, x, d);
      return x + a*d;
    }
  };

  
  // gradient descent solver class
  struct gradientDescent: public lineSearchSolver {

    gradientDescent(bool wolfe=false)
      : lineSearchSolver(wolfe) {}

    // search direction d
    VectorXd dir(problem& prob, VectorXd& x) override
    {
      return - prob.f.grad(x);
    }
  };


  struct NewtonsMethod: public lineSearchSolver {

    bool use_line_search;

    NewtonsMethod(bool line_search=false, bool wolfe=false)
      : lineSearchSolver(wolfe), use_line_search(line_search) {}
    
    // search direction d
    VectorXd dir(problem& prob, VectorXd& x) override
    {
      return (prob.f.hesse(x)).colPivHouseholderQr().solve(-(prob.f.grad(x)));
    }

    // step size alpha
    double alpha(problem& prob, VectorXd& x, VectorXd& d) override
    {
      return (not use_line_search) ? 1.0 : lineSearchSolver::alpha(prob, x, d);
    }
  };
  
}

#endif // __CONTINUOUS_HPP_INCLUDED__
