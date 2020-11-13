#ifndef __CONTINUOUS_HPP_INCLUDED__
#define __CONTINUOUS_HPP_INCLUDED__


#include <iostream>
#include <functional>
#include <vector>
#include <cmath>
#include <cstdio>
#include <Eigen/Core>

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
    objFunc f;
    constraint* c;

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


    bool converge(problem& prob, VectorXd x) // convergence test
    {
      std::printf("in converge(): %p\n", &x);
      double grad_norm = (prob.f.grad(x)).norm();
      return (grad_norm < eps);
    }

    
    VectorXd operator()(problem& prob, VectorXd x0) // body
    {
      VectorXd x = x0;
      std::printf("in oprator(): %p\n", &x);
      while (not converge(prob, x))      
	x = update(prob, x);	
      return x;
    }

    virtual VectorXd update(problem& prob, VectorXd x)=0; // updates approximate solution x
  };

  

  //// base class for solver with line search algorithm
  struct lineSearchSolver: public iterativeSolver {
    virtual VectorXd dir(problem& prob, VectorXd x)=0; // computes searching direction
    virtual double alpha(problem& prob, VectorXd x, VectorXd d)=0; // computes step size
    
    VectorXd update(problem& prob, VectorXd x) override
    {
      std::printf("in update(): %p\n", &x);
      VectorXd d = dir(prob, x);
      double a = alpha(prob, x, d);
      return x + a*d;
    }
  };

  
  // gradient descent solver class
  struct gradientDescent: public lineSearchSolver {
    // constants in Wolfe condition
    double c1; // 減少条件
    double c2; // 曲率条件
    // constants in backtracking
    double rho; // 縮小率
    double alpha0; // initial step size

    
    gradientDescent()
      : c1(0.0001), c2(0.9), rho(0.5), alpha0(1.0) {}


    // armijo's condition
    bool Armijo(problem& prob, VectorXd x, double a, VectorXd d)
    {
      double lhs, rhs;
      lhs = ( prob.f.grad(x + a*d) ).dot( d );
      rhs = ( prob.f.grad(x) ).dot( d ) * c1;
      return lhs < rhs;
    }


    // search direction d
    VectorXd dir(problem& prob, VectorXd x)
    {
      VectorXd grad = prob.f.grad(x);
      return -grad;
    }

    // step size alpha
    double alpha(problem& prob, VectorXd x, VectorXd d)
    {
      double a = alpha0;
      while(not Armijo(prob, x, a, d))
	a *= rho;
      return a;
    }
  };

  
}

#endif // __CONTINUOUS_HPP_INCLUDED__
