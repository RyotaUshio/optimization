#ifndef __CONTINUOUS_HPP_INCLUDED__
#define __CONTINUOUS_HPP_INCLUDED__

#include <iostream>
#include <functional>
#include <vector>
#include <cmath>
#include <cstdio>
#include <Eigen/Dense>

using funcType = std::function<double(Eigen::VectorXd)>;
using gradType = std::function<Eigen::VectorXd(Eigen::VectorXd)>;
using hesseType = std::function<Eigen::MatrixXd(Eigen::VectorXd)>;


namespace Continuous {
  //// Continuous Optimization Problem ////
  
  struct objFunc {
    //// objective function class
    const funcType func; // objective function itself
    const gradType grad; // gradient
    const hesseType hesse; // Hessian matrix


    objFunc(funcType f);
    objFunc(funcType f, gradType g);
    objFunc(funcType f, gradType g, hesseType h);

    double operator()(Eigen::VectorXd x) const;
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

    problem(objFunc& func);
    problem(objFunc& func, constraint& cons);
  };



  //// optimization problem solver with iterative method
  struct iterativeSolver {
    
    double eps; // tolerant norm of gradient vector

    iterativeSolver();


    bool converge(problem& prob, Eigen::VectorXd& x); // convergence test
    Eigen::VectorXd operator()(problem& prob, Eigen::VectorXd& x0); // body    
    virtual Eigen::VectorXd update(problem& prob, Eigen::VectorXd& x)=0; // updates approximate solution x
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


    lineSearchSolver(bool wolfe=false);
    virtual Eigen::VectorXd dir(problem& prob, Eigen::VectorXd& x)=0; // computes searching direction
    // step size alpha
    virtual double alpha(problem& prob, Eigen::VectorXd& x, Eigen::VectorXd& d);
    // Armijo's condition
    bool Armijo(problem& prob, Eigen::VectorXd& x, double a, Eigen::VectorXd& d);
    // curvature condition of Wolfe's condition
    bool curvature_condition(problem& prob, Eigen::VectorXd& x, double a, Eigen::VectorXd& d);
    // backtracking
    double alpha_armijo(problem& prob, Eigen::VectorXd& x, Eigen::VectorXd& d);
    // return alpha that satisfying Wolfe's condition
    double alpha_wolfe(problem& prob, Eigen::VectorXd& x, Eigen::VectorXd& d);
    Eigen::VectorXd update(problem& prob, Eigen::VectorXd& x) override;
  };

  
  //// Gradient Descent solver class
  struct gradientDescent: public lineSearchSolver {

    gradientDescent(bool wolfe=false);
    // search direction d: steepest descent direction
    Eigen::VectorXd dir(problem& prob, Eigen::VectorXd& x) override;
  };


  //// Newton's Method solver class
  struct NewtonsMethod: public lineSearchSolver {
    bool use_line_search;
    
    NewtonsMethod(bool line_search=false, bool wolfe=false);
    // search direction d: the Newton direction
    Eigen::VectorXd dir(problem& prob, Eigen::VectorXd& x) override;
    // step size alpha: fixed to 1.0 or obtained with Armijo/Wolfe's method
    double alpha(problem& prob, Eigen::VectorXd& x, Eigen::VectorXd& d) override;
  };
  
}

#endif // __CONTINUOUS_HPP_INCLUDED__
