#ifndef __CONTINUOUS_HPP_INCLUDED__
#define __CONTINUOUS_HPP_INCLUDED__

#include <iostream>
#include <functional>
#include <vector>
#include <string>
#include <fstream>
#include <limits>
#include <cmath>
#include <cstdio>
#include <Eigen/Dense>

using funcType = std::function<double(Eigen::VectorXd)>;
using gradType = std::function<Eigen::VectorXd(Eigen::VectorXd)>;
using hesseType = std::function<Eigen::MatrixXd(Eigen::VectorXd)>;


namespace Continuous { //// Continuous Optimization Problem ////


  void parseArgs(Eigen::VectorXd& x0, int argc, char* argv[]); // set the inital guess x0 from command line

  
  struct objFunc //// objective function class
  { 
    const funcType func; // objective function itself
    const gradType grad; // gradient
    const hesseType hesse; // Hessian matrix

    objFunc();
    objFunc(funcType f);
    objFunc(funcType f, gradType g);
    objFunc(funcType f, gradType g, hesseType h);

    double operator()(Eigen::VectorXd x) const;
  };

  
  struct constraint {}; //// base class for constraints

  
  struct eqConstraint: public constraint //// equality constraints
  {
    std::vector<objFunc> func;
    Eigen::MatrixXd Jacobian(Eigen::VectorXd x) const; // Jacobian matrix of g(x)

    eqConstraint(); // default constructor
    eqConstraint(std::vector<objFunc> funcs);

    Eigen::VectorXd operator()(Eigen::VectorXd x) const;
  };


  struct ineqConstraint: public eqConstraint //// inequality constraints
  {
    std::vector<objFunc> ineqcons;
  };


  objFunc makeLagrangian(objFunc& func, eqConstraint& eqcons);


  struct problem //// Optimization problem class
  {
    const objFunc f; // objective function
    // const eqConstraint g; // equality constraint
    // const objFunc L; // Lagrangian

    problem(objFunc& func);
    problem(objFunc& func, eqConstraint& eqcons);
  };


  struct iterativeSolver
  //// optimization problem solver with iterative method
  {
    static double eps; // tolerant norm of gradient vector
    static int k_max; // maximum number of iteration
    int k; // iteration counter

    iterativeSolver();

    bool converge(problem& prob, Eigen::VectorXd& x) const; // convergence test
    Eigen::VectorXd operator()(problem& prob, Eigen::VectorXd& x0); // body    
    virtual Eigen::VectorXd update(problem& prob, Eigen::VectorXd& x)=0; // updates approximate solution x
  };


  struct lineSearchSolver: public iterativeSolver
  //// base class for solver with line search algorithm
  {
    // whether use Wolfe's condition or not
    bool use_wolfe;
    // constants in Wolfe's condition
    double c1; // 減少条件
    double c2; // 曲率条件
    // constants in backtracking
    double rho; // 縮小率
    double alpha0; // initial step size
    // output a log file
    bool log;
    std::string logname;
    std::ofstream logout;
    static std::string delimiter;
    

    lineSearchSolver(bool wolfe=false);
    lineSearchSolver(const char* filename, bool wolfe=false);
    virtual Eigen::VectorXd dir(problem& prob, Eigen::VectorXd& x)=0; // computes searching direction
    // step size alpha
    virtual double alpha(problem& prob, Eigen::VectorXd& x, Eigen::VectorXd& d);
    // Armijo's condition
    bool Armijo(problem& prob, Eigen::VectorXd& x, double a, Eigen::VectorXd& d) const;
    // curvature condition of Wolfe's condition
    bool curvature_condition(problem& prob, Eigen::VectorXd& x, double a, Eigen::VectorXd& d) const;
    // backtracking
    double alpha_armijo(problem& prob, Eigen::VectorXd& x, Eigen::VectorXd& d);
    // return alpha that satisfying Wolfe's condition
    double alpha_wolfe(problem& prob, Eigen::VectorXd& x, Eigen::VectorXd& d);
    Eigen::VectorXd update(problem& prob, Eigen::VectorXd& x) override;
  };

  
  struct gradientDescent: public lineSearchSolver
  //// Gradient Descent solver class
  {
    gradientDescent(bool wolfe=false);
    gradientDescent(const char* filename, bool wolfe=false);
    // search direction d: steepest descent direction
    Eigen::VectorXd dir(problem& prob, Eigen::VectorXd& x) override;
  };


  struct NewtonsMethod: public lineSearchSolver
  //// Newton's Method solver class
  {
    bool use_line_search;
    
    NewtonsMethod(bool line_search=false, bool wolfe=false);
    NewtonsMethod(const char* filename, bool line_search=false, bool wolfe=false);
    
    // search direction d: the Newton direction
    Eigen::VectorXd dir(problem& prob, Eigen::VectorXd& x) override;
    // step size alpha: fixed to 1.0 or Armijo/Wolfe's method
    double alpha(problem& prob, Eigen::VectorXd& x, Eigen::VectorXd& d) override;
  };


  //// Quasi-Newton Method solver class
  class quasiNewtonMethod: public lineSearchSolver {
  private:
    std::string hesse_method; // Hessian approximation: BFGS or DFP
    Eigen::MatrixXd H;
    
  public:
    quasiNewtonMethod(Eigen::MatrixXd H0, std::string method="BFGS", bool wolfe=true);
    quasiNewtonMethod(const char* filename, Eigen::MatrixXd H0, std::string method="BFGS", bool wolfe=true);

    void set_H(Eigen::MatrixXd H0);
    void set_hesse_method(std::string method);
    Eigen::VectorXd dir(problem& prob, Eigen::VectorXd& x) override; // search direction d
    void BFGS (problem& prob, Eigen::VectorXd& x_old, Eigen::VectorXd& x_new); // update H with the BFGS formula
    void DFP(problem& prob, Eigen::VectorXd& x_old, Eigen::VectorXd& x_new); // update H with the DFP formula
    Eigen::VectorXd update(problem& prob, Eigen::VectorXd& x) override;
  };
}


#endif // __CONTINUOUS_HPP_INCLUDED__
