#ifndef __CONTINUOUS_HPP_INCLUDED__
#define __CONTINUOUS_HPP_INCLUDED__


#include <functional>
#include <vector>
#include <string>
#include <cmath>
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


    objFunc(funcType f)
      : func(f) {}
    objFunc(funcType f, gradType g)
      : func(f), grad(g) {}
    objFunc(funcType f, gradType g, hesseType h)
      : func(f), grad(g), hesse(h) {}

    double operator()(Eigen::VectorXd x) const
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


    bool converge(problem& prob, Eigen::VectorXd& x) // convergence test
    {
      double grad_norm = (prob.f.grad(x)).norm();
      return (grad_norm < eps);
    }

    
    Eigen::VectorXd operator()(problem& prob, Eigen::VectorXd& x0) // body
    {
      Eigen::VectorXd x = x0;
      while (not converge(prob, x))
	x = update(prob, x);
      return x;
    }
    
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


    lineSearchSolver(bool wolfe=false)
      : c1(0.0001), c2(0.9), rho(0.5), alpha0(1.0), use_wolfe(wolfe) {}

    
    virtual Eigen::VectorXd dir(problem& prob, Eigen::VectorXd& x)=0; // computes searching direction

    // step size alpha
    virtual double alpha(problem& prob, Eigen::VectorXd& x, Eigen::VectorXd& d)
    {
      return use_wolfe ? alpha_wolfe(prob, x, d) : alpha_armijo(prob, x, d);
    }
    

    // Armijo's condition
    bool Armijo(problem& prob, Eigen::VectorXd& x, double a, Eigen::VectorXd& d)
    {
      double lhs, rhs;
      lhs = prob.f(x + a*d);
      rhs = prob.f(x) + c1*a*(prob.f.grad(x)).dot(d);
      return lhs <= rhs;
    }

    
    // curvature condition of Wolfe's condition
    bool curvature_condition(problem& prob, Eigen::VectorXd& x, double a, Eigen::VectorXd& d)
    {
      double lhs, rhs;
      lhs = (prob.f.grad(x + a*d)).dot(d);
      rhs = (prob.f.grad(x)).dot(d) * c2;
      return lhs >= rhs;
    }


    // backtracking
    double alpha_armijo(problem& prob, Eigen::VectorXd& x, Eigen::VectorXd& d)
    {
      double a = alpha0;
      while(not Armijo(prob, x, a, d))
	a *= rho;
      return a;
    }

    // return alpha that satisfying Wolfe's condition
    double alpha_wolfe(problem& prob, Eigen::VectorXd& x, Eigen::VectorXd& d)
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

    
    Eigen::VectorXd update(problem& prob, Eigen::VectorXd& x) override
    {
      Eigen::VectorXd d = dir(prob, x);
      double a = alpha(prob, x, d);
      return x + a*d;
    }
  };

  
  //// Gradient Descent solver class
  struct gradientDescent: public lineSearchSolver {

    gradientDescent(bool wolfe=false)
      : lineSearchSolver(wolfe) {}

    // search direction d: steepest descent direction
    Eigen::VectorXd dir(problem& prob, Eigen::VectorXd& x) override
    {
      return - prob.f.grad(x);
    }
  };


  //// Newton's Method solver class
  struct NewtonsMethod: public lineSearchSolver {

    bool use_line_search;

    NewtonsMethod(bool line_search=false, bool wolfe=false)
      : lineSearchSolver(wolfe), use_line_search(line_search) {}
    
    // search direction d: the Newton direction
    Eigen::VectorXd dir(problem& prob, Eigen::VectorXd& x) override
    {
      return (prob.f.hesse(x)).colPivHouseholderQr().solve(-(prob.f.grad(x)));
    }

    // step size alpha
    double alpha(problem& prob, Eigen::VectorXd& x, Eigen::VectorXd& d) override
    {
      return (not use_line_search) ? 1.0 : lineSearchSolver::alpha(prob, x, d);
    }
  };


  //// Quasi-Newton Method solver class
  class quasiNewtonMethod: public lineSearchSolver {
  private:
    std::string hesse_method; // Hessian approximation: BFGS or DFP
    Eigen::MatrixXd H;
    
  public:
    quasiNewtonMethod(Eigen::MatrixXd H0, std::string method="BFGS", bool wolfe=true)
      : lineSearchSolver(wolfe)
    {
      if (H0.rows() != H0.cols())
	{
	  std::cerr << "Continuous::quasiNewtonMethod::quasiNewtonMethod(): "
		    << "H0 must be a square matrix"
		    << std::endl;
	  std::exit(1);
	}
      H = H0;
      set_hesse_method(method);
    }

    void set_hesse_method(std::string method)
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
    Eigen::VectorXd dir(problem& prob, Eigen::VectorXd& x) override
    {
      return -H*prob.f.grad(x);
    }

    // update H with the BFGS formula
    void BFGS (problem& prob, Eigen::VectorXd& x_old, Eigen::VectorXd& x_new)
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
    void DFP(problem& prob, Eigen::VectorXd& x_old, Eigen::VectorXd& x_new)
    {
      Eigen::VectorXd
	s = x_new - x_old,
	y = prob.f.grad(x_new) - prob.f.grad(x_old);
      H = H - H*y*(y.transpose())*H/(y.dot(H*y)) + s*(s.transpose())/(y.dot(s));
    }

    Eigen::VectorXd update(problem& prob, Eigen::VectorXd& x) override
    {
      // update x
      Eigen::VectorXd x_new = lineSearchSolver::update(prob, x);
      // update H
      (hesse_method == "BFGS") ? BFGS(prob, x, x_new) : DFP(prob, x, x_new);
      return x_new;
    }
  };    
}


#endif // __CONTINUOUS_HPP_INCLUDED__
