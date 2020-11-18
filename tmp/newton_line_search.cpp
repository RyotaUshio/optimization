#include "continuous.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

using namespace Eigen;
using namespace Continuous;

int main()
{
  // ニュートン法・演習問題2
  // ヘッセ行列がつねに正定値であるにもかかわらず発散してしまう例。
  objFunc f(
	    [](VectorXd x) -> double {
	      double x_ = 1 + std::abs(x(0));
	      return x_ * std::log(x_) - x_;
	    },
	    [](VectorXd x) -> VectorXd {
	      VectorXd grad(1);
	      grad << (double)((int)(x(0)>0) - (int)(x(0)<0)) * std::log(1 + std::abs(x(0)));
	      return grad;
		},
	    [](VectorXd x) -> MatrixXd {
	      MatrixXd hesse(1, 1);
	      hesse << 1.0 / (1 + std::abs(x(0)));
	      return hesse;
	    });
  
  problem prob(f);
  VectorXd x0(1);
  x0 << exp(2) - 0.9;
  NewtonsMethod solver("newton_line_search_armijo_log.out");
  solver.use_line_search = true; // 直線探索を組み合わせることで収束させることができる
  VectorXd x_star = solver(prob, x0);
  std::cout << x_star.transpose() << std::endl;
}
