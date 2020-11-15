%module continuous
%{
  #include "continuous.hpp"
  #include <Eigen/Dense>
  using namespace Eigen;
%}
%include "continuous.hpp"
%include <Eigen/Dense>
using namespace Eigen;
