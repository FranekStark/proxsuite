//
// Copyright (c) 2026 INRIA
//
// Regression tests for the primal infeasibility check. A certificate (dy, dz)
// satisfies A^T dy + C^T dz = 0 and b^T dy + u^T [dz]_+ - l^T [-dz]_+ < 0. The
// lower bound term used to be added with the wrong sign: certificates that need
// a lower bound were never accepted, and feasible problems with very negative
// lower bounds could be declared infeasible.
#include <limits>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <Eigen/Core>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/sparse/sparse.hpp>

using T = double;
using namespace proxsuite;

namespace {

using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

// min 1/2 h x^2 + g x  s.t.  l_i <= c_i x <= u_i  (x scalar)
struct Problem
{
  T h;
  T g;
  std::vector<T> c;
  std::vector<T> l;
  std::vector<T> u;
};

Mat
constraint_matrix(const Problem& p)
{
  Mat C(p.c.size(), 1);
  for (size_t i = 0; i < p.c.size(); ++i) {
    C(i, 0) = p.c[i];
  }
  return C;
}

Vec
to_vec(const std::vector<T>& v)
{
  return Eigen::Map<const Vec>(v.data(), v.size());
}

template<typename QP>
void
set_settings(QP& qp)
{
  qp.settings.eps_abs = 1e-8;
  qp.settings.eps_rel = 0;
  // enough iterations for a problem of size 1; a missed certificate ends with
  // PROXQP_MAX_ITER_REACHED
  qp.settings.max_iter = 500;
}

proxqp::QPSolverOutput
solve_dense(const Problem& p)
{
  proxqp::isize n_in = p.c.size();
  proxqp::dense::QP<T> qp(1, 0, n_in);
  set_settings(qp);
  qp.init(Mat::Constant(1, 1, p.h),
          Vec::Constant(1, p.g),
          nullopt,
          nullopt,
          constraint_matrix(p),
          to_vec(p.l),
          to_vec(p.u));
  qp.solve();
  return qp.results.info.status;
}

proxqp::QPSolverOutput
solve_dense_box(const Problem& p, T l_box, T u_box)
{
  proxqp::isize n_in = p.c.size();
  proxqp::dense::QP<T> qp(1, 0, n_in, true);
  set_settings(qp);
  qp.init(Mat::Constant(1, 1, p.h),
          Vec::Constant(1, p.g),
          nullopt,
          nullopt,
          constraint_matrix(p),
          to_vec(p.l),
          to_vec(p.u),
          Vec::Constant(1, l_box),
          Vec::Constant(1, u_box));
  qp.solve();
  return qp.results.info.status;
}

proxqp::QPSolverOutput
solve_sparse(const Problem& p)
{
  proxqp::isize n_in = p.c.size();
  proxqp::sparse::QP<T, int> qp(1, 0, n_in);
  set_settings(qp);
  Eigen::SparseMatrix<T, Eigen::ColMajor, int> H =
    Mat::Constant(1, 1, p.h).sparseView();
  Eigen::SparseMatrix<T, Eigen::ColMajor, int> C =
    constraint_matrix(p).sparseView();
  qp.init(
    H, Vec::Constant(1, p.g), nullopt, nullopt, C, to_vec(p.l), to_vec(p.u));
  qp.solve();
  return qp.results.info.status;
}

// infinite bounds given as infinity and as a large finite value
const T infinities[] = { std::numeric_limits<T>::infinity(), T(1e20) };

} // namespace

TEST_CASE("primal infeasibility: certificate with a lower bound")
{
  for (T inf : infinities) {
    // x <= 0 and x >= 1: the certificate combines an upper and a lower bound
    Problem p{ 1., 0., { 1., 1. }, { -inf, 1. }, { 0., inf } };
    CHECK(solve_dense(p) == proxqp::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE);
    CHECK(solve_sparse(p) == proxqp::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE);
  }
}

TEST_CASE("primal infeasibility: feasible problem is not reported")
{
  for (T inf : infinities) {
    // feasible (0.3 x = 0.3 gives x = 1), nearly flat cost, one-sided rows with
    // an infinite lower bound
    Problem p{ 1e-8,
               0.5,
               { -0.01, 0.3, -0.6, 1., -1. },
               { -2.5, 0.3, -1., -inf, -inf },
               { 1.7, 0.3, -0.5, 6., 6. } };
    CHECK(solve_dense(p) == proxqp::QPSolverOutput::PROXQP_SOLVED);
    CHECK(solve_sparse(p) == proxqp::QPSolverOutput::PROXQP_SOLVED);
  }
}

TEST_CASE("primal infeasibility: dense box constraints")
{
  for (T inf : infinities) {
    // x <= 0 as a row, 1 <= x <= 2 as a box: the certificate needs the box
    // lower bound
    Problem row_upper{ 1., 0., { 1. }, { -inf }, { 0. } };
    CHECK(solve_dense_box(row_upper, 1., 2.) ==
          proxqp::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE);

    // the feasible problem from above with x <= 6 as box, its lower side
    // infinite
    Problem feasible{
      1e-8, 0.5, { -0.01, 0.3, -0.6 }, { -2.5, 0.3, -1. }, { 1.7, 0.3, -0.5 }
    };
    CHECK(solve_dense_box(feasible, -inf, 6.) ==
          proxqp::QPSolverOutput::PROXQP_SOLVED);
  }
}
