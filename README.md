# CS-based-Sparse-Signal-Reconstruction
This repository contains code pertaining to the project of CS-based sparse signal reconstructionin the course Applied Convex Optimization
In this project the problem of sparse signal reconstruction is formulated into least l1-norm problem and Lasso regularization problem.
1. CVX_main calls the CVX solver to addresses the two proposed problem formulations.
2. ProjectedGradient_main uses the projected (sub)gradient method to solve least l1-norm problem.
3. SparseRepresentation_main addresses the Lasso regularization problem using quadratic programming method.
4. Barrier_main solves the Lasso regularization problem using barrier method. it calls barrier.m to do outer iterations and objval.m to compute current value of objective function. barrier.m calls backtracing_newton.m to do inner iterations and dualobjval.m to compute current value of dual function. backtracing_newton.m calls g_H_comp.m to compute the gradient and Hessian.
