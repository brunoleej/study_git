from cvxopt import matrix, solvers

Q = matrix(np.diag([2.0, 2.0]))
c = matrix(np.array([0.0, 0.0]))
A = matrix(np.array([[1.0, 1.0]]))
b = matrix(np.array([[1.0]]))
sol = solvers.qp(Q, c, A=A, b=b)

print(np.array(sol['x']))
