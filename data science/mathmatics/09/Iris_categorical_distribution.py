import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def plot_dirichlet(alpha, n):
    def project(x):
        n1 = np.array([1, 0, 0])
        n2 = np.array([0, 1, 0])
        n3 = np.array([0, 0, 1])
        n12 = (n1 + n2)/2
        m1 = np.array([1, -1, 0])
        m2 = n3 - n12
        m1 = m1/np.linalg.norm(m1)
        m2 = m2/np.linalg.norm(m2)
        return np.dstack([(x-n12).dot(m1), (x-n12).dot(m2)])[0]

def project_reverse(x):
    n1 = np.array([1, 0, 0])
    n2 = np.array([0, 1, 0])
    n3 = np.array([0, 0, 1])
    n12 = (n1 + n2)/2
    m1 = np.array([1, -1, 0])
    m2 = n3 - n12
    m1 = m1/np.linalg.norm(m1)
    m2 = m2/np.linalg.norm(m2)
    return x[:, 0][:, np.newaxis] * m1 + x[:, 1][:, np.newaxis] * m2 + n12
 
eps = np.finfo(float).eps * 10
X = project([[1-eps, 0, 0], [0, 1-eps, 0], [0, 0, 1-eps]])
 
import matplotlib.tri as mtri
triang = mtri.Triangulation(X[:, 0], X[:, 1], [[0, 1, 2]])
refiner = mtri.UniformTriRefiner(triang)
triang2 = refiner.refine_triangulation(subdiv=6)
XYZ = project_reverse(
np.dstack([triang2.x, triang2.y, 1-triang2.x-triang2.y])[0])
pdf = sp.stats.dirichlet(alpha).pdf(XYZ.T)
plt.tricontourf(triang2, pdf, cmap=plt.cm.bone_r)
plt.axis("equal")
plt.title("정규분포 확률변수의 모수를 베이즈 추정법으로 추정한 결과: {} 추정".format(n))
plt.show()

mu0 = np.array([0.3, 0.5, 0.2])
np.random.seed(0)

a0 = np.ones(3)
plot_dirichlet(a0, "초기")