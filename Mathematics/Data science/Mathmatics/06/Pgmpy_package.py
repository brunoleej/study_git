import pgmpy
import numpy as np
from pgmpy.factors.discrete import JointProbabilityDistribution as JPD

px = JPD(['X'], [2], np.array([12, 8]) / 20)

print(px)