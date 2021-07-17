from pgmpy.factors.discrete import TabularCPD
cpd_X = TabularCPD('X', 2, [[1 - 0.002, 0.002]])
print(cpd_X)

cpd_Y_on_X = TabularCPD('Y', 2, np.array([[0.95, 0.01], [0.05, 0.99]]),evidence=['X'], evidence_card=[2])
print(cpd_Y_on_X)