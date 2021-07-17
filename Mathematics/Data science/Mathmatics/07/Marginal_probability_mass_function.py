import pandas as pd

grades = ["A", "B", "C", "D", "E", "F"]
scores = pd.DataFrame(
    [[1, 2, 1, 0, 0, 0],
    [0, 2, 3, 1, 0, 0],
    [0, 4, 7, 4, 1, 0],
    [0, 1, 4, 5, 4, 0],
    [0, 0, 1, 3, 2, 0],
    [0, 0, 0, 1, 2, 1]], 
    columns=grades, index=grades)


pmf = scores / scores.values.sum()
pmf_marginal_x = pmf.sum(axis=0)
print(pmf_marginal_x)
