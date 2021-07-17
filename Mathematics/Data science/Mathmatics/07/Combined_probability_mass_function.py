import seaborn as sns
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
scores.index.name = "Y"
scores.columns.name = "X"

print(scores)

pmf = scores / scores.values.sum()
print(pmf)

sns.heatmap(pmf, cmap=mpl.cm.bone_r, annot=True,
            xticklabels=['A', 'B', 'C', 'D', 'E', 'F'],
            yticklabels=['A', 'B', 'C', 'D', 'E', 'F'])
plt.title("결합확률질량함수 p(x,y)")
plt.tight_layout()
plt.show()