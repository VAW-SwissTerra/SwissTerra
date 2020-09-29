import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

filepath = "/remotes/erik-ryzen/home/erik/Documents/Projects/SwissTopo/Rhonegletscher/temp/principal_point_sesitivity.csv"

data = pd.read_csv(filepath).dropna(how="any")
data = data[data["error"] < 1000]
data["offset"] = np.linalg.norm(data[["offset_x", "offset_y"]], axis=1)

best_solutions = data.groupby(pd.cut(data["offset"], np.arange(0, 100, 1))).min().dropna()
best_solutions.index = [index.mid for index in best_solutions.index]

model = np.poly1d(np.polyfit(best_solutions["offset"], best_solutions["error"], deg=2))
xs = np.linspace(data["offset"].min(), data["offset"].max(), num=50)

plt.figure(figsize=(8, 4), dpi=150)
plt.plot(xs, model(xs))
plt.scatter(data["offset"], data["error"], s=1)
plt.ylabel("Tie point reprojection error (px)")
plt.xlabel("Principal point offset (px)")
plt.savefig("figures/principal_point_sensitivity.jpg", dpi=300)
plt.show()
