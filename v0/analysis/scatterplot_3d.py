import pandas as pd
import matplotlib.pyplot as plt

DIRECTORY = "formatted"
N_SAMPLES = 400
X_COLUMN = "8-302-0"
Y_COLUMN = "8-303-0"
Z_COLUMN = "8-303-2"

# データの読み込み
df_8302 = pd.read_csv(f"{DIRECTORY}/formatted_8-302_t=40.csv")
df_8303 = pd.read_csv(f"{DIRECTORY}/formatted_8-303_t=40.csv")

# データをランダムにサンプリング
df_8302 = df_8302.sample(n=N_SAMPLES)
df_8303 = df_8303.sample(n=N_SAMPLES)

# データの結合
df = pd.concat([df_8302, df_8303])


# 3次元散布図の描画
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
label_colors = {'8-302': 'red', '8-303': 'blue'}
for label, color in label_colors.items():
    subset = df[df['label'] == label]
    ax.scatter(subset[X_COLUMN], subset[Y_COLUMN], subset[Z_COLUMN], c=color, label=label)
ax.set_xlabel(X_COLUMN)
ax.set_ylabel(Y_COLUMN)
ax.set_zlabel(Z_COLUMN)

# Show the plot
plt.legend()
plt.show()

