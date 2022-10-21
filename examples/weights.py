import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


def plot_weights(support, weights_func, xlabels, xticks):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    print(weights_func(support))
    ax.plot(support, weights_func(support))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=16)
    ax.set_ylim(-0.1, 1.1)
    return ax


norms = sm.robust.norms

c = 2
support = np.linspace(-3 * c, 3 * c, 1000)
trimmed = norms.TrimmedMean(c=c)
weight_fnc = trimmed.weights
print(weight_fnc(support))

# c = 2
# support = np.linspace(-3 * c, 3 * c, 1000)
# trimmed = norms.TrimmedMean(c=c)
# plot_weights(support, trimmed.weights, ["-3*c", "0", "3*c"], [-3 * c, 0, 3 * c])
# plt.show()

c = 4.685
support = np.linspace(-3 * c, 3 * c, 1000)
tukey = norms.TukeyBiweight(c=c)
plot_weights(support, tukey.weights, ["-3*c", "0", "3*c"], [-3 * c, 0, 3 * c])


# a = 1.339
# support = np.linspace(-np.pi * a, np.pi * a, 100)
# andrew = norms.AndrewWave(a=a)
# plot_weights(
#     support, andrew.weights, ["$-\pi*a$", "0", "$\pi*a$"], [-np.pi * a, 0, np.pi * a]
# )
# plt.show()
