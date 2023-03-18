import matplotlib.pyplot as plt
import pickle
import matplotlib.ticker as ticker

with open(
        "/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/DBSCAN/aggregation/Highest values/aggregation-DBSCAN(f1)-default cost-Min_pts 20.pkl",
        'rb') as fid:
    fig = pickle.load(fid)

# Get the first subplot from the saved figure
ax1 = fig.axes

# Create a new figure with two identical subplots
fig2, ax = plt.subplots(1, 3)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.90)

# Plot the data from the first subplot in the new figure
ax[0].plot(ax1.lines[0].get_xdata(), ax1.lines[0].get_ydata(), "r--", label="first")
ax[0].plot(ax1.lines[1].get_xdata(), ax1.lines[1].get_ydata(), "b--", label="second")

ax[0].legend(loc="upper right")
ax[0].set_title("Homogeneity")
ax[0].set_ylabel("Score")

ymax = max(ax1.lines[0].get_ydata())
ymax2 = max(ax1.lines[1].get_ydata())

# Set y-axis tick locators
ax[0].yaxis.set_major_locator(ticker.FixedLocator([ymax, ymax2]))

# Set y-axis tick labels with different colors
tick_values = ax[0].yaxis.get_major_locator()._get_tick_locations()
red_index = [i for i in range(len(tick_values)) if tick_values[i] == ymax][0]
for tick, label in zip(ax[0].yaxis.get_major_ticks(), ax[0].yaxis.get_majorticklabels()):
    if tick.index == red_index:
        label.set_color('red')
    else:
        label.set_color('blue')

plt.show()
