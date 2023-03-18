import matplotlib.pyplot as plt
import pickle
import matplotlib.ticker as ticker


with open("/home/arch/PycharmProjects/Dimensionality reduction results/Version 0.3/default cost function/CommonNN/Processed datasets/aggregation-commonNN(F1)-default cost.pkl", 'rb') as fid:
    fig = pickle.load(fid)

# Get the first subplot from the saved figure
ax1 = fig[0].axes

# Create a new figure with two identical subplots
fig2, ax = plt.subplots(2, 3)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.90)
# Plot the data from the first subplot in the new figure
ax[0][0].plot(ax1.lines[0].get_xdata(), ax1.lines[0].get_ydata(), "r--", label="first")
ax[0][0].plot(ax1.lines[1].get_xdata(), ax1.lines[1].get_ydata(), "b--", label="second")

ax[0][0].legend(loc="upper right")
ax[0][0].set_title("Homogeneity")
ax[0][0].set_ylabel("Score")

ymax= max(ax1.lines[0].get_ydata())
#ax[0].yaxis.set_major_locator(ticker.MultipleLocator(ymax))

ymax2= max(ax1.lines[1].get_ydata())
# ax[0].yaxis.set_major_locator(ticker.FixedLocator([ymax, ymax2]))
# ax[0].set_yticklabels([ymax, ymax2], color=['red', 'blue'])

# ax[0].yaxis.set_major_locator(ticker.AutoLocator([ymax]))
# ax[0].set_yticklabels([ymax], color='red')
ax[0][0].yaxis.set_major_locator(ticker.FixedLocator([ymax, ymax2]))
#ax[0].set_yticks([ymax, ymax2], color=['blue','red'])
# ax[0].set_yticklabels(ymax2, color='red')

flag = 0
for t in ax[0][0].yaxis.get_ticklabels():
    if flag == 0:
        t.set_color("red")
        flag =1
    else:
        t.set_color("blue")

#plt.setp(ax[0].get_yticklabels(), rotation=90, horizontalalignment='right')

ax[1][0].plot(ax1.lines[0].get_xdata(), ax1.lines[0].get_ydata(), "r--", label="first")
ax[1][0].plot(ax1.lines[1].get_xdata(), ax1.lines[1].get_ydata(), "b--", label="second")

ax[1][0].legend(loc="upper right")
ax[1][0].set_title("AMI")
ax[1][0].set_xlabel("epsilon distances")

ax[1][0].plot(ax1.lines[0].get_xdata(), ax1.lines[0].get_ydata(), "r--", label="first")
ax[1][0].plot(ax1.lines[1].get_xdata(), ax1.lines[1].get_ydata(), "b--", label="second")

ax[1][0].legend(loc="upper right")
ax[1][0].set_title("test2")

plt.show()

