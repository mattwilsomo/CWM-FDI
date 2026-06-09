# !/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import re

# parameters to modify 
filename="ping_slow.log"
label='Ping times'
xlabel = 'Number of measurements'
ylabel = 'CUmulative time'
title='CDF of ping times'
fig_name='assignment2_question2.png'
bins=100 #adjust the number of bins to your plot

## load data from input file
if re.search(r"^ping.*\.log$", filename): 
	times = []
	with open(filename, "r") as f:
		for line in f:
			m = re.search(r"time=([0-9.]+)", line)
			if m:
				times.append(float(m.group(1)))
	t = np.array(times)

else: 	
	t = np.loadtxt(filename, delimiter=" ", dtype="float")

## if your data is "X Y" (2 cols), use the following line
#plt.plot(t[:,0], t[:,1], label=label)  # Plot some data on the (implicit) axes.

## if your data is "X" (1 col), use the following line
#plt.plot(t, label=label)  # Plot some data on the (implicit) axes.

## comment the lines above and uncomment the line below to plot a simple CDF
#plt.hist(t[:], bins, density=True, histtype='step', cumulative=True, label=label)

## comment the lines above and uncomment the 4 lines below for a nicer CDF
n = np.arange(1,len(t)+1) / float(len(t))
ts = np.sort(t)
fig, ax = plt.subplots()
ax.step(ts,n)

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend()
plt.savefig(fig_name)
plt.show()
