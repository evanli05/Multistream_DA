import numpy as np
import matplotlib.pyplot as plt

val1 = np.genfromtxt('sen1.csv', delimiter=',')
val2 = np.genfromtxt('sen2.csv', delimiter=',')

x1 = val1[0, :]
x2 = val2[0, :]

y11 = val1[1, :]
y12 = val1[2, :]
y21 = val2[1, :]
y22 = val2[2, :]

fig, ax1 = plt.subplots()
ax1.plot(x1, y11, 'bs-')
ax1.set_xlabel(r'$\beta$')
ax1.set_ylim(10, 60)
ax1.set_ylabel('Error rate', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
# s2 = np.sin(2 * np.pi * t)
ax2.plot(x1, y12, 'rv-')
ax2.set_ylim(200, 300)
ax2.set_ylabel('Time', color='r')
ax2.tick_params('y', colors='r')

plt.show()

# plt.subplot(2, 1, 2)
# plt.plot(x2, y22, 'r^-')

# plt.show()


# fig, ax1 = plt.subplots()
# t = np.arange(0.01, 10.0, 0.01)
# s1 = np.exp(t)
# ax1.plot(t, s1, 'b-')
# ax1.set_xlabel('time (s)')
# # Make the y-axis label, ticks and tick labels match the line color.
# ax1.set_ylabel('exp', color='b')
# ax1.tick_params('y', colors='b')

# ax2 = ax1.twinx()
# s2 = np.sin(2 * np.pi * t)
# ax2.plot(t, s2, 'r.')
# ax2.set_ylabel('sin', color='r')
# ax2.tick_params('y', colors='r')