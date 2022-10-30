import numpy as np
import matplotlib.pyplot as plt


def calc_prob_encoding_x(x):
    if x>0.:
        prob = np.exp(-1./((x)/0.1))/1. 
        return prob
    elif x<0.:
        prob = np.exp(1./((1*x)/0.1))/1. 
        return  prob

def calc_time_to_point(x):
    return np.sqrt(x)*4

import matplotlib.pyplot as plt

y_plus = []
y_neg = []
x = np.arange(-1.0, 1.01, 0.01)
# y = np.exp(1./((x)/2))/2. + 0.5
for i in range(len(x)):
#     # calc_prob_encoding_x(x[i])
    y_plus.append(calc_prob_encoding_x(x[i]))
    # y_neg.append(calc_prob_encoding_x(x[i])[1])

plt.plot(x[:int(len(y_plus)/2)], y_plus[:int(len(y_plus)/2)], label='Negative node')
plt.plot(x[int(len(y_plus)/2):], y_plus[int(len(y_plus)/2):], label='Positive node')
# plt.plot(x, y_neg, c='#ff9900', label='Negative node')
plt.xlabel(r"$\Delta$" + ' divergence setpoint (constant)')
plt.ylabel('Probability *100%')
plt.legend()
plt.title('Setpoint vs. Firing Probability')



# x_time = np.arange(0., 5.01, 0.1)

# plt.title('Setpoint vs. Firing Probability')
# plt.plot(x_time, calc_time_to_point(x_time))
# plt.xlabel('distance (m)')
# plt.ylabel('time given (s)')
# plt.title('Time given to reach setpoint')
# plt.savefig('againmeeting_time11-10.png')