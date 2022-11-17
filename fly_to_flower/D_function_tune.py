import numpy as np
import matplotlib.pyplot as plt


def calc_prob_encoding_x(x):
    # if x>0.:
    #     prob = 1./  ( 1.+ np.exp(-5.*x))-0.5
    #     # prob = 1- 0.7/np.exp(6*x)
    #     # prob = x/np.sqrt(1+x**2)
    #     return prob
    # elif x<0.:
    #     prob = 1./(1. + np.exp(1*x))
    #     return  prob
    
    prob_1 = 1./  ( 1.+ np.exp(-20.*x)) - 0.1
    prob_2 = 1./  ( 1.+ np.exp(20*x)) - 0.1
    if prob_1<0.:
        prob_1 = 0.
    if prob_2<0.:
        prob_2 = 0.
    return prob_1,prob_2

def calc_time_to_point(x):
    return np.sqrt(x)*4

import matplotlib.pyplot as plt

y_plus = []
y_neg = []
x = np.arange(-1.0, 1.01, 0.01)
# y = np.exp(1./((x)/2))/2. + 0.5
for i in range(len(x)):
#     # calc_prob_encoding_x(x[i])
    y_plus.append(calc_prob_encoding_x(x[i])[0])
    y_neg.append(calc_prob_encoding_x(x[i])[1])
    # y_neg.append(calc_prob_encoding_x(x[i])[1])

# plt.plot(x[:int(len(y_plus)/2)], y_plus[:int(len(y_plus)/2)], label='Negative node')
# plt.plot(x[int(len(y_plus)/2):], y_plus[int(len(y_plus)/2):], label='Positive node')
plt.plot(x, y_plus, label='Positive node')
plt.plot(x, y_neg, label='Negative node')
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
# plt.savefig('againmeeting_time0811.png')