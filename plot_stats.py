# stats.append([np.power(2, i), acc_train, auc_train, nll_train, mse_train, acc_test, auc_test, nll_test, mse_test])

import numpy as np
import matplotlib.pyplot as plt

stats = np.load('stats_isicsctcwad.npy')
x = stats[:, 0]
print('final stats:')
print(stats[-1])
acc_train = stats[:, 1]
auc_train = stats[:, 2]
nll_train = stats[:, 3]
mse_train = stats[:, 4]
acc_test = stats[:, 5]
auc_test = stats[:, 6]
nll_test = stats[:, 7]
mse_test = stats[:, 8]

plt.plot(x, acc_train, label='train')
plt.plot(x, acc_test, label='test')
plt.legend()
plt.title('acc')
plt.xlabel('# of users')
plt.show()

plt.plot(x, auc_train, label='train')
plt.plot(x, auc_test, label='test')
plt.legend()
plt.title('auc')
plt.xlabel('# of users')
plt.show()

plt.plot(x, nll_train, label='train')
plt.plot(x, nll_test, label='test')
plt.legend()
plt.title('nll')
plt.xlabel('# of users')
plt.show()

plt.plot(x, mse_train, label='train')
plt.plot(x, mse_test, label='test')
plt.legend()
plt.title('mse')
plt.xlabel('# of users')
plt.show()