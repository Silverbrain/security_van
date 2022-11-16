import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
import scipy.stats as stats

#['population_size', 'tournament_size', 'mutation_rate', 'lap', 'value', 'weight', 'fitness', 'convergence_gen','runing_duration']
p_10_data = pd.read_csv('results_normal.csv')
p_100_data = pd.read_csv('results_p_100.csv')


gs = gridspec.GridSpec(1, 2)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
# ax3 = plt.subplot(gs[1, 0])
# ax4 = plt.subplot(gs[1, 1])

#ax3 = plt.subplot(gs[1, :])
ax1.hist(p_10_data['value'], bins = 30, alpha= 0.50, label='p = 10')
ax1.hist(p_100_data['value'], bins = 30, alpha= 0.50, label='p = 100')
ax1.set_xlabel('value')
ax1.legend()

# ax3.hist(normal_data['value'], bins = 30, alpha= 0.50, label='normal')
# ax3.hist(no_cros_data['value'], bins = 30, alpha= 0.50, label='no crossover', color='C2')
# ax3.set_xlabel('value')
# ax3.legend()

ax2.hist(p_10_data['convergence_gen'], bins = 30, alpha= 0.50, label='p = 10')
ax2.hist(p_100_data['convergence_gen'], bins = 10, alpha= 0.50, label='p = 100')
ax2.set_xlabel('convergence')
ax2.legend()

# ax4.hist(normal_data['convergence_gen'], bins = 30, alpha= 0.50, label='normal')
# ax4.hist(no_cros_data['convergence_gen'], bins = 30, alpha= 0.50, label='no crossover', color='C2')
# ax4.set_xlabel('convergence')
# ax4.legend()

#ax3.(data['value'], data['convergence_gen'])
plt.show()