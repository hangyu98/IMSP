import numpy as np
import statsmodels.stats.multitest as mt

p_values = np.array([0.0001,
                     0.0004,
                     0.0019,
                     0.0095,
                     0.0201,
                     0.0278,
                     0.0298,
                     0.0344,
                     0.0459,
                     0.324,
                     0.4262,
                     0.5719,
                     0.6528,
                     0.759,
                     1])

reject, pvals_corrected, alphacSidak, alphacBonf = mt.multipletests(p_values, 0.05, 'fdr_bh')
print(pvals_corrected)
print(reject)
