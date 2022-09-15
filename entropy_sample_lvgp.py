import pandas as pd
import numpy as np
from lvgp_pytorch.optim import noise_tune, fit_model_scipy
import matplotlib.pyplot as plt
from scipy.stats import differential_entropy, norm
import gpytorch
import torch
from lvgp_pytorch.models import LVGPR

data = pd.read_csv("/home/hzz6536/PSED/OQMD_xtalsys.csv", index_col=0).drop(columns=['bandgap',
        'structure', 'crystal_structure', 'ntypes'])
# Change crystal system to ordinal coding, for LVGP training
categories_code = data['crystal_system'].astype('category').cat.categories
data['crystal_system'] = data['crystal_system'].astype('category').cat.codes
n_test = 1000  # size of data left out as unlabeled
# Select all unstable orthorhombics -> unlabeled
select1 = data.loc[(data['crystal_system'] == 3) & (data['energy'] > 0)]
remain1 = data.drop(select1.index, inplace=False)
# Randomly select others, avoid triclinic
wo_triclinic = remain1.drop(data.loc[data['crystal_system'] == 5].index)
select2 = wo_triclinic.sample(n=(n_test-select1.shape[0]))

data_l = remain1.drop(select2.index, inplace=False)
data_u = pd.concat([select1, select2])

y_labeled = data_l['energy']
x_labeled = data_l.drop(columns=['energy'])
y_unlabel = data_u['energy']
x_unlabel = data_u.drop(columns=['energy'])

# Info entropy for every crystal sys, within labeled data
# entropies = dict.fromkeys(x_unlabel.crystal_system.unique())
entropies = dict.fromkeys(data.crystal_system.unique())
for key in entropies:
    entropies[key] = differential_entropy(y_labeled[x_labeled['crystal_system'] == key])
# Column = crystal sys, row = iteration
entropies = pd.DataFrame.from_dict([entropies])

def train_LVGP(train_x, train_y, qual_index, quant_index, num_levels_per_var):    
    train_x = torch.tensor(train_x.values)
    train_y = torch.tensor(train_y.values)
    model = LVGPR(
        train_x=train_x,
        train_y=train_y,
        qual_index=qual_index,
        quant_index=quant_index,
        num_levels_per_var=num_levels_per_var,
        quant_correlation_class='RBFKernel'
    ).double()
    
    nll_inc_tuned, opt_history = noise_tune(
        model,
        num_restarts=3
    )
    # marginal log likelihood loss

    return model

# Expected improvement, maximizing y
def calculate_acf(pred_mean, pred_std, y_max):
    improve = pred_mean - y_max
    z_score = np.divide(improve, pred_std + 1e-9)
    acf = np.multiply(improve, norm.cdf(z_score)) + np.multiply(pred_std, norm.pdf(z_score))
    return acf


n_iter = 2  # Total iterations of sampling
n_sample = 1000  # Samples drawn for each unlabeled data point
qual_index = [10]
quant_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
num_levels_per_var = [len(pd.unique(x_labeled.iloc[:,i])) for i in qual_index]
sample_path = np.zeros(n_iter)

for i in range(n_iter):
    # Find the lowest entropy class with unlabeled sample available
    available_sys = x_unlabel.crystal_system.unique()
    entropies_available = entropies.iloc[-1][available_sys]
    lowest_h_sys = entropies_available.idxmin()
    h_curr = entropies_available[lowest_h_sys]  # Incumbent target value 

    # Now only use the lowest entropy crystal system
    # x_target_l = x_labeled.loc[x_labeled['crystal_system'] == lowest_h_sys].copy()
    # x_target_l.drop(columns=['crystal_system'], inplace=True)
    y_target_l = y_labeled.loc[x_labeled['crystal_system'] == lowest_h_sys].copy()

    LVGP_model = train_LVGP(x_labeled, y_labeled, qual_index, quant_index, num_levels_per_var)
    x_target_u = x_unlabel.loc[x_unlabel['crystal_system'] == lowest_h_sys].copy()
    # x_target_u.drop(columns=['crystal_system'], inplace=True)
    
    # Acquisition function values for all unlabeled points
    acq_func = pd.Series(index=x_target_u.index, dtype=float)
    # Compute the new entropy mean and variance
    for index, row in x_target_u.iterrows():
        x_test = torch.tensor(row.values[np.newaxis,:])
        f_preds = LVGP_model(x_test)
        y_samples = f_preds.sample(sample_shape=torch.Size((n_sample,)))

        # Monte Carlo sampling the entropies if a sample is added into labeled
        # h_new = [differential_entropy(np.append(y_target_l.values, y_mc)) for y_mc in y_samples]
        y_with_new_sample = np.concatenate((np.tile(y_target_l.values, (n_sample,1)), y_samples.numpy()), axis=1)
        # h_new = np.apply_along_axis(differential_entropy, axis=1, arr=y_with_new_sample)
        h_new = differential_entropy(y_with_new_sample, axis=1)
        h_new_mean = np.mean(h_new)
        h_new_std = np.std(h_new)
        acq_func[index] = calculate_acf(h_new_mean, h_new_std, h_curr)
    
    # Select the unlabeled datapoint with max acq func
    next_sample_idx = acq_func.idxmax()
    # Evaluate the sample and add to dataset
    x_labeled = pd.concat([x_labeled, x_unlabel.loc[[next_sample_idx]]])
    y_labeled = pd.concat([y_labeled, y_unlabel.loc[[next_sample_idx]]])
    x_unlabel.drop(index=next_sample_idx, inplace=True)
    y_unlabel.drop(index=next_sample_idx, inplace=True)
    sample_path[i] = next_sample_idx

    # entropies[lowest_h_sys] = differential_entropy(y_labeled[x_labeled['crystal_system'] == lowest_h_sys])
    entropies = pd.concat([entropies, entropies.iloc[-1].to_frame().transpose()], ignore_index=True)
    # Change the entropy of crystal sys with new data added
    entropies.iloc[-1][lowest_h_sys] = differential_entropy(y_labeled[x_labeled['crystal_system'] == lowest_h_sys])

    if not i % 10:
        print("Iteration " + str(i) + " completed.")


fig, ax = plt.subplots(dpi=200)
for col in entropies:
    ax.plot(entropies[col], label=categories_code[col])
ax.legend(loc=2)
plt.title('Information entropies')
plt.savefig('/home/hzz6536/PSED/info_entropy_evolution.png', dpi=200)


# Save results to files
entropies.to_pickle('/home/hzz6536/PSED/info_entropy_evolution.pkl')
np.save('/home/hzz6536/PSED/sample_path.npy', sample_path)
np.save('/home/hzz6536/PSED/initial_labeled.npy', data_l.index.to_numpy())