import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns

dataset = "cifar"
with open('{}.pkl'.format(dataset), "rb") as f:
    trus, fas = pickle.load(f)

fig, ax = plt.subplots()
sns.set_style("whitegrid", {
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "serif"]
})
sns.lineplot(
    ax=ax,
    x=np.linspace(0, 10, 100),
    y=trus,
    label="True Pruning Error",
    color='black',
    linewidth=2.8,     
)
sns.lineplot(
    ax=ax,
    x=np.linspace(0, 10, 100),
    y=fas,
    label="Bound on Pruning Error",
    color='grey',
    linewidth=2.8,     
)
plt.xlabel(r'$d$', fontsize=16)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=16)

    
fig.savefig("pruningerror_{}.pdf".format(dataset))
fig.clf()


with open("{}_num_counts.pkl".format(dataset), "rb") as f:
    trues, predicted = pickle.load(f)
fig, ax = plt.subplots()
sns.set_style("whitegrid", {
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "serif"]
})
sns.lineplot(
    ax=ax,
    x=np.linspace(0, 10, 100),
    y=predicted,
    label="Predicted Sparsity",
    color='black',
    linewidth=2.8,     
)
sns.lineplot(
    ax=ax,
    x=np.linspace(0, 10, 100),
    y=trues,
    label=r"$\max(j_r, j_c)$",
    color='grey',
    linewidth=2.8,     
)
plt.xlabel(r'$d$', fontsize=16)
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=16)
plt.tight_layout()
    
fig.savefig("numcounts_{}.pdf".format(dataset))
fig.clf()


