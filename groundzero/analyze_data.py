import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")
    # Set the font to be serif, rather than sans
    sns.set(font='serif', font_scale=1.5)
    sns.set_palette('muted')
    # Make the background white, and specify the
    # specific font family
    sns.set_style("whitegrid", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })

set_style()
with open("results_dir/mnist.pkl", "rb") as f:
   ours, trues, margins, alphas, neyshaburs, bartletts, neyshabur_seconds, epochs = pickle.load(f)
breakpoint()
ours = torch.tensor(ours)/torch.max(torch.tensor(ours))
trues = torch.tensor(trues)/torch.max(torch.tensor(trues))
indices = np.asarray(list(range(len(trues))))
indices *= 5
plt.plot(indices,ours, label="Ours")
plt.plot(indices,trues, label="True")
plt.legend()
plt.xlabel("Epochs")
plt.savefig("results_dir/mlp_large_5l_300e_truemargin.png")