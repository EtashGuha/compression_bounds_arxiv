"""Main script for training, validation, and testing."""

# Imports Python builtins.
import os
import os.path as osp
import resource
import pickle
from copy import deepcopy
from scipy.stats import bernoulli
# Imports Python packages.
from PIL import ImageFile
import numpy as np
import matplotlib.pyplot as plt
# Imports PyTorch packages.
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
# from pytorch_lightning.utilities import seed_everything
import torch
import copy
import scipy
from tqdm import tqdm
# Imports groundzero packages.
from groundzero.args import parse_args
from groundzero.imports import valid_models_and_datamodules
import pickle
# Prevents PIL from throwing invalid error on large image files.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Prevents DataLoader memory error.
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

def lpq_norm(A, p, q):
    return np.power(np.sum(np.power(np.linalg.norm(np.power(abs(A), p), axis=0),q/p), axis = 0), 1/q)

def load_datamodule(args, datamodule_class):
    """Loads DataModule for training and validation.

    Args:
        args: The configuration dictionary.
        datamodule_class: A class which inherits from groundzero.datamodules.DataModule.

    Returns:
        An instance of datamodule_class parameterized by args.
    """

    datamodule = datamodule_class(args)
    print(datamodule.load_msg())

    return datamodule

def load_model(args, model_class):
    """Loads model for training and validation.

    Args:
        args: The configuration dictionary.
        model_class: A class which inherits from groundzero.models.Model.

    Returns:
        An instance of model_class parameterized by args.
    """

    model = model_class(args)
    print(model.load_msg())
 
    args.ckpt_path = None
    if args.weights:
        if args.resume_training:
            # Resumes training state (weights, optimizer, epoch, etc.) from args.weights.
            args.ckpt_path = args.weights
            print(f"Resuming training state from {args.weights}.")
        else:
            # Loads just the weights from args.weights.
            checkpoint = torch.load(args.weights, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            model.load_state_dict(state_dict, strict=False)
            print(f"Weights loaded from {args.weights}.")     

    return model

def load_trainer(args, addtl_callbacks=None):
    """Loads PL Trainer for training and validation.

    Args:
        args: The configuration dictionary.
        addtl_callbacks: Any desired callbacks besides ModelCheckpoint and TQDMProgressBar.

    Returns:
        An instance of pytorch_lightning.Trainer parameterized by args.
    """

    if args.val_split:
        # Checkpoints model at the specified number of epochs.
        checkpointer1 = ModelCheckpoint(
            filename="{epoch:02d}-{val_loss:.3f}-{val_acc1:.3f}",
            save_top_k=-1,
            every_n_epochs=args.ckpt_every_n_epoch,
        )

        # Checkpoints model with respect to validation loss.
        checkpointer2 = ModelCheckpoint(
            filename="best-{epoch:02d}-{val_loss:.3f}-{val_acc1:.3f}",
            monitor="val_loss",
        )
    else:
        # Checkpoints model with respect to training loss.
        args.check_val_every_n_epoch = 0
        args.num_sanity_val_steps = 0

        # Checkpoints model at the specified number of epochs.
        checkpointer1 = ModelCheckpoint(
            filename="{epoch:02d}-{train_loss:.3f}-{train_acc1:.3f}",
            save_top_k=-1,
            every_n_epochs=args.ckpt_every_n_epoch,
        )

        checkpointer2 = ModelCheckpoint(
            filename="best-{epoch:02d}-{train_loss:.3f}-{train_acc1:.3f}",
            monitor="train_loss",
        )

    progress_bar = TQDMProgressBar(refresh_rate=args.refresh_rate)

    # Sets DDP strategy for multi-GPU training.
    # args.devices = int(args.devices)
    args.accelerator == "cpu"
    # args.strategy = "ddp" if args.devices > 1 else None

    callbacks = [checkpointer1, checkpointer2, progress_bar]
    if isinstance(addtl_callbacks, list):
        callbacks.extend(addtl_callbacks)
    # trainer = Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer = Trainer(accelerator="cpu", max_epochs=100)

    return trainer
def prune_weights(original_weights, alpha):
    original_weights_np = original_weights.detach().numpy()
    mask = get_mask(original_weights, alpha)
    new_weights = mask * original_weights_np
    return torch.from_numpy(new_weights)

def get_mask(original_weights, alpha):
    original_weights_np = original_weights.detach().numpy()
    psi = np.var(original_weights_np)
    weights = -1 * np.square(original_weights_np)/(alpha * psi)
    weights = 1 - np.exp(weights)
    sampler = bernoulli(weights) 
    return sampler.rvs()
def set_weights(model, index, new_weights):
    module = list(model.named_modules())[index]
    length, width = module[1].weight.shape
    module[1].weight.data = copy.deepcopy( new_weights.float().cuda())

def get_norm_interpretable(A):
    u, s, vh = np.linalg.svd(A, full_matrices=False)

    # use first column of u as x
    x = u[:, 0]

    # use first column of vh as y
    y = vh[0, :]

    # maximize xAy
    max_value = np.dot(x, np.dot(A, y))/(np.linalg.norm(x) * np.linalg.norm(y))
    return max_value

def get_margin(model, dataloader):
    batch_margin_error = 0
    total_samples = 0
    min_margin = np.inf
    model = model.cuda()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.cuda(), target.cuda()

        output = model(data)

        masker = torch.zeros_like(output)
        correct_scores = output.gather(1, target.view(-1, 1)).squeeze()
        indices = np.asarray(torch.stack((torch.tensor(list(range(output.shape[0]))).cuda(), target)).T.cpu())
        real_indices = []
        for i in list(indices):
            masker[i[0], i[1]] = -10000000
        margins = correct_scores.cpu() - (output.detach().cpu() + masker.detach().cpu()).max(dim=1).values

        min_margin = min(min_margin, torch.min(margins))

        if min_margin < 0:
            break
    return min_margin

def empirical_margin_loss(model, dataloader, m):
    batch_margin_error = 0
    total_samples = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.cuda(), target.cuda()
        output = model(data)

        masker = torch.zeros_like(output)
        correct_scores = output.gather(1, target.view(-1, 1)).squeeze()
        indices = np.asarray(torch.stack((torch.tensor(list(range(output.shape[0]))).cuda(), target)).T.cpu())
        real_indices = []
        for i in list(indices):
            masker[i[0], i[1]] = -10000000
        margins = correct_scores.cpu() - (output.detach().cpu() + masker.detach().cpu()).max(dim=1).values
        margins = torch.where(margins < m, torch.ones_like(margins), 0)
        batch_margin_error += torch.sum(margins)
        total_samples += data.shape[0]
    return batch_margin_error / total_samples, total_samples



def main(args, model_class, datamodule_class, callbacks=None, model_hooks=None):
    """Main method for training and validation.

    Args:
        args: The configuration dictionary.
        model_class: A class which inherits from groundzero.models.Model.
        datamodule_class: A class which inherits from groundzero.datamodules.DataModule.
        callbacks: Any desired callbacks besides ModelCheckpoint and TQDMProgressBar.
        model_hooks: Any desired functions to run on the model before training.

    Returns:
        The trained model with its validation and test metrics.
    """
    os.makedirs(args.out_dir, exist_ok=True)

    # Sets global seed for reproducibility. Due to CUDA operations which can't
    # be made deterministic, the results may not be perfectly reproducible.
    # seed_everything(seed=args.seed, workers=True)

    datamodule = load_datamodule(args, datamodule_class)
    args.num_classes = datamodule.num_classes
    model = load_model(args, model_class)
    
    if model_hooks:
        for hook in model_hooks:
            hook(model)

    def get_error(alpha, model, trainer):
        model = model.cuda()
        new_generalization_bound = 0
        with torch.no_grad():
            new_model = copy.deepcopy(model)
            for index, (name, module) in enumerate(model.named_modules()):
                if isinstance(module, torch.nn.Linear):
                    set_weights(new_model,index, prune_weights(module.weight.cpu(), alpha))
        max_difference = -1
        model = model.cuda()
        new_model = new_model.cuda()
        metrics = trainer.test(new_model, datamodule)
        
        return metrics
                
    def find_alpha_true(alpha, model):
        new_generalization_bound = 0
        model = model.cpu()
        with torch.no_grad():
            new_model = copy.deepcopy(model).cpu()
            for index, (name, module) in enumerate(model.named_modules()):
                if isinstance(module, torch.nn.Linear):
                    set_weights(new_model,index, prune_weights(module.weight.cpu(), alpha))
        max_difference = -1
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(datamodule.train_dataloader()):
                data = data.cpu()
                target = target.cpu()
                model = model.cpu()
                new_model = new_model.cpu()


                original_output = model(data)
                edited_output = new_model(data)
                original_correct_scores = original_output.gather(1, target.view(-1, 1)).squeeze()
                edited_correct_scores = edited_output.gather(1, target.view(-1, 1)).squeeze()
                max_difference = max(torch.max(abs(original_correct_scores - edited_correct_scores)), max_difference)

        sum_of_diffs = 0
        product_of_norms = 1
        product_of_constants = 1
        diffs = []

        for index, (name, module) in enumerate(model.named_modules()):
            if isinstance(module, torch.nn.Linear):
                product_of_norms *= np.linalg.norm(module.weight.detach().cpu().numpy(), ord=2)
                hat_matrix = list(new_model.cpu().named_modules())[index][1].weight.detach().numpy()
                diffs.append(np.linalg.norm(module.weight.detach().cpu().numpy()) - np.linalg.norm(hat_matrix))           
                diff = np.linalg.norm(module.weight.detach().cpu().numpy() - hat_matrix, ord=2)
                sum_of_diffs += diff/np.linalg.norm(module.weight.detach().cpu().numpy(), ord=2)
                d_1, d_2 = module.weight.T.shape
                if d_2 < d_1:
                    t_val = scipy.optimize.bisect(lambda t: 1 - len(datamodule.train_dataloader().dataset)*np.exp(-t * t * d_1)-.99, 0, 1, xtol=.01)
                    constant = t_val * np.sqrt(d_2) 
                    if constant > 1:
                        constant = 1
                else:
                    constant = 1
                product_of_constants *= constant

        new_model = new_model.cpu()
        max_input_norm = 0
        model = model.cpu()
        new_model = new_model.cpu()
        for batch_idx, (data, target) in enumerate(datamodule.train_dataloader()):
            data.cpu()
            flattened_data = data.flatten(2,3).squeeze(1)
            max_input_norm = max(max_input_norm, torch.max(torch.norm(flattened_data, dim=1, p=2)))
        needed_margin =  np.e * max_input_norm * sum_of_diffs * product_of_norms * product_of_constants
        return max_difference, needed_margin, np.mean(diffs)

    def calc_neyshabur(model, gamma):
        initial_value = 1
        for index, (name, module) in enumerate(model.named_modules()):
            if isinstance(module, torch.nn.Linear):
                initial_value *= np.square(np.linalg.norm(module.weight.detach().cpu().numpy(), 'fro'))
        initial_value *= 1/(gamma * gamma)
        return initial_value

    def calc_bartlett(model, gamma):
        initial_value = 1
        product_of_norms = 1
        sum_of_ratios = 0
        for index, (name, module) in enumerate(model.named_modules()):
            if isinstance(module, torch.nn.Linear):
                product_of_norms *= np.square(np.linalg.norm(module.weight.detach().cpu().numpy(), ord=2))
                sum_of_ratios += np.square(lpq_norm(module.weight.detach().cpu().numpy(), 1, 2))/np.square(np.linalg.norm(module.weight.detach().cpu().numpy(), ord=2))
        initial_value *= 1/(gamma * gamma) * product_of_norms * sum_of_ratios
        return initial_value

    def calc_second_neyshabur(model, gamma):
        initial_value = 1
        product_of_norms = 1
        sum_of_ratios = 0
        for index, (name, module) in enumerate(model.named_modules()):
            if isinstance(module, torch.nn.Linear):
                product_of_norms *= np.square(np.linalg.norm(module.weight.detach().cpu().numpy(), ord=2))
                sum_of_ratios += np.square(np.linalg.norm(module.weight.detach().cpu().numpy(), ord="fro"))/np.square(np.linalg.norm(module.weight.detach().cpu().numpy(), ord=2)) * module.weight.shape[1]
        initial_value *= 1/(gamma * gamma) * product_of_norms * sum_of_ratios
        return initial_value

    def find_alpha(alpha, model, pre_input_margin):
        model = model.cuda()
        new_generalization_bound = 0
        with torch.no_grad():
            new_model = copy.deepcopy(model)
            for index, (name, module) in enumerate(model.named_modules()):
                if isinstance(module, torch.nn.Linear):
                    set_weights(new_model,index, prune_weights(module.weight.cpu(), alpha))
    
        product_of_norms = 1
        product_of_constants = 1
        sum_of_diffs = 0
        diffs = []
        for index, (name, module) in enumerate(model.named_modules()):
            if isinstance(module, torch.nn.Linear):
                product_of_norms *= np.linalg.norm(module.weight.detach().cpu().numpy(), ord=2)
                hat_matrix = list(new_model.cpu().named_modules())[index][1].weight.detach().numpy()   
                diffs.append(np.linalg.norm(hat_matrix) - np.linalg.norm(module.weight.detach().cpu().numpy(), ord=2))         
                diff = np.linalg.norm(module.weight.detach().cpu().numpy() - hat_matrix, ord=2)
                sum_of_diffs += diff/np.linalg.norm(module.weight.detach().cpu().numpy(), ord=2)
                d_1, d_2 = module.weight.T.shape
                if d_2 < d_1:
                    t_val = scipy.optimize.bisect(lambda t: 1 - len(datamodule.train_dataloader().dataset)*np.exp(-t * t * d_1)-.99, 0, 1, xtol=.01)
                    constant = t_val * np.sqrt(d_2) 
                    if constant > 1:
                        constant = 1
                else:
                    constant = 1
                product_of_constants *= constant
        
        max_input_norm = 0
        model = model.cuda()
        new_model = new_model.cuda()
        for batch_idx, (data, target) in enumerate(datamodule.train_dataloader()):
            flattened_data = data.flatten(2,3).squeeze(1)
            max_input_norm = max(max_input_norm, torch.max(torch.norm(flattened_data, dim=1, p=2)))

        needed_margin =  np.e * max_input_norm * sum_of_diffs * product_of_norms * product_of_constants
        return pre_input_margin - needed_margin

    def get_number_of_zeros(model, alpha):
        max_weight = 1
        with torch.no_grad():
            new_model = copy.deepcopy(model)
            for index, (name, module) in enumerate(model.named_modules()):
                if isinstance(module, torch.nn.Linear):
                    prune_weights(module.weight.cpu(), alpha)
                    if module.weight.shape[1] > max_weight:
                        num_zero = torch.sum(prune_weights(module.weight.cpu(), alpha) != 0, dim=1)
                        max_weight = module.weight.shape[1]
                        predicted = 3 * module.weight.shape[1] * (np.sqrt(alpha + 2) - np.sqrt(alpha))/np.sqrt(alpha + 2)

        return np.max(num_zero.detach().numpy()), predicted
    def compute_best_generalization_bound(model, alpha):
        model = model.cuda()

        new_generalization_bound = 0
        with torch.no_grad():
            new_model = copy.deepcopy(model)
            for index, (name, module) in enumerate(model.named_modules()):
                if isinstance(module, torch.nn.Linear):
                    set_weights(new_model,index, prune_weights(module.weight.cpu(), alpha))
                    d1 = module.weight.shape[0]
                    d2 = module.weight.shape[1]
                    new_generalization_bound +=  d1 * d2 * (np.sqrt(alpha +2) - np.sqrt(alpha))/np.sqrt(alpha +2) * np.square(np.log(max(d1, d2)))
        num_samples = len(datamodule.train_dataloader().dataset)
        product_of_norms = 1
        product_of_constants = 1
        sum_of_diffs = 0
        for index, (name, module) in enumerate(model.named_modules()):
            if isinstance(module, torch.nn.Linear):
                product_of_norms *= np.linalg.norm(module.weight.detach().cpu().numpy(), ord=2)
                hat_matrix = list(new_model.cpu().named_modules())[index][1].weight.detach().numpy()            
                diff = np.linalg.norm(module.weight.detach().cpu().numpy() - hat_matrix, ord=2)
                sum_of_diffs += diff/np.linalg.norm(module.weight.detach().cpu().numpy(), ord=2)
                d_1, d_2 = module.weight.T.shape
                if d_2 < d_1:
                    t_val = scipy.optimize.bisect(lambda t: 1 - len(datamodule.train_dataloader().dataset)*np.exp(-t * t * d_1)-.99, 0, 1, xtol=.01)
                    constant = t_val * np.sqrt(d_2) 
                    if constant > 1:
                        constant = 1
                else:
                    constant = 1
                product_of_constants *= constant

        new_model = new_model.cuda()
        max_input_norm = 0
        model = model.cuda()
        new_model = new_model.cuda()
        for batch_idx, (data, target) in enumerate(datamodule.train_dataloader()):
            flattened_data = data.flatten(2,3).squeeze(1)
            max_input_norm = max(max_input_norm, torch.max(torch.norm(flattened_data, dim=1, p=2)))

        needed_margin =  np.e * max_input_norm * sum_of_diffs * product_of_norms * product_of_constants
        compressed_test_metrics_new_model = trainer.test(new_model, datamodule=datamodule)
        return empirical_margin_loss(model.cuda(), datamodule.train_dataloader(), needed_margin)[0] + np.sqrt(new_generalization_bound/num_samples),  empirical_margin_loss(new_model.cuda(), datamodule.test_dataloader(), 0)[0]
    prev_alpha = 0
    our_generalization_bounds = []
    true_generalizations = []
    neyshaburs = []
    bartletts = [] 
    neyshabur_seconds = [] 
    alphas = []
    margins = []
    num_epochs = 300
    epochs = []
    trainer = load_trainer(args, addtl_callbacks=callbacks)
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)
    trus=[]
    fas = []
    for i in np.linspace(0, 10, 100):
        a, b, _ = find_alpha_true(i, model)
        trus.append(a.item())
        fas.append(b.item())

    predicted_counts = []
    true_counts = []
    for i in np.linspace(0, 10, 100):
        true, predicted = get_number_of_zeros(model, i)
        true_counts.append(true)
        predicted_counts.append(predicted)
    with open("cifar.pkl" , "wb") as f:
        pickle.dump((trus, fas), f)
    with open("cifar_num_counts.pkl", "wb") as f:
        pickle.dump((true_counts, predicted_counts), f)

    breakpoint()
    # for epoch_iteration in range(num_epochs):
    # for epoch in range(num_epochs):
    #     trainer = load_trainer(args, addtl_callbacks=callbacks)
    #     trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)
    #     model_margin = get_margin(model, datamodule.train_dataloader()) 
    #     if model_margin > 0:
    #         best_alpha = scipy.optimize.bisect(find_alpha,  0, 10, args= (model, model_margin), xtol=1e-3)
    #     else: 
    #         model_margin = 0
    #         best_alpha = 0
    #     our_generalization_bound, true_generalization = compute_best_generalization_bound(model, best_alpha)
    #     our_generalization_bounds.append(our_generalization_bound)
    #     true_generalizations.append(true_generalization)
    #     neyshabur_seconds.append(calc_second_neyshabur(model, model_margin))
    #     margins.append(model_margin)
    #     neyshaburs.append(calc_neyshabur(model, model_margin))
    #     bartletts.append(calc_bartlett(model, model_margin))
    #     epochs.append(epoch)
    #     alphas.append(best_alpha)
        
    
    #     with open("results_dir/{}.pkl".format(args.datamodule), "wb") as f:
    #         pickle.dump((our_generalization_bounds, true_generalizations, margins, alphas, neyshaburs, bartletts, neyshabur_seconds, epochs), f)
    
    args.ckpt_path = None
    return model


if __name__ == "__main__":
    args = parse_args()
    
    models, datamodules = valid_models_and_datamodules()

    main(args, models[args.model], datamodules[args.datamodule])

