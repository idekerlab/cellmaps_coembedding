"""
ProMERGE co-embedding algorithm.

This module trains a base-context ProteinProjector model, uses the resulting
anchor embeddings, and then learns query-context embeddings with ProMERGE.
"""

import os
import collections
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import cellmaps_coembedding.proteinprojector as proteingps
from cellmaps_coembedding.exceptions import CellmapsCoEmbeddingError
from cellmaps_coembedding.proteinprojector.architecture import TrainingDataWrapper, Protein_Dataset
from .model import CoEmbed

MODALITY_SEP = '___'

__all__ = [
    'MODALITY_SEP',
    'add_gaussian_noise',
    'variance_regularizer',
    'save_results',
    'fit_predict',
]


def add_gaussian_noise(df, frac=0.05, per_dim=True, seed=None):
    """
    Adds Gaussian noise to an embedding dataframe.

    :param df: Dataframe containing embedding values.
    :type df: pandas.DataFrame
    :param frac: Fraction of the data standard deviation to use as noise scale.
    :type frac: float
    :param per_dim: If ``True``, calculate noise scale per embedding dimension.
    :type per_dim: bool
    :param seed: Optional random seed.
    :type seed: int or None
    :return: Dataframe with added Gaussian noise.
    :rtype: pandas.DataFrame
    """
    rng = np.random.default_rng(seed)
    X = df.to_numpy(dtype=np.float32)

    if per_dim:
        scale = X.std(axis=0, keepdims=True)  # (1, d)
    else:
        scale = X.std()

    noise = rng.normal(loc=0.0, scale=frac * scale, size=X.shape).astype(X.dtype)
    X_noisy = X + noise
    return pd.DataFrame(X_noisy, index=df.index, columns=df.columns)


def variance_regularizer(z, target: float = 1.0):
    """
    Computes variance regularization loss for a latent tensor.

    :param z: Latent tensor.
    :type z: torch.Tensor
    :param target: Target standard deviation.
    :type target: float
    :return: Mean squared error between observed and target standard deviation.
    :rtype: torch.Tensor
    """
    std = z.std(dim=0)
    return F.mse_loss(std, torch.full_like(std, target))


def _canonicalize_modality_names(modality_names, cond_str_list, mod_str_list):
    """
    Converts user-facing embedding names to ProMERGE canonical modality names.

    :param modality_names: Names passed with the input embeddings.
    :type modality_names: list[str]
    :param cond_str_list: Context strings to locate in each name.
    :type cond_str_list: list[str]
    :param mod_str_list: Modality strings to locate in each name.
    :type mod_str_list: list[str]
    :return: Canonical names in ``modality-context`` format.
    :rtype: list[str]
    :raises CellmapsCoEmbeddingError: If names cannot be mapped unambiguously.
    """
    canonical_names = []
    seen_names = set()

    for name in modality_names:
        matching_conditions = [cond for cond in cond_str_list if cond in name]
        matching_modalities = [mod for mod in mod_str_list if mod in name]

        if len(matching_conditions) != 1 or len(matching_modalities) != 1:
            raise CellmapsCoEmbeddingError(
                'ProMERGE embedding name "{}" must contain exactly one context from {} '
                'and exactly one modality from {}'.format(name, cond_str_list, mod_str_list)
            )

        canonical_name = '{}-{}'.format(matching_modalities[0], matching_conditions[0])
        if canonical_name in seen_names:
            raise CellmapsCoEmbeddingError(
                'ProMERGE received duplicate context/modality embedding "{}"'.format(canonical_name)
            )
        canonical_names.append(canonical_name)
        seen_names.add(canonical_name)

    return canonical_names


def _get_modalities_per_condition(modality_names, cond_str_list):
    """
    Groups canonical modality names by context.

    :param modality_names: Canonical names in ``modality-context`` format.
    :type modality_names: list[str]
    :param cond_str_list: Context strings to locate in each name.
    :type cond_str_list: list[str]
    :return: Dictionary mapping each context to available modalities.
    :rtype: dict
    """
    result = defaultdict(set)
    for name in modality_names:
        modality, condition = name.rsplit('-', 1)
        if condition in cond_str_list:
            result[condition].add(modality)
    return dict(result)


def save_results(model, protein_dataset, data_wrapper, anchor_emb, results_suffix=''):
    """
    Evaluates the model, saves the state, and exports embeddings for each protein.

    :param model: The neural network model.
    :type model: torch.nn.Module
    :param protein_dataset: The dataset containing protein data.
    :type protein_dataset: cellmaps_coembedding.proteinprojector.architecture.Protein_Dataset
    :param data_wrapper: Data handling and configurations as an object.
    :type data_wrapper: TrainingDataWrapper
    :param results_suffix: Suffix to append to results directory for saving.
    :type results_suffix: str
    """
    resultsdir = data_wrapper.resultsdir + results_suffix
    model.eval()
    torch.save(model.state_dict(), '{}_model.pth'.format(resultsdir))

    all_latents = dict()
    all_outputs = dict()
    all_disentangles = dict() ##
    for input_modality in data_wrapper.modalities_dict.keys():
        all_latents[input_modality] = dict()
        all_disentangles[input_modality] = dict() ##
        for output_modality in data_wrapper.modalities_dict.keys():
            output_key = input_modality + MODALITY_SEP + output_modality
            all_outputs[output_key] = dict()

    embeddings_by_protein = dict()
    disentangles_by_protein = dict() ##
    with torch.no_grad():
        for i in np.arange(len(protein_dataset)):
            protein, mask, protein_index = protein_dataset[i]
            protein_name = protein_dataset.protein_ids[protein_index]
            embeddings_by_protein[protein_name] = dict()
            disentangles_by_protein[protein_name] = dict() ##
            
            ## reformat the protein for model()
            protein_input = {}
            for k, v in protein.items():
                protein_input[k] = v.unsqueeze(0)
            latents, outputs, disentangles = model(protein_input, [protein_name], anchor_emb)
            
            for modality, latent in latents.items():
                if mask[modality] > 0:
                    protein_embedding = latent.detach().cpu().numpy()[0]
                    all_latents[modality][protein_name] = protein_embedding
                    embeddings_by_protein[protein_name][modality] = protein_embedding
            for modality, output in outputs.items():
                input_modality = modality.split(MODALITY_SEP)[0]
                # Only the input modality must be present to save this reconstruction.
                if (mask[input_modality] > 0):
                    all_outputs[modality][protein_name] = output.detach().cpu().numpy()[0]
            
            ## also save disentangles
            for modality, disentangle in disentangles.items():
                if mask[modality] > 0:
                    protein_embedding = disentangle.detach().cpu().numpy()[0]
                    all_disentangles[modality][protein_name] = protein_embedding
                    disentangles_by_protein[protein_name][modality] = protein_embedding
                    
    # save latent embeddings
    for modality, latents in all_latents.items():
        filepath = '{}_{}_latent.tsv'.format(resultsdir, modality)
        proteingps.write_embedding_dictionary_to_file(
            filepath, latents, data_wrapper.latent_dim
        )

    # save averaged coembedding
    filepath = '{}_latent.tsv'.format(resultsdir)
    proteingps.write_embedding_dictionary_to_file(
        filepath, embeddings_by_protein, data_wrapper.latent_dim
    )
    
    ## save disentangle embeddings
    for modality, disentangles in all_disentangles.items():
        filepath = '{}_{}_disentangle.tsv'.format(resultsdir, modality)
        proteingps.write_embedding_dictionary_to_file(
            filepath, disentangles, data_wrapper.latent_dim
        )
    
    ## save averaged disentangle
    filepath = '{}_disentangle.tsv'.format(resultsdir)
    proteingps.write_embedding_dictionary_to_file(
        filepath, disentangles_by_protein, data_wrapper.latent_dim
    )

    # save reconstructed embeddings
    for modality, outputs in all_outputs.items():
        filepath = '{}_{}_reconstructed.tsv'.format(resultsdir, modality)
        output_modality = modality.split(MODALITY_SEP)[1]
        output_modality_dim = data_wrapper.modalities_dict[output_modality].input_dim
        proteingps.write_embedding_dictionary_to_file(
            filepath, outputs, output_modality_dim
        )

    return embeddings_by_protein


def fit_predict(
    resultsdir,
    modality_data,
    modality_names=(),
    latent_dim=128,
    n_epochs=300,
    save_update_epochs=True,
    batch_size=16,
    triplet_margin=0.5,
    dropout=0,
    l2_norm=True,
    mean_losses=False,
    learn_rate=1e-4,
    hidden_size_1=512,
    hidden_size_2=256,
    negative_from_batch=False,

    cond_str_list=None,
    mod_str_list=None,
    mod_str_list_mine=None,
    lambda_reconstruction=1.0,
    lambda_disentangle=1.0,
    lambda_triplet_disentangle=1.0,
    lambda_l2_disentangle=0,
    lambda_l2_latent=0,
    lambda_var=0.1,
    disentangle_method="MINE",

    save_epoch=50,
    base_proteingps_parameters=None
):
    """
    Trains and predicts query-context co-embeddings with ProMERGE.

    :param resultsdir: Directory to save ProMERGE intermediate files.
    :type resultsdir: str
    :param modality_data: Input embeddings, one list per modality/context.
    :type modality_data: list
    :param modality_names: Names corresponding to ``modality_data``.
    :type modality_names: list[str]
    :param latent_dim: Dimensionality of the learned latent embeddings.
    :type latent_dim: int
    :param n_epochs: Number of training epochs.
    :type n_epochs: int
    :param save_update_epochs: Whether to save intermediate epoch outputs.
    :type save_update_epochs: bool
    :param batch_size: Batch size for training.
    :type batch_size: int
    :param triplet_margin: Margin for triplet disentanglement loss.
    :type triplet_margin: float
    :param dropout: Dropout rate for neural network layers.
    :type dropout: float
    :param l2_norm: Whether to L2 normalize latent embeddings.
    :type l2_norm: bool
    :param mean_losses: Whether to average losses instead of summing them.
    :type mean_losses: bool
    :param learn_rate: Learning rate for optimizers.
    :type learn_rate: float
    :param hidden_size_1: Size of the first hidden layer.
    :type hidden_size_1: int
    :param hidden_size_2: Size of the second hidden layer.
    :type hidden_size_2: int
    :param negative_from_batch: Whether to sample negatives from the same batch.
    :type negative_from_batch: bool
    :param cond_str_list: Base and query context strings to find in embedding names.
    :type cond_str_list: list[str] or None
    :param mod_str_list: Modality strings to find in embedding names.
    :type mod_str_list: list[str] or None
    :param mod_str_list_mine: Modalities used for MINE disentanglement.
    :type mod_str_list_mine: list[str] or None
    :param lambda_reconstruction: Weight for reconstruction loss.
    :type lambda_reconstruction: float
    :param lambda_disentangle: Weight for disentanglement loss.
    :type lambda_disentangle: float
    :param lambda_triplet_disentangle: Weight for triplet disentanglement loss.
    :type lambda_triplet_disentangle: float
    :param lambda_l2_disentangle: Weight for L2 regularization on disentangled values.
    :type lambda_l2_disentangle: float
    :param lambda_l2_latent: Weight for L2 regularization on latent values.
    :type lambda_l2_latent: float
    :param lambda_var: Weight for variance regularization.
    :type lambda_var: float
    :param disentangle_method: Disentanglement method. One of ``MINE`` or ``subtract``.
    :type disentangle_method: str
    :param save_epoch: Epoch interval for intermediate saves.
    :type save_epoch: int
    :param base_proteingps_parameters: Overrides for the base-context ProteinProjector run.
    :type base_proteingps_parameters: dict or None
    :return: Generator of co-embedding rows.
    :rtype: generator
    :raises CellmapsCoEmbeddingError: If context or modality names are invalid.
    """
    modality_data = list(modality_data)
    modality_names = list(modality_names)
    cond_str_list = list(cond_str_list) if cond_str_list is not None else ['base', 'query']
    mod_str_list = list(mod_str_list) if mod_str_list is not None else ['mod1', 'mod2']
    if mod_str_list_mine is not None:
        mod_str_list_mine = list(mod_str_list_mine)
    base_proteingps_parameters = dict(base_proteingps_parameters or {})

    if len(cond_str_list) != 2:
        raise CellmapsCoEmbeddingError(
            'ProMERGE currently supports exactly two contexts: one base context and one query context'
        )
    if len(mod_str_list) < 2:
        raise CellmapsCoEmbeddingError('ProMERGE requires at least two modality strings')
    if len(modality_names) != len(modality_data):
        raise CellmapsCoEmbeddingError('ProMERGE requires one embedding name for each embedding input')
    if disentangle_method not in {'MINE', 'subtract'}:
        raise CellmapsCoEmbeddingError('disentangle_method should be MINE or subtract')

    modality_names = _canonicalize_modality_names(modality_names, cond_str_list, mod_str_list)
    modalities_per_cond = _get_modalities_per_condition(modality_names, cond_str_list)
    print("Modalities for each context:")
    print(f"\t{modalities_per_cond}")

    for cond in cond_str_list:
        if cond not in modalities_per_cond:
            raise CellmapsCoEmbeddingError('No ProMERGE embeddings found for context "{}"'.format(cond))

    if len(modalities_per_cond[cond_str_list[0]]) < len(modalities_per_cond[cond_str_list[1]]):
        raise CellmapsCoEmbeddingError('ProMERGE base context should have equal or more modalities than query context')

    if len(modalities_per_cond[cond_str_list[1]]) >= 2:
        if (mod_str_list_mine is None) or (mod_str_list_mine == []):
            print("No mod_str_list_mine specified. Using all modalities in query for MINE.")
            mod_str_list_mine = sorted(modalities_per_cond[cond_str_list[1]])
        else:
            if all(ii in modalities_per_cond[cond_str_list[1]] for ii in mod_str_list_mine):
                print("Using modalities in mod_str_list_mine for MINE.")
            else:
                raise CellmapsCoEmbeddingError("mod_str_list_mine and modalities in query context do not match")
    elif len(modalities_per_cond[cond_str_list[1]]) == 1:
        print("Query context has only one available modality")
        if (mod_str_list_mine is None) or (mod_str_list_mine == []):
            print("No mod_str_list_mine specified. Using all modalities in base for MINE.")
            mod_str_list_mine = sorted(modalities_per_cond[cond_str_list[0]])
        else:
            print("Using modalities in mod_str_list_mine for MINE.")
        
        if any(ii in modalities_per_cond[cond_str_list[1]] for ii in mod_str_list_mine):
            print("Using the one available query modality to impute the rest modalities needed for MINE")
            for m in range(len(modality_names)):
                m_query_list = list(modalities_per_cond[cond_str_list[1]])
                if (m_query_list[0] in modality_names[m]) and (cond_str_list[1] in modality_names[m]):
                    break
            query_valid_modality_data = modality_data[m]
            
            mod_to_fill = sorted(set(mod_str_list_mine) - modalities_per_cond[cond_str_list[1]])
            for mod in mod_to_fill:
                mod_impute_name = f"{mod}-{cond_str_list[1]}"
                mod_impute_data = add_gaussian_noise(
                    pd.DataFrame(query_valid_modality_data).set_index(0)
                )
                modality_names.append(mod_impute_name)
                modality_data.append(mod_impute_data.reset_index().values.tolist())
        else:
            raise CellmapsCoEmbeddingError("mod_str_list_mine should contain the modality in query context")

    if disentangle_method == "MINE" and len(mod_str_list_mine) < 2:
        raise CellmapsCoEmbeddingError("MINE requires at least two modalities")
    if disentangle_method == "subtract" and len(mod_str_list_mine) < 1:
        raise CellmapsCoEmbeddingError("subtract disentanglement requires at least one modality")
    anchor_mod_str_list = sorted(set(mod_str_list_mine).union(modalities_per_cond[cond_str_list[1]]))

    #### other set up
    
    # for each context in the cond_str_list,
    # return a list of index in the embedding list that representing that context
    cond2cond_idx = {}
    for cond in cond_str_list:
        cond2cond_idx[cond] = [
            ii for ii in range(len(modality_names))
            if cond in modality_names[ii]
        ]
    
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    
    #### First condition (anchor condition) ####
    cond = cond_str_list[0]
    
    # check if base context anchor emb by all MINE needed modality already exist
    exist_list = []
    print("Checking base context file")
    for mod in anchor_mod_str_list:
        file_check = f"{resultsdir}/{cond}/{cond}_{mod}-{cond}_latent.tsv"
        print(f"\t{file_check}")
        print(f"\tExists: {os.path.exists(file_check)}")
        exist_list.append(os.path.exists(file_check))
        
    # if any anchor emb not exist, run ProteinGPS first
    if not np.all(exist_list):
        print("Some base context embeddings were NOT found. Starting ProteinGPS on base context, using base_proteingps_parameters")
        anchor_resultsdir = os.path.join(resultsdir, cond)
        os.makedirs(anchor_resultsdir, exist_ok=True)
        base_proteingps_parameters.setdefault('latent_dim', latent_dim)
        base_proteingps_parameters.setdefault('n_epochs', n_epochs)
        base_proteingps_parameters.setdefault('save_update_epochs', save_update_epochs)
        base_proteingps_parameters.setdefault('batch_size', batch_size)
        base_proteingps_parameters.setdefault('triplet_margin', triplet_margin)
        base_proteingps_parameters.setdefault('dropout', dropout)
        base_proteingps_parameters.setdefault('l2_norm', l2_norm)
        base_proteingps_parameters.setdefault('mean_losses', mean_losses)
        base_proteingps_parameters.setdefault('learn_rate', learn_rate)
        base_proteingps_parameters.setdefault('hidden_size_1', hidden_size_1)
        base_proteingps_parameters.setdefault('hidden_size_2', hidden_size_2)
        base_proteingps_parameters.setdefault('negative_from_batch', negative_from_batch)
        base_proteingps_parameters.setdefault('lambda_reconstruction', lambda_reconstruction)
        base_proteingps_parameters.setdefault('lambda_triplet', lambda_triplet_disentangle)
        base_proteingps_parameters.setdefault('lambda_l2', lambda_l2_latent)
        anchor_generator = proteingps.fit_predict(
            resultsdir = os.path.join(anchor_resultsdir, cond),
            modality_data=[modality_data[ii] for ii in cond2cond_idx[cond]],
            modality_names=[modality_names[ii] for ii in cond2cond_idx[cond]],
            **base_proteingps_parameters
        )
        for _ in anchor_generator:
            pass
    
    #### Second condition ####
    cond = cond_str_list[1]
    print(f"Start ProMERGE on {cond} context")
    cond_resultsdir = os.path.join(resultsdir, cond)
    os.makedirs(cond_resultsdir, exist_ok=True)
    source_file = open('{}/{}.txt'.format(cond_resultsdir, cond), 'w')
    
    # load the anchor emb by modality
    # the anchor is always the first condition (cond_str_list[0])
    print("Loading base context")
    anchor_emb = {}
    for mod in anchor_mod_str_list:
        filename = (
            f"{resultsdir}/{cond_str_list[0]}/"
            f"{cond_str_list[0]}_{mod}-{cond_str_list[0]}_latent.tsv"
        )
        anchor_emb[f'{mod}-{cond}'] = pd.read_csv(filename, sep="\t", index_col=0)
    
    #### Data loader ####
    modality_data_cond = [modality_data[ii] for ii in cond2cond_idx[cond]]
    modality_names_cond = [modality_names[ii] for ii in cond2cond_idx[cond]]
    data_wrapper = TrainingDataWrapper(
        modality_data_cond, modality_names_cond,
        device, l2_norm, dropout, latent_dim,
        hidden_size_1, hidden_size_2, os.path.join(cond_resultsdir, cond)
    )
    data_wrapper.disentangle_method = disentangle_method
    dataset = Protein_Dataset(data_wrapper.modalities_dict)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    #### model ####
    model = CoEmbed(data_wrapper).to(device)
    
    #### optimizer ####
    # define different optimizer for each MINE module
    if disentangle_method == "MINE":
        MINE_param_dict = {
            name: list(mine.parameters())
            for name, mine in model.mines.items()
        }
        mine_optimizers = {
            name: torch.optim.Adam(params, lr=learn_rate / 2)
            for name, params in MINE_param_dict.items()
        }
        all_MINE_params = set(p for plist in MINE_param_dict.values() for p in plist)
        other_params = [p for p in model.parameters() if p not in all_MINE_params]
        main_optimizer = torch.optim.Adam(other_params, lr=learn_rate)

        ema_marginal_exp = {f'{mod}-{cond}': 1.0 for mod in mod_str_list_mine}
        ema_rate = 0.01
    else:
        main_optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
        
    #### training loop (epoch) ####
    for epoch in tqdm(range(n_epochs)):
        
        # first epoch only update MINE for warmup
        warmup_mine = (disentangle_method == "MINE" and epoch == 0)
        
        # init loss variables
        total_loss = []
        total_reconstruction_loss = []
        total_triplet_disentangle_loss = []
        total_l2_disentangle_loss = []
        total_l2_latent_loss = []
        total_disentangle_loss = []
        total_var_loss = []
        
        total_reconstruction_loss_by_modality = collections.defaultdict(list)
        total_triplet_disentangle_loss_by_modality = collections.defaultdict(list)
        total_l2_disentangle_loss_by_modality = collections.defaultdict(list)
        total_l2_latent_loss_by_modality = collections.defaultdict(list)
        total_disentangle_loss_by_modality = collections.defaultdict(list)
        total_var_loss_by_modality = collections.defaultdict(list)
        
        #### batch ####
        model.train()
        for _step, (batch_data, batch_mask, batch_proteins) in enumerate(train_loader):
            # find the protein in the batch that overlap between base and query
            # disentangle will only be computed on these proteins
            batch_proteins_names = [dataset.protein_ids[ii.item()] for ii in batch_proteins]
            batch_protein_in_mod_bool = {}
            batch_protein_in_mod_name = {}
            for mod in mod_str_list:
                protein_bool = []
                anchor_key = f'{mod}-{cond}'
                for ii in np.array(batch_proteins_names):
                    protein_bool.append(
                        (ii in anchor_emb[anchor_key].index.values)
                        if (anchor_key in anchor_emb)
                        else False
                    )
                batch_protein_in_mod_bool[mod] = protein_bool
                batch_protein_in_mod_name[mod] = np.array(batch_proteins_names)[protein_bool]

            #### forward pass ####
            model.train()
            latents, outputs, disentangles = model(batch_data, batch_proteins_names, anchor_emb)
            
            if disentangle_method == "MINE":
                # optimize MINE networks
                for mod in mod_str_list_mine:
                    if not any(batch_protein_in_mod_bool.get(mod, [])):
                        continue
                    z = disentangles[f'{mod}-{cond}'][batch_protein_in_mod_bool[mod]].detach()
                    s = anchor_emb[f'{mod}-{cond}'].loc[batch_protein_in_mod_name[mod]]
                    s = torch.tensor(s.values, dtype=torch.float32).to(device)

                    # mod-cond specific MINE optimizer and update
                    opt = mine_optimizers[f'{mod}-{cond}']
                    model.mines[f'{mod}-{cond}'].train()

                    # If need to train MINE network more (e.g. 5 steps), start here with `for _ in range(5)`:
                    s_shuffle = s[torch.randperm(s.size(0))].to(device)
                    joint_preds = model.mines[f'{mod}-{cond}'](z, s)
                    marginal_preds = model.mines[f'{mod}-{cond}'](z, s_shuffle)

                    opt.zero_grad()
                    joint_mean = torch.mean(joint_preds)
                    marginal_exp_mean = torch.mean(torch.exp(marginal_preds.detach()))
                    ema_key = f'{mod}-{cond}'
                    ema_marginal_exp[ema_key] = (
                        (1 - ema_rate) * ema_marginal_exp[ema_key]
                        + ema_rate * marginal_exp_mean.item()
                    )
                    stable_marginal = torch.log(
                        torch.tensor(ema_marginal_exp[ema_key] + 1e-6, device=device)
                    )
                    loss = -(joint_mean - stable_marginal)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.mines[f'{mod}-{cond}'].parameters(), max_norm=1.0
                    )
                    opt.step()
                    # end of optional `for _ in range(5)`
            
            #### LOSS ####
            # L2 loss on the disentangle
            batch_l2_disentangle_losses = torch.tensor([]).to(device)
            for input_modality in batch_data.keys():
                l2_loss = torch.norm(disentangles[input_modality], p=2, dim=1)
                batch_l2_disentangle_losses = torch.cat(
                    (batch_l2_disentangle_losses, l2_loss)
                )
                total_l2_disentangle_loss_by_modality[input_modality].append(
                    torch.mean(l2_loss).detach().cpu().numpy()
                )
                
            # L2 loss on the latent
            batch_l2_latent_losses = torch.tensor([]).to(device)
            for input_modality in batch_data.keys():
                l2_loss = torch.norm(latents[input_modality], p=2, dim=1)
                batch_l2_latent_losses = torch.cat(
                    (batch_l2_latent_losses, l2_loss)
                )
                total_l2_latent_loss_by_modality[input_modality].append(
                    torch.mean(l2_loss).detach().cpu().numpy()
                )
            
            # variance regularizing loss on the disentangles
            batch_var_losses = torch.tensor([]).to(device)
            for input_modality in batch_data.keys():
                var_loss = variance_regularizer(disentangles[input_modality])
                batch_var_losses = torch.cat(
                    (batch_var_losses, var_loss.unsqueeze(0))
                )
                total_var_loss_by_modality[input_modality].append(
                    torch.mean(var_loss).detach().cpu().numpy()
                )
            
            # The reconstruction loss
            batch_reconstruction_losses = torch.tensor([]).to(device)
            for input_modality in batch_data.keys():
                for output_modality in batch_data.keys():
                    
                    # protein present in both modalities mask
                    mask = (
                        batch_mask[input_modality].bool()
                        & batch_mask[output_modality].bool()
                    )
                    if torch.sum(mask) == 0:
                        continue  # no overlap

                    output_key = input_modality + MODALITY_SEP + output_modality

                    # compare OUTPUT modality original embedding to output embedding
                    pairwise_dist_input_output = 1 - F.cosine_similarity(
                        batch_data[output_modality], outputs[output_key], dim=1
                    )
                    reconstruction_loss = pairwise_dist_input_output[mask]
                    batch_reconstruction_losses = torch.cat(
                        (batch_reconstruction_losses, reconstruction_loss)
                    )
                    total_reconstruction_loss_by_modality[output_key].append(
                        torch.mean(reconstruction_loss).detach().cpu().numpy()
                    )
            
            # The triplet loss on the disentangle
            batch_triplet_disentangle_losses = torch.tensor([]).to(device)
            for anchor_modality in batch_data.keys():
                posneg_modality = random.choice(
                    list([x for x in batch_data.keys() if x != anchor_modality])
                )
                
                # protein present in both modalities mask
                mask_protein_in_both_mod = (
                    batch_mask[anchor_modality].bool()
                    & batch_mask[posneg_modality].bool()
                )
                # if a protein from both modality both have corresponding base context emb
                # then its disentangle are both disentangle, align
                # otherwise cannot align
                anchor_mod = anchor_modality.split("-")[0]
                posneg_mod = posneg_modality.split("-")[0]
                mask_both_mod_has_anchor_cond = (
                    torch.tensor(batch_protein_in_mod_bool[anchor_mod]).bool()
                     & torch.tensor(batch_protein_in_mod_bool[posneg_mod]).bool()
                )
                # combine mask
                mask = mask_protein_in_both_mod & mask_both_mod_has_anchor_cond
                
                # need at least 1 valid protein in the batch to construct a positive
                if torch.sum(mask) == 0:
                    continue
                # if negative sample from batch, then need at least two valid proteins
                if negative_from_batch and (torch.sum(mask) < 2):
                    continue
                
                # triplet anchor from one modality
                # positive point from another modality
                anchor_disentangles = disentangles[anchor_modality]
                positive_disentangles = disentangles[posneg_modality]
                positive_dist = 1 - F.cosine_similarity(
                    anchor_disentangles, positive_disentangles, dim=1
                )

                # for a anchor/positive protein, pick a different protein as negative
                posneg_modality_indices = np.arange(
                    len(data_wrapper.modalities_dict[posneg_modality].train_labels)
                )
                protein_indexes_not_in_batch = list(
                    set(posneg_modality_indices) 
                    - set(batch_proteins)
                )
                negative_indices = random.sample(
                    protein_indexes_not_in_batch, len(positive_dist)
                )
                negative_data = {
                    posneg_modality: data_wrapper.modalities_dict[posneg_modality]
                    .train_features[negative_indices]
                }
                negative_proteins_names = list(
                    np.array(data_wrapper.modalities_dict[posneg_modality]
                    .train_labels)[negative_indices]
                )
                negative_latents_dict, _, negative_disentangles_dict = model(
                    negative_data, negative_proteins_names, anchor_emb
                )
                negative_disentangles = negative_disentangles_dict[posneg_modality]
                negative_dist = 1 - F.cosine_similarity(
                    anchor_disentangles, negative_disentangles, dim=1
                )

                # triplet is max of 0 or positive - negative + margin
                triplet_disentangle_loss = torch.maximum(
                    positive_dist - negative_dist + triplet_margin,
                    torch.zeros(len(positive_dist)).to(device)
                )
                # only valid proteins contribute to the loss
                triplet_disentangle_loss = triplet_disentangle_loss[mask]

                batch_triplet_disentangle_losses = torch.cat(
                    (batch_triplet_disentangle_losses, triplet_disentangle_loss)
                )
                total_triplet_disentangle_loss_by_modality[
                    anchor_modality + MODALITY_SEP + posneg_modality
                ].append(
                    torch.mean(triplet_disentangle_loss).detach().cpu().numpy()
                )
                
            # The disentangle loss
            batch_disentangle_losses = torch.tensor([]).to(device)    
            if disentangle_method == "MINE":
                for mod in mod_str_list_mine:
                    if not any(batch_protein_in_mod_bool.get(mod, [])):
                        continue
                    z = disentangles[f'{mod}-{cond}'][batch_protein_in_mod_bool[mod]]
                    s = anchor_emb[f'{mod}-{cond}'].loc[batch_protein_in_mod_name[mod]]
                    s = torch.tensor(s.values, dtype=torch.float32).to(device)

                    s_shuffle = s[torch.randperm(s.size(0))].to(device)
                    model.mines[f'{mod}-{cond}'].eval()
                    joint_preds = model.mines[f'{mod}-{cond}'](z, s)
                    marginal_preds = model.mines[f'{mod}-{cond}'](z, s_shuffle)
                    joint_mean = torch.mean(joint_preds)
                    marginal_logsumexp = (
                        torch.logsumexp(marginal_preds, dim=0) 
                        - torch.log(
                            torch.tensor(marginal_preds.size(0), dtype=torch.float32).to(device)
                        )
                    )
                    mine_loss = joint_mean - marginal_logsumexp
                    batch_disentangle_losses = torch.cat(
                        (batch_disentangle_losses, mine_loss.unsqueeze(0))
                    )
                    total_disentangle_loss_by_modality[mod].append(
                        torch.mean(mine_loss.unsqueeze(0)).detach().cpu().numpy()
                    )
            else:
                mine_loss = torch.tensor(0, dtype=torch.float32).to(device)
                batch_disentangle_losses = torch.cat(
                    (batch_disentangle_losses, mine_loss.unsqueeze(0))
                )
                total_disentangle_loss_by_modality[disentangle_method].append(
                    torch.mean(mine_loss.unsqueeze(0)).detach().cpu().numpy()
                )
            if len(batch_disentangle_losses) == 0:
                batch_disentangle_losses = torch.tensor([0.0], device=device)
            
            # check valid matches 
            if (
                len(batch_reconstruction_losses) == 0
                or len(batch_triplet_disentangle_losses) == 0
            ):
                continue  # didn't have any overlapping proteins btw modalities, or btw conditions

            #### total loss ####
            if mean_losses:
                reconstruction_loss = torch.mean(batch_reconstruction_losses)
                triplet_disentangle_loss = torch.mean(batch_triplet_disentangle_losses)
                l2_disentangle_loss = torch.mean(batch_l2_disentangle_losses)
                l2_latent_loss = torch.mean(batch_l2_latent_losses)
                disentangle_loss = torch.mean(batch_disentangle_losses)
                var_loss = torch.mean(batch_var_losses)
            else:
                reconstruction_loss = torch.sum(batch_reconstruction_losses)
                triplet_disentangle_loss = torch.sum(batch_triplet_disentangle_losses)
                l2_disentangle_loss = torch.sum(batch_l2_disentangle_losses)
                l2_latent_loss = torch.sum(batch_l2_latent_losses)
                disentangle_loss = torch.sum(batch_disentangle_losses)
                var_loss = torch.sum(batch_var_losses)
            
            if disentangle_method == "subtract":
                lambda_disentangle = 0
                disentangle_loss = torch.tensor(0.0, device=device)
            batch_total_loss = (
                lambda_reconstruction * reconstruction_loss
                + lambda_triplet_disentangle * triplet_disentangle_loss
                + lambda_l2_disentangle * l2_disentangle_loss
                + lambda_l2_latent * l2_latent_loss
                + lambda_disentangle * disentangle_loss
                + lambda_var * var_loss
            )
            
            #### step update ####
            if not warmup_mine:
                main_optimizer.zero_grad()
                batch_total_loss.backward()
                main_optimizer.step()
            else:
                # Warmup epoch: do not update main model parameters.
                pass

            total_loss.append(batch_total_loss.detach().cpu().numpy())
            total_reconstruction_loss.append(reconstruction_loss.detach().cpu().numpy())
            total_triplet_disentangle_loss.append(triplet_disentangle_loss.detach().cpu().numpy())
            total_l2_disentangle_loss.append(l2_disentangle_loss.detach().cpu().numpy())
            total_l2_latent_loss.append(l2_latent_loss.detach().cpu().numpy())
            total_disentangle_loss.append(disentangle_loss.detach().cpu().numpy())
            total_var_loss.append(var_loss.detach().cpu().numpy())

        # log loss for each epoch
        result_string = (
            f"epoch:{epoch}\t"
            f"total_loss:{np.mean(total_loss):03.5f}\t"
            f"reconstruction_loss:{np.mean(total_reconstruction_loss):03.5f}\t"
            f"triplet_disentangle_loss:{np.mean(total_triplet_disentangle_loss):03.5f}\t"
            f"l2_disentangle_loss:{np.mean(total_l2_disentangle_loss):03.5f}\t"
            f"l2_latent_loss:{np.mean(total_l2_latent_loss):03.5f}\t"
            f"disentangle_loss:{np.mean(total_disentangle_loss):03.5f}\t"
            f"var_loss:{np.mean(total_var_loss):03.5f}\t"
        )
        # log per modality loss for each epoch
        for modality, loss in total_reconstruction_loss_by_modality.items():
            result_string += '%s_reconstruction_loss:%03.5f\t' % (modality, np.mean(loss))
        for modality, loss in total_triplet_disentangle_loss_by_modality.items():
            result_string += '%s_triplet_disentangle_loss:%03.5f\t' % (modality, np.mean(loss))
        for modality, loss in total_l2_disentangle_loss_by_modality.items():
            result_string += '%s_l2_disentangle_loss:%03.5f\t' % (modality, np.mean(loss))
        for modality, loss in total_l2_latent_loss_by_modality.items():
            result_string += '%s_l2_latent_loss:%03.5f\t' % (modality, np.mean(loss))
        for modality, loss in total_disentangle_loss_by_modality.items():
            result_string += '%s_disentangle_loss:%03.5f\t' % (modality, np.mean(loss))
        for modality, loss in total_var_loss_by_modality.items():
            result_string += '%s_var_loss:%03.5f\t' % (modality, np.mean(loss))
        print(result_string, file=source_file)
        
        # save result for every N epochs
        if (save_update_epochs) & (epoch % save_epoch == 0):
            save_results(model, dataset, data_wrapper, anchor_emb, results_suffix='_epoch{}'.format(epoch))
    
    # save final results
    embeddings_by_protein = save_results(model, dataset, data_wrapper, anchor_emb)
    source_file.close()
    
    # average embeddings for each protein and return as coemembedding
    for protein, embeddings in embeddings_by_protein.items():
        average_embedding = np.mean(list(embeddings.values()), axis=0)
        row = [protein]
        row.extend(average_embedding)
        yield row
