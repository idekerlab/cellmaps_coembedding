import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

MODALITY_SEP = '___'

class MINE(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(MINE, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim*2, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, 1)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, z, s):
        h = self.LeakyReLU(self.FC_hidden(torch.cat((z, s), 1)))
        h = self.LeakyReLU(self.FC_hidden2(h))
        T = self.FC_output(h)
        return torch.clamp(T, min=-50.0, max=50.0)
    

class CoEmbed(nn.Module):
    def __init__(self, data_wrapper):
        super(CoEmbed, self).__init__()
        self.l2_norm = data_wrapper.l2_norm

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        
        self.disentangle_method = data_wrapper.disentangle_method
        if self.disentangle_method not in {"MINE", "subtract"}:
            raise Exception("disentangle_method should be MINE or subtract")
        if self.disentangle_method == "MINE":
            self.mines = nn.ModuleDict()
        
        for modality_name, modality in data_wrapper.modalities_dict.items():
            self.input_dim = data_wrapper.latent_dim
            
            # set up encoder and decoder for each modality
            encoder = nn.Sequential(
                nn.Dropout(data_wrapper.dropout),
                nn.Linear(modality.input_dim, data_wrapper.hidden_size_1),
                nn.ReLU(),
                nn.Dropout(data_wrapper.dropout),
                nn.Linear(data_wrapper.hidden_size_1, data_wrapper.hidden_size_2),
                nn.ReLU(),
                nn.Linear(data_wrapper.hidden_size_2, data_wrapper.latent_dim)
            )
            self.encoders[modality.name] = encoder
            
            decoder = nn.Sequential(
                nn.Dropout(data_wrapper.dropout),
                nn.Linear(data_wrapper.latent_dim, data_wrapper.hidden_size_2),
                nn.ReLU(),
                nn.Linear(data_wrapper.hidden_size_2, data_wrapper.hidden_size_1),
                nn.ReLU(),
                nn.Linear(data_wrapper.hidden_size_1, modality.input_dim)
            )
            self.decoders[modality.name] = decoder
            
            # the MI estimator network
            # not needed for subtract method
            if self.disentangle_method == "MINE":
                mine = MINE(
                    latent_dim=data_wrapper.latent_dim,
                    hidden_dim=data_wrapper.hidden_size_1)
                self.mines[modality.name] = mine
            

    def forward(self, inputs, proteins, anchor_emb):
        hiddens = dict()
        latents = dict()
        disentangles = dict()
        outputs = dict()
        device = next(self.parameters()).device
        
        # input -> encoder -> hidden
        for modality_name, modality_values in inputs.items():
            hiddens[modality_name] = self.encoders[modality_name](modality_values)
        
        # hidden -> disentangle
        # With MINE, this hidden will be disentangle enforced by the loss function
        if self.disentangle_method != "subtract":
            disentangles = hiddens
        # With simple subtract, the hidden is used to subtract the anchor to get disentangle
        else:
            for modality_name, modality_values in hiddens.items():
                # for each proteins, if exist in anchor, then fetch its anchor emb
                # otherwise use zero emb
                df = anchor_emb[modality_name]
                arr = df.reindex(proteins, fill_value=0).to_numpy(dtype="float32")
                s = torch.from_numpy(arr).to(device)
                
                disentangles[modality_name] = modality_values - s
        
        # disentangle -> latent
        for modality_name, modality_values in disentangles.items():               
            # If L2 normalization, latent is the normalized disentangled
            if self.l2_norm:
                if len(modality_values.shape) > 1:
                    latents[modality_name] = nn.functional.normalize(modality_values, p=2, dim=1)
                else:
                    latents[modality_name] = nn.functional.normalize(modality_values, p=2, dim=0)
            # Otherwise, latent is just disentangled
            else:
                latents[modality_name] = modality_values
        
        # latent -> decoder -> reconstruct
        for modality_name, modality_values in latents.items():
            for output_name, _ in inputs.items():
                out_key = modality_name + MODALITY_SEP + output_name
                outputs[out_key] = self.decoders[output_name](modality_values)

        return latents, outputs, disentangles