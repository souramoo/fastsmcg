from rdkit import Chem 
from rdkit.Chem import AllChem  
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdDistGeom

import sys
from confgen.molecule.graph import rdk2graph
from confgen.model.gnn import GNN

import torch
from torch_geometric.data import Batch, Data
from torch_sparse import SparseTensor

import re
import pickle

print('import ok!!!')

#load model
checkpoint_file = 'checkpoint_94.pt'
model_param = {
	'mlp_hidden_size': 1024,
	'mlp_layers': 2,
	'latent_size': 256,
	'use_layer_norm': False,
	'num_message_passing_steps': 6,
	'global_reducer': 'sum',
	'node_reducer': 'sum',
	'dropedge_rate': 0.1,
	'dropnode_rate': 0.1,
	'dropout': 0.1,
	'layernorm_before': False,
	'encoder_dropout': 0.0,
	'use_bn': True,
	'vae_beta': 1.0,
	'decoder_layers': None,
	'reuse_prior': True,
	'cycle': 1,
	'pred_pos_residual': True,
	'node_attn': True,
	'global_attn': False,
	'shared_decoder': False,
	'sample_beta': 1.2,
	'shared_output': True
}
model = GNN(**model_param)
#print(model)

checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)['model_state_dict']
print(model.load_state_dict(checkpoint,strict=False))
print('model ok!!!')

#create graph
class CustomData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if isinstance(value, SparseTensor):
            return (0, 1)
        elif bool(re.search("(index|face|nei_tgt_mask)", key)):
            return -1
        return 0


#input : [Dy] has been replaced by C
smiles=[
	 'C#CCOc1ccc(CNc2nc(NCC3CCCN3c3cccnn3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCC3CCCN3c3cccnn3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCC3CCCN3c3cccnn3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCC3CCCN3c3cccnn3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCC3CCCN3c3cccnn3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCC3CCCN3c3cccnn3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCC3CCCN3c3cccnn3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCC3CCCN3c3cccnn3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCC3CCCN3c3cccnn3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCC3CCCN3c3cccnn3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCC3CCCN3c3cccnn3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCC3CCCN3c3cccnn3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCC3CCCN3c3cccnn3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCC3CCCN3c3cccnn3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1',
	 'C#CCOc1ccc(CNc2nc(NCC3CCCN3c3cccnn3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3cc(C(C)(C)C)[nH]n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1ccc(CNc2nc(Nc3nc(Cl)nc4cc(OC)c(OC)cc34)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1',
	 'C#CCOc1cccc(CNc2nc(NCc3nc4ccccc4s3)nc(N[C@@H](CC#C)CC(=O)NC)n2)c1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCC23CCC(C)(CO2)C3)nc(Nc2cccc(S(=O)(=O)NC(C)(C)C)c2)n1',
	 'C#CC[C@@H](CC(=O)NC)Nc1nc(NCc2cccnc2N(C)C)nc(Nc2cnc(Cl)nc2OC)n1',
	 'C#CC[C@@](C)(Nc1nc(NCc2ccccc2N2CCOCC2)nc(NCc2ncccc2N(C)C)n1)C(=O)NC',
	 'CNC(=O)c1cc(OC)c(Nc2nc(NCCC3CC3)nc(NCc3nnc4c(=O)[nH]ccn34)n2)cc1N',
	 'CNC(=O)[C@H](CCCN=[N+]=[N-])Nc1nc(Nc2noc3ccc(F)cc23)nc(Nc2noc3ccc(F)cc23)n1'
]
a = smiles
# make graph
kaggle_mol = [
	Chem.MolFromSmiles(s) for s in a
]
graph = []
for m in kaggle_mol:
	g = rdk2graph(m)
	assert len(g["edge_attr"]) == g["edge_index"].shape[1]
	assert len(g["node_feat"]) == g["num_nodes"]

	data = CustomData()
	data.edge_index = torch.from_numpy(g["edge_index"]).to(torch.int64)
	data.edge_attr = torch.from_numpy(g["edge_attr"]).to(torch.int64)
	data.x = torch.from_numpy(g["node_feat"]).to(torch.int64)
	data.n_nodes = g["n_nodes"]
	data.n_edges = g["n_edges"]
	data.nei_src_index = torch.from_numpy(g["nei_src_index"]).to(torch.int64)
	data.nei_tgt_index = torch.from_numpy(g["nei_tgt_index"]).to(torch.int64)
	data.nei_tgt_mask = torch.from_numpy(g["nei_tgt_mask"]).to(torch.bool)
	# data.pos = torch.from_numpy(mol.GetConformer(0).GetPositions()).to(torch.float)
	# data.isomorphisms = isomorphic_core(mol)
	graph.append(data)

# collate to a batch and send to network
batch_graph = Batch.from_data_list(graph)
print(batch_graph)
print('data ok!!!')

#start conformer generator here !!!!!!!!!!!

num_mol  = len(kaggle_mol)
num_conf = 3 #we generator 3 conformers per mol
device ='cuda:0' #'cuda:0'

model = model.eval()
if device!='cpu':
	model.cuda()

batch_graph = batch_graph.to(device)

def split_prediction(pred,batch_graph):
    split = []
    batch_size = batch_graph.num_graphs
    n_nodes = batch_graph.n_nodes.tolist()
    n = 0
    for i in range(batch_size):
        p = pred[n: n + n_nodes[i]]
        n += n_nodes[i]
        split.append(p)
    return split
import time

predict = [[] for j in range(num_mol)]
for i in range(num_conf):
	with torch.no_grad():
		s_t = time.time()
		out, _ = model(batch_graph, sample=True)
		print('time:', time.time() - s_t, 'batch:', len(graph))

	pos = out[-1]  # 3d xyz position
	pos = split_prediction(pos, batch_graph)

	for j in range(num_mol):
		predict[j].append(pos[j])

print('completed!')
print('smiles[0]', kaggle_smiles[0])
print('estimated xyz pos:')
print('\tconformer 0:')
print(predict[0][0][:5],'...\n')
print('\tconformer 1:')
print(predict[0][1][:5],'...\n')
print('\tconformer 2:')
print(predict[0][2][:5],'...\n')


pickle_file = 'kaggle_mol.pickle'
with open(pickle_file, 'wb') as f:
	pickle.dump(kaggle_mol, f, pickle.HIGHEST_PROTOCOL)

print('infer ok!')