import torch
from transformers import EsmForProteinFolding
from faesm.esm import FAEsmForMaskedLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_faesmfold(device):

    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", device_map="cpu").eval()
    model.esm = None
    model.esm = FAEsmForMaskedLM.from_pretrained("facebook/esm2_t36_3B_UR50D", use_fa=True).to(torch.float16).to(device).eval()
    model = model.to(device)
    return model

model = get_faesmfold(device)

protein_name = "1pga_faesmfold.pdb"
sequence = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE" * 20

print(len(sequence))

outputs = model.infer_pdb(sequence)

with open(protein_name, "w") as f:
    f.write(outputs)

