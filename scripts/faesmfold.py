import torch
from transformers import EsmForProteinFolding
from faesm.esm import FAEsmForMaskedLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", device_map="cpu")
model.esm = FAEsmForMaskedLM.from_pretrained("facebook/esm2_t36_3B_UR50D").to(torch.float16)
model = model.to(device).eval()

protein_name = "1pga_faesmfold.pdb"
sequence = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"

outputs = model.infer_pdb(sequence)

with open(protein_name, "w") as f:
    f.write(outputs)
