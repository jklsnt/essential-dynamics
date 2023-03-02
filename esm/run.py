import torch
import esm
import time

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

# Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
# Lower sizes will have lower memory requirements at the cost of increased speed.
#ATT_SIZE = 4
# model.set_chunk_size(ATT_SIZE)

sequence = "GSPQAIKCVVVGDGAVGKTCLLISYTTNAFPGEYIPTVFDNYSANVMVDGKPVNLGLWDTAGQEDYDRLRPLSYPQTDVSLICFSLVSPASFENVRAKWYPEVRHHCPNTPIILVGTKLDLRDDKDTIEKLKEKKLTPITYPQGLAMAKEIGAVKYLECSALTQRGLKTVFDEAIRAVLCPPPVKK"

# seq = list(sequence)
# seq[178] = "D"
# seq[179] = "M"
# seq[180] = "V"
# seq[181] = "V"
# seq[182] = "V"
# seq = "".join(seq)

# print(seq)
# print(sequence)
# with torch.no_grad():
#     output = model.infer_pdb(seq)

# with open(f"/out/prots/arc.pdb", "w") as f:
#     f.write(output)

# seq = list(sequence)
# seq[0] = "A"
# seq[1] = "A"
# seq = "".join(seq)
# with torch.no_grad():
#     output = model.infer_pdb(seq)

# with open(f"/out/prots/beg_short.pdb", "w") as f:
#     f.write(output)

from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def run_seq(seq, name, replacements=[]):
    sf = open(f"/out/{name}_summary.txt", "w")
    sf.write(f"run {name}\n")
    for i, reps in enumerate(powerset(replacements)):
        if reps == (): continue
        seq = list(seq)
        for rep in reps:
            seq[rep[0]] = rep[1]
        seq = "".join(seq)

        start = time.time()
        with torch.no_grad():
            output = model.infer_pdb(seq)

        with open(f"/out/prots/{name}_{i}.pdb", "w") as f:
            f.write(output)
        print(f"Finished run {i} of {name} in {time.time()-start}s")

        sf.write(f"replacements {reps} are in file {i}\n")

seq = list(sequence)
seq[110] = "V"
seq[111] = "K"
seq[112] = "K"
seq[113] = "D"
seq[114] = "P"
seq[115] = "A"
seq[116] = "M"
seq = "".join(seq)
with torch.no_grad():
    output = model.infer_pdb(seq)

with open(f"/out/prots/small_peptide.pdb", "w") as f:
    f.write(output)

seq = list(sequence)
seq[165] = "A"
seq[166] = "D"
seq[167] = "I"
seq[168] = "M"
seq[169] = "P"
seq[170] = "D"
seq[171] = "L"
seq[172] = "I"
seq[173] = "G"
seq[174] = "K"
seq[175] = "Y"
seq[176] = "S"
seq[177] = "P"
seq = "".join(seq)
with torch.no_grad():
    output = model.infer_pdb(seq)

with open(f"/out/prots/long_peptide.pdb", "w") as f:
    f.write(output)

# run_seq(
#     sequence,
#     "structural_test2",
#     replacements=[
#         (25, "Y"),
#         (34, "D")
#     ]
# )

# # import biotite.structure.io as bsio
# struct = bsio.load_structure("result.pdb", extra_fields=["b_factor"])
# print(struct.b_factor.mean())  # this will be the pLDDT
# # 88.3
