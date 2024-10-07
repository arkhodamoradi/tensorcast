import torch
import argparse

DEBUG = True

DIMS = {
    "o": 0, # OFM
    "i": 1, # IFM
    "r": 2, # ROW
    "c": 3, # COL
}

def print_debug(s, debug):
    if debug:
        print(s)

def depermute(perm):
    """Calculates the inverse permutation pattern."""
    inverse_perm = [None]*len(perm)
    for i, p in enumerate(perm):
        inverse_perm[p] = i
    return inverse_perm

# permute, reshape
parser = argparse.ArgumentParser()
parser.add_argument("--path-str", type=str, help="The path string to reshape the tensor")
args = parser.parse_args()
path_str = args.path_str
print_debug(f"path_str: {path_str}", DEBUG)
divide = path_str.find("_")
shape_str = path_str[:divide]
advancement_str = path_str[divide+1:]
# search for the keys
shape_pos = {"o": shape_str.find("o"), "i": shape_str.find("i"), "r": shape_str.find("r"), "c": shape_str.find("c")}
advancement_pos = {"o": advancement_str.find("o"), "i": advancement_str.find("i"), "r": advancement_str.find("r"), "c": advancement_str.find("c")}
# remove the ones that are not found
shape_pos = {k: v for k, v in shape_pos.items() if v != -1}
advancement_pos = {k: v for k, v in advancement_pos.items() if v != -1}
# sort them
shape_pos = {k: v for k, v in sorted(shape_pos.items(), key=lambda item: item[1])}
advancement_pos = {k: v for k, v in sorted(advancement_pos.items(), key=lambda item: item[1])}
# create the size dictionary
shape_keys = list(shape_pos.keys())
shape_size = {k: shape_str[shape_pos[shape_keys[i]]+1:shape_pos[shape_keys[i+1]]] for i, k in enumerate(shape_keys[:-1])}
shape_size[shape_keys[-1]] = shape_str[shape_pos[shape_keys[-1]]+1:]
# adjust pos values for ordering
shape_pos = {k: i for i, k in enumerate(shape_keys)}
print_debug(f"shape_pos: {shape_pos}", DEBUG)
print_debug(f"shape_size: {shape_size}", DEBUG)
print_debug(f"advancement_pos: {advancement_pos}", DEBUG)

if len(shape_pos.keys()) == 4:
    # create a 4D tesnor
    conv = torch.nn.Conv2d(8, 16, 4) # weight shape: [o, i, r, c]
    tensor_check = conv.weight.data.detach().clone()
    tensor = conv.weight.data.detach().clone()
    shape_org = tensor.shape
    print_debug(f"Tensor shape org: {shape_org}", DEBUG)

elif len(shape_pos.keys()) == 2:
    # create a 4D tesnor
    linear = torch.nn.Linear(8, 16) # weight shape: [o, i, r, c]
    tensor_check = linear.weight.data.detach().clone()
    tensor = linear.weight.data.detach().clone()
    shape_org = tensor.shape
    print_debug(f"Tensor shape org: {shape_org}", DEBUG)

# replace `` with correct size
shape_size = {k: v if v!='' else str(shape_org[DIMS[k]]) for k, v in shape_size.items()}
print_debug(f"reshape sizes: {shape_size}", DEBUG)
sizes = [int(v) for k, v in shape_size.items()]
print_debug(f"sizes: {sizes} -> {list(reversed(sizes))}", DEBUG)
sizes = list(reversed(sizes))

# get the permute pattern
permute = [DIMS[k] for k in shape_keys]
print_debug(f"permute: {permute} -> {list(reversed(permute))}", DEBUG)
permute = list(reversed(permute))

# get the advancement pattern
advancement_in_permute = [permute.index(DIMS[k]) for k in advancement_pos.keys()]
print_debug(f"advancement_in_permute: {advancement_in_permute}", DEBUG) # -> {list(reversed(advancement_in_permute))}")
#advancement_in_permute = list(reversed(advancement_in_permute)) # this is the order of advancement in the permuted tensor

# permute the tensor
tensor = tensor.permute(permute)
shape_after_permute = tensor.shape
print_debug(f"Tensor shape permute: {tensor.shape}", DEBUG)
reshape_pattern = [[i//s, s] for i, s in zip(tensor.shape, sizes)]
reshape_pattern = [i for s in reshape_pattern for i in s]
tensor = tensor.reshape(*reshape_pattern)
print_debug(f"Tensor shape reshape: {tensor.shape}", DEBUG)

# prepare for the advancement pattern
advancement_in_permute_reshaped = [ [i*2, (i*2)+1] for i in advancement_in_permute ]
advancement_in_permute_reshaped = [i for s in advancement_in_permute_reshaped for i in s]
print_debug(f"advancement_in_permute_reshaped: {advancement_in_permute_reshaped}", DEBUG)
tensor = tensor.permute(advancement_in_permute_reshaped)
print_debug(f"Tensor shape advancement permute: {tensor.shape}", DEBUG)

current_shape = tensor.shape
# flatten the tensor
tensor_flattened = tensor.flatten()

# Quantize the tensor_reshaped

# reshape the tensor
tensor = tensor_flattened.reshape(current_shape)

# de-permute the advancement pattern
tensor = tensor.permute(depermute(advancement_in_permute_reshaped))
print_debug(f"Tensor shape advancement de-permute: {tensor.shape}", DEBUG)
# de-reshape to pre-advancement
tensor = tensor.reshape(shape_after_permute)
print_debug(f"Tensor shape de-reshape: {tensor.shape}", DEBUG)
# de-permute the tensor
tensor = tensor.permute(depermute(permute))
print_debug(f"Tensor shape de-permute: {tensor.shape}", DEBUG)
print_debug(f"Tensors are equal: {torch.equal(tensor, tensor_check)}", DEBUG)
