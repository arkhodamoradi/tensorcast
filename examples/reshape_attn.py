import torch

def manual_reshape_3D(tensor, b3, b2, b1, equation="abc->abc"):
    a3, a2, a1 = tensor.shape
    assert a3 % b3 == 0
    assert a2 % b2 == 0
    assert a1 % b1 == 0
    # get the orders from the equation
    order_from, order_to = equation.split("->")
    A = [a3, a2, a1]
    B = [b3, b2, b1]
    loop_A = [0, 0, 0]
    loop_B = [0, 0, 0]
    for i, o_f in enumerate(order_from):
        for j, o_t in enumerate(order_to):
            if o_f == o_t:
                # loop j should be over dimension i
                loop_A[j] = A[i]
                loop_B[j] = B[i]

    _tensor = torch.zeros(tensor.numel())
    tensor_1d = tensor.reshape(-1)
    _idx = 0
    _idx_1d = 0

    for i3 in range(loop_A[0]//loop_B[0]): # 1357/1=1357
        P3 = i3*loop_B[0]*loop_A[1]*loop_A[2] # move to address

        for i2 in range(loop_A[1]//loop_B[1]): # 1/1=1
            P2 = i2*loop_B[1]*loop_A[2] # move to address

            for i1 in range(loop_A[2]//loop_B[2]): # 1536/64=24
                P1 = i1*loop_B[2] # move to address

                for j3 in range(loop_B[0]): # 1
                    p3 = P3 + P2 + P1 + j3*loop_A[1]*loop_A[2] # move to address

                    for j2 in range(loop_B[1]): # 1
                        p2 = p3 + j2*loop_A[2] # move to address

                        _tensor[_idx : _idx+loop_B[2]] = tensor_1d[p2 : p2+loop_B[2]] # at this address, copy b1 elements from tensor to _tensor
                        _idx += loop_B[2]
    return _tensor

query = torch.randn(1357, 1536)
query_base = query.view(-1, 24, 64).transpose(0, 1)
print(f"query_base shape: {query_base.shape}")
# (1357, 1536) -> (24, 1357, 64)
queryRT = manual_reshape_3D(query.unsqueeze(0), 1, 1357, 64, "abc->bac")
if torch.allclose(query_base.reshape(-1), queryRT):
    print("Manual reshape for RT is correct")
else:
    print("ERROR: Manual reshape for RT is incorrect")
# (24, 1357, 64) -> (1357, 1536)
queryTR = manual_reshape_3D(queryRT.view(24, 1357, 64), 24, 1, 64)
if torch.allclose(query.reshape(-1), queryTR):
    print("Manual reshape for TR is correct")
else:
    print("ERROR: Manual reshape for TR is incorrect")
