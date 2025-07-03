import torch
from pointnet_util import index_points, square_distance

# Define the desired size of the tensor
k = 4
line = torch.tensor([[[0, 0, 0],
         [1, 1, 1],
         [2, 2, 2],
         [3, 3, 3],
         [4, 4, 4],
         [5, 5, 5],
         [6, 6, 6],
         [7, 7, 7],
         [8, 8, 8],
         [9, 9, 9]]], dtype=torch.int32)
dist = square_distance(line, line) # dis_{i,j} = dist i to j: d_{0,2}= 2^2+2^2+2^2= 3*4 = 12, and so on 
knn_idx = dist.argsort()[:, :, :k]  # b x n x k = bx(nk_{ij}=indx of closer k points to n_i. In this case: n_{0i}=[0,1,2,3], n_{ij}=[i,i-1,i+1,i-2]
knn_xyz = index_points(line, knn_idx) # b x n x k x 3 = bx(n_{ij}=k points closer to i )x3
#  bx(n_{ij}x3 where n_{ij}= j closest point to i. eg. n_{0j}=[closest, closer, less_close]
in_delta =line[:, :, None] - knn_xyz

print(f"knn_xyz")

