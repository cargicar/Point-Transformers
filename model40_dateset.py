import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataset import ModelNetDataLoader
from models.Menghao.model import PointTransformerCls
import hydra
import omegaconf
import logging
from pathlib import Path
import os
import argparse


def plot_point_cloud(points):
  """
  Plots a 3D point cloud.

  Args:
    points: A NumPy array of shape (N, 3), where N is the number of points 
            and each row represents the (x, y, z) coordinates of a point.
  """
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1) 
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plt.savefig('plots/points2.png') 


train_dataset = ModelNetDataLoader(root='/home/carlos/Rnet_local/point_transformer/Point-Transformers/modelnet40_normal_resampled', npoint=1024, split='train', normal_channel=True)
test_dataset = ModelNetDataLoader(root='/home/carlos/Rnet_local/point_transformer/Point-Transformers/modelnet40_normal_resampled', npoint=1024, split='test', normal_channel=True)

# Assuming you have loaded the ModelNet40_Normal_Resampled dataset 
# and have access to a single point cloud sample:
#model = torch.load('~/Rnet_local/point_transformer/Point-Transformers/log/cls/best_model.pth')
############################################################


@hydra.main(config_path='config', config_name='cls')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)
    #device =torch.device("cuda:%d" % args.gpu if args.use_gpu else "cpu")
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)
    args.num_class = 40
    args.input_dim = 6 if args.normal else 3
    print(f"args are {args}")

    rdn_int = np.random.randint(10) 
    point_cloud = train_dataset[1][0]  # Assuming dataset[0][0] contains the point cloud data
    point_cloud = test_dataset[1][0]  # Assuming dataset[0][0] contains the point cloud data
    plot_point_cloud(point_cloud)

    points, target = train_dataset[rdn_int]

    model = PointTransformerCls(args)
    checkpoint = torch.load('/home/carlos/Rnet_local/point_transformer/Point-Transformers/log/cls/Menghao/best_model.pth')
    #model.load_state_dict(torch.load('./log/cls/Menghao/best_model.pth', weights_only=True))
    model.load_state_dict(checkpoint["model_state_dict"])
    classifier = model.eval()
    pred = classifier(points)
    pred_choice = pred.data.max(1)[1]
    print(f"target {target}, pred {pred_choice}")


if __name__ == '__main__':
    main()


