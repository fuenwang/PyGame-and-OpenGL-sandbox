import os
import sys
import cv2
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

class Depth2Points(nn.Module):
    def __init__(self, xyz_grid):
        #
        # xyz_grid is [6 x h x w x 3]
        # grid order is ['back', 'down', 'front', 'left', 'right', 'up']

        super(Depth2Points, self).__init__()
        self.xyz_grid = xyz_grid
        self.order = ['back', 'down', 'front', 'left', 'right', 'up']

    def forward(self, x):
        #
        # x is [6*bs x 1 x h x w]
        #
        #


        [bs, c, h, w] = x.size()

        if bs % 6 != 0 or c != 1:
            print("Batch size mismatch in Depth2Points")
            exit()

        bs = bs / 6
        grid = self.xyz_grid
        all_pts = [] 
        for i in range(bs):
            cubemap = x[i*6:(i+1)*6, 0, :, :] # 6 x h x w
            for j, face in enumerate(self.order):
                if face == 'back' or face == 'front':
                    # depth is z axis
                    # cubemap[j, :, :] is h x w
                    # grid[j, :, :, 0] is h x w
                    #
                    scale = cubemap[j, :, :] / torch.abs(grid[j, :, :, 2])
                elif face == 'down' or face == 'up':
                    # depth is y axis
                    scale = cubemap[j, :, :] / torch.abs(grid[j, :, :, 1])
                elif face == 'left' or face == 'right':
                    # depth is x axis
                    scale = cubemap[j, :, :] / torch.abs(grid[j, :, :, 0])
                else:
                    print('Order error in Depth2Points')
                    exit()

                pt_x = (scale * grid[j, :, :, 0]).view(1, h, w, 1)
                pt_y = (scale * grid[j, :, :, 1]).view(1, h, w, 1)
                pt_z = (scale * grid[j, :, :, 2]).view(1, h, w, 1)
                pt = torch.cat([pt_x, pt_y, pt_z], dim=3) 
                all_pts.append(pt)
        point_cloud = torch.cat(all_pts, dim=0)
        #print point_cloud
        return point_cloud

class Equirec2Cube:
    def __init__(self, batch_size, equ_h, equ_w, out_dim, FOV, RADIUS=128, CUDA=True):
        R_lst = []
        theta_lst = np.array([-90, 0, 90, 180], np.float) / 180 * np.pi
        phi_lst = np.array([90, -90], np.float) / 180 * np.pi
        self.equ_h = equ_h
        self.equ_w = equ_w
        self.CUDA = CUDA
        for theta in theta_lst:
            angle_axis = theta * np.array([0, 1, 0], np.float)
            R = cv2.Rodrigues(angle_axis)[0]
            R_lst.append(R)

        for phi in phi_lst:
            angle_axis = phi * np.array([1, 0, 0], np.float)
            R = cv2.Rodrigues(angle_axis)[0]
            R_lst.append(R)
        
        if CUDA:
            R_lst = [Variable(torch.FloatTensor(x)).cuda() for x in R_lst]
        else:
            R_lst = [Variable(torch.FloatTensor(x)) for x in R_lst]
        
        self.out_dim = out_dim
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0
        c_x = (out_dim - 1) / 2.0
        c_y = (out_dim - 1) / 2.0
        
        wangle = (180 - FOV) / 2.0
        w_len = 2 * RADIUS * np.sin(np.radians(FOV / 2.0)) / np.sin(np.radians(wangle))

        f = RADIUS / w_len * out_dim
        cx = c_x
        cy = c_y
        self.intrisic = {
                    'f': f,
                    'cx': cx,
                    'cy': cy
                }

        interval = w_len / (out_dim - 1) 
        
        z_map = np.zeros([out_dim, out_dim], np.float32) + RADIUS
        x_map = np.tile((np.arange(out_dim) - c_x) * interval, [out_dim, 1])
        y_map = np.tile((np.arange(out_dim) - c_y) * interval, [out_dim, 1]).T
        D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
        xyz = np.zeros([out_dim, out_dim, 3], np.float)
        xyz[:, :, 0] = (RADIUS / D) * x_map[:, :]
        xyz[:, :, 1] = (RADIUS / D) * y_map[:, :]
        xyz[:, :, 2] = (RADIUS / D) * z_map[:, :]
        #print xyz[:, :, 0]
        #print xyz[:, :, 1]
        #print np.min(np.linalg.norm(xyz, axis=2))
        #exit()
        if CUDA:
            xyz = Variable(torch.FloatTensor(xyz)).cuda()
        else:
            xyz = Variable(torch.FloatTensor(xyz))
        
        reshape_xyz = xyz.view(out_dim * out_dim, 3).transpose(0, 1)
        self.batch_size = batch_size # NOTE: Might give an error when batch_size smaller than real batch_size of the batch input
        self.loc = []
        self.grid = []
        for i, R in enumerate(R_lst):
            result = torch.matmul(R, reshape_xyz).transpose(0, 1)
            tmp_xyz = result.contiguous().view(1, out_dim, out_dim, 3)
            self.grid.append(tmp_xyz)
            lon = torch.atan2(result[:, 0] , result[:, 2]).view(1, out_dim, out_dim, 1) / np.pi
            lat = torch.asin(result[:, 1] / RADIUS).view(1, out_dim, out_dim, 1) / (np.pi / 2)

            self.loc.append(torch.cat([lon.repeat(batch_size, 1, 1, 1), lat.repeat(batch_size, 1, 1, 1)], dim=3))
    
    def _ToCube(self, batch):
        batch_size = batch.size()[0]
        if batch_size != self.batch_size:
            print('Batch error! Expect to have {} but got {}'.format(self.batch_size, batch_size))

        #lst = ['left', 'front', 'right', 'back', 'up', 'down']
        #lst = ['back', 'down', 'front', 'left', 'right', 'up']
        new_lst = [3, 5, 1, 0, 2, 4]
        out = []
        for i in new_lst:
            coor = self.loc[i]
            result = F.grid_sample(batch, coor[:batch_size, :, :, :])
            out.append(result)
        return out

    def GetGrid(self):
        #lst = ['left', 'front', 'right', 'back', 'up', 'down']
        new_lst = [3, 5, 1, 0, 2, 4]
        out = [self.grid[x] for x in new_lst]
        out = torch.cat(out, dim=0)
        return out


    def ToCubeNumpy(self, batch):
        out = self._ToCube(batch)
        result = [x.data.cpu().numpy() for x in out]
        return result

    def ToCubeTensor(self, batch):
        batch_size = batch.size()[0]
        cube = self._ToCube(batch)
        out_batch = None
        for batch_idx in range(batch_size):
            for cube_idx in range(6):
                patch = torch.unsqueeze(cube[cube_idx][batch_idx, :, :, :], 0)
                if out_batch is None:
                    out_batch = patch
                else:
                    out_batch = torch.cat([out_batch, patch], dim=0)
        #print out_batch
        #exit()
        return out_batch


