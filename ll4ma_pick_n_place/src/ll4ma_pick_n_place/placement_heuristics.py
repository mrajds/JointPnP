#!/usr/bin/env python3

import torch
import numpy
from gpu_sd_map.transforms_lib import transform_points, vector2mat

class place_heuristics:
    def cost(self, x):
        x = torch.from_numpy(x)
        return self.cost_torch(x).item()

    def grad(self, x):
        x = torch.from_numpy(x)
        x.requires_grad = True
        y = torch.autograd.grad(self.cost_torch(x), x)[0].numpy()
        return y.reshape((1,6))

class corner_heuristic(place_heuristics):
    def __init__(self):
        self.table_corner = torch.zeros(2)
        
    def cost_torch(self, x):
        return torch.linalg.norm(x[:2] - self.table_corner)

    def cost(self, x):
        x = torch.from_numpy(x)
        return self.cost_torch(x).item()

    def grad(self, x):
        x = torch.from_numpy(x)
        x.requires_grad = True
        y = torch.autograd.grad(self.cost_torch(x), x)[0].numpy()
        return y.reshape((1,6))


class position_target(place_heuristics):
    def __init__(self, target, frame_id='world'):
        self.target = torch.tensor(target)
        self.frame_id = frame_id

    def cost_torch(self, x):
        return torch.linalg.norm(x[:3] - self.target[:3])


class packing_heuristic(place_heuristics):
    def __init__(self, env_list, gobj):
        self.min_bnds, self.max_bnds = self.get_bb_dims(env_list)
        #self.odims = torch.tensor(gobj.true_size)
        self.offset, self.odims = self.compute_yaw(gobj)
        print(f"Min BB dims: {self.odims} at {self.offset}")
        self.place_bounds = (self.min_bnds.detach().cpu().numpy(), self.max_bnds.detach().cpu().numpy())
        self.place_bounds[1][1] += 0.25

    def cost_torch(self, x):
        Edim = self.max_bnds - self.min_bnds
        yaw = x[5] - self.offset
        l = self.odims[0]*torch.cos(yaw) + self.odims[1]*torch.sin(yaw)
        l *= 0.5
        l += x[0] - self.min_bnds[0]
        b = self.odims[1]*torch.cos(yaw) + self.odims[0]*torch.sin(yaw)
        b *= 0.5
        b += x[1] - self.min_bnds[1]
        return l*b + torch.maximum(Edim[0], l) * torch.maximum(Edim[1], b)

    def compute_yaw(self, gobj):
        def bb_area(gobj, yaw):
            trans = vector2mat(rpy=[0, 0, yaw])
            pts = torch.from_numpy(transform_points(gobj.obj_pts, trans))
            min_bnds = pts.min(axis=0)[0][:2]
            max_bnds = pts.max(axis=0)[0][:2]
            dim = max_bnds - min_bnds
            return dim.prod(), dim
        yaw = 0
        step = 0.1
        damp = 0.5
        area = bb_area(gobj, yaw)[0]
        yaw += step
        area_ = bb_area(gobj, yaw)[0]
        while area != area_:
            if area_ > area:
                step = -step * damp

            area = area_
            yaw += step
            area_, dim = bb_area(gobj, yaw)
        return yaw, dim
            
        

    def get_bb_dims(self, env_list):
        min_bnds = torch.zeros(2) + torch.inf
        max_bnds = torch.zeros(2) - torch.inf
        for ze in env_list.values():
            pts = torch.from_numpy(transform_points(ze.obj_pts, ze.Trans))
            min_bnds = torch.minimum(min_bnds, pts.min(axis=0)[0][:2])
            max_bnds = torch.maximum(max_bnds, pts.max(axis=0)[0][:2])
        return min_bnds, max_bnds



class y_inline_heuristic(place_heuristics):
    def __init__(self, x):
        self.xt = x
        self.place_bounds = ([self.xt-0.01, 0.1], [self.xt+0.01, 1.0])

    def cost_torch(self, x):
        v = x[0] - self.xt
        return -torch.exp(10*v*v) + x[1]

class minimize_height(place_heuristics):
    def __init__(self, obj_grid, xt, env_list):
        self.odims = obj_grid.get_bounds_at_config()
        self.odims = self.odims[1] - self.odims[0]
        max_ht = 0
        for ze in env_list.values():
            ht = ze.get_bounds_at_config()[1][2]
            max_ht = max(max_ht, ht)
        mindim = self.odims.min()
        ht_bound = max(max_ht + mindim/2, 0.17)
        self.place_bounds = ([xt[0]-0.01, xt[1]-0.1, ht_bound], \
                             [xt[0]+0.01, xt[1]+0.1, ht_bound + 0.01])

    def cost_torch(self, x):
        L = self.odims[0]
        W = self.odims[1]
        H = self.odims[2]
        a = x[3]
        b = x[4]
        g = x[5]
        ca = torch.cos(a)
        sa = torch.sin(a)
        cb = torch.cos(b)
        sb = torch.sin(b)
        cg = torch.cos(g)
        sg = torch.sin(g)

        Lz = -ca*sb*cg + sa*sg
        Lz *= L
        Wz = ca*sb*sg + sa*cg
        Wz *= W
        Hz = ca*cb
        Hz *= H

        Lx = L*cb*cg
        Wx = -W*cb*sg
        Hx = H*sb

        return 10*(Lz+Wz+Hz)**2 + (Lx+Wx+Hx)**2
