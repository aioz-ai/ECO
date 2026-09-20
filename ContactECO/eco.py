import os
import os.path as osp
import torch
import torch.utils.data
from torch import nn
import numpy as np
import trimesh
import openmesh as om
import posa_utils
import MinkowskiEngine as ME 
from MinkowskiSparseTensor import SparseTensor

# convert Sparse Tensor to torch Tensor
def convert_me_to_torch(x): 
  x,_,_ = x.dense() 
  return x.permute(0,2,1)

# some operators support for Sparse Tensor
def reshape_me_tensor(A, h,w): # reshape (C,H,W) to (C,HW,1)
    # feats = A.F # (CW, H) 
    # new feature of tensor should have shape (C, HW)
    c= convert_me_to_torch(A).shape[0]
    feats = [A.F[i:i+w].mT.reshape(1,-1) for i in range (0,c*w,w) ] 
    feats = torch.cat(feats, dim = 0)
    coords = torch.tensor([[i,0] for i in range(c)]).cuda()
    return ME.SparseTensor(coordinates = coords, 
                           features = feats
                           )
def cat_me_tensor(A,B): 
    if (A.coordinate_map_key != B.coordinate_map_key or A.coordinate_manager != B.coordinate_manager):
      bf = B.F 
      B = ME.SparseTensor(#coordinates = cc,
                            features = bf , 
                            coordinate_manager=A.coordinate_manager,  
                            coordinate_map_key=A.coordinate_map_key )
    return ME.cat(A,B)

def permute_me_tensor(M,  # SparseTensor with shape (C,H,W) 
            h,w): #reshaped to (C,W,H)
    c = convert_me_to_torch(M).shape[0]
    m_F = torch.cat([M.F[i:i+w, :].mT for i in range(0,c*w,w)], dim = 0)  
    m_C = torch.tensor([[i,j] for i in range(c) for j in range(h)]).cuda() 

    return ME.SparseTensor(coordinates=m_C.int(),features=m_F)


def expand_me_tensor(A,w ):
  # assume A has shape (bs,h,1) 
  # expanded output has shape (bs,h,w) 
  bs = convert_me_to_torch(A).shape[0]
  coord = torch.tensor([[i,j] for i in range(bs) for j in range(w)]).cuda()
  feats = A.F 
  feats = [z.expand(w,-1) for z in feats]
  feats = torch.cat(feats, dim=0)
  rst = ME.SparseTensor(coordinates = coord, features = feats)
  return rst

def index_select_me_tensor(X,h,w, # SparseTensor, 
                           dim, indices): 
    c = convert_me_to_torch(X).shape[0]
    _,d = indices.size()
    # expect X has shape (C,H,W) 
    # indices is a tensor with shape (W,D) 
    # -> output will have the shape (C,H,WD) -> (C,HD,W)
    feats = X.F 
    feats = [torch.index_select(feats[i:i+w,:], 0, indices.view(-1)) for i in range(0,c*w,w)] 
    feats = torch.cat(feats,dim = 0) 
    feats = feats.reshape(c*w,h*d)
    coords = X.C
    return SparseTensor(coordinates = coords, features = feats)

class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1):
        super(SpiralConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)   # 9
        self.layer = ME.MinkowskiLinear(in_channels * self.seq_length, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.linear.weight)
        torch.nn.init.constant_(self.layer.linear.bias, 0)

    def forward(self, x):
        # x: (bs, 655, 46), indices: (655, 9)
        n_nodes, d = self.indices.size()
        h = x.F.shape[1]
        x = index_select_me_tensor(x,h,n_nodes,self.dim, self.indices)  # (64, 655 * 9, 46)
        x = self.layer(x)
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)


class GraphLin(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphLin, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer = ME.MinkowskiLinear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.linear.weight)
        torch.nn.init.constant_(self.layer.linear.bias, 0)

    def forward(self, x):
        x = self.layer(x)
        return x


class GraphLin_block(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out=False, non_lin=True):
        super(GraphLin_block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.non_lin = non_lin
        self.drop_out = drop_out
        self.conv = GraphLin(in_channels, out_channels)
        self.normalization = ME.MinkowskiBatchNorm(self.out_channels)
        if self.non_lin:
            self.relu = ME.MinkowskiReLU()
        if self.drop_out:
            self.drop_out_layer = ME.MinkowskiDropout()

    def forward(self, x):
        x = self.conv(x)
        x = self.normalization(x)
        if self.non_lin:
            x = self.relu(x)
        if self.drop_out:
            x = self.drop_out_layer(x)
        return x


class Spiral_block(nn.Module):
    def __init__(self, in_channels, out_channels, indices, non_lin=True):
        # in_channels = 46/64, out_channels = 64, indices: (655, 9).
        super(Spiral_block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = ME.MinkowskiBatchNorm(out_channels)
        self.non_lin = non_lin

        self.conv = SpiralConv( in_channels, out_channels, indices)  # indices: (655, 9) or (164, 9)
        if self.non_lin:
            self.relu = ME.MinkowskiReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.normalization(x)
        if self.non_lin:
            x = self.relu(x)
        return x


class fc_block(nn.Module):
    def __init__(self, in_features, out_features, #normalization_mode=None, 
                 drop_out=False, non_lin=True):
        super(fc_block, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.non_lin = non_lin
        self.drop_out = drop_out
        self.lin = ME.MinkowskiLinear(in_features, out_features)
        self.norm = ME.MinkowskiBatchNorm(out_features)

        if self.non_lin:
            self.relu = ME.MinkowskiReLU()
        if self.drop_out:
            self.drop_out_layer = ME.MinkowskiDropout(0.5)

    def forward(self, x):
        x = self.lin(x)
        x = self.norm(x)
        if self.non_lin:
            x = self.relu(x)
        return x


class ds_us_fn(nn.Module):
    def __init__(self, M):
        super(ds_us_fn, self).__init__()
        
        self.M = M
        self.layer = ME.MinkowskiLinear(M.shape[1], M.shape[0], bias = False) 
        self.layer.requires_grad = True
        self.layer.linear.weight.data = self.M

    def forward(self, x):
        bs = convert_me_to_torch(x).shape[0]
        h = x.F.shape[1] 
        w = x.F.shape[0]//bs
        x = permute_me_tensor(x,h,w)
        x = self.layer(x) 
        x = permute_me_tensor(x,self.M.shape[0],h)
        return x



def load_ds_us_param(ds_us_dir, level, seq_length, use_cuda=True):
    ds_us_dir = osp.abspath(ds_us_dir)
    device = torch.device("cuda" if use_cuda else "cpu")
    level = level + 2  # Start from 2
    # m.vertices: (N, 3), m.faces: (F, 3), m--T_pose, for getting spiral indices.
    m = trimesh.load(osp.join(ds_us_dir, 'mesh_{}.obj'.format(level)), process=False)
    # spiral_indices: (N, 9), nv: N
    spiral_indices = torch.tensor(posa_utils.extract_spirals(om.TriMesh(m.vertices, m.faces), seq_length)).to(device)
    nv = m.vertices.shape[0]
    verts_T_pose = torch.tensor(m.vertices, dtype=torch.float32).to(device)

    # A: adjacency matrix; D: downsampling matrix; U: upsampling matrix
    # A: (N, N); U: (N', N); D: (N, N'); N'>N
    A, U, D = posa_utils.get_graph_params(ds_us_dir, level, use_cuda=use_cuda)
    A = A.to_dense()
    U = U.to_dense()
    D = D.to_dense()
    return nv, spiral_indices, A, U, D, verts_T_pose


class Encoder(nn.Module):
    def __init__(self,h_dim=512, z_dim=256, channels=64, ds_us_dir='./mesh_ds', seq_length=9, no_obj_classes=8, use_cuda=True, **kwargs):
        super(Encoder, self).__init__()

        self.f_dim = no_obj_classes
        self.spiral_indices = []
        self.nv = []
        self.D = []
        levels = [0, 1, 2]
        for level in levels:
            nv, spiral_indices, _, _, D, _ = load_ds_us_param(ds_us_dir, level, seq_length, use_cuda)
            self.nv.append(nv)  # nv: [655, 164, 41]
            self.D.append(D)    # D: [(655, 2619),(164, 655),(41, 164)]
            self.spiral_indices.append(spiral_indices) # spiral_indices: (655->164->41, 9)
        self.channels = (channels * np.ones(4)).astype(int).tolist() # [64, 64, 64, 64]

        self.en_spiral = nn.ModuleList()
        self.en_spiral.append(
            Spiral_block( 3 + self.f_dim, self.channels[0], self.spiral_indices[0]))
        for i in levels:
            self.en_spiral.append(
                Spiral_block( self.channels[i], self.channels[i + 1], self.spiral_indices[i]))
            if i != len(levels) - 1:
                self.en_spiral.append(ds_us_fn( self.D[i + 1]))

        self.en_spiral = nn.Sequential(*self.en_spiral)
        self.en_fc = nn.ModuleList()
        self.en_fc.append(fc_block(self.nv[-1] * self.channels[-1], h_dim))
        self.en_fc = nn.Sequential(*self.en_fc)
        self.en_mu = ME.MinkowskiLinear(h_dim, z_dim)
        self.en_log_var = ME.MinkowskiLinear(h_dim, z_dim)

    def forward(self, x, vertices):
        x = cat_me_tensor(vertices,x)  # (bs,11,655)
        x = self.en_spiral(x)  #(bs,41,64)
        w = self.nv[-1]
        h = self.channels[-1]
        x = reshape_me_tensor(x,h,w)  #(bs, 41*64,1)
        x = self.en_fc(x)   # (bs, 512)

        mu = self.en_mu(x)
        logvar = self.en_log_var(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, z_dim=256, num_hidden_layers=3, channels=64, ds_us_dir='./mesh_ds', seq_length=9, no_obj_classes=8, use_cuda=True, **kwargs):
        super(Decoder, self).__init__()
        self.f_dim = no_obj_classes
        self.spiral_indices = []
        self.nv = []
        levels = [0, 1, 2]
        for level in levels:
            nv, spiral_indices, _, _, _, _ = load_ds_us_param(ds_us_dir, level, seq_length, use_cuda)
            self.nv.append(nv)
            self.spiral_indices.append(spiral_indices)
        self.channels = (channels * np.ones(4)).astype(int).tolist()
        self.de_spiral = nn.ModuleList()
        self.graphin = GraphLin_block(3 + z_dim, z_dim // 2)
        self.de_spiral.append(GraphLin_block(z_dim // 2, self.channels[0]))
        for _ in range(num_hidden_layers):
            self.de_spiral.append(
                Spiral_block(self.channels[0], self.channels[0], self.spiral_indices[0]))
        self.de_spiral.append(SpiralConv(self.channels[0], self.f_dim, self.spiral_indices[0]))
        self.de_spiral = nn.Sequential(*self.de_spiral)

    def forward(self, x, vertices):
        bs = convert_me_to_torch(vertices).shape[0]
        w = vertices.F.shape[0]//bs
        x = expand_me_tensor(x,w)  
        x = cat_me_tensor(vertices,x) 
        x = self.graphin(x)
        x = self.de_spiral(x)
        return x

class ECO(nn.Module):
    def __init__(self, num_mask = 5, sparsity_ratio = 0.5, **kwargs):
        super(ECO, self).__init__()

        # POSA network with Sparse Layer
        self.encoder = Encoder(**kwargs)
        self.decoder = Decoder(**kwargs)
        
        self.sparsity_ratio = sparsity_ratio
        self.num_mask = num_mask
        self.vertices_mask = None # list of masks applied to vertices

        # weigh contr for vertices
        self.vertices_contr = nn.Linear(1,self.num_mask, bias = False) # (num_mask, 1)


    def reparameterize(self, mu, logvar):
        lv_c = logvar.C 
        lv_f = logvar.F 

        std_f = torch.exp(0.5*lv_f) 
        std_c = lv_c 
        std = ME.SparseTensor(coordinates = std_c, 
                              features = std_f,
                              coordinate_manager=mu.coordinate_manager) 
        eps_f = torch.randn_like(std_f)
        eps_c = std_c 
        eps = ME.SparseTensor(coordinates = eps_c, 
                              features = eps_f,
                              coordinate_manager=mu.coordinate_manager) 
        return mu + eps * std

    def forward(self, x, vertices=None):

        # preprocess data: apply mask to dense input -> sparse input
        # vertices: (bs, no_ver, 3)
        mu_list = [] 
        out_list = [] 
        logvar_list = []
        no_ver = vertices.shape[1] # 655
        # init masks
        if self.vertices_mask is None: 
          self.vertices_mask = [np.random.choice(list(range(no_ver)), size = int(no_ver*self.sparsity_ratio), replace = False) for _ in range(self.num_mask)]
        # scale contr of vertices to (0,1)
        w_ver =  nn.functional.softmax(self.vertices_contr.weight.data, dim =0)
        for i in range(self.num_mask): 
            # applied masks to vertices and contacts
            sparse_ver_i = torch.clone(vertices) 
            sparse_ver_i[:,self.vertices_mask[i],:] = 0
            con = torch.clone(x) 
            con[:,self.vertices_mask[i],:] = 0
            # convert sparse vertices and contact to sparse tensors
            ver = w_ver[i][0]*(sparse_ver_i)
            ver = ver.permute(0,2,1) 
            ver = ME.to_sparse_all(ver) 
            con = con.permute(0,2,1)
            con = ME.to_sparse_all(con)

            # forward to POSA model
            muu, lgvr = self.encoder(con, ver)  # mu, logvar = (bs, 256)
            z = self.reparameterize(muu, lgvr)
            outt = self.decoder(z, ver) # out = (bs, n_verts, 8)

            # convert back to torch.Tensor
            outt = convert_me_to_torch(outt)
            muu = convert_me_to_torch(muu)
            lgvr = convert_me_to_torch(lgvr)
            muu = muu.squeeze() 
            lgvr = lgvr.squeeze()
            out_list.append(outt) 
            mu_list.append(muu) 
            logvar_list.append(lgvr) 
        # sum all the results.
        mu = torch.stack(mu_list, dim=0).sum(dim=0)
        out = torch.stack(out_list, dim=0).sum(dim=0)
        logvar = torch.stack(logvar_list, dim=0).sum(dim=0)
        return out, mu, logvar