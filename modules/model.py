import torch
from torch import nn
import torch.nn.functional as F
    
class TimeBlock(nn.Module):
    """
    Neural network block that applies 3 temporal convolutions to each node of a graph in isolation.
    """

    def __init__(self, n_layers, in_channels, out_channels, kernel_size=5, activ = F.relu, dropout = 0, pool_kernel_size = (1,1)):
        """
        :param in_channels: Number of input features at each node in each time step.
        :param out_channels: Desired number of output channels at each node in each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels, out_channels, (1, kernel_size)))
        for i in range(1, n_layers):
            self.layers.append(nn.Conv2d(out_channels, out_channels, (1, kernel_size)))
            
        self.activ = activ
        self.dropout = nn.Dropout2d(dropout)
        self.pooling = nn.MaxPool2d(pool_kernel_size)
        
        for l in self.layers:
            nn.init.orthogonal_(l.weight)

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_features=in_channels, num_nodes, num_timesteps)
        :return: Output data of shape (batch_size, num_features=out_channels, num_nodes, num_timesteps_out)
        """
        for i in range(len(self.layers)):
            X = self.layers[i](X)
            X = self.activ(X)
            X = self.dropout(X)
        X = self.pooling(X)
                        
        return X
        
class CNN(nn.Module):

    def __init__(self, in_channels, dropout = 0):
        
        super(CNN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(TimeBlock(n_layers = 3, in_channels = in_channels, out_channels = 8, kernel_size = 5, activ = F.relu, dropout = dropout, pool_kernel_size = (1,4)))
        self.layers.append(TimeBlock(n_layers = 3, in_channels = 8, out_channels = 16, kernel_size = 5, activ = F.relu, dropout = dropout, pool_kernel_size = (1,4)))
        self.layers.append(TimeBlock(n_layers = 3, in_channels = 16, out_channels = 32, kernel_size = 5, activ = F.relu, dropout = dropout, pool_kernel_size = (1,4)))
        self.layers.append(TimeBlock(n_layers = 2, in_channels = 32, out_channels = 64, kernel_size = 5, activ = F.relu, dropout = dropout))
        self.layers.append(TimeBlock(n_layers = 1, in_channels = 64, out_channels = 64, kernel_size = 5, activ = F.tanh, dropout = 0, pool_kernel_size = (1,16)))
        
        # Taille du signal
        # 2048 -> 2044 -> 2040 -> 2036 -> 509
        # 509 -> 505 -> 501 -> 497 -> 124
        # 124 -> 120 -> 116 -> 112 -> 28
        # 28 -> 24 -> 20 -> 16 -> 1

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_features=in_channels, num_nodes, 2048)
        :return: Output data of shape (batch_size, num_features=64, num_nodes, 1)
        """
        
        for l in self.layers:
            X = l(X)
            
        return X
        
class GraphConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, adj, activ = F.relu, dropout = 0):
        """
        :param in_channels: Number of input features at each node in each time step.
        :param out_channels: Desired number of output features at each node in each time step.
        :param adj: Normalized adjacency matrix.
        """
        super(GraphConvLayer, self).__init__()

        adj_sym = (adj+torch.t(adj))/2
        D = torch.sum(adj_sym, dim=1).reshape((-1,))
        diag = torch.diag(torch.rsqrt(D))
        norm_adj = torch.matmul(diag, torch.matmul(adj_sym, diag))

        self.adj = norm_adj 
        self.Theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.activ = activ
        self.dropout = nn.Dropout2d(dropout)

        nn.init.orthogonal_(self.Theta)
            

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_features=in_channels, num_nodes, num_timesteps)
        :return: Output data of shape (batch_size, num_features=out_channels, num_nodes, num_timesteps)
        """
        
        lfs = torch.einsum("ij,kmjl->kmil", [self.adj, X])
        X = torch.einsum("kmil,mn->knil", [lfs, self.Theta])
        X = self.activ(X)
        X = self.dropout(X)
        return X
        
        
class DynamicGraphConvLayer(nn.Module):


    def __init__(self, in_channels, out_channels, K, dist, pos, activ = F.relu, dropout = 0, version = 3):
        """
        :param in_channels: Number of input features at each node in each time step.
        :param out_channels: Desired number of output features at each node in each time step.
        :param K: Nearest neighbors hyperparameter
        :param_pos: position of stations (2, num_nodes)
        """
        super(DynamicGraphConvLayer, self).__init__()

        self.version = version

        self.K = K

        self.pos = pos
        self.diff = pos.unsqueeze(1)-pos.unsqueeze(2)                               # Shape (2, num_nodes, num_nodes)
        self.ind_st_D = torch.topk(dist, self.K, largest=False)[1]                  # Shape (num_nodes, K)

        if self.version in [2.5, 2.7, 2.9, 3, 2.55, 2.75]:
            self.D1 = nn.Linear(in_channels*2 + 4, out_channels//2)
            self.D2 = nn.Linear(out_channels//2, out_channels)
            nn.init.orthogonal_(self.D1.weight)
            nn.init.orthogonal_(self.D2.weight)
            if self.version in [2.5, 2.9, 3, 2.55]:
                self.F1 = nn.Linear(in_channels*2, out_channels//2)
                self.F2 = nn.Linear(out_channels//2, out_channels)
                nn.init.orthogonal_(self.F1.weight)
                nn.init.orthogonal_(self.F2.weight)

        elif self.version in [2.2, 2.51, 2.71, 2.8, 2.91, 3.1, 2.25, 2.56, 2.76, 2.85]:
            self.D1 = nn.Linear(in_channels + 2, out_channels//2)
            self.D2 = nn.Linear(out_channels//2, out_channels)
            nn.init.orthogonal_(self.D1.weight)
            nn.init.orthogonal_(self.D2.weight)
            if self.version in [2.51, 2.8, 2.91, 3.1, 2.56, 2.85]:
                self.F1 = nn.Linear(in_channels, out_channels//2)
                self.F2 = nn.Linear(out_channels//2, out_channels)
                nn.init.orthogonal_(self.F1.weight)
                nn.init.orthogonal_(self.F2.weight)
            if self.version in [2.51, 2.71, 2.91, 3.1, 2.56, 2.76]:
                self.D1bis = nn.Linear(in_channels + 2, out_channels//2)
                self.D2bis = nn.Linear(out_channels//2, out_channels)
                nn.init.orthogonal_(self.D1bis.weight)
                nn.init.orthogonal_(self.D2bis.weight)
                if self.version in [2.51, 2.91, 3.1, 2.56]:
                    self.F1bis = nn.Linear(in_channels, out_channels//2)
                    self.F2bis = nn.Linear(out_channels//2, out_channels)
                    nn.init.orthogonal_(self.F1bis.weight)
                    nn.init.orthogonal_(self.F2bis.weight)

        elif self.version in [3.7]:
            self.F1 = nn.Linear(in_channels*2 + 4, out_channels//2)
            self.F2 = nn.Linear(out_channels//2, out_channels)
            nn.init.orthogonal_(self.F1.weight)
            nn.init.orthogonal_(self.F2.weight)

        else:   # 2.01, 2.1, 2.3, 2.4, 2.6
            self.ThetaD = nn.Parameter(torch.FloatTensor(in_channels + 2, out_channels))
            nn.init.orthogonal_(self.ThetaD)
            if self.version in [2.3, 2.6, 2.35, 2.65]:
                self.ThetaF = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
                nn.init.orthogonal_(self.ThetaF)
            if self.version in [2.4, 2.6, 2.45, 2.65]:
                self.ThetaDbis = nn.Parameter(torch.FloatTensor(in_channels + 2, out_channels))
                nn.init.orthogonal_(self.ThetaDbis)
            if self.version in [2.6, 2.65]:
                self.ThetaFbis = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
                nn.init.orthogonal_(self.ThetaFbis)

        self.activ = activ
        self.dropout = nn.Dropout2d(dropout)
        if self.version in [2.1, 2.6, 2.7, 2.71, 2.8, 2.9, 2.91, 3, 3.1, 2.15, 2.65, 2.75, 2.76, 2.85]:
            self.pooling = nn.MaxPool3d((K, 1, 1))
        else:   # 2.01, 2.2, 2.3, 2.4, 2.5, 2.51
            self.pooling = nn.AvgPool3d((K, 1, 1))

    def knn(self, X):
        """
        :param X: Features of shape (batch_size, num_features=in_channels, num_nodes, num_timesteps)
        return tensor of shape (batch_size, num_features=in_channels, num_nodes, num_nodes, num_timesteps)
        """
        
        diff = X.unsqueeze(2)-X.unsqueeze(3)                                            # Shape (batch_size, in_channels, num_nodes, num_nodes, num_timesteps)
        dist = torch.sum(diff**2, (1,4))                                                # Shape (batch_size, num_nodes, num_nodes)
        indices = torch.topk(dist, self.K, largest=False)[1]                          # Shape (batch_size, num_nodes, K)
        return indices, diff

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_features=in_channels, num_nodes, num_timesteps)
        """

        X_F_1 = X.repeat(self.K, 1, 1, 1, 1)                                        # Shape (K, batch_size, in_channels, num_nodes, num_timesteps)
        X_F_1 = X_F_1.permute(1, 3, 0, 4, 2)                                        # Shape (batch_size, num_nodes, K, num_timesteps, in_channels)
        ind_st_F, diffX = self.knn(X)
        ind_st_F = ind_st_F.repeat(X.shape[1], X.shape[3], 1, 1, 1)                 # Shape (in_channels, num_timesteps, batch_size, num_nodes, K)
        ind_st_F = ind_st_F.permute(2, 0, 3, 4, 1)                                  # Shape (batch_size, in_channels, num_nodes, K, num_timesteps)
        X_F_2 = torch.gather(diffX, 3, ind_st_F)                                    # Shape (batch_size, in_channels, num_nodes, K, num_timesteps)
        X_F_2 = X_F_2.permute(0, 2, 3, 4, 1)                                        # Shape (batch_size, num_nodes, K, num_timesteps, in_channels)

        X_F = torch.cat((X_F_1, X_F_2), 4)                                          # Shape (batch_size, num_nodes, K, num_timesteps, 2*in_channels)
        
        Y_F = X_F_1 + X_F_2                                                         # pour les variantes

        Y = self.pos.repeat(X.shape[0], X.shape[3], 1, 1).permute(0, 2, 3, 1)       # Shape (batch_size, 2, num_nodes, num_timesteps)
        X_Y = torch.cat((X, Y), 1)                                                  # Shape (batch_size, in_channels + 2, num_nodes, num_timesteps)
        X_D_1 = X_Y.repeat(self.K, 1, 1, 1, 1)                                      # Shape (K, batch_size, in_channels + 2, num_nodes, num_timesteps)
        X_D_1 = X_D_1.permute(1, 3, 0, 4, 2)                                        # Shape (batch_size, num_nodes, K, num_timesteps, in_channels + 2)
        ind_st_D = self.ind_st_D.repeat(X.shape[0], X_Y.shape[1], X.shape[3], 1, 1) # Shape (batch_size, in_channels + 2, num_timesteps, num_nodes, K)
        ind_st_D = ind_st_D.permute(0, 1, 3, 4, 2)                                  # Shape (batch_size, in_channels + 2, num_nodes, K, num_timesteps)
        diffY = self.diff.repeat(X.shape[0], X.shape[3], 1, 1, 1)                   # Shape (batch_size, num_timesteps, 2, num_nodes, num_nodes)
        diffY = diffY.permute(0, 2, 3, 4, 1)                                        # Shape (batch_size, 2, num_nodes, num_nodes, num_timesteps)
        diff = torch.cat((diffX, diffY), 1)                                         # Shape (batch_size, in_channels + 2, num_nodes, num_nodes, num_timesteps)
        X_D_2 = torch.gather(diff, 3, ind_st_D)                                     # Shape (batch_size, in_channels + 2, num_nodes, K, num_timesteps)
        X_D_2 = X_D_2.permute(0, 2, 3, 4, 1)                                        # Shape (batch_size, num_nodes, K, num_timesteps, in_channels + 2)
        
        X_D = torch.cat((X_D_1, X_D_2), 4)                                          # Shape (batch_size, num_nodes, K, num_timesteps, 2*in_channels + 4)
     
        Y_D = X_D_1 + X_D_2                                                         # pour les variantes

        if self.version in [2.5, 2.7, 2.9, 3, 2.55, 2.75]:
            X_D = self.activ(self.D1(X_D))
            X_D = self.activ(self.D2(X_D))
            if self.version in [2.5, 2.9, 3, 2.55]:
                X_F = self.activ(self.F1(X_F))
                X_F = self.activ(self.F2(X_F))

        elif self.version in [2.2, 2.51, 2.71, 2.91, 3.1, 2.25, 2.56, 2.76]:
            X_D_1 = self.activ(self.D1(X_D_1))
            X_D_1 = self.activ(self.D2(X_D_1))
            X_D = X_D_1

            if self.version in [2.51, 2.91, 3.1, 2.56]:
                X_F_1 = self.activ(self.F1(X_F_1))
                X_F_1 = self.activ(self.F2(X_F_1))
                X_F = X_F_1

            if self.version in [2.51, 2.71, 2.91, 3.1, 2.56, 2.76]:
                X_D_2 = self.activ(self.D1bis(X_D_2))
                X_D_2 = self.activ(self.D2bis(X_D_2))
                X_D = X_D_1 + X_D_2

                if self.version in [2.51, 2.91, 3.1, 2.56]:
                    X_F_2 = self.activ(self.F1bis(X_F_2))
                    X_F_2 = self.activ(self.F2bis(X_F_2))
                    X_F = X_F_1 + X_F_2

        elif self.version in [2.8, 2.85]:
            X_D = self.activ(self.D1(Y_D))
            X_D = self.activ(self.D2(X_D))
            X_F = self.activ(self.F1(Y_F))
            X_F = self.activ(self.F2(X_F))

        elif self.version in [2.4, 2.6, 2.45, 2.65]:   
            X_D_1 = torch.einsum("ij,bnkti->bnktj", [self.ThetaD, X_D_1])
            X_D_2 = torch.einsum("ij,bnkti->bnktj", [self.ThetaDbis, X_D_2])
            X_D = X_D_1 + X_D_2

            if self.version in [2.6, 2.65]:
                X_F_1 = torch.einsum("ij,bnkti->bnktj", [self.ThetaF, X_F_1])
                X_F_2 = torch.einsum("ij,bnkti->bnktj", [self.ThetaFbis, X_F_2])
                X_F = X_F_1 + X_F_2

        elif self.version in [3.7]:

            Y = self.pos.repeat(X.shape[0], X.shape[3], 1, 1).permute(0, 2, 3, 1)       # Shape (batch_size, 2, num_nodes, num_timesteps)
            X_Y = torch.cat((X, Y), 1)                                                  # Shape (batch_size, in_channels + 2, num_nodes, num_timesteps)
            X_F_1 = X_Y.repeat(self.K, 1, 1, 1, 1)                                      # Shape (K, batch_size, in_channels + 2, num_nodes, num_timesteps)
            X_F_1 = X_F_1.permute(1, 3, 0, 4, 2)                                        # Shape (batch_size, num_nodes, K, num_timesteps, in_channels + 2)
            ind_st_F, diffX = self.knn(X)
            ind_st_F = ind_st_F.repeat(X_Y.shape[1], X.shape[3], 1, 1, 1)               # Shape (in_channels + 2, num_timesteps, batch_size, num_nodes, K)
            ind_st_F = ind_st_F.permute(2, 0, 3, 4, 1)                                  # Shape (batch_size, in_channels + 2, num_nodes, K, num_timesteps)
            diffY = self.diff.repeat(X.shape[0], X.shape[3], 1, 1, 1)                   # Shape (batch_size, num_timesteps, 2, num_nodes, num_nodes)
            diffY = diffY.permute(0, 2, 3, 4, 1)                                        # Shape (batch_size, 2, num_nodes, num_nodes, num_timesteps)
            diff = torch.cat((diffX, diffY), 1)                                         # Shape (batch_size, in_channels + 2, num_nodes, num_nodes, num_timesteps)
            X_F_2 = torch.gather(diff, 3, ind_st_F)                                     # Shape (batch_size, in_channels + 2, num_nodes, K, num_timesteps)
            X_F_2 = X_F_2.permute(0, 2, 3, 4, 1)                                        # Shape (batch_size, num_nodes, K, num_timesteps, in_channels + 2)

            X_F = torch.cat((X_F_1, X_F_2), 4)                                          # Shape (batch_size, num_nodes, K, num_timesteps, 2*in_channels + 4)

            X_F = self.activ(self.F1(X_F))
            X_F = self.activ(self.F2(X_F))
            

        else:   # 2.01, 2.1, 2.3
            X_D = torch.einsum("ij,bnkti->bnktj", [self.ThetaD, Y_D])
            if self.version in [2.3, 2.35]:
                X_F = torch.einsum("ij,bnkti->bnktj", [self.ThetaF, Y_F])

        if self.version in [2.3, 2.5, 2.51, 2.6, 2.8, 2.9, 2.91, 3, 3.1, 2.35, 2.55, 2.56, 2.65, 2.85]:
            X = self.pooling(X_D) + self.pooling(X_F)
        elif self.version in [3.7]:
            X = self.pooling(X_F)
        else:   # 2.01, 2.2, 2.4, 2.7, 2.71
            X = self.pooling(X_D)

        if self.version in [2.01, 2.1, 2.3, 2.4, 2.6, 2.06, 2.15, 2.35, 2.45, 2.65]:
            X = self.activ(X)

        X = torch.squeeze(X, dim = 2).permute(0, 3, 1, 2)
        X = self.dropout(X)

        return X
        
class MLP(nn.Module):

    def __init__(self, num_nodes, in_channels, dropout = 0):

        super(MLP, self).__init__()

        self.pooling = nn.MaxPool2d((num_nodes,1))
        self.MLP1 = nn.Linear(in_channels, 128)
        self.MLP2 = nn.Linear(128, 4)
        nn.init.orthogonal_(self.MLP1.weight)
        nn.init.orthogonal_(self.MLP2.weight)
        self.dropout = nn.Dropout(dropout)
            

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, in_channels, num_nodes, 1)
        :return: Output data of shape (batch_size, 4)
        """
        X = torch.squeeze(self.pooling(X))
        X = F.relu(self.MLP1(X))
        X = self.dropout(X)
        X = F.tanh(self.MLP2(X))
        return X
        
        
class GNN1(nn.Module):

    def __init__(self, in_channels, pos, dropout = 0):
        """
        :param in_channels: Number of input features at each node in each time step.
        :param pos: Positions of stations : shape (2, num_nodes)
        """
        
        super(GNN1, self).__init__()
        
        self.pos = pos
        
        self.CNNlayer = CNN(in_channels = in_channels, dropout = dropout)
        self.sblock = TimeBlock(n_layers = 2, in_channels = 66, out_channels = 128, kernel_size = 1, activ = F.relu, dropout = dropout)
        self.MLP = MLP(num_nodes = pos.shape[1], in_channels = 128, dropout = dropout)

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_features=in_channels, num_nodes, 2048)
        :param pos: Positions of stations : shape (2, num_nodes)
        :return: Output data of shape (batch_size, num_features=4)
        """
        
        X = self.CNNlayer(X)                                                    # Shape (batch_size, 64, num_nodes, 1)
        pos = self.pos.repeat(X.shape[0], 1, 1, 1).permute(0, 2, 3, 1)          # Shape (batch_size, 2, num_nodes, 1)
        X = torch.cat((X, pos), 1)                                              # Shape (batch_size, 66, num_nodes, 1)
        X = self.sblock(X)
        X = self.MLP(X)
        return X

class GNN1_depth(nn.Module):

    def __init__(self, in_channels, pos, dropout = 0, concat=False, nlayers = 4):
        """
        :param in_channels: Number of input features at each node in each time step.
        :param pos: Positions of stations : shape (2, num_nodes)
        """

        super(GNN1_depth, self).__init__()

        self.pos = pos

        self.CNNlayer = CNN(in_channels = in_channels, dropout = dropout)
        self.layers = nn.ModuleList()
        for L in range(nlayers):
            self.layers.append(TimeBlock(n_layers = 1, in_channels = 66, out_channels = 64, kernel_size = 1, activ = F.relu, dropout = dropout))

        self.concat = concat
        if concat:
            self.MLP = MLP(num_nodes = pos.shape[1], in_channels = 64 * (nlayers + 1), dropout = dropout)
        else:
            self.MLP = MLP(num_nodes = pos.shape[1], in_channels = 64, dropout = dropout)



    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_features=in_channels, num_nodes, 2048)
        :param pos: Positions of stations : shape (2, num_nodes)
        :return: Output data of shape (batch_size, num_features=4)
        """

        X = self.CNNlayer(X)                                                                    # Shape (batch_size, 64, num_nodes, 1)

        X_layers = [X]
        for L in self.layers:
            pos = self.pos.repeat(X_layers[-1].shape[0], 1, 1, 1).permute(0, 2, 3, 1)           # Shape (batch_size, 2, num_nodes, 1)
            inp = torch.cat((X_layers[-1], pos), 1)                                             # Shape (batch_size, 66, num_nodes, 1)
            X_layers.append(L(inp))

        if self.concat:
            X = torch.cat(X_layers, 1)
            X = self.MLP(X)
        else:
            X = self.MLP(X_layers[-1])

        return X



class GNN2(nn.Module):

    def __init__(self, in_channels, adj, pos, dropout = 0):
        """
        :param in_channels: Number of input features at each node in each time step.
        :param adj: Normalized adjacency matrix.
        :param pos: Positions of stations : shape (2, num_nodes)
        """
        
        super(GNN2, self).__init__()
        
        self.pos = pos
        
        self.CNNlayer = CNN(in_channels = in_channels, dropout = dropout)
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvLayer(in_channels = 66, out_channels = 128, adj = adj, activ = F.relu, dropout = dropout))
        self.layers.append(GraphConvLayer(in_channels = 128, out_channels = 128, adj = adj, activ = F.relu, dropout = dropout))
        self.layers.append(MLP(num_nodes = pos.shape[1], in_channels = 128, dropout = dropout))
        

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_features=in_channels, num_nodes, 2048)
        :return: Output data of shape (batch_size, num_features=4)
        """
        
        X = self.CNNlayer(X)                                                    # Shape (batch_size, 64, num_nodes, 1)
        pos = self.pos.repeat(X.shape[0], 1, 1, 1).permute(0, 2, 3, 1)          # Shape (batch_size, 2, num_nodes, 1)
        X = torch.cat((X, pos), 1)                                              # Shape (batch_size, 66, num_nodes, 1)
        
        for l in self.layers:
            X = l(X)
            
        return X

class GNN2_depth(nn.Module):

    def __init__(self, in_channels, adj, pos, dropout = 0, concat=False, nlayers = 4):
        """
        :param in_channels: Number of input features at each node in each time step.
        :param adj: Normalized adjacency matrix.
        :param pos: Positions of stations : shape (2, num_nodes)
        """

        super(GNN2_depth, self).__init__()

        self.pos = pos

        self.CNNlayer = CNN(in_channels = in_channels, dropout = dropout)
        self.layers = nn.ModuleList()
        for L in range(nlayers):
            self.layers.append(GraphConvLayer(in_channels = 66, out_channels = 64, adj = adj, activ = F.relu, dropout = dropout))

        self.concat = concat
        if concat:
            self.MLP = MLP(num_nodes = pos.shape[1], in_channels = 64 * (nlayers + 1), dropout = dropout)
        else:
            self.MLP = MLP(num_nodes = pos.shape[1], in_channels = 64, dropout = dropout)


    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_features=in_channels, num_nodes, 2048)
        :return: Output data of shape (batch_size, num_features=4)
        """

        X = self.CNNlayer(X)                                                                    # Shape (batch_size, 64, num_nodes, 1)

        X_layers = [X]
        for L in self.layers:
            pos = self.pos.repeat(X_layers[-1].shape[0], 1, 1, 1).permute(0, 2, 3, 1)           # Shape (batch_size, 2, num_nodes, 1)
            inp = torch.cat((X_layers[-1], pos), 1)                                             # Shape (batch_size, 66, num_nodes, 1)
            X_layers.append(L(inp))

        if self.concat:
            X = torch.cat(X_layers, 1)
            X = self.MLP(X)
        else:
            X = self.MLP(X_layers[-1])

        return X

        
class GNN3(nn.Module):

    def __init__(self, in_channels, K, dist, pos, dropout = 0, version = 3, nlayers = 4):
        """
        :param in_channels: Number of input features at each node in each time step.
        :param pos: Positions of stations : shape (2, num_nodes)
        """

        super(GNN3, self).__init__()

        self.version = version
        
        self.CNNlayer = CNN(in_channels = in_channels, dropout = dropout)
        
        self.slayers = nn.ModuleList()
        for L in range(nlayers):
            self.slayers.append(DynamicGraphConvLayer(in_channels = 64, out_channels = 64, K = K, dist = dist, pos = pos, activ = F.relu, dropout = dropout, version = version))
        
        if self.version in [2.01, 2.1, 2.2, 2.3, 2.4, 2.5, 2.51, 2.6, 2.7, 2.71, 2.8, 2.9, 2.91]:
            self.MLP = MLP(num_nodes = pos.shape[1], in_channels = 64, dropout = dropout)
        else:
            self.MLP = MLP(num_nodes = pos.shape[1], in_channels = 64 * (nlayers + 1), dropout = dropout)

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_features=in_channels, num_nodes=N, num_timesteps)
        """
        
        X = self.CNNlayer(X)
        
        X_layers = [X]
        for L in self.slayers:
            X_layers.append(L(X_layers[-1]))
            
        if self.version in [2.01, 2.1, 2.2, 2.3, 2.4, 2.5, 2.51, 2.6, 2.7, 2.71, 2.8, 2.9, 2.91]:
            X = self.MLP(X_layers[-1])
        else:
            X = torch.cat(X_layers, 1)
            X = self.MLP(X)
        
        return X
