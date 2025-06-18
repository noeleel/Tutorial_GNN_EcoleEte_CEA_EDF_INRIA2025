import torch
import torch.nn as nn
from graphviz import Digraph
import modules.model as md

def short_label(layer):
    if isinstance(layer, nn.Conv2d):
        return f'Conv2d\n{layer.in_channels}→{layer.out_channels}'
    elif isinstance(layer, nn.MaxPool2d):
        return f'MaxPool2d\n{layer.kernel_size}'
    elif isinstance(layer, nn.Dropout2d):
        return f'Dropout2d\np={layer.p}'
    else:
        return layer.__class__.__name__

def visualize_flat_model(model: nn.Module, output_file="flat_model", view=True):
    dot = Digraph(comment='Flat ML Diagram', format='png')
    dot.attr(rankdir='LR')  # Left to right
    dot.attr('node', shape='box', fontsize='12', style='filled', fillcolor='lightgray')

    node_id = 0
    prev_id = None

    for name, module in model.named_children():
        # Handle ModuleList separately
        if isinstance(module, nn.ModuleList):
            for submod in module:
                label = short_label(submod)
                dot.node(f"{node_id}", label)
                if prev_id is not None:
                    dot.edge(f"{prev_id}", f"{node_id}")
                prev_id = f"{node_id}"
                node_id += 1
        else:
            label = short_label(module)
            dot.node(f"{node_id}", label)
            if prev_id is not None:
                dot.edge(f"{prev_id}", f"{node_id}")
            prev_id = f"{node_id}"
            node_id += 1

    dot.render(output_file, view=view)

if __name__=="__main__":
    visualize_flat_model(md.GNN1(x_train.shape[1], pos, dropout_rate), output_file="../GNN1")
    visualize_flat_model(md.GNN2(x_train.shape[1], adj, pos, dropout_rate), output_file="../GNN2")
    visualize_flat_model(md.GNN3(x_train.shape[1], K, dist, pos, dropout_rate, version, nlayers), output_file="GNN3")
    visualize_flat_model(md.CNN(in_channels = x_train.shape[1], dropout = 0.15), output_file="../CNN")
    visualize_flat_model(md.TimeBlock(n_layers = 2, in_channels = 66, out_channels = 128, kernel_size = 1, activ = F.relu, dropout = 0.15), output_file="../TimeBlock")
    visualize_flat_model(md.GraphConvLayer(in_channels = 66, out_channels = 128, adj = adj, activ = F.relu, dropout = 0.15), output_file="../GraphConvLayer")
    visualize_flat_model(md.DynamicGraphConvLayer(in_channels = 64, out_channels = 64, K = K, dist = dist, pos = pos, activ = F.relu, dropout = 0.15, version = 3), output_file="../DynamicGraphConvLayer")
    visualize_flat_model(md.MLP(num_nodes = pos.shape[1], in_channels = 128, dropout = 0.15), output_file="MLP")