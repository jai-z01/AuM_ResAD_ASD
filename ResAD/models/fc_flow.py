from torch import nn
# FrEIA (https://github.com/VLL-HD/FrEIA/)
import FrEIA.framework as Ff
import FrEIA.modules as Fm


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 2*dims_in), nn.ReLU(), nn.Linear(2*dims_in, dims_out))


def flow_model(args, in_channels, **kwargs):
    coder = Ff.SequenceINN(in_channels)
    print('Normalizing Flow => Feature Dimension: ', in_channels)
    coupling_layers = kwargs.get('coupling_layers', None)
    coupling_layers = coupling_layers if coupling_layers is not None else args.coupling_layers
    for k in range(coupling_layers):
        coder.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, affine_clamping=args.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder


def conditional_flow_model(args, in_channels, **kwargs):
    coder = Ff.SequenceINN(in_channels)
    print('Conditional Normalizing Flow => Feature Dimension: ', in_channels)
    coupling_layers = kwargs.get('coupling_layers', None)
    coupling_layers = coupling_layers if coupling_layers is not None else args.coupling_layers
    for k in range(coupling_layers):  # 8
        coder.append(Fm.AllInOneBlock, cond=0, cond_shape=(args.pos_embed_dim,), subnet_constructor=subnet_fc, affine_clamping=args.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder


def load_flow_model(args, in_channels, **kwargs):
    if args.flow_arch == 'flow_model':
        model = flow_model(args, in_channels, **kwargs)
    elif args.flow_arch == 'conditional_flow_model':
        model = conditional_flow_model(args, in_channels, **kwargs)
    else:
        raise NotImplementedError('{} is not supported Normalizing Flow!'.format(args.flow_arch))
    
    return model