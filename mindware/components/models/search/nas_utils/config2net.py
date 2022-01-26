from mindware.components.models.search.nas_utils.nb1_utils import nasbench1, nasbench1_spec, bench101_opt_choices
from mindware.components.models.search.nas_utils.nb2_utils import nasbench2


def bench101_config2spec(config):
    matrix = []
    vertex = 7
    total_edges = sum(list(range(vertex)))
    active_edges = 0
    e = 0
    while e < total_edges:
        row = [0] * 7
        for i in range(len(matrix) + 1, vertex):
            row[i] = config['edge_%d' % e]
            if config['edge_%d' % e] == 1:
                active_edges += 1
            e += 1
        matrix.append(row)
    matrix.append([0] * 7)
    opt = ['input']
    for v in range(1, vertex - 1):
        opt.append(config['vertex_%d' % v])
    opt.append("output")
    try:
        spec = nasbench1_spec._ToModelSpec(matrix, opt)
        spec.hash_spec(bench101_opt_choices)
    except:
        spec = None
    return spec


def get_net_from_config(space, config, n_classes, **kargs):
    if space == 'nasbench101':
        spec = bench101_config2spec(config)
        net = nasbench1.Network(spec, stem_out=128, num_stacks=3, num_mods=3, num_classes=n_classes)
    elif space == 'nasbench201':
        init_channels = kargs.get('init_channels', 16)
        config = '|%s~0|+|%s~0|%s~1|+|%s~0|%s~1|%s~2|' % (config['op_0'],
                                                          config['op_1'], config['op_2'],
                                                          config['op_3'], config['op_4'], config['op_5'])
        net = nasbench2.get_model_from_arch_str(config, n_classes, init_channels=init_channels)
    else:
        raise ValueError('%s is not supported' % space)
    return net
