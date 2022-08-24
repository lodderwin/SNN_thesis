import graphviz
import warnings
import os
import numpy as np
import networkx as nx
import matplotlib.pylab as plt


# os.environ["PATH"] += os.pathsep + '/Users/erwinlodder/miniforge3/envs/snn_env/lib/python3.8/site-packages/graphviz/'
def draw_net(genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
# def draw_net(input_node_list, output_node_list, list_with_genes, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
            #  node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    # If requested, use a copy of the genome which omits all components that won't affect the output.
    # if prune_unused:
    #     genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)
    # dot.attr(compound='true', rankdir="TB" )
    # dot_top = graphviz.Digraph('subgraph')
    # dot_top.graph_attr.update(rank='min')
    inputs = set()
    for k in [x.id for x in genome.input_neurons]:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style' : 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    # dot.subgraph(dot_top)

    # dot_bottom = graphviz.Digraph('subgraph')
    # dot_bottom.graph_attr.update(rank='max')
    outputs = set()
    for k in [x.id for x in genome.output_neurons]:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    # dot.subgraph(dot_bottom)

    # dot_ middle = graphviz.Digraph('subgraph')
    # dot_middle.graph_attr.update(rank='same')
    # used_nodes = set(genome.nodes.keys())
    used_nodes = list(genome.neurons.keys())
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)
    # dot.subgraph(dot_middle)

    # for cg in genome.connections.values():
    for cg in list(genome.genes.values()):    # for x in list(a.species[1].genomes[0].genes.values())
        if not cg.enabled:
        # if cg.input not in used_nodes or cg.output not in used_nodes:
           continue
        input_node, output_node = cg.input_neuron.id, cg.output_neuron.id
        a = node_names.get(input_node, str(input_node))
        b = node_names.get(output_node, str(output_node))
        style = 'solid' #if cg.enabled else 'dotted'
        color = 'green' #if cg.weight > 0 else 'red'
        width = str(.8) #str(0.1 + abs(cg.weight / 5.0))
        dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)
    # dot.view()
    return dot


def draw_networkx_net(genome):

    locations_input_neurons = np.linspace(0., 1., int(genome.num_input_neurons))
    locations_output_neurons = np.linspace(0., 1., int(genome.num_output_neurons))
    fixed_positions = {}
    for neuron_num in [x.id for x in genome.input_neurons]:
        fixed_positions[neuron_num] = (0, locations_input_neurons[neuron_num-1]) #dict with two of the positions set
    for neuron_num in range(len([x.id for x in genome.output_neurons])):
        fixed_positions[[x.id for x in genome.output_neurons][neuron_num]] = (1, locations_output_neurons[neuron_num])
    
    
    fixed_nodes = fixed_positions.keys()
    pos = nx.spring_layout(genome.networkx_network,pos=fixed_positions, fixed = fixed_nodes)
    
    nx.draw_networkx(genome.networkx_network,pos)
    # axis = plt.gca()
    # axis.set_xlim(0,1)
    # plt.show()