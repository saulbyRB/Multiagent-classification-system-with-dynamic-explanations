import matplotlib.pyplot as plt
import networkx as nx

def plot_agent_architecture(agents):
    """
    agents: dict {agent_name: {'type': 'classifier/NN', 'connections': [other_agent_names]}}
    """
    G = nx.DiGraph()
    for agent_name, info in agents.items():
        G.add_node(agent_name, type=info.get('type', 'classifier'))
        for conn in info.get('connections', []):
            G.add_edge(agent_name, conn)
    pos = nx.spring_layout(G)
    node_colors = ['skyblue' if G.nodes[n]['type']=='classifier' else 'lightgreen' for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2000, font_size=10, font_weight='bold', arrowsize=20)
    plt.show()
