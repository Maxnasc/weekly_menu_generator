import matplotlib.pyplot as plt
import networkx as nx
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

def draw_neural_net(model):
    # Pega apenas camadas Dense com units definidas
    layer_sizes = [layer.units for layer in model.layers if isinstance(layer, Dense)]
    
    if not layer_sizes:
        raise ValueError("Nenhuma camada Dense com unidades encontradas. Verifique se o modelo foi corretamente carregado.")

    G = nx.DiGraph()
    pos = {}
    layer_y_gap = 1.5

    for layer_idx, layer_size in enumerate(layer_sizes):
        x_gap = 1.0 / (layer_size + 1)
        for i in range(layer_size):
            node_id = f"L{layer_idx}_N{i}"
            G.add_node(node_id)
            pos[node_id] = (i * x_gap + 0.5, -layer_idx * layer_y_gap)

            if layer_idx > 0:
                for j in range(layer_sizes[layer_idx - 1]):
                    prev_node_id = f"L{layer_idx - 1}_N{j}"
                    G.add_edge(prev_node_id, node_id)

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_size=1000, node_color="skyblue", edge_color='gray')
    plt.title("Topologia da Rede Neural")
    plt.axis('off')
    plt.show()

# Carregar modelo e desenhar
model = load_model('weekly_menu_generator/model/modelo_cardapio.keras')
draw_neural_net(model)
