import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from berkeley_pacman import layout

def build_graph_from_layout(layout):
    walls = layout.walls  #2D array of booleans
    width = layout.width #Layout width
    height = layout.height #Layout height

    G = nx.Graph()
    
    #We add all nodes for walkable tiles
    for x in range(width):
        for y in range(height):
            if not walls[x][y]: #walkable tiles. true = wall and false = walkable
                G.add_node((x, y)) #Add to graph, should work???

    #We add edges between neighboring tiles (a little hacky)
    for x, y in G.nodes:
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]: 
            nx2, ny2 = x + dx, y + dy
            if (nx2, ny2) in G.nodes:
                G.add_edge((x, y), (nx2, ny2))

    visual_of_nodes_and_edges(G)
    
    return G

def debug_layout(layout_name: str = "originalClassic"):
    
    lay = layout.getLayout(layout_name + ".lay")

    G = build_graph_from_layout(lay)

    print(f" Layout Name: {layout_name}") #Layout name
    print(f" Number of walkable nodes: {G.number_of_nodes()}") #Number of nodes
    print(f" Number of edges: {G.number_of_edges()}") #Number of edges

    for pos in [(1,1),(2,1),(5,5)]:
        if pos in G:
            print(f" Nodes that are adjacent to {pos} are: {list(G.neighbors(pos))}") #Should return;  Nodes that are adjacent to (1, 1): [(2, 1), (1, 2)] and Nodes that are adjacent to (2, 1): [(1, 1), (3, 1)]
        else:
            print(f"  {pos} is a wall") #Should return (5,5) as a wall

    return G

def visual_of_nodes_and_edges(G):
    #nx.draw_spring(G, with_labels=True)
    pos = { (x,y): (x, -y) for (x,y) in G.nodes }  #-y flips vertically for nicer view
    nx.draw(G, pos, with_labels=False, node_size=50)
    plt.gca().set_aspect("equal")   #keep square proportions
    plt.show()

if __name__ == "__main__":
    debug_layout("originalClassic")
