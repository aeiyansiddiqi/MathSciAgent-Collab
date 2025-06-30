import sys
#sys.path.append("c:/Users/Admin/Desktop/Samaneh_Proj/GraphReasoning")


from GraphReasoning.graph_tools import *
from GraphReasoning.utils import *
from GraphReasoning.graph_analysis import *

from PyPDF2 import PdfReader
from glob import glob
import copy
import re
from IPython.display import display, Markdown
import markdown2
import pdfkit
import uuid
import pandas as pd
import numpy as np
import networkx as nx
import os
from pathlib import Path
import random
from pyvis.network import Network
from tqdm.notebook import tqdm
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
palette = "hls"
import pickle
import heapq
import time


from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture
from scipy.spatial import Voronoi, voronoi_plot_2d
import transformers
from transformers import logging
logging.set_verbosity_error()
import requests

import openai
from openai import OpenAI
import base64
from datetime import datetime

# from timm.data import ImageNetInfo
# print(ImageNetInfo)
########################################################################
graph_path = "C:/Users/Admin/Desktop/Samaneh_Proj/data_output_KG_45/graph_root_graphML.graphml"
graph = nx.read_graphml(graph_path)
data_dir="C:/Users/Admin/Desktop/Samaneh_Proj/embbedings/Visualizations"

def analyze_network(G,  data_dir='./', root = 'graph_analysis'):
    # Compute the degrees of the nodes
    # Compute the degrees of the nodes
    degrees = [d for n, d in G.degree()]
    
    # Compute maximum, minimum, and median node degrees
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = np.median(degrees)
    
    # Number of nodes and edges
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # Average node degree
    avg_degree = np.mean(degrees)
    
    # Density of the network
    density = nx.density(G)
    
    # Number of communities (using connected components as a simple community proxy)
    num_communities = nx.number_connected_components(G)
    
    # Print the results
    print(f"Maximum Degree: {max_degree}")
    print(f"Minimum Degree: {min_degree}")
    print(f"Median Degree: {median_degree}")
    print(f"Number of Nodes: {num_nodes}")
    print(f"Number of Edges: {num_edges}")
    print(f"Average Node Degree: {avg_degree:.2f}")
    print(f"Density: {density:.4f}")
    print(f"Number of Communities: {num_communities}")
    
    # Plot the results
    fig, axs = plt.subplots(5, 1, figsize=(10, 15))

    metrics = [
        ('Number of Nodes', num_nodes),
        ('Number of Edges', num_edges),
        ('Avg Node Degree', avg_degree),
        ('Density', density),
        ('Number of Communities', num_communities)
    ]
    
    for ax, (label, value) in zip(axs, metrics):
        ax.barh(label, value, color='blue')
        ax.set_xlim(0, max(value * 1.1, 1.1))  # Adding some padding for better visualization
        ax.set_xlabel('Value')
        ax.set_title(label)
    
    plt.tight_layout()
    plt.savefig(f'{data_dir}/community_structure_{root}.svg')
    # Show the plot
    plt.show()
    
    return max_degree, min_degree, median_degree

def graph_statistics_and_plots(G, data_dir='./'):
    # Calculate statistics
    degrees = [degree for node, degree in G.degree()]
    degree_distribution = np.bincount(degrees)
    average_degree = np.mean(degrees)
    clustering_coefficients = nx.clustering(G)
    average_clustering_coefficient = nx.average_clustering(G)
    triangles = sum(nx.triangles(G).values()) / 3
    connected_components = nx.number_connected_components(G)
    density = nx.density(G)
    
    # Diameter and Average Path Length (for connected graphs or components)
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        average_path_length = nx.average_shortest_path_length(G)
    else:
        diameter = "Graph not connected"
        component_lengths = [nx.average_shortest_path_length(G.subgraph(c)) for c in nx.connected_components(G)]
        average_path_length = np.mean(component_lengths)
    
    # Plot Degree Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1), alpha=0.75, color='blue')
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.savefig(f'{data_dir}/degree_distribution.svg')
    #plt.close()
    plt.show()
    
    # Plot Clustering Coefficient Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(list(clustering_coefficients.values()), bins=10, alpha=0.75, color='green')
    plt.title('Clustering Coefficient Distribution')
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Frequency')
    plt.savefig(f'{data_dir}/clustering_coefficient_distribution.svg')
    plt.show()
    #plt.close()
    
    statistics = {
        'Degree Distribution': degree_distribution,
        'Average Degree': average_degree,
        'Clustering Coefficients': clustering_coefficients,
        'Average Clustering Coefficient': average_clustering_coefficient,
        'Number of Triangles': triangles,
        'Connected Components': connected_components,
        'Diameter': diameter,
        'Density': density,
        'Average Path Length': average_path_length,
    }
    
    return statistics
 
def graph_statistics_and_plots_for_large_graphs (G, data_dir='./', include_centrality=False,
                                                 make_graph_plot=False,root='graph', log_scale=True, 
                                                 log_hist_scale=True,density_opt=False, bins=50,
                                                ):
    # Basic statistics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    degrees = [degree for node, degree in G.degree()]
    log_degrees = np.log1p(degrees)  # Using log1p for a better handle on zero degrees
    #degree_distribution = np.bincount(degrees)
    average_degree = np.mean(degrees)
    density = nx.density(G)
    connected_components = nx.number_connected_components(G)
    
    # Centrality measures
    if include_centrality:
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    # Community detection with Louvain method
    partition = community_louvain.best_partition(G)
    num_communities = len(set(partition.values()))

    # Plotting
    # Degree Distribution on a log-log scale
    plt.figure(figsize=(10, 6))
     
    if log_scale:
        counts, bins, patches = plt.hist(log_degrees, bins=bins, alpha=0.75, color='blue', log=log_hist_scale, density=density_opt)
    
        plt.xscale('log')
        plt.yscale('log')
        xlab_0='Log(1 + Degree)'
        if density_opt:
            ylab_0='Probability Distribution'
        else: 
            ylab_0='Probability Distribution'
        ylab_0=ylab_0 + log_hist_scale*' (log)'    
        
        
        plt_title='Histogram of Log-Transformed Node Degrees with Log-Log Scale'
        
    else:
        counts, bins, patches = plt.hist(degrees, bins=bins, alpha=0.75, color='blue', log=log_hist_scale, density=density_opt)
        xlab_0='Degree'
        if density_opt:
            ylab_0='Probability Distribution'
        else: 
            ylab_0='Probability Distribution'
        ylab_0=ylab_0 + log_hist_scale*' (log)'     
        plt_title='Histogram of Node Degrees'

    plt.title(plt_title)
    plt.xlabel(xlab_0)
    plt.ylabel(ylab_0)
    plt.savefig(f'{data_dir}/{plt_title}_{root}.svg')
    plt.show()
    
    if make_graph_plot:
        
        # Additional Plots
        # Plot community structure
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(G)  # for better visualization
        cmap = plt.get_cmap('viridis')
        nx.draw_networkx(G, pos, node_color=list(partition.values()), node_size=20, cmap=cmap, with_labels=False)
        plt.title('Community Structure')
        plt.savefig(f'{data_dir}/community_structure_{root}.svg')
        plt.show()
        plt.close()

    # Save statistics
    statistics = {
        'Number of Nodes': num_nodes,
        'Number of Edges': num_edges,
        'Average Degree': average_degree,
        'Density': density,
        'Connected Components': connected_components,
        'Number of Communities': num_communities,
        # Centrality measures could be added here as well, but they are often better analyzed separately due to their detailed nature
    }
    if include_centrality:
        centrality = {
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness_centrality,
            'closeness_centrality': closeness_centrality,
            'eigenvector_centrality': eigenvector_centrality,
        }
    else:
        centrality=None
 
    
    return statistics, include_centrality

## Now add these colors to communities and make another dataframe
def colors2Community(communities) -> pd.DataFrame:
    ## Define a color palette
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    df_colors = pd.DataFrame(rows)
    return df_colors


 
def graph_Louvain (G, 
                  graph_GraphML=None, palette = "hls"):
    # Assuming G is your graph and data_dir is defined
    
    # Compute the best partition using the Louvain algorithm
    partition = community_louvain.best_partition(G)
    
    # Organize nodes into communities based on the Louvain partition
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)
    
    communities_list = list(communities.values())
    print("Number of Communities =", len(communities_list))
    print("Communities: ", communities_list)
    
    # Assuming colors2Community can work with the communities_list format
    colors = colors2Community(communities_list)
    print("Colors: ", colors)
    
    # Assign attributes to nodes based on their community membership
    for index, row in colors.iterrows():
        node = row['node']
        G.nodes[node]['group'] = row['group']
        G.nodes[node]['color'] = row['color']
        G.nodes[node]['size'] = G.degree[node]
    
    print("Done, assigned colors and groups...")
    
    # Write the graph with community information to a GraphML file
    if graph_GraphML != None:
        try:
            nx.write_graphml(G, graph_GraphML)
    
            print("Written GraphML.")

        except:
            print ("Error saving GraphML file.")
    return G
    
def save_graph (G, 
                  graph_GraphML=None, ):
    if graph_GraphML != None:
        nx.write_graphml(G, graph_GraphML)
    
        print("Written GraphML")
    else:
        print("Error, no file name provided.")
    return 



if __name__ == "__main__":
   
    # 1) Load your graph
    graph_path = "C:/Users/Admin/Desktop/Samaneh_Proj/data_output_KG_45/graph_root_graphML.graphml"
    G = nx.read_graphml(graph_path)

    # create an output folder
    out_dir = "C:/Users/Admin/Desktop/Samaneh_Proj/statistics/graph_analysis_outputs"
    os.makedirs(out_dir, exist_ok=True)

    # 2) Basic network metrics + bar‑chart
    max_deg, min_deg, med_deg = analyze_network(G, data_dir=out_dir, root="full_graph")

    # 3) Degree & clustering statistics + histograms
    stats = graph_statistics_and_plots(G, data_dir=out_dir)
    print("Graph statistics:", stats)

    # 4) Large‑graph version (with log‑scale, Louvain communities, centrality)
    lg_stats, centrality = graph_statistics_and_plots_for_large_graphs(
        G,
        data_dir=out_dir,
        include_centrality=True,
        make_graph_plot=True,
        root="full_graph",
        log_scale=True,
        log_hist_scale=True,
        density_opt=False,
        bins=50
    )
    print("Large‑graph summary:", lg_stats)

    # 5) Color & save Louvain communities back into GraphML
    colored = graph_Louvain(G, graph_GraphML=os.path.join(out_dir, "graph_louvain_colored.graphml"))
    
    # 6) (Optional) Save final graph again
    save_graph(colored, graph_GraphML=os.path.join(out_dir, "graph_final.graphml"))
    
    print(f"All analysis plots & files written to {out_dir}")