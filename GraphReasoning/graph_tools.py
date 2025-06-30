import sys
import json
sys.path.append("..")  # Go one level up


#from GraphReasoning.graph_tools import *
#from GraphReasoning.utils import *
#from GraphReasoning.graph_analysis import *

from PyPDF2 import PdfReader
from glob import glob
import json
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
from sklearn.metrics.pairwise import cosine_similarity

# from timm.data import ImageNetInfo
# print(ImageNetInfo)


###############################################################
#/home/Aeiyan/testingNougat/SciAgentMath/graphgenerationOutput
#graph_path = "./graphgenerationOutput/graph_root_graphML.graphml"
#graph = nx.read_graphml(graph_path)
#data_dir="C:/Users/Admin/Desktop/Samaneh_Proj/embbedings/Visualizations"


#tokenizer_model = "BAAI/bge-large-en-v1.5"
#embedding_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
#embedding_model = AutoModel.from_pretrained(tokenizer_model)

# Function to generate embeddinggraphs
def generate_node_embeddings(graph, tokenizer, model):
    embeddings = {}
    for node in tqdm(graph.nodes()):
        inputs = tokenizer(str(node), return_tensors="pt")
        outputs = model(**inputs)
        embeddings[node] = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings



def save_embeddings(embeddings, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)
def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings
 
def find_best_fitting_node(keyword, embeddings, tokenizer, model):
    inputs = tokenizer(keyword, return_tensors="pt")
    outputs = model(**inputs)
    keyword_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()  # Flatten to ensure 1-D
    
    # Calculate cosine similarity and find the best match
    best_node = None
    best_similarity = float('-inf')  # Initialize with negative infinity
    for node, embedding in embeddings.items():
        # Ensure embedding is 1-D
        embedding = embedding.flatten()  # Flatten to ensure 1-D
        similarity = 1 - cosine(keyword_embedding, embedding)  # Cosine similarity
        if similarity > best_similarity:
            best_similarity = similarity
            best_node = node
            
    return best_node, best_similarity

def find_best_fitting_node_list(keyword, embeddings, tokenizer, model, N_samples=5):
    inputs = tokenizer(keyword, return_tensors="pt")
    outputs = model(**inputs)
    keyword_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()  # Flatten to ensure 1-D
    
    # Initialize a min-heap
    min_heap = []
    heapq.heapify(min_heap)
    
    for node, embedding in embeddings.items():
        # Ensure embedding is 1-D
        embedding = embedding.flatten()  # Flatten to ensure 1-D
        similarity = 1 - cosine(keyword_embedding, embedding)  # Cosine similarity
        
        # If the heap is smaller than N_samples, just add the current node and similarity
        if len(min_heap) < N_samples:
            heapq.heappush(min_heap, (similarity, node))
        else:
            # If the current similarity is greater than the smallest similarity in the heap
            if similarity > min_heap[0][0]:
                heapq.heappop(min_heap)  # Remove the smallest
                heapq.heappush(min_heap, (similarity, node))  # Add the current node and similarity
                
    # Convert the min-heap to a sorted list in descending order of similarity
    best_nodes = sorted(min_heap, key=lambda x: -x[0])
    
    # Return a list of tuples (node, similarity)
    return [(node, similarity) for similarity, node in best_nodes]


# Example usage
def visualize_embeddings_2d(embeddings , data_dir='./'):
    # Generate embeddings
    #embeddings = generate_node_embeddings(graph, tokenizer, model)
    
    # Extract the embedding vectors
    node_ids = list(embeddings.keys())
    vectors = np.array([embeddings[node].flatten() for node in node_ids])
    
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.5)
    for i, node_id in enumerate(node_ids):
        plt.text(vectors_2d[i, 0], vectors_2d[i, 1], str(node_id), fontsize=9)
    plt.title('Node Embeddings Visualization')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.savefig(f'{data_dir}/node_embeddings_2d.svg')  # Save the figure as SVG
    plt.show()


def visualize_embeddings_2d_notext(embeddings, n_clusters=3, data_dir='./'):
    # Extract the embedding vectors
    node_ids = list(embeddings.keys())
    vectors = np.array([embeddings[node].flatten() for node in node_ids])
    
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vectors)
    labels = kmeans.labels_
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=labels, alpha=0.5, cmap='viridis')
    plt.title('Node Embeddings Visualization')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.savefig(f'{data_dir}/node_embeddings_2d_clusters.svg')  # Save the figure as SVG
    plt.show()


def visualize_embeddings_2d_pretty(embeddings, n_clusters=3,  data_dir='./'):
    # Extract the embedding vectors
    node_ids = list(embeddings.keys())
    vectors = np.array([embeddings[node].flatten() for node in node_ids])
    
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vectors)
    labels = kmeans.labels_
    
    # Count the number of points in each cluster
    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.set(style='whitegrid')  # Set seaborn style for prettier plots
    
    # Use seaborn's color palette and matplotlib's scatter plot
    palette = sns.color_palette("hsv", n_clusters)  # Use a different color palette
    for cluster in range(n_clusters):
        cluster_points = vectors_2d[labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster} (n={cluster_counts[cluster]})', alpha=0.7, edgecolors='w', s=100, cmap=palette)
    
    plt.title('Node Embeddings Visualization with Clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(scatterpoints=1)  # Add a legend to show cluster labels and counts
    plt.savefig(f'{data_dir}/node_embeddings_2d_clusters_pretty.svg')  # Save the figure as SVG
    plt.show()
    
    # Optionally print the counts for each cluster
    for cluster, count in cluster_counts.items():
        print(f'Cluster {cluster}: {count} items')



def visualize_embeddings_2d_pretty_and_sample(embeddings, n_clusters=3, n_samples=5, data_dir='./',
                                             alpha=0.7, edgecolors='none', s=50,):
    # Extract the embedding vectors
    node_ids = list(embeddings.keys())
    vectors = np.array([embeddings[node].flatten() for node in node_ids])
    
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vectors)
    labels = kmeans.labels_
    
    # Count the number of points in each cluster
    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.set(style='whitegrid')  # Set seaborn style for prettier plots
    palette = sns.color_palette("hsv", n_clusters)
    for cluster in range(n_clusters):
        cluster_points = vectors_2d[labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster} (n={cluster_counts[cluster]})'
                    , alpha=alpha, edgecolors=edgecolors, s=s, cmap=palette,#alpha=0.7, edgecolors='w', s=100, cmap=palette)
                   )
    
    plt.title('Node Embeddings Visualization with Clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(scatterpoints=1)
    plt.savefig(f'{data_dir}/node_embeddings_2d_clusters_pretty.svg')
    plt.show()
    
    # Output N_sample terms from the center of each cluster
    centroids = kmeans.cluster_centers_
    for cluster in range(n_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        cluster_vectors = vectors[cluster_indices]
        cluster_node_ids = np.array(node_ids)[cluster_indices]
        
        # Calculate distances of points in this cluster to the centroid
        distances = cdist(cluster_vectors, [centroids[cluster]], 'euclidean').flatten()
        
        # Get indices of N_samples closest points
        closest_indices = np.argsort(distances)[:n_samples]
        closest_node_ids = cluster_node_ids[closest_indices]
        
        print(f'Cluster {cluster}: {len(cluster_vectors)} items')
        print(f'Closest {n_samples} node IDs to centroid:', closest_node_ids)



def visualize_embeddings_with_gmm_density_voronoi_and_print_top_samples(embeddings, n_clusters=5, top_n=3, data_dir='./',s=50):
    # Extract the embedding vectors
    descriptions = list(embeddings.keys())
    node_ids = list(embeddings.keys())
    vectors = np.array([embeddings[node].flatten() for node in node_ids])
    
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # Fit a Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(vectors_2d)
    labels = gmm.predict(vectors_2d)
    
    # Generate Voronoi regions
    vor = Voronoi(gmm.means_)
    
    # Plotting
    plt.figure(figsize=(10, 10))
    # Plot Voronoi diagram
    voronoi_plot_2d(vor, show_vertices=False, show_points=False, line_colors='black', line_width=1, line_alpha=0.7, point_size=2)
    
    # Color points based on their cluster
    for i in range(n_clusters):
        plt.scatter(vectors_2d[labels == i, 0], vectors_2d[labels == i, 1], s=s, label=f'Cluster {i}')
    
    plt.title('Embedding Vectors with GMM Density and Voronoi Tessellation')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.savefig(f'{data_dir}/node_embeddings_2d_clusters_voronoi.svg')
    
    plt.show()
    # Print top-ranked sample texts
    for i in range(n_clusters):
        cluster_center = gmm.means_[i]
        cluster_points = vectors_2d[labels == i]
        
        distances = euclidean_distances(cluster_points, [cluster_center])
        distances = distances.flatten()
        
        closest_indices = np.argsort(distances)[:top_n]
        
        print(f"\nTop {top_n} closest samples to the center of Cluster {i}:")
        for idx in closest_indices:
            original_idx = np.where(labels == i)[0][idx]
            desc = descriptions[original_idx]
            print(f"- Description: {desc}, Distance: {distances[idx]:.2f}")


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



def simplify_graph(graph_, node_embeddings, tokenizer, model, similarity_threshold=0.9, use_llm=False,
                   data_dir_output='./', graph_root='simple_graph', verbatim=False, max_tokens=2048, 
                   temperature=0.3, generate=None):
    """
    Simplifies a graph by merging similar nodes and optionally renaming them using a language model.
    """

    graph = graph_.copy()
    
    nodes = list(node_embeddings.keys())
    embeddings_matrix = np.array([node_embeddings[node].flatten() for node in nodes])

    similarity_matrix = cosine_similarity(embeddings_matrix)
    to_merge = np.where(similarity_matrix > similarity_threshold)

    node_mapping = {}
    nodes_to_recalculate = set()
    merged_nodes = set()  # Keep track of nodes that have been merged
    if verbatim:
        print("Start...")
    for i, j in tqdm(zip(*to_merge), total=len(to_merge[0])):
        if i != j and nodes[i] not in merged_nodes and nodes[j] not in merged_nodes:  # Check for duplicates
            node_i, node_j = nodes[i], nodes[j]
            
            try:
                if graph.degree(node_i) >= graph.degree(node_j):
                #if graph.degree[node_i] >= graph.degree[node_j]:
                    node_to_keep, node_to_merge = node_i, node_j
                else:
                    node_to_keep, node_to_merge = node_j, node_i
    
                if verbatim:
                    print("Node to keep and merge:", node_to_keep, "<--", node_to_merge)
    
                #if use_llm and node_to_keep in nodes_to_recalculate:
                #    node_to_keep = simplify_node_name_with_llm(node_to_keep, max_tokens=max_tokens, temperature=temperature)
    
                node_mapping[node_to_merge] = node_to_keep
                nodes_to_recalculate.add(node_to_keep)
                merged_nodes.add(node_to_merge)  # Mark the merged node to avoid duplicate handling
            except:
                print (end="")
    if verbatim:
        print ("Now relabel. ")
    # Create the simplified graph by relabeling nodes.
    new_graph = nx.relabel_nodes(graph, node_mapping, copy=True)
    if verbatim:
        print ("New graph generated, nodes relabled. ")
    # Recalculate embeddings for nodes that have been merged or renamed.
    recalculated_embeddings = regenerate_node_embeddings(new_graph, nodes_to_recalculate, tokenizer, model)
    if verbatim:
        print ("Relcaulated embeddings... ")
    # Update the embeddings dictionary with the recalculated embeddings.
    updated_embeddings = {**node_embeddings, **recalculated_embeddings}

    # Remove embeddings for nodes that no longer exist in the graph.
    for node in merged_nodes:
        updated_embeddings.pop(node, None)
    if verbatim:
        print ("Now save graph... ")

    # Save the simplified graph to a file.
    graph_path = f'{data_dir_output}/{graph_root}_graphML_simplified.graphml'
    nx.write_graphml(new_graph, graph_path)

    if verbatim:
        print(f"Graph simplified and saved to {graph_path}")

    return new_graph, updated_embeddings

def remove_small_fragents (G_new, size_threshold):
    if size_threshold >0:
        
        # Find all connected components, returned as sets of nodes
        components = list(nx.connected_components(G_new))
        
        # Iterate through components and remove those smaller than the threshold
        for component in components:
            if len(component) < size_threshold:
                # Remove the nodes in small components
                G_new.remove_nodes_from(component)
    return G_new


def update_node_embeddings(embeddings, graph_new, tokenizer, model, remove_embeddings_for_nodes_no_longer_in_graph=True,
                          verbatim=False):
    """
    Update embeddings for new nodes in an updated graph, ensuring that the original embeddings are not altered.

    Args:
    - embeddings (dict): Existing node embeddings.
    - graph_new: The updated graph object.
    - tokenizer: Tokenizer object to tokenize node names.
    - model: Model object to generate embeddings.

    Returns:
    - Updated embeddings dictionary with embeddings for new nodes, without altering the original embeddings.
    """
    # Create a deep copy of the original embeddings
    embeddings_updated = copy.deepcopy(embeddings)
    
    # Iterate through new graph nodes
    for node in tqdm(graph_new.nodes()):
        # Check if the node already has an embedding in the copied dictionary
        if node not in embeddings_updated:
            if verbatim:
                print(f"Generating embedding for new node: {node}")
            inputs = tokenizer(node, return_tensors="pt")
            outputs = model(**inputs)
            # Update the copied embeddings dictionary with the new node's embedding
            embeddings_updated[node] = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    
    if remove_embeddings_for_nodes_no_longer_in_graph:
        # Remove embeddings for nodes that no longer exist in the graph from the copied dictionary
        nodes_in_graph = set(graph_new.nodes())
        for node in list(embeddings_updated):
            if node not in nodes_in_graph:
                if verbatim:
                    print(f"Removing embedding for node no longer in graph: {node}")
                del embeddings_updated[node]

    return embeddings_updated


def make_HTML (G,data_dir='./', graph_root='graph_root'):

    net = Network(
            #notebook=False,
            notebook=True,
            # bgcolor="#1a1a1a",
            cdn_resources="remote",
            height="900px",
            width="100%",
            select_menu=True,
            # font_color="#cccccc",
            filter_menu=False,
        )
        
    net.from_nx(G)
    # net.repulsion(node_distance=150, spring_length=400)
    net.force_atlas_2based(central_gravity=0.015, gravity=-31)
    # net.barnes_hut(gravity=-18100, central_gravity=5.05, spring_length=380)
    
    #net.show_buttons(filter_=["physics"])
    net.show_buttons()
    
    #net.show(graph_output_directory, notebook=False)
    graph_HTML= f'{data_dir}/{graph_root}_graphHTML.html'
    
    net.show(graph_HTML, #notebook=True
            )

    return graph_HTML


def regenerate_node_embeddings(graph, nodes_to_recalculate, tokenizer, model):
    """
    Regenerate embeddings for specific nodes.
    """
    new_embeddings = {}
    for node in tqdm(nodes_to_recalculate):
        inputs = tokenizer(node, return_tensors="pt")
        outputs = model(**inputs)
        new_embeddings[node] = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return new_embeddings


def load_graph_with_text_as_JSON(data_dir='./', graph_name='my_graph.graphml'):
    # Ensure correct path joining
    import os
    fname = os.path.join(data_dir, graph_name)

    G = nx.read_graphml(fname)

    for node, data in tqdm(G.nodes(data=True)):
        for key, value in data.items():
            if isinstance(value, str):  # Only attempt to deserialize strings
                try:
                    data[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass  # If the value is not a valid JSON string, do nothing

    for _, _, data in tqdm(G.edges(data=True)):
        for key, value in data.items():
            if isinstance(value, str):
                try:
                    data[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass

    return G



            

if __name__ == "__main__":
    import networkx as nx
    from transformers import AutoTokenizer, AutoModel
    
    # Load your graph
    graph = nx.read_graphml("./graphgenerationOutput/graph_root_graphML.graphml")
    
    # Load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    
    # Generate and save embeddings
    print("Generating node embeddings...")
    embeddings = generate_node_embeddings(graph, tokenizer, model)


    print("Simplifying graph and regenerating embeddings...")
    simplified_graph, updated_embeddings = simplify_graph(
        graph,
        embeddings,
        tokenizer,
        model,
        similarity_threshold=0.9,
        verbatim=True,
        data_dir_output='./embeddings',
        graph_root='simple_graph'
    )



    save_embeddings(updated_embeddings, "./embeddings/node_embeddings.pkl")
    print("Embeddings saved.")
    
    # Visualize
    # Try other visualization methods one by one:
    #visualize_embeddings_2d_notext(embeddings, n_clusters=4)
    #visualize_embeddings_2d_pretty_and_sample(embeddings, n_clusters=5, n_samples=3)
    #isualize_embeddings_with_gmm_density_voronoi_and_print_top_samples(embeddings, n_clusters=5, top_n=3)
