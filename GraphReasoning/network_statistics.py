
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
######################################################
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
model     = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")

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


def simplify_node_name_with_llm(node_name, generate, max_tokens=2048, temperature=0.3):
    # Generate a prompt for the LLM to simplify or describe the node name
    system_prompt='You are an ontological graph maker. You carefully rename nodes in complex networks.'
    prompt = f"Provide a simplified, more descriptive name for a network node named '{node_name}' that reflects its importance or role within a network."
   
    # Assuming 'generate' is a function that calls the LLM with the given prompt
    #simplified_name = generate(system_prompt=system_prompt, prompt)
    simplified_name = generate(system_prompt=system_prompt, prompt=prompt, max_tokens=max_tokens, temperature=temperature)
   
    return simplified_name



def simplify_graph_simple(graph_, node_embeddings, tokenizer, model, similarity_threshold=0.9, use_llm=False,
                  data_dir_output='./',
                  graph_root='simple_graph', verbatim=False,max_tokens=2048, temperature=0.3,generate=None,
                  ):
    graph = graph_.copy()
    nodes = list(node_embeddings.keys())
    embeddings_matrix = np.array([node_embeddings[node].flatten() for node in nodes])

    similarity_matrix = cosine_similarity(embeddings_matrix)
    to_merge = np.where(similarity_matrix > similarity_threshold)

    node_mapping = {}
    nodes_to_recalculate = set()
    for i, j in tqdm(zip(*to_merge)):
        if i != j:  # ignore self-similarity
            node_i, node_j = nodes[i], nodes[j]
            if graph.degree(node_i) >= graph.degree(node_j):
                node_to_keep, node_to_merge = node_i, node_j
            else:
                node_to_keep, node_to_merge = node_j, node_i
            if verbatim:
                print ("node to keep and merge: ",  node_to_keep,"<--",  node_to_merge)
            # Optionally use LLM to generate a simplified or more descriptive name
            if use_llm:
                original_node_to_keep = node_to_keep
                node_to_keep = simplify_node_name_with_llm(node_to_keep, generate, max_tokens=max_tokens, temperature=temperature)
                # Add the original and new node names to the list for recalculation
                nodes_to_recalculate.add(original_node_to_keep)
                nodes_to_recalculate.add(node_to_keep)
            
            node_mapping[node_to_merge] = node_to_keep

    new_graph = nx.relabel_nodes(graph, node_mapping, copy=True)

    # Recalculate embeddings for nodes that have been merged or renamed
    recalculated_embeddings = regenerate_node_embeddings(new_graph, nodes_to_recalculate, tokenizer, model)
    
    # Update the embeddings dictionary with the recalculated embeddings
    updated_embeddings = {**node_embeddings, **recalculated_embeddings}

    # Remove embeddings for nodes that no longer exist
    for node in node_mapping.keys():
        if node in updated_embeddings:
            del updated_embeddings[node]

    graph_GraphML=  f'{data_dir_output}/{graph_root}_graphML_simplified.graphml'  #  f'{data_dir}/resulting_graph.graphml',
        #print (".")
    nx.write_graphml(new_graph, graph_GraphML)
    
    return new_graph, updated_embeddings



# Assuming regenerate_node_embeddings is defined as provided earlier


def simplify_node_name_with_llm(node_name, max_tokens, temperature):
    # This is a placeholder for the actual function that uses a language model
    # to generate a simplified or more descriptive node name.
    return node_name  

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

def return_giant_component_of_graph (G_new ):
    connected_components = sorted(nx.connected_components(G_new), key=len, reverse=True)
    G_new = G_new.subgraph(connected_components[0]).copy()
    return G_new 
    
def return_giant_component_G_and_embeddings (G_new, node_embeddings):
    connected_components = sorted(nx.connected_components(G_new), key=len, reverse=True)
    G_new = G_new.subgraph(connected_components[0]).copy()
    node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model, verbatim=False)
    return G_new, node_embeddings

def extract_number(filename):
    # This function extracts numbers from a filename and converts them to an integer.
    # It finds all sequences of digits in the filename and returns the first one as an integer.
    # If no number is found, it returns -1.
    match = re.search(r'(\d+)', filename)
    return int(match.group(0)) if match else -1
 
def get_list_of_graphs_and_chunks (graph_q='graph_*_graph_clean.csv',  chunk_q='graph_*_chunks_clean.csv', data_dir='./',verbatim=False):
    graph_pattern = os.path.join(data_dir, graph_q)
    chunk_pattern = os.path.join(data_dir, chunk_q)
    
    # Use glob to find all files matching the patterns
    graph_files = glob.glob(graph_pattern)
    chunk_files = glob.glob(chunk_pattern)
    
    # Sort the files using the custom key function
    graph_file_list = sorted(graph_files, key=extract_number)
    chunk_file_list = sorted(chunk_files, key=extract_number)

    if verbatim:
        # Print the lists to verify
        print ('\n'.join(graph_file_list[:10]), '\n\n', '\n'.join(chunk_file_list[:10]),'\n')
        
        print('# graph files:', len (graph_file_list))
        print('# chunk files:', len (chunk_file_list))
    
    return graph_file_list, chunk_file_list

def print_graph_nodes_with_texts(G, separator="; ", N=64):
    """
    Prints out each node in the graph along with the associated texts, concatenated into a single string.

    Parameters:
    - G: A NetworkX graph object where each node has a 'texts' attribute containing a list of texts.
    - separator: A string separator used to join texts. Default is "; ".
    """
    print("Graph Nodes and Their Associated Texts (Concatenated):")
    for node, data in G.nodes(data=True):
        texts = data.get('texts', [])
        concatenated_texts = separator.join(texts)
        print(f"Node: {node}, Texts: {concatenated_texts[:N]}")      
       
def print_graph_nodes (G, separator="; ", N=64):
    """
    Prints out each node in the graph along with the associated texts, concatenated into a single string.

    Parameters:
    - G: A NetworkX graph object where each node has a 'texts' attribute containing a list of texts.
    - separator: A string separator used to join texts. Default is "; ".
    """
    i=0
    print("Graph Nodes and Their Associated Texts (Concatenated):")
    for node in G.nodes :
        print(f"Node {i}: {node}")  
        i=i+1
def get_text_associated_with_node(G, node_identifier ='bone', ):
        
    # Accessing and printing the 'texts' attribute for the node
    if 'texts' in G.nodes[node_identifier]:
        texts = G.nodes[node_identifier]['texts']
        concatenated_texts = "; ".join(texts)  # Assuming you want to concatenate the texts
        print(f"Texts associated with node '{node_identifier}': {concatenated_texts}")
    else:
        print(f"No 'texts' attribute found for node {node_identifier}")
        concatenated_texts=''
    return concatenated_texts 

import networkx as nx
import json
from copy import deepcopy
from tqdm import tqdm

def save_graph_with_text_as_JSON(G_or, data_dir='./', graph_name='my_graph.graphml'):
    G = deepcopy(G_or)

    # Ensure correct path joining
    import os
    fname = os.path.join(data_dir, graph_name)

    for _, data in tqdm(G.nodes(data=True)):
        for key in data:
            if isinstance(data[key], (list, dict, set, tuple)):  # Extend this as needed
                data[key] = json.dumps(data[key])

    for _, _, data in tqdm(G.edges(data=True)):
        for key in data:
            if isinstance(data[key], (list, dict, set, tuple)):  # Extend this as needed
                data[key] = json.dumps(data[key])

    nx.write_graphml(G, fname)
    return fname

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


def save_graph_without_text(G_or, data_dir='./', graph_name='my_graph.graphml'):
    G = deepcopy(G_or)

    # Process nodes: remove 'texts' attribute and convert others to string
    for _, data in tqdm(G.nodes(data=True), desc="Processing nodes"):
        if 'texts' in data:
            del data['texts']  # Remove the 'texts' attribute
        # Convert all other attributes to strings
        for key in data:
            data[key] = str(data[key])

    # Process edges: similar approach, remove 'texts' and convert attributes
    for i, (_, _, data) in enumerate(tqdm(G.edges(data=True), desc="Processing edges")):
    #for _, _, data in tqdm(G.edges(data=True), desc="Processing edges"):
        data['id'] = str(i)  # Assign a unique ID
        if 'texts' in data:
            del data['texts']  # Remove the 'texts' attribute
        # Convert all other attributes to strings
        for key in data:
            data[key] = str(data[key])
    
    # Ensure correct directory path and file name handling
    fname = os.path.join(data_dir, graph_name)
    
    # Save the graph to a GraphML file
    nx.write_graphml(G, fname, edge_id_from_attribute='id')
    return fname

def print_nodes_and_labels (G, N=10):
    # Printing out the first 10 nodes
    ch_list=[]
    
    print("First 10 nodes:")
    for node in list(G.nodes())[:10]:
        print(node)
    
    print("\nFirst 10 edges with titles:")
    for (node1, node2, data) in list(G.edges(data=True))[:10]:
        edge_title = data.get('title')  # Replace 'title' with the attribute key you're interested in
        ch=f"Node labels: ({node1}, {node2}) - Title: {edge_title}"
        ch_list.append (ch)
        
        print (ch)
        

    return ch_list



def make_graph_from_text_withtext(graph_file_list, chunk_file_list,
                                  include_contextual_proximity=False,
                                  graph_root='graph_root',
                                  repeat_refine=0, verbatim=False,
                                  data_dir='./data_output_KG/',
                                  save_PDF=False, save_HTML=True, N_max=10,
                                  idx_start=0):
    """
    Constructs a graph from text data, ensuring edge labels do not incorrectly include node names.
    """

    # Initialize an empty DataFrame to store all texts
    all_texts_df = pd.DataFrame()

    # Initialize an empty graph
    G_total = nx.Graph()

    for idx in tqdm(range(idx_start, min(len(graph_file_list), N_max)), desc="Processing graphs"):
        try:
            # Load graph and chunk data
            graph_df = pd.read_csv(graph_file_list[idx])
            text_df = pd.read_csv(chunk_file_list[idx])
            
            # Append the current text_df to the all_texts_df
            all_texts_df = pd.concat([all_texts_df, text_df], ignore_index=True)
    
            # Clean and aggregate the graph data
            graph_df.replace("", np.nan, inplace=True)
            graph_df.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
            graph_df['count'] = 4  # Example fixed count, adjust as necessary
            
            # Aggregate edges and combine attributes
            graph_df = (graph_df.groupby(["node_1", "node_2"])
                        .agg({"chunk_id": ",".join, "edge": ','.join, 'count': 'sum'})
                        .reset_index())
            
            if verbatim:
                print("Shape of graph DataFrame: ", graph_df.shape)
    
            # Add edges to the graph
            for _, row in graph_df.iterrows():
                G_total.add_edge(row['node_1'], row['node_2'], chunk_id=row['chunk_id'],
                                 title=row['edge'], weight=row['count'] / 4)
    
        except Exception as e:
            print(f"Error in graph generation for idx={idx}: {e}")
   
    # Ensure no duplicate chunk_id entries
    all_texts_df = all_texts_df.drop_duplicates(subset=['chunk_id'])
    
    # Map chunk_id to text
    chunk_id_to_text = pd.Series(all_texts_df.text.values, index=all_texts_df.chunk_id).to_dict()

    # Initialize node texts collection
    node_texts = {node: set() for node in G_total.nodes()}

    # Associate texts with nodes based on edges
    for (node1, node2, data) in tqdm(G_total.edges(data=True), desc="Mapping texts to nodes"):
        chunk_ids = data.get('chunk_id', '').split(',')
        for chunk_id in chunk_ids:
            text = chunk_id_to_text.get(chunk_id, "")
            if text:  # If text is found for the chunk_id
                node_texts[node1].add(text)
                node_texts[node2].add(text)

    # Update nodes with their texts
    for node, texts in node_texts.items():
        G_total.nodes[node]['texts'] = list(texts)  # Convert from set to list

    return G_total
import numpy as np
from tqdm import tqdm
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

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
    
def simplify_graph_with_text(graph_, node_embeddings, tokenizer, model, similarity_threshold=0.9, use_llm=False,
                   data_dir_output='./', graph_root='simple_graph', verbatim=False, max_tokens=2048, 
                   temperature=0.3, generate=None):
    """
    Simplifies a graph by merging similar nodes and optionally renaming them using a language model.
    Also, merges 'texts' node attribute ensuring no duplicates.
    """

    graph = deepcopy(graph_)
    
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
                    node_to_keep, node_to_merge = node_i, node_j
                else:
                    node_to_keep, node_to_merge = node_j, node_i
    
                # Handle 'texts' attribute by merging and removing duplicates
                texts_to_keep = set(graph.nodes[node_to_keep].get('texts', []))
                texts_to_merge = set(graph.nodes[node_to_merge].get('texts', []))
                merged_texts = list(texts_to_keep.union(texts_to_merge))
                graph.nodes[node_to_keep]['texts'] = merged_texts
    
                if verbatim:
                    print("Node to keep and merge:", node_to_keep, "<--", node_to_merge)
    
                node_mapping[node_to_merge] = node_to_keep
                nodes_to_recalculate.add(node_to_keep)
                merged_nodes.add(node_to_merge)  # Mark the merged node to avoid duplicate handling
            except Exception as e:
                print("Error during merging:", e)
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
    if verbatim:
        print ("Done recalculate embeddings... ")
    
    # Remove embeddings for nodes that no longer exist in the graph.
    for node in merged_nodes:
        updated_embeddings.pop(node, None)
    if verbatim:
        print ("Now save graph... ")

    # Save the simplified graph to a file.
    graph_path = f'{graph_root}_graphML_simplified_JSON.graphml'
    save_graph_with_text_as_JSON (new_graph, data_dir=data_dir_output, graph_name=graph_path)
    
    if verbatim:
        print(f"Graph simplified and saved to {graph_path}")

    return new_graph, updated_embeddings


if __name__=="__main__":
    import os
    import networkx as nx
    from transformers import AutoTokenizer, AutoModel
    # … (other imports, community_louvain, etc.) …

    # 1) Load your graph
    graph_path = "C:/Users/Admin/Desktop/Samaneh_Proj/data_output_KG_45/graph_root_graphML.graphml"
    G = nx.read_graphml(graph_path)

    # 2) Load your saved embeddings from disk
    emb_path = r"C:\Users\Admin\Desktop\Samaneh_Proj\embbedings\node_embeddings.pkl"
    if os.path.exists(emb_path):
        embeddings = load_embeddings(emb_path)
        print(f"Loaded {len(embeddings)} embeddings from {emb_path}")
    else:
        # fallback: generate & save
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        model     = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
        embeddings = generate_node_embeddings(G, tokenizer, model)
        save_embeddings(embeddings, emb_path)
        print(f"Generated and saved embeddings to {emb_path}")

    # 3) (Optional) prune away tiny disconnected fragments
    G = remove_small_fragents(G, size_threshold=5)

    # 4) (Optional) keep only giant component (and update embeddings to match)
    G, embeddings = return_giant_component_G_and_embeddings(G, embeddings)

    # 5) Update embeddings if graph changed
    embeddings = update_node_embeddings(embeddings, G, tokenizer, model)

    # 6) Simplify the graph by merging near‑duplicate nodes
    G_simple, embeddings_simple = simplify_graph(
        G, embeddings, tokenizer, model,
        similarity_threshold=0.9,
        use_llm=False,
        data_dir_output=".",
        graph_root="simplified",
        verbatim=True
    )

    # 7) Write out simplified graph (with texts) as JSON‐friendly GraphML
    save_graph_with_text_as_JSON(G_simple, data_dir=".", graph_name="simplified_graph.graphml")

    # 8) Produce an interactive HTML view
    html_path = make_HTML(G_simple, data_dir=".", graph_root="simplified")
    print("HTML visualization written to", html_path)

    print("Done running full pipeline.")