import sys
import sys
sys.path.append("..")  # Go one level up

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
import json

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
import transformers
from transformers import logging
logging.set_verbosity_error()
import requests

import openai
from openai import OpenAI
import base64
from datetime import datetime

import ollama

# Code based on: https://github.com/rahulnyk/knowledge_graph

#BASE_URL = 'http://localhost:11434'
BASE_URL = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

# Generate a response for a given prompt with a provided model. This is a streaming endpoint, so will be a series of responses.
# The final response object will include statistics and additional data from the request. Use the callback function to override
# the default handler.
def generate(prompt, model_name='zephyr:latest', system=None, template=None, context=None, options=None, callback=None):
    try:
        url = f"{BASE_URL}/api/generate"
        payload = {
            "model": model_name, 
            "prompt": prompt, 
            "system": system, 
            "template": template, 
            "context": context, 
            "options": options
        }
        
        # Remove keys with None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            
            # Creating a variable to hold the context history of the final chunk
            final_context = None
            
            # Variable to hold concatenated response strings if no callback is provided
            full_response = ""

            # Iterating over the response line by line and displaying the details
            for line in response.iter_lines():
                if line:
                    try:
                        line = line.decode("utf-8").strip()
                        if not line:
                            continue  # Skip blank or whitespace-only lines
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON: {line}")
                        continue
                    
                    # If a callback function is provided, call it with the chunk
                    if callback:
                        callback(chunk)
                    else:
                        # If this is not the last chunk, add the "response" field value to full_response and print it
                        if not chunk.get("done"):
                            response_piece = chunk.get("response", "")
                            full_response += response_piece
                            print(response_piece, end="", flush=True)
                    
                    # Check if it's the last chunk (done is true)
                    if chunk.get("done"):
                        final_context = chunk.get("context")
            
            # Return the full response and the final context
            return full_response, final_context
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None, None

def extract (string, start='[', end=']'):
    start_index = string.find(start)
    end_index = string.rfind(end)
     
    return string[start_index :end_index+1]
def documents2Dataframe(documents) -> pd.DataFrame:
    rows = []
    for chunk in documents:
        row = {
            "text": chunk,
           # **chunk.metadata,
            "chunk_id": uuid.uuid4().hex,
        }
        rows = rows + [row]

    df = pd.DataFrame(rows)
    return df

def concepts2Df(concepts_list) -> pd.DataFrame:
    ## Remove all NaN entities
    concepts_dataframe = pd.DataFrame(concepts_list).replace(" ", np.nan)
    concepts_dataframe = concepts_dataframe.dropna(subset=["entity"])
    concepts_dataframe["entity"] = concepts_dataframe["entity"].apply(
        lambda x: x.lower()
    )

    return concepts_dataframe


def df2Graph(dataframe: pd.DataFrame, generate, repeat_refine=0, verbatim=False,
          
            ) -> list:
  
    results = dataframe.apply(
        lambda row: graphPrompt(row.text, generate, {"chunk_id": row.chunk_id}, repeat_refine=repeat_refine,
                                verbatim=verbatim,#model
                               ), axis=1
    )
    # invalid json results in NaN
    results = results.dropna()
    results = results.reset_index(drop=True)

    ## Flatten the list of lists to one single list of entities.
    concept_list = np.concatenate(results).ravel().tolist()
    return concept_list


def graph2Df(nodes_list) -> pd.DataFrame:
    ## Remove all NaN entities
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
    graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: str(x).lower())
    graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: str(x).lower())

    return graph_dataframe



def graphPrompt(input: str, generate, metadata={}, repeat_refine=0, verbatim=False):
    import json
    from tqdm import tqdm


    def is_valid_ontology_json(response):
        try:
            data = json.loads(response)
            return isinstance(data, list) and all(
                isinstance(d, dict) and {"node_1", "node_2", "edge"} <= d.keys()
                for d in data
            )
        except:
            return False

    def normalize_ontology(result):
        for item in result:
            item['node_1'] = item['node_1'].strip().title()
            item['node_2'] = item['node_2'].strip().title()
            item['edge'] = item['edge'].strip().lower()
        return result


    

    SYS_PROMPT_GRAPHMAKER = (
     "You are a mathematical ontology extractor specializing in extracting formal conceptual relationships from advanced mathematics texts. "
    "You are provided with a context (delimited by triple backticks) that includes definitions, theorems, lemmas, or theoretical explanations. "
    "Your task is to extract ontology triples representing key mathematical entities and the logical, structural, or definitional relationships between them.\n\n"

    
    "Format your output as a list of JSON. Each element of the list contains a pair of terms "
    "and the relation between them, like the following:\n"
    "[\n"
    "   {{\n"
    '       "node_1": "a well-defined mathematical concept (e.g., group, theorem, construction)",\n'
    '       "node_2": "a related concept from the context",\n'
    '       "edge": "a precise relationship between the two, using clear academic terms (e.g., "is equivalent to", "characterizes", "is used to define", "generalizes", "is bounded by")."\n'
    "   }}, {{...}}\n"
    "]\n\n"

    "Avoid vague verbs like \"related to\". Ensure the triples reflect formal logical or structural connections found in mathematical discourse."
    "Examples:\n"
    "Context: ```Nonprincipal ultrafilters are used to produce asymptotic cones of a metric space.```\n"
    "[\n"
    "   {{\n"
    '       "node_1": ""Nonprincipal ultrafilters",\n'
    '       "node_2": "Asymptotic Cones of a Metric Space",\n'
    '       "edge": "are used to produce"\n'
    "   }},\n"
    "   {{\n"
    '       "node_1": "The Fundamental Theorem of Bass-Serre Theory",\n'
    '       "node_2": "Groups G acting on a tree",\n'
    '       "edge": "characterizes up to isomorphism"\n'
    "   }},\n"
    "   {{\n"
    '       "node_1": "A finitely generated group G has polynomial growth",\n'
    '       "node_2": "A finitely generated group G a nilpotent subgroup of finite index",\n'
    '       "edge": "is an equivalent statement to"\n'
    "   }}\n"
    "]\n"
    "Analyze the text carefully and produce triplets, making sure they reflect consistent mathematical ontologies.\n"
)










    

    SYS_PROMPT_FORMAT = (
        "You respond in this format: [ {\"node_1\": ..., \"node_2\": ..., \"edge\": ...}, {...} ]"
    )

    def run_generate(prompt, system=None):
        return generate(prompt=prompt, system=system, options={"temperature": 0.3, "top_p": 0.9, "num_predict": 512})[0]

    print(".", end="")
    response = run_generate(SYS_PROMPT_GRAPHMAKER.format(input))

    if verbatim:
        print("\n--- Initial Extraction ---\n", response)

    USER_PROMPT = (
    f"Read this context: ```{input}```."
    f" Read this ontology: ```{response}```"
    f"\n\nImprove the ontology by renaming nodes so that they have consistent labels that are widely used in the field of Mathematics. "
)



    ############## response,_= turn into this after meeting
 

    response =  run_generate(USER_PROMPT, system=SYS_PROMPT_FORMAT)

    if verbatim:
        print("\n--- After Label Normalization ---\n", response)

    if not is_valid_ontology_json(response):
        response = run_generate(f"Context: ```{response}```\n\nFix to make sure it is proper JSON format.", system=SYS_PROMPT_FORMAT)

    for rep in range(repeat_refine):
        USER_PROMPT = (
            f"Insert new triplets. Context: ```{input}```."
            f" Read current ontology: ```{response}```"
            "\n\nAdd new meaningful triplets. Keep consistent format."
        )
        new_response = run_generate(USER_PROMPT, system=SYS_PROMPT_GRAPHMAKER)

        if verbatim:
            print(f"\n--- Refinement {rep+1} ---\n", new_response)

        USER_PROMPT = (
            f"Merge and clean triplets. Context: ```{input}```."
            f" Ontology: ```{new_response}```"
            "\n\nEnsure proper JSON formatting and consistent naming."
        )
        response = run_generate(USER_PROMPT, system=SYS_PROMPT_FORMAT)

    if not is_valid_ontology_json(response):
        response = run_generate(f"Fix the format of the following JSON list: ```{response}```", system=SYS_PROMPT_FORMAT)

    try:
        result = json.loads(response)
        result = normalize_ontology(result)
        result = [dict(item, **metadata) for item in result]
    except Exception as e:
        print("\n\nERROR ### Final parsing failed:\n", response, "\n", str(e))
        result = None

    return result

def colors2Community(communities) -> pd.DataFrame:
    
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

def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    ## Melt the dataframe into a list of nodes
    df['node_1'] = df['node_1'].astype(str)
    df['node_2'] = df['node_2'].astype(str)
    df['edge'] = df['edge'].astype(str)
    dfg_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )
    dfg_long.drop(columns=["variable"], inplace=True)
    # Self join with chunk id as the key will create a link between terms occuring in the same text chunk.
    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
    # drop self loops
    self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
    dfg2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)
    ## Group and count edges.
    dfg2 = (
        dfg2.groupby(["node_1", "node_2"])
        .agg({"chunk_id": [",".join, "count"]})
        .reset_index()
    )
    dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
    dfg2.replace("", np.nan, inplace=True)
    dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
    # Drop edges with 1 count
    dfg2 = dfg2[dfg2["count"] != 1]
    dfg2["edge"] = "contextual proximity"
    return dfg2
    
def make_graph_from_text (generate,
                          include_contextual_proximity=False,
                          graph_root='graph_root',
                          repeat_refine=0,verbatim=False,
                          data_dir='./graphgenerationOutput/',
                          save_PDF=False,#TO DO
                          save_HTML=True,
                         ): 
    print("🧩 STEP 1: Entered make_graph_from_text")
      
    #json_dir = r"C:/Users/Admin/Desktop/Samaneh_Proj/distillated_chunks_02"
    json_dir = r"/home/aeiyan/testingSciAgentsDiscovery/SciAgentMath/distilOutput3"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    outputdirectory = Path(f"./{data_dir}/")

    pages = []
    json_files = glob(os.path.join(json_dir, "*.json"))

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON file: {json_file}")
                continue

            if not isinstance(data, dict) or "chunks" not in data:
                print(f"Warning: Unexpected format in {json_file}")
                continue

            for chunk in data["chunks"]:
                # Build combined text: title + summary + bullet points
                equations_text = "\n".join(
                    f"{eq.get('expr', '')}: {eq.get('description', '')}"
                    for eq in chunk.get("equations", [])
                )

                formal_constructs_text = "\n".join(
                    f"{fc.get('type', '')} {fc.get('label', '')}: {fc.get('statement', '')}"
                    for fc in chunk.get("formal_constructs", [])
                )

                text_parts = [
                    chunk.get("title", ""),
                    chunk.get("summary", ""),
                    "\n".join(chunk.get("bullet_points", [])),
                    equations_text,
                    formal_constructs_text
                    ]

                combined_text = "\n".join(part for part in text_parts if part.strip())

                pages.append({
                    "page_content": combined_text,
                    "metadata": {"source": json_file}
                })

    print(f"Loaded {len(pages)} chunks from {json_dir}")

    df = documents2Dataframe([p["page_content"] for p in pages])
    df.to_csv(outputdirectory / "chunks.csv", index=False)

    regenerate = True
    
    if regenerate:
        concepts_list = df2Graph(df,generate,repeat_refine=repeat_refine,verbatim=verbatim) #model='zephyr:latest' )
        # with open(outputdirectory /"concepts_list.pkl", "wb") as f:
        #     pickle.dump(concepts_list, f)
        dfg1 = graph2Df(concepts_list)
        if not os.path.exists(outputdirectory):
            os.makedirs(outputdirectory)
        
        dfg1.to_csv(outputdirectory/f"{graph_root}_graph.csv", sep="|", index=False)
        df.to_csv(outputdirectory/f"{graph_root}_chunks.csv", sep="|", index=False)
        dfg1.to_csv(outputdirectory/f"{graph_root}_graph_clean.csv", #sep="|", index=False
                   )
        #print("🧩 STEP 4: Graph generated by LLM and saved as CSV")
        df.to_csv(outputdirectory/f"{graph_root}_chunks_clean.csv", #sep="|", index=False
                 )
    else:
        dfg1 = pd.read_csv(outputdirectory/f"{graph_root}_graph.csv", sep="|")
    
    dfg1.replace("", np.nan, inplace=True)
    dfg1 = dfg1.infer_objects(copy=False)
    dfg1.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
    dfg1['count'] = 4 
      
    if verbatim:
        print("Shape of graph DataFrame: ", dfg1.shape)
    dfg1.head()### 
    
    if include_contextual_proximity:
        dfg2 = contextual_proximity(dfg1)
        dfg = pd.concat([dfg1, dfg2], axis=0)
        #dfg2.tail()
    else:
        dfg=dfg1



    
    #modified by samaneh
    dfg = (
        dfg.groupby(["node_1", "node_2"])
        .agg({
            "chunk_id": lambda x: ",".join(map(str, x)), 
            "edge": lambda x: ','.join(map(str,x)),
            'count': 'sum'})
        .reset_index()
    )
    # #dfg
    # dfg = (
    #     dfg.groupby(["node_1", "node_2"])
    #     .agg({"chunk_id": ",".join, "edge": ','.join, 'count': 'sum'})
    #     .reset_index()
    # )
        
    nodes = pd.concat([dfg['node_1'], dfg['node_2']], axis=0).unique()
    print ("Nodes shape: ", nodes.shape)
    print("🧩 STEP 5: Finished grouping edges and collected unique nodes")
    
    G = nx.Graph()
    node_list=[]
    node_1_list=[]
    node_2_list=[]
    title_list=[]
    weight_list=[]
    chunk_id_list=[]
    
    ## Add nodes to the graph
    for node in nodes:
        G.add_node(
            str(node)
        )
        node_list.append (node)
    
    ## Add edges to the graph
    for _, row in dfg.iterrows():
        
        G.add_edge(
            str(row["node_1"]),
            str(row["node_2"]),
            title=row["edge"],
            weight=row['count']/4
        )
        
        node_1_list.append (row["node_1"])
        node_2_list.append (row["node_2"])
        title_list.append (row["edge"])
        weight_list.append (row['count']/4)
         
        chunk_id_list.append (row['chunk_id'] )
    
    print("🧩 STEP 6: Finished adding edges to the NetworkX graph")
    try:
            
        df_nodes = pd.DataFrame({"nodes": node_list} )    
        df_nodes.to_csv(f'{data_dir}/{graph_root}_nodes.csv')
        df_nodes.to_json(f'{data_dir}/{graph_root}_nodes.json')
        
        df_edges = pd.DataFrame({"node_1": node_1_list, "node_2": node_2_list,"edge_list": title_list, "weight_list": weight_list } )    
        df_edges.to_csv(f'{data_dir}/{graph_root}_edges.csv')
        df_edges.to_json(f'{data_dir}/{graph_root}_edges.json')
        
    except:
        
        print ("Error saving CSV/JSON files.")
    
    communities_generator = nx.community.girvan_newman(G)
    #top_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)
    communities = sorted(map(sorted, next_level_communities))
    
    if verbatim:
        print("Number of Communities = ", len(communities))
        
    if verbatim:
        print("Communities: ", communities)
    
    colors = colors2Community(communities)
    if verbatim:
        print ("Colors: ", colors)
    
    for index, row in colors.iterrows():
        G.nodes[row['node']]['group'] = row['group']
        G.nodes[row['node']]['color'] = row['color']
        G.nodes[row['node']]['size'] = G.degree[row['node']]
            
    net = Network(
             
            notebook=True,
         
            cdn_resources="remote",
            height="900px",
            width="100%",
            select_menu=True,
            
            filter_menu=False,
        )
        
    net.from_nx(G)
    net.force_atlas_2based(central_gravity=0.015, gravity=-31)
   
    net.show_buttons()
    
    graph_HTML= f'{data_dir}/{graph_root}_grapHTML.html'
    graph_GraphML=  f'{data_dir}/{graph_root}_graphML.graphml'  #  f'{data_dir}/resulting_graph.graphml',
    nx.write_graphml(G, graph_GraphML)
    print("🧩 STEP 7: Saved GraphML file")
    
    if save_HTML:
        # Solution 1: Using the patched method
        with open(graph_HTML, 'w', encoding='utf-8') as f:
            f.write(net.generate_html())

    if save_PDF:
        output_pdf=f'{data_dir}/{graph_root}_PDF.pdf'
        pdfkit.from_file(graph_HTML,  output_pdf)
    else:
        output_pdf=None

    # try:   
    #     res_stat=graph_statistics_and_plots_for_large_graphs(G, data_dir=data_dir,include_centrality=False,
    #                                                          make_graph_plot=False,)
    #     print ("Graph statistics: ", res_stat)

    # except Exception as e:
    #     print("❌ ERROR in graph_statistics_and_plots_for_large_graphs:", e)   
    print("✅ STEP 8: Finished everything, returning results")
    return graph_HTML, graph_GraphML, G, net, output_pdf


# we aremt isomg tjos
def add_new_subgraph_from_text(txt,generate,node_embeddings,tokenizer, model,
                               original_graph_path_and_fname,
                               data_dir_output='./data_temp/', verbatim=True,
                               size_threshold=10,chunk_size=10000,
                               do_Louvain_on_new_graph=True,include_contextual_proximity=False,repeat_refine=0,similarity_threshold=0.95, do_simplify_graph=True,#whether or not to simplify, uses similiraty_threshold defined above
                               return_only_giant_component=False,
                               save_common_graph=False,G_to_add=None,graph_GraphML_to_add=None,
                              ):
    

    print("🟢 Step 1: Entered add_new_subgraph_from_text")

    display (Markdown(txt[:256]+"...."))
    graph_GraphML=None
     
    G_new=None
    res=None
    assert not (G_to_add is not None and graph_GraphML_to_add is not None), "G_to_add and graph_GraphML_to_add cannot be used together. Pick one or the other to provide a graph to be added."
 
    try:
        print("Step 2: Starting graph generation")
        start_time = time.time() 
        idx=0
        
        if verbatim:
            print ("Now create or load new graph...")

        if graph_GraphML_to_add==None and G_newlymade==None: #make new if no existing one provided
            print("Step 3: Generating new graph from text...")
            print ("Make new graph from text...")
            _, graph_GraphML_to_add, G_to_add, _, _ =make_graph_from_text (txt,generate,
                                      include_contextual_proximity=include_contextual_proximity,
                                      
                                     data_dir=data_dir_output,
                                     graph_root=f'graph_new_{idx}',
                                    
                                        chunk_size=chunk_size,   repeat_refine=repeat_refine, 
                                      verbatim=verbatim,
                                       
                                  )
            print("✅ Step 3.1: Graph generated from text")
            
            if verbatim:
                print ("Generated new graph from text provided: ", graph_GraphML_to_add)

        else:
            if verbatim:
                print ("Instead of generating graph, loading it or using provided graph...(any txt data provided will be ignored...)")

            if graph_GraphML_to_add!=None:
                print ("Loading graph: ", graph_GraphML_to_add)
        
        print("--- %s seconds ---" % (time.time() - start_time))
        print("🔷 Step 4: Graph generation or loading complete")

    except Exception as e:
        print(f"❌ Exception occurred: {e}")
    
    print ("Now add node to existing graph...")
    
    try:
        #Load original graph
        print("🔶 Step 5: Now loading the original base graph")
        G = nx.read_graphml(original_graph_path_and_fname)
        
        if G_to_add!=None:
            G_loaded=H = deepcopy(G_to_add)
            if verbatim:
                print ("Using provided graph to add (any txt data provided will be ignored...)")
        else:
            if verbatim:
                print ("Loading graph to be added either newly generated or provided.")
            G_loaded = nx.read_graphml(graph_GraphML_to_add)
        
        res_newgraph=graph_statistics_and_plots_for_large_graphs(G_loaded, data_dir=data_dir_output,include_centrality=False,
                                                       make_graph_plot=False,root='new_graph')
        print (res_newgraph)
        
        G_new = nx.compose(G,G_loaded)
        print("🟠 Step 6: Graphs combined with nx.compose")

        if save_common_graph:
            print ("Identify common nodes and save...")
            try:
                
                common_nodes = set(G.nodes()).intersection(set(G_loaded.nodes()))
    
                subgraph = G_new.subgraph(common_nodes)
                graph_GraphML=  f'{data_dir_output}/{graph_root}_common_nodes_before_simple.graphml' 
                nx.write_graphml(subgraph, graph_GraphML)
            except: 
                print ("Common nodes identification failed.")
            print ("Done!")
        
        if verbatim:
            print ("Now update node embeddings")
        print("🟣 Step 7: Updating node embeddings")    
        node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model)
        print ("Done update node embeddings.")
        print("✅ Step 7.1: Node embeddings updated")
        if do_simplify_graph:
            print("🟤 Step 8: Starting graph simplification")
            if verbatim:
                print ("Now simplify graph.")
            G_new, node_embeddings =simplify_graph (G_new, node_embeddings, tokenizer, model , 
                                                    similarity_threshold=similarity_threshold, use_llm=False, data_dir_output=data_dir_output,
                                    verbatim=verbatim,)
            if verbatim:
                print ("Done simplify graph.")
            print("✅ Step 8.1: Graph simplified")    
            
        if verbatim:
            print ("Done update graph")
        
        if size_threshold >0:
            if verbatim:
                print ("Remove small fragments") 
            print("🧊 Step 9: Removing small graph fragments")               
            G_new=remove_small_fragents (G_new, size_threshold=size_threshold)
            node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model, verbatim=verbatim)
        
        if return_only_giant_component:
            if verbatim:
                print ("Select only giant component...")   
            connected_components = sorted(nx.connected_components(G_new), key=len, reverse=True)
            G_new = G_new.subgraph(connected_components[0]).copy()
            node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model, verbatim=verbatim)
            
        print (".")
        if do_Louvain_on_new_graph:
            print("🧩 Step 10: Starting Louvain clustering")
            G_new=graph_Louvain (G_new, 
                      graph_GraphML=None)
            if verbatim:
                print ("Don Louvain...")

        print (".")
         
        graph_root=f'graph'
        graph_GraphML=  f'{data_dir_output}/{graph_root}_augmented_graphML_integrated.graphml'  #  f'{data_dir}/resulting_graph.graphml',
        print (".")
        nx.write_graphml(G_new, graph_GraphML)
        print ("Done...written: ", graph_GraphML)
        res=graph_statistics_and_plots_for_large_graphs(G_new, data_dir=data_dir_output,include_centrality=False,
                                                       make_graph_plot=False,root='assembled')
        
        print ("Graph statistics: ", res)

    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        print (end="")


    
    
    print("🏁 Step 11: Done! Writing final GraphML and returning results")
    return graph_GraphML, G_new, G_loaded, G, node_embeddings, res

#################################################################################################
# api_key='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'
# def generate(system, prompt, model="llama3-8b-8192", max_retries=5):
#     url = "https://api.groq.com/openai/v1/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {'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'}",  # Replace with your actual key or handle externally
#         "Content-Type": "application/json"
#     }
#     data = {
#         "model": model,
#         "messages": [
#             {"role": "system", "content": system},
#             {"role": "user", "content": prompt}
#         ]
#     }

#     retry_delay = 3  # seconds
#     for attempt in range(max_retries):
#         response = requests.post(url, headers=headers, json=data)
#         if response.status_code == 200:
#             return response.json()["choices"][0]["message"]["content"]
#         elif response.status_code == 429:
#             print(f"⚠️ Rate limit hit. Retrying in {retry_delay:.2f}s... (attempt {attempt+1}/{max_retries})")
#             time.sleep(retry_delay)
#             retry_delay *= 1.5  # exponential backoff
#         else:
#             raise Exception(f"Request failed: {response.status_code}, {response.text}")

#     raise Exception("❌ Max retries exceeded due to rate limits.")




######################################################
# openai_api_key="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"

# def generate ( system, prompt,
#               temperature=0.2,max_tokens=2048,timeout=120,
#              frequency_penalty=0, 
#              presence_penalty=0, 
#              top_p=1.0,  
#             gpt_model='gpt-4o', organization='',
#              ):
#     client = openai.OpenAI(api_key="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",
#                       organization ="org-cMVAuZIuiPCh4hwT8E80HZAK")

#     chat_completion = client.chat.completions.create(
#         messages=[
#             {
#                 "role": "system",
#                 "content": system,
#             },
#             {
#                 "role": "user",
#                 "content": prompt,
#             }
#         ],
#         temperature=temperature,
#         max_tokens=max_tokens,
#         model=gpt_model,
#         timeout=timeout,
#         frequency_penalty=frequency_penalty,
#         presence_penalty=presence_penalty,
#         top_p=top_p,
#     )
#     return chat_completion.choices[0].message.content


# def read_text_from_pdf(file_path):
#     reader = PdfReader(file_path)
#     text = ""
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# txt_content = read_text_from_pdf(r'C:/Users/Admin/Documents/Postdoc_KG_0/test_0_papers/LPSF.pdf')
# clean_txt_content = remove_markdown_symbols(txt_content) #added by samaneh using utils.py


if __name__ == "__main__":
    # Code to run if the script is executed directly

    txt=None

    print("📌 MAIN STEP 1: About to call make_graph_from_text")
    
    graph_HTML, graph_GraphML, G, net, output_pdf = make_graph_from_text( generate,
                            include_contextual_proximity=False,
                            graph_root='graph_root',
                            repeat_refine=0,verbatim=False,
                            ##data_dir='./data_output_KG_45/',
                            data_dir='./graphgenerationOutput/',
                            save_PDF=True,#TO DO
                            save_HTML=True,
                            )
    
    data_dir ='./graphgenerationOutput/'
    graph_root = 'graph_root'
    
    generated_files = [
    f'{graph_root}_graph.csv',
    f'{graph_root}_chunks.csv',
    f'{graph_root}_graph_clean.csv',
    f'{graph_root}_chunks_clean.csv',
    f'{graph_root}_nodes.csv',
    f'{graph_root}_nodes.json',
    f'{graph_root}_edges.csv',
    f'{graph_root}_edges.json',
    f'{graph_root}_grapHTML.html',
    f'{graph_root}_graphML.graphml',
    f'{graph_root}_PDF.pdf'
    ]

    for file_name in generated_files:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            print(f"File {file_name} generated successfully.")
        else:
            print(f"File {file_name} not found.")
    
    #print("❌ ERROR in make_graph_from_text:", e)

        
    

    # print("📌 MAIN STEP 2: Finished calling make_graph_from_text")
    # #print("graph_HTML:", graph_HTML)
    # #print("graph_GraphML:", graph_GraphML)
    # #print("G:", type(G))
    # #print("net:", type(net))
    # #print("output_pdf:", output_pdf) 
    print("SinShin*************************************************************************SINsSHIn")

