import sys
#sys.path.append("/home/aeiyan/testingSciAgentsDiscovery/SciAgentMath/GraphReasoning")


#from GraphReasoning.graph_tools import *
#from GraphReasoning.utils import *
#from GraphReasoning.graph_analysis import *

from GraphReasoning.graph_tools import *
from GraphReasoning.utils import *
from GraphReasoning.graph_analysis import *


from PyPDF2 import PdfReader
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



# Set the path to your graphml file
#data_dir = './data_output_KG_41/'
data_dir = './embeddings/'
graphml_path = data_dir + 'simple_graph_graphML_simplified.graphml'  # replace with actual filename

# Load the graph
G = nx.read_graphml(graphml_path)

# Compute statistics
res_stat = graph_statistics_and_plots_for_large_graphs(
    G,
    data_dir=data_dir,
    include_centrality=False,
    make_graph_plot=False
)

# Optional: print or inspect the result
print(res_stat)
