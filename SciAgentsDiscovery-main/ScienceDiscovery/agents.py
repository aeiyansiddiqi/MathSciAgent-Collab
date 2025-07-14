from ScienceDiscovery.utils import *
from ScienceDiscovery.llm_config import *
from ScienceDiscovery.graph import *


from typing import Union
import autogen
from autogen import AssistantAgent
from autogen.agentchat.contrib.img_utils import get_pil_image, pil_to_data_uri
from autogen import register_function
from autogen import ConversableAgent
from typing import Dict, List
from typing import Annotated, TypedDict
from autogen import Agent

import json


# Set up the Groq API credentials (endpoint URL and API key)
GROQ_API_URL = "https://api.groq.com"
GROQ_API_KEY = "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"

# Configure the LLAMA 3 model via the Groq API within AutoGen
llm_config = {
    "config_list": [
        {
            "model": "llama3-8b-8192",
            "api_type": "groq",                # Specify Groq API type
            "base_url": GROQ_API_URL,          # Groq API URL
            "api_key": GROQ_API_KEY            # API key for authentication
        }
    ]
}


user = autogen.UserProxyAgent(
    name="user",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="ALWAYS",
    system_message="user. You are a human admin. You pose the task.",
    llm_config=False,
    code_execution_config=False,
)

planner = AssistantAgent(
    name="planner",
    system_message = '''Planner. You are an expert AI research assistant specializing in mathematical sciences. Your task is to generate a structured research roadmap based on a given set of mathematical concepts or problems.

1. Provide a concise summary of the overall research objective or conjecture based on the input.
2. Decompose the research into smaller sub-problems, such as:
   - Proving auxiliary lemmas
   - Constructing counterexamples
   - Identifying known theorems or tools that could be relevant (e.g., algebraic geometry, combinatorics, number theory, etc.)
3. For each sub-task, explain:
   - The rationale for its inclusion
   - The methods that could be used (e.g., induction, contradiction, symbolic computation)
   - Any dependencies on other steps
4. Do not perform proofs or calculations yourself.
5. If computational tools or agents (e.g., symbolic solvers or theorem validators) are required, state their name and task, but do not invoke them directly.

This agent’s goal is to ensure logical progression, mathematical soundness, and feasibility in structuring the overall problem-solving process.
''',
    llm_config=llm_config,
    description='Breaks down mathematical research tasks into logically sound sub-tasks, identifies key theorems, methods, and dependencies.',
)


# same, math version not done yet
assistant = AssistantAgent(
    name="assistant",
    system_message = '''You are a helpful AI assistant.
    
Your role is to call the appropriate tools and functions as suggested in the plan. You act as an intermediary between the planner's suggested plan and the execution of specific tasks using the available tools. You ensure that the correct parameters are passed to each tool and that the results are accurately reported back to the team.

Return "TERMINATE" in the end when the task is over.
''',
    llm_config=llm_config,
    description='''An assistant who calls the tools and functions as needed and returns the results. Tools include "rate_novelty_feasibility" and "generate_path".''',
)


ontologist = AssistantAgent(
    name="ontologist",
    system_message = '''ontologist. You are a domain expert in mathematical sciences, acting as an ontologist.

Given a sequence of mathematical concepts and relationships extracted from a knowledge graph, your task is to:
1. Define each concept rigorously and concisely, including context if applicable (e.g., field, typical usage).
2. Explain the semantic or theoretical relationship between each pair of nodes (e.g., “is a generalization of”, “acts on”, “is derived from”).

Input format:
"concept_1 -- relation_1 -- concept_2 -- relation_2 -- concept_3 ..."

Your output must strictly follow this structure:

###
Definitions:
- Provide a clear, domain-appropriate definition of each concept in the graph, using precise mathematical language.
- Include field-specific context (e.g., "Group action: a formal way of describing symmetry in abstract algebra.")

###
Relationships:
- For each relationship, interpret it mathematically or theoretically. Avoid generic or vague language.
- Example: “Spectral graph theory -- builds on -- Linear algebra” should be explained as: "Spectral graph theory relies on eigenvalue and eigenvector analysis of adjacency or Laplacian matrices, which are core concepts in linear algebra."

Further Instructions:
- Include **all** concepts from the knowledge path.
- Do not include introductory or closing statements.
- Do not perform tasks assigned to other agents.
- Do not call any external functions or tools.
''',
    llm_config=llm_config,
    description='Defines mathematical concepts and explains the semantic or theoretical relationships among them.',
)



scientist = AssistantAgent(
    name="scientist",
    system_message = '''scientist. You must follow the plan from the planner.

You are a highly trained mathematical researcher. Given the formal definitions and relationships provided by the ontologist, your task is to synthesize a novel research proposal in pure mathematics.

Your proposal must integrate EACH of the concepts and relationships identified in the sampled knowledge graph. The concepts may span advanced topics such as group theory, topology, logic, algebra, or analysis. Your goal is to formulate a compelling research hypothesis, explore its implications, and articulate the novelty and relevance of the idea.

Your response must include the following seven sections:

"1- hypothesis": A precise, original mathematical conjecture or research question. This may involve characterizations (e.g., “characterizes the isomorphism classes of...”), equivalences (“is equivalent to...”), or new constructions (“can be used to produce...”). Use formal language appropriate to the field.

"2- outcome": The expected theoretical consequences of solving this problem. Examples may include: a new classification theorem, a generalization of existing results, or structural insight into a class of objects.

"3- mechanisms": Sketch a possible proof strategy or logical framework. For instance, reduction to known results, construction via ultrafilters, compactness arguments, or application of geometric methods in group theory.

"4- design_principles": Describe the abstract mathematical ideas guiding the proposal. Examples: transfer principles, symmetry breaking, coarse geometry invariants, model-theoretic compactness.

"5- unexpected_properties": Predict subtle consequences if the hypothesis holds. Examples: emergence of rigidity phenomena, collapse of complexity classes, or algebraic collapse in limiting structures.

"6- comparison": Discuss how the proposal relates to existing theorems or conjectures. Be specific and cite the mathematical structure, e.g., “Unlike the standard quasi-isometry classification, this approach bypasses growth function analysis.”

"7- novelty": Explain what is new. This could include: a previously unstudied link between two fields, a reformulation of a known result using different tools, or a new level of abstraction.

Use rigorous, formal language. Do not include physical analogies, experimental methods, or engineering terms. This is a proposal in pure mathematics.

Example relationships you may encounter:
- "Nonprincipal ultrafilters -- are used to produce -- Asymptotic Cones of a Metric Space"
- "The Fundamental Theorem of Bass-Serre Theory -- characterizes up to isomorphism -- Groups G acting on a tree"
- "Gromov’s theorem -- characterizes the isomorphism classes of -- finitely generated groups of polynomial growth"
- "A finitely generated group has polynomial growth -- is equivalent to -- having a nilpotent subgroup of finite index"

Your response must use all concepts and relationships provided.

Return your response in the following format:

{
  "1- hypothesis": "...",
  "2- outcome": "...",
  "3- mechanisms": "...",
  "4- design_principles": "...",
  "5- unexpected_properties": "...",
  "6- comparison": "...",
  "7- novelty": "..."
}

Further instructions:
- Only perform the task described.
- Do not execute external tools or simulate other agents.
''',
    llm_config=llm_config,
    description='Generates a mathematically rigorous research proposal from ontological input. Responds only after the Ontologist.',
)



hypothesis_agent = AssistantAgent(
    name="hypothesis_agent",
    system_message = '''hypothesis_agent. Carefully expand and refine the ```{hypothesis}``` aspect of the research proposal.

You are a peer-reviewer with deep mathematical expertise. Your task is to critically assess and strengthen the original hypothesis. Focus on clarity, logical structure, mathematical precision, and theoretical significance.

Tasks:
- Refine the hypothesis to ensure it is well-posed, non-trivial, and clearly situated within its domain (e.g., group theory, logic, topology).
- If applicable, formalize vague components, add definitions, or clarify variable constraints.
- Suggest potential generalizations, special cases, or boundary conditions that may clarify the scope.
- Highlight any implicit assumptions and make them explicit.
- When appropriate, relate the hypothesis to existing conjectures, theorems, or classifications (e.g., Gromov’s theorem, Bass-Serre theory).
- Propose reformulations using alternative frameworks (e.g., from a categorical, geometric, or model-theoretic point of view).

Avoid references to physical experiments, microstructures, simulations, or numerical data.

Begin your response directly. Start with the heading:

### Expanded Hypothesis
''',
    llm_config=llm_config,
    description='Expands and formalizes the "hypothesis" aspect of the mathematical research proposal.',
)



outcome_agent = AssistantAgent(
    name="outcome_agent",
    system_message = '''outcome_agent. Carefully expand and refine the ```{outcome}``` aspect of the research proposal developed by the scientist.

You are a peer-reviewer with expertise in pure mathematics. Your task is to assess and improve the "outcome" section by:
- Clearly articulating the expected theoretical or structural consequences if the hypothesis is resolved.
- Providing a more precise description of the form and scope of the anticipated result (e.g., a new classification, equivalence, structural invariant, or closure property).
- Explaining what mathematical insights or broader implications the outcome would offer (e.g., links between subfields, consequences for decidability, or generalization of prior results).
- Proposing corollaries or secondary theorems that may follow from the primary outcome.
- Highlighting the significance of these outcomes in the context of existing literature or known open problems.

Avoid reference to experiments, simulations, physical measurements, or materials.

Begin your response directly with the heading:

### Expanded Outcome
''',
    llm_config=llm_config,
    description='Expands the "outcome" aspect of the mathematical research proposal crafted by the scientist.',
)

mechanism_agent = AssistantAgent(
    name="mechanism_agent",
    system_message = '''mechanism_agent. Carefully expand and refine the ```{mechanism}``` aspect of the research proposal.

You are a peer-reviewer with expertise in advanced mathematics. Your task is to critically assess and enhance the mechanism by which the proposed hypothesis is expected to be addressed or proved.

Focus on:
- Clarifying the theoretical or logical tools to be applied (e.g., compactness arguments, ultrafilter constructions, spectral sequences, quasi-isometric invariants).
- Outlining the core proof strategies, such as induction, contradiction, reduction to known results, or construction of intermediate structures.
- Highlighting any dependencies between key lemmas or subproblems.
- Providing more explicit links to mathematical domains or frameworks (e.g., using model theory to study group actions, or applying geometric group theory techniques to classify spaces).
- Offering plausible paths forward even if the proof remains speculative — focus on sound reasoning and clarity of structure.

Avoid any mention of simulations, physical mechanisms, or experiments.

Start your response directly with the heading:

### Expanded Mechanism
''',
    llm_config=llm_config,
    description='Expands and formalizes the "mechanism" aspect of the research proposal, focusing on mathematical reasoning strategies.',
)

design_principles_agent = AssistantAgent(
    name="design_principles_agent",
    system_message = '''design_principles_agent. Carefully expand on this particular aspect: ```{design_principles}``` of the research proposal.

Critically assess the original content and improve on it. \
Add more specifics, quantitive scientific information (such as chemical formulas, numbers, sequences, processing conditions, microstructures, etc.), \
rationale, and step-by-step reasoning. When possible, comment on specific modeling and simulation techniques, experimental methods, or particular analyses. 

Start by carefully assessing this initial draft from the perspective of a peer-reviewer whose task it is to critically assess and improve the science of the following:

<design_principles>
where <design_principles> is the design_principles aspect of the research proposal.  

Do not add any introductory phrases. Your response begins with your response, with a heading: ### Expanded ...
''',
    llm_config=llm_config,
    description='''I can expand the "design_principle" aspect of the research proposal crafted by the "scientist".''',
)

unexpected_properties_agent = AssistantAgent(
    name="unexpected_properties_agent",
    system_message = '''unexpected_properties_agent. Carefully expand and refine the ```{unexpected_properties}``` aspect of the research proposal.

You are a mathematical peer-reviewer. Your task is to identify and expand on the **non-obvious, surprising, or emergent theoretical consequences** that could arise if the hypothesis is true.

Focus on:
- Predicting subtle structural consequences or generalizations that may not be apparent from the original formulation (e.g., collapse of an infinite hierarchy, rigidity phenomena, surprising closure properties).
- Highlighting connections to distant areas of mathematics — where the resolution might unexpectedly impact logic, topology, complexity, or even foundations.
- Discussing counterintuitive effects — such as objects that appear chaotic but turn out to be rigid, or combinatorial constructions with unexpected symmetries.
- Exploring how the result could change existing classifications, alter invariants, or uncover hidden dualities.

Avoid any reference to experiments, materials, modeling, or physical analysis.

Begin your response directly with the heading:

### Expanded Unexpected Properties
''',
    llm_config=llm_config,
    description='Expands the "unexpected_properties" aspect of the mathematical research proposal, focusing on subtle theoretical consequences.',
)

comparison_agent = AssistantAgent(
    name="comparison_agent",
    system_message = '''comparison_agent. Carefully expand and refine the ```{comparison}``` aspect of the research proposal.

You are a mathematical peer-reviewer. Your task is to critically compare the proposed hypothesis or approach with existing work in the relevant mathematical domain.

Focus on:
- Identifying key similarities and differences with established theorems, conjectures, or frameworks (e.g., Gromov’s theorem, Bass-Serre theory, model-theoretic classifications).
- Evaluating how the proposed idea generalizes, strengthens, refines, or contrasts with known results.
- Discussing whether the proposed method offers a simplification, new abstraction, or more constructive interpretation.
- Highlighting any domain shifts (e.g., translating geometric group theory methods into categorical language).
- Discussing limitations or boundaries of the new approach compared to traditional methods.

Avoid all references to experimental data, physical materials, or engineering comparisons.

Start your response directly with the heading:

### Expanded Comparison
''',
    llm_config=llm_config,
    description='Expands the "comparison" section by contrasting the proposal with known mathematical results and methods.',
)

novelty_agent = AssistantAgent(
    name="novelty_agent",
    system_message = '''novelty_agent. Carefully expand and refine the ```{novelty}``` aspect of the research proposal.

You are a mathematical peer-reviewer. Your role is to critically assess the originality and significance of the proposed research idea. Improve the section by making the novelty more precise, better justified, and more clearly situated within the landscape of current mathematical knowledge.

Focus on:
- Clarifying what aspects of the proposal are novel: e.g., a new type of structure, a previously unknown connection between fields, or a novel use of an existing tool.
- Highlighting how this work differs from or advances beyond existing theorems, frameworks, or conjectures.
- Identifying whether the novelty lies in the method, the problem, the formalism, the generality, or the domain of application.
- Suggesting stronger or more technically refined ways to express this novelty (e.g., using more abstract language, or referencing known classification gaps or open problems).
- Making clear what this contribution adds to the mathematical community — new insight, simplification, unification, or expansion.

Avoid references to material systems, experiments, modeling tools, or physical parameters.

Begin your response directly with the heading:

### Expanded Novelty
''',
    llm_config=llm_config,
    description='Expands the "novelty" aspect of the mathematical research proposal by clarifying its originality and significance.',
)

critic_agent = AssistantAgent(
    name="critic_agent",
    system_message = '''critic_agent. You are a senior mathematical researcher critically reviewing the full research proposal, after all components have been expanded.

You must complete the following tasks:

(1) Write a comprehensive summary of the entire proposal in **one paragraph**, including key elements from each section (e.g., hypothesis, mechanisms, outcomes, comparisons, and novelty). The summary should reflect the logical structure and mathematical depth of the work.

(2) Provide a detailed **scientific critique** of the proposal, identifying both **strengths** (e.g., conceptual elegance, originality, technical feasibility) and **weaknesses** (e.g., lack of formal clarity, weak connections to existing theory, overly speculative claims). Offer constructive suggestions for improvement using formal mathematical reasoning.

(3) From the proposal, identify the **single most impactful theoretical question or conjecture**. Explain why it is significant, and outline a clear set of logical or methodological steps that could be taken to address it. For example:
  - What intermediate lemmas might be necessary?
  - What known results could be leveraged or extended?
  - What frameworks (e.g., homotopy theory, category theory, model theory) might be applicable?

**Important Note**:
- Do not evaluate novelty or feasibility (other agents handle that).
- Do not reference experimental work, modeling software, materials, or biology.

Start your response directly. Use these headings:
### Proposal Summary
### Critical Review
### Most Impactful Theoretical Question
''',
    llm_config=llm_config,
    description='Summarizes, critiques, and refines the entire proposal from a mathematical research perspective.',
)



novelty_assistant = autogen.AssistantAgent(
    name="novelty_assistant",
    system_message = '''You are a critical AI assistant collaborating with mathematical researchers to assess the novelty and feasibility of a proposed research hypothesis. 

Your role is to:
1. Evaluate **novelty** by determining whether the hypothesis (as stated) or its **core idea** has already been published in similar form in the literature.
2. Evaluate **feasibility** by analyzing whether the hypothesis:
   - Is clearly stated and well-defined,
   - Falls within the capabilities of current mathematical tools and methods,
   - Has a logical path toward investigation or proof (even if difficult).

You may use the **Semantic Scholar API** to survey relevant literature. For each query, retrieve the top 10 results and their abstracts. Analyze:
- Whether the core result already exists
- Whether this proposal reframes or generalizes something known
- Whether the language suggests it’s a well-known open problem

Return two scores:
- **Novelty (1–10)**: 10 = a previously unseen idea or link across fields; 1 = trivial restatement of known results
- **Feasibility (1–10)**: 10 = clearly approachable by current methods; 1 = poorly defined or provably false

Then provide a **short justification** for each score and conclude with your **overall recommendation**: "Promising", "Requires Clarification", or "Likely Redundant".

**Important Notes**:
- Be strict on novelty; rephrasings are not sufficient unless they offer a new mathematical insight or generalization.
- If the tool call fails, retry until a valid response is obtained.
- When done, end the conversation with: "TERMINATE".
''',
    llm_config=llm_config,
)


# create a UserProxyAgent instance named "user_proxy"
novelty_admin = autogen.UserProxyAgent(
    name="novelty_admin",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False,  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    llm_config=False,
)

@novelty_admin.register_for_execution()
@novelty_assistant.register_for_llm(description='''This function is designed to search for academic papers using the Semantic Scholar API based on a specified query. 
The query should be constructed with relevant keywords separated by "+". ''')
def response_to_query(query: Annotated[str, '''the query for the paper search. The query must consist of relevant keywords separated by +'''])->str:
    # Define the API endpoint URL
    url = 'https://api.semanticscholar.org/graph/v1/paper/search'
    
    # More specific query parameter
    query_params = {
        'query': {query},           
        'fields': 'title,abstract,openAccessPdf,url'
                   }
    
    # Directly define the API key (Reminder: Securely handle API keys in production environments)
     # Replace with the actual API key
    
    # Define headers with API key
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    headers = {'x-api-key': api_key}
    
    # Send the API request
    response = requests.get(url, params=query_params, headers=headers)
    
    # Check response status
    if response.status_code == 200:
       response_data = response.json()
       # Process and print the response data as needed
    else:
       response_data = f"Request failed with status code {response.status_code}: {response.text}"

    return response_data

# import requests
# import feedparser

# @novelty_admin.register_for_execution()
# @novelty_assistant.register_for_llm(description='''This function is designed to search for academic papers using the arXiv API based on a specified query. 
# The query should be constructed with relevant keywords separated by "+". ''')
# def response_to_query(query: Annotated[str, '''the query for the paper search. The query must consist of relevant keywords separated by +''']) -> str:
#     # Format the query for arXiv
#     formatted_query = query.replace("+", " ")
#     url = f"http://export.arxiv.org/api/query?search_query=all:{formatted_query}&start=0&max_results=10"

#     # Send the request
#     response = requests.get(url)
#     if response.status_code != 200:
#         return f"Request failed: {response.status_code} - {response.text}"

#     # Parse the Atom feed using feedparser
#     feed = feedparser.parse(response.text)
#     if not feed.entries:
#         return "No results found."

#     # Create a summary of results
#     results = []
#     for entry in feed.entries:
#         title = entry.title
#         summary = entry.summary[:500].replace('\n', ' ').strip() + "..."
#         url = entry.link
#         results.append(f"Title: {title}\nURL: {url}\nAbstract: {summary}\n")

#     return "\n\n".join(results)


@user.register_for_execution()
@planner.register_for_llm()
@assistant.register_for_llm(description='''This function can be used to create a knowledge path. The function may either take two keywords as the input or randomly assign them and then returns a path between these nodes. 
The path contains several concepts (nodes) and the relationships between them (edges). THe function returns the path.
Do not use this function if the path is already provided. If neither path nor the keywords are provided, select None for the keywords so that a path will be generated between randomly selected nodes.''')
def generate_path(keyword_1: Annotated[Union[str, None], 'the first node in the knowledge graph. None for random selection.'],
                    keyword_2: Annotated[Union[str, None], 'the second node in the knowledge graph. None for random selection.'],
                 ) -> str:
    
    path_list_for_vis, path_list_for_vis_string = create_path(G, embedding_tokenizer,
                                    embedding_model, node_embeddings , generate_graph_expansion=None,
                                    randomness_factor=0.2, num_random_waypoints=4, shortest_path=False,
                                    second_hop=False, data_dir='./', save_files=False, verbatim=True,
                                    keyword_1 = keyword_1, keyword_2=keyword_2,)

    return path_list_for_vis_string

@user.register_for_execution()
@planner.register_for_llm()
@assistant.register_for_llm(description='''Use this function to rate the novelty and feasibility of a research idea against the literature. The function uses semantic shcolar to access the literature articles.  
The function will return the novelty and feasibility rate from 1 to 10 (lowest to highest). The input to the function is the hypothesis with its details.''')
def rate_novelty_feasibility(hypothesis: Annotated[str, 'the research hypothesis.']) -> str:
    res = novelty_admin.initiate_chat(
    novelty_assistant,
        clear_history=True,
        silent=False,
        max_turns=10,
    message=f'''Rate the following research hypothesis\n\n{hypothesis}. \n\nCall the function three times at most, but not in parallel. Wait for the results before calling the next function. ''',
        summary_method="reflection_with_llm",
        summary_args={"summary_prompt" : "Return all the results of the analysis as is."}
    )

    return res.summary


planner.reset()
assistant.reset()
ontologist.reset()
scientist.reset()
critic_agent.reset()


groupchat = autogen.GroupChat(
    agents=[user, planner, assistant, ontologist, scientist,
            hypothesis_agent, outcome_agent, mechanism_agent, design_principles_agent, unexpected_properties_agent, comparison_agent, novelty_agent, critic_agent#sequence_retriever,
               ], messages=[], max_round=50, admin_name='user', send_introductions=True, allow_repeat_speaker=True,
    speaker_selection_method='auto',
)

manager = autogen.GroupChatManager(groupchat=groupchat, 
                                   llm_config=llm_config,
                                   system_message='you dynamically select a speaker.')
