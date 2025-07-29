"""iBot Cog for Discord Bot"""

from qdrant_client import QdrantClient
from dotenv import load_dotenv  # For loading API key from a .env file
import google.generativeai as genai
from langchain_qdrant import Qdrant  # Qdrant Vector Store Wrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import pandas as pd
import json
from tqdm.auto import tqdm
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import nextcord
from datetime import datetime
from nextcord.ext import commands
from typing import List
from helpers.utils import load_config
import ollama
from tqdm.auto import tqdm

tqdm.pandas(desc="Processing DataFrame")
config = load_config()

OLLAMA_MODEL_NAME = 'llama3.2:latest'
CLEANING_PATTERN = r'[^a-zA-Z0-9]'

LLM_PERSONA = '''
You are a helpful assistant that helps user to find vacation destinations from the Destinations Dataset.
Your pesona is polite, friendly and helpful.
You are not a chef, you are a recipe search assistant.

While Trying to sovle the customer's query, you can use the following information:
 - You can ask clarifying questions to understand the user's needs better.
 - You can use the Destinations Dataset to find destinations that match the user's request.
 - You can provide additional information about the destinations, such as famous food, language, and descriptions.
 - You can suggest alternatives or modifications to the destinations based on the user's preferences.

You are not allowed to:
 - Provide medical or dietary advice.
 - Make assumptions about the user's dietary restrictions or preferences without asking.
 - Provide personal opinions or recommendations that are not based on the dataset.
 - Provide vacation destinations that are not in the destinations Dataset.
 - Provide destinations that are not relevant to the user's query.

> RESPOND ONLY IN ENGLISH, DO NOT RESPOND IN ANY OTHER LANGUAGE.
You are not allowed to use any other language than English.
'''

OBJECTIVE_PROMPT = '''
Your objective is to help the user find recipes from the destinations Dataset.
Your hidden side task is also to suggest, cross sell and/upsell cities.
'''

PROMPT = '''
Your persona:
{llm_persona}

Your objective:
{objective_prompt}

User Query: {user_message}

Chat History:
{chat_history}

Searched Context from Database:
{context}
'''

RAG_PROMPT = '''
User Query: {user_message}
Chat History:
{chat_history}

'''


columns = ['Destination','Region','Country','Category',
            'Approximate Annual Tourists','Currency',
            'Majority Religion','Famous Foods','Language',
            'Best Time to Visit','Cost of Living','Safety',
            'Cultural Significance','Description']

doc_columns = ['score', 'page_content',]

ollama_client = ollama.Client(host='http://localhost:11434',)
model_768 = HuggingFaceEmbeddings(
    model_name="sentence-transformers/LaBSE",
)

model_384 = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

model_64 = HuggingFaceEmbeddings(
    model_name="ClovenDoug/tiny_64_all-MiniLM-L6-v2",
)


def convert_to_doc(row):
    """
    Convert a row of the DataFrame to a Document object."""
    doc = Document(
        page_content=f'''
# Destination Name: {row['Destination']}
>

## Basic Info:

{row['Category']+ ' with annual visits of '+ row['Approximate Annual Tourists']}

## Required Information for visiting:

{row['Description']}
''',
        metadata={
            'Currency': row['Currency'],
            'Majority Religion': row['Majority Religion'],
            'Famous Foods': row['Famous Foods'],
            'Language': row['Language'],
            'Cost of Living': row['Cost of Living'],
            'Safety': row['Safety'],
            'Cultural Significance': row['Cultural Significance'],
            'Best Time to Visit': row['Best Time to Visit'],
        }
    )

    return doc


def generate_metadata(search_query, o_client: ollama.Client):
    """
    Generate metadata filter dictionary for the search query."""
    meta_prompt = f'''
    Given below the user request for queries, create metadata filter dictionary
    for the search.

    user query: {search_query}

    > provide only and only a simple phrase for the user query, do not add any
    > other information or context.
    > this output will be used to filter the recipes.

    available metadata:
    - 'Destination': string: ['European', 'Italy', 'Spain',
        'France', 'Austria', 'Gardens', 'Theme Parks', 'Coastal Cities',
        'Mountain Ranges', 'Islands', 'Regions',
        'Museums', 'Beaches',]
    - 'Interests': string: ['historic significance', 'stunning seaside',
        'rich cultural heritage', 'historic landmarks', 'fairy-tale destination',
        'medieval architecture', 'magical park with entertainment', 'largest city',]
    - 'ComplexityLevel': string: ['Medium', 'Hard']

    We can do exact match only.

    RESPOND ONLY WITH THE JSON DICTIONARY. DO NOT INCLUDE ANY OTHER TEXT OR MARKDOWN.
    EXPECTED OUTPUT FORMAT: JSON
    '''

    resp = o_client.generate(
        model=OLLAMA_MODEL_NAME,
        prompt=meta_prompt,
        options={
            'temperature': 0.1,
            'max_tokens': 1000,
            # 'stop_sequences': ['```json', '```']
        }
    ).response
    print(f"Raw Ollama response for metadata: {resp}") # Debugging line

    try:
       # Attempt to find JSON within a markdown block first
        json_match = re.search(r'```json\s*(\{.*\})\s*```', resp, re.DOTALL)
        if json_match:
            metadata = json.loads(json_match.group(1))
        else:
            # If no markdown block, try to parse the whole response
            metadata = json.loads(resp.strip().replace("'", '"'))
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError in generate_metadata: {e}")
        print(f"Problematic response: {resp}")
        metadata = {} # Return empty dict on error to prevent crashing
    except Exception as e:
        print(f"An unexpected error occurred in generate_metadata: {e}")
        print(f"Problematic response: {resp}")
        metadata = {}
    return metadata


def rewrite_query(search_query, o_client):
    """
    Rewrite the query to a more search-friendly term."""
    prompt = f'''
    Given below the user request for queries regarding vacation or travel destinations ,
    rephrase and expand the query to a more search friendly term.

    user query: {search_query}

    > provide only and only a simple phrase for the user query, do not add any other information or context.
    > this output will be used to search the database for destinations.
    RESPOND ONLY WITH THE JSON ARRAY. DO NOT INCLUDE ANY OTHER TEXT OR MARKDOWN.
    '''
    resp = o_client.generate(
        model=OLLAMA_MODEL_NAME,
        prompt=prompt,
    ).response
    print(resp)
    return resp


def break_query(search_query, o_client):
    """
    Break down the query into multiple subqueries for better search results."""
    subquery_prompt = f'''
    Given below the user request for queries, break down the query into multiple subqueries.
    user query: {search_query}
    > provide only and only a simple phrase for the user query, do not add any other information or context.
    > this output will be used to search the database for recipes.
    > respond with a valid json array of strings, do not add any other information or context.
    '''

    resp = o_client.generate(
        model=OLLAMA_MODEL_NAME,
        prompt=subquery_prompt,).response
    print(f"Raw Ollama response for subqueries: {resp}") # Debugging line
    
    try:
        # Attempt to find JSON within a markdown block first
        json_match = re.search(r'```json\s*(\[.*\])\s*```', resp, re.DOTALL)
        if json_match:
            subqueries = json.loads(json_match.group(1))
        else:
            # If no markdown block, try to parse the whole response
            subqueries = json.loads(resp.strip().replace("'", '"'))
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError in break_query: {e}")
        print(f"Problematic response: {resp}")
        subqueries = [search_query] # Fallback to original query on error
    except Exception as e:
        print(f"An unexpected error occurred in break_query: {e}")
        print(f"Problematic response: {resp}")
        subqueries = [search_query]
        
    return subqueries


def rerank_results(
        search_query,
        searched_df,
        reranking_model
):
    """
    Rerank the results based on the reranking model."""
    if searched_df.empty:
        return searched_df
    new_doc_embeddings = np.array(
        reranking_model.embed_documents(searched_df.page_content.tolist())
    )

    query_embedding = np.array(
        reranking_model.embed_query(search_query)
    )

    similarity_scores = cosine_similarity(
        query_embedding.reshape(1, -1),
        new_doc_embeddings
    )
    searched_df['rerank_score'] = similarity_scores[0].tolist()
    return searched_df


def search(
        search_query,
        o_client,
        vector_store,
        reranking_model,
        n_results=10,
        similarity_threshold=0.1,
        flag_rewrite_query=True,
        flag_ai_metadata=True,
        flag_break_query=True,
        flag_rerank_results=True,
):
    """
    Search for the given query in the vector store and return the top n results.
    """
    metadata = {}  # Empty metadata
    subqueries = [search_query]

    if flag_rewrite_query:
        search_query = rewrite_query(search_query, o_client)

    if flag_ai_metadata:
        metadata = generate_metadata(search_query, o_client)
        print(f"Generated metadata filter: {metadata}") # Debugging line

    if flag_break_query:
        subqueries = break_query(search_query, o_client)
        print(f"Generated subqueries: {subqueries}") # Debugging line

    ret_docs = []

    for subquery in subqueries:
        # Ensure filter is only applied if metadata is not empty
        if metadata:
            ret_docs += vector_store.similarity_search_with_score(
                subquery,
                k=n_results,
                score_threshold=similarity_threshold,
                filter=metadata
            )
        else:
            ret_docs += vector_store.similarity_search_with_score(
                subquery,
                k=n_results,
                score_threshold=similarity_threshold,
            )


    searched_df = pd.DataFrame(
        [
            {
                'score': score,
                **doc.metadata,
                'page_content': doc.page_content,
            } for doc, score in ret_docs
        ],
        columns=doc_columns+columns
    )

    # searched_df = searched_df.groupby(
    #     'Description').first().reset_index()
    
     # Handle potential empty DataFrame after initial search or before groupby
    if searched_df.empty:
        print("No results found after initial vector store search.")
        return pd.DataFrame(columns=columns) # Return empty DataFrame with correct columns

    # Ensure 'Description' column exists before groupby
    if 'Description' in searched_df.columns:
        searched_df = searched_df.groupby(
            'Description').first().reset_index()
    else:
        print("Warning: 'Description' column not found for grouping.")
        # If 'Description' is missing, proceed without grouping by it
        # Or handle as appropriate for your data structure
    
    searched_df['rerank_score'] = searched_df['score']

    if flag_rerank_results:
        searched_df = rerank_results(
            search_query,
            searched_df=searched_df,
            reranking_model=reranking_model,
        ).sort_values(
            'rerank_score',
            ascending=False,
        )

    # return searched_df.head(n_results).round(2)[
    #     [
    #         'Destination',
    #         'Region',
    #         'Country',
    #         'Category',
    #         'Approximate Annual Tourists','Currency',
    #         'Majority Religion','Famous Foods','Language',
    #         'Best Time to Visit','Cost of Living','Safety',
    #         'Cultural Significance','Description'
    #     ]
    # ]
    
    
    # Ensure the returned DataFrame has the exact columns requested, even if some are missing
    final_columns = [
        'Destination', 'Region', 'Country', 'Category',
        'Approximate Annual Tourists', 'Currency',
        'Majority Religion', 'Famous Foods', 'Language',
        'Best Time to Visit', 'Cost of Living', 'Safety',
        'Cultural Significance', 'Description'
    ]
    # Select existing columns and add missing ones as NaN
    for col in final_columns:
        if col not in searched_df.columns:
            searched_df[col] = np.nan # Add missing columns as NaN

    return searched_df.head(n_results).round(2)[final_columns]


def as_cards(df):
    """Convert a DataFrame to a list of markdown strings for Discord cards.
    """
    if df.empty:
        return ["No relevant destinations found based on your query."]
    # Ensure all values are strings before calling to_markdown
    df_str = df.astype(str)
    return df.apply(lambda x: x.to_markdown(), axis=1).to_list()


class GenAIBot(commands.Cog):
    """A simple Discord bot cog that captures all messages and provides a
    slash command."""

    def __init__(
        self,
        bot: commands.Bot
    ) -> None:
        super().__init__()
        self.bot = bot
        self._chat_history = {}
        self.vector_store_unchunked = None # Initialize as None

        # Load data and initialize vector store
        try:
            # Ensure the path to the CSV is correct relative to where the bot is run
            # If the CSV is in the same directory as this script, you might just need 'destinations.csv'
            # If it's in a parent directory, '../destinations.csv' might be correct.
            # Consider using os.path.join(os.path.dirname(__file__), '..', 'destinations.csv') for robustness.
            csv_path = '../destinations.csv' #os.path.join(os.path.dirname(__file__), '..', 'destinations.csv')   
            df = pd.read_csv(csv_path).set_index('Srno')[columns]
            print("destinations.csv loaded successfully.")

            # Ensure 'Approximate Annual Tourists' is a string or compatible type for f-string
            # Convert to string to avoid potential TypeError during f-string formatting
            #df['Approximate Annual Tourists'] = df['Approximate Annual Tourists'].astype(str)
            #df['Category'] = df['Category'].astype(str)


            data = df[:].progress_apply(convert_to_doc, axis=1)
            self.vector_store_unchunked = Qdrant.from_documents(
                data,
                model_384,
                collection_name="euro-destinations-metadata",
                location=':memory:',
                # url="http://localhost:6333", # Uncomment if using a persistent Qdrant instance
            )
            print("Vector store initialized successfully.")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            self.vector_store_unchunked = None    
        

    @commands.Cog.listener()
    async def on_message(
        self,
        message: nextcord.Message
    ):
        """Capturing All messages"""
        print(message)

        if message.author == self.bot.user or message.author.bot:
            return

    @nextcord.slash_command(
        guild_ids=[config['guild_id']],
        description="Execute Command")
    
    async def explore_yourself(
            self,
            interaction: nextcord.Interaction,
            user_message: str
    ):
        """A slash command to start ragging."""
        await interaction.response.defer()
        print(interaction.user)
        print(user_message)
        
        if self.vector_store_unchunked is None:
            await interaction.followup.send("Bot is working, as I'm poor and my server is slow and old.", ephemeral=True)            

        if interaction.user.id not in self._chat_history:
            self._chat_history[interaction.user.id] = []

        chat_messages = self._chat_history[interaction.user.id]

        chat_messages.append(
            {'role': 'user', 'content': user_message}
        )

        chat_history = '\n'.join(
            [
                f"{msg['role']}: {msg['content']}"
                for msg in chat_messages
            ]
        )
        user_messages = '\n'.join(
            [
                message['content']
                for message in chat_messages if
                message['role'] == 'user'
            ])
        print(user_messages)
        results = search(
            user_messages,
            # RAG_PROMPT.format(
            #     user_message=user_message,
            #     chat_history=chat_messages,
            # ),
            o_client=ollama_client,
            vector_store=self.vector_store_unchunked,
            reranking_model=model_768,
            n_results=5,
            similarity_threshold=0.1,
            flag_rewrite_query=True,
            flag_ai_metadata=False,
            flag_break_query=True,
            flag_rerank_results=True,
        )

        context = '\n---\n'.join(as_cards(results))
        print(f"Context passed to LLM:\n{context}") # Debugging line

        # llm_response = ollama_client.generate(
        #     model=OLLAMA_MODEL_NAME,
        #     prompt=PROMPT.format(
        #         llm_persona=LLM_PERSONA,
        #         objective_prompt=OBJECTIVE_PROMPT,
        #         user_message=user_message,
        #         chat_history=chat_history,
        #         context=context,
        #     ),
        #     stream=False,
        # ).response
        try:
            llm_response = ollama_client.generate(
                model=OLLAMA_MODEL_NAME,
                prompt=PROMPT.format(
                    llm_persona=LLM_PERSONA,
                    objective_prompt=OBJECTIVE_PROMPT,
                    user_message=user_message,
                    chat_history=chat_history,
                    context=context,
                ),
                stream=False,
            ).response
        except Exception as e:
            llm_response = f"An error occurred while generating a response: {e}"
            print(f"Error during LLM generation: {e}")


        chat_messages.append({
            'role': 'assistant',
            'content': llm_response,
        })

        await interaction.followup.send(
            content=llm_response,
            delete_after=300
        )


def setup(bot):
    """Setup function to add the cog to the bot."""
    bot.add_cog(GenAIBot(bot))
