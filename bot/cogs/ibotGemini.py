"""iBot Cog for Discord Bot"""

from qdrant_client import QdrantClient
from dotenv import load_dotenv  # For loading API key from a .env file
import google.generativeai as genai # Import Google Generative AI
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
# import ollama # Remove ollama import
from tqdm.auto import tqdm

# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv()

tqdm.pandas(desc="Processing DataFrame")
config = load_config()

# Configure Google Generative AI
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
    print("Google Generative AI configured successfully.")
except Exception as e:
    print(f"ERROR: Failed to configure Google Generative AI: {e}")
    print("Please ensure GOOGLE_API_KEY is set in your .env file or environment.")
    # You might want to exit or disable LLM functionality if API key is missing
    exit() # Exiting for critical dependency

# Use a suitable Gemini model
LLM_MODEL_NAME = 'gemini-2.5-flash' # Or 'gemini-pro', 'gemini-2.5-pro' depending on needs
# Initialize the GenerativeModel
model = genai.GenerativeModel(LLM_MODEL_NAME)


CLEANING_PATTERN = r'[^a-zA-Z0-9]'

LLM_PERSONA = '''
You are a helpful assistant that helps user to find vacation destinations from the Destinations Dataset.
Your persona is polite, friendly and helpful.
You are not a chef, you are a recipe search assistant.

While Trying to solve the customer's query, you can use the following information:
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

# No need for ollama_client here anymore
# ollama_client = ollama.Client(host='http://localhost:11434',)

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
# Destination Name: {row['Destination']}, {row['Region']}, {row['Country']}

## Basic Info:

{row['Category']} with annual visits of {row['Approximate Annual Tourists']}

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


def generate_metadata(search_query, genai_model: genai.GenerativeModel):
    """
    Generate metadata filter dictionary for the search query using Gemini.
    """
    meta_prompt = f'''
    Given the user query, create a metadata filter dictionary for searching vacation destinations.

    User Query: {search_query}

    Available metadata fields and their possible string values (exact match only):
    - 'Destination': ['European', 'Italy', 'Spain', 'France', 'Austria', 'Gardens', 'Theme Parks', 'Coastal Cities', 'Mountain Ranges', 'Islands', 'Regions', 'Museums', 'Beaches']
    - 'Interests': ['historic significance', 'stunning seaside', 'rich cultural heritage', 'historic landmarks', 'fairy-tale destination', 'medieval architecture', 'magical park with entertainment', 'largest city']
    - 'ComplexityLevel': ['Medium', 'Hard']

    Return only a valid JSON dictionary. Do not include any other text or markdown.
    Example output: {{"Destination": "Italy", "Interests": "historic significance"}}
    '''
    
    response = genai_model.generate_content(
        meta_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=1000,
            response_mime_type="application/json", # Request JSON output
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "Destination": {"type": "STRING", "nullable": True},
                    "Interests": {"type": "STRING", "nullable": True},
                    "ComplexityLevel": {"type": "STRING", "nullable": True}
                }
            }
        )
    )
    try:
        # Gemini's structured response is directly in response.text (as a string)
        metadata_str = response.text
        print(f"Raw Gemini response for metadata: {metadata_str}")
        metadata = json.loads(metadata_str)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError in generate_metadata: {e}")
        print(f"Problematic response: {response.text if 'response' in locals() else 'No response received'}")
        metadata = {} # Return empty dict on error to prevent crashing
    except Exception as e:
        print(f"An unexpected error occurred in generate_metadata: {e}")
        print(f"Problematic response: {response.text if 'response' in locals() else 'No response received'}")
        metadata = {}
    return metadata


def rewrite_query(search_query, genai_model: genai.GenerativeModel):
    """
    Rewrite the query to a more search-friendly term using Gemini.
    """
    prompt = f'''
    Given the user's query regarding vacation or travel destinations,
    rephrase and expand the query to a more search-friendly term.

    User Query: {search_query}

    Provide only the rephrased phrase. Do not add any other information or context.
    This output will be used to search the database for destinations.
    '''
    try:
        response = genai_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=200 # Keep it concise
            )
        )
        resp_text = response.text
        print(f"Rewritten query: {resp_text}") # Debugging line
        return resp_text
    except Exception as e:
        print(f"Error rewriting query with Gemini: {e}")
        return search_query # Fallback to original query on error


def break_query(search_query, genai_model: genai.GenerativeModel):
    """
    Break down the query into multiple subqueries for better search results using Gemini.
    """
    subquery_prompt = f'''
    Given the user's query, break it down into multiple subqueries for searching destinations.

    User Query: {search_query}

    Return only a valid JSON array of strings. Do not include any other information or markdown.
    Example output: ["subquery 1", "subquery 2"]
    '''
    
    response = genai_model.generate_content(
        subquery_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=1000,
            response_mime_type="application/json", # Request JSON array output
            response_schema={
                "type": "ARRAY",
                "items": {"type": "STRING"}
            }
        )
    )
    try:
        subqueries_str = response.text
        print(f"Raw Gemini response for subqueries: {subqueries_str}") # Debugging line
        subqueries = json.loads(subqueries_str)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError in break_query: {e}")
        print(f"Problematic response: {response.text if 'response' in locals() else 'No response received'}")
        subqueries = [search_query] # Fallback to original query on error
    except Exception as e:
        print(f"An unexpected error occurred in break_query: {e}")
        print(f"Problematic response: {response.text if 'response' in locals() else 'No response received'}")
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
        reranking_model.embed_documents(searched_df.page_content.tolist()) # Ensure it's a list
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
        genai_model, # Renamed from o_client to reflect Gemini model
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
        search_query = rewrite_query(search_query, genai_model)

    if flag_ai_metadata:
        metadata = generate_metadata(search_query, genai_model)
        print(f"Generated metadata filter: {metadata}") # Debugging line

    if flag_break_query:
        subqueries = break_query(search_query, genai_model)
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
        columns=doc_columns + columns # Ensure all columns are present
    )

    # Handle potential empty DataFrame after initial search or before groupby
    if searched_df.empty:
        print("No results found after initial vector store search.")
        return pd.DataFrame(columns=columns) # Return empty DataFrame with correct columns

    # Ensure 'Cultural Significance' column exists before groupby
    if 'Cultural Significance' in searched_df.columns:
        searched_df = searched_df.groupby(
            'Cultural Significance').first().reset_index()
    else:
        print("Warning: 'Cultural Significance' column not found for grouping.")
        # If 'Cultural Significance' is missing, proceed without grouping by it
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
    return df_str.apply(lambda x: x.to_markdown(), axis=1).to_list()

# --- NEW FUNCTION FOR CHUNKING MESSAGES ---
def chunk_message(text: str, max_length: int = 2000) -> List[str]:
    """
    Splits a string into chunks of max_length, attempting to break at natural
    points like newlines or sentence endings to avoid cutting words.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    current_chunk = ""
    lines = text.split('\n') # Split by newlines first

    for line in lines:
        # Check if adding the current line (plus a newline character) exceeds max_length
        if len(current_chunk) + len(line) + 1 <= max_length:
            current_chunk += line + '\n'
        else:
            # If current_chunk is not empty, add it to chunks before processing the long line
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # Process the long line itself
            words = line.split(' ')
            temp_line_chunk = ""
            for word in words:
                # Check if adding the current word (plus a space) exceeds max_length
                if len(temp_line_chunk) + len(word) + 1 <= max_length:
                    temp_line_chunk += word + ' '
                else:
                    # If adding the word exceeds, add the current temp_line_chunk
                    if temp_line_chunk:
                        chunks.append(temp_line_chunk.strip())
                    temp_line_chunk = word + ' ' # Start a new temp_line_chunk with the current word

            # Add any remaining part of the line to current_chunk for next iteration
            if temp_line_chunk:
                current_chunk += temp_line_chunk.strip() + '\n'

    # Add any final remaining content
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Final pass to ensure no chunk exceeds max_length, splitting aggressively if needed
    # This handles cases where a single word or a part of a sentence is longer than max_length
    # or if the initial splitting logic didn't perfectly adhere to the limit.
    final_chunks = []
    for chunk in chunks:
        while len(chunk) > max_length:
            # Try to find a natural break point (newline, period, space) within the limit
            split_point = -1
            # Search backwards from max_length to find the last good break
            for i in range(min(max_length, len(chunk)) - 1, -1, -1):
                if chunk[i] in ['\n', '.', ' ']:
                    split_point = i + 1 # Include the delimiter in the first chunk
                    break
            
            if split_point == -1 or split_point == 0: # No natural break found or break at start
                split_point = max_length # Force split at max_length

            final_chunks.append(chunk[:split_point])
            chunk = chunk[split_point:].strip() # Remove leading/trailing whitespace after split
        if chunk:
            final_chunks.append(chunk)

    return final_chunks
# --- END NEW FUNCTION ---


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
            # Construct the absolute path to destinations.csv
            # This assumes destinations.csv is one directory up from the current script.
            csv_path = '../destinations.csv' #os.path.join(os.path.dirname(__file__), '..', 'destinations.csv')
            print(f"Attempting to load destinations.csv from: {csv_path}")

            df = pd.read_csv(csv_path).set_index('Srno')[columns]
            print("destinations.csv loaded successfully.")

            # Ensure 'Approximate Annual Tourists' is a string or compatible type for f-string
            # Convert to string to avoid potential TypeError during f-string formatting
            df['Approximate Annual Tourists'] = df['Approximate Annual Tourists'].astype(str)
            df['Category'] = df['Category'].astype(str)


            data = df[:].progress_apply(convert_to_doc, axis=1)
            self.vector_store_unchunked = Qdrant.from_documents(
                data,
                model_384,
                collection_name="euro-destinations-metadata",
                location=':memory:',
                # url="http://localhost:6333", # Uncomment if using a persistent Qdrant instance
            )
            print("Vector store initialized successfully.")
        except FileNotFoundError:
            print(f"ERROR: destinations.csv not found at {csv_path}.")
            print("Please ensure the 'destinations.csv' file exists in the correct location relative to your bot script.")
            self.vector_store_unchunked = None # Explicitly set to None on failure
        except KeyError as e:
            print(f"ERROR: Missing expected column in destinations.csv: {e}")
            print("Please ensure all required columns are present in your CSV file.")
            self.vector_store_unchunked = None
        except Exception as e:
            print(f"ERROR: An unexpected error occurred during vector store initialization: {e}")
            self.vector_store_unchunked = None # Explicitly set to None on failure


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
        print(f"Interaction user: {interaction.user}")
        print(f"User message: {user_message}")

        if self.vector_store_unchunked is None:
            await interaction.followup.send(
                "Bot is not ready. The destination data could not be loaded. Please inform the bot owner to check the logs for errors during startup.",
                ephemeral=True
            )
            return

        if interaction.user.id not in self._chat_history:
            self._chat_history[interaction.user.id] = []

        chat_messages = self._chat_history[interaction.user.id]

        chat_messages.append(
            {'role': 'user', 'content': user_message}
        )

        # Gemini API expects chat history in a specific format (role: user/model)
        # Convert your existing chat_messages to this format
        gemini_chat_history = []
        for msg in chat_messages:
            # Gemini uses 'model' for bot responses, not 'assistant'
            role = 'user' if msg['role'] == 'user' else 'model'
            gemini_chat_history.append({'role': role, 'parts': [msg['content']]})
        
        # The last message in chat_messages is the current user_message.
        # For the main prompt, we will explicitly add user_message and context.
        # So, the chat_history for the LLM should exclude the current user message.
        # This prevents the current user message from being duplicated in the prompt if it's already in chat_history.
        # For the prompt, we will use the `chat_history` string as before, which is fine.
        # For the Gemini `generate_content` method, we will pass the `gemini_chat_history` as `contents`.
        # The current user message and context will be added as the last part of `contents`.

        # Format chat_history for the PROMPT string (as it was before)
        chat_history_str = '\n'.join(
            [
                f"{msg['role']}: {msg['content']}"
                for msg in chat_messages
            ]
        )

        user_messages_for_search = '\n'.join(
            [
                message['content']
                for message in chat_messages if
                message['role'] == 'user'
            ])
        print(f"User messages for search: {user_messages_for_search}")
        
        try:
            results = search(
                user_messages_for_search,
                genai_model=model, # Pass the Gemini model here
                vector_store=self.vector_store_unchunked,
                reranking_model=model_768,
                n_results=5,
                similarity_threshold=0.1,
                flag_rewrite_query=True,
                flag_ai_metadata=False,
                flag_break_query=True,
                flag_rerank_results=True,
            )
        except Exception as e:
            print(f"ERROR: An error occurred during the search operation: {e}")
            await interaction.followup.send(
                f"An internal error occurred while searching for destinations: {e}. Please try again later.",
                ephemeral=True
            )
            return


        context = '\n---\n'.join(as_cards(results))
        print(f"Context passed to LLM:\n{context}") # Debugging line

        try:
            # Construct the full prompt for the main LLM call
            full_llm_prompt = PROMPT.format(
                llm_persona=LLM_PERSONA,
                objective_prompt=OBJECTIVE_PROMPT,
                user_message=user_message, # Current user message
                chat_history=chat_history_str, # Full chat history string
                context=context,
            )

            # For Gemini, it's better to pass chat history as a list of dicts directly
            # and the current prompt as the last item.
            # The `PROMPT` string is now mainly for structuring the final message part for Gemini.
            contents_for_gemini = gemini_chat_history[:-1] # All previous messages
            contents_for_gemini.append({
                'role': 'user',
                'parts': [full_llm_prompt] # The current user query + context + persona
            })

            response = model.generate_content(
                contents_for_gemini,
                generation_config=genai.GenerationConfig(
                    temperature=0.7, # Adjust temperature as needed
                    max_output_tokens=2000 # Max tokens for the LLM response
                )
            )
            llm_response = response.text # Extract the text content

            # --- MODIFIED FIX FOR 400 Bad Request (error code: 50035) ---
            # Instead of truncating, chunk the message and send multiple parts
            MAX_DISCORD_MESSAGE_LENGTH = 2000
            message_chunks = chunk_message(llm_response, MAX_DISCORD_MESSAGE_LENGTH)

            # Send the first chunk as the initial followup response
            if message_chunks:
                await interaction.followup.send(
                    content=message_chunks[0],
                    # delete_after=300 # Consider if you want all chunks to disappear after 300s
                )
                # Send subsequent chunks as new messages in the same channel
                for i, chunk in enumerate(message_chunks[1:]):
                    await interaction.channel.send(
                        content=chunk,
                        # delete_after=300 # Apply to all if desired
                    )
            else:
                await interaction.followup.send(
                    "The LLM generated an empty response.",
                    ephemeral=True
                )

            # --- END MODIFIED FIX ---

        except Exception as e:
            llm_response = f"An error occurred while generating a response from the LLM: {e}"
            print(f"Error during LLM generation: {e}")
            await interaction.followup.send(
                content=llm_response,
                ephemeral=True # Send error message ephemerally
            )


        # Only append to chat history if the LLM response was successfully generated
        # and not an error message from the try-except block
        if not llm_response.startswith("An error occurred"):
            chat_messages.append({
                'role': 'assistant',
                'content': llm_response, # This will be the full response, not the chunked one
            })

    # --- NEW SLASH COMMAND TO RESET SEARCH ---
    @nextcord.slash_command(
        guild_ids=[config['guild_id']],
        description="Starts a new search by clearing your chat history."
    )
    async def reset_search(
        self,
        interaction: nextcord.Interaction
    ):
        """Clears the user's chat history to start a new search."""
        await interaction.response.defer(ephemeral=True) # Defer and make it ephemeral

        user_id = interaction.user.id
        if user_id in self._chat_history:
            del self._chat_history[user_id]
            await interaction.followup.send("Your search history has been cleared. You can now start a new search!", ephemeral=True)
            print(f"Chat history for user {user_id} cleared.")
        else:
            await interaction.followup.send("You don't have an active search history to clear.", ephemeral=True)
            print(f"User {user_id} tried to clear history, but none existed.")
    # --- END NEW SLASH COMMAND ---


def setup(bot):
    """Setup function to add the cog to the bot."""
    bot.add_cog(GenAIBot(bot))