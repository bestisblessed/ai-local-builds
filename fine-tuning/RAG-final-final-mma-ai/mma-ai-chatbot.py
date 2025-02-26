import ollama
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any
import subprocess
# Clean imports for llama_index - these should work with newer versions
from llama_index.core import Document, VectorStoreIndex, ServiceContext
from llama_index.llms.ollama import Ollama
# from llama_index.core.embeddings import HuggingFaceEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Load data
fighter_info_df = pd.read_csv('data-raw/fighter_info.csv')
event_data_df = pd.read_csv('data-raw/event_data_sherdog.csv')

# Process fighter data
def clean_fighter_data(df):
    # Convert column names to lowercase and replace spaces with underscores
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Clean up fighter names
    if 'fighter' in df.columns:
        df.rename(columns={'fighter': 'name'}, inplace=True)
    
    return df

fighter_info_df = clean_fighter_data(fighter_info_df)

# Process event data
def clean_event_data(df):
    # Convert column names to lowercase and replace spaces with underscores
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    return df

event_data_df = clean_event_data(event_data_df)

# Create fighter profiles
def create_fighter_profiles(fighter_df, event_df):
    profiles = {}
    
    for _, fighter in fighter_df.iterrows():
        name = fighter['name']
        
        # Get fighter's fights
        fighter_fights = event_df[(event_df['fighter_1'] == name) | (event_df['fighter_2'] == name)]
        
        # Calculate win/loss record from events
        wins = fighter_fights[fighter_fights['winning_fighter'] == name].shape[0]
        losses = fighter_fights[(fighter_fights['fighter_1'] == name) | 
                               (fighter_fights['fighter_2'] == name)].shape[0] - wins
        
        # Create profile
        profile = {
            'name': name,
            'height': fighter.get('height', 'Unknown'),
            'weight_class': fighter.get('weight_class', 'Unknown'),
            'wins': fighter.get('wins', wins),
            'losses': fighter.get('losses', losses),
            'country': fighter.get('nationality', 'Unknown'),
            'team': fighter.get('association', 'Unknown'),
            'fights': []
        }
        
        # Add fight history
        for _, fight in fighter_fights.iterrows():
            opponent = fight['fighter_2'] if fight['fighter_1'] == name else fight['fighter_1']
            result = 'Win' if fight['winning_fighter'] == name else 'Loss'
            
            fight_info = {
                'opponent': opponent,
                'result': result,
                'method': fight['winning_method'],
                'event': fight['event_name'],
                'date': fight['event_date'],
                'round': fight['winning_round'],
                'time': fight['winning_time']
            }
            
            profile['fights'].append(fight_info)
            
        profiles[name] = profile
    
    return profiles

fighter_profiles = create_fighter_profiles(fighter_info_df, event_data_df)

# Create documents for RAG
def create_documents(fighter_profiles, event_df):
    documents = []
    
    # Fighter profile documents
    for name, profile in fighter_profiles.items():
        # Create a detailed text representation of the fighter
        fight_history = "\n".join([
            f"- {fight['event']} ({fight['date']}): {fight['result']} against {fight['opponent']} via {fight['method']} in round {fight['round']} at {fight['time']}"
            for fight in profile['fights']
        ])
        
        text = f"""
        Fighter Profile: {name}
        Height: {profile['height']}
        Weight Class: {profile['weight_class']}
        Record: {profile['wins']}-{profile['losses']}
        Country: {profile['country']}
        Team: {profile['team']}
        
        Fight History:
        {fight_history}
        """
        
        documents.append(Document(text=text, metadata={"type": "fighter", "name": name}))
    
    # Event documents
    for _, event in event_df.drop_duplicates('event_name').iterrows():
        event_fights = event_df[event_df['event_name'] == event['event_name']]
        
        fights_text = "\n".join([
            f"- {fight['fighter_1']} vs {fight['fighter_2']}: {fight['winning_fighter']} won via {fight['winning_method']} in round {fight['winning_round']} at {fight['winning_time']}"
            for _, fight in event_fights.iterrows()
        ])
        
        text = f"""
        Event: {event['event_name']}
        Location: {event['event_location']}
        Date: {event['event_date']}
        
        Fights:
        {fights_text}
        """
        
        documents.append(Document(text=text, metadata={"type": "event", "name": event['event_name']}))
    
    # Add statistical documents
    # Weight class distribution
    weight_class_counts = fighter_info_df['weight_class'].value_counts().to_dict()
    weight_class_text = "\n".join([f"- {wc}: {count} fighters" for wc, count in weight_class_counts.items()])
    
    documents.append(Document(text=f"""
    Weight Class Distribution:
    {weight_class_text}
    """, metadata={"type": "stats", "name": "weight_classes"}))
    
    # Winning methods distribution
    winning_methods = event_data_df['winning_method'].value_counts().to_dict()
    winning_methods_text = "\n".join([f"- {method}: {count} fights" for method, count in winning_methods.items()])
    
    documents.append(Document(text=f"""
    Winning Methods Distribution:
    {winning_methods_text}
    """, metadata={"type": "stats", "name": "winning_methods"}))
    
    return documents

# Chat function using direct Ollama API instead of llama_index
def chat():
    print("🥊 MMA AI Chatbot 🥊")
    print("Ask me anything about MMA fighters and events! Type 'exit' to quit.")
    
    # Prepare context from fighter profiles and events
    context = ""
    for name, profile in list(fighter_profiles.items())[:50]:  # Limit to 50 fighters for context size
        context += f"Fighter: {name}, Weight: {profile['weight_class']}, Record: {profile['wins']}-{profile['losses']}\n"
    
    for _, event in event_data_df.drop_duplicates('event_name').head(20).iterrows():
        context += f"Event: {event['event_name']} ({event['event_date']}): {event['fighter_1']} vs {event['fighter_2']}\n"
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        
        try:
            # Create prompt with context
            prompt = f"""
            You are an MMA expert assistant. Use the following information to answer the user's question:
            
            {context}
            
            User question: {user_input}
            
            If you don't know the answer based on the provided information, say so and provide general MMA knowledge instead.
            """
            
            # Get response from Ollama
            response = ollama.chat(model="llama3", messages=[
                {"role": "system", "content": "You are an MMA expert assistant."},
                {"role": "user", "content": prompt}
            ])
            
            print(f"\nMMA AI: {response['message']['content']}")
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    # Check if Ollama is running and has the required model
    try:
        # Use subprocess to check if Ollama is running and has the required model
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        
        # Check if llama3 model exists in the output
        if 'llama3' not in result.stdout:
            print("Downloading llama3 model. This might take a while...")
            print("Please run 'ollama pull llama3' manually if this fails.")
            try:
                subprocess.run(['ollama', 'pull', 'llama3'], check=True)
                print("Successfully pulled llama3 model.")
            except subprocess.SubprocessError:
                print("Could not automatically pull the model. Please run 'ollama pull llama3' manually.")
        else:
            print("llama3 model is available.")
    except FileNotFoundError:
        print("Error: Ollama command not found.")
        print("Please make sure Ollama is installed and in your PATH.")
        print("You can install it from https://ollama.com/")
        exit(1)
    except subprocess.SubprocessError as e:
        print(f"Error running Ollama: {e}")
        print("Please make sure Ollama is installed and running.")
        print("You can install it from https://ollama.com/")
        exit(1)
    
    # Start the chat
    chat() 