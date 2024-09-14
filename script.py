import requests
import transformers
from transformers import AutoTokenizer
import re
import string
import pandas as pd
# import nltk
import torch

from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from crewai import Agent , Task ,Crew , Process

from params import *

import json



class Summarizer():
    
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.summarizer_agent = None
        self.summarizer_task = None
        
    def init_model(self):
        
        """Initialize model and tokenizer"""
        
        self.model = Ollama(model= self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        
    def clean_text(self, text):
        
        """clean text --> returns text"""
        
        text = re.sub(r'[^A-Za-z0-9\s.\(\)[\]{\}]+', '' , text)
        text = text.lower()
        text = " ".join(text.split())
        return text
    
    def count_tokens(self, text):
        tokens = self.tokenizer.encode(text , add_special_tokens = True)
        return(len(tokens))
        
    def summarizer(self):
        
        """set up the summarizer agent and task"""
        
        self.summarizer_agent= Agent(
        role='Summarizer',
        goal="Summarize the full text in no more than 2000 tokens. Include main challenges and successes in bullet points with metrics",
        verbose=True,
        memory=True,
        backstory=(
            "Equipped with advanced summarization techniques, "
            "the goal is to distill complex information into concise summaries."
        ),
        llm=self.model,
        allow_delegation=False  # No need for search_tool if performing local summarization
    )
        
    def create_summary_task(self, description_template, expected_output_template, stock_transcript):
        
        description = description_template.format(stock_transcript=stock_transcript)
        expected_output = expected_output_template.format(stock_transcript=stock_transcript)
        
        retun= Task(
            description=description,
            expected_output=expected_output,
            agent=self.summarizer_agent
        )
        
    def run_crew(self):
        
        self.summarizer_task  = self.create_summary_task(
            description_template="Summarize the transcript of {stock_transcript}.",
            expected_output_template="A concise summary of the transcript of {stock_transcript}.",
            stock_transcript=clean_transcripts,
            agent=self.summarizer_agent
        )
        
        crew = Crew(
            agents=[self.summarizer_agent],
            tasks=[self.summarizer_task],
            process=Process.sequential
        )
        
        crew_result = crew.kickoff()


if __name__ == '__main__':
    
    # RETREIEVING DATA
    js = json.load(open("json_data/transcripts_2024.json"))
    
    df = pd.DataFrame()

    for i in js:
        df_sub = pd.DataFrame(i)
        df = pd.concat([df,df_sub])
    
    #INSTANTIATE THE SUMMARIZER
    summ = Summarizer(model= "llama3.1:latest", tokenizer="allenai/longformer-base-4096")
    summ.init_model()
    
    df['clean_content'] = df['content'].apply(summ.clean_text)
    df['count_tokens'] = df['clean_content'].apply(summ.count_tokens)
    
    clean_transcripts = df[['clean_content']][:2]
    
    summ.summarizer()
    
    
    
    
    