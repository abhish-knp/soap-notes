import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.callbacks.manager import CallbackManagerForLLMRun

# Dataset and evaluation
from datasets import load_dataset
import evaluate
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# UI
import streamlit as st
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SOAPNote:
    """Structure for SOAP note components"""
    subjective: str
    objective: str
    assessment: str
    plan: str
    
    def to_string(self) -> str:
        return f"""SUBJECTIVE:
{self.subjective}

OBJECTIVE:
{self.objective}

ASSESSMENT:
{self.assessment}

PLAN:
{self.plan}"""

class SOAPDatasetProcessor:
    """Handles loading and preprocessing of SOAP notes dataset"""
    
    def __init__(self, dataset_name: str = "adesouza1/soap_notes"):
        self.dataset_name = dataset_name
        self.dataset = None
        
    def load_dataset(self) -> pd.DataFrame:
        """Load the SOAP notes dataset"""
        try:
            self.dataset = load_dataset(self.dataset_name)
            df = pd.DataFrame(self.dataset['train'])
            logger.info(f"Loaded dataset with {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            # Fallback to mock data for demonstration
            return self._create_mock_dataset()
    
    def _create_mock_dataset(self) -> pd.DataFrame:
        """Create mock SOAP notes for demonstration"""
        mock_data = [
            {
                "conversation": "Patient: I've been having chest pain for 2 days. Doctor: Can you describe the pain? Patient: It's sharp and gets worse when I breathe. Doctor: Any shortness of breath? Patient: Yes, especially when lying down.",
                "soap_note": "SUBJECTIVE:\nPatient reports chest pain for 2 days, sharp in nature, worsens with breathing. Associated shortness of breath, particularly when supine.\n\nOBJECTIVE:\nVital signs stable. Patient appears uncomfortable.\n\nASSESSMENT:\nChest pain, possibly pleuritic. Rule out pneumonia, pulmonary embolism.\n\nPLAN:\nChest X-ray, CBC, D-dimer. Follow up in 24 hours."
            },
            {
                "conversation": "Patient: I have a terrible headache and fever. Doctor: How long have you had these symptoms? Patient: Started yesterday morning. Doctor: Any neck stiffness? Patient: A little bit.",
                "soap_note": "SUBJECTIVE:\nPatient complains of severe headache and fever since yesterday morning. Reports mild neck stiffness.\n\nOBJECTIVE:\nFever 101.5°F, patient appears ill.\n\nASSESSMENT:\nFever with headache and neck stiffness. Rule out meningitis.\n\nPLAN:\nLumbar puncture, blood cultures, empiric antibiotics."
            }
        ]
        return pd.DataFrame(mock_data)
    
    def prepare_documents(self, df: pd.DataFrame) -> List[Document]:
        """Prepare documents for vector storage"""
        documents = []
        for idx, row in df.iterrows():
            # Create document with conversation and SOAP note
            content = f"Conversation: {row['conversation']}\n\nSOAP Note: {row['soap_note']}"
            doc = Document(
                page_content=content,
                metadata={
                    "id": idx,
                    "conversation": row['conversation'],
                    "soap_note": row['soap_note']
                }
            )
            documents.append(doc)
        return documents

class VectorStoreManager:
    """Manages vector storage and retrieval using FAISS or Chroma"""
    
    def __init__(self, store_type: str = "faiss"):
        self.store_type = store_type
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def create_vector_store(self, documents: List[Document], persist_path: str = "./vector_store"):
        """Create and persist vector store"""
        # Split documents
        split_docs = self.text_splitter.split_documents(documents)
        
        if self.store_type == "faiss":
            self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
            self.vector_store.save_local(persist_path)
        elif self.store_type == "chroma":
            self.vector_store = Chroma.from_documents(
                split_docs, 
                self.embeddings,
                persist_directory=persist_path
            )
            self.vector_store.persist()
        
        logger.info(f"Created {self.store_type} vector store with {len(split_docs)} chunks")
    
    def load_vector_store(self, persist_path: str = "./vector_store"):
        """Load existing vector store"""
        try:
            if self.store_type == "faiss":
                self.vector_store = FAISS.load_local(persist_path, self.embeddings)
            elif self.store_type == "chroma":
                self.vector_store = Chroma(
                    persist_directory=persist_path,
                    embedding_function=self.embeddings
                )
            logger.info(f"Loaded {self.store_type} vector store")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
        return True
    
    def get_retriever(self, k: int = 3):
        """Get retriever for similarity search"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        return self.vector_store.as_retriever(search_kwargs={"k": k})

class SOAPGenerator:
    """Main SOAP note generation system using RAG"""
    
    def __init__(self, llm_model: str = "gpt-3.5-turbo", vector_store_type: str = "faiss"):
        self.llm_model = llm_model
        self.vector_manager = VectorStoreManager(vector_store_type)
        self.llm = self._initialize_llm()
        self.prompt_template = self._create_prompt_template()
        self.chain = None
        
    def _initialize_llm(self):
        """Initialize the language model"""
        if "gpt" in self.llm_model.lower():
            return ChatOpenAI(model_name=self.llm_model, temperature=0.1)
        else:
            # For other models, you might need to use different LangChain integrations
            return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)
    
    def _create_prompt_template(self):
        """Create prompt template for SOAP note generation"""
        template = """You are a medical assistant specialized in creating SOAP notes from doctor-patient conversations.

Use the following examples as reference for the format and style:

{context}

Based on the conversation below, generate a well-structured SOAP note following the standard format:
- SUBJECTIVE: Patient's reported symptoms and history
- OBJECTIVE: Observable findings and measurements
- ASSESSMENT: Clinical impression and diagnosis
- PLAN: Treatment plan and follow-up

Conversation:
{question}

Generate a comprehensive SOAP note:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def setup_rag_chain(self, documents: List[Document]):
        """Setup the RAG chain with vector store and LLM"""
        # Create or load vector store
        if not self.vector_manager.load_vector_store():
            logger.info("Creating new vector store...")
            self.vector_manager.create_vector_store(documents)
        
        # Create retrieval chain
        retriever = self.vector_manager.get_retriever(k=3)
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True
        )
        
        logger.info("RAG chain setup complete")
    
    def generate_soap_note(self, conversation: str) -> Dict[str, Any]:
        """Generate SOAP note from conversation"""
        if self.chain is None:
            raise ValueError("RAG chain not initialized. Call setup_rag_chain first.")
        
        try:
            result = self.chain({"query": conversation})
            soap_text = result["result"]
            source_docs = result["source_documents"]
            
            return {
                "soap_note": soap_text,
                "source_documents": [doc.page_content for doc in source_docs],
                "success": True
            }
        except Exception as e:
            logger.error(f"Error generating SOAP note: {e}")
            return {"error": str(e), "success": False}

class SOAPEvaluator:
    """Evaluate generated SOAP notes using ROUGE and BLEU metrics"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
    
    def evaluate_single(self, generated: str, reference: str) -> Dict[str, float]:
        """Evaluate a single generated SOAP note"""
        # ROUGE scores
        rouge_scores = self.rouge_scorer.score(reference, generated)
        
        # BLEU score
        reference_tokens = reference.lower().split()
        generated_tokens = generated.lower().split()
        bleu_score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=self.smoothing)
        
        return {
            "rouge1_f": rouge_scores['rouge1'].fmeasure,
            "rouge2_f": rouge_scores['rouge2'].fmeasure,
            "rougeL_f": rouge_scores['rougeL'].fmeasure,
            "bleu": bleu_score
        }
    
    def evaluate_batch(self, generated_notes: List[str], reference_notes: List[str]) -> Dict[str, float]:
        """Evaluate a batch of generated SOAP notes"""
        if len(generated_notes) != len(reference_notes):
            raise ValueError("Generated and reference notes must have same length")
        
        scores = [self.evaluate_single(gen, ref) for gen, ref in zip(generated_notes, reference_notes)]
        
        # Calculate averages
        avg_scores = {}
        for metric in scores[0].keys():
            avg_scores[f"avg_{metric}"] = np.mean([s[metric] for s in scores])
        
        return avg_scores

# CLI Interface
def cli_interface():
    """Command line interface for SOAP note generation"""
    parser = argparse.ArgumentParser(description="Generate SOAP notes from conversations")
    parser.add_argument("--conversation", "-c", required=True, help="Doctor-patient conversation text")
    parser.add_argument("--model", "-m", default="gpt-3.5-turbo", help="LLM model to use")
    parser.add_argument("--vector-store", "-v", default="faiss", choices=["faiss", "chroma"], help="Vector store type")
    
    args = parser.parse_args()
    
    # Initialize system
    processor = SOAPDatasetProcessor()
    df = processor.load_dataset()
    documents = processor.prepare_documents(df)
    
    generator = SOAPGenerator(llm_model=args.model, vector_store_type=args.vector_store)
    generator.setup_rag_chain(documents)
    
    # Generate SOAP note
    result = generator.generate_soap_note(args.conversation)
    
    if result["success"]:
        print("Generated SOAP Note:")
        print("=" * 50)
        print(result["soap_note"])
        print("\nSource Documents Used:")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"\n{i}. {doc[:200]}...")
    else:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    # Set up environment variables (you'll need to add your API keys)
    # os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
    
    cli_interface()
  
