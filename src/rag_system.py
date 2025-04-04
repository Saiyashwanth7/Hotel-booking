# src/rag_system.py
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
import logging

class RAGSystem:
    def __init__(self, data_df: pd.DataFrame, embedding_model: str = "all-MiniLM-L6-v2"):
        self.data = data_df
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.documents = []
        self.query_history = []
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("RAGSystem")
        
        # Initialize LLM (requires HuggingFace API token)
        self.huggingface_api_token = os.environ.get("HUGGINGFACE_API_TOKEN")
        if not self.huggingface_api_token:
            self.logger.warning("HUGGINGFACE_API_TOKEN not found in environment variables")
        
        self.llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingface_api_token=self.huggingface_api_token,
            model_kwargs={"temperature": 0.1, "max_length": 512}
        )
        
        # Define prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template="""
            You are an AI assistant that helps answer questions about hotel booking data.
            
            Here is the relevant context from the hotel booking dataset:
            {context}
            
            Please answer the following question based on the context above:
            {question}
            
            If the information is not available in the context, please say "I don't have enough information to answer this question."
            """
        )
    
    def create_document_embeddings(self):
        """Create document embeddings for the hotel booking data"""
        self.logger.info("Creating document embeddings...")
        
        # Create text chunks from the DataFrame
        documents = []
        for idx, row in self.data.iterrows():
            if idx % 100 == 0:  # Create chunks of data
                chunk = self.data.iloc[idx:idx+100]
                
                # Create monthly summaries
                if 'arrival_date' in chunk.columns:
                    month = pd.Timestamp(chunk['arrival_date'].iloc[0]).strftime('%B %Y')
                    avg_price = chunk['adr'].mean()
                    total_revenue = chunk['revenue'].sum()
                    cancellations = chunk['is_canceled'].sum()
                    
                    doc_text = f"For {month}, the average daily rate was {avg_price:.2f}, " \
                              f"total revenue was {total_revenue:.2f}, " \
                              f"and there were {cancellations} cancellations."
                    documents.append({"text": doc_text, "metadata": {"month": month, "chunk_id": idx}})
            
            # Create individual booking records
            doc_text = f"Booking ID {idx}: "
            for col in ['hotel', 'arrival_date', 'adr', 'country', 'is_canceled', 'reservation_status']:
                if col in row:
                    doc_text += f"{col}: {row[col]}, "
            
            documents.append({"text": doc_text, "metadata": {"booking_id": idx}})
        
        self.documents = documents
        
        # Create FAISS index
        self.logger.info(f"Creating FAISS index with {len(documents)} documents...")
        texts = [doc["text"] for doc in documents]
        embeddings = self.embedding_model.encode(texts)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity with normalized vectors
        self.index.add(embeddings)
        
        self.logger.info("Document embeddings and FAISS index created successfully.")
        return len(documents)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for the query"""
        if self.index is None:
            self.create_document_embeddings()
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Get retrieved documents
        retrieved_docs = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):  # Check if index is valid
                doc = self.documents[idx]
                retrieved_docs.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": float(scores[0][i])
                })
        
        return retrieved_docs
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using RAG"""
        self.logger.info(f"Answering question: {question}")
        
        # Track query history
        self.query_history.append({
            "question": question,
            "timestamp": pd.Timestamp.now()
        })
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(question)
        context = "\n".join([doc["text"] for doc in retrieved_docs])
        
        try:
            # Generate prompt
            prompt = self.prompt_template.format(question=question, context=context)
            
            # Get answer from LLM
            answer = self.llm(prompt)
            
            # For questions about specific metrics, compute from the data
            lower_question = question.lower()
            
            if "total revenue" in lower_question and any(month in lower_question for month in ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]):
                # Extract month from question
                for month in ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]:
                    if month in lower_question:
                        if "2017" in lower_question or "2016" in lower_question:
                            year = "2017" if "2017" in lower_question else "2016"
                            # Calculate actual revenue for the specified month
                            month_data = self.data[
                                (self.data['arrival_date'].dt.month == pd.to_datetime(month, format='%B').month) & 
                                (self.data['arrival_date'].dt.year == int(year))
                            ]
                            actual_revenue = month_data['revenue'].sum()
                            answer = f"The total revenue for {month.capitalize()} {year} was {actual_revenue:.2f}."
                        break
            
            elif "average price" in lower_question or "average daily rate" in lower_question:
                avg_price = self.data['adr'].mean()
                answer = f"The average price (ADR) of a hotel booking is {avg_price:.2f}."
            
            elif "highest booking cancellations" in lower_question or "most cancellations" in lower_question:
                if "country" in lower_question or "location" in lower_question:
                    cancellations_by_country = self.data[self.data['is_canceled'] == 1]['country'].value_counts().reset_index()
                    cancellations_by_country.columns = ['country', 'cancellations']
                    top_countries = cancellations_by_country.head(5)
                    answer = f"The locations with the highest booking cancellations are: {', '.join(top_countries['country'].tolist())}."
                else:
                    cancellations_by_hotel = self.data[self.data['is_canceled'] == 1]['hotel'].value_counts().reset_index()
                    cancellations_by_hotel.columns = ['hotel', 'cancellations']
                    answer = f"The hotel type with the highest booking cancellations is {cancellations_by_hotel.iloc[0]['hotel']} with {cancellations_by_hotel.iloc[0]['cancellations']} cancellations."
            
            return {
                "question": question,
                "answer": answer,
                "retrieved_documents": retrieved_docs
            }
            
        except Exception as e:
            self.logger.error(f"Error answering question: {str(e)}")
            return {
                "question": question,
                "answer": "Sorry, I encountered an error while trying to answer your question.",
                "error": str(e)
            }
    
    def get_query_history(self):
        """Get the query history"""
        return self.query_history