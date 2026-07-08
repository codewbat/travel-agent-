import os
import json
from typing import List, Dict, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain_community.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class FoodRAG:
    """
    RAG system for food / restaurant recommendations.
    Flow: Raw Restaurant Data → Documents → Chunks → Vector DB → Retrieval
    """
    
    def __init__(self, persist_dir: str = "./chroma_food_db"):
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=80,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.vector_store = None
        
    def create_documents(self, restaurants: List[Dict], city: str) -> List[Document]:
        """Convert raw restaurant dictionary data to Document objects."""
        documents = []
        
        for i, rest in enumerate(restaurants):
            name = rest.get('name') or rest.get('displayName', {}).get('text') or 'Unknown Restaurant'
            address = rest.get('formattedAddress') or rest.get('address') or 'N/A'
            rating = rest.get('rating') or 'N/A'
            types = rest.get('types') or [rest.get('type')] or []
            types_str = ", ".join([t for t in types if t]) if types else "restaurant"
            price_level = rest.get('priceLevel', 'N/A')
            
            text = f"""
            RESTAURANT INFORMATION:
            Name: {name}
            City: {city}
            Cuisine/Categories: {types_str}
            Address: {address}
            Rating: {rating} out of 5
            Price Level: {price_level}
            """
            
            loc_obj = rest.get('location') or {}
            lat = loc_obj.get('latitude') or rest.get('latitude') or 0.0
            lng = loc_obj.get('longitude') or rest.get('longitude') or 0.0
            
            metadata = {
                'restaurant_name': name,
                'city': city,
                'rating': str(rating),
                'price_level': str(price_level),
                'latitude': float(lat),
                'longitude': float(lng),
                'index': i,
                'source': 'GooglePlaces/OSM'
            }
            
            doc = Document(
                page_content=text,
                metadata=metadata
            )
            documents.append(doc)
            
        print(f"✅ Created {len(documents)} documents from restaurant data")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        chunks = self.text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} restaurant chunks from {len(documents)} documents")
        return chunks
    
    def store_in_vector_db(self, chunks: List[Document]) -> None:
        """Store chunks in vector database."""
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        self.vector_store.persist()
        print(f"✅ Stored restaurant chunks in vector database at: {self.persist_dir}")
    
    def load_vector_db(self) -> None:
        """Load existing vector database."""
        if os.path.exists(self.persist_dir):
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            print(f"✅ Loaded restaurants database from: {self.persist_dir}")
        else:
            print(f"⚠️ Restaurants database not found at: {self.persist_dir}")
    
    def retrieve(self, query: str, k: int = 8) -> List[Document]:
        """Retrieve relevant restaurants based on query."""
        if not self.vector_store:
            self.load_vector_db()
            
        if not self.vector_store:
            return []
        
        results = self.vector_store.similarity_search(query, k=k)
        print(f"✅ Retrieved {len(results)} relevant restaurant chunks")
        return results
    
    def retrieve_and_format_for_llm(self, query: str, k: int = 8) -> str:
        """Retrieve and format restaurant data specifically for LLM context."""
        docs = self.retrieve(query, k=k)
        
        if not docs:
            return "No matching restaurants found in the database."
        
        summaries = []
        seen_names = set()
        count = 1
        for doc in docs:
            metadata = doc.metadata
            name = metadata.get('restaurant_name', 'Unknown')
            if name in seen_names:
                continue
            seen_names.add(name)
            summaries.append(
                f"Restaurant {count}: Name: {name} | Rating: {metadata.get('rating', 'N/A')}/5 | "
                f"Price Level: {metadata.get('price_level', 'N/A')} | "
                f"Lat: {metadata.get('latitude', 0.0)} | Lng: {metadata.get('longitude', 0.0)}"
            )
            count += 1
        
        return "\n".join(summaries)

# ========== SINGLETON INSTANCE ==========
_food_rag_instance = None

def get_food_rag() -> FoodRAG:
    """Get or create singleton FoodRAG instance."""
    global _food_rag_instance
    if _food_rag_instance is None:
        _food_rag_instance = FoodRAG()
    return _food_rag_instance
