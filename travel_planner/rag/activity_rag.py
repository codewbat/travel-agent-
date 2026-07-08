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

class ActivityRAG:
    """
    RAG system for activities / tourist attractions data.
    Flow: Raw Attractions Data → Documents → Chunks → Vector DB → Retrieval
    """
    
    def __init__(self, persist_dir: str = "./chroma_activities_db"):
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
        
    def create_documents(self, attractions: List[Dict], city: str) -> List[Document]:
        """Convert raw attractions dictionary data to Document objects."""
        documents = []
        
        for i, attr in enumerate(attractions):
            # Parse names and attributes safely
            name = attr.get('name') or attr.get('displayName', {}).get('text') or 'Unknown Attraction'
            address = attr.get('formattedAddress') or attr.get('address') or 'N/A'
            rating = attr.get('rating') or 'N/A'
            types = attr.get('types') or [attr.get('type')] or []
            types_str = ", ".join([t for t in types if t]) if types else "tourist attraction"
            
            text = f"""
            ATTRACTION INFORMATION:
            Name: {name}
            City: {city}
            Category/Type: {types_str}
            Address: {address}
            Rating: {rating} out of 5
            """
            
            loc_obj = attr.get('location') or {}
            lat = loc_obj.get('latitude') or attr.get('latitude') or 0.0
            lng = loc_obj.get('longitude') or attr.get('longitude') or 0.0
            
            metadata = {
                'attraction_name': name,
                'city': city,
                'rating': str(rating),
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
            
        print(f"✅ Created {len(documents)} documents from attractions data")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        chunks = self.text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} activity chunks from {len(documents)} documents")
        return chunks
    
    def store_in_vector_db(self, chunks: List[Document]) -> None:
        """Store chunks in vector database."""
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        self.vector_store.persist()
        print(f"✅ Stored activity chunks in vector database at: {self.persist_dir}")
    
    def load_vector_db(self) -> None:
        """Load existing vector database."""
        if os.path.exists(self.persist_dir):
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            print(f"✅ Loaded activities database from: {self.persist_dir}")
        else:
            print(f"⚠️ Activities database not found at: {self.persist_dir}")
    
    def retrieve(self, query: str, k: int = 8) -> List[Document]:
        """Retrieve relevant activities based on query."""
        if not self.vector_store:
            self.load_vector_db()
            
        if not self.vector_store:
            return []
        
        results = self.vector_store.similarity_search(query, k=k)
        print(f"✅ Retrieved {len(results)} relevant activity chunks")
        return results
    
    def retrieve_and_format_for_llm(self, query: str, k: int = 8) -> str:
        """Retrieve and format attraction data specifically for LLM context."""
        docs = self.retrieve(query, k=k)
        
        if not docs:
            return "No matching attractions found in the database."
        
        summaries = []
        seen_names = set()
        count = 1
        for doc in docs:
            metadata = doc.metadata
            name = metadata.get('attraction_name', 'Unknown')
            if name in seen_names:
                continue
            seen_names.add(name)
            summaries.append(
                f"Attraction {count}: Name: {name} | Rating: {metadata.get('rating', 'N/A')}/5 | "
                f"Lat: {metadata.get('latitude', 0.0)} | Lng: {metadata.get('longitude', 0.0)}"
            )
            count += 1
        
        return "\n".join(summaries)

# ========== SINGLETON INSTANCE ==========
_activity_rag_instance = None

def get_activity_rag() -> ActivityRAG:
    """Get or create singleton ActivityRAG instance."""
    global _activity_rag_instance
    if _activity_rag_instance is None:
        _activity_rag_instance = ActivityRAG()
    return _activity_rag_instance
