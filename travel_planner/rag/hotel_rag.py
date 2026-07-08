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
from travel_planner.models import HotelData

class HotelRAG:
    """
    RAG system for hotel data.
    Flow: API Data → Documents → Chunks → Vector DB → Retrieval
    """
    
    def __init__(self, persist_dir: str = "./chroma_hotels_db"):
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.vector_store = None
        
    def create_documents(self, hotels: List[HotelData], city: str) -> List[Document]:
        """
        Step 1: Convert HotelData objects to Documents.
        """
        documents = []
        
        for hotel in hotels:
            # Create rich text representation for semantic search
            text = f"""
            HOTEL INFORMATION:
            Name: {hotel.name}
            City: {city}
            Star Rating: {hotel.star_rating} out of 5
            Price: {hotel.currency} {hotel.current_price} per night
            Address: {hotel.address}
            Neighborhood: {hotel.neighborhood}
            
            GUEST RATINGS:
            Rating: {hotel.guest_rating} out of 10
            Badge: {hotel.rating_badge}
            Reviews: {hotel.reviews_count}
            
            AMENITIES:
            {', '.join(hotel.amenities[:10])}
            
            FREEBIES:
            {', '.join(hotel.freebies)}
            
            ROOM TYPES:
            {', '.join([f"{r.name} (${r.price})" for r in hotel.room_types[:3]])}
            
            LOCATION:
            Latitude: {hotel.latitude}, Longitude: {hotel.longitude}
            
            NEARBY ATTRACTIONS:
            {', '.join(hotel.nearby_attractions[:5])}
            
            TRANSPORTATION:
            Airports: {', '.join([f"{a.name} ({a.time})" for a in hotel.airports[:2]])}
            Train Stations: {', '.join([f"{t.name} ({t.time})" for t in hotel.train_stations[:2]])}
            
            CHECK-IN/OUT:
            Check-in: {hotel.check_in_time}
            Check-out: {hotel.check_out_time}
            
            POLICIES:
            {', '.join(hotel.policies)}
            """
            
            # Create metadata for filtering
            metadata = {
                'hotel_name': hotel.name,
                'city': city,
                'star_rating': hotel.star_rating,
                'price_numeric': hotel.price_numeric,
                'guest_rating': hotel.guest_rating,
                'reviews_count': hotel.reviews_count,
                'neighborhood': hotel.neighborhood,
                'latitude': hotel.latitude,
                'longitude': hotel.longitude,
                'currency': hotel.currency,
                'source': 'RapidAPI'
            }
            
            doc = Document(
                page_content=text,
                metadata=metadata
            )
            documents.append(doc)
            
        print(f"✅ Created {len(documents)} documents from hotel data")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Step 2: Split documents into smaller chunks.
        """
        chunks = self.text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def store_in_vector_db(self, chunks: List[Document]) -> None:
        """
        Step 3: Store chunks in vector database.
        """
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        self.vector_store.persist()
        print(f"✅ Stored chunks in vector database at: {self.persist_dir}")
    
    def load_vector_db(self) -> None:
        """
        Load existing vector database.
        """
        if os.path.exists(self.persist_dir):
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            print(f"✅ Loaded vector database from: {self.persist_dir}")
        else:
            print(f"⚠️ Vector database not found at: {self.persist_dir}")
    
    def retrieve(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Document]:
        """
        Step 4: Retrieve relevant hotel data based on query.
        """
        if not self.vector_store:
            self.load_vector_db()
            
        if not self.vector_store:
            return []
        
        if filter_dict:
            results = self.vector_store.similarity_search(
                query, 
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search(query, k=k)
        
        print(f"✅ Retrieved {len(results)} relevant hotel chunks")
        return results
    
    def retrieve_hotel_info(self, query: str, k: int = 3) -> str:
        """
        Public method: Retrieve hotel info as formatted string.
        """
        docs = self.retrieve(query, k=k)
        
        if not docs:
            return "No relevant hotel information found."
        
        formatted_results = []
        for i, doc in enumerate(docs, 1):
            formatted_results.append(f"""
            === HOTEL {i} ===
            {doc.page_content}
            ---
            """)
        
        return "\n".join(formatted_results)
    
    def retrieve_and_format_for_llm(self, query: str, k: int = 3) -> str:
        """
        Retrieve and format hotel data specifically for LLM context.
        """
        docs = self.retrieve(query, k=k)
        
        if not docs:
            return "No hotel data found matching your criteria."
        
        hotel_summaries = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata
            hotel_summaries.append(f"""
            HOTEL {i}:
            Name: {metadata.get('hotel_name', 'Unknown')}
            Star Rating: {metadata.get('star_rating', 0)}/5
            Price: ${metadata.get('price_numeric', 0)}
            Guest Rating: {metadata.get('guest_rating', 'N/A')}/10
            Neighborhood: {metadata.get('neighborhood', 'N/A')}
            Reviews: {metadata.get('reviews_count', 0)}
            {doc.page_content[:300]}...
            """)
        
        return "\n".join(hotel_summaries)

# ========== SINGLETON INSTANCE ==========
_hotel_rag_instance = None

def get_hotel_rag() -> HotelRAG:
    """Get or create singleton HotelRAG instance."""
    global _hotel_rag_instance
    if _hotel_rag_instance is None:
        _hotel_rag_instance = HotelRAG()
    return _hotel_rag_instance

# ========== RAG TOOL ==========
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class RAGQueryInput(BaseModel):
    query: str = Field(description="Search query for hotels (e.g., 'luxury hotels with pool under $300')")
    city: str = Field(description="City name to filter results")
    k: int = Field(default=3, description="Number of results to return")

@tool(args_schema=RAGQueryInput)
def retrieve_hotel_info(query: str, city: str, k: int = 3) -> str:
    """
    Retrieve relevant hotel information from vector database using semantic search.
    """
    rag = get_hotel_rag()
    city_db_path = f"./chroma_hotels_{city.lower().replace(' ', '_')}_db"
    if rag.persist_dir != city_db_path:
        rag.persist_dir = city_db_path
        rag.load_vector_db()
    
    return rag.retrieve_hotel_info(query, k)
