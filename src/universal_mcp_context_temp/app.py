from universal_mcp.applications import APIApplication
from universal_mcp.integrations import Integration
from contextlib import contextmanager
import numpy as np
from sqlmodel import SQLModel, Field, Session, create_engine, select
from sqlalchemy import JSON, cast, Float
from pgvector.sqlalchemy import Vector
from openai import AzureOpenAI
from typing import Optional, List, Dict, Any
from universal_mcp_markitdown.app import MarkitdownApp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from universal_mcp_context_temp.settings import settings

# 5. Collection suppprt / Project support
# 7. Indexing support ( hnsw index)
# 8. Full text support ( vector + text) , hybrid search

client = AzureOpenAI(
    api_version=settings.embedding_api_version,
    azure_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_api_key,
)

VECTOR_DIM = 3072  # Set to your embedding size

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(settings.database_url, echo=False)
        SQLModel.metadata.create_all(_engine)
    return _engine

def generate_embedding(text: str):
    embedding = client.embeddings.create(
        input=[text],
        model=settings.embedding_model_name,
    )
    return embedding.data[0].embedding

@contextmanager
def get_session():
    engine = get_engine()
    with Session(engine) as session:
        yield session

markitdown = MarkitdownApp()

class Document(SQLModel, table=True):
    __tablename__ = "documents"
    id: Optional[int] = Field(default=None, primary_key=True)
    collection: str = Field(index=True)
    content: str
    filepath: Optional[str] = Field(default=None, index=False)
    embedding: List[float] = Field(sa_type=Vector(VECTOR_DIM))
    meta: dict = Field(
        default_factory=dict,
        sa_type=JSON,
    )

class ContextApp(APIApplication):
    """
    Base class for Universal MCP Applications.
    """
    def __init__(self, integration: Integration = None, **kwargs) -> None:
        super().__init__(name="context", integration=integration, **kwargs)


    async def insert_document(self, collection, content: str = None, filepath: str = None, metadata=None) -> List[int]:
        """
        Adds a document to the context, automatically chunking it if it's large.

        Args:
            collection (str): The name of the collection to which the document belongs.
            content (str, optional): The content of the document. Required if filepath is not provided.
            filepath (str, optional): The path to the document file. If provided, its content will be extracted.
            metadata (dict, optional): Base metadata to associate with the document and all its chunks.

        Returns:
            List[int]: A list of IDs for all the document chunks that were inserted.

        Raises:
            ValueError: If neither content nor filepath is provided.
            
        Tags:
            insert, content, document, important
        """
        final_content = ""
        if filepath:
            final_content = await markitdown.convert_to_markdown(filepath)
        elif content:
            final_content = content
        else:
            raise ValueError("Either 'content' or 'filepath' must be provided.")

        # 1. Initialize the text splitter
        # These values can be tuned. chunk_size is the max size of a chunk.
        # chunk_overlap keeps some text from the end of the previous chunk at the start of the next.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        # 2. Split the document content into chunks
        chunks = text_splitter.split_text(final_content)

        # 3. Process and insert each chunk as a separate document
        with get_session() as session:
            
            newly_created_docs = []
            for i, chunk_content in enumerate(chunks):
                
                # Generate an embedding for the specific chunk
                embedding = generate_embedding(chunk_content)

                # Create a copy of user-provided metadata and add chunk-specific info
                chunk_metadata = (metadata or {}).copy()
                chunk_metadata.update({
                    "source_filepath": filepath,
                    "chunk_number": i + 1,  # Human-friendly chunk number (starts at 1)
                    "total_chunks": len(chunks)
                })

                doc = Document(
                    collection=collection,
                    content=chunk_content,
                    filepath=filepath,
                    embedding=embedding,
                    meta=chunk_metadata
                )
                session.add(doc)
                newly_created_docs.append(doc)

            # Commit all the new document chunks in a single transaction
            session.commit()

            # After committing, refresh each object to get its database-assigned ID
            for doc in newly_created_docs:
                session.refresh(doc)
                
            # Return the list of all new document IDs
            return [doc.id for doc in newly_created_docs]

    def delete_document(self, doc_id):
        """
        Deletes a document from the context.

        Args:
            doc_id (int): The ID of the document to delete.

        Returns:
            None
        
        Tags:
            delete, important
        """
        with get_session() as session:
            doc = session.get(Document, doc_id)
            if doc:
                session.delete(doc)
                session.commit()

    def query_similar(self, collection: str, query: str, top_k: int = 5, metadata_filter: List[Dict[str, Any]] = None):
        """
        Queries the context for similar documents, with advanced metadata filtering.

        The metadata_filter should be a list of dictionaries, where each dictionary
        defines a single condition.
        
        Supported operators ('op'):
        - '==' or 'eq': Equals (for strings or numbers)
        - '!=' or 'ne': Not Equals
        - '>' or 'gt': Greater Than (for numbers)
        - '>=' or 'gte': Greater Than or Equal To (for numbers)
        - '<' or 'lt': Less Than (for numbers)
        - '<=' or 'lte': Less Than or Equal To (for numbers)
        - 'in': Value is in a list (e.g., {"field": "type", "op": "in", "value": ["pdf", "txt"]})

        Example:
        metadata_filter = [
            {"field": "subject", "op": "==", "value": "DSAI"},
            {"field": "chunk_number", "op": ">", "value": 5}
        ]

        Args:
            collection (str): The name of the collection to search within.
            query (str): The query string to find similar documents.
            top_k (int, optional): The maximum number of similar documents to return. Defaults to 5.
            metadata_filter (List[Dict], optional): A list of filter conditions to apply.

        Returns:
            List[dict]: A list of dictionaries representing the similar documents.
            
        Tags:
            query, important
        """
        query_embedding = generate_embedding(query)
        
        with get_session() as session:
            stmt = select(Document).where(Document.collection == collection)

            if metadata_filter:
                for condition in metadata_filter:
                    field = condition.get("field")
                    op = condition.get("op", "==").lower()
                    value = condition.get("value")

                    if not all([field, op, value is not None]):
                        continue
                    
                    json_field = Document.meta[field]

                    if op in ('>', 'gt', '>=', 'gte', '<', 'lt', '<=', 'lte'):
                        numeric_field = cast(json_field.as_string(), Float)
                        if op in ('>', 'gt'):
                            stmt = stmt.where(numeric_field > value)
                        elif op in ('>=', 'gte'):
                            stmt = stmt.where(numeric_field >= value)
                        elif op in ('<', 'lt'):
                            stmt = stmt.where(numeric_field < value)
                        elif op in ('<=', 'lte'):
                            stmt = stmt.where(numeric_field <= value)
                    
                    elif op == 'in':
                        if isinstance(value, list):
                            stmt = stmt.where(json_field.as_string().in_([str(v) for v in value]))

                    elif op in ('!=', 'ne'):
                        stmt = stmt.where(json_field.as_string() != str(value))
                    
                    else: # Default case for '==' or 'eq'
                        stmt = stmt.where(json_field.as_string() == str(value))

            stmt = stmt.order_by(Document.embedding.cosine_distance(query_embedding)).limit(top_k)
            
            results = session.exec(stmt).all()
            return [doc.model_dump(exclude={"embedding"}) for doc in results]
        
    def list_tools(self):
        """
        Lists the available tools (methods) for this application.
        """
        return [
            self.insert_document, 
            self.delete_document, 
            self.query_similar
        ]
 