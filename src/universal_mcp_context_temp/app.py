from universal_mcp.applications import APIApplication
from universal_mcp.integrations import Integration
from contextlib import contextmanager
import numpy as np
from sqlmodel import SQLModel, Field, Session, create_engine, select, Relationship
from sqlalchemy import JSON, cast, Float, func, Index
from pgvector.sqlalchemy import Vector
from openai import AzureOpenAI
from typing import Optional, List, Dict, Any
from universal_mcp_markitdown.app import MarkitdownApp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from universal_mcp_context_temp.settings import settings
from datetime import datetime, timezone


# 5. Collection suppprt / Project support
# 8. Full text support ( vector + text) , hybrid search

client = AzureOpenAI(
    api_version=settings.embedding_api_version,
    azure_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_api_key,
)
VECTOR_DIM = 1536
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

class DocumentChunk(SQLModel, table=True):
    """
    Represents a single chunk of content derived from a SourceDocument.
    This is the 'child' table.
    """
    __tablename__ = "document_chunks"
    id: Optional[int] = Field(default=None, primary_key=True)
    
    content: str
    embedding: List[float] = Field(sa_type=Vector(VECTOR_DIM))
    chunk_sequence: int = Field(index=True) # The order of the chunk
    meta: dict = Field(default_factory=dict, sa_type=JSON)
    
    source_document_id: Optional[int] = Field(default=None, foreign_key="source_documents.id")
    source_document: "SourceDocument" = Relationship(back_populates="chunks")

    __table_args__ = (
        Index(
            'hnsw_index_on_embedding',
            'embedding',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_cosine_ops'}
        ),
    )

class SourceDocument(SQLModel, table=True):
    """
    Represents a single source document (e.g., a file).
    This is the 'parent' table.
    """
    __tablename__ = "source_documents"
    id: Optional[int] = Field(default=None, primary_key=True)

    filepath: str = Field(index=True, unique=True) # Assuming filepath is a unique identifier
    collection: str = Field(index=True)
    chunk_count: int = Field(default=0)
    meta: dict = Field(default_factory=dict, sa_type=JSON)
    
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc), nullable=False)
    updated_at: datetime = Field(default_factory=datetime.now(timezone.utc), nullable=False)
    
    chunks: List[DocumentChunk] = Relationship(back_populates="source_document", sa_relationship_kwargs={"cascade": "all, delete-orphan"})

class ContextApp(APIApplication):
    """
    Base class for Universal MCP Applications.
    """
    def __init__(self, integration: Integration = None, **kwargs) -> None:
        super().__init__(name="context", integration=integration, **kwargs)

    def _get_or_create_source_document(self, session: Session, collection: str, filepath: str, metadata: dict) -> SourceDocument:
        """
        Helper function to either retrieve an existing document or create a new one.
        If the document already exists, its old chunks are deleted.
        """
        stmt = select(SourceDocument).where(SourceDocument.filepath == filepath)
        existing_doc = session.exec(stmt).first()

        if existing_doc:
            existing_doc.chunks.clear()
            existing_doc.meta = metadata # Update metadata
            existing_doc.updated_at = datetime.now(timezone.utc)
            return existing_doc
        else:
            # Document does not exist, create a new one
            new_doc = SourceDocument(
                filepath=filepath,
                collection=collection,
                meta=metadata or {},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            session.add(new_doc)
            return new_doc


    async def insert_document(self, collection, content: str = None, filepath: str = None, metadata=None) -> int:
        """
        Adds or updates a document in the context, chunking it and storing it
        in a relational schema.

        Args:
            collection (str): The name of the collection for the document.
            content (str, optional): The content of the document.
            filepath (str, optional): The path to the document file. This is now the
                                     preferred and unique identifier for a document.
            metadata (dict, optional): Base metadata for the source document.

        Returns:
            int: The ID of the SourceDocument that was created or updated.

        Raises:
            ValueError: If filepath is not provided.
            
        Tags:
            insert, content, document, important
        """
        if not filepath:
            raise ValueError("'filepath' is required and must be a unique identifier for the document.")

        final_content = ""
        if content:
            final_content = content
        else:
            final_content = await markitdown.convert_to_markdown(filepath)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(final_content)

        with get_session() as session:
            source_doc = self._get_or_create_source_document(session, collection, filepath, metadata)
            
            session.commit()
            session.refresh(source_doc)

            new_chunks = []
            for i, chunk_content in enumerate(chunks):
                embedding = generate_embedding(chunk_content)
                chunk = DocumentChunk(
                    source_document_id=source_doc.id, # Link to the parent
                    content=chunk_content,
                    embedding=embedding,
                    chunk_sequence=i + 1 # Set the order
                )
                new_chunks.append(chunk)

            session.add_all(new_chunks)
            
            source_doc.chunk_count = len(chunks)
            source_doc.updated_at = datetime.now(timezone.utc)
            session.add(source_doc)

            session.commit()
            session.refresh(source_doc)
            
            return source_doc.id

    def delete_document(self, doc_id: int) -> bool:
        """
        Deletes a source document and all of its associated chunks from the context.

        Args:
            doc_id (int): The ID of the SourceDocument to delete.

        Returns:
            bool: True if the document was found and deleted, False otherwise.
        
        Tags:
            delete, important
        """
        with get_session() as session:
            doc_to_delete = session.get(SourceDocument, doc_id)

            if doc_to_delete:
                session.delete(doc_to_delete)
                session.commit()
                return True
            
            return False
            
    def query_similar(self, collection: str, query: str, top_k: int = 5, metadata_filter: List[Dict[str, Any]] = None) -> List[dict]:
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
            stmt = select(DocumentChunk).join(SourceDocument)

            stmt = stmt.where(SourceDocument.collection == collection)

            if metadata_filter:
                for condition in metadata_filter:
                    field = condition.get("field")
                    op = condition.get("op", "==").lower()
                    value = condition.get("value")

                    if not all([field, op, value is not None]):
                        continue
                    
                    json_field = SourceDocument.meta[field]

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

            stmt = stmt.order_by(DocumentChunk.embedding.cosine_distance(query_embedding)).limit(top_k)
            
            results = session.exec(stmt).all()

            output = []
            for chunk in results:
                output.append({
                    "chunk_id": chunk.id,
                    "content": chunk.content,
                    "chunk_sequence": chunk.chunk_sequence,
                    "chunk_meta": chunk.meta,
                    "source_document": {
                        "id": chunk.source_document.id,
                        "filepath": chunk.source_document.filepath,
                        "collection": chunk.source_document.collection,
                        "meta": chunk.source_document.meta,
                    }
                })
            return output
        
    def list_tools(self):
        """
        Lists the available tools (methods) for this application.
        """
        return [
            self.insert_document, 
            self.delete_document, 
            self.query_similar
        ]
 