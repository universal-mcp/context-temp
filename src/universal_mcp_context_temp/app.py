from universal_mcp.applications import APIApplication
from universal_mcp.integrations import Integration
from contextlib import contextmanager
import numpy as np
from sqlmodel import SQLModel, Field, Session, create_engine, select, Relationship
from sqlalchemy import JSON, cast, Float, func, Index, UniqueConstraint
from pgvector.sqlalchemy import Vector
from openai import AzureOpenAI
from typing import Optional, List, Dict, Any
from universal_mcp_markitdown.app import MarkitdownApp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from universal_mcp_context_temp.settings import settings
from datetime import datetime, timezone
import uuid

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

    project: str = Field(index=True)
    filepath: str = Field(index=True) # Unique within a project
    chunk_count: int = Field(default=0)
    meta: dict = Field(default_factory=dict, sa_type=JSON)
    
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc), nullable=False)
    updated_at: datetime = Field(default_factory=datetime.now(timezone.utc), nullable=False)
    
    chunks: List[DocumentChunk] = Relationship(back_populates="source_document", sa_relationship_kwargs={"cascade": "all, delete-orphan"})

    __table_args__ = (UniqueConstraint('project', 'filepath', name='_project_filepath_uc'),)


class ContextApp(APIApplication):
    """
    Base class for Universal MCP Applications.
    """
    def __init__(self, integration: Integration = None, **kwargs) -> None:
        super().__init__(name="context", integration=integration, **kwargs)

    def _get_or_create_source_document(self, session: Session, project: str, filepath: str, metadata: dict) -> SourceDocument:
        """
        Helper function to either retrieve an existing document or create a new one.
        If the document already exists, its old chunks are deleted.
        """
        stmt = select(SourceDocument).where(SourceDocument.project == project, SourceDocument.filepath == filepath)
        existing_doc = session.exec(stmt).first()

        if existing_doc:
            existing_doc.chunks.clear()
            existing_doc.meta = metadata # Update metadata
            existing_doc.updated_at = datetime.now(timezone.utc)
            return existing_doc
        else:
            # Document does not exist, create a new one
            new_doc = SourceDocument(
                project=project,
                filepath=filepath,
                meta=metadata or {},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            session.add(new_doc)
            return new_doc

    async def insert_document(self, project: str, content: str = None, filepath: str = None, filename: str = None, metadata=None) -> int:
        """
        Adds or updates a document in the context.
        This function supports two primary modes of operation:
        1.  **By Filepath**: Provide a `filepath` to a document. The content will be loaded
            from this path, and the `filepath` itself will serve as the unique identifier
            within the project. If a document with the same `project` and `filepath`
            already exists, it will be updated.
        2.  **By Content**: Provide the document's `content` directly. In this mode, you
            must also provide a `filename`, which will serve as the unique identifier
            within the project. This allows for readable identification and enables future
            updates to the document by referencing the same `project` and `filename`.

        Args:
            project (str): The name of the project to associate with the document.
            content (str, optional): The raw text content of the document. If provided,
                                     `filename` must also be specified. Mutually exclusive
                                     with `filepath`.
            filepath (str, optional): The path to the document (e.g., file path, URL).
                                      Mutually exclusive with `content` and `filename`.
            filename (str, optional): A unique name for the document when providing `content`
                                      directly. It is used as the `filepath` identifier.
            metadata (dict, optional): Base metadata for the source document.

        Returns:
            int: The ID of the SourceDocument that was created or updated.

        Raises:
            ValueError: If validation fails. For example:
                        - If 'filepath' is provided along with 'content' or 'filename'.
                        - If 'content' is provided without a 'filename'.
                        - If neither ('content' and 'filename') nor 'filepath' are provided.
                        - If loading content from the 'filepath' fails.
            
        Tags:
            insert, content, document, important
        """

        if filepath and (content or filename):
            raise ValueError("If 'filepath' is provided, 'content' and 'filename' must be omitted.")

        if content and not filename:
            raise ValueError("If 'content' is provided, you must also provide a 'filename'.")

        if not filepath and not content:
            raise ValueError("You must provide either 'filepath' or 'content'.")
        
        source_identifier = ""
        final_content = ""

        if filepath:
            # Use case: Content is loaded from a path-like identifier.
            source_identifier = filepath
            try:
                final_content = await markitdown.convert_to_markdown(filepath)
            except Exception as e:
                raise ValueError(f"Failed to load or convert content from filepath '{filepath}'. Reason: {e}")

        elif content: # We know 'filename' is guaranteed to be present here
            # Use case: Content is provided directly with a filename.
            source_identifier = filename
            final_content = content

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(final_content)

        with get_session() as session:
            source_doc = self._get_or_create_source_document(session, project, source_identifier, metadata)
            
            session.commit()
            session.refresh(source_doc)

            new_chunks = []
            for i, chunk_content in enumerate(chunks):
                embedding = generate_embedding(chunk_content)
                chunk = DocumentChunk(
                    source_document_id=source_doc.id,
                    content=chunk_content,
                    embedding=embedding,
                    chunk_sequence=i + 1
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
            
    def query_similar(self, project: str, query: str, top_k: int = 5, metadata_filter: List[Dict[str, Any]] = None) -> List[dict]:
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
        For example, metadata_filter = [
            {"field": "author", "op": "==", "value": "Ankit Ranjan"},
            {"field": "chunk_number", "op": ">", "value": 5}
        ]

        Args:
            project (str): The name of the project to search within.
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

            stmt = stmt.where(SourceDocument.project == project)

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
                        "project": chunk.source_document.project,
                        "filepath": chunk.source_document.filepath,
                        "meta": chunk.source_document.meta,
                    }
                })
            return output
        
    def list_projects(self) -> List[str]:
        """
        Lists all unique project names available in the context.

        Returns:
            List[str]: A list of unique project names.
        
        Tags:
            list, query, project, important
        """
        with get_session() as session:
            stmt = select(SourceDocument.project).distinct()
            results = session.exec(stmt).all()

            return results

    def list_documents_in_project(self, project: str) -> List[Dict[str, Any]]:
        """
        Lists all documents (and their metadata) within a specific project.

        Args:
            project (str): The name of the project to list documents for.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a document
                                  with its ID, filepath, and other metadata.
        
        Tags:
            list, query, document, important
        """
        with get_session() as session:
            stmt = select(SourceDocument).where(SourceDocument.project == project).order_by(SourceDocument.filepath)
            documents = session.exec(stmt).all()
            
            return [
                {
                    "id": doc.id,
                    "filepath": doc.filepath,
                    # "chunk_count": doc.chunk_count,
                    # "meta": doc.meta,
                    # "created_at": doc.created_at,
                    # "updated_at": doc.updated_at,
                }
                for doc in documents
            ]
            
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Lists all documents across all projects available in the context.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a document
                                  with its ID, project, filepath, and other metadata.
        
        Tags:
            list, query, document, important
        """
        with get_session() as session:
            stmt = select(SourceDocument).order_by(SourceDocument.project, SourceDocument.filepath)
            documents = session.exec(stmt).all()
            
            return [
                {
                    "id": doc.id,
                    "project": doc.project, # Include the project name for clarity
                    "filepath": doc.filepath,
                    # "chunk_count": doc.chunk_count,
                    # "meta": doc.meta,
                    # "created_at": doc.created_at,
                    # "updated_at": doc.updated_at,
                }
                for doc in documents
            ]

    def list_tools(self):
        
        return [
            self.insert_document, 
            self.delete_document, 
            self.query_similar,
            self.list_projects,
            self.list_documents_in_project,
            self.list_documents,
        ]