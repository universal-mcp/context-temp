from universal_mcp.applications import APIApplication
from universal_mcp.integrations import Integration
from contextlib import contextmanager
from sqlmodel import SQLModel, Field, Session, create_engine, select, Relationship
from sqlalchemy import JSON, cast, Float, func, Index, UniqueConstraint, desc, text
from sqlalchemy.orm import selectinload
from pgvector.sqlalchemy import Vector
from openai import AzureOpenAI
from typing import Optional, List, Dict, Any
from universal_mcp_markitdown.app import MarkitdownApp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from universal_mcp_context_temp.settings import settings
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

embedding_client = AzureOpenAI(
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
    embedding = embedding_client.embeddings.create(
        input=[text],
        model=settings.embedding_model_name,
    )
    return embedding.data[0].embedding

def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a list of texts in a single, efficient API call.

    Args:
        texts (List[str]): A list of text strings to embed.

    Returns:
        List[List[float]]: A list of embedding vectors.
    """
    if not texts:
        return []

    embedding = embedding_client.embeddings.create(
        input=texts,
        model=settings.embedding_model_name,
    )
    return [item.embedding for item in embedding.data]

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
        Index(
            'fts_index_on_content',
            text("to_tsvector('english', content)"),
            postgresql_using='gin'
        ),
    )

class SourceDocument(SQLModel, table=True):
    """
    Represents a single source document (e.g., a file).
    This is the 'parent' table.
    """
    __tablename__ = "source_documents"
    id: Optional[int] = Field(default=None, primary_key=True)

    collection: str = Field(index=True)
    filepath: str = Field(index=True) # Unique within a collection
    chunk_count: int = Field(default=0)
    meta: dict = Field(default_factory=dict, sa_type=JSON)
    
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc), nullable=False)
    updated_at: datetime = Field(default_factory=datetime.now(timezone.utc), nullable=False)
    
    chunks: List[DocumentChunk] = Relationship(back_populates="source_document", sa_relationship_kwargs={"cascade": "all, delete-orphan"})

    __table_args__ = (UniqueConstraint('collection', 'filepath', name='_collection_filepath_uc'),)


class ContextApp(APIApplication):
    """
    Base class for Universal MCP Applications.
    """
    def __init__(self, integration: Integration = None, **kwargs) -> None:
        super().__init__(name="context", integration=integration, **kwargs)

        self.chat_llm = AzureChatOpenAI(
            openai_api_version=settings.embedding_api_version,
            deployment_name=settings.chat_model_name,
            temperature=0.2,
            max_retries=3,
        )

    def _generate_contextual_text(self, full_document_content: str, chunk_content: str) -> str:
        """
        Calls an LLM via LangChain to generate context for a chunk based on the full document.
        """
        try:
            prompt_text = f"""Given the full document below, provide a short, succinct context to situate the following chunk within it. This will be used to improve search retrieval of the chunk. Answer only with the context and nothing else.

<document>
{full_document_content[:20000]}
</document>

<chunk>
{chunk_content}
</chunk>
"""
            messages = [
                SystemMessage(content="You are a helpful assistant that provides concise contextual information for text chunks."),
                HumanMessage(content=prompt_text),
            ]
            response = self.chat_llm.invoke(messages)

            generated_context = response.content.strip()
            return f"{generated_context}\n---\n{chunk_content}"

        except Exception as e:
            print(f"Warning: Failed to generate context for chunk. Reason: {e}. Using original chunk for embedding.")
            return chunk_content

    def _get_or_create_source_document(self, session: Session, collection: str, filepath: str, metadata: dict) -> SourceDocument:
        """
        Helper function to either retrieve an existing document or create a new one.
        If the document already exists, its old chunks are deleted.
        """
        stmt = select(SourceDocument).where(SourceDocument.collection == collection, SourceDocument.filepath == filepath)
        existing_doc = session.exec(stmt).first()

        if existing_doc:
            existing_doc.chunks.clear()
            session.flush() # Persist the deletion of old chunks
            existing_doc.meta = metadata # Update metadata
            existing_doc.updated_at = datetime.now(timezone.utc)
            return existing_doc
        else:
            # Document does not exist, create a new one
            new_doc = SourceDocument(
                collection=collection,
                filepath=filepath,
                meta=metadata or {},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            session.add(new_doc)
            return new_doc

    def _process_and_store_document(self, session: Session, collection: str, source_identifier: str, content: str, metadata: Optional[dict]) -> int:
        """
        Internal helper to process content, create contextual embeddings, and store in the database.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(content)

        if not chunks:
            return None # No content to process

        source_doc = self._get_or_create_source_document(session, collection, source_identifier, metadata)
        
        session.commit()
        session.refresh(source_doc)

        enriched_texts_for_embedding = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self._generate_contextual_text, content, chunk) for chunk in chunks]
            for future in futures:
                enriched_texts_for_embedding.append(future.result())

        embeddings = generate_embeddings_batch(enriched_texts_for_embedding)

        new_chunks = []
        for i, original_chunk_content in enumerate(chunks):
            chunk = DocumentChunk(
                source_document_id=source_doc.id,
                content=original_chunk_content,
                embedding=embeddings[i],
                chunk_sequence=i + 1,
                meta={"embedding_strategy": "contextual"} # Add metadata to track strategy
            )
            new_chunks.append(chunk)

        session.add_all(new_chunks)
        
        # Update the parent document's metadata
        source_doc.chunk_count = len(chunks)
        source_doc.updated_at = datetime.now(timezone.utc)
        session.add(source_doc)

        # Final commit to save chunks and the updated parent
        session.commit()
        session.refresh(source_doc)
        
        return source_doc.id

    async def insert_document_from_file(self, collection: str, filepath: str, metadata: Optional[dict] = None) -> int:
        """
        Adds or updates a document in the context from a file path.
        The content will be loaded from the specified path, and the `filepath` itself 
        will serve as the unique identifier within the collection. If a document with the 
        same `collection` and `filepath` already exists, its content and metadata will be updated.

        Args:
            collection (str): The name of the collection to associate with the document.
            filepath (str): The path to the document (e.g., local file path, URL).
            metadata (dict, optional): Base metadata for the source document.

        Returns:
            int: The ID of the SourceDocument that was created or updated.

        Raises:
            ValueError: If loading or converting content from the filepath fails.
        
        Tags:
            insert, content, document, file, important
        """
        try:
            content = await markitdown.convert_to_markdown(filepath)
        except Exception as e:
            raise ValueError(f"Failed to load or convert content from filepath '{filepath}'. Reason: {e}")

        with get_session() as session:
            doc_id = self._process_and_store_document(
                session=session,
                collection=collection,
                source_identifier=filepath,
                content=content,
                metadata=metadata
            )
            return doc_id
        
    def search(self, collection: str, query: str, top_k: int = 5, metadata_filter: List[Dict[str, Any]] = None) -> List[dict]:
        """
        Performs a hybrid search using both vector similarity (for semantic meaning)
        and full-text search (for keyword matching), with advanced metadata filtering.
        Results are combined using Reciprocal Rank Fusion (RRF).
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
            collection (str): The name of the collection to search within.
            query (str): The query string to find similar documents.
            top_k (int, optional): The maximum number of similar documents to return. Defaults to 5.
            metadata_filter (List[Dict], optional): A list of filter conditions to apply.

        Returns:
            List[dict]: A list of dictionaries representing the similar document chunks.
            
        Tags:
            query, important, hybrid
        """
        enriched_query_for_embedding = f"Search query: {query}" # Simple enrichment, can be more complex
        query_embedding = generate_embeddings_batch([enriched_query_for_embedding])[0]

        tsquery = func.plainto_tsquery('english', query)
        k_reciprocal = 60  # RRF ranking constant

        with get_session() as session:
            # 1. Build a list of filter clauses from the metadata_filter argument
            filter_clauses = [SourceDocument.collection == collection]
            if metadata_filter:
                for condition in metadata_filter:
                    field = condition.get("field")
                    op = condition.get("op", "==").lower()
                    value = condition.get("value")

                    if not all([field, op, value is not None]):
                        continue
                    
                    json_field = SourceDocument.meta[field]
                    clause = None
                    if op in ('>', 'gt', '>=', 'gte', '<', 'lt', '<=', 'lte'):
                        numeric_field = cast(json_field.as_string(), Float)
                        if op in ('>', 'gt'): 
                            clause = numeric_field > value
                        elif op in ('>=', 'gte'): 
                            clause = numeric_field >= value
                        elif op in ('<', 'lt'): 
                            clause = numeric_field < value
                        elif op in ('<=', 'lte'): 
                            clause = numeric_field <= value
                    elif op == 'in':
                        if isinstance(value, list):
                            clause = json_field.as_string().in_([str(v) for v in value])
                    elif op in ('!=', 'ne'):
                        clause = json_field.as_string() != str(value)
                    else:  # Default case for '==' or 'eq'
                        clause = json_field.as_string() == str(value)
                    
                    if clause is not None:
                        filter_clauses.append(clause)

            # 2. Vector Search CTE: Rank documents by cosine distance
            vector_search_cte = (
                select(
                    DocumentChunk.id.label("id"),
                    func.row_number().over(
                        order_by=DocumentChunk.embedding.cosine_distance(query_embedding)
                    ).label("rank")
                )
                .join(SourceDocument)
                .where(*filter_clauses)
                .cte("vector_search")
            )

            # 3. Full-Text Search (FTS) CTE: Rank documents by text relevance
            fts_filter_clauses = filter_clauses + [
                func.to_tsvector('english', DocumentChunk.content).op('@@')(tsquery)
            ]
            fts_search_cte = (
                select(
                    DocumentChunk.id.label("id"),
                    func.row_number().over(
                        order_by=func.ts_rank(
                            func.to_tsvector('english', DocumentChunk.content),
                            tsquery
                        ).desc()
                    ).label("rank")
                )
                .join(SourceDocument)
                .where(*fts_filter_clauses)
                .cte("fts_search")
            )

            # 4. Combine results with Reciprocal Rank Fusion (RRF)
            all_chunk_ids = select(vector_search_cte.c.id).union(select(fts_search_cte.c.id)).subquery()

            final_stmt = (
                select(
                    DocumentChunk,
                    (
                        func.coalesce(1.0 / (k_reciprocal + vector_search_cte.c.rank), 0.0) +
                        func.coalesce(1.0 / (k_reciprocal + fts_search_cte.c.rank), 0.0)
                    ).label("rrf_score")
                )
                .join(all_chunk_ids, DocumentChunk.id == all_chunk_ids.c.id)
                .outerjoin(vector_search_cte, DocumentChunk.id == vector_search_cte.c.id)
                .outerjoin(fts_search_cte, DocumentChunk.id == fts_search_cte.c.id)
                .options(selectinload(DocumentChunk.source_document))  # Eager load parent document
                .order_by(desc("rrf_score"))
                .limit(top_k)
            )

            # 5. Execute query and format the output
            results = session.exec(final_stmt).all()
            
            output = []
            for row in results:
                chunk = row.DocumentChunk
                output.append({
                    "chunk_id": chunk.id,
                    "content": chunk.content,
                    "chunk_sequence": chunk.chunk_sequence,
                    "chunk_meta": chunk.meta,
                    "source_document": {
                        "id": chunk.source_document.id,
                        "collection": chunk.source_document.collection,
                        "filepath": chunk.source_document.filepath,
                        "meta": chunk.source_document.meta,
                    }
                })
            return output

    async def insert_document_from_content(self, collection: str, content: str, filename: str, metadata: Optional[dict] = None) -> int:
        """
        Adds or updates a document in the context from raw content.
        You must provide the document's `content` directly and a `filename` which 
        will serve as the unique identifier within the collection. This allows for readable
        identification and enables future updates to the document by referencing the 
        same `collection` and `filename`.

        Args:
            collection (str): The name of the collection to associate with the document.
            content (str): The raw text content of the document.
            filename (str): A unique name for the document. This is used as the `filepath` 
                            identifier in the database.
            metadata (dict, optional): Base metadata for the source document.

        Returns:
            int: The ID of the SourceDocument that was created or updated.
        
        Tags:
            insert, content, document, text, important
        """
        with get_session() as session:
            doc_id = self._process_and_store_document(
                session=session,
                collection=collection,
                source_identifier=filename,
                content=content,
                metadata=metadata
            )
            return doc_id

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
        For example, metadata_filter = [
            {"field": "author", "op": "==", "value": "Ankit Ranjan"},
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
                        "collection": chunk.source_document.collection,
                        "filepath": chunk.source_document.filepath,
                        "meta": chunk.source_document.meta,
                    }
                })
            return output
             
    def list_collections(self) -> List[str]:
        """
        Lists all unique collection names available in the context.

        Returns:
            List[str]: A list of unique collection names.
        
        Tags:
            list, query, collection, important
        """
        with get_session() as session:
            stmt = select(SourceDocument.collection).distinct()
            results = session.exec(stmt).all()

            return results

    def list_documents_in_collection(self, collection: str) -> List[Dict[str, Any]]:
        """
        Lists all documents (and their metadata) within a specific collection.

        Args:
            collection (str): The name of the collection to list documents for.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a document
                                  with its ID, filepath, and other metadata.
        
        Tags:
            list, query, document, important
        """
        with get_session() as session:
            stmt = select(SourceDocument).where(SourceDocument.collection == collection).order_by(SourceDocument.filepath)
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
        Lists all documents across all collections available in the context.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a document
                                  with its ID, collection, filepath, and other metadata.
        
        Tags:
            list, query, document, important
        """
        with get_session() as session:
            stmt = select(SourceDocument).order_by(SourceDocument.collection, SourceDocument.filepath)
            documents = session.exec(stmt).all()
            
            return [
                {
                    "id": doc.id,
                    "collection": doc.collection, # Include the collection name for clarity
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
            # self.insert_document_from_file,
            # self.insert_document_from_content, 
            # self.delete_document, 
            # self.query_similar,
            # self.list_collections,
            # self.list_documents_in_collection,
            # self.list_documents,
            self.search,
        ]