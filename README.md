<<<<<<< Updated upstream
## Table of Contents

- Project Overview  
- System Architecture  
- Pipeline Stages  
  - Data Ingestion  
  - Domain Classification  
  - Data Cleaning & Chunking  
  - Indexing Layer  
  - Query Understanding  
  - Hybrid Retrieval + RRF Fusion  
  - Graph Layer (Knowledge Graph Expansion)  
  - Cross-Encoder Re-Ranking  
  - Prompt Builder  
  - LLM Generation  
- Knowledge Graph Schema  
- Technology Stack  
- Project Directory Structure  
- Data Sources  
- Evaluation Strategy  
- Future Extensions

## 1. Project Overview

The Legal Advisory System is a Retrieval-Augmented Generation (RAG) application designed to assist users in navigating legal documents, case law, statutes, and contracts.

It extends standard RAG with a GraphRAG layer, where a knowledge graph encodes relationships between legal entities such as precedents, citations, amendments, and cross-references.

### Core Design Goals

- Domain-aware retrieval  
- High-precision answers with citations  
- Scalable ingestion pipeline  
- Explainable retrieval using source tracking

## 2. System Architecture

The system is divided into two main pipelines:

---

### Offline Pipeline

- Legal documents (case law, statutes, contracts, regulations) are ingested into the system  
- Domain classification is performed to categorize documents into legal domains  
- Documents undergo cleaning and semantic chunking for better retrieval performance  
- Indexing is performed using:
  - Sparse index (keyword-based retrieval using BM25)
  - Dense index (semantic embeddings for similarity search)
- A knowledge graph is constructed to represent legal relationships such as:
  - Case citations  
  - Precedents  
  - Amendments  
  - Cross-references  

---

### Online Pipeline

- User query is analyzed for intent, entities, and legal domain detection  
- Hybrid retrieval is performed using:
  - BM25 (sparse retrieval)
  - Dense vector embeddings (semantic retrieval)  
- Graph expansion enables multi-hop legal reasoning using the knowledge graph  
- Retrieved results are re-ranked using a cross-encoder model  
- A structured prompt is constructed using the refined context  
- Final response is generated using an LLM with citations and traceable sources


## 3. Pipeline Stages

---

### 3.1 Data Ingestion

The system supports multiple legal document formats:

- PDF (scanned and digital documents)

#### Processing Steps:

- Format detection (identifying scanned vs digital PDFs)  
- OCR applied for scanned documents  
- Metadata extraction, including:
  - Title  
  - Date  
  - Jurisdiction  
  - Court information  
- Conversion into a structured document format for downstream processing  

---

### 3.2 Domain Classification

Each document is classified into its respective legal domain.

This classification improves retrieval precision by filtering out irrelevant legal domains and ensuring that queries are matched with contextually appropriate documents.

### 3.3 Data Cleaning & Chunking

Legal documents require structured preprocessing to ensure high-quality retrieval.

#### Cleaning Process:

- Removal of headers, footers, and watermarks  
- Correction of OCR errors  
- Normalization of formatting  
- Deduplication of repeated content  

#### Chunking Strategy:

- Statutes → split by section and sub-section  
- Case law → split into headnotes, issues, and judgments  
- Contracts → split into clauses and sub-clauses  

#### Each Chunk Preserves:

- Document ID  
- Domain label  
- Section / clause identifiers  
- Metadata (jurisdiction, court, date, etc.)  

---

### 3.4 Indexing Layer

The system uses three parallel indexing mechanisms:

#### Sparse Index (BM25)

- Keyword-based retrieval  
- Captures exact legal terminology such as:
  - Section numbers  
  - Case citations  
  - Legal references  

#### Dense Index (Embeddings)

- Semantic search using legal language models  
- Captures contextual meaning beyond exact keywords  

#### Knowledge Graph (Neo4j)

Stores structured relationships between legal entities such as:

- Cases  
- Statutes  
- Sections  
- Clauses  
- Courts  
- Legal concepts


### 3.5 Query Understanding

Each user query is analyzed across multiple dimensions:

#### Intent Detection:
- Find precedent  
- Interpret statute  
- Draft legal guidance  
- General legal question  

#### Entity Extraction:
- Case names  
- Legal sections  
- Statutes / laws  

#### Domain Routing:
- Identifies the relevant legal domain to improve retrieval accuracy  

---

### 3.6 Hybrid Retrieval + RRF Fusion

Retrieval is performed using a hybrid approach:

- BM25 keyword-based search  
- Dense vector semantic search  

Results from both methods are merged using **Reciprocal Rank Fusion (RRF)**, ensuring stable ranking even when scoring systems differ.

---

### 3.7 Graph Layer (Knowledge Graph Expansion)

After initial retrieval:

- Top-ranked documents are treated as seed nodes  
- The knowledge graph is traversed (1–2 hops)  

This enables retrieval of:

- Precedent cases  
- Statutory interpretations  
- Overruled or related judgments  

This significantly improves multi-hop legal reasoning.

---

### 3.8 Cross-Encoder Re-Ranking

A cross-encoder model re-ranks retrieved chunks by jointly evaluating:

- Query–document relevance  

This step improves precision compared to dense retrieval alone.

---

### 3.9 Prompt Builder

A structured prompt is constructed containing:

- System instructions (legal assistant role definition)  
- Retrieved legal sources  
- Domain context  
- User query  

Each retrieved source is clearly labeled to support citation tracking and explainability.

---

### 3.10 LLM Generation

A large language model generates the final response using:

- Retrieved legal context  
- Graph-expanded knowledge  
- Re-ranked sources  

#### Output Includes:

- Final answer  
- Source references  
- Confidence score  
- Domain label  
- Legal disclaimer  

---

## 4. Knowledge Graph Schema

The knowledge graph represents structured relationships between legal entities.

### Core Relationships:

- Case → cites → Case  
- Case → overrules → Case  
- Case → interprets → Section  
- Statute → contains → Section  
- Section → applies to → Concept  
- Contract → contains → Clause  

### Entity Types:

- Cases  
- Statutes  
- Sections  
- Clauses  
- Courts  
- Parties  
- Legal Concepts  

<img 
  src="https://github.com/user-attachments/assets/7e3763d5-34d5-484f-b426-978820ccb993" 
  alt="Legal RAG System Architecture"
  width="100%"
/>
=======
>>>>>>> Stashed changes
