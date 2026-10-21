"""
rag_pipeline.py
================
Repaired and Enhanced RAG Pipeline implementation:
- Hybrid Retrieval: Semantic FAISS (IndexFlatIP) + Lexical Keyword Matching.
- Query Intent Detection: Identity, Education, Internships, Skills, Projects, Publications.
- Multi-chunk aggregation for list questions (ensures ALL internships/skills are retrieved).
- Grounded generation with strict prompt rules and structured evidence collection.
- Traceable citations and robust refusal for missing facts (e.g. unstated CGPA).
- Graceful extractive fallback when Ollama is offline.
"""

import os
import sys
import json
import pickle
import re
import urllib.request
import urllib.error
import warnings
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import numpy as np

# Suppress harmless sklearn/scipy version mismatch warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Safeguard against optional torchaudio DLL loading issues on Windows
if "torchaudio" not in sys.modules:
    sys.modules["torchaudio"] = None

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class RAGPipeline:
    """
    Manages dense and lexical hybrid indexing, query-specific multi-chunk retrieval,
    and hallucination-free answer generation.
    """

    # Default Configuration (Centrally configured)
    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_SIMILARITY_THRESHOLD = 0.20   # Balanced threshold (prevents rejecting headers/institutions)
    DEFAULT_TOP_K = 6                     # Top chunks passed to LLM
    DEFAULT_CANDIDATE_COUNT = 12          # Chunks evaluated during hybrid retrieval
    UNAVAILABLE_RESPONSE = "I could not find this information in the provided knowledge base."
    NO_RESEARCH_EVIDENCE_RESPONSE = "Based on the provided context, the student's research interests are not explicitly stated, and there is insufficient research-related information to infer them."

    GROUNDED_SYSTEM_PROMPT = (
        "You are an expert personal knowledge-base assistant specializing in resume and document comprehension. "
        "Answer questions strictly and faithfully using only evidence found in the supplied document excerpts. "
        "A resume may use unconventional layouts, sidebars, tables, or lack standard section headings. "
        "Understand facts from their context: recognize candidate names (which may appear at the top, in headers, or alongside contact info), "
        "contact details, educational degrees and institutions, job titles and employers (which may be listed together or across adjacent lines), "
        "employment dates, technical skills, projects, and measurable achievements. "
        "For list or history questions (such as all jobs, employers, internships, or skills), enumerate EVERY distinct position or item found across all supplied excerpts. "
        "Do not stop after the first item. Do not invent or extrapolate missing personal details. "
        "If a specific detail is not found in the supplied excerpts, state clearly: 'I could not find this information in the provided knowledge base.' "
        "Cite source filenames for each answer."
    )

    # Stopwords for lexical term extraction
    STOPWORDS = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "and", "or", "but", "if", "then", "else", "what",
        "which", "who", "whom", "this", "that", "these", "those", "am", "i", "me",
        "my", "mine", "you", "your", "yours", "student", "tell", "show", "list",
        "give", "about", "all", "any", "how", "can"
    }

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        vector_store_dir: str = "vector_store",
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llama3:latest",
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        candidate_count: int = DEFAULT_CANDIDATE_COUNT
    ):
        self.embedding_model_name = embedding_model_name
        self.vector_store_dir = Path(vector_store_dir)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.candidate_count = candidate_count

        self._embedder: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.chunks_metadata: List[Dict[str, Any]] = []
        self.embedding_dimension: int = 384
        self.indexed_file_hashes: Dict[str, str] = {}

    @property
    def embedder(self) -> SentenceTransformer:
        """Loads and caches the SentenceTransformer model."""
        if self._embedder is None:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is not installed. Run: pip install sentence-transformers")
            self._embedder = SentenceTransformer(self.embedding_model_name)
            self.embedding_dimension = self._embedder.get_sentence_embedding_dimension()
        return self._embedder

    # ==========================================
    # Ollama Health & Model Discovery
    # ==========================================
    def check_ollama_status(self) -> Dict[str, Any]:
        """Pings local Ollama service and retrieves available models."""
        try:
            req = urllib.request.Request(
                f"{self.ollama_url}/api/tags",
                headers={"User-Agent": "PersonalRAG/2.0"}
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                    return {
                        "available": True,
                        "models": models,
                        "message": f"Ollama is running with {len(models)} model(s) installed."
                    }
        except Exception as e:
            return {
                "available": False,
                "models": [],
                "message": f"Ollama is unreachable at {self.ollama_url} ({str(e)})."
            }

        return {
            "available": False,
            "models": [],
            "message": f"Ollama is not running at {self.ollama_url}."
        }

    # ==========================================
    # Vector Indexing & Persistence
    # ==========================================
    def build_index(self, chunks: List[Dict[str, Any]], file_hashes: Optional[Dict[str, str]] = None) -> int:
        """
        Builds a normalized FAISS IndexFlatIP from provided chunks.
        Replaces any existing index to avoid stale or duplicate data.
        """
        if not chunks:
            self.clear_index()
            return 0

        if faiss is None:
            raise ImportError("faiss is not installed. Run: pip install faiss-cpu")

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalizing enables exact Cosine Similarity via Inner Product
        )

        dimension = embeddings.shape[1]
        self.embedding_dimension = dimension

        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype("float32"))

        self.index = index
        self.chunks_metadata = [dict(c) for c in chunks]
        self.indexed_file_hashes = file_hashes or {}

        self.save_index()
        return len(chunks)

    def save_index(self) -> None:
        """Persists FAISS index, metadata, and hashes to disk."""
        if self.index is None or not self.chunks_metadata:
            return

        index_file = self.vector_store_dir / "index.faiss"
        metadata_file = self.vector_store_dir / "metadata.pkl"

        faiss.write_index(self.index, str(index_file))
        with open(metadata_file, "wb") as f:
            pickle.dump({
                "chunks": self.chunks_metadata,
                "model_name": self.embedding_model_name,
                "dimension": self.embedding_dimension,
                "file_hashes": self.indexed_file_hashes
            }, f)

    def load_index(self) -> bool:
        """Loads serialized FAISS index and metadata."""
        index_file = self.vector_store_dir / "index.faiss"
        metadata_file = self.vector_store_dir / "metadata.pkl"

        if not index_file.exists() or not metadata_file.exists() or faiss is None:
            return False

        try:
            self.index = faiss.read_index(str(index_file))
            with open(metadata_file, "rb") as f:
                data = pickle.load(f)
                self.chunks_metadata = data.get("chunks", [])
                self.embedding_dimension = data.get("dimension", 384)
                self.indexed_file_hashes = data.get("file_hashes", {})
            return True
        except Exception:
            return False

    def clear_index(self) -> None:
        """Wipes the active index and disk artifacts."""
        self.index = None
        self.chunks_metadata = []
        self.indexed_file_hashes = {}
        index_file = self.vector_store_dir / "index.faiss"
        metadata_file = self.vector_store_dir / "metadata.pkl"
        if index_file.exists():
            index_file.unlink()
        if metadata_file.exists():
            metadata_file.unlink()

    # ==========================================
    # Query Intent Classification
    # ==========================================
    def classify_query(self, query: str) -> str:
        """
        Classifies user query to select the optimal retrieval strategy:
        - IDENTITY: Full name, candidate profile, contact details
        - EXPERIENCE: All jobs, internships, work experience, companies, employers, roles
        - EDUCATION: College, university, degree, coursework, GPA/CGPA
        - SKILLS: Technical skills, design tools, software proficiencies, frameworks
        - PROJECTS: Capstone, academic projects, software or designs built
        - PUBLICATIONS: Research papers, conferences, publications, author
        - RESEARCH_INTERESTS: Research interests, areas of research
        - ACHIEVEMENTS: Measurable impact, performance metrics, accomplishments
        - GENERAL: Other factual inquiries
        """
        q = query.lower()

        # 1. Experience / Employment / Jobs (Check early: questions often mention 'candidate's jobs')
        if re.search(r"\b(job|jobs|internship|internships|intern|interned|experience|work experience|work history|employment|positions?|employers?|companies|company|worked at|worked for|career|roles?)\b", q):
            return "EXPERIENCE"

        # 2. Research Interests
        if re.search(r"\b(research\s+interests?|areas?\s+of\s+research|research\s+areas?|fields?\s+of\s+research|topics?\s+of\s+research|research\s+topics?)\b", q) or ("research" in q and any(w in q for w in ["interest", "interests", "area", "areas", "field", "fields", "topic", "topics"])):
            return "RESEARCH_INTERESTS"

        # 3. Publications
        if re.search(r"\b(publication|publications|paper|papers|conference|published|ieee|springer|author|authored)\b", q):
            return "PUBLICATIONS"

        # 4. Education
        if re.search(r"\b(college|university|school|institute|attend|enrolled|degree|bachelor|b\.?tech|b\.?f\.?a\.?|master|ph\.?d|cgpa|gpa|percentage|marksheet|coursework|education|studied|major)\b", q):
            return "EDUCATION"

        # 5. Skills
        if re.search(r"\b(skill|skills|programming language|languages|technologies|tech stack|tools|libraries|frameworks|database|software|proficiencies)\b", q):
            return "SKILLS"

        # 6. Projects
        if re.search(r"\b(project|projects|capstone|developed|built|system design|application)\b", q):
            return "PROJECTS"

        # 7. Achievements
        if re.search(r"\b(achievement|achievements|accomplishment|accomplishments|impact|metrics|award|awards|honors?)\b", q):
            return "ACHIEVEMENTS"

        # 8. Identity / Contact (Avoid bare 'candidate' or 'student' so they don't hijack other queries)
        if re.search(r"\b(name|who is|who am i|my name|full name|student('s)? name|candidate('s)? name|contact|email|phone|location|address|linkedin)\b", q):
            return "IDENTITY"

        return "GENERAL"

    # ==========================================
    # Hybrid Retrieval (Semantic + Lexical)
    # ==========================================
    @staticmethod
    def _stem_word(w: str) -> str:
        """Simple morphological stemmer for English query keywords."""
        word = w.lower()
        for suffix in ["ships", "ship", "ies", "ing", "ed", "es", "s"]:
            if word.endswith(suffix) and len(word) > len(suffix) + 3:
                return word[:-len(suffix)]
        return word

    def _extract_query_tokens(self, query: str) -> List[str]:
        """Extracts meaningful keywords from user query for lexical scoring."""
        words = re.findall(r"\b[a-zA-Z0-9_\+\#\.\-]{2,}\b", query.lower())
        return [w for w in words if w not in self.STOPWORDS]

    def _compute_lexical_score(self, chunk_text: str, tokens: List[str]) -> float:
        """Computes lexical relevance score with root/stem matching."""
        if not tokens or not chunk_text:
            return 0.0

        text_lower = chunk_text.lower()
        matched = 0
        bonus = 0.0

        for token in tokens:
            stem = self._stem_word(token)
            if token in text_lower or stem in text_lower:
                matched += 1
                freq = text_lower.count(token) + (text_lower.count(stem) if stem != token else 0)
                bonus += min(freq * 0.05, 0.25)

        base_score = matched / len(tokens)
        return min(base_score + bonus, 1.0)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval tailored to the detected query type:
        1. Classifies query intent.
        2. Gathers candidate chunks via FAISS dense search.
        3. Computes lexical scores.
        4. Applies intent-specific boosting & multi-chunk aggregation.
        5. Returns deduplicated relevant chunks sorted by combined score.
        """
        if self.index is None or not self.chunks_metadata:
            return []

        query_clean = query.strip()
        if not query_clean:
            return []

        k = top_k or self.top_k
        threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold
        query_type = self.classify_query(query_clean)
        tokens = self._extract_query_tokens(query_clean)

        # 1. Semantic Search across candidates
        query_vec = self.embedder.encode(
            [query_clean],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        candidate_n = min(self.candidate_count, len(self.chunks_metadata))
        dense_scores, dense_indices = self.index.search(query_vec, candidate_n)

        # Map dense scores
        chunk_score_map = {}
        for score, idx in zip(dense_scores[0], dense_indices[0]):
            if idx != -1 and idx < len(self.chunks_metadata):
                chunk_score_map[idx] = float(score)

        # 2. Score all candidate chunks
        all_candidate_indices = set(chunk_score_map.keys())

        # For list questions, identity, or research interests, scan all chunks for content matches
        if query_type in {"IDENTITY", "EXPERIENCE", "INTERNSHIPS", "EDUCATION", "SKILLS", "PROJECTS", "PUBLICATIONS", "RESEARCH_INTERESTS", "ACHIEVEMENTS"}:
            for idx, ch in enumerate(self.chunks_metadata):
                ch_text = ch["text"].lower()
                sec = ch.get("section_hint", "")

                if query_type == "IDENTITY" and (ch.get("is_header") or ch.get("chunk_id", -1) == 0 or "@" in ch_text or "phone" in ch_text or "contact" in ch_text):
                    all_candidate_indices.add(idx)
                elif query_type in {"EXPERIENCE", "INTERNSHIPS"} and (
                    sec == "EXPERIENCE" or
                    re.search(r"\b(20\d\d|19\d\d)\s*[-–—to]+\s*(present|current|20\d\d|\d{4})\b", ch_text) or
                    any(k in ch_text for k in ["intern", "experience", "designer", "developer", "engineer", "manager", "lead", "worked with", "collaborated with", "company", "employer"])
                ):
                    all_candidate_indices.add(idx)
                elif query_type == "EDUCATION" and (
                    sec == "EDUCATION" or
                    any(k in ch_text for k in ["university", "college", "school", "degree", "b.tech", "b.f.a", "bachelor", "master", "coursework", "cgpa", "gpa"])
                ):
                    all_candidate_indices.add(idx)
                elif query_type == "SKILLS" and (
                    sec == "SKILLS" or
                    any(k in ch_text for k in ["skills", "technologies", "tools", "adobe", "python", "programming", "software", "photoshop", "illustrator", "html", "css"])
                ):
                    all_candidate_indices.add(idx)
                elif query_type == "PROJECTS" and (sec == "PROJECTS" or "project" in ch_text or "capstone" in ch_text):
                    all_candidate_indices.add(idx)
                elif query_type == "ACHIEVEMENTS" and (
                    re.search(r"\b\d+%\b", ch_text) or
                    any(k in ch_text for k in ["improved", "boosted", "slashed", "increased", "reduced", "launched", "reached over"])
                ):
                    all_candidate_indices.add(idx)
                elif query_type == "PUBLICATIONS" and (
                    sec == "PUBLICATIONS" or
                    any(k in ch_text for k in [
                        "publication", "paper", "conference", "accepted at",
                        "ieee", "springer", "scopus", "journal", "proceedings",
                        "arxiv", "lnns", "lncs", "iccisd", "indea"
                    ])
                ):
                    all_candidate_indices.add(idx)
                elif query_type == "RESEARCH_INTERESTS" and (
                    sec in {"PUBLICATIONS", "PROJECTS"} or
                    any(k in ch_text for k in ["research", "publication", "paper", "conference", "springer", "ieee", "patent"])
                ):
                    all_candidate_indices.add(idx)

        # 3. Combine Dense and Lexical Scores
        ranked_chunks = []
        for idx in all_candidate_indices:
            ch = dict(self.chunks_metadata[idx])
            dense_score = chunk_score_map.get(idx, 0.0)
            lex_score = self._compute_lexical_score(ch["text"], tokens)

            # Intent-specific boosts
            intent_boost = 0.0
            sec = ch.get("section_hint", "")
            ch_lower = ch["text"].lower()

            if query_type == "IDENTITY" and (ch.get("is_header") or "@" in ch["text"]):
                intent_boost += 0.50
            elif query_type in {"EXPERIENCE", "INTERNSHIPS"}:
                if sec == "EXPERIENCE" or re.search(r"\b(20\d\d|19\d\d)\s*[-–—to]+\s*(present|current|20\d\d|\d{4})\b", ch["text"]):
                    intent_boost += 0.45
                elif any(k in ch_lower for k in ["designer", "engineer", "intern", "developer", "worked with"]):
                    intent_boost += 0.35
            elif query_type == "EDUCATION" and (sec == "EDUCATION" or "university" in ch_lower or "college" in ch_lower or "degree" in ch_lower):
                intent_boost += 0.40
            elif query_type == "SKILLS" and (sec == "SKILLS" or any(s in ch_lower for s in ["photoshop", "illustrator", "python", "skills", "tools"])):
                intent_boost += 0.35
            elif query_type == "ACHIEVEMENTS" and re.search(r"\b\d+%\b", ch["text"]):
                intent_boost += 0.40
            elif query_type == "PROJECTS" and sec == "PROJECTS":
                intent_boost += 0.35
            elif query_type == "PUBLICATIONS":
                if sec == "PUBLICATIONS" or "publication" in ch_lower:
                    intent_boost += 0.55
                elif any(k in ch_lower for k in ["accepted at", "ieee", "springer", "scopus", "conference", "paper", "journal", "proceedings"]):
                    intent_boost += 0.45
            elif query_type == "RESEARCH_INTERESTS":
                if "research interest" in ch_lower or "areas of research" in ch_lower:
                    intent_boost += 0.55
                elif sec == "PUBLICATIONS" or "publication" in ch_lower or "conference" in ch_lower:
                    intent_boost += 0.45
                elif sec == "PROJECTS" or "project" in ch_lower:
                    intent_boost += 0.35

            # Hybrid formula
            hybrid_score = (0.55 * dense_score) + (0.30 * lex_score) + intent_boost

            ch["similarity_score"] = round(float(dense_score), 4)
            ch["lexical_score"] = round(float(lex_score), 4)
            ch["hybrid_score"] = round(float(hybrid_score), 4)

            # Accept chunk if dense >= threshold OR if hybrid/lexical match is strong
            passes_threshold = (
                dense_score >= threshold or
                lex_score >= 0.35 or
                intent_boost > 0.25 or
                hybrid_score >= (threshold + 0.10)
            )

            if passes_threshold:
                ranked_chunks.append(ch)

        # 4. Sort by hybrid score descending
        ranked_chunks.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # For list questions, return all matching chunks to guarantee complete enumeration
        if query_type in {"EXPERIENCE", "INTERNSHIPS", "SKILLS", "PROJECTS", "PUBLICATIONS", "RESEARCH_INTERESTS", "ACHIEVEMENTS"}:
            effective_k = min(max(k, 10), len(ranked_chunks))
        else:
            effective_k = min(k, len(ranked_chunks))

        return ranked_chunks[:effective_k]

    # ==========================================
    # Structured Evidence Extractor for Lists
    # ==========================================
    def extract_structured_evidence(self, chunks: List[Dict[str, Any]], query_type: str) -> List[str]:
        """
        Extracts clean, discrete factual entries from chunks to ensure
        the LLM does not omit items found in secondary passages.
        Works across diverse fields, layouts, and document formats.
        """
        evidence_items = []
        seen = set()

        if query_type in {"EXPERIENCE", "INTERNSHIPS"}:
            date_pat = re.compile(r"\b(20\d\d|19\d\d)\s*[-–—to]+\s*(present|current|20\d\d|\d{4})\b", re.I)
            role_pat = re.compile(r"\b(designer|developer|engineer|intern|manager|lead|director|specialist|analyst|associate|consultant|officer|assistant|founder|co-founder|architect|researcher)\b", re.I)

            for ch in chunks:
                lines = [l.strip() for l in ch["text"].split("\n") if l.strip()]
                for i, l in enumerate(lines):
                    # Filter out generic section headers
                    if l.upper() in {"EXPERIENCE", "WORK EXPERIENCE", "EMPLOYMENT", "PROFESSIONAL EXPERIENCE", "INTERNSHIP EXPERIENCE"}:
                        continue

                    # Pattern 1: Inline formatted entry: Company — Role (Dates)
                    if ("—" in l or " - " in l) and (date_pat.search(l) or role_pat.search(l)):
                        normalized = re.sub(r"\s+", " ", l)
                        if normalized not in seen and len(normalized) > 8:
                            seen.add(normalized)
                            evidence_items.append(f"• {normalized} (Source: {ch['source']})")
                        continue

                    # Pattern 2: Multiline structure (Role \n Company \n Dates)
                    if date_pat.search(l):
                        role_str = ""
                        comp_str = ""
                        if i >= 2 and not date_pat.search(lines[i-1]) and not date_pat.search(lines[i-2]):
                            comp_str = lines[i-1]
                            role_str = lines[i-2]
                        elif i >= 1 and not date_pat.search(lines[i-1]):
                            role_str = lines[i-1]

                        if role_str:
                            entry = f"{role_str}{' at ' + comp_str if comp_str else ''} ({l})"
                            if entry not in seen:
                                seen.add(entry)
                                evidence_items.append(f"• {entry} (Source: {ch['source']})")
                        elif l not in seen and len(l) > 6:
                            seen.add(l)
                            evidence_items.append(f"• {l} (Source: {ch['source']})")

                    # Pattern 3: Explicit role indicator
                    elif role_pat.search(l) and len(l) < 60 and not l.endswith("."):
                        if i + 1 < len(lines) and not date_pat.search(lines[i+1]) and len(lines[i+1]) < 50:
                            entry = f"{l} at {lines[i+1]}"
                        else:
                            entry = l
                        if entry not in seen and len(entry) > 6:
                            seen.add(entry)
                            evidence_items.append(f"• {entry} (Source: {ch['source']})")

        elif query_type == "ACHIEVEMENTS":
            metric_pat = re.compile(r"(\b\d+%\b|\$\d+|\b\d+\s+(assets|clients|hospitals|members|teams|projects)\b|improved|boosted|slashed|increased|reduced)", re.I)
            for ch in chunks:
                lines = [l.strip() for l in ch["text"].split("\n") if l.strip()]
                for l in lines:
                    if metric_pat.search(l) and len(l) > 15:
                        if l not in seen:
                            seen.add(l)
                            evidence_items.append(f"• {l} (Source: {ch['source']})")

        elif query_type == "EDUCATION":
            for ch in chunks:
                lines = [l.strip() for l in ch["text"].split("\n") if l.strip()]
                for l in lines:
                    if re.search(r"\b(b\.?s\.?|b\.?tech|b\.?f\.?a\.?|bachelor|master|ph\.?d|degree|university|college|institute)\b", l, re.I):
                        if l.upper() not in {"EDUCATION", "ACADEMIC BACKGROUND"}:
                            if l not in seen and len(l) > 6:
                                seen.add(l)
                                evidence_items.append(f"• {l} (Source: {ch['source']})")

        elif query_type == "PUBLICATIONS":
            for ch in chunks:
                lines = ch["text"].split("\n")
                for i, line in enumerate(lines):
                    line_clean = line.strip()
                    if any(term in line_clean.lower() for term in ["accepted at", "ieee", "springer", "scopus", "conference", "journal", "proceedings"]):
                        title = ""
                        for prev_idx in range(i-1, -1, -1):
                            prev_line = lines[prev_idx].strip()
                            if len(prev_line) > 15 and not prev_line.isupper() and not prev_line.upper().startswith("RESEARCH"):
                                title = prev_line
                                break
                        entry = f"{title + ' — ' if title else ''}{line_clean}"
                        if entry not in seen:
                            seen.add(entry)
                            evidence_items.append(f"• {entry} (Source: {ch['source']})")
                    elif len(line_clean) > 25 and not line_clean.isupper() and any(k in line_clean.lower() for k in ["framework", "model", "benchmarking", "federated", "analysis"]):
                        if line_clean not in seen:
                            seen.add(line_clean)
                            evidence_items.append(f"• {line_clean} (Source: {ch['source']})")

        elif query_type == "RESEARCH_INTERESTS":
            for ch in chunks:
                lines = ch["text"].split("\n")
                for line in lines:
                    line_clean = line.strip()
                    if len(line_clean) > 12 and not line_clean.isupper():
                        if any(k in line_clean.lower() for k in ["accepted at", "conference", "ieee", "springer", "paper", "research", "journal", "proceedings", "framework", "pipeline"]):
                            if line_clean not in seen:
                                seen.add(line_clean)
                                evidence_items.append(f"• {line_clean} (Source: {ch['source']})")

        return evidence_items

    @staticmethod
    def inspect_research_evidence(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Inspects chunks for explicit research interest declarations and research-related artifacts
        (publications, conference papers, research topics, patents, research projects).
        """
        has_explicit = False
        explicit_statements = []
        has_research_artifacts = False
        artifacts = []

        explicit_pattern = re.compile(
            r"(research\s+interests?|areas?\s+of\s+research|research\s+focus|fields?\s+of\s+research|topics?\s+of\s+research)\s*[:\-]",
            re.IGNORECASE
        )

        for ch in chunks:
            text = ch["text"]
            text_lower = text.lower()
            sec = ch.get("section_hint", "")

            # 1. Check for explicit statements
            for line in text.split("\n"):
                line_clean = line.strip()
                if explicit_pattern.search(line_clean):
                    has_explicit = True
                    explicit_statements.append(f"{line_clean} (Source: {ch['source']})")

            # 2. Check for research artifacts (publications, conference papers, research projects)
            is_pub = (sec == "PUBLICATIONS" or any(k in text_lower for k in [
                "accepted at", "conference", "ieee", "springer", "proceedings",
                "scopus", "journal", "arxiv", "paper"
            ]))
            is_proj = (sec == "PROJECTS" or any(k in text_lower for k in [
                "research project", "diagnostic engine", "deep learning pipeline",
                "multimodal", "graph neural", "federated", "transformer",
                "explainable ai", "grad-cam", "patent", "capstone"
            ]))

            if is_pub or is_proj:
                has_research_artifacts = True
                for line in text.split("\n"):
                    l_str = line.strip()
                    if len(l_str) > 15 and not l_str.isupper():
                        if any(term in l_str.lower() for term in ["accepted", "paper", "journal", "conference", "developed", "proposed", "model", "framework", "pipeline"]):
                            artifacts.append(f"• [{ch['source']}] {l_str}")

        return {
            "has_explicit": has_explicit,
            "explicit_statements": explicit_statements,
            "has_research_artifacts": has_research_artifacts,
            "artifacts": artifacts
        }

    # ==========================================
    # LLM Invocation
    # ==========================================
    def call_ollama(self, prompt: str, model: Optional[str] = None) -> Tuple[bool, str]:
        """Calls local Ollama API with grounded prompt."""
        active_model = model or self.ollama_model
        payload = {
            "model": active_model,
            "prompt": prompt,
            "system": self.GROUNDED_SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": 0.05,  # Very low temperature for strict adherence
                "top_p": 0.9
            }
        }

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=req_data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status == 200:
                    resp_body = json.loads(resp.read().decode("utf-8"))
                    answer = resp_body.get("response", "").strip()
                    return True, answer
                return False, f"Ollama HTTP error {resp.status}"
        except urllib.error.URLError as e:
            return False, f"Ollama connection error: {str(e.reason)}"
        except Exception as e:
            return False, f"Ollama call failed: {str(e)}"

    # ==========================================
    # End-to-End Answer Pipeline
    # ==========================================
    def answer_question(
        self,
        query: str,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes complete RAG workflow:
        1. Classifies query intent.
        2. Performs hybrid multi-chunk retrieval.
        3. If evidence is lacking, responds with standard unavailable message.
        4. Injects structured evidence and prompt constraints.
        5. Generates grounded answer via Ollama or executes extractive fallback.
        """
        query_clean = query.strip()
        if not query_clean:
            return {
                "answer": "Please enter a valid, non-empty question.",
                "retrieved_chunks": [],
                "sources": [],
                "query_type": "EMPTY",
                "llm_used": None,
                "status": "empty_query"
            }

        if self.index is None or not self.chunks_metadata:
            return {
                "answer": "Knowledge base is empty. Please upload or index your personal documents first.",
                "retrieved_chunks": [],
                "sources": [],
                "query_type": "EMPTY_INDEX",
                "llm_used": None,
                "status": "empty_index"
            }

        query_type = self.classify_query(query_clean)

        # 1. Retrieve hybrid chunks
        retrieved_chunks = self.retrieve(
            query=query_clean,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        if not retrieved_chunks:
            return {
                "answer": self.UNAVAILABLE_RESPONSE,
                "retrieved_chunks": [],
                "sources": [],
                "query_type": query_type,
                "llm_used": None,
                "status": "no_relevant_context"
            }

        # Check if the query asks for CGPA / GPA when it is absent
        if "cgpa" in query_clean.lower() or "gpa" in query_clean.lower() or "percentage" in query_clean.lower():
            has_gpa_text = any(
                re.search(r"\b(cgpa|gpa|percentage|marks)\s*[:\-=]\s*[0-9]", ch["text"], re.I)
                for ch in retrieved_chunks
            )
            if not has_gpa_text:
                return {
                    "answer": self.UNAVAILABLE_RESPONSE,
                    "retrieved_chunks": retrieved_chunks,
                    "sources": sorted(list({c["source"] for c in retrieved_chunks})),
                    "query_type": query_type,
                    "llm_used": None,
                    "status": "missing_fact"
                }

        sources = sorted(list({c["source"] for c in retrieved_chunks}))

        # Check if the query asks for Research Interests
        if query_type == "RESEARCH_INTERESTS":
            research_info = self.inspect_research_evidence(retrieved_chunks)
            if not research_info["has_explicit"] and not research_info["has_research_artifacts"]:
                return {
                    "answer": self.NO_RESEARCH_EVIDENCE_RESPONSE,
                    "retrieved_chunks": retrieved_chunks,
                    "sources": sources,
                    "query_type": query_type,
                    "llm_used": None,
                    "status": "insufficient_research_evidence"
                }

        # 2. Structured Evidence Collection
        structured_evidence = self.extract_structured_evidence(retrieved_chunks, query_type)
        evidence_block = ""
        if structured_evidence:
            evidence_block = "\nIdentified Key Entries across all excerpts:\n" + "\n".join(structured_evidence) + "\n"

        # 3. Format Context Passages
        context_parts = []
        for i, c in enumerate(retrieved_chunks, 1):
            page_info = f" | Page {c.get('page_number', 1)}" if c.get("doc_type") == ".pdf" else ""
            context_parts.append(
                f"[Document: {c['source']}{page_info} | Chunk ID: {c['chunk_id']} | Section: {c.get('section_hint', 'GENERAL')} | Relevance: {c.get('hybrid_score', 0):.2f}]\n"
                f"{c['text']}"
            )
        formatted_context = "\n\n---\n\n".join(context_parts)

        # 4. Construct Prompt based on Intent
        if query_type == "RESEARCH_INTERESTS":
            user_prompt = (
                f"Context Information:\n"
                f"---------------------\n"
                f"{formatted_context}\n"
                f"{evidence_block}"
                f"---------------------\n\n"
                f"Question: {query_clean}\n"
                f"Detected Query Category: RESEARCH_INTERESTS\n\n"
                f"Specific Instructions for Answering Research Interests:\n"
                f"1. Check if the provided context explicitly states the student's research interests (for example, under a 'Research Interests' or 'Areas of Research' heading).\n"
                f"   - If explicitly stated: State them directly as explicitly declared research interests and cite the source filename.\n"
                f"2. If research interests are NOT explicitly stated, infer likely research interests from the student's published papers, research topics, patents, and research projects in the retrieved excerpts:\n"
                f"   - State clearly: 'The student\\'s research interests are not explicitly declared in the provided documents.'\n"
                f"   - Synthesize recurring themes and map them to recognized AI/ML research areas under the clear heading: '**Inferred Research Interests**'.\n"
                f"   - For each inferred research interest, provide a brief explanation of which specific publication(s) or project(s) support it.\n"
                f"   - Cite the source document filenames for all referenced publications and projects.\n"
                f"   - Do NOT present inferred interests as explicitly declared facts.\n"
                f"3. If the context contains neither explicit research interests nor sufficient research evidence to make any inference, respond with EXACTLY:\n"
                f"   \"{self.NO_RESEARCH_EVIDENCE_RESPONSE}\"\n\n"
                f"Answer:"
            )
        else:
            user_prompt = (
                f"Context Information:\n"
                f"---------------------\n"
                f"{formatted_context}\n"
                f"{evidence_block}"
                f"---------------------\n\n"
                f"Question: {query_clean}\n"
                f"Detected Query Category: {query_type}\n\n"
                f"Instructions:\n"
                f"1. Answer using ONLY the supplied context above. Do not assume or extrapolate.\n"
                f"2. If the question asks about jobs, employment history, internships, or positions, enumerate EVERY distinct position, employer, and employment date range found across all supplied excerpts. Mention company names, roles, dates, and key achievements. Do not stop after the first job.\n"
                f"3. If the question asks about research publications or papers, enumerate EVERY distinct research paper found across all supplied excerpts with full titles, venues, and citations. Do not stop after the first paper.\n"
                f"4. For list questions (such as skills, software proficiencies, or projects), enumerate EVERY distinct entry found across all excerpts.\n"
                f"5. If the context does not contain the answer, say exactly: \"{self.UNAVAILABLE_RESPONSE}\"\n"
                f"6. Never invent grades, names, dates, companies, or technologies.\n"
                f"7. Cite the source document filename for your answer.\n\n"
                f"Answer:"
            )

        # 5. Invoke Ollama
        active_model = model or self.ollama_model
        success, response_text = self.call_ollama(user_prompt, model=active_model)

        if success and response_text:
            return {
                "answer": response_text,
                "retrieved_chunks": retrieved_chunks,
                "sources": sources,
                "query_type": query_type,
                "llm_used": active_model,
                "status": "success"
            }

        # 6. Fallback Mode (Extractive Retrieval)
        if query_type == "RESEARCH_INTERESTS":
            research_info = self.inspect_research_evidence(retrieved_chunks)
            if research_info["has_explicit"]:
                fallback_msg = (
                    f"*[Notice: Local Ollama generation is currently offline. "
                    f"Displaying explicitly stated research interests directly from your documents.]*\n\n"
                    f"**Explicitly Stated Research Interests:**\n"
                )
                for stmt in research_info["explicit_statements"]:
                    fallback_msg += f"- {stmt}\n"
                return {
                    "answer": fallback_msg,
                    "retrieved_chunks": retrieved_chunks,
                    "sources": sources,
                    "query_type": query_type,
                    "llm_used": "Fallback (Extractive Retrieval)",
                    "status": "fallback"
                }
            elif research_info["has_research_artifacts"]:
                fallback_msg = (
                    f"*[Notice: Local Ollama generation is currently offline. "
                    f"Displaying research evidence directly from your documents.]*\n\n"
                    f"The student's research interests are not explicitly declared in the provided documents.\n\n"
                    f"**Inferred Research Interests** (Derived from publications and projects in the knowledge base):\n"
                )
                for art in research_info["artifacts"][:6]:
                    fallback_msg += f"{art}\n"
                fallback_msg += f"\n*Supported by sources:* {', '.join(sources)}"
                return {
                    "answer": fallback_msg,
                    "retrieved_chunks": retrieved_chunks,
                    "sources": sources,
                    "query_type": query_type,
                    "llm_used": "Fallback (Extractive Retrieval)",
                    "status": "fallback"
                }
            else:
                return {
                    "answer": self.NO_RESEARCH_EVIDENCE_RESPONSE,
                    "retrieved_chunks": retrieved_chunks,
                    "sources": sources,
                    "query_type": query_type,
                    "llm_used": None,
                    "status": "insufficient_research_evidence"
                }

        fallback_msg = (
            f"*[Notice: Local Ollama generation is currently offline ({response_text}). "
            f"Displaying grounded information retrieved directly from your indexed documents.]*\n\n"
        )

        if structured_evidence:
            fallback_msg += "**Document Entries Found:**\n" + "\n".join(structured_evidence) + "\n\n"

        fallback_msg += "**Relevant Extracted Excerpts:**\n"
        for i, c in enumerate(retrieved_chunks, 1):
            fallback_msg += f"- **[{c['source']} (Chunk #{c['chunk_id']}, Score: {c.get('hybrid_score', 0):.2f})]**:\n> {c['text']}\n\n"

        return {
            "answer": fallback_msg,
            "retrieved_chunks": retrieved_chunks,
            "sources": sources,
            "query_type": query_type,
            "llm_used": "Fallback (Extractive Retrieval)",
            "status": "fallback"
        }
