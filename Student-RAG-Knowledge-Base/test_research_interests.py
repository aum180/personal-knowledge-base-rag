"""
test_research_interests.py
==========================
Validates the three research interest query scenarios:
Case 1: Explicitly stated research interests.
Case 2: Inferred research interests from published papers and projects.
Case 3: No research-related evidence in documents.
"""

import os
import sys
import shutil
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from document_processor import DocumentProcessor
from rag_pipeline import RAGPipeline


class TestResearchInterestsHandling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(__file__).resolve().parent / "temp_research_test_data"
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.processor = DocumentProcessor(chunk_size=600, chunk_overlap=100)

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_case_1_explicit_research_interests(self):
        """Case 1: Document explicitly states research interests -> returned directly."""
        v_dir = self.test_dir / "vstore_case1"
        v_dir.mkdir(parents=True, exist_ok=True)
        doc = self.test_dir / "explicit_doc.txt"
        doc.write_text(
            "Student Name: Maya Lin\n"
            "Department of Computer Science\n"
            "Research Interests: Reinforcement Learning, Multi-Agent Systems, and Quantum Computing.\n"
            "Core Courses: Advanced Algorithms, Operating Systems.\n",
            encoding="utf-8"
        )

        chunks = self.processor.chunk_text(doc.read_text(encoding="utf-8"), "explicit_doc.txt")
        pipeline = RAGPipeline(vector_store_dir=str(v_dir))
        pipeline.build_index(chunks)

        res = pipeline.answer_question("What are the student's research interests?")
        self.assertEqual(res["query_type"], "RESEARCH_INTERESTS")
        self.assertIn("explicit_doc.txt", res["sources"])

        ans = res["answer"]
        print("\n--- CASE 1 (EXPLICIT) ANSWER ---")
        print(ans)

        # Must mention reinforcement learning, quantum computing or multi-agent systems
        self.assertTrue(
            "reinforcement learning" in ans.lower() or
            "quantum" in ans.lower() or
            "multi-agent" in ans.lower()
        )

    def test_case_2_inferred_research_interests(self):
        """Case 2: Research interests not explicitly stated, but papers/projects exist -> inferred and labeled."""
        v_dir = self.test_dir / "vstore_case2"
        v_dir.mkdir(parents=True, exist_ok=True)
        doc = self.test_dir / "publications_doc.txt"
        doc.write_text(
            "Student Profile: David Zhang\n"
            "RESEARCH PUBLICATIONS\n"
            "Paper 1: Towards Trustworthy Multimodal LLMs: Benchmarking Hallucination in Vision-Language Models\n"
            "Accepted at INDEA 2026 (Springer LNNS)\n"
            "Developed an interpretability scoring benchmark for vision transformers and multimodal explanation grounding.\n\n"
            "Paper 2: Physics-Informed Graph Neural Networks for Decentralized Smart-Grid State Estimation\n"
            "Accepted at IEEE Conference on Smart Systems 2026\n"
            "Engineered federated graph learning algorithms incorporating physical Kirchhoff constraints on electrical power grids.\n\n"
            "ACADEMIC PROJECTS\n"
            "Project MedVision: Explainable AI pipeline with Grad-CAM for medical retinal scan pathology segmentation.\n",
            encoding="utf-8"
        )

        chunks = self.processor.chunk_text(doc.read_text(encoding="utf-8"), "publications_doc.txt")
        pipeline = RAGPipeline(vector_store_dir=str(v_dir))
        pipeline.build_index(chunks)

        res = pipeline.answer_question("What are the student's research interests?")
        self.assertEqual(res["query_type"], "RESEARCH_INTERESTS")
        ans = res["answer"]
        print("\n--- CASE 2 (INFERRED) ANSWER ---")
        print(ans)

        # Must clearly include label 'Inferred Research Interests'
        self.assertIn("inferred research interests", ans.lower())
        # Must connect to publications / projects
        self.assertTrue(
            "multimodal" in ans.lower() or
            "graph" in ans.lower() or
            "federated" in ans.lower() or
            "indea" in ans.lower() or
            "ieee" in ans.lower()
        )

    def test_case_3_no_research_evidence(self):
        """Case 3: Document has no research evidence -> returns exact unavailable response."""
        v_dir = self.test_dir / "vstore_case3"
        v_dir.mkdir(parents=True, exist_ok=True)
        doc = self.test_dir / "no_research_doc.txt"
        doc.write_text(
            "Student Profile: Kevin Cole\n"
            "Education: High School Diploma, General Studies\n"
            "Activities: University Tennis Club, Library Assistant\n"
            "Work Experience: Cashier at Retail Grocery Store (2022-2023)\n",
            encoding="utf-8"
        )

        chunks = self.processor.chunk_text(doc.read_text(encoding="utf-8"), "no_research_doc.txt")
        pipeline = RAGPipeline(vector_store_dir=str(v_dir))
        pipeline.build_index(chunks)

        res = pipeline.answer_question("What are the student's research interests?")
        ans = res["answer"]
        print("\n--- CASE 3 (NO RESEARCH EVIDENCE) ANSWER ---")
        print(ans)

        self.assertEqual(res["query_type"], "RESEARCH_INTERESTS")
        self.assertEqual(
            ans.strip(),
            pipeline.NO_RESEARCH_EVIDENCE_RESPONSE.strip()
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
