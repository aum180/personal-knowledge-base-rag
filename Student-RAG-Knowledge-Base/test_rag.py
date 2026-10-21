"""
test_rag.py
===========
Automated Regression Test Suite for Experiment 8.
Verifies all 10 requirements using temporary synthetic fixtures:
1. Resume containing full name and university name extraction.
2. Multi-chunk chunking across distinct internship entries.
3. Retrieval of ALL internships across multiple chunks.
4. Retrieval of university/institution chunk.
5. Aggregation of skills across multiple chunks.
6. Missing CGPA returns unavailable response (no hallucination).
7. Explicit facts are not rejected by similarity threshold.
8. Stale index replacement when documents change.
9. Empty and scanned image PDF reporting.
10. Full system resilience and extractive fallback when Ollama is offline.
"""

import os
import sys
import shutil
import unittest
import warnings
from pathlib import Path

# Suppress harmless sklearn/scipy version mismatch warnings
warnings.filterwarnings("ignore", category=UserWarning)

if "torchaudio" not in sys.modules:
    sys.modules["torchaudio"] = None

from document_processor import DocumentProcessor
from rag_pipeline import RAGPipeline


class TestPersonalRAGRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Creates temporary test directory and synthetic documents."""
        cls.test_dir = Path(__file__).resolve().parent / "temp_test_suite_data"
        cls.test_vector_dir = cls.test_dir / "test_vector_store"
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.test_vector_dir.mkdir(parents=True, exist_ok=True)

        cls.processor = DocumentProcessor(chunk_size=600, chunk_overlap=120)
        cls.pipeline = RAGPipeline(
            vector_store_dir=str(cls.test_vector_dir),
            similarity_threshold=0.20,
            top_k=6,
            candidate_count=12
        )

        # 1. Synthetic Resume with Name, University, 3 Distinct Internships, Skills, Projects (NO CGPA)
        cls.resume_file = cls.test_dir / "synthetic_student_resume.txt"
        cls.resume_content = (
            "ELENA ROSTOVA\n"
            "Email: elena.rostova.test@example.com | Phone: +1-555-0199 | Location: Boston, MA\n\n"
            "PROFESSIONAL SUMMARY\n"
            "Computer Science student specializing in Artificial Intelligence and Machine Learning.\n\n"
            "EDUCATION\n"
            "Bachelor of Science in Computer Science and Engineering\n"
            "Metropolitan Institute of Technology | Enrolled: 2022 - 2026\n"
            "Core Subjects: Data Structures, Deep Learning, Distributed Systems, Database Management.\n\n"
            "TECHNICAL SKILLS\n"
            "Programming Languages: Python, C++, Go, Rust, TypeScript\n"
            "Machine Learning Frameworks: PyTorch, JAX, Scikit-learn, HuggingFace Transformers\n"
            "Cloud and DevOps Tools: Docker, Kubernetes, Terraform, AWS, Linux\n\n"
            "INTERNSHIP EXPERIENCE\n\n"
            "Machine Learning Intern | Apex Robotics Labs (Jun 2024 - Aug 2024)\n"
            "Developed real-time computer vision perception pipelines for autonomous mobile robots. "
            "Optimized YOLOv8 inference latency on embedded NVIDIA Jetson devices using TensorRT.\n\n"
            "Backend Engineering Intern | CloudScale Systems (Jan 2024 - May 2024)\n"
            "Engineered distributed REST microservices in Go and PostgreSQL handling 5,000 requests per second. "
            "Integrated Redis distributed caching to decrease database read load by 40 percent.\n\n"
            "Data Science Intern | FinTech Analytics Group (Jun 2023 - Aug 2023)\n"
            "Built fraud detection classification models utilizing XGBoost and LightGBM on transaction streams. "
            "Evaluated feature importance and deployed model scoring pipelines.\n\n"
            "KEY PROJECTS\n"
            "Project MedScan: Multimodal diagnostic assistant built with PyTorch and FastAPI for MRI segmentations.\n"
        )
        cls.resume_file.write_text(cls.resume_content, encoding="utf-8")

        # 2. Synthetic Scanned PDF (0 text)
        cls.scanned_pdf_file = cls.test_dir / "scanned_certificate.pdf"
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(str(cls.scanned_pdf_file), pagesize=letter)
            # Empty page or blank image canvas
            c.showPage()
            c.save()
        except Exception:
            # Fallback dummy bytes if reportlab not available
            cls.scanned_pdf_file.write_bytes(b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R>>endobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000052 00000 n \n0000000108 00000 n \ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n150\n%%EOF")

        # 3. Empty 0-byte file
        cls.empty_file = cls.test_dir / "blank_note.txt"
        cls.empty_file.write_bytes(b"")

    @classmethod
    def tearDownClass(cls):
        """Removes all temporary synthetic data."""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_resume_name_and_university_extraction(self):
        """Test 1: Full name and university extraction from resume."""
        report = self.processor.inspect_and_extract_file(str(self.resume_file))
        self.assertFalse(report["is_empty"])
        self.assertIn("ELENA ROSTOVA", report["text"])
        self.assertIn("Metropolitan Institute of Technology", report["text"])
        self.assertGreater(report["char_count"], 500)

    def test_02_resume_multi_internship_chunking(self):
        """Test 2: Verification of multi-chunk chunking across distinct internships."""
        chunks = self.processor.chunk_text(self.resume_content, "synthetic_student_resume.txt")
        self.assertGreaterEqual(len(chunks), 3)

        chunk_texts = " ".join([c["text"] for c in chunks])
        self.assertIn("Apex Robotics Labs", chunk_texts)
        self.assertIn("CloudScale Systems", chunk_texts)
        self.assertIn("FinTech Analytics Group", chunk_texts)

    def test_03_all_internships_retrieval(self):
        """Test 3: Query requesting all internships retrieves all three distinct companies."""
        chunks = self.processor.chunk_text(self.resume_content, "synthetic_student_resume.txt")
        self.pipeline.build_index(chunks)

        query = "Which internships has the student undertaken?"
        result = self.pipeline.answer_question(query)

        self.assertIn(result["status"], ["success", "fallback"])
        retrieved_texts = " ".join([c["text"] for c in result["retrieved_chunks"]])

        self.assertIn("Apex Robotics Labs", retrieved_texts)
        self.assertIn("CloudScale Systems", retrieved_texts)
        self.assertIn("FinTech Analytics Group", retrieved_texts)

    def test_04_university_question_retrieval(self):
        """Test 4: University question retrieves the institution chunk."""
        query = "Which college or university does the student attend?"
        result = self.pipeline.answer_question(query)

        self.assertIn(result["status"], ["success", "fallback"])
        retrieved_texts = " ".join([c["text"] for c in result["retrieved_chunks"]])
        self.assertIn("Metropolitan Institute of Technology", retrieved_texts)

    def test_05_skills_multi_chunk_retrieval(self):
        """Test 5: Skills query retrieves languages and frameworks."""
        query = "What technical skills does the student know?"
        result = self.pipeline.answer_question(query)

        self.assertIn(result["status"], ["success", "fallback"])
        retrieved_texts = " ".join([c["text"] for c in result["retrieved_chunks"]])
        self.assertIn("Python", retrieved_texts)
        self.assertIn("PyTorch", retrieved_texts)

    def test_06_missing_cgpa_returns_unavailable(self):
        """Test 6: Missing CGPA returns unavailable response without hallucinating a number."""
        query = "What is the student's exact CGPA or percentage?"
        result = self.pipeline.answer_question(query)

        answer = result["answer"].lower()
        self.assertTrue(
            "could not find" in answer or
            result["status"] in ["missing_fact", "no_relevant_context"] or
            "unavailable" in answer
        )
        # Ensure no fake grade numbers are present
        self.assertNotIn("3.", answer)
        self.assertNotIn("4.", answer)
        self.assertNotIn("9.", answer)

    def test_07_explicit_fact_not_rejected(self):
        """Test 7: Fact explicitly present (Project MedScan) is not rejected by retrieval."""
        query = "What is Project MedScan?"
        result = self.pipeline.answer_question(query)

        self.assertNotEqual(result["status"], "no_relevant_context")
        retrieved_texts = " ".join([c["text"] for c in result["retrieved_chunks"]])
        self.assertIn("MedScan", retrieved_texts)

    def test_08_stale_index_replaced_on_change(self):
        """Test 8: Re-indexing updates content and removes stale entries."""
        # Add new temporary document
        new_doc = self.test_dir / "temp_award.txt"
        new_doc.write_text("Elena won the International Open Source Fellowship 2025.", encoding="utf-8")

        chunks, succ, fail = self.processor.process_directory(str(self.test_dir))
        self.pipeline.build_index(chunks)

        res = self.pipeline.retrieve("International Open Source Fellowship", top_k=2)
        sources = [c["source"] for c in res]
        self.assertIn("temp_award.txt", sources)

        # Delete document and rebuild
        new_doc.unlink()
        chunks_after, succ_after, _ = self.processor.process_directory(str(self.test_dir))
        self.pipeline.build_index(chunks_after)

        res_after = self.pipeline.retrieve("International Open Source Fellowship", top_k=2)
        sources_after = [c["source"] for c in res_after]
        self.assertNotIn("temp_award.txt", sources_after)

    def test_09_empty_or_scanned_pdf_reported(self):
        """Test 9: Empty file and scanned PDF are detected and reported."""
        empty_rep = self.processor.inspect_and_extract_file(str(self.empty_file))
        self.assertTrue(empty_rep["is_empty"])
        self.assertIsNotNone(empty_rep["error"])

        scanned_rep = self.processor.inspect_and_extract_file(str(self.scanned_pdf_file))
        self.assertTrue(scanned_rep["is_scanned"] or scanned_rep["char_count"] < 40)
        self.assertIn("OCR", scanned_rep.get("error", ""))

    def test_10_offline_ollama_fallback(self):
        """Test 10: Extractive fallback remains fully operational when Ollama is offline."""
        offline_pipeline = RAGPipeline(
            vector_store_dir=str(self.test_vector_dir),
            ollama_url="http://localhost:9998",  # Non-existent port
            similarity_threshold=0.20
        )
        offline_pipeline.load_index()

        result = offline_pipeline.answer_question("What is the student's full name?")
        self.assertEqual(result["status"], "fallback")
        self.assertIn("ELENA ROSTOVA", result["answer"])
        self.assertIn("Fallback", result["llm_used"])

    def test_11_research_interests_inference(self):
        """Test 11: Research interests query correctly infers and labels Inferred Research Interests."""
        result = self.pipeline.answer_question("What are the student's research interests?")
        self.assertEqual(result["query_type"], "RESEARCH_INTERESTS")
        ans = result["answer"].lower()
        self.assertTrue(
            "inferred research interests" in ans or
            "not explicitly stated" in ans
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
