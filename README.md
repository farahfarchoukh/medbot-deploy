#MedBot: Fine-Tuning LLMs for Medical Q&A

Ranked Top 4 in the LLM Fine-Tuning Challenge (Zaka AI – August 2025)

This project focuses on building MedBot, a medical question-answering chatbot, by fine-tuning large language models (LLMs) using LoRA and 4-bit quantization with Hugging Face Transformers. The challenge emphasized efficiency, innovation, and practical deployment under tight GPU and budget constraints.

#Project Summary

The objective was to fine-tune a large language model on a curated medical instruction-response dataset to deliver reliable answers to patient-related questions. Despite limited compute resources, MedBot achieved strong results, placing 4th overall in the competition.

#Tech Stack

Hugging Face Transformers & PEFT

BitsAndBytes (4-bit quantization)

LoRA (Low-Rank Adaptation)

PyTorch

Google Colab / RunPod (A100 GPU)

#Dataset

Medical Meadow WikiDoc

Used as the primary instruction-response dataset for training the chatbot to handle real-world medical Q&A.

#Training Highlights

Fine-tuned GPT-NeoX-20B with LoRA + 4-bit quantization

Training: 1 epoch (~2.5 hours) with gradient accumulation

Only 0.08% of model parameters updated, making training compute-efficient

Debugged and resolved multiple inference issues and CUDA crashes

Attempted fallback on GPT-Neo 2.7B (training crashed at 0.87/1 epoch)

Full pipeline: dataset preprocessing → model setup → fine-tuning → saving → inference

#Results & Takeaways

Delivered concise and medically aligned responses under limited resources

Ranked Top 4 in Zaka AI’s LLM Fine-Tuning Challenge (Aug 2025)

Validated the effectiveness of LoRA + 4-bit quantization for scaling large medical models on budget hardware

Future improvements: integrate retrieval-augmented generation (RAG) with trusted sources (e.g., WHO, CDC), and expand safety guardrails for clinical use
