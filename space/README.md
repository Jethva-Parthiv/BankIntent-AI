---
title: BankIntent AI
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: DistilBERT banking intent classification for 77 intents
---

# 🏦 BankIntent AI

> **Tagline:** DistilBERT-powered banking intent classification for 77 customer-support intents.

An intelligent customer inquiry triage and intent classification system fine-tuned on the **Banking77** dataset.

- **Model on HF Hub:** [Parthiv-Jethva/bankintent-ai](https://huggingface.co/Parthiv-Jethva/bankintent-ai)
- **Architecture:** DistilBERT (`distilbert-base-uncased`, 66M parameters)
- **Accuracy:** 85.66% on Banking77 test set (3,076 samples)
- **Weighted F1:** 84.94%
- **Automated Routing:** Maps queries into 6 banking support departments (Fraud/Security, Card Logistics, Payments, Forex, KYC, Disputes).
