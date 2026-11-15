"""Catalog of 54+ Real-Industry & Specialized Domain Datasets on Hugging Face.

Every entry is verified to load cleanly with `datasets` (no legacy dataset scripts
and no gated access required). Users can prepare any domain via:
    python 06_prepare_hf_dataset.py --domain <id>
or with 1 click in Tab 1 of the Visual UI.
"""

DOMAIN_CATALOG: dict[str, dict] = {
    # --- Finance, Accounting & Economics ---
    "finance_alpaca": {"repo": "gbharti/finance-alpaca", "config": None, "category": "Finance", "label": "Finance Advisory & Investing Q&A (finance-alpaca)", "kind": "instruct"},
    "finance_fiqasa": {"repo": "ChanceFocus/flare-fiqasa", "config": None, "category": "Finance", "label": "Financial Sentiment & QA (FinQA-SA)", "kind": "instruct"},
    "finance_news_sent": {"repo": "zeroshot/twitter-financial-news-sentiment", "config": None, "category": "Finance", "label": "Market & Financial News Sentiment", "kind": "classify"},
    "finance_bench": {"repo": "PatronusAI/financebench", "config": None, "category": "Finance", "label": "FinanceBench — Corporate SEC Filings", "kind": "qa_context"},
    "finance_headlines": {"repo": "AdaptLLM/finance-tasks", "config": "Headline", "category": "Finance", "label": "Financial Headlines Analysis", "kind": "instruct"},
    "econ_econometrics": {"repo": "cais/mmlu", "config": "econometrics", "category": "Finance", "label": "Econometrics & Quantitative Economics", "kind": "mcqa"},
    "biz_accounting": {"repo": "cais/mmlu", "config": "professional_accounting", "category": "Finance", "label": "Professional Accounting & Audit", "kind": "mcqa"},

    # --- Healthcare, Medicine & Life Sciences ---
    "med_mcqa": {"repo": "openlifescienceai/medmcqa", "config": None, "category": "Healthcare", "label": "Medical Entrance Exam Q&A (MedMCQA)", "kind": "mcqa"},
    "med_quad": {"repo": "lavita/MedQuAD", "config": None, "category": "Healthcare", "label": "NIH Medical Q&A (MedQuAD)", "kind": "instruct"},
    "med_flashcards": {"repo": "medalpaca/medical_meadow_medical_flashcards", "config": None, "category": "Healthcare", "label": "Clinical Medical Flashcards", "kind": "instruct"},
    "med_text": {"repo": "BI55/MedText", "config": None, "category": "Healthcare", "label": "Clinical MedText Patient Scenarios", "kind": "instruct"},
    "med_clinical": {"repo": "cais/mmlu", "config": "clinical_knowledge", "category": "Healthcare", "label": "Clinical Medicine Knowledge", "kind": "mcqa"},
    "med_professional": {"repo": "cais/mmlu", "config": "professional_medicine", "category": "Healthcare", "label": "Professional Medicine Licensing", "kind": "mcqa"},
    "med_virology": {"repo": "cais/mmlu", "config": "virology", "category": "Healthcare", "label": "Virology & Infectious Disease", "kind": "mcqa"},
    "med_nutrition": {"repo": "cais/mmlu", "config": "nutrition", "category": "Healthcare", "label": "Clinical Nutrition & Dietetics", "kind": "mcqa"},

    # --- Legal, Compliance & Policy ---
    "legal_scotus": {"repo": "coastalcph/lex_glue", "config": "scotus", "category": "Legal", "label": "US Supreme Court Opinions (SCOTUS)", "kind": "classify"},
    "legal_ecthr": {"repo": "coastalcph/lex_glue", "config": "ecthr_a", "category": "Legal", "label": "European Human Rights Cases (ECtHR)", "kind": "classify"},
    "legal_eurlex": {"repo": "coastalcph/lex_glue", "config": "eurlex", "category": "Legal", "label": "EU Legislation & Regulation (EUR-Lex)", "kind": "classify"},
    "legal_unfair_tos": {"repo": "coastalcph/lex_glue", "config": "unfair_tos", "category": "Legal", "label": "Unfair Terms of Service Detection", "kind": "classify"},
    "legal_billsum": {"repo": "FiscalNote/billsum", "config": None, "category": "Legal", "label": "US Congressional Bills Summarization", "kind": "instruct"},
    "legal_stackexchange": {"repo": "jonathanli/law-stack-exchange", "config": None, "category": "Legal", "label": "Law StackExchange Practical Legal Q&A", "kind": "instruct"},
    "legal_prof_law": {"repo": "cais/mmlu", "config": "professional_law", "category": "Legal", "label": "Professional Bar Exam Law", "kind": "mcqa"},
    "legal_intl_law": {"repo": "cais/mmlu", "config": "international_law", "category": "Legal", "label": "International Law & Treaties", "kind": "mcqa"},

    # --- Software Engineering, Data & Cybersecurity ---
    "code_mbpp": {"repo": "google-research-datasets/mbpp", "config": None, "category": "Tech & Code", "label": "Python Programming Tasks (MBPP)", "kind": "instruct"},
    "code_humaneval": {"repo": "openai/openai_humaneval", "config": None, "category": "Tech & Code", "label": "Python Code Synthesis (HumanEval)", "kind": "instruct"},
    "code_sql": {"repo": "b-mc2/sql-create-context", "config": None, "category": "Tech & Code", "label": "Enterprise Text-to-SQL Generation", "kind": "qa_context"},
    "code_magicoder": {"repo": "ise-uiuc/Magicoder-Evol-Instruct-110K", "config": None, "category": "Tech & Code", "label": "Software Engineering Instruction Tuning", "kind": "instruct"},
    "cyber_secqa": {"repo": "zefang-liu/secqa", "config": "secqa_v1", "category": "Tech & Code", "label": "Cybersecurity Exam Q&A (SecQA)", "kind": "mcqa"},
    "cyber_compsec": {"repo": "cais/mmlu", "config": "computer_security", "category": "Tech & Code", "label": "Computer Security & Cryptography", "kind": "mcqa"},
    "tech_ml": {"repo": "cais/mmlu", "config": "machine_learning", "category": "Tech & Code", "label": "Machine Learning & Statistics", "kind": "mcqa"},

    # --- Science, Math & Engineering ---
    "math_gsm8k": {"repo": "openai/gsm8k", "config": "main", "category": "STEM", "label": "Step-by-Step Math Reasoning (GSM8K)", "kind": "instruct"},
    "stem_openbookqa": {"repo": "allenai/openbookqa", "config": "main", "category": "STEM", "label": "Science Reasoning (OpenBookQA)", "kind": "mcqa"},
    "stem_arc": {"repo": "allenai/ai2_arc", "config": "ARC-Challenge", "category": "STEM", "label": "Hard Science Exam (ARC-Challenge)", "kind": "mcqa"},
    "stem_physics": {"repo": "cais/mmlu", "config": "college_physics", "category": "STEM", "label": "College Physics", "kind": "mcqa"},
    "stem_chemistry": {"repo": "cais/mmlu", "config": "college_chemistry", "category": "STEM", "label": "College Chemistry", "kind": "mcqa"},
    "stem_biology": {"repo": "cais/mmlu", "config": "college_biology", "category": "STEM", "label": "College Biology", "kind": "mcqa"},
    "stem_astronomy": {"repo": "cais/mmlu", "config": "astronomy", "category": "STEM", "label": "Astronomy & Astrophysics", "kind": "mcqa"},
    "stem_ee": {"repo": "cais/mmlu", "config": "electrical_engineering", "category": "STEM", "label": "Electrical Engineering", "kind": "mcqa"},

    # --- Customer Support, Banking, Retail & Enterprise Ops ---
    "ops_banking77": {"repo": "PolyAI/banking77", "config": None, "category": "Enterprise Ops", "label": "Retail Banking Intent Routing (77 intents)", "kind": "classify"},
    "ops_clinc": {"repo": "clinc/clinc_oos", "config": "small", "category": "Enterprise Ops", "label": "Customer Service & Out-of-Scope Intents", "kind": "classify"},
    "ops_bitext_support": {"repo": "bitext/Bitext-customer-support-llm-chatbot-training-dataset", "config": None, "category": "Enterprise Ops", "label": "Customer Support Chatbot Dialogues", "kind": "instruct"},
    "ops_amazon_reviews": {"repo": "SetFit/amazon_reviews_multi_en", "config": None, "category": "Enterprise Ops", "label": "E-Commerce Product Reviews Sentiment", "kind": "classify"},
    "ops_ag_news": {"repo": "fancyzhx/ag_news", "config": None, "category": "Enterprise Ops", "label": "Global News Topic Routing", "kind": "classify"},
    "ops_dbpedia": {"repo": "fancyzhx/dbpedia_14", "config": None, "category": "Enterprise Ops", "label": "Enterprise Entity & Knowledge Graph", "kind": "classify"},

    # --- Factuality, Hallucination & Reasoning Benchmarks ---
    "eval_squad_v2": {"repo": "rajpurkar/squad_v2", "config": None, "category": "Hallucination & Eval", "label": "SQuAD v2 (Answerable + Unanswerable QA)", "kind": "qa_context"},
    "eval_truthfulqa": {"repo": "truthfulqa/truthful_qa", "config": "generation", "category": "Hallucination & Eval", "label": "TruthfulQA (Myths & Hallucination Traps)", "kind": "instruct"},
    "eval_boolq": {"repo": "google/boolq", "config": None, "category": "Hallucination & Eval", "label": "BoolQ Reading Comprehension (True/False)", "kind": "qa_context"},
    "eval_hotpotqa": {"repo": "hotpotqa/hotpot_qa", "config": "distractor", "category": "Hallucination & Eval", "label": "HotpotQA Multi-Hop Reasoning", "kind": "qa_context"},
    "eval_commonsense": {"repo": "tau/commonsense_qa", "config": None, "category": "Hallucination & Eval", "label": "Commonsense Reasoning QA", "kind": "mcqa"},

    # --- Climate/ESG, Management, Marketing & Ethics ---
    "esg_climate": {"repo": "climatebert/climate_sentiment", "config": None, "category": "Business & ESG", "label": "Climate & ESG Disclosure Sentiment", "kind": "classify"},
    "biz_management": {"repo": "cais/mmlu", "config": "management", "category": "Business & ESG", "label": "Business Management & Strategy", "kind": "mcqa"},
    "biz_marketing": {"repo": "cais/mmlu", "config": "marketing", "category": "Business & ESG", "label": "Marketing & Consumer Behavior", "kind": "mcqa"},
    "biz_ethics": {"repo": "cais/mmlu", "config": "business_ethics", "category": "Business & ESG", "label": "Corporate & Business Ethics", "kind": "mcqa"},
    "biz_pr": {"repo": "cais/mmlu", "config": "public_relations", "category": "Business & ESG", "label": "Public Relations & Communications", "kind": "mcqa"},
}
