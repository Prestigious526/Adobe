from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

def refine(section):
    """
    Summarize section text using T5-small.
    """
    text = section["text"]
    # Pre-trim long text
    text = text[:2000]

    # Summarize
    inputs = tokenizer("summarize: " + text, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=30)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    section["subsection"] = {"refined_text": summary[:800]}
    return section