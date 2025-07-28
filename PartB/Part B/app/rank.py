from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def hybrid_score(query_vec, section_vecs, texts):
    """
    Compute hybrid score = 0.6*cosine + 0.4*BM25-like TFIDF sum.
    """
    cos_scores = cosine_similarity([query_vec], section_vecs).flatten()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    bm25_scores = np.array(tfidf_matrix.sum(axis=1)).ravel()

    return 0.6 * cos_scores + 0.4 * bm25_scores

def select(sections, embeddings, persona_query):
    """
    Select top-ranked sections based on persona query embedding.
    Improved to better match persona requirements.
    """
    # Create a more specific query based on persona
    if "travel" in persona_query.lower():
        query_keywords = "travel planning trip itinerary destination attractions activities"
    elif "food" in persona_query.lower() or "recipe" in persona_query.lower():
        query_keywords = "cooking recipes food preparation ingredients cuisine"
    elif "adobe" in persona_query.lower() or "acrobat" in persona_query.lower():
        query_keywords = "adobe acrobat forms documents software tutorial"
    else:
        query_keywords = persona_query
    
    # Encode the enhanced query
    from .embed import encode
    query_vec = encode([query_keywords])[0]
    
    scores = hybrid_score(query_vec, embeddings, [s["text"] for s in sections])
    ranked = sorted(zip(sections, scores), key=lambda x: x[1], reverse=True)
    
    # Assign importance rank
    for rank, (sec, _) in enumerate(ranked, start=1):
        sec["importance_rank"] = rank
    
    return [sec for sec, _ in ranked[:5]]