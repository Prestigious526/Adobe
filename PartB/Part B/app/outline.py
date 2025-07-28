from sklearn.cluster import KMeans
import numpy as np
import re

def build(blocks):
    """
    Cluster font sizes to classify headings into H1, H2, H3.
    Extract cleaner section titles.
    """
    if not blocks:
        return []

    font_sizes = np.array([b['font_size'] for b in blocks]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=min(3, len(blocks)), random_state=0).fit(font_sizes)
    labels = kmeans.labels_
    centroids = sorted(
        [(i, np.mean(font_sizes[labels == i])) for i in range(len(set(labels)))],
        key=lambda x: -x[1]
    )
    label_map = {label: f"H{i+1}" for i, (label, _) in enumerate(centroids)}

    sections = []
    for idx, block in enumerate(blocks):
        # Clean up the title - take first sentence or first 100 chars
        title = block["text"]
        # Remove extra whitespace and newlines
        title = re.sub(r'\s+', ' ', title.strip())
        # Take first sentence or first 100 characters
        if len(title) > 100:
            # Try to find a sentence break
            sentence_end = title.find('.')
            if sentence_end > 0 and sentence_end < 100:
                title = title[:sentence_end + 1]
            else:
                title = title[:100].rstrip() + "..."
        
        sections.append({
            "page": block["page"],
            "level": label_map[labels[idx]],
            "title": title,
            "start_idx": idx,
            "end_idx": idx
        })
    return sections