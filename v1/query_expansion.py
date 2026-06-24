"""
Expands a natural-language question into a richer query string before it's
embedded for Template 1 / Template 2 vector search.

Strategy: dictionary first (free, deterministic, covers known domain
vocabulary), LLM fallback only when no dictionary term matches (covers
novel phrasing the dictionary didn't anticipate).

The expanded string is concatenated with the original question before
embedding -- this doesn't replace the question, it adds context the
embedding model can use.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Domain synonym dictionary, built from the actual cfKeyw/cfResInt vocabulary
# in the graph (see v3_neo4j_agri_research_FINAL.cypher). Keys are lowercase
# trigger terms; values are related terms actually present in the graph's
# keyword fields, so expansion stays grounded in real data rather than
# introducing terms that don't exist anywhere in the graph.
DOMAIN_SYNONYMS = {
    "durian": ["ทุเรียน", "Phytophthora", "ripeness", "fruit physiology"],
    "longan": ["ลำไย", "postharvest", "grading", "downgrade"],
    "rice": ["ข้าว", "glutinous rice", "paddy", "jasmine rice", "germplasm"],
    "biocontrol": ["Trichoderma", "Bacillus", "Phytophthora", "plant disease", "IPM"],
    "biocontrol agents": ["Trichoderma", "Bacillus", "Phytophthora", "plant disease"],
    "packaging": ["MAP", "shelf life", "biodegradable", "Modified Atmosphere Packaging"],
    "packaging technology": ["MAP", "shelf life", "biodegradable"],
    "irrigation": ["water management", "smart irrigation", "paddy"],
    "iot": ["smart farming", "precision agriculture", "sensor"],
    "machine vision": ["AI", "computer vision", "crop detection", "grading"],
    "organic farming": ["soil", "fertilizer", "PGPR", "biofertilizer"],
    "value chain": ["agricultural economics", "market", "premium market"],
}


def expand_with_dictionary(question: str) -> list:
    """
    Check the question (lowercased) for any dictionary trigger terms.
    Returns the related terms for every match found -- a question can match
    multiple triggers (e.g. "durian packaging" hits both entries).
    """
    question_lower = question.lower()
    expansions = []
    for trigger, related_terms in DOMAIN_SYNONYMS.items():
        if trigger in question_lower:
            expansions.extend(related_terms)
    return list(dict.fromkeys(expansions))  # dedupe, preserve order


def expand_with_llm(question: str) -> list:
    """
    Fallback when no dictionary term matched. Asks gpt-4o-mini for a short
    list of related agricultural-research terms, grounded in the same kind
    of vocabulary the dictionary uses (crop names, techniques, methods) --
    not asked to invent arbitrary synonyms.
    """
    prompt = f"""You are helping expand a search query for a Thai agricultural
research knowledge graph. Given the question below, list 3-5 closely related
technical or domain terms (crop names, research techniques, methods) that
might appear in a researcher's or project's keyword list, even if not
explicitly mentioned in the question.

Question: {question}

Return ONLY a comma-separated list of terms, nothing else."""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()
    return [term.strip() for term in raw.split(",") if term.strip()]


def expand_query(question: str) -> str:
    """
    Full expansion pipeline: dictionary first, LLM fallback only if the
    dictionary found nothing. Returns the original question concatenated
    with expansion terms, ready to pass into embed_query().
    """
    expansions = expand_with_dictionary(question)
    source = "dictionary"

    if not expansions:
        expansions = expand_with_llm(question)
        source = "llm"

    if not expansions:
        return question  # nothing to add, embed the original as-is

    expanded_text = f"{question} (related terms: {', '.join(expansions)})"
    print(f"  [expansion via {source}] {expansions}")
    return expanded_text


if __name__ == "__main__":
    test_questions = [
        "Find researchers who work on biocontrol agents.",
        "Find projects related to packaging technology.",
        "Who studies aquaponics systems?",  # not in dictionary -> LLM fallback
    ]
    for q in test_questions:
        print(f"\nQ: {q}")
        print(f"Expanded: {expand_query(q)}")