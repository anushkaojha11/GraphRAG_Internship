"""
Template 1: find_experts_by_topic

Parameterized retrieval template for "who works on / who is expert in X"
style questions. Replaces free-form Text2Cypher entity-name guessing with:
  1. Embed the natural-language query (same model as vector_indexing.py)
  2. Vector search against the `person_embeddings` index for top-N candidates
  3. Deterministic Cypher enrichment (org affiliation + project titles) --
     no LLM-generated Cypher in this path at all.

This is intentionally narrow: it answers "which person(s)" questions only.
Fund/facility/sub-project/collaborator-chain questions are out of scope
here -- that's Template 2 / query decomposition.
"""

from neo4j import GraphDatabase
from openai import OpenAI
import os
from query_expansion import expand_query


from dotenv import load_dotenv
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

EMBEDDING_MODEL = "text-embedding-3-small"  # must match vector_indexing.py
VECTOR_INDEX_NAME = "person_embeddings"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
openai_client = OpenAI()


def embed_query(text: str) -> list:
    """Embed the natural-language question with the same model used to
    build person_embeddings, so the vector space matches."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def vector_search_persons(query_embedding: list, top_n: int = 5) -> list:
    """
    Step 1: pure vector similarity search against person_embeddings.
    Returns candidate cfPersId values ranked by cosine similarity score.
    """
    cypher = """
    CALL db.index.vector.queryNodes($index_name, $top_n, $embedding)
    YIELD node, score
    RETURN node.cfPersId AS personId, score
    ORDER BY score DESC
    """
    with driver.session() as session:
        result = session.run(
            cypher,
            index_name=VECTOR_INDEX_NAME,
            top_n=top_n,
            embedding=query_embedding,
        )
        return [{"personId": r["personId"], "score": r["score"]} for r in result]


def enrich_persons(person_ids: list) -> list:
    """
    Step 2: deterministic, fixed-shape Cypher enrichment.
    For each candidate person, attach org affiliation and project titles.
    This part never changes regardless of the question -- that's the point.
    """
    cypher = """
    MATCH (p:cfPers) WHERE p.cfPersId IN $person_ids
    OPTIONAL MATCH (p)-[:cfPers_OrgUnit]->(org:cfOrgUnit)
    OPTIONAL MATCH (proj:cfProj)-[:cfProj_Pers]->(p)
    RETURN
        p.cfPersId AS personId,
        p.cfPersNameEN AS name,
        p.cfKeyw AS keywords,
        org.cfNameEN AS organization,
        collect(DISTINCT proj.cfTitleEN) AS projects
    """
    with driver.session() as session:
        result = session.run(cypher, person_ids=person_ids)
        return [dict(r) for r in result]


def find_experts_by_topic(question: str, top_n: int = 3) -> list:
    """
    Full Template 1 pipeline: question -> embedding -> vector search ->
    enrichment -> ranked, enriched expert list.
    """
    expanded_question = expand_query(question)
    query_embedding = embed_query(expanded_question)
    candidates = vector_search_persons(query_embedding, top_n=top_n)

    if not candidates:
        return []

    person_ids = [c["personId"] for c in candidates]
    enriched = enrich_persons(person_ids)

    # Preserve the vector search's similarity ranking -- enrichment query
    # above does not guarantee order, so re-sort by the original rank.
    score_by_id = {c["personId"]: c["score"] for c in candidates}
    enriched_by_id = {e["personId"]: e for e in enriched}

    ranked_results = []
    for pid in person_ids:
        if pid in enriched_by_id:
            row = enriched_by_id[pid]
            row["score"] = score_by_id[pid]
            ranked_results.append(row)

    return ranked_results


if __name__ == "__main__":
    # Quick smoke test
    test_questions = [
        "Find researchers who work on durian.",
        "Who invented the high-precision longan grading device patent?",
    ]
    for q in test_questions:
        print(f"\nQ: {q}")
        for r in find_experts_by_topic(q, top_n=3):
            print(f"  {r['name']} (score={r['score']:.4f}) -- {r['organization']} -- {r['projects']}")