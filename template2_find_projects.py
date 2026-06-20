"""
template2_find_projects.py

Template 2: find_projects_by_topic_with_experts

Parameterized retrieval template for "find projects about X, and who works
on them" style questions. Mirrors Template 1's structure:
  1. Embed the natural-language query (same model as vector_indexing.py)
  2. Vector search against the `project_embeddings` index for top-N candidates
  3. Deterministic Cypher enrichment (researchers + their org affiliations)
     -- no LLM-generated Cypher in this path at all.

Scope: answers "which project(s)" questions only. Fund/facility/sub-project
lookups for a *specific* project are out of scope here -- that needs either
an extended enrichment query or query decomposition.
"""

from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv
from query_expansion import expand_query
import os

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

EMBEDDING_MODEL = "text-embedding-3-small"  # must match vector_indexing.py
VECTOR_INDEX_NAME = "project_embeddings"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_query(text: str) -> list:
    """Embed the natural-language question with the same model used to
    build project_embeddings, so the vector space matches."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def vector_search_projects(query_embedding: list, top_n: int = 5) -> list:
    """
    Step 1: pure vector similarity search against project_embeddings.
    Returns candidate cfProjId values ranked by cosine similarity score.
    """
    cypher = """
    CALL db.index.vector.queryNodes($index_name, $top_n, $embedding)
    YIELD node, score
    RETURN node.cfProjId AS projectId, score
    ORDER BY score DESC
    """
    with driver.session() as session:
        result = session.run(
            cypher,
            index_name=VECTOR_INDEX_NAME,
            top_n=top_n,
            embedding=query_embedding,
        )
        return [{"projectId": r["projectId"], "score": r["score"]} for r in result]


def enrich_projects(project_ids: list) -> list:
    """
    Step 2: deterministic, fixed-shape Cypher enrichment.
    For each candidate project, attach its researchers and their org
    affiliations. This part never changes regardless of the question.
    """
    cypher = """
    MATCH (proj:cfProj) WHERE proj.cfProjId IN $project_ids
    OPTIONAL MATCH (proj)-[:cfProj_Pers]->(p:cfPers)
    OPTIONAL MATCH (p)-[:cfPers_OrgUnit]->(org:cfOrgUnit)
    RETURN
        proj.cfProjId AS projectId,
        proj.cfTitleEN AS title,
        proj.textDescription AS summary,
        collect(DISTINCT p.cfPersNameEN) AS researchers,
        collect(DISTINCT org.cfNameEN) AS organizations
    """
    with driver.session() as session:
        result = session.run(cypher, project_ids=project_ids)
        return [dict(r) for r in result]


def find_projects_by_topic_with_experts(question: str, top_n: int = 3) -> list:
    """
    Full Template 2 pipeline: question -> embedding -> vector search ->
    enrichment -> ranked, enriched project list.
    """
    expanded_question = expand_query(question)
    query_embedding = embed_query(expanded_question)
    candidates = vector_search_projects(query_embedding, top_n=top_n)

    if not candidates:
        return []

    project_ids = [c["projectId"] for c in candidates]
    enriched = enrich_projects(project_ids)

    # Preserve the vector search's similarity ranking -- enrichment query
    # above does not guarantee order, so re-sort by the original rank.
    score_by_id = {c["projectId"]: c["score"] for c in candidates}
    enriched_by_id = {e["projectId"]: e for e in enriched}

    ranked_results = []
    for pid in project_ids:
        if pid in enriched_by_id:
            row = enriched_by_id[pid]
            row["score"] = score_by_id[pid]
            ranked_results.append(row)

    return ranked_results


if __name__ == "__main__":
    # Quick smoke test
    test_questions = [
        "Find projects related to packaging technology.",
        "Who leads the NIR durian ripeness detection project?",
    ]
    for q in test_questions:
        print(f"\nQ: {q}")
        for r in find_projects_by_topic_with_experts(q, top_n=3):
            print(f"  {r['title']} (score={r['score']:.4f}) -- researchers: {r['researchers']} -- orgs: {r['organizations']}")