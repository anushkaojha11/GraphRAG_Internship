from neo4j import GraphDatabase
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


TEXT2CYPHER_SYSTEM_PROMPT = """
You convert natural language questions into Cypher queries for a Neo4j 
knowledge graph about Thai agricultural research (TNRR).

## Schema

Node: cfPers (researcher)
Properties: cfPersId, cfPersNameEN, cfKeyw, cfResInt, textDescription, embedding

Node: cfProj (research project)
Properties: cfProjId, cfTitleEN, cfKeyw, cfAbstr, cfAcro, textDescription, embedding

Node: cfOrgUnit (organization)
Properties: cfOrgUnitId, cfNameEN

Node: cfResPubl (publication)
Properties: cfResPublId, cfTitle

Node: cfResPat (patent)
Properties: cfResPatId, cfTitle

Relationships:
(cfProj)-[:cfProj_Pers]->(cfPers)        project has this person (leader/researcher)
(cfPers)-[:cfPers_OrgUnit]->(cfOrgUnit)  person belongs to org
(cfPers)-[:cfPers_ResPubl]->(cfResPubl)  person authored publication
(cfPers)-[:cfPers_ResPat]->(cfResPat)    person holds patent

## Examples

Question: Find all researchers
Cypher: MATCH (p:cfPers) RETURN p.cfPersNameEN as name, p.cfKeyw as keywords

Question: Who works on longan grading?
Cypher: MATCH (p:cfPers) WHERE p.cfKeyw CONTAINS 'longan' AND p.cfKeyw CONTAINS 'grading' RETURN p.cfPersNameEN as name, p.cfKeyw as keywords

Question: What projects has Dr. Somchai worked on?
Cypher: MATCH (proj:cfProj)-[:cfProj_Pers]->(p:cfPers) WHERE p.cfPersNameEN CONTAINS 'Somchai' RETURN proj.cfTitleEN as title

Question: Show me a researcher's full profile including projects, org, publications and patents
Cypher: MATCH (p:cfPers) WHERE p.cfPersNameEN CONTAINS 'Pannipa Sophon' OPTIONAL MATCH (proj:cfProj)-[:cfProj_Pers]->(p) OPTIONAL MATCH (p)-[:cfPers_OrgUnit]->(org:cfOrgUnit) OPTIONAL MATCH (p)-[:cfPers_ResPubl]->(pub:cfResPubl) OPTIONAL MATCH (p)-[:cfPers_ResPat]->(pat:cfResPat) RETURN p, collect(distinct proj.cfTitleEN) as projects, org.cfNameEN as organization, collect(distinct pub.cfTitle) as publications, collect(distinct pat.cfTitle) as patents

## Rules

- Only use the node labels, relationships and properties listed above
- Never invent properties or relationships that don't exist
- Always return readable field names using "as"
- When matching a person's name, use WHERE p.cfPersNameEN CONTAINS '<partial name>' instead of exact equality, since users may not provide the full name
- Return ONLY the Cypher query, no explanation, no markdown code blocks, no backticks
"""


def generate_cypher(question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": TEXT2CYPHER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\nCypher:"}
        ]
    )
    cypher = response.choices[0].message.content.strip()
    # Safety: strip markdown formatting if GPT adds it anyway
    cypher = cypher.replace("```cypher", "").replace("```", "").strip()
    return cypher


def run_cypher(cypher_query):
    with driver.session() as session:
        result = session.run(cypher_query)
        return [record.data() for record in result]


def ask(question):
    print(f"\nQuestion: {question}")
    cypher = generate_cypher(question)
    print(f"Generated Cypher:\n{cypher}\n")

    results = run_cypher(cypher)
    print(f"Results ({len(results)} rows):")
    for row in results:
        print(f"  {row}")

    return results