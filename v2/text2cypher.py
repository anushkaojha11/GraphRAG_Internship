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

Node: cfPers (researcher) — 25 researchers
Properties: cfPersId, cfPersNameEN, cfKeyw, cfResInt, textDescription, embedding

Node: cfProj (research project) — 50 projects
Properties: cfProjId, cfTitleEN, cfKeyw, cfAbstr, cfAcro, textDescription, embedding

Node: cfOrgUnit (organization) — 10 organizations
Properties: cfOrgUnitId, cfNameEN

Node: cfResPubl (publication) — 25 publications
Properties: cfResPublId, cfTitleEN

Node: cfResPat (patent) — 3 patents
Properties: cfResPatId, cfTitleEN

Node: cfResProd (research product) — 13 products
Properties: cfResProdId, cfNameEN, cfDescr

Node: cfFund (funding source) — 10 funds
Properties: cfFundId, cfNameEN, cfAmount

Node: cfFacil (facility) — 5 facilities
Properties: cfFacilId, cfNameEN

Node: cfEquip (equipment) — 5 equipment
Properties: cfEquipId, cfNameEN

Node: cfIndic (indicator) — 6 indicators
Properties: cfIndicId, cfNameEN

Node: cfMeas (measurement) — 50+ measurements
Properties: cfMeasId, cfCountInt, cfValFloatP

## Relationships

(cfProj)-[:cfProj_Pers]->(cfPers)              project has this person (leader/researcher)
(cfPers)-[:cfPers_OrgUnit]->(cfOrgUnit)        person affiliated with organization
(cfPers)-[:cfPers_ResPubl]->(cfResPubl)        person authored publication
(cfPers)-[:cfPers_ResPat]->(cfResPat)          person holds patent
(cfPers)-[:cfPers_ResProd]->(cfResProd)        person created product
(cfPers)-[:cfPers_Pers]->(cfPers)              person collaborates with person
(cfProj)-[:cfProj_OrgUnit]->(cfOrgUnit)        project coordinated/partnered by org
(cfProj)-[:cfProj_Fund]->(cfFund)              project funded by fund
(cfProj)-[:cfProj_ResPubl]->(cfResPubl)        project originated publication
(cfProj)-[:cfProj_ResPat]->(cfResPat)          project originated patent
(cfProj)-[:cfProj_ResProd]->(cfResProd)        project originated product
(cfProj)-[:cfProj_Facil]->(cfFacil)            project uses facility
(cfProj)-[:cfProj_Equip]->(cfEquip)            project uses equipment
(cfProj)-[:cfProj_Proj]->(cfProj)              project is sub-project/continuation of project
(cfProj)-[:cfProj_Meas]->(cfMeas)              project has measurement (e.g. TRL)
(cfIndic)-[:cfIndic_Meas]->(cfMeas)            indicator links to measurement
(cfOrgUnit)-[:cfOrgUnit_OrgUnit]->(cfOrgUnit)  org is part-of/partner-of org
(cfOrgUnit)-[:cfOrgUnit_Fund]->(cfFund)        org is funder/recipient of fund

## Important field notes
- Publications and patents use cfTitleEN (English), NOT cfTitle (Thai)
- Always use cfTitleEN for publication and patent titles
- Use cfNameEN for organizations, facilities, equipment, products, funds

## Examples

Question: Find all researchers
Cypher: MATCH (p:cfPers) RETURN p.cfPersNameEN as name, p.cfKeyw as keywords

Question: Who works on longan grading?
Cypher: MATCH (p:cfPers) WHERE p.cfKeyw CONTAINS 'longan' AND p.cfKeyw CONTAINS 'grading' RETURN p.cfPersNameEN as name, p.cfKeyw as keywords

Question: What projects has Dr. Somchai worked on?
Cypher: MATCH (proj:cfProj)-[:cfProj_Pers]->(p:cfPers) WHERE p.cfPersNameEN CONTAINS 'Somchai' RETURN proj.cfTitleEN as title

Question: Which researchers hold patents?
Cypher: MATCH (p:cfPers)-[:cfPers_ResPat]->(pat:cfResPat) RETURN DISTINCT p.cfPersNameEN as researcher, pat.cfTitleEN as patent

Question: What fund supports a project?
Cypher: MATCH (proj:cfProj)-[:cfProj_Fund]->(f:cfFund) WHERE proj.cfTitleEN CONTAINS 'longan' RETURN proj.cfTitleEN as project, f.cfNameEN as fund

Question: Find researchers working on longan or durian projects with TRL >= 7
Cypher: MATCH (p:cfPers)<-[:cfProj_Pers]-(proj:cfProj)-[:cfProj_Meas]->(meas:cfMeas)<-[:cfIndic_Meas]-(ind:cfIndic) WHERE ind.cfIndicId = 'INDIC003' AND (toLower(proj.cfKeyw) CONTAINS 'longan' OR toLower(proj.cfKeyw) CONTAINS 'durian') AND toInteger(coalesce(meas.cfCountInt, 0)) >= 7 RETURN DISTINCT p.cfPersNameEN as researcher, proj.cfTitleEN as project, meas.cfCountInt as trl

Question: Show me a researcher's full profile including projects, org, publications and patents
Cypher: MATCH (p:cfPers) WHERE p.cfPersNameEN CONTAINS 'Pannipa Sophon' OPTIONAL MATCH (proj:cfProj)-[:cfProj_Pers]->(p) OPTIONAL MATCH (p)-[:cfPers_OrgUnit]->(org:cfOrgUnit) OPTIONAL MATCH (p)-[:cfPers_ResPubl]->(pub:cfResPubl) OPTIONAL MATCH (p)-[:cfPers_ResPat]->(pat:cfResPat) RETURN p.cfPersNameEN as name, collect(distinct proj.cfTitleEN) as projects, org.cfNameEN as organization, collect(distinct pub.cfTitleEN) as publications, collect(distinct pat.cfTitleEN) as patents

## Rules

- Only use the node labels, relationships and properties listed above
- Never invent properties or relationships that don't exist
- Always use cfTitleEN (not cfTitle) for publications and patents
- Always return readable field names using "as"
- When matching a person's name, use WHERE p.cfPersNameEN CONTAINS '<partial name>'
- Use toLower() when doing keyword matching for robustness
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