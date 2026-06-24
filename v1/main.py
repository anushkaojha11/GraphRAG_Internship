from neo4j import GraphDatabase
from dotenv import load_dotenv
from openai import OpenAI
import os

# Load environment variables
load_dotenv()

# Neo4j connection
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# OpenAI connection
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Test Neo4j
def test_neo4j():
    with driver.session() as session:
        result = session.run(
            "MATCH (p:cfPers) RETURN p.cfPersNameEN as name LIMIT 3"
        )
        print("Neo4j test:")
        for record in result:
            print(f"  - {record['name']}")

# Test OpenAI
def test_openai():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Say 'OpenAI connected successfully' and nothing else."}
        ]
    )
    print("\nOpenAI test:")
    print(f"  {response.choices[0].message.content}")

# Run both tests
print("Testing connections...\n")
test_neo4j()
test_openai()
print("\nAll connections successful!")
