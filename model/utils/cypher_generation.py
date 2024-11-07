from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import re

from .init_settings import init_settings


person_schema = """
**Person**
- 'income': INTEGER Min: -1616, Max: 927313 description: Individual annual income of the person in USD.
- 'age': INTEGER Min: 17, Max: 94 description: Age of the person.
"""

# - 'vehicles': STRING Available options: ['1', '0', '2', '3_plus'] description: Number of vehicles available to the person.
# - 'family_structure': STRING Available options: ['family_single', 'married_couple', 'nonfamily_single', 'living_alone'] description: Family structure of the person.
# - 'household_size': STRING Available options: ['4_person', '5_person', '3_person', '2_person', '1_person', '7_plus_person', '1_person_group_quarters', '6_person'] description: Number of people in the household this person belongs to.


cypher_template = """Based on the Neo4j graph schema below, complete the Cypher query to find the {number} nodes similar to the profile:
{schema}
Profile: {profile}

Groups:
- 'income' : 
  - 'Debt' : <0
  - 'Low' : >=0 AND <30000
  - 'Moderate' : >=30000 AND <60000
  - 'High' : >=60000 AND <90000
  - 'Very High' : >=90000 AND <120000
  - 'Ultra High' : >=120000

- 'age' : 
  - 'Teen' : <18
  - 'Young Adult' : >=18 AND <30
  - 'Adult' : >=30 AND <40
  - 'Middle Age' : >=40 AND <50
  - 'Senior' : >=50 AND <60
  - 'Elderly' : >=60

**IMPORTANT:**
- **DO NOT** include other information except age and income.
- **DO NOT** include queries for filters not mentioned in the profile. For example, if the profile is "a young man" and there is no information about income, **DO NOT** generate a query for income.
- Map the relevant age or income information into these groups. For example, "a young man" should translate to `p.age ≥18 AND p.age<30` for age.
- Keep other parts of the query as is, and answer with the whole query.
- Wrap the final answer in triple backticks (```cypher ```) to indicate a code block.

Answer: 
age group: e.g. 'High'
income group: e.g. 'Moderate'
```cypher
MATCH (p:Person)
WHERE [[your response query here]]
WITH p ORDER BY rand() LIMIT {number}
MATCH (p)-[r1:WANT_TO]-(d:Desire {{desire:'{desire}'}})-[r2:GO_TO]-(i:Intention)
RETURN COLLECT(DISTINCT p) AS person, 
    COLLECT(DISTINCT d) AS desire, 
    COLLECT(DISTINCT i) AS intention,
    COLLECT(DISTINCT r1) AS want_to,
    COLLECT(DISTINCT r2) AS go_to
```
"""


class CypherOutputParser(StrOutputParser):
    def parse(self, text: str) -> str:
        code_pattern = re.compile(r'```(?:cypher)?(.*?)```', re.DOTALL)
        match = code_pattern.search(text)
        if match:
            return match.group(1).strip()
        return "No cypher found"


def get_llm_cypher(profile, top_k, desire):
    cypher_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given an input query, convert it to a Cypher query. No pre-amble.",
            ),
            ("human", cypher_template),
        ]
    )
    cypher_response = (
        RunnablePassthrough.assign(
            schema=lambda _: person_schema,
        )
        | cypher_prompt
        | init_settings.cypher_model.bind()
        | CypherOutputParser()
    )

    cypher = cypher_response.invoke({
        "profile": profile,
        "number": top_k,
        "desire": desire
    })
    return cypher


def get_cypher_query(cypher):
    query_results = init_settings.graph.query(cypher)
    return query_results[0]
