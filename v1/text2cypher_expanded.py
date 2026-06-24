"""
text2cypher_expanded.py

Wraps the frozen text2cypher.ask() with query expansion applied to the
question text BEFORE Cypher generation. text2cypher.py itself stays
untouched (it remains the frozen baseline) -- this is a separate thin
layer so the comparison is clean: baseline vs baseline+expansion, same
underlying Text2Cypher logic.

Unlike the Template 1/2 case (where expansion enriches an embedding),
here the expanded terms are surfaced directly in the question text the
LLM sees, since Text2Cypher's only lever is the WHERE clause it writes --
there's no embedding step to enrich.
"""

from query_expansion import expand_query
from text2cypher import ask


def ask_with_expansion(question: str) -> list:
    """
    Expand the question, then pass an augmented version into the frozen
    Text2Cypher pipeline. The augmentation explicitly tells the model these
    are alternative terms to match with OR, not additional required terms
    to AND together -- that distinction is the actual baseline bug (Q02
    and Q04 failed because the model ANDed two specific words instead of
    treating them as alternatives).
    """
    expanded = expand_query(question)

    if expanded == question:
        # No expansion found (dictionary + LLM both came up empty) --
        # just run the original question through, unmodified.
        return ask(question)

    augmented_question = (
        f"{question}\n\n"
        f"(Note: when filtering by keyword, treat the following as "
        f"alternative terms to match with OR, not all of which need to "
        f"be present: {expanded.split('related terms: ')[1].rstrip(')')})"
    )
    return ask(augmented_question)


if __name__ == "__main__":
    test_questions = [
        "Find researchers who work on biocontrol agents.",
        "Find projects related to packaging technology.",
    ]
    for q in test_questions:
        print(f"\nQ: {q}")
        for row in ask_with_expansion(q):
            print(f"  {row}")