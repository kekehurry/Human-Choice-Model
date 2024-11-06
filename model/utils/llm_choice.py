from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from .init_settings import init_settings

choice_template = """
What {choice_type} would a person who is {profile} and lives in {city} choose, when he want to {desire}? {additional_condition}
-
Context: This is the reference of people who have similar profile with the given profile in Boston and their choice.
if scored is provided, higher socre of related choice in context means more likely to be chosen by people with similar profile.
{recommendation}

Note: 
- Output the weights of all options provided in a json format.
- You should consider the profile, the choices other people with similar profile did, and the possible cultural difference between Boston and {city}.
- The sum of all weights should equals to 1.

Answer Format:
{{
'thought': 'Based on the profile, xxxx. Consider the choices of people with similar profile, xxx. And the culture in {city}, xxx. I think xxx',
'final_answer':[
{{'choice': option1. 'weight': possibility for option1}},{{'choice': option1. 'weight': possibility for option2}},...
],
}}

"""


def get_llm_choice(profile, desire, choice_type, recommendation, city='Boston', additional_condition=''):

    choose_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert analyst of human behavior, given the context of profile and behaviors of similar people, evaluate the possibility of different options.",
            ),
            ("human", choice_template),
        ]
    )
    choice_response = (
        choose_prompt
        | init_settings.choice_model.bind()
        | StrOutputParser()
    )

    answer = choice_response.invoke({
        "profile": profile,
        "desire": desire,
        "choice_type": choice_type,
        "recommendation": recommendation,
        "city": city,
        "additional_condition": additional_condition
    })

    return answer
