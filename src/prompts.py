from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

ANSWER_PROMPT = ChatPromptTemplate.from_template("""\

The following is your task:

Answer the question based only on the following context from files. If the context is irrelevant, don't use your own knowledge but say that the context doesn't contain the answer.
You are a chatbot, and you have a record of your conversation with the user.
History:

{chat_history}

You are also given the following context from the files.
Context:

{context}


If you find the context irrelevant, say so and don't use your own knowledge to give an answer.

Answer to the input based on everything above: {question}
"""
)

DOCUMENT_PROMPT = PromptTemplate.from_template("Source {source}:\n{content}")

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""\
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language, do not change details and include every relevant detail without chainging them.
Keep the question as close to the original as possible, but make sure it is a standalone question. Make sure the standalone question matches the original question in meaning, intent, tone, and is as close as possible to the original question (with the added context).
DON'T BE TALKATIVE, DON'T USE COURTEOUS PHRASES, JUST GIVE ME THE STANDALONE QUESTION REQUESTED."

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
"""
)