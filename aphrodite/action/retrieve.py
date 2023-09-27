from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.document_transformers import LongContextReorder
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma

from aphrodite.action import Action

STUFF_PROMPT_TEMPLATE = """Given this text extracts:
---
{context}
---
Please answer the following question:
{query}
"""

PERSONA_TEXT = [
    "Minji is a 28 years old.",
    "Minji likes to play football, video game, and soccer.",
    "Her favorite video game is Genshin.",
    "Minji hates insect especially spider.",
    "Minji loves BTS, i.e., K-pop idol",
    "Minji is a big fan of BlackPink (i.e., K-pop idol).",
    "Minji likes to eat chicken with salad.",
    "Minji cannot drive a car.",
    "Minji is not good at studying science.",
    "Minji's favorite food is as follows: chicken, salad, and orange juice.",
    "Her favorite song is FGSM.",
]


class PersonaRetrieve(Action):
    def __init__(self, question: str, doc_type: str, **kwargs) -> None:
        self._question = question
        self._doc_type = doc_type
        self._chat_model = ChatOpenAI(openai_api_key=kwargs["config"]["API_KEY"])
        self._h_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def do_h_emb_retrive(self, query: str) -> str:
        retriever = Chroma.from_texts(
            PERSONA_TEXT, embedding=self._h_embedding
        ).as_retriever(search_kwargs={"k": 5})

        docs = retriever.get_relevant_documents(query)
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(docs)
        response = self._polishing_docs_with_chain(reordered_docs, query)
        return response

    def _polishing_docs_with_chain(self, reordered_docs: list, query: str) -> str:
        document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )
        document_variable_name = "context"
        prompt = PromptTemplate(
            template=STUFF_PROMPT_TEMPLATE, input_variables=["context", "query"]
        )

        llm_chain = LLMChain(llm=self._chat_model, prompt=prompt)
        chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name,
        )
        return chain.run(input_documents=reordered_docs, query=query)


if __name__ == "__main__":
    import json

    with open("config.json", "r", encoding="UTF8") as f:
        json_data = json.load(f)

    query = "What is Minji's favorite K-pop idol?"
    doc_type = "markdown"
    retriever = PersonaRetrieve(question=query, doc_type=doc_type, **json_data)
    response = retriever.do_h_emb_retrive(query=query)
    print(response["output_text"])
