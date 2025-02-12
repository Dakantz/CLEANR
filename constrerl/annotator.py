from .erl_schema import (
    build_grammar,
    build_model,
    EnumERLModel,
    StringERLModel,
    convert_to_enum_model,
    convert_to_string_model,
)
from .annotations_schema import (
    Metadata,
    Entity,
    Relation,
    BinaryTagBasedRelation,
    TernaryTagBasedRelation,
    TernaryMentionBasedRelation,
    Article,
)
from llama_cpp import Llama, ChatCompletionRequestMessage, LlamaGrammar
from tqdm import tqdm
import json
import json_repair
from langchain_core.language_models.chat_models import BaseChatModel, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

ANNOTATION_SYSTEM_PROMPT = """You are annotating a medical scientific title and abstract. You return only the most relevant relations within the title and abstract as JSON. The few returned relations include the relation type and text. It is very important to return only the most relevant relations.
"""


class Annotator:
    def __init__(
        self,
        model: Llama = None,
        langchain: BaseChatModel = None,
        gen_tokens=4096,
        system_prompt=ANNOTATION_SYSTEM_PROMPT,
    ):
        self.model = model
        self.langchain = langchain
        self.gen_tokens = gen_tokens
        self.example_messages: list[ChatCompletionRequestMessage] = [
            {"role": "system", "content": system_prompt}
        ]
        self.erl_grammar = build_grammar()
        self.llama_grammar = LlamaGrammar(_grammar=self.erl_grammar)

        if langchain is not None:
            self.structured_llm = langchain.with_structured_output(StringERLModel)

    @classmethod
    def __prompt_article(self, metadata: Metadata):
        return {
            "role": "user",
            "content": f"Title: {metadata.title}\nAbstract: {metadata.abstract}",
        }

    @classmethod
    def prompt(self, article: Article) -> list[ChatCompletionRequestMessage]:
        relation_jsons = [
            relation.model_dump()
            for relation in article.ternary_mention_based_relations
        ]
        relation_json = {"relations": relation_jsons}
        for relation in relation_json["relations"]:
            subject_entity: Entity = next(
                (
                    entity
                    for entity in article.entities
                    if entity.text_span == relation["subject_text_span"]
                ),
                None,
            )
            if subject_entity is not None:
                relation["subject_label"] = subject_entity.label
                relation["subject_location"] = subject_entity.location
            object_entity: Entity = next(
                (
                    entity
                    for entity in article.entities
                    if entity.text_span == relation["object_text_span"]
                ),
                None,
            )
            if object_entity is not None:
                relation["object_label"] = object_entity.label
                relation["object_location"] = object_entity.location
        simplified_relations = EnumERLModel.model_validate(relation_json)
        simplified_relations_string = convert_to_string_model(simplified_relations)

        return [
            self.__prompt_article(article.metadata),
            {
                "role": "assistant",
                "content": simplified_relations_string.model_dump_json(),
            },
        ]

    def add_prompt_examples(self, articles: list[Article]):
        for article in articles:
            self.example_messages.extend(self.prompt(article))

    def __message_to_langchain(self, message: ChatCompletionRequestMessage):
        if message["role"] == "system":
            return SystemMessage(message["content"])
        if message["role"] == "user":
            return HumanMessage(message["content"])
        if message["role"] == "assistant":
            return AIMessage(message["content"])

    def annotate(
        self, articles: dict[str, Metadata]
    ) -> dict[str, TernaryMentionBasedRelation]:
        annotated_relations = {}
        progress = tqdm(articles.items(), desc="Annotating articles")
        for id, article in progress:
            if self.langchain is not None:
                langchain_messages = [
                    self.__message_to_langchain(message)
                    for message in self.example_messages
                ]
                few_shot_prompt = ChatPromptTemplate.from_messages(langchain_messages)
                chain = (
                    {"query": RunnablePassthrough()}
                    | few_shot_prompt
                    | self.structured_llm
                )
                chat_response = chain.invoke(self.__prompt_article(article))
                annotated_relations[id] = chat_response

            else:
                messages = self.example_messages + [self.__prompt_article(article)]
                chat_response = self.model.create_chat_completion(
                    messages,
                    max_tokens=self.gen_tokens,
                    grammar=self.llama_grammar,
                )
                response = chat_response["choices"][-1]["message"]["content"]
                try:
                    fixed_response = json_repair.repair_json(response)
                    relation_response = self.erl_type.model_validate_json(
                        fixed_response, strict=False
                    )
                    relation_response_enum = convert_to_enum_model(relation_response)
                    annotated_relations[id] = relation_response_enum
                except Exception as e:
                    print(f"Error in article {id}")
                    print(response)
                    print(e)
            progress.set_postfix({"id": id})
        return annotated_relations


def load_train(file_path: str):
    with open(file_path, "r") as file:
        data = json.load(file)
    articles: dict[str, Article] = {}
    for id, article in data.items():
        articles[id] = Article.model_validate(article)
    return articles


def load_test(file_path: str):
    with open(file_path, "r") as file:
        data = json.load(file)
    articles: dict[str, Metadata] = {}
    for id, article in data.items():
        articles[id] = Metadata.model_validate(article)
    return
