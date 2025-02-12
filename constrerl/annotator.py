from .erl_schema import build_grammar, build_model
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

ANNOTATION_SYSTEM_PROMPT = """You are annotating a medical scientific article. You return every relation within the article as JSON. The returned relations include the relation type and text. You also include whether the relation occurs in the abstract or title.
"""


class Annotator:
    def __init__(
        self, model: Llama, gen_tokens=4096, system_prompt=ANNOTATION_SYSTEM_PROMPT
    ):
        self.model = model
        self.gen_tokens = gen_tokens
        self.example_messages: list[ChatCompletionRequestMessage] = [
            {"role": "system", "content": system_prompt}
        ]
        self.erl_type = build_model()
        self.erl_grammar = build_grammar()
        self.llama_grammar = LlamaGrammar(_grammar=self.erl_grammar)

    def __prompt_article(self, metadata: Metadata):
        return {
            "role": "user",
            "content": f"Title: {metadata.title}\nAbstract: {metadata.abstract}",
        }

    def __prompt(self, article: Article):
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
        simplified_relations = self.erl_type.model_validate(relation_json)

        return [
            self.__prompt_article(article.metadata),
            {"role": "assistant", "content": simplified_relations.model_dump_json()},
        ]

    def add_prompt_examples(self, articles: list[Article]):
        for article in articles:
            self.example_messages.extend(self.__prompt(article))

    def annotate(
        self, articles: dict[str, Metadata]
    ) -> list[TernaryMentionBasedRelation]:
        annotated_relations = {}
        progress = tqdm(articles.items(), desc="Annotating articles")
        for id, article in progress:
            messages = self.example_messages + [self.__prompt_article(article)]
            chat_response = self.model.create_chat_completion(
                messages,
                max_tokens=self.gen_tokens,
                grammar=self.llama_grammar,
            )
            relation_response = self.erl_type.model_validate_json(
                chat_response.choices[-1].message.content
            )
            annotated_relations[id] = relation_response
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
