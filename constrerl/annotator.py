from .erl_schema import (
    build_grammar,
    EnumERLModel,
    StringERLModel,
    ExtendedStringERLModel,
    ExtendedEnumERLModel,
    clean_label,
    convert_to_string_model,
    entity_labels,
)
from .annotation_model import (
    Conceptlevelrelation,
    FullRelation,
    Metadata,
    Entity,
    AnnotatedArticle,
    Relation,
)
from llama_cpp import Llama, ChatCompletionRequestMessage, LlamaGrammar
from tqdm import tqdm
import json
import json_repair
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

from sqlalchemy import create_engine, text, select
from sqlalchemy.orm import Session
from .db_schema import Base, RelDocument

from sentence_transformers import SentenceTransformer
import torch as th
import numpy as np
from pydantic import BaseModel
import json
import re

ANNOTATION_SYSTEM_PROMPT = (
    """You are a medical expert annotating a medical scientific title and abstract."""
)


class Sentence(BaseModel):
    start_idx: int
    from_article: Metadata
    text: str
    title: bool = False

    entities: list[Entity] | None = None
    relations: list[FullRelation | Relation] | None = None


def article_to_sentences(
    article: Metadata, relations: list[Relation] = [], entities: list[Entity] = []
):
    sentences: list[Sentence] = []

    title_entities = [ent for ent in entities if ent.location.lower() == "title"]
    title_relations = [
        rel
        for rel in relations
        if (rel.object_location.lower() == "title")
        or (rel.subject_location.lower() == "title")
    ]
    sentences.append(
        Sentence(
            start_idx=0,
            from_article=article,
            text=article.title,
            title=True,
            entities=title_entities,
            relations=title_relations,
        )
    )
    sentence_text = re.split(r"\.[\\n ]+", article.abstract)  # A *very* basic heuristic
    last_idx = 0
    for sentence in sentence_text:
        start_idx = article.abstract.find(sentence, max(0, last_idx - 2))
        if start_idx == -1:
            raise ValueError(
                f"Sentnce {sentence} not contained in abstract, start_idx {last_idx}."
            )
        end_idx = start_idx + len(sentence)
        sentence_entities: list[Entity] = [
            ent
            for ent in entities
            if ent.start_idx >= start_idx and ent.end_idx <= end_idx
        ]
        for ent in sentence_entities:
            ent.start_idx = ent.start_idx - start_idx
            ent.end_idx = ent.end_idx - start_idx
        sentence_relations: list[Relation] = [
            rel
            for rel in relations
            if (rel.object_start_idx >= start_idx and rel.object_end_idx <= end_idx)
            or (rel.subject_start_idx >= start_idx and rel.subject_end_idx <= end_idx)
        ]
        for rel in sentence_relations:
            rel.subject_start_idx = rel.subject_start_idx - start_idx
            rel.subject_end_idx = rel.subject_end_idx - start_idx
            rel.object_start_idx = rel.object_start_idx - start_idx
            rel.object_end_idx = rel.object_end_idx - start_idx
        sentences.append(
            Sentence(
                start_idx=start_idx,
                from_article=article,
                text=sentence,
                entities=sentence_entities,
                relations=sentence_relations,
            )
        )
        last_idx = end_idx
    return sentences


def annotated_sentences_to_article(
    sentences: list[Sentence], metadata: Metadata
) -> AnnotatedArticle:
    all_entities: list[Entity] = []
    all_relations: list[FullRelation] = []
    for sentence in sentences:
        if sentence.entities is not None:
            for sen in sentence.entities:
                sen.start_idx = sen.start_idx + sentence.start_idx
                sen.end_idx = sen.end_idx + sentence.start_idx
            all_entities.extend(sentence.entities)
        if sentence.relations is not None:
            for rel in sentence.relations:
                rel.subject_start_idx = rel.subject_start_idx + sentence.start_idx
                rel.subject_end_idx = rel.subject_end_idx + sentence.start_idx
                rel.object_start_idx = rel.object_start_idx + sentence.start_idx
                rel.object_end_idx = rel.object_end_idx + sentence.start_idx
            all_relations.extend(sentence.relations)
    return AnnotatedArticle(
        metadata=metadata,
        entities=all_entities,
        relations=all_relations,
    )


class Annotator:
    def __init__(
        self,
        model: Llama = None,
        langchain: BaseChatModel = None,
        gen_tokens=4096,
        system_prompt=ANNOTATION_SYSTEM_PROMPT,
        embedding_model="NeuML/pubmedbert-base-embeddings",
        top_k=10,
        add_few_shot=False,
        add_rag=False,
        reorder=False,
        add_entity_labels=False,
        score_reweights={
            "platinum": 1.0,
            "gold": 0.9,
            "silver": 0.8,
            "bronze": 0.7,
        },
    ):
        self.model = model
        self.langchain = langchain
        self.gen_tokens = gen_tokens
        if add_entity_labels:
            system_prompt = (
                system_prompt
                + " The possible entities are:\n"
                + "\n".join(
                    [f"{clean_label(e['label'])}: {e['desc']}" for e in entity_labels]
                )
            )
        self.system_message: list[ChatCompletionRequestMessage] = [
            {"role": "system", "content": system_prompt}
        ]
        self.example_messages = [*self.system_message]

        self.embedding_model = SentenceTransformer(embedding_model).to(
            "cuda" if th.cuda.is_available() else "mps"
        )

        self.top_k = top_k
        self.few_shot = add_few_shot
        self.rag = add_rag
        self.score_reweights = score_reweights
        self.reorder = reorder

        self.loaded_articles: dict[str, AnnotatedArticle] = {}
        self.loaded_sentences: dict[str, Sentence] = {}
        self.embeddings: dict[str, th.Tensor] = {}
        self.embeddings_sentences: dict[str, th.Tensor] = {}

    @classmethod
    def __prompt_article(self, metadata: Metadata):
        return {
            "role": "user",
            "content": f"{metadata.title}\n{metadata.abstract}",
        }

    @classmethod
    def relations_to_str(self, relations: list[Relation], sep="\n", sep_rel="|"):
        return sep.join(
            [
                sep_rel.join(
                    [
                        rel.subject_label,
                        rel.predicate,
                        rel.object_label,
                    ]
                )
                for rel in relations
            ]
        )

    @classmethod
    def entities_to_str(self, entities: list[Entity], sep="\n"):
        return sep.join([f"{ent.label} ({ent.text_span})" for ent in entities])

    @classmethod
    def prompt_and_response_entities(
        self, sent: Sentence
    ) -> list[ChatCompletionRequestMessage]:
        entities_str = Annotator.entities_to_str(sent.entities)

        return [
            self.__prompt_article(sent.from_article),
            {
                "role": "assistant",
                "content": entities_str,
            },
        ]

    @classmethod
    def prompt_and_response_relations(
        self, sent: Sentence
    ) -> list[ChatCompletionRequestMessage]:
        relations_str = Annotator.relations_to_str(sent.relations)
        return [
            self.__prompt_article(sent.from_article),
            {
                "role": "assistant",
                "content": relations_str,
            },
        ]

    def embed_article(self, article: AnnotatedArticle):
        search_embedding = self.embedding_model.encode(
            [article.title + "\n" + article.abstract]
        )
        return search_embedding

    def load_articles(self, articles: dict[str, AnnotatedArticle]):
        for id, article in tqdm(list(articles.items()), desc="Embedding articles"):
            self.embeddings[id] = self.embed_article(article.metadata)
            sentences = article_to_sentences(
                article.metadata, article.relations, article.entities
            )
            sentence_texts = [sentence.text for sentence in sentences]
            embedded_sentences = self.embedding_model.encode(sentence_texts)
            for i, sentence in enumerate(sentences):
                sid = f"{id}_{sentence.start_idx}"
                self.loaded_sentences[sid] = sentence
                self.embeddings_sentences[sid] = embedded_sentences[i, :]
        self.loaded_articles = articles

    def add_prompt_examples(self, articles: list[AnnotatedArticle]):
        self.example_messages = [
            self.prompt_and_respone(article) for article in articles
        ]

    def __message_to_langchain(self, message: ChatCompletionRequestMessage):
        if message["role"] == "system":
            return SystemMessage(message["content"])
        if message["role"] == "user":
            return HumanMessage(message["content"])
        if message["role"] == "assistant":
            return AIMessage(message["content"])

    def find_similar_examples(self, txt: str, id: str | None):
        # search_embedding = self.embedding_model.encode(
        #     [article.title + "\n" + article.abstract],
        #     batch_size=12,
        #     max_length=8192,  # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
        # )["dense_vecs"]
        if id in self.embeddings_sentences.keys():
            search_embedding = self.embeddings_sentences[id]
        else:
            search_embedding = self.embedding_model.encode([txt])[0]
        # search_embedding = search_embedding[0]
        ids = [
            k
            for k in self.embeddings_sentences.keys()
            if id is None or (not k.startswith(id))
        ]
        dense_matrix = np.stack([self.embeddings_sentences[id] for id in ids])
        similarities = dense_matrix @ search_embedding
        max_idx = np.argsort(similarities, axis=0)[-self.top_k :][::-1]
        best_matches_sentences = [self.loaded_sentences[ids[idx]] for idx in max_idx]
        return best_matches_sentences

    def annotate(self, articles: dict[str, Metadata]) -> dict[str, AnnotatedArticle]:
        annotated_articles = {}
        progress = tqdm(articles.items(), desc="Annotating articles")
        for id, article in progress:
            annotated_articles[id] = StringERLModel(
                relations=[],
            )
            prompts = [*self.system_message]
            if self.few_shot:
                for ex in self.example_messages:
                    prompts.extend(ex)
            if self.rag:
                similar_articles = self.find_similar_examples(article, id)
                similar_article_messages = [
                    self.prompt_and_respone(similar_article)
                    for similar_article in similar_articles
                ]
                for ex in similar_article_messages:
                    prompts.extend(ex)

            if self.langchain is not None:
                langchain_messages = [
                    self.__message_to_langchain(message) for message in prompts
                ]
                few_shot_prompt = ChatPromptTemplate.from_messages(langchain_messages)
                chain = (
                    {"query": RunnablePassthrough()}
                    | few_shot_prompt
                    | self.structured_llm
                )
                chat_response = chain.invoke(self.__prompt_article(article))
                # relation_response_enum = convert_to_enum_model(chat_response)
                annotated_articles[id] = chat_response

            else:
                messages = prompts + [self.__prompt_article(article)]

                try:
                    chat_response = self.model.create_chat_completion(
                        messages,
                        max_tokens=self.gen_tokens,
                        grammar=self.llama_grammar,
                    )
                    response = chat_response["choices"][-1]["message"]["content"]
                    resp = json_repair.repair_json(response, logging=True)
                    log = []
                    fixed_response = ""
                    if isinstance(resp, tuple):
                        fixed_response = resp[0]
                        log = resp[1]
                    else:
                        fixed_response = resp
                    if len(log) > 0:
                        print(f"Error in article {id}")
                        print(log)
                        print("removing last element")
                        fixed_response["relations"] = fixed_response["relations"][:-1]
                    resp_str = json.dumps(fixed_response)
                    relation_response = self.erl_model.model_validate_json(
                        resp_str, strict=False
                    )
                    # relation_response_enum = convert_to_enum_model(relation_response)
                    annotated_articles[id] = relation_response
                except Exception as e:
                    print(f"Error in article {id}")
                    print(response)
                    print(e)
            progress.set_postfix({"id": id})
        return annotated_articles


def load_train(file_path: str):
    with open(file_path, "r") as file:
        data = json.load(file)
    articles: dict[str, AnnotatedArticle] = {}
    for id, article in data.items():
        articles[id] = AnnotatedArticle.model_validate(article)
    return articles


def load_test(file_path: str):
    with open(file_path, "r") as file:
        data = json.load(file)
    articles: dict[str, Metadata] = {}
    for id, article in data.items():
        articles[id] = Metadata.model_validate(article)
    return articles


def article_to_enum_model(article: AnnotatedArticle, model=EnumERLModel):
    relation_jsons = [
        relation.model_dump() for relation in article.ternary_mention_based_relations
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
    simplified_relations = model.model_validate(relation_json)
    return simplified_relations
