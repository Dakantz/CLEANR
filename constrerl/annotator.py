from .erl_schema import (
    build_grammar,
    build_model,
    EnumERLModel,
    StringERLModel,
    ExtendedStringERLModel,
    ExtendedEnumERLModel,
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

from sqlalchemy import create_engine, text, select
from sqlalchemy.orm import Session
from .db_schema import Base, RelDocument

from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel

ANNOTATION_SYSTEM_PROMPT = """You are annotating a medical scientific title and abstract. You return all relation between entities within the title and abstract as JSON. The returned data include the relation type and text and should cover all relations over all relevant entities occurring in the text."""


class Annotator:
    def __init__(
        self,
        model: Llama = None,
        langchain: BaseChatModel = None,
        gen_tokens=4096,
        system_prompt=ANNOTATION_SYSTEM_PROMPT,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        conn_str="postgresql+psycopg://pgvector:pgvector@localhost:5450/pgvector",
        top_k=10,
        add_few_shot=False,
        add_rag=False,
        reorder=False,
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
        self.system_message: list[ChatCompletionRequestMessage] = [
            {"role": "system", "content": system_prompt}
        ]
        self.example_messages = [*self.system_message]
        self.erl_grammar = build_grammar()
        self.llama_grammar = LlamaGrammar(_grammar=self.erl_grammar)

        if langchain is not None:
            self.structured_llm = langchain.with_structured_output(StringERLModel)
        self.erl_model = StringERLModel
        self.extended_erl_model = ExtendedStringERLModel
        self.engine = create_engine(conn_str)
        self.embedding_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        self.top_k = top_k
        self.few_shot = add_few_shot
        self.rag = add_rag
        self.score_reweights = score_reweights
        self.reorder = reorder

    @classmethod
    def __prompt_article(self, metadata: Metadata):
        return {
            "role": "user",
            "content": f"Title: {metadata.title}\nAbstract: {metadata.abstract}",
        }

    @classmethod
    def prompt_and_respone(
        self, article: Article
    ) -> list[ChatCompletionRequestMessage]:
        simplified_relations = article_to_enum_model(
            article, model=ExtendedEnumERLModel
        )
        simplified_relations_string = convert_to_string_model(
            simplified_relations, ExtendedStringERLModel
        )

        return [
            self.__prompt_article(article.metadata),
            {
                "role": "assistant",
                "content": simplified_relations_string.model_dump_json(),
            },
        ]

    def add_prompt_examples(self, articles: list[Article]):
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

    def find_similar_examples(self, article: Metadata):
        search_embedding = self.embedding_model.encode(
            [article.title + "\n" + article.abstract],
            batch_size=12,
            max_length=8192,  # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
        )["dense_vecs"]
        with Session(self.engine) as session:
            collection_docs: dict[str, list] = {}
            if self.reorder:
                for collection in self.score_reweights.keys():
                    collection_docs[collection] = session.execute(
                        select(
                            RelDocument.doc_meta.label("doc_meta"),
                            RelDocument.vectors.cosine_distance(
                                search_embedding[0]
                            ).label("score"),
                        )
                        .filter(RelDocument.collection.contains(collection))
                        .order_by(
                            RelDocument.vectors.cosine_distance(search_embedding[0])
                        )
                        .limit(self.top_k)
                    ).all()
                best_matches: list[tuple[RelDocument, float]] = []
                for collection, docs in collection_docs.items():
                    for r in docs:
                        document: str = r.doc_meta
                        score = r.score * self.score_reweights[collection]
                        best_matches.append(
                            (Article.model_validate_json(document), score)
                        )
                best_matches = sorted(best_matches, key=lambda x: x[1], reverse=False)[
                    -self.top_k :
                ]
                articles = [match for match, score in best_matches]
                return articles
            else:
                best_matches_documents = session.execute(
                    select(
                        RelDocument.doc_meta.label("doc_meta"),
                        RelDocument.vectors.cosine_distance(search_embedding[0]).label(
                            "score"
                        ),
                    )
                    .order_by(RelDocument.vectors.cosine_distance(search_embedding[0]))
                    .limit(self.top_k)
                ).all()
                best_matches: list[Article] = []
                for r in best_matches_documents:
                    document: str = r.doc_meta
                    best_matches.append(Article.model_validate_json(document))
                return best_matches

    def annotate(self, articles: dict[str, Metadata]) -> dict[str, StringERLModel]:
        annotated_relations = {}
        progress = tqdm(articles.items(), desc="Annotating articles")
        for id, article in progress:
            prompts = [*self.system_message]
            if self.few_shot:
                for ex in self.example_messages:
                    prompts.extend(ex)
            if self.rag:
                similar_articles = self.find_similar_examples(article)
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
                annotated_relations[id] = chat_response

            else:
                messages = prompts + [self.__prompt_article(article)]
                chat_response = self.model.create_chat_completion(
                    messages,
                    max_tokens=self.gen_tokens,
                    grammar=self.llama_grammar,
                )
                response = chat_response["choices"][-1]["message"]["content"]
                try:
                    fixed_response = json_repair.repair_json(response)
                    relation_response = self.erl_model.model_validate_json(
                        fixed_response, strict=False
                    )
                    # relation_response_enum = convert_to_enum_model(relation_response)
                    annotated_relations[id] = relation_response
                except Exception as e:
                    print(f"Error in article {id}")
                    print(response)
                    print(e)
            progress.set_postfix({"id": id})
        return annotated_relations

    def embed_articles(self, articles: list[Article], setup_db=True, collection="all"):
        if setup_db:
            with Session(self.engine) as session:
                session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                session.commit()
            Base.metadata.drop_all(self.engine)
            Base.metadata.create_all(self.engine)

        with Session(self.engine) as session:
            for i, article in enumerate(tqdm(articles, desc="Embedding articles")):
                article_json = article.model_dump_json()
                embeddings = self.embedding_model.encode(
                    [article.metadata.title + "\n" + article.metadata.abstract],
                    batch_size=12,
                    max_length=8192,  # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                )["dense_vecs"]
                document = RelDocument(
                    title=article.metadata.title,
                    abstract=article.metadata.abstract,
                    vectors=embeddings[0, :].astype(float),
                    doc_meta=article_json,
                    collection=collection,
                )
                session.add(document)
                if i % 100 == 0:
                    session.commit()
            session.commit()


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


def article_to_enum_model(article: Article, model=EnumERLModel):
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
