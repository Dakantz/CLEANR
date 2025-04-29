from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class Metadata(BaseModel):
    title: str
    author: str
    journal: str
    year: str
    abstract: str
    annotator: str | None = None


class Entity(BaseModel):
    start_idx: int
    end_idx: int
    location: str
    text_span: str
    label: str


class Relation(BaseModel):
    subject_start_idx: int
    subject_end_idx: int
    subject_location: str
    subject_text_span: str
    subject_label: str
    predicate: str
    object_start_idx: int
    object_end_idx: int
    object_location: str
    object_text_span: str
    object_label: str


class BinaryTagBasedRelation(BaseModel):
    subject_label: str
    object_label: str


class TernaryTagBasedRelation(BaseModel):
    subject_label: str
    predicate: str
    object_label: str


class TernaryMentionBasedRelation(BaseModel):
    subject_text_span: str
    subject_label: str
    predicate: str
    object_text_span: str
    object_label: str


class Article(BaseModel):
    metadata: Optional[Metadata] = None
    entities: Optional[List[Entity]] = None
    relations: Optional[List[Relation]] = None
    binary_tag_based_relations: Optional[List[BinaryTagBasedRelation]] = None
    ternary_tag_based_relations: Optional[List[TernaryTagBasedRelation]] = None
    ternary_mention_based_relations: Optional[List[TernaryMentionBasedRelation]] = None
