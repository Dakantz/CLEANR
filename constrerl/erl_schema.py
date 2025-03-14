from __future__ import annotations

from pydantic import BaseModel, create_model, Field
from typing import List, Tuple, Union, Annotated, Literal
from enum import Enum
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import (
    generate_gbnf_grammar_from_pydantic_models,
)
from .annotations_schema import (
    Relation,
    TernaryMentionBasedRelation,
    TernaryTagBasedRelation,
    Article,
)


entity_labels = [
    {
        "label": "Anatomical Location",
        "uri": "NCIT_C13717",
        "desc": "Named locations of or within the body. ",
    },
    {
        "label": "Animal",
        "uri": "NCIT_C14182",
        "desc": "A non-human living organism that has membranous cell walls, requires oxygen and organic foods, and is capable of voluntary movement, as distinguished from a plant or mineral.",
    },
    {
        "label": "Biomedical Technique",
        "uri": "NCIT_C15188",
        "desc": "Research concerned with the application of biological and physiological principles to clinical medicine.",
    },
    {
        "label": "Bacteria",
        "uri": "NCBITaxon_2",
        "desc": "One of the three domains of life (the others being Eukarya and ARCHAEA), also called Eubacteria. They are unicellular prokaryotic microorganisms which generally possess rigid cell walls, multiply by cell division, and exhibit three principal forms: round or coccal, rodlike or bacillary, and spiral or spirochetal.",
    },
    {
        "label": "Chemical",
        "uri": "CHEBI_59999",
        "desc": "A chemical substance is a portion of matter of constant composition, composed of molecular entities of the same type or of different types. This category also includes metabolites, which in biochemistry are the intermediate or end product of metabolism, and neurotransmitters, which are endogenous compounds used to transmit information across the synapses.",
    },
    {
        "label": "Dietary Supplement",
        "uri": "MESH_68019587",
        "desc": "Products in capsule, tablet or liquid form that provide dietary ingredients, and that are intended to be taken by mouth to increase the intake of nutrients. Dietary supplements can include macronutrients, such as proteins, carbohydrates, and fats; and/or micronutrients, such as vitamins; minerals; and phytochemicals.",
    },
    {
        "label": "DDF",
        "uri": "NCIT_C7057",
        "desc": "A condition that is relevant to human neoplasms and non-neoplastic disorders. This includes observations, test results, history and other concepts relevant to the characterization of human pathologic conditions.",
    },
    {
        "label": "Drug",
        "uri": "CHEBI_23888",
        "desc": "Any substance which when absorbed into a living organism may modify one or more of its functions. The term is generally accepted for a substance taken for a therapeutic purpose, but is also commonly used for abused substances.",
    },
    {
        "label": "Food",
        "uri": "NCIT_C1949",
        "desc": "A substance consumed by humans and animals for nutritional purpose",
    },
    {
        "label": "Gene",
        "uri": "SNOMEDCT_67261001",
        "desc": "A functional unit of heredity which occupies a specific position on a particular chromosome and serves as the template for a product that contributes to a phenotype or a biological function.",
    },
    {
        "label": "Human",
        "uri": "NCBITaxon_9606",
        "desc": "Members of the species Homo sapiens.",
    },
    {
        "label": "Microbiome",
        "uri": "OHMI_0000003",
        "desc": "This term refers to the entire habitat, including the microorganisms (bacteria, archaea, lower and higher eukaryotes, and viruses), their genomes (i.e., genes), and the surrounding environmental conditions.",
    },
    {
        "label": "Statistical Technique",
        "uri": "NCIT_C19044A",
        "desc": "method of calculating, analyzing, or representing statistical data.",
    },
]

relations = [
    {
        "heads": ["Anatomical Location"],
        "tails": ["Human", "Animal"],
        "predicate": ["Located in"],
    },
    {
        "heads": ["Bacteria"],
        "tails": ["Bacteria", "Chemical", "Drug"],
        "predicate": ["Interact"],
    },
    {"heads": ["Bacteria"], "tails": ["DDF"], "predicate": ["Influence"]},
    {"heads": ["Bacteria"], "tails": ["Gene"], "predicate": ["Change expression"]},
    {
        "heads": ["Bacteria"],
        "tails": ["Human", "Animal"],
        "predicate": ["Located in"],
    },
    {"heads": ["Bacteria"], "tails": ["Microbiome"], "predicate": ["Part of"]},
    {
        "heads": ["Chemical"],
        "tails": ["Anatomical Location", "Human", "Animal"],
        "predicate": ["Located in"],
    },
    {
        "heads": ["Chemical"],
        "tails": ["Chemical"],
        "predicate": ["Interact", "Part of"],
    },
    {
        "heads": ["Chemical"],
        "tails": ["Microbiome"],
        "predicate": ["Impact", "Produced by"],
    },
    {
        "heads": ["Chemical", "Dietary Supplement", "Drug", "Food"],
        "tails": ["Bacteria", "Microbiome"],
        "predicate": ["Impact"],
    },
    {
        "heads": ["Chemical", "Dietary Supplement", "Drug", "Food"],
        "tails": ["DDF"],
        "predicate": ["Influence"],
    },
    {
        "heads": ["Chemical", "Dietary Supplement", "Drug", "Food"],
        "tails": ["Gene"],
        "predicate": ["Change expression"],
    },
    {
        "heads": ["Chemical", "Dietary Supplement", "Drug", "Food"],
        "tails": ["Human", "Animal"],
        "predicate": ["Administered"],
    },
    {"heads": ["DDF"], "tails": ["Anatomical Location"], "predicate": ["Strike"]},
    {
        "heads": ["DDF"],
        "tails": ["Bacteria", "Microbiome"],
        "predicate": ["Change abundance"],
    },
    {"heads": ["DDF"], "tails": ["DDF"], "predicate": ["Affect", "Is a"]},
    {"heads": ["DDF"], "tails": ["Human", "Animal"], "predicate": ["Target"]},
    {
        "heads": ["Drug", "DDF"],
        "tails": ["Chemical", "Drug"],
        "predicate": ["Interact"],
    },
    {"heads": ["Drug"], "tails": ["DDF"], "predicate": ["Change effect"]},
    {
        "heads": ["Microbiome"],
        "tails": ["Anatomical Location", "Human", "Animal"],
        "predicate": ["Located in"],
    },
    {
        "heads": ["Microbiome", "Human"],  # TODO: check with gianmaria
        "tails": ["Biomedical Technique"],
        "predicate": ["Used by"],
    },
    {
        "heads": ["Microbiome"],
        "tails": ["Gene"],
        "predicate": ["Change expression"],
    },
    {"heads": ["Microbiome"], "tails": ["DDF"], "predicate": ["Is linked to"]},
    {
        "heads": ["Microbiome"],
        "tails": ["Microbiome"],
        "predicate": ["Compared to"],
    },
]
extended_relations = [
    {"heads": ["DDF"], "tails": ["human"], "predicate": ["affect"]},
    {"heads": ["DDF"], "tails": ["DDF"], "predicate": ["influence"]},
    {"heads": ["chemical"], "tails": ["DDF"], "predicate": ["change effect"]},
    {"heads": ["dietary supplement"], "tails": ["DDF"], "predicate": ["change effect"]},
    {"heads": ["human"], "tails": ["statistical technique"], "predicate": ["used by"]},
    {"heads": ["gene"], "tails": ["DDF"], "predicate": ["affect"]},
    {"heads": ["gene"], "tails": ["animal"], "predicate": ["located in"]},
    {"heads": ["human"], "tails": ["human"], "predicate": ["target"]},
    {
        "heads": ["microbiome"],
        "tails": ["statistical technique"],
        "predicate": ["used by"],
    },
    {
        "heads": ["anatomical location"],
        "tails": ["biomedical technique"],
        "predicate": ["used by"],
    },
    {"heads": ["drug"], "tails": ["animal"], "predicate": ["located in"]},
    {"heads": ["drug"], "tails": ["drug"], "predicate": ["change effect"]},
    {"heads": ["chemical"], "tails": ["DDF"], "predicate": ["is linked to"]},
    {"heads": ["bacteria"], "tails": ["chemical"], "predicate": ["influence"]},
    {
        "heads": ["chemical"],
        "tails": ["anatomical location"],
        "predicate": ["influence"],
    },
    {"heads": ["microbiome"], "tails": ["DDF"], "predicate": ["affect"]},
    {"heads": ["microbiome"], "tails": ["DDF"], "predicate": ["influence"]},
    {"heads": ["microbiome"], "tails": ["microbiome"], "predicate": ["impact"]},
    {"heads": ["microbiome"], "tails": ["chemical"], "predicate": ["is linked to"]},
    {"heads": ["microbiome"], "tails": ["drug"], "predicate": ["is linked to"]},
    {"heads": ["human"], "tails": ["gene"], "predicate": ["used by"]},
    {"heads": ["DDF"], "tails": ["bacteria"], "predicate": ["impact"]},
    {"heads": ["animal"], "tails": ["biomedical technique"], "predicate": ["used by"]},
    {"heads": ["DDF"], "tails": ["microbiome"], "predicate": ["impact"]},
    {"heads": ["bacteria"], "tails": ["human"], "predicate": ["administered"]},
    {"heads": ["dietary supplement"], "tails": ["drug"], "predicate": ["influence"]},
    {"heads": ["bacteria"], "tails": ["drug"], "predicate": ["influence"]},
    {"heads": ["chemical"], "tails": ["anatomical location"], "predicate": ["impact"]},
    {"heads": ["dietary supplement"], "tails": ["DDF"], "predicate": ["affect"]},
    {
        "heads": ["microbiome"],
        "tails": ["dietary supplement"],
        "predicate": ["used by"],
    },
    {"heads": ["microbiome"], "tails": ["human"], "predicate": ["target"]},
    {"heads": ["DDF"], "tails": ["anatomical location"], "predicate": ["affect"]},
    {"heads": ["gene"], "tails": ["biomedical technique"], "predicate": ["used by"]},
    {"heads": ["animal"], "tails": ["statistical technique"], "predicate": ["used by"]},
    {"heads": ["microbiome"], "tails": ["animal"], "predicate": ["is linked to"]},
    {"heads": ["drug"], "tails": ["DDF"], "predicate": ["affect"]},
    {
        "heads": ["microbiome"],
        "tails": ["biomedical technique"],
        "predicate": ["is linked to"],
    },
    {"heads": ["anatomical location"], "tails": ["DDF"], "predicate": ["is linked to"]},
    {
        "heads": ["microbiome"],
        "tails": ["statistical technique"],
        "predicate": ["is linked to"],
    },
    {"heads": ["gene"], "tails": ["DDF"], "predicate": ["influence"]},
    {"heads": ["gene"], "tails": ["microbiome"], "predicate": ["impact"]},
    {
        "heads": ["chemical"],
        "tails": ["biomedical technique"],
        "predicate": ["used by"],
    },
    {"heads": ["drug"], "tails": ["human"], "predicate": ["target"]},
    {
        "heads": ["biomedical technique"],
        "tails": ["DDF"],
        "predicate": ["is linked to"],
    },
    {"heads": ["biomedical technique"], "tails": ["DDF"], "predicate": ["affect"]},
    {
        "heads": ["microbiome"],
        "tails": ["anatomical location"],
        "predicate": ["is linked to"],
    },
    {"heads": ["gene"], "tails": ["anatomical location"], "predicate": ["located in"]},
    {"heads": ["drug"], "tails": ["drug"], "predicate": ["influence"]},
    {"heads": ["gene"], "tails": ["chemical"], "predicate": ["interact"]},
    {"heads": ["chemical"], "tails": ["anatomical location"], "predicate": ["strike"]},
    {"heads": ["microbiome"], "tails": ["human"], "predicate": ["is linked to"]},
    {"heads": ["chemical"], "tails": ["chemical"], "predicate": ["influence"]},
    {"heads": ["bacteria"], "tails": ["microbiome"], "predicate": ["located in"]},
    {"heads": ["animal"], "tails": ["animal"], "predicate": ["located in"]},
    {
        "heads": ["bacteria"],
        "tails": ["anatomical location"],
        "predicate": ["located in"],
    },
    {"heads": ["microbiome"], "tails": ["drug"], "predicate": ["used by"]},
    {"heads": ["drug"], "tails": ["human"], "predicate": ["change effect"]},
    {"heads": ["chemical"], "tails": ["animal"], "predicate": ["target"]},
    {"heads": ["drug"], "tails": ["animal"], "predicate": ["target"]},
    {"heads": ["chemical"], "tails": ["microbiome"], "predicate": ["influence"]},
    {"heads": ["animal"], "tails": ["DDF"], "predicate": ["used by"]},
    {"heads": ["bacteria"], "tails": ["animal"], "predicate": ["influence"]},
    {"heads": ["chemical"], "tails": ["animal"], "predicate": ["influence"]},
    {"heads": ["bacteria"], "tails": ["DDF"], "predicate": ["is linked to"]},
    {"heads": ["biomedical technique"], "tails": ["DDF"], "predicate": ["influence"]},
    {"heads": ["human"], "tails": ["bacteria"], "predicate": ["change abundance"]},
    {"heads": ["human"], "tails": ["microbiome"], "predicate": ["change abundance"]},
    {"heads": ["DDF"], "tails": ["chemical"], "predicate": ["affect"]},
    {"heads": ["DDF"], "tails": ["dietary supplement"], "predicate": ["influence"]},
    {"heads": ["bacteria"], "tails": ["microbiome"], "predicate": ["produced by"]},
    {
        "heads": ["chemical"],
        "tails": ["anatomical location"],
        "predicate": ["produced by"],
    },
    {"heads": ["animal"], "tails": ["microbiome"], "predicate": ["produced by"]},
    {"heads": ["biomedical technique"], "tails": ["human"], "predicate": ["target"]},
    {"heads": ["drug"], "tails": ["chemical"], "predicate": ["influence"]},
    {"heads": ["dietary supplement"], "tails": ["DDF"], "predicate": ["is linked to"]},
    {"heads": ["anatomical location"], "tails": ["human"], "predicate": ["target"]},
    {"heads": ["DDF"], "tails": ["anatomical location"], "predicate": ["target"]},
    {"heads": ["gene"], "tails": ["chemical"], "predicate": ["influence"]},
    {"heads": ["chemical"], "tails": ["microbiome"], "predicate": ["located in"]},
    {"heads": ["DDF"], "tails": ["DDF"], "predicate": ["change effect"]},
    {"heads": ["drug"], "tails": ["animal"], "predicate": ["impact"]},
    {"heads": ["DDF"], "tails": ["gene"], "predicate": ["change abundance"]},
    {"heads": ["chemical"], "tails": ["DDF"], "predicate": ["affect"]},
    {"heads": ["bacteria"], "tails": ["DDF"], "predicate": ["change effect"]},
    {
        "heads": ["bacteria"],
        "tails": ["biomedical technique"],
        "predicate": ["used by"],
    },
    {"heads": ["bacteria"], "tails": ["animal"], "predicate": ["administered"]},
    {"heads": ["DDF"], "tails": ["DDF"], "predicate": ["is linked to"]},
    {"heads": ["DDF"], "tails": ["microbiome"], "predicate": ["target"]},
    {"heads": ["bacteria"], "tails": ["human"], "predicate": ["target"]},
    {"heads": ["human"], "tails": ["DDF"], "predicate": ["influence"]},
    {
        "heads": ["dietary supplement"],
        "tails": ["microbiome"],
        "predicate": ["influence"],
    },
    {"heads": ["chemical"], "tails": ["human"], "predicate": ["target"]},
    {"heads": ["human"], "tails": ["human"], "predicate": ["used by"]},
    {
        "heads": ["bacteria"],
        "tails": ["biomedical technique"],
        "predicate": ["is linked to"],
    },
    {"heads": ["DDF"], "tails": ["human"], "predicate": ["located in"]},
    {
        "heads": ["dietary supplement"],
        "tails": ["chemical"],
        "predicate": ["influence"],
    },
    {"heads": ["dietary supplement"], "tails": ["human"], "predicate": ["target"]},
    {"heads": ["drug"], "tails": ["microbiome"], "predicate": ["influence"]},
    {"heads": ["gene"], "tails": ["microbiome"], "predicate": ["influence"]},
    {"heads": ["statistical technique"], "tails": ["DDF"], "predicate": ["affect"]},
    {"heads": ["animal"], "tails": ["animal"], "predicate": ["used by"]},
    {"heads": ["chemical"], "tails": ["gene"], "predicate": ["influence"]},
    {"heads": ["DDF"], "tails": ["microbiome"], "predicate": ["strike"]},
    {"heads": ["microbiome"], "tails": ["chemical"], "predicate": ["used by"]},
    {"heads": ["food"], "tails": ["chemical"], "predicate": ["influence"]},
    {"heads": ["chemical"], "tails": ["microbiome"], "predicate": ["change abundance"]},
    {"heads": ["dietary supplement"], "tails": ["human"], "predicate": ["influence"]},
    {"heads": ["bacteria"], "tails": ["DDF"], "predicate": ["affect"]},
    {"heads": ["microbiome"], "tails": ["bacteria"], "predicate": ["impact"]},
    {"heads": ["drug"], "tails": ["microbiome"], "predicate": ["produced by"]},
    {"heads": ["DDF"], "tails": ["drug"], "predicate": ["affect"]},
    {"heads": ["chemical"], "tails": ["chemical"], "predicate": ["impact"]},
    {"heads": ["food"], "tails": ["anatomical location"], "predicate": ["located in"]},
    {"heads": ["drug"], "tails": ["chemical"], "predicate": ["change effect"]},
    {
        "heads": ["DDF"],
        "tails": ["anatomical location"],
        "predicate": ["change abundance"],
    },
    {"heads": ["anatomical location"], "tails": ["DDF"], "predicate": ["influence"]},
    {"heads": ["dietary supplement"], "tails": ["animal"], "predicate": ["influence"]},
    {"heads": ["DDF"], "tails": ["bacteria"], "predicate": ["strike"]},
    {"heads": ["chemical"], "tails": ["bacteria"], "predicate": ["located in"]},
    {"heads": ["anatomical location"], "tails": ["DDF"], "predicate": ["affect"]},
    {"heads": ["bacteria"], "tails": ["bacteria"], "predicate": ["impact"]},
    {"heads": ["microbiome"], "tails": ["animal"], "predicate": ["target"]},
    {"heads": ["bacteria"], "tails": ["microbiome"], "predicate": ["impact"]},
    {"heads": ["anatomical location"], "tails": ["animal"], "predicate": ["target"]},
    {
        "heads": ["dietary supplement"],
        "tails": ["microbiome"],
        "predicate": ["produced by"],
    },
    {"heads": ["microbiome"], "tails": ["microbiome"], "predicate": ["produced by"]},
    {"heads": ["microbiome"], "tails": ["DDF"], "predicate": ["used by"]},
    {"heads": ["animal"], "tails": ["animal"], "predicate": ["target"]},
    {"heads": ["chemical"], "tails": ["microbiome"], "predicate": ["is linked to"]},
    {
        "heads": ["dietary supplement"],
        "tails": ["bacteria"],
        "predicate": ["influence"],
    },
    {"heads": ["animal"], "tails": ["chemical"], "predicate": ["used by"]},
    {"heads": ["microbiome"], "tails": ["food"], "predicate": ["is linked to"]},
    {
        "heads": ["dietary supplement"],
        "tails": ["anatomical location"],
        "predicate": ["impact"],
    },
    {"heads": ["dietary supplement"], "tails": ["chemical"], "predicate": ["interact"]},
    {"heads": ["drug"], "tails": ["chemical"], "predicate": ["impact"]},
    {"heads": ["DDF"], "tails": ["human"], "predicate": ["is a"]},
    {"heads": ["DDF"], "tails": ["DDF"], "predicate": ["target"]},
    {
        "heads": ["biomedical technique"],
        "tails": ["chemical"],
        "predicate": ["influence"],
    },
    {"heads": ["bacteria"], "tails": ["bacteria"], "predicate": ["is linked to"]},
    {
        "heads": ["biomedical technique"],
        "tails": ["DDF"],
        "predicate": ["change effect"],
    },
    {
        "heads": ["biomedical technique"],
        "tails": ["animal"],
        "predicate": ["administered"],
    },
    {"heads": ["food"], "tails": ["DDF"], "predicate": ["change effect"]},
    {"heads": ["dietary supplement"], "tails": ["DDF"], "predicate": ["impact"]},
    {"heads": ["dietary supplement"], "tails": ["animal"], "predicate": ["target"]},
    {"heads": ["human"], "tails": ["drug"], "predicate": ["used by"]},
    {
        "heads": ["anatomical location"],
        "tails": ["anatomical location"],
        "predicate": ["located in"],
    },
    {"heads": ["chemical"], "tails": ["bacteria"], "predicate": ["is linked to"]},
    {"heads": ["biomedical technique"], "tails": ["animal"], "predicate": ["target"]},
    {"heads": ["microbiome"], "tails": ["DDF"], "predicate": ["located in"]},
]


def clean_label(label: str) -> str:
    if label == "DDF":
        return label
    return label.lower().strip()


def enumize_label(label: str) -> Enum:
    return label.replace(" ", "_").replace("-", "_").replace("/", "_")


class LabelLocation(Enum):
    ABSTRACT = "abstract"
    TEXT = "text"
    TITLE = "title"


def build_model(relations=relations):
    possible_links = {}
    for relation in relations:
        heads = [clean_label(head) for head in relation["heads"]]
        tails = [clean_label(tail) for tail in relation["tails"]]
        predicates = [clean_label(pred) for pred in relation["predicate"]]

        for head in heads:
            enum_head = enumize_label(head)
            for tail in tails:
                enum_tail = enumize_label(tail)
                for pred in predicates:
                    enum_pred = enumize_label(pred)
                    possible_links["_".join([enum_head, enum_pred, enum_tail])] = (
                        " | ".join([head, pred, tail])
                    )
    link_type = Enum("LinkType", possible_links)
    relation_type = create_model(
        "Relation",
        link_type=(link_type, ...),
        subject_text_span=(str, ...),
        subject_location=(LabelLocation, ...),
        object_text_span=(str, ...),
        object_location=(LabelLocation, ...),
    )
    relation_union = create_model("Relations", relations=(list[relation_type], ...))
    return relation_union


StringERLModel = build_model()
ExtendedStringERLModel = build_model(extended_relations + relations)


def build_enum_model(relations=relations):
    relation_models = []
    for relation in relations:
        heads = [clean_label(head) for head in relation["heads"]]
        tails = [clean_label(tail) for tail in relation["tails"]]
        predicates = [clean_label(pred) for pred in relation["predicate"]]

        enum_heads = Enum("Entity", {enumize_label(head): head for head in heads})
        enum_tails = Enum("Entity", {enumize_label(tail): tail for tail in tails})
        enum_predicates = Enum(
            "Predicate", {enumize_label(pred): pred for pred in predicates}
        )
        model_names = []
        for head in heads:
            for tail in tails:
                for pred in predicates:
                    model_names.append(
                        "_".join([enumize_label(lbl) for lbl in [head, tail, pred]])
                    )
        model_name = "_".join(model_names)
        relation_models.append(
            create_model(
                f"ERL_{model_name}",
                subject_label=(enum_heads, ...),
                subject_text_span=(str, ...),
                subject_location=(LabelLocation, ...),
                predicate=(enum_predicates, ...),
                object_label=(enum_tails, ...),
                object_text_span=(str, ...),
                object_location=(LabelLocation, ...),
            )
        )
    relation_type = Union[tuple(relation_models)]
    relation_union = create_model("Relations", relations=(list[relation_type], ...))
    return relation_union


EnumERLModel = build_enum_model()
ExtendedEnumERLModel = build_enum_model(extended_relations + relations)


def convert_to_enum_model(
    string_data: "StringERLModel", model=EnumERLModel
) -> "EnumERLModel":
    converted_relations = []
    for relation in string_data.relations:
        data = relation.model_dump()
        spo = relation.link_type.value.split(" | ")
        spo = [e.strip() for e in spo]
        data["subject_label"] = spo[0]
        data["predicate"] = spo[1]
        data["object_label"] = spo[2]
        converted_relations.append(data)
    return model.model_validate({"relations": converted_relations})


def convert_to_output(string_data: "StringERLModel", model=EnumERLModel):
    converted = convert_to_enum_model(string_data, model)
    converted_output_relations = Article(
        ternary_mention_based_relations=[], ternary_tag_based_relations=[]
    )
    for relation in converted.relations:
        tmbr = TernaryMentionBasedRelation(
            subject_text_span=relation.subject_text_span,
            subject_label=relation.subject_label.value,
            predicate=relation.predicate.value,
            object_text_span=relation.object_text_span,
            object_label=relation.object_label.value,
        )
        ttbr = TernaryTagBasedRelation(
            subject_label=relation.subject_label.value,
            predicate=relation.predicate.value,
            object_label=relation.object_label.value,
        )
        converted_output_relations.ternary_tag_based_relations.append(ttbr)
        converted_output_relations.ternary_mention_based_relations.append(tmbr)
    return converted_output_relations


def convert_to_string_model(
    enum_data: "EnumERLModel", model=StringERLModel
) -> "StringERLModel":
    converted_relations = []
    for relation in enum_data.relations:
        spo = relation.subject_label, relation.predicate, relation.object_label
        spo_values = [e.value for e in spo]
        data = relation.dict()
        data["link_type"] = " | ".join(spo_values)
        converted_relations.append(data)
    return model.model_validate({"relations": converted_relations})


def build_grammar():
    rel_model = build_model()
    return generate_gbnf_grammar_from_pydantic_models([rel_model])
