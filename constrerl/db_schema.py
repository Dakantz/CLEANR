from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import DeclarativeBase, relationship, mapped_column, Mapped


class Base(DeclarativeBase):
    pass


N_DIMS = 1024


class RelDocument(Base):
    __tablename__ = "rel_document"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column()
    abstract: Mapped[str] = mapped_column()
    vectors: Mapped[Vector] = mapped_column(Vector(N_DIMS))
    annon_type: Mapped[str] = mapped_column(nullable=True)
    doc_meta: Mapped[str] = mapped_column()
    collection: Mapped[str] = mapped_column()
