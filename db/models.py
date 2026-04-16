import uuid
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Household(Base):
    __tablename__ = "households"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    users: Mapped[list["User"]] = relationship(back_populates="household")
    categories: Mapped[list["Category"]] = relationship(back_populates="household")


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    household_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("households.id"), nullable=False
    )
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    person_code: Mapped[str] = mapped_column(String(1), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    household: Mapped["Household"] = relationship(back_populates="users")


class Category(Base):
    __tablename__ = "categories"
    __table_args__ = (
        UniqueConstraint("household_id", "name", name="uq_category_household_name"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    household_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("households.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    slug: Mapped[str] = mapped_column(Text, nullable=False)
    category_type: Mapped[str] = mapped_column(Text, nullable=False)
    household_tier: Mapped[str] = mapped_column(Text, nullable=False)
    tax_context: Mapped[str] = mapped_column(Text, nullable=False)
    default_split: Mapped[str] = mapped_column(Text, nullable=False)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False)

    household: Mapped["Household"] = relationship(back_populates="categories")


class BudgetEnvelope(Base):
    __tablename__ = "budget_envelopes"
    __table_args__ = (
        UniqueConstraint(
            "category_id", "user_id", "period_year",
            name="uq_envelope_category_user_year",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    household_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("households.id"), nullable=False
    )
    category_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("categories.id"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False
    )
    period_year: Mapped[int] = mapped_column(Integer, nullable=False)
    amount_cad: Mapped[str] = mapped_column(Numeric(18, 4), nullable=False)

    category: Mapped["Category"] = relationship()
    user: Mapped["User"] = relationship()


class Statement(Base):
    __tablename__ = "statements"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    household_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("households.id"), nullable=False
    )
    issuer: Mapped[str] = mapped_column(Text, nullable=False)
    period_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    period_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    layout_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    parsed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    parse_status: Mapped[str] = mapped_column(Text, nullable=False)


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    household_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("households.id"), nullable=False
    )
    statement_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("statements.id"), nullable=True
    )
    cash_date: Mapped[date] = mapped_column(Date, nullable=False)
    accounting_period_year: Mapped[int] = mapped_column(Integer, nullable=False)
    accounting_period_month: Mapped[int] = mapped_column(Integer, nullable=False)
    accounting_period_is_override: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_description: Mapped[str] = mapped_column(Text, nullable=False)
    original_amount: Mapped[str] = mapped_column(Numeric(18, 4), nullable=False)
    original_currency: Mapped[str] = mapped_column(Text, nullable=False)
    fx_rate: Mapped[str | None] = mapped_column(Numeric(12, 6), nullable=True)
    fx_rate_source: Mapped[str | None] = mapped_column(Text, nullable=True)
    cad_amount: Mapped[str | None] = mapped_column(Numeric(18, 4), nullable=True)
    category_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("categories.id"), nullable=False
    )
    split_method: Mapped[str] = mapped_column(Text, nullable=False)
    andrew_amount: Mapped[str | None] = mapped_column(Numeric(18, 4), nullable=True)
    kristy_amount: Mapped[str | None] = mapped_column(Numeric(18, 4), nullable=True)
    classifier_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    classifier_source: Mapped[str | None] = mapped_column(Text, nullable=True)
    needs_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_manually_reviewed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    source_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    category: Mapped["Category"] = relationship()
    statement: Mapped["Statement | None"] = relationship()


class Correction(Base):
    __tablename__ = "corrections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    household_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("households.id"), nullable=False
    )
    transaction_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("transactions.id"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False
    )
    prev_category_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("categories.id"), nullable=True
    )
    new_category_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("categories.id"), nullable=False
    )
    prev_split_method: Mapped[str | None] = mapped_column(Text, nullable=True)
    new_split_method: Mapped[str] = mapped_column(Text, nullable=False)
    corrected_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    transaction: Mapped["Transaction"] = relationship()


class ExactMatchCache(Base):
    __tablename__ = "exact_match_cache"
    __table_args__ = (
        UniqueConstraint(
            "household_id", "normalized_merchant",
            name="uq_cache_household_merchant",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    household_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("households.id"), nullable=False
    )
    normalized_merchant: Mapped[str] = mapped_column(Text, nullable=False)
    category_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("categories.id"), nullable=False
    )
    responsibility: Mapped[str] = mapped_column(Text, nullable=False)
    confirmation_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1
    )
    last_confirmed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    category: Mapped["Category"] = relationship()


class ParseError(Base):
    __tablename__ = "parse_errors"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    statement_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("statements.id"), nullable=False
    )
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str] = mapped_column(Text, nullable=False)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    statement: Mapped["Statement"] = relationship()
