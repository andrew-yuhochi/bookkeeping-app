"""Seed the database with the default household, users, and categories from config/categories.yaml.

Idempotent: running multiple times will not create duplicate rows.
Uses ON CONFLICT DO NOTHING logic via check-before-insert.
"""

import logging
import sys
from pathlib import Path

import yaml
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

# Add project root to path so db package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.models import Base, Category, Household, User  # noqa: E402
from src.config import settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "categories.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def seed_household(session: Session, config: dict) -> Household:
    """Create or retrieve the default household."""
    household_name = config["household"]["name"]
    existing = session.execute(
        select(Household).where(Household.name == household_name)
    ).scalar_one_or_none()

    if existing:
        logger.info("Household '%s' already exists (id=%s)", household_name, existing.id)
        return existing

    household = Household(id=settings.household_id, name=household_name)
    session.add(household)
    session.flush()
    logger.info("Created household '%s' (id=%s)", household_name, household.id)
    return household


def seed_users(session: Session, config: dict, household: Household) -> list[User]:
    """Create or retrieve users for the household."""
    users = []
    for user_cfg in config["users"]:
        existing = session.execute(
            select(User).where(
                User.household_id == household.id,
                User.person_code == user_cfg["person_code"],
            )
        ).scalar_one_or_none()

        if existing:
            logger.info(
                "User '%s' (code=%s) already exists",
                existing.display_name,
                existing.person_code,
            )
            users.append(existing)
            continue

        user = User(
            household_id=household.id,
            display_name=user_cfg["display_name"],
            person_code=user_cfg["person_code"],
        )
        session.add(user)
        session.flush()
        logger.info("Created user '%s' (code=%s, id=%s)", user.display_name, user.person_code, user.id)
        users.append(user)

    return users


def seed_categories(session: Session, config: dict, household: Household) -> list[Category]:
    """Create or retrieve all expense + income categories."""
    categories = []
    all_cats = [
        (cat, "expense") for cat in config["expense_categories"]
    ] + [
        (cat, "income") for cat in config["income_categories"]
    ]

    for cat_cfg, cat_type in all_cats:
        existing = session.execute(
            select(Category).where(
                Category.household_id == household.id,
                Category.name == cat_cfg["name"],
            )
        ).scalar_one_or_none()

        if existing:
            logger.info("Category '%s' already exists", existing.name)
            categories.append(existing)
            continue

        category = Category(
            household_id=household.id,
            name=cat_cfg["name"],
            slug=cat_cfg["slug"],
            category_type=cat_type,
            household_tier=cat_cfg["household_tier"],
            tax_context=cat_cfg["tax_context"],
            default_split=cat_cfg["default_split"],
            sort_order=cat_cfg["sort_order"],
        )
        session.add(category)
        session.flush()
        logger.info(
            "Created category '%s' (type=%s, tier=%s, sort=%d)",
            category.name,
            cat_type,
            category.household_tier,
            category.sort_order,
        )
        categories.append(category)

    return categories


def main() -> None:
    logger.info("Loading config from %s", CONFIG_PATH)
    config = load_config()

    engine = create_engine(settings.database_url, echo=False)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        household = seed_household(session, config)
        users = seed_users(session, config, household)
        categories = seed_categories(session, config, household)
        session.commit()

    logger.info(
        "Seed complete: 1 household, %d users, %d categories (%d expense + %d income)",
        len(users),
        len(categories),
        len(config["expense_categories"]),
        len(config["income_categories"]),
    )


if __name__ == "__main__":
    main()
