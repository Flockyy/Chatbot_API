# Import all the models, so that Base has them before being
# imported by Alembic
from src.db.base_class import Base  # noqa
from src.models.question import Question  # noqa
from src.models.answer import Answer  # noqa
from src.models.chat import Chat  # noqa
