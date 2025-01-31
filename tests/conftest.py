import pytest

@pytest.fixture(autouse=True)
def enable_db(db):
    # Added a comment to comply with atomic line length
    pass
