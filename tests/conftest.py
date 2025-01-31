import pytest

@pytest.fixture(autouse=True)
def enable_db(db):
    # Fixture to enable database
    pass
