import pytest

@pytest.fixture(autouse=True)
def enable_db(db):
    pass
