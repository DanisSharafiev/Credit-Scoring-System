from sqlmodel import SQLModel, create_engine, Session
from typing import Optional

username = "username"
password = "password"
dbname = "dbname"

DATABASE_URL = f"postgresql://{username}:{password}@localhost:5432/{dbname}"

engine = create_engine(DATABASE_URL)

class Request(SQLModel, table=True):
    id: Optional[int] = None
    CODE_GENDER: str
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: float
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    FLAG_WORK_PHONE: int
    FLAG_PHONE: int
    FLAG_EMAIL: int
    OCCUPATION_TYPE: str
    CNT_FAM_MEMBERS: int
    DECISION: int

def init_db():
    SQLModel.metadata.create_all(engine)

def get_db():
    with Session(engine) as session:
        yield session