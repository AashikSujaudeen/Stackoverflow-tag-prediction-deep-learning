from pydantic import BaseModel

class UserInput(BaseModel):
    customInputFlag: str
    questions: str
    questionCount: int
