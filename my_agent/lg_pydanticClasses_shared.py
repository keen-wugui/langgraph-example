
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError, HttpUrl
from pydantic.v1 import validator


class singleTopic(BaseModel):
    topic_number: int = Field(description="The number of a subtopic")
    topic_subheading: str = Field(description="The content of a subtopic")
    topic_description: str = Field(description="The 100-word description of a subtopic") 

    def access_as_text(self):
        return f"**{self.topic_subheading}** \n {self.topic_description} \n\n" 

class singleTopicResearchRequest(singleTopic):
    research_is_requested: bool = Field(description="Whether research is requested for the subtopic")
    reason_for_research: str = Field(description="The reason for researching the subtopic")

class researchReference(BaseModel):
    reference_content: str = Field(description="A reference for a subtopic")
    reference_source: HttpUrl = Field(description="The source URL or citation of the reference")

class singleTopicWithReferences(BaseModel):
    topic_number: int = Field(description="The number of a subtopic")
    topic_subheading: str = Field(description="The content of a subtopic")
    topic_description: str = Field(description="The description of a subtopic") 
    references: list[researchReference] = Field(description="A list of references for the subtopic")

    def access_as_text(self):
        return f"**{self.topic_subheading}** \n {self.topic_description} \n\n" 
    
    def save_to_blob(self,dbName="graph", collectionName="blobs"):
        from lg_loader_inline import lg_loader_inline   
        return lg_loader_inline(docs=[self.access_as_text()],dbName=dbName, collectionName=collectionName)

class essay(BaseModel):
    essay_content: str = Field(description="The content of the essay")
    writer_questions: list[str] = Field(description="Questions and assumptions collected from the writer")

