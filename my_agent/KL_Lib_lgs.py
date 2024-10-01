from typing import Annotated, Literal, Union, Optional, List, Any

from typing_extensions import TypedDict


def append_object(list_obj: List[Any], new_objects: List[Any]) -> List[Any]:
    """
    A custom function to append any objects to the list.
    This function can be used to add any type of objects to a list,
    allowing for greater flexibility.
    """
    if type(new_objects) != list:
        new_objects = [new_objects]

    return list_obj + new_objects

from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError, HttpUrl # , validator
class ResponderWithRetries:
    def __init__(self, runnable, validator, retry_limit=3):                                            
        self.runnable = runnable
        self.validator = validator
        self.retry_limit = retry_limit

    def respond(self, state: list):
        response = []
        for attempt in range(self.retry_limit):
            response = self.runnable.invoke(
                {"messages": state}, {"tags": [f"attempt:{attempt}"]}
            )
            try:
                if self.validator:                   
                    self.validator.invoke(response)
                    return response
                else:
                    return response
            except ValidationError as e:
                state = state + [
                    response,
                    ToolMessage(
                        content=f"{repr(e)}\n\nPay close attention to the function schema.\n\n"
                        + self.validator.schema_json()
                        + " Respond by fixing all validation errors.",
                        tool_call_id=response.tool_calls[0]["id"],
                    ),
                ]
        return response
