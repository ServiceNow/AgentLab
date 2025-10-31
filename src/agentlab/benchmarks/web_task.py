
from pydantic import BaseModel


class AbstractWebTask(BaseModel):
    name: str
    validate_per_step: bool = False
    
    def get_setup_js(self) -> str:
        raise NotImplementedError

    def get_teardown_js(self) -> str:
        raise NotImplementedError

    def get_task_validate_js(self) -> str:
        raise NotImplementedError

    def get_step_validate_js(self) -> str:
        raise NotImplementedError

    def parse_validation_result(self, validate_result: str) -> tuple[float, dict]:
        raise NotImplementedError
