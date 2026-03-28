from typing import List, Optional
from datetime import datetime
from typing import Optional, List
from datetime import datetime
from agentlab.traces.trace_utils import update_index
from traces import upload_trace,upload_study

class Experiment:
    """Represents a single experiment with relevant metadata."""
    
    def __init__(
        self, 
        exp_id: str, 
        study_id: str, 
        name: str,
        llm: str, 
        benchmark: str, 
        license: str, 
        dir: str,
            ):

        self.exp_id = exp_id
        self.study_id = study_id
        self.name = name
        self.llm = llm
        self.benchmark = benchmark
        self.license = license
        self.dir = dir
        self.timestamp = datetime.now().isoformat()

    def __repr__(self):
        return (
            f"Experiment(exp_id={self.exp_id}, study_id={self.study_id}, "
            f"name={self.name}, llm={self.llm}, benchmark={self.benchmark}, "
            
        )


class Study:
    """Represents a study containing multiple experiments."""
    

    def __init__(self, study_id: str, study_name: str, description: str, experiments: List[Experiment]):
        self.study_id = study_id
        self.study_name = study_name
        self.description = description
        self.experiments = experiments

    def add_experiment(self, experiment: Experiment) -> None:
        """Add an experiment to the study."""
        self.experiments.append(experiment)

    def remove_experiment(self, experiment_id: str) -> None:
        """Remove an experiment from the study by ID."""
        self.experiments = [exp for exp in self.experiments if exp.exp_id != experiment_id]

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Retrieve an experiment by ID."""
        for experiment in self.experiments:
            if experiment.exp_id == experiment_id:
                return experiment
        return None
    
    def upload(self) -> None:
            """Upload all experiment traces in the study to Hugging Face."""
        self.upload_Study()
        for exp in self.experiments:
            trace_pointer = upload_trace(exp.exp_id,exp.dir,exp.benchmark)
            # Assign a license based on LLM and benchmark
            LICENSES = {
            ("GPT-4", "benchmark1"): "MIT",
            ("Llama2", "benchmark2"): "Apache-2.0",
             }
            license_type = LICENSES.get((exp.exp_llm, exp.exp_benchmark), "Unknown")
            update_index(exp.exp_id,self.study_id,exp.llm,exp.benchmark,license_type,trace_pointer)
         
    def upload_Study(self):
        """Upload study to the study dataset"""
        study_data = {
            "study_id": [self.study_id],
            "study_name": [self.study_name],
            "description": [self.description],
        }
        upload_study(study_data)

    def __repr__(self):
        return f"Study(id={self.study_id}, name={self.study_name}, experiments={len(self.experiments)})"
