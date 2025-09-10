
from dotenv import load_dotenv
import argparse

load_dotenv()

import logging
import argparse

from agentlab.agents.generic_agent_hinter.generic_agent import GenericAgentArgs
from agentlab.agents.generic_agent_hinter.agent_configs import CHAT_MODEL_ARGS_DICT, FLAGS_GPT_4o
from bgym import DEFAULT_BENCHMARKS
from agentlab.experiments.study import Study

logging.getLogger().setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--llm-config", required=True)
    parser.add_argument("--relaunch", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--parallel-backend", type=str, default="ray")
    parser.add_argument("--reproducibility-mode", action="store_true")
    # hint flags
    parser.add_argument("--hint-type", type=str, default="docs")
    parser.add_argument("--hint-index-type", type=str, default="sparse")
    parser.add_argument("--hint-query-type", type=str, default="direct")
    parser.add_argument("--hint-index-path", type=str, default="indexes/servicenow-docs-bm25")
    parser.add_argument("--hint-retriever-path", type=str, default="google/embeddinggemma-300m")
    parser.add_argument("--hint-num-results", type=int, default=5)
    args = parser.parse_args()

    flags = FLAGS_GPT_4o
    flags.use_task_hint = True
    flags.hint_type = args.hint_type
    flags.hint_index_type = args.hint_index_type
    flags.hint_query_type = args.hint_query_type
    flags.hint_index_path = args.hint_index_path
    flags.hint_retriever_path = args.hint_retriever_path
    flags.hint_num_results = args.hint_num_results

    # instantiate agent
    agent_args = [GenericAgentArgs(
        chat_model_args=CHAT_MODEL_ARGS_DICT[args.llm_config],
        flags=flags,
    )]
    
    benchmark = DEFAULT_BENCHMARKS[args.benchmark]()


    if args.relaunch:
        #  relaunch an existing study
        study = Study.load_most_recent(contains=None)
        study.find_incomplete(include_errors=True)

    else:
        study = Study(
            agent_args,
            benchmark,
            logging_level=logging.WARNING,
            logging_level_stdout=logging.WARNING,
        )
        
    study.run(
        n_jobs=args.n_jobs,
        parallel_backend=args.parallel_backend,
        strict_reproducibility=args.reproducibility_mode,
        n_relaunch=1,
    )

        

if __name__ == "__main__":
    main()