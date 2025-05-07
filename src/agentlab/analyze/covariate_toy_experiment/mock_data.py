from dataclasses import dataclass
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@dataclass
class Task:
    difficulty: float
    type: int = None
    uuid: str = None

    def __post_init__(self):
        self.uuid = str(uuid4())


class Agent:

    def __init__(
        self,
        competence: float,
        benchmark: list[Task],
        type: int = None,
        consistancy: float = 10,
        rng: np.random.RandomState = np.random,
    ):
        self.competence = competence
        self.type = type
        self.task_success_rate = {}
        fit_count = 0
        for task in benchmark:

            agent_task_competence = competence
            if task.type is not None and type is not None:
                if task.type != type:
                    agent_task_competence = competence * 0.5
                else:
                    fit_count += 1

            # task_competence = agent_task_competence * (1.001 - task.difficulty)
            task_success_rate = sigmoid(consistancy * (agent_task_competence - task.difficulty))

            self.task_success_rate[task.uuid] = task_success_rate
        self.fit_ratio = fit_count / len(benchmark)

    def get_task_success_rate(self, task: Task):
        return self.task_success_rate[task.uuid]

    def get_success_rate(self):
        return np.mean(list(self.task_success_rate.values()))

    def __str__(self):

        return f"Agent(competence={self.competence:.3f}, type={self.type}, success_rate={self.get_success_rate():.3f}, fit_ratio={self.fit_ratio:.3f})"


def agent_on_benchmark(
    agent: Agent,
    benchmark: list[Task],
    n_samples_per_task=None,
    rng: np.random.RandomState = np.random,
):

    all_rewards = []
    for task in benchmark:
        task_success_rate = agent.get_task_success_rate(task)

        # sample n_samples_per_task from bernoulli distribution
        rewards = rng.binomial(1, task_success_rate, n_samples_per_task)

        all_rewards.append(rewards)
    return np.array(all_rewards)


def plot_task_difficulty(difficulties):
    """
    Plot the difficulty of each task in the benchmark.
    """

    plt.hist(difficulties, bins=20)
    plt.xlabel("Task Difficulty")
    plt.ylabel("Frequency")
    plt.title("Distribution of Task Difficulty")


def plot_gaussian(mu, sigma, label=None):
    """
    Plot a Gaussian distribution with mean mu and standard deviation sigma.
    """
    x = np.linspace(0, 1, 1000)
    plt.plot(
        x, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2), label=label
    )


def _augment_with_average(matrix: np.ndarray) -> np.ndarray:
    """Return a new array with row averages as the last column,
    column averages as the last row, and the overall average in the bottom-right."""
    row_avg = matrix.mean(axis=1)
    col_avg = matrix.mean(axis=0)
    overall_avg = matrix.mean()

    # sort columns by their average
    sorted_indices = np.argsort(col_avg)
    matrix = matrix[:, sorted_indices]
    col_avg = col_avg[sorted_indices]

    # sort rows by their average
    sorted_indices = np.argsort(row_avg)
    matrix = matrix[sorted_indices, :]
    row_avg = row_avg[sorted_indices]

    aug = np.zeros((matrix.shape[0] + 1, matrix.shape[1] + 1))
    aug[:-1, :-1] = matrix
    aug[:-1, -1] = row_avg
    aug[-1, :-1] = col_avg
    aug[-1, -1] = overall_avg

    return aug
