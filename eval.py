import torch
from dataclasses import dataclass
from typing import Any, Callable
import click
import pandas as pd 
from contextlib import contextmanager
import triton


@dataclass
class OperatorTestCase:
    op: str
    overload: str
    correctness_tests: list[Any]
    performance_tests: list[Any]


class Backend:
    @contextmanager
    def activate(self):
        yield


class AtenBackend(Backend):
    name: str = "aten"

    @contextmanager
    def activate(self):
        yield


def run_correctness_test(backend: Backend, op: OperatorTestCase, testcase: str):
    args, kwargs = eval(testcase)
    op_fn = getattr(getattr(torch.ops.aten, op.op), op.overload)
    ref = op_fn(*args, **kwargs)
    with backend.activate():
        res = op_fn(*args, **kwargs)
    return torch.allclose(ref, res)


def run_performance_test(backend: Backend, op: OperatorTestCase, testcase: str):
    args, kwargs = eval(testcase)
    op_fn = getattr(getattr(torch.ops.aten, op.op), op.overload)
    with backend.activate():
        return triton.testing.do_bench(lambda: op_fn(*args, **kwargs))


OPS = [
    OperatorTestCase(
        "relu",
        "default",
        [
            "((torch.randn(1, 1),), {})",
        ],
        [
            "((torch.randn(2**20),), {})",
        ],
    )
]

BACKENDS = [
    AtenBackend(),
]

@click.command()
@click.option('--ops', default=None, help='Comma-separated list of operations to test.')
@click.option('--backends', default=None, help='Comma-separated list of backends to use.')
def main(ops, backends):
    """
    For each backend, check every op for correctness and performance.  Record those results.  Produce a report.  I guess to really produce a score, we need to compare against a baseline.  Maybe we can do that by comparing against the default backend.
    Sort of a hierarchical structure for the data:
    - Backend
    - Op
        - Overload
        - Correctness test results
        - Performance test results
    But for ease of analysis we actually kind of want to flatten that out:
    Backend, Op, Overload, Correctness test case, Correctness result, Performance test case, Performance result

    Yeah.  And then we can turn that into a dataframe and do whatever we want with it.
    """
    if ops:
        ops = ops.split(',')
    else:
        ops = OPS  # Default operation

    if backends:
        backends = backends.split(',')
    else:
        backends = ["default"]  # Default backend

    print(f"Testing operations: {ops}")
    print(f"Using backends: {backends}")

    # For each backend, for each op, run the correctness and performance tests and record the results in a pandas dataframe.

    # Initialize an empty list to store results
    correctness_results = []
    performance_results = []

    # Iterate over each backend and operation
    for backend in BACKENDS:
        for op in OPS:
            for correctness_test in op.correctness_tests:
                correctness_result = run_correctness_test(backend, op, correctness_test)
                correctness_results.append({
                    "backend": backend.name,
                    "op": op.op,
                    "overload": op.overload,
                    "test": correctness_test,
                    "result": correctness_result,
                })
            for performance_test in op.performance_tests:
                performance_result = run_performance_test(backend, op, performance_test)
                performance_results.append({
                    "backend": backend.name,
                    "op": op.op,
                    "overload": op.overload,
                    "test": performance_test,
                    "result": performance_result,
                })


    # Convert the results into a pandas DataFrame
    correctness_df = pd.DataFrame(correctness_results)
    performance_df = pd.DataFrame(performance_results)

    print(correctness_df)
    print(performance_df)




if __name__ == "__main__":
    main()
