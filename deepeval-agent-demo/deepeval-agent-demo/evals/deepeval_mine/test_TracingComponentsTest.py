import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath( __file__ ) ) ) )
from deepeval.contextvars import get_current_golden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.tracing import observe, update_current_trace
from deepeval.test_case import ToolCall
from deepeval.models import AnthropicModel

from agent_instrumented import support_agent as _support_agent

#Trace have actual values and expected values
@observe(name="support_agent")
def support_agent(user_input: str):
    golden = get_current_golden()
    if golden:
        if golden.expected_tools:
            update_current_trace(expected_tools=golden.expected_tools)
        if golden.expected_output:
            update_current_trace(expected_output=golden.expected_output)
    return _support_agent(user_input)


anthropic_model = AnthropicModel(model="claude-sonnet-4-6")
task_completion_metric = TaskCompletionMetric(threshold=0.7,model=anthropic_model)
tool_correctness_metric = ToolCorrectnessMetric(model=anthropic_model)
dataset = EvaluationDataset(goldens = [
    Golden(input= "where is my order ORD-1042?",
           expected_tools=[ToolCall(name= "get_order_status")]),
    Golden(input= "what is the refund for electronics?",
           expected_tools=[ToolCall(name= "get_refund_policy")])
])
metrics=[task_completion_metric]
for golden in dataset.evals_iterator(metrics = [task_completion_metric, tool_correctness_metric]):
    support_agent(golden.input)