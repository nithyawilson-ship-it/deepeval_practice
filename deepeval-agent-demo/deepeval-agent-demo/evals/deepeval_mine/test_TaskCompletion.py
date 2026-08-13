import sys, os

from langchain_core.callbacks import file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath( __file__ ) ) ) )
from deepeval.evaluate import evaluate
from deepeval.models import AnthropicModel
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase

from agent_instrumented import support_agent


actual_output = support_agent("where is my order ORD-1042?")
test_case = LLMTestCase(
    input = "where is my order ORD-1042?",
    actual_output = actual_output
)
anthropic_model = AnthropicModel(model="claude-sonnet-4-6")
evaluate(test_cases = [test_case],
        metrics= [TaskCompletionMetric(threshold=0.7,model=anthropic_model)]
)