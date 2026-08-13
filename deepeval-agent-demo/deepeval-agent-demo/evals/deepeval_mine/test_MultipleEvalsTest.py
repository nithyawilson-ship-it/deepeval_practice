import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath( __file__ ) ) ) ) )

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import PromptAlignmentMetric, StepEfficiencyMetric, AnswerRelevancyMetric
from deepeval.models import AnthropicModel
from deepeval.tracing import observe

from agent_instrumented import support_agent as _support_agent



#Trace have actual values and expected values
@observe(name="support_agent")
def support_agent(user_input: str):
    return _support_agent(user_input)

#Assign object for anthropic model
anthropic_model = AnthropicModel(model="claude-sonnet-4-6")
#Test Case 1: Prompt Alignment Metric
prompt_alignment_metrics = PromptAlignmentMetric(
    prompt_instructions=[
        "You are a friendly customer-support agent. "
        "Use the available tools to answer order and refund questions. "
        "Keep replies short and helpful."
    ], threshold=0.7,model=anthropic_model
)
#Test Case 2: Step Efficiency Metric
step_efficiency_metrics = StepEfficiencyMetric(threshold=0.5,model=anthropic_model)

#Test Case 3: Step Efficiency Metric
answer_relevancy_metrics = AnswerRelevancyMetric(threshold=0.7,model=anthropic_model)

#Test Data
dataset = EvaluationDataset(goldens=[
    Golden(input="Where is my order ORD-1042?"),
    Golden(input="What is the refund policy for electronics?"),
    Golden(input="I wabt to return order ORD-1099, what should I do?"),
])

for golden in dataset.evals_iterator(metrics=[prompt_alignment_metrics, step_efficiency_metrics, answer_relevancy_metrics]):
    support_agent(golden.input)