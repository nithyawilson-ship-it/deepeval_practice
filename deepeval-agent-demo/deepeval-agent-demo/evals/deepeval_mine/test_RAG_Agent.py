import sys, os

from deepeval.test_case import SingleTurnParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath( __file__ ) ) ) ) )
from deepeval.contextvars import get_current_golden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, AnswerRelevancyMetric, \
    FaithfulnessMetric, GEval, PIILeakageMetric, ToxicityMetric
from deepeval.models import AnthropicModel
from deepeval.tracing import observe, update_current_trace

from rag_agent import rag_support_agent as _rag_support_agent

@observe(name = "rag_support_agent")
def rag_support_agent(user_input: str) -> str:
        golden = get_current_golden()
        if golden:
            if golden.expected_output:
                update_current_trace(expected_output=golden.expected_output)
        return _rag_support_agent(user_input)

Dataset = EvaluationDataset(goldens=[
    Golden(
        input = "What is the return policy for electronics?",
        expected_output= "Electronics can be returned within 15 days of delivery if unopened"
                       "and in original packaging. Refund takes 5 - 7 business days."
    ),
    Golden(
        input = "How long does express shipping take and what does it cost?",
        expected_output= "Express shipping takes 1-2 business days and costs $15."
                        "Orders placed before 2 PM are dispatched on the same day."
    )
])

anthropic_model = AnthropicModel(model="claude-sonnet-4-6")
correctness= GEval(
    name ="Correctness",
    criteria = (
        "Determine whether the actual output conveys the same factual information"
        "as the expected output. Minor wording differences are acceptable;"
        "missing or wrong facts are not."
    ),
    model=anthropic_model,
    threshold=0.8,
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
        SingleTurnParams.ACTUAL_OUTPUT
    ]
)
contextual_precision_metrics = ContextualPrecisionMetric(threshold=0.7, model=anthropic_model)
contextual_recall_metrics = ContextualRecallMetric(threshold=0.7, model=anthropic_model)
answer_relevancy_metrics = AnswerRelevancyMetric(threshold=0.7, model=anthropic_model)
faithfulness_metrics = FaithfulnessMetric(threshold=0.7, model=anthropic_model)
pileakage_metrics = PIILeakageMetric(threshold=0.7, model=anthropic_model)
toxicity_metrics = ToxicityMetric(threshold=0.7,model=anthropic_model)

for golden in Dataset.evals_iterator(metrics=[contextual_precision_metrics,contextual_recall_metrics,answer_relevancy_metrics,faithfulness_metrics,correctness,pileakage_metrics,toxicity_metrics]):
    rag_support_agent(golden.input)
