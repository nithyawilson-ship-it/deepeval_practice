from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import OllamaModel

# Point directly at your local Ollama model
local_model = OllamaModel(
    model="qwen2.5:latest",
    base_url="http://localhost:11434"
)

def test_with_context():
    question = "What is our refund policy?"
    answer = "You can get a refund within 90 days of purchase."
    context = ["Our company allows refunds within 30 days of the original purchase date."]

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=context
    )

    relevancy = AnswerRelevancyMetric(threshold=0.7, model=local_model)
    faithfulness = FaithfulnessMetric(threshold=0.7, model=local_model)

    assert_test(test_case, [relevancy, faithfulness])