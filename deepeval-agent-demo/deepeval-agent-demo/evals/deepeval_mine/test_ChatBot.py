import sys, os

from deepeval.models import AnthropicModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath( __file__ ) ) ) ) )

from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.metrics import TurnRelevancyMetric, KnowledgeRetentionMetric, ConversationCompletenessMetric
from deepeval.evaluate import evaluate
from chatbot import chat


turns = []
history = []
for user_msg in [
    "Hi I placed an order las week, the order ID is ORD-1042.",
    "Is it going to arrive on time?",
    "what was the ETA you just mentioned?", #testing memory retention
    "Can I upgrade to express shipping?"
]:
    reply,history,_ = chat(user_msg,history)
    turns.append(Turn(role = "user", content = user_msg))
    turns.append(Turn(role = "assistant", content = reply))

anthropic_model = AnthropicModel(model="claude-sonnet-4-6")
turn_relevancy_metrics = TurnRelevancyMetric(threshold=0.5,model=anthropic_model)
retention_metrics = KnowledgeRetentionMetric(threshold=0.5,model=anthropic_model)
completeness_metrics = ConversationCompletenessMetric(threshold=0.5,model=anthropic_model)

test_case = ConversationalTestCase(
    turns = turns
)

evaluate(test_cases = [test_case], metrics = [turn_relevancy_metrics, retention_metrics,completeness_metrics])