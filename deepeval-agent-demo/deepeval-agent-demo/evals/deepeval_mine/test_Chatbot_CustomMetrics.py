import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath( __file__ ) ) ) ) )

from deepeval.metrics import ConversationalGEval
from deepeval.evaluate import evaluate
from deepeval.models import AnthropicModel
from deepeval.test_case import Turn, ConversationalTestCase, MultiTurnParams
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

correctness= ConversationalGEval(
    name ="Correctness",
    criteria = (
        "Did the chatbot fully resolve the customer's issue"
        "It should use tools when needed and provide accurate answers."
    ),
    model=anthropic_model,
    threshold=0.8,
    evaluation_params=[
        MultiTurnParams.ROLE,
        MultiTurnParams.CONTENT
    ]
)



test_case = ConversationalTestCase(
    turns = turns
)

evaluate(test_cases= [test_case], metrics = [correctness])