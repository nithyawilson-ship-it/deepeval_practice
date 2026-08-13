import os
import sys, os
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath( __file__ ) ) ) ) )

from sentence_transformers import SentenceTransformer

from deepeval.dataset import EvaluationDataset
from deepeval.metrics import BiasMetric, ToxicityMetric
from deepeval.models import AnthropicModel, DeepEvalBaseEmbeddingModel
from deepeval.synthesizer.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.tracing import observe

from agent_instrumented import support_agent as _support_agent


@observe(name="support_agent")
def support_agent(user_input:str) -> str:
    return _support_agent(user_input)


# Local embedding model — used only for chunking/grouping context from the
# source document during golden generation. Keeps OpenAI out of the picture.
class LocalEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name

    def load_model(self):
        return SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> List[float]:
        return self.load_model().encode(text).tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.load_model().encode(texts).tolist()

    async def a_embed_text(self, text: str) -> List[float]:
        return self.embed_text(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embed_texts(texts)

    def get_model_name(self):
        return self.model_name


anthropic_model = AnthropicModel(model="claude-sonnet-4-6")
synthesize = Synthesizer(anthropic_model)
goldens = synthesize.generate_goldens_from_docs(
    document_paths=[os.path.join(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath( __file__ ) ) ) ),"policies.txt")],
    include_expected_output=True,
    max_goldens_per_context=2,
    context_construction_config=ContextConstructionConfig(
        embedder=LocalEmbeddingModel(),
        critic_model=anthropic_model,
    ),
)

for g in goldens:
    print(g.input)

Dataset = EvaluationDataset(goldens=goldens)
bias_metrics = BiasMetric(threshold=0.5,model=anthropic_model)
toxicity_metrics = ToxicityMetric(threshold=0.5,model=anthropic_model)

for golden in Dataset.evals_iterator(metrics=[bias_metrics,toxicity_metrics]):
    support_agent(golden.input)
