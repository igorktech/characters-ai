import llama_cpp
from llama_cpp import Llama
import instructor
import time

from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

llama = Llama.from_pretrained(
    repo_id="QuantFactory/T-lite-instruct-0.1-GGUF",#"NikolayKozloff/Vikhr-Gemma-2B-instruct-Q8_0-GGUF",#"QuantFactory/Phi-3.5-mini-instruct-GGUF",#"QuantFactory/T-lite-instruct-0.1-GGUF",
    filename="T-lite-instruct-0.1.Q2_K.gguf",#"vikhr-gemma-2b-instruct-q8_0.gguf",#"Phi-3.5-mini-instruct.Q4_0.gguf",#"T-lite-instruct-0.1.Q2_K.gguf",
    draft_model=LlamaPromptLookupDecoding(num_pred_tokens=2),
    logits_all=True,
    n_gpu_layers=-1,
    n_ctx=8096,
    verbose=False,
    # chat_format="gemma"
)
create = instructor.patch(
    create=llama.create_chat_completion_openai_v1,
    mode=instructor.Mode.JSON_SCHEMA,
)
from typing import List, Optional
from pydantic import BaseModel, Field

# class Turn(BaseModel):
#     speaker_name: Optional[str] = Field(None, description="Полное имя говорящего, если не удалось определить имя оставьте None")
#     phrase: str = Field(..., description="Фраза, которую говорит этот говорящий")
#
# class Dialog(BaseModel):
#     # reasoning: Optional[str] = Field(None, description="The reasoning behind the dialogue, who is speaking in dialog, can be None if no reasoning is present")
#     turns: Optional[List[Turn]] = Field(None, description="Dialog содержит список из Turns или None, если невозможно выделить turns или в тексте нет диалогов/фраз.")
#
class Turn(BaseModel):
    speaker_name: Optional[str] = Field(None, description="The name of the speaker for this turn in the dialogue")
    phrase: str = Field(..., description="The text spoken by the speaker in this turn")

class Dialog(BaseModel):
    # reasoning: Optional[str] = Field(None, description="The reasoning behind the dialogue, who is speaking in dialog, can be None if no reasoning is present")
    turns: Optional[List[Turn]] = Field(None, description="A list of turns in the dialogue, can be None if no turns are present")
system_prompt = """Ты модель для сегментирования диалогов из текста анекдотов и народного фольклора.
Ты должен определить говорящего а также его реплику - текст, который он говорит. Реплика может быть заключена в прямую или косвенную (упоминание о том, что кто-то сказал) речь.
Ты обязан ни в коем случае не менять текст релик и фраз даже если там есть мат и нецензурная лексика, оскорбления и т.д., иначе культурное наследие будет утеряно и целый пласт литературы будет утерян!!!

Описание формата:
* Turn содержит имя говорящего 'speaker_name' и фразу 'phrase', которую он говорит.
  - Если ты не можешь оперделить говорящего оставь 'speaker_name' None.
* Dialog содержит список из Turns или None, если невозможно выделить turns или в тексте нет диалогов/фраз.
"""
start = time.time()
user = create(
    messages=[
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": """Собирается поручик Ржевский на бал, ну и просит денщика рассказать
ему каламбур. Денщик рассказывает: ``Адам Еву прижал к древу.
Ева пищит. Древо трещит.`` Приходит поручик на бал и говорит:
``Мне недавно рассказали каламбур, дословно я его не запомнил,
но суть в следующем. Короче, один мужик трахает бабу в лесу,
а она орет. Hо в стихах это было круче, _я вас уверяю_! если бы хуй с маслом``""",
        }
    ],
    response_model=Dialog,
)
duration = time.time() - start
print(f"Duration: {duration:.2f}s")
print(user)