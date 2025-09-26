import requests
import json
from langfuse import Langfuse, get_client

# Langfuse 초기화
langfuse = Langfuse(
  secret_key="sk-lf-5d1a1869-fe21-45f4-8385-f0062a62a2f4",
  public_key="pk-lf-9d02c54e-9dad-4c22-aae2-a57f393cd2ff",
  host="https://cloud.langfuse.com"
)

# trace_id 생성 (필요하면)
trace_id = Langfuse.create_trace_id()

prompt = "Explain Newton's laws in simple terms."

# Ollama 호출
response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3.1:8b", "prompt": prompt},
    stream=True
)

output_text = ""
for line in response.iter_lines():
    if line:
        try:
            obj = json.loads(line.decode("utf-8"))
            if "response" in obj:
                output_text += obj["response"]
        except json.JSONDecodeError:
            continue

# 대체: 직접 span/generation context manager가 없는 경우, observe 데코레이터 활용
# 혹은 get_client() 와 함께 start_as_current_span / start_as_current_generation 만약 지원되면 사용
# 예:
client = get_client()

with client.start_as_current_observation(
    name="llama3.1:8b-gen",
    as_type="generation",
    model="llama3.1:8b"
) as gen:
    gen.update(output=output_text)


print("Ollama Response:", output_text)
