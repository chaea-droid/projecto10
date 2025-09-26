import requests
import json
from langfuse import Langfuse, get_client

# Langfuse 초기화
langfuse = Langfuse(
  secret_key="sk-lf-5d1a1869-fe21-45f4-8385-f0062a62a2f4",
  public_key="pk-lf-9d02c54e-9dad-4c22-aae2-a57f393cd2ff",
  host="https://cloud.langfuse.com"
)

client = get_client()

# v2 프롬프트
v2_prompt = """
너는 채점관이다.
학생 답안을 루브릭에 따라 평가하라.
각 항목(정확도, 피드백 구체성 등)에 대해 0~5점으로 채점하고,
항목별로 구체적인 피드백을 작성하라.
"""

# 예시 학생 답안 5개
student_answers = [
    "뉴턴의 제1법칙은 관성의 법칙이다.",
    "뉴턴의 법칙은 모두 중력에 관한 것이다.",
    "뉴턴의 제3법칙은 작용과 반작용의 법칙이다.",
    "물체는 힘이 작용하지 않으면 계속 같은 운동을 한다.",
    "뉴턴의 법칙은 총 4개다."   # 일부러 틀린 답
]

# 각 답안마다 모델 호출
for i, answer in enumerate(student_answers, start=1):
    # 프롬프트에 학생 답안 삽입
    prompt = f"{v2_prompt}\n\n학생 답안: {answer}"

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

    # Langfuse에 기록
    with client.start_as_current_observation(
        name=f"llama3.1:8b-gen-v2-{i}",
        as_type="generation",
        model="llama3.1:8b"
    ) as gen:
        gen.update(
            input=prompt,
            output=output_text
        )

    # 결과 출력
    print(f"\n=== 학생 답안 {i} ===")
    print("학생 답안:", answer)
    print("평가 결과:", output_text)
