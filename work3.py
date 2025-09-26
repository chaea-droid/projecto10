import ollama
import json
import csv

# ----------------------------
# Grading Prompt (force JSON output)
# ----------------------------
judge_prompt = """
You are a grader.
Evaluate the student's answer against the question and reference answer according to a rubric.
Give each category a score from 0 to 5, and write short feedback for each.
Output must be in JSON format only.

Example output:
{
  "Accuracy": {"score": 4, "feedback": "Core concepts are correct but details are missing"},
  "Feedback Specificity": {"score": 3, "feedback": "Lacks examples"},
  "Total": 7
}
"""

# ----------------------------
# Grading Function
# ----------------------------
def grade_answer(question: str, reference_answer: str, student_answer: str, model="llama3.1:8b") -> dict:
    user_prompt = f"""
Question: {question}
Reference Answer: {reference_answer}
Student Answer: {student_answer}
"""
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    output_text = response["message"]["content"]

    # Attempt JSON parsing
    try:
        result = json.loads(output_text)
    except json.JSONDecodeError:
        result = {"raw_output": output_text, "error": "JSON parsing failed"}

    return result

# ----------------------------
# Example Run
# ----------------------------
if __name__ == "__main__":
    question = "What is Newton's First Law?"
    reference = "An object will remain at rest or in uniform motion unless acted upon by an external force."
    student_answers = [
        "Newton's First Law is the law of inertia.",
        "Newton's laws are all about gravity.",
        "Newton's Third Law is the law of action and reaction.",
        "An object keeps moving the same way if no force acts on it.",
        "There are 4 Newton's laws."  # intentionally incorrect
    ]

    results = []
    for i, ans in enumerate(student_answers, start=1):
        print(f"\n=== Student Answer {i} ===")
        print("Student Answer:", ans)

        graded = grade_answer(question, reference, ans)
        print("Grading Result:", graded)

        results.append({"No": i, "Student Answer": ans, "Grading Result": graded})

    # ----------------------------
    # Save to CSV
    # ----------------------------
    with open("grading_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["No", "Student Answer", "Grading Result (JSON)"])
        for r in results:
            writer.writerow([r["No"], r["Student Answer"], json.dumps(r["Grading Result"], ensure_ascii=False)])
