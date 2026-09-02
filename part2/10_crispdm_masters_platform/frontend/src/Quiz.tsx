import { useState } from "react";

export interface QuizQuestion {
  question: string;
  options: string[];
  correctIndex: number;
  explanation: string;
}

function Question({ q, index }: { q: QuizQuestion; index: number }) {
  const [selected, setSelected] = useState<number | null>(null);
  return (
    <div className="quiz-q">
      <div className="question">{index + 1}. {q.question}</div>
      {q.options.map((opt, i) => {
        let cls = "quiz-opt";
        if (selected !== null) {
          if (i === q.correctIndex) cls += " correct";
          else if (i === selected) cls += " incorrect";
        }
        return (
          <button key={i} className={cls} disabled={selected !== null} onClick={() => setSelected(i)}>
            {opt}
          </button>
        );
      })}
      {selected !== null && <div className="quiz-explain">{q.explanation}</div>}
    </div>
  );
}

export default function Quiz({ title, questions }: { title: string; questions: QuizQuestion[] }) {
  return (
    <div className="card">
      <h4>Quiz: {title}</h4>
      {questions.map((q, i) => <Question key={i} q={q} index={i} />)}
    </div>
  );
}
