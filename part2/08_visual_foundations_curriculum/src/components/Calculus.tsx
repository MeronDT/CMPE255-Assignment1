import { useEffect, useRef, useState } from "react";
import { InlineMath, BlockMath } from "react-katex";
import "katex/dist/katex.min.css";
import Quiz, { type QuizQuestion } from "./Quiz";

const questions: QuizQuestion[] = [
  {
    question: "What does a derivative f'(x) tell you geometrically?",
    options: [
      "The area under the curve up to x",
      "The slope of the line tangent to f at x",
      "The value of f at x",
      "The average value of f over all x",
    ],
    correctIndex: 1,
    explanation: "The derivative is the instantaneous rate of change -- the slope of the tangent line at that exact point, found by taking the limit of the slope of secant lines as they get infinitesimally close.",
  },
  {
    question: "In gradient descent, why do we move OPPOSITE the gradient direction?",
    options: [
      "The gradient points in the direction of steepest INCREASE, and we want to minimize the loss",
      "It's a mathematical convention with no geometric meaning",
      "The gradient always points toward zero",
      "Moving with the gradient would make training faster",
    ],
    correctIndex: 0,
    explanation: "The gradient points toward steepest ascent. To minimize a loss function, you step in the negative gradient direction -- steepest descent.",
  },
  {
    question: "What happens if the learning rate in gradient descent is set too high?",
    options: [
      "Convergence is just slower",
      "The parameter update can overshoot the minimum and diverge or oscillate",
      "Nothing changes, learning rate only affects final accuracy",
      "The gradient becomes zero immediately",
    ],
    correctIndex: 1,
    explanation: "Too large a step can jump past the minimum entirely, and on a steep enough curve this compounds -- the loss can oscillate or diverge instead of converging.",
  },
];

function f(x: number) { return 0.15 * (x - 3) * (x - 3) + 1; }
function fPrime(x: number) { return 0.3 * (x - 3); }

const W = 560, H = 300, PAD = 40;
const X_MIN = -3, X_MAX = 9, Y_MIN = 0, Y_MAX = 6;
const sx = (x: number) => PAD + ((x - X_MIN) / (X_MAX - X_MIN)) * (W - 2 * PAD);
const sy = (y: number) => H - PAD - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * (H - 2 * PAD);

function CurveSvg({ children }: { children?: React.ReactNode }) {
  const pathPoints = [];
  for (let x = X_MIN; x <= X_MAX; x += 0.1) pathPoints.push(`${sx(x)},${sy(f(x))}`);
  return (
    <svg width={W} height={H} style={{ background: "#0f1420", borderRadius: 8 }}>
      <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#2a3550" />
      <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#2a3550" />
      <polyline points={pathPoints.join(" ")} fill="none" stroke="#6c8cff" strokeWidth={2} />
      {children}
    </svg>
  );
}

function DerivativeDemo() {
  const [x0, setX0] = useState(1);
  const slope = fPrime(x0);
  const y0 = f(x0);
  const tanX1 = x0 - 2.5, tanX2 = x0 + 2.5;
  const tanY1 = y0 + slope * (tanX1 - x0), tanY2 = y0 + slope * (tanX2 - x0);

  return (
    <div className="card">
      <h4>Live: The Tangent Line</h4>
      <div className="sim-controls">
        <label>x <span className="value">{x0.toFixed(2)}</span>
          <input type="range" min={X_MIN + 0.5} max={X_MAX - 0.5} step={0.05} value={x0} onChange={(e) => setX0(+e.target.value)} />
        </label>
      </div>
      <CurveSvg>
        <line x1={sx(tanX1)} y1={sy(tanY1)} x2={sx(tanX2)} y2={sy(tanY2)} stroke="#e6a24d" strokeWidth={2} />
        <circle cx={sx(x0)} cy={sy(y0)} r={5} fill="#4dd6b6" />
      </CurveSvg>
      <p style={{ fontSize: 14 }}>
        f(x) = 0.15(x-3)² + 1. At x = {x0.toFixed(2)}: f'(x) = {slope.toFixed(3)} — the tangent line's slope.
        Where the curve is steep, |f'(x)| is large; at the minimum (x=3), f'(x) = 0 exactly.
      </p>
    </div>
  );
}

function GradientDescentDemo() {
  const [x, setX] = useState(-2);
  const [lr, setLr] = useState(2.5);
  const [history, setHistory] = useState<number[]>([-2]);
  const [running, setRunning] = useState(false);
  const timerRef = useRef<number | null>(null);

  const step = () => {
    setX((prev) => {
      const grad = fPrime(prev);
      const next = prev - lr * grad;
      const clamped = Math.max(X_MIN, Math.min(X_MAX, next));
      setHistory((h) => [...h, clamped]);
      return clamped;
    });
  };

  const reset = () => {
    setX(-2);
    setHistory([-2]);
    setRunning(false);
    if (timerRef.current) window.clearInterval(timerRef.current);
  };

  useEffect(() => {
    if (running) {
      timerRef.current = window.setInterval(step, 500);
    } else if (timerRef.current) {
      window.clearInterval(timerRef.current);
    }
    return () => { if (timerRef.current) window.clearInterval(timerRef.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, lr]);

  return (
    <div className="card">
      <h4>Live: Gradient Descent</h4>
      <div className="sim-controls">
        <label>Learning rate <span className="value">{lr.toFixed(2)}</span>
          <input type="range" min={0.1} max={7} step={0.1} value={lr} onChange={(e) => { setLr(+e.target.value); }} />
        </label>
        <button className="btn" onClick={step}>Step</button>
        <button className={running ? "btn" : "btn secondary"} onClick={() => setRunning(!running)}>{running ? "Pause" : "Run"}</button>
        <button className="btn secondary" onClick={reset}>Reset</button>
      </div>
      <CurveSvg>
        {history.slice(0, -1).map((hx, i) => (
          <line key={i} x1={sx(hx)} y1={sy(f(hx))} x2={sx(history[i + 1])} y2={sy(f(history[i + 1]))} stroke="#4dd6b6" strokeWidth={1.5} opacity={0.6} />
        ))}
        <circle cx={sx(x)} cy={sy(f(x))} r={6} fill="#e65c5c" />
      </CurveSvg>
      <p style={{ fontSize: 14 }}>
        x = {x.toFixed(3)}, f(x) = {f(x).toFixed(3)}, gradient f'(x) = {fPrime(x).toFixed(3)}. Update rule:{" "}
        <InlineMath math="x_{t+1} = x_t - \eta \cdot f'(x_t)" /> with η (learning rate) = {lr.toFixed(2)}.
        {" "}Try η &gt; 6 and watch it overshoot and diverge instead of converging.
      </p>
    </div>
  );
}

export default function Calculus() {
  return (
    <div>
      <h2>Differential Calculus &amp; Gradient Descent</h2>
      <p className="lede">Derivatives measure rates of change. Gradient descent uses that measurement to find a function's minimum, one small step at a time.</p>

      <h3>The Derivative</h3>
      <p>
        The derivative of f at a point x is the slope of the tangent line there — formally, the limit of the
        average slope between x and a nearby point, as that point gets infinitesimally close:
      </p>
      <div className="katex-block">
        <BlockMath math="f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}" />
      </div>
      <p>
        Intuitively: it answers "if I nudge x slightly, how much (and in which direction) does f(x) change?"
        That single piece of local information is exactly what an optimizer needs to decide which way to move.
      </p>
      <DerivativeDemo />

      <h3>From Derivatives to Gradient Descent</h3>
      <p>
        Training a model means minimizing a loss function over its parameters. Gradient descent repeats a
        simple idea: compute the derivative (gradient, in multiple dimensions) of the loss with respect to
        the parameters, then take a small step in the <em>opposite</em> direction — since the gradient points
        toward steepest <em>increase</em>, its negative points toward steepest decrease.
      </p>
      <div className="katex-block">
        <BlockMath math="\theta_{t+1} = \theta_t - \eta \nabla_\theta \, \mathcal{L}(\theta_t)" />
      </div>
      <p>Where η (eta) is the <strong>learning rate</strong> — how big a step to take. Too small and training crawls; too large and it can overshoot the minimum entirely, as the simulation below shows.</p>
      <GradientDescentDemo />

      <Quiz title="Calculus & Gradient Descent" questions={questions} />
    </div>
  );
}
