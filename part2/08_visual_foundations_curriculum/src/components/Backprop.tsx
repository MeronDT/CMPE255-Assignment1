import { useMemo, useState } from "react";
import { InlineMath, BlockMath } from "react-katex";
import "katex/dist/katex.min.css";
import Quiz, { type QuizQuestion } from "./Quiz";

const questions: QuizQuestion[] = [
  {
    question: "The chain rule computes the derivative of a composition f(g(x)). What is the formula?",
    options: [
      "f'(g(x)) + g'(x)",
      "f'(g(x)) · g'(x)",
      "f(g'(x))",
      "f'(x) · g'(x)",
    ],
    correctIndex: 1,
    explanation: "d/dx f(g(x)) = f'(g(x)) * g'(x) -- the derivative of the outer function (evaluated at the inner function's output) times the derivative of the inner function.",
  },
  {
    question: "Why is backpropagation just repeated application of the chain rule?",
    options: [
      "A neural network's loss is a deep composition of functions (layers), and backprop computes each parameter's gradient by chaining local derivatives from the loss backward through every layer",
      "Backpropagation doesn't use calculus at all, it's pure linear algebra",
      "The chain rule is only used in the first layer",
      "Backpropagation replaces the chain rule with a faster approximation",
    ],
    correctIndex: 0,
    explanation: "Each layer is a function composed with the next. To get d(Loss)/d(early-layer-weight), you chain together the local derivative at every layer in between -- exactly the multi-variable chain rule, applied systematically and cached for reuse (that reuse is what makes it efficient).",
  },
  {
    question: "In backprop, why compute gradients backward (output to input) rather than forward?",
    options: [
      "Forward-mode is mathematically impossible for neural networks",
      "Reverse-mode lets you compute the gradient of ONE scalar loss with respect to ALL parameters in a single backward pass, which is far cheaper than one forward-mode pass per parameter",
      "Backward computation uses less memory in all cases",
      "It's purely a historical convention with no efficiency reason",
    ],
    correctIndex: 1,
    explanation: "This is the key efficiency insight (reverse-mode automatic differentiation): with millions of parameters but one scalar loss, one backward pass gives every parameter's gradient, vs. needing millions of forward passes otherwise.",
  },
];

function sigmoid(z: number) { return 1 / (1 + Math.exp(-z)); }

export default function Backprop() {
  const [w, setW] = useState(0.6);
  const [b, setB] = useState(0.1);
  const [x] = useState(2.0);
  const [y] = useState(1.0);

  const trace = useMemo(() => {
    const z = w * x + b;
    const a = sigmoid(z);
    const L = 0.5 * (a - y) * (a - y);

    // Backward pass, chain rule step by step:
    const dL_da = a - y;
    const da_dz = a * (1 - a); // sigmoid derivative
    const dz_dw = x;
    const dz_db = 1;

    const dL_dz = dL_da * da_dz;
    const dL_dw = dL_dz * dz_dw;
    const dL_db = dL_dz * dz_db;

    return { z, a, L, dL_da, da_dz, dL_dz, dz_dw, dz_db, dL_dw, dL_db };
  }, [w, b, x, y]);

  return (
    <div>
      <h2>Chain Rule &amp; Backpropagation</h2>
      <p className="lede">Backpropagation is the chain rule, applied systematically through a computational graph, with intermediate results cached for reuse.</p>

      <h3>The Chain Rule</h3>
      <p>For a composition of functions, the chain rule lets you differentiate step by step:</p>
      <div className="katex-block">
        <BlockMath math="\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)" />
      </div>
      <p>
        Extended to a chain of many functions (exactly what a neural network's layers are), each link
        contributes a local derivative, and you multiply them all together:
      </p>
      <div className="katex-block">
        <BlockMath math="\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}" />
      </div>

      <h3>Live: A Single Neuron's Forward + Backward Pass</h3>
      <p>
        The smallest possible "neural network" — one neuron, one weight, sigmoid activation, squared-error
        loss against a fixed target. Adjust w and b and watch the full forward pass compute a prediction, and
        the full backward pass compute exactly how much each parameter should change.
      </p>
      <div className="card">
        <div className="sim-controls">
          <label>Weight w <span className="value">{w.toFixed(2)}</span>
            <input type="range" min={-2} max={2} step={0.05} value={w} onChange={(e) => setW(+e.target.value)} />
          </label>
          <label>Bias b <span className="value">{b.toFixed(2)}</span>
            <input type="range" min={-2} max={2} step={0.05} value={b} onChange={(e) => setB(+e.target.value)} />
          </label>
          <label>Input x (fixed) <span className="value">{x}</span></label>
          <label>Target y (fixed) <span className="value">{y}</span></label>
        </div>

        <h4 style={{ marginTop: 16 }}>Forward Pass</h4>
        <table className="mini">
          <thead><tr><th>z = wx + b</th><th>a = sigmoid(z)</th><th>L = ½(a - y)²</th></tr></thead>
          <tbody><tr>
            <td>{trace.z.toFixed(4)}</td>
            <td>{trace.a.toFixed(4)}</td>
            <td>{trace.L.toFixed(4)}</td>
          </tr></tbody>
        </table>

        <h4 style={{ marginTop: 16 }}>Backward Pass (chain rule, one link at a time)</h4>
        <table className="mini">
          <thead><tr><th>∂L/∂a</th><th>∂a/∂z</th><th>∂L/∂z = ∂L/∂a · ∂a/∂z</th><th>∂z/∂w</th><th>∂L/∂w = ∂L/∂z · ∂z/∂w</th></tr></thead>
          <tbody><tr>
            <td>{trace.dL_da.toFixed(4)}</td>
            <td>{trace.da_dz.toFixed(4)}</td>
            <td>{trace.dL_dz.toFixed(4)}</td>
            <td>{trace.dz_dw.toFixed(4)}</td>
            <td style={{ color: "#4dd6b6", fontWeight: 700 }}>{trace.dL_dw.toFixed(4)}</td>
          </tr></tbody>
        </table>
        <p style={{ fontSize: 13, color: "#8a97b5", marginTop: 10 }}>
          Notice <InlineMath math="\partial L / \partial z" /> is computed <em>once</em> and reused for both{" "}
          <InlineMath math="\partial L/\partial w" /> and <InlineMath math="\partial L/\partial b" /> —
          that reuse of shared intermediate gradients, multiplied out across a whole network of layers, is
          exactly what makes backpropagation efficient instead of recomputing everything from scratch for
          every single parameter.
        </p>
      </div>

      <Quiz title="Chain Rule & Backpropagation" questions={questions} />
    </div>
  );
}
