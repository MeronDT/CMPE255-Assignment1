import { useMemo, useState } from "react";
import { InlineMath, BlockMath } from "react-katex";
import "katex/dist/katex.min.css";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine,
} from "recharts";
import Quiz, { type QuizQuestion } from "./Quiz";

const questions: QuizQuestion[] = [
  {
    question: "A medical test says a healthy patient has a disease. What type of error is this?",
    options: ["Type I error (false positive)", "Type II error (false negative)", "Not an error, it's a true positive", "Sampling bias"],
    correctIndex: 0,
    explanation: "Type I error = false positive = rejecting a true null hypothesis ('patient is healthy') when it was actually true.",
  },
  {
    question: "Why is accuracy a misleading metric on a 99%-vs-1% imbalanced dataset?",
    options: [
      "Accuracy can't be computed on imbalanced data",
      "A trivial 'always predict majority class' model scores 99% accuracy while catching zero minority cases",
      "Accuracy always equals precision on imbalanced data",
      "Imbalanced datasets don't have a defined accuracy",
    ],
    correctIndex: 1,
    explanation: "This is exactly the trap in fraud/anomaly detection: a classifier that never flags anything looks great on accuracy but is useless. AUPRC or recall-at-fixed-precision are better metrics under heavy imbalance.",
  },
  {
    question: "You increase the classification threshold (require higher confidence to predict positive). What happens to precision and recall?",
    options: [
      "Both increase",
      "Precision typically increases, recall typically decreases",
      "Precision typically decreases, recall typically increases",
      "Both stay the same",
    ],
    correctIndex: 1,
    explanation: "A higher bar for 'positive' means fewer, more confident positive predictions -- fewer false positives (higher precision) but more missed true positives (lower recall). This is the precision/recall tradeoff.",
  },
  {
    question: "What does a ROC-AUC of 0.5 mean?",
    options: [
      "The model is 50% accurate",
      "The model's ranking of positive vs. negative examples is no better than random guessing",
      "The model catches half of all positive cases",
      "The model has a 50% false positive rate",
    ],
    correctIndex: 1,
    explanation: "ROC-AUC measures ranking quality: the probability a random positive example scores higher than a random negative one. 0.5 = coin flip, 1.0 = perfect ranking.",
  },
];

// Synthetic score distributions: negatives centered lower, positives centered higher, with overlap.
function gaussianSample(mean: number, std: number, rng: () => number) {
  const u1 = rng(), u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + z * std;
}

function mulberry32(seed: number) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildDataset() {
  const rng = mulberry32(42);
  const negatives = Array.from({ length: 500 }, () => Math.max(0, Math.min(1, gaussianSample(0.35, 0.15, rng))));
  const positives = Array.from({ length: 500 }, () => Math.max(0, Math.min(1, gaussianSample(0.65, 0.15, rng))));
  return { negatives, positives };
}

const DATASET = buildDataset();

function confusionAt(threshold: number) {
  const tp = DATASET.positives.filter((s) => s >= threshold).length;
  const fn = DATASET.positives.length - tp;
  const fp = DATASET.negatives.filter((s) => s >= threshold).length;
  const tn = DATASET.negatives.length - fp;
  return { tp, fn, fp, tn };
}

function rocCurve() {
  const points = [];
  for (let t = 0; t <= 1.0001; t += 0.02) {
    const { tp, fn, fp, tn } = confusionAt(t);
    const tpr = tp / (tp + fn);
    const fpr = fp / (fp + tn);
    points.push({ fpr, tpr });
  }
  return points.sort((a, b) => a.fpr - b.fpr);
}

function auc(points: { fpr: number; tpr: number }[]) {
  let area = 0;
  for (let i = 1; i < points.length; i++) {
    const dx = points[i].fpr - points[i - 1].fpr;
    area += dx * (points[i].tpr + points[i - 1].tpr) / 2;
  }
  return area;
}

const ROC_POINTS = rocCurve();
const AUC_VALUE = auc(ROC_POINTS);

export default function ModelEvaluation() {
  const [threshold, setThreshold] = useState(0.5);
  const [costFP, setCostFP] = useState(1);
  const [costFN, setCostFN] = useState(5);

  const { tp, fn, fp, tn } = useMemo(() => confusionAt(threshold), [threshold]);
  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const f1 = (2 * precision * recall) / (precision + recall) || 0;
  const totalCost = fp * costFP + fn * costFN;

  return (
    <div>
      <h2>Evaluating a Model</h2>
      <p className="lede">Confusion matrices, error types, ROC-AUC, cost-sensitive evaluation, and the precision/recall tradeoff — all live on one synthetic dataset.</p>

      <h3>The Confusion Matrix &amp; Error Types</h3>
      <p>
        Every binary classifier's predictions fall into exactly four buckets. <strong>Type I error</strong>{" "}
        (false positive) is predicting positive when the truth is negative — like a smoke alarm going off
        with no fire. <strong>Type II error</strong> (false negative) is predicting negative when the truth
        is positive — like a smoke alarm staying silent during an actual fire. The two error types trade off
        against each other as you move the decision threshold.
      </p>

      <h3>Live Simulation: Threshold, Confusion Matrix, Precision/Recall</h3>
      <p>1,000 synthetic "model scores" (500 true negatives, 500 true positives, overlapping distributions — a realistic imperfect classifier). Drag the threshold and watch everything update.</p>
      <div className="card">
        <div className="sim-controls">
          <label>
            Decision threshold <span className="value">{threshold.toFixed(2)}</span>
            <input type="range" min={0} max={1} step={0.01} value={threshold} onChange={(e) => setThreshold(+e.target.value)} />
          </label>
        </div>
        <table className="mini">
          <thead><tr><th></th><th>Predicted Positive</th><th>Predicted Negative</th></tr></thead>
          <tbody>
            <tr><th>Actual Positive</th><td style={{ color: "#4dd6b6" }}>TP: {tp}</td><td style={{ color: "#e65c5c" }}>FN: {fn} (Type II)</td></tr>
            <tr><th>Actual Negative</th><td style={{ color: "#e65c5c" }}>FP: {fp} (Type I)</td><td style={{ color: "#4dd6b6" }}>TN: {tn}</td></tr>
          </tbody>
        </table>
        <div className="grid" style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginTop: 14 }}>
          <div><div style={{ color: "#8a97b5", fontSize: 12 }}>Precision</div><div style={{ fontSize: 20, fontWeight: 700 }}>{(precision * 100).toFixed(1)}%</div></div>
          <div><div style={{ color: "#8a97b5", fontSize: 12 }}>Recall</div><div style={{ fontSize: 20, fontWeight: 700 }}>{(recall * 100).toFixed(1)}%</div></div>
          <div><div style={{ color: "#8a97b5", fontSize: 12 }}>F1</div><div style={{ fontSize: 20, fontWeight: 700 }}>{(f1 * 100).toFixed(1)}%</div></div>
        </div>
        <p style={{ fontSize: 13, color: "#8a97b5", marginTop: 10 }}>
          Slide the threshold toward 1.0: fewer positive predictions, higher precision, lower recall. Slide toward 0.0: the opposite. There is no threshold that maximizes both simultaneously on overlapping distributions.
        </p>
      </div>

      <h3>ROC Curve &amp; AUC</h3>
      <p>
        The ROC curve plots True Positive Rate against False Positive Rate across <em>every possible
        threshold</em> at once — unlike precision/recall, it doesn't depend on your choice of threshold.
        The Area Under the Curve (AUC) summarizes ranking quality: <InlineMath math="P(\text{score}(\text{random positive}) > \text{score}(\text{random negative}))" />.
      </p>
      <div className="card">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={ROC_POINTS} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
            <CartesianGrid stroke="#2a3550" />
            <XAxis dataKey="fpr" type="number" domain={[0, 1]} stroke="#8a97b5" label={{ value: "False Positive Rate", position: "insideBottom", offset: -5, fill: "#8a97b5", fontSize: 12 }} />
            <YAxis dataKey="tpr" type="number" domain={[0, 1]} stroke="#8a97b5" label={{ value: "True Positive Rate", angle: -90, position: "insideLeft", fill: "#8a97b5", fontSize: 12 }} />
            <Tooltip contentStyle={{ background: "#161d2e", border: "1px solid #2a3550" }} labelStyle={{ color: "#e6ebf5" }} itemStyle={{ color: "#e6ebf5" }} cursor={{ fill: "rgba(108,140,255,0.12)", stroke: "#6c8cff", strokeOpacity: 0.4 }} />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#8a97b5" strokeDasharray="4 4" />
            <Line type="monotone" dataKey="tpr" stroke="#6c8cff" strokeWidth={2} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
        <p style={{ fontSize: 14 }}>This model's AUC on the synthetic data: <strong style={{ color: "#4dd6b6" }}>{AUC_VALUE.toFixed(3)}</strong> (dashed line = random guessing, AUC 0.5).</p>
      </div>

      <h3>Cost Matrix: When Errors Aren't Equal</h3>
      <p>
        In fraud detection, a missed fraud (false negative) usually costs far more than a false alarm
        (false positive). A cost matrix makes this explicit and lets you find the threshold that minimizes
        <em> business cost</em>, not just error count.
      </p>
      <div className="card">
        <div className="sim-controls">
          <label>Cost per false positive <span className="value">{costFP}</span>
            <input type="range" min={1} max={20} step={1} value={costFP} onChange={(e) => setCostFP(+e.target.value)} />
          </label>
          <label>Cost per false negative <span className="value">{costFN}</span>
            <input type="range" min={1} max={20} step={1} value={costFN} onChange={(e) => setCostFN(+e.target.value)} />
          </label>
        </div>
        <p>
          At threshold {threshold.toFixed(2)}: {fp} false positives × {costFP} + {fn} false negatives × {costFN}{" "}
          = <strong style={{ color: "#e6a24d" }}>total cost {totalCost}</strong>. Try sliding the threshold with a high FN cost — the
          cost-minimizing threshold shifts lower (catch more positives, tolerate more false alarms) than the
          "optimal" threshold by accuracy alone.
        </p>
      </div>

      <div className="katex-block">
        <BlockMath math="\text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{TP + FN} \qquad F_1 = \frac{2 \cdot P \cdot R}{P + R}" />
      </div>

      <Quiz title="Model Evaluation" questions={questions} />
    </div>
  );
}
