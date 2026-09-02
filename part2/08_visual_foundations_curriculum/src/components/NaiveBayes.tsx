import { useMemo, useState } from "react";
import { InlineMath, BlockMath } from "react-katex";
import "katex/dist/katex.min.css";
import Quiz, { type QuizQuestion } from "./Quiz";

const questions: QuizQuestion[] = [
  {
    question: "What does the 'naive' in Naive Bayes refer to?",
    options: [
      "The model is simple to implement",
      "It assumes all features are conditionally independent given the class",
      "It only works on small datasets",
      "It ignores the prior probability",
    ],
    correctIndex: 1,
    explanation: "The 'naive' assumption is conditional independence of features given the class -- rarely exactly true in reality, but the model works remarkably well anyway.",
  },
  {
    question: "Why do implementations compute log-probabilities instead of raw probabilities?",
    options: [
      "Logs are faster to compute on CPUs",
      "To avoid numerical underflow from multiplying many small probabilities",
      "Because Bayes' theorem requires logarithms",
      "To convert the classifier into a regression model",
    ],
    correctIndex: 1,
    explanation: "Multiplying dozens of probabilities (each < 1) underflows to 0 in floating point. Summing log-probabilities avoids this and turns products into sums.",
  },
  {
    question: "What is Laplace (additive) smoothing used for?",
    options: [
      "Speeding up training",
      "Preventing a single unseen feature value from zeroing out the entire posterior",
      "Reducing the number of features",
      "Normalizing the prior probabilities",
    ],
    correctIndex: 1,
    explanation: "Without smoothing, a feature value never seen with a given class gets probability 0, which zeroes the whole product regardless of other evidence. Adding a small pseudo-count fixes this.",
  },
];

function classify(pSpamPrior: number, pWordGivenSpam: number, pWordGivenHam: number, hasWord: boolean) {
  const pHamPrior = 1 - pSpamPrior;
  const pW_spam = hasWord ? pWordGivenSpam : 1 - pWordGivenSpam;
  const pW_ham = hasWord ? pWordGivenHam : 1 - pWordGivenHam;
  const jointSpam = pSpamPrior * pW_spam;
  const jointHam = pHamPrior * pW_ham;
  const posteriorSpam = jointSpam / (jointSpam + jointHam);
  return { posteriorSpam, jointSpam, jointHam };
}

export default function NaiveBayes() {
  const [prior, setPrior] = useState(0.3);
  const [pFreeGivenSpam, setPFreeGivenSpam] = useState(0.7);
  const [pFreeGivenHam, setPFreeGivenHam] = useState(0.05);
  const [hasWord, setHasWord] = useState(true);

  const result = useMemo(
    () => classify(prior, pFreeGivenSpam, pFreeGivenHam, hasWord),
    [prior, pFreeGivenSpam, pFreeGivenHam, hasWord]
  );

  return (
    <div>
      <h2>Naive Bayes</h2>
      <p className="lede">A probabilistic classifier built directly from Bayes' theorem, with one bold simplifying assumption.</p>

      <h3>The Intuition</h3>
      <p>
        Suppose you want to classify an email as spam or not-spam. Bayes' theorem lets you flip the question
        around: instead of asking "given this email, what's the probability it's spam?" (hard to estimate
        directly), you ask "given that an email <em>is</em> spam, how likely is it to contain these words?"
        (easy to estimate by counting spam emails) — and then use Bayes' theorem to invert that into what
        you actually want.
      </p>

      <h3>The Math</h3>
      <p>Bayes' theorem, applied to classification:</p>
      <div className="katex-block">
        <BlockMath math="P(y \mid x_1, \dots, x_n) = \frac{P(y)\, P(x_1, \dots, x_n \mid y)}{P(x_1, \dots, x_n)}" />
      </div>
      <p>
        The denominator is the same for every class, so it's just a normalizing constant — we can drop it and
        compare the numerator across classes. The "naive" independence assumption then factorizes the hard
        joint likelihood into a product of easy per-feature likelihoods:
      </p>
      <div className="katex-block">
        <BlockMath math="P(x_1, \dots, x_n \mid y) \approx \prod_{i=1}^{n} P(x_i \mid y)" />
      </div>
      <p>So the final classification rule (dropping the constant denominator) is:</p>
      <div className="katex-block">
        <BlockMath math="\hat{y} = \arg\max_{y} \; P(y) \prod_{i=1}^{n} P(x_i \mid y)" />
      </div>
      <p>
        In practice this is computed in log-space to avoid numerical underflow (see quiz below):{" "}
        <InlineMath math="\hat{y} = \arg\max_y \left[ \log P(y) + \sum_i \log P(x_i \mid y) \right]" />
      </p>

      <h3>Live Simulation: One-Word Spam Filter</h3>
      <p>
        A toy spam classifier using a single feature: does the email contain the word "free"? Adjust the
        priors and likelihoods and watch the posterior probability update in real time via Bayes' theorem.
      </p>
      <div className="card">
        <div className="sim-controls">
          <label>
            P(spam) — prior <span className="value">{prior.toFixed(2)}</span>
            <input type="range" min={0.01} max={0.99} step={0.01} value={prior} onChange={(e) => setPrior(+e.target.value)} />
          </label>
          <label>
            P("free" | spam) <span className="value">{pFreeGivenSpam.toFixed(2)}</span>
            <input type="range" min={0.01} max={0.99} step={0.01} value={pFreeGivenSpam} onChange={(e) => setPFreeGivenSpam(+e.target.value)} />
          </label>
          <label>
            P("free" | ham) <span className="value">{pFreeGivenHam.toFixed(2)}</span>
            <input type="range" min={0.01} max={0.99} step={0.01} value={pFreeGivenHam} onChange={(e) => setPFreeGivenHam(+e.target.value)} />
          </label>
          <label>
            Email contains "free"?
            <button className={hasWord ? "btn" : "btn secondary"} onClick={() => setHasWord(!hasWord)}>
              {hasWord ? "Yes" : "No"}
            </button>
          </label>
        </div>
        <table className="mini">
          <thead><tr><th>Joint: spam</th><th>Joint: ham</th><th>Posterior P(spam | evidence)</th></tr></thead>
          <tbody>
            <tr>
              <td>{result.jointSpam.toFixed(4)}</td>
              <td>{result.jointHam.toFixed(4)}</td>
              <td style={{ color: result.posteriorSpam > 0.5 ? "#e65c5c" : "#4dd6b6", fontWeight: 700 }}>
                {(result.posteriorSpam * 100).toFixed(1)}%
              </td>
            </tr>
          </tbody>
        </table>
        <p style={{ fontSize: 13, color: "#8a97b5" }}>
          Notice: even a strong prior can be overturned by strong evidence, and a weak prior needs strong
          evidence to shift it — this is Bayesian updating in miniature. Try setting the prior near 0.01 and
          the likelihoods far apart to see how much evidence it takes to flip the classification.
        </p>
      </div>

      <Quiz title="Naive Bayes" questions={questions} />
    </div>
  );
}
