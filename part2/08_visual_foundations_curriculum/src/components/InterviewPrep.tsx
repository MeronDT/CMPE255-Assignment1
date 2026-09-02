interface Item { q: string; a: string; }

const NAIVE_BAYES: Item[] = [
  { q: "Why does Naive Bayes still work well even though the independence assumption is usually false?",
    a: "It only needs to rank classes correctly, not estimate perfectly calibrated probabilities. Correlated features get 'double counted' identically across classes in many cases, which biases the magnitude but often leaves the argmax ranking intact -- so the classification decision survives even when the assumption is violated." },
  { q: "How would you handle a continuous feature (like age) in Gaussian Naive Bayes?",
    a: "Assume P(x_i | y) follows a Gaussian distribution per class, estimate its mean and variance from the training data for each class, and plug the resulting density into the likelihood product instead of a discrete probability." },
  { q: "What's the difference between Multinomial and Bernoulli Naive Bayes for text classification?",
    a: "Multinomial NB models word counts/frequencies (how many times a word appears); Bernoulli NB models binary presence/absence (does the word appear at all, ignoring count). Multinomial usually wins when word frequency carries signal; Bernoulli can work better on short texts." },
];

const EVALUATION: Item[] = [
  { q: "A model has 99% accuracy on fraud detection. Is it good?",
    a: "Not necessarily -- if fraud is 1% of transactions, a model that always predicts 'not fraud' gets 99% accuracy while catching zero fraud. Ask for precision/recall or AUPRC on the minority class before judging." },
  { q: "When would you optimize for precision over recall, or vice versa?",
    a: "Optimize for precision when false positives are costly relative to false negatives (e.g., spam filter -- you don't want to lose real email). Optimize for recall when false negatives are costly (e.g., cancer screening -- missing a real case is worse than a false alarm requiring follow-up)." },
  { q: "Explain ROC-AUC to a non-technical stakeholder.",
    a: "It measures how well the model separates the two classes overall, across every possible decision threshold, rather than committing to one threshold. A value of 1.0 means perfect separation; 0.5 means the model does no better than a coin flip at telling the classes apart." },
  { q: "Why can ROC-AUC be misleading on a heavily imbalanced dataset?",
    a: "ROC-AUC's false-positive-rate denominator is dominated by the (huge) negative class, so even a moderate absolute number of false positives barely moves FPR -- the curve can look great while precision (which cares about the ratio among predicted positives) is actually poor. AUPRC is more sensitive to this in practice." },
];

const CALCULUS: Item[] = [
  { q: "Why do we need the gradient rather than just the function value to train a model?",
    a: "The function value tells you how bad the current parameters are, but not which direction to move to improve. The gradient gives both the direction of steepest change and its magnitude, which is exactly the information an iterative optimizer needs." },
  { q: "What's the difference between batch, stochastic, and mini-batch gradient descent?",
    a: "Batch GD computes the gradient over the entire dataset per step (accurate but slow per step). Stochastic GD (SGD) uses one example per step (fast, noisy, can escape shallow local minima). Mini-batch is the practical middle ground used almost everywhere -- a small batch balances gradient accuracy against compute efficiency and enables GPU parallelism." },
  { q: "Why might gradient descent get stuck, and how do modern optimizers address it?",
    a: "Plain gradient descent can stall at saddle points or local minima, or oscillate in narrow valleys with a fixed learning rate. Momentum-based optimizers (Adam, RMSProp) accumulate a running estimate of past gradients to smooth updates and adapt the effective step size per parameter, helping escape these issues." },
];

const BACKPROP: Item[] = [
  { q: "Why is backpropagation described as 'reverse-mode automatic differentiation'?",
    a: "It computes the gradient of a single scalar loss with respect to every parameter in one backward traversal of the computational graph, reusing intermediate derivatives -- vastly cheaper than computing each parameter's gradient independently (forward-mode), which would need one full pass per parameter." },
  { q: "What causes the vanishing gradient problem, and how does it connect to the chain rule?",
    a: "Backprop multiplies many local derivatives together across layers (the chain rule). If each layer's derivative is consistently less than 1 (common with sigmoid/tanh saturating activations), the product shrinks exponentially with depth, so early layers get a near-zero gradient signal and barely learn." },
  { q: "Why do ReLU activations help with vanishing gradients compared to sigmoid?",
    a: "ReLU's derivative is exactly 1 for any positive input (not a fraction less than 1 like sigmoid's, which maxes out at 0.25), so gradients don't get shrunk by that link in the chain -- though ReLU introduces its own issue (dead neurons) when inputs are consistently negative." },
];

function Section({ title, items }: { title: string; items: Item[] }) {
  return (
    <>
      <h3>{title}</h3>
      {items.map((item, i) => (
        <details key={i} className="interview-item">
          <summary>{item.q}</summary>
          <div className="answer">{item.a}</div>
        </details>
      ))}
    </>
  );
}

export default function InterviewPrep() {
  return (
    <div>
      <h2>Interview Prep Questions</h2>
      <p className="lede">Common data science interview questions for each topic in this curriculum, with model answers. Click a question to reveal its answer.</p>
      <Section title="Naive Bayes" items={NAIVE_BAYES} />
      <Section title="Model Evaluation" items={EVALUATION} />
      <Section title="Calculus & Gradient Descent" items={CALCULUS} />
      <Section title="Chain Rule & Backpropagation" items={BACKPROP} />
    </div>
  );
}
