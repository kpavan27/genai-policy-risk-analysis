# Policy risk memo: where moderation leaves a GenAI usage policy uncovered

**Scope.** 5,082 real user prompts sent to a public chatbot demo (ToxicChat, Lin et al.,
2023), checked against the 20 clauses of Google's Generative AI Prohibited Use Policy
(17 Dec 2024), using the OpenAI moderation scores shipped with the dataset as the
"off-the-shelf" signal. Label-based figures use the 2,801 human-annotated prompts only.
Every number below comes from [`metrics.json`](metrics.json).

## Summary

An off-the-shelf moderation signal at its default threshold misses most of what humans
judge to be policy-violating prompts. It also has no category at all for 12 of the
policy's 20 clauses. The largest uncovered risk in real traffic is **attempts to
circumvent safety filters**. The second is **calibration**: covered categories are
scored, but far below the flag threshold. Recommendations follow each risk.

## Risk 1: jailbreak attempts are almost invisible to content moderation (clause 2c)

* Of 113 human-labelled jailbreak prompts, moderation flags **14%** at threshold 0.5
  (95% CI 9-22%). Their median top score is 0.033.
* That is expected, not a malfunction. A jailbreak is about *intent towards the system*,
  and its text is often polite and contains nothing toxic. Content categories cannot see it.
* A 10-pattern lexicon written for this analysis catches **35%** of labelled jailbreaks at
  **75%** precision, and lexicon plus moderation together catch **47%**.
* Sized on all traffic, about **10 in every 1,000 prompts** match the jailbreak lexicon,
  and moderation flags 6% of those.

**Recommendation.** Treat circumvention as its own signal with its own classifier and
review queue, not as a sub-case of toxicity. A lexicon is a cheap first layer for
known templates (DAN-style personas, "ignore previous instructions"), and the 53% that
neither signal catches needs a trained model. Track jailbreak-template families over
time, because they are copied and iterated.

## Risk 2: covered categories are scored, but well below the flag threshold

* For human-labelled toxic prompts, moderation at 0.5 has **15% recall** (CI 12-19%) at
  91% precision.
* Lowering the threshold to 0.024, the best-F1 point (chosen in-sample, so optimistic),
  raises recall to **68%**. Precision falls to 66%, and flagged volume rises from **23 to
  142 per 1,000 prompts**, roughly six times the review load.
* Of the 326 toxic prompts moderation misses at 0.5, the highest-scoring category was
  *sexual* for 171, *harassment* for 66 and *violence* for 55. These categories *exist*;
  the scores are just low. Topic modelling of the misses points to sexual content framed
  as fiction ("write a story..."), demeaning "jokes" about groups, and role-play set-ups.

**Recommendation.** Threshold choice is a policy decision with a measurable cost, so
set it per category from labelled samples and the review capacity available, rather
than inheriting a global 0.5. Fiction and role-play framing deserve explicit guidance in
the policy, since that is where the score and the human judgement diverge most.

## Risk 3: instructions for regulated goods have no category (clause 1e)

* Prompts asking how to make drugs or explosives: 16 in the full set, 13 human-annotated.
  **All 13** are labelled toxic by humans, and moderation flags **none** (0/13, CI 0-23%).
  The highest score among them is 0.34.
* Volume is low, about 3 per 1,000, but severity per item is high, and there is no
  category under which these would ever cross a threshold.

**Recommendation.** Add a dedicated "dangerous instructions" signal (drugs, weapons,
explosives) mapped to clause 1e, because tuning existing categories cannot recover it.

## What the data cannot tell us

Fraud and scams (4a), malware and phishing (2a), privacy and tracking (1f/1g) and
civic/health misinformation (4d) each matched 9 or fewer prompts, at low precision. In
this traffic they are either rare or phrased in ways a lexicon does not catch. Absence
of evidence here is a **monitoring gap, not a clean bill of health**: these clauses have
no moderation signal at all, so a rise in volume would go unnoticed. A labelled sample
from a product's own traffic is the next step for sizing them.

## Method notes and caveats

* ToxicChat labels non-annotated prompts as non-toxic after a moderation pre-filter.
  Evaluating moderation on those rows would be circular, so they are excluded from every
  label-based figure.
* Moderation scores are those published with the dataset (Jan 2024). Current moderation
  models differ, so the method transfers but the exact figures may not.
* Lexicon detectors are simple and auditable. One jailbreak pattern was
  replaced after inspecting a sample of hits. Both versions are reported (precision
  58% → 75%, recall unchanged).
* Severity weights (1-3) are a judgement and only order risks of similar size. The
  ranking is unchanged without them.
* The prompts come from one research demo in 2023, and a production assistant's traffic
  mix will differ.
