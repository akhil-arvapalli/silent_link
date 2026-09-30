import glossesJson from './glosses.json';

export interface GlossEntry {
  words: string[];
  type: string;
}

export interface GlossesConfig {
  language: string;
  version: string;
  finger_spelling: { letters: string[]; model: string };
  glosses: Record<string, GlossEntry>;
}

const config = glossesJson as unknown as GlossesConfig;

/**
 * Ordered gloss keys (class index order).
 *
 * MUST match the order used at training time (the Python training script uses
 * `sorted(data["glosses"].keys())`), so argmax over the model logits maps to
 * the correct gloss.
 */
export const GLOSSES: string[] = Object.keys(config.glosses).sort();

export function glossWords(gloss: string): string[] {
  return config.glosses[gloss]?.words ?? [];
}

export function glossLabel(gloss: string): string {
  return glossWords(gloss)[0] ?? gloss;
}

export function topGloss(logits: ArrayLike<number>): { gloss: string; label: string; confidence: number } {
  let best = 0;
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > logits[best]) best = i;
  }
  const gloss = GLOSSES[best] ?? 'unknown';
  const exp = logits as number[];
  const max = exp[best];
  const sum = exp.reduce((a, b) => a + Math.exp(b - max), 0);
  const confidence = Math.exp(exp[best] - max) / sum;
  return { gloss, label: glossLabel(gloss), confidence };
}
