/**
 * Mirror of model/src/synthesis/normalize.py — rule-based sentence -> gloss
 * mapping for the on-device Text->Sign renderer.
 *
 * Reads the same canonical gloss specs the Python side uses (words per gloss)
 * and maps free-form input onto the gloss vocabulary, longest-phrase-first.
 */

export interface GlossSpec {
  words: string[] | string;
  type: string;
}

const PUNCT = /[^\w\s']/g;

function canonical(text: string): string {
  return text.replace(PUNCT, ' ').trim().toLowerCase();
}

export class GlossNormalizer {
  private byPhrase = new Map<string, string>();
  private phrases: string[] = [];

  constructor(glosses: Record<string, GlossSpec>) {
    for (const [gloss, spec] of Object.entries(glosses)) {
      const words = Array.isArray(spec.words) ? spec.words : [spec.words];
      for (const phrase of words) {
        const key = canonical(phrase);
        if (key) this.byPhrase.set(key, gloss);
      }
    }
    this.phrases = [...this.byPhrase.keys()].sort((a, b) => b.length - a.length);
  }

  /** Map an input sentence onto the canonical gloss sequence (empty if none). */
  normalize(text: string): string[] {
    const tokens = canonical(text).split(' ').filter(Boolean);
    const out: string[] = [];
    let i = 0;
    while (i < tokens.length) {
      let matched = false;
      for (const phrase of this.phrases) {
        const n = phrase.split(' ').length;
        if (n <= 0) continue;
        const window = tokens.slice(i, i + n).join(' ');
        if (window === phrase) {
          const gloss = this.byPhrase.get(phrase)!;
          if (out[out.length - 1] !== gloss) out.push(gloss);
          i += n;
          matched = true;
          break;
        }
      }
      if (!matched) i += 1;
    }
    return out;
  }
}
