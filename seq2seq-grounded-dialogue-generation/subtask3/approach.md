# Approach — Grounded Hinglish Dialogue

## Inputs
The model receives two sources:
1. Multi-turn dialogue history with speaker and turn order.
2. The linked grounding document or document passages.

## Architecture
Use two recurrent encoders. The decoder performs separate attention over conversation states and document states, then fuses the two context vectors before generating each next token.

## Preprocessing
Preserve code mixing. Normalize whitespace and obvious artifacts but do not translate Hindi into English. Build the vocabulary from training data only and retain speaker/turn structure.

## Baselines
- Retrieval-style response picker.
- Ungrounded generator: history only.
- Full generator: history + document.

## Evaluation
Use BLEU/ROUGE as overlap metrics, then manually rate relevance, grounding, fluency, code-mixing quality and hallucination. Break results down by mostly-English and heavily code-mixed turns.

## Research question
Does explicit document conditioning improve factual grounding while preserving conversational relevance?
