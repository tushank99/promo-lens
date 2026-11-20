"""Prompts for the propose and judge stages. Text only; no logic."""
from __future__ import annotations

PROPOSE_SYSTEM = (
    "You are a senior editor of vertical microdrama promos (ReelShort / DramaBox style). "
    "You pick promo moments from a timestamped screenplay. Reply with a single JSON object only."
)

PROPOSE_TEMPLATE = """Find the best promo moments in this episode.

A promo is a 20-60 second continuous clip posted on TikTok/Instagram/YouTube to pull in NEW viewers.
It should open on a hook, make sense to someone who has never seen the show, and leave the viewer wanting more.

EPISODE LOGLINE: {logline}
CONTINUITY NOTES: {continuity}

TIMELINE FACTS (computed by code from the timestamps; the checker uses exactly these numbers):
{timeline}

SCREENPLAY. One unit per line: id | start-end seconds | type | text. Units are in file order and may overlap
in time (camera and action units usually overlap dialogue).
{units}

CANDIDATE SELECTION CONSTRAINTS (checked by code; a violating window is rejected, never repaired):
- Propose continuous windows using unit IDs only. Never write timestamps.
- A window runs from start_unit_id to end_unit_id inclusive, in file order. It plays from the EARLIEST start
  to the LATEST end of any unit in it, and that total must be 20-60 seconds.
- A window must stay inside ONE scene (scene unit ranges are in the timeline facts).
- Do not start or end in the middle of a spoken line. Include whole dialogue units, and include both sides
  of any dialogue that overlaps in time.
- Silent or visual-only stretches are allowed: camera units carry visual storytelling (a reveal, a reaction).
  Gaps with nothing scripted are allowed too, but will be flagged for human review.
- Prefer emotionally compelling moments with clear stakes, escalation, betrayal, confrontation or payoff.
- A strong promo does not require an unresolved cliffhanger. If the window ends on one, give
  cliffhanger_unit_id (a unit in the last 20% of the window); otherwise use null.
- Propose up to {max_candidates} DISTINCT moments, not different cuts of the same moment. Do not pad.
  Return fewer candidates, or an empty list, if no suitable window exists.

THE HOOK (most rejections come from getting this wrong):
- The hook is the EARLIEST compelling event of the clip: whatever would stop a cold viewer scrolling in the first
  3 seconds. It can be VISUAL (a camera or action unit: a reveal, a fire, a scar, a slap), SPOKEN (a dialogue
  line) or AUDIO (a sound-FX unit, or a scream/shout). Visual hooks are as valid as spoken ones.
- If the window opens on a striking image, THAT image is the hook: name its unit, with modality "visual". Do not
  name a later line just because it is spoken. A line that explains the image later is fine; it is not the hook.
- hook.unit_ids: the unit(s) that establish the hook. The earliest of them must BEGIN within 3 seconds of the clip
  start (code measures this from the timestamps). Usually start the window on, or right before, the hook.
- hook.modality must match the units: visual = camera/action units, spoken = dialogue, audio = sfx or dialogue.

Return exactly:
{{"candidates": [
  {{"start_unit_id": "...", "end_unit_id": "...",
    "hook": {{"unit_ids": ["..."], "modality": "visual|spoken|audio",
              "description": "what the viewer sees/hears", "evidence": "why it stops a scroller"}},
    "cliffhanger_unit_id": "..." or null, "reason": "one short sentence"}}
]}}"""

RETRY_TEMPLATE = """Some promo windows you proposed for this episode were rejected by the code checker. For each one,
propose ONE new window for the same moment that satisfies every rule, or give up if no compliant window would
still be a compelling promo. This is a fresh proposal: do not just nudge the old IDs.

EPISODE LOGLINE: {logline}

RULES: 20-60 seconds from the earliest start to the latest end of the units in the window; one scene only;
the earliest hook unit must begin within 3 seconds of the clip start; never start/end inside a spoken line;
include both sides of overlapping dialogue; if you give cliffhanger_unit_id it must start in the last 20%.

THE HOOK (most rejections come from getting this wrong):
- The hook is the EARLIEST compelling event of the clip: whatever would stop a cold viewer scrolling in the first
  3 seconds. It can be VISUAL (a camera or action unit: a reveal, a fire, a scar, a slap), SPOKEN (a dialogue
  line) or AUDIO (a sound-FX unit, or a scream/shout). Visual hooks are as valid as spoken ones.
- If the window opens on a striking image, THAT image is the hook: name its unit, with modality "visual". Do not
  name a later line just because it is spoken. A line that explains the image later is fine; it is not the hook.
- hook.unit_ids: the unit(s) that establish the hook. The earliest of them must BEGIN within 3 seconds of the clip
  start (code measures this from the timestamps). Usually start the window on, or right before, the hook.
- hook.modality must match the units: visual = camera/action units, spoken = dialogue, audio = sfx or dialogue.

REJECTED WINDOWS:
{rejected}

Return exactly:
{{"retries": [
  {{"retry_of": "c02", "start_unit_id": "...", "end_unit_id": "...",
    "hook": {{"unit_ids": ["..."], "modality": "visual|spoken|audio", "description": "...", "evidence": "..."}},
    "cliffhanger_unit_id": "..." or null, "reason": "one short sentence"}},
  {{"retry_of": "c03", "give_up": true, "reason": "why no compliant window works"}}
]}}"""

JUDGE_SYSTEM = (
    "You are a strict, independent judge of promo clips for vertical microdramas. You score ONLY what is on the "
    "page. Reply with a single JSON object only."
)

JUDGE_TEMPLATE = """Judge this candidate promo clip. It will be posted to strangers who have never seen the show.

You see only the clip's units. Judge the clip from these units alone, as a cold viewer would.

EPISODE LOGLINE (use ONLY to decide spoils_main_twist; do NOT use it to fill gaps when scoring clarity):
{logline}
CONTINUITY NOTES: {continuity}

CLIP: {clip_start}s to {clip_end}s ({duration}s). Units below: id | seconds from clip start | type | text.
{units}

SCALE (use for hook, clarity, emotion, escalation, ending):
0 absent or unusable | 1 very weak | 2 weak | 3 good enough to work | 4 strong | 5 exceptional
Do not inflate. 3 means it works, 5 is rare.

- hook: how strongly the FIRST 3 SECONDS make a scrolling viewer stop (a slap, a reveal, a shocking line or image).
- clarity: could a stranger state the central conflict (who vs who, what is at stake) from this clip alone?
  Mystery is fine. Confusion about who is who or why it matters is not.
- emotion: intensity and authenticity of the emotion on the page.
- escalation: does tension or stakes rise across the clip rather than stay flat?
- ending: how strongly the final moments make a viewer want to keep watching. Classify the ending type
  carefully; most endings are NOT cliffhangers:
    "cliffhanger": the clip stops on an UNRESOLVED question, threat, reveal-in-progress or imminent consequence
                   (we do not yet know what happens or what it means).
    "payoff": the clip lands an answer: a betrayal revealed, an accusation confirmed, a secret admitted.
    "emotional_resolution": the clip ends on a character's emotional turn or transformation (e.g. hurt becomes resolve).
    "transition": the clip simply stops without a strong narrative beat.

For every score cite evidence_unit_ids: unit ids FROM THIS CLIP ONLY that justify the score. For ending, cite the
unit(s) in the final moments of the clip.

Also return:
- hook_unit_id: the unit you consider the real hook (it should be near the start).
- hook_type: one of slap_or_violence | shocking_line | reveal | accusation | threat | public_humiliation | confession | visual_shock | other
- emotion_type: one of shock | heartbreak | rage | revenge | humiliation | fear | romance | suspense | other
- trope: a short phrase (e.g. "betrayed bride", "hidden identity")
- central_conflict: ONE sentence a stranger could say after watching
- spoils_main_twist: true only if the clip gives away a major later twist that new viewers should discover by
  watching the series, as opposed to establishing the premise

Return exactly:
{{"hook": {{"score": 0, "evidence_unit_ids": []}},
  "clarity": {{"score": 0, "evidence_unit_ids": []}},
  "emotion": {{"score": 0, "evidence_unit_ids": []}},
  "escalation": {{"score": 0, "evidence_unit_ids": []}},
  "ending": {{"score": 0, "type": "cliffhanger|payoff|emotional_resolution|transition", "evidence_unit_ids": []}},
  "hook_unit_id": "", "hook_type": "", "emotion_type": "", "trope": "",
  "central_conflict": "", "spoils_main_twist": false}}"""
