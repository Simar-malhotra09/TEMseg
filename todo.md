# - of vertices: rel area not enough, maybe look at model conf as well. 
# - Make sure the model loading changes doesnt break anything on a new build.

- Allow users to modify existing and define their own custom configs. Curently we have a cinfig files with labesl adn rules.
Can we expose a easy client ui for user to edit? Should be simple powerful and not allow them to do shit that would break up the thing
Obv the default should not be modified. I want you to just write your reccomended plan here. 

Plan: keep `shape_config.toml` as the live file but ship a `shape_config.default.toml` next to
it that's never touched — "reset to defaults" just copies that over the live file. Add two
backend endpoints, GET/PUT `/config/shape-rules`, that read/write the live file as JSON (rules
array only, `default_shape` stays fixed). PUT never writes raw TOML from the client — it builds
the TOML server-side and validates it through the existing `load_shape_classification_config`
parser (same function `compute_stats.py` already calls, so no duplicate validation logic); a
bad rule gets rejected with the same error it'd raise today, and the file on disk is untouched
until validation passes. Frontend gets a small rule-builder panel (gear icon, own modal/section):
each rule is a label + list of {metric, op, value} conditions, with add/remove rule and
add/remove condition, plus up/down reorder since rules are evaluated top-to-bottom. `metric` and
`op` are closed dropdowns sourced from the same `_SHAPE_METRICS` / operator enums the backend
validates against, so users can only recombine existing metrics — never invent a field. No
restart needed: `load_shape_classification_config` is already called fresh per request, so an
edit applies on the next segment/recompute.

- Delete scribble in rf mode inaugemtn tba. Just right clikign on alr exisitng scribble deleted iot. Make sure logic is 
consistent. 

- How does the rf eciction glgoc work? just writye here. grep for eviction in backend as a tip. 

Answer: `rf_cache.py` keeps one trained `RFRecovery` per session in an in-memory dict keyed on
`session_id`, no TTL or size cap — `evict(session_key)` just pops that key. It's called
whenever the cached classifier would go stale relative to the mask it was trained on: at the
top of `/rf/train` and `/rf/propose` (both evict then retrain fresh, since the cache is only
keyed on session_id and can't tell the mask changed underneath it), and after any mask edit —
`masks.py` and `segment.py` call `rf_cache.evict(session_id)` right after a refine save or a
new segmentation so the next `/rf/propose` doesn't train on a stale mask. There's also a manual
`DELETE /rf/cache/{session_key}` endpoint for evicting on demand. Nothing evicts on session
deletion/expiry itself — same as the rest of the filesystem-based session state.

- What are the min scribble limits. again just write here 

Answer: two separate floors. On the canvas (`ScribbleCanvas.tsx`), a stroke only gets kept on
mouseup if it has at least 2 points (`points.length >= 4`, i.e. not a single click with no
drag) — that's just a "don't save a zero-length stroke" guard, not a real limit. The real limit
is server-side in `rf.py`: `MIN_BG_FRACTION = 0.04`, so `/rf/propose` refuses to train unless
the rasterized bg scribbles cover at least 4% of the image's pixels, returning an error naming
how many px were marked vs how many are required. Below that floor the RF barely sees what
background looks like and everything slightly different from the sampled pixels drifts to the
foreground, so it refuses to train rather than return junk (see the comment above
`MIN_BG_FRACTION`).

- Same thing with tooltip in refine mode config that we do with shape labels. Allow mutability for user to modify or extend. 
In this case, obv user cant invent fields, they can choose to removbe/add which fields they want to see in the tooltip. 
The backend/fronetnd source of truth for fields avaolbel are in commits 24a7d196 and c863fca3 respectively. 

