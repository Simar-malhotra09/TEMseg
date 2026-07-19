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

- What are the min scribble limits. again just write here 

- Same thing with tooltip in refine mode config that we do with shape labels. Allow mutability for user to modify or extend. 
In this case, obv user cant invent fields, they can choose to removbe/add which fields they want to see in the tooltip. 
The backend/fronetnd source of truth for fields avaolbel are in commits 24a7d196 and c863fca3 respectively. 

