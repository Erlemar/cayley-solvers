import time, bridge

BOT = "artgor_cayley_solver_bot"
bridge.BOT_USERNAME = BOT

def make(text):
    return {"message_id": 999001, "date": int(time.time()),
            "from": {"id": 239578004, "first_name": "Andrey", "last_name": "Lukyanenko",
                     "username": "artgor"},
            "chat": {"id": -5300047051, "title": "CayleyPy_Agent", "type": "group"},
            "text": text,
            "entities": [{"type": "mention", "offset": 0, "length": len("@" + BOT)}]}

print("=== markdown flattening ===")
sample = "## Head\nThe best is **28,094** and __second__ is 28,398.\n```python\nx=1\n```"
print(repr(bridge.to_plain_text(sample)))

print("\n=== language detection ===")
for t in ["what is our best tetraminx score?", "какой у нас лучший результат?", "beam 65536?"]:
    print("  %-40r -> %s" % (t[:38], bridge.language_of(t)))

for label, q in [
    ("EN", "@%s what is our best tetraminx score, and how far ahead of second place are we?" % BOT),
    ("RU", "@%s какая ширина бима используется в продакшн-рецепте для мегаминкса?" % BOT),
]:
    print("\n=== %s question, full path (NOT posted) ===" % label)
    t0 = time.time()
    raw = bridge.run_agent(bridge.build_prompt("CayleyPy_Agent", -5300047051, make(q)))
    final = bridge.scrub(bridge.to_plain_text(raw))
    print("elapsed %.0fs | markdown left: %s" % (time.time()-t0, "**" in final or "##" in final))
    print("-" * 74)
    print(final[:1100])
