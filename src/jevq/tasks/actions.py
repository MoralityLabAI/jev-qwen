"""Tool-call / structured-action decisions: pick the tool a request should be routed to."""

from __future__ import annotations

import random

# name -> (description, direct requests, indirect requests)
TOOLS = {
    "get_weather": (
        "Get the weather forecast for a city.",
        ["What's the weather in {city}?", "Give me the forecast for {city}."],
        ["Should I bring an umbrella in {city} today?", "Will I need a coat in {city} tonight?"],
    ),
    "send_email": (
        "Send an email to a recipient.",
        ["Email {person} the meeting notes.", "Send {person} an email about the delay."],
        ["Let {person} know in writing that the report is late.", "Drop {person} a note saying I'm out sick."],
    ),
    "create_calendar_event": (
        "Create an event on the user's calendar.",
        ["Schedule a meeting with {person} on Friday.", "Add a dentist appointment to my calendar."],
        ["Block out Tuesday afternoon so {person} and I can talk.", "Make sure I don't forget lunch with {person} next week."],
    ),
    "search_docs": (
        "Search the internal documentation.",
        ["Search the docs for the deployment guide.", "Look up the API rate limits in the documentation."],
        ["How do we rotate credentials here?", "Where is the onboarding checklist written down?"],
    ),
    "refund_payment": (
        "Refund a customer payment.",
        ["Refund order {order}.", "Issue a refund for order {order}."],
        ["The customer for order {order} was charged twice; give the extra money back.", "Order {order} arrived broken and the customer wants their money back."],
    ),
    "lookup_order": (
        "Look up the status of an order.",
        ["Look up order {order}.", "What is the status of order {order}?"],
        ["Has order {order} shipped yet?", "Where is the package for order {order}?"],
    ),
    "escalate_to_human": (
        "Hand the conversation to a human agent.",
        ["Escalate this conversation to a human.", "Transfer me to a human agent."],
        ["I want to speak to a real person.", "This bot is useless, get me someone who works there."],
    ),
    "translate_text": (
        "Translate text into another language.",
        ["Translate 'good morning' into French.", "Translate this sentence to Spanish."],
        ["How do you say 'thank you' in Japanese?", "What is 'where is the station' in German?"],
    ),
}

# What the user is after, for the worked solutions of the reasoning-trace readout.
NEEDS = {
    "get_weather": "a weather forecast",
    "send_email": "to send someone an email",
    "create_calendar_event": "something put on the calendar",
    "search_docs": "to find something in the documentation",
    "refund_payment": "money returned for an order",
    "lookup_order": "the status of an order",
    "escalate_to_human": "to talk to a human",
    "translate_text": "a translation",
}

CITIES = ["Lisbon", "Denver", "Nairobi", "Osaka", "Bergen"]
PEOPLE = ["Priya", "Marco", "Lena", "Tomas", "Aiko"]


def _fill(rng: random.Random, template: str) -> str:
    return template.format(city=rng.choice(CITIES), person=rng.choice(PEOPLE), order=rng.randint(1000, 9999))


def _render(tool_names: list[str], rules: list[str], request: str) -> str:
    lines = ["Tools:"] + [f"- {name}: {TOOLS[name][0]}" for name in tool_names]
    lines += [f"Rule: {rule}" for rule in rules]
    lines += [f"Request: {request}", "Which tool should be called?"]
    return "\n".join(lines)


def gen_tool_select(rng: random.Random, difficulty: int):
    """1: direct request, 4 tools. 2: indirect request, 6 tools. 3+: the answer depends on a
    routing rule (refund limit); 4 phrases the request indirectly; 5 adds a VIP exception."""
    names = list(TOOLS)
    if difficulty <= 2:
        target = rng.choice(names)
        n_tools = 4 if difficulty == 1 else 6
        shown = [target] + rng.sample([n for n in names if n != target], n_tools - 1)
        rng.shuffle(shown)
        request = _fill(rng, rng.choice(TOOLS[target][1 if difficulty == 1 else 2]))
        meta = {"rule_based": False, "rationale": f"The user wants {NEEDS[target]}."}
        return _render(shown, [], request), target, shown, meta

    limit = rng.choice([50, 100, 200])
    amount = rng.choice([a for a in (20, 40, 80, 120, 150, 250, 400) if a != limit])
    order = rng.randint(1000, 9999)
    rules = [f"refunds over ${limit} must use escalate_to_human instead of refund_payment."]
    effective_limit = limit
    customer = "The customer"
    rationale = f"This is a refund of ${amount}."
    if difficulty >= 5:
        vip_limit = limit * 4
        is_vip = rng.random() < 0.5
        rules.append(f"VIP customers may be refunded directly up to ${vip_limit}.")
        customer = "The VIP customer" if is_vip else "The regular customer"
        if is_vip:
            effective_limit = vip_limit
        rationale += f" The customer is {'a VIP' if is_vip else 'not a VIP'}, so the limit is ${effective_limit}."
    else:
        rationale += f" The limit is ${limit}."
    over = amount > effective_limit
    target = "escalate_to_human" if over else "refund_payment"
    rationale += f" ${amount} is {'over' if over else 'not over'} ${effective_limit}."
    if difficulty >= 4:
        request = f"{customer} for order {order} was overcharged and wants ${amount} back."
    else:
        request = f"Refund ${amount} for order {order}."

    fixed = ["refund_payment", "escalate_to_human"]
    shown = fixed + rng.sample([n for n in names if n not in fixed], 4)
    rng.shuffle(shown)
    meta = {"rule_based": True, "amount": amount, "limit": limit, "rationale": rationale}
    return _render(shown, rules, request), target, shown, meta
