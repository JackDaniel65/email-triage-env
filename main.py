from env.environment import EmailTriageEnv
from utils.models import Action

print("EmailTriageEnv - Sanity Check")

print("\n--- Task 1: Spam Detection ---")
env = EmailTriageEnv(task_id=1, shuffle=False)
obs = env.reset()
actions = ["spam","not_spam","not_spam","spam","not_spam","not_spam","spam","not_spam","not_spam","not_spam"]
for a in actions:
    if env.state()["done"]:
        break
    result = env.step(Action(label=a))
    print(f"  {obs.email_id} | label={a} | reward={result.reward} | {result.info.get('reason')}")
    obs = result.observation
print(f"  Avg reward: {env.state()['avg_reward']}")

print("\n--- Task 2: Priority + Category ---")
env = EmailTriageEnv(task_id=2, shuffle=False)
obs = env.reset()
actions2 = [("low","other"),("urgent","billing"),("normal","support"),("low","other"),("low","feedback"),("urgent","support"),("low","other"),("urgent","billing"),("low","feedback"),("urgent","support")]
for p,c in actions2:
    if env.state()["done"]:
        break
    result = env.step(Action(priority=p, category=c))
    print(f"  {obs.email_id} | p={p} c={c} | reward={result.reward}")
    obs = result.observation
print(f"  Avg reward: {env.state()['avg_reward']}")

print("\n--- Task 3: Reply Drafting ---")
env = EmailTriageEnv(task_id=3, shuffle=False)
obs = env.reset()
actions3 = [
    None,
    "Sorry about the billing issue. Our team will review your invoice and issue a refund if needed.",
    "Yes we support Linux. Happy to help with any other questions.",
    None,
    "Thank you for your feedback! We appreciate you sharing this with our team.",
    "We sincerely apologize. Our team is investigating and will restore your account shortly.",
    None,
    "Sorry for the billing confusion. We will review and issue a refund if applicable.",
    "Thank you! We will note the dark mode request for our mobile team.",
    "Sorry to hear about the login issue. Our team will investigate right away.",
]
for r in actions3:
    if env.state()["done"]:
        break
    result = env.step(Action(reply=r))
    print(f"  {obs.email_id} | reward={result.reward}")
    obs = result.observation
print(f"  Avg reward: {env.state()['avg_reward']}")

print("\nAll tasks done!")
