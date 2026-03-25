import random
from utils.models import Observation, Action, StepResult
from utils.data import EMAILS
from graders.grader1 import grade_task1
from graders.grader2 import grade_task2
from graders.grader3 import grade_task3


class EmailTriageEnv:
    def __init__(self, task_id: int = 1, shuffle: bool = True):
        assert task_id in [1, 2, 3], "task_id must be 1, 2, or 3"
        self.task_id = task_id
        self.shuffle = shuffle
        self._emails = EMAILS.copy()
        self._index = 0
        self._step_num = 0
        self._done = False
        self._total_reward = 0.0
        self._episode_rewards = []
        if self.shuffle:
            random.shuffle(self._emails)

    def reset(self) -> Observation:
        self._emails = EMAILS.copy()
        if self.shuffle:
            random.shuffle(self._emails)
        self._index = 0
        self._step_num = 0
        self._done = False
        self._total_reward = 0.0
        self._episode_rewards = []
        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        current_email = self._emails[self._index]
        reward_signal = self._grade(action, current_email)
        self._total_reward += reward_signal.score
        self._episode_rewards.append(reward_signal.score)
        self._step_num += 1
        self._index += 1
        done = self._index >= len(self._emails)
        self._done = done
        next_obs = self._make_observation(final=done)
        info = {
            **reward_signal.info,
            "email_id": current_email["email_id"],
            "step": self._step_num,
            "total_reward": round(self._total_reward, 4),
            "avg_reward": round(self._total_reward / self._step_num, 4),
        }
        return StepResult(observation=next_obs, reward=reward_signal.score, done=done, info=info)

    def state(self) -> dict:
        return {
            "task_id": self.task_id,
            "step_number": self._step_num,
            "emails_remaining": max(0, len(self._emails) - self._index),
            "total_emails": len(self._emails),
            "total_reward": round(self._total_reward, 4),
            "avg_reward": round(self._total_reward / max(self._step_num, 1), 4),
            "done": self._done,
            "episode_rewards": self._episode_rewards,
        }

    def _make_observation(self, final: bool = False) -> Observation:
        if final or self._index >= len(self._emails):
            return Observation(email_id="END", subject="Episode Complete", body="All emails processed.", sender="system@env", task_id=self.task_id, step_number=self._step_num)
        e = self._emails[self._index]
        return Observation(email_id=e["email_id"], subject=e["subject"], body=e["body"], sender=e["sender"], task_id=self.task_id, step_number=self._step_num)

    def _grade(self, action: Action, email: dict):
        if self.task_id == 1:
            return grade_task1(action, email)
        elif self.task_id == 2:
            return grade_task2(action, email)
        elif self.task_id == 3:
            return grade_task3(action, email)
