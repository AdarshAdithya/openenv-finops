import random
from .models import Observation, Action, Resource


class FinOpsEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.steps = 0
        self.resources = [
            Resource(id=f"dev-{i}", type="compute", cost=10.0, utilization=0.0, is_prod=False)
            for i in range(5)
        ]
        self.resources.append(
            Resource(id="prod-db", type="database", cost=50.0, utilization=80.0, is_prod=True)
        )
        return self.state()

    def state(self) -> Observation:
        return Observation(
            resources=self.resources,
            total_cost=sum(r.cost for r in self.resources),
        )

    def step(self, action: Action):
        self.steps = getattr(self, "steps", 0) + 1
        reward = -0.1
        done = self.steps >= 10

        target = next((r for r in self.resources if r.id == action.target_id), None)

        if action.cmd == "terminate" and target:
            if target.is_prod:
                reward = -50.0
                done = True
            else:
                reward = target.cost
                self.resources.remove(target)
        elif action.cmd == "resize" and target:
            if not target.is_prod and target.utilization < 50.0:
                reward = target.cost / 2.0
                target.cost /= 2.0

        return self.state(), reward, done, {}