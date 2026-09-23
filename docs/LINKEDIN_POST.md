# LinkedIn Post — Spontaneous

## Post (copy/paste)

was just playing around with RL this weekend and somehow got my two-wheeled robot to balance 😅

started with a simple MuJoCo model (two wheels + a stick basically) and a PPO agent that had no clue what it was doing — falling every 40 steps.

2M steps later... it just stands there. 5000 steps, random pushes, even velocity commands — and it stays up.

no fancy tricks, just a lot of trial and error on the reward and a lot of coffee ☕

still not perfect but watching it catch itself after a 13° tilt for the first time felt unreal

code is here if you want to play with it too: https://github.com/<you>/balancia-robot

video below 👇

#ReinforcementLearning #Robotics #MuJoCo #LearningByDoing

---

## Variant 2 — even shorter / more casual

ok this was supposed to be a quick RL experiment...

built a little two-wheeled robot in MuJoCo and told an agent "just don't fall"

a few days and 2 million steps later it actually learned to balance 😄

kinda addictive watching it go from faceplanting every 2 seconds to handling random tilts like nothing happened

put the code on github for anyone who wants to try — it's messy but it works!

#Robotics #PPO

---

## Variant 3 — with a bit more context but still spontaneous

Been messing with reinforcement learning in my free time

My challenge: make a two-wheeled robot balance like a Segway

At first it was hilarious — just flopping over instantly. Tried a PID controller, it worked but was shaky.

Then switched to PPO, let it play in simulation... after ~2M steps it finally clicked. Now it recovers on its own and can even follow speed commands.

That moment when it first balanced for a full episode (5000 steps) — I literally just stared at the viewer for a minute 😂

Super fun to build. If you're curious, everything is open: https://github.com/<you>/balancia-robot

---

## Visuals

Just one video is enough for this tone — 10-15 sec screen recording of `python scripts/play_rl.py` with the robot wobbling then catching itself. No need for TensorBoard screenshots, keeps it spontaneous.

Caption the video: "from 0 to 5000 steps — no code changes, just learning"
