# LinkedIn Post — Balancia Robot

## Option 1: Technical / Story (recommended)

🤖 Balancia Robot — Teaching a two-wheeled robot to balance with Deep RL

After weeks of tuning MuJoCo + Gymnasium + PPO, my Balancia robot now balances for 5000 steps without falling and tracks velocity commands up to 0.5 m/s.

**What I built:**
• Custom Gymnasium env (7-dim obs: pitch, pitch_rate, yaw_rate, velocities + target_vel)
• Symmetric drive: 1-D torque → both wheels (no differential)
• Reward shaping: +1 alive -7.0*pitch² -0.4*vel_error² -0.05*action_rate² … and smooth velocity ramping
• PPO (SB3, MlpPolicy): 2M steps, 8 parallel envs, ~7000 FPS → ep_len 4958/5000, peak ep_rew 4599

**Key lesson:** Increasing a reward weight doesn't just make the policy "more aware" — it rebalances the whole trade-off. Crank pitch_weight too high and you get jittery over-correction; too low and it learns to lean 8° to chase velocity. The sweet spot was `PITCH_WEIGHT=7.0`, `VEL_WEIGHT=0.4` + `ACTION_RATE=0.05` to kill chattering.

**Classical vs RL:** PID (KP=9, KD=0.5) balances, but RL recovers from random ±13° tilt + 0.5 rad/s kicks and follows changing velocity goals mid-episode (resampled every 300 steps).

🔗 GitHub: https://github.com/<you>/balancia-robot
📊 TensorBoard + checkpoints + viewer: `tensorboard --logdir logs` / `python scripts/play_rl.py --target-vel 0.3`

Huge thanks to the MuJoCo & SB3 communities. Next: sim2real on the physical Balancia!

#ReinforcementLearning #Robotics #MuJoCo #DeepLearning #PPO #Gymnasium #Python #ControlSystems #Sim2Real

---

## Option 2: Short / Punchy

Two wheels. One torque. Zero falls.

My PPO agent learned to balance a MuJoCo two-wheeled robot for 5000 steps — even with random 13° tilts and velocity commands to 0.5 m/s. 2M steps later, it beats my hand-tuned PID.

Code + env + training pipeline open: https://github.com/<you>/balancia-robot

Try it: `python scripts/play_rl.py --target-vel 0.3`

What would you add next — domain randomization or sim2real?

#Robotics #ReinforcementLearning #MuJoCo

---

## Visuals to attach (3-4):

1. **Video** 15s: `play_rl.py` viewer — robot catching fall + tracking `target_vel 0.3` (record with `ffmpeg` or phone)
2. **TensorBoard screenshot**: `rollout/ep_len_mean` → 4958 and `ep_rew_mean` → 4599 peak
3. **Architecture card**: obs/action/reward table (from docs/ENV_REFERENCE.md)
4. **Before/After**: early checkpoint (wobbly, falls @ 500 steps) vs latest (stable 5000 steps)

**Post timing:** Tue-Thu 09:00-11:00 CET, tag 3 colleagues in robotics/RL.

**First comment (for algorithm):**
> Full write-up in README — env details, hyperparams, and how ramped target_vel + action_rate penalty fixed velocity-tracking failures. Happy to share the 2015808-step checkpoint that gets 0/20 falls vs latest that overfits. DM for the .zip!
